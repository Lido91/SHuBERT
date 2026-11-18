"""SHuBERT section 4.3 hyperparameters, applied to a pose-only checkpoint."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.pose_inference import load_model  # noqa: E402
from shubert_pose.wlasl import (PoseFullFTClassifier, PoseLoRAClassifier, WLASLPoseDataset, atomic_json, collate_clips,
                               frozen_digest, metrics_from_counts, sha256)  # noqa: E402


def atomic_save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()


def to_device(batch, device):
    source, padding, labels, ids = batch
    return ({s: x.to(device, non_blocking=True) for s, x in source.items()},
            padding.to(device, non_blocking=True), labels.to(device, non_blocking=True), ids)


@torch.inference_mode()
def evaluate(model, loader, device, max_batches=0, predictions=False):
    model.eval()
    counts = torch.zeros(2000, 4, device=device, dtype=torch.float64)
    loss_sum, records, total = 0.0, [], 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        source, padding, labels, ids = to_device(batch, device)
        with amp(device):
            logits = model(source, padding)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite evaluation logits")
        loss_sum += nn.functional.cross_entropy(logits.float(), labels, reduction="sum").item()
        top = logits.topk(10, dim=1).indices
        correct = top.eq(labels[:, None])
        counts[:, 0].scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float64))
        for column, k in enumerate((1, 5, 10), 1):
            counts[:, column].scatter_add_(0, labels, correct[:, :k].any(1).double())
        total += len(labels)
        if predictions:
            records.extend({"id": name, "label": label, "top10": candidates}
                           for name, label, candidates in zip(ids, labels.cpu().tolist(), top.cpu().tolist()))
    result = metrics_from_counts(counts)
    result["loss"] = loss_sum / total
    if not max_batches and total != len(loader.dataset):
        raise RuntimeError("evaluation coverage mismatch")
    if predictions and len({r["id"] for r in records}) != len(records):
        raise RuntimeError("duplicate test predictions")
    return result, records


def run(args):
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("the full WLASL evaluation requires a GPU")
    device = torch.device("cuda")
    seed_all(args.seed)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    score_role = "dev" if args.selection_split == "dev" else "test_as_dev"
    result_name = "test_metrics.json" if args.selection_split == "dev" else "test_as_dev_metrics.json"
    backbone = load_model(args.checkpoint, torch.device("cpu"))
    if list(backbone.cfg.stream_dims) != [14, 42, 42]:
        raise ValueError("this WLASL conversion requires the three 2D pose streams")
    classifier = PoseFullFTClassifier if args.finetune_mode == "full" else PoseLoRAClassifier
    model = classifier(backbone)
    before = frozen_digest(model)
    lora = [p for p in model.backbone.parameters() if p.requires_grad]
    tracked_weight = model.backbone.projections["body"].weight.detach().clone()
    config = {"checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": sha256(args.checkpoint),
              "paper": "https://aclanthology.org/2025.acl-long.1397.pdf#page=7", "section": "4.3",
              "epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size,
              "seed": args.seed, "finetune_mode": args.finetune_mode,
              "rank": 1 if args.finetune_mode == "lora" else None,
              "alpha": 1.0 if args.finetune_mode == "lora" else None,
              "head_lr": 1e-4, "lora_lr": 1e-5 if args.finetune_mode == "lora" else None,
              "backbone_lr": 1e-5, "selection_start_epoch": args.selection_start_epoch,
              "optimizer": "Adam", "weight_decay": 1e-4, "label_smoothing": 0.1,
              "selection": f"maximum {score_role} per-instance top-1; earliest epoch on ties",
              "selection_split": args.selection_split, "score_role": score_role,
              "independent_test": args.selection_split == "dev",
              "warm_start": str(Path(args.warm_start).resolve()) if args.warm_start else None,
              "warm_start_sha256": sha256(args.warm_start) if args.warm_start else None,
              "pooling": "final Transformer residual hidden states; mean over non-padding time; BatchNorm + Linear",
              "lora_layers": model.lora_layers,
              "lora_parameters": sum(p.numel() for p in lora) if args.finetune_mode == "lora" else 0,
              "backbone_trainable_parameters": sum(p.numel() for p in lora),
              "head_parameters": sum(p.numel() for p in model.head.parameters()),
              "frozen_sha256": before, "data": json.loads((Path(args.data) / "ready.json").read_text()),
              "precision": "bfloat16 autocast, float32 parameters and losses",
              "implementation_choices_not_specified_in_paper": {
                  "patience": args.patience, "seed": args.seed,
                  "alpha": 1 if args.finetune_mode == "lora" else None,
                  "scheduler": "constant", "frames": "all frames; no temporal resampling or augmentation",
                  "layer": "last", "encoder_layerdrop": 0, "activation_checkpointing": True},
              "scope": ("LoRA downstream settings on coordinate-only SHuBERT" if args.finetune_mode == "lora"
                        else "full feature-extractor fine-tuning on coordinate-only SHuBERT"),
              "code_sha256": sha256(__file__), "model_code_sha256": sha256(Path(__file__).parents[1] / "shubert_pose/wlasl.py"),
              "torch": torch.__version__}
    config_path = output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("output already contains a different checkpoint, data, or evaluation protocol")
        if (output / result_name).exists():
            print("ALREADY_COMPLETE " + str(output), flush=True)
            return
        if not args.resume:
            raise ValueError("existing output requires --resume")
    else:
        atomic_json(config_path, config)
    model.to(device)
    optimizer = torch.optim.Adam([{"params": model.head.parameters(), "lr": 1e-4},
                                  {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": 1e-5}],
                                 weight_decay=1e-4)
    used_splits = (("train", "test") if args.selection_split == "test" and args.selection_start_epoch == 1
                   else ("train", "dev", "test"))
    datasets = {split: WLASLPoseDataset(args.data, split) for split in used_splits}
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size if split == "train" else 64,
                                 shuffle=(split == "train"), num_workers=args.workers,
                                 persistent_workers=(args.workers > 0), pin_memory=True,
                                 collate_fn=collate_clips) for split, dataset in datasets.items()}
    first_epoch, best, best_epoch, bad, steps = 1, -1.0, 0, 0, 0
    resuming = args.resume and (output / "last.pth").exists()
    if resuming or args.warm_start:
        path = output / "last.pth" if resuming else Path(args.warm_start)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model.restore_adapted(saved["adapted"])
        optimizer.load_state_dict(saved["optimizer"])
        first_epoch, steps = saved["epoch"] + 1, saved["steps"]
        if resuming:
            best, best_epoch, bad = saved["best"], saved["best_epoch"], saved["bad_epochs"]
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        torch.set_rng_state(saved["rng"]["torch"])
        torch.cuda.set_rng_state(saved["rng"]["cuda"])
    print("START " + json.dumps({"checkpoint": args.checkpoint, "batch": args.batch_size,
                                 "lora_parameters": config["lora_parameters"],
                                 "first_epoch": first_epoch, "selection_split": args.selection_split,
                                 "splits": {k: len(v) for k, v in datasets.items()}}), flush=True)
    if args.warm_start and not resuming:
        baseline, _ = evaluate(model, loaders[args.selection_split], device,
                               max_batches=2 if args.mode == "smoke" else 0)
        best, best_epoch, bad = baseline["pi_top1"], first_epoch - 1, 0
        saved.update({"best": best, "best_epoch": best_epoch, "bad_epochs": bad})
        atomic_save(output / "best.pth", saved)
        atomic_json(output / "selection_baseline.json", {"epoch": best_epoch, score_role: baseline})
        print("SELECTION_BASELINE " + json.dumps({"epoch": best_epoch, score_role: baseline}), flush=True)
    start = time.monotonic()
    for epoch in range(first_epoch, args.epochs + 1):
        if bad >= args.patience:
            break
        epoch_start = time.monotonic()
        seed_all(args.seed + epoch)  # Same shuffling and dropout stream for every source checkpoint.
        model.train()
        loss_sum, correct, samples = 0.0, 0, 0
        for i, batch in enumerate(loaders["train"]):
            if args.mode == "smoke" and i >= args.smoke_steps:
                break
            source, padding, labels, _ = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with amp(device):
                logits = model(source, padding)
                loss = nn.functional.cross_entropy(logits.float(), labels, label_smoothing=0.1)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            for name, p in model.named_parameters():
                if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
                    raise FloatingPointError("missing/nonfinite gradient: " + name)
            optimizer.step()
            steps += 1
            samples += len(labels)
            loss_sum += loss.item() * len(labels)
            correct += logits.argmax(1).eq(labels).sum().item()
            if i % 20 == 0:
                atomic_json(output / "progress.json", {"phase": "training", "epoch": epoch,
                            "batch": i + 1, "batches": len(loaders["train"]), "steps": steps,
                            "train_loss": loss_sum / samples, "updated_unix": time.time()})
        selection_active = args.mode == "smoke" or epoch >= args.selection_start_epoch
        evaluated_split = args.selection_split if selection_active else "dev"
        metric_role = score_role if selection_active else "dev"
        dev, _ = evaluate(model, loaders[evaluated_split], device,
                          max_batches=2 if args.mode == "smoke" else 0)
        improved = selection_active and dev["pi_top1"] > best
        if improved:
            best, best_epoch, bad = dev["pi_top1"], epoch, 0
        elif selection_active:
            bad += 1
        row = {"epoch": epoch, "steps": steps, "train_loss": loss_sum / samples,
               "train_pi_top1": 100 * correct / samples, "train_samples": samples,
               metric_role: dev, "selection_split": evaluated_split, "selection_active": selection_active,
               "best_epoch": best_epoch,
               "best_selection_pi_top1": best, "bad_epochs": bad,
               "seconds": time.monotonic() - epoch_start,
               "max_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30}
        saved = {"adapted": model.adapted_state(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "steps": steps, "best": best, "best_epoch": best_epoch, "bad_epochs": bad,
                 "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                         "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}}
        atomic_save(output / "last.pth", saved)
        if improved:
            atomic_save(output / "best.pth", saved)
        atomic_json(output / "progress.json", {"phase": "epoch_finished", **row, "updated_unix": time.time()})
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        print("EPOCH " + json.dumps(row), flush=True)
        if args.mode == "smoke":
            if frozen_digest(model) != before:
                raise RuntimeError("original frozen weights changed")
            if args.finetune_mode == "lora" and not any(
                    p.detach().abs().sum().item() > 0 for name, p in model.named_parameters() if name.endswith(".b")):
                raise RuntimeError("LoRA adapters did not update")
            if args.finetune_mode == "full" and torch.equal(
                    tracked_weight, model.backbone.projections["body"].weight.detach().cpu()):
                raise RuntimeError("full fine-tuning did not update the backbone projection")
            sample = to_device(next(iter(loaders[args.selection_split])), device)
            with torch.inference_mode(), amp(device):
                reference = model(sample[0], sample[1])
                loaded = torch.load(output / "last.pth", map_location="cpu", weights_only=False)
                model.restore_adapted(loaded["adapted"])
                restored = model(sample[0], sample[1])
            torch.testing.assert_close(reference, restored, rtol=0, atol=0)
            atomic_json(output / "smoke_result.json", {**row, "frozen_weights_unchanged": True,
                                                       "finetune_mode": args.finetune_mode,
                                                       "backbone_weights_updated": args.finetune_mode == "full",
                                                       "checkpoint_roundtrip_exact": True})
            print("SMOKE_OK", flush=True)
            return
    if frozen_digest(model) != before:
        raise RuntimeError("original frozen weights changed")
    selected = torch.load(output / "best.pth", map_location="cpu", weights_only=False)
    model.restore_adapted(selected["adapted"])
    test, predictions = evaluate(model, loaders["test"], device, predictions=True)
    path = output / "test_predictions.jsonl"
    with path.with_suffix(".jsonl.tmp").open("w") as handle:
        for record in predictions:
            handle.write(json.dumps(record) + "\n")
    path.with_suffix(".jsonl.tmp").replace(path)
    result = {"checkpoint": config["checkpoint"], "checkpoint_sha256": config["checkpoint_sha256"],
              "selected_epoch": best_epoch, "best_selection_pi_top1": best, "score_role": score_role,
              "selection_split": args.selection_split, "independent_test": args.selection_split == "dev",
              "selection_started_epoch": (json.loads((output / "selection_baseline.json").read_text())["epoch"]
                                          if (output / "selection_baseline.json").exists()
                                          else args.selection_start_epoch),
              "finetune_mode": args.finetune_mode,
              "backbone_weights_updated": (not torch.equal(tracked_weight,
                                            model.backbone.projections["body"].weight.detach().cpu())),
              "warm_start": config["warm_start"],
              ("test" if args.selection_split == "dev" else "test_as_dev"): test,
              "frozen_weights_unchanged": True, "elapsed_seconds": time.monotonic() - start}
    atomic_json(output / result_name, result)
    atomic_json(output / "progress.json", {"phase": "complete", **result, "updated_unix": time.time()})
    print("FINAL_TEST " + json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=125)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selection-split", choices=("dev", "test"), default="dev")
    parser.add_argument("--finetune-mode", choices=("lora", "full"), default="lora")
    parser.add_argument("--selection-start-epoch", type=int, default=1)
    parser.add_argument("--warm-start", help="Continue optimizer/adapters from an earlier selection phase")
    parser.add_argument("--mode", choices=("train", "smoke"), default="train")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
