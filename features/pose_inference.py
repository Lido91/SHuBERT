"""Extract [selected layers, frames, hidden] features from a trained pose-only checkpoint."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shubert_pose import SHubertPoseConfig, SHubertPoseModel  # noqa: E402
from shubert_pose.data import load_stream, read_manifest  # noqa: E402


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "cfg" not in checkpoint or "model" not in checkpoint:
        raise ValueError("require a Fairseq pose checkpoint containing both cfg and model")
    saved = OmegaConf.create(checkpoint["cfg"])
    if saved.model.get("_name") != "shubert_pose":
        raise ValueError("this extractor requires a shubert_pose checkpoint")
    values = OmegaConf.to_container(saved.model, resolve=True)
    values.pop("_name", None)
    cfg = OmegaConf.merge(OmegaConf.structured(SHubertPoseConfig()), values)
    model = SHubertPoseModel(cfg)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def parse_layers(value, total):
    """'all' or comma-separated 1-based Transformer layer numbers, e.g. '12' or '4,8,12'."""
    if value == "all":
        return list(range(1, total + 1))
    layers = [int(x) for x in value.split(",")]
    if not layers or len(set(layers)) != len(layers) or any(not 1 <= x <= total for x in layers):
        raise ValueError(f"--layers must be 'all' or unique layer numbers in [1, {total}]")
    return layers


def extract(model, manifest, output_dir, device, layers="all", resume=False):
    rows = read_manifest(manifest, model.streams)
    layers = parse_layers(layers, model.cfg.encoder_layers)
    output_dir = Path(output_dir)
    if not resume and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir}: choose an empty output directory, or pass --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.inference_mode():
        for row in rows:
            output = output_dir / f"{row['id']}.npy"
            expected = (len(layers), row["frames"], model.cfg.encoder_embed_dim)
            if output.exists():
                if not resume:
                    raise FileExistsError(f"{output}: choose a fresh output directory")
                # A kept file must be what this run would have written.
                if np.load(output, mmap_mode="r").shape != expected:
                    raise ValueError(f"{output}: existing shape differs from {expected}; use a fresh directory")
            else:
                source = {s: load_stream(row["paths"][s], row["frames"], d).unsqueeze(0).to(device)
                          for s, d in zip(model.streams, model.cfg.stream_dims)}
                # The encoder stops after the deepest requested layer.
                result = model.extract_features(source, layer=max(layers) - 1)
                # Fairseq layer tuples are (hidden states, attention, feed-forward states).
                features = np.stack([result["layer_results"][n - 1][0][:, 0].float().cpu().numpy() for n in layers])
                if features.shape != expected or not np.isfinite(features).all():
                    raise ValueError(f"{row['id']}: invalid output {features.shape}, expected {expected}")
                # Write then rename, so an interrupted run never leaves a truncated file for --resume.
                temporary = output.with_name(f".{output.name}.tmp")
                with temporary.open("wb") as handle:
                    np.save(handle, features)
                temporary.replace(output)
            records.append({"id": row["id"], "frames": row["frames"], "shape": list(expected), "output": str(output)})
    (output_dir / "manifest.json").write_text(json.dumps({"layers": layers, "files": records}, indent=2) + "\n")
    print(f"Saved {len(records)} pose feature files to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="all", help="'all' or 1-based layer numbers, e.g. 12 or 4,8,12")
    parser.add_argument("--resume", action="store_true", help="keep feature files already in the output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    extract(load_model(args.checkpoint, device), args.manifest, args.output_dir, device, args.layers, args.resume)


if __name__ == "__main__":
    main()
