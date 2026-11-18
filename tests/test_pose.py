import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.nn import functional as F

from shubert_pose import SHubertPoseConfig, SHubertPoseModel
from shubert_pose.dataset import PoseDataset


def tiny_config(**kwargs):
    defaults = dict(stream_names=["body", "left_hand", "right_hand"], stream_dims=[4, 6, 6],
                    stream_embed_dim=8, codebook_size=5, encoder_layers=2, encoder_embed_dim=16,
                    encoder_ffn_embed_dim=32, encoder_attention_heads=2, conv_pos=3, conv_pos_groups=2,
                    required_seq_len_multiple=1, dropout=0.0, attention_dropout=0.0, activation_dropout=0.0,
                    encoder_layerdrop=0.0)
    defaults.update(kwargs)
    return SHubertPoseConfig(**defaults)


class PoseModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = SHubertPoseModel(tiny_config())
        self.source = {s: torch.randn(2, 6, d) for s, d in zip(self.model.streams, self.model.cfg.stream_dims)}
        self.padding = torch.tensor([[False] * 6, [False] * 3 + [True] * 3])
        self.labels = {s: torch.randint(5, (2, 6)) for s in self.model.streams}

    def test_masked_only_loss_and_gradients(self):
        mask = torch.zeros(2, 6, 3, dtype=torch.bool)
        mask[:, 1] = True
        mask[1, 4] = True  # Padded positions must never contribute.
        result = self.model(self.source, self.padding, self.labels, mask_indices=mask)
        self.assertEqual(result["sample_size"], 6)
        self.assertFalse(result["masked_positions"][self.padding].any())
        changed = {s: target.clone() for s, target in self.labels.items()}
        for i, s in enumerate(self.model.streams):
            changed[s][~result["masked_positions"][:, :, i]] = -100
        other = self.model(self.source, self.padding, changed, mask_indices=mask)
        torch.testing.assert_close(result["losses"]["kmeans_loss"], other["losses"]["kmeans_loss"])
        result["losses"]["kmeans_loss"].mean().backward()
        for s in self.model.streams:
            self.assertGreater(self.model.heads[s].weight.grad.abs().sum().item(), 0)
            self.assertGreater(self.model.projections[s].weight.grad.abs().sum().item(), 0)

    def test_loss_matches_manual_cluster_prediction(self):
        mask = torch.zeros(2, 6, 3, dtype=torch.bool)
        mask[:, 1] = True
        features = self.model(self.source, self.padding, features_only=True, mask_indices=mask)["x"]
        expected = torch.cat([F.cross_entropy(self.model.heads[s](features[:, 1]).float(),
                                            self.labels[s][:, 1], reduction="none") for s in self.model.streams])
        result = self.model(self.source, self.padding, self.labels, mask_indices=mask)
        torch.testing.assert_close(result["losses"]["kmeans_loss"], expected)

    def test_padding_values_do_not_change_valid_features(self):
        self.model.eval()
        before = self.model.extract_features(self.source, self.padding)["x"]
        changed = {s: x.clone() for s, x in self.source.items()}
        for x in changed.values():
            x[self.padding] = 1000
        after = self.model.extract_features(changed, self.padding)["x"]
        torch.testing.assert_close(before[~self.padding], after[~self.padding])

    def test_body_only_short_sequences_and_feature_extraction(self):
        for strategy in ("random", "time"):
            model = SHubertPoseModel(tiny_config(stream_names=["body"], stream_dims=[14], mask_strategy=strategy))
            source = {"body": torch.randn(2, 2, 14)}
            result = model(source, labels={"body": torch.zeros(2, 2, dtype=torch.long)})
            self.assertTrue(torch.isfinite(result["losses"]["kmeans_loss"]).all())
            self.assertEqual(result["sample_size"], 2)
            result["losses"]["kmeans_loss"].sum().backward()
            features = model.eval().extract_features(source)
            self.assertEqual(features["x"].shape, (2, 2, 16))
            self.assertEqual(len(features["layer_results"]), 2)

    def test_sampled_masks_respect_padding_and_keep_context(self):
        padding = torch.zeros(4, 40, dtype=torch.bool)
        padding[1, 25:] = True
        padding[2, 1:] = True
        padding[3, 2:] = True
        for strategy in ("random", "time", "flat"):
            model = SHubertPoseModel(tiny_config(mask_strategy=strategy))
            for _ in range(20):
                mask = model._sample_mask(padding)
                self.assertEqual(mask.shape, (4, 40, 3))
                self.assertFalse(mask[padding].any())
                for i in range(4):
                    valid = mask[i][~padding[i]]
                    self.assertTrue(valid.any())
                    if valid.size(0) > 1:  # a one-frame clip has nothing else to observe
                        self.assertFalse(valid.all())
                if strategy == "time":
                    self.assertTrue((mask == mask[:, :, :1]).all())

    def test_flat_masking_spans_streams_within_a_frame(self):
        torch.manual_seed(0)
        padding = torch.zeros(16, 200, dtype=torch.bool)
        runs = {}
        for strategy in ("random", "flat"):
            mask = SHubertPoseModel(tiny_config(mask_strategy=strategy))._sample_mask(padding)
            self.assertGreater(mask.float().mean().item(), 0.45)
            self.assertLess(mask.float().mean().item(), 0.75)
            # Mean temporal run length of masked positions within one stream.
            starts = mask & ~torch.nn.functional.pad(mask, (0, 0, 1, 0))[:, :-1]
            runs[strategy] = mask.sum().item() / starts.sum().item()
        self.assertLess(runs["flat"], runs["random"])

    def test_extract_selected_layers_and_resume(self):
        from features.pose_inference import extract
        model = self.model.eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"id": "clip", "frames": 6}
            for s in model.streams:
                np.save(root / f"{s}.npy", self.source[s][0].numpy())
                row[s] = f"{s}.npy"
            with (root / "valid.tsv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", *model.streams, "frames"], delimiter="\t")
                writer.writeheader()
                writer.writerow(row)
            extract(model, root / "valid.tsv", root / "all", torch.device("cpu"))
            extract(model, root / "valid.tsv", root / "last", torch.device("cpu"), layers="2")
            everything, last = np.load(root / "all" / "clip.npy"), np.load(root / "last" / "clip.npy")
            self.assertEqual(everything.shape, (2, 6, 16))
            np.testing.assert_allclose(last, everything[1:], rtol=1e-5, atol=1e-6)
            with self.assertRaises(FileExistsError):
                extract(model, root / "valid.tsv", root / "last", torch.device("cpu"), layers="2")
            extract(model, root / "valid.tsv", root / "last", torch.device("cpu"), layers="2", resume=True)
            with self.assertRaisesRegex(ValueError, "existing shape"):
                extract(model, root / "valid.tsv", root / "last", torch.device("cpu"), resume=True)

    def test_checkpoint_round_trip(self):
        from features.pose_inference import load_model
        cfg = OmegaConf.structured(self.model.cfg)
        cfg._name = "shubert_pose"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"cfg": {"model": OmegaConf.to_container(cfg, enum_to_str=True)},
                        "model": self.model.state_dict()}, path)
            loaded = load_model(path, torch.device("cpu"))
        expected = self.model.eval().extract_features(self.source)["x"]
        torch.testing.assert_close(expected, loaded.extract_features(self.source)["x"])


class PoseDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rows = []
        for i, length in enumerate((1, 4, 7)):
            path = self.root / f"clip{i}.npy"
            np.save(path, np.repeat(np.arange(length, dtype=np.float32)[:, None], 4, axis=1))
            rows.append({"id": f"clip{i}", "body": path.name, "frames": length})
        self.manifest = self.root / "train.tsv"
        with self.manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "body", "frames"], delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        (self.root / "train.body.km").write_text("0\n0 1 2 3\n0 1 2 3 4 5 6\n")

    def test_filter_crop_and_labels_stay_aligned(self):
        data = PoseDataset(self.manifest, ["body"], [4], 7, min_frames=2, max_frames=5)
        self.assertEqual(len(data), 2)
        self.assertEqual(data.rows[0]["id"], "clip1")
        np.random.seed(4)
        batch = data.collater([data[0], data[1]])
        source, labels = batch["net_input"]["source"]["body"], batch["net_input"]["labels"]["body"]
        valid = ~batch["net_input"]["padding_mask"]
        torch.testing.assert_close(source[:, :, 0][valid], labels[valid].float())
        self.assertEqual(batch["ntokens"], 9)
        self.assertEqual(labels[0, -1], -100)

    def test_label_frame_mismatch_is_rejected(self):
        (self.root / "train.body.km").write_text("0\n0 1\n0 1 2 3 4 5 6\n")
        with self.assertRaisesRegex(ValueError, "frame/label mismatch"):
            PoseDataset(self.manifest, ["body"], [4], 7)

    def test_bad_coordinate_shape_is_rejected(self):
        data = PoseDataset(self.manifest, ["body"], [14], 7)
        with self.assertRaisesRegex(ValueError, "expected"):
            data[0]

    def test_reordered_manifest_is_rejected_with_codebook_metadata(self):
        (self.root / "codebooks").mkdir()
        metadata = {"clusters": 7, "streams": {"body": 4},
                    "manifest_sha256": {"train": hashlib.sha256(self.manifest.read_bytes()).hexdigest()}}
        (self.root / "codebooks" / "metadata.json").write_text(json.dumps(metadata))
        PoseDataset(self.manifest, ["body"], [4], 7)
        lines = self.manifest.read_text().splitlines()
        self.manifest.write_text("\n".join([lines[0], *reversed(lines[1:])]) + "\n")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            PoseDataset(self.manifest, ["body"], [4], 7)


if __name__ == "__main__":
    unittest.main()
