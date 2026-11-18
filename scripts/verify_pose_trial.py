"""Verify a finished real-data pose trial and write a compact durable result."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def verify(root):
    root = Path(root)
    checkpoint = torch.load(root / "checkpoints/checkpoint_last.pt", map_location="cpu", weights_only=False)
    updates = checkpoint["optimizer_history"][-1]["num_updates"]
    if updates != 200:
        raise ValueError(f"expected 200 training updates, got {updates}")
    if not all(torch.isfinite(value).all() for value in checkpoint["model"].values()):
        raise ValueError("checkpoint contains non-finite model weights")
    with (root / "data/valid.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    feature_manifest = json.loads((root / "valid_features/manifest.json").read_text())
    if feature_manifest["layers"] != [12] or [r["id"] for r in feature_manifest["files"]] != [r["id"] for r in rows]:
        raise ValueError("feature manifest does not match the requested clips and layer")
    for row in rows:
        array = np.load(root / "valid_features" / f"{row['id']}.npy", allow_pickle=False)
        expected = (1, int(row["frames"]), 768)
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(f"invalid features for {row['id']}: {array.shape}, expected {expected}")
    summary = {"status": "verified", "updates": updates, "validation_clips": len(rows),
               "validation_frames": sum(int(row["frames"]) for row in rows),
               "feature_layers": [12], "feature_dimension": 768,
               "checkpoint": str(root / "checkpoints/checkpoint_last.pt"),
               "feature_directory": str(root / "valid_features")}
    temporary = root / "verification.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(root / "verification.json")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-root", required=True)
    verify(parser.parse_args().trial_root)
