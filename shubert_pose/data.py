"""Aligned coordinate streams and per-frame cluster labels."""

import csv
from pathlib import Path

import numpy as np
import torch


def read_manifest(path, streams):
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"id", "frames", *streams}
        if reader.fieldnames is None or set(reader.fieldnames) != expected or len(reader.fieldnames) != len(expected):
            raise ValueError(f"{path}: expected header columns {sorted(expected)}")
        rows, seen = [], set()
        for line, row in enumerate(reader, 2):
            if set(row) != expected or any(value is None or not value for value in row.values()):
                raise ValueError(f"{path}:{line}: missing or extra manifest fields")
            sample_id = row["id"]
            if not sample_id or sample_id in seen or Path(sample_id).name != sample_id or sample_id in {".", ".."}:
                raise ValueError(f"{path}:{line}: invalid or duplicate sample ID {sample_id!r}")
            frames = int(row["frames"])
            if frames < 1:
                raise ValueError(f"{path}:{line}: frame count must be positive")
            paths = {}
            for stream in streams:
                value = Path(row[stream])
                paths[stream] = value if value.is_absolute() else path.parent / value
            rows.append({"id": sample_id, "frames": frames, "paths": paths})
            seen.add(sample_id)
    if not rows:
        raise ValueError(f"{path}: empty manifest")
    return rows


def load_stream(path, frames, dimension):
    array = np.load(path, allow_pickle=False)
    if array.shape != (frames, dimension):
        raise ValueError(f"{path}: expected {(frames, dimension)}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: contains non-finite coordinates")
    return torch.from_numpy(np.asarray(array, dtype=np.float32).copy())
