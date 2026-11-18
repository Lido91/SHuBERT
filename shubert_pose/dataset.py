import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fairseq.data import FairseqDataset

from .data import load_stream, read_manifest


class PoseDataset(FairseqDataset):
    def __init__(self, manifest, streams, dimensions, codebook_size, min_frames=2, max_frames=5000, shuffle=True):
        super().__init__()
        self.streams = list(streams)
        self.dimensions = list(dimensions)
        if len(self.streams) != len(self.dimensions):
            raise ValueError("stream_names and stream_dims must have equal lengths")
        if min_frames < 1 or max_frames < min_frames:
            raise ValueError("require 1 <= min_frames <= max_frames")
        metadata_path = Path(manifest).parent / "codebooks" / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            digest = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
            if metadata["manifest_sha256"].get(Path(manifest).stem) != digest:
                raise ValueError(f"{manifest}: manifest changed; regenerate cluster labels")
            if metadata["clusters"] != codebook_size or metadata["streams"] != dict(zip(streams, dimensions)):
                raise ValueError("codebook metadata does not match configured streams, dimensions, or cluster count")
        rows = read_manifest(manifest, streams)
        self.labels = {}
        for stream in streams:
            label_path = Path(manifest).with_suffix(f".{stream}.km")
            labels = [torch.tensor([int(x) for x in line.split()], dtype=torch.long)
                      for line in label_path.read_text().splitlines()]
            if len(labels) != len(rows):
                raise ValueError(f"{label_path}: expected {len(rows)} label rows, got {len(labels)}")
            for row, label in zip(rows, labels):
                if label.numel() != row["frames"]:
                    raise ValueError(f"{label_path}: frame/label mismatch for {row['id']}")
                if (label < 0).any() or (label >= codebook_size).any():
                    raise ValueError(f"{label_path}: cluster IDs must be in [0, {codebook_size})")
            self.labels[stream] = labels
        # Filter coordinates and labels by the same original row index.
        kept = [i for i, row in enumerate(rows) if row["frames"] >= min_frames]
        self.rows = [rows[i] for i in kept]
        self.labels = {s: [labels[i] for i in kept] for s, labels in self.labels.items()}
        if not kept:
            raise ValueError(f"{manifest}: no samples meet min_frames={min_frames}")
        self.sizes = np.array([min(row["frames"], max_frames) for row in self.rows], dtype=np.int64)
        self.max_frames = max_frames
        self.shuffle = shuffle

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {
            "id": index,
            "source": {s: load_stream(row["paths"][s], row["frames"], d)
                       for s, d in zip(self.streams, self.dimensions)},
            "labels": {s: self.labels[s][index] for s in self.streams},
        }

    def collater(self, samples):
        if not samples:
            return {}
        lengths = [min(next(iter(s["source"].values())).size(0), self.max_frames) for s in samples]
        size = max(lengths)
        source = {s: torch.zeros(len(samples), size, d) for s, d in zip(self.streams, self.dimensions)}
        labels = {s: torch.full((len(samples), size), -100, dtype=torch.long) for s in self.streams}
        padding = torch.ones(len(samples), size, dtype=torch.bool)
        for i, (sample, length) in enumerate(zip(samples, lengths)):
            original_length = next(iter(sample["source"].values())).size(0)
            start = np.random.randint(original_length - length + 1) if self.shuffle else 0
            padding[i, :length] = False
            for stream in self.streams:
                source[stream][i, :length] = sample["source"][stream][start:start + length]
                labels[stream][i, :length] = sample["labels"][stream][start:start + length]
        return {
            "id": torch.tensor([s["id"] for s in samples]),
            "nsentences": len(samples),
            "ntokens": sum(lengths),
            "net_input": {"source": source, "labels": labels, "padding_mask": padding},
        }

    def num_tokens(self, index):
        return int(self.sizes[index])

    def num_tokens_vec(self, indices):
        return self.sizes[indices]

    def size(self, index):
        return self.num_tokens(index)

    def ordered_indices(self):
        order = np.random.permutation(len(self)) if self.shuffle else np.arange(len(self))
        return order[np.argsort(self.sizes[order], kind="stable")]
