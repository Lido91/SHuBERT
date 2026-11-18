"""Memory-mapped coordinate streams and labels for corpus-scale pretraining."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fairseq.data import FairseqDataset
from .dataset import PoseDataset


class PackedPoseDataset(FairseqDataset):
    collater = PoseDataset.collater
    num_tokens = PoseDataset.num_tokens
    num_tokens_vec = PoseDataset.num_tokens_vec
    size = PoseDataset.size
    ordered_indices = PoseDataset.ordered_indices

    def __init__(self, manifest, streams, dimensions, codebook_size, min_frames=2, max_frames=256, shuffle=True):
        super().__init__()
        self.manifest = Path(manifest)
        self.metadata = json.loads(self.manifest.read_text())
        if self.metadata.get("format") != "shubert-pose-pack-v1":
            raise ValueError("unsupported packed pose format")
        self.streams, self.dimensions = list(streams), list(dimensions)
        if (self.metadata["dimensions"] != dict(zip(streams, dimensions))
                or self.metadata["codebook_size"] != codebook_size):
            raise ValueError("packed streams, dimensions or cluster count do not match configuration")
        if min_frames < 1 or max_frames < min_frames:
            raise ValueError("require 1 <= min_frames <= max_frames")
        for key in ("offsets", "indices"):
            path = Path(self.metadata[key])
            if hashlib.sha256(path.read_bytes()).hexdigest() != self.metadata[f"{key}_sha256"]:
                raise ValueError(f"packed {key} changed; regenerate labels and manifests")
        self.offsets = np.load(self.metadata["offsets"], mmap_mode="r", allow_pickle=False)
        indices = np.load(self.metadata["indices"], allow_pickle=False)
        if (self.offsets.ndim != 1 or self.offsets[0] != 0 or (np.diff(self.offsets) < 1).any()
                or indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices)
                or (indices < 0).any() or (indices >= len(self.offsets) - 1).any()):
            raise ValueError("invalid packed offsets or clip indices")
        lengths = self.offsets[indices + 1] - self.offsets[indices]
        self.indices = indices[lengths >= min_frames]
        self.sizes = np.minimum(lengths[lengths >= min_frames], max_frames).astype(np.int64)
        if not len(self.indices):
            raise ValueError("no packed clips meet min_frames")
        self.max_frames, self.shuffle = max_frames, shuffle
        self._open_arrays()

    def _open_arrays(self):
        total = int(self.offsets[-1])
        self.features = {s: np.load(self.metadata["features"][s], mmap_mode="r", allow_pickle=False)
                         for s in self.streams}
        self.labels = {s: np.load(self.metadata["labels"][s], mmap_mode="r", allow_pickle=False)
                       for s in self.streams}
        for stream, dimension in zip(self.streams, self.dimensions):
            if self.features[stream].shape != (total, dimension) or self.labels[stream].shape != (total,):
                raise ValueError(f"{stream}: packed array shape mismatch")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        clip = self.indices[index]
        begin, end = map(int, self.offsets[clip:clip + 2])
        length = min(end - begin, self.max_frames)
        begin += np.random.randint(end - begin - length + 1) if self.shuffle else 0
        end = begin + length
        source, labels = {}, {}
        for stream in self.streams:
            coordinates = np.array(self.features[stream][begin:end], dtype=np.float32, copy=True)
            targets = np.array(self.labels[stream][begin:end], dtype=np.int64, copy=True)
            if (not np.isfinite(coordinates).all() or (targets < 0).any()
                    or (targets >= self.metadata["codebook_size"]).any()):
                raise ValueError(f"invalid packed coordinates or labels for clip {clip}, {stream}")
            source[stream], labels[stream] = torch.from_numpy(coordinates), torch.from_numpy(targets)
        return {"id": index, "source": source, "labels": labels}

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("features", "labels", "offsets"):
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.offsets = np.load(self.metadata["offsets"], mmap_mode="r", allow_pickle=False)
        self._open_arrays()
