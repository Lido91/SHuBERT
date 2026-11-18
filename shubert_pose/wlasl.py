"""Rank-one LoRA and WLASL classification for coordinate-only SHuBERT."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class LoRALinear(nn.Module):
    """Expose the adapted weight for Fairseq's functional attention fast path."""

    def __init__(self, base, rank=1, alpha=1.0):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank

    @property
    def weight(self):
        return self.base.weight + (self.b @ self.a) * self.scale

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


def add_lora(module, prefix=""):
    names = []
    for name, child in list(module.named_children()):
        qualified = prefix + name
        if qualified == "heads":
            continue  # Cluster-prediction heads are not used for downstream features.
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child))
            names.append(qualified)
        else:
            names.extend(add_lora(child, qualified + "."))
    return names


class CheckpointLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, *args, **kwargs):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.layer, *args, use_reentrant=False, **kwargs)
        return self.layer(*args, **kwargs)


class PoseLoRAClassifier(nn.Module):
    def __init__(self, backbone, classes=2000, checkpoint_layers=True):
        super().__init__()
        self.backbone = backbone.requires_grad_(False)
        self.lora_layers = add_lora(backbone)
        if not self.lora_layers:
            raise ValueError("no feature-extractor linear layers received LoRA")
        # Keep every adapter in the training graph; no stochastic layer skipping.
        backbone.encoder.layerdrop = 0.0
        if checkpoint_layers:
            backbone.encoder.layers = nn.ModuleList([CheckpointLayer(x) for x in backbone.encoder.layers])
        self.head = nn.Sequential(nn.BatchNorm1d(backbone.cfg.encoder_embed_dim),
                                  nn.Linear(backbone.cfg.encoder_embed_dim, classes))

    def forward(self, source, padding_mask):
        features = self.backbone.extract_features(source, padding_mask=padding_mask, mask=False)["x"]
        valid = (~padding_mask).unsqueeze(-1)
        pooled = (features * valid).sum(1) / valid.sum(1).clamp_min(1)
        return self.head(pooled.float())

    def adapted_state(self):
        trainable = {name for name, p in self.named_parameters() if p.requires_grad}
        return {name: value.detach().cpu() for name, value in self.state_dict().items()
                if name in trainable or name.startswith("head.")}

    def restore_adapted(self, state):
        expected = set(self.adapted_state())
        if set(state) != expected:
            raise ValueError("adapter/classifier state keys do not match this model")
        current = self.state_dict()
        current.update(state)
        self.load_state_dict(current, strict=True)


class PoseFullFTClassifier(PoseLoRAClassifier):
    """Train all downstream feature weights and the classifier, without adapters."""

    def __init__(self, backbone, classes=2000, checkpoint_layers=True):
        nn.Module.__init__(self)
        self.backbone = backbone.requires_grad_(True)
        # These pretraining-only parameters are absent from the downstream forward graph.
        backbone.heads.requires_grad_(False)
        backbone.mask_embedding.requires_grad_(False)
        self.lora_layers = []
        backbone.encoder.layerdrop = 0.0
        if checkpoint_layers:
            backbone.encoder.layers = nn.ModuleList([CheckpointLayer(x) for x in backbone.encoder.layers])
        self.head = nn.Sequential(nn.BatchNorm1d(backbone.cfg.encoder_embed_dim),
                                  nn.Linear(backbone.cfg.encoder_embed_dim, classes))


class WLASLPoseDataset(torch.utils.data.Dataset):
    def __init__(self, root, split):
        root = Path(root)
        self.metadata = json.loads((root / split / "meta.json").read_text())
        self.names, self.labels = self.metadata["names"], self.metadata["labels"]
        self.offsets = np.load(root / split / "offsets.npy", mmap_mode="r")
        self.features = {s: np.load(root / split / (s + ".npy"), mmap_mode="r")
                         for s in ("body", "left_hand", "right_hand")}

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        begin, end = map(int, self.offsets[index:index + 2])
        return {"id": self.names[index], "label": self.labels[index],
                "source": {s: torch.from_numpy(np.array(x[begin:end], copy=True))
                           for s, x in self.features.items()}}


def collate_clips(rows):
    lengths = [len(x["source"]["body"]) for x in rows]
    frames = max(lengths)
    padding = torch.arange(frames)[None] >= torch.tensor(lengths)[:, None]
    source = {}
    for s in rows[0]["source"]:
        array = torch.zeros(len(rows), frames, rows[0]["source"][s].shape[-1])
        for i, row in enumerate(rows):
            array[i, :lengths[i]] = row["source"][s]
        source[s] = array
    return source, padding, torch.tensor([r["label"] for r in rows]), [r["id"] for r in rows]


def metrics_from_counts(counts):
    counts = counts.cpu().double()
    present = counts[:, 0] > 0
    total = counts[:, 0].sum().item()
    if total < 1:
        raise ValueError("empty evaluation split")
    result = {"samples": int(total), "classes_present": int(present.sum())}
    for column, k in enumerate((1, 5, 10), 1):
        result[f"pi_top{k}"] = 100 * counts[:, column].sum().item() / total
        result[f"pc_top{k}"] = 100 * (counts[present, column] / counts[present, 0]).mean().item()
    return result


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, value in model.backbone.named_parameters():
        if not value.requires_grad:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
