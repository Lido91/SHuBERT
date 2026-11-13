"""Coordinate projections, masked temporal context, and one cluster head per stream."""

from dataclasses import dataclass, field
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2 import TransformerEncoder, Wav2Vec2Config


@dataclass
class SHubertPoseConfig(Wav2Vec2Config):
    stream_names: List[str] = field(default_factory=lambda: ["body", "left_hand", "right_hand"])
    stream_dims: List[int] = field(default_factory=lambda: [14, 42, 42])
    stream_embed_dim: int = 256
    codebook_size: int = 256
    mask_strategy: str = "random"
    mask_prob: float = 0.8
    mask_length: int = 3


@register_model("shubert_pose", dataclass=SHubertPoseConfig)
class SHubertPoseModel(BaseFairseqModel):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.streams = list(cfg.stream_names)
        if (not self.streams or len(set(self.streams)) != len(self.streams)
                or any(not s or "." in s for s in self.streams)
                or len(self.streams) != len(cfg.stream_dims) or any(d < 1 for d in cfg.stream_dims)):
            raise ValueError("require unique stream names and one positive dimension per stream")
        if cfg.codebook_size < 2 or cfg.stream_embed_dim < 1:
            raise ValueError("require codebook_size >= 2 and stream_embed_dim >= 1")
        if not 0 < cfg.mask_prob < 1 or cfg.mask_length < 1 or cfg.mask_strategy not in {"random", "time", "flat"}:
            raise ValueError("require 0 < mask_prob < 1, mask_length >= 1, and random/time/flat masking")
        self.input_norms = nn.ModuleDict({s: nn.LayerNorm(d) for s, d in zip(self.streams, cfg.stream_dims)})
        self.projections = nn.ModuleDict({s: nn.Linear(d, cfg.stream_embed_dim)
                                          for s, d in zip(self.streams, cfg.stream_dims)})
        self.mask_embedding = nn.Parameter(torch.empty(len(self.streams), cfg.stream_embed_dim).uniform_())
        self.post_extract_proj = nn.Linear(len(self.streams) * cfg.stream_embed_dim, cfg.encoder_embed_dim)
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.encoder = TransformerEncoder(cfg)
        self.heads = nn.ModuleDict({s: nn.Linear(cfg.encoder_embed_dim, cfg.codebook_size) for s in self.streams})

    @classmethod
    def build_model(cls, cfg, task=None):
        return cls(cfg)

    def _span_mask(self, size, device):
        """Mask spans over `size` consecutive positions, always leaving one observed when size > 1."""
        span = min(self.cfg.mask_length, max(1, size - 1))
        count = max(1, int(self.cfg.mask_prob * size / span + torch.rand(()).item()))
        starts = torch.randperm(size - span + 1, device=device)[:count]
        mask = torch.zeros(size, dtype=torch.bool, device=device)
        mask[(starts[:, None] + torch.arange(span, device=device)).flatten()] = True
        if size > 1 and mask.all():
            mask[torch.randint(size, (), device=device)] = False
        return mask

    def _sample_mask(self, padding):
        batch, frames = padding.shape
        streams = len(self.streams)
        mask = torch.zeros(batch, frames, streams, dtype=torch.bool, device=padding.device)
        for i in range(batch):
            length = int((~padding[i]).sum())
            if length < 1:
                raise ValueError("each sample must contain at least one valid frame")
            if self.cfg.mask_strategy == "flat":
                # Original SHuBERT: spans over the frame-major [T * streams] sequence, so a span
                # mostly covers neighbouring streams of one frame rather than neighbouring frames.
                mask[i, :length] = self._span_mask(length * streams, padding.device).view(length, streams)
            elif self.cfg.mask_strategy == "time":
                mask[i, :length] = self._span_mask(length, padding.device)[:, None]
            else:
                for channel in range(streams):
                    mask[i, :length, channel] = self._span_mask(length, padding.device)
        return mask

    def forward(self, source, padding_mask=None, labels=None, mask=True, features_only=False, mask_indices=None, layer=None):
        if set(source) != set(self.streams):
            raise ValueError(f"expected coordinate streams {self.streams}, got {list(source)}")
        first = source[self.streams[0]]
        if first.dim() != 3:
            raise ValueError("coordinate inputs must have shape [B, T, D]")
        batch, frames = first.shape[:2]
        for stream, dimension in zip(self.streams, self.cfg.stream_dims):
            if source[stream].shape != (batch, frames, dimension):
                raise ValueError(f"{stream}: expected shape {(batch, frames, dimension)}")
        if padding_mask is None:
            padding_mask = torch.zeros(batch, frames, dtype=torch.bool, device=first.device)
        if padding_mask.shape != (batch, frames) or padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be boolean with shape [B, T]")
        if padding_mask[:, :-1].logical_and(~padding_mask[:, 1:]).any():
            raise ValueError("only right padding is supported")
        projected = torch.stack([self.projections[s](self.input_norms[s](source[s])) for s in self.streams], dim=2)
        selected = torch.zeros(batch, frames, len(self.streams), dtype=torch.bool, device=first.device)
        if mask:
            selected = self._sample_mask(padding_mask) if mask_indices is None else mask_indices
            if selected.shape != (batch, frames, len(self.streams)) or selected.dtype != torch.bool:
                raise ValueError("mask_indices must be boolean with shape [B, T, streams]")
            selected = selected & ~padding_mask.unsqueeze(-1)
            projected = torch.where(selected.unsqueeze(-1), self.mask_embedding.to(projected.dtype), projected)
        x = self.dropout_input(self.post_extract_proj(projected.flatten(2)))
        x, layers = self.encoder(x, padding_mask=padding_mask, layer=layer)
        if features_only:
            return {"x": x, "padding_mask": padding_mask, "layer_results": layers}
        if not mask or not selected.any():
            raise ValueError("cluster-prediction training requires at least one masked valid position")
        if labels is None or set(labels) != set(self.streams):
            raise ValueError("training requires one cluster-label tensor per coordinate stream")
        losses = []
        for i, stream in enumerate(self.streams):
            if labels[stream].shape != (batch, frames) or labels[stream].dtype != torch.long:
                raise ValueError(f"{stream}: cluster labels must be int64 with shape [B, T]")
            chosen = selected[:, :, i]
            losses.append(F.cross_entropy(self.heads[stream](x[chosen]).float(), labels[stream][chosen], reduction="none"))
        losses = torch.cat(losses)
        return {"losses": {"kmeans_loss": losses}, "sample_size": losses.numel(), "masked_positions": selected}

    def extract_features(self, source, padding_mask=None, mask=False, layer=None):
        return self.forward(source, padding_mask=padding_mask, mask=mask, features_only=True, layer=layer)
