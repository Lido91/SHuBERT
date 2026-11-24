import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from omegaconf import MISSING

from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

from .dataset import PoseDataset
from .packed_dataset import PackedPoseDataset


@dataclass
class PosePretrainingConfig(FairseqDataclass):
    data: str = MISSING
    stream_names: List[str] = field(default_factory=lambda: ["body", "left_hand", "right_hand"])
    stream_dims: List[int] = field(default_factory=lambda: [14, 42, 42])
    codebook_size: int = 256
    min_frames: int = 2
    max_frames: int = 5000
    captions: str = ""


@register_task("pose_pretraining", dataclass=PosePretrainingConfig)
class PosePretrainingTask(FairseqTask):
    def load_dataset(self, split, task_cfg=None, **kwargs):
        cfg = task_cfg or self.cfg
        packed = Path(cfg.data) / f"{split}.pack.json"
        dataset_class = PackedPoseDataset if packed.exists() else PoseDataset
        if cfg.captions and not packed.exists():
            raise ValueError("task.captions requires packed pose data")
        extra = {"captions": cfg.captions} if cfg.captions else {}
        self.datasets[split] = dataset_class(
            packed if packed.exists() else Path(cfg.data) / f"{split}.tsv", cfg.stream_names, cfg.stream_dims,
            cfg.codebook_size, cfg.min_frames, cfg.max_frames, shuffle=(split == "train"), **extra,
        )

    def build_model(self, cfg, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)
        if (list(model.cfg.stream_names) != list(self.cfg.stream_names)
                or list(model.cfg.stream_dims) != list(self.cfg.stream_dims)
                or model.cfg.codebook_size != self.cfg.codebook_size):
            raise ValueError("model and task stream names, dimensions, and codebook_size must match")
        if model.semantic is not None and not from_checkpoint:
            if not self.cfg.captions:
                raise ValueError("model.semantic requires task.captions")
            model.semantic.text_encoder()  # Fail before training if the frozen text encoder is unavailable.
        if model.cfg.init_checkpoint and not from_checkpoint:
            model.initialize_from(model.cfg.init_checkpoint)
        return model

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None

    def max_positions(self):
        return sys.maxsize, sys.maxsize
