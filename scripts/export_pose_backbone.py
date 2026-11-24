"""Write a semantic-alignment checkpoint without its alignment heads, for the existing downstream tools."""

import argparse
import hashlib
from pathlib import Path
import sys

from omegaconf import OmegaConf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.pose_inference import load_model  # noqa: E402
from shubert_pose.semantic import SEMANTIC_PREFIX  # noqa: E402


def export(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"{output}: choose a new output path")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(checkpoint["cfg"])
    if cfg.model.get("_name") != "shubert_pose" or not cfg.model.get("semantic", False):
        raise ValueError("require a shubert_pose checkpoint trained with model.semantic=true")
    removed = sorted(name for name in checkpoint["model"] if name.startswith(SEMANTIC_PREFIX))
    cfg.model.semantic, cfg.model.text_model, cfg.model.init_checkpoint = False, "", ""
    if "captions" in cfg.task:
        cfg.task.captions = ""
    exported = {"cfg": OmegaConf.to_container(cfg, resolve=True),
                "model": {name: value for name, value in checkpoint["model"].items() if name not in removed},
                "export": {"source": str(source.resolve()),
                           "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                           "removed": removed}}
    temporary = output.with_name(f".{output.name}.tmp")
    torch.save(exported, temporary)
    load_model(temporary, torch.device("cpu"))  # The remaining weights must load strictly.
    temporary.replace(output)
    print(f"Removed {len(removed)} alignment tensors; wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.output)
