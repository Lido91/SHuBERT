"""Snapshot the completed OpenASL run's selected full model for How2Sign SFT."""

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shubert_pose.sft_handoff import validate_source  # noqa: E402


def prepare(args):
    import torch
    from shubert_pose.wlasl import atomic_json, sha256
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    completed_path = source / 'train/completed.json'
    if not completed_path.exists():
        raise RuntimeError('OpenASL completion record is missing; refusing premature How2Sign SFT')
    completed = json.loads(completed_path.read_text())
    checkpoint_path = source / 'train/checkpoint_best.pth'
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu', mmap=True, weights_only=False)
    provenance = validate_source(checkpoint, completed, source / 'data')
    del checkpoint
    output.mkdir(parents=True, exist_ok=True)
    initialization = output / 'initialization'
    initialization.mkdir(exist_ok=True)
    target = initialization / 'source_openasl_best.pth'
    source_sha256 = sha256(checkpoint_path)
    if target.exists():
        if sha256(target) != source_sha256:
            raise RuntimeError('existing How2Sign initialization differs from OpenASL best')
    else:
        temporary = target.with_suffix('.pth.tmp')
        shutil.copyfile(checkpoint_path, temporary)
        if sha256(temporary) != source_sha256:
            raise RuntimeError('OpenASL checkpoint snapshot checksum failed')
        temporary.replace(target)
    provenance.update(source_checkpoint=str(checkpoint_path), snapshot=str(target),
                      sha256=source_sha256, completed_sha256=sha256(completed_path))
    handoff_path = output / 'handoff.json'
    if handoff_path.exists() and json.loads(handoff_path.read_text()) != provenance:
        raise RuntimeError('How2Sign handoff provenance changed')
    atomic_json(handoff_path, provenance)
    config = json.loads(Path(args.config).read_text())
    config.update(init_slt_checkpoint=str(target), init_slt_step=provenance['source_step'])
    atomic_json(output / 'sft_config.json', config)
    print(json.dumps(provenance), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='/scratch/lidowu91/SHuBERT/outputs/openasl_pose_mt5_sft_20261116')
    parser.add_argument('--output', default='/scratch/lidowu91/SHuBERT/outputs/how2sign_pose_mt5_after_openasl_20261116')
    parser.add_argument('--config', default=str(ROOT / 'configs/how2sign_pose_mt5_sft.json'))
    prepare(parser.parse_args())
