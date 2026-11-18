"""Snapshot an explicitly selected How2Sign model for a fresh SFT schedule."""

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_checkpoint(checkpoint, source_data, expected_step, expected_bleu4):
    config = checkpoint['identity']['config']
    if config.get('dataset') != 'How2Sign' or Path(config['data']).resolve() != source_data:
        raise ValueError('continuation checkpoint did not originate from the requested How2Sign run')
    score = checkpoint['validation']['bleu4']
    if checkpoint['step'] != expected_step or not math.isfinite(score) or abs(score - expected_bleu4) > 1e-7:
        raise ValueError('continuation checkpoint differs from the explicitly selected step/BLEU-4')
    keys = checkpoint['model']
    if (not {'projection.weight', 'projection.bias'} <= set(keys)
            or not any(key.startswith('encoder.') for key in keys)
            or not any(key.startswith('mt5_model.') for key in keys)):
        raise ValueError('continuation requires the complete encoder, projection, and mT5 model')
    return {'source_dataset': 'How2Sign', 'target_dataset': 'How2Sign',
            'source_step': checkpoint['step'], 'source_epoch_index': checkpoint.get('epoch'),
            'selected_bleu4': score, 'selection_split': checkpoint['validation']['split'],
            'independent_test': False, 'complete_full_model': True,
            'optimizer_reset': True, 'scheduler_reset': True}


def prepare(args):
    import torch
    from shubert_pose.wlasl import atomic_json, sha256

    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    source_data = source / 'data'
    config = json.loads(Path(args.config).read_text())
    if config.get('dataset') != 'How2Sign' or Path(config['data']).resolve() != source_data:
        raise ValueError('continuation must use the same How2Sign data')
    initialization = output / 'initialization'
    initialization.mkdir(parents=True, exist_ok=True)
    source_checkpoint = source / 'train/checkpoint_best.pth'
    target = initialization / f'source_how2sign_best_step{args.expected_step}.pth'
    candidate = target if target.exists() else target.with_suffix('.pth.tmp')
    if not target.exists():
        shutil.copyfile(source_checkpoint, candidate)
    # Inspect the copied file itself, so an atomic source checkpoint replacement
    # cannot change the model selected for this continuation.
    checkpoint = torch.load(str(candidate), map_location='cpu', mmap=True, weights_only=False)
    provenance = validate_checkpoint(checkpoint, source_data, args.expected_step, args.expected_bleu4)
    del checkpoint
    checksum = sha256(candidate)
    if candidate != target:
        candidate.replace(target)
    provenance.update(source_checkpoint=str(source_checkpoint), snapshot=str(target), sha256=checksum)
    handoff = output / 'handoff.json'
    if handoff.exists() and json.loads(handoff.read_text()) != provenance:
        raise RuntimeError('existing continuation provenance changed')
    config.update(init_slt_checkpoint=str(target), init_slt_step=args.expected_step)
    resolved = output / 'sft_config.json'
    if resolved.exists() and json.loads(resolved.read_text()) != config:
        if any((output / mode / 'run_config.json').exists() for mode in ('smoke', 'train')):
            raise RuntimeError('existing continuation configuration changed after model startup')
    atomic_json(handoff, provenance)
    atomic_json(resolved, config)
    print(json.dumps(provenance), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='/scratch/lidowu91/SHuBERT/outputs/how2sign_pose_mt5_after_openasl_20261116')
    parser.add_argument('--output', default='/scratch/lidowu91/SHuBERT/outputs/how2sign_pose_mt5_continue_best1518_20261117')
    parser.add_argument('--config', default=str(ROOT / 'configs/how2sign_pose_mt5_continue.json'))
    parser.add_argument('--expected-step', type=int, default=3876)
    parser.add_argument('--expected-bleu4', type=float, default=15.175059951878245)
    prepare(parser.parse_args())
