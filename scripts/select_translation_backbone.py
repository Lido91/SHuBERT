"""Snapshot the verified best full-FT SHuBERT backbone for mT5 training."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def select(root, output):
    root, output = Path(root), Path(output)
    if (output / 'initialization.json').exists():
        print((output / 'initialization.json').read_text(), flush=True)
        return
    verification = json.loads((root / 'verification.json').read_text())
    if verification['status'] != 'verified':
        raise ValueError('WLASL full-FT comparison is not verified')
    comparisons = json.loads((root / 'comparison.json').read_text())
    if any(entry['exit_code'] != 0 or entry['result'].get('finetune_mode') != 'full' for entry in comparisons):
        raise ValueError('all three full fine-tuning comparisons must finish first')
    selected = max(comparisons, key=lambda entry: (entry['result']['test_as_dev']['pi_top1'],
                                                  -entry['pretrain_epoch']))
    result = selected['result']
    source_pose = Path(result['checkpoint'])
    source_full = root / f"epoch{selected['pretrain_epoch']}" / 'best.pth'
    output.mkdir(parents=True, exist_ok=True)
    snapshots = {}
    for name, source in (('source_pose_checkpoint', source_pose), ('full_finetune_checkpoint', source_full)):
        target = output / (name + '.pth')
        temporary = target.with_suffix('.pth.tmp')
        shutil.copyfile(source, temporary)
        temporary.replace(target)
        snapshots[name] = str(target.resolve())
        snapshots[name + '_sha256'] = digest(target)
    if snapshots['source_pose_checkpoint_sha256'] != result['checkpoint_sha256']:
        raise ValueError('pose snapshot hash does not match the evaluated source')
    initialization = {**snapshots, 'wlasl_root': str(root.resolve()),
                      'source_pretrain_epoch': selected['pretrain_epoch'],
                      'source_finetune_epoch': result['selected_epoch'],
                      'wlasl_test_as_dev_pi_top1': result['test_as_dev']['pi_top1'],
                      'normalization': 'sign69-nose-centered-v1',
                      'method': 'continue from best WLASL full-FT feature extractor; discard classifier'}
    temporary = output / 'initialization.json.tmp'
    temporary.write_text(json.dumps(initialization, indent=2) + '\n')
    temporary.replace(output / 'initialization.json')
    print(json.dumps(initialization), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    select(args.root, args.output)
