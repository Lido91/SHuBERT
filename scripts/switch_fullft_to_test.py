"""Preserve the active full-FT states and restart selection using only the test split."""

import json
import os
from pathlib import Path
import shutil
import signal
import time


def main():
    source = Path('/scratch/lidowu91/SHuBERT/outputs/wlasl_pose_fullft_e141_143_20261115_test_as_dev')
    destination = source.with_name('wlasl_pose_fullft_e141_143_20261115_test_only_selection')
    destination.mkdir(parents=True, exist_ok=True)
    status = json.loads((source / 'progress.json').read_text())
    jobs = status.get('jobs', [])
    for job in jobs:
        path = Path(f"/proc/{job['pid']}/cmdline")
        if not path.exists():
            continue
        command = path.read_bytes().replace(b'\0', b' ').decode()
        expected = str(source / f"epoch{job['epoch']}")
        if 'scripts/eval_wlasl_pose.py' not in command or expected not in command:
            raise RuntimeError('refusing to stop a process outside this full-FT comparison')
        os.kill(job['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if all(not Path(f"/proc/{job['pid']}/cmdline").exists()
               or not Path(f"/proc/{job['pid']}/cmdline").read_bytes() for job in jobs):
            break
        time.sleep(0.5)
    else:
        raise RuntimeError('full-FT workers have not stopped; refusing to copy active state')
    copied = []
    warm = destination / 'warm_start'
    warm.mkdir(exist_ok=True)
    for epoch in (141, 142, 143):
        folder = source / f'epoch{epoch}'
        config = json.loads((folder / 'config.json').read_text())
        if config['finetune_mode'] != 'full' or config['lora_parameters'] != 0:
            raise ValueError('expected a full-fine-tuning state, not LoRA adapters')
        target = warm / f'epoch{epoch}.pth'
        temporary = target.with_suffix('.pth.tmp')
        shutil.copyfile(folder / 'last.pth', temporary)
        temporary.replace(target)
        rows = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
        copied.append({'pretrain_epoch': epoch, 'last_completed_epoch': rows[-1]['epoch'],
                       'state': str(target)})
    transition = {'reason': 'User requires test-as-dev selection without a dev-selection stage',
                  'source': str(source), 'retained_training': True, 'selection_split': 'test',
                  'selection_start_epoch': 1, 'independent_test': False,
                  'prior_dev_scores_excluded': True, 'checkpoints': copied, 'updated_unix': time.time()}
    (destination / 'transition.json').write_text(json.dumps(transition, indent=2) + '\n')
    print(json.dumps(transition), flush=True)


if __name__ == '__main__':
    main()
