"""Print the current three-checkpoint WLASL comparison without importing ML libraries."""

import json
from pathlib import Path
import time


root = Path('/scratch/lidowu91/SHuBERT/outputs/wlasl_pose_e141_143_20261115_test_as_dev')
for epoch in (141, 142, 143):
    output = root / f'epoch{epoch}'
    progress = output / 'progress.json'
    row = {'checkpoint_epoch': epoch}
    if progress.exists():
        current = json.loads(progress.read_text())
        for key in ('phase', 'epoch', 'batch', 'batches', 'steps', 'train_loss'):
            if key in current:
                row[key] = current[key]
        row['seconds_since_update'] = round(time.time() - current['updated_unix'], 1)
    metrics = output / 'metrics.jsonl'
    if metrics.exists():
        records = [json.loads(line) for line in metrics.read_text().splitlines()]
        if records:
            last = records[-1]
            role = 'test_as_dev' if 'test_as_dev' in last else 'dev'
            values = last[role]
            row.update({'completed_epoch': last['epoch'], role + '_pi': round(values['pi_top1'], 3),
                        role + '_pc': round(values['pc_top1'], 3),
                        'best_pi': round(last['best_selection_pi_top1'], 3),
                        'bad_epochs': last['bad_epochs'], 'seconds_per_epoch': round(last['seconds'], 2)})
    result = output / 'test_as_dev_metrics.json'
    if result.exists():
        row['test_result'] = json.loads(result.read_text())
    print(json.dumps(row), flush=True)
print('SWEEP', (root / 'progress.json').read_text())
