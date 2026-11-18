"""Stop only this comparison's workers and preserve their resumable states."""

import json
import os
from pathlib import Path
import shutil
import signal
import time


root = Path('/scratch/lidowu91/SHuBERT/outputs/wlasl_pose_e141_143_20261115')
destination = root.with_name(root.name + '_test_as_dev')
destination.mkdir(parents=True, exist_ok=True)
status = json.loads((root / 'progress.json').read_text())
for job in status.get('jobs', []):
    pid = job['pid']
    cmdline = Path(f'/proc/{pid}/cmdline')
    if not cmdline.exists():
        continue
    command = cmdline.read_bytes().replace(b'\0', b' ').decode()
    if ('scripts/eval_wlasl_pose.py' not in command or str(root / f"epoch{job['epoch']}") not in command):
        raise RuntimeError('Refusing to signal a process outside this comparison')
    os.kill(pid, signal.SIGTERM)
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    if all(not Path(f"/proc/{job['pid']}/cmdline").exists()
           or not Path(f"/proc/{job['pid']}/cmdline").read_bytes() for job in status.get('jobs', [])):
        break
    time.sleep(1)
for epoch in (141, 142, 143):
    target = destination / 'warm_start' / f'epoch{epoch}.pth'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / f'epoch{epoch}' / 'last.pth', target)
code = root / 'executed_source'
code.mkdir(exist_ok=True)
repository = Path(__file__).resolve().parents[1]
for relative in ('scripts/eval_wlasl_pose.py', 'shubert_pose/wlasl.py', 'scripts/run_wlasl_pose_sweep.py'):
    shutil.copyfile(repository / relative, code / Path(relative).name)
(destination / 'transition.json').write_text(json.dumps({
    'reason': 'User explicitly requested using the test set as the validation/selection set',
    'previous_run': str(root), 'retained_training': True, 'selection_split': 'test',
    'score_role': 'test_as_dev', 'independent_test': False, 'switched_unix': time.time(),
    'previous_selection_history': 'dev-only; reset for this new test-as-dev selection phase'
}, indent=2) + '\n')
print('SWITCH_READY ' + str(destination), flush=True)
