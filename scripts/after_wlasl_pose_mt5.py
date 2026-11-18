"""Persistently wait for verified WLASL completion, then start four-GPU translation training."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time


def write(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'handoff.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = Path(args.wlasl_root)
    while True:
        progress = json.loads((source / 'progress.json').read_text())
        if progress['phase'] == 'failed':
            write(output / 'handoff.json', {'phase': 'blocked_by_wlasl_failure', 'updated_unix': time.time()})
            raise RuntimeError('WLASL comparison failed; translation has not started')
        verified = source / 'verification.json'
        if progress['phase'] == 'complete' and verified.exists():
            if json.loads(verified.read_text())['status'] == 'verified':
                break
        write(output / 'handoff.json', {'phase': 'waiting_for_wlasl', 'updated_unix': time.time()})
        time.sleep(20)
    write(output / 'handoff.json', {'phase': 'launching_translation', 'updated_unix': time.time()})
    with (output / 'launcher.log').open('a') as log:
        result = subprocess.run(['srun', f'--jobid={args.gpu_job}', '--overlap', '--nodes=1', '--ntasks=1',
                                 '--cpus-per-task=16', '--mem=96G', '--gres=gpu:4',
                                 'bash', '/home/lidowu91/Workspace/SHuBERT/scripts/run_pose_mt5.sh'],
                                stdout=log, stderr=subprocess.STDOUT)
    complete = output / 'train/completed.json'
    write(output / 'handoff.json', {'phase': 'complete' if complete.exists() else 'paused_or_failed',
                                    'exit_code': result.returncode, 'updated_unix': time.time()})
    if result.returncode:
        raise RuntimeError(f'translation launcher exited with {result.returncode}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-job', required=True)
    parser.add_argument('--wlasl-root', default=(
        '/scratch/lidowu91/SHuBERT/outputs/wlasl_pose_fullft_e141_143_20261115_test_only_selection'))
    parser.add_argument('--output', default='/scratch/lidowu91/SHuBERT/outputs/pose_mt5_youtubeasl_20261115')
    main(parser.parse_args())
