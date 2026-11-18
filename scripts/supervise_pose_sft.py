"""Supervise an SFT step without tying its lifetime to the GPU reservation."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = Path('/scratch/lidowu91/SHuBERT/outputs')
STAGES = {
    'openasl': ('openasl_pose_mt5_sft_20261116', 'run_openasl_pose_mt5_sft.sh'),
    'how2sign': ('how2sign_pose_mt5_after_openasl_20261116', 'run_how2sign_pose_mt5_sft.sh'),
    'how2sign_continue': ('how2sign_pose_mt5_continue_best1518_20261117', 'run_how2sign_pose_mt5_continue.sh'),
    'how2sign_grpo': ('how2sign_pose_mt5_grpo_bleu_consensus_v2_20261117', 'run_how2sign_pose_grpo.sh'),
    'pose_face': ('youtubeasl_pose_face18_e150_20261117',
                  '/scratch/lidowu91/SHuBERT/outputs/youtubeasl_pose_face18_e150_20261117/run_pretraining.sh'),
}


def write_status(path, **values):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(dict(values, updated_unix=time.time()), indent=2) + '\n')
    temporary.replace(path)
    if values.get('state') == 'failed':
        progress = path.parent / 'train/progress.json'
        if progress.exists():
            training = json.loads(progress.read_text())
            if training.get('state') not in ('complete', 'paused'):
                training.update(state='failed', last_train_progress_unix=training.get('updated_unix'),
                                supervisor_attempt=values.get('attempt'), exit_code=values.get('exit_code'),
                                updated_unix=time.time())
                temporary = progress.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(training, indent=2) + '\n')
                temporary.replace(progress)


def main(args):
    name, runner = STAGES[args.stage]
    output = OUTPUTS / name
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        supervise(args, output, runner)


def supervise(args, output, runner):
    status = output / 'supervisor.json'
    target = output.parent / STAGES['how2sign'][0] if args.stage == 'openasl' else output
    for attempt in range(1, args.retries + 2):
        state = subprocess.run(['squeue', '-h', '-j', args.gpu_job, '-o', '%T'],
                               capture_output=True, text=True, check=True).stdout.strip()
        if state != 'RUNNING':
            write_status(status, state='reservation_not_running', gpu_job=args.gpu_job,
                         stage=args.stage, attempt=attempt, reservation_state=state)
            return
        command = ['srun', f'--jobid={args.gpu_job}', '--overlap', '--nodes=1', '--ntasks=1',
                   '--cpus-per-task=16', '--mem=96G', '--gres=gpu:4', 'bash', str(ROOT / 'scripts' / runner)]
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL)
        started = time.time()
        write_status(status, state='running', gpu_job=args.gpu_job, stage=args.stage,
                     attempt=attempt, step_launcher_pid=process.pid, command=command)
        stalled = False
        completed_at = None
        while process.poll() is None:
            # Completion is written before distributed/DataLoader teardown. A
            # completed progress record is intentionally absent from the running
            # candidates below, so handle it before evaluating startup timeouts.
            if (target / 'train/completed.json').exists():
                if completed_at is None:
                    completed_at = time.time()
                if time.time() - completed_at <= 120:
                    time.sleep(1)
                    continue
                print('Training completed; stopping a launcher stuck in exit cleanup.', flush=True)
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            candidates = [output / 'train/progress.json', target / 'train/progress.json']
            progress = [p for p in candidates if p.exists() and p.stat().st_mtime >= started
                        and json.loads(p.read_text()).get('state') == 'running']
            latest = max([started] + [p.stat().st_mtime for p in progress])
            # Allow initial preparation/model loading; thereafter detect missing progress.
            timeout = 45 * 60 if not progress else 20 * 60
            if time.time() - latest > timeout:
                stalled = True
                print('SFT progress stalled; terminating its step launcher for recovery.', flush=True)
                process.terminate()
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            time.sleep(10)
        code = process.returncode
        if (target / 'train/completed.json').exists():
            write_status(status, state='complete', gpu_job=args.gpu_job, stage=args.stage,
                         attempt=attempt, exit_code=code)
            return
        paused = False
        for path in (output / 'train/progress.json', target / 'train/progress.json'):
            if path.exists() and json.loads(path.read_text()).get('state') == 'paused':
                paused = True
        if paused or (output / 'train/STOP').exists() or (target / 'train/STOP').exists():
            write_status(status, state='paused', gpu_job=args.gpu_job, stage=args.stage,
                         attempt=attempt, exit_code=code)
            return
        write_status(status, state='retrying' if attempt <= args.retries else 'failed',
                     gpu_job=args.gpu_job, stage=args.stage, attempt=attempt,
                     exit_code=code, stalled=stalled)
        if attempt <= args.retries:
            print(f'SFT attempt {attempt} exited with {code}; retrying from saved state.', flush=True)
            time.sleep(30 * attempt)
    raise SystemExit('SFT failed after bounded retries; see supervisor.json and train.log')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-job', required=True)
    parser.add_argument('--stage', choices=STAGES, default='openasl')
    parser.add_argument('--retries', type=int, default=3)
    main(parser.parse_args())
