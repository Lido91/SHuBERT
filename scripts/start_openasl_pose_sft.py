"""Detach OpenASL SFT inside an existing four-GPU allocation."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from supervise_pose_sft import OUTPUTS, STAGES


def main(args):
    root = OUTPUTS / STAGES[args.stage][0]
    root.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).with_name('supervise_pose_sft.py')),
               '--gpu-job', args.gpu_job, '--stage', args.stage]
    with (root / 'launcher.log').open('a') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    value = {'pid': process.pid, 'gpu_job': args.gpu_job, 'stage': args.stage, 'command': command,
             'started_unix': time.time(), 'scope': 'training step inside an independent GPU reservation'}
    (root / 'launch.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-job', required=True)
    parser.add_argument('--stage', choices=STAGES, default='openasl')
    main(parser.parse_args())
