"""Verify a running WLASL comparison automatically when all three runs finish."""

import argparse
import json
from pathlib import Path
import time

from verify_wlasl_comparison import verify


def watch(root):
    while True:
        progress = root / 'progress.json'
        if not progress.exists():
            time.sleep(20)
            continue
        status = json.loads(progress.read_text())
        if status['phase'] == 'complete':
            verify(root)
            return
        if status['phase'] == 'failed':
            raise RuntimeError(f'comparison failed; inspect {root}')
        time.sleep(20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    watch(parser.parse_args().root)
