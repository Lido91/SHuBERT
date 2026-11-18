"""Check one real train/test clip against the coordinate translation contract."""

import gzip
import json
from pathlib import Path
import pickle

import numpy as np

from prepare_openasl_translation import convert_record


def main():
    root = Path('/scratch/lidowu91/Uni-Sign-Lido/data/How2Sign')
    for split in ('train', 'test'):
        with gzip.open(root / f'labels.{split}', 'rb') as handle:
            row = next(iter(pickle.load(handle).values()))
        _, streams, error = convert_record((row, str(root / f'{split}-pose-rtmpose-192'), split == 'test'))
        if error:
            raise RuntimeError(error)
        if (not all(np.isfinite(value).all() for value in streams.values())
                or [streams[s].shape[1] for s in ('body', 'left_hand', 'right_hand')] != [14, 42, 42]):
            raise RuntimeError('How2Sign coordinate contract failed')
        print(json.dumps({'split': split, 'id': row['name'],
                          'stream_shapes': {s: list(value.shape) for s, value in streams.items()},
                          'finite': True}), flush=True)


if __name__ == '__main__':
    main()
