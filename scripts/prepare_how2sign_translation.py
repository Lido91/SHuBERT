"""Prepare How2Sign train/test pairs for coordinate-only translation SFT."""

import argparse
import fcntl
import gzip
import json
from multiprocessing import Pool
from pathlib import Path
import pickle

import numpy as np

from prepare_openasl_translation import STREAM_DIMS, atomic_json, convert_record
from prepare_youtubeasl_pack import digest


def read_labels(path):
    with gzip.open(path, 'rb') as handle:
        rows = list(pickle.load(handle).values())
    if len({row['name'] for row in rows}) != len(rows):
        raise ValueError('duplicate How2Sign clip IDs')
    return rows


def write_split(output, split, rows, pose_root, args):
    features = {stream: [] for stream in STREAM_DIMS}
    records, lengths, excluded = [], [], []
    jobs = ((row, pose_root, split != 'train') for row in rows)
    with Pool(args.workers) as pool:
        for index, (row, streams, error) in enumerate(pool.imap(convert_record, jobs, chunksize=16), 1):
            if error:
                excluded.append({'id': row['name'], 'video_path': row['video_path'], 'error': error})
            else:
                records.append({'id': row['name'], 'text': row['text'].strip(), 'pack_index': len(records)})
                lengths.append(len(streams['body']))
                for stream in STREAM_DIMS:
                    features[stream].append(streams[stream])
            if index % 5000 == 0:
                print(json.dumps({'split': split, 'processed': index, 'total': len(rows),
                                  'kept': len(records), 'excluded': len(excluded)}), flush=True)
    atomic_json(output / f'excluded_{split}.json', excluded)
    if (not records or (split != 'train' and excluded)
            or len(excluded) / len(rows) > args.max_excluded_fraction):
        raise RuntimeError(f'unusable How2Sign {split} records; inspect excluded_{split}.json')
    offsets = output / f'{split}.offsets.npy'
    np.save(offsets, np.concatenate(([0], np.cumsum(lengths))))
    paths = {}
    for stream in STREAM_DIMS:
        path = output / f'{split}.{stream}.npy'
        np.save(path, np.concatenate(features.pop(stream)))
        paths[stream] = str(path)
    records_path = output / f'{split}.records.jsonl'
    records_path.write_text(''.join(json.dumps(row) + '\n' for row in records))
    metadata = {'records': str(records_path), 'samples': len(records),
                'offsets': str(offsets), 'features': paths}
    if split == 'validation':
        metadata['evaluation_split'] = 'How2Sign test used as validation for How2Sign SFT'
    atomic_json(output / f'{split}.json', metadata)
    return {'source_samples': len(rows), 'samples': len(records), 'frames': sum(lengths),
            'excluded': len(excluded), 'records_sha256': digest(records_path)}


def prepare(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = {'dataset': 'How2Sign', 'normalization': 'sign69-nose-centered-v1',
                'selection_split': 'How2Sign test', 'independent_test': False,
                'train_labels_sha256': digest(args.train_labels),
                'validation_labels_sha256': digest(args.validation_labels),
                'train_pose_root': str(Path(args.train_poses).resolve()),
                'validation_pose_root': str(Path(args.validation_poses).resolve()),
                'code_sha256': {name: digest(Path(__file__).with_name(name)) for name in
                                ('prepare_how2sign_translation.py', 'prepare_openasl_translation.py',
                                 'prepare_youtubeasl_pack.py')}}
    ready_path = output / 'ready.json'
    if ready_path.exists():
        ready = json.loads(ready_path.read_text())
        if ready['identity'] != identity:
            raise RuntimeError('How2Sign preparation inputs changed')
        print(json.dumps(ready), flush=True)
        return
    train, validation = read_labels(args.train_labels), read_labels(args.validation_labels)
    if {row['name'] for row in train} & {row['name'] for row in validation}:
        raise ValueError('How2Sign train and test IDs overlap')
    summary = {'train': write_split(output, 'train', train, args.train_poses, args),
               'validation': write_split(output, 'validation', validation, args.validation_poses, args)}
    ready = {'identity': identity, 'splits': summary, 'train_samples': summary['train']['samples'],
             'validation_samples': summary['validation']['samples']}
    atomic_json(ready_path, ready)
    print(json.dumps(ready), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-labels', default='/scratch/lidowu91/Uni-Sign-Lido/data/How2Sign/labels.train')
    parser.add_argument('--validation-labels', default='/scratch/lidowu91/Uni-Sign-Lido/data/How2Sign/labels.test')
    parser.add_argument('--train-poses', default='/scratch/lidowu91/Uni-Sign-Lido/data/How2Sign/train-pose-rtmpose-192')
    parser.add_argument('--validation-poses', default='/scratch/lidowu91/Uni-Sign-Lido/data/How2Sign/test-pose-rtmpose-192')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-excluded-fraction', type=float, default=0.05)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'prepare.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prepare(args)
