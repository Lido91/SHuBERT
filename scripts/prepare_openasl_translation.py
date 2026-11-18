"""Prepare OpenASL train poses with the exact normalization used by PoseSHuBERT SLT."""

import argparse
import gzip
import json
from multiprocessing import Pool
from pathlib import Path
import pickle

import numpy as np

from prepare_youtubeasl_pack import convert_sign69, digest

INDICES = ([0] + list(range(3, 11)) + list(range(91, 133))
           + list(range(23, 40, 2)) + list(range(83, 91)) + [53])
STREAM_DIMS = {'body': 14, 'left_hand': 42, 'right_hand': 42}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def convert_record(job):
    row, pose_root = job[:2]
    allow_zero_body = len(job) > 2 and job[2]
    path = Path(pose_root) / Path(row['video_path']).with_suffix('.pkl')
    try:
        if not row['text'].strip():
            raise ValueError('empty caption')
        with path.open('rb') as handle:
            pose = pickle.load(handle)
        xy, score = np.asarray(pose['keypoints']), np.asarray(pose['scores'])
        if xy.ndim == 4:
            xy, score = xy[:, 0], score[:, 0]
        begin, end = int(pose.get('start', 0)), int(pose.get('end', len(score)))
        selected = np.concatenate((xy[begin:end, INDICES], score[begin:end, INDICES, None]), axis=-1)
        streams = convert_sign69(selected)
        if not allow_zero_body and not np.any(streams['body']):
            raise ValueError('unusable body anchors')
        return row, streams, None
    except Exception as error:
        return row, None, f'{type(error).__name__}: {error}'


def prepare(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.translation_data).resolve()
    identity = {'train_labels': str(Path(args.train_labels).resolve()),
                'train_labels_sha256': digest(args.train_labels),
                'pose_root': str(Path(args.pose_root).resolve()),
                'validation_metadata_sha256': digest(source / 'validation.json'),
                'validation_records_sha256': digest(source / 'validation.records.jsonl'),
                'normalization': 'sign69-nose-centered-v1', 'selection_split': 'OpenASL test',
                'independent_test': False, 'preparation_sha256': digest(__file__),
                'conversion_sha256': digest(Path(__file__).with_name('prepare_youtubeasl_pack.py'))}
    ready_path = output / 'ready.json'
    if ready_path.exists():
        ready = json.loads(ready_path.read_text())
        if ready['identity'] != identity:
            raise RuntimeError('OpenASL preparation inputs changed')
        print(json.dumps(ready), flush=True)
        return
    with gzip.open(args.train_labels, 'rb') as handle:
        rows = list(pickle.load(handle).values())
    names = [row['name'] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError('duplicate OpenASL training IDs')
    validation = json.loads((source / 'validation.json').read_text())
    validation_records = [json.loads(line) for line in Path(validation['records']).read_text().splitlines()]
    if set(names) & {row['id'] for row in validation_records}:
        raise ValueError('OpenASL train and validation IDs overlap')
    features = {stream: [] for stream in STREAM_DIMS}
    records, lengths, excluded = [], [], []
    jobs = ((row, args.pose_root) for row in rows)
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
                print(json.dumps({'phase': 'convert', 'processed': index, 'total': len(rows),
                                  'kept': len(records), 'excluded': len(excluded)}), flush=True)
    atomic_json(output / 'excluded_train.json', excluded)
    if not records or len(excluded) / len(rows) > args.max_excluded_fraction:
        raise RuntimeError('too many unusable OpenASL training clips; inspect excluded_train.json')
    offsets = output / 'train.offsets.npy'
    np.save(offsets, np.concatenate(([0], np.cumsum(lengths))))
    paths = {}
    for stream in STREAM_DIMS:
        path = output / f'train.{stream}.npy'
        np.save(path, np.concatenate(features.pop(stream)))
        paths[stream] = str(path)
    records_path = output / 'train.records.jsonl'
    records_path.write_text(''.join(json.dumps(row) + '\n' for row in records))
    atomic_json(output / 'train.json', {'records': str(records_path), 'samples': len(records),
                                      'offsets': str(offsets), 'features': paths})
    validation.update(evaluation_split='OpenASL test used as validation for OpenASL SFT')
    atomic_json(output / 'validation.json', validation)
    ready = {'identity': identity, 'source_train_samples': len(rows), 'train_samples': len(records),
             'excluded_train_samples': len(excluded), 'train_frames': sum(lengths),
             'validation_samples': len(validation_records),
             'records_sha256': digest(records_path)}
    atomic_json(ready_path, ready)
    print(json.dumps(ready), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-labels', default='/scratch/lidowu91/Uni-Sign-Lido/data/OpenASL/labels.train')
    parser.add_argument('--pose-root', default='/scratch/lidowu91/Uni-Sign-Lido/data/OpenASL/pose-rtmpose-192')
    parser.add_argument('--translation-data', default=(
        '/scratch/lidowu91/SHuBERT/outputs/pose_mt5_youtubeasl_20261115/data'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-excluded-fraction', type=float, default=0.05)
    prepare(parser.parse_args())
