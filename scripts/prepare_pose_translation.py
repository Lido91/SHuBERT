"""Reuse all available YouTubeASL pairs and prepare OpenASL test exclusively as validation."""

import argparse
import gzip
import json
from pathlib import Path
import pickle

import numpy as np

from prepare_youtubeasl_pack import convert_sign69, digest

INDICES = ([0] + list(range(3, 11)) + list(range(91, 133))
           + list(range(23, 40, 2)) + list(range(83, 91)) + [53])
STREAM_DIMS = {'body': 14, 'left_hand': 42, 'right_hand': 42}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def prepare(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    coordinates, pack = Path(args.coordinates), Path(args.pose_pack)
    identity = {'manifest': str(Path(args.manifest).resolve()), 'manifest_sha256': digest(args.manifest),
                'coordinate_preparation_sha256': digest(coordinates / 'preparation.json'),
                'source_pose_metadata_sha256': digest(pack / 'meta.json'),
                'eval_labels_sha256': digest(args.eval_labels), 'eval_pose_root': str(Path(args.eval_poses).resolve()),
                'normalization': 'sign69-nose-centered-v1', 'validation': 'OpenASL test',
                'selection_metric': 'bleu4', 'independent_test': False}
    if (output / 'ready.json').exists():
        saved = json.loads((output / 'ready.json').read_text())
        if saved['identity'] != identity:
            raise ValueError('translation data settings changed')
        print(json.dumps(saved), flush=True)
        return
    meta = json.loads((pack / 'meta.json').read_text())
    done = np.load(coordinates / 'converted.npy', mmap_mode='r')
    offsets = np.load(coordinates / 'offsets.npy', mmap_mode='r')
    if not np.array_equal(offsets, np.load(pack / 'offsets.npy', mmap_mode='r')):
        raise ValueError('source pair indices and coordinate offsets do not align')
    seen, kept, excluded = set(), 0, 0
    train_records = output / 'train.records.jsonl'
    temporary = train_records.with_suffix('.jsonl.tmp')
    with open(args.manifest) as source, temporary.open('w') as target:
        for line in source:
            row = json.loads(line)
            index = row['pack_index']
            if meta['names'][index] != row['name'] or row['name'] in seen:
                raise ValueError('duplicate or misaligned YouTubeASL pair')
            seen.add(row['name'])
            if done[index] != 1 or not row['text'].strip():
                excluded += 1
                continue
            target.write(json.dumps({'id': row['name'], 'text': row['text'].strip(), 'pack_index': index}) + '\n')
            kept += 1
    temporary.replace(train_records)
    write_json(output / 'train.json', {'records': str(train_records), 'samples': kept,
               'offsets': str((coordinates / 'offsets.npy').resolve()),
               'features': {s: str((coordinates / f'{s}.npy').resolve()) for s in STREAM_DIMS}})
    with gzip.open(args.eval_labels, 'rb') as handle:
        labels = list(pickle.load(handle).values())
    raw, records, lengths, zero_pose = [], [], [], []
    for index, row in enumerate(labels):
        path = Path(args.eval_poses) / row['video_path'].replace('.mp4', '.pkl')
        with path.open('rb') as handle:
            pose = pickle.load(handle)
        xy, score = np.asarray(pose['keypoints']), np.asarray(pose['scores'])
        if xy.ndim == 4:
            xy, score = xy[:, 0], score[:, 0]
        begin, end = int(pose.get('start', 0)), int(pose.get('end', len(score)))
        selected = np.concatenate((xy[begin:end, INDICES], score[begin:end, INDICES, None]), axis=-1)
        streams = convert_sign69(selected)
        if len(selected) < 1 or not row['text'].strip():
            raise ValueError('unusable OpenASL validation clip: ' + row['name'])
        if not np.any(streams['body']):
            zero_pose.append(row['name'])
        raw.append(streams)
        lengths.append(len(selected))
        records.append({'id': row['name'], 'text': row['text'].strip(), 'pack_index': index})
    validation = output / 'validation'
    validation.mkdir(exist_ok=True)
    np.save(validation / 'offsets.npy', np.concatenate(([0], np.cumsum(lengths))))
    for stream in STREAM_DIMS:
        np.save(validation / f'{stream}.npy', np.concatenate([row[stream] for row in raw]))
    validation_records = output / 'validation.records.jsonl'
    validation_records.write_text(''.join(json.dumps(row) + '\n' for row in records))
    write_json(output / 'validation.json', {'records': str(validation_records), 'samples': len(records),
               'offsets': str(validation / 'offsets.npy'),
               'features': {s: str(validation / f'{s}.npy') for s in STREAM_DIMS}})
    ready = {'identity': identity, 'train_samples': kept, 'excluded_train_pairs': excluded,
             'source_train_pairs': len(seen), 'validation_samples': len(records),
             'validation_zero_pose_clips': zero_pose,
             'records_sha256': {s: digest(output / f'{s}.records.jsonl') for s in ('train', 'validation')}}
    write_json(output / 'ready.json', ready)
    print(json.dumps(ready), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--manifest', default='/scratch/lidowu91/Uni-Sign-Lido/data/YouTubeASL_slim_v2/train.jsonl')
    parser.add_argument('--pose-pack', default='/scratch/lidowu91/SLiM/data/youtube_asl_rtmpose_all/train')
    parser.add_argument('--coordinates', default='/scratch/lidowu91/SHuBERT/outputs/youtubeasl_pose_full_e150_20261114/data')
    parser.add_argument('--eval-labels', default='/scratch/lidowu91/Uni-Sign-Lido/data/OpenASL/labels.test')
    parser.add_argument('--eval-poses', default='/scratch/lidowu91/Uni-Sign-Lido/data/OpenASL/pose-rtmpose-192')
    prepare(parser.parse_args())
