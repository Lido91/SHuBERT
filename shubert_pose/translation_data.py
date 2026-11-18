"""Paired captions and memory-mapped SHuBERT coordinate streams."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

STREAMS = ('body', 'left_hand', 'right_hand')


def frame_indices(length, maximum, training, seed):
    if length < 1:
        raise ValueError('empty pose interval')
    if length <= maximum:
        return np.arange(length)
    if not training:
        return np.linspace(0, length - 1, maximum, dtype=np.int64)
    edges = np.linspace(0, length, maximum + 1, dtype=np.int64)
    rng = np.random.RandomState(seed % (2**32))
    return np.array([rng.randint(begin, end) for begin, end in zip(edges[:-1], edges[1:])])


class TranslationDataset(Dataset):
    def __init__(self, root, split, max_frames=256, seed=42, limit=0):
        self.root, self.split = Path(root), split
        self.metadata = json.loads((self.root / f'{split}.json').read_text())
        with Path(self.metadata['records']).open() as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if limit:
            self.records = self.records[:limit]
        if not self.records or max_frames < 1:
            raise ValueError('require paired records and a positive frame limit')
        self.max_frames, self.seed, self.epoch = max_frames, seed, 0
        self._open_arrays()

    def _open_arrays(self):
        self.offsets = np.load(self.metadata['offsets'], mmap_mode='r')
        self.features = {s: np.load(self.metadata['features'][s], mmap_mode='r') for s in STREAMS}

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        clip = row['pack_index']
        begin, end = map(int, self.offsets[clip:clip + 2])
        chosen = frame_indices(end - begin, self.max_frames, self.split == 'train',
                               self.seed + 1000003 * self.epoch + clip) + begin
        source = {s: torch.from_numpy(np.array(value[chosen], dtype=np.float32, copy=True))
                  for s, value in self.features.items()}
        if not row['text'].strip() or not all(torch.isfinite(value).all() for value in source.values()):
            raise ValueError('invalid paired sample: ' + row['id'])
        return {'id': row['id'], 'text': row['text'], 'source': source}

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('features', None)
        state.pop('offsets', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_arrays()


class TranslationCollator:
    def __init__(self, tokenizer, max_text_length=128):
        self.tokenizer, self.max_text_length = tokenizer, max_text_length

    def __call__(self, rows):
        lengths = [len(row['source']['body']) for row in rows]
        frames = max(lengths)
        padding = torch.arange(frames)[None] >= torch.tensor(lengths)[:, None]
        source = {}
        for stream in STREAMS:
            result = torch.zeros(len(rows), frames, rows[0]['source'][stream].shape[1])
            for index, row in enumerate(rows):
                result[index, :lengths[index]] = row['source'][stream]
            source[stream] = result
        targets = self.tokenizer([row['text'] for row in rows], padding=True, truncation=True,
                                 max_length=self.max_text_length, return_tensors='pt')
        labels = targets['input_ids'].clone()
        labels[~targets['attention_mask'].bool()] = -100
        prefix = self.tokenizer(['Translate sign language video to English: '] * len(rows),
                                padding=True, return_tensors='pt')
        return {'source': source, 'padding_mask': padding, 'labels': labels,
                'prefix_ids': prefix['input_ids'], 'prefix_mask': prefix['attention_mask'],
                'ids': [row['id'] for row in rows], 'texts': [row['text'] for row in rows]}
