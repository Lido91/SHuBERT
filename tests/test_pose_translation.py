import importlib.util
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from shubert_pose import SHubertPoseConfig, SHubertPoseModel
from shubert_pose.translation import PoseMT5, restore_fullft_backbone, restore_translation_model
from shubert_pose.translation_data import TranslationCollator, TranslationDataset, frame_indices
from shubert_pose.wlasl import PoseFullFTClassifier


def tiny_pose():
    return SHubertPoseModel(SHubertPoseConfig(stream_embed_dim=8, encoder_layers=2, encoder_embed_dim=16,
        encoder_ffn_embed_dim=32, encoder_attention_heads=2, conv_pos=3, conv_pos_groups=2,
        required_seq_len_multiple=1, dropout=0, attention_dropout=0, activation_dropout=0, encoder_layerdrop=0))


class TinyMT5(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(d_model=16)
        self.encoder = nn.Module()
        self.encoder.embed_tokens = nn.Embedding(32, 16)
        self.output = nn.Linear(16, 32)

    def _shift_right(self, labels):
        result = torch.zeros_like(labels)
        result[:, 1:] = labels[:, :-1].clamp_min(0)
        return result

    def forward(self, inputs_embeds, attention_mask, decoder_input_ids, **kwargs):
        pooled = (inputs_embeds * attention_mask[..., None]).sum(1) / attention_mask.sum(1)[:, None]
        hidden = self.encoder.embed_tokens(decoder_input_ids) + pooled[:, None]
        return SimpleNamespace(logits=self.output(hidden))


class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        lengths = [min(4, len(text.split()) + 1) for text in texts]
        ids = torch.zeros(len(texts), max(lengths), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, length in enumerate(lengths):
            ids[index, :length] = torch.arange(1, length + 1)
            mask[index, :length] = 1
        return {'input_ids': ids, 'attention_mask': mask}


class TranslationTests(unittest.TestCase):
    def test_how2sign_continuation_pins_selected_full_model_with_new_schedule(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/prepare_how2sign_continuation.py'
        spec = importlib.util.spec_from_file_location('pose_continuation_script', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'source', root / 'continuation'
            (source / 'train').mkdir(parents=True)
            model = PoseMT5(tiny_pose(), TinyMT5(), checkpoint_layers=False)
            checkpoint = {'model': model.state_dict(), 'step': 3876, 'epoch': 11,
                          'validation': {'bleu4': 15.175059951878245, 'split': 'How2Sign test used as validation'},
                          'identity': {'config': {'dataset': 'How2Sign', 'data': str(source / 'data')}}}
            checkpoint_path = source / 'train/checkpoint_best.pth'
            torch.save(checkpoint, checkpoint_path)
            config = root / 'template.json'
            config.write_text(json.dumps({'dataset': 'How2Sign', 'data': str(source / 'data'),
                                          'lr': 1e-4, 'epochs': 10}))
            args = SimpleNamespace(source=str(source), output=str(target), config=str(config),
                                   expected_step=3876, expected_bleu4=15.175059951878245)
            module.prepare(args)
            resolved = json.loads((target / 'sft_config.json').read_text())
            snapshot = Path(resolved['init_slt_checkpoint'])
            self.assertEqual(snapshot.read_bytes(), checkpoint_path.read_bytes())
            self.assertEqual(resolved['init_slt_step'], 3876)
            self.assertEqual(resolved['lr'], 1e-4)
            self.assertTrue(json.loads((target / 'handoff.json').read_text())['optimizer_reset'])
            restored = PoseMT5(tiny_pose(), TinyMT5(), checkpoint_layers=False)
            restore_translation_model(restored, torch.load(str(snapshot), mmap=True, weights_only=False), 3876)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key])
            # Later changes to the original run must not change this selected model.
            torch.save(dict(checkpoint, step=6460), checkpoint_path)
            module.prepare(args)
            self.assertEqual(torch.load(str(snapshot), mmap=True, weights_only=False)['step'], 3876)
            for changes, message in (({'step': 1}, 'selected'),
                                     ({'validation': {'bleu4': 14.9}}, 'selected'),
                                     ({'model': {'encoder.weight': None}}, 'complete encoder'),
                                     ({'identity': {'config': {'dataset': 'OpenASL',
                                                              'data': str(source / 'data')}}}, 'originate')):
                with self.subTest(changes=list(changes)), self.assertRaisesRegex(ValueError, message):
                    module.validate_checkpoint(dict(checkpoint, **changes), source / 'data',
                                               args.expected_step, args.expected_bleu4)
            template = json.loads(config.read_text())
            config.write_text(json.dumps(dict(template, warmup_steps=200)))
            module.prepare(args)
            (target / 'train').mkdir()
            (target / 'train/run_config.json').write_text('{}')
            config.write_text(json.dumps(dict(template, warmup_steps=100)))
            with self.assertRaisesRegex(RuntimeError, 'after model startup'):
                module.prepare(args)

    def test_handoff_snapshots_selected_checkpoint_and_writes_concrete_sft_config(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/prepare_how2sign_sft.py'
        spec = importlib.util.spec_from_file_location('pose_sft_handoff_script', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'openasl', root / 'how2sign'
            (source / 'train').mkdir(parents=True)
            model = PoseMT5(tiny_pose(), TinyMT5(), checkpoint_layers=False)
            checkpoint = {'model': model.state_dict(), 'step': 12, 'epoch': 1,
                          'validation': {'bleu4': 22.9, 'split': 'OpenASL test used as validation'},
                          'identity': {'config': {'dataset': 'OpenASL', 'data': str(source / 'data')}}}
            torch.save(checkpoint, source / 'train/checkpoint_best.pth')
            (source / 'train/completed.json').write_text(json.dumps(
                {'state': 'complete', 'step': 20, 'best_bleu4': 22.9}))
            config = root / 'template.json'
            config.write_text(json.dumps({'dataset': 'How2Sign', 'init_slt_checkpoint': None}))
            args = SimpleNamespace(source=str(source), output=str(target), config=str(config))
            module.prepare(args)
            resolved = json.loads((target / 'sft_config.json').read_text())
            self.assertEqual(resolved['init_slt_step'], 12)
            self.assertEqual(Path(resolved['init_slt_checkpoint']).read_bytes(),
                             (source / 'train/checkpoint_best.pth').read_bytes())
            self.assertEqual(json.loads((target / 'handoff.json').read_text())['selected_bleu4'], 22.9)
            module.prepare(args)

    def test_handoff_requires_completed_openasl_best_full_model(self):
        from shubert_pose.sft_handoff import validate_source
        checkpoint = {'identity': {'config': {'dataset': 'OpenASL', 'data': '/tmp/openasl/data'}},
                      'step': 100, 'epoch': 2,
                      'validation': {'bleu4': 20.0, 'split': 'OpenASL test used as validation'},
                      'model': {'projection.weight': None, 'projection.bias': None,
                                'encoder.weight': None, 'mt5_model.weight': None}}
        completed = {'state': 'complete', 'step': 200, 'best_bleu4': 20.0}
        result = validate_source(checkpoint, completed, '/tmp/openasl/data')
        self.assertEqual(result['source_step'], 100)
        self.assertEqual(result['target_dataset'], 'How2Sign')
        with self.assertRaisesRegex(ValueError, 'not complete'):
            validate_source(checkpoint, dict(completed, state='paused'), '/tmp/openasl/data')
        with self.assertRaisesRegex(ValueError, 'not the selected'):
            validate_source(checkpoint, dict(completed, best_bleu4=21.0), '/tmp/openasl/data')
        with self.assertRaisesRegex(ValueError, 'did not originate'):
            validate_source(checkpoint, completed, '/tmp/wrong/data')
        incomplete = dict(checkpoint, model={'encoder.weight': None})
        with self.assertRaisesRegex(ValueError, 'must contain'):
            validate_source(incomplete, completed, '/tmp/openasl/data')

    def test_slt_initialization_restores_encoder_projection_and_text_model(self):
        source = PoseMT5(tiny_pose(), TinyMT5(), checkpoint_layers=False)
        target = PoseMT5(tiny_pose(), TinyMT5(), checkpoint_layers=False)
        saved = {'model': source.state_dict(), 'step': 64000, 'epoch': 10,
                 'validation': {'bleu4': 16.36249987310074}}
        provenance = restore_translation_model(target, saved, expected_step=64000)
        self.assertTrue(provenance['model_state_restored_strictly'])
        for key, value in source.state_dict().items():
            torch.testing.assert_close(value, target.state_dict()[key])
        with self.assertRaisesRegex(ValueError, 'step differs'):
            restore_translation_model(target, saved, expected_step=1)
        incomplete = dict(saved, model=dict(saved['model']))
        incomplete['model'].pop('projection.weight')
        with self.assertRaises(RuntimeError):
            restore_translation_model(target, incomplete)

    def test_fullft_checkpoint_restores_raw_backbone(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        classifier = PoseFullFTClassifier(tiny_pose()).to(device).eval()
        restored = tiny_pose().to(device).eval()
        adapted = classifier.adapted_state()
        restore_fullft_backbone(restored, adapted)
        source = {s: torch.randn(2, 5, d, device=device)
                  for s, d in zip(restored.streams, restored.cfg.stream_dims)}
        padding = torch.zeros(2, 5, dtype=torch.bool, device=device)
        torch.testing.assert_close(classifier.backbone.extract_features(source, padding)['x'],
                                   restored.extract_features(source, padding)['x'])
        adapted.pop(next(key for key in adapted if key.startswith('backbone.')))
        with self.assertRaisesRegex(ValueError, 'keys differ'):
            restore_fullft_backbone(restored, adapted)

    def test_translation_masks_padding_and_trains_both_models(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PoseMT5(tiny_pose(), TinyMT5()).to(device)
        source = {s: torch.randn(2, 6, d, device=device)
                  for s, d in zip(model.encoder.streams, model.encoder.cfg.stream_dims)}
        padding = torch.zeros(2, 6, dtype=torch.bool, device=device)
        padding[1, 3:] = True
        prefix = torch.ones(2, 2, dtype=torch.long, device=device)
        labels = torch.tensor([[2, 3, 4], [4, 5, -100]], device=device)
        model.eval()
        before = model.translation_inputs(source, padding, prefix, prefix)
        changed = {s: value.clone() for s, value in source.items()}
        for value in changed.values():
            value[padding] = 999
        after = model.translation_inputs(changed, padding, prefix, prefix)
        torch.testing.assert_close(before[0], after[0])
        self.assertEqual(before[1].tolist(), [[1] * 8, [1] * 5 + [0] * 3])
        model.train()
        loss = model(source, padding, prefix, prefix, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for module in (model.projection, model.mt5_model.output, model.encoder.projections['body']):
            self.assertIsNotNone(module.weight.grad)
            self.assertTrue(torch.isfinite(module.weight.grad).all())
            self.assertGreater(module.weight.grad.abs().sum().item(), 0)

    def test_paired_sampling_collation_and_mmap_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / 'offsets.npy', np.array([0, 4, 14]))
            features = {}
            for stream, dimension in (('body', 14), ('left_hand', 42), ('right_hand', 42)):
                path = root / f'{stream}.npy'
                np.save(path, np.repeat(np.arange(14, dtype=np.float32)[:, None], dimension, axis=1))
                features[stream] = str(path)
            records = root / 'records.jsonl'
            records.write_text(json.dumps({'id': 'second', 'text': 'two words', 'pack_index': 1}) + '\n'
                               + json.dumps({'id': 'first', 'text': 'one', 'pack_index': 0}) + '\n')
            meta = {'records': str(records), 'offsets': str(root / 'offsets.npy'), 'features': features}
            for split in ('train', 'validation'):
                (root / f'{split}.json').write_text(json.dumps(meta))
            data = TranslationDataset(root, 'validation', max_frames=6)
            expected = np.linspace(4, 13, 6, dtype=np.int64)
            np.testing.assert_array_equal(data[0]['source']['body'][:, 0], expected)
            self.assertEqual(data[0]['text'], 'two words')
            restored = pickle.loads(pickle.dumps(data))
            self.assertIsInstance(restored.features['body'], np.memmap)
            batch = TranslationCollator(TinyTokenizer())([data[0], data[1]])
            self.assertEqual(batch['padding_mask'][1].tolist(), [False] * 4 + [True] * 2)
            self.assertEqual(batch['labels'][1, -1], -100)
            sampled = frame_indices(20, 6, True, 42)
            np.testing.assert_array_equal(sampled, frame_indices(20, 6, True, 42))
            self.assertFalse(np.array_equal(sampled, frame_indices(20, 6, True, 43)))


if __name__ == '__main__':
    unittest.main()
