import hashlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

import numpy as np
import torch

from shubert_pose.packed_dataset import PackedPoseDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_youtubeasl_pack import convert_sign69  # noqa: E402


class PackedPoseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        np.save(self.root / "offsets.npy", np.array([0, 4, 11]))
        np.save(self.root / "indices.npy", np.array([0, 1]))
        np.save(self.root / "body.npy", np.repeat(np.arange(11, dtype=np.float32)[:, None], 4, axis=1))
        np.save(self.root / "labels.npy", np.arange(11, dtype=np.uint16))
        meta = {"format": "shubert-pose-pack-v1", "dimensions": {"body": 4}, "codebook_size": 16,
                "features": {"body": str(self.root / "body.npy")},
                "labels": {"body": str(self.root / "labels.npy")}}
        for key in ("offsets", "indices"):
            path = self.root / f"{key}.npy"
            meta[key] = str(path)
            meta[f"{key}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.manifest = self.root / "train.pack.json"
        self.manifest.write_text(json.dumps(meta))

    def data(self, **kwargs):
        return PackedPoseDataset(self.manifest, ["body"], [4], 16, max_frames=5, **kwargs)

    def test_crop_labels_padding_and_sizes(self):
        data = self.data()
        self.assertEqual(data.sizes.tolist(), [4, 5])
        for _ in range(10):
            batch = data.collater([data[0], data[1]])
            valid = ~batch["net_input"]["padding_mask"]
            torch.testing.assert_close(batch["net_input"]["source"]["body"][:, :, 0][valid],
                                       batch["net_input"]["labels"]["body"][valid].float())
            self.assertEqual(batch["net_input"]["labels"]["body"][0, -1], -100)

    def test_pickling_reopens_memmaps(self):
        data = self.data(shuffle=False)
        payload = pickle.dumps(data)
        restored = pickle.loads(payload)
        self.assertIsInstance(restored.features["body"], np.memmap)
        torch.testing.assert_close(data[1]["source"]["body"], restored[1]["source"]["body"])
        self.assertLess(len(payload), 10000)

    def test_spawn_worker_reads_aligned_slices(self):
        data = self.data(shuffle=False)
        loader = torch.utils.data.DataLoader(data, batch_size=2, num_workers=1,
                                            multiprocessing_context="spawn", collate_fn=data.collater)
        batch = next(iter(loader))
        valid = ~batch["net_input"]["padding_mask"]
        torch.testing.assert_close(batch["net_input"]["source"]["body"][:, :, 0][valid],
                                   batch["net_input"]["labels"]["body"][valid].float())

    def test_modified_indices_and_invalid_label_are_rejected(self):
        data = self.data()
        array = np.load(self.root / "labels.npy", mmap_mode="r+")
        array[0] = 99
        array.flush()
        with self.assertRaisesRegex(ValueError, "invalid packed"):
            data[0]
        np.save(self.root / "indices.npy", np.array([1, 0]))
        with self.assertRaisesRegex(ValueError, "indices changed"):
            self.data()

    def test_coordinate_normalization_and_missing_hands(self):
        raw = np.random.default_rng(0).uniform(100, 400, (8, 69, 3)).astype(np.float32)
        raw[..., 2] = 1
        raw[:, 3, :2], raw[:, 4, :2] = [300, 250], [200, 250]
        raw[:, 0, :2] = [250, 150]
        before = convert_sign69(raw)
        moved = raw.copy()
        moved[..., :2] = moved[..., :2] * 2.5 + [100, -50]
        after = convert_sign69(moved)
        for stream in before:
            self.assertTrue(np.isfinite(before[stream]).all())
            np.testing.assert_allclose(before[stream], after[stream], atol=1e-5)
        np.testing.assert_allclose(before["body"][:, :2], 0, atol=1e-6)
        raw[2, 9:30, 2] = 0
        self.assertTrue((convert_sign69(raw)["left_hand"][2] == 0).all())

    def test_face_coordinates_preserve_three_streams_and_handle_missing_anchor(self):
        raw = np.random.default_rng(42).uniform(100, 400, (8, 69, 3)).astype(np.float32)
        raw[..., 2] = 1
        raw[:, 0, :2], raw[:, 3, :2], raw[:, 4, :2] = [250, 150], [300, 250], [200, 250]
        raw[:, 68, :2] = [250, 160]
        before = convert_sign69(raw)
        four = convert_sign69(raw, include_face=True)
        self.assertEqual(four['face'].shape, (8, 36))
        for stream in before:
            np.testing.assert_array_equal(before[stream], four[stream])
        np.testing.assert_allclose(four['face'][:, -2:], 0)
        moved = raw.copy()
        moved[..., :2] = 2.5 * moved[..., :2] + [120, -50]
        np.testing.assert_allclose(four['face'], convert_sign69(moved, include_face=True)['face'], atol=1e-5)
        raw[1, 68, 2], raw[2, 51:69, 2] = 0, 0
        np.testing.assert_array_equal(convert_sign69(raw, include_face=True)['face'][1:3], 0)

    def test_four_stream_model_trains_face_projection_and_cluster_head(self):
        from shubert_pose import SHubertPoseConfig, SHubertPoseModel
        model = SHubertPoseModel(SHubertPoseConfig(
            stream_names=['body', 'left_hand', 'right_hand', 'face'], stream_dims=[14, 42, 42, 36],
            stream_embed_dim=8, codebook_size=2, encoder_layers=1, encoder_embed_dim=16,
            encoder_ffn_embed_dim=32, encoder_attention_heads=2, conv_pos=3, conv_pos_groups=2,
            required_seq_len_multiple=1, dropout=0, attention_dropout=0, activation_dropout=0, encoder_layerdrop=0))
        source = {s: torch.randn(2, 6, d) for s, d in zip(model.streams, model.cfg.stream_dims)}
        labels = {s: torch.randint(2, (2, 6)) for s in model.streams}
        mask = torch.zeros(2, 6, 4, dtype=torch.bool)
        mask[:, 1] = True
        loss = model(source, labels=labels, mask_indices=mask)['losses']['kmeans_loss'].mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for module in (model.heads['face'], model.projections['face']):
            self.assertTrue(torch.isfinite(module.weight.grad).all())
            self.assertGreater(float(module.weight.grad.abs().sum()), 0)

    def test_four_stream_pack_fits_face_codebook_and_rejects_three_stream_reuse(self):
        from types import SimpleNamespace
        from prepare_youtubeasl_pack import prepare, stream_dimensions
        source, output = self.root / 'source', self.root / 'four'
        source.mkdir()
        raw = np.random.default_rng(42).uniform(100, 400, (11, 69, 3)).astype(np.float32)
        raw[..., 2] = 1
        raw[:, 3, :2], raw[:, 4, :2] = [300, 250], [200, 250]
        np.save(source / 'poses.npy', raw)
        np.save(source / 'offsets.npy', np.array([0, 4, 11]))
        joints = [0] + list(range(3, 11)) + list(range(91, 133)) + list(range(23, 40, 2)) + list(range(83, 91)) + [53]
        (source / 'meta.json').write_text(json.dumps({
            'format': 'slim-sign-pack-v1', 'layout': 'unisign69', 'coco_wholebody_indices': joints,
            'channels': ['x', 'y', 'score'], 'num_clips': 2, 'num_frames': 11,
            'names': ['videoA-000', 'videoB-000']}))
        args = SimpleNamespace(source=str(source), output=str(output), include_face=True,
            workers=1, threshold=0.3, clusters=2, sample_frames=4, seed=42, valid_videos=1)
        prepare(args)
        dims = stream_dimensions(True)
        data = PackedPoseDataset(output / 'train.pack.json', list(dims), list(dims.values()), 2)
        self.assertEqual(data[0]['source']['face'].shape[-1], 36)
        self.assertTrue((output / 'codebooks/face.joblib').exists())
        prepare(args)
        with self.assertRaisesRegex(ValueError, 'fresh output'):
            prepare(SimpleNamespace(**dict(vars(args), include_face=False)))

    def test_real_sign69_face_coordinates_are_finite_and_aligned(self):
        root = Path('/scratch/lidowu91/SLiM/data/youtube_asl_rtmpose_all/train')
        if not (root / 'poses.npy').exists():
            self.skipTest('real sign69 data unavailable')
        raw, offsets = np.load(root / 'poses.npy', mmap_mode='r'), np.load(root / 'offsets.npy', mmap_mode='r')
        for index in range(min(32, len(offsets) - 1)):
            begin, end = map(int, offsets[index:index + 2])
            streams = convert_sign69(raw[begin:end], include_face=True)
            if np.any(streams['face']):
                self.assertEqual(streams['face'].shape, (end - begin, 36))
                self.assertTrue(all(len(value) == end - begin and np.isfinite(value).all()
                                    for value in streams.values()))
                return
        self.fail('no usable face coordinates in the real samples inspected')


if __name__ == "__main__":
    unittest.main()
