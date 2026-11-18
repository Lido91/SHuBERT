import tempfile
from pathlib import Path
import unittest

import torch
from torch import nn

from shubert_pose import SHubertPoseConfig, SHubertPoseModel
from shubert_pose.wlasl import LoRALinear, PoseFullFTClassifier, PoseLoRAClassifier, metrics_from_counts


class WLASLPoseTests(unittest.TestCase):
    def test_lora_functional_weight_has_gradients(self):
        layer = LoRALinear(nn.Linear(4, 6))
        x = torch.randn(3, 4)
        torch.testing.assert_close(layer(x), layer.base(x))
        torch.nn.functional.linear(x, layer.weight, layer.bias).square().mean().backward()
        self.assertIsNone(layer.base.weight.grad)
        self.assertIsNotNone(layer.a.grad)
        self.assertGreater(layer.b.grad.abs().sum().item(), 0)

    def test_attention_adapters_receive_gradients_and_restore(self):
        torch.manual_seed(1)
        cfg = SHubertPoseConfig(stream_embed_dim=8, encoder_layers=2, encoder_embed_dim=16,
                               encoder_ffn_embed_dim=32, encoder_attention_heads=2, conv_pos=3,
                               conv_pos_groups=2, required_seq_len_multiple=1,
                               dropout=0, attention_dropout=0, activation_dropout=0, encoder_layerdrop=0)
        backbone = SHubertPoseModel(cfg)
        source = {s: torch.randn(3, 7, d) for s, d in zip(cfg.stream_names, cfg.stream_dims)}
        padding = torch.zeros(3, 7, dtype=torch.bool)
        padding[1, 4:] = True
        backbone.eval()
        expected = backbone.extract_features(source, padding_mask=padding)["x"].detach()
        model = PoseLoRAClassifier(backbone, classes=12)
        actual = model.backbone.eval().extract_features(source, padding_mask=padding)["x"]
        torch.testing.assert_close(expected, actual)
        model.train()
        model(source, padding).square().mean().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
            else:
                self.assertIsNone(p.grad, name)
        for layer in model.modules():
            if isinstance(layer, LoRALinear):
                self.assertGreater(layer.b.grad.abs().sum().item(), 0)
        state = model.adapted_state()
        self.assertFalse(any('.base.' in k for k in state))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'adapters.pt'
            torch.save(state, path)
            model.restore_adapted(torch.load(path, weights_only=False))
        model.eval()
        changed = {s: x.clone() for s, x in source.items()}
        for x in changed.values():
            x[1, 4:] = 999
        torch.testing.assert_close(model(source, padding), model(changed, padding), atol=1e-5, rtol=1e-5)

    def test_per_instance_and_per_class_are_distinct(self):
        counts = torch.tensor([[10, 5, 8, 9], [2, 2, 2, 2], [0, 0, 0, 0]])
        values = metrics_from_counts(counts)
        self.assertEqual(values['samples'], 12)
        self.assertEqual(values['classes_present'], 2)
        self.assertAlmostEqual(values['pi_top1'], 100 * 7 / 12)
        self.assertEqual(values['pc_top1'], 75)

    def test_full_finetuning_updates_backbone_and_restores(self):
        torch.manual_seed(3)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cfg = SHubertPoseConfig(stream_embed_dim=8, encoder_layers=2, encoder_embed_dim=16,
                               encoder_ffn_embed_dim=32, encoder_attention_heads=2, conv_pos=3,
                               conv_pos_groups=2, required_seq_len_multiple=1,
                               dropout=0, attention_dropout=0, activation_dropout=0, encoder_layerdrop=0)
        model = PoseFullFTClassifier(SHubertPoseModel(cfg), classes=12).to(device)
        self.assertFalse(any(isinstance(layer, LoRALinear) for layer in model.modules()))
        for name, parameter in model.backbone.named_parameters():
            unused = name == 'mask_embedding' or name.startswith('heads.')
            self.assertEqual(parameter.requires_grad, not unused, name)
        source = {s: torch.randn(3, 7, d, device=device) for s, d in zip(cfg.stream_names, cfg.stream_dims)}
        padding = torch.zeros(3, 7, dtype=torch.bool, device=device)
        before = model.backbone.projections['body'].weight.detach().clone()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        torch.nn.functional.cross_entropy(model(source, padding), torch.tensor([0, 1, 2], device=device)).backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            else:
                self.assertIsNone(parameter.grad, name)
        gradient_max = model.backbone.projections['body'].weight.grad.abs().max().item()
        self.assertGreater(gradient_max, 0, 'full-FT body projection has a zero gradient')
        optimizer.step()
        self.assertFalse(torch.equal(before, model.backbone.projections['body'].weight),
                         f'Adam did not update projection: grad_max={gradient_max}, '
                         f'threads={torch.get_num_threads()}, device={before.device}')
        model.eval()
        expected = model(source, padding).detach()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'full.pth'
            torch.save(model.adapted_state(), path)
            restored = PoseFullFTClassifier(SHubertPoseModel(cfg), classes=12).to(device).eval()
            restored.restore_adapted(torch.load(path, weights_only=False))
        torch.testing.assert_close(expected, restored(source, padding))


if __name__ == '__main__':
    unittest.main()
