import unittest

import torch
from transformers import MT5Config, MT5ForConditionalGeneration

from shubert_pose import SHubertPoseConfig, SHubertPoseModel
from shubert_pose.grpo import GRPOPolicy, completion_mask, group_advantages, grouped_rewards, grpo_loss
from shubert_pose.translation import PoseMT5


def tiny_policy_inputs(device):
    encoder = SHubertPoseModel(SHubertPoseConfig(
        stream_embed_dim=8, encoder_layers=1, encoder_embed_dim=16, encoder_ffn_embed_dim=32,
        encoder_attention_heads=2, conv_pos=3, conv_pos_groups=2, required_seq_len_multiple=1))
    translator = MT5ForConditionalGeneration(MT5Config(
        vocab_size=32, d_model=16, d_ff=32, d_kv=4, num_heads=4, num_layers=1,
        num_decoder_layers=1, dropout_rate=0.5, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
    policy = GRPOPolicy(PoseMT5(encoder, translator)).to(device).eval()
    source = {s: torch.randn(1, 5, d, device=device) for s, d in zip(encoder.streams, encoder.cfg.stream_dims)}
    prefix = torch.tensor([[3, 1]], device=device)
    sequence = torch.tensor([[0, 5, 6, 1], [0, 7, 1, 0]], device=device)
    return policy, dict(source=source, padding_mask=torch.zeros(1, 5, dtype=torch.bool, device=device),
                        prefix_ids=prefix, prefix_mask=torch.ones_like(prefix), sequences=sequence)


class GRPOTests(unittest.TestCase):
    def test_real_mt5_tokenizer_output_padding_is_never_sampled(self):
        import json
        from pathlib import Path
        from transformers import T5Tokenizer
        from transformers.generation.logits_process import SuppressTokensLogitsProcessor
        root = Path('/scratch/lidowu91/Uni-Sign-Lido/checkpoints/mt5-base')
        if not root.exists():
            self.skipTest('real mT5 tokenizer assets are unavailable')
        tokenizer = T5Tokenizer.from_pretrained(str(root), local_files_only=True, legacy=False)
        size = json.loads((root / 'config.json').read_text())['vocab_size']
        decodable = set(tokenizer.get_vocab().values())
        suppressed = sorted(set(range(size)) - decodable)
        self.assertTrue(suppressed)
        scores = torch.zeros(1, size)
        scores[:, suppressed] = 1000
        constrained = SuppressTokensLogitsProcessor(suppressed)(torch.zeros(1, 1, dtype=torch.long), scores)
        samples = torch.multinomial(constrained.softmax(-1), 1024, replacement=True)
        self.assertTrue(all(int(token) in decodable for token in samples.flatten()))
        tokenizer.batch_decode(samples.reshape(-1, 1), skip_special_tokens=True)

    def test_padded_vocabulary_is_excluded_from_sampling_and_all_policy_probabilities(self):
        class Tokenizer:
            def get_vocab(self):
                # ID 31 models an added token beyond a SentencePiece base vocabulary.
                return {str(index): index for index in range(32) if index != 30}

        torch.manual_seed(42)
        policy, inputs = tiny_policy_inputs(torch.device('cpu'))
        self.assertEqual(policy.restrict_tokenizer_vocabulary(Tokenizer()), [30])
        self.assertEqual(policy.model.mt5_model.generation_config.suppress_tokens, [30])
        self.assertNotIn('suppressed_token_ids', policy.state_dict())

        def padded_token_dominates(module, arguments, logits):
            logits = logits.clone()
            logits[..., 30] = 10000.
            return logits

        hook = policy.model.mt5_model.lm_head.register_forward_hook(padded_token_dominates)
        try:
            arguments = {key: value for key, value in inputs.items() if key != 'sequences'}
            with torch.no_grad():
                generated = policy.model.generate(**arguments, do_sample=True, num_beams=1,
                    top_k=0, num_return_sequences=8, max_new_tokens=4)
            self.assertFalse(bool(generated.eq(30).any()))
            log_probs = policy(**inputs)
            self.assertTrue(torch.isfinite(log_probs).all())
            self.assertGreater(float(log_probs.min()), -100.)
            (-log_probs.mean()).backward()
            self.assertTrue(torch.isfinite(policy.model.projection.weight.grad).all())
            self.assertGreater(float(policy.model.projection.weight.grad.abs().sum()), 0.)
        finally:
            hook.remove()

    def test_tokenizer_larger_than_model_is_rejected(self):
        class Tokenizer:
            def get_vocab(self):
                return {'unsupported': 32}

        policy, _ = tiny_policy_inputs(torch.device('cpu'))
        with self.assertRaisesRegex(ValueError, 'vocabulary'):
            policy.restrict_tokenizer_vocabulary(Tokenizer())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for BF16 backward regression')
    def test_bf16_policy_backward_outside_autocast(self):
        policy, inputs = tiny_policy_inputs(torch.device('cuda', 0))
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            old = policy(**inputs)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            current = policy(**inputs)
            loss, _ = grpo_loss(current, old, old, torch.tensor([1., -1.], device='cuda'),
                                completion_mask(inputs['sequences'], 1))
        loss.backward()
        self.assertTrue(torch.isfinite(policy.model.projection.weight.grad).all())
        self.assertGreater(float(policy.model.projection.weight.grad.abs().sum()), 0.)

    def test_rewards_use_reference_bleu_and_leave_one_out_consensus(self):
        import sacrebleu
        candidates = ['A small red house', 'a small red house', 'a blue ocean']
        bleu, consensus = grouped_rewards([candidates], ['a small red house'])
        self.assertEqual(tuple(bleu.shape), (1, 3))
        self.assertAlmostEqual(float(bleu[0, 0]), 1.)
        self.assertEqual(float(bleu[0, 0]), float(bleu[0, 1]))
        expected = sum(sacrebleu.sentence_bleu(candidates[0], [other], lowercase=True,
                                              tokenize='13a').score / 100 for other in candidates[1:]) / 2
        self.assertAlmostEqual(float(consensus[0, 0]), expected, places=6)
        _, changed_consensus = grouped_rewards([candidates], ['completely different reference'])
        torch.testing.assert_close(consensus, changed_consensus)
        self.assertLess(float(consensus[0, 2]), float(consensus[0, 0]))

    def test_each_reward_is_normalized_before_weighting_within_each_clip(self):
        bleu = torch.tensor([[0., 1., 2., 4.], [1., 1., 1., 1.]])
        consensus = torch.tensor([[20., 0., 10., 5.], [2., 2., 2., 2.]])
        expected = ((bleu - bleu.mean(-1, keepdim=True)) / (bleu.std(-1, keepdim=True) + 1e-4)
                    + 0.5 * (consensus - consensus.mean(-1, keepdim=True))
                    / (consensus.std(-1, keepdim=True) + 1e-4))
        actual = group_advantages(bleu, consensus)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[1], torch.zeros(4))
        combined = bleu + 0.5 * consensus
        wrong = (combined - combined.mean(-1, keepdim=True)) / (combined.std(-1, keepdim=True) + 1e-4)
        self.assertFalse(torch.allclose(actual[0], wrong[0]))

    def test_eos_is_scored_but_decoder_start_and_post_eos_padding_are_excluded(self):
        sequences = torch.tensor([[0, 5, 1, 0, 0], [0, 1, 0, 0, 0], [0, 4, 5, 6, 7]])
        self.assertEqual(completion_mask(sequences, 1).tolist(),
                         [[True, True, False, False], [True, False, False, False], [True] * 4])

    def test_loss_is_sentence_mean_with_correct_policy_gradient_and_masking(self):
        current = torch.tensor([[-2., -2., -9.], [-3., -4., -5.]], requires_grad=True)
        old, reference = current.detach().clone(), current.detach().clone()
        mask = torch.tensor([[True, True, False], [True, False, False]])
        loss, metrics = grpo_loss(current, old, reference, torch.tensor([1., -1.]), mask)
        self.assertAlmostEqual(float(loss), 0.)
        self.assertAlmostEqual(float(metrics['kl']), 0.)
        loss.backward()
        torch.testing.assert_close(current.grad, torch.tensor([[-0.25, -0.25, 0.], [0.5, 0., 0.]]))
        self.assertAlmostEqual(float(metrics['ratio_mean']), 1.)

    def test_clipping_and_k3_kl_match_the_given_formula(self):
        old = torch.tensor([[-2.], [-2.]])
        current = torch.tensor([[-1.], [-3.]], requires_grad=True)
        reference = torch.tensor([[-1.5], [-2.5]])
        advantage = torch.tensor([1., -1.])
        mask = torch.ones_like(current, dtype=torch.bool)
        loss, metrics = grpo_loss(current, old, reference, advantage, mask)
        ratio = (current - old).exp()
        policy = -torch.minimum(ratio * advantage[:, None], ratio.clamp(0.8, 1.2) * advantage[:, None])
        delta = reference - current
        expected = (policy + 0.02 * (delta.exp() - delta - 1)).mean()
        torch.testing.assert_close(loss, expected)
        self.assertGreater(float(metrics['kl']), 0.)

    def test_shared_encoder_scoring_matches_teacher_forcing_and_trains_all_components(self):
        torch.manual_seed(42)
        encoder = SHubertPoseModel(SHubertPoseConfig(
            stream_embed_dim=8, encoder_layers=1, encoder_embed_dim=16, encoder_ffn_embed_dim=32,
            encoder_attention_heads=2, conv_pos=3, conv_pos_groups=2, required_seq_len_multiple=1))
        translator = MT5ForConditionalGeneration(MT5Config(
            vocab_size=32, d_model=16, d_ff=32, d_kv=4, num_heads=4, num_layers=1,
            num_decoder_layers=1, dropout_rate=0.5, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
        policy = GRPOPolicy(PoseMT5(encoder, translator)).eval()
        source = {s: torch.randn(1, 5, d) for s, d in zip(encoder.streams, encoder.cfg.stream_dims)}
        padding = torch.zeros(1, 5, dtype=torch.bool)
        prefix = torch.tensor([[3, 1]])
        sequence = torch.tensor([[0, 5, 6, 1], [0, 7, 1, 0]])
        inputs = dict(source=source, padding_mask=padding, prefix_ids=prefix,
                      prefix_mask=torch.ones_like(prefix), sequences=sequence)
        with torch.no_grad():
            old = policy(**inputs)
            embedded, attention = policy.model.translation_inputs(source, padding, prefix, torch.ones_like(prefix))
            logits = translator(inputs_embeds=embedded.expand(2, -1, -1),
                attention_mask=attention.expand(2, -1), decoder_input_ids=sequence[:, :-1],
                use_cache=False).logits
            expected = logits.float().log_softmax(-1).gather(-1, sequence[:, 1:, None]).squeeze(-1)
            torch.testing.assert_close(old, expected, atol=2e-5, rtol=2e-5)
        current = policy(**inputs)
        torch.testing.assert_close(current, old, atol=2e-5, rtol=2e-5)
        loss, _ = grpo_loss(current, old, old, torch.tensor([1., -1.]), completion_mask(sequence, 1))
        loss.backward()
        for module in (policy.model.encoder.projections['body'], policy.model.projection, translator.lm_head):
            self.assertIsNotNone(module.weight.grad)
            self.assertTrue(torch.isfinite(module.weight.grad).all())
            self.assertGreater(float(module.weight.grad.abs().sum()), 0.)
        self.assertFalse(any(module.training for module in policy.modules()))


if __name__ == '__main__':
    unittest.main()
