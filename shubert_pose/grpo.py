"""Grouped BLEU/consensus rewards and token-level GRPO for pose translation."""

import torch
from torch import nn
from torch.nn import functional as F


def grouped_rewards(hypotheses, references):
    """Score each clip's candidates against its reference and other candidates."""
    import sacrebleu

    if len(hypotheses) != len(references) or not hypotheses:
        raise ValueError('require one reference per candidate group')
    size = len(hypotheses[0])
    if size < 2 or any(len(group) != size for group in hypotheses):
        raise ValueError('require equally sized groups with at least two candidates')

    def bleu(hypothesis, reference):
        return sacrebleu.sentence_bleu(hypothesis, [reference], lowercase=True, tokenize='13a').score / 100

    supervised, consensus = [], []
    for candidates, reference in zip(hypotheses, references):
        supervised.append([bleu(candidate, reference) for candidate in candidates])
        consensus.append([sum(bleu(candidate, other) for j, other in enumerate(candidates) if j != i)
                          / (size - 1) for i, candidate in enumerate(candidates)])
    return torch.tensor(supervised, dtype=torch.float32), torch.tensor(consensus, dtype=torch.float32)


def group_advantages(bleu, consensus, weight=0.5, epsilon=1e-4):
    if bleu.ndim != 2 or bleu.shape != consensus.shape or bleu.shape[1] < 2:
        raise ValueError('rewards must have shape [clips, candidates], candidates >= 2')

    def normalize(reward):
        # Match torch.std's sample standard deviation, independently per reward.
        return (reward - reward.mean(-1, keepdim=True)) / (reward.std(-1, keepdim=True) + epsilon)

    return normalize(bleu) + weight * normalize(consensus)


def completion_mask(sequences, eos_token_id):
    """Exclude the decoder start and tokens after the first EOS; include EOS."""
    tokens = sequences[:, 1:]
    eos = tokens.eq(eos_token_id)
    return (eos.cumsum(-1) - eos.long()).eq(0)


def grpo_loss(log_probs, old_log_probs, reference_log_probs, advantages, mask, beta=0.02, clip=0.2):
    if log_probs.shape != old_log_probs.shape or log_probs.shape != reference_log_probs.shape:
        raise ValueError('policy, sampling-policy and reference log probabilities must align')
    if mask.shape != log_probs.shape or advantages.shape != log_probs.shape[:1]:
        raise ValueError('mask and per-sentence advantages do not align with log probabilities')
    ratio = (log_probs - old_log_probs.detach()).exp()
    advantage = advantages.detach()[:, None]
    policy = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    delta = reference_log_probs.detach() - log_probs
    kl = torch.expm1(delta) - delta
    valid = mask.to(log_probs.dtype)
    denominator = valid.sum(-1).clamp_min(1)

    def sentence_mean(value):
        return ((value * valid).sum(-1) / denominator).mean()

    loss = sentence_mean(policy + beta * kl)
    return loss, {'policy_loss': sentence_mean(policy).detach(), 'kl': sentence_mean(kl).detach(),
                  'ratio_mean': sentence_mean(ratio).detach(),
                  'ratio_max_error': (ratio.detach() - 1).abs().masked_fill(~mask, 0).max(),
                  'completion_tokens': valid.sum(-1).mean().detach()}


class GRPOPolicy(nn.Module):
    """Score one pose clip's candidate group, sharing its encoder computation."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('suppressed_token_ids', torch.empty(0, dtype=torch.long), persistent=False)

    def restrict_tokenizer_vocabulary(self, tokenizer):
        """Use the same decodable token support for sampling and policy scoring."""
        size = self.model.mt5_model.config.vocab_size
        vocabulary = set(tokenizer.get_vocab().values())
        if not vocabulary or min(vocabulary) < 0 or max(vocabulary) >= size:
            raise ValueError('tokenizer vocabulary does not fit the model output vocabulary')
        generation = self.model.mt5_model.generation_config
        suppressed = sorted((set(range(size)) - vocabulary) | set(generation.suppress_tokens or []))
        self.suppressed_token_ids = torch.tensor(suppressed, dtype=torch.long,
                                                 device=self.suppressed_token_ids.device)
        generation.suppress_tokens = suppressed or None
        return suppressed

    def forward(self, source, padding_mask, prefix_ids, prefix_mask, sequences):
        from transformers.modeling_outputs import BaseModelOutput

        if padding_mask.shape[0] != 1:
            raise ValueError('score exactly one clip and its entire candidate group')
        inputs, attention = self.model.translation_inputs(source, padding_mask, prefix_ids, prefix_mask)
        hidden = self.model.mt5_model.encoder(inputs_embeds=inputs, attention_mask=attention,
                                              return_dict=True).last_hidden_state
        count = sequences.shape[0]
        result = self.model.mt5_model(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden.expand(count, -1, -1)),
            attention_mask=attention.expand(count, -1), decoder_input_ids=sequences[:, :-1],
            use_cache=False, return_dict=True)
        labels = sequences[:, 1:]
        logits = result.logits.float()
        if self.suppressed_token_ids.numel():
            if bool(torch.isin(labels, self.suppressed_token_ids).any()):
                raise ValueError('sampled completion contains a suppressed token ID')
            logits = logits.index_fill(-1, self.suppressed_token_ids, float('-inf'))
        return -F.cross_entropy(logits.transpose(1, 2), labels, reduction='none')
