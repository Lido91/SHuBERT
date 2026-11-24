"""SeRV-style semantic alignment of pose features to a frozen T5 text encoder.

Two symmetric InfoNCE losses are computed on clips that have a caption: a sentence-level loss on
pooled features and a FILIP-style token-wise late-interaction loss on frame and text tokens.
Projection heads and the handling of other processes' candidates follow SSVP-SLT.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

SEMANTIC_PREFIX = "semantic."
# Upper bound on the elements of one [texts, clips, text tokens, frames] similarity block.
BLOCK_ELEMENTS = 2 ** 26


def head(inputs, outputs):
    """SSVP-SLT's projection: two bias-free linear layers around a GELU."""
    hidden = (inputs + outputs) // 2
    return nn.Sequential(nn.Linear(inputs, hidden, bias=False), nn.GELU(), nn.Linear(hidden, outputs, bias=False))


def process_group():
    """(rank, world size) of the active process group; (0, 1) without distributed training."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def gather_counts(values, device):
    """[ranks, len(values)] integers: every rank's `values`, ordered by rank."""
    local = torch.tensor(values, dtype=torch.long, device=device)
    parts = [torch.zeros_like(local) for _ in range(process_group()[1])]
    torch.distributed.all_gather(parts, local)
    return torch.stack(parts)


def gather_rows(local, counts, rank):
    """All ranks' rows, ordered by rank; `counts` holds each rank's row count.

    Only this rank's own rows keep their gradient. Summed over ranks, the gradients of a loss on the
    gathered rows are the gradients of that loss computed in one process.
    """
    padded = local.new_zeros((int(counts.max()),) + local.shape[1:])
    padded[:len(local)] = local.detach()
    parts = [torch.zeros_like(padded) for _ in counts]
    torch.distributed.all_gather(parts, padded)
    rows = [part[:count] for part, count in zip(parts, counts.tolist())]
    rows[rank] = local
    return torch.cat(rows)


def pad_tokens(tokens, mask, length):
    """Right-pad the token axis of [N, L, D] tokens and their [N, L] validity mask."""
    extra = length - tokens.size(1)
    return F.pad(tokens, (0, 0, 0, extra)), F.pad(mask, (0, extra))


def interaction(tokens, mask, candidates, candidate_mask):
    """[N, C] late-interaction scores: each valid token's best match in a candidate, averaged per row."""
    scores = tokens.new_zeros(len(tokens), len(candidates))
    floor = torch.finfo(tokens.dtype).min / 2
    step = max(1, BLOCK_ELEMENTS // max(1, len(candidates) * tokens.size(1) * candidates.size(1)))
    for begin in range(0, len(tokens), step):
        rows = mask[begin:begin + step].to(tokens.dtype)
        similarity = torch.einsum("nld,cmd->nclm", tokens[begin:begin + step], candidates)
        best = similarity.masked_fill(~candidate_mask[None, :, None, :], floor).amax(dim=-1)
        scores[begin:begin + step] = (best * rows[:, None]).sum(-1) / rows.sum(-1).clamp_min(1)[:, None]
    return scores


def keep_top_tokens(tokens, mask, scores, ratio):
    keep = max(1, min(tokens.size(1), math.ceil(tokens.size(1) * ratio)))
    chosen = scores.masked_fill(~mask, torch.finfo(scores.dtype).min).topk(keep, dim=1).indices
    return tokens.gather(1, chosen[..., None].expand(-1, -1, tokens.size(-1))), mask.gather(1, chosen)


@torch.no_grad()
def best_local_matches(text, text_mask, frames, frame_mask):
    """Each text token's and each frame's strongest cross-modal match within this process."""
    floor = torch.finfo(text.dtype).min / 2
    text_scores = text.new_full(text.shape[:2], floor)
    frame_scores = frames.new_full(frames.shape[:2], floor)
    step = max(1, BLOCK_ELEMENTS // max(1, len(frames) * text.size(1) * frames.size(1)))
    for begin in range(0, len(text), step):
        similarity = torch.einsum("nld,cmd->nclm", text[begin:begin + step], frames)
        text_scores[begin:begin + step] = similarity.masked_fill(
            ~frame_mask[None, :, None, :], floor).amax(dim=(1, 3))
        frame_scores = torch.maximum(frame_scores, similarity.masked_fill(
            ~text_mask[begin:begin + step, None, :, None], floor).amax(dim=(0, 2)))
    return text_scores, frame_scores


def symmetric_nce(scores, reverse_scores, temperature):
    """Symmetric InfoNCE over [N, N] scores in both directions, with matching pairs on the diagonal."""
    labels = torch.arange(len(scores), device=scores.device)
    loss = 0.5 * (F.cross_entropy(scores / temperature, labels)
                  + F.cross_entropy(reverse_scores / temperature, labels))
    return loss, (scores.argmax(-1) == labels).float().mean()


class SemanticAlignment(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if not 0 < cfg.token_top_ratio <= 1 or cfg.sent_tau <= 0 or cfg.token_tau <= 0 or cfg.align_dim < 1:
            raise ValueError("require 0 < token_top_ratio <= 1, positive temperatures, and align_dim >= 1")
        self.text_model, self.text_dim = cfg.text_model, cfg.text_dim
        self.sent_tau, self.token_tau = cfg.sent_tau, cfg.token_tau
        self.token_align, self.token_top_ratio = cfg.token_align, cfg.token_top_ratio
        self.gather = cfg.semantic_gather
        self.pose_sentence = head(cfg.encoder_embed_dim, cfg.align_dim)
        self.text_sentence = head(cfg.text_dim, cfg.align_dim)
        if self.token_align:
            self.pose_token = head(cfg.encoder_embed_dim, cfg.align_dim)
            self.text_token = head(cfg.text_dim, cfg.align_dim)
        # A plain list keeps the frozen text encoder out of the state dict, .half(), and the optimizer.
        self._text_encoder = []

    def text_encoder(self, device=None):
        if not self._text_encoder:
            if not self.text_model:
                raise ValueError("semantic alignment requires model.text_model")
            from transformers import AutoConfig, MT5EncoderModel, T5EncoderModel
            kind = MT5EncoderModel if AutoConfig.from_pretrained(self.text_model).model_type == "mt5" else T5EncoderModel
            encoder = kind.from_pretrained(self.text_model)
            if encoder.config.d_model != self.text_dim:
                raise ValueError(f"model.text_dim={self.text_dim} differs from the text encoder's "
                                 f"{encoder.config.d_model}")
            self._text_encoder.append(encoder.float().eval().requires_grad_(False))
        encoder = self._text_encoder[0]
        if device is not None and next(encoder.parameters()).device != device:
            encoder.to(device)
        return encoder

    @torch.no_grad()
    def encode_text(self, text_ids, text_mask):
        encoder = self.text_encoder(text_ids.device)
        if int(text_ids.max()) >= encoder.config.vocab_size:
            raise ValueError("caption token IDs exceed the text encoder vocabulary")
        with torch.autocast(text_ids.device.type, enabled=False):
            tokens = encoder(input_ids=text_ids, attention_mask=text_mask.long()).last_hidden_state.float()
        weights = text_mask[..., None].float()
        return tokens, (tokens * weights).sum(1) / weights.sum(1).clamp_min(1)

    def forward(self, x, padding_mask, text_ids, text_mask, sample_size, gather=True):
        """Losses over all captioned clips, in float32, and logs.

        `text_mask` is True at caption tokens; clips without any are skipped. With `gather`, the losses
        cover every process's clips. `sample_size` is the weight the criterion gives this process's loss.
        """
        if text_ids is None or text_mask is None:
            raise ValueError("semantic alignment requires text_ids and text_mask; set task.captions")
        if text_ids.shape != text_mask.shape or text_mask.dtype != torch.bool or len(text_ids) != len(x):
            raise ValueError("text_ids and boolean text_mask must have shape [B, L]")
        paired = text_mask.any(1) & ~padding_mask.all(1)
        rank, world = process_group()
        distributed = gather and world > 1
        words_kept, frames_kept = (max(1, math.ceil(size * self.token_top_ratio))
                                   for size in (text_ids.size(1), x.size(1)))
        pairs, share = int(paired.sum()), 1.0
        if distributed:
            # Every rank runs the same collectives, even with no captioned clip of its own.
            counts, sizes, word_sizes, frame_sizes = gather_counts(
                [pairs, sample_size, words_kept, frames_kept], x.device).unbind(1)
            pairs = int(counts.sum())
            # Each rank holds a share of the gradient of one global loss. Scaled like this, the shares
            # add up to that loss once after the criterion weights every rank by its own sample size.
            share = float(sizes.sum()) / sample_size
        zero = x.float().sum() * 0
        losses = {"sent_nce_loss": zero}
        logs = {"sent_acc": 0.0, "caption_pairs": float(paired.sum())}
        if self.token_align:
            losses["token_nce_loss"], logs["token_acc"] = zero, 0.0
        if not pairs:
            return losses, logs

        def scaled(loss):
            return zero + loss.detach() + (loss - loss.detach()) * share

        frames, valid = x[paired], ~padding_mask[paired]
        weights = valid[..., None].to(frames.dtype)
        pose = F.normalize(self.pose_sentence((frames * weights).sum(1) / weights.sum(1)).float(), dim=-1)
        text_tokens, sentence = self.encode_text(text_ids[paired], text_mask[paired]) if paired.any() else (
            x.new_zeros(0, text_ids.size(1), self.text_dim, dtype=torch.float),
            x.new_zeros(0, self.text_dim, dtype=torch.float))
        sentence = F.normalize(self.text_sentence(sentence.to(frames.dtype)).float(), dim=-1)
        if distributed:
            pose, sentence = gather_rows(pose, counts, rank), gather_rows(sentence, counts, rank)
        scores = pose @ sentence.t()
        loss, accuracy = symmetric_nce(scores, scores.t(), self.sent_tau)
        losses["sent_nce_loss"], logs["sent_acc"] = scaled(loss), float(accuracy)
        if not self.token_align:
            return losses, logs

        words = F.normalize(self.text_token(text_tokens.to(frames.dtype)).float(), dim=-1)
        steps = F.normalize(self.pose_token(frames).float(), dim=-1)
        word_mask = text_mask[paired]
        if self.token_top_ratio < 1:
            word_scores, step_scores = best_local_matches(words, word_mask, steps, valid)
            words, word_mask = keep_top_tokens(words, word_mask, word_scores, self.token_top_ratio)
            steps, valid = keep_top_tokens(steps, valid, step_scores, self.token_top_ratio)
        if distributed:
            words, word_mask = pad_tokens(words, word_mask, int(word_sizes.max()))
            steps, valid = pad_tokens(steps, valid, int(frame_sizes.max()))
            words, steps = gather_rows(words, counts, rank), gather_rows(steps, counts, rank)
            word_mask = gather_rows(word_mask.long(), counts, rank).bool()
            valid = gather_rows(valid.long(), counts, rank).bool()
        loss, accuracy = symmetric_nce(interaction(steps, valid, words, word_mask),
                                       interaction(words, word_mask, steps, valid), self.token_tau)
        losses["token_nce_loss"], logs["token_acc"] = scaled(loss), float(accuracy)
        return losses, logs
