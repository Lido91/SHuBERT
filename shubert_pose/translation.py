"""Pose-only SHuBERT features projected into an mT5 translation model."""

import re

import torch
from torch import nn
from torch.nn import functional as F

from .wlasl import CheckpointLayer


def restore_translation_model(model, checkpoint, expected_step=None):
    """Restore the complete translator, without carrying its optimizer into SFT."""
    if expected_step is not None and checkpoint.get('step') != expected_step:
        raise ValueError('translation initialization step differs from the requested checkpoint')
    model.load_state_dict(checkpoint['model'], strict=True)
    return {'step': checkpoint['step'], 'epoch': checkpoint.get('epoch'),
            'validation': checkpoint.get('validation', checkpoint.get('last_eval')),
            'model_state_restored_strictly': True}


def restore_fullft_backbone(backbone, adapted):
    expected = {name for name, _ in backbone.named_parameters()
                if name != 'mask_embedding' and not name.startswith('heads.')}
    values = {}
    for name, value in adapted.items():
        if name.startswith('backbone.'):
            name = name[len('backbone.'):]
            name = re.sub(r'^(encoder\.layers\.\d+)\.layer\.', r'\1.', name)
            if name in values:
                raise ValueError('duplicate full-FT backbone parameter: ' + name)
            values[name] = value
    if set(values) != expected:
        raise ValueError(f'full-FT backbone keys differ: missing={sorted(expected - set(values))}, '
                         f'extra={sorted(set(values) - expected)}')
    state = backbone.state_dict()
    state.update(values)
    backbone.load_state_dict(state, strict=True)


class PoseMT5(nn.Module):
    def __init__(self, encoder, translator, label_smoothing=0.2, checkpoint_layers=True):
        super().__init__()
        self.encoder = encoder.requires_grad_(True)
        encoder.mask_embedding.requires_grad_(False)
        encoder.heads.requires_grad_(False)
        encoder.encoder.layerdrop = 0.0
        if checkpoint_layers:
            encoder.encoder.layers = nn.ModuleList([CheckpointLayer(layer) for layer in encoder.encoder.layers])
        self.projection = nn.Linear(encoder.cfg.encoder_embed_dim, translator.config.d_model)
        self.mt5_model = translator
        self.label_smoothing = label_smoothing

    def translation_inputs(self, source, padding_mask, prefix_ids, prefix_mask):
        features = self.encoder.extract_features(source, padding_mask=padding_mask, mask=False)['x']
        projected = self.projection(features)
        projected = projected.masked_fill(padding_mask[..., None], 0)
        prefix = self.mt5_model.encoder.embed_tokens(prefix_ids)
        inputs = torch.cat((prefix, projected), dim=1)
        attention = torch.cat((prefix_mask, (~padding_mask).long()), dim=1)
        return inputs, attention

    def forward(self, source, padding_mask, prefix_ids, prefix_mask, labels):
        inputs, attention = self.translation_inputs(source, padding_mask, prefix_ids, prefix_mask)
        result = self.mt5_model(inputs_embeds=inputs, attention_mask=attention,
                                decoder_input_ids=self.mt5_model._shift_right(labels),
                                use_cache=False, return_dict=True)
        return F.cross_entropy(result.logits.float().flatten(0, 1), labels.flatten(),
                               ignore_index=-100, label_smoothing=self.label_smoothing)

    @torch.inference_mode()
    def generate(self, source, padding_mask, prefix_ids, prefix_mask, **kwargs):
        inputs, attention = self.translation_inputs(source, padding_mask, prefix_ids, prefix_mask)
        return self.mt5_model.generate(inputs_embeds=inputs, attention_mask=attention, use_cache=True, **kwargs)
