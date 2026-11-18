"""Validate full-model provenance when transferring from OpenASL to How2Sign."""

import math
from pathlib import Path


def validate_source(checkpoint, completed, source_data):
    if completed.get('state') != 'complete':
        raise ValueError('OpenASL training is not complete; How2Sign handoff is not ready')
    config = checkpoint['identity']['config']
    if config.get('dataset') != 'OpenASL' or Path(config['data']).resolve() != Path(source_data).resolve():
        raise ValueError('handoff checkpoint did not originate from the requested OpenASL SFT run')
    score = checkpoint['validation']['bleu4']
    if not math.isfinite(score) or abs(score - completed['best_bleu4']) > 1e-7:
        raise ValueError('handoff checkpoint is not the selected OpenASL best model')
    if checkpoint['step'] > completed['step']:
        raise ValueError('handoff checkpoint step exceeds completed OpenASL training')
    keys = checkpoint['model']
    if (not {'projection.weight', 'projection.bias'} <= set(keys)
            or not any(k.startswith('encoder.') for k in keys)
            or not any(k.startswith('mt5_model.') for k in keys)):
        raise ValueError('handoff must contain encoder, projection, and mT5 weights')
    return {'source_dataset': 'OpenASL', 'source_step': checkpoint['step'],
            'source_epoch_index': checkpoint.get('epoch'), 'selected_bleu4': score,
            'selection_split': checkpoint['validation']['split'], 'independent_test': False,
            'target_dataset': 'How2Sign', 'complete_full_model': True}
