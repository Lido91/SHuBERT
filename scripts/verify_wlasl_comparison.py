"""Independently recount WLASL predictions and validate checkpoint selection."""

import argparse
import csv
import json
import math
from pathlib import Path


def verify(root):
    comparison = json.loads((root / 'comparison.json').read_text())
    if {row['pretrain_epoch'] for row in comparison} != {141, 142, 143}:
        raise ValueError('comparison does not contain the three requested pretraining snapshots')
    sources = {row['epoch']: row['sha256'] for row in json.loads((root / 'checkpoints.json').read_text())}
    records = []
    reference_protocol = None
    for entry in comparison:
        epoch, result = entry['pretrain_epoch'], entry['result']
        if entry['exit_code'] != 0 or not result:
            raise ValueError(f'epoch {epoch} did not complete successfully')
        output = root / f'epoch{epoch}'
        config = json.loads((output / 'config.json').read_text())
        protocol = {key: config[key] for key in (
            'epochs', 'patience', 'batch_size', 'seed', 'rank', 'alpha', 'head_lr', 'lora_lr',
            'optimizer', 'weight_decay', 'label_smoothing', 'precision', 'selection_split',
            'code_sha256', 'model_code_sha256')}
        reference_protocol = protocol if reference_protocol is None else reference_protocol
        if protocol != reference_protocol:
            raise ValueError('the three checkpoints were evaluated with different protocols')
        mode = config.get('finetune_mode', 'lora')
        if mode == 'full' and (config['lora_parameters'] != 0 or config['lora_layers']
                               or config['backbone_trainable_parameters'] < 90000000
                               or not result.get('backbone_weights_updated')):
            raise ValueError('full fine-tuning did not update the intended backbone weights')
        if (config['checkpoint_sha256'] != sources[epoch]
                or result['checkpoint_sha256'] != sources[epoch]):
            raise ValueError('source pretraining checkpoint hashes disagree')
        if (result['score_role'] != 'test_as_dev' or result['independent_test']
                or not result['frozen_weights_unchanged'] or config['selection_split'] != 'test'):
            raise ValueError('score role, selection split, or frozen-weight invariant is incorrect')
        data = Path(config['data']['source']) / 'test' / 'meta.json'
        metadata = json.loads(data.read_text())
        expected = dict(zip(metadata['names'], metadata['labels']))
        predictions = [json.loads(line) for line in (output / 'test_predictions.jsonl').read_text().splitlines()]
        ids = [p['id'] for p in predictions]
        if len(set(ids)) != len(ids) or set(ids) != set(expected) or len(predictions) != 2878:
            raise ValueError('test prediction coverage or ID uniqueness failed')
        counts = [[0, 0, 0, 0] for _ in range(2000)]
        for prediction in predictions:
            label, top = prediction['label'], prediction['top10']
            if (label != expected[prediction['id']] or len(set(top)) != 10
                    or min(top) < 0 or max(top) >= 2000):
                raise ValueError('prediction labels or ranked candidates are inconsistent')
            counts[label][0] += 1
            for column, k in enumerate((1, 5, 10), 1):
                counts[label][column] += label in top[:k]
        present = [row for row in counts if row[0]]
        metrics = result['test_as_dev']
        if metrics['samples'] != len(predictions) or metrics['classes_present'] != len(present):
            raise ValueError('reported test sample/class counts disagree with predictions')
        for column, k in enumerate((1, 5, 10), 1):
            pi = 100 * sum(row[column] for row in counts) / len(predictions)
            pc = 100 * sum(row[column] / row[0] for row in present) / len(present)
            if (not math.isclose(pi, metrics[f'pi_top{k}'], abs_tol=1e-8)
                    or not math.isclose(pc, metrics[f'pc_top{k}'], abs_tol=1e-8)):
                raise ValueError('reported metrics differ from independent prediction recount')
        baseline_path = output / 'selection_baseline.json'
        baseline = [json.loads(baseline_path.read_text())] if baseline_path.exists() else []
        history = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
        candidates = baseline + [row for row in history if 'test_as_dev' in row]
        if not candidates:
            raise ValueError('no test-as-dev selection history')
        best = max(candidates, key=lambda row: row['test_as_dev']['pi_top1'])
        if (result['selected_epoch'] != best['epoch']
                or not math.isclose(result['best_selection_pi_top1'], best['test_as_dev']['pi_top1'], abs_tol=1e-8)
                or not math.isclose(metrics['pi_top1'], result['best_selection_pi_top1'], abs_tol=1e-8)):
            raise ValueError('selected checkpoint is not the earliest maximum test-as-dev Top-1')
        records.append({'pretrain_epoch': epoch, 'finetune_mode': mode,
                        'selected_finetune_epoch': result['selected_epoch'],
                        'finetune_epochs_completed': history[-1]['epoch'],
                        'pi_top1': metrics['pi_top1'], 'pc_top1': metrics['pc_top1'],
                        'pi_top5': metrics['pi_top5'], 'pi_top10': metrics['pi_top10'],
                        'samples': len(predictions), 'classes': len(present),
                        'selection_started_epoch': result['selection_started_epoch']})
    records.sort(key=lambda row: row['pretrain_epoch'])
    with (root / 'comparison.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {'status': 'verified', 'score_role': 'test_as_dev', 'independent_test': False,
               'checks': ['all three processes completed', 'prediction ID/label coverage',
                          'independent P-I/P-C Top-1/5/10 recount', 'earliest maximum selection',
                          'parameter update invariants', 'identical protocols', 'source checkpoint hashes'],
               'results': records}
    (root / 'verification.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(
        '/scratch/lidowu91/SHuBERT/outputs/wlasl_pose_e141_143_20261115_test_as_dev'))
    verify(parser.parse_args().root)
