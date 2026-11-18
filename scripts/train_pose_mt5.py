"""Train or fine-tune SHuBERT + mT5 on paired poses and captions."""

import argparse
import contextlib
import datetime
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.pose_inference import load_model  # noqa: E402
from shubert_pose.translation import PoseMT5, restore_fullft_backbone, restore_translation_model  # noqa: E402
from shubert_pose.translation_data import TranslationCollator, TranslationDataset  # noqa: E402
from shubert_pose.wlasl import atomic_json, sha256  # noqa: E402


def atomic_save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def parallel():
    return dist.is_initialized()


def rng_state(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device)}


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'], device)


def amp(device, enabled):
    return torch.autocast('cuda', dtype=torch.bfloat16) if enabled else contextlib.nullcontext()


def finite(value):
    flag = (~torch.isfinite(value).all()).to(torch.int32)
    if parallel():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return not bool(flag.item())


def move(batch, device):
    return {key: ({s: x.to(device, non_blocking=True) for s, x in value.items()} if key == 'source'
                  else value.to(device, non_blocking=True)) for key, value in batch.items()
            if key not in ('ids', 'texts')}


class RemainingBatches(Sampler):
    def __init__(self, sampler, size, start=0):
        self.sampler, self.size, self.start = sampler, size, start

    def __len__(self):
        return max(0, len(self.sampler) // self.size - self.start)

    def __iter__(self):
        indices = list(self.sampler)
        for begin in range(self.start * self.size, len(indices) - self.size + 1, self.size):
            yield indices[begin:begin + self.size]


@torch.inference_mode()
def evaluate(model, dataset, collate, args, rank, world, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=args.eval_batch_size,
                        sampler=list(range(rank, len(dataset), world)), num_workers=args.num_workers,
                        collate_fn=collate, pin_memory=True)
    predictions = []
    for batch in loader:
        tensors = move(batch, device)
        tensors.pop('labels')
        with amp(device, args.bf16):
            generated = model.generate(**tensors, num_beams=args.num_beams, max_new_tokens=args.max_text_length)
        texts = collate.tokenizer.batch_decode(generated, skip_special_tokens=True)
        predictions.extend(zip(batch['ids'], batch['texts'], texts))
    if parallel():
        gathered = [None] * world
        dist.all_gather_object(gathered, predictions)
        predictions = [row for part in gathered for row in part] if rank == 0 else []
    result = None
    if rank == 0:
        import sacrebleu
        predictions.sort()
        expected = {row['id']: row['text'] for row in dataset.records}
        if (len(predictions) != len(dataset) or len({row[0] for row in predictions}) != len(dataset)
                or {row[0]: row[1] for row in predictions} != expected):
            raise RuntimeError('validation coverage or pose/caption pairing failed')
        score = sacrebleu.corpus_bleu([row[2] for row in predictions], [[row[1] for row in predictions]],
                                     tokenize='13a')
        result = {'bleu4': score.score, 'bleu_precisions': score.precisions, 'samples': len(predictions),
                  'split': dataset.metadata.get('evaluation_split', 'OpenASL test used as validation'),
                  'selection_metric': 'bleu4', 'bleu_tokenizer': '13a', 'independent_test': False}
        atomic_json(Path(args.output) / 'validation_predictions.json', predictions)
    if parallel():
        values = [result]
        dist.broadcast_object_list(values, src=0)
        result = values[0]
    model.train()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/translation_mt5.json'))
    parser.add_argument('--initialization', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-steps', type=int, default=0)
    parser.add_argument('--limit-samples', type=int, default=0)
    parser.add_argument('--eval-limit', type=int, default=0)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--max-text-length', type=int)
    parser.add_argument('--stop-at-timestamp', type=float, default=0)
    parser.add_argument('--mode', choices=('train', 'smoke'), default='train')
    options = parser.parse_args()
    config = json.loads(Path(options.config).read_text())
    for key, value in vars(options).items():
        if value is not None:
            config[key] = value
    return argparse.Namespace(**config)


def train(args):
    torch.set_num_threads(1)
    rank, local, world = [int(os.environ.get(key, default)) for key, default in
                          (('RANK', 0), ('LOCAL_RANK', 0), ('WORLD_SIZE', 1))]
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=45))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    from transformers import MT5ForConditionalGeneration, T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.text_encoder, local_files_only=True, legacy=False)
    collate = TranslationCollator(tokenizer, args.max_text_length)
    training = TranslationDataset(args.data, 'train', args.max_frames, args.seed, args.limit_samples)
    validation = TranslationDataset(args.data, 'validation', args.max_frames, args.seed, args.eval_limit)
    initialization = json.loads(Path(args.initialization).read_text())
    encoder = load_model(initialization['source_pose_checkpoint'], torch.device('cpu'))
    full = torch.load(initialization['full_finetune_checkpoint'], map_location='cpu', weights_only=False)
    restore_fullft_backbone(encoder, full['adapted'])
    del full
    translator = MT5ForConditionalGeneration.from_pretrained(args.text_encoder, local_files_only=True)
    translator.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    translator.config.use_cache = False
    model = PoseMT5(encoder, translator, args.label_smoothing)
    del encoder, translator
    slt_initialization = None
    if getattr(args, 'init_slt_checkpoint', None):
        saved_slt = torch.load(args.init_slt_checkpoint, map_location='cpu', weights_only=False)
        slt_initialization = restore_translation_model(model, saved_slt, getattr(args, 'init_slt_step', None))
        del saved_slt
        if rank == 0:
            atomic_json(output / 'slt_initialization.json', slt_initialization)
            print('FULL_SLT_CHECKPOINT_LOADED', json.dumps(slt_initialization), flush=True)
    model = model.to(device)
    identity = None
    if rank == 0:
        config = {k: v for k, v in vars(args).items() if k not in ('stop_at_timestamp',)}
        files = [Path(args.initialization), Path(args.data) / 'ready.json', Path(__file__),
                 ROOT / 'shubert_pose/translation.py', ROOT / 'shubert_pose/translation_data.py',
                 ROOT / 'shubert_pose/model.py', ROOT / 'shubert_pose/wlasl.py',
                 Path(initialization['source_pose_checkpoint']), Path(initialization['full_finetune_checkpoint']),
                 Path(args.text_encoder) / 'config.json', Path(args.text_encoder) / 'pytorch_model.bin',
                 Path(args.text_encoder) / 'spiece.model']
        if slt_initialization is not None:
            files.append(Path(args.init_slt_checkpoint))
        identity = {'config': config, 'world_size': world, 'initialization': initialization,
                    'files': {str(path): sha256(path) for path in files},
                    'train_samples': len(training), 'validation_samples': len(validation),
                    'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                    'method': 'full SHuBERT + projection + mT5 translation cross-entropy; no OT objective',
                    'validation_role': 'OpenASL test used for validation/selection; no independent test'}
        if slt_initialization is not None:
            dataset_name = getattr(args, 'dataset', 'OpenASL')
            identity.update(slt_initialization=slt_initialization,
                            method=f'{dataset_name} full-model SFT from selected SLT checkpoint; translation CE',
                            validation_role=(validation.metadata.get('evaluation_split',
                                'OpenASL test used as validation') + '; selection only; no independent test'))
        for key in ('source_pose_checkpoint', 'full_finetune_checkpoint'):
            if identity['files'][initialization[key]] != initialization[key + '_sha256']:
                raise RuntimeError('initialization snapshot changed: ' + key)
        existing = output / 'run_config.json'
        if existing.exists() and json.loads(existing.read_text()) != identity:
            raise RuntimeError('translation output belongs to a different experiment')
        atomic_json(existing, identity)
    if parallel():
        shared = [identity]
        dist.broadcast_object_list(shared, src=0)
        identity = shared[0]
    groups = []
    for is_encoder in (True, False):
        values = [p for name, p in model.named_parameters()
                  if p.requires_grad and name.startswith('encoder.') == is_encoder]
        groups.append({'params': values, 'lr': args.lr * (args.encoder_lr_scale if is_encoder else 1)})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    sampler = DistributedSampler(training, world, rank, shuffle=True, seed=args.seed, drop_last=True)
    batches_per_epoch = len(sampler) // args.batch_size
    if not batches_per_epoch:
        raise ValueError('not enough paired samples for a full batch on each GPU')
    total_steps = batches_per_epoch * args.epochs
    warmup = min(args.warmup_steps, max(1, total_steps // 10))

    def rate(step):
        if step < warmup:
            return (step + 1) / warmup
        fraction = min(1, (step - warmup) / max(1, total_steps - warmup))
        return 0.01 + 0.99 * (1 + math.cos(math.pi * fraction)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    step, start_epoch, start_batch, best, skipped, last_eval = 0, 0, 0, -1., 0, None
    last = output / 'checkpoint_last.pth'
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise RuntimeError('resume checkpoint source/configuration changed')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        step, start_epoch, start_batch, best, skipped, last_eval = [saved[key] for key in
            ('step', 'epoch', 'next_batch', 'best_score', 'skipped_updates', 'last_eval')]
        restore_rng(saved['rng'][rank], device)
        del saved
    elif slt_initialization is not None and getattr(args, 'evaluate_initial', False) and args.mode != 'smoke':
        last_eval = evaluate(model, validation, collate, args, rank, world, device)
        best = last_eval['bleu4']
        if rank == 0:
            atomic_json(output / 'initial_eval.json', {'step': 0, **last_eval})
            atomic_save(output / 'checkpoint_best.pth', {'model': model.state_dict(), 'step': 0,
                        'epoch': 0, 'validation': last_eval, 'identity': identity})
            print('INITIAL_VALIDATION', json.dumps(last_eval), flush=True)
    if not last.exists():
        torch.manual_seed(args.seed + rank)
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
    ddp = DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False) if parallel() else model
    parameters = [p for p in model.parameters() if p.requires_grad]
    stop_requested = [False]
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, lambda *_: stop_requested.__setitem__(0, True))

    def save(epoch, batch, is_best=False):
        states = [rng_state(device)]
        if parallel():
            states = [None] * world
            dist.all_gather_object(states, rng_state(device))
        if rank == 0:
            value = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'step': step, 'epoch': epoch, 'next_batch': batch,
                     'best_score': best, 'skipped_updates': skipped, 'last_eval': last_eval,
                     'rng': states, 'identity': identity}
            atomic_save(last, value)
            if is_best:
                atomic_save(output / 'checkpoint_best.pth', {'model': value['model'], 'step': step,
                             'epoch': epoch, 'validation': last_eval, 'identity': identity})
            atomic_json(output / 'checkpoint_state.json', {key: value[key] for key in
                        ('step', 'epoch', 'next_batch', 'best_score', 'last_eval')})
        if parallel():
            dist.barrier()

    if rank == 0:
        print('TRAIN_START', json.dumps({'world_size': world, 'batch_per_gpu': args.batch_size,
              'train_samples': len(training), 'validation_samples': len(validation), 'epochs': args.epochs,
              'batches_per_epoch': batches_per_epoch, 'resume_step': step,
              'trainable_parameters': identity['trainable_parameters']}), flush=True)
    initial_projection = model.projection.weight.detach().clone() if args.mode == 'smoke' else None
    recent = []
    for epoch in range(start_epoch, args.epochs):
        training.set_epoch(epoch)
        sampler.set_epoch(epoch)
        beginning = start_batch if epoch == start_epoch else 0
        loader = DataLoader(training, batch_sampler=RemainingBatches(sampler, args.batch_size, beginning),
                            num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
                            generator=torch.Generator().manual_seed(args.seed + epoch + rank))
        model.train()
        for index, batch in enumerate(loader, beginning):
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            with amp(device, args.bf16):
                loss = ddp(**move(batch, device))
            applied, grad = False, None
            if finite(loss):
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(parameters, args.clip_grad)
                if finite(grad):
                    optimizer.step()
                    scheduler.step()
                    step += 1
                    applied = True
            else:
                (torch.nan_to_num(loss) * 0).backward()
            if not applied:
                skipped += 1
                if skipped > 20:
                    raise FloatingPointError('too many non-finite translation updates')
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            seconds = time.monotonic() - started
            recent = (recent + [seconds])[-50:]
            at_end = index + 1 == batches_per_epoch
            smoke_end = args.max_steps > 0 and step >= args.max_steps
            improved = False
            if at_end or smoke_end or (applied and step % args.eval_every == 0):
                last_eval = evaluate(model, validation, collate, args, rank, world, device)
                improved = last_eval['bleu4'] > best
                best = max(best, last_eval['bleu4'])
                if rank == 0:
                    print('VALIDATION', json.dumps({'step': step, **last_eval}), flush=True)
            stop = (stop_requested[0] or smoke_end or (output / 'STOP').exists()
                    or (args.stop_at_timestamp and time.time() >= args.stop_at_timestamp))
            flag = torch.tensor(int(bool(stop)), device=device)
            if parallel():
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            stop = bool(flag.item())
            next_epoch, next_batch = (epoch + 1, 0) if at_end else (epoch, index + 1)
            if at_end or stop or improved or (applied and step % args.save_every == 0):
                save(next_epoch, next_batch, improved)
            if rank == 0:
                status = {'state': 'paused' if stop else 'running', 'step': step,
                          'epoch': epoch + (index + 1) / batches_per_epoch,
                          'loss': float(loss.detach()) if torch.isfinite(loss) else None,
                          'grad_norm': float(grad) if grad is not None and torch.isfinite(grad) else None,
                          'update_applied': applied, 'skipped_updates': skipped,
                          'seconds_per_update': sum(recent) / len(recent),
                          'remaining_train_seconds': (total_steps - epoch * batches_per_epoch - index - 1)
                                                      * sum(recent) / len(recent),
                          'lr': optimizer.param_groups[1]['lr'], 'encoder_lr': optimizer.param_groups[0]['lr'],
                          'best_bleu4': best, 'last_eval': last_eval, 'next_epoch': next_epoch,
                          'next_batch': next_batch, 'gpu_peak_gib': torch.cuda.max_memory_allocated(device) / 2**30,
                          'updated_unix': time.time()}
                atomic_json(output / 'progress.json', status)
                with (output / 'metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(status, allow_nan=False) + '\n')
                if step <= 3 or step % args.log_every == 0 or stop:
                    print('TRAIN_METRICS', json.dumps(status), flush=True)
            if stop:
                if args.mode == 'smoke':
                    if torch.equal(initial_projection, model.projection.weight.detach()):
                        raise RuntimeError('translation projection did not update')
                    restored = torch.load(last, map_location='cpu', weights_only=False)
                    model.load_state_dict(restored['model'], strict=True)
                    optimizer.load_state_dict(restored['optimizer'])
                    if restored['step'] != step:
                        raise RuntimeError('translation checkpoint step mismatch')
                    if rank == 0:
                        atomic_json(output / 'smoke_result.json', {'status': 'passed', 'steps': step,
                                    'finite_loss_and_gradients': True, 'projection_updated': True,
                                    'checkpoint_restored_strictly': True, 'validation': last_eval})
                if parallel():
                    dist.destroy_process_group()
                return
        start_batch = 0
    if rank == 0:
        status = json.loads((output / 'progress.json').read_text())
        status.update(state='complete', updated_unix=time.time())
        atomic_json(output / 'progress.json', status)
        atomic_json(output / 'completed.json', status)
        print('TRAIN_COMPLETE', step, flush=True)
    if parallel():
        dist.destroy_process_group()


if __name__ == '__main__':
    options = parse_args()
    try:
        train(options)
    except BaseException as error:
        if int(os.environ.get('RANK', 0)) == 0:
            output = Path(options.output)
            output.mkdir(parents=True, exist_ok=True)
            atomic_json(output / 'failure.json', {'error': repr(error), 'updated_unix': time.time()})
        raise
