"""Reinforce PoseSHuBERT/mT5 with separately standardized BLEU and consensus."""

import argparse
import contextlib
import datetime
import json
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
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.pose_inference import load_model  # noqa: E402
from shubert_pose.grpo import GRPOPolicy, completion_mask, group_advantages, grouped_rewards, grpo_loss  # noqa: E402
from shubert_pose.translation import PoseMT5, restore_translation_model  # noqa: E402
from shubert_pose.translation_data import TranslationCollator, TranslationDataset  # noqa: E402
from shubert_pose.wlasl import atomic_json, sha256  # noqa: E402
from train_pose_mt5 import (RemainingBatches, amp, atomic_save, evaluate, finite, move,
                            parallel, restore_rng, rng_state)  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/how2sign_pose_grpo.json'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('train', 'smoke'), default='train')
    parser.add_argument('--max-steps', type=int, default=0)
    parser.add_argument('--limit-samples', type=int, default=0)
    parser.add_argument('--eval-limit', type=int, default=0)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--stop-at-timestamp', type=float, default=0)
    options = parser.parse_args()
    config = json.loads(Path(options.config).read_text())
    config.update({key: value for key, value in vars(options).items() if value is not None})
    return argparse.Namespace(**config)


def slice_inputs(batch, begin, end):
    return {key: ({name: tensor[begin:end] for name, tensor in value.items()} if key == 'source'
                  else value[begin:end]) for key, value in batch.items() if key != 'labels'}


def build_policy(args):
    from transformers import MT5ForConditionalGeneration

    architecture = json.loads(Path(args.initialization).read_text())
    encoder = load_model(architecture['source_pose_checkpoint'], torch.device('cpu'))
    text_model = MT5ForConditionalGeneration.from_pretrained(args.text_encoder, local_files_only=True)
    model = PoseMT5(encoder, text_model, label_smoothing=0)
    checkpoint = torch.load(str(args.init_slt_checkpoint), map_location='cpu', mmap=True, weights_only=False)
    config = checkpoint['identity']['config']
    if config.get('dataset') != 'How2Sign' or Path(config['data']).resolve() != Path(args.data).resolve():
        raise ValueError('RL source must be the selected How2Sign SFT model on the same data')
    if abs(checkpoint['validation']['bleu4'] - args.init_bleu4) > 1e-7:
        raise ValueError('RL source BLEU-4 differs from the explicitly selected SFT checkpoint')
    provenance = restore_translation_model(model, checkpoint, args.init_slt_step)
    del checkpoint
    return GRPOPolicy(model).eval(), provenance


def train(args):
    from transformers import T5Tokenizer

    torch.set_num_threads(1)
    rank, local, world = [int(os.environ.get(key, default)) for key, default in
                          (('RANK', 0), ('LOCAL_RANK', 0), ('WORLD_SIZE', 1))]
    if world * args.clips_per_rank * args.gradient_accumulation != args.global_clips_per_update:
        raise ValueError('distributed batch must contain exactly the configured global number of clips')
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=45))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = T5Tokenizer.from_pretrained(args.text_encoder, local_files_only=True, legacy=False)
    collate = TranslationCollator(tokenizer, args.max_text_length)
    training = TranslationDataset(args.data, 'train', args.max_frames, args.seed, args.limit_samples)
    validation = TranslationDataset(args.data, 'validation', args.max_frames, args.seed, args.eval_limit)
    policy, provenance = build_policy(args)
    # Fairseq's legacy weight-normalized layers cannot be deep-copied safely.
    reference, _ = build_policy(args)
    suppressed_tokens = policy.restrict_tokenizer_vocabulary(tokenizer)
    if reference.restrict_tokenizer_vocabulary(tokenizer) != suppressed_tokens:
        raise ValueError('policy and reference output vocabularies differ')
    reference = reference.requires_grad_(False).to(device).eval()
    policy = policy.to(device).eval()
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1., (step + 1) / args.warmup_steps))
    local_clips = args.clips_per_rank * args.gradient_accumulation
    sampler = DistributedSampler(training, world, rank, shuffle=True, seed=args.seed, drop_last=True)
    updates_per_epoch = len(sampler) // local_clips
    if updates_per_epoch < 1:
        raise ValueError('not enough training clips for one global update')
    identity = None
    if rank == 0:
        source_sha256 = sha256(args.init_slt_checkpoint)
        if source_sha256 != args.init_slt_sha256:
            raise ValueError('immutable SFT/reference checkpoint checksum changed')
        config = {key: value for key, value in vars(args).items() if key != 'stop_at_timestamp'}
        files = [Path(__file__), ROOT / 'shubert_pose/grpo.py', ROOT / 'shubert_pose/translation.py',
                 ROOT / 'shubert_pose/translation_data.py', ROOT / 'scripts/train_pose_mt5.py',
                 Path(args.data) / 'ready.json', Path(args.initialization)]
        identity = {'config': config, 'world_size': world, 'source_sha256': source_sha256,
                    'files': {str(path): sha256(path) for path in files}, 'source': provenance,
                    'train_samples': len(training), 'validation_samples': len(validation),
                    'global_clips_per_update': local_clips * world,
                    'rollouts_per_update': local_clips * world * args.group_size,
                    'updates_per_epoch': updates_per_epoch,
                    'method': 'separately normalized BLEU + 0.5 consensus advantages; clipped GRPO + k3 KL',
                    'validation_role': validation.metadata['evaluation_split'] + '; no independent test',
                    'dropout': 'disabled for sampling and gradient-enabled policy scoring',
                    'suppressed_token_ids': suppressed_tokens,
                    'reference': 'frozen complete SFT starting model; never updated'}
        existing = output / 'run_config.json'
        if existing.exists() and json.loads(existing.read_text()) != identity:
            if (output / 'checkpoint_last.pth').exists() or (output / 'metrics.jsonl').exists():
                raise ValueError('RL output belongs to a different experiment/configuration')
            # A startup/smoke failure before any saved update may be repaired.
            # Once a step is saved, resume identity must remain exact.
            atomic_json(output / 'previous_run_config.json', json.loads(existing.read_text()))
        atomic_json(existing, identity)
        atomic_json(output / 'slt_initialization.json', provenance)
        print('RL_SOURCE_LOADED', json.dumps(provenance), flush=True)
    if parallel():
        shared = [identity]
        dist.broadcast_object_list(shared, src=0)
        identity = shared[0]
    step, start_epoch, start_batch, best, last_eval, skipped = 0, 0, 0, -1., None, 0
    last = output / 'checkpoint_last.pth'
    if last.exists():
        saved = torch.load(str(last), map_location='cpu', mmap=True, weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('RL resume source/configuration changed')
        policy.model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        step, start_epoch, start_batch, best, last_eval, skipped = [saved[key] for key in
            ('step', 'epoch', 'next_batch', 'best_score', 'last_eval', 'skipped_updates')]
        restore_rng(saved['rng'][rank], device)
        del saved
    elif args.mode != 'smoke':
        last_eval = evaluate(policy.model, validation, collate, args, rank, world, device)
        policy.eval()
        if abs(last_eval['bleu4'] - args.init_bleu4) > 1e-7:
            raise ValueError('full initial validation failed to reproduce the selected SFT BLEU-4')
        best = last_eval['bleu4']
        if rank == 0:
            atomic_json(output / 'initial_eval.json', {'step': 0, **last_eval})
            atomic_save(output / 'checkpoint_best.pth', {'model': policy.model.state_dict(),
                        'step': 0, 'epoch': 0, 'validation': last_eval, 'identity': identity})
            print('INITIAL_VALIDATION', json.dumps(last_eval), flush=True)
    if not last.exists():
        torch.manual_seed(args.seed + rank)
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
    ddp = DistributedDataParallel(policy, device_ids=[local], broadcast_buffers=False) if parallel() else policy
    stop_requested = [False]
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, lambda *_: stop_requested.__setitem__(0, True))

    def save(epoch, batch, improved=False):
        states = [rng_state(device)]
        if parallel():
            states = [None] * world
            dist.all_gather_object(states, rng_state(device))
        if rank == 0:
            value = {'model': policy.model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'step': step, 'epoch': epoch, 'next_batch': batch,
                     'best_score': best, 'last_eval': last_eval, 'skipped_updates': skipped,
                     'rng': states, 'identity': identity}
            atomic_save(last, value)
            if improved:
                atomic_save(output / 'checkpoint_best.pth', {'model': value['model'], 'step': step,
                            'epoch': epoch, 'validation': last_eval, 'identity': identity})
            atomic_json(output / 'checkpoint_state.json', {key: value[key] for key in
                        ('step', 'epoch', 'next_batch', 'best_score', 'last_eval')})
        if parallel():
            dist.barrier()

    initial_projection = policy.model.projection.weight.detach().clone() if args.mode == 'smoke' else None
    reference_projection = reference.model.projection.weight.detach().clone()
    recent = []
    if rank == 0:
        print('RL_TRAIN_START', json.dumps({'resume_step': step, 'updates_per_epoch': updates_per_epoch,
              'epochs': args.epochs, 'global_clips_per_update': args.global_clips_per_update,
              'group_size': args.group_size, 'total_updates': updates_per_epoch * args.epochs}), flush=True)
    for epoch in range(start_epoch, args.epochs):
        training.set_epoch(epoch)
        sampler.set_epoch(epoch)
        beginning = start_batch if epoch == start_epoch else 0
        loader = DataLoader(training, batch_sampler=RemainingBatches(sampler, local_clips, beginning),
                            num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
                            generator=torch.Generator().manual_seed(args.seed + epoch + rank))
        for index, batch in enumerate(loader, beginning):
            policy.eval()
            reference.eval()
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            tensors = move(batch, device)
            statistics = []
            rewards = []
            for begin in range(0, local_clips, args.clips_per_rank):
                inputs = slice_inputs(tensors, begin, begin + args.clips_per_rank)
                with torch.no_grad(), amp(device, args.bf16):
                    sequences = policy.model.generate(**inputs, do_sample=True, num_beams=1,
                        num_return_sequences=args.group_size, temperature=args.temperature, top_k=0, top_p=1.,
                        max_new_tokens=args.rollout_max_tokens).clone()
                decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)
                groups = [decoded[j:j + args.group_size] for j in range(0, len(decoded), args.group_size)]
                bleu, consensus = grouped_rewards(groups, batch['texts'][begin:begin + args.clips_per_rank])
                advantages = group_advantages(bleu, consensus, args.consensus_weight, args.reward_epsilon).to(device)
                for clip in range(args.clips_per_rank):
                    group = sequences[clip * args.group_size:(clip + 1) * args.group_size]
                    clip_inputs = slice_inputs(inputs, clip, clip + 1)
                    mask = completion_mask(group, tokenizer.eos_token_id)
                    with torch.no_grad(), amp(device, args.bf16):
                        old_log_probs = policy(**clip_inputs, sequences=group).detach()
                        reference_log_probs = reference(**clip_inputs, sequences=group).detach()
                    synchronize = begin + clip + 1 == local_clips
                    context = ddp.no_sync() if parallel() and not synchronize else contextlib.nullcontext()
                    with context:
                        with amp(device, args.bf16):
                            log_probs = ddp(**clip_inputs, sequences=group)
                            loss, metrics = grpo_loss(log_probs, old_log_probs, reference_log_probs,
                                advantages[clip], mask, beta=args.beta, clip=args.clip_epsilon)
                        if not finite(loss):
                            raise FloatingPointError('non-finite grouped policy/KL loss')
                        (loss / local_clips).backward()
                    statistics.append({key: float(value) for key, value in metrics.items()})
                    rewards.append({'bleu_reward': float(bleu[clip].mean()),
                                    'consensus_reward': float(consensus[clip].mean()),
                                    'advantage_std': float(advantages[clip].std()), 'loss': float(loss.detach())})
                    if rank == 0 and step == 0 and begin == 0 and clip == 0:
                        atomic_json(output / 'first_rollout.json', {'clip': batch['ids'][0],
                            'reference': batch['texts'][0], 'hypotheses': groups[0],
                            'bleu_rewards': bleu[0].tolist(), 'consensus_rewards': consensus[0].tolist(),
                            'advantages': advantages[0].tolist(), 'mask_lengths': mask.sum(-1).tolist()})
                    del log_probs, old_log_probs, reference_log_probs, loss
            gradient_modules = (policy.model.encoder, policy.model.projection, policy.model.mt5_model)
            nonzero_gradients = [any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                                    for p in module.parameters()) for module in gradient_modules]
            grad = torch.nn.utils.clip_grad_norm_(parameters, args.clip_grad)
            applied = finite(grad)
            lr_used = optimizer.param_groups[0]['lr']
            if applied:
                optimizer.step()
                scheduler.step()
                step += 1
            else:
                skipped += 1
                if skipped > 20:
                    raise FloatingPointError('too many non-finite RL updates')
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            recent = (recent + [time.monotonic() - started])[-30:]
            at_end = index + 1 == updates_per_epoch
            smoke_end = args.max_steps > 0 and step >= args.max_steps
            improved = False
            if index + 1 in {(updates_per_epoch + 1) // 2, updates_per_epoch} or smoke_end:
                last_eval = evaluate(policy.model, validation, collate, args, rank, world, device)
                policy.eval()
                improved = last_eval['bleu4'] > best
                best = max(best, last_eval['bleu4'])
                if rank == 0:
                    print('RL_VALIDATION', json.dumps({'step': step, **last_eval}), flush=True)
            stop = (stop_requested[0] or smoke_end or (output / 'STOP').exists()
                    or (args.stop_at_timestamp and time.time() >= args.stop_at_timestamp))
            flag = torch.tensor(int(bool(stop)), device=device)
            if parallel():
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            stop = bool(flag.item())
            next_epoch, next_batch = (epoch + 1, 0) if at_end else (epoch, index + 1)
            if step == 1 or at_end or stop or improved or (applied and step % args.save_every == 0):
                save(next_epoch, next_batch, improved)
            if rank == 0:
                combined = [{**stat, **reward} for stat, reward in zip(statistics, rewards)]
                means = {key: sum(row[key] for row in combined) / len(combined) for key in combined[0]}
                status = {'state': 'paused' if stop else 'running', 'step': step,
                          'epoch': epoch + (index + 1) / updates_per_epoch,
                          **means, 'grad_norm': float(grad) if torch.isfinite(grad) else None,
                          'update_applied': applied, 'skipped_updates': skipped, 'lr_used': lr_used,
                          'lr': optimizer.param_groups[0]['lr'], 'encoder_lr': optimizer.param_groups[0]['lr'],
                          'global_clips_per_update': args.global_clips_per_update,
                          'rollouts_per_update': args.global_clips_per_update * args.group_size,
                          'seconds_per_update': sum(recent) / len(recent),
                          'remaining_train_seconds': (updates_per_epoch * args.epochs
                              - epoch * updates_per_epoch - index - 1) * sum(recent) / len(recent),
                          'best_bleu4': best, 'last_eval': last_eval, 'next_epoch': next_epoch,
                          'next_batch': next_batch, 'gradient_flow_encoder_projection_mt5': nonzero_gradients,
                          'gpu_peak_gib': torch.cuda.max_memory_allocated(device) / 2**30,
                          'updated_unix': time.time()}
                atomic_json(output / 'progress.json', status)
                with (output / 'metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(status, allow_nan=False) + '\n')
                if step <= 3 or step % args.log_every == 0 or stop:
                    print('RL_TRAIN_METRICS', json.dumps(status), flush=True)
            if stop:
                if args.mode == 'smoke':
                    if torch.equal(initial_projection, policy.model.projection.weight.detach()):
                        raise RuntimeError('RL projection did not update')
                    if not all(nonzero_gradients):
                        raise RuntimeError('RL gradients did not reach encoder, projection and mT5')
                    if (any(p.requires_grad or p.grad is not None for p in reference.parameters())
                            or not torch.equal(reference_projection, reference.model.projection.weight.detach())):
                        raise RuntimeError('frozen RL reference changed')
                    restored = torch.load(str(last), map_location='cpu', mmap=True, weights_only=False)
                    policy.model.load_state_dict(restored['model'], strict=True)
                    optimizer.load_state_dict(restored['optimizer'])
                    scheduler.load_state_dict(restored['scheduler'])
                    if restored['step'] != step:
                        raise RuntimeError('RL checkpoint step mismatch')
                    if rank == 0:
                        atomic_json(output / 'smoke_result.json', {'status': 'passed', 'steps': step,
                            'global_clips_per_update': args.global_clips_per_update,
                            'rollouts_per_update': args.global_clips_per_update * args.group_size,
                            'finite_loss_and_gradients': True, 'encoder_projection_mt5_gradients': True,
                            'frozen_reference_unchanged': True, 'checkpoint_and_optimizer_restored': True})
                if parallel():
                    dist.destroy_process_group()
                return
        start_batch = 0
    if rank == 0:
        status = json.loads((output / 'progress.json').read_text())
        status.update(state='complete', updated_unix=time.time())
        atomic_json(output / 'progress.json', status)
        atomic_json(output / 'completed.json', status)
        print('RL_TRAIN_COMPLETE', step, flush=True)
    if parallel():
        dist.destroy_process_group()


if __name__ == '__main__':
    options = parse_args()
    try:
        train(options)
    except BaseException as error:
        rank = int(os.environ.get('RANK', 0))
        output = Path(options.output)
        output.mkdir(parents=True, exist_ok=True)
        failure = {'error': repr(error), 'rank': rank, 'updated_unix': time.time()}
        atomic_json(output / f'failure_rank{rank}.json', failure)
        if rank == 0:
            atomic_json(output / 'failure.json', failure)
        raise
