# Pose-only SHuBERT

This variant learns contextual representations from coordinate sequences. Each
anatomical stream has a normalization layer and a linear projection. Masked
stream features are replaced with learned embeddings; the concatenated streams
enter the same Fairseq temporal Transformer family used by SHuBERT. Separate
heads predict offline k-means cluster IDs, with cross-entropy evaluated only at
masked, non-padding positions.

The implementation is in `shubert_pose/`, registered as model `shubert_pose` and
task `pose_pretraining`. The original four-stream implementation is unchanged.
This is a new trainable variant; no trained pose-only weights or downstream
accuracy results are provided. Original SHuBERT checkpoints are not directly
compatible with its input projections and prediction heads.

## Coordinate inputs

The default in `configs/pose.yaml` is three streams:

| Stream | Default dimensions per frame | Meaning |
| --- | --- | --- |
| `body` | 14 | Seven upper-body landmarks, flattened 2D coordinates |
| `left_hand` | 42 | 21 left-hand landmarks, flattened 2D coordinates |
| `right_hand` | 42 | 21 right-hand landmarks, flattened 2D coordinates |

For a new four-stream pretraining experiment, `scripts/prepare_youtubeasl_pack.py
--include-face` additionally exports the sign69 pack's 18 facial landmarks as
`face: 36`, fits a separate face codebook using training frames only, and writes
four-stream manifests. Use a fresh output directory. Face coordinates are
relative to the packed facial nose point and scaled by half the shoulder width;
frames without a reliable face anchor are zeroed. These 18 points cover jaw,
inner lips, and nose; they do not include eyes or eyebrows.

Train that new pack with `train_pose.sh` and the overrides
`'task.stream_names=[body,left_hand,right_hand,face]'` and
`'task.stream_dims=[14,42,42,36]'`. Existing three-stream checkpoints require
explicit initialization adaptation or a new four-stream training run. This
does not retrofit face information into the existing translation/RL weights.
On 2026-11-17, nine packed-data tests passed, including four-stream packing and
clustering, nonzero face-projection/head gradients, missing-face handling, and
finite, aligned coordinates from a real YouTubeASL clip. Full four-stream
pretraining and downstream translation have not yet been run.

`scripts/convert_rtmpose.py` produces exactly these streams from COCO-WholeBody-133
RTMPose pickles (see below). For any other source, the following applies.

These are input contracts, not an additional pose detector. Prepare one finite
float `.npy` array of shape `[T, D]` per clip and stream. Use the same clip IDs,
frame timestamps, coordinate convention, and landmark order across the dataset.
Resolve missing detections and normalize coordinates consistently before
clustering and training. The code does not infer handedness or replace missing
coordinates. In particular, the existing MediaPipe export's unordered hand list
must not be assumed to be anatomical left/right.

You can use the existing 14D body-feature arrays alone, 3D joints, or another
flattened pose representation by changing `task.stream_names` and
`task.stream_dims`. The model settings follow these task values automatically.
The number of streams is arbitrary; every configured stream gets its own cluster
head. A body-only run has one head.

## Convert RTMPose-133 pickles

For one pickle per clip holding `keypoints` (`[T, 1, 133, 2]` or `[T, 133, 2]`,
pixels) and `scores`, with optional `start`/`end` frame bounds:

```bash
python scripts/convert_rtmpose.py --input /path/to/pose_pkls --output data/pose_streams --workers 16
```

This writes `body/`, `left_hand/` and `right_hand/` under the output directory,
ready for `prepare_pose.py manifest`. Nested inputs get IDs such as
`sub__dir__clip`. Existing outputs are kept, so an interrupted run can be
restarted; unreadable pickles are listed in `failed.txt`.

- `body` follows `features/body_features.py`: nose, shoulders, elbows and wrists
  in a signing-space box of 6 x 7 head units (one unit is half the shoulder
  width), centred on the signer. Joints below the confidence threshold hold
  their last confident value.
- Each hand is expressed relative to its own wrist and divided by the same head
  unit. Hand position is therefore carried only by the body stream's wrists, and
  the hand codebooks describe handshape and orientation instead of where the
  hand is in the frame. This matters because the model applies a per-frame
  LayerNorm to every stream, which removes a hand's absolute position anyway.
- A hand whose wrist or most joints are below the threshold is all zeros for
  that frame, so "no hand" stays a distinct state.

## Prepare manifests and targets

Activate `shubert_train_env` and install the bundled Fairseq as documented in
`QUICKSTART.md`. Run the following from the repository root. Add the local
Fairseq to the import path, including for preprocessing:

```bash
export PYTHONPATH="$PWD/fairseq:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Create a manifest using exact clip IDs from a split file (one ID per line):

```bash
python scripts/prepare_pose.py manifest \
  --stream body=/path/to/body_coordinates \
  --stream left_hand=/path/to/left_hand_coordinates \
  --stream right_hand=/path/to/right_hand_coordinates \
  --ids /path/to/train_ids.txt \
  --output data/pose/train.tsv
```

Repeat with the validation IDs and `--output data/pose/valid.tsv` if needed.
Without `--ids`, all stream directories must contain exactly the same IDs.
The builder checks dimensions, frame counts, and finite values, and writes a
tab-separated header and rows:

```text
id    body    left_hand    right_hand    frames
clip1    /absolute/path/body/clip1.npy    /absolute/path/left/clip1.npy    /absolute/path/right/clip1.npy    120
```

Fit codebooks **only on the training split**, then assign targets to the desired
splits using those same codebooks:

```bash
python scripts/prepare_pose.py cluster \
  --data data/pose --streams body left_hand right_hand \
  --splits train valid --clusters 256 --sample-frames 100000 --seed 0
```

For training data alone, use `--splits train`. This produces one
`SPLIT.STREAM.km` file per split and stream, with one label row per manifest row,
and saves the fitted codebooks and sampling metadata under `codebooks/`.
The loader checks recorded manifest hashes, stream dimensions, and cluster count
when that metadata is present, so a reordered manifest cannot reuse stale labels.
Sampling is uniform over training frames and bounds the matrix used for fitting;
individual coordinate files must still fit in memory during label assignment.
Regenerate labels whenever the manifest order, coordinates, or codebook changes.

## Train

The launcher runs in the foreground using the active environment's `python`:

```bash
bash train_pose.sh data/pose outputs/pose/ckpt
```

For an initial short run:

```bash
bash train_pose.sh data/pose outputs/pose_smoke/ckpt \
  optimization.max_update=2 lr_scheduler.warmup_updates=0 \
  dataset.num_workers=0 dataset.max_tokens=128 \
  task.max_frames=64 model.encoder_layers=2 \
  model.encoder_embed_dim=128 model.encoder_ffn_embed_dim=256 \
  model.encoder_attention_heads=4 model.conv_pos_groups=4 \
  model.encoder_layerdrop=0.0
```

Variable-length batches use right padding. The loader applies identical temporal
crops to coordinates and targets; padding is excluded from masking and loss.
`model.mask_strategy=random` samples temporal spans independently per stream;
`time` shares the temporal mask across streams; `flat` follows the original
model and samples spans over the frame-major `[frames * streams]` sequence. With
the default span of 3, a `flat` span mostly covers neighbouring streams of one
frame, so a masked value is usually still visible in the adjacent frames
(about 2.5 consecutive masked frames per stream, against about 5.5 for
`random`, at the same 60% masked ratio). `random` is the harder task and stays
the default; use `model.mask_strategy=flat` to match the original. Every
strategy keeps observed context for sequences longer than one frame.

`configs/pose.yaml` keeps the original schedule, which assumes 8 GPUs. It sets
`optimization.update_freq=[8]` for one GPU; keep
`distributed_world_size * update_freq = 8`, for example
`distributed_training.distributed_world_size=4 'optimization.update_freq=[2]'`.
Mixed precision is on (`common.fp16=true`).

For **body-only** input, build a manifest with just `--stream body=...`, cluster
with `--streams body`, and train with:

```bash
bash train_pose.sh data/pose_body outputs/pose_body/ckpt \
  'task.stream_names=[body]' 'task.stream_dims=[14]'
```

For a single 133D input array, for example, name that stream `pose` and use
`'task.stream_names=[pose]' 'task.stream_dims=[133]'`. This does not perform a
representation conversion. Codebooks must be fitted on the same representation.

## YouTubeASL trial on the current cluster

The machine-specific launchers in `scripts/pose_runtime.sh` and
`scripts/train_youtubeasl_pose_trial.sh` reuse SLiM's Python 3.9/PyTorch runtime
with an isolated Hydra 1.0.7/OmegaConf 2.0.6 installation in `.venv/`.
The prepared trial uses 4,096 training clips and a 128-clip, video-disjoint
holdout drawn from SLiM's YouTubeASL training manifest. Three 256-way codebooks
are fitted on 100,000 training frames. Generated files live in
`/scratch/lidowu91/SHuBERT/outputs/youtubeasl_pose_20261113`; the repository's
`outputs` path links to scratch to avoid the home-directory quota.

The trial keeps the default 12-layer, 768-dimensional encoder and runs 200
updates on four GPUs. It uses `dataset.num_workers=0`: spawning workers with
the in-memory per-clip label tensors exceeded the file-descriptor limit in the
initial attempt. That attempt failed before a training update or checkpoint.

To start automatically when an existing four-GPU reservation begins, submit:

```bash
sbatch --dependency=after:GPU_JOB_ID --kill-on-invalid-dep=yes \
  scripts/autostart_pose_trial.sbatch GPU_JOB_ID
```

This submits a small CPU launcher that runs the trial inside the existing GPU
allocation. After training it strictly reloads the checkpoint, extracts layer
12 for the holdout, and checks the update count, finite weights, clip/frame
alignment, and `[1, T, 768]` feature shapes. Success is recorded in
`verification.json`; execution logs are `autostart-JOB_ID.log` and
`auto_train.log`. The GPU reservation remains available after the trial.

## WLASL checkpoint comparison

`scripts/run_wlasl_pose.sh` evaluates the retained formal-pretraining snapshots
at epochs 141, 142, and 143, with one independent run per GPU. It uses the
published downstream hyperparameters from
[SHuBERT, section 4.3](https://aclanthology.org/2025.acl-long.1397.pdf): rank-1
LoRA in the feature extractor's linear layers, frozen original weights,
temporal mean pooling, a BatchNorm plus linear classification head, Adam,
batch size 128, head learning rate 0.0001, LoRA learning rate 0.00001,
weight decay 0.0001, label smoothing 0.1, and at most 125 epochs. Selection
normally uses validation per-instance Top-1, followed by a test evaluation.
Per-instance and per-class Top-1 are reported, with Top-5/10 as extra metrics.

This applies the paper's disclosed downstream settings to the coordinate-only
variant, whose input features differ from the original RGB/DINO model. The
paper does not specify all implementation details: this runner uses seed 0,
LoRA alpha 1, constant learning rates, final-layer residual hidden states,
all available frames without extra augmentation, and a patience of 20
non-improving validation epochs. It disables encoder layer dropping and uses
bfloat16 autocast with activation checkpointing. These choices are recorded
in each run's `config.json`.

`scripts/prepare_wlasl_pose.py` reuses the formal pretraining run's exact
`sign69-nose-centered-v1` normalization. It preserves the existing WLASL2000
splits and vocabulary, checks for duplicate/overlapping IDs, records source
hashes, and retains all 14,289 training, 3,916 validation, and 2,878 test clips.
The evaluator first checks attention-adapter gradients and frozen weights, then
runs a real-data GPU smoke check before starting the three runs. Fairseq's
functional attention reads adapted weights directly; adapters therefore apply
both in its functional fast path and through ordinary linear forward calls.

Outputs are under
`/scratch/lidowu91/SHuBERT/outputs/wlasl_pose_e141_143_20261115/` for the initial
dev-selection phase. The user subsequently requested using the test set as the
validation set. The active run is now in the sibling directory
`wlasl_pose_e141_143_20261115_test_as_dev/`, with `--selection-split test`.
It preserves and resumes the adapters and optimizer state from the initial
phase, resets the selection history, evaluates a selection baseline on the
resumed state, and selects subsequent epochs by test per-instance Top-1. These
scores are explicitly `test_as_dev`, with `independent_test=false`.
`transition.json` and `selection_baseline.json` record the protocol transition;
no earlier test-selection history is claimed.
Each `epochNNN/` contains resumable adapters and optimizer state,
`metrics.jsonl`, `progress.json`, and, on completion, `test_metrics.json` and
test predictions with original clip IDs. The active run writes
`test_as_dev_metrics.json` instead of `test_metrics.json`.
`comparison.json` is written after
all three runs finish. Early pretraining epochs were removed by the original
rolling retention policy; the 4,096-clip trial snapshots are not included in
this formal-checkpoint comparison.

### Full fine-tuning and translation initialization

`scripts/run_wlasl_pose_fullft.sh` also supports full encoder fine-tuning,
without LoRA adapters. The three completed 125-epoch runs are in
`/scratch/lidowu91/SHuBERT/outputs/wlasl_pose_fullft_e141_143_20261115_test_only_selection`.
After the protocol transition at fine-tuning epoch 33, selection uses only
WLASL test per-instance Top-1, as requested. These are test-as-validation
results, with `independent_test=false`:

| Pretraining epoch | Selected fine-tuning epoch | P-I Top-1 | P-C Top-1 |
| --- | --- | --- | --- |
| 141 | 124 | 57.3315 | 54.5256 |
| 142 | 112 | 57.7137 | 55.2259 |
| 143 | 122 | 57.8874 | 55.0884 |

The comparison verifier checked prediction coverage for all 2,878 test clips,
the 1,999 represented classes, and the full encoder update contract.
`verification.json` records the result. The best P-I checkpoint, pretraining
epoch 143 followed by fine-tuning epoch 122, initializes the translation run
below. Its WLASL classifier is discarded.

## YouTubeASL translation with mT5

`scripts/run_pose_mt5.sh` attaches a learned 768-to-768 projection and local
`mt5-base` to the selected, fully fine-tuned pose encoder. All feature encoder,
projection, and mT5 weights train together; unused cluster heads and mask
embeddings remain frozen. Initialization snapshots and hashes are saved before
training, and feature weights must load with exactly the expected keys.
This follows SLiM's translation training settings with a translation
cross-entropy objective; it does not add SLiM's optimal-transport objective.

`scripts/prepare_pose_translation.py` pairs the SLiM YouTubeASL caption manifest
with the full normalized pose pack. Of 565,040 source pairs, 564,409 have usable
poses. OpenASL **test** supplies all 976 validation clips, using the same
`sign69-nose-centered-v1` normalization. OpenASL clips never contribute training
gradients. Their BLEU-4 selects checkpoints, so this run has no independent
OpenASL test score.

`configs/translation_mt5.json` sets 20 epochs, four-GPU DDP, batch size 24 per
GPU (96 globally), at most 256 pose frames and 128 target tokens, bfloat16,
and activation checkpointing. AdamW uses learning rates 0.0002 for the
projection/mT5 and 0.00002 for the pose encoder, weight decay 0.0001,
200 warmup updates, cosine decay, gradient clipping at 1, and label smoothing
0.2. Validation uses four beams and SacreBLEU's `13a` tokenizer every 2,000
updates and at each epoch end, checking exact clip-ID and caption coverage.

Outputs are in
`/scratch/lidowu91/SHuBERT/outputs/pose_mt5_youtubeasl_20261115`:

- `initialization/` holds immutable source snapshots and their provenance.
- `smoke/` records a separate, production-size four-GPU check: two optimizer
  updates, finite loss/gradients, a changed projection, generation, and strict
  checkpoint/optimizer restoration. These updates are not reused by training.
- `train/progress.json` and `train/metrics.jsonl` record actual training updates.
- `train/checkpoint_last.pth` saves model, optimizer, scheduler, each rank's RNG,
  sampler epoch, and next batch every 1,000 updates, at epoch ends, on a new
  best score, and when pausing. It replaces the previous last checkpoint.
- `train/checkpoint_best.pth` keeps the model with the highest validation
  BLEU-4. `validation_predictions.json` contains the latest validation output.
- `train/completed.json` is written only after all 20 epochs finish.

The launcher holds a run lock and validates source, data, code, and configuration
hashes when resuming. Under SLURM it saves ten minutes before allocation expiry
and queues `scripts/resume_pose_mt5.sbatch` on four GPUs with an `afterany`
dependency. Completion cancels the unused successor. Re-running the launcher
resumes from the last saved batch; a `train/STOP` file requests a checkpointed
pause. Remove that file before resuming intentionally.

On 2026-11-16 the separate four-GPU smoke check passed, then the formal run
started with all 564,409 training pairs and 976 validation clips. The first
343 updates completed without skipped updates. This establishes training and
checkpoint functionality; it is not a completed translation result.
`tests/test_pose_translation.py` additionally covers exact full-fine-tune
restoration, padding invariance, encoder/projection/text gradients, pairing,
temporal sampling, token padding, and mmap reopening.

## OpenASL SFT from the selected translation checkpoint

The YouTubeASL translation run was stopped on 2026-11-16 after saving step
94,859 (16.135 epochs). OpenASL SFT starts from its retained best full model,
step 64,000 (10.886 epochs), with validation BLEU-4 16.3625. The encoder,
projection, and mT5 state are restored strictly; optimizer and schedule start
fresh. The original best checkpoint is copied into the SFT initialization
directory before training.

`scripts/prepare_openasl_translation.py` converts the 96,476 OpenASL training
records with `sign69-nose-centered-v1`, records every unusable training clip,
and rejects excessive exclusions and overlapping train/validation clip IDs.
It reuses the exact 976 normalized OpenASL test clips from translation
pretraining as validation. This follows the earlier OpenASL experiment's
test-as-validation protocol; scores are not independent test results.

`configs/openasl_pose_mt5_sft.json` requests 20 epochs, batch 24 per GPU on four
GPUs, learning rate 0.0001 for projection/mT5 and 0.00001 for the pose encoder.
The launcher runs the focused tests and a separate two-update, four-GPU smoke
check before the formal run. The formal run evaluates its initialization and
keeps that baseline as the first best checkpoint, so a worse SFT model cannot
replace it. Outputs are under
`/scratch/lidowu91/SHuBERT/outputs/openasl_pose_mt5_sft_20261116`.

GPU allocation is independent of this workload:

```bash
sbatch --parsable scripts/reserve_four_gpus.sbatch
sbatch --dependency=after:GPU_JOB_ID --kill-on-invalid-dep=yes \
  scripts/autostart_openasl_pose_mt5.sbatch GPU_JOB_ID
```

The reservation holds four GPUs for 24 hours even after SFT finishes or fails.
Only its training step and small CPU launcher exit when SFT ends. The training
launcher saves ten minutes before the reservation expires; it does not request
another allocation. Reservation 36160306 and dependent CPU launcher 36160313
were submitted on 2026-11-16. Submission is not evidence that SFT has started.

### OpenASL to How2Sign automatic handoff

`scripts/run_openasl_pose_mt5_sft.sh` now runs two successive 20-epoch SFT
stages on the same independent GPU reservation. Both stages use their dataset's
train split for gradients and test split for validation/checkpoint selection,
as requested. OpenASL uses 976 validation clips; How2Sign uses all 2,349 test
clips. Neither stage reports an independent test score.

How2Sign cannot start until OpenASL writes `train/completed.json` with
`state=complete`. `scripts/prepare_how2sign_sft.py` checks that
`train/checkpoint_best.pth` belongs to this exact OpenASL run, that its BLEU-4
equals the completed run's best score, and that it contains encoder,
projection, and mT5 state. It snapshots and hashes the selected model, writes
`handoff.json` and a concrete `sft_config.json`, and starts a fresh optimizer
and schedule. It never substitutes the last checkpoint for the best model.

`scripts/prepare_how2sign_translation.py` normalizes the 31,086 train records
and all test records with the same three-stream coordinate contract. Training
exclusions are recorded, validation failures stop the run, and a preparation
lock prevents overlapping writers. `scripts/run_how2sign_pose_mt5_sft.sh` runs
the separate four-GPU smoke check before formal training. How2Sign outputs are
under `/scratch/lidowu91/SHuBERT/outputs/how2sign_pose_mt5_after_openasl_20261116`.

On 2026-11-16, all five translation tests passed, including rejection of an
unfinished run, a non-best source, the wrong source dataset/path, and an
incomplete model. One real How2Sign training clip and one real test clip were
also converted with finite, aligned `[T,14]`, `[T,42]`, `[T,42]` outputs. These
are input and handoff checks, not completed SFT results.

For a detached supervised launch inside the reservation, use:

```bash
python3 scripts/start_openasl_pose_sft.py --gpu-job GPU_JOB_ID --stage how2sign
```

`scripts/supervise_pose_sft.py` detects nonzero step exits and missing progress,
retries up to three times using the saved SFT state, and writes the attempt,
exit code, and final state to the stage's `supervisor.json`. It respects an
explicit STOP file or paused checkpoint and never cancels the GPU reservation.
Startup preparation has a 45-minute grace period; a running training stage has
a 20-minute progress timeout. Exhausted retries are recorded as `failed`.

On 2026-11-17 the How2Sign handoff's PyTorch 2.1 memory-mapped load was corrected
to pass a string filename instead of a `Path`. A real serialized checkpoint
handoff regression test now covers snapshot checksums and the generated SFT
configuration. All six translation tests and both supervisor recovery tests
passed. The actual selected OpenASL model (step 6,000, BLEU-4 22.9031) was
snapshotted with checksum verification, and the four-GPU How2Sign smoke check
passed after this correction.

### Continue How2Sign SFT from a selected How2Sign model

The 2026-11-17 continuation explicitly selects the first How2Sign run's
step 3,876 model (epoch 12 validation BLEU-4 15.175059951878245), including
the pose encoder, projection, and mT5. `scripts/prepare_how2sign_continuation.py`
checks the dataset, data path, exact selected step/score, and all three model
components. It snapshots the selected checkpoint and records its SHA-256 before
training; subsequent source-run changes cannot change that snapshot.

`configs/how2sign_pose_mt5_continue.json` starts a fresh optimizer and cosine
schedule for 10 more epochs, keeping the original learning rates of 1e-4 for
projection/mT5 and 1e-5 for the pose encoder, and 200 warmup steps. Data, batch size, preprocessing,
and test-as-validation selection remain the same. The runner waits for the
original training step to release its lock, runs a separate four-GPU smoke
check, and then evaluates the entire test split before the first formal update.
The initial model is eligible for best-checkpoint selection even if further
training does not improve it.

```bash
python3 scripts/start_openasl_pose_sft.py --gpu-job GPU_JOB_ID --stage how2sign_continue
```

Outputs and recovery state are separate, under
`/scratch/lidowu91/SHuBERT/outputs/how2sign_pose_mt5_continue_best1518_20261117`.
The supervisor checks completion of this continuation, rather than the
already completed original run, and leaves the independent GPU reservation
running after training finishes.

## How2Sign grouped reinforcement learning

`scripts/rft_pose_grpo.py` implements the requested BLEU/consensus recipe for
the complete PoseSHuBERT + projection + mT5 model. Policy and frozen reference
both load the immutable How2Sign SFT step 3,876 snapshot (BLEU-4
15.175059951878245, SHA-256 recorded in `configs/how2sign_pose_grpo.json`).
The stopped SFT continuation separately reached 15.185968970398537 at its
third epoch; that later model is retained in its original output directory.

Each clip produces eight sampled completions, with temperature 1, `top_k=0`,
`top_p=1`, and at most 60 new tokens. Rewards are lowercased sacreBLEU sentence
BLEU divided by 100: against the training reference, and the average against
the other seven candidates. Each reward is standardized independently within
its group using sample standard deviation (`torch.std`, correction 1) plus
1e-4; advantages are `norm(BLEU) + 0.5 * norm(consensus)`.

The token objective uses the clipped ratio to the sampling policy (0.8--1.2)
and k3 KL to the fixed SFT reference, weighted by 0.02. EOS is included, the
decoder start and all tokens after EOS are excluded. Token losses are averaged
within each sentence, then over candidates and clips. The pose/mT5 encoder
output is shared across a clip's candidates without detaching it, so both
encoders, projection, and decoder receive gradients. Dropout is disabled for
sampling, reference scoring, and gradient-enabled policy scoring.

Four ranks each process two clips per microbatch and accumulate twice:
16 clips and 128 completions per optimizer update globally. AdamW uses 2e-6
for all trainable parameters, 20 linear warmup steps followed by a constant
learning rate, zero weight decay, and gradient clipping at 1.0. Each rollout
is used for one optimizer update. Two epochs contain 3,884 updates; the last
14 clips per epoch are dropped to keep complete global batches.

Full test-as-validation scoring runs initially and twice per epoch with beam
4. Its 128-token decoding limit preserves the previous SFT evaluation
protocol; the 60-token cap applies to sampled training rollouts. The initial
score must reproduce 15.175059951878245 before formal RL updates. The best
checkpoint starts from that baseline, and the last checkpoint includes the
optimizer, scheduler, and all ranks' random state for recovery.

```bash
python3 scripts/start_openasl_pose_sft.py --gpu-job GPU_JOB_ID --stage how2sign_grpo
```

`scripts/run_how2sign_pose_grpo.sh` waits for the stopped SFT step to release
its lock and runs a separate two-update, four-GPU smoke check before formal
training. Outputs are under
`/scratch/lidowu91/SHuBERT/outputs/how2sign_pose_mt5_grpo_bleu_consensus_v2_20261117`.
The independent GPU reservation stays running after the RL workload exits.
Seven focused tests cover rewards, separate normalization, EOS masking,
sentence averaging, clipping/KL, shared-encoder gradient flow, and CUDA BF16
backward outside autocast. These passed on 2026-11-17 using the actual runtime.
The actual four-GPU, two-update smoke check also passed, including a frozen
reference, gradients through all three model components, and checkpoint/optimizer
restoration. Full initial validation reproduced BLEU-4 15.175059951878245 over
all 2,349 test clips, and formal RL updates began successfully on 2026-11-17.

That initial attempt stopped after 24 updates: mT5's output vocabulary has
250,112 positions, while its tokenizer can decode only IDs 0 through 250,099.
Unrestricted stochastic sampling reached one of the twelve padded positions
and SentencePiece raised `IndexError`. The corrected runner suppresses every
unsupported ID during generation and applies exactly the same mask before
normalizing policy, old-policy, and frozen-reference log probabilities. Added
token IDs present in the tokenizer remain valid. Each failing rank now saves
its own error record. Regression tests force a padded token to dominate the
raw logits and check that sampling excludes it while valid scores and gradients
remain finite. Failed outputs are retained in the original
`how2sign_pose_mt5_grpo_bleu_consensus_20261117` directory. The separate v2
experiment starts again from the same selected SFT model with a fresh optimizer
and repeats the smoke check and full initial validation. All 10 RL tests passed,
including an actual mT5 tokenizer test with deliberately dominant invalid logits.
Four supervisor tests passed; exhausted retries now also mark existing training
progress as failed instead of leaving a stale `running` state.

### Compatibility with the supplied Uni-Sign `rft_grpo.py`

The active adaptation corresponds to this branch of the supplied implementation:

| Reference setting | Active PoseSHuBERT setting |
| --- | --- |
| `rag_mode=empty`, `alpha=0` | No retrieval utility reward; preserve the SFT prompt/pose input contract |
| `quality=bleu`, `cons_weight=0.5`, `cons_view=self` | Reference BLEU plus separately normalized leave-one-out consensus |
| `adv_norm=group`, sequence loss mean | Per-reward group standardization and per-sequence token means |
| `ppo_epochs=1`, `clip_eps=0.2`, `beta=0.02` | One update per rollout; same clipping and k3 KL |
| Single GPU: `batch_size=8`, `accum=2` | Four GPUs: two clips per rank, accumulation two; same 16 clips globally |
| `freeze_pose=False` | Pose encoder, projection, and mT5 all train |
| `bad_words_ids=no_piece` | Suppress padding IDs in generation and normalize scores on the same vocabulary |

Adapters use the existing paired coordinate packs and `PoseMT5` in place of
`S2T_Dataset_RAG` and `Uni_Sign_SignSMD`. The SFT input stays
`[translate prompt][pose]`; no extra empty-hint token is introduced. Retrieval
utilities, distillation, visual contrast, mirrored views, prefix-guided rollouts,
and alternative rewards from the supplied script are not enabled or claimed
as ported by this experiment.

## Extract features

### Full YouTubeASL, 150 epochs

`scripts/run_youtubeasl_pose_150.sh` prepares SLiM's complete raw sign69 pack
and trains a fresh pose model for 150 epochs in
`/scratch/lidowu91/SHuBERT/outputs/youtubeasl_pose_full_e150_20261114`.
The input has 598,661 clips and 69,273,524 frames. Preparation excludes 908
short or unusable clips and holds out all clips from 32 source videos:
595,492 training clips and 2,261 validation clips remain.

`scripts/prepare_youtubeasl_pack.py` writes continuous `[all frames, D]`
float32 coordinate arrays, uint16 cluster labels, clip offsets and split
indices. `train.pack.json` and `valid.pack.json` describe these files;
`PackedPoseDataset` checks the index/offset hashes, dimensions and cluster
count, then crops coordinates and labels together. Workers reopen memory maps
when spawned. The task selects this format when a `SPLIT.pack.json` exists.

The raw sign69 pack includes the nose, shoulders and hands but omits eye
landmarks. Its body stream uses a nose-centred frame, with horizontal/vertical
scales of six/seven half-shoulder-width units. Hands remain wrist-relative,
in half-shoulder-width units. New 256-way codebooks are fitted on 100,000
training frames using this normalization. The preparation metadata records
the convention as `sign69-nose-centered-v1`.

The full run uses four GPUs, `max_frames=256`, 15,000 frames per device,
gradient accumulation of two, and two data workers per rank. Fairseq's actual
batching gives 560 update groups per epoch, so the polynomial schedule spans
84,000 updates with 6,720 warmup updates. `optimization.max_epoch=150` is the
stopping condition and `optimization.max_update=0` removes the update cap.

Checkpoint policy: save every epoch and retain the latest three epoch files;
also save every 1,000 updates and retain the latest two interval files. Keep
`checkpoint_last.pt` and the lowest-validation-loss `checkpoint_best.pt`,
including optimizer and scheduler state for resumption. Running the same
launcher again restores the last checkpoint.

On 2026-11-15 all 23 focused tests passed, including packed crop/label
alignment, mmap serialization, a spawned data worker, invalid-label and
changed-index rejection, and sign69 normalization. The full four-GPU run
completed epoch 1, saved `checkpoint1.pt`, and entered epoch 2. Its first
validation loss was 3.826. This is evidence of an active training run, not a
completed 150-epoch result or downstream accuracy measurement.

Use a checkpoint produced by this variant, with its saved Fairseq configuration:

```bash
python features/pose_inference.py \
  --manifest data/pose/valid.tsv \
  --checkpoint outputs/pose/ckpt/checkpoint_last.pt \
  --output-dir outputs/pose/valid_features --layers 12 --device cuda
```

Inference requires coordinates only, without `.km` labels. It restores the model
configuration, loads weights strictly, disables masking, and saves Transformer
hidden states as `[selected layers, frames, hidden_dimension]` in `CLIP_ID.npy`,
plus `manifest.json`. `--layers` takes 1-based layer numbers (`12`, `4,8,12`) or
`all` (the default). All 12 layers of the base model cost about 37 KB per frame,
so select layers for anything larger than a benchmark split. These are the
residual-stream hidden states; the original `features/shubert_inference.py`
exports each layer's feed-forward branch output instead. CPU inference is
supported with `--device cpu`. Choose a fresh output directory, or pass
`--resume` to keep files already written with the same layer selection.

## Validation

```bash
python -m unittest discover -s tests -v
bash -n train_pose.sh
```

The focused tests cover masked-only loss, gradients through the coordinate
projections, padding invariance, short/body-only inputs, checkpoint restoration,
and alignment of filtered/cropped coordinates and labels. Synthetic smoke tests
establish functionality, not representation quality or downstream performance.

On 2026-11-13, all 18 focused tests passed with Python 3.9, PyTorch 2.1.1,
Hydra 1.0.7, and OmegaConf 2.0.6. A full 12-layer, 768-dimensional model also
completed a CPU forward/backward check on two real YouTubeASL clips cropped
to 32 frames, with finite loss and finite, nonzero gradients through all three
coordinate projections. This does not establish successful multi-GPU training;
the first attempt hit the DataLoader file-descriptor issue described above.

On 2026-10-21, the nine tests that existed then passed with Python 3.10, PyTorch 2.3.0,
Hydra 1.0.7, and OmegaConf 2.0.6. A synthetic CPU run fitted three codebooks on
three training clips, completed two Fairseq updates, reloaded the resulting
checkpoint strictly, and exported finite `[2, 9, 16]` hidden states for a separate
nine-frame clip. This used a reduced two-layer model, not the default full model.
