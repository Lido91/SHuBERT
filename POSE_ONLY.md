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

## Extract features

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

`tests/test_convert_rtmpose.py` needs only NumPy and passed on 2026-11-13. The
flat masking, layer selection and resume changes of that date, and their tests in
`tests/test_pose.py`, have not yet been run under Fairseq.

On 2026-10-21, the nine tests that existed then passed with Python 3.10, PyTorch 2.3.0,
Hydra 1.0.7, and OmegaConf 2.0.6. A synthetic CPU run fitted three codebooks on
three training clips, completed two Fairseq updates, reloaded the resulting
checkpoint strictly, and exported finite `[2, 9, 16]` hidden states for a separate
nine-frame clip. This used a reduced two-layer model, not the default full model.
