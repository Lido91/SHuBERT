# Repository guidance

## Project and layout

SHuBERT is the ACL 2025 sign-language representation model from
`https://github.com/ShesterG/SHuBERT.git`. It learns masked cluster prediction
over face, left-hand, right-hand, and body-posture feature streams.

- Start with `README.md` and `QUICKSTART.md`. `DATASETS.md` and `FEATURES.md`
  contain some outdated script paths; verify commands against the actual files.
- `dataset/`: YouTube-ASL downloads, clip creation, signer cropping, MediaPipe
  landmark extraction, and face/hand crops.
- `features/`: body features, DINOv2 face/hand features, and SHuBERT inference.
- `fairseq/examples/shubert/models/shubert.py`: model and configuration classes.
- `fairseq/examples/shubert/models/simple_kmeans/`: cluster fitting and labels.
- `fairseq/examples/shubert/config/base_random.yaml`: pretraining configuration.
- `fairseq/fairseq/tasks/audio_pretraining.py` and
  `fairseq/fairseq/data/audio/raw_audio_dataset.py`: modified visual-stream data
  pipeline, despite their inherited audio names.
- `train_shubert.sh`: SLURM pretraining template. `fairseq/tests/` contains
  inherited Fairseq tests; there is no dedicated SHuBERT test suite here.

The bundled `fairseq/` contains project-specific changes. Preserve those changes
and use this checkout when installing Fairseq.

The coordinate-only variant lives in `shubert_pose/`, with `configs/pose.yaml`,
`scripts/prepare_pose.py`, `train_pose.sh`, and `features/pose_inference.py`.
See `POSE_ONLY.md` for its separate manifest and checkpoint contracts and
`tests/test_pose.py` for focused tests. It supports configurable coordinate
streams and does not use the original four-stream data loader.

## Environments and setup

The repository provides separate Conda environments:

| Specification | Environment | Main use |
| --- | --- | --- |
| `environment_feature_extraction.yml` | `feature_extraction` | Crops, landmarks, body features; Python 3.10 |
| `environment_dino.yml` | `dino` | DINOv2 face/hand features; Python 3.9 |
| `environment_shubert.yml` | `shubert_train_env` | SHuBERT training/inference; Python 3.10 |

For model work, the documented setup from the repository root is:

```bash
conda env create -f environment_shubert.yml
conda activate shubert_train_env
python -m pip install -e ./fairseq
```

Use the relevant environment for each pipeline stage. The model environment pins
Hydra 1.0.7 and OmegaConf 2.0.6; check compatibility before changing dependencies.
Environment creation and runtime functionality must be verified on the current
machine before reporting setup as complete.

## Running the pipeline

Follow `QUICKSTART.md`: signer crop, landmarks, face/hand crops, body and DINOv2
features, then SHuBERT inference. External assets include YOLO, MediaPipe task
files, fine-tuned DINOv2 weights, and the SHuBERT checkpoint.

The shell launchers contain empty environment/SLURM settings and placeholder
paths. Many submit 256 jobs in a loop. Inspect their arguments and working
directory before running them. `train_shubert.sh` also needs its Fairseq/config
paths checked after its `cd fairseq`.

For a single inference shard, activate `shubert_train_env` and run from the
repository root, replacing the example paths with real assets:

```bash
PYTHONPATH="$PWD/fairseq${PYTHONPATH:+:$PYTHONPATH}" \
  python features/shubert_inference.py \
  --index 0 --batch_size 1 \
  --csv_path /path/to/manifest.csv \
  --checkpoint_path /path/to/shubert.pt \
  --output_dir /path/to/output
```

In this script, `--batch_size` controls the number of manifest rows in a shard;
samples are processed individually. `--index` selects that shard. Inference uses
CUDA unconditionally.

## Data and model contracts

- Inference reads each row with `csv.reader`, then splits `row[0]` on tabs.
  Supply four tab-separated paths in this order: face, left hand, right hand,
  body posture; the script does not skip a header.
- Default per-frame inputs have 384 dimensions for each visual crop stream and
  14 for body posture. All four streams must share the same frame alignment.
- Model inputs are a list of sample dictionaries with those four streams and
  matching `label_*` keys. Inference supplies dummy labels and disables masking.
- Inference saves layer features with shape `[L, T, D]`. Check dimensions,
  finiteness, sample names, and frame counts on a representative real sample.
- Checkpoint loading uses `strict=False`; successful loading alone does not
  establish that all intended weights matched. Inspect missing/unexpected keys
  when modifying the model or loading a new checkpoint.
- `write_list.py` writes gzip-compressed pickle data, despite the `.list` suffix.
  It has hardcoded input/output paths and uses `os.listdir`; verify that list
  entries have the path format required by the consuming script.
- Keep generated videos, features, manifests, logs, and checkpoints out of source
  changes. Existing outputs are historical evidence, not a fresh validation run.

## Changes and validation

Keep edits focused on the requested pipeline stage. Follow surrounding Python
style; Fairseq's `setup.cfg` specifies Flake8 with a 127-character line limit.
Avoid reformatting the bundled library or unrelated research scripts.

For Python changes, check syntax and run relevant tests in the appropriate
environment. For example, from `fairseq/`, a focused inherited test command is
`python -m unittest tests.test_utils`. Choose tests that exercise the changed
behavior; passing generic Fairseq tests does not validate the visual pipeline.
For shell changes, use `bash -n path/to/script.sh` before execution.

For feature/model changes, use a small real sample when assets are available and
check the output contract above. Report the exact checks performed and any
missing prerequisites. Distinguish syntax checks, smoke inference, and complete
dataset processing. Documentation-only edits need a diff/whitespace review.
