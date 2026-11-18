#!/usr/bin/env bash
# Run on the allocated GPU node: train, reload the checkpoint, and verify real-data features.
set -euo pipefail
pose_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$pose_repo"
export POSE_TRIAL_ROOT="${POSE_TRIAL_ROOT:-/scratch/lidowu91/SHuBERT/outputs/youtubeasl_pose_20261113}"
mkdir -p "$POSE_TRIAL_ROOT"
exec 9> "$POSE_TRIAL_ROOT/trial.lock"
flock -n 9 || { echo 'Another process is already running this trial.' >&2; exit 1; }
ulimit -n 65536
exec > >(tee -a "$POSE_TRIAL_ROOT/auto_train.log") 2>&1
date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv
bash scripts/pose_runtime.sh python scripts/cache_pose_native.py
bash scripts/train_youtubeasl_pose_trial.sh
bash scripts/pose_runtime.sh python features/pose_inference.py \
    --manifest "$POSE_TRIAL_ROOT/data/valid.tsv" \
    --checkpoint "$POSE_TRIAL_ROOT/checkpoints/checkpoint_last.pt" \
    --output-dir "$POSE_TRIAL_ROOT/valid_features" --layers 12 --device cuda --resume
bash scripts/pose_runtime.sh python scripts/verify_pose_trial.py --trial-root "$POSE_TRIAL_ROOT"
