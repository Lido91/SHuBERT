#!/usr/bin/env bash
set -euo pipefail
pose_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$pose_repo"
export POSE_FORMAL_ROOT="${POSE_FORMAL_ROOT:-/scratch/lidowu91/SHuBERT/outputs/youtubeasl_pose_full_e150_20261114}"
mkdir -p "$POSE_FORMAL_ROOT"
exec 9> "$POSE_FORMAL_ROOT/run.lock"
flock -n 9 || { echo 'This formal run is already active.' >&2; exit 1; }
exec > >(tee -a "$POSE_FORMAL_ROOT/train.log") 2>&1
ulimit -n 65536
date --iso-8601=seconds
hostname
bash scripts/pose_runtime.sh python scripts/cache_pose_native.py
bash scripts/pose_runtime.sh python scripts/prepare_youtubeasl_pack.py \
    --source /scratch/lidowu91/SLiM/data/youtube_asl_rtmpose_all/train \
    --output "$POSE_FORMAL_ROOT/data" --workers 8 --valid-videos 32
read -r total_updates warmup_updates < <(/usr/bin/python3 - "$POSE_FORMAL_ROOT/data/ready.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
print(value['total_updates_for_lr'], value['warmup_updates'])
PY
)
bash scripts/pose_runtime.sh bash train_pose.sh "$POSE_FORMAL_ROOT/data" "$POSE_FORMAL_ROOT/checkpoints" \
    distributed_training.distributed_world_size=4 'optimization.update_freq=[2]' \
    optimization.max_epoch=150 optimization.max_update=0 optimization.stop_min_lr=-1 \
    "+lr_scheduler.total_num_update=$total_updates" "lr_scheduler.warmup_updates=$warmup_updates" \
    task.max_frames=256 dataset.max_tokens=15000 dataset.max_tokens_valid=15000 \
    dataset.num_workers=2 dataset.disable_validation=false dataset.validate_interval=1 \
    checkpoint.no_epoch_checkpoints=false checkpoint.save_interval=1 checkpoint.keep_last_epochs=3 \
    checkpoint.save_interval_updates=1000 checkpoint.keep_interval_updates=2 \
    checkpoint.best_checkpoint_metric=loss checkpoint.maximize_best_checkpoint_metric=false \
    checkpoint.no_save_optimizer_state=false common.log_interval=20 "$@"
date --iso-8601=seconds
