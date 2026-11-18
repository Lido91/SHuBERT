#!/usr/bin/env bash
# Bounded real-data trial; rerunning resumes checkpoint_last.pt in the same directory.
set -euo pipefail
pose_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$pose_repo"
trial_root="${POSE_TRIAL_ROOT:-$pose_repo/outputs/youtubeasl_pose_20261113}"
exec bash scripts/pose_runtime.sh bash train_pose.sh "$trial_root/data" "$trial_root/checkpoints" \
    distributed_training.distributed_world_size=4 \
    'optimization.update_freq=[2]' optimization.max_update=200 \
    lr_scheduler.warmup_updates=20 \
    task.max_frames=256 dataset.max_tokens=4096 dataset.max_tokens_valid=2048 \
    dataset.num_workers=0 dataset.disable_validation=false \
    dataset.validate_interval_updates=50 checkpoint.save_interval_updates=50 \
    checkpoint.save_interval=100000 common.log_interval=10 \
    "$@"
