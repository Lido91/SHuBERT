#!/usr/bin/env bash
set -euo pipefail
cd /home/lidowu91/Workspace/SHuBERT
ulimit -n 65536
bash scripts/pose_runtime.sh python scripts/cache_pose_native.py
exec bash scripts/pose_runtime.sh python -u scripts/run_wlasl_pose_sweep.py \
    --output /scratch/lidowu91/SHuBERT/outputs/wlasl_pose_e141_143_20261115_test_as_dev \
    --checkpoints /scratch/lidowu91/SHuBERT/outputs/youtubeasl_pose_full_e150_20261114/checkpoints \
    --data /scratch/lidowu91/SHuBERT/outputs/wlasl_pose_e141_143_20261115/data \
    --selection-split test \
    --warm-start-root /scratch/lidowu91/SHuBERT/outputs/wlasl_pose_e141_143_20261115_test_as_dev/warm_start
