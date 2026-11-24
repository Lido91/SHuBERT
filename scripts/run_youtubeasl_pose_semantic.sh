#!/usr/bin/env bash
# Second stage on the formal YouTubeASL pack: cluster prediction plus SeRV-style semantic alignment.
set -euo pipefail
pose_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$pose_repo"
formal="${POSE_FORMAL_ROOT:-/scratch/lidowu91/SHuBERT/outputs/youtubeasl_pose_full_e150_20261114}"
root="${POSE_SEMANTIC_ROOT:?set POSE_SEMANTIC_ROOT to a new output directory}"
text_model="${POSE_TEXT_MODEL:-/scratch/lidowu91/Uni-Sign-Lido/checkpoints/mt5-base}"
initial="${POSE_INIT_CHECKPOINT:-$formal/checkpoints/checkpoint_last.pt}"
records="${POSE_CAPTION_RECORDS:-/scratch/lidowu91/SHuBERT/outputs/pose_mt5_youtubeasl_20261115/data/train.records.jsonl}"
epochs="${POSE_SEMANTIC_EPOCHS:-50}"
mkdir -p "$root"
exec 9> "$root/run.lock"
flock -n 9 || { echo 'This semantic run is already active.' >&2; exit 1; }
exec > >(tee -a "$root/train.log") 2>&1
ulimit -n 65536
date --iso-8601=seconds
hostname
bash scripts/pose_runtime.sh python scripts/cache_pose_native.py
bash scripts/pose_runtime.sh python scripts/prepare_pose_captions.py \
    --records "$records" --pack "$formal/data/train.pack.json" \
    --text-model "$text_model" --output "$root/captions"
read -r total_updates warmup_updates < <(/usr/bin/python3 - "$formal/data/ready.json" "$epochs" <<'PY'
import json, sys
total = json.load(open(sys.argv[1]))['updates_per_epoch'] * int(sys.argv[2])
print(total, max(1, int(total * 0.08)))
PY
)
POSE_CONFIG=pose_semantic bash scripts/pose_runtime.sh bash train_pose.sh "$formal/data" "$root/checkpoints" \
    "task.captions=$root/captions/captions.json" "model.text_model=$text_model" \
    "model.init_checkpoint=$initial" \
    distributed_training.distributed_world_size=4 'optimization.update_freq=[2]' \
    "optimization.max_epoch=$epochs" optimization.max_update=0 optimization.stop_min_lr=-1 \
    "+lr_scheduler.total_num_update=$total_updates" "lr_scheduler.warmup_updates=$warmup_updates" \
    task.max_frames=256 dataset.max_tokens=15000 dataset.max_tokens_valid=15000 \
    dataset.num_workers=2 dataset.disable_validation=false dataset.validate_interval=1 \
    checkpoint.no_epoch_checkpoints=false checkpoint.save_interval=1 checkpoint.keep_last_epochs=3 \
    checkpoint.save_interval_updates=1000 checkpoint.keep_interval_updates=2 \
    checkpoint.best_checkpoint_metric=loss checkpoint.maximize_best_checkpoint_metric=false \
    checkpoint.no_save_optimizer_state=false common.log_interval=20 "$@"
date --iso-8601=seconds
