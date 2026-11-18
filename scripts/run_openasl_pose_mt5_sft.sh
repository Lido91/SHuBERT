#!/usr/bin/env bash
set -euo pipefail
cd /home/lidowu91/Workspace/SHuBERT
sft_root=/scratch/lidowu91/SHuBERT/outputs/openasl_pose_mt5_sft_20261116
source_root=/scratch/lidowu91/SHuBERT/outputs/pose_mt5_youtubeasl_20261115
mkdir -p "$sft_root/initialization"
exec 9> "$sft_root/run.lock"
flock -n 9 || { echo 'OpenASL SFT is already active.' >&2; exit 1; }
exec > >(tee -a "$sft_root/train.log") 2>&1
/usr/bin/python3 - "$sft_root/launch.json" <<'PY'
import json
import os
from pathlib import Path
import sys
import time
Path(sys.argv[1]).write_text(json.dumps({'gpu_job': os.environ.get('SLURM_JOB_ID'),
    'step': os.environ.get('SLURM_STEP_ID'), 'node': os.environ.get('SLURMD_NODENAME'),
    'started_unix': time.time(), 'phase': 'preparing',
    'scope': 'training step inside independent GPU reservation'}, indent=2) + '\n')
PY
ulimit -n 65536
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
bash scripts/pose_runtime.sh python scripts/cache_pose_native.py
CUDA_VISIBLE_DEVICES='' bash scripts/pose_runtime.sh python -m unittest discover -s tests -p test_pose_translation.py -v
if [[ ! -f "$sft_root/initialization/source_best_step64000.pth" ]]; then
    cp "$source_root/train/checkpoint_best.pth" "$sft_root/initialization/source_best_step64000.pth.tmp"
    mv "$sft_root/initialization/source_best_step64000.pth.tmp" "$sft_root/initialization/source_best_step64000.pth"
fi
bash scripts/pose_runtime.sh python -u scripts/prepare_openasl_translation.py --output "$sft_root/data" --workers 8
if [[ ! -f "$sft_root/smoke/smoke_result.json" ]]; then
    bash scripts/pose_runtime.sh python -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
        scripts/train_pose_mt5.py --config configs/openasl_pose_mt5_sft.json \
        --initialization "$source_root/initialization/initialization.json" \
        --output "$sft_root/smoke" --mode smoke --max-steps 2 --limit-samples 192 --eval-limit 4 \
        --batch-size 24 --num-workers 0
fi
stop_at=0
if [[ -n ${SLURM_JOB_ID:-} ]]; then
    job_info="$(scontrol show job "$SLURM_JOB_ID" -o)"
    for field in $job_info; do
        if [[ "$field" == EndTime=* && "$field" != EndTime=Unknown ]]; then
            stop_at=$(( $(date -d "${field#EndTime=}" +%s) - 600 ))
        fi
    done
fi
# Save before this allocation ends. Further allocations require an explicit launch.
if [[ ! -f "$sft_root/train/completed.json" ]]; then
    bash scripts/pose_runtime.sh python -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
        scripts/train_pose_mt5.py --config configs/openasl_pose_mt5_sft.json \
        --initialization "$source_root/initialization/initialization.json" \
        --output "$sft_root/train" --stop-at-timestamp "$stop_at"
fi
if [[ -f "$sft_root/train/completed.json" ]]; then
    echo 'OpenASL complete; starting How2Sign SFT from its selected best full checkpoint.'
    bash scripts/run_how2sign_pose_mt5_sft.sh
else
    echo 'OpenASL paused before completion; How2Sign handoff remains pending.'
fi
