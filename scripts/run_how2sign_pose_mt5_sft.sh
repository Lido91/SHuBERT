#!/usr/bin/env bash
set -euo pipefail
cd /home/lidowu91/Workspace/SHuBERT
sft_root=/scratch/lidowu91/SHuBERT/outputs/how2sign_pose_mt5_after_openasl_20261116
source_root=/scratch/lidowu91/SHuBERT/outputs/pose_mt5_youtubeasl_20261115
mkdir -p "$sft_root"
exec 8> "$sft_root/run.lock"
flock -n 8 || { echo 'How2Sign SFT is already active.' >&2; exit 1; }
exec > >(tee -a "$sft_root/train.log") 2>&1
ulimit -n 65536
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
bash scripts/pose_runtime.sh python scripts/cache_pose_native.py
bash scripts/pose_runtime.sh python -u scripts/prepare_how2sign_sft.py
bash scripts/pose_runtime.sh python -u scripts/prepare_how2sign_translation.py --output "$sft_root/data" --workers 8
if [[ ! -f "$sft_root/smoke/smoke_result.json" ]]; then
    bash scripts/pose_runtime.sh python -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
        scripts/train_pose_mt5.py --config "$sft_root/sft_config.json" \
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
if [[ ! -f "$sft_root/train/completed.json" ]]; then
    bash scripts/pose_runtime.sh python -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
        scripts/train_pose_mt5.py --config "$sft_root/sft_config.json" \
        --initialization "$source_root/initialization/initialization.json" \
        --output "$sft_root/train" --stop-at-timestamp "$stop_at"
fi
