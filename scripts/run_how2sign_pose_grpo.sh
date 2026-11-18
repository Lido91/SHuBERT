#!/usr/bin/env bash
set -euo pipefail
cd /home/lidowu91/Workspace/SHuBERT
rl_root=/scratch/lidowu91/SHuBERT/outputs/how2sign_pose_mt5_grpo_bleu_consensus_v2_20261117
sft_root=/scratch/lidowu91/SHuBERT/outputs/how2sign_pose_mt5_continue_best1518_20261117
mkdir -p "$rl_root"
exec 8> "$rl_root/run.lock"
flock -n 8 || { echo 'How2Sign RL is already active.' >&2; exit 1; }
exec > >(tee -a "$rl_root/train.log") 2>&1
ulimit -n 65536
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec 7> "$sft_root/run.lock"
echo 'Waiting for stopped SFT to finish its final checkpoint save.'
flock 7
bash scripts/pose_runtime.sh python scripts/cache_pose_native.py
if [[ ! -f "$rl_root/smoke/smoke_result.json" ]]; then
    bash scripts/pose_runtime.sh python -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
        scripts/rft_pose_grpo.py --config configs/how2sign_pose_grpo.json \
        --output "$rl_root/smoke" --mode smoke --max-steps 2 --limit-samples 64 --eval-limit 4 --num-workers 0
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
if [[ ! -f "$rl_root/train/completed.json" ]]; then
    bash scripts/pose_runtime.sh python -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
        scripts/rft_pose_grpo.py --config configs/how2sign_pose_grpo.json \
        --output "$rl_root/train" --stop-at-timestamp "$stop_at"
fi
