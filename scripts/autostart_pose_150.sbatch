#!/usr/bin/env bash
#SBATCH --job-name=shubert-pose-150
#SBATCH --account=general
#SBATCH --partition=normal
#SBATCH --container=el9hw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=48:00:00
#SBATCH --chdir=/home/lidowu91/Workspace/SHuBERT
#SBATCH --output=/scratch/lidowu91/SHuBERT/outputs/youtubeasl_pose_full_e150_20261114/launcher-%j.log
set -euo pipefail
gpu_job="${1:?GPU reservation job ID required}"
gpu_state="$(squeue --noheader --jobs="$gpu_job" --format='%T')"
if [[ "$gpu_state" != RUNNING ]]; then
    echo "GPU reservation $gpu_job is not running: $gpu_state" >&2
    exit 1
fi
exec srun --jobid="$gpu_job" --overlap --nodes=1 --ntasks=1 \
    --cpus-per-task=16 --mem=96G --gres=gpu:4 \
    bash /home/lidowu91/Workspace/SHuBERT/scripts/run_youtubeasl_pose_150.sh
