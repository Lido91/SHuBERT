#!/usr/bin/env bash
# Submit with --dependency=after:GPU_JOB_ID and pass that GPU_JOB_ID as the first argument.
#SBATCH --job-name=shubert-autostart
#SBATCH --account=general
#SBATCH --partition=normal
#SBATCH --container=el9hw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=04:10:00
#SBATCH --chdir=/home/lidowu91/Workspace/SHuBERT
#SBATCH --output=/scratch/lidowu91/SHuBERT/outputs/youtubeasl_pose_20261113/autostart-%j.log

set -euo pipefail
gpu_job="${1:?Usage: sbatch --dependency=after:GPU_JOB_ID scripts/autostart_pose_trial.sbatch GPU_JOB_ID}"
gpu_state="$(squeue --noheader --jobs="$gpu_job" --format='%T')"
if [[ "$gpu_state" != RUNNING ]]; then
    echo "GPU reservation $gpu_job is not running (state=$gpu_state); no training launched." >&2
    exit 1
fi
exec srun --jobid="$gpu_job" --overlap --nodes=1 --ntasks=1 \
    --cpus-per-task=16 --mem=96G --gres=gpu:4 --time=04:00:00 \
    bash /home/lidowu91/Workspace/SHuBERT/scripts/run_youtubeasl_pose_trial.sh
