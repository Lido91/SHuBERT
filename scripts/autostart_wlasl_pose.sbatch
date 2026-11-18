#!/usr/bin/env bash
#SBATCH --job-name=shubert-wlasl-eval
#SBATCH --account=general
#SBATCH --partition=normal
#SBATCH --container=el9hw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=25:00:00
#SBATCH --chdir=/home/lidowu91/Workspace/SHuBERT
#SBATCH --output=/scratch/lidowu91/SHuBERT/outputs/wlasl_pose_e141_143_20261115/launcher-%j.log
set -euo pipefail
gpu_job="${1:?GPU reservation job ID required}"
exec srun --jobid="$gpu_job" --overlap --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G --gres=gpu:4 \
    bash /home/lidowu91/Workspace/SHuBERT/scripts/run_wlasl_pose.sh
