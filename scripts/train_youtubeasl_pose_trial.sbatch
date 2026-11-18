#!/usr/bin/env bash
#SBATCH --job-name=shubert-pose-ytasl
#SBATCH --account=general
#SBATCH --partition=sooner_gpu_test_quad
#SBATCH --container=el9hw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/lidowu91/Workspace/SHuBERT
#SBATCH --output=/home/lidowu91/Workspace/SHuBERT/outputs/youtubeasl_pose_20261113/slurm-%j.log

set -euo pipefail
cd "${SHUBERT_REPO:-/home/lidowu91/Workspace/SHuBERT}"
ulimit -n 65536
bash scripts/pose_runtime.sh python scripts/cache_pose_native.py
exec bash scripts/train_youtubeasl_pose_trial.sh "$@"
