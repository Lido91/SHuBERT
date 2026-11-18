#!/usr/bin/env bash
#SBATCH --job-name=shubert-mt5-ytasl
#SBATCH --account=general
#SBATCH --partition=sooner_gpu_test_quad
#SBATCH --container=el9hw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --chdir=/home/lidowu91/Workspace/SHuBERT
#SBATCH --output=/scratch/lidowu91/SHuBERT/outputs/pose_mt5_youtubeasl_20261115/slurm-%j.log
set -euo pipefail
if [[ -f /scratch/lidowu91/SHuBERT/outputs/pose_mt5_youtubeasl_20261115/train/completed.json ]]; then
    exit 0
fi
exec bash /home/lidowu91/Workspace/SHuBERT/scripts/run_pose_mt5.sh
