#!/usr/bin/env bash
#SBATCH --job-name=shubert-mt5-openasl
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
#SBATCH --output=/scratch/lidowu91/SHuBERT/outputs/openasl_pose_mt5_sft_20261116/slurm-%j.log
set -euo pipefail
exec bash /home/lidowu91/Workspace/SHuBERT/scripts/run_openasl_pose_mt5_sft.sh
