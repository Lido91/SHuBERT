#!/usr/bin/env bash
#SBATCH --job-name=gpu-standby
#SBATCH --account=general
#SBATCH --partition=sooner_gpu_test_quad
#SBATCH --container=el9hw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/lidowu91/Workspace/SHuBERT
#SBATCH --output=/scratch/lidowu91/SHuBERT/outputs/openasl_pose_mt5_sft_20261116/reservation-%j.log
set -euo pipefail
echo "Four GPUs reserved independently of workloads: job=$SLURM_JOB_ID node=$SLURMD_NODENAME"
date --iso-8601=seconds
trap 'exit 0' TERM INT
while true; do
    sleep 60 &
    wait "$!" || true
done
