#!/usr/bin/env bash
#SBATCH --job-name=openasl-sft-launcher
#SBATCH --account=general
#SBATCH --partition=normal
#SBATCH --container=el9hw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/lidowu91/Workspace/SHuBERT
#SBATCH --output=/scratch/lidowu91/SHuBERT/outputs/openasl_pose_mt5_sft_20261116/autostart-%j.log
set -euo pipefail
gpu_job="${1:?GPU reservation job ID required}"
gpu_state="$(squeue --noheader --jobs="$gpu_job" --format='%T')"
if [[ "$gpu_state" != RUNNING ]]; then
    echo "GPU reservation $gpu_job is not running: $gpu_state"
    exit 1
fi
python3 - "$gpu_job" "$SLURM_JOB_ID" <<'PY'
import json
from pathlib import Path
import sys
import time
path = Path('/scratch/lidowu91/SHuBERT/outputs/openasl_pose_mt5_sft_20261116/launch.json')
path.write_text(json.dumps({'gpu_job': sys.argv[1], 'launcher_job': sys.argv[2],
                           'started_unix': time.time(),
                           'scope': 'training step inside independent GPU reservation'}, indent=2) + '\n')
PY
# Only this step ends when SFT finishes; the independent reservation stays alive.
exec srun --jobid="$gpu_job" --overlap --nodes=1 --ntasks=1 \
    --cpus-per-task=16 --mem=96G --gres=gpu:4 \
    bash /home/lidowu91/Workspace/SHuBERT/scripts/run_openasl_pose_mt5_sft.sh
