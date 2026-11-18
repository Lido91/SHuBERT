#!/usr/bin/env bash
# Isolate Fairseq's legacy Hydra stack while reusing the existing SLiM PyTorch runtime.
set -euo pipefail
pose_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
slim_repo="${SLIM_REPO:-/home/lidowu91/Workspace/SLiM}"
export PYTHONPATH="$pose_repo/.venv/lib/python3.9/site-packages:$slim_repo/.venv/stdlib:$slim_repo/.venv/lib/python3.9/site-packages:$pose_repo/fairseq:$pose_repo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# Uni-Sign's missing libssl leaves its older libcrypto paired with the EL9 system libssl.
export LD_PRELOAD="/lib64/libcrypto.so.3${LD_PRELOAD:+:$LD_PRELOAD}"
export TRITON_CACHE_DIR="/tmp/shubert-pose-${UID}/triton"
export XDG_CACHE_HOME="$pose_repo/outputs/runtime_cache"
export WANDB_MODE=disabled
export TF_CPP_MIN_LOG_LEVEL=2
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
export PATH="$pose_repo/.venv/bin:$PATH"
exec "$@"
