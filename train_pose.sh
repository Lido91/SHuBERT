#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: bash train_pose.sh DATA_DIRECTORY CHECKPOINT_DIRECTORY [Hydra overrides...]" >&2
    exit 2
fi
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pose_data="$(realpath -- "$1")"
pose_output="$(realpath -m -- "$2")"
shift 2
export PYTHONPATH="$repo_root/fairseq:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
exec python -m fairseq_cli.hydra_train \
    --config-dir "$repo_root/configs" --config-name pose \
    "common.user_dir=$repo_root/shubert_pose" \
    "task.data=$pose_data" "checkpoint.save_dir=$pose_output" "$@"
