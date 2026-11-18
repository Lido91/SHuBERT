#!/usr/bin/env bash
set -euo pipefail
cd /home/lidowu91/Workspace/SHuBERT
pose_mt5_root="${POSE_MT5_ROOT:-/scratch/lidowu91/SHuBERT/outputs/pose_mt5_youtubeasl_20261115}"
mkdir -p "$pose_mt5_root"
exec 9> "$pose_mt5_root/run.lock"
flock -n 9 || { echo 'Pose+mT5 training is already active.' >&2; exit 1; }
exec > >(tee -a "$pose_mt5_root/train.log") 2>&1
ulimit -n 65536
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
bash scripts/pose_runtime.sh python scripts/cache_pose_native.py
bash scripts/pose_runtime.sh python scripts/prepare_pose_translation.py --output "$pose_mt5_root/data"
/usr/bin/python3 scripts/select_translation_backbone.py \
    --root /scratch/lidowu91/SHuBERT/outputs/wlasl_pose_fullft_e141_143_20261115_test_only_selection \
    --output "$pose_mt5_root/initialization"
if [[ ! -f "$pose_mt5_root/smoke/smoke_result.json" ]]; then
    bash scripts/pose_runtime.sh python -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
        scripts/train_pose_mt5.py --initialization "$pose_mt5_root/initialization/initialization.json" \
        --output "$pose_mt5_root/smoke" --mode smoke --max-steps 2 --limit-samples 192 --eval-limit 4 \
        --batch-size 24 --num-workers 0 --max-frames 256 --max-text-length 128
fi
stop_at=0
if [[ -n ${SLURM_JOB_ID:-} ]]; then
    job_info="$(scontrol show job "$SLURM_JOB_ID" -o)"
    for field in $job_info; do
        if [[ "$field" == EndTime=* && "$field" != EndTime=Unknown ]]; then
            stop_at=$(( $(date -d "${field#EndTime=}" +%s) - 600 ))
        fi
    done
fi
if [[ -f "$pose_mt5_root/train/completed.json" ]]; then
    echo 'All translation epochs already completed.'
    exit 0
fi
# Queue a successor before training so wall-time expiry cannot strand the 20-epoch run.
if [[ -n ${SLURM_JOB_ID:-} ]]; then
    existing_job=
    if [[ -f "$pose_mt5_root/continuation.json" ]]; then
        existing_job="$(/usr/bin/python3 - "$pose_mt5_root/continuation.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))['job_id'])
PY
)"
    fi
    existing_state=
    if [[ -n "$existing_job" && "$existing_job" != "$SLURM_JOB_ID" ]]; then
        existing_state="$(squeue -h -j "$existing_job" -o '%T' 2>/dev/null || true)"
    fi
    if [[ "$existing_state" != PENDING && "$existing_state" != RUNNING ]]; then
        next_job="$(sbatch --parsable --dependency="afterany:$SLURM_JOB_ID" scripts/resume_pose_mt5.sbatch)"
        next_job="${next_job%%;*}"
        /usr/bin/python3 - "$pose_mt5_root/continuation.json" "$next_job" "$SLURM_JOB_ID" <<'PY'
import json, pathlib, sys
path=pathlib.Path(sys.argv[1])
temporary=path.with_suffix('.json.tmp')
temporary.write_text(json.dumps({'job_id':sys.argv[2], 'after_job':sys.argv[3]}, indent=2)+'\n')
temporary.replace(path)
PY
        echo "Queued continuation job: $next_job"
    fi
fi
bash scripts/pose_runtime.sh python -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pose_mt5.py --initialization "$pose_mt5_root/initialization/initialization.json" \
    --output "$pose_mt5_root/train" --stop-at-timestamp "$stop_at"
if [[ -f "$pose_mt5_root/train/completed.json" && -f "$pose_mt5_root/continuation.json" ]]; then
    next_job="$(/usr/bin/python3 - "$pose_mt5_root/continuation.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))['job_id'])
PY
)"
    if [[ "$next_job" != "${SLURM_JOB_ID:-}" ]]; then
        scancel "$next_job"
    fi
fi
