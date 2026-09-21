#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sdb1/lm/MixerCSeg
PYTHON=/home/lm/miniconda3/envs/MixerCSeg/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1

if [[ "${1:-}" == "--dry-run" ]]; then
    CUDA_VISIBLE_DEVICES="" "$PYTHON" tools/validate_wma_id_dgb.py dry-run
    exit 0
fi

STAMP="${STAMP:-$(date '+%Y%m%d_%H%M%S')}"
RUN_DIR="${RUN_DIR:-$PWD/work_dirs/${STAMP}_wma_id_dgb_staged_gpu012}"
PREFIX="${PREFIX:-$PWD/logs/${STAMP}_wma_id_dgb_staged_gpu012}"
mkdir -p "$RUN_DIR"
exec 9>"$PWD/logs/.wma_id_dgb_staged_gpu012.lock"
flock -n 9 || { echo "Another wma_id_dgb staged queue is running"; exit 1; }
printf '%s\n' "$$" > "${PREFIX}.pid"
printf '%s\n' "${UNIT_NAME:-unknown}" > "${PREFIX}.unit"
printf '%s\n' "$RUN_DIR" > "${PREFIX}.run_dir"
trap 'code=$?; if ((code != 0)); then echo "wma_id_dgb staged queue stopped with failures=1 exit=$code"; fi' EXIT

echo "[$(date -Is)] PREPARE run_dir=$RUN_DIR"
CUDA_VISIBLE_DEVICES="" "$PYTHON" -u tools/validate_wma_id_dgb.py prepare --run-dir "$RUN_DIR"

run_stage() {
    local stage="$1" failures=0 pid0 pid1 pid2
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u tools/validate_wma_id_dgb.py worker --stage "$stage" --gpu 0 \
        --run-dir "$RUN_DIR" > "${PREFIX}.${stage}.gpu0.log" 2>&1 & pid0=$!
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" -u tools/validate_wma_id_dgb.py worker --stage "$stage" --gpu 1 \
        --run-dir "$RUN_DIR" > "${PREFIX}.${stage}.gpu1.log" 2>&1 & pid1=$!
    if [[ "$stage" == "stage1" ]]; then
        CUDA_VISIBLE_DEVICES=2 "$PYTHON" -u tools/validate_wma_id_dgb.py worker --stage "$stage" --gpu 2 \
            --run-dir "$RUN_DIR" > "${PREFIX}.${stage}.gpu2.log" 2>&1 & pid2=$!
    else
        pid2=""
    fi
    wait "$pid0" || failures=$((failures + 1))
    wait "$pid1" || failures=$((failures + 1))
    if [[ -n "$pid2" ]]; then wait "$pid2" || failures=$((failures + 1)); fi
    echo "[$(date -Is)] stage=$stage failures=$failures"
    return "$failures"
}

run_stage stage1
set +e
CUDA_VISIBLE_DEVICES="" "$PYTHON" -u tools/validate_wma_id_dgb.py stage1-summary --run-dir "$RUN_DIR"
decision_status=$?
set -e
if [[ "$decision_status" -eq 3 ]]; then
    CUDA_VISIBLE_DEVICES=2 "$PYTHON" -u tools/validate_wma_id_dgb.py profile --gpu 2 --run-dir "$RUN_DIR"
    echo "[$(date -Is)] stage1_rejected failures=0"
    exit 0
elif [[ "$decision_status" -ne 0 ]]; then
    exit "$decision_status"
fi

run_stage stage2
set +e
CUDA_VISIBLE_DEVICES="" "$PYTHON" -u tools/validate_wma_id_dgb.py final-summary --run-dir "$RUN_DIR"
decision_status=$?
set -e
CUDA_VISIBLE_DEVICES=2 "$PYTHON" -u tools/validate_wma_id_dgb.py profile --gpu 2 --run-dir "$RUN_DIR"
if [[ "$decision_status" -eq 0 ]]; then
    echo "[$(date -Is)] final_promoted failures=0"
elif [[ "$decision_status" -eq 4 ]]; then
    echo "[$(date -Is)] final_rejected failures=0"
else
    exit "$decision_status"
fi
