#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sdb1/lm/MixerCSeg
PYTHON=/home/lm/miniconda3/envs/MixerCSeg/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1

if [[ "${1:-}" == "--dry-run" ]]; then
    CUDA_VISIBLE_DEVICES="" "$PYTHON" tools/paper_audit.py dry-run
    exit 0
fi

STAMP="${STAMP:-$(date '+%Y%m%d_%H%M%S')}"
RUN_DIR="${RUN_DIR:-$PWD/work_dirs/${STAMP}_paper_v7_val_reselect_drop4_gpu12}"
PREFIX="${PREFIX:-$PWD/logs/${STAMP}_paper_v7_val_reselect_drop4_gpu12}"
mkdir -p "$RUN_DIR"
exec 9>"$PWD/logs/.paper_v7_val_reselect_drop4_gpu12.lock"
flock -n 9 || { echo "Another paper validation queue is running"; exit 1; }
printf '%s\n' "$$" > "${PREFIX}.pid"
printf '%s\n' "${UNIT_NAME:-unknown}" > "${PREFIX}.unit"
printf '%s\n' "$RUN_DIR" > "${PREFIX}.run_dir"
trap 'code=$?; if ((code != 0)); then echo "paper-v7 validation queue stopped with failures=1 exit=$code"; fi' EXIT

echo "[$(date -Is)] PREPARE run_dir=$RUN_DIR"
CUDA_VISIBLE_DEVICES="" "$PYTHON" -u tools/paper_audit.py prepare --run-dir "$RUN_DIR"

run_pair() {
    local phase="$1" failures=0 p1 p2
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" -u tools/paper_audit.py "$phase" \
        --gpu 1 --run-dir "$RUN_DIR" > "${PREFIX}.${phase}.gpu1.log" 2>&1 &
    p1=$!
    CUDA_VISIBLE_DEVICES=2 "$PYTHON" -u tools/paper_audit.py "$phase" \
        --gpu 2 --run-dir "$RUN_DIR" > "${PREFIX}.${phase}.gpu2.log" 2>&1 &
    p2=$!
    wait "$p1" || failures=$((failures + 1))
    wait "$p2" || failures=$((failures + 1))
    echo "[$(date -Is)] phase=$phase failures=$failures"
    return "$failures"
}

run_pair reselect
CUDA_VISIBLE_DEVICES="" "$PYTHON" tools/paper_audit.py summarize --run-dir "$RUN_DIR"
run_pair removals
CUDA_VISIBLE_DEVICES="" "$PYTHON" tools/paper_audit.py summarize --run-dir "$RUN_DIR" --require-removals
echo "[$(date -Is)] paper-v7 validation reselect18 drop4 GPU1+GPU2 finished with failures=0"
