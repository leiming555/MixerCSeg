#!/usr/bin/env bash
set -euo pipefail

cd /mnt/sdb1/lm/MixerCSeg
PYTHON=/home/lm/miniconda3/envs/MixerCSeg/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1

if [[ "${1:-}" == "--dry-run" ]]; then
    CUDA_VISIBLE_DEVICES="" "$PYTHON" tools/reselect_v3_factorial.py dry-run
    exit 0
fi

STAMP="${STAMP:-$(date '+%Y%m%d_%H%M%S')}"
RUN_DIR="${RUN_DIR:-$PWD/work_dirs/${STAMP}_v3_factorial_reselect9_gpu012}"
PREFIX="${PREFIX:-$PWD/logs/${STAMP}_v3_factorial_reselect9_gpu012}"
mkdir -p "$RUN_DIR"
exec > >(tee -a "${PREFIX}.outer.log") 2>&1
exec 9>"$PWD/logs/.v3_factorial_reselect9_gpu012.lock"
flock -n 9 || { echo "Another V3 factorial reselection queue is running"; exit 1; }
printf '%s\n' "$$" > "${PREFIX}.pid"
printf '%s\n' "${UNIT_NAME:-unknown}" > "${PREFIX}.unit"
printf '%s\n' "$RUN_DIR" > "${PREFIX}.run_dir"
trap 'code=$?; if ((code != 0)); then echo "v3 factorial reselect9 gpu012 stopped with failures=1 exit=$code"; fi' EXIT

echo "[$(date -Is)] PREPARE run_dir=$RUN_DIR"
CUDA_VISIBLE_DEVICES="" "$PYTHON" -u tools/reselect_v3_factorial.py prepare --run-dir "$RUN_DIR"

failures=0
pids=()
for gpu in 0 1 2; do
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u tools/reselect_v3_factorial.py worker \
        --gpu "$gpu" --run-dir "$RUN_DIR" > "${PREFIX}.gpu${gpu}.log" 2>&1 &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "$pid" || failures=$((failures + 1))
done
echo "[$(date -Is)] workers finished failures=$failures"

if ((failures != 0)); then
    exit 1
fi
CUDA_VISIBLE_DEVICES="" "$PYTHON" -u tools/reselect_v3_factorial.py summarize --run-dir "$RUN_DIR"
echo "[$(date -Is)] v3 factorial reselect9 gpu012 finished with failures=0"
