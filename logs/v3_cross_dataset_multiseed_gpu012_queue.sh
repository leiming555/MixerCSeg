#!/usr/bin/env bash
set -euo pipefail

cd /mnt/sdb1/lm/MixerCSeg
PYTHON=/home/lm/miniconda3/envs/MixerCSeg/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
WAIT_UNIT="${WAIT_UNIT:-mixercseg-v3-simple-20260926_153810.service}"
WAIT_LOG="${WAIT_LOG:-$PWD/logs/20260926_153810_V3_simplified_candidates_staged_gpu012.outer.log}"

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "DRYRUN wait_unit=$WAIT_UNIT"
    CUDA_VISIBLE_DEVICES="" "$PYTHON" tools/validate_v3_cross_dataset_multiseed.py dry-run
    exit 0
fi

STAMP="${STAMP:-$(date '+%Y%m%d_%H%M%S')}"
RUN_DIR="${RUN_DIR:-$PWD/work_dirs/${STAMP}_v3_cross_dataset_multiseed_gpu012}"
PREFIX="${PREFIX:-$PWD/logs/${STAMP}_V3_cross_dataset_multiseed_gpu012}"
mkdir -p "$RUN_DIR"
exec > >(tee -a "${PREFIX}.outer.log") 2>&1
exec 9>"$PWD/logs/.v3_cross_dataset_multiseed_gpu012.lock"
flock -n 9 || { echo "Another V3 cross-dataset multiseed queue is running"; exit 1; }
printf '%s\n' "$$" > "${PREFIX}.pid"
printf '%s\n' "${UNIT_NAME:-unknown}" > "${PREFIX}.unit"
printf '%s\n' "$RUN_DIR" > "${PREFIX}.run_dir"
trap 'code=$?; if ((code != 0)); then echo "V3 cross-dataset multiseed queue stopped with failures=1 exit=$code"; fi' EXIT

while systemctl --user --quiet is-active "$WAIT_UNIT"; do
    echo "[$(date -Is)] waiting for $WAIT_UNIT"
    sleep 60
done
if [[ -f "$WAIT_LOG" ]] && ! grep -Eq '(stage1_rejected|final_promoted|final_rejected) failures=0' "$WAIT_LOG"; then
    echo "Prerequisite queue stopped without a successful terminal marker: $WAIT_LOG"
    exit 1
fi

echo "[$(date -Is)] PREPARE run_dir=$RUN_DIR"
CUDA_VISIBLE_DEVICES="" "$PYTHON" -u tools/validate_v3_cross_dataset_multiseed.py \
    prepare --run-dir "$RUN_DIR"

declare -a pids=()
for gpu in 0 1 2; do
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u tools/validate_v3_cross_dataset_multiseed.py \
        worker --gpu "$gpu" --run-dir "$RUN_DIR" \
        > "${PREFIX}.gpu${gpu}.log" 2>&1 &
    pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
    wait "$pid" || failures=$((failures + 1))
done
if ((failures != 0)); then
    echo "[$(date -Is)] worker failures=$failures"
    exit 1
fi

CUDA_VISIBLE_DEVICES="" "$PYTHON" -u tools/validate_v3_cross_dataset_multiseed.py \
    collect --run-dir "$RUN_DIR"
echo "[$(date -Is)] v3 cross-dataset multiseed gpu012 finished with failures=0"
