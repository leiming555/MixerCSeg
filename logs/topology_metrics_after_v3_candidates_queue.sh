#!/usr/bin/env bash
set -euo pipefail

cd /mnt/sdb1/lm/MixerCSeg
PYTHON=/home/lm/miniconda3/envs/MixerCSeg/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
WAIT_UNIT="mixercseg-v3-simple-20260926_153810.service"
WAIT_LOG="$PWD/logs/20260926_153810_V3_simplified_candidates_staged_gpu012.outer.log"
CANDIDATE_RUN="$PWD/work_dirs/20260926_153810_v3_simplified_candidates_staged_gpu012"
POLL_SECONDS="${POLL_SECONDS:-60}"

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "DRYRUN wait_unit=$WAIT_UNIT"
    "$PYTHON" tools/summarize_topology_after_v3.py dry-run \
        --candidate-run "$CANDIDATE_RUN"
    exit 0
fi

STAMP="${STAMP:-$(date '+%Y%m%d_%H%M%S')}"
RUN_DIR="${RUN_DIR:-$PWD/work_dirs/${STAMP}_topology_metrics_after_v3_candidates}"
PREFIX="${PREFIX:-$PWD/logs/${STAMP}_topology_metrics_after_v3_candidates}"
mkdir -p "$RUN_DIR"
exec > >(tee -a "${PREFIX}.outer.log") 2>&1
exec 9>"$PWD/logs/.topology_metrics_after_v3_candidates.lock"
flock -n 9 || { echo "Another topology-metric queue is running"; exit 1; }
printf '%s\n' "$$" > "${PREFIX}.pid"
printf '%s\n' "${UNIT_NAME:-unknown}" > "${PREFIX}.unit"
printf '%s\n' "$RUN_DIR" > "${PREFIX}.run_dir"

echo "[$(date -Is)] waiting for $WAIT_UNIT"
while systemctl --user is-active --quiet "$WAIT_UNIT"; do
    sleep "$POLL_SECONDS"
done

if ! grep -Eq '(stage1_rejected|final_promoted|final_rejected) failures=0' "$WAIT_LOG"; then
    echo "[$(date -Is)] prerequisite queue did not finish successfully"
    exit 1
fi

echo "[$(date -Is)] topology evaluation started"
CUDA_VISIBLE_DEVICES="" "$PYTHON" -u tools/summarize_topology_after_v3.py collect \
    --run-dir "$RUN_DIR" \
    --candidate-run "$CANDIDATE_RUN"
echo "[$(date -Is)] topology metrics finished with failures=0"
