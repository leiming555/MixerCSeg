#!/usr/bin/env bash
set -uo pipefail

cd /mnt/sdb1/lm/MixerCSeg || exit 1

PYTHON=/home/lm/miniconda3/envs/MixerCSeg/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
POLL_SECONDS="${POLL_SECONDS:-60}"
OUT_ROOT="work_dirs/triple_stack_v11_crackmap_full16_valselect_gpu12"
RESULT_ROOT="results/probability_maps/triple_stack_v11_crackmap"
RUN_LOG_ROOT="logs/triple_stack_v11_crackmap_full16_valselect_gpu12_runs"
REFERENCE_TSV="work_dirs/20260919_085924_paper_v7_val_reselect_drop4_gpu12/reselected.metrics.tsv"
STAMP="${STAMP:-$(date '+%Y%m%d_%H%M%S')}"
PREFIX="${PREFIX:-$PWD/logs/${STAMP}_TripleStackV11_full16_valselect_gpu12}"
VALIDATION_TSV="${PREFIX}.validation.tsv"
TOP2_TSV="${PREFIX}.top2.tsv"
TEST_TSV="${PREFIX}.test.tsv"
COMPARISON_TSV="${PREFIX}.comparison.tsv"
BEST_MODE_FILE="${PREFIX}.best_mode.txt"
PROFILE_LOG="${PREFIX}.profile.log"
PID_FILE="${PREFIX}.pid"
UNIT_FILE="${PREFIX}.unit"

GPU1_MODES=(
  ram_rcv_cgb
  ram_rcv_pbs
  ram_pcv_cgb
  ram_pcv_pbs
  scm_rcv_cgb
  scm_rcv_pbs
  scm_pcv_cgb
  scm_pcv_pbs
)

GPU2_MODES=(
  fpm_rcv_cgb
  fpm_rcv_pbs
  fpm_pcv_cgb
  fpm_pcv_pbs
  lwm_rcv_cgb
  lwm_rcv_pbs
  lwm_pcv_cgb
  lwm_pcv_pbs
)

if [[ "${1:-}" == "--dry-run" ]]; then
  for mode in "${GPU1_MODES[@]}"; do
    echo "DRYRUN TRAIN mode=${mode} dataset=CrackMap seed=42 epochs=50 gpu=1 validation_selection=true"
  done
  for mode in "${GPU2_MODES[@]}"; do
    echo "DRYRUN TRAIN mode=${mode} dataset=CrackMap seed=42 epochs=50 gpu=2 validation_selection=true"
  done
  exit 0
fi

mkdir -p logs "$OUT_ROOT" "$RESULT_ROOT" "$RUN_LOG_ROOT"
exec > >(tee -a "${PREFIX}.outer.log") 2>&1
exec 9>"$PWD/logs/.triple_stack_v11_full16_valselect_gpu12.lock"
flock -n 9 || { echo "Another TripleStack-v11 queue is running"; exit 1; }
printf '%s\n' "$$" > "$PID_FILE"
printf '%s\n' "${UNIT_NAME:-unknown}" > "$UNIT_FILE"

now() { date "+%Y-%m-%d %H:%M:%S"; }

wait_for_gpu() {
  local gpu="$1"
  while true; do
    local line memory utilization
    line=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null || true)
    memory=$(printf '%s' "$line" | cut -d',' -f1 | tr -d ' ')
    utilization=$(printf '%s' "$line" | cut -d',' -f2 | tr -d ' ')
    if [[ "$memory" =~ ^[0-9]+$ ]] && [[ "$utilization" =~ ^[0-9]+$ ]] \
      && ((memory < 1000)) && ((utilization < 10)); then
      echo "[$(now)] selected GPU ${gpu} memory=${memory}MiB utilization=${utilization}%"
      return 0
    fi
    echo "[$(now)] waiting GPU ${gpu} memory=${memory:-unknown}MiB utilization=${utilization:-unknown}%"
    sleep "$POLL_SECONDS"
  done
}

find_valid_checkpoint() {
  local job_root="$1"
  local mode="$2"
  PYTHONWARNINGS=ignore "$PYTHON" - "$job_root" "$mode" <<'PY'
import sys
from pathlib import Path

from tools.summarize_triple_stack_v11 import completed_run

try:
    _, completion = completed_run(Path(sys.argv[1]), sys.argv[2])
except ValueError:
    raise SystemExit(1)
print(completion["checkpoint"])
PY
}

run_train() {
  local mode="$1"
  local gpu="$2"
  local job_id="v11_${mode}_seed42"
  local job_out="$OUT_ROOT/$job_id"
  local checkpoint=""
  mkdir -p "$job_out"

  checkpoint=$(find_valid_checkpoint "$job_out" "$mode" 2>/dev/null || true)
  if [[ -n "$checkpoint" ]]; then
    echo "[$(now)] TRAIN SKIP mode=${mode} gpu=${gpu} checkpoint=${checkpoint}"
    return 0
  fi

  wait_for_gpu "$gpu"
  local attempt_stamp runlog status
  attempt_stamp=$(date '+%Y%m%d_%H%M%S')
  runlog="$RUN_LOG_ROOT/${job_id}_${attempt_stamp}.train.log"
  echo "[$(now)] TRAIN START mode=${mode} dataset=CrackMap seed=42 epochs=50 gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u main.py \
    --dataset_path dataset/CrackMap \
    --nbins 180 \
    --seed 42 \
    --epochs 50 \
    --BCELoss_ratio 0.87 \
    --DiceLoss_ratio 0.13 \
    --output_dir "$job_out" \
    --validation_selection \
    --use_triple_stack_v11 \
    --triple_stack_v11_mode "$mode" 2>&1 | tee "$runlog"
  status=${PIPESTATUS[0]}
  if ((status != 0)); then
    echo "[$(now)] TRAIN FAIL mode=${mode} gpu=${gpu} status=${status}"
    return 1
  fi
  checkpoint=$(find_valid_checkpoint "$job_out" "$mode" 2>/dev/null || true)
  if [[ -z "$checkpoint" ]]; then
    echo "[$(now)] TRAIN FAIL mode=${mode} gpu=${gpu} reason=invalid_completion"
    return 1
  fi
  echo "[$(now)] TRAIN DONE mode=${mode} gpu=${gpu} checkpoint=${checkpoint}"
}

run_worker() {
  local gpu="$1"
  shift
  local failures=0 mode
  for mode in "$@"; do
    run_train "$mode" "$gpu" || failures=$((failures + 1))
  done
  echo "[$(now)] WORKER DONE gpu=${gpu} failures=${failures}"
  return "$failures"
}

run_test() {
  local mode="$1"
  local gpu="$2"
  local checkpoint="$3"
  local job_id="v11_${mode}_seed42"
  local result_dir="$RESULT_ROOT/$job_id"
  local inferlog="$RUN_LOG_ROOT/${job_id}.infer.log"
  local evallog="$RUN_LOG_ROOT/${job_id}.eval.log"
  local checkpoint_marker="$result_dir/tested_checkpoint.txt"
  local status

  if [[ -f "$result_dir/summary.csv" ]] && [[ -f "$checkpoint_marker" ]] \
    && [[ "$(cat "$checkpoint_marker")" == "$checkpoint" ]]; then
    echo "[$(now)] TEST SKIP mode=${mode} gpu=${gpu} result=${result_dir}"
    return 0
  fi
  wait_for_gpu "$gpu"
  mkdir -p "$result_dir"
  echo "[$(now)] TEST START mode=${mode} gpu=${gpu} checkpoint=${checkpoint}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u tools/infer_probability.py \
    --dataset_path dataset/CrackMap \
    --checkpoint "$checkpoint" \
    --save_dir "$result_dir" \
    --infer_phase test \
    --overwrite 2>&1 | tee "$inferlog"
  status=${PIPESTATUS[0]}
  if ((status != 0)); then
    echo "[$(now)] TEST FAIL mode=${mode} gpu=${gpu} step=infer status=${status}"
    return 1
  fi
  "$PYTHON" -u tools/eval_standard.py \
    --result_dir "$result_dir" \
    --method "TripleStackV11_${mode}_seed42" 2>&1 | tee "$evallog"
  status=${PIPESTATUS[0]}
  if ((status != 0)) || [[ ! -f "$result_dir/summary.csv" ]]; then
    echo "[$(now)] TEST FAIL mode=${mode} gpu=${gpu} step=standard_eval status=${status}"
    return 1
  fi
  printf '%s\n' "$checkpoint" > "$checkpoint_marker"
  echo "[$(now)] TEST DONE mode=${mode} gpu=${gpu}"
}

echo "[$(now)] TripleStack-v11 full16 validation-selection GPU1+GPU2 queue started"
echo "[$(now)] python=${PYTHON} jobs=16 epochs=50 bce=0.87 dice=0.13 test_policy=validation_top2_only"

if [[ ! -x "$PYTHON" ]] || [[ ! -f "$REFERENCE_TSV" ]]; then
  echo "[$(now)] PREFLIGHT FAIL python=${PYTHON} reference=${REFERENCE_TSV}"
  exit 1
fi

failures=0
run_worker 1 "${GPU1_MODES[@]}" &
gpu1_pid=$!
run_worker 2 "${GPU2_MODES[@]}" &
gpu2_pid=$!
wait "$gpu1_pid" || failures=$((failures + $?))
wait "$gpu2_pid" || failures=$((failures + $?))
echo "[$(now)] TRAINING STAGE DONE failures=${failures}"

if ((failures == 0)); then
  "$PYTHON" -u tools/summarize_triple_stack_v11.py collect \
    --out-root "$OUT_ROOT" \
    --validation-tsv "$VALIDATION_TSV" \
    --top-tsv "$TOP2_TSV" \
    --top-k 2 || failures=$((failures + 1))
fi

test_pids=()
if ((failures == 0)); then
  rank=0
  while IFS=$'\t' read -r _ mode _ _ _ _ _ _ checkpoint _; do
    [[ "$mode" == "mode" ]] && continue
    rank=$((rank + 1))
    gpu=$rank
    run_test "$mode" "$gpu" "$checkpoint" &
    test_pids+=("$!")
  done < "$TOP2_TSV"
  if ((rank != 2)); then
    echo "[$(now)] TEST PREP FAIL expected=2 actual=${rank}"
    failures=$((failures + 1))
  fi
fi

for pid in "${test_pids[@]}"; do
  wait "$pid" || failures=$((failures + 1))
done

if ((failures == 0)); then
  "$PYTHON" -u tools/summarize_triple_stack_v11.py finalize \
    --top-tsv "$TOP2_TSV" \
    --result-root "$RESULT_ROOT" \
    --reference-tsv "$REFERENCE_TSV" \
    --test-tsv "$TEST_TSV" \
    --comparison-tsv "$COMPARISON_TSV" \
    --best-mode "$BEST_MODE_FILE" \
    --top-k 2 || failures=$((failures + 1))
fi

if ((failures == 0)); then
  best_mode=$(head -1 "$BEST_MODE_FILE")
  wait_for_gpu 2
  CUDA_VISIBLE_DEVICES=2 "$PYTHON" -u tools/profile_model.py \
    --dataset_path dataset/CrackMap \
    --nbins 180 \
    --warmup 20 \
    --repeat 100 \
    --use_triple_stack_v11 \
    --triple_stack_v11_mode "$best_mode" 2>&1 | tee "$PROFILE_LOG" || true
fi

echo "[$(now)] TripleStack-v11 full16 validation-selection GPU1+GPU2 finished with failures=${failures}"
exit "$failures"
