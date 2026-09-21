#!/usr/bin/env bash

cd /mnt/sdb1/lm/MixerCSeg || exit 1
source /home/lm/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate MixerCSeg || exit 1
set -u

POLL_SECONDS="${POLL_SECONDS:-60}"
OUT_ROOT="work_dirs/triple_stack_v10_anchored12_full50_gpu12"
RESULT_ROOT="results/probability_maps/triple_stack_v10_anchored_crackmap"
RUN_LOG_ROOT="logs/triple_stack_v10_anchored12_full50_gpu12_runs"
PREFIX="${PREFIX:-logs/$(date '+%Y%m%d_%H%M%S')_TripleStackV10_anchored12_full50_gpu12}"
GPU1_METRICS="${PREFIX}.gpu1.metrics.tsv"
GPU2_METRICS="${PREFIX}.gpu2.metrics.tsv"
METRICS="${PREFIX}.metrics.tsv"
COMPARISON="${PREFIX}.comparison.tsv"
BEST_MODE="${PREFIX}.best_mode.txt"
PROFILE_LOG="${PREFIX}.profile.log"
PID_FILE="${PREFIX}.pid"
UNIT_FILE="${PREFIX}.unit"
UNIT_NAME="${UNIT_NAME:-none}"
V7_REFERENCE="logs/20260906_145204_MixerCSeg_pending_gpu0_resume.v7.metrics.tsv"

GPU1_MODES=(
  ard_dbr_bpg
  ard_pcr_dmc
  wfd_dbr_bpg
  wfd_pcr_dmc
  tcd_dbr_bpg
  tcd_pcr_dmc
)

GPU2_MODES=(
  ard_dbr_dmc
  ard_pcr_bpg
  wfd_dbr_dmc
  wfd_pcr_bpg
  tcd_dbr_dmc
  tcd_pcr_bpg
)

if [ "${1:-}" = "--dry-run" ]; then
  for mode in "${GPU1_MODES[@]}"; do
    echo "DRYRUN mode=${mode} dataset=CrackMap seed=42 epochs=50 gpu=1"
  done
  for mode in "${GPU2_MODES[@]}"; do
    echo "DRYRUN mode=${mode} dataset=CrackMap seed=42 epochs=50 gpu=2"
  done
  exit 0
fi

mkdir -p logs "$OUT_ROOT" "$RESULT_ROOT" "$RUN_LOG_ROOT"
printf '%s\n' "$$" > "$PID_FILE"
printf '%s\n' "$UNIT_NAME" > "$UNIT_FILE"

now() { date "+%Y-%m-%d %H:%M:%S"; }

metrics_header() {
  printf "mode\tseed\tgpu\tlegacy_mIoU\tlegacy_F1\tlegacy_ODS\tlegacy_OIS\tmIoU_fixed\tF1_fixed\tP_fixed\tR_fixed\tODS_F1\tOIS_F1\tcheckpoint\tresult_dir\n"
}

wait_for_gpu() {
  gpu="$1"
  while true; do
    gpu_line=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null || true)
    memory=$(printf "%s" "$gpu_line" | cut -d',' -f1 | tr -d ' ')
    utilization=$(printf "%s" "$gpu_line" | cut -d',' -f2 | tr -d ' ')
    if [[ "$memory" =~ ^[0-9]+$ ]] && [[ "$utilization" =~ ^[0-9]+$ ]] \
      && [ "$memory" -lt 1000 ] && [ "$utilization" -lt 10 ]; then
      echo "[$(now)] selected GPU ${gpu} memory=${memory}MiB utilization=${utilization}%"
      return 0
    fi
    echo "[$(now)] waiting GPU ${gpu} memory=${memory:-unknown}MiB utilization=${utilization:-unknown}%"
    sleep "$POLL_SECONDS"
  done
}

find_checkpoint() {
  job_out="$1"
  find "$job_out" -type f -name checkpoint_best.pth -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -1 | cut -d' ' -f2-
}

parse_legacy_metric() {
  metric_name="$1"
  line="$2"
  printf "%s\n" "$line" | sed -n "s/.*${metric_name} -> \([^ |]*\).*/\1/p"
}

append_metrics() {
  mode="$1"
  gpu="$2"
  checkpoint="$3"
  result_dir="$4"
  runlog="$5"
  metrics_path="$6"

  legacy_line=$(grep -a "Best metrics | experiment ->" "$runlog" 2>/dev/null | tail -1 || true)
  legacy_miou=$(parse_legacy_metric mIoU "$legacy_line")
  legacy_f1=$(parse_legacy_metric F1 "$legacy_line")
  legacy_ods=$(parse_legacy_metric ODS "$legacy_line")
  legacy_ois=$(parse_legacy_metric OIS "$legacy_line")
  standard_values=$(python - "$result_dir/summary.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as handle:
    row = next(csv.DictReader(handle))
fields = ["mIoU_fixed", "F1_fixed", "P_fixed", "R_fixed", "ODS_F1", "OIS_F1"]
print("\t".join(row[field] for field in fields))
PY
)
  IFS=$'\t' read -r miou_fixed f1_fixed p_fixed r_fixed ods_f1 ois_f1 <<< "$standard_values"
  printf "%s\t42\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$mode" "$gpu" "$legacy_miou" "$legacy_f1" "$legacy_ods" "$legacy_ois" \
    "$miou_fixed" "$f1_fixed" "$p_fixed" "$r_fixed" "$ods_f1" "$ois_f1" \
    "$checkpoint" "$result_dir" >> "$metrics_path"
}

run_job() {
  mode="$1"
  gpu="$2"
  metrics_path="$3"
  job_id="v10_${mode}_seed42"
  job_out="$OUT_ROOT/$job_id"
  result_dir="$RESULT_ROOT/$job_id"
  runlog="$RUN_LOG_ROOT/${job_id}.train.log"
  inferlog="$RUN_LOG_ROOT/${job_id}.infer.log"
  evallog="$RUN_LOG_ROOT/${job_id}.eval.log"
  mkdir -p "$job_out" "$result_dir"

  checkpoint=$(find_checkpoint "$job_out")
  if [ -f "$result_dir/summary.csv" ] && [ -n "$checkpoint" ]; then
    echo "[$(now)] RESUME DONE mode=${mode} gpu=${gpu} checkpoint=${checkpoint}"
    append_metrics "$mode" "$gpu" "$checkpoint" "$result_dir" "$runlog" "$metrics_path"
    return 0
  fi

  if [ -z "$checkpoint" ]; then
    wait_for_gpu "$gpu"
    echo "[$(now)] START mode=${mode} dataset=CrackMap seed=42 epochs=50 gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="$gpu" python -u main.py \
      --dataset_path dataset/CrackMap \
      --nbins 180 \
      --seed 42 \
      --epochs 50 \
      --BCELoss_ratio 0.87 \
      --DiceLoss_ratio 0.13 \
      --output_dir "$job_out" \
      --use_triple_stack_v10 \
      --triple_stack_v10_mode "$mode" 2>&1 | tee "$runlog"
    status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then
      echo "[$(now)] FAIL mode=${mode} gpu=${gpu} step=train status=${status}"
      return 1
    fi
    checkpoint=$(find_checkpoint "$job_out")
    if [ -z "$checkpoint" ]; then
      echo "[$(now)] FAIL mode=${mode} gpu=${gpu} step=train reason=no_checkpoint_best"
      return 1
    fi
  else
    wait_for_gpu "$gpu"
    echo "[$(now)] RESUME EVAL mode=${mode} gpu=${gpu} checkpoint=${checkpoint}"
  fi

  CUDA_VISIBLE_DEVICES="$gpu" python -u tools/infer_probability.py \
    --dataset_path dataset/CrackMap \
    --checkpoint "$checkpoint" \
    --save_dir "$result_dir" \
    --infer_phase test \
    --overwrite 2>&1 | tee "$inferlog"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "[$(now)] FAIL mode=${mode} gpu=${gpu} step=infer status=${status}"
    return 1
  fi

  python -u tools/eval_standard.py \
    --result_dir "$result_dir" \
    --method "TripleStackV10_${mode}_seed42" 2>&1 | tee "$evallog"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ] || [ ! -f "$result_dir/summary.csv" ]; then
    echo "[$(now)] FAIL mode=${mode} gpu=${gpu} step=standard_eval status=${status}"
    return 1
  fi

  append_metrics "$mode" "$gpu" "$checkpoint" "$result_dir" "$runlog" "$metrics_path"
  echo "[$(now)] DONE mode=${mode} dataset=CrackMap seed=42 gpu=${gpu}"
  return 0
}

run_worker() {
  gpu="$1"
  metrics_path="$2"
  shift 2
  modes=("$@")
  worker_failures=0
  metrics_header > "$metrics_path"
  for mode in "${modes[@]}"; do
    run_job "$mode" "$gpu" "$metrics_path" || worker_failures=$((worker_failures + 1))
  done
  echo "[$(now)] worker gpu=${gpu} finished failures=${worker_failures}"
  return "$worker_failures"
}

merge_metrics() {
  metrics_header > "$METRICS"
  for source in "$GPU1_METRICS" "$GPU2_METRICS"; do
    if [ -f "$source" ]; then
      tail -n +2 "$source" >> "$METRICS"
    fi
  done
}

write_comparison() {
  python - "$METRICS" "$V7_REFERENCE" "$COMPARISON" "$BEST_MODE" <<'PY'
import csv
import sys

metrics_path, reference_path, output_path, best_mode_path = sys.argv[1:]
with open(metrics_path, newline="") as handle:
    candidates = list(csv.DictReader(handle, delimiter="\t"))
if not candidates:
    raise RuntimeError("No successful TripleStack-v10 rows")
with open(reference_path, newline="") as handle:
    references = list(csv.DictReader(handle, delimiter="\t"))
reference = next(
    row for row in references
    if row.get("mode") == "wma_ctv_dgb" and row.get("seed") == "42"
)
metrics = ["mIoU_fixed", "F1_fixed", "P_fixed", "R_fixed", "ODS_F1", "OIS_F1"]
candidates.sort(
    key=lambda row: (float(row["mIoU_fixed"]), float(row["F1_fixed"])),
    reverse=True,
)
header = ["rank", "model", "mode", *metrics, *(f"delta_{name}" for name in metrics), "eligible"]
with open(output_path, "w", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(header)
    writer.writerow([
        0,
        "TripleStack-v7",
        "wma_ctv_dgb",
        *(reference[name] for name in metrics),
        *("0" for _ in metrics),
        "reference",
    ])
    for rank, row in enumerate(candidates, 1):
        deltas = {name: float(row[name]) - float(reference[name]) for name in metrics}
        both_aux_drop = deltas["ODS_F1"] < -0.002 and deltas["OIS_F1"] < -0.002
        eligible = (
            float(row["mIoU_fixed"]) > 0.830017
            and float(row["F1_fixed"]) > 0.804438
            and not both_aux_drop
        )
        writer.writerow([
            rank,
            "TripleStack-v10",
            row["mode"],
            *(row[name] for name in metrics),
            *(f"{deltas[name]:.12f}" for name in metrics),
            str(eligible),
        ])

best = candidates[0]
with open(best_mode_path, "w") as handle:
    handle.write(best["mode"] + "\n")
print("V10 FULL50 SORTED RESULTS")
for rank, row in enumerate(candidates, 1):
    print(f"{rank}\t{row['mode']}\t{float(row['mIoU_fixed']):.6f}\t{float(row['F1_fixed']):.6f}")
print(f"BEST_V10_MODE={best['mode']}")
PY
}

echo "[$(now)] TripleStack-v10 anchored12 full50 GPU1+GPU2 queue started"
echo "[$(now)] conda=${CONDA_DEFAULT_ENV:-unknown} jobs=12 epochs=50 bce=0.87 dice=0.13"

if [ ! -f "$V7_REFERENCE" ]; then
  echo "[$(now)] FAIL preflight missing_reference=${V7_REFERENCE}"
  exit 1
fi

failures=0
run_worker 1 "$GPU1_METRICS" "${GPU1_MODES[@]}" &
gpu1_pid=$!
run_worker 2 "$GPU2_METRICS" "${GPU2_MODES[@]}" &
gpu2_pid=$!
wait "$gpu1_pid" || failures=$((failures + $?))
wait "$gpu2_pid" || failures=$((failures + $?))
merge_metrics

if [ "$(($(wc -l < "$METRICS") - 1))" -gt 0 ]; then
  if write_comparison; then
    echo "[$(now)] V10 VS V7 COMPARISON"
    column -t -s $'\t' "$COMPARISON" || true
    best_mode=$(head -1 "$BEST_MODE")
    wait_for_gpu 2
    CUDA_VISIBLE_DEVICES=2 python -u tools/profile_model.py \
      --dataset_path dataset/CrackMap \
      --nbins 180 \
      --warmup 20 \
      --repeat 100 \
      --use_triple_stack_v10 \
      --triple_stack_v10_mode "$best_mode" 2>&1 | tee "$PROFILE_LOG" || true
  else
    echo "[$(now)] FAIL stage=write_comparison"
    failures=$((failures + 1))
  fi
else
  echo "[$(now)] FAIL stage=full reason=no_completed_rows"
  failures=$((failures + 1))
fi

echo "[$(now)] TripleStack-v10 anchored12 full50 GPU1+GPU2 finished with failures=${failures}"
exit "$failures"
