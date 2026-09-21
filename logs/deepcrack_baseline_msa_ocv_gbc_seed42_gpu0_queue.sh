#!/usr/bin/env bash

cd /mnt/sdb1/lm/MixerCSeg || exit 1
source /home/lm/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate MixerCSeg || exit 1
set -u

GPU=0
POLL_SECONDS="${POLL_SECONDS:-60}"
PREDECESSOR_UNIT="${PREDECESSOR_UNIT:-mixercseg-v6-20260808_111819}"
OUT_ROOT="work_dirs/deepcrack_baseline_msa_ocv_gbc_seed42_gpu0"
RESULT_ROOT="results/probability_maps/deepcrack_baseline_msa_ocv_gbc"
RUN_LOG_ROOT="logs/deepcrack_baseline_msa_ocv_gbc_seed42_gpu0_runs"
PREFIX="${PREFIX:-logs/$(date '+%Y%m%d_%H%M%S')_DeepCrack_baseline_msa_ocv_gbc_seed42_gpu0}"
METRICS="${PREFIX}.metrics.tsv"
COMPARISON="${PREFIX}.comparison.tsv"
DRY_RUN=0

if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
fi

JOBS=(baseline msa_ocv_gbc)

mkdir -p logs "$OUT_ROOT" "$RESULT_ROOT" "$RUN_LOG_ROOT"

now() { date "+%Y-%m-%d %H:%M:%S"; }

if [ "$DRY_RUN" -eq 1 ]; then
  for job in "${JOBS[@]}"; do
    echo "DRYRUN job=${job} dataset=DeepCrack seed=42 epochs=50 gpu=${GPU} nbins=180 bce=0.87 dice=0.13"
  done
  exit 0
fi

wait_for_predecessor() {
  while systemctl --user is-active --quiet "$PREDECESSOR_UNIT"; do
    echo "[$(now)] waiting for ${PREDECESSOR_UNIT}"
    sleep "$POLL_SECONDS"
  done
  echo "[$(now)] predecessor ${PREDECESSOR_UNIT} is inactive"
}

wait_for_gpu0() {
  while true; do
    gpu_line=$(nvidia-smi --id="$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)
    memory=$(printf "%s" "$gpu_line" | cut -d',' -f1 | tr -d ' ')
    utilization=$(printf "%s" "$gpu_line" | cut -d',' -f2 | tr -d ' ')
    if [[ "$memory" =~ ^[0-9]+$ ]] && [[ "$utilization" =~ ^[0-9]+$ ]] \
      && [ "$memory" -lt 1000 ] && [ "$utilization" -lt 10 ]; then
      echo "[$(now)] selected GPU ${GPU} memory=${memory}MiB utilization=${utilization}%"
      return 0
    fi
    echo "[$(now)] waiting GPU ${GPU} memory=${memory:-unknown}MiB utilization=${utilization:-unknown}%"
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
  job="$1"
  checkpoint="$2"
  result_dir="$3"
  runlog="$4"

  legacy_line=$(grep -a "Best metrics | experiment ->" "$runlog" | tail -1 || true)
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
  printf "%s\t42\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$job" "$legacy_miou" "$legacy_f1" "$legacy_ods" "$legacy_ois" \
    "$miou_fixed" "$f1_fixed" "$p_fixed" "$r_fixed" "$ods_f1" "$ois_f1" \
    "$checkpoint" "$result_dir" >> "$METRICS"
}

run_job() {
  job="$1"
  job_id="deepcrack_${job}_seed42"
  job_out="$OUT_ROOT/$job_id"
  result_dir="$RESULT_ROOT/$job_id"
  runlog="$RUN_LOG_ROOT/${job_id}.train.log"
  inferlog="$RUN_LOG_ROOT/${job_id}.infer.log"
  evallog="$RUN_LOG_ROOT/${job_id}.eval.log"
  mkdir -p "$job_out" "$result_dir"

  checkpoint=$(find_checkpoint "$job_out")
  if [ -f "$result_dir/summary.csv" ] && [ -n "$checkpoint" ]; then
    echo "[$(now)] RESUME DONE job=${job} checkpoint=${checkpoint}"
    append_metrics "$job" "$checkpoint" "$result_dir" "$runlog"
    return 0
  fi

  model_args=()
  if [ "$job" = "msa_ocv_gbc" ]; then
    model_args=(--use_triple_stack_v3 --triple_stack_v3_mode msa_ocv_gbc)
  fi

  if [ -z "$checkpoint" ]; then
    echo "[$(now)] START job=${job} dataset=DeepCrack seed=42 epochs=50 gpu=${GPU}"
    CUDA_VISIBLE_DEVICES="$GPU" python -u main.py \
      --dataset_path dataset/DeepCrack \
      --nbins 180 \
      --seed 42 \
      --epochs 50 \
      --BCELoss_ratio 0.87 \
      --DiceLoss_ratio 0.13 \
      --output_dir "$job_out" \
      "${model_args[@]}" 2>&1 | tee "$runlog"
    status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then
      echo "[$(now)] FAIL job=${job} stage=train status=${status}"
      return 1
    fi
    checkpoint=$(find_checkpoint "$job_out")
    if [ -z "$checkpoint" ]; then
      echo "[$(now)] FAIL job=${job} stage=train reason=no_checkpoint_best"
      return 1
    fi
  else
    echo "[$(now)] RESUME EVAL job=${job} checkpoint=${checkpoint}"
  fi

  CUDA_VISIBLE_DEVICES="$GPU" python -u tools/infer_probability.py \
    --dataset_path dataset/DeepCrack \
    --checkpoint "$checkpoint" \
    --save_dir "$result_dir" \
    --infer_phase test \
    --overwrite 2>&1 | tee "$inferlog"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "[$(now)] FAIL job=${job} stage=infer status=${status}"
    return 1
  fi

  python -u tools/eval_standard.py \
    --result_dir "$result_dir" \
    --method "DeepCrack_${job}_seed42" 2>&1 | tee "$evallog"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ] || [ ! -f "$result_dir/summary.csv" ]; then
    echo "[$(now)] FAIL job=${job} stage=standard_eval status=${status}"
    return 1
  fi

  append_metrics "$job" "$checkpoint" "$result_dir" "$runlog"
  echo "[$(now)] DONE job=${job} dataset=DeepCrack seed=42"
  return 0
}

write_comparison() {
  python - "$METRICS" "$COMPARISON" <<'PY'
import csv
import sys

metrics_path, output_path = sys.argv[1:]
fields = ["mIoU_fixed", "F1_fixed", "ODS_F1", "OIS_F1", "P_fixed", "R_fixed"]
with open(metrics_path, newline="") as handle:
    rows = {row["job"]: row for row in csv.DictReader(handle, delimiter="\t")}

header = ["row_type", "model", *fields]
output = []
for model in ("baseline", "msa_ocv_gbc"):
    row = rows[model]
    output.append(["result", model, *(row[field] for field in fields)])
baseline = rows["baseline"]
proposed = rows["msa_ocv_gbc"]
output.append([
    "delta",
    "msa_ocv_gbc-baseline",
    *(f"{float(proposed[field]) - float(baseline[field]):.9f}" for field in fields),
])

with open(output_path, "w", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(header)
    writer.writerows(output)
PY
}

printf "job\tseed\tlegacy_mIoU\tlegacy_F1\tlegacy_ODS\tlegacy_OIS\tmIoU_fixed\tF1_fixed\tP_fixed\tR_fixed\tODS_F1\tOIS_F1\tcheckpoint\tresult_dir\n" > "$METRICS"

echo "[$(now)] DeepCrack baseline + msa_ocv_gbc seed42 gpu0 queue started"
echo "[$(now)] conda=${CONDA_DEFAULT_ENV:-unknown} gpu=${GPU} jobs=2 epochs=50 nbins=180 bce=0.87 dice=0.13"
wait_for_predecessor
wait_for_gpu0

failures=0
for job in "${JOBS[@]}"; do
  run_job "$job" || failures=$((failures + 1))
done

if [ "$failures" -eq 0 ]; then
  write_comparison
  echo "[$(now)] STANDARD COMPARISON"
  column -t -s $'\t' "$COMPARISON" || true
fi

echo "[$(now)] DeepCrack baseline + msa_ocv_gbc seed42 gpu0 finished with failures=${failures}"
exit "$failures"
