#!/usr/bin/env bash

cd /mnt/sdb1/lm/MixerCSeg || exit 1
source /home/lm/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate MixerCSeg || exit 1
set -u

GPU=0
POLL_SECONDS="${POLL_SECONDS:-60}"
OUT_ROOT="work_dirs/triple_stack_v4_crackmap_search12_validate8_gpu0"
RESULT_ROOT="results/probability_maps/triple_stack_v4_crackmap"
RUN_LOG_ROOT="logs/triple_stack_v4_crackmap_search12_validate8_gpu0_runs"
PREFIX="${PREFIX:-logs/$(date '+%Y%m%d_%H%M%S')_CrackMap_triple_stack_v4_search12_validate8_gpu0}"
METRICS="${PREFIX}.metrics.tsv"
VALIDATION="${PREFIX}.validation.tsv"
PROFILE_LOG="${PREFIX}.profile.log"
DRY_RUN=0

if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
fi

V4_MODES=(
  amc_eov_tgc
  amc_eov_pgs
  amc_dct_tgc
  amc_dct_pgs
  spc_eov_tgc
  spc_eov_pgs
  spc_dct_tgc
  spc_dct_pgs
  adc_eov_tgc
  adc_eov_pgs
  adc_dct_tgc
  adc_dct_pgs
)

mkdir -p logs "$OUT_ROOT" "$RESULT_ROOT" "$RUN_LOG_ROOT"

now() { date "+%Y-%m-%d %H:%M:%S"; }

job_id_for() {
  group="$1"
  model="$2"
  mode="$3"
  seed="$4"
  if [ "$group" = "search" ]; then
    printf "search_v4_%s_seed%s" "$mode" "$seed"
  elif [ "$model" = "baseline" ]; then
    printf "validate_baseline_seed%s" "$seed"
  else
    printf "validate_v3_%s_seed%s" "$mode" "$seed"
  fi
}

print_job() {
  group="$1"
  model="$2"
  mode="$3"
  seed="$4"
  job_id=$(job_id_for "$group" "$model" "$mode" "$seed")
  echo "DRYRUN job=${job_id} group=${group} model=${model} mode=${mode} seed=${seed} gpu=${GPU}"
}

print_manifest() {
  print_job validate baseline none 42
  print_job validate v3 msa_ocv_gbc 42
  for mode in "${V4_MODES[@]}"; do
    print_job search v4 "$mode" 42
  done
  for seed in 3407 2026 1234; do
    print_job validate baseline none "$seed"
    print_job validate v3 msa_ocv_gbc "$seed"
  done
}

if [ "$DRY_RUN" -eq 1 ]; then
  print_manifest
  exit 0
fi

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
  job_id="$1"
  group="$2"
  model="$3"
  mode="$4"
  seed="$5"
  checkpoint="$6"
  result_dir="$7"
  runlog="$8"

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
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$job_id" "$group" "$model" "$mode" "$seed" \
    "$legacy_miou" "$legacy_f1" "$legacy_ods" "$legacy_ois" \
    "$miou_fixed" "$f1_fixed" "$p_fixed" "$r_fixed" "$ods_f1" "$ois_f1" \
    "$checkpoint" "$result_dir" >> "$METRICS"
}

run_job() {
  group="$1"
  model="$2"
  mode="$3"
  seed="$4"
  job_id=$(job_id_for "$group" "$model" "$mode" "$seed")
  job_out="$OUT_ROOT/$job_id"
  result_dir="$RESULT_ROOT/$job_id"
  runlog="$RUN_LOG_ROOT/${job_id}.train.log"
  inferlog="$RUN_LOG_ROOT/${job_id}.infer.log"
  evallog="$RUN_LOG_ROOT/${job_id}.eval.log"
  mkdir -p "$job_out" "$result_dir"

  checkpoint=$(find_checkpoint "$job_out")
  if [ -f "$result_dir/summary.csv" ] && [ -n "$checkpoint" ]; then
    echo "[$(now)] RESUME DONE job=${job_id} checkpoint=${checkpoint}"
    append_metrics "$job_id" "$group" "$model" "$mode" "$seed" "$checkpoint" "$result_dir" "$runlog"
    return 0
  fi

  if [ -z "$checkpoint" ]; then
    cmd=(
      python -u main.py
      --dataset_path dataset/CrackMap
      --nbins 180
      --seed "$seed"
      --epochs 50
      --BCELoss_ratio 0.87
      --DiceLoss_ratio 0.13
      --output_dir "$job_out"
    )
    if [ "$model" = "v3" ]; then
      cmd+=(--use_triple_stack_v3 --triple_stack_v3_mode "$mode")
    elif [ "$model" = "v4" ]; then
      cmd+=(--use_triple_stack_v4 --triple_stack_v4_mode "$mode")
    fi

    echo "[$(now)] START job=${job_id} group=${group} model=${model} mode=${mode} seed=${seed} gpu=${GPU}"
    CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}" 2>&1 | tee "$runlog"
    status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then
      echo "[$(now)] FAIL job=${job_id} stage=train status=${status}"
      return 1
    fi
    checkpoint=$(find_checkpoint "$job_out")
    if [ -z "$checkpoint" ]; then
      echo "[$(now)] FAIL job=${job_id} stage=train reason=no_checkpoint_best"
      return 1
    fi
  else
    echo "[$(now)] RESUME EVAL job=${job_id} checkpoint=${checkpoint}"
  fi

  CUDA_VISIBLE_DEVICES="$GPU" python -u tools/infer_probability.py \
    --dataset_path dataset/CrackMap \
    --checkpoint "$checkpoint" \
    --save_dir "$result_dir" \
    --infer_phase test \
    --overwrite 2>&1 | tee "$inferlog"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "[$(now)] FAIL job=${job_id} stage=infer status=${status}"
    return 1
  fi

  python -u tools/eval_standard.py \
    --result_dir "$result_dir" \
    --method "$job_id" 2>&1 | tee "$evallog"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ] || [ ! -f "$result_dir/summary.csv" ]; then
    echo "[$(now)] FAIL job=${job_id} stage=standard_eval status=${status}"
    return 1
  fi

  append_metrics "$job_id" "$group" "$model" "$mode" "$seed" "$checkpoint" "$result_dir" "$runlog"
  echo "[$(now)] DONE job=${job_id} group=${group} model=${model} mode=${mode} seed=${seed}"
  return 0
}

write_validation_summary() {
  python - "$METRICS" "$VALIDATION" <<'PY'
import csv
import statistics
import sys

metrics_path, output_path = sys.argv[1:]
metric_names = ["mIoU_fixed", "F1_fixed", "P_fixed", "R_fixed", "ODS_F1", "OIS_F1"]
with open(metrics_path, newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

validation = [row for row in rows if row["group"] == "validate"]
by_model = {
    model: {int(row["seed"]): row for row in validation if row["model"] == model}
    for model in ("baseline", "v3")
}

header = ["row_type", "model", "seed", "n"]
for metric in metric_names:
    header.extend([f"{metric}_mean", f"{metric}_std"])

output_rows = []
for model in ("baseline", "v3"):
    model_rows = list(by_model[model].values())
    row = ["summary", model, "all", str(len(model_rows))]
    for metric in metric_names:
        values = [float(item[metric]) for item in model_rows]
        mean = statistics.mean(values) if values else float("nan")
        std = statistics.stdev(values) if len(values) > 1 else float("nan")
        row.extend([f"{mean:.9f}", f"{std:.9f}"])
    output_rows.append(row)

paired_seeds = sorted(set(by_model["baseline"]) & set(by_model["v3"]))
delta_values = {metric: [] for metric in metric_names}
for seed in paired_seeds:
    row = ["paired_delta", "v3-baseline", str(seed), "1"]
    for metric in metric_names:
        delta = float(by_model["v3"][seed][metric]) - float(by_model["baseline"][seed][metric])
        delta_values[metric].append(delta)
        row.extend([f"{delta:.9f}", ""])
    output_rows.append(row)

row = ["paired_delta_summary", "v3-baseline", "all", str(len(paired_seeds))]
for metric in metric_names:
    values = delta_values[metric]
    mean = statistics.mean(values) if values else float("nan")
    std = statistics.stdev(values) if len(values) > 1 else float("nan")
    row.extend([f"{mean:.9f}", f"{std:.9f}"])
output_rows.append(row)

with open(output_path, "w", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(header)
    writer.writerows(output_rows)
PY
}

best_v4_mode() {
  python - "$METRICS" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as handle:
    rows = [row for row in csv.DictReader(handle, delimiter="\t") if row["model"] == "v4"]
if rows:
    best = max(rows, key=lambda row: (float(row["mIoU_fixed"]), float(row["F1_fixed"]), float(row["ODS_F1"])))
    print(best["mode"])
PY
}

printf "job_id\tgroup\tmodel\tmode\tseed\tlegacy_mIoU\tlegacy_F1\tlegacy_ODS\tlegacy_OIS\tmIoU_fixed\tF1_fixed\tP_fixed\tR_fixed\tODS_F1\tOIS_F1\tcheckpoint\tresult_dir\n" > "$METRICS"

echo "[$(now)] TripleStack-v4 CrackMap search12 validate8 queue started"
echo "[$(now)] conda=${CONDA_DEFAULT_ENV:-unknown} gpu=${GPU} epochs=50 bce=0.87 dice=0.13 jobs=20"
wait_for_gpu0

failures=0
run_job validate baseline none 42 || failures=$((failures + 1))
run_job validate v3 msa_ocv_gbc 42 || failures=$((failures + 1))

for mode in "${V4_MODES[@]}"; do
  run_job search v4 "$mode" 42 || failures=$((failures + 1))
done

for seed in 3407 2026 1234; do
  run_job validate baseline none "$seed" || failures=$((failures + 1))
  run_job validate v3 msa_ocv_gbc "$seed" || failures=$((failures + 1))
done

write_validation_summary

echo "[$(now)] RESULT SORTED BY STANDARD mIoU/F1/ODS"
python - "$METRICS" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
for row in sorted(rows, key=lambda item: (float(item["mIoU_fixed"]), float(item["F1_fixed"]), float(item["ODS_F1"])), reverse=True):
    print("\t".join([row["job_id"], row["mIoU_fixed"], row["F1_fixed"], row["ODS_F1"], row["OIS_F1"]]))
PY

best_mode=$(best_v4_mode)
if [ -n "$best_mode" ]; then
  echo "[$(now)] PROFILE baseline, v3 msa_ocv_gbc, v4 ${best_mode}" | tee "$PROFILE_LOG"
  CUDA_VISIBLE_DEVICES="$GPU" python -u tools/profile_model.py --nbins 180 --warmup 20 --repeat 100 2>&1 | tee -a "$PROFILE_LOG" || true
  CUDA_VISIBLE_DEVICES="$GPU" python -u tools/profile_model.py --nbins 180 --warmup 20 --repeat 100 \
    --use_triple_stack_v3 --triple_stack_v3_mode msa_ocv_gbc 2>&1 | tee -a "$PROFILE_LOG" || true
  CUDA_VISIBLE_DEVICES="$GPU" python -u tools/profile_model.py --nbins 180 --warmup 20 --repeat 100 \
    --use_triple_stack_v4 --triple_stack_v4_mode "$best_mode" 2>&1 | tee -a "$PROFILE_LOG" || true
fi

echo "[$(now)] TripleStack-v4 CrackMap search12 validate8 gpu0 finished with failures=${failures}"
exit "$failures"
