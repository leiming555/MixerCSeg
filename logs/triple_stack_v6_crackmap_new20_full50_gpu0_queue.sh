#!/usr/bin/env bash

cd /mnt/sdb1/lm/MixerCSeg || exit 1
source /home/lm/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate MixerCSeg || exit 1
set -u

GPU=0
POLL_SECONDS="${POLL_SECONDS:-60}"
OUT_ROOT="work_dirs/triple_stack_v6_crackmap_new20_full50_gpu0"
RESULT_ROOT="results/probability_maps/triple_stack_v6_crackmap"
RUN_LOG_ROOT="logs/triple_stack_v6_crackmap_new20_full50_gpu0_runs"
PREFIX="${PREFIX:-logs/$(date '+%Y%m%d_%H%M%S')_CrackMap_triple_stack_v6_new20_full50_gpu0}"
METRICS="${PREFIX}.metrics.tsv"
COMPARISON="${PREFIX}.comparison.tsv"
PROFILE_LOG="${PREFIX}.profile.log"
REFERENCE_METRICS="logs/20260806_220017_CrackMap_triple_stack_v4_search12_validate8_gpu0.metrics.tsv"
DRY_RUN=0

if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
fi

MODES=(
  hfr_ert_apb
  hfr_ert_ugc
  hfr_soc_apb
  hfr_soc_ugc
  wdr_ert_apb
  wdr_ert_ugc
  wdr_soc_apb
  wdr_soc_ugc
  cax_ert_apb
  cax_ert_ugc
  cax_soc_apb
  cax_soc_ugc
  gdp_ert_apb
  gdp_ert_ugc
  gdp_soc_apb
  gdp_soc_ugc
  rst_ert_apb
  rst_ert_ugc
  rst_soc_apb
  rst_soc_ugc
)

mkdir -p logs "$OUT_ROOT" "$RESULT_ROOT" "$RUN_LOG_ROOT"

now() { date "+%Y-%m-%d %H:%M:%S"; }

if [ "$DRY_RUN" -eq 1 ]; then
  for mode in "${MODES[@]}"; do
    echo "DRYRUN mode=${mode} seed=42 epochs=50 gpu=${GPU} bce=0.87 dice=0.13"
  done
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
  mode="$1"
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
    "$mode" "$legacy_miou" "$legacy_f1" "$legacy_ods" "$legacy_ois" \
    "$miou_fixed" "$f1_fixed" "$p_fixed" "$r_fixed" "$ods_f1" "$ois_f1" \
    "$checkpoint" "$result_dir" >> "$METRICS"
}

run_job() {
  mode="$1"
  job_id="v6_${mode}_seed42"
  job_out="$OUT_ROOT/$job_id"
  result_dir="$RESULT_ROOT/$job_id"
  runlog="$RUN_LOG_ROOT/${job_id}.train.log"
  inferlog="$RUN_LOG_ROOT/${job_id}.infer.log"
  evallog="$RUN_LOG_ROOT/${job_id}.eval.log"
  mkdir -p "$job_out" "$result_dir"

  checkpoint=$(find_checkpoint "$job_out")
  if [ -f "$result_dir/summary.csv" ] && [ -n "$checkpoint" ]; then
    echo "[$(now)] RESUME DONE mode=${mode} checkpoint=${checkpoint}"
    append_metrics "$mode" "$checkpoint" "$result_dir" "$runlog"
    return 0
  fi

  if [ -z "$checkpoint" ]; then
    echo "[$(now)] START mode=${mode} seed=42 epochs=50 gpu=${GPU}"
    CUDA_VISIBLE_DEVICES="$GPU" python -u main.py \
      --dataset_path dataset/CrackMap \
      --nbins 180 \
      --seed 42 \
      --epochs 50 \
      --BCELoss_ratio 0.87 \
      --DiceLoss_ratio 0.13 \
      --output_dir "$job_out" \
      --use_triple_stack_v6 \
      --triple_stack_v6_mode "$mode" 2>&1 | tee "$runlog"
    status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then
      echo "[$(now)] FAIL mode=${mode} stage=train status=${status}"
      return 1
    fi
    checkpoint=$(find_checkpoint "$job_out")
    if [ -z "$checkpoint" ]; then
      echo "[$(now)] FAIL mode=${mode} stage=train reason=no_checkpoint_best"
      return 1
    fi
  else
    echo "[$(now)] RESUME EVAL mode=${mode} checkpoint=${checkpoint}"
  fi

  CUDA_VISIBLE_DEVICES="$GPU" python -u tools/infer_probability.py \
    --dataset_path dataset/CrackMap \
    --checkpoint "$checkpoint" \
    --save_dir "$result_dir" \
    --infer_phase test \
    --overwrite 2>&1 | tee "$inferlog"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "[$(now)] FAIL mode=${mode} stage=infer status=${status}"
    return 1
  fi

  python -u tools/eval_standard.py \
    --result_dir "$result_dir" \
    --method "triple_stack_v6_${mode}" 2>&1 | tee "$evallog"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ] || [ ! -f "$result_dir/summary.csv" ]; then
    echo "[$(now)] FAIL mode=${mode} stage=standard_eval status=${status}"
    return 1
  fi

  append_metrics "$mode" "$checkpoint" "$result_dir" "$runlog"
  echo "[$(now)] DONE mode=${mode} seed=42"
  return 0
}

write_comparison() {
  python - "$METRICS" "$REFERENCE_METRICS" "$COMPARISON" <<'PY'
import csv
import sys

v6_path, reference_path, output_path = sys.argv[1:]
metrics = ["mIoU_fixed", "F1_fixed", "ODS_F1", "OIS_F1", "P_fixed", "R_fixed"]
with open(v6_path, newline="") as handle:
    v6_rows = list(csv.DictReader(handle, delimiter="\t"))
best = max(v6_rows, key=lambda row: tuple(float(row[key]) for key in metrics[:4]))

references = {}
with open(reference_path, newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        if row["seed"] == "42" and row["model"] in {"baseline", "v3"}:
            references[row["model"]] = row

header = ["row_type", "model", "mode", *metrics]
rows = [["result", "v6", best["mode"], *(best[key] for key in metrics)]]
for model in ("baseline", "v3"):
    reference = references[model]
    rows.append(["reference", model, reference["mode"], *(reference[key] for key in metrics)])
    rows.append([
        "delta",
        f"v6-{model}",
        best["mode"],
        *(f"{float(best[key]) - float(reference[key]):.9f}" for key in metrics),
    ])

with open(output_path, "w", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(header)
    writer.writerows(rows)
print(best["mode"])
PY
}

printf "mode\tseed\tlegacy_mIoU\tlegacy_F1\tlegacy_ODS\tlegacy_OIS\tmIoU_fixed\tF1_fixed\tP_fixed\tR_fixed\tODS_F1\tOIS_F1\tcheckpoint\tresult_dir\n" > "$METRICS"

echo "[$(now)] TripleStack-v6 CrackMap new20 full50 gpu0 queue started"
echo "[$(now)] conda=${CONDA_DEFAULT_ENV:-unknown} gpu=${GPU} modes=20 epochs=50 bce=0.87 dice=0.13"
wait_for_gpu0

failures=0
for mode in "${MODES[@]}"; do
  run_job "$mode" || failures=$((failures + 1))
done

echo "[$(now)] RESULT SORTED BY STANDARD mIoU/F1/ODS/OIS"
python - "$METRICS" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
for row in sorted(
    rows,
    key=lambda item: tuple(float(item[key]) for key in ("mIoU_fixed", "F1_fixed", "ODS_F1", "OIS_F1")),
    reverse=True,
):
    print("\t".join([row["mode"], row["mIoU_fixed"], row["F1_fixed"], row["ODS_F1"], row["OIS_F1"]]))
PY

if [ "$failures" -eq 0 ] && [ -f "$REFERENCE_METRICS" ]; then
  best_mode=$(write_comparison | tail -1)
  echo "[$(now)] PROFILE best_v6=${best_mode}" | tee "$PROFILE_LOG"
  CUDA_VISIBLE_DEVICES="$GPU" python -u tools/profile_model.py \
    --nbins 180 --warmup 20 --repeat 100 \
    --use_triple_stack_v6 --triple_stack_v6_mode "$best_mode" 2>&1 | tee -a "$PROFILE_LOG" || true
fi

echo "[$(now)] TripleStack-v6 CrackMap new20 full50 gpu0 finished with failures=${failures}"
exit "$failures"
