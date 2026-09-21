#!/usr/bin/env bash

cd /mnt/sdb1/lm/MixerCSeg || exit 1
source /home/lm/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate MixerCSeg || exit 1
set -u

GPU=0
POLL_SECONDS="${POLL_SECONDS:-60}"
PREFIX="${PREFIX:-logs/$(date '+%Y%m%d_%H%M%S')_MixerCSeg_pending_gpu0_resume}"
V6_OUT_ROOT="work_dirs/triple_stack_v6_resume3_full50_gpu0"
V6_RESULT_ROOT="results/probability_maps/triple_stack_v6_resume3_crackmap"
V6_RUN_LOG_ROOT="logs/triple_stack_v6_resume3_full50_gpu0_runs"
V6_METRICS="${PREFIX}.v6_resume3.metrics.tsv"
V6_ALL_METRICS="${PREFIX}.v6_all20.metrics.tsv"
V6_OLD_METRICS="logs/20260808_111819_CrackMap_triple_stack_v6_new20_full50_gpu0.metrics.tsv"
DRY_RUN=0

if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
fi

V6_MODES=(rst_ert_ugc rst_soc_apb rst_soc_ugc)

mkdir -p logs "$V6_OUT_ROOT" "$V6_RESULT_ROOT" "$V6_RUN_LOG_ROOT"

now() { date "+%Y-%m-%d %H:%M:%S"; }

if [ "$DRY_RUN" -eq 1 ]; then
  for mode in "${V6_MODES[@]}"; do
    echo "DRYRUN stage=v6 mode=${mode} dataset=CrackMap seed=42 epochs=50 gpu=${GPU}"
  done
  echo "DRYRUN stage=deepcrack jobs=baseline,msa_ocv_gbc gpu=${GPU}"
  echo "DRYRUN stage=camcrack789 jobs=baseline,msa_ocv_gbc gpu=${GPU}"
  echo "DRYRUN stage=v7 modes=20 dataset=CrackMap seed=42 epochs=50 gpu=${GPU}"
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
  root="$1"
  find "$root" -type f -name checkpoint_best.pth -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -1 | cut -d' ' -f2-
}

parse_legacy_metric() {
  metric_name="$1"
  line="$2"
  printf "%s\n" "$line" | sed -n "s/.*${metric_name} -> \([^ |]*\).*/\1/p"
}

append_v6_metrics() {
  mode="$1"
  checkpoint="$2"
  result_dir="$3"
  runlog="$4"

  legacy_line=""
  if [ -f "$runlog" ]; then
    legacy_line=$(grep -a "Best metrics | experiment ->" "$runlog" | tail -1 || true)
  fi
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
    "$checkpoint" "$result_dir" >> "$V6_METRICS"
}

run_v6_job() {
  mode="$1"
  job_root="$V6_OUT_ROOT/v6_${mode}_seed42"
  result_dir="$V6_RESULT_ROOT/v6_${mode}_seed42"
  mkdir -p "$job_root" "$result_dir"

  checkpoint=$(find_checkpoint "$job_root")
  latest_runlog=$(find "$V6_RUN_LOG_ROOT" -maxdepth 1 -type f -name "v6_${mode}_seed42_*.train.log" \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
  if [ -f "$result_dir/summary.csv" ] && [ -n "$checkpoint" ]; then
    echo "[$(now)] V6 RESUME DONE mode=${mode} checkpoint=${checkpoint}"
    append_v6_metrics "$mode" "$checkpoint" "$result_dir" "$latest_runlog"
    return 0
  fi

  attempt=$(date '+%Y%m%d_%H%M%S')
  attempt_out="$job_root/attempt_${attempt}"
  runlog="$V6_RUN_LOG_ROOT/v6_${mode}_seed42_${attempt}.train.log"
  inferlog="$V6_RUN_LOG_ROOT/v6_${mode}_seed42_${attempt}.infer.log"
  evallog="$V6_RUN_LOG_ROOT/v6_${mode}_seed42_${attempt}.eval.log"
  mkdir -p "$attempt_out"

  echo "[$(now)] V6 START mode=${mode} seed=42 epochs=50 gpu=${GPU} attempt=${attempt}"
  CUDA_VISIBLE_DEVICES="$GPU" python -u main.py \
    --dataset_path dataset/CrackMap \
    --nbins 180 \
    --seed 42 \
    --epochs 50 \
    --BCELoss_ratio 0.87 \
    --DiceLoss_ratio 0.13 \
    --output_dir "$attempt_out" \
    --use_triple_stack_v6 \
    --triple_stack_v6_mode "$mode" 2>&1 | tee "$runlog"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "[$(now)] V6 FAIL mode=${mode} stage=train status=${status}"
    return 1
  fi

  checkpoint=$(find_checkpoint "$attempt_out")
  if [ -z "$checkpoint" ]; then
    echo "[$(now)] V6 FAIL mode=${mode} stage=train reason=no_checkpoint_best"
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$GPU" python -u tools/infer_probability.py \
    --dataset_path dataset/CrackMap \
    --checkpoint "$checkpoint" \
    --save_dir "$result_dir" \
    --infer_phase test \
    --overwrite 2>&1 | tee "$inferlog"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "[$(now)] V6 FAIL mode=${mode} stage=infer status=${status}"
    return 1
  fi

  python -u tools/eval_standard.py \
    --result_dir "$result_dir" \
    --method "triple_stack_v6_${mode}" 2>&1 | tee "$evallog"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ] || [ ! -f "$result_dir/summary.csv" ]; then
    echo "[$(now)] V6 FAIL mode=${mode} stage=standard_eval status=${status}"
    return 1
  fi

  append_v6_metrics "$mode" "$checkpoint" "$result_dir" "$runlog"
  echo "[$(now)] V6 DONE mode=${mode} seed=42"
  return 0
}

merge_v6_metrics() {
  python - "$V6_OLD_METRICS" "$V6_METRICS" "$V6_ALL_METRICS" <<'PY'
import csv
import sys

old_path, resumed_path, output_path = sys.argv[1:]
rows = {}
fieldnames = None
for path in (old_path, resumed_path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames
        for row in reader:
            rows[row["mode"]] = row
with open(output_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows.values())
print(f"V6 merged completed={len(rows)}")
if rows:
    best = max(
        rows.values(),
        key=lambda row: tuple(float(row[key]) for key in ("mIoU_fixed", "F1_fixed", "ODS_F1", "OIS_F1")),
    )
    print("V6 best=" + "\t".join(
        best[key] for key in ("mode", "mIoU_fixed", "F1_fixed", "ODS_F1", "OIS_F1")
    ))
PY
}

run_child_queue() {
  stage="$1"
  script="$2"
  child_prefix="$3"
  echo "[$(now)] CHILD START stage=${stage} script=${script}"
  PREDECESSOR_UNIT="mixercseg-no-predecessor" PREFIX="$child_prefix" bash "$script"
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "[$(now)] CHILD FAIL stage=${stage} status=${status}"
    return "$status"
  fi
  echo "[$(now)] CHILD DONE stage=${stage}"
  return 0
}

printf "mode\tseed\tlegacy_mIoU\tlegacy_F1\tlegacy_ODS\tlegacy_OIS\tmIoU_fixed\tF1_fixed\tP_fixed\tR_fixed\tODS_F1\tOIS_F1\tcheckpoint\tresult_dir\n" > "$V6_METRICS"

echo "[$(now)] MixerCSeg pending GPU0 recovery queue started"
echo "[$(now)] conda=${CONDA_DEFAULT_ENV:-unknown} gpu=${GPU} v6_remaining=3 deepcrack=2 camcrack789=2 v7=20"
wait_for_gpu0

failures=0
for mode in "${V6_MODES[@]}"; do
  run_v6_job "$mode" || failures=$((failures + 1))
done
merge_v6_metrics || failures=$((failures + 1))

run_child_queue \
  deepcrack \
  logs/deepcrack_baseline_msa_ocv_gbc_seed42_gpu0_queue.sh \
  "${PREFIX}.deepcrack" || failures=$((failures + 1))

run_child_queue \
  camcrack789 \
  logs/camcrack789_baseline_msa_ocv_gbc_seed42_gpu0_queue.sh \
  "${PREFIX}.camcrack789" || failures=$((failures + 1))

run_child_queue \
  triple_stack_v7 \
  logs/triple_stack_v7_crackmap_new20_full50_gpu0_queue.sh \
  "${PREFIX}.v7" || failures=$((failures + 1))

echo "[$(now)] MixerCSeg pending GPU0 recovery queue finished with failures=${failures}"
exit "$failures"
