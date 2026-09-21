#!/usr/bin/env bash

cd /mnt/sdb1/lm/MixerCSeg || exit 1
source /home/lm/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate MixerCSeg || exit 1
set -u

GPU=2
POLL_SECONDS="${POLL_SECONDS:-60}"
SCREEN_OUT_ROOT="work_dirs/triple_stack_v8_crackmap_screen12_gpu2"
FULL_OUT_ROOT="work_dirs/triple_stack_v8_crackmap_top4_full50_gpu2"
RESULT_ROOT="results/probability_maps/triple_stack_v8_crackmap"
RUN_LOG_ROOT="logs/triple_stack_v8_crackmap_screen12_top4_gpu2_runs"
PREFIX="${PREFIX:-logs/$(date '+%Y%m%d_%H%M%S')_TripleStackV8_CrackMap_screen12_top4_gpu2}"
SCREEN_METRICS="${PREFIX}.screen.metrics.tsv"
FULL_METRICS="${PREFIX}.full.metrics.tsv"
TOP4_MODES="${PREFIX}.top4_modes.txt"
BEST_MODE="${PREFIX}.best_mode.txt"
COMPARISON="${PREFIX}.comparison.tsv"
PROFILE_LOG="${PREFIX}.profile.log"
PID_FILE="${PREFIX}.pid"
UNIT_FILE="${PREFIX}.unit"
UNIT_NAME="${UNIT_NAME:-none}"
V7_REFERENCE="logs/20260906_145204_MixerCSeg_pending_gpu0_resume.v7.metrics.tsv"

MODES=(
  cwc_lsf_obp
  cwc_lsf_trc
  cwc_cot_obp
  cwc_cot_trc
  swa_lsf_obp
  swa_lsf_trc
  swa_cot_obp
  swa_cot_trc
  awc_lsf_obp
  awc_lsf_trc
  awc_cot_obp
  awc_cot_trc
)

if [ "${1:-}" = "--dry-run" ]; then
  for mode in "${MODES[@]}"; do
    echo "DRYRUN stage=screen mode=${mode} dataset=CrackMap seed=42 epochs=20 gpu=${GPU}"
  done
  for rank in 1 2 3 4; do
    echo "DRYRUN stage=full rank=${rank} mode=TOP${rank}_FROM_SCREEN dataset=CrackMap seed=42 epochs=50 gpu=${GPU}"
  done
  exit 0
fi

mkdir -p logs "$SCREEN_OUT_ROOT" "$FULL_OUT_ROOT" "$RESULT_ROOT" "$RUN_LOG_ROOT"
printf '%s\n' "$$" > "$PID_FILE"
printf '%s\n' "$UNIT_NAME" > "$UNIT_FILE"

now() { date "+%Y-%m-%d %H:%M:%S"; }

wait_for_gpu2() {
  while true; do
    gpu_line=$(nvidia-smi --id="$GPU" --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null || true)
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
  stage="$1"
  mode="$2"
  epochs="$3"
  checkpoint="$4"
  result_dir="$5"
  runlog="$6"
  metrics_path="$7"

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
  printf "%s\t%s\t%s\t42\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$stage" "$mode" "$epochs" "$legacy_miou" "$legacy_f1" "$legacy_ods" "$legacy_ois" \
    "$miou_fixed" "$f1_fixed" "$p_fixed" "$r_fixed" "$ods_f1" "$ois_f1" \
    "$checkpoint" "$result_dir" >> "$metrics_path"
}

run_job() {
  stage="$1"
  mode="$2"
  epochs="$3"
  out_root="$4"
  metrics_path="$5"
  job_id="${stage}_${mode}_seed42"
  job_out="$out_root/$job_id"
  result_dir="$RESULT_ROOT/$job_id"
  runlog="$RUN_LOG_ROOT/${job_id}.train.log"
  inferlog="$RUN_LOG_ROOT/${job_id}.infer.log"
  evallog="$RUN_LOG_ROOT/${job_id}.eval.log"
  mkdir -p "$job_out" "$result_dir"

  checkpoint=$(find_checkpoint "$job_out")
  if [ -f "$result_dir/summary.csv" ] && [ -n "$checkpoint" ]; then
    echo "[$(now)] RESUME DONE stage=${stage} mode=${mode} checkpoint=${checkpoint}"
    append_metrics "$stage" "$mode" "$epochs" "$checkpoint" "$result_dir" "$runlog" "$metrics_path"
    return 0
  fi

  if [ -z "$checkpoint" ]; then
    wait_for_gpu2
    echo "[$(now)] ${stage^^} START mode=${mode} dataset=CrackMap seed=42 epochs=${epochs} gpu=${GPU}"
    CUDA_VISIBLE_DEVICES="$GPU" python -u main.py \
      --dataset_path dataset/CrackMap \
      --nbins 180 \
      --seed 42 \
      --epochs "$epochs" \
      --BCELoss_ratio 0.87 \
      --DiceLoss_ratio 0.13 \
      --output_dir "$job_out" \
      --use_triple_stack_v8 \
      --triple_stack_v8_mode "$mode" 2>&1 | tee "$runlog"
    status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then
      echo "[$(now)] FAIL stage=${stage} mode=${mode} step=train status=${status}"
      return 1
    fi
    checkpoint=$(find_checkpoint "$job_out")
    if [ -z "$checkpoint" ]; then
      echo "[$(now)] FAIL stage=${stage} mode=${mode} step=train reason=no_checkpoint_best"
      return 1
    fi
  else
    wait_for_gpu2
    echo "[$(now)] RESUME EVAL stage=${stage} mode=${mode} checkpoint=${checkpoint}"
  fi

  CUDA_VISIBLE_DEVICES="$GPU" python -u tools/infer_probability.py \
    --dataset_path dataset/CrackMap \
    --checkpoint "$checkpoint" \
    --save_dir "$result_dir" \
    --infer_phase test \
    --overwrite 2>&1 | tee "$inferlog"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "[$(now)] FAIL stage=${stage} mode=${mode} step=infer status=${status}"
    return 1
  fi

  python -u tools/eval_standard.py \
    --result_dir "$result_dir" \
    --method "TripleStackV8_${stage}_${mode}_seed42" 2>&1 | tee "$evallog"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ] || [ ! -f "$result_dir/summary.csv" ]; then
    echo "[$(now)] FAIL stage=${stage} mode=${mode} step=standard_eval status=${status}"
    return 1
  fi

  append_metrics "$stage" "$mode" "$epochs" "$checkpoint" "$result_dir" "$runlog" "$metrics_path"
  echo "[$(now)] ${stage^^} DONE mode=${mode} dataset=CrackMap seed=42"
  return 0
}

select_top4() {
  python - "$SCREEN_METRICS" "$TOP4_MODES" <<'PY'
import csv
import sys

metrics_path, output_path = sys.argv[1:]
with open(metrics_path, newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if len(rows) < 4:
    raise RuntimeError(f"Need at least four successful screen rows, found {len(rows)}")
rows.sort(key=lambda row: (float(row["mIoU_fixed"]), float(row["F1_fixed"])), reverse=True)
with open(output_path, "w") as handle:
    for row in rows[:4]:
        handle.write(row["mode"] + "\n")
print("SCREEN TOP 4")
for rank, row in enumerate(rows[:4], 1):
    print(f"{rank}\t{row['mode']}\t{float(row['mIoU_fixed']):.6f}\t{float(row['F1_fixed']):.6f}")
PY
}

write_comparison() {
  python - "$FULL_METRICS" "$V7_REFERENCE" "$COMPARISON" "$BEST_MODE" <<'PY'
import csv
import sys

full_path, reference_path, output_path, best_mode_path = sys.argv[1:]
with open(full_path, newline="") as handle:
    candidates = list(csv.DictReader(handle, delimiter="\t"))
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
        eligible = deltas["mIoU_fixed"] > 0.0 and deltas["F1_fixed"] > 0.0 and not both_aux_drop
        writer.writerow([
            rank,
            "TripleStack-v8",
            row["mode"],
            *(row[name] for name in metrics),
            *(f"{deltas[name]:.12f}" for name in metrics),
            str(eligible),
        ])

best = candidates[0]
with open(best_mode_path, "w") as handle:
    handle.write(best["mode"] + "\n")
print("FULL50 SORTED RESULTS")
for rank, row in enumerate(candidates, 1):
    print(f"{rank}\t{row['mode']}\t{float(row['mIoU_fixed']):.6f}\t{float(row['F1_fixed']):.6f}")
print(f"BEST_V8_MODE={best['mode']}")
PY
}

printf "stage\tmode\tepochs\tseed\tlegacy_mIoU\tlegacy_F1\tlegacy_ODS\tlegacy_OIS\tmIoU_fixed\tF1_fixed\tP_fixed\tR_fixed\tODS_F1\tOIS_F1\tcheckpoint\tresult_dir\n" > "$SCREEN_METRICS"
printf "stage\tmode\tepochs\tseed\tlegacy_mIoU\tlegacy_F1\tlegacy_ODS\tlegacy_OIS\tmIoU_fixed\tF1_fixed\tP_fixed\tR_fixed\tODS_F1\tOIS_F1\tcheckpoint\tresult_dir\n" > "$FULL_METRICS"

echo "[$(now)] TripleStack-v8 CrackMap screen12 top4 GPU2 queue started"
echo "[$(now)] conda=${CONDA_DEFAULT_ENV:-unknown} gpu=${GPU} screen_jobs=12 full_jobs=4 bce=0.87 dice=0.13"

if [ ! -f "$V7_REFERENCE" ]; then
  echo "[$(now)] FAIL preflight missing_reference=${V7_REFERENCE}"
  exit 1
fi

failures=0
for mode in "${MODES[@]}"; do
  run_job screen "$mode" 20 "$SCREEN_OUT_ROOT" "$SCREEN_METRICS" || failures=$((failures + 1))
done

if select_top4; then
  mapfile -t selected_modes < "$TOP4_MODES"
  for mode in "${selected_modes[@]}"; do
    run_job full "$mode" 50 "$FULL_OUT_ROOT" "$FULL_METRICS" || failures=$((failures + 1))
  done
else
  echo "[$(now)] FAIL stage=select_top4"
  failures=$((failures + 1))
fi

if [ "$(($(wc -l < "$FULL_METRICS") - 1))" -gt 0 ]; then
  if write_comparison; then
    echo "[$(now)] V8 VS V7 COMPARISON"
    column -t -s $'\t' "$COMPARISON" || true
    best_mode=$(head -1 "$BEST_MODE")
    wait_for_gpu2
    CUDA_VISIBLE_DEVICES="$GPU" python -u tools/profile_model.py \
      --dataset_path dataset/CrackMap \
      --nbins 180 \
      --warmup 20 \
      --repeat 100 \
      --use_triple_stack_v8 \
      --triple_stack_v8_mode "$best_mode" 2>&1 | tee "$PROFILE_LOG" || true
  else
    echo "[$(now)] FAIL stage=write_comparison"
    failures=$((failures + 1))
  fi
else
  echo "[$(now)] FAIL stage=full reason=no_completed_rows"
  failures=$((failures + 1))
fi

echo "[$(now)] TripleStack-v8 CrackMap screen12 top4 GPU2 finished with failures=${failures}"
exit "$failures"
