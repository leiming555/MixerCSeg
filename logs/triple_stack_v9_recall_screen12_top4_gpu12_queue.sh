#!/usr/bin/env bash

cd /mnt/sdb1/lm/MixerCSeg || exit 1
source /home/lm/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate MixerCSeg || exit 1
set -u

POLL_SECONDS="${POLL_SECONDS:-60}"
SCREEN_OUT_ROOT="work_dirs/triple_stack_v9_recall_screen12_gpu12"
FULL_OUT_ROOT="work_dirs/triple_stack_v9_recall_top4_full50_gpu12"
RESULT_ROOT="results/probability_maps/triple_stack_v9_recall_crackmap"
RUN_LOG_ROOT="logs/triple_stack_v9_recall_screen12_top4_gpu12_runs"
PREFIX="${PREFIX:-logs/$(date '+%Y%m%d_%H%M%S')_TripleStackV9_recall_screen12_top4_gpu12}"
SCREEN_GPU1_METRICS="${PREFIX}.screen.gpu1.metrics.tsv"
SCREEN_GPU2_METRICS="${PREFIX}.screen.gpu2.metrics.tsv"
SCREEN_METRICS="${PREFIX}.screen.metrics.tsv"
FULL_GPU1_METRICS="${PREFIX}.full.gpu1.metrics.tsv"
FULL_GPU2_METRICS="${PREFIX}.full.gpu2.metrics.tsv"
FULL_METRICS="${PREFIX}.full.metrics.tsv"
TOP4_MODES="${PREFIX}.top4_modes.txt"
BEST_MODE="${PREFIX}.best_mode.txt"
COMPARISON="${PREFIX}.comparison.tsv"
PROFILE_LOG="${PREFIX}.profile.log"
PID_FILE="${PREFIX}.pid"
UNIT_FILE="${PREFIX}.unit"
UNIT_NAME="${UNIT_NAME:-none}"
V7_REFERENCE="logs/20260906_145204_MixerCSeg_pending_gpu0_resume.v7.metrics.tsv"

GPU1_MODES=(
  eub_oeb_prg
  eub_oeb_cfr
  eub_gtp_prg
  eub_gtp_cfr
  dcb_oeb_prg
  dcb_oeb_cfr
)

GPU2_MODES=(
  dcb_gtp_prg
  dcb_gtp_cfr
  mcb_oeb_prg
  mcb_oeb_cfr
  mcb_gtp_prg
  mcb_gtp_cfr
)

if [ "${1:-}" = "--dry-run" ]; then
  for mode in "${GPU1_MODES[@]}"; do
    echo "DRYRUN stage=screen mode=${mode} dataset=CrackMap seed=42 epochs=20 gpu=1"
  done
  for mode in "${GPU2_MODES[@]}"; do
    echo "DRYRUN stage=screen mode=${mode} dataset=CrackMap seed=42 epochs=20 gpu=2"
  done
  echo "DRYRUN stage=full rank=1 mode=TOP1_FROM_SCREEN dataset=CrackMap seed=42 epochs=50 gpu=1"
  echo "DRYRUN stage=full rank=2 mode=TOP2_FROM_SCREEN dataset=CrackMap seed=42 epochs=50 gpu=2"
  echo "DRYRUN stage=full rank=3 mode=TOP3_FROM_SCREEN dataset=CrackMap seed=42 epochs=50 gpu=1"
  echo "DRYRUN stage=full rank=4 mode=TOP4_FROM_SCREEN dataset=CrackMap seed=42 epochs=50 gpu=2"
  exit 0
fi

mkdir -p logs "$SCREEN_OUT_ROOT" "$FULL_OUT_ROOT" "$RESULT_ROOT" "$RUN_LOG_ROOT"
printf '%s\n' "$$" > "$PID_FILE"
printf '%s\n' "$UNIT_NAME" > "$UNIT_FILE"

now() { date "+%Y-%m-%d %H:%M:%S"; }

metrics_header() {
  printf "stage\tmode\tepochs\tseed\tgpu\tlegacy_mIoU\tlegacy_F1\tlegacy_ODS\tlegacy_OIS\tmIoU_fixed\tF1_fixed\tP_fixed\tR_fixed\tODS_F1\tOIS_F1\tcheckpoint\tresult_dir\n"
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
  stage="$1"
  mode="$2"
  epochs="$3"
  gpu="$4"
  checkpoint="$5"
  result_dir="$6"
  runlog="$7"
  metrics_path="$8"

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
    "$stage" "$mode" "$epochs" "$gpu" "$legacy_miou" "$legacy_f1" "$legacy_ods" \
    "$legacy_ois" "$miou_fixed" "$f1_fixed" "$p_fixed" "$r_fixed" "$ods_f1" \
    "$ois_f1" "$checkpoint" "$result_dir" >> "$metrics_path"
}

run_job() {
  stage="$1"
  mode="$2"
  epochs="$3"
  gpu="$4"
  out_root="$5"
  metrics_path="$6"
  job_id="${stage}_${mode}_seed42"
  job_out="$out_root/$job_id"
  result_dir="$RESULT_ROOT/$job_id"
  runlog="$RUN_LOG_ROOT/${job_id}.train.log"
  inferlog="$RUN_LOG_ROOT/${job_id}.infer.log"
  evallog="$RUN_LOG_ROOT/${job_id}.eval.log"
  mkdir -p "$job_out" "$result_dir"

  checkpoint=$(find_checkpoint "$job_out")
  if [ -f "$result_dir/summary.csv" ] && [ -n "$checkpoint" ]; then
    echo "[$(now)] RESUME DONE stage=${stage} mode=${mode} gpu=${gpu} checkpoint=${checkpoint}"
    append_metrics "$stage" "$mode" "$epochs" "$gpu" "$checkpoint" "$result_dir" "$runlog" "$metrics_path"
    return 0
  fi

  if [ -z "$checkpoint" ]; then
    wait_for_gpu "$gpu"
    echo "[$(now)] ${stage^^} START mode=${mode} dataset=CrackMap seed=42 epochs=${epochs} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="$gpu" python -u main.py \
      --dataset_path dataset/CrackMap \
      --nbins 180 \
      --seed 42 \
      --epochs "$epochs" \
      --BCELoss_ratio 0.87 \
      --DiceLoss_ratio 0.13 \
      --output_dir "$job_out" \
      --use_triple_stack_v9 \
      --triple_stack_v9_mode "$mode" 2>&1 | tee "$runlog"
    status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then
      echo "[$(now)] FAIL stage=${stage} mode=${mode} gpu=${gpu} step=train status=${status}"
      return 1
    fi
    checkpoint=$(find_checkpoint "$job_out")
    if [ -z "$checkpoint" ]; then
      echo "[$(now)] FAIL stage=${stage} mode=${mode} gpu=${gpu} step=train reason=no_checkpoint_best"
      return 1
    fi
  else
    wait_for_gpu "$gpu"
    echo "[$(now)] RESUME EVAL stage=${stage} mode=${mode} gpu=${gpu} checkpoint=${checkpoint}"
  fi

  CUDA_VISIBLE_DEVICES="$gpu" python -u tools/infer_probability.py \
    --dataset_path dataset/CrackMap \
    --checkpoint "$checkpoint" \
    --save_dir "$result_dir" \
    --infer_phase test \
    --overwrite 2>&1 | tee "$inferlog"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "[$(now)] FAIL stage=${stage} mode=${mode} gpu=${gpu} step=infer status=${status}"
    return 1
  fi

  python -u tools/eval_standard.py \
    --result_dir "$result_dir" \
    --method "TripleStackV9_${stage}_${mode}_seed42" 2>&1 | tee "$evallog"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ] || [ ! -f "$result_dir/summary.csv" ]; then
    echo "[$(now)] FAIL stage=${stage} mode=${mode} gpu=${gpu} step=standard_eval status=${status}"
    return 1
  fi

  append_metrics "$stage" "$mode" "$epochs" "$gpu" "$checkpoint" "$result_dir" "$runlog" "$metrics_path"
  echo "[$(now)] ${stage^^} DONE mode=${mode} dataset=CrackMap seed=42 gpu=${gpu}"
  return 0
}

run_worker() {
  stage="$1"
  epochs="$2"
  gpu="$3"
  out_root="$4"
  metrics_path="$5"
  shift 5
  modes=("$@")
  worker_failures=0
  metrics_header > "$metrics_path"
  for mode in "${modes[@]}"; do
    run_job "$stage" "$mode" "$epochs" "$gpu" "$out_root" "$metrics_path" \
      || worker_failures=$((worker_failures + 1))
  done
  echo "[$(now)] worker stage=${stage} gpu=${gpu} finished failures=${worker_failures}"
  return "$worker_failures"
}

merge_metrics() {
  output="$1"
  shift
  metrics_header > "$output"
  for source in "$@"; do
    if [ -f "$source" ]; then
      tail -n +2 "$source" >> "$output"
    fi
  done
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
        eligible = (
            float(row["mIoU_fixed"]) > 0.830017
            and float(row["F1_fixed"]) > 0.804438
            and not both_aux_drop
        )
        writer.writerow([
            rank,
            "TripleStack-v9",
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
print(f"BEST_V9_MODE={best['mode']}")
PY
}

echo "[$(now)] TripleStack-v9 recall screen12 top4 GPU1+GPU2 queue started"
echo "[$(now)] conda=${CONDA_DEFAULT_ENV:-unknown} screen_jobs=12 full_jobs=4 bce=0.87 dice=0.13"

if [ ! -f "$V7_REFERENCE" ]; then
  echo "[$(now)] FAIL preflight missing_reference=${V7_REFERENCE}"
  exit 1
fi

failures=0
run_worker screen 20 1 "$SCREEN_OUT_ROOT" "$SCREEN_GPU1_METRICS" "${GPU1_MODES[@]}" &
gpu1_pid=$!
run_worker screen 20 2 "$SCREEN_OUT_ROOT" "$SCREEN_GPU2_METRICS" "${GPU2_MODES[@]}" &
gpu2_pid=$!
wait "$gpu1_pid" || failures=$((failures + $?))
wait "$gpu2_pid" || failures=$((failures + $?))
merge_metrics "$SCREEN_METRICS" "$SCREEN_GPU1_METRICS" "$SCREEN_GPU2_METRICS"

if select_top4; then
  mapfile -t selected_modes < "$TOP4_MODES"
  gpu1_full_modes=("${selected_modes[0]}" "${selected_modes[2]}")
  gpu2_full_modes=("${selected_modes[1]}" "${selected_modes[3]}")
  run_worker full 50 1 "$FULL_OUT_ROOT" "$FULL_GPU1_METRICS" "${gpu1_full_modes[@]}" &
  gpu1_pid=$!
  run_worker full 50 2 "$FULL_OUT_ROOT" "$FULL_GPU2_METRICS" "${gpu2_full_modes[@]}" &
  gpu2_pid=$!
  wait "$gpu1_pid" || failures=$((failures + $?))
  wait "$gpu2_pid" || failures=$((failures + $?))
  merge_metrics "$FULL_METRICS" "$FULL_GPU1_METRICS" "$FULL_GPU2_METRICS"
else
  echo "[$(now)] FAIL stage=select_top4"
  failures=$((failures + 1))
  metrics_header > "$FULL_METRICS"
fi

if [ "$(($(wc -l < "$FULL_METRICS") - 1))" -gt 0 ]; then
  if write_comparison; then
    echo "[$(now)] V9 VS V7 COMPARISON"
    column -t -s $'\t' "$COMPARISON" || true
    best_mode=$(head -1 "$BEST_MODE")
    wait_for_gpu 2
    CUDA_VISIBLE_DEVICES=2 python -u tools/profile_model.py \
      --dataset_path dataset/CrackMap \
      --nbins 180 \
      --warmup 20 \
      --repeat 100 \
      --use_triple_stack_v9 \
      --triple_stack_v9_mode "$best_mode" 2>&1 | tee "$PROFILE_LOG" || true
  else
    echo "[$(now)] FAIL stage=write_comparison"
    failures=$((failures + 1))
  fi
else
  echo "[$(now)] FAIL stage=full reason=no_completed_rows"
  failures=$((failures + 1))
fi

echo "[$(now)] TripleStack-v9 recall screen12 top4 GPU1+GPU2 finished with failures=${failures}"
exit "$failures"
