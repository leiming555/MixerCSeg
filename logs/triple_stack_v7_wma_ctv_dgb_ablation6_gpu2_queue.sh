#!/usr/bin/env bash

cd /mnt/sdb1/lm/MixerCSeg || exit 1
source /home/lm/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate MixerCSeg || exit 1
set -u

GPU=2
POLL_SECONDS="${POLL_SECONDS:-60}"
OUT_ROOT="work_dirs/triple_stack_v7_wma_ctv_dgb_ablation6_gpu2"
RESULT_ROOT="results/probability_maps/triple_stack_v7_wma_ctv_dgb_ablation6_gpu2"
RUN_LOG_ROOT="logs/triple_stack_v7_wma_ctv_dgb_ablation6_gpu2_runs"
PREFIX="${PREFIX:-logs/$(date '+%Y%m%d_%H%M%S')_TripleStackV7_wma_ctv_dgb_ablation6_gpu2}"
METRICS="${PREFIX}.metrics.tsv"
ABLATION="${PREFIX}.ablation.tsv"
EFFECTS="${PREFIX}.effects.tsv"
PREFLIGHT_LOG="${PREFIX}.preflight.log"
PID_FILE="${PREFIX}.pid"
UNIT_FILE="${PREFIX}.unit"
UNIT_NAME="${UNIT_NAME:-none}"

BASE_REFERENCE="logs/20260806_220017_CrackMap_triple_stack_v4_search12_validate8_gpu0.metrics.tsv"
FULL_REFERENCE="logs/20260906_145204_MixerCSeg_pending_gpu0_resume.v7.metrics.tsv"

MODES=(
  wma_ocv_gbc
  msa_ctv_gbc
  msa_ocv_dgb
  wma_ctv_gbc
  wma_ocv_dgb
  msa_ctv_dgb
)

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
fi

if [ "$DRY_RUN" -eq 1 ]; then
  for mode in "${MODES[@]}"; do
    echo "DRYRUN dataset=CrackMap model=${mode} seed=42 epochs=50 gpu=${GPU} nbins=180 bce=0.87 dice=0.13"
  done
  exit 0
fi

mkdir -p logs "$OUT_ROOT" "$RESULT_ROOT" "$RUN_LOG_ROOT"
printf '%s\n' "$$" > "$PID_FILE"
printf '%s\n' "$UNIT_NAME" > "$UNIT_FILE"

now() { date "+%Y-%m-%d %H:%M:%S"; }

wait_for_gpu2() {
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
  printf "%s\t42\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$mode" "$legacy_miou" "$legacy_f1" "$legacy_ods" "$legacy_ois" \
    "$miou_fixed" "$f1_fixed" "$p_fixed" "$r_fixed" "$ods_f1" "$ois_f1" \
    "$checkpoint" "$result_dir" >> "$METRICS"
}

run_job() {
  mode="$1"
  job_id="ablation_${mode}_seed42"
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
    wait_for_gpu2
    echo "[$(now)] START mode=${mode} dataset=CrackMap seed=42 epochs=50 gpu=${GPU}"
    CUDA_VISIBLE_DEVICES="$GPU" python -u main.py \
      --dataset_path dataset/CrackMap \
      --nbins 180 \
      --seed 42 \
      --epochs 50 \
      --BCELoss_ratio 0.87 \
      --DiceLoss_ratio 0.13 \
      --output_dir "$job_out" \
      --use_triple_stack_v7 \
      --triple_stack_v7_mode "$mode" 2>&1 | tee "$runlog"
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
    wait_for_gpu2
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
    --method "TripleStackV7_ablation_${mode}_seed42" 2>&1 | tee "$evallog"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ] || [ ! -f "$result_dir/summary.csv" ]; then
    echo "[$(now)] FAIL mode=${mode} stage=standard_eval status=${status}"
    return 1
  fi

  append_metrics "$mode" "$checkpoint" "$result_dir" "$runlog"
  echo "[$(now)] DONE mode=${mode} dataset=CrackMap seed=42"
  return 0
}

write_ablation_tables() {
  python - "$METRICS" "$BASE_REFERENCE" "$FULL_REFERENCE" "$ABLATION" "$EFFECTS" <<'PY'
import csv
import sys

metrics_path, base_path, full_path, ablation_path, effects_path = sys.argv[1:]
metrics = ["mIoU_fixed", "F1_fixed", "P_fixed", "R_fixed", "ODS_F1", "OIS_F1"]


def read_tsv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def values(row):
    return {metric: float(row[metric]) for metric in metrics}


rows = {row["mode"]: values(row) for row in read_tsv(metrics_path)}
base_candidates = [
    row
    for row in read_tsv(base_path)
    if row.get("group") == "validate"
    and row.get("model") == "v3"
    and row.get("mode") == "msa_ocv_gbc"
    and row.get("seed") == "42"
]
full_candidates = [
    row
    for row in read_tsv(full_path)
    if row.get("mode") == "wma_ctv_dgb" and row.get("seed") == "42"
]
if len(base_candidates) != 1 or len(full_candidates) != 1:
    raise RuntimeError("Expected one seed42 reference row for both base and full models")

rows["msa_ocv_gbc"] = values(base_candidates[0])
rows["wma_ctv_dgb"] = values(full_candidates[0])

design = [
    ("msa_ocv_gbc", 0, 0, 0),
    ("wma_ocv_gbc", 1, 0, 0),
    ("msa_ctv_gbc", 0, 1, 0),
    ("msa_ocv_dgb", 0, 0, 1),
    ("wma_ctv_gbc", 1, 1, 0),
    ("wma_ocv_dgb", 1, 0, 1),
    ("msa_ctv_dgb", 0, 1, 1),
    ("wma_ctv_dgb", 1, 1, 1),
]
missing = [mode for mode, *_ in design if mode not in rows]
if missing:
    raise RuntimeError(f"Missing ablation rows: {missing}")

base = rows["msa_ocv_gbc"]
ablation_header = [
    "mode", "WMA", "CTV", "DGB", *metrics,
    *(f"delta_vs_base_{metric}" for metric in metrics),
]
with open(ablation_path, "w", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(ablation_header)
    for mode, wma, ctv, dgb in design:
        writer.writerow([
            mode,
            wma,
            ctv,
            dgb,
            *(f"{rows[mode][metric]:.12f}" for metric in metrics),
            *(f"{rows[mode][metric] - base[metric]:.12f}" for metric in metrics),
        ])

codes = {mode: (wma, ctv, dgb) for mode, wma, ctv, dgb in design}


def factorial_effect(selected_factors, metric):
    contrast = 0.0
    for mode, factor_values in codes.items():
        sign = 1.0
        for factor_index in selected_factors:
            sign *= 1.0 if factor_values[factor_index] else -1.0
        contrast += sign * rows[mode][metric]
    return contrast / 4.0


effects = [
    ("WMA", (0,)),
    ("CTV", (1,)),
    ("DGB", (2,)),
    ("WMAxCTV", (0, 1)),
    ("WMAxDGB", (0, 2)),
    ("CTVxDGB", (1, 2)),
    ("WMAxCTVxDGB", (0, 1, 2)),
]
effect_values = {}
with open(effects_path, "w", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(["effect", *metrics])
    for effect_name, factor_indices in effects:
        current = {
            metric: factorial_effect(factor_indices, metric)
            for metric in metrics
        }
        effect_values[effect_name] = current
        writer.writerow([
            effect_name,
            *(f"{current[metric]:.12f}" for metric in metrics),
        ])

best_mode = max(design, key=lambda item: (rows[item[0]]["mIoU_fixed"], rows[item[0]]["F1_fixed"]))[0]
full = rows["wma_ctv_dgb"]
full_is_top = all(
    full[metric] >= max(rows[mode][metric] for mode, *_ in design)
    for metric in ("mIoU_fixed", "F1_fixed")
)
main_effects_positive = all(
    effect_values[effect][metric] > 0.0
    for effect in ("WMA", "CTV", "DGB")
    for metric in ("mIoU_fixed", "F1_fixed")
)
decision = "retain_three_point_wma_ctv_dgb" if full_is_top and main_effects_positive else f"review_best_{best_mode}"

print("ABLATION SORTED BY STANDARD mIoU/F1")
for mode, *_ in sorted(
    design,
    key=lambda item: (rows[item[0]]["mIoU_fixed"], rows[item[0]]["F1_fixed"]),
    reverse=True,
):
    print(
        f"{mode}\t{rows[mode]['mIoU_fixed']:.6f}\t{rows[mode]['F1_fixed']:.6f}"
        f"\t{rows[mode]['ODS_F1']:.6f}\t{rows[mode]['OIS_F1']:.6f}"
    )
print(f"ABLATION_DECISION={decision}")
PY
}

printf "mode\tseed\tlegacy_mIoU\tlegacy_F1\tlegacy_ODS\tlegacy_OIS\tmIoU_fixed\tF1_fixed\tP_fixed\tR_fixed\tODS_F1\tOIS_F1\tcheckpoint\tresult_dir\n" > "$METRICS"

echo "[$(now)] TripleStack-v7 wma_ctv_dgb ablation6 GPU2 queue started"
echo "[$(now)] conda=${CONDA_DEFAULT_ENV:-unknown} gpu=${GPU} modes=${#MODES[@]} epochs=50 nbins=180 bce=0.87 dice=0.13"

for required in "$BASE_REFERENCE" "$FULL_REFERENCE"; do
  if [ ! -f "$required" ]; then
    echo "[$(now)] FAIL preflight missing_reference=${required}"
    exit 1
  fi
done

wait_for_gpu2
echo "[$(now)] CUDA PREFLIGHT mode=wma_ocv_gbc gpu=${GPU}" | tee "$PREFLIGHT_LOG"
CUDA_VISIBLE_DEVICES="$GPU" python -u tools/profile_model.py \
  --nbins 180 --warmup 1 --repeat 1 \
  --use_triple_stack_v7 --triple_stack_v7_mode wma_ocv_gbc 2>&1 | tee -a "$PREFLIGHT_LOG"
preflight_status=${PIPESTATUS[0]}
if [ "$preflight_status" -ne 0 ]; then
  echo "[$(now)] FAIL preflight status=${preflight_status}"
  exit "$preflight_status"
fi

failures=0
for mode in "${MODES[@]}"; do
  run_job "$mode" || failures=$((failures + 1))
done

if [ "$failures" -eq 0 ]; then
  if write_ablation_tables; then
    echo "[$(now)] FULL FACTORIAL ABLATION"
    column -t -s $'\t' "$ABLATION" || true
    echo "[$(now)] FACTORIAL EFFECTS"
    column -t -s $'\t' "$EFFECTS" || true
  else
    echo "[$(now)] FAIL stage=ablation_summary"
    failures=$((failures + 1))
  fi
fi

echo "[$(now)] TripleStack-v7 wma_ctv_dgb ablation6 GPU2 finished with failures=${failures}"
exit "$failures"
