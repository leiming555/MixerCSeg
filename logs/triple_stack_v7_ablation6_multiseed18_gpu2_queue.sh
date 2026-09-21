#!/usr/bin/env bash

cd /mnt/sdb1/lm/MixerCSeg || exit 1
source /home/lm/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate MixerCSeg || exit 1
set -u

GPU=2
POLL_SECONDS="${POLL_SECONDS:-60}"
OUT_ROOT="work_dirs/triple_stack_v7_ablation6_multiseed18_gpu2"
RESULT_ROOT="results/probability_maps/triple_stack_v7_ablation6_multiseed18_gpu2"
RUN_LOG_ROOT="logs/triple_stack_v7_ablation6_multiseed18_gpu2_runs"
PREFIX="${PREFIX:-logs/$(date '+%Y%m%d_%H%M%S')_TripleStackV7_ablation6_multiseed18_gpu2}"
METRICS="${PREFIX}.metrics.tsv"
ABLATION="${PREFIX}.ablation.tsv"
EFFECTS="${PREFIX}.effects.tsv"
PID_FILE="${PREFIX}.pid"
UNIT_FILE="${PREFIX}.unit"
UNIT_NAME="${UNIT_NAME:-none}"

BASE_REFERENCE="logs/20260806_220017_CrackMap_triple_stack_v4_search12_validate8_gpu0.metrics.tsv"
FULL_SEED42_REFERENCE="logs/20260906_145204_MixerCSeg_pending_gpu0_resume.v7.metrics.tsv"
FULL_OTHER_SEEDS_REFERENCE="logs/20260908_221618_TripleStackV7_wma_ctv_dgb_core5_gpu2.metrics.tsv"
ABLATION_SEED42_REFERENCE="logs/20260909_232712_TripleStackV7_wma_ctv_dgb_ablation6_gpu2.metrics.tsv"

MODES=(
  wma_ocv_gbc
  msa_ctv_gbc
  msa_ocv_dgb
  wma_ctv_gbc
  wma_ocv_dgb
  msa_ctv_dgb
)
SEEDS=(3407 2026 1234)

if [ "${1:-}" = "--dry-run" ]; then
  for seed in "${SEEDS[@]}"; do
    for mode in "${MODES[@]}"; do
      echo "DRYRUN dataset=CrackMap mode=${mode} seed=${seed} epochs=50 gpu=${GPU} nbins=180 bce=0.87 dice=0.13"
    done
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
  seed="$2"
  checkpoint="$3"
  result_dir="$4"
  runlog="$5"

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
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$mode" "$seed" "$legacy_miou" "$legacy_f1" "$legacy_ods" "$legacy_ois" \
    "$miou_fixed" "$f1_fixed" "$p_fixed" "$r_fixed" "$ods_f1" "$ois_f1" \
    "$checkpoint" "$result_dir" >> "$METRICS"
}

run_job() {
  mode="$1"
  seed="$2"
  job_id="ablation_${mode}_seed${seed}"
  job_out="$OUT_ROOT/$job_id"
  result_dir="$RESULT_ROOT/$job_id"
  runlog="$RUN_LOG_ROOT/${job_id}.train.log"
  inferlog="$RUN_LOG_ROOT/${job_id}.infer.log"
  evallog="$RUN_LOG_ROOT/${job_id}.eval.log"
  mkdir -p "$job_out" "$result_dir"

  checkpoint=$(find_checkpoint "$job_out")
  if [ -f "$result_dir/summary.csv" ] && [ -n "$checkpoint" ]; then
    echo "[$(now)] RESUME DONE mode=${mode} seed=${seed} checkpoint=${checkpoint}"
    append_metrics "$mode" "$seed" "$checkpoint" "$result_dir" "$runlog"
    return 0
  fi

  if [ -z "$checkpoint" ]; then
    wait_for_gpu2
    echo "[$(now)] START mode=${mode} dataset=CrackMap seed=${seed} epochs=50 gpu=${GPU}"
    CUDA_VISIBLE_DEVICES="$GPU" python -u main.py \
      --dataset_path dataset/CrackMap \
      --nbins 180 \
      --seed "$seed" \
      --epochs 50 \
      --BCELoss_ratio 0.87 \
      --DiceLoss_ratio 0.13 \
      --output_dir "$job_out" \
      --use_triple_stack_v7 \
      --triple_stack_v7_mode "$mode" 2>&1 | tee "$runlog"
    status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then
      echo "[$(now)] FAIL mode=${mode} seed=${seed} stage=train status=${status}"
      return 1
    fi
    checkpoint=$(find_checkpoint "$job_out")
    if [ -z "$checkpoint" ]; then
      echo "[$(now)] FAIL mode=${mode} seed=${seed} stage=train reason=no_checkpoint_best"
      return 1
    fi
  else
    wait_for_gpu2
    echo "[$(now)] RESUME EVAL mode=${mode} seed=${seed} checkpoint=${checkpoint}"
  fi

  CUDA_VISIBLE_DEVICES="$GPU" python -u tools/infer_probability.py \
    --dataset_path dataset/CrackMap \
    --checkpoint "$checkpoint" \
    --save_dir "$result_dir" \
    --infer_phase test \
    --overwrite 2>&1 | tee "$inferlog"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "[$(now)] FAIL mode=${mode} seed=${seed} stage=infer status=${status}"
    return 1
  fi

  python -u tools/eval_standard.py \
    --result_dir "$result_dir" \
    --method "TripleStackV7_ablation_${mode}_seed${seed}" 2>&1 | tee "$evallog"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ] || [ ! -f "$result_dir/summary.csv" ]; then
    echo "[$(now)] FAIL mode=${mode} seed=${seed} stage=standard_eval status=${status}"
    return 1
  fi

  append_metrics "$mode" "$seed" "$checkpoint" "$result_dir" "$runlog"
  echo "[$(now)] DONE mode=${mode} dataset=CrackMap seed=${seed}"
  return 0
}

write_multiseed_tables() {
  python - "$METRICS" "$BASE_REFERENCE" "$FULL_SEED42_REFERENCE" \
    "$FULL_OTHER_SEEDS_REFERENCE" "$ABLATION_SEED42_REFERENCE" "$ABLATION" "$EFFECTS" <<'PY'
import csv
import statistics
import sys

(
    new_path,
    base_path,
    full_seed42_path,
    full_other_path,
    ablation_seed42_path,
    ablation_path,
    effects_path,
) = sys.argv[1:]

metric_names = ["mIoU_fixed", "F1_fixed", "P_fixed", "R_fixed", "ODS_F1", "OIS_F1"]
seed_order = ["42", "3407", "2026", "1234"]
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


def read_tsv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def values(row):
    return {name: float(row[name]) for name in metric_names}


results = {mode: {} for mode, *_ in design}

for row in read_tsv(base_path):
    if (
        row.get("group") == "validate"
        and row.get("model") == "v3"
        and row.get("mode") == "msa_ocv_gbc"
        and row.get("seed") in seed_order
    ):
        results["msa_ocv_gbc"][row["seed"]] = values(row)

for row in read_tsv(full_seed42_path):
    if row.get("mode") == "wma_ctv_dgb" and row.get("seed") == "42":
        results["wma_ctv_dgb"]["42"] = values(row)

for row in read_tsv(full_other_path):
    if row.get("dataset") == "CrackMap" and row.get("model") == "wma_ctv_dgb":
        results["wma_ctv_dgb"][row["seed"]] = values(row)

for row in read_tsv(ablation_seed42_path):
    if row.get("mode") in results:
        results[row["mode"]]["42"] = values(row)

for row in read_tsv(new_path):
    results[row["mode"]][row["seed"]] = values(row)

for mode, rows in results.items():
    missing = [seed for seed in seed_order if seed not in rows]
    if missing:
        raise RuntimeError(f"Missing rows for {mode}: {missing}")

codes = {mode: (wma, ctv, dgb) for mode, wma, ctv, dgb in design}
ablation_header = [
    "row_type", "mode", "WMA", "CTV", "DGB", "seed", "n",
    *metric_names,
    *(f"{name}_std" for name in metric_names),
    *(f"delta_vs_base_{name}" for name in metric_names),
]

means = {}
with open(ablation_path, "w", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(ablation_header)
    for mode, wma, ctv, dgb in design:
        for seed in seed_order:
            current = results[mode][seed]
            base = results["msa_ocv_gbc"][seed]
            writer.writerow([
                "result", mode, wma, ctv, dgb, seed, 1,
                *(f"{current[name]:.12f}" for name in metric_names),
                *("" for _ in metric_names),
                *(f"{current[name] - base[name]:.12f}" for name in metric_names),
            ])

        means[mode] = {
            name: statistics.mean(results[mode][seed][name] for seed in seed_order)
            for name in metric_names
        }
        stds = {
            name: statistics.stdev(results[mode][seed][name] for seed in seed_order)
            for name in metric_names
        }
        base_mean = {
            name: statistics.mean(results["msa_ocv_gbc"][seed][name] for seed in seed_order)
            for name in metric_names
        }
        writer.writerow([
            "summary", mode, wma, ctv, dgb, "all", len(seed_order),
            *(f"{means[mode][name]:.12f}" for name in metric_names),
            *(f"{stds[name]:.12f}" for name in metric_names),
            *(f"{means[mode][name] - base_mean[name]:.12f}" for name in metric_names),
        ])

effects = [
    ("WMA", (0,)),
    ("CTV", (1,)),
    ("DGB", (2,)),
    ("WMAxCTV", (0, 1)),
    ("WMAxDGB", (0, 2)),
    ("CTVxDGB", (1, 2)),
    ("WMAxCTVxDGB", (0, 1, 2)),
]


def effect_for_seed(selected_factors, metric_name, seed):
    contrast = 0.0
    for mode, factor_values in codes.items():
        sign = 1.0
        for factor_index in selected_factors:
            sign *= 1.0 if factor_values[factor_index] else -1.0
        contrast += sign * results[mode][seed][metric_name]
    return contrast / 4.0


effect_header = [
    "row_type", "effect", "seed", "n", *metric_names,
    *(f"{name}_std" for name in metric_names),
]
with open(effects_path, "w", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(effect_header)
    for effect_name, selected_factors in effects:
        per_seed = {}
        for seed in seed_order:
            per_seed[seed] = {
                name: effect_for_seed(selected_factors, name, seed)
                for name in metric_names
            }
            writer.writerow([
                "result", effect_name, seed, 1,
                *(f"{per_seed[seed][name]:.12f}" for name in metric_names),
                *("" for _ in metric_names),
            ])
        writer.writerow([
            "summary", effect_name, "all", len(seed_order),
            *(
                f"{statistics.mean(per_seed[seed][name] for seed in seed_order):.12f}"
                for name in metric_names
            ),
            *(
                f"{statistics.stdev(per_seed[seed][name] for seed in seed_order):.12f}"
                for name in metric_names
            ),
        ])

best_mode = max(
    means,
    key=lambda mode: (means[mode]["mIoU_fixed"], means[mode]["F1_fixed"]),
)
path = ["msa_ocv_gbc", "wma_ocv_gbc", "wma_ocv_dgb", "wma_ctv_dgb"]
increments_positive = all(
    means[right][name] > means[left][name]
    for left, right in zip(path, path[1:])
    for name in ("mIoU_fixed", "F1_fixed")
)
full_is_top = all(
    means["wma_ctv_dgb"][name] >= max(means[mode][name] for mode in means)
    for name in ("mIoU_fixed", "F1_fixed")
)
decision = (
    "retain_three_point_wma_ctv_dgb"
    if best_mode == "wma_ctv_dgb" and full_is_top and increments_positive
    else f"review_best_{best_mode}"
)

print("FOUR-SEED ABLATION SORTED BY MEAN mIoU/F1")
for mode in sorted(
    means,
    key=lambda item: (means[item]["mIoU_fixed"], means[item]["F1_fixed"]),
    reverse=True,
):
    print(
        f"{mode}\t{means[mode]['mIoU_fixed']:.6f}\t{means[mode]['F1_fixed']:.6f}"
        f"\t{means[mode]['ODS_F1']:.6f}\t{means[mode]['OIS_F1']:.6f}"
    )
print(f"MULTISEED_ABLATION_DECISION={decision}")
PY
}

printf "mode\tseed\tlegacy_mIoU\tlegacy_F1\tlegacy_ODS\tlegacy_OIS\tmIoU_fixed\tF1_fixed\tP_fixed\tR_fixed\tODS_F1\tOIS_F1\tcheckpoint\tresult_dir\n" > "$METRICS"

echo "[$(now)] TripleStack-v7 ablation6 multiseed18 GPU2 queue started"
echo "[$(now)] conda=${CONDA_DEFAULT_ENV:-unknown} gpu=${GPU} jobs=18 epochs=50 nbins=180 bce=0.87 dice=0.13"

for required in "$BASE_REFERENCE" "$FULL_SEED42_REFERENCE" "$FULL_OTHER_SEEDS_REFERENCE" "$ABLATION_SEED42_REFERENCE"; do
  if [ ! -f "$required" ]; then
    echo "[$(now)] FAIL preflight missing_reference=${required}"
    exit 1
  fi
done

failures=0
for seed in "${SEEDS[@]}"; do
  for mode in "${MODES[@]}"; do
    run_job "$mode" "$seed" || failures=$((failures + 1))
  done
done

if [ "$failures" -eq 0 ]; then
  if write_multiseed_tables; then
    echo "[$(now)] FOUR-SEED FULL FACTORIAL ABLATION"
    column -t -s $'\t' "$ABLATION" || true
    echo "[$(now)] FOUR-SEED FACTORIAL EFFECTS"
    column -t -s $'\t' "$EFFECTS" || true
  else
    echo "[$(now)] FAIL stage=multiseed_summary"
    failures=$((failures + 1))
  fi
fi

echo "[$(now)] TripleStack-v7 ablation6 multiseed18 GPU2 finished with failures=${failures}"
exit "$failures"
