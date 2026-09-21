#!/usr/bin/env bash

cd /mnt/sdb1/lm/MixerCSeg || exit 1
source /home/lm/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate MixerCSeg || exit 1
set -u

GPU=2
POLL_SECONDS="${POLL_SECONDS:-60}"
OUT_ROOT="work_dirs/triple_stack_v7_wma_ctv_dgb_core5_gpu2"
RESULT_ROOT="results/probability_maps/triple_stack_v7_wma_ctv_dgb_core5_gpu2"
RUN_LOG_ROOT="logs/triple_stack_v7_wma_ctv_dgb_core5_gpu2_runs"
PREFIX="${PREFIX:-logs/$(date '+%Y%m%d_%H%M%S')_TripleStackV7_wma_ctv_dgb_core5_gpu2}"
METRICS="${PREFIX}.metrics.tsv"
VALIDATION="${PREFIX}.validation.tsv"
PREFLIGHT_LOG="${PREFIX}.preflight.log"
PID_FILE="${PREFIX}.pid"
UNIT_FILE="${PREFIX}.unit"
UNIT_NAME="${UNIT_NAME:-none}"

CRACKMAP_REFERENCE="logs/20260806_220017_CrackMap_triple_stack_v4_search12_validate8_gpu0.metrics.tsv"
V7_SEED42_REFERENCE="logs/20260906_145204_MixerCSeg_pending_gpu0_resume.v7.metrics.tsv"
DEEPCRACK_REFERENCE="logs/20260906_145204_MixerCSeg_pending_gpu0_resume.deepcrack.metrics.tsv"
CAMCRACK789_REFERENCE="logs/20260906_145204_MixerCSeg_pending_gpu0_resume.camcrack789.metrics.tsv"

JOBS=(
  "CrackMap|3407"
  "CrackMap|2026"
  "CrackMap|1234"
  "DeepCrack|42"
  "CamCrack789|42"
)

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
fi

if [ "$DRY_RUN" -eq 1 ]; then
  for entry in "${JOBS[@]}"; do
    IFS='|' read -r dataset seed <<< "$entry"
    echo "DRYRUN dataset=${dataset} model=wma_ctv_dgb seed=${seed} epochs=50 gpu=${GPU} nbins=180 bce=0.87 dice=0.13"
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

dataset_slug() {
  case "$1" in
    CrackMap) echo "crackmap" ;;
    DeepCrack) echo "deepcrack" ;;
    CamCrack789) echo "camcrack789" ;;
    *) return 1 ;;
  esac
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
  dataset="$1"
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
  printf "%s\twma_ctv_dgb\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$dataset" "$seed" "$legacy_miou" "$legacy_f1" "$legacy_ods" "$legacy_ois" \
    "$miou_fixed" "$f1_fixed" "$p_fixed" "$r_fixed" "$ods_f1" "$ois_f1" \
    "$checkpoint" "$result_dir" >> "$METRICS"
}

run_job() {
  dataset="$1"
  seed="$2"
  slug=$(dataset_slug "$dataset") || return 1
  job_id="${slug}_wma_ctv_dgb_seed${seed}"
  job_out="$OUT_ROOT/$job_id"
  result_dir="$RESULT_ROOT/$job_id"
  runlog="$RUN_LOG_ROOT/${job_id}.train.log"
  inferlog="$RUN_LOG_ROOT/${job_id}.infer.log"
  evallog="$RUN_LOG_ROOT/${job_id}.eval.log"
  mkdir -p "$job_out" "$result_dir"

  checkpoint=$(find_checkpoint "$job_out")
  if [ -f "$result_dir/summary.csv" ] && [ -n "$checkpoint" ]; then
    echo "[$(now)] RESUME DONE dataset=${dataset} model=wma_ctv_dgb seed=${seed} checkpoint=${checkpoint}"
    append_metrics "$dataset" "$seed" "$checkpoint" "$result_dir" "$runlog"
    return 0
  fi

  if [ -z "$checkpoint" ]; then
    wait_for_gpu2
    echo "[$(now)] START dataset=${dataset} model=wma_ctv_dgb seed=${seed} epochs=50 gpu=${GPU}"
    CUDA_VISIBLE_DEVICES="$GPU" python -u main.py \
      --dataset_path "dataset/$dataset" \
      --nbins 180 \
      --seed "$seed" \
      --epochs 50 \
      --BCELoss_ratio 0.87 \
      --DiceLoss_ratio 0.13 \
      --output_dir "$job_out" \
      --use_triple_stack_v7 \
      --triple_stack_v7_mode wma_ctv_dgb 2>&1 | tee "$runlog"
    status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then
      echo "[$(now)] FAIL dataset=${dataset} model=wma_ctv_dgb seed=${seed} stage=train status=${status}"
      return 1
    fi
    checkpoint=$(find_checkpoint "$job_out")
    if [ -z "$checkpoint" ]; then
      echo "[$(now)] FAIL dataset=${dataset} model=wma_ctv_dgb seed=${seed} stage=train reason=no_checkpoint_best"
      return 1
    fi
  else
    wait_for_gpu2
    echo "[$(now)] RESUME EVAL dataset=${dataset} model=wma_ctv_dgb seed=${seed} checkpoint=${checkpoint}"
  fi

  CUDA_VISIBLE_DEVICES="$GPU" python -u tools/infer_probability.py \
    --dataset_path "dataset/$dataset" \
    --checkpoint "$checkpoint" \
    --save_dir "$result_dir" \
    --infer_phase test \
    --overwrite 2>&1 | tee "$inferlog"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "[$(now)] FAIL dataset=${dataset} model=wma_ctv_dgb seed=${seed} stage=infer status=${status}"
    return 1
  fi

  python -u tools/eval_standard.py \
    --result_dir "$result_dir" \
    --method "${dataset}_wma_ctv_dgb_seed${seed}" 2>&1 | tee "$evallog"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ] || [ ! -f "$result_dir/summary.csv" ]; then
    echo "[$(now)] FAIL dataset=${dataset} model=wma_ctv_dgb seed=${seed} stage=standard_eval status=${status}"
    return 1
  fi

  append_metrics "$dataset" "$seed" "$checkpoint" "$result_dir" "$runlog"
  echo "[$(now)] DONE dataset=${dataset} model=wma_ctv_dgb seed=${seed}"
  return 0
}

write_validation() {
  python - "$METRICS" "$CRACKMAP_REFERENCE" "$V7_SEED42_REFERENCE" \
    "$DEEPCRACK_REFERENCE" "$CAMCRACK789_REFERENCE" "$VALIDATION" <<'PY'
import csv
import statistics
import sys

(
    new_path,
    crackmap_reference_path,
    v7_seed42_path,
    deepcrack_reference_path,
    camcrack789_reference_path,
    output_path,
) = sys.argv[1:]

metrics = ["mIoU_fixed", "F1_fixed", "P_fixed", "R_fixed", "ODS_F1", "OIS_F1"]
std_fields = [f"{field}_std" for field in metrics]
seeds = ["42", "3407", "2026", "1234"]


def read_tsv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def numeric(row):
    return {field: float(row[field]) for field in metrics}


new_rows = read_tsv(new_path)
crackmap_reference = read_tsv(crackmap_reference_path)
v7_seed42_reference = read_tsv(v7_seed42_path)

crackmap = {"baseline": {}, "msa_ocv_gbc": {}, "wma_ctv_dgb": {}}
for row in crackmap_reference:
    if row.get("group") != "validate" or row.get("seed") not in seeds:
        continue
    if row.get("model") == "baseline":
        crackmap["baseline"][row["seed"]] = numeric(row)
    elif row.get("model") == "v3" and row.get("mode") == "msa_ocv_gbc":
        crackmap["msa_ocv_gbc"][row["seed"]] = numeric(row)

for row in v7_seed42_reference:
    if row.get("mode") == "wma_ctv_dgb" and row.get("seed") == "42":
        crackmap["wma_ctv_dgb"]["42"] = numeric(row)

for row in new_rows:
    if row["dataset"] == "CrackMap":
        crackmap["wma_ctv_dgb"][row["seed"]] = numeric(row)

for model, rows in crackmap.items():
    missing = [seed for seed in seeds if seed not in rows]
    if missing:
        raise RuntimeError(f"Missing CrackMap rows for {model}: {missing}")

header = [
    "row_type", "dataset", "model", "reference", "seed", "n",
    *metrics, *std_fields,
]
output = []


def append_result(dataset, model, seed, values, reference=""):
    output.append([
        "result", dataset, model, reference, seed, "1",
        *(f"{values[field]:.12f}" for field in metrics),
        *("" for _ in std_fields),
    ])


def append_summary(dataset, model, values_by_seed):
    means = {
        field: statistics.mean(values[field] for values in values_by_seed.values())
        for field in metrics
    }
    stds = {
        field: statistics.stdev(values[field] for values in values_by_seed.values())
        for field in metrics
    }
    output.append([
        "summary", dataset, model, "", "all", str(len(values_by_seed)),
        *(f"{means[field]:.12f}" for field in metrics),
        *(f"{stds[field]:.12f}" for field in metrics),
    ])
    return means


def append_paired_deltas(dataset, proposed, reference):
    deltas = {}
    for seed in seeds:
        values = {
            field: crackmap[proposed][seed][field] - crackmap[reference][seed][field]
            for field in metrics
        }
        deltas[seed] = values
        output.append([
            "paired_delta", dataset, proposed, reference, seed, "1",
            *(f"{values[field]:.12f}" for field in metrics),
            *("" for _ in std_fields),
        ])
    means = {
        field: statistics.mean(values[field] for values in deltas.values())
        for field in metrics
    }
    stds = {
        field: statistics.stdev(values[field] for values in deltas.values())
        for field in metrics
    }
    output.append([
        "paired_delta_summary", dataset, proposed, reference, "all", str(len(seeds)),
        *(f"{means[field]:.12f}" for field in metrics),
        *(f"{stds[field]:.12f}" for field in metrics),
    ])


for model in ("baseline", "msa_ocv_gbc", "wma_ctv_dgb"):
    for seed in seeds:
        append_result("CrackMap", model, seed, crackmap[model][seed])

crackmap_means = {
    model: append_summary("CrackMap", model, crackmap[model])
    for model in ("baseline", "msa_ocv_gbc", "wma_ctv_dgb")
}
append_paired_deltas("CrackMap", "wma_ctv_dgb", "baseline")
append_paired_deltas("CrackMap", "wma_ctv_dgb", "msa_ocv_gbc")

cross_dataset_pass = True
for dataset, reference_path in (
    ("DeepCrack", deepcrack_reference_path),
    ("CamCrack789", camcrack789_reference_path),
):
    reference_rows = {row["job"]: numeric(row) for row in read_tsv(reference_path)}
    proposed_rows = [
        numeric(row)
        for row in new_rows
        if row["dataset"] == dataset and row["seed"] == "42"
    ]
    if len(proposed_rows) != 1:
        raise RuntimeError(f"Expected one new {dataset} row, found {len(proposed_rows)}")
    values_by_model = {
        "baseline": reference_rows["baseline"],
        "msa_ocv_gbc": reference_rows["msa_ocv_gbc"],
        "wma_ctv_dgb": proposed_rows[0],
    }
    for model, values in values_by_model.items():
        append_result(dataset, model, "42", values)
    for reference in ("baseline", "msa_ocv_gbc"):
        delta = {
            field: values_by_model["wma_ctv_dgb"][field] - values_by_model[reference][field]
            for field in metrics
        }
        output.append([
            "delta", dataset, "wma_ctv_dgb", reference, "42", "1",
            *(f"{delta[field]:.12f}" for field in metrics),
            *("" for _ in std_fields),
        ])
    cross_dataset_pass = cross_dataset_pass and all(
        values_by_model["wma_ctv_dgb"][field] >= values_by_model["msa_ocv_gbc"][field]
        for field in ("mIoU_fixed", "F1_fixed")
    )

crackmap_pass = all(
    crackmap_means["wma_ctv_dgb"][field] > crackmap_means["msa_ocv_gbc"][field]
    for field in ("mIoU_fixed", "F1_fixed")
)
promote = crackmap_pass and cross_dataset_pass
decision = "promote_wma_ctv_dgb" if promote else "keep_msa_ocv_gbc"
output.append([
    "decision", "all", decision, "paper_main_model", "all", "",
    *("" for _ in metrics), *("" for _ in std_fields),
])

with open(output_path, "w", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(header)
    writer.writerows(output)

print(
    "CrackMap wma_ctv_dgb mean: "
    + " ".join(f"{field}={crackmap_means['wma_ctv_dgb'][field]:.6f}" for field in metrics)
)
print(
    "CrackMap msa_ocv_gbc mean: "
    + " ".join(f"{field}={crackmap_means['msa_ocv_gbc'][field]:.6f}" for field in metrics)
)
print(f"PROMOTION_DECISION={decision}")
PY
}

printf "dataset\tmodel\tseed\tlegacy_mIoU\tlegacy_F1\tlegacy_ODS\tlegacy_OIS\tmIoU_fixed\tF1_fixed\tP_fixed\tR_fixed\tODS_F1\tOIS_F1\tcheckpoint\tresult_dir\n" > "$METRICS"

echo "[$(now)] TripleStack-v7 wma_ctv_dgb core5 GPU2 queue started"
echo "[$(now)] conda=${CONDA_DEFAULT_ENV:-unknown} gpu=${GPU} jobs=${#JOBS[@]} epochs=50 nbins=180 bce=0.87 dice=0.13"

for required in "$CRACKMAP_REFERENCE" "$V7_SEED42_REFERENCE" "$DEEPCRACK_REFERENCE" "$CAMCRACK789_REFERENCE"; do
  if [ ! -f "$required" ]; then
    echo "[$(now)] FAIL preflight missing_reference=${required}"
    exit 1
  fi
done

wait_for_gpu2
echo "[$(now)] CUDA PREFLIGHT model=wma_ctv_dgb gpu=${GPU}" | tee "$PREFLIGHT_LOG"
CUDA_VISIBLE_DEVICES="$GPU" python -u tools/profile_model.py \
  --nbins 180 --warmup 1 --repeat 1 \
  --use_triple_stack_v7 --triple_stack_v7_mode wma_ctv_dgb 2>&1 | tee -a "$PREFLIGHT_LOG"
preflight_status=${PIPESTATUS[0]}
if [ "$preflight_status" -ne 0 ]; then
  echo "[$(now)] FAIL preflight status=${preflight_status}"
  exit "$preflight_status"
fi

failures=0
for entry in "${JOBS[@]}"; do
  IFS='|' read -r dataset seed <<< "$entry"
  run_job "$dataset" "$seed" || failures=$((failures + 1))
done

if [ "$failures" -eq 0 ]; then
  if write_validation; then
    echo "[$(now)] VALIDATION SUMMARY"
    column -t -s $'\t' "$VALIDATION" || true
  else
    echo "[$(now)] FAIL stage=validation_summary"
    failures=$((failures + 1))
  fi
fi

echo "[$(now)] TripleStack-v7 wma_ctv_dgb core5 GPU2 finished with failures=${failures}"
exit "$failures"
