#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1

set +u
source /home/lm/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate MixerCSeg || exit 1
set -u

GPU="${GPU:-1}"
POLL_SECONDS="${POLL_SECONDS:-60}"
DATASET_PATH="${DATASET_PATH:-dataset/CrackMap}"
NBINS="${NBINS:-180}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-50}"
BCE_RATIO="${BCE_RATIO:-0.87}"
DICE_RATIO="${DICE_RATIO:-0.13}"

DATASET_NAME="$(basename "${DATASET_PATH}")"
OUT_ROOT="${OUT_ROOT:-work_dirs/triple_stack_v3_ablation_crackmap_gpu1}"
RESULT_ROOT="${RESULT_ROOT:-results/probability_maps/triple_stack_v3_ablation_crackmap_gpu1}"
RUN_LOG_ROOT="${RUN_LOG_ROOT:-logs/triple_stack_v3_ablation_crackmap_gpu1_runs}"
PREFIX="${PREFIX:-logs/$(date '+%Y%m%d_%H%M%S')_${DATASET_NAME}_triple_stack_v3_ablation_gpu1}"
METRICS="${PREFIX}.metrics.tsv"
ABLATION_SUMMARY="${PREFIX}.ablation.tsv"
DRY_RUN=0

if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
fi

V3_MODES=(
  id_id_id
  msa_id_id
  id_ocv_id
  id_id_gbc
  msa_ocv_id
  msa_id_gbc
  id_ocv_gbc
  msa_ocv_gbc
)

now() { date "+%Y-%m-%d %H:%M:%S"; }

job_id_for() {
  model="$1"
  mode="$2"
  if [ "${model}" = "baseline" ]; then
    printf "baseline_seed%s" "${SEED}"
  else
    printf "v3_%s_seed%s" "${mode}" "${SEED}"
  fi
}

print_manifest() {
  echo "DRYRUN job=$(job_id_for baseline none) model=baseline mode=none seed=${SEED} gpu=${GPU}"
  for mode in "${V3_MODES[@]}"; do
    echo "DRYRUN job=$(job_id_for v3 "${mode}") model=v3 mode=${mode} seed=${SEED} gpu=${GPU}"
  done
}

if [ "${DRY_RUN}" -eq 1 ]; then
  print_manifest
  exit 0
fi

mkdir -p logs "${OUT_ROOT}" "${RESULT_ROOT}" "${RUN_LOG_ROOT}"

wait_for_gpu() {
  while true; do
    gpu_line=$(nvidia-smi --id="${GPU}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)
    memory=$(printf "%s" "${gpu_line}" | cut -d',' -f1 | tr -d ' ')
    utilization=$(printf "%s" "${gpu_line}" | cut -d',' -f2 | tr -d ' ')
    if [[ "${memory}" =~ ^[0-9]+$ ]] && [[ "${utilization}" =~ ^[0-9]+$ ]] \
      && [ "${memory}" -lt 1000 ] && [ "${utilization}" -lt 10 ]; then
      echo "[$(now)] selected GPU ${GPU} memory=${memory}MiB utilization=${utilization}%"
      return 0
    fi
    echo "[$(now)] waiting GPU ${GPU} memory=${memory:-unknown}MiB utilization=${utilization:-unknown}%"
    sleep "${POLL_SECONDS}"
  done
}

find_checkpoint() {
  job_out="$1"
  find "${job_out}" -type f -name checkpoint_best.pth -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -1 | cut -d' ' -f2-
}

parse_legacy_metric() {
  metric_name="$1"
  line="$2"
  printf "%s\n" "${line}" | sed -n "s/.*${metric_name} -> \([^ |]*\).*/\1/p"
}

append_metrics() {
  job_id="$1"
  model="$2"
  mode="$3"
  checkpoint="$4"
  result_dir="$5"
  runlog="$6"

  legacy_line=$(grep -a "Best metrics | experiment ->" "${runlog}" 2>/dev/null | tail -1 || true)
  legacy_miou=$(parse_legacy_metric mIoU "${legacy_line}")
  legacy_f1=$(parse_legacy_metric F1 "${legacy_line}")
  legacy_ods=$(parse_legacy_metric ODS "${legacy_line}")
  legacy_ois=$(parse_legacy_metric OIS "${legacy_line}")

  standard_values=$(python - "${result_dir}/summary.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as handle:
    row = next(csv.DictReader(handle))
fields = ["mIoU_fixed", "F1_fixed", "P_fixed", "R_fixed", "ODS_F1", "OIS_F1"]
print("\t".join(row[field] for field in fields))
PY
)
  IFS=$'\t' read -r miou_fixed f1_fixed p_fixed r_fixed ods_f1 ois_f1 <<< "${standard_values}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${job_id}" "${model}" "${mode}" "${SEED}" \
    "${legacy_miou}" "${legacy_f1}" "${legacy_ods}" "${legacy_ois}" \
    "${miou_fixed}" "${f1_fixed}" "${p_fixed}" "${r_fixed}" "${ods_f1}" "${ois_f1}" \
    "${checkpoint}" >> "${METRICS}"
}

run_job() {
  model="$1"
  mode="$2"
  job_id=$(job_id_for "${model}" "${mode}")
  job_out="${OUT_ROOT}/${job_id}"
  result_dir="${RESULT_ROOT}/${job_id}"
  runlog="${RUN_LOG_ROOT}/${job_id}.train.log"
  inferlog="${RUN_LOG_ROOT}/${job_id}.infer.log"
  evallog="${RUN_LOG_ROOT}/${job_id}.eval.log"
  train_done="${job_out}/.train_done"
  mkdir -p "${job_out}" "${result_dir}"

  checkpoint=$(find_checkpoint "${job_out}")
  if [ -f "${result_dir}/summary.csv" ] && [ -f "${train_done}" ] && [ -n "${checkpoint}" ]; then
    echo "[$(now)] RESUME DONE job=${job_id} checkpoint=${checkpoint}"
    append_metrics "${job_id}" "${model}" "${mode}" "${checkpoint}" "${result_dir}" "${runlog}"
    return 0
  fi

  if [ ! -f "${train_done}" ] || [ -z "${checkpoint}" ]; then
    cmd=(
      python -u main.py
      --dataset_path "${DATASET_PATH}"
      --nbins "${NBINS}"
      --seed "${SEED}"
      --epochs "${EPOCHS}"
      --BCELoss_ratio "${BCE_RATIO}"
      --DiceLoss_ratio "${DICE_RATIO}"
      --output_dir "${job_out}"
    )
    if [ "${model}" = "v3" ]; then
      cmd+=(--use_triple_stack_v3 --triple_stack_v3_mode "${mode}")
    fi

    wait_for_gpu
    echo "[$(now)] START job=${job_id} model=${model} mode=${mode} seed=${SEED} gpu=${GPU}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${cmd[@]}" 2>&1 | tee "${runlog}"
    status=${PIPESTATUS[0]}
    if [ "${status}" -ne 0 ]; then
      echo "[$(now)] FAIL job=${job_id} stage=train status=${status}"
      return 1
    fi
    checkpoint=$(find_checkpoint "${job_out}")
    if [ -z "${checkpoint}" ]; then
      echo "[$(now)] FAIL job=${job_id} stage=train reason=no_checkpoint_best"
      return 1
    fi
    touch "${train_done}"
  else
    echo "[$(now)] RESUME EVAL job=${job_id} checkpoint=${checkpoint}"
  fi

  wait_for_gpu
  CUDA_VISIBLE_DEVICES="${GPU}" python -u tools/infer_probability.py \
    --dataset_path "${DATASET_PATH}" \
    --checkpoint "${checkpoint}" \
    --save_dir "${result_dir}" \
    --infer_phase test \
    --overwrite 2>&1 | tee "${inferlog}"
  status=${PIPESTATUS[0]}
  if [ "${status}" -ne 0 ]; then
    echo "[$(now)] FAIL job=${job_id} stage=infer status=${status}"
    return 1
  fi

  python -u tools/eval_standard.py \
    --result_dir "${result_dir}" \
    --method "${job_id}" 2>&1 | tee "${evallog}"
  status=${PIPESTATUS[0]}
  if [ "${status}" -ne 0 ] || [ ! -f "${result_dir}/summary.csv" ]; then
    echo "[$(now)] FAIL job=${job_id} stage=standard_eval status=${status}"
    return 1
  fi

  append_metrics "${job_id}" "${model}" "${mode}" "${checkpoint}" "${result_dir}" "${runlog}"
  echo "[$(now)] DONE job=${job_id} model=${model} mode=${mode} seed=${SEED}"
  return 0
}

write_ablation_summary() {
  python - "${METRICS}" "${ABLATION_SUMMARY}" <<'PY'
import csv
import sys

metrics_path, output_path = sys.argv[1:]
metric_names = ["mIoU_fixed", "F1_fixed", "ODS_F1", "OIS_F1"]
with open(metrics_path, newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

baseline = next((row for row in rows if row["model"] == "baseline"), None)
header = ["model", "mode", "MSA", "OCV", "GBC", "wrapper_control", *metric_names]
header.extend(f"delta_{metric}" for metric in metric_names)

with open(output_path, "w", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(header)
    for row in rows:
        mode = row["mode"]
        parts = mode.split("_") if row["model"] == "v3" else []
        flags = ["1" if name in parts else "0" for name in ("msa", "ocv", "gbc")]
        control = "1" if mode == "id_id_id" else "0"
        values = [row[metric] for metric in metric_names]
        if baseline is None:
            deltas = [""] * len(metric_names)
        else:
            deltas = [f"{float(row[metric]) - float(baseline[metric]):.9f}" for metric in metric_names]
        writer.writerow([row["model"], mode, *flags, control, *values, *deltas])
PY
}

printf "job_id\tmodel\tmode\tseed\tlegacy_mIoU\tlegacy_F1\tlegacy_ODS\tlegacy_OIS\tmIoU_fixed\tF1_fixed\tP_fixed\tR_fixed\tODS_F1\tOIS_F1\tcheckpoint\n" > "${METRICS}"

echo "[$(now)] TripleStack-v3 MSA/OCV/GBC ablation started"
echo "[$(now)] conda=${CONDA_DEFAULT_ENV:-unknown} python=$(command -v python) gpu=${GPU} dataset=${DATASET_PATH} seed=${SEED} epochs=${EPOCHS}"

failures=0
run_job baseline none || failures=$((failures + 1))
for mode in "${V3_MODES[@]}"; do
  run_job v3 "${mode}" || failures=$((failures + 1))
done

write_ablation_summary

echo "[$(now)] RESULT ablation_summary=${ABLATION_SUMMARY}"
column -t -s $'\t' "${ABLATION_SUMMARY}" 2>/dev/null || cat "${ABLATION_SUMMARY}"
echo "[$(now)] TripleStack-v3 MSA/OCV/GBC ablation finished with failures=${failures}"
exit "${failures}"
