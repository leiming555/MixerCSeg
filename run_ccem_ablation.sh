#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-MixerCSeg}"
GPU="${GPU:-0}"
DATASET_PATH="${DATASET_PATH:-dataset/Crack500}"
NBINS="${NBINS:-36}"
SEED="${SEED:-42}"
LOG_DIR="${LOG_DIR:-logs}"
WORK_DIR="${WORK_DIR:-work_dirs}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

DATASET_NAME="$(basename "${DATASET_PATH}")"

run_experiment() {
    local mode="$1"
    local exp_name="${DATASET_NAME}_seed${SEED}_ccem_${mode}"
    local log_file="${LOG_DIR}/${exp_name}.log"

    echo "================================================================"
    echo "Running ${exp_name}"
    echo "Log: ${log_file}"
    echo "================================================================"

    if [[ "${mode}" == "baseline" ]]; then
        CUDA_VISIBLE_DEVICES="${GPU}" python main.py \
            --dataset_path "${DATASET_PATH}" \
            --nbins "${NBINS}" \
            --seed "${SEED}" \
            --output_dir "${WORK_DIR}" \
            > "${log_file}" 2>&1
    else
        CUDA_VISIBLE_DEVICES="${GPU}" python main.py \
            --dataset_path "${DATASET_PATH}" \
            --nbins "${NBINS}" \
            --seed "${SEED}" \
            --output_dir "${WORK_DIR}" \
            --use_ccem \
            --ccem_mode "${mode}" \
            > "${log_file}" 2>&1
    fi
}

run_experiment baseline
run_experiment full
run_experiment no_local
run_experiment no_strip
run_experiment no_dilation
run_experiment no_gate

echo "All CCEM ablation experiments finished."
