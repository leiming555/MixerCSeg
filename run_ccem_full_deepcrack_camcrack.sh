#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-MixerCSeg}"
GPU="${GPU:-0}"
NBINS="${NBINS:-180}"
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

run_ccem_full() {
    local dataset_path="$1"
    local dataset_name
    dataset_name="$(basename "${dataset_path}")"
    local exp_name="${dataset_name}_seed${SEED}_ccem_full"
    local log_file="${LOG_DIR}/${exp_name}.log"

    echo "================================================================"
    echo "Running ${exp_name}"
    echo "Dataset: ${dataset_path}"
    echo "NBINS: ${NBINS}"
    echo "Log: ${log_file}"
    echo "================================================================"

    CUDA_VISIBLE_DEVICES="${GPU}" python main.py \
        --dataset_path "${dataset_path}" \
        --nbins "${NBINS}" \
        --seed "${SEED}" \
        --output_dir "${WORK_DIR}" \
        --use_ccem \
        --ccem_mode full \
        > "${log_file}" 2>&1
}

run_ccem_full "dataset/DeepCrack"
run_ccem_full "dataset/CamCrack789"

echo "DeepCrack and CamCrack789 CCEM full experiments finished."
