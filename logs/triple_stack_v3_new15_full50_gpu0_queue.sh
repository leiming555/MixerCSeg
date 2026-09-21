#!/usr/bin/env bash

cd /mnt/sdb1/lm/MixerCSeg || exit 1
source /home/lm/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate MixerCSeg || exit 1
set -u

if [ -z "${PREFIX:-}" ]; then
  TS=$(date "+%Y%m%d_%H%M%S")
  PREFIX="logs/${TS}_CrackMap_triple_stack_v3_new15_full50_gpu0"
fi
OUTDIR="${OUTDIR:-work_dirs/triple_stack_v3_new15_full50_gpu0}"
mkdir -p logs "$OUTDIR"

MODES=(
  msa_llc_prb
  msa_llc_egc
  msa_llc_bns
  msa_ocv_bns
  msa_htm_bns
  msa_rtm_bns
  dsc_htm_bns
  dsc_rtm_bns
  dsc_ocv_bns
  dsc_llc_prb
  dsc_llc_egc
  fsc_ocv_gbc
  fsc_ocv_prb
  fsc_rtm_prb
  sta_ocv_gbc
)

GPU=0
SUMMARY="${PREFIX}.metrics.tsv"
: > "$SUMMARY"

now() { date "+%Y-%m-%d %H:%M:%S"; }

parse_metric() {
  metric_name="$1"
  line="$2"
  printf "%s\n" "$line" | sed -n "s/.*${metric_name} -> \([^ |]*\).*/\1/p"
}

run_train() {
  mode="$1"
  runlog="${PREFIX}.${mode}.log"
  echo "[$(now)] START mode=${mode} epochs=50 outdir=${OUTDIR} gpu=${GPU}"
  CUDA_VISIBLE_DEVICES="$GPU" python -u main.py \
    --dataset_path dataset/CrackMap \
    --nbins 180 \
    --seed 42 \
    --epochs 50 \
    --output_dir "$OUTDIR" \
    --use_triple_stack_v3 \
    --triple_stack_v3_mode "$mode" 2>&1 | tee "$runlog"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "[$(now)] FAIL mode=${mode} status=${status}"
    return 1
  fi
  line=$(grep -a "Best metrics | experiment ->" "$runlog" | tail -1 || true)
  if [ -z "$line" ]; then
    echo "[$(now)] FAIL mode=${mode} reason=no_best_metrics_line"
    return 1
  fi
  miou=$(parse_metric mIoU "$line")
  f1=$(parse_metric F1 "$line")
  ods=$(parse_metric ODS "$line")
  ois=$(parse_metric OIS "$line")
  precision=$(parse_metric Precision "$line")
  recall=$(parse_metric Recall "$line")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$mode" "$miou" "$f1" "$ods" "$ois" "$precision" "$recall" >> "$SUMMARY"
  echo "[$(now)] DONE mode=${mode} mIoU=${miou} F1=${f1} ODS=${ods} OIS=${ois} Precision=${precision} Recall=${recall}"
  return 0
}

echo "[$(now)] triple-stack-v3 new15 full50 gpu0 queue started"
echo "[$(now)] modes=${#MODES[@]} epochs=50 gpu=${GPU}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits

failures=0
for mode in "${MODES[@]}"; do
  if ! run_train "$mode"; then
    failures=$((failures + 1))
  fi
done

echo "[$(now)] RESULT SORTED"
TAB=$(printf "\t")
sort -t "$TAB" -k2,2gr -k3,3gr "$SUMMARY" || true
echo "[$(now)] triple-stack-v3 new15 full50 gpu0 finished with failures=${failures}"
