#!/usr/bin/env bash

cd /mnt/sdb1/lm/MixerCSeg || exit 1
source /home/lm/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate MixerCSeg || exit 1
set -u

if [ -z "${PREFIX:-}" ]; then
  TS=$(date "+%Y%m%d_%H%M%S")
  PREFIX="logs/${TS}_CrackMap_triple_stack_v3_screen20_top5_onegpu_no_gpu1"
fi
SCREEN_OUTDIR="${SCREEN_OUTDIR:-work_dirs/triple_stack_v3_screen20}"
FULL_OUTDIR="${FULL_OUTDIR:-work_dirs/triple_stack_v3_full50}"
mkdir -p logs "$SCREEN_OUTDIR" "$FULL_OUTDIR"

MODES=(
  dsc_htm_gbc dsc_htm_prb dsc_htm_egc dsc_rtm_gbc dsc_rtm_prb
  dsc_ocv_gbc dsc_llc_bns fsc_htm_gbc fsc_htm_prb fsc_rtm_gbc
  fsc_ocv_egc fsc_llc_bns msa_htm_gbc msa_htm_egc msa_rtm_prb
  msa_ocv_gbc sta_htm_gbc sta_rtm_prb sta_ocv_egc sta_llc_bns
)

SCREEN_METRICS="${PREFIX}.screen_metrics.tsv"
TOP5_FILE="${PREFIX}.top5_modes.txt"
: > "$SCREEN_METRICS"
: > "$TOP5_FILE"

now() { date "+%Y-%m-%d %H:%M:%S"; }
trim() { sed "s/^ *//;s/ *$//"; }

select_gpu() {
  while true; do
    while IFS="," read -r idx mem util; do
      idx=$(printf "%s" "$idx" | trim)
      mem=$(printf "%s" "$mem" | trim)
      util=$(printf "%s" "$util" | trim)
      if { [ "$idx" = "0" ] || [ "$idx" = "2" ]; } && [ "$mem" -lt 1000 ] && [ "$util" -lt 10 ]; then
        printf "%s\n" "$idx"
        return 0
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
    echo "[$(now)] no allowed idle GPU yet; waiting 60s"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
    sleep 60
  done
}

parse_metric() {
  metric_name="$1"
  line="$2"
  printf "%s\n" "$line" | sed -n "s/.*${metric_name} -> \([^ |]*\).*/\1/p"
}

run_train() {
  phase="$1"
  mode="$2"
  epochs="$3"
  outdir="$4"
  label=$(printf "%s" "$phase" | tr "[:lower:]" "[:upper:]")
  runlog="${PREFIX}.${phase}.${mode}.log"
  echo "[$(now)] ${label} START mode=${mode} epochs=${epochs} outdir=${outdir} gpu=${GPU}"
  CUDA_VISIBLE_DEVICES="$GPU" python -u main.py \
    --dataset_path dataset/CrackMap \
    --nbins 180 \
    --seed 42 \
    --epochs "$epochs" \
    --output_dir "$outdir" \
    --use_triple_stack_v3 \
    --triple_stack_v3_mode "$mode" 2>&1 | tee "$runlog"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "[$(now)] ${label} FAIL mode=${mode} status=${status}"
    return 1
  fi
  line=$(grep -a "Best metrics | experiment ->" "$runlog" | tail -1 || true)
  if [ -z "$line" ]; then
    echo "[$(now)] ${label} FAIL mode=${mode} reason=no_best_metrics_line"
    return 1
  fi
  miou=$(parse_metric mIoU "$line")
  f1=$(parse_metric F1 "$line")
  ods=$(parse_metric ODS "$line")
  ois=$(parse_metric OIS "$line")
  if [ "$phase" = "screen" ]; then
    printf "%s\t%s\t%s\t%s\t%s\n" "$mode" "$miou" "$f1" "$ods" "$ois" >> "$SCREEN_METRICS"
  fi
  echo "[$(now)] ${label} DONE mode=${mode} mIoU=${miou} F1=${f1} ODS=${ods} OIS=${ois}"
  return 0
}

echo "[$(now)] triple-stack-v3 screen20 top5 onegpu no-gpu1 queue started"
echo "[$(now)] modes=${#MODES[@]} screen_epochs=20 full_epochs=50 allowed_gpus=0,2"
GPU=$(select_gpu)
echo "[$(now)] selected GPU ${GPU}"

failures=0
for mode in "${MODES[@]}"; do
  if ! run_train screen "$mode" 20 "$SCREEN_OUTDIR"; then
    failures=$((failures + 1))
  fi
done

if [ -s "$SCREEN_METRICS" ]; then
  TAB=$(printf "\t")
  sort -t "$TAB" -k2,2gr -k3,3gr "$SCREEN_METRICS" | head -5 | cut -f1 > "$TOP5_FILE"
  echo "[$(now)] SCREEN TOP5"
  sort -t "$TAB" -k2,2gr -k3,3gr "$SCREEN_METRICS" | head -5
  while read -r mode; do
    [ -n "$mode" ] || continue
    if ! run_train full "$mode" 50 "$FULL_OUTDIR"; then
      failures=$((failures + 1))
    fi
  done < "$TOP5_FILE"
else
  echo "[$(now)] SCREEN produced no metrics; skip full retraining"
  failures=$((failures + 1))
fi

echo "[$(now)] triple-stack-v3 screen20 top5 onegpu no-gpu1 finished with failures=${failures}"
