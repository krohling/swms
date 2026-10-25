#!/bin/bash
#SBATCH -J swm-compose
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o %x-%j.out
# Compose baseline (exp2 step 1): how well a scoring checkpoint's per-question
# predictions blend (with the STORED gate) into the planner's decision.
# Args: $1 ckpt dir, $2 train yaml, $3 out json (defaults: exp1 final).
set -x
W=$WORK/swm-followup/swms-exp2
V=$WORK/swm-followup/swms-lt/.venv
cd $W
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKPT=${1:-../swms-exp1/outputs_swm_lora/exp1_full/ckpt_step84687}
TCFG=${2:-swm-next/swm_lora/train_exp1.yaml}
OUT=${3:-experiments/exp2/compose_exp1_final.json}
$V/bin/python swm-next/swm_lora/eval_compose.py \
  --ckpt "$CKPT" --train-config "$TCFG" \
  --h5 datasets_swm/bg_hybrid_v2_k8_r10.h5 \
  --max-cycles 4000 --device cuda --out "$OUT" \
  || { echo COMPOSE_JOB_FAILED; exit 1; }
echo COMPOSE_JOB_DONE
