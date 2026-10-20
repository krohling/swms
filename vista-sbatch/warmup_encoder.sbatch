#!/bin/bash
#SBATCH -J swm-exp1-warmup
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o %x-%j.out
# Experiment 1 stage 1: encoder warm-up (projector only, no LoRA, fixed LR).
# Runs from the swms-exp1 checkout (branch exp1-allinear-warmup).
set -x
W=$WORK/swm-followup/swms-exp1
V=$WORK/swm-followup/swms-lt/.venv
cd $W
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $WORK/swm-followup/.wandb_env
export WANDB_DATA_DIR=$WORK/wandb_data
$V/bin/python swm-next/swm_lora/warmup_encoder.py --config swm-next/swm_lora/warmup_encoder.yaml || { echo WARMUP_JOB_FAILED; exit 1; }
echo WARMUP_JOB_DONE
