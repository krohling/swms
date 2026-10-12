#!/bin/bash
#SBATCH -J swm-run5sel
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -o %x-%j.out
# Run 5 (selection arm) — runs from the swms-sel checkout (branch selection-arm),
# fully isolated from the run4 scoring arm in swms-datagen.
set -x
W=$WORK/swm-followup/swms-sel
V=$WORK/swm-followup/swms-lt/.venv
cd $W
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $WORK/swm-followup/.wandb_env
$V/bin/python swm-next/swm_lora/train_sel.py --config swm-next/swm_lora/train_sel_hybrid.yaml --resume auto || { echo TRAIN_JOB_FAILED; exit 1; }
echo TRAIN_JOB_DONE
