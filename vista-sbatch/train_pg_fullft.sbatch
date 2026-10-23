#!/bin/bash
#SBATCH -J swm-pg-fullft
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -o %x-%j.out
# PaliGemma full-weight score regression on v2-R10 (branch paligemma-fullft).
# Submit leg 1 with --dependency=afterok:<warmup job>, leg 2 afterany:<leg1>
# (--resume auto picks up the newest resume state).
set -x
W=$WORK/swm-followup/swms-pg
V=$WORK/swm-followup/swms-lt/.venv
cd $W
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $WORK/swm-followup/.wandb_env
export WANDB_DATA_DIR=$WORK/wandb_data
$V/bin/python swm-next/swm_lora/train_pg_fullft.py --config swm-next/swm_lora/train_pg_fullft.yaml --resume auto || { echo TRAIN_JOB_FAILED; exit 1; }
echo TRAIN_JOB_DONE
