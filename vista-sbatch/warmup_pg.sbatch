#!/bin/bash
#SBATCH -J swm-warmup-pg
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -o %x-%j.out
# PaliGemma projector warm-up (branch paligemma-fullft): winning recipe —
# uniform, no scalar, run to 6000, final ckpt feeds train_pg_fullft.
set -x
W=$WORK/swm-followup/swms-pg
V=$WORK/swm-followup/swms-lt/.venv
cd $W
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $WORK/swm-followup/.wandb_env
export WANDB_DATA_DIR=$WORK/wandb_data
$V/bin/python swm-next/swm_lora/warmup_encoder.py --config swm-next/swm_lora/warmup_pg.yaml || { echo WARMUP_JOB_FAILED; exit 1; }
echo WARMUP_JOB_DONE
