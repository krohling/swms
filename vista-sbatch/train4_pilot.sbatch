#!/bin/bash
#SBATCH -J swm-run4pilot
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o %x-%j.out
set -x
W=$WORK/swm-followup/swms-datagen
V=$WORK/swm-followup/swms-lt/.venv
cd $W
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $WORK/swm-followup/.wandb_env
$V/bin/python swm-next/swm_lora/train4.py --config swm-next/swm_lora/train4_pilot.yaml --resume auto
echo TRAIN_JOB_DONE
