#!/bin/bash
#SBATCH -J swm-exp1b
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -o %x-%j.out
# Experiment 1b: exp1 recipe on v2-R10 data, no-scale warm-start.
# Submit as a 2-leg afterany chain (--resume auto) like run4.
set -x
W=$WORK/swm-followup/swms-exp1
V=$WORK/swm-followup/swms-lt/.venv
cd $W
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $WORK/swm-followup/.wandb_env
export WANDB_DATA_DIR=$WORK/wandb_data
$V/bin/python swm-next/swm_lora/train_exp1.py --config swm-next/swm_lora/train_exp1b.yaml --resume auto || { echo TRAIN_JOB_FAILED; exit 1; }
echo TRAIN_JOB_DONE
