#!/bin/bash
#SBATCH -J swm-exp2a
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -o %x-%j.out
# Experiment 2a: planner-aligned selector, listwise soft-CE arm.
# Prereq: exp1b finished + outputs_swm_lora/exp1b_full/ckpt_final symlink.
# Submit as a 2-leg afterany chain (--resume auto) like exp1b.
set -x
W=$WORK/swm-followup/swms-exp2
V=$WORK/swm-followup/swms-lt/.venv
cd $W
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $WORK/swm-followup/.wandb_env
export WANDB_DATA_DIR=$WORK/wandb_data
$V/bin/python swm-next/swm_lora/train_exp2.py --config swm-next/swm_lora/train_exp2a.yaml --resume auto || { echo TRAIN_JOB_FAILED; exit 1; }
echo TRAIN_JOB_DONE
