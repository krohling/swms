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
# pre-flight: never burn the queue slot without the init artifacts.
# exp1b must have FINISHED (not merely ended) and the selection-warmed
# encoder must already be staged; ckpt_final resolves to exp1b's last step.
E1B=$W/outputs_swm_lora/exp1b_full
L=$(ls -t $WORK/swm-followup/swms-exp1/swm-exp1b-*.out 2>/dev/null | head -1)
grep -aq TRAIN_JOB_DONE "$L" || { echo INIT_MISSING_EXP1B_DONE; exit 1; }
LAST=$(ls -d $E1B/ckpt_step* 2>/dev/null | sort -V | tail -1)
[ -n "$LAST" ] || { echo INIT_MISSING_EXP1B_CKPT; exit 1; }
ln -sfn $(basename $LAST) $E1B/ckpt_final
echo "ckpt_final -> $(readlink $E1B/ckpt_final)"
[ -f $W/outputs_swm_lora/warmup_sel/ckpt_step6000/projector.pt ] || { echo INIT_MISSING_WARMUP_SEL; exit 1; }
$V/bin/python swm-next/swm_lora/train_exp2.py --config swm-next/swm_lora/train_exp2a.yaml --resume auto || { echo TRAIN_JOB_FAILED; exit 1; }
echo TRAIN_JOB_DONE
