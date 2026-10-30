#!/bin/bash
#SBATCH -J swm-exp2c84k
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -o %x-%j.out
# exp2c-84k: fresh full-budget cold-LoRA listwise selector (team call
# 2026-10-30). ~84.7k steps @ ~2.3k steps/h ~ 37h -> one 48h leg should
# finish; submit a second leg with --dependency=afterany:<leg1> as walltime/
# crash insurance. Same script both legs: --resume auto starts fresh on an
# empty out_dir and resumes otherwise. Legs are serial (afterany), so there
# is no live-writer race; the DONE guard stops a redundant leg-2.
set -x
W=$WORK/swm-followup/swms-exp2
V=$WORK/swm-followup/swms-lt/.venv
cd $W
# anchored: under set -x this script's own trace lines quote the pattern
if grep -q '^TRAIN_JOB_DONE' $W/swm-exp2c84k-*.out 2>/dev/null; then echo RESCUE_NOOP_ALREADY_DONE; exit 0; fi
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $WORK/swm-followup/.wandb_env
export WANDB_DATA_DIR=$WORK/wandb_data
[ -f $W/outputs_swm_lora/warmup_sel/ckpt_step6000/projector.pt ] || { echo INIT_MISSING_WARMUP_SEL; exit 1; }
[ -f $W/outputs_swm_lora/warmup_sel/ckpt_step6000/selector_extra.pt ] || { echo INIT_MISSING_WARMUP_SEL_EXTRA; exit 1; }
$V/bin/python swm-next/swm_lora/train_exp2.py --config swm-next/swm_lora/train_exp2c_84k.yaml --resume auto || { echo TRAIN_JOB_FAILED; exit 1; }
echo TRAIN_JOB_DONE
