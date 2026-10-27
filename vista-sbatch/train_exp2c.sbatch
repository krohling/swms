#!/bin/bash
#SBATCH -J swm-exp2c
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -o %x-%j.out
# Experiment 2c: planner-aligned selector, listwise, COLD LoRA adapters.
# Ablation of exp2a's LoRA warm start. Needs ONLY the selection warm-up
# artifact (trajectory adapter + cand_embed) — no exp1b dependency, so it
# is submitted WITHOUT --dependency and may start before exp2a/2b.
# Submit as a 2-leg afterany chain (--resume auto) like the others.
set -x
W=$WORK/swm-followup/swms-exp2
V=$WORK/swm-followup/swms-lt/.venv
cd $W
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $WORK/swm-followup/.wandb_env
export WANDB_DATA_DIR=$WORK/wandb_data
# pre-flight: the one init artifact this arm needs.
[ -f $W/outputs_swm_lora/warmup_sel/ckpt_step6000/projector.pt ] || { echo INIT_MISSING_WARMUP_SEL; exit 1; }
[ -f $W/outputs_swm_lora/warmup_sel/ckpt_step6000/selector_extra.pt ] || { echo INIT_MISSING_WARMUP_SEL_EXTRA; exit 1; }
$V/bin/python swm-next/swm_lora/train_exp2.py --config swm-next/swm_lora/train_exp2c.yaml --resume auto || { echo TRAIN_JOB_FAILED; exit 1; }
echo TRAIN_JOB_DONE
