#!/bin/bash
#SBATCH -J swm-exp4c
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -o %x-%j.out
# Experiment 4c: exp4a at double LoRA capacity, r32/a64 (train_exp4c.yaml).
# Same two-leg pattern as exp4a: submit a leg-2 with
# --dependency=afterany:<leg1> as crash/walltime insurance (--resume auto;
# the DONE guard no-ops a redundant leg-2). Warm-starts the trajectory
# adapter from wu4a's picked noscale ckpt (guarded below); the r32 LoRA
# trains from scratch. Runs from $WORK/swm-followup/swms-exp4.
set -x
W=$WORK/swm-followup/swms-exp4
V=$WORK/swm-followup/swms-lt/.venv
cd $W
# anchored: under set -x this script's own trace lines quote the pattern
if grep -q '^TRAIN_JOB_DONE' $W/swm-exp4c-*.out 2>/dev/null; then echo RESCUE_NOOP_ALREADY_DONE; exit 0; fi
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $WORK/swm-followup/.wandb_env
export WANDB_DATA_DIR=$WORK/wandb_data
[ -f $W/datasets_swm/bg_v3a_k1tau1_s42000.h5 ] || { echo INIT_MISSING_V3A_H5; exit 1; }
[ -f $W/ckpts/ogbench_base_diffusion/blue_cube_green_cube.pt ] || { echo INIT_MISSING_DIFFUSION_CKPT; exit 1; }
[ -f $W/outputs_swm_lora/exp4a_warmup_v3a/ckpt_step5500/projector.pt ] || { echo INIT_MISSING_WARMUP_PROJECTOR; exit 1; }
$V/bin/python swm-next/swm_lora/train_exp1.py --config swm-next/swm_lora/train_exp4c.yaml --resume auto || { echo TRAIN_JOB_FAILED; exit 1; }
echo TRAIN_JOB_DONE
