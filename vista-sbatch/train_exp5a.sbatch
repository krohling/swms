#!/bin/bash
#SBATCH -J swm-exp5a
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -o %x-%j.out
# Experiment 5a: exp4a's recipe on the v5 OGBench-oracle play dataset
# (train_exp5a.yaml). ~92k steps, one 48h leg should cover it; submit a leg-2
# with --dependency=afterany:<leg1> as crash/walltime insurance (--resume
# auto; the DONE guard no-ops a redundant leg-2). Runs from the exp5 checkout
# $WORK/swm-followup/swms-exp5 (branch exp5-ogb-play).
# Submit ONLY at pick-lock (after wu5's checkpoint is chosen and wired into
# train_exp5a.yaml) — never submit-and-hold: a Slurm hold resets accrued age
# priority on release (verified 2026-11-07, cost exp4a/b ~1 day).
set -x
W=$WORK/swm-followup/swms-exp5
V=$WORK/swm-followup/swms-lt/.venv
cd $W
# anchored: under set -x this script's own trace lines quote the pattern
if grep -q '^TRAIN_JOB_DONE' $W/swm-exp5a-*.out 2>/dev/null; then echo RESCUE_NOOP_ALREADY_DONE; exit 0; fi
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $WORK/swm-followup/.wandb_env
export WANDB_DATA_DIR=$WORK/wandb_data
[ -f $W/datasets_swm/bg_v5_ogbplay_s70000.h5 ] || { echo INIT_MISSING_V5_H5; exit 1; }
[ -f $W/ckpts/ogbench_base_diffusion/blue_cube_green_cube.pt ] || { echo INIT_MISSING_DIFFUSION_CKPT; exit 1; }
# the warm-up pick must be wired in (uncommented) and present on disk
PICK=$(grep -E '^init_projector:' $W/swm-next/swm_lora/train_exp5a.yaml | awk '{print $2}')
[ -n "$PICK" ] || { echo INIT_PICK_NOT_LOCKED; exit 1; }
[ -f $W/$PICK/projector.pt ] || { echo INIT_MISSING_WARMUP_PROJECTOR; exit 1; }
$V/bin/python swm-next/swm_lora/train_exp1.py --config swm-next/swm_lora/train_exp5a.yaml --resume auto || { echo TRAIN_JOB_FAILED; exit 1; }
echo TRAIN_JOB_DONE
