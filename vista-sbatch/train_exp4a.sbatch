#!/bin/bash
#SBATCH -J swm-exp4a
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -o %x-%j.out
# Experiment 4a: exp1 recipe on the v3a k=1 tau=1 dataset (train_exp4a.yaml).
# ~91k steps expected (2 planner questions/cycle; see train_exp4a.yaml) but
# k=1 single-candidate steps are ~8x lighter than exp1's 8-lists, so one 48h
# leg should still cover it; submit a leg-2 with --dependency=afterany:<leg1>
# as crash/walltime insurance (same script; --resume auto resumes, and the
# DONE guard no-ops a redundant leg-2).
# Runs from the pointwise-k1 harness checkout $WORK/swm-followup/swms-exp4.
# Staging pre-reqs (verified by the guards below): the v3a HDF5 in
# datasets_swm/ and the diffusion ckpt copied from swms-exp1. The trajectory
# adapter initializes FRESH (Kevin 2026-11-04) — no warm-up ckpt needed.
set -x
W=$WORK/swm-followup/swms-exp4
V=$WORK/swm-followup/swms-lt/.venv
cd $W
# anchored: under set -x this script's own trace lines quote the pattern
if grep -q '^TRAIN_JOB_DONE' $W/swm-exp4a-*.out 2>/dev/null; then echo RESCUE_NOOP_ALREADY_DONE; exit 0; fi
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $WORK/swm-followup/.wandb_env
export WANDB_DATA_DIR=$WORK/wandb_data
[ -f $W/datasets_swm/bg_v3a_k1tau1_s42000.h5 ] || { echo INIT_MISSING_V3A_H5; exit 1; }
[ -f $W/ckpts/ogbench_base_diffusion/blue_cube_green_cube.pt ] || { echo INIT_MISSING_DIFFUSION_CKPT; exit 1; }
$V/bin/python swm-next/swm_lora/train_exp1.py --config swm-next/swm_lora/train_exp4a.yaml --resume auto || { echo TRAIN_JOB_FAILED; exit 1; }
echo TRAIN_JOB_DONE
