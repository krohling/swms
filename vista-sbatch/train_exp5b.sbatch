#!/bin/bash
#SBATCH -J swm-exp5b
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -o %x-%j.out
# Experiment 5b: the paper-blend set (2,100 play + 2,100 noisy episodes) at 96
# examples/step for 80,367 steps (train_exp5b.yaml); ~53 h at exp6a's 2.36 s/step,
# so submit twice: leg 1, then leg 2 with --dependency=afterany:<leg1>
# (--resume auto; the DONE guard no-ops a redundant leg 2). Runs from
# $WORK/swm-followup/swms-exp5 (branch exp5-ogb-play).
# Submit ONLY at pick-lock (warm-up checkpoint chosen and wired into
# train_exp5b.yaml) — never submit-and-hold.
set -x
W=$WORK/swm-followup/swms-exp5
V=$WORK/swm-followup/swms-lt/.venv
cd $W
# anchored: under set -x this script's own trace lines quote the pattern
if grep -q '^TRAIN_JOB_DONE' $W/swm-exp5b-*.out 2>/dev/null; then echo RESCUE_NOOP_ALREADY_DONE; exit 0; fi
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $WORK/swm-followup/.wandb_env
export WANDB_DATA_DIR=$WORK/wandb_data
[ -f $W/datasets_swm/bg_v5_ogbplay_s70000_n2100.h5 ] || { echo INIT_MISSING_V5_PLAY_H5; exit 1; }
[ -f $W/datasets_swm/bg_v5_ogbnoisy_s80000.h5 ] || { echo INIT_MISSING_V5_NOISY_H5; exit 1; }
[ -f $W/ckpts/ogbench_base_diffusion/blue_cube_green_cube.pt ] || { echo INIT_MISSING_DIFFUSION_CKPT; exit 1; }
# the warm-up pick must be wired in (uncommented) and present on disk
PICK=$(grep -E '^init_projector:' $W/swm-next/swm_lora/train_exp5b.yaml | awk '{print $2}')
[ -n "$PICK" ] || { echo INIT_PICK_NOT_LOCKED; exit 1; }
[ -f $W/$PICK/projector.pt ] || { echo INIT_MISSING_WARMUP_PROJECTOR; exit 1; }
$V/bin/python swm-next/swm_lora/train_exp1.py --config swm-next/swm_lora/train_exp5b.yaml --resume auto || { echo TRAIN_JOB_FAILED; exit 1; }
echo TRAIN_JOB_DONE
