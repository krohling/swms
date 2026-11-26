#!/bin/bash
#SBATCH -J swm-wu5b
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o %x-%j.out
# exp5a trajectory-adapter warm-up, BALANCED variant (warmup_v5_balanced.yaml;
# Kevin 2026-11-08, after the unbalanced wu5 plateaued near the prior-only
# BCE). One 2h gh-dev leg; ckpt_every 250 keeps the pick (argmin of
# val_bce_balanced over saved ckpts) on disk — no resume needed. Extend with
# one more 2h leg via init_projector only if the smoothed balanced tail is
# still falling at the wall. train_exp5a.yaml's init_projector then points at
# the pick and exp5a is submitted at pick-lock (never submit-and-hold: a Slurm
# hold resets accrued age priority on release, verified 2026-11-07).
set -x
W=$WORK/swm-followup/swms-exp5
V=$WORK/swm-followup/swms-lt/.venv
cd $W
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $WORK/swm-followup/.wandb_env
export WANDB_DATA_DIR=$WORK/wandb_data
[ -f $W/datasets_swm/bg_v5_ogbplay_s70000.h5 ] || { echo INIT_MISSING_V5_H5; exit 1; }
[ -f $W/ckpts/ogbench_base_diffusion/blue_cube_green_cube.pt ] || { echo INIT_MISSING_DIFFUSION_CKPT; exit 1; }
$V/bin/python swm-next/swm_lora/warmup_encoder.py --config swm-next/swm_lora/warmup_v5_balanced.yaml || { echo WARMUP_JOB_FAILED; exit 1; }
echo WARMUP_JOB_DONE
