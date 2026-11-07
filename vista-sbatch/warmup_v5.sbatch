#!/bin/bash
#SBATCH -J swm-wu5
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o %x-%j.out
# exp5a trajectory-adapter warm-up (warmup_v5.yaml: noscale recipe +
# label-balanced sampling) on gh-dev, from the exp5 checkout
# $WORK/swm-followup/swms-exp5 (branch exp5-ogb-play), so outputs land where
# train_exp5a.yaml's init_projector will point
# (outputs_swm_lora/exp5a_warmup_v5/ckpt_stepN/projector.pt). ckpt_every 250:
# if the 2h cap cuts the 6000-step ceiling short, the pick is still on disk.
# Extension rule (Kevin): if val_bce_balanced is still falling at the wall,
# one more 2h leg warm-loads the last ckpt via init_projector.
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
$V/bin/python swm-next/swm_lora/warmup_encoder.py --config swm-next/swm_lora/warmup_v5.yaml || { echo WARMUP_JOB_FAILED; exit 1; }
echo WARMUP_JOB_DONE
