#!/bin/bash
#SBATCH -J swm-wu4a
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o %x-%j.out
# exp4a trajectory-adapter warm-up (warmup_v3a.yaml, noscale recipe) on
# gh-dev. Runs inside the exp4 training checkout so outputs land exactly
# where train_exp4a.yaml's init_projector will point
# (outputs_swm_lora/exp4a_warmup_v3a/ckpt_stepN/projector.pt). LARG was the
# planned host but the A100s now show the A40s' CUDA vision-path fault
# (probed 2026-11-05). If the 2h cap cuts the 6000-step ceiling short,
# ckpt_every 250 means the plateau pick is still on disk — no resume needed.
set -x
W=$WORK/swm-followup/swms-exp4
V=$WORK/swm-followup/swms-lt/.venv
cd $W
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $WORK/swm-followup/.wandb_env
export WANDB_DATA_DIR=$WORK/wandb_data
[ -f $W/datasets_swm/bg_v3a_k1tau1_s42000.h5 ] || { echo INIT_MISSING_V3A_H5; exit 1; }
[ -f $W/ckpts/ogbench_base_diffusion/blue_cube_green_cube.pt ] || { echo INIT_MISSING_DIFFUSION_CKPT; exit 1; }
$V/bin/python swm-next/swm_lora/warmup_encoder.py --config swm-next/swm_lora/warmup_v3a.yaml || { echo WARMUP_JOB_FAILED; exit 1; }
echo WARMUP_JOB_DONE
