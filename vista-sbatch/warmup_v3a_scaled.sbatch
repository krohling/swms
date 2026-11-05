#!/bin/bash
#SBATCH -J swm-wu4as
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o %x-%j.out
# exp4a trajectory-adapter warm-up, SCALED (exp1 treatment) — see
# warmup_v3a.sbatch for the pattern notes. Runs inside the exp4 checkout so
# outputs land where train_exp4a.yaml's init_projector will point
# (outputs_swm_lora/exp4a_warmup_v3a_scaled/ckpt_stepN).
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
$V/bin/python swm-next/swm_lora/warmup_encoder.py --config swm-next/swm_lora/warmup_v3a_scaled.yaml || { echo WARMUP_JOB_FAILED; exit 1; }
echo WARMUP_JOB_DONE
