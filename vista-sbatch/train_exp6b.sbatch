#!/bin/bash
#SBATCH -J swm-exp6b
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -o %x-%j.out
# Experiment 6b: exp6a's recipe on the original v3b file only (train_exp6b.yaml),
# the batch-size control for exp6a. As long as exp6a (about 53 h), so submit
# twice: leg 1, then leg 2 with --dependency=afterany:<leg1> (--resume auto;
# the DONE guard makes leg 2 a no-op if leg 1 finishes). Never hold either leg.
# wandb is offline for both legs (the stored key is rejected since Nov 19) and
# the key file is not read: each leg writes wandb/offline-run-*-exp6b-v3b1x-b96,
# to be uploaded in start order with `wandb sync` once a key works. The trainer
# logs no checkpoint artifacts in that mode.
# Runs from $WORK/swm-followup/swms-exp6.
set -x
W=$WORK/swm-followup/swms-exp6
V=$WORK/swm-followup/swms-lt/.venv
cd $W
# anchored: under set -x this script's own trace lines quote the pattern
if grep -q '^TRAIN_JOB_DONE' $W/swm-exp6b-*.out 2>/dev/null; then echo RESCUE_NOOP_ALREADY_DONE; exit 0; fi
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline WANDB_DATA_DIR=$WORK/wandb_data
[ -f $W/datasets_swm/bg_v3b_k8uni_s60000.h5 ] || { echo INIT_MISSING_V3B_H5; exit 1; }
[ -f $W/ckpts/ogbench_base_diffusion/blue_cube_green_cube.pt ] || { echo INIT_MISSING_DIFFUSION_CKPT; exit 1; }
[ -f $W/outputs_swm_lora/exp4b_warmup_v3b/ckpt_step5250/projector.pt ] || { echo INIT_MISSING_PROJECTOR; exit 1; }
[ -f $W/swm-next/swm_lora/splits/exp4b_split.json ] || { echo INIT_MISSING_FIXED_VAL_SPLIT; exit 1; }
$V/bin/python swm-next/swm_lora/train_exp1.py --config swm-next/swm_lora/train_exp6b.yaml --resume auto || { echo TRAIN_JOB_FAILED; exit 1; }
echo TRAIN_JOB_DONE
