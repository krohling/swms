#!/bin/bash
#SBATCH -J swm-exp6c
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -o %x-%j.out
# Experiment 6c: exp6a's recipe on the 8x v3b set (train_exp6c.yaml), the
# interim run of the 16x expansion. As long as exp6a (about 53 h), so submit
# twice: leg 1, then leg 2 with --dependency=afterany:<leg1> (--resume auto;
# the DONE guard makes leg 2 a no-op if leg 1 finishes). Never hold either leg.
# wandb is offline for both legs (as for exp6b: no checkpoint artifacts) and
# the key file is not read: each leg writes wandb/offline-run-*-exp6c-v3b8x-b96,
# to be uploaded in start order with `wandb sync` after each leg. The trainer
# logs no checkpoint artifacts in that mode.
# Runs from $WORK/swm-followup/swms-exp6.
set -x
W=$WORK/swm-followup/swms-exp6
V=$WORK/swm-followup/swms-lt/.venv
cd $W
# anchored: under set -x this script's own trace lines quote the pattern
if grep -q '^TRAIN_JOB_DONE' $W/swm-exp6c-*.out 2>/dev/null; then echo RESCUE_NOOP_ALREADY_DONE; exit 0; fi
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline WANDB_DATA_DIR=$WORK/wandb_data
for s in 60000 100000 102700 105400 108100 110800 113500 120000 122700 125400 128100 130800 133500 136200 138900; do
  [ -f $W/datasets_swm/bg_v3b_k8uni_s$s.h5 ] || { echo INIT_MISSING_V3B_H5_s$s; exit 1; }
done
[ -f $W/ckpts/ogbench_base_diffusion/blue_cube_green_cube.pt ] || { echo INIT_MISSING_DIFFUSION_CKPT; exit 1; }
[ -f $W/outputs_swm_lora/exp4b_warmup_v3b/ckpt_step5250/projector.pt ] || { echo INIT_MISSING_PROJECTOR; exit 1; }
[ -f $W/swm-next/swm_lora/splits/exp4b_split.json ] || { echo INIT_MISSING_FIXED_VAL_SPLIT; exit 1; }
$V/bin/python swm-next/swm_lora/train_exp1.py --config swm-next/swm_lora/train_exp6c.yaml --resume auto || { echo TRAIN_JOB_FAILED; exit 1; }
echo TRAIN_JOB_DONE
