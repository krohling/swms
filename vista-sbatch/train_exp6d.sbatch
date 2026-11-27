#!/bin/bash
#SBATCH -J swm-exp6d
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -o %x-%j.out
# Experiment 6d: exp6a's recipe on the full 16x v3b set at 160,734 steps
# (train_exp6d.yaml). About 105 h at exp6c's pace, so submit three times:
# leg 1, leg 2 with --dependency=afterany:<leg1>, leg 3 with
# --dependency=afterany:<leg2> and -t 24:00:00 (it needs about 10 h).
# --resume auto; the DONE guard makes a later leg a no-op if an earlier one
# finishes. Never hold a leg.
# wandb is offline for all legs (as for exp6c: no checkpoint artifacts) and
# the key file is not read: each leg writes wandb/offline-run-*-exp6d-v3b16x-b96,
# to be uploaded in start order with `wandb sync` after each leg.
# Runs from $WORK/swm-followup/swms-exp6.
set -x
W=$WORK/swm-followup/swms-exp6
V=$WORK/swm-followup/swms-lt/.venv
cd $W
# anchored: under set -x this script's own trace lines quote the pattern
if grep -q '^TRAIN_JOB_DONE' $W/swm-exp6d-*.out 2>/dev/null; then echo RESCUE_NOOP_ALREADY_DONE; exit 0; fi
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline WANDB_DATA_DIR=$WORK/wandb_data
for s in 60000 100000 102700 105400 108100 110800 113500 120000 122700 125400 128100 130800 133500 136200 138900 \
         141600 144300 147000 149700 152400 155100 157800 160500 \
         163200 165900 168600 171300 174000 176700 179400 182100; do
  [ -f $W/datasets_swm/bg_v3b_k8uni_s$s.h5 ] || { echo INIT_MISSING_V3B_H5_s$s; exit 1; }
done
[ -f $W/ckpts/ogbench_base_diffusion/blue_cube_green_cube.pt ] || { echo INIT_MISSING_DIFFUSION_CKPT; exit 1; }
[ -f $W/outputs_swm_lora/exp4b_warmup_v3b/ckpt_step5250/projector.pt ] || { echo INIT_MISSING_PROJECTOR; exit 1; }
[ -f $W/swm-next/swm_lora/splits/exp4b_split.json ] || { echo INIT_MISSING_FIXED_VAL_SPLIT; exit 1; }
$V/bin/python swm-next/swm_lora/train_exp1.py --config swm-next/swm_lora/train_exp6d.yaml --resume auto || { echo TRAIN_JOB_FAILED; exit 1; }
echo TRAIN_JOB_DONE
