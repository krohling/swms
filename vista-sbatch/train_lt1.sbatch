#!/bin/bash
#SBATCH -J swm-lt1
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -o %x-%j.out
# LangTable run lt1: exp6a's recipe (96 examples/step x 80,367 steps) on the
# 13,500-episode LangTable set (train_lt1.yaml): batch 0 (50 units) plus
# batches 1-4 (200 units), five tasks in equal shares, one model.
# Longer than one 48 h leg, so submit the legs as a chain: leg 1, then each
# next leg with --dependency=afterany:<previous leg> (--resume auto; the DONE
# guard makes a leg a no-op once an earlier one has finished). Never hold a leg.
# wandb is offline for every leg (as for exp6c: no checkpoint artifacts) and
# the key file is not read: each leg writes wandb/offline-run-*-<run_name>,
# to be uploaded in start order with `wandb sync` after each leg.
# Runs from $WORK/swm-followup/swms-lttrain.
set -x
W=$WORK/swm-followup/swms-lttrain
V=$WORK/swm-followup/swms-lt/.venv
cd $W
# anchored: under set -x this script's own trace lines quote the pattern
if grep -q '^TRAIN_JOB_DONE' $W/swm-lt1-*.out 2>/dev/null; then echo RESCUE_NOOP_ALREADY_DONE; exit 0; fi
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline WANDB_DATA_DIR=$WORK/wandb_data
n0=$(ls $W/datasets_swm/lt_v1/lt_v1_b0/units/b0_*_c[0-9].h5 2>/dev/null | wc -l)
[ "$n0" -eq 50 ] || { echo INIT_MISSING_LT_B0_UNITS_have_$n0; exit 1; }
n1=$(ls $W/datasets_swm/lt_v1/lt_v1_main/units/b[1-4]_*_c[0-9].h5 2>/dev/null | wc -l)
[ "$n1" -eq 200 ] || { echo INIT_MISSING_LT_MAIN_UNITS_have_$n1; exit 1; }
P=$(grep -E '^init_projector:' swm-next/swm_lora/train_lt1.yaml | awk '{print $2}')
[ -n "$P" ] && [ -f $W/$P/projector.pt ] || { echo INIT_MISSING_PROJECTOR; exit 1; }
[ -f $W/swm-next/swm_lora/splits/lt_v1_b0_split.json ] || { echo INIT_MISSING_FIXED_VAL_SPLIT; exit 1; }
$V/bin/python swm-next/swm_lora/train_exp1.py --config swm-next/swm_lora/train_lt1.yaml --resume auto || { echo TRAIN_JOB_FAILED; exit 1; }
echo TRAIN_JOB_DONE
