#!/bin/bash
#SBATCH -J swm-wu-lt1
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o %x-%j.out
# LangTable trajectory-adapter warm-up (warmup_lt1.yaml, noscale recipe) on
# gh-dev. LangTable actions are 2-D, so the OGB projector cannot be reused.
# Same pattern as warmup_v3b.sbatch, on LangTable batch 0 (50 units) with the
# training run's fixed validation split.
# The 6,000 steps do not fit one 2 h job (the OGB warm-up reached step 5,700 in
# 2 h at 112 tokens per example; LangTable frames at 640x360 give 268, so
# expect three jobs), so submit this as a chain: job 1, then each next job with
# --dependency=afterany:<previous job>. Every job runs --resume auto and goes on
# from the newest ckpt_step* (projector only; optimizer moments start fresh);
# the DONE guard makes a job a no-op once an earlier one has finished. gh-dev
# runs one job per user at a time and takes at most three submitted.
# Runs from $WORK/swm-followup/swms-lttrain.
set -x
W=$WORK/swm-followup/swms-lttrain
V=$WORK/swm-followup/swms-lt/.venv
cd $W
# anchored: under set -x this script's own trace lines quote the pattern
if grep -q '^WARMUP_JOB_DONE' $W/swm-wu-lt1-*.out 2>/dev/null; then echo RESCUE_NOOP_ALREADY_DONE; exit 0; fi
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# no trace while the key file is read: the trace would copy the key into the job log
{ set +x; } 2>/dev/null
source $WORK/swm-followup/.wandb_env
set -x
export WANDB_DATA_DIR=$WORK/wandb_data
n=$(ls $W/datasets_swm/lt_v1/lt_v1_b0/units/b0_*_c[0-9].h5 2>/dev/null | wc -l)
[ "$n" -eq 50 ] || { echo INIT_MISSING_LT_UNITS_have_$n; exit 1; }
[ -f $W/swm-next/swm_lora/splits/lt_v1_b0_split.json ] || { echo INIT_MISSING_FIXED_VAL_SPLIT; exit 1; }
$V/bin/python swm-next/swm_lora/warmup_encoder.py --config swm-next/swm_lora/warmup_lt1.yaml --resume auto || { echo WARMUP_JOB_FAILED; exit 1; }
echo WARMUP_JOB_DONE
