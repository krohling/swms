#!/bin/bash
#SBATCH -J swm-exp2c-rescue
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -o %x-%j.out
# Rescue v2 for exp2c job 1103346 (wedged at step 21.5k since 2026-10-28 23:05 CT,
# node c610-091 Lustre client hung mid-checkpoint-write; .tmp safetensors frozen
# at 174MB). Replaces rescue v1 (1109485, cancelled while still pending): v1
# gated aliveness on the ORIGINAL LOG's mtime, but the wedged job's wandb
# sidecar appends telemetry-error lines (observed 11:14 and 20:31 CT Oct 29)
# and Slurm epilogues also touch the log - both advance mtime without trainer
# progress, so v1 could spuriously no-op; and a live leg-2 writes its OWN log,
# so v1 could also scancel a healthy running leg-2. v2 gates on the one signal
# every legitimate writer (original, leg-2, rescue) refreshes: resume_step*.pt
# mtime in the shared out_dir, written every 500 steps (~15-20 min).
# Pre-flight (one-writer guarantee, same scancel list as v1):
#   ^TRAIN_JOB_DONE in any exp2c log            -> no-op
#   newest resume_step*.pt fresher than 40 min  -> a trainer is active; no-op
#   otherwise                                   -> scancel 1103346 1103347 and
#                                                  resume from newest resume_step*.pt
set -x
W=$WORK/swm-followup/swms-exp2
V=$WORK/swm-followup/swms-lt/.venv
O=$W/outputs_swm_lora/exp2c_full
cd $W
# anchored: under set -x this script's own trace lines quote the pattern
if grep -q '^TRAIN_JOB_DONE' $W/swm-exp2c-*.out 2>/dev/null; then echo RESCUE_NOOP_ALREADY_DONE; exit 0; fi
newest=$(ls -t $O/resume_step*.pt 2>/dev/null | head -1)
if [ -n "$newest" ]; then
  m=$(stat -c %Y "$newest"); now=$(date +%s)
  if [ $((now - m)) -lt 2400 ]; then echo "RESCUE_NOOP_TRAINER_ACTIVE ($(basename $newest) written $(( (now - m) / 60 )) min ago)"; exit 0; fi
fi
echo "RESCUE_TAKEOVER: no live writer (newest resume: $(basename ${newest:-none})); cancelling 1103346 1103347"
scancel 1103346 1103347 2>/dev/null
sleep 90  # let the old allocation release; its trainer is in uninterruptible I/O and dies with it
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $WORK/swm-followup/.wandb_env
export WANDB_DATA_DIR=$WORK/wandb_data
[ -f $W/outputs_swm_lora/warmup_sel/ckpt_step6000/projector.pt ] || { echo INIT_MISSING_WARMUP_SEL; exit 1; }
[ -f $W/outputs_swm_lora/warmup_sel/ckpt_step6000/selector_extra.pt ] || { echo INIT_MISSING_WARMUP_SEL_EXTRA; exit 1; }
[ -n "$(ls $O/resume_step*.pt 2>/dev/null)" ] || { echo RESUME_FILE_MISSING; exit 1; }
$V/bin/python swm-next/swm_lora/train_exp2.py --config swm-next/swm_lora/train_exp2c.yaml --resume auto || { echo TRAIN_JOB_FAILED; exit 1; }
echo TRAIN_JOB_DONE
