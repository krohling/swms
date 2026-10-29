#!/bin/bash
#SBATCH -J swm-exp2c-rescue
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -o %x-%j.out
# Rescue leg for exp2c job 1103346, wedged at step 21.5k since 2026-10-28
# 23:05 CT by a node-side Lustre client hang (c610-091) mid-checkpoint-write.
# Submitted INDEPENDENT (no dependency) so it accrues queue priority
# immediately — the afterany leg 1103347 accrues none while held — and with a
# SHORT walltime (8h >> ~4h of remaining work: ~8.6k steps at ~2.3k/h) to be
# backfill-eligible.
# Pre-flight guarantees exactly one writer for outputs_swm_lora/exp2c_full:
#   original finished              -> no-op exit
#   original log advanced <30 min  -> original resumed; no-op exit
#   original still frozen          -> scancel 1103346 1103347 and take over
#                                     from the latest resume_step*.pt
#                                     (resume_step21000.pt; ~500 steps redone)
set -x
W=$WORK/swm-followup/swms-exp2
V=$WORK/swm-followup/swms-lt/.venv
cd $W
OLDLOG=$W/swm-exp2c-1103346.out
if grep -q TRAIN_JOB_DONE $OLDLOG 2>/dev/null; then echo RESCUE_NOOP_ALREADY_DONE; exit 0; fi
now=$(date +%s); m=$(stat -c %Y $OLDLOG 2>/dev/null || echo 0)
if [ $((now - m)) -lt 1800 ]; then echo RESCUE_NOOP_ORIGINAL_ALIVE; exit 0; fi
echo "RESCUE_TAKEOVER: original frozen (log idle $(( (now - m) / 60 )) min); cancelling 1103346 1103347"
scancel 1103346 1103347 2>/dev/null
sleep 90  # let the old allocation release; its trainer is in uninterruptible I/O and dies with it
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $WORK/swm-followup/.wandb_env
export WANDB_DATA_DIR=$WORK/wandb_data
[ -f $W/outputs_swm_lora/warmup_sel/ckpt_step6000/projector.pt ] || { echo INIT_MISSING_WARMUP_SEL; exit 1; }
[ -f $W/outputs_swm_lora/warmup_sel/ckpt_step6000/selector_extra.pt ] || { echo INIT_MISSING_WARMUP_SEL_EXTRA; exit 1; }
[ -n "$(ls $W/outputs_swm_lora/exp2c_full/resume_step*.pt 2>/dev/null)" ] || { echo RESUME_FILE_MISSING; exit 1; }
$V/bin/python swm-next/swm_lora/train_exp2.py --config swm-next/swm_lora/train_exp2c.yaml --resume auto || { echo TRAIN_JOB_FAILED; exit 1; }
echo TRAIN_JOB_DONE
