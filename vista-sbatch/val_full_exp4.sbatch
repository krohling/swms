#!/bin/bash
#SBATCH -J swm-valfull
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o %x-%j.out
# Full held-out val pass over exp4 checkpoints (Kevin 2026-11-10): the
# trainer's val_bce reads only the first 200 val items, ~3 episodes at k=1.
# Usage: sbatch -J swm-vf4<arm> vista-sbatch/val_full_exp4.sbatch <arm> <step> [<step> ...]
# Code runs from the exp5-ogb-play checkout; data, split.json, configs and
# checkpoints are read from the swms-exp4 checkout the arm trained in (read
# only). Results go outside both checkouts. Resumable: finished steps skip.
set -x
ARM=$1; shift
W=$WORK/swm-followup/swms-exp5
E4=$WORK/swm-followup/swms-exp4
V=$WORK/swm-followup/swms-lt/.venv
O=$WORK/swm-followup/val_full/exp4$ARM
cd $W
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f $E4/outputs_swm_lora/exp4${ARM}_full/split.json ] || { echo INIT_MISSING_SPLIT; exit 1; }
for s in "$@"; do
  [ -f $E4/outputs_swm_lora/exp4${ARM}_full/ckpt_step$s/projector.pt ] || { echo INIT_MISSING_CKPT_$s; exit 1; }
done
$V/bin/python swm-next/swm_lora/eval_val_full.py --config $E4/swm-next/swm_lora/train_exp4${ARM}.yaml \
  --root $E4 --steps "$@" --out-dir $O || { echo VAL_FULL_JOB_FAILED; exit 1; }
echo VAL_FULL_JOB_DONE
