#!/bin/bash
#SBATCH -J swm-smokelt1
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -o %x-%j.out
# Sizing smoke for the LangTable run lt1 (train_lt1_smoke.yaml): 60 steps of
# exp6a's geometry at each candidate input size, to measure pace and peak GPU
# memory before the frame size and the number of 48 h legs are fixed.
#   448    frames resized to 448x256 (112 image tokens)
#   640    frames as stored, 640x360 (the processor makes 640x352, 220 tokens)
#   640m8  only if 640 fails: 8 sequences per forward x 12 (same 96 per step)
# The 640 configs are made from the 448 one with sed. Each run gets the job id
# in its run name and output dir, so a second smoke never reads the first's files.
# Needs at least 5 batch 0 units in datasets_swm/lt_v1/units. wandb stays
# offline. The job name and end marker deliberately do NOT match
# train_lt1.sbatch's DONE guard (swm-lt1-*.out / TRAIN_JOB_DONE).
# Pace per size = (mtime ckpt_step40 - mtime ckpt_step20) / 20; memory in
# gpu_smokelt1_<size>_<job>.log.
set -x
W=$WORK/swm-followup/swms-lttrain
V=$WORK/swm-followup/swms-lt/.venv
J=${SLURM_JOB_ID:-manual}
cd $W
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline WANDB_DATA_DIR=$WORK/wandb_data
n=$(ls $W/datasets_swm/lt_v1/units/b0_*_c[0-9].h5 2>/dev/null | wc -l)
[ "$n" -ge 5 ] || { echo INIT_MISSING_LT_UNITS_have_$n; exit 1; }
S=swm-next/swm_lora/train_lt1_smoke.yaml
mkdir -p outputs_swm_lora
mk() {   # <size tag> [sed edits]: this size's config, from the 448 one
  local tag=$1; shift
  sed -e "s/lt1-smoke-448/lt1-smoke-$tag-$J/" -e "s/lt1_smoke_448/lt1_smoke_${tag}_$J/" "$@" $S \
    > outputs_swm_lora/train_lt1_smoke_${tag}_$J.yaml
}
mk 448
mk 640 -e '/^image_size:/d'
mk 640m8 -e '/^image_size:/d' -e 's/^micro_cycles: 16$/micro_cycles: 8/' -e 's/^accum_cycles: 6 /accum_cycles: 12 /'
run() {  # <size tag>
  local g=$W/gpu_smokelt1_$1_$J.log
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -l 15 > $g &
  local ns=$!
  date +%s
  $V/bin/python swm-next/swm_lora/train_exp1.py --config outputs_swm_lora/train_lt1_smoke_$1_$J.yaml --resume none
  local rc=$?
  date +%s
  kill $ns
  stat -c '%Y %n' outputs_swm_lora/lt1_smoke_$1_$J/ckpt_step* 2>/dev/null
  echo "SMOKE_RUN_$1 exit $rc peak $(sort -n $g | tail -n 1)"
  return $rc
}
run 448; r448=$?
run 640; r640=$?
[ $r640 -eq 0 ] || run 640m8
[ $r448 -eq 0 ] || { echo SMOKE_JOB_FAILED; exit 1; }
echo SMOKE_JOB_DONE
