#!/bin/bash
#SBATCH -J swm-dgv3b
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -o %x-%j.out
# v3b 16x expansion, Vista share (Kevin 2026-11-24): the m5 launcher
# (swm-next/verifier/datagen_v3b_x16_m5.sh) works up from batch 0; these jobs
# take batches from the top of the same seed plan so 16x lands sooner. Batch j
# covers seeds 120000 + 2700*j; one job runs TWO batches side by side on one
# node (as m5 does), same recipe (datagen_v3.py + datagen_v3b.yaml, device
# switched to cuda), same file names, so the files drop into the 16x set.
# Usage: sbatch vista-sbatch/datagen_v3b_x16.sbatch <seed_start_a> <seed_start_b>
# Resumable: datagen_v3.py skips episodes already in the output file, so the
# same line can be submitted again after a time-out.
# FAILS LOUDLY (exit 1) unless both batches print DATAGEN_DONE.
# Runs from $WORK/swm-followup/swms-exp6.
set -u
W=$WORK/swm-followup/swms-exp6
V=$WORK/swm-followup/swms-lt/.venv
N=2700
cd $W
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f $W/ckpts/ogbench_base_diffusion/blue_cube_green_cube.pt ] || { echo INIT_MISSING_DIFFUSION_CKPT; exit 1; }
CFG=/tmp/dgv3b_cuda_${SLURM_JOB_ID}.yaml
sed "s/^device:.*/device: cuda/" swm-next/configs/datagen_v3b.yaml > $CFG
grep -q "^device: cuda" $CFG || { echo INIT_MISSING_CUDA_CONFIG; exit 1; }
pids=""
for SS in "$1" "$2"; do
  log=dgv3b_vista_s${SS}.log
  # the recipe must stay the one the earlier files were made with: log what
  # this batch runs from
  echo "START seeds $SS-$((SS + N - 1)) $(date '+%m-%d %H:%M') head=$(git rev-parse --short HEAD) gen=$(git hash-object swm-next/verifier/datagen_v3.py | cut -c1-10) cfg=$(git hash-object swm-next/configs/datagen_v3b.yaml | cut -c1-10)"
  $V/bin/python swm-next/verifier/datagen_v3.py --config $CFG --episodes $N \
      --seed-start $SS --out datasets_swm/bg_v3b_k8uni_s${SS}.h5 >> $log 2>&1 &
  pids="$pids $!"
done
wait $pids
rc=0
for SS in "$1" "$2"; do
  if grep -q "^DATAGEN_DONE" dgv3b_vista_s${SS}.log; then
    echo "DATAGEN_BATCH_DONE $SS $(date '+%m-%d %H:%M')"
  else
    echo "DATAGEN_BATCH_FAILED $SS $(date '+%m-%d %H:%M')"; rc=1
  fi
done
exit $rc
