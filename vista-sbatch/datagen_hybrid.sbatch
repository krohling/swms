#!/bin/bash
#SBATCH -J hybrid-datagen
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -o %x-%j.out
# Hybrid k=8 dataset generation, ONE shard per job (k=8 is ~8x the judge work
# of the k=1 run — a 4-worker single node would blow the 48h wall).
# Usage: sbatch datagen_hybrid.sbatch <seed_start> <episodes>
# FAILS LOUDLY (exit 1) unless the shard prints DATAGEN_DONE.
set -x
SS=$1; EPS=$2
W=$WORK/swm-followup/swms-datagen
V=$WORK/swm-followup/swms-lt/.venv
cd $W
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
sed "s/^device:.*/device: cuda/" swm-next/configs/datagen_hybrid_bg.yaml > /tmp/dg_hybrid_cuda_${SS}.yaml
$V/bin/python swm-next/verifier/datagen.py --config /tmp/dg_hybrid_cuda_${SS}.yaml \
    --episodes $EPS --seed-start $SS \
    --out datasets_swm/bg_hybrid_k8_s${SS}.h5 > hybridgen_s${SS}.log 2>&1
grep -q "DATAGEN_DONE" hybridgen_s${SS}.log || { echo "HYBRID_SHARD_FAILED $SS"; exit 1; }
echo "HYBRID_SHARD_DONE $SS"
