#!/bin/bash
# Teacher labeling pass on Vista. One array task per file (HDF5 allows only
# one writer, so no intra-file sharding; the two files run in parallel):
#
#   smoke (gh-dev, 1 traj per file):
#     sbatch -p gh-dev -t 00:45:00 --array=0-1 label_teacher.sbatch --limit-trajs 1
#   production (both files, ~5-8h each):
#     sbatch --array=0-1 label_teacher.sbatch
#
# Qwen3-VL-8B must be pre-cached under $WORK/hf_cache (compute nodes are
# offline). Reruns resume: completed trajectories are skipped.
#
#SBATCH -J swm-teacher-label
#SBATCH -o /work/11599/krohling/vista/swm-repro/logs/%x-%A-%a.out
#SBATCH -e /work/11599/krohling/vista/swm-repro/logs/%x-%A-%a.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -A ASC25021
set -uo pipefail
S=$WORK/swm-followup/swms
source $S/.venv/bin/activate
export HF_HUB_CACHE=$WORK/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

D=$WORK/swm-repro/data
FILES=(noisy play)
F=${FILES[${SLURM_ARRAY_TASK_ID:-0}]}

cd $WORK/swm-repro/swm-next
python label_teacher.py --hdf5 $D/$F.hdf5 --out $D/$F.teacher.hdf5 \
    --shard 0/1 --batch 96 "$@"
