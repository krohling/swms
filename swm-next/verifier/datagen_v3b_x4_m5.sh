#!/bin/bash
# v3b 4x expansion worker on m5 (Kevin 2026-11-11): 16,200 more episodes of
# the unchanged v3b recipe (datagen_v3b.yaml: k=8 eval ladder, one proposal
# picked uniformly, executed and judge-labeled), added to the original 5,400
# (seeds 60000-65399). Two workers (the judge saturates at 2 processes on
# MPS), three batches of 2,700 episodes each, so the 2x / 3x / 4x sets are
# complete files as each batch pair lands:
#   worker 0: seeds 100000, 105400, 110800     worker 1: 102700, 108100, 113500
# -> 100000-116199, clear of eval 6000-6099, every earlier dataset and the
# smoke seeds (90000-90299). Resumable: datagen_v3.py skips episodes already
# in the output file. A batch waits while datasets_swm/DGV3B_X4_PAUSE exists
# (frees the GPU for evals between batches without killing anything).
# usage: bash swm-next/verifier/datagen_v3b_x4_m5.sh <0|1>
set -u
cd /Users/kevin/projects/research/semantic-world-modeling/workspace/swms
I=$1
export PYTHONPATH=$PWD:$PWD/swm-next PYTHONUNBUFFERED=1
N=2700
for b in 0 1 2; do
  SS=$((100000 + N * (2 * b + I)))
  out=datasets_swm/bg_v3b_k8uni_s${SS}.h5
  log=dgv3b_m5_s${SS}.log
  while [ -f datasets_swm/DGV3B_X4_PAUSE ]; do sleep 60; done
  if grep -q "^DATAGEN_DONE" "$log" 2>/dev/null; then
    echo "SKIP batch $b seeds $SS (done)"
    continue
  fi
  echo "START batch $b seeds $SS-$((SS + N - 1)) $(date '+%m-%d %H:%M')"
  nice -n 10 .venv/bin/python swm-next/verifier/datagen_v3.py \
    --config swm-next/configs/datagen_v3b.yaml --episodes $N \
    --seed-start $SS --out "$out" >> "$log" 2>&1
  rc=$?
  echo "END batch $b seeds $SS rc=$rc $(date '+%m-%d %H:%M') $(grep -E '^DATAGEN_DONE|Error|Traceback' "$log" | tail -1)"
  [ $rc -eq 0 ] || { echo "WORKER_${I}_FAILED batch $b"; exit 1; }
done
echo "WORKER_${I}_DONE $(date '+%m-%d %H:%M')"
