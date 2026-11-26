#!/bin/bash
# exp4 SR-ladder worker on m5: closed-loop evals (eval_closed_loop.py, k=8,
# n=50, seeds 6000-6049) over the 16k/42k/64k checkpoints of exp4a/b/c/d on
# bg/by/yr. Worker I of N takes every N-th job of the fixed list below, in
# order (arm a -> b -> c -> d, step 16k -> 42k -> 64k, task bg -> by -> yr).
# Resumable: a job whose result JSON exists is skipped.
# usage: bash sr_ladder_exp4_worker.sh <I> <N>
set -u
cd /Users/kevin/projects/research/semantic-world-modeling/workspace/swms
I=$1; N=$2
export PYTHONPATH=$PWD:$PWD/swm-next PYTHONUNBUFFERED=1
# (bash 3.2 on macOS: no associative arrays)
cfg_for() {
  case "$1" in
    bg) echo swm-next/configs/verifier_mpc.yaml ;;
    by) echo swm-next/configs/verifier_mpc_blue_cube_yellow_cube.yaml ;;
    yr) echo swm-next/configs/verifier_mpc_yellow_cube_red_cube.yaml ;;
    *) echo "unknown task $1" >&2; exit 2 ;;
  esac
}
jobs=()
for arm in a b c d; do
  for step in 16000 42000 64000; do
    for task in bg by yr; do jobs+=("$arm $step $task"); done
  done
done
k=0
for j in "${jobs[@]}"; do
  if [ $((k % N)) -eq "$I" ]; then
    read -r arm step task <<< "$j"
    out=experiments/exp4${arm}/eval_cl_step${step}_${task}_k8.json
    log=outputs_swm_lora/eval_logs/exp4${arm}_step${step}_${task}.log
    mkdir -p experiments/exp4${arm} outputs_swm_lora/eval_logs
    if [ -f "$out" ]; then
      echo "SKIP exp4$arm $step $task (result exists)"
    elif [ ! -f outputs_swm_lora/exp4${arm}_full/ckpt_step${step}/adapter_model.safetensors ]; then
      echo "MISSING_CKPT exp4$arm $step"
    else
      echo "START exp4$arm $step $task $(date '+%m-%d %H:%M')"
      .venv/bin/python swm-next/swm_lora/eval_closed_loop.py \
        --ckpt outputs_swm_lora/exp4${arm}_full/ckpt_step${step} \
        --config "$(cfg_for "$task")" \
        --train-config swm-next/swm_lora/train_exp4${arm}.yaml \
        --episodes 50 --k 8 --out "$out" > "$log" 2>&1
      echo "END exp4$arm $step $task $(date '+%m-%d %H:%M') $(grep -E 'EVAL_DONE|Error|Traceback' "$log" | tail -1)"
    fi
  fi
  k=$((k + 1))
done
echo "WORKER_${I}_OF_${N}_DONE $(date '+%m-%d %H:%M')"
