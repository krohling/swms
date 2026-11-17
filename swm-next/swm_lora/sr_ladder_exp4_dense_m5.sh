#!/bin/bash
# exp4 dense SR-ladder queue on m5 (Kevin 2026-11-12: as many exp4a/exp4b
# closed-loop datapoints as possible before the Nov 13 14:00 PT check-in).
# Same eval as sr_ladder_exp4_m5.sh (eval_closed_loop.py, k=8, n=50, seeds
# 6000-6049, bg/by/yr). One process: a second worker adds no throughput on MPS.
# The queue file holds "<arm> <step> <task>" lines in priority order and is
# re-read before every job, so it can be reordered or extended while running:
# each pass runs the first job whose result JSON is missing and whose
# checkpoint is on disk. Waits while the only jobs left lack a checkpoint.
# No job starts after the cutoff. Resumable: existing results are skipped.
# usage: bash swm-next/swm_lora/sr_ladder_exp4_dense_m5.sh <queue file> "<YYYY-mm-dd HH:MM cutoff>"
set -u
cd /Users/kevin/projects/research/semantic-world-modeling/workspace/swms
Q=$1
CUT=$(date -j -f "%Y-%m-%d %H:%M" "$2" +%s) || exit 2
export PYTHONPATH=$PWD:$PWD/swm-next PYTHONUNBUFFERED=1
cfg_for() {
  case "$1" in
    bg) echo swm-next/configs/verifier_mpc.yaml ;;
    by) echo swm-next/configs/verifier_mpc_blue_cube_yellow_cube.yaml ;;
    yr) echo swm-next/configs/verifier_mpc_yellow_cube_red_cube.yaml ;;
    *) echo "unknown task $1" >&2; exit 2 ;;
  esac
}
mkdir -p outputs_swm_lora/eval_logs
while :; do
  if [ "$(date +%s)" -ge "$CUT" ]; then
    echo "QUEUE_CUTOFF $(date '+%m-%d %H:%M')"; exit 0
  fi
  pick=""; waiting=0
  while read -r arm step task; do
    case "${arm:-}" in ""|\#*) continue ;; esac
    [ -f experiments/exp4${arm}/eval_cl_step${step}_${task}_k8.json ] && continue
    if [ ! -f outputs_swm_lora/exp4${arm}_full/ckpt_step${step}/adapter_model.safetensors ] ||
       [ ! -f outputs_swm_lora/exp4${arm}_full/ckpt_step${step}/projector.pt ]; then
      waiting=1; continue
    fi
    pick="$arm $step $task"; break
  done < "$Q"
  if [ -z "$pick" ]; then
    if [ $waiting -eq 1 ]; then sleep 300; continue; fi
    echo "QUEUE_DONE $(date '+%m-%d %H:%M')"; exit 0
  fi
  read -r arm step task <<< "$pick"
  out=experiments/exp4${arm}/eval_cl_step${step}_${task}_k8.json
  log=outputs_swm_lora/eval_logs/exp4${arm}_step${step}_${task}.log
  mkdir -p experiments/exp4${arm}
  echo "START exp4$arm $step $task $(date '+%m-%d %H:%M')"
  .venv/bin/python swm-next/swm_lora/eval_closed_loop.py \
    --ckpt outputs_swm_lora/exp4${arm}_full/ckpt_step${step} \
    --config "$(cfg_for "$task")" \
    --train-config swm-next/swm_lora/train_exp4${arm}.yaml \
    --episodes 50 --k 8 --out "$out" > "$log" 2>&1
  echo "END exp4$arm $step $task $(date '+%m-%d %H:%M') $(grep -E 'EVAL_DONE|Error|Traceback' "$log" | tail -1)"
  # a job that leaves no result would be picked again forever: stop instead
  [ -f "$out" ] || { echo "QUEUE_FAILED exp4$arm $step $task $(date '+%m-%d %H:%M')"; exit 1; }
done
