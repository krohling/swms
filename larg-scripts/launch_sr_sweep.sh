#!/bin/bash
# usage: launch_sr_sweep.sh <gpu_index> <queue file> [experiment prefix, default exp4]
# Closed-loop SR sweep worker for one experiment's checkpoints on one LARG GPU
# (Kevin 2026-11-16: 8k-step checkpoint grid for exp4a/exp4b, then the same
# grid for exp6a). Same eval as the m5 runner sr_ladder_exp4_dense_m5.sh
# (eval_closed_loop.py, k=8, n=50, seeds 6000-6049, bg/by/yr), with the launch
# guards of launch_datagen_v3.sh: idle re-check, GPU pinning, host-load guard,
# >=1 GPU left free, offline HF cache, EGL rendering, nice, detached tmux
# session.
# The queue file holds "<arm> <step> <task>" lines in priority order and is
# shared by every worker on every host (scratch is one filer): a worker takes
# the first job that has a staged checkpoint, no result and no claim, and
# claims it with an atomic mkdir. The guards run again before every job, and
# a worker stops rather than waits or retries when its GPU has been taken,
# the host is loaded, its next job would leave no free GPU, or a job leaves
# no result (that job's claim stays, so no other worker repeats it).
# `touch <log dir>/STOP` ends every worker launched before the touch, after its
# current job (a STOP file older than a worker is ignored, so a forgotten one
# cannot block a later launch).
# <gpu_index> is nvidia-smi's index. CUDA is pinned by that GPU's UUID, not by
# the index: CUDA and EGL both skip a GPU the driver cannot use (pulisic GPU 0
# on 2026-11-16, utilization [N/A]), which shifts their numbering against
# nvidia-smi's — index 1 there landed on nvidia-smi GPU 2.
# Results go to experiments/<prefix><arm>_larg/ so a LARG score never
# overwrites the m5-scored file of the same name. Runs from the harness
# checkout this script sits in (swms-exp1 on branch pointwise-k1 for exp4,
# swms-exp6 on branch exp6-v3b-x4 for exp6); queue, claims and logs live in
# that checkout. The tmux session is named per GPU, not per experiment, so a
# GPU never carries two sweep workers.
set -e
GPU=${1:?gpu index}; Q=${2:?queue file}; EXP=${3:-exp4}
W=$(cd "$(dirname "$0")/.." && pwd)
LOGDIR=$W/outputs_swm_lora/eval_logs/sweep8k_larg
HOST=$(hostname -s)
cd $W
[ -f "$Q" ] || { echo "queue file $Q not found"; exit 1; }
while read -r arm _; do
  case "${arm:-}" in ""|\#*) continue ;; esac
  [ -f swm-next/swm_lora/train_${EXP}${arm}.yaml ] || { echo "$W has no train_${EXP}${arm}.yaml — wrong checkout or prefix"; exit 1; }
done < "$Q"
for t in blue_cube_green_cube blue_cube_yellow_cube yellow_cube_red_cube; do
  [ -f ckpts/ogbench_base_diffusion/$t.pt ] || { echo "missing diffusion checkpoint $t.pt"; exit 1; }
done
tmux has-session -t sr8k-g$GPU 2>/dev/null && { echo "tmux session sr8k-g$GPU already exists on $HOST"; exit 1; }
mkdir -p $LOGDIR/claims $LOGDIR/cfg
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU)
case "$UUID" in GPU-*) ;; *) echo "no UUID for GPU $GPU on $HOST: $UUID"; exit 1 ;; esac
# EGL device index = position among the GPUs the driver can use
EGL_ID=0
for g in $(seq 0 $((GPU - 1))); do
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g)
  if [ "$u" -ge 0 ] 2>/dev/null; then EGL_ID=$((EGL_ID + 1)); fi
done
WORKER=$LOGDIR/worker_${HOST}_g${GPU}.sh
cat > $WORKER <<INNER
#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$UUID
export MUJOCO_EGL_DEVICE_ID=$EGL_ID
export HF_HUB_CACHE=/scratch/cluster/krohling/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONPATH=$W:$W/swm-next
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export OMP_NUM_THREADS=8
# torch's compiled-kernel cache defaults to ~/.cache/torch/kernels, which holds
# zero-byte entries on LARG (found 2026-11-16); loading one fails every integer
# .prod() on the GPU with "CUDA driver error: invalid argument" (the Qwen image
# path calls it). Keep the cache on scratch, next to the venv it belongs to.
export PYTORCH_KERNEL_CACHE_PATH=/scratch/cluster/krohling/swm-followup/torch_kernel_cache
cd $W
# free = <1GB used and <5% util, as everywhere on LARG
gpu_free() {
  local mu ut
  mu=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i \$1)
  ut=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i \$1)
  [ "\$mu" -lt 1024 ] 2>/dev/null && [ "\$ut" -lt 5 ] 2>/dev/null
}
guard() {
  gpu_free $GPU || { echo "GPU $GPU is in use"; return 1; }
  # nproc alone would return OMP_NUM_THREADS
  awk -v c="\$(nproc --all)" -v l="\$(cut -d" " -f1 /proc/loadavg)" 'BEGIN{exit !(l < c/2)}' \\
    || { echo "host load \$(cut -d" " -f1 /proc/loadavg) too high"; return 1; }
  # Kevin 2026-11-02: a placement must leave >=1 free GPU on the machine
  local g others=0
  for g in \$(seq 0 \$((\$(nvidia-smi -L | wc -l) - 1))); do
    [ "\$g" -eq $GPU ] && continue
    gpu_free \$g && others=\$((others + 1))
  done
  [ "\$others" -ge 1 ] || { echo "GPU $GPU would leave 0 free GPUs on this host"; return 1; }
}
cfg_for() {
  case "\$1" in
    bg) echo swm-next/configs/verifier_mpc.yaml ;;
    by) echo swm-next/configs/verifier_mpc_blue_cube_yellow_cube.yaml ;;
    yr) echo swm-next/configs/verifier_mpc_yellow_cube_red_cube.yaml ;;
  esac
}
echo "WORKER_START $HOST g$GPU \$(date '+%m-%d %H:%M') $UUID egl=$EGL_ID"
first=1
while :; do
  [ $LOGDIR/STOP -nt $WORKER ] && { echo "WORKER_STOPPED $HOST g$GPU \$(date '+%m-%d %H:%M')"; exit 0; }
  # let the previous job's memory and utilization readings clear
  [ \$first -eq 1 ] || sleep 20
  first=0
  why=\$(guard) || { echo "WORKER_YIELD $HOST g$GPU \$(date '+%m-%d %H:%M') \$why"; exit 3; }
  pick=""; waiting=0
  while read -r arm step task; do
    case "\${arm:-}" in ""|\\#*) continue ;; esac
    [ -n "\$(cfg_for "\$task")" ] || { echo "WORKER_FAILED $HOST g$GPU unknown task \$task"; exit 2; }
    [ -f experiments/${EXP}\${arm}_larg/eval_cl_step\${step}_\${task}_k8.json ] && continue
    [ -d $LOGDIR/claims/${EXP}\${arm}_step\${step}_\${task} ] && continue
    if [ ! -f outputs_swm_lora/${EXP}\${arm}_full/ckpt_step\${step}/adapter_model.safetensors ] ||
       [ ! -f outputs_swm_lora/${EXP}\${arm}_full/ckpt_step\${step}/projector.pt ]; then
      waiting=1; continue
    fi
    mkdir $LOGDIR/claims/${EXP}\${arm}_step\${step}_\${task} 2>/dev/null || continue
    pick="\$arm \$step \$task"; break
  done < "$Q"
  if [ -z "\$pick" ]; then
    if [ \$waiting -eq 1 ]; then sleep 300; continue; fi
    echo "WORKER_QUEUE_EMPTY $HOST g$GPU \$(date '+%m-%d %H:%M')"; exit 0
  fi
  read -r arm step task <<< "\$pick"
  claim=$LOGDIR/claims/${EXP}\${arm}_step\${step}_\${task}
  echo "$HOST g$GPU \$(date '+%m-%d %H:%M')" > \$claim/owner
  out=experiments/${EXP}\${arm}_larg/eval_cl_step\${step}_\${task}_k8.json
  log=$LOGDIR/${EXP}\${arm}_step\${step}_\${task}.log
  cfg=$LOGDIR/cfg/${HOST}_g${GPU}_\${task}.yaml
  sed "s/^device:.*/device: cuda/" "\$(cfg_for "\$task")" > \$cfg
  grep -q "^device: cuda" \$cfg || { echo "WORKER_FAILED $HOST g$GPU no device key in \$cfg"; exit 2; }
  mkdir -p experiments/${EXP}\${arm}_larg
  echo "START ${EXP}\$arm \$step \$task $HOST g$GPU \$(date '+%m-%d %H:%M')"
  timeout 5h nice -n 10 /scratch/cluster/krohling/swm-followup/.venv/bin/python \\
    swm-next/swm_lora/eval_closed_loop.py \\
    --ckpt outputs_swm_lora/${EXP}\${arm}_full/ckpt_step\${step} \\
    --config \$cfg --train-config swm-next/swm_lora/train_${EXP}\${arm}.yaml \\
    --episodes 50 --seed-start 6000 --k 8 --out "\$out" > "\$log" 2>&1
  echo "END ${EXP}\$arm \$step \$task $HOST g$GPU \$(date '+%m-%d %H:%M') \$(grep -E 'EVAL_DONE|Error|Traceback' "\$log" | tail -1)"
  # a job that leaves no result keeps its claim and stops this worker
  [ -f "\$out" ] || { echo "WORKER_FAILED $HOST g$GPU ${EXP}\$arm \$step \$task \$(date '+%m-%d %H:%M')"; exit 1; }
  rm -rf \$claim
done
INNER
chmod +x $WORKER
# same guard as the worker runs before each job, checked here before launching
export CUDA_DEVICE_ORDER=PCI_BUS_ID
gpu_free() {
  local mu ut
  mu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1)
  ut=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $1)
  [ "$mu" -lt 1024 ] 2>/dev/null && [ "$ut" -lt 5 ] 2>/dev/null
}
gpu_free $GPU || { echo "GPU $GPU busy on $HOST — refusing"; exit 1; }
CORES=$(nproc --all); LOAD=$(cut -d" " -f1 /proc/loadavg)
awk -v c="$CORES" -v l="$LOAD" 'BEGIN{exit !(l < c/2)}' || { echo "host load $LOAD too high"; exit 1; }
FREE_OTHERS=0
for g in $(seq 0 $(($(nvidia-smi -L | wc -l) - 1))); do
  [ "$g" -eq "$GPU" ] && continue
  if gpu_free $g; then FREE_OTHERS=$((FREE_OTHERS + 1)); fi
done
[ "$FREE_OTHERS" -ge 1 ] || { echo "GPU $GPU would leave 0 free GPUs on $HOST — refusing (keep >=1 open)"; exit 1; }
tmux new-session -d -s sr8k-g$GPU "$WORKER 2>&1 | tee -a $LOGDIR/worker_${HOST}_g${GPU}.log"
echo "launched $EXP SR sweep worker in tmux:sr8k-g$GPU on $HOST GPU $GPU ($UUID, EGL device $EGL_ID), queue $Q, checkout $W"
