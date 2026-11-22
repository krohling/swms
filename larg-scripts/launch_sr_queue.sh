#!/bin/bash
# usage: launch_sr_queue.sh <gpu_index> <queue file> [check|keep]
# Closed-loop eval worker on one LARG GPU, driven by a job queue that can mix
# experiments, seed blocks and the base-policy reference (Kevin 2026-11-16:
# base diffusion policy at n=500 per task, then exp4a/exp4b from n=50 to n=200
# per task with the last checkpoints first; exp6a's n=50 grid points go in
# ahead as its checkpoints are staged). launch_sr_sweep.sh stays as the record
# of the n=50 exp4 sweep; this is the same worker with the job spelled out per
# queue line instead of fixed in the script, and the same launch guards: idle
# re-check, GPU pinning by UUID, host-load guard, >=1 GPU left free, offline
# HF cache, EGL rendering, nice, detached tmux session.
#
# Queue lines, in priority order ("#" lines and blank lines are skipped):
#   cl   <checkout> <run> <step> <task> <episodes> <seed_start>
#   base <checkout> -     -      <task> <episodes> <seed_start>
#   teacher <checkout> -  -      <task> <episodes> <seed_start>
#   random  <checkout> -  -      <task> <episodes> <seed_start>
#   swm  <checkout> <ckpt dir> - <task> <episodes> <seed_start>
# cl   = eval_closed_loop.py on <run>'s ckpt_step<step>, k=8, as the n=50 sweep.
#        Result: <checkout>/experiments/<run>_larg/eval_cl_step<step>_<task>_k8.json
#        for the first block (50 episodes from seed 6000); any other block adds
#        _s<seed_start>n<episodes> before .json, so a new seed block never
#        overwrites an earlier one.
# base = verifier/mpc.py at k=1, tau=1: the base diffusion policy alone, the
#        recipe of experiments/anchors_tau1/REPORT.md. No Qwen is loaded.
#        Result: <checkout>/experiments/base_larg/base_k1_tau1_<task>_s<seed_start>n<episodes>/
#                mpc_softgate_ema_templadder_k1_L16.json
# teacher = verifier/mpc.py with the zero-shot Qwen judge choosing among k=8
#        proposals (Kevin 2026-11-17: the original teacher configuration at
#        n=200 per task): each proposal is rolled out 16 steps in the simulator,
#        Qwen scores its end frame, the best one is executed. Scorer and
#        temperature ladder are those of the September teacher row
#        (outputs_verifier_mpc_softgate_ema_k8_n50, 64 / 68 / 62 on m5).
#        Result: <checkout>/experiments/teacher_larg/teacher_k8_<task>_s<seed_start>n<episodes>/
#                mpc_softgate_ema_templadder_k8_L16.json
# random = the teacher's matched control (Kevin 2026-11-17): the same mpc.py
#        run, the same k=8 ladder and 16-step rollouts, but the executed
#        proposal is drawn uniformly at random and no Qwen is loaded
#        (--scorer random, the code of the October outputs_random row,
#        48 / 56 / 38 on m5). Queue it on the teacher's seeds.
#        Result: <checkout>/experiments/random_larg/random_k8_<task>_s<seed_start>n<episodes>/
#                mpc_random_templadder_k8_L16.json
# swm  = the authors' own closed-loop eval of a PaliGemma SWM checkpoint (Kevin
#        2026-11-17: the released jacob3333/paligemma_wm_ogbench at n=200 per
#        task, the ceiling row; the paper measured it at n=50):
#        scripts/evaluate_swm_hydra.py with the shipped
#        configs/eval_gradient_full_ogbench.yaml, i.e. gradient planning on one
#        base-policy proposal. Only the task, the seeds, the checkpoint path and
#        the output places are overridden. <ckpt dir> is a directory under
#        <checkout>/ckpts that also holds the processor files (true of the
#        released checkpoint, not of our Trainer checkpoints). The script writes
#        results.txt once, after the last seed, and skips a seed whose
#        reward.pkl is already there, so a job cut short can be run again.
#        It runs in the eval venv with one difference: the authors' pinned
#        transformers 4.57.1 (and the huggingface_hub 0.36.2 it requires) from
#        swm-followup/pydeps_tf457 ahead of the venv's 5.12.1 on PYTHONPATH,
#        because the released model code fails under 5.12.1 (see TF457 below).
#        Simulator, torch and base policy are the ones every other kind uses.
#        Result: <checkout>/experiments/swm_larg/<ckpt dir>_<task>_s<seed_start>n<episodes>/results.txt
# swmlt = the same eval on LangTable (Kevin 2026-11-21: the released
#        jacob3333/paligemma_wm_lt at n=200 per task): the shipped
#        configs/eval_gradient_full.yaml, one of its five tasks per job. <task>
#        is gcbm, rmgs, rpbm, yprm or ysbc (see lt_combo_for). The base
#        policies are the authors' lt_base_diffusion files in the
#        swm-followup/swms checkout (1.8 GB, not copied per checkout), and the
#        simulator is the language-table checkout beside it, on PYTHONPATH.
#        Same venv and transformers 4.57.1 overlay as swm.
#        Result: <checkout>/experiments/swmlt_larg/<ckpt dir>_<task>_s<seed_start>n<episodes>/results.txt
# <checkout> is a harness checkout under /scratch/cluster/krohling (swms-exp1 =
# branch pointwise-k1 for exp4, swms-exp6 = exp6-v3b-x4 for exp6). Each job
# runs from its own checkout with that checkout on PYTHONPATH. <task> is bg, by
# or yr. Eval seeds 6050-6099 are the held block B: keep them out of the queue.
#
# The queue is shared by every worker on every host (scratch is one filer). A
# worker takes the first job that has no result and no claim, and claims it
# with an atomic mkdir; a cl job whose checkpoint is not staged yet is passed
# over, and a worker with only such jobs left waits for them. A first-block
# job that a launch_sr_sweep.sh worker has claimed counts as claimed here too.
# A claim outlives its job: another host can go on seeing "no result file" for
# up to a minute after the result is written (NFS attribute cache), and a claim
# removed in that window let salah repeat pulisic's base_yr_s6400n150 on
# 2026-11-17. To run a job again, delete its result and its claim.
# The host-load and >=1-free-GPU guards are acquisition rules (Kevin
# 2026-11-16): they are checked once, at launch, and a worker that has its GPU
# keeps it between jobs. Before each job a worker only re-checks that nobody
# else is on its GPU, and it stops rather than waits or retries when the GPU
# has been taken or a job leaves no result (that job's claim stays, so no other
# worker repeats it). `touch <log dir>/STOP` ends every worker launched before
# the touch, after its current job.
# The tmux session is sr8k-g<gpu>, the name launch_sr_sweep.sh uses, so a GPU
# never carries two of our eval workers.
# With "check" as the third argument nothing is launched: the worker script is
# generated beside the real one and lists every queue line with its state
# (done / claimed / unstaged / todo), failing on a malformed line.
# With "keep" the GPU is one we already hold: the call waits for the worker now
# in tmux session sr8k-g<gpu> to end (its queue ran out, or its log dir got a
# STOP), then starts this queue's worker on the same GPU without the two
# acquisition guards. It refuses when there is no such session, so it cannot
# take a GPU we do not have. It blocks while it waits: run it in a tmux session
# of its own whose name does not start with sr8k-g. The new worker script is
# written beside the running one and moved in after the wait, so the order
# "start the keep calls, then touch STOP" replaces every worker at the end of
# its current job.
set -e
GPU=${1:?gpu index}; Q=${2:?queue file}; MODE=${3:-launch}
W=$(cd "$(dirname "$0")/.." && pwd)
LOGDIR=$W/outputs_swm_lora/eval_logs/srq_larg
HOST=$(hostname -s)
[ -f "$Q" ] || { echo "queue file $Q not found"; exit 1; }
# jobs change directory, so the worker needs the queue by absolute path
Q=$(cd "$(dirname "$Q")" && pwd)/$(basename "$Q")
case "$MODE" in launch|check|keep) ;; *) echo "third argument is 'check', 'keep' or nothing"; exit 1 ;; esac
if [ $MODE = launch ]; then
  tmux has-session -t sr8k-g$GPU 2>/dev/null && { echo "tmux session sr8k-g$GPU already exists on $HOST"; exit 1; }
fi
if [ $MODE = keep ]; then
  tmux has-session -t sr8k-g$GPU 2>/dev/null || { echo "no worker of ours in tmux session sr8k-g$GPU on $HOST — launch without 'keep'"; exit 1; }
fi
mkdir -p $LOGDIR/claims $LOGDIR/cfg
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU)
case "$UUID" in GPU-*) ;; *) echo "no UUID for GPU $GPU on $HOST: $UUID"; exit 1 ;; esac
# CUDA and EGL both skip a GPU the driver cannot use (pulisic GPU 0 on
# 2026-11-16, utilization [N/A]), which shifts their numbering against
# nvidia-smi's: EGL device index = position among the GPUs the driver can use
EGL_ID=0
for g in $(seq 0 $((GPU - 1))); do
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g)
  if [ "$u" -ge 0 ] 2>/dev/null; then EGL_ID=$((EGL_ID + 1)); fi
done
WORKER=$LOGDIR/worker_${HOST}_g${GPU}.sh
[ $MODE = check ] && WORKER=$LOGDIR/worker_${HOST}_g${GPU}.check.sh
# with "keep" the worker being replaced may be running $WORKER itself
GEN=$WORKER
[ $MODE = keep ] && GEN=$WORKER.next
# the launch-time values go in as a header; the body below is written verbatim
{
  echo "#!/bin/bash"
  echo "GPU=$GPU; HOST=$HOST; UUID=$UUID; EGL_ID=$EGL_ID"
  echo "Q=$Q"
  echo "LOGDIR=$LOGDIR"
  echo "WORKER=$WORKER"
  cat <<'INNER'
ROOT=/scratch/cluster/krohling
PY=$ROOT/swm-followup/.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$UUID
export MUJOCO_EGL_DEVICE_ID=$EGL_ID
export HF_HUB_CACHE=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export OMP_NUM_THREADS=8
# torch's compiled-kernel cache defaults to ~/.cache/torch/kernels, which holds
# zero-byte entries on LARG (found 2026-11-16); loading one fails every integer
# .prod() on the GPU with "CUDA driver error: invalid argument" (the Qwen image
# path calls it). Keep the cache on scratch, next to the venv it belongs to.
export PYTORCH_KERNEL_CACHE_PATH=$ROOT/swm-followup/torch_kernel_cache
# The released PaliGemma SWM code is written for the authors' pinned
# transformers==4.57.1: under the venv's 5.12.1 its first forward pass dies in
# create_masks_for_generate (missing inputs_embeds; path check 2026-11-17).
# 4.57.1 and huggingface_hub 0.36.2 are installed alone in this directory
# (pip --target --no-deps from the venv's pip); only the swm kind puts it on
# PYTHONPATH, ahead of the venv, and no other kind ever sees it.
TF457=$ROOT/swm-followup/pydeps_tf457
# the teacher's k=8 ladder, as the September result files record it
TEACHER_TEMPS=0.0,0.2857,0.5714,0.8571,1.1429,1.4286,1.7143,2.0
# free = <1GB used and <5% util, as everywhere on LARG
gpu_free() {
  local mu ut
  mu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1)
  ut=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $1)
  [ "$mu" -lt 1024 ] 2>/dev/null && [ "$ut" -lt 5 ] 2>/dev/null
}
# host load and >=1 free GPU are checked at launch only: a worker keeps its GPU
guard() {
  gpu_free $GPU || { echo "GPU $GPU is in use"; return 1; }
}
cfg_for() {
  case "$1" in
    bg) echo swm-next/configs/verifier_mpc.yaml ;;
    by) echo swm-next/configs/verifier_mpc_blue_cube_yellow_cube.yaml ;;
    yr) echo swm-next/configs/verifier_mpc_yellow_cube_red_cube.yaml ;;
  esac
}
policy_for() {
  case "$1" in
    bg) echo blue_cube_green_cube ;;
    by) echo blue_cube_yellow_cube ;;
    yr) echo yellow_cube_red_cube ;;
  esac
}
# the same pair as the authors' config spells it (tasks[].block_combo)
combo_for() {
  case "$1" in
    bg) echo blue_cube,green_cube ;;
    by) echo blue_cube,yellow_cube ;;
    yr) echo yellow_cube,red_cube ;;
  esac
}
# LangTable: the five tasks of the authors' eval_gradient_full.yaml
lt_combo_for() {
  case "$1" in
    gcbm) echo green_cube,blue_moon ;;
    rmgs) echo red_moon,green_star ;;
    rpbm) echo red_pentagon,blue_moon ;;
    yprm) echo yellow_pentagon,red_moon ;;
    ysbc) echo yellow_star,blue_cube ;;
  esac
}
LTPOL=$ROOT/swm-followup/swms/ckpts/lt_base_diffusion
LTSIM=$ROOT/swm-followup/language-table
is_num() { case "$1" in ""|*[!0-9]*) return 1 ;; esac; }
# one queue line -> JW (its checkout), id (claim and log name), out and ckpt
# (both relative to JW), sweep_claim (launch_sr_sweep.sh's claim on the same
# job, empty when there cannot be one); non-zero on a line that cannot be run
job() {
  kind=$1; co=$2; run=$3; step=$4; task=$5; eps=$6; seed0=$7
  if [ "$kind" = swmlt ]; then
    [ -n "$co" ] && [ -n "$(lt_combo_for "$task")" ] && is_num "$eps" && is_num "$seed0" || return 1
    case "$run" in ""|-|*/*) return 1 ;; esac
    JW=$ROOT/$co
    ckpt=ckpts/$run
    [ -f "$JW/$ckpt/model.safetensors.index.json" ] && [ -f "$JW/$ckpt/preprocessor_config.json" ] \
      && [ -f "$JW/scripts/evaluate_swm_hydra.py" ] && [ -f "$JW/configs/eval_gradient_full.yaml" ] \
      && [ -f "$LTPOL/$(lt_combo_for "$task" | tr , _).pt" ] && [ -d "$LTSIM/language_table" ] \
      && [ -d "$TF457/transformers-4.57.1.dist-info" ] && [ -d "$TF457/huggingface_hub-0.36.2.dist-info" ] || return 1
    sweep_claim=""
    id=swmlt_${run}_${task}_s${seed0}n${eps}
    out=experiments/swmlt_larg/${run}_${task}_s${seed0}n${eps}/results.txt
    return 0
  fi
  [ -n "$co" ] && [ -n "$(cfg_for "$task")" ] && is_num "$eps" && is_num "$seed0" || return 1
  JW=$ROOT/$co
  [ -f "$JW/$(cfg_for "$task")" ] && [ -f "$JW/ckpts/ogbench_base_diffusion/$(policy_for "$task").pt" ] || return 1
  local sfx=_s${seed0}n${eps}
  sweep_claim=""
  case "$kind" in
    cl)
      is_num "$step" && [ -f "$JW/swm-next/swm_lora/train_${run}.yaml" ] || return 1
      ckpt=outputs_swm_lora/${run}_full/ckpt_step${step}
      id=${run}_step${step}_${task}${sfx}
      # the first block keeps the n=50 sweep's file name
      if [ $sfx = _s6000n50 ]; then
        sfx=""
        sweep_claim=$JW/outputs_swm_lora/eval_logs/sweep8k_larg/claims/${run}_step${step}_${task}
      fi
      out=experiments/${run}_larg/eval_cl_step${step}_${task}_k8${sfx}.json ;;
    base)
      ckpt=""
      id=base_${task}${sfx}
      out=experiments/base_larg/base_k1_tau1_${task}${sfx}/mpc_softgate_ema_templadder_k1_L16.json ;;
    teacher)
      ckpt=""
      id=teacher_${task}${sfx}
      out=experiments/teacher_larg/teacher_k8_${task}${sfx}/mpc_softgate_ema_templadder_k8_L16.json ;;
    random)
      ckpt=""
      id=random_${task}${sfx}
      out=experiments/random_larg/random_k8_${task}${sfx}/mpc_random_templadder_k8_L16.json ;;
    swm)
      case "$run" in ""|-|*/*) return 1 ;; esac
      ckpt=ckpts/$run
      [ -f "$JW/$ckpt/model.safetensors.index.json" ] && [ -f "$JW/$ckpt/preprocessor_config.json" ] \
        && [ -f "$JW/scripts/evaluate_swm_hydra.py" ] && [ -f "$JW/configs/eval_gradient_full_ogbench.yaml" ] \
        && [ -d "$TF457/transformers-4.57.1.dist-info" ] && [ -d "$TF457/huggingface_hub-0.36.2.dist-info" ] || return 1
      id=swm_${run}_${task}${sfx}
      out=experiments/swm_larg/${run}_${task}${sfx}/results.txt ;;
    *) return 1 ;;
  esac
}
staged() { [ -f "$JW/$ckpt/adapter_model.safetensors" ] && [ -f "$JW/$ckpt/projector.pt" ]; }
if [ "${1:-}" = --check ]; then
  bad=0
  while read -r kind co run step task eps seed0 _; do
    case "${kind:-}" in ""|\#*) continue ;; esac
    job "$kind" "$co" "$run" "$step" "$task" "$eps" "$seed0" \
      || { echo "bad queue line: $kind $co $run $step $task $eps $seed0"; bad=1; continue; }
    state=todo
    [ $kind = cl ] && ! staged && state=unstaged
    [ -d "$LOGDIR/claims/$id" ] && state=claimed
    [ -n "$sweep_claim" ] && [ -d "$sweep_claim" ] && state=claimed
    [ -f "$JW/$out" ] && state=done
    echo "$state $id $co/$out"
  done < "$Q"
  exit $bad
fi
echo "WORKER_START $HOST g$GPU $(date '+%m-%d %H:%M') $UUID egl=$EGL_ID"
first=1
while :; do
  [ $LOGDIR/STOP -nt $WORKER ] && { echo "WORKER_STOPPED $HOST g$GPU $(date '+%m-%d %H:%M')"; exit 0; }
  # let the previous job's memory and utilization readings clear
  [ $first -eq 1 ] || sleep 20
  first=0
  why=$(guard) || { echo "WORKER_YIELD $HOST g$GPU $(date '+%m-%d %H:%M') $why"; exit 3; }
  pick=0; waiting=0
  while read -r kind co run step task eps seed0 _; do
    case "${kind:-}" in ""|\#*) continue ;; esac
    job "$kind" "$co" "$run" "$step" "$task" "$eps" "$seed0" \
      || { echo "WORKER_FAILED $HOST g$GPU bad queue line: $kind $co $run $step $task $eps $seed0"; exit 2; }
    [ -f "$JW/$out" ] && continue
    [ -d "$LOGDIR/claims/$id" ] && continue
    [ -n "$sweep_claim" ] && [ -d "$sweep_claim" ] && continue
    if [ $kind = cl ] && ! staged; then waiting=1; continue; fi
    mkdir "$LOGDIR/claims/$id" 2>/dev/null || continue
    pick=1; break
  done < "$Q"
  if [ $pick -eq 0 ]; then
    if [ $waiting -eq 1 ]; then sleep 300; continue; fi
    echo "WORKER_QUEUE_EMPTY $HOST g$GPU $(date '+%m-%d %H:%M')"; exit 0
  fi
  # kind, run, task, eps, seed0, JW, id, out and ckpt now describe the claimed job
  claim=$LOGDIR/claims/$id
  echo "$HOST g$GPU $(date '+%m-%d %H:%M')" > $claim/owner
  log=$LOGDIR/$id.log
  if [ $kind != swmlt ]; then
    cfg=$LOGDIR/cfg/${HOST}_g${GPU}_${task}.yaml
    sed "s/^device:.*/device: cuda/" "$JW/$(cfg_for "$task")" > $cfg
    grep -q "^device: cuda" $cfg || { echo "WORKER_FAILED $HOST g$GPU no device key in $cfg"; exit 2; }
  fi
  cd $JW
  export PYTHONPATH=$JW:$JW/swm-next
  mkdir -p "$(dirname "$out")"
  echo "START $id $HOST g$GPU $(date '+%m-%d %H:%M')"
  if [ $kind = cl ]; then
    timeout 5h nice -n 10 $PY swm-next/swm_lora/eval_closed_loop.py \
      --ckpt $ckpt --config $cfg --train-config swm-next/swm_lora/train_${run}.yaml \
      --episodes $eps --seed-start $seed0 --k 8 --out "$out" > "$log" 2>&1
  elif [ $kind = teacher ]; then
    timeout 8h nice -n 10 $PY swm-next/verifier/mpc.py --config $cfg \
      --temps $TEACHER_TEMPS --scorer softgate_ema --lookahead 16 \
      --seeds $eps --seed-start $seed0 --out "$(dirname "$out")" > "$log" 2>&1
  elif [ $kind = random ]; then
    timeout 5h nice -n 10 $PY swm-next/verifier/mpc.py --config $cfg \
      --temps $TEACHER_TEMPS --scorer random --lookahead 16 \
      --seeds $eps --seed-start $seed0 --out "$(dirname "$out")" > "$log" 2>&1
  elif [ $kind = swm ]; then
    # hydra changes into its run dir; the shipped config resolves its paths
    # from the launch directory, and the overrides here are absolute
    PYTHONPATH=$TF457:$PYTHONPATH timeout 10h nice -n 10 $PY scripts/evaluate_swm_hydra.py --config-name eval_gradient_full_ogbench \
      seed_start=$seed0 num_seeds=$eps \
      paths.model_ckpt_path=$JW/$ckpt \
      paths.root_save_path=$JW/$(dirname "$out") \
      hydra.run.dir=$JW/$(dirname "$out")/hydra \
      "tasks=[{block_combo:[$(combo_for "$task")],diffusion_path:\"$JW/ckpts/ogbench_base_diffusion/$(policy_for "$task").pt\"}]" \
      > "$log" 2>&1
  elif [ $kind = swmlt ]; then
    # the script skips a seed whose reward.pkl is there, so a job the timeout
    # cuts short can be run again (delete its claim)
    PYTHONPATH=$TF457:$LTSIM:$PYTHONPATH timeout 30h nice -n 10 $PY scripts/evaluate_swm_hydra.py --config-name eval_gradient_full \
      seed_start=$seed0 num_seeds=$eps \
      paths.model_ckpt_path=$JW/$ckpt \
      paths.root_save_path=$JW/$(dirname "$out") \
      hydra.run.dir=$JW/$(dirname "$out")/hydra \
      "tasks=[{block_combo:[$(lt_combo_for "$task")],diffusion_path:\"$LTPOL/$(lt_combo_for "$task" | tr , _).pt\"}]" \
      > "$log" 2>&1
  else
    timeout 5h nice -n 10 $PY swm-next/verifier/mpc.py --config $cfg \
      --k 1 --temps 1.0 --scorer softgate_ema --lookahead 16 \
      --seeds $eps --seed-start $seed0 --out "$(dirname "$out")" > "$log" 2>&1
  fi
  echo "END $id $HOST g$GPU $(date '+%m-%d %H:%M') $(grep -E 'EVAL_DONE|MPC_DONE|^Success rate|Error|Traceback' "$log" | tail -1)"
  # a job that leaves no result keeps its claim and stops this worker
  [ -f "$out" ] || { echo "WORKER_FAILED $HOST g$GPU $id $(date '+%m-%d %H:%M')"; exit 1; }
  # the claim stays (see the header): mkdir on it fails on every host at once
  date '+%m-%d %H:%M' > $claim/done
done
INNER
} > $GEN
chmod +x $GEN
if [ $MODE = check ]; then
  bash -n $WORKER
  $WORKER --check
  exit $?
fi
$GEN --check > /dev/null || { $GEN --check | grep "^bad"; echo "fix $Q first"; exit 1; }
gpu_free() {
  local mu ut
  mu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1)
  ut=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $1)
  [ "$mu" -lt 1024 ] 2>/dev/null && [ "$ut" -lt 5 ] 2>/dev/null
}
if [ $MODE = keep ]; then
  echo "KEEP_WAIT $HOST g$GPU $(date '+%m-%d %H:%M') waiting for tmux:sr8k-g$GPU to end"
  while tmux has-session -t sr8k-g$GPU 2>/dev/null; do sleep 5; done
  # the last job's memory and utilization readings take a few seconds to clear
  for i in $(seq 1 12); do
    if gpu_free $GPU; then break; fi
    sleep 5
  done
  # a worker ends on a STOP newer than its script: the new script has to be
  # newer than the STOP that ended the old worker
  mv $GEN $WORKER
  touch $WORKER
fi
gpu_free $GPU || { echo "GPU $GPU busy on $HOST — refusing"; exit 1; }
if [ $MODE = launch ]; then
  # acquisition guards, checked here only: host load under half the cores
  # (nproc alone would return OMP_NUM_THREADS), >=1 other GPU left free
  # (Kevin 2026-11-02)
  CORES=$(nproc --all); LOAD=$(cut -d" " -f1 /proc/loadavg)
  awk -v c="$CORES" -v l="$LOAD" 'BEGIN{exit !(l < c/2)}' || { echo "host load $LOAD too high"; exit 1; }
  FREE_OTHERS=0
  for g in $(seq 0 $(($(nvidia-smi -L | wc -l) - 1))); do
    [ "$g" -eq "$GPU" ] && continue
    if gpu_free $g; then FREE_OTHERS=$((FREE_OTHERS + 1)); fi
  done
  [ "$FREE_OTHERS" -ge 1 ] || { echo "GPU $GPU would leave 0 free GPUs on $HOST — refusing (keep >=1 open)"; exit 1; }
fi
tmux new-session -d -s sr8k-g$GPU "$WORKER 2>&1 | tee -a $LOGDIR/worker_${HOST}_g${GPU}.log"
echo "launched eval-queue worker in tmux:sr8k-g$GPU on $HOST GPU $GPU ($UUID, EGL device $EGL_ID), queue $Q"
