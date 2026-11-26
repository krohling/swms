#!/bin/bash
# LangTable k=1 dataset generation on ONE LARG card (Kevin 2026-11-25): one
# micro-batching Qwen judge on the GPU (judge_server_mb.py, bound to
# 127.0.0.1), shared by N generator workers (datagen_lt.py). The workers are
# simulator-bound CPU work; the judge is the GPU work.
#
# Usage: datagen_lt_larg.sh <set> plan smoke          25 units of 4 episodes
#        datagen_lt_larg.sh <set> plan batch <b>      50 units of 54 episodes
#        datagen_lt_larg.sh <set> start <gpu index> <N> <cpu|cuda>   (N may be 0)
#        datagen_lt_larg.sh <set> add <N>             more workers, same judge
#        datagen_lt_larg.sh <set> status | stop | stopworkers
#
# A unit = one task, one seed range, one HDF5 file. Seeds: batch b, task index
# t (order of TASKS), episode i of 540 -> 200000 + b*2700 + t*540 + i; smoke
# (20 episodes per task, in small units so that a production number of
# workers runs at once) 190000 + t*100 + i. A unit is written as
# <name>.part.h5, verified
# (verify_dataset_lt.py), then renamed to <name>.h5. The m5 side copies
# finished units away and leaves <name>.relayed, which counts as done here.
# A killed unit resumes from its .part.h5 (finished episodes are kept).
set -u
SET=${1:?set}; CMD=${2:?command}
SELF=$(readlink -f "$0")
ROOT=/scratch/cluster/krohling
C=${LTGEN_CHECKOUT:-$ROOT/swms-ltgen}
OUT=$C/experiments/ltgen_larg/$SET
PY=$ROOT/swm-followup/.venv/bin/python
LTPOL=$ROOT/swm-followup/swms/ckpts/lt_base_diffusion
LTSIM=$ROOT/swm-followup/language-table
PORT=8902
MIN_FREE_GB=6                 # scratch quota guard: no new unit under this
CHECK_CYCLES=${LTGEN_CHECK_CYCLES:-2}   # restore check: first N cycles of every episode
TASKS=(green_cube_blue_moon red_moon_green_star red_pentagon_blue_moon
       yellow_pentagon_red_moon yellow_star_blue_cube)
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_CACHE=$ROOT/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export PYTORCH_KERNEL_CACHE_PATH=$ROOT/swm-followup/torch_kernel_cache
export MPLCONFIGDIR=$ROOT/swm-followup/mpl_cache
export PYTHONPATH=$LTSIM:$C:$C/swm-next:$C/swm-next/verifier
mkdir -p $OUT/units $OUT/logs $OUT/pids
alive() { [ -f $1 ] && kill -0 "$(cat $1)" 2>/dev/null; }
n_alive() { local n=0 p; for p in $OUT/pids/worker*.pid; do alive $p && n=$((n+1)); done; echo $n; }
load_ok() {
  # nproc alone honors OMP_NUM_THREADS (set above) and would report 2
  local l1; l1=$(cut -d. -f1 /proc/loadavg)
  echo "load1 $l1 of $(nproc --all) cores"
  [ "$l1" -lt $(( $(nproc --all) / 2 )) ]
}
start_workers() {
  local n=$1 i last
  last=$(ls $OUT/pids 2>/dev/null | sed -n 's/^worker\([0-9]*\)\.pid$/\1/p' | sort -n | tail -n 1)
  for i in $(seq $(( ${last:-0} + 1 )) $(( ${last:-0} + n ))); do
    nohup bash $SELF $SET worker $i >> $OUT/logs/worker$i.log 2>&1 < /dev/null &
    echo $! > $OUT/pids/worker$i.pid
    sleep 2
  done
}
case $CMD in
plan)
  touch $OUT/units.txt
  add() { grep -q "^$1 " $OUT/units.txt || echo "$1 $2 $3 $4" >> $OUT/units.txt; }
  if [ "${3:-}" = smoke ]; then
    for c in 0 1 2 3 4; do for t in 0 1 2 3 4; do
      add smoke_${TASKS[$t]}_c$c ${TASKS[$t]} $((190000 + t*100 + c*4)) 4
    done; done
  elif [ "${3:-}" = batch ]; then
    b=${4:?batch number}
    for c in 0 1 2 3 4 5 6 7 8 9; do for t in 0 1 2 3 4; do
      add b${b}_${TASKS[$t]}_c$c ${TASKS[$t]} $((200000 + b*2700 + t*540 + c*54)) 54
    done; done
  else echo "plan smoke | plan batch <b>"; exit 1; fi
  echo "units planned: $(wc -l < $OUT/units.txt)";;
start)
  G=${3:?gpu index}; N=${4:?workers}; POLDEV=${5:?cpu|cuda}
  [ -s $OUT/units.txt ] || { echo "no units planned"; exit 1; }
  [ "$(n_alive)" = 0 ] && ! alive $OUT/pids/judge.pid || { echo "already running"; exit 1; }
  # acquisition: the GPU free on a fresh survey, load under half the cores; ONE launch
  mu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G)
  ut=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $G)
  UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $G)
  echo "survey $(hostname -s) $(date '+%m-%d %H:%M CT') gpu $G $UUID used ${mu}MiB util ${ut}%"
  [ "$mu" -lt 1024 ] && [ "$ut" -lt 5 ] || { echo "ABORT gpu $G is in use"; exit 1; }
  load_ok || { echo "ABORT load too high"; exit 1; }
  ss -ltn | grep -q ":$PORT " && { echo "ABORT port $PORT taken"; exit 1; }
  echo $UUID > $OUT/GPU_UUID; hostname -s > $OUT/HOST; echo $POLDEV > $OUT/POLICY_DEVICE
  rm -rf $OUT/claims            # nothing of ours is alive: any claim left is stale
  rm -f $OUT/pids/*.pid
  { cd $C && git rev-parse HEAD && git status --porcelain -- swm swm-next | head -n 5
    git -C $LTSIM rev-parse HEAD; } > $OUT/CODE_STATE.txt
  cd $C
  # a log per judge start: "serving on" of an earlier start must not count
  [ -f $OUT/logs/judge_server.log ] && mv $OUT/logs/judge_server.log $OUT/logs/judge_server.$(date +%m%d_%H%M%S).log
  CUDA_VISIBLE_DEVICES=$UUID nohup nice -n 10 $PY swm-next/verifier/judge_server_mb.py \
    --model Qwen/Qwen3-VL-8B-Instruct --device cuda --host 127.0.0.1 --port $PORT \
    >> $OUT/logs/judge_server.log 2>&1 < /dev/null &
  echo $! > $OUT/pids/judge.pid
  for i in $(seq 1 150); do
    grep -q "serving on" $OUT/logs/judge_server.log && break
    alive $OUT/pids/judge.pid || break
    sleep 2
  done
  grep -q "serving on" $OUT/logs/judge_server.log || { echo "judge did not come up"; tail -n 3 $OUT/logs/judge_server.log; alive $OUT/pids/judge.pid && kill "$(cat $OUT/pids/judge.pid)"; exit 1; }
  start_workers $N
  echo "started judge + $N workers (policy on $POLDEV)";;
add)
  N=${3:?workers}
  alive $OUT/pids/judge.pid || { echo "judge is not running: use start"; exit 1; }
  load_ok || { echo "ABORT load too high"; exit 1; }
  start_workers $N
  echo "added $N workers: $(n_alive) alive";;
worker)
  W=${3:?id}
  UUID=$(cat $OUT/GPU_UUID); POLDEV=$(cat $OUT/POLICY_DEVICE)
  [ $POLDEV = cuda ] && CVD=$UUID || CVD=""
  CODE=$(tr '\n' ' ' < $OUT/CODE_STATE.txt)
  cd $C || exit 1
  mkdir -p $OUT/claims
  while [ ! -f $OUT/STOP ]; do
    NAME=""
    while read -r n t s e; do
      [ -f $OUT/units/$n.h5 ] && continue
      [ -f $OUT/units/$n.relayed ] && continue
      mkdir $OUT/claims/$n 2>/dev/null || continue
      NAME=$n; T=$t; S=$s; E=$e; break
    done < $OUT/units.txt
    [ -n "$NAME" ] || break
    while :; do
      free=$(df -BG --output=avail $ROOT | tail -n 1 | tr -dc 0-9)
      [ "${free:-0}" -ge $MIN_FREE_GB ] && break
      date "+DISK_WAIT %m-%d %H:%M:%S CT w$W $NAME free ${free}G"; sleep 300
    done
    P=$OUT/units/$NAME.part.h5
    date "+START %m-%d %H:%M:%S CT w$W $NAME"
    CUDA_VISIBLE_DEVICES=$CVD nice -n 10 $PY swm-next/verifier/datagen_lt.py \
      --config swm-next/configs/verifier_mpc_lt_$T.yaml --set device=$POLDEV \
      --set judge_url=http://127.0.0.1:$PORT --set diffusion_path=$LTPOL/$T.pt \
      --episodes $E --seed-start $S --restore-check-cycles $CHECK_CYCLES --code-state "$CODE" \
      --out $P >> $OUT/logs/$NAME.log 2>&1 < /dev/null
    rc=$?
    if [ $rc = 0 ]; then
      CUDA_VISIBLE_DEVICES="" nice -n 10 $PY swm-next/verifier/verify_dataset_lt.py $P \
        --task $T --seed-start $S --episodes $E --replay 2 \
        > $OUT/units/$NAME.verify.txt 2>> $OUT/logs/$NAME.log < /dev/null
      rc=$?
      [ $rc = 0 ] && mv $P $OUT/units/$NAME.h5 || rc="verify$rc"
    fi
    echo "RC=$rc w$W $NAME $(date '+%m-%d %H:%M:%S')"
    # a failed unit keeps its claim, so no worker picks it up again by itself
  done
  echo "WORKER_DONE w$W";;
status)
  alive $OUT/pids/judge.pid && echo "judge alive" || echo "judge DEAD"
  echo "workers alive: $(n_alive)"
  tot=$(wc -l < $OUT/units.txt 2>/dev/null); tot=${tot:-0}
  dn=$(ls $OUT/units 2>/dev/null | grep -c -E '\.(relayed|h5)$' ); pt=$(ls $OUT/units 2>/dev/null | grep -c '\.part\.h5$')
  echo "units done: $((dn - pt)) of $tot (in progress: $pt, relayed: $(ls $OUT/units | grep -c '\.relayed$'))"
  cat $OUT/logs/smoke_*.log $OUT/logs/b[0-9]*.log 2>/dev/null | grep -a -E '^\[[0-9]+/[0-9]+\] seed' \
    | awk '{n++; if ($0 ~ /: success/) w++; split($0, a, " @ "); split(a[2], b, " "); c += b[1]}
           END {printf "episodes done: %d, success %.1f%%, cycles %d (%.1f per episode)\n", n, n ? 100*w/n : 0, c, n ? c/n : 0}'
  grep -a -h "^RC=" $OUT/logs/worker*.log 2>/dev/null | grep -v "^RC=0" | tail -n 3
  grep -a -l -E "Traceback|RESTORE_CHECK_FAILED" $OUT/logs/*.log 2>/dev/null | head -n 3 | sed 's/^/error in: /'
  grep -a "items in" $OUT/logs/judge_server.log 2>/dev/null | tail -n 1
  echo "workers finished: $(cat $OUT/logs/worker*.log 2>/dev/null | grep -a -c WORKER_DONE)"
  df -h $ROOT | tail -n 1 | awk '{print "scratch free: " $4}';;
stopworkers)
  touch $OUT/STOP
  for p in $OUT/pids/worker*.pid; do alive $p && { pkill -P "$(cat $p)"; kill "$(cat $p)"; }; done
  sleep 2
  # no worker is left, so every claim is stale: `add` can pick the units up again
  [ "$(n_alive)" = 0 ] && rm -rf $OUT/claims
  rm -f $OUT/STOP; echo "workers stopped ($(n_alive) alive)";;
stop)
  for p in $OUT/pids/worker*.pid; do alive $p && { pkill -P "$(cat $p)"; kill "$(cat $p)"; }; done
  alive $OUT/pids/judge.pid && kill "$(cat $OUT/pids/judge.pid)"; echo stopped;;
*) echo "unknown command $CMD"; exit 1;;
esac
