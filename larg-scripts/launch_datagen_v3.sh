#!/bin/bash
# usage: launch_datagen_v3.sh <a|b> <gpu_index> <seed_start> <episodes>
# v3 pointwise-k1 collection launcher (swm-next/verifier/datagen_v3.py),
# cloned from the proven dgv2 launcher (larg_launch_datagen.sh): launch-time
# idle re-check, PCI-ordered GPU pinning, host-load guard, scratch-free guard,
# offline HF cache, EGL rendering, nice+ionice, detached tmux session.
# Arms: a = k=1 tau=1.0 (datagen_v3a.yaml, production seeds 42000+)
#       b = k=8 ladder uniform-pick (datagen_v3b.yaml, production seeds 60000+)
# Smoke/benchmark runs use seeds 90000+. Runs from the swms-exp1 harness
# checkout on /scratch — switch it to branch pointwise-k1 before launching.
set -e
ARM=${1:?arm a|b}; GPU=${2:?gpu index}; SS=${3:?seed_start}; EPS=${4:?episodes}
case $ARM in a|b) ;; *) echo "arm must be a or b"; exit 1 ;; esac
export CUDA_DEVICE_ORDER=PCI_BUS_ID
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $GPU)
[ "$USED" -lt 1024 ] && [ "$UTIL" -lt 5 ] || { echo "GPU $GPU busy (${USED}MiB, ${UTIL}%)"; exit 1; }
CORES=$(nproc); LOAD=$(cut -d" " -f1 /proc/loadavg)
awk -v c="$CORES" -v l="$LOAD" 'BEGIN{exit !(l < c/2)}' || { echo "host load $LOAD too high"; exit 1; }
FREE_GB=$(df -BG --output=avail /scratch/cluster/krohling | tail -1 | tr -dc '0-9')
FLOOR_GB=${FREE_FLOOR_GB:-10}
[ "$FREE_GB" -ge "$FLOOR_GB" ] || { echo "scratch only ${FREE_GB}G free (floor ${FLOOR_GB}G) — refusing"; exit 1; }
# Kevin 2026-11-02: placing a job must leave >=1 free GPU on the machine
# for other users (free = <1GB used and <5% util).
NGPU=$(nvidia-smi -L | wc -l)
FREE_OTHERS=0
for g in $(seq 0 $((NGPU-1))); do
  [ "$g" -eq "$GPU" ] && continue
  mu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  ut=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g)
  [ "$mu" -lt 1024 ] && [ "$ut" -lt 5 ] && FREE_OTHERS=$((FREE_OTHERS+1))
done
[ "$FREE_OTHERS" -ge 1 ] || { echo "GPU $GPU would leave 0 free GPUs on this host — refusing (keep >=1 open)"; exit 1; }
W=/scratch/cluster/krohling/swms-exp1
cd $W
[ -f swm-next/verifier/datagen_v3.py ] || { echo "checkout missing datagen_v3.py — checkout branch pointwise-k1 first"; exit 1; }
case $ARM in
  a) OUT=bg_v3a_k1tau1_s${SS}.h5 ;;
  b) OUT=bg_v3b_k8uni_s${SS}.h5 ;;
esac
cat > $W/run_dgv3${ARM}_${SS}.sh <<INNER
#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU
export MUJOCO_EGL_DEVICE_ID=$GPU
export HF_HUB_CACHE=/scratch/cluster/krohling/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONPATH=$W:$W/swm-next
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
cd $W
sed "s/^device:.*/device: cuda/" swm-next/configs/datagen_v3${ARM}.yaml > /tmp/dgv3${ARM}_larg_${SS}.yaml
exec nice -n 10 ionice -c 3 /scratch/cluster/krohling/swm-followup/.venv/bin/python \
  swm-next/verifier/datagen_v3.py --config /tmp/dgv3${ARM}_larg_${SS}.yaml \
  --episodes $EPS --seed-start $SS \
  --out datasets_swm/$OUT
INNER
chmod +x $W/run_dgv3${ARM}_${SS}.sh
tmux new-session -d -s dgv3${ARM}-$SS "$W/run_dgv3${ARM}_${SS}.sh 2>&1 | tee $W/dgv3${ARM}_larg_s${SS}.log"
echo "launched dgv3$ARM s${SS} in tmux:dgv3${ARM}-$SS on GPU $GPU (PCI order) -> datasets_swm/$OUT"
