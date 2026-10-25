#!/bin/bash
# Selector encoder warm-up on LARG (exp2 prerequisite).
# usage: launch_warmup_sel.sh <gpu_index>
# Pre-flight per lab discipline: the GPU must be empty and host load sane;
# wandb local files go to node-local /tmp (scratch quota is at 99%).
set -e
GPU=${1:?gpu index required}
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $GPU)
[ "$USED" -lt 1024 ] && [ "$UTIL" -lt 5 ] || { echo "GPU $GPU busy (${USED}MiB, ${UTIL}%)"; exit 1; }
LOAD=$(awk "{print int(\$1)}" /proc/loadavg)
CORES=$(nproc)
[ "$LOAD" -lt $((CORES / 2)) ] || { echo "host load $LOAD too high"; exit 1; }
W=/scratch/cluster/krohling/swms-exp1
cd $W
cat > $W/run_warmup_sel_inner.sh <<INNER
#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU
export HF_HUB_CACHE=/scratch/cluster/krohling/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONPATH=$W:$W/swm-next
export PYTHONUNBUFFERED=1
export WANDB_DIR=/tmp/krohling-wandb-sel
export WANDB_DATA_DIR=/tmp/krohling-wandb-sel
mkdir -p /tmp/krohling-wandb-sel
set -a; source /scratch/cluster/krohling/swm-followup/.env; set +a
cd $W
exec nice -n 10 /scratch/cluster/krohling/swm-followup/.venv/bin/python \
  swm-next/swm_lora/warmup_encoder.py --config swm-next/swm_lora/warmup_sel.yaml
INNER
chmod +x $W/run_warmup_sel_inner.sh
tmux new-session -d -s swm-warmup-sel "$W/run_warmup_sel_inner.sh 2>&1 | tee $W/warmup_sel_larg.log"
echo "launched tmux:swm-warmup-sel on GPU $GPU (PCI order)"
