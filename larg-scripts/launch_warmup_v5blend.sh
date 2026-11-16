#!/bin/bash
# usage: launch_warmup_v5blend.sh <gpu_index>
# exp5b trajectory-adapter warm-up on the paper-blend set, on one LARG GPU
# (Kevin 2026-11-16: run the exp5b warm-up on a LARG A100 rather than wait for
# the gh-dev queue). Same recipe as vista-sbatch/warmup_v5blend_balanced.sbatch:
# the config is derived here from warmup_v5blend_balanced.yaml with only the
# wandb run name and the out dir changed (suffix -larg / _larg), so this run
# and the Vista leg never share a wandb run or a checkpoint dir. There is no
# wall clock on LARG — the run ends at the config's max_steps (WARMUP_DONE).
# Pick rule unchanged: argmin of val_bce_balanced over saved checkpoints.
# Launch guards per lab discipline: idle re-check, host-load guard, >=1 GPU
# left free on the machine, offline HF cache, nice, detached tmux session.
# <gpu_index> is nvidia-smi's index; CUDA is pinned by that GPU's UUID, because
# CUDA skips a GPU the driver cannot use and its own numbering then shifts
# against nvidia-smi's (pulisic GPU 0, 2026-11-16).
# Runs from the swms-exp5 harness checkout on /scratch (branch exp5-ogb-play).
set -e
GPU=${1:?gpu index}
W=/scratch/cluster/krohling/swms-exp5
SRC=swm-next/swm_lora/warmup_v5blend_balanced.yaml
RUN=warmup-v5blend-noscale-balanced-larg
OUT=outputs_swm_lora/exp5b_warmup_v5blend_balanced_larg
HOST=$(hostname -s)
cd $W
[ -f $SRC ] || { echo "checkout missing $SRC — checkout branch exp5-ogb-play first"; exit 1; }
for f in datasets_swm/bg_v5_ogbplay_s70000_n2100.h5 datasets_swm/bg_v5_ogbnoisy_s80000.h5 \
         ckpts/ogbench_base_diffusion/blue_cube_green_cube.pt; do
  [ -f $f ] || { echo "missing $f"; exit 1; }
done
tmux has-session -t swm-wu5c 2>/dev/null && { echo "tmux session swm-wu5c already exists on $HOST"; exit 1; }
# warmup_encoder.py does not resume: a second launch would restart at step 0
# over the first run's checkpoints and wandb history
ls -d $OUT/ckpt_step* >/dev/null 2>&1 && { echo "$OUT already holds checkpoints — refusing"; exit 1; }
mkdir -p $OUT
sed -e "s|^run_name:.*|run_name: $RUN|" -e "s|^out_dir:.*|out_dir: $OUT|" $SRC > $OUT/warmup_larg.yaml
grep -q "^run_name: $RUN\$" $OUT/warmup_larg.yaml && grep -q "^out_dir: $OUT\$" $OUT/warmup_larg.yaml \
  || { echo "could not derive the LARG config from $SRC"; exit 1; }
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU)
case "$UUID" in GPU-*) ;; *) echo "no UUID for GPU $GPU on $HOST: $UUID"; exit 1 ;; esac
cat > $OUT/run_warmup_larg.sh <<INNER
#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$UUID
export HF_HUB_CACHE=/scratch/cluster/krohling/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONPATH=$W:$W/swm-next
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# torch's compiled-kernel cache defaults to ~/.cache/torch/kernels, which holds
# zero-byte entries on LARG (found 2026-11-16); loading one fails every integer
# .prod() on the GPU with "CUDA driver error: invalid argument" (the Qwen image
# path calls it). Keep the cache on scratch, next to the venv it belongs to.
export PYTORCH_KERNEL_CACHE_PATH=/scratch/cluster/krohling/swm-followup/torch_kernel_cache
# wandb local files go to node-local /tmp, not the scratch quota
export WANDB_DIR=/tmp/krohling-wandb-wu5c
export WANDB_DATA_DIR=/tmp/krohling-wandb-wu5c
mkdir -p /tmp/krohling-wandb-wu5c
set -a; source /scratch/cluster/krohling/swm-followup/.env; set +a
cd $W
echo "WARMUP_START $HOST g$GPU \$(date '+%m-%d %H:%M') $UUID"
nice -n 10 /scratch/cluster/krohling/swm-followup/.venv/bin/python \\
  swm-next/swm_lora/warmup_encoder.py --config $OUT/warmup_larg.yaml \\
  || { echo "WARMUP_JOB_FAILED \$(date '+%m-%d %H:%M')"; exit 1; }
echo "WARMUP_JOB_DONE \$(date '+%m-%d %H:%M')"
INNER
chmod +x $OUT/run_warmup_larg.sh
# free = <1GB used and <5% util, as everywhere on LARG
gpu_free() {
  local mu ut
  mu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1)
  ut=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $1)
  [ "$mu" -lt 1024 ] 2>/dev/null && [ "$ut" -lt 5 ] 2>/dev/null
}
gpu_free $GPU || { echo "GPU $GPU busy on $HOST — refusing"; exit 1; }
# nproc alone would return OMP_NUM_THREADS when that is exported
CORES=$(nproc --all); LOAD=$(cut -d" " -f1 /proc/loadavg)
awk -v c="$CORES" -v l="$LOAD" 'BEGIN{exit !(l < c/2)}' || { echo "host load $LOAD too high"; exit 1; }
# Kevin 2026-11-02: a placement must leave >=1 free GPU on the machine
FREE_OTHERS=0
for g in $(seq 0 $(($(nvidia-smi -L | wc -l) - 1))); do
  [ "$g" -eq "$GPU" ] && continue
  if gpu_free $g; then FREE_OTHERS=$((FREE_OTHERS + 1)); fi
done
[ "$FREE_OTHERS" -ge 1 ] || { echo "GPU $GPU would leave 0 free GPUs on $HOST — refusing (keep >=1 open)"; exit 1; }
tmux new-session -d -s swm-wu5c "$W/$OUT/run_warmup_larg.sh 2>&1 | tee -a $W/$OUT/warmup_larg.log"
echo "launched exp5b warm-up in tmux:swm-wu5c on $HOST GPU $GPU ($UUID), log $OUT/warmup_larg.log"
