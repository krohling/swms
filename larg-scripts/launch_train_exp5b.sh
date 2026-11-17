#!/bin/bash
# usage: launch_train_exp5b.sh <gpu_index>
# exp5b main-stage training on one LARG A100 (Kevin 2026-11-16: start it on
# debruyne rather than wait out the Vista gh queue; the Vista pair 1147760 +
# 1147761 stays queued as the fallback). Same recipe as
# vista-sbatch/train_exp5b.sbatch: the config is derived here from
# train_exp5b.yaml with only the wandb run name and the out dir changed (suffix
# -larg / _larg). The trainer uses the run name as the wandb run id, so the two
# copies must not share it. There is no wall clock on LARG — one launch runs to
# total_steps (TRAIN_JOB_DONE). After a dead run, launch again: the trainer
# resumes from the newest resume state (--resume auto, saved every 500 steps).
# Needs an 80 GB card (41 GB peak at 16 examples per micro-batch on Vista).
# Disk, all on scratch: a checkpoint every 500 steps (~183 MB each, ~30 GB over
# the run) plus wandb's artifact cache (~15 GB). Node-local / and /tmp are one
# shared 79 GB volume, so nothing of this run goes there.
# Launch guards per lab discipline: idle re-check, host-load guard, >=1 GPU
# left free on the machine, offline HF cache, nice, detached tmux session.
# <gpu_index> is nvidia-smi's index; CUDA is pinned by that GPU's UUID, because
# CUDA skips a GPU the driver cannot use and its own numbering then shifts
# against nvidia-smi's (pulisic GPU 0, 2026-11-16).
# Runs from the swms-exp5 harness checkout on /scratch (branch exp5-ogb-play).
set -e
GPU=${1:?gpu index}
W=/scratch/cluster/krohling/swms-exp5
SRC=swm-next/swm_lora/train_exp5b.yaml
RUN=exp5b-ogbblend-b96-larg
OUT=outputs_swm_lora/exp5b_full_larg
WB=/scratch/cluster/krohling/wandb-data/exp5b-larg
HOST=$(hostname -s)
cd $W
[ -f $SRC ] || { echo "checkout missing $SRC — checkout branch exp5-ogb-play first"; exit 1; }
for f in datasets_swm/bg_v5_ogbplay_s70000_n2100.h5 datasets_swm/bg_v5_ogbnoisy_s80000.h5 \
         ckpts/ogbench_base_diffusion/blue_cube_green_cube.pt; do
  [ -f $f ] || { echo "missing $f"; exit 1; }
done
# the warm-up pick must be wired in (uncommented) and present on disk
PICK=$(grep -E '^init_projector:' $SRC | awk '{print $2}')
[ -n "$PICK" ] || { echo "no init_projector in $SRC — pick not locked"; exit 1; }
[ -f $PICK/projector.pt ] || { echo "missing $PICK/projector.pt"; exit 1; }
tmux has-session -t swm-exp5b 2>/dev/null && { echo "tmux session swm-exp5b already exists on $HOST"; exit 1; }
grep -q '^TRAIN_JOB_DONE' $OUT/train_larg.log 2>/dev/null && { echo "$OUT/train_larg.log already says TRAIN_JOB_DONE"; exit 1; }
# scratch is one filer across LARG hosts: two trainers on one out dir would
# overwrite each other's resume states
[ -f $OUT/owner ] && [ "$(cat $OUT/owner)" != "$HOST" ] && {
  echo "$OUT belongs to a run on $(cat $OUT/owner) — confirm it is dead there, then remove $OUT/owner"; exit 1; }
mkdir -p $OUT $WB/cache
sed -e "s|^run_name:.*|run_name: $RUN|" -e "s|^out_dir:.*|out_dir: $OUT|" $SRC > $OUT/train_larg.yaml.new
grep -q "^run_name: $RUN\$" $OUT/train_larg.yaml.new && grep -q "^out_dir: $OUT\$" $OUT/train_larg.yaml.new \
  || { echo "could not derive the LARG config from $SRC"; exit 1; }
# a relaunch resumes this run, so it must not pick up a changed recipe
if [ -f $OUT/train_larg.yaml ]; then
  cmp -s $OUT/train_larg.yaml $OUT/train_larg.yaml.new \
    || { echo "$SRC no longer matches the config this run started with ($OUT/train_larg.yaml) — refusing"; exit 1; }
  rm $OUT/train_larg.yaml.new
else
  mv $OUT/train_larg.yaml.new $OUT/train_larg.yaml
fi
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU)
case "$UUID" in GPU-*) ;; *) echo "no UUID for GPU $GPU on $HOST: $UUID"; exit 1 ;; esac
cat > $OUT/run_train_larg.sh <<INNER
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
# wandb run files, artifact staging and artifact cache: scratch, not \$HOME or /tmp
export WANDB_DIR=$WB
export WANDB_DATA_DIR=$WB
export WANDB_CACHE_DIR=$WB/cache
set -a; source /scratch/cluster/krohling/swm-followup/.env; set +a
cd $W
echo "TRAIN_START $HOST g$GPU \$(date '+%m-%d %H:%M') $UUID"
nice -n 10 /scratch/cluster/krohling/swm-followup/.venv/bin/python \\
  swm-next/swm_lora/train_exp1.py --config $OUT/train_larg.yaml --resume auto \\
  || { echo "TRAIN_JOB_FAILED \$(date '+%m-%d %H:%M')"; exit 1; }
echo "TRAIN_JOB_DONE \$(date '+%m-%d %H:%M')"
INNER
chmod +x $OUT/run_train_larg.sh
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
echo $HOST > $OUT/owner
tmux new-session -d -s swm-exp5b "$W/$OUT/run_train_larg.sh 2>&1 | tee -a $W/$OUT/train_larg.log"
echo "launched exp5b training in tmux:swm-exp5b on $HOST GPU $GPU ($UUID), log $OUT/train_larg.log"
