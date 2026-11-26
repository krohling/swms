#!/bin/bash
#SBATCH -J swm-smoke6a
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o %x-%j.out
# 60-step smoke test of train_exp6a.yaml's new pieces (train_exp6a_smoke.yaml):
# fixed val split, explicit total_steps, 16-sequence forward batch at k=1.
# wandb stays offline so no smoke checkpoint lands in the shared artifact
# collection. The job name and end marker deliberately do NOT match
# train_exp6a.sbatch's DONE guard (swm-exp6a-*.out / TRAIN_JOB_DONE).
# Pace = (mtime ckpt_step40 - mtime ckpt_step20) / 20; memory in gpu_smoke6a.log.
set -x
W=$WORK/swm-followup/swms-exp6
V=$WORK/swm-followup/swms-lt/.venv
cd $W
export HF_HOME=$WORK/hf_cache PYTHONPATH=$W:$W/swm-next PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline WANDB_DATA_DIR=$WORK/wandb_data
[ -f $W/datasets_swm/bg_v3b_k8uni_s60000.h5 ] || { echo INIT_MISSING_V3B_H5; exit 1; }
[ -f $W/ckpts/ogbench_base_diffusion/blue_cube_green_cube.pt ] || { echo INIT_MISSING_DIFFUSION_CKPT; exit 1; }
[ -f $W/outputs_swm_lora/exp4b_warmup_v3b/ckpt_step5250/projector.pt ] || { echo INIT_MISSING_PROJECTOR; exit 1; }
[ -f $W/swm-next/swm_lora/splits/exp4b_split.json ] || { echo INIT_MISSING_FIXED_VAL_SPLIT; exit 1; }
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -l 15 > $W/gpu_smoke6a.log &
date +%s
$V/bin/python swm-next/swm_lora/train_exp1.py --config swm-next/swm_lora/train_exp6a_smoke.yaml --resume none || { echo SMOKE_JOB_FAILED; exit 1; }
date +%s
stat -c '%Y %n' outputs_swm_lora/exp6a_smoke/ckpt_step*
echo SMOKE_JOB_DONE
