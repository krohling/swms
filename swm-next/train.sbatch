#!/bin/bash
# Production launcher. Any argument is forwarded to train.py as a config
# override, so the same script serves the full run and any short variant:
#
#   sbatch train.sbatch                                        # 64k steps, ~38 h
#   sbatch -p gh-dev -t 02:00:00 train.sbatch max_steps=200    # short check
#
#SBATCH -J swm-repro
#SBATCH -o /work/11599/krohling/vista/swm-repro/logs/%x-%j.out
#SBATCH -e /work/11599/krohling/vista/swm-repro/logs/%x-%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -A ASC25021
set -uo pipefail
S=$WORK/swm-followup/swms
source $S/.venv/bin/activate
source $WORK/swm-followup/.wandb_env          # compute nodes cannot reach 1Password
export SWMS_REPO=$S PYTHONPATH=$S
# Base PaliGemma is cached under $WORK/hf_cache in the pre-hub/ layout, and
# compute nodes have no outbound network -- fail loudly instead of hanging.
export HF_HUB_CACHE=$WORK/hf_cache HF_HUB_OFFLINE=1

cd $WORK/swm-repro/swm-next
python -u train.py --config configs/repro.yaml "$@"
