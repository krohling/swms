#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# get the base diffusion checkpoints for lang table
hf download jacob3333/lt_base_diffusion --local-dir "$SCRIPT_DIR/lt_base_diffusion"

# get the base diffusion checkpoints for ogbench
hf download jacob3333/ogbench_base_diffusion --local-dir "$SCRIPT_DIR/ogbench_base_diffusion"

# get the paligemma_wm checkpoint for lang table
hf download jacob3333/paligemma_wm_lt --local-dir "$SCRIPT_DIR/paligemma_wm_lt"

# # get the paligemma_wm checkpoint for ogbench
hf download jacob3333/paligemma_wm_ogbench --local-dir "$SCRIPT_DIR/paligemma_wm_ogbench"
