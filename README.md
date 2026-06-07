# Semantic World Models

Evaluation code for **Semantic World Models** — a PaliGemma-based world model
with action conditioning, used as the reward signal for sampling-based and
gradient-based planners.

Project page: https://weirdlabuw.github.io/swm

## Installation

```bash
git clone https://github.com/weirdlabuw/semantic_world_models.git
cd semantic_world_models
uv sync
source .venv/bin/activate
```
## Downloading model checkpoints:
There is a script to download all of the checkpoints to the checkpoints directory. Run it by calling 
```bash
bash ckpts/download_checkpoints.sh
```


## Running evaluation

To run the evaluation, use any of the provided configs:

```bash
python scripts/evaluate_swm_hydra.py --config-name eval_gradient_full
```

Example configs provided are in `configs/` and are the following:

- `eval_base_diffusion_lt.yaml` — `lang_table` base diffusion (expert-policy) evaluation for block-pushing tasks.
- `eval_base_ogbench.yaml` — `ogbench` base diffusion (expert-policy) evaluation for cube-stacking tasks.
- `eval_gradient_full.yaml` — `lang_table` block-pushing tasks with gradient planning.
- `eval_gradient_full_ogbench.yaml` — `ogbench` cube-stacking tasks with gradient planning.

Each run writes a video, heat maps, and a `results.txt` summary under
`<root_save_path>`.

