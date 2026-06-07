import os

# Must be set before the CUDA context is created (i.e. before any torch CUDA
# op runs); otherwise torch.use_deterministic_algorithms(True) will refuse to
# enable cuBLAS deterministic kernels.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm.auto import tqdm
from swm.evaluation import eval
import absl.logging
from swm.constants import ANSWER_OPTIONS
from swm.semantic_world_model import SWMGradModel
import ogbench # import used to resolve the ogbench registration

absl.logging.set_verbosity(absl.logging.WARNING)


def _resolve_tasks(cfg: DictConfig):
    """Build a list of (block_combo, diffusion_path) tasks from the config.

    If `cfg.tasks` is provided it is used as-is. Otherwise we fall back to the
    legacy single-task layout (`cfg.block_combo` + `cfg.paths.diffusion_path`)
    so older configs keep working.
    """
    raw_tasks = OmegaConf.select(cfg, "tasks", default=None)
    if raw_tasks is not None and len(raw_tasks) > 0:
        tasks = []
        for i, t in enumerate(raw_tasks):
            if "block_combo" not in t or "diffusion_path" not in t:
                raise ValueError(
                    f"tasks[{i}] must define both 'block_combo' and 'diffusion_path'; got {t}"
                )
            tasks.append({
                "block_combo": tuple(t.block_combo),
                "diffusion_path": t.diffusion_path,
            })
        return tasks

    return [{
        "block_combo": tuple(cfg.block_combo),
        "diffusion_path": cfg.paths.diffusion_path,
    }]


@hydra.main(config_path="../configs", config_name="eval_gradient_full", version_base=None)
def run_evaluation(cfg: DictConfig):
    """
    Run evaluation over multiple seeds and one or more tasks.

    Each task is an (block_combo, diffusion_path) pair. The same SWM model
    checkpoint is reused across tasks; only the diffusion policy and target
    block combo change per task.
    """
    if cfg.get("deterministic", False):
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


    root_save_path = cfg.paths.root_save_path
    if not os.path.exists(root_save_path):
        os.makedirs(root_save_path)

    planning_cfg = cfg.planning
    tasks = _resolve_tasks(cfg)

    overall_results = []
    overall_success = 0
    overall_total = 0
    overall_time = 0.0
    device = cfg.get("device", "cuda")
    model = None
    if not planning_cfg.expert_diffusion:
        model = SWMGradModel(checkpoint_path=cfg.paths.model_ckpt_path, processor_path=cfg.paths.processor_path, tokens=ANSWER_OPTIONS,
                             precision=torch.bfloat16, device=device)

    for task_idx, task in enumerate(tasks):
        block_combo = task["block_combo"]
        diffusion_path = task["diffusion_path"]

        results = []
        success_count = 0
        total_time = 0.0

        task_desc = f"task {task_idx + 1}/{len(tasks)}: {block_combo[0]} -> {block_combo[1]}"
        for seed in tqdm(
            range(cfg.seed_start, cfg.seed_start + cfg.num_seeds),
            desc=f"Running {task_desc}",
        ):
            uid = f"{cfg.name}/{planning_cfg.planning_name}/{block_combo[0]}_{block_combo[1]}"
            run_name = f"{uid}/{seed}"
            output_dir = f"{root_save_path}/{run_name}/"

            reward_kwargs = {
                "block_combo": block_combo,
                "ood": False,
            }

            success, time_taken = eval(
                seed=seed,
                reward_type=cfg.goal_type,
                env_type=cfg.env_type,
                device=device,

                output_dir=output_dir,
                ckpt_path=cfg.paths.model_ckpt_path,
                processor_path=cfg.paths.processor_path,
                model=model,
                diffusion_path=diffusion_path,

                diffusion=planning_cfg.diffusion,
                mppi=planning_cfg.mppi,
                gradient=planning_cfg.gradient,
                expert_diffusion=planning_cfg.expert_diffusion,

                precision=torch.bfloat16,
                action_skip=planning_cfg.action_skip,
                model_batch_size=cfg.model_batch_size,

                reward_kwargs=reward_kwargs,
                action_dim=cfg.action_dim,

                num_steps=planning_cfg.num_steps,
                num_actions_executed=planning_cfg.num_actions_executed,
                pred_horizon=planning_cfg.pred_horizon,
                num_samples=planning_cfg.num_samples,
                num_planning_iters=planning_cfg.num_planning_iters,

                gradient_lr=planning_cfg.gradient_lr,
                gradient_clipping_value=planning_cfg.gradient_clipping_value,

                mppi_temperature=planning_cfg.mppi_temperature,

                intermediate_hm=cfg.intermediate_hm,
            )

            results.append((seed, int(success), time_taken))
            success_count += int(success)
            total_time += time_taken

            print(f"\n[{block_combo[0]}->{block_combo[1]}] Seed {seed}: Success={success}, Time={time_taken:.2f} minutes")

        print("\n" + "=" * 80)
        print(f"TASK SUMMARY: {block_combo[0]} -> {block_combo[1]}")
        print("=" * 80)
        print(f"Configuration: {uid}")
        print(f"Diffusion checkpoint: {diffusion_path}")
        print(f"Total runs: {cfg.num_seeds}")
        print(f"Successful runs: {success_count}")
        print(f"Success rate: {success_count / cfg.num_seeds:.2%}")
        print(f"Average time: {total_time / cfg.num_seeds:.2f} minutes")
        print("=" * 80)

        results_file = os.path.join(root_save_path, "results.txt")
        with open(results_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Configuration: {uid}\n")
            f.write(f"Block combo: {block_combo}\n")
            f.write(f"Diffusion checkpoint: {diffusion_path}\n")
            f.write(f"Success rate: {success_count}/{cfg.num_seeds} ({success_count / cfg.num_seeds:.2%})\n")
            f.write(f"Average time: {total_time / cfg.num_seeds:.2f} minutes\n")
            f.write(f"Individual results:\n")
            for seed, success, time_taken in results:
                f.write(f"  Seed {seed}: Success={success}, Time={time_taken:.2f} min\n")
            f.write(f"{'='*80}\n")

        overall_results.append({
            "block_combo": block_combo,
            "diffusion_path": diffusion_path,
            "success_count": success_count,
            "num_seeds": cfg.num_seeds,
            "total_time": total_time,
        })
        overall_success += success_count
        overall_total += cfg.num_seeds
        overall_time += total_time

    if len(tasks) > 1:
        print("\n" + "#" * 80)
        print("OVERALL SUMMARY (all tasks)")
        print("#" * 80)
        for r in overall_results:
            bc = r["block_combo"]
            print(
                f"  {bc[0]} -> {bc[1]}: "
                f"{r['success_count']}/{r['num_seeds']} "
                f"({r['success_count'] / r['num_seeds']:.2%}), "
                f"avg {r['total_time'] / r['num_seeds']:.2f} min"
            )
        print(
            f"TOTAL: {overall_success}/{overall_total} "
            f"({overall_success / overall_total:.2%}), "
            f"avg {overall_time / overall_total:.2f} min"
        )
        print("#" * 80)

        with open(os.path.join(root_save_path, "results.txt"), "a") as f:
            f.write(f"\n{'#'*80}\nOVERALL SUMMARY (all tasks)\n{'#'*80}\n")
            for r in overall_results:
                bc = r["block_combo"]
                f.write(
                    f"  {bc[0]} -> {bc[1]}: "
                    f"{r['success_count']}/{r['num_seeds']} "
                    f"({r['success_count'] / r['num_seeds']:.2%}), "
                    f"avg {r['total_time'] / r['num_seeds']:.2f} min\n"
                )
            f.write(
                f"TOTAL: {overall_success}/{overall_total} "
                f"({overall_success / overall_total:.2%}), "
                f"avg {overall_time / overall_total:.2f} min\n"
            )
            f.write(f"{'#'*80}\n")

    print(f"\nResults saved to: {os.path.join(root_save_path, 'results.txt')}")


if __name__ == "__main__":
    run_evaluation()
