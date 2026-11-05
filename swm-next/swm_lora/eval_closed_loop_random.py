"""Closed-loop random-pick control (matched-k): same env, same temperature
ladder and per-episode/cycle seeding as the model evals, but the executed
candidate is drawn uniformly at random. Gives the no-knowledge k=8 anchor on
any seed block (the original 48/56/38 control row was block A only).

Usage (m5):
  PYTHONPATH=$PWD:$PWD/swm-next .venv/bin/python \
    swm-next/swm_lora/eval_closed_loop_random.py \
    --config swm-next/configs/datagen_hybrid_bg.yaml \
    --episodes 50 --seed-start 6050 --k 8 \
    --out experiments/exp2/eval_cl_random_k8_bg_seeds6050.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "verifier"))

from swm.constants import ANSWER_OPTIONS
from swm.diffusion_policy import DiffusionPolicy
from swm.utils.goal_generators import get_ogbench_goal

from mpc import make_env  # noqa: E402


def run_episode(env, goal, policy_path, cfg, seed, k, temps):
    torch.manual_seed(hash((seed, "mpc")) % 2**31)
    np.random.seed(hash(("mpc", seed)) % 2**31)

    frame = goal.reset_env(seed=seed)
    goal.reset_hook()
    policy = DiffusionPolicy.load(policy_path, device=cfg["device"])
    policy.add_obs(frame)

    n_exec = cfg["actions_per_cycle"]
    done = False
    cycle = -1
    for cycle in range(cfg["max_cycles"]):
        torch.manual_seed(hash((seed, cycle)) % 2**31)
        candidates = np.asarray(policy.sample_trajs(k, temperatures=temps))
        pick = int(np.random.randint(k))
        for a in candidates[pick][:n_exec]:
            frame = env.step(np.asarray(a))
            policy.add_obs(frame)
            if goal.get_done():
                done = True
                break
        if done:
            break
    return done, cycle + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed-start", type=int, default=6000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    k = args.k
    env = make_env(cfg)
    goal = get_ogbench_goal("stack_blocks", env, None, ANSWER_OPTIONS,
                            {"block_combo": cfg["block_combo"]})
    temps = [i * 2.0 / (k - 1) for i in range(k)]

    results = []
    wins = 0
    for i in range(args.episodes):
        seed = args.seed_start + i
        with torch.no_grad():
            success, cycles = run_episode(
                env, goal, cfg["diffusion_path"], cfg, seed, k, temps)
        wins += success
        results.append(dict(seed=seed, success=bool(success), cycles=cycles))
        print(f"[{i+1}/{args.episodes}] seed {seed}: "
              f"{'success' if success else 'fail'} @ {cycles} "
              f"(SR {wins/(i+1):.0%})", flush=True)

    summary = dict(task="_".join(cfg["block_combo"]), k=k,
                   episodes=args.episodes, seed_start=args.seed_start,
                   scorer="random_pick",
                   sr=wins / args.episodes, episodes_detail=results)
    print(f"EVAL_DONE sr={summary['sr']:.2%}", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(summary, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
