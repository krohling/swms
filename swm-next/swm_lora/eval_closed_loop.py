"""Closed-loop planning eval: trained SWM checkpoint as the MPC scorer.

Replaces the (sim rollout + judge end-frame) scoring of verifier-MPC with the
SWM's direct estimate p_yes(frame_t, question, trajectory) per candidate — the
deployment configuration the SWM is trained for. Selection mirrors
datagen/verifier-MPC exactly:

  gate  g   = EMA(p_grasp, alpha 0.35); p_grasp = p_yes(frame_t, q_hold, ZERO
              trajectory) — the zero-traj probe is the SWM's frame-only readout
  score_i   = (1-g) * p_app_i + g * (0.6 * p_place_i + 0.4 * p_hold_i)
  (q_hold defaults to q_app, so p_hold_i = p_app_i without an extra forward)

Candidates come from the same diffusion-policy temperature ladder (0..2, k
rungs) as verifier-MPC — no min_sep rejection here: eval must match the
deployment proposal distribution, not the training-data recipe.

Eval seeds 6000+ (reserved range, disjoint from all generation seeds).

Usage (m5):
  PYTHONPATH=$PWD:$PWD/swm-next .venv/bin/python \
    swm-next/swm_lora/eval_closed_loop.py \
    --ckpt outputs_swm_lora_local/run4_ckpt_step28000 \
    --config swm-next/configs/datagen_hybrid_bg.yaml \
    --train-config swm-next/swm_lora/train4_hybrid.yaml \
    --episodes 50 --k 8 --out results/run4_28k_bg.json
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

from mpc import make_env, build_questions  # noqa: E402
from model import QwenSWM                  # noqa: E402
from data import action_stats, normalize   # noqa: E402

GATE_ALPHA = 0.35
PLACE_W, HOLD_W = 0.6, 0.4


def run_episode(env, goal, swm, policy_path, cfg, q_app, q_place, q_hold,
                seed, k, temps, amin, amax):
    torch.manual_seed(hash((seed, "mpc")) % 2**31)
    np.random.seed(hash(("mpc", seed)) % 2**31)

    frame = goal.reset_env(seed=seed)
    goal.reset_hook()
    policy = DiffusionPolicy.load(policy_path, device=cfg["device"])
    policy.add_obs(frame)

    n_exec = cfg["actions_per_cycle"]
    H, A = swm.horizon, len(amin)
    zero_traj = torch.zeros(1, H, A)

    gate_ema = None
    done = False
    cycle = -1
    for cycle in range(cfg["max_cycles"]):
        torch.manual_seed(hash((seed, cycle)) % 2**31)
        candidates = np.asarray(policy.sample_trajs(k, temperatures=temps))
        frame_t = np.asarray(frame, dtype=np.uint8)
        from PIL import Image
        img = Image.fromarray(frame_t).convert("RGB")

        trajs = torch.from_numpy(
            normalize(candidates[:, :H].astype(np.float32), amin, amax))
        p_app = swm.p_yes([img] * k, [q_app] * k, trajs).cpu().numpy()
        p_place = swm.p_yes([img] * k, [q_place] * k, trajs).cpu().numpy()
        p_hold = p_app if q_hold == q_app else \
            swm.p_yes([img] * k, [q_hold] * k, trajs).cpu().numpy()

        # frame-only gate via the zero-trajectory probe
        rg = float(swm.p_yes([img], [q_hold], zero_traj)[0])
        gate_ema = rg if gate_ema is None else \
            GATE_ALPHA * rg + (1 - GATE_ALPHA) * gate_ema
        g = gate_ema
        scores = (1.0 - g) * p_app + g * (PLACE_W * p_place + HOLD_W * p_hold)
        pick = int(np.argmax(scores))

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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True,
                    help="datagen/env config (env, questions, diffusion_path)")
    ap.add_argument("--train-config", required=True,
                    help="training yaml (traj_cumsum, proj_init_scale, dims)")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed-start", type=int, default=6000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tcfg = yaml.safe_load(open(args.train_config))

    swm = QwenSWM(model_id=tcfg["model_id"], action_dim=tcfg["action_dim"],
                  horizon=tcfg["horizon"], lora_r=tcfg["lora_r"],
                  lora_alpha=tcfg["lora_alpha"], lora_dropout=0.0,
                  device=cfg["device"],
                  proj_init_scale=float(tcfg.get("proj_init_scale", 0.1)),
                  traj_cumsum=bool(tcfg.get("traj_cumsum", False)))
    swm.load_adapters(args.ckpt)
    swm.model.eval()

    env = make_env(cfg)
    goal = get_ogbench_goal("stack_blocks", env, None, ANSWER_OPTIONS,
                            {"block_combo": cfg["block_combo"]})
    tmpl = build_questions(cfg)
    qcfg = cfg.get("questions") or {}
    q_app = qcfg.get("approach", tmpl["grasp"])
    q_place = qcfg.get("place", tmpl["ontop"])
    q_hold = qcfg.get("hold", q_app)

    amin, amax = action_stats(cfg["diffusion_path"])
    temps = [i * 2.0 / (args.k - 1) for i in range(args.k)] if args.k > 1 else [0.0]

    results = []
    wins = 0
    for i in range(args.episodes):
        seed = args.seed_start + i
        with torch.no_grad():
            success, cycles = run_episode(
                env, goal, swm, cfg["diffusion_path"], cfg, q_app, q_place,
                q_hold, seed, args.k, temps, amin, amax)
        wins += success
        results.append(dict(seed=seed, success=bool(success), cycles=cycles))
        print(f"[{i+1}/{args.episodes}] seed {seed}: "
              f"{'success' if success else 'fail'} @ {cycles} "
              f"(SR {wins/(i+1):.0%})", flush=True)

    summary = dict(ckpt=args.ckpt, task="_".join(cfg["block_combo"]),
                   k=args.k, episodes=args.episodes,
                   seed_start=args.seed_start,
                   sr=wins / args.episodes, episodes_detail=results)
    print(f"EVAL_DONE sr={summary['sr']:.2%}", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(summary, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
