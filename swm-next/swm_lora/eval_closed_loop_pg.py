"""Closed-loop planning eval for the PaliGemma full-FT SWM (pg-fullft arm).

Port of eval_closed_loop.py (add-linear-and-warmup @ cc76b68) with the model
swapped for PaliGemmaSWM. The deployment scoring loop is byte-identical —
soft-gate EMA blend, zero-trajectory gate probe, diffusion temperature
ladder, eval seeds 6000+ — so SR numbers are directly comparable with the
exp1/exp1b/teacher columns.

  gate  g   = EMA(p_grasp, alpha 0.35); p_grasp = p_yes(frame_t, q_hold, ZERO
              trajectory) — the zero-traj probe is the SWM's frame-only readout
  score_i   = (1-g) * p_app_i + g * (0.6 * p_place_i + 0.4 * p_hold_i)
  (q_hold defaults to q_app, so p_hold_i = p_app_i without an extra forward)

Checkpoint loading (save_full contract): --ckpt is an HF save_pretrained dir;
PaliGemmaSWM(base_model=ckpt) restores the LM/vision weights, then
load_adapters(ckpt) restores the trajectory adapter (projector.pt), which
__init__ re-initialises fresh after from_pretrained.

Usage (m5, paligemma-fullft worktree):
  PYTHONPATH=$PWD:$PWD/swm-next ../swms/.venv/bin/python \
    swm-next/swm_lora/eval_closed_loop_pg.py \
    --ckpt outputs_swm_lora_local/pg_ckpt_step80367 \
    --config swm-next/configs/datagen_hybrid_bg.yaml \
    --train-config swm-next/swm_lora/train_pg_fullft.yaml \
    --episodes 50 --k 8 --out experiments/pg_fullft/eval_cl_step80367_bg_k8.json
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

from mpc import make_env, build_questions   # noqa: E402
from model_pg import PaliGemmaSWM           # noqa: E402
from data import action_stats, normalize    # noqa: E402

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
    ap.add_argument("--ckpt", required=True,
                    help="save_full dir (HF shards + projector.pt)")
    ap.add_argument("--config", required=True,
                    help="datagen/env config (env, questions, diffusion_path)")
    ap.add_argument("--train-config", required=True,
                    help="train_pg_fullft.yaml (dims, cumsum, processor_path)")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed-start", type=int, default=6000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tcfg = yaml.safe_load(open(args.train_config))

    swm = PaliGemmaSWM(
        base_model=args.ckpt,
        processor_path=tcfg.get("processor_path", "ckpts/paligemma_wm_ogbench"),
        action_dim=tcfg["action_dim"], horizon=tcfg["horizon"],
        device=cfg["device"], proj_hidden=tcfg.get("proj_hidden", 512),
        traj_cumsum=bool(tcfg.get("traj_cumsum", True)),
        full_finetune=False)
    # __init__ swaps in a fresh projector AFTER from_pretrained — restore the
    # trained trajectory adapter from the checkpoint's projector.pt.
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
