"""Closed-loop planning eval for the SELECTION arm: the QwenSWMSelector picks
among k candidates directly (digit softmax), gate-blended across questions.

Per cycle:
  P_app  = softmax(selection_logits(frame, q_app,  all k trajs))   # (k,)
  P_place= softmax(selection_logits(frame, q_place, all k trajs))
  g      = EMA(p_yes(frame, q_hold, ZERO traj), alpha 0.35)  # inherited yes/no
                                                             # readout = frame-
                                                             # only gate probe
  score  = (1-g) * P_app + g * (0.6 * P_place + 0.4 * P_app)  # q_hold == q_app
  pick   = argmax(score)

This mirrors the scorer-arm gate blend (datagen/eval_closed_loop.py) with
selection softmaxes in place of per-candidate p_yes, so results are comparable.
Candidates come from the same temperature ladder; candidate order is NOT
shuffled at eval (the selector was trained order-invariant; ladder order is
deployment-realistic and keeps runs deterministic).

Usage (m5):
  PYTHONPATH=$PWD:$PWD/swm-next .venv/bin/python \
    swm-next/swm_lora/eval_closed_loop_sel.py \
    --ckpt outputs_swm_lora_local/run5b_ckpt_step21000 \
    --config swm-next/configs/datagen_hybrid_bg.yaml \
    --train-config swm-next/swm_lora/train_sel_hybrid.yaml \
    --episodes 50 --out results/run5b_21k_bg.json
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
from model import QwenSWMSelector          # noqa: E402
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
        from PIL import Image
        img = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")

        trajs = torch.from_numpy(
            normalize(candidates[:, :H].astype(np.float32), amin, amax)
        ).unsqueeze(0)                                        # (1, k, H, A)
        z = swm.selection_logits([img] * 2, [q_app, q_place],
                                 trajs.expand(2, -1, -1, -1))
        p_app, p_place = torch.softmax(z, dim=-1).cpu().numpy()

        rg = float(torch.sigmoid(
            swm.answer_logit([img], [q_hold], zero_traj))[0])
        gate_ema = rg if gate_ema is None else \
            GATE_ALPHA * rg + (1 - GATE_ALPHA) * gate_ema
        g = gate_ema
        p_hold = p_app if q_hold == q_app else None
        assert p_hold is not None, "distinct hold question not supported"
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
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-config", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed-start", type=int, default=6000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tcfg = yaml.safe_load(open(args.train_config))
    k = int(tcfg.get("k", 8))                 # k is baked into the selector

    swm = QwenSWMSelector(
        k=k, model_id=tcfg["model_id"], action_dim=tcfg["action_dim"],
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
    temps = [i * 2.0 / (k - 1) for i in range(k)]

    results = []
    wins = 0
    for i in range(args.episodes):
        seed = args.seed_start + i
        with torch.no_grad():
            success, cycles = run_episode(
                env, goal, swm, cfg["diffusion_path"], cfg, q_app, q_place,
                q_hold, seed, k, temps, amin, amax)
        wins += success
        results.append(dict(seed=seed, success=bool(success), cycles=cycles))
        print(f"[{i+1}/{args.episodes}] seed {seed}: "
              f"{'success' if success else 'fail'} @ {cycles} "
              f"(SR {wins/(i+1):.0%})", flush=True)

    summary = dict(ckpt=args.ckpt, task="_".join(cfg["block_combo"]),
                   k=k, episodes=args.episodes, seed_start=args.seed_start,
                   scorer="selector_softmax_gateblend",
                   sr=wins / args.episodes, episodes_detail=results)
    print(f"EVAL_DONE sr={summary['sr']:.2%}", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(summary, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
