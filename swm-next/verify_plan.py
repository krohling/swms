"""Verifier-planning experiment: can zero-shot Qwen select successful rollouts?

Stage A (rollout): for each seed, run the base diffusion policy K times from
the identical start state (env reset is seed-deterministic; policy sampling
varies by a per-(seed,k) torch seed). Bank start/end frames, oracle success,
and sim states to an HDF5 -- verification then never re-pays rollout cost.

Stage B (verify): score every banked rollout with pooled-p_yes Qwen prompts
(QwenJudge from label_teacher.py) under multiple prompt variants, and report
per-variant: AUC vs oracle success, accuracy@0.5, per-seed argmax selection
SR, plus the random-selection baseline and the oracle best-of-K ceiling.

Run from the repo root with the repo on PYTHONPATH:

    python swm-next/verify_plan.py --config swm-next/configs/verifier.yaml --stage all
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import h5py
import numpy as np
import torch
import yaml

from swm.constants import ANSWER_OPTIONS
from swm.diffusion_policy import DiffusionPolicy
from swm.utils.envs import get_ogbench_env
from swm.utils.goal_generators import get_ogbench_goal


def load_config():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["rollout", "verify", "all"], default="all")
    ap.add_argument("overrides", nargs="*", help="key=value")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    for ov in args.overrides:
        k, v = ov.split("=", 1)
        cur = cfg.get(k)
        cfg[k] = type(cur)(v) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else v
    return cfg, args.stage


def rollout_stage(cfg):
    import ogbench  # noqa: F401  (env registration)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    env = get_ogbench_env({"ood": False})
    goal = get_ogbench_goal("stack_blocks", env, None, ANSWER_OPTIONS,
                            {"block_combo": cfg["block_combo"]})
    path = os.path.join(cfg["out_dir"], "rollouts.h5")
    n_done = 0
    with h5py.File(path, "w") as out:
        out.attrs["block_combo"] = json.dumps(cfg["block_combo"])
        for seed in range(cfg["seed_start"], cfg["seed_start"] + cfg["num_seeds"]):
            for k in range(cfg["k_rollouts"]):
                # Identical start state per seed; per-k policy stochasticity.
                torch.manual_seed(hash((seed, k)) % 2**31)
                np.random.seed(hash((k, seed)) % 2**31)
                frame0 = goal.reset_env(seed=seed)
                goal.reset_hook()
                instruction = goal.get_instruction()
                dm = DiffusionPolicy.load(cfg["diffusion_path"], device=cfg["device"])
                dm.add_obs(frame0)
                current, success, cycles = frame0, False, 0
                state0 = env.get_state()
                for _ in range(cfg["max_cycles"]):
                    cycles += 1
                    for action in dm.get_action()[: cfg["actions_per_cycle"]]:
                        current = env.step(action)
                        dm.add_obs(current)
                        if goal.get_done():
                            success = True
                            break
                    if success:
                        break
                g = out.create_group(f"s{seed}_k{k}")
                g.create_dataset("start_frame", data=np.asarray(frame0, dtype=np.uint8))
                g.create_dataset("end_frame", data=np.asarray(current, dtype=np.uint8))
                g.attrs["success"] = bool(success)
                g.attrs["cycles"] = cycles
                g.attrs["seed"] = seed
                g.attrs["k"] = k
                g.attrs["instruction"] = instruction
                g.create_dataset("state_start", data=np.frombuffer(
                    pickle.dumps(state0), dtype=np.uint8))
                g.create_dataset("state_end", data=np.frombuffer(
                    pickle.dumps(env.get_state()), dtype=np.uint8))
                n_done += 1
            done_rate = _bank_success_rate(path)
            print(f"seed {seed}: banked {n_done} rollouts (running success {done_rate:.1%})",
                  flush=True)
    print(f"ROLLOUTS_DONE: {path}", flush=True)


def _bank_success_rate(path):
    with h5py.File(path, "r") as f:
        s = [f[g].attrs["success"] for g in f.keys()]
    return float(np.mean(s)) if s else 0.0


def _prompts(cfg, instruction, top, bottom):
    """Each variant: (name, uses_pair, question_text)."""
    return [
        ("end_state", False,
         f"Is the {top} stacked on top of the {bottom}?"),
        ("completion_pair", True,
         f"The robot was instructed to: {instruction}. "
         "Did the robot successfully complete this task?"),
    ]


def verify_stage(cfg):
    import sys
    sys.dont_write_bytecode = True
    from label_teacher import QwenJudge

    path = os.path.join(cfg["out_dir"], "rollouts.h5")
    judge = QwenJudge(model_id=cfg["model_id"], device=cfg["device"])
    top, bottom = [b.replace("_", " ") for b in cfg["block_combo"]]

    rollouts = []
    with h5py.File(path, "r") as f:
        for name in sorted(f.keys()):
            g = f[name]
            rollouts.append(dict(
                name=name, seed=int(g.attrs["seed"]), k=int(g.attrs["k"]),
                success=bool(g.attrs["success"]),
                instruction=str(g.attrs["instruction"]),
                start=np.asarray(g["start_frame"]), end=np.asarray(g["end_frame"])))

    results = {}
    variants = _prompts(cfg, rollouts[0]["instruction"], top, bottom)
    for vname, pair, question in variants:
        scores = np.zeros(len(rollouts))
        masses = np.zeros(len(rollouts))
        B = int(cfg.get("batch", 8))
        for b0 in range(0, len(rollouts), B):
            chunk = rollouts[b0:b0 + B]
            imgs = [((r["start"], r["end"]) if pair else r["end"]) for r in chunk]
            p, m = judge.p_yes(imgs, [question] * len(chunk))
            scores[b0:b0 + len(chunk)] = p
            masses[b0:b0 + len(chunk)] = m
            print(f"[{vname}] {min(b0 + B, len(rollouts))}/{len(rollouts)}", flush=True)
        results[vname] = _metrics(rollouts, scores, masses, cfg)

    out = dict(config={k: v for k, v in cfg.items()}, variants=results)
    with open(os.path.join(cfg["out_dir"], "verify_results.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    _report(rollouts, results, cfg)


def _metrics(rollouts, scores, masses, cfg):
    succ = np.array([r["success"] for r in rollouts], dtype=float)
    # AUC via rank statistic
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = succ.sum(), (1 - succ).sum()
    auc = float((ranks[succ == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)) \
        if pos and neg else float("nan")
    sel, best = [], []
    seeds = sorted({r["seed"] for r in rollouts})
    for s in seeds:
        idx = [i for i, r in enumerate(rollouts) if r["seed"] == s]
        sel.append(succ[idx[int(np.argmax(scores[idx]))]])
        best.append(float(succ[idx].max()))
    return dict(
        auc=auc,
        acc_at_half=float(((scores >= 0.5) == (succ == 1)).mean()),
        selection_sr=float(np.mean(sel)),
        random_baseline=float(succ.mean()),
        best_of_k_ceiling=float(np.mean(best)),
        mean_mass=float(masses.mean()),
        scores=[round(float(x), 4) for x in scores],
    )


def _report(rollouts, results, cfg):
    succ = np.array([r["success"] for r in rollouts], dtype=float)
    print(f"\n=== verifier report (n={len({r['seed'] for r in rollouts})} seeds, "
          f"K={cfg['k_rollouts']}, {len(rollouts)} rollouts) ===")
    print(f"rollout success rate (random-selection baseline): {succ.mean():.1%}")
    for vname, m in results.items():
        print(f"[{vname:16s}] AUC={m['auc']:.3f}  acc@0.5={m['acc_at_half']:.3f}  "
              f"selection SR={m['selection_sr']:.1%}  "
              f"best-of-K ceiling={m['best_of_k_ceiling']:.1%}  mass={m['mean_mass']:.3f}")


def main():
    cfg, stage = load_config()
    if stage in ("rollout", "all"):
        rollout_stage(cfg)
    if stage in ("verify", "all"):
        verify_stage(cfg)


if __name__ == "__main__":
    main()
