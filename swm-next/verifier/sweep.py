"""Variable-horizon verification sweep over banked rollouts.

Question under test (Jiaheng, 2026-09-04): is zero-shot Qwen a useful
INCREMENTAL trajectory-selection signal -- can it pick the best of K partial
rollouts partway through an episode, the way our SWM emits a value signal at
every replan cycle -- rather than only judging finished episodes?

For each horizon t (in replan cycles) and each prompt variant, all K partial
rollouts per seed are scored at time t and we report:

    selection_sr  -- argmax over the K scores picks a rollout that
                     EVENTUALLY succeeds (per-seed, averaged)
    auc           -- rank-AUC of score vs eventual success, all rollouts
    n_terminated  -- rollouts already past their oracle-success cycle at t

Needs a bank from verifier/capture.py (per-cycle frames). Successful rollouts
end early; their last frame persists for later horizons.

Usage (repo root on PYTHONPATH):

    PYTHONPATH=$PWD:$PWD/swm-next python swm-next/verifier/sweep.py --config swm-next/configs/verifier.yaml
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import h5py
import numpy as np
import yaml


@dataclass
class Rollout:
    seed: int
    k: int
    success: bool          # eventual oracle success
    done_cycle: int        # -1 if never succeeded
    frames: np.ndarray     # (C+1, H, W, 3); index 0 is pre-action

    def frame_at(self, t: int) -> np.ndarray:
        return self.frames[min(t, len(self.frames) - 1)]


def load_bank(path: str) -> list[Rollout]:
    rollouts = []
    with h5py.File(path, "r") as f:
        for name in sorted(f.keys()):
            g = f[name]
            if "cycle_frames" not in g:
                raise RuntimeError(
                    f"{name} lacks cycle_frames -- bank was made by the old "
                    "capture; re-run verifier/capture.py")
            rollouts.append(Rollout(
                seed=int(g.attrs["seed"]), k=int(g.attrs["k"]),
                success=bool(g.attrs["success"]),
                done_cycle=int(g.attrs["done_cycle"]),
                frames=np.asarray(g["cycle_frames"])))
    return rollouts


# --------------------------------------------------------------- variants
#
# A variant scores all rollouts at horizon t: (judge, rollouts, t) -> scores.
# To experiment with a new prompt or score composition, add a function here
# and register it in VARIANTS at the bottom of this block.

TOP, BOTTOM = "blue cube", "green cube"   # overwritten from config in main()


def _score(judge, images, question, batch=8):
    """Pooled-p_yes for one question over a list of images (or image pairs)."""
    out = np.zeros(len(images))
    for i in range(0, len(images), batch):
        p, _ = judge.p_yes(images[i:i + batch], [question] * len(images[i:i + batch]))
        out[i:i + len(images[i:i + batch])] = p
    return out


def end_state(judge, rollouts, t):
    """Terminal state question on the frame at t. The original experiment's
    winner; by construction it cannot discriminate early in the episode."""
    q = f"Is the {TOP} stacked on top of the {BOTTOM}?"
    return _score(judge, [r.frame_at(t) for r in rollouts], q)


def progress_pair(judge, rollouts, t):
    """(start frame, frame at t) pair with a progress question."""
    q = (f"The robot was instructed to: Stack the {TOP} on top of the {BOTTOM}. "
         "Is the robot making progress toward completing this task?")
    return _score(judge, [(r.frames[0], r.frame_at(t)) for r in rollouts], q)


def phase_schedule(judge, rollouts, t):
    """Planning's phase-1 objective on the frame at t, exact phrasings from
    StackBlocksGoal: 0.6 * p(on top) + 0.4 * p(grasping). The most faithful
    analog of the SWM's role at a replan step."""
    frames = [r.frame_at(t) for r in rollouts]
    p_ontop = _score(judge, frames, f"Is the {TOP} on top of the {BOTTOM}?")
    p_grasp = _score(judge, frames, f"Is the robot grasping the {TOP}?")
    return 0.6 * p_ontop + 0.4 * p_grasp


VARIANTS = {
    "end_state": end_state,
    "progress_pair": progress_pair,
    "phase_sched": phase_schedule,
}


# ---------------------------------------------------------------- metrics

def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = labels.sum(), (1 - labels).sum()
    if not (pos and neg):
        return float("nan")
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def selection_sr(scores: np.ndarray, rollouts: list[Rollout]) -> float:
    """Per seed: does argmax over the K scores pick an eventual success?"""
    picks = []
    for seed in sorted({r.seed for r in rollouts}):
        idx = [i for i, r in enumerate(rollouts) if r.seed == seed]
        picks.append(rollouts[idx[int(np.argmax(scores[idx]))]].success)
    return float(np.mean(picks))


def metrics_at(scores, rollouts, t) -> dict:
    labels = np.array([r.success for r in rollouts], dtype=float)
    return dict(
        selection_sr=selection_sr(scores, rollouts),
        auc=rank_auc(scores, labels),
        n_terminated=int(sum(0 <= r.done_cycle <= t for r in rollouts)),
        scores=[round(float(s), 4) for s in scores],
    )


# ------------------------------------------------------------------ main

def main():
    global TOP, BOTTOM
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    TOP, BOTTOM = [b.replace("_", " ") for b in cfg["block_combo"]]

    import sys
    sys.dont_write_bytecode = True
    from label_teacher import QwenJudge
    judge = QwenJudge(model_id=cfg["model_id"], device=cfg["device"])

    rollouts = load_bank(os.path.join(cfg["out_dir"], "rollouts.h5"))
    horizons = cfg.get("horizons", [1, 2, 4, 8, 16, 32, 50])
    baseline = float(np.mean([r.success for r in rollouts]))
    ceiling = selection_sr(np.array([float(r.success) for r in rollouts]), rollouts)
    print(f"{len(rollouts)} rollouts | random baseline {baseline:.1%} | "
          f"best-of-K ceiling {ceiling:.1%}", flush=True)

    results = {name: {} for name in VARIANTS}
    for t in horizons:
        for name, variant in VARIANTS.items():
            results[name][t] = metrics_at(variant(judge, rollouts, t), rollouts, t)
        print(f"t={t:3d}: " + "  ".join(
            f"{name} SR={results[name][t]['selection_sr']:.0%} "
            f"AUC={results[name][t]['auc']:.2f}" for name in VARIANTS), flush=True)

    out_path = os.path.join(cfg["out_dir"], "sweep_results.json")
    with open(out_path, "w") as fh:
        json.dump(dict(config=cfg, horizons=horizons,
                       random_baseline=baseline, best_of_k_ceiling=ceiling,
                       variants=results), fh, indent=2)
    print(f"SWEEP_DONE: {out_path}", flush=True)


if __name__ == "__main__":
    main()
