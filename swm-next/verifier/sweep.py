"""Stage 2: score banked rollouts with the judge. Pure data capture.

Every question in the config is evaluated at every eval position (and, for
pair questions, over every (start, start+h) window) across all rollouts.
Raw pooled p_yes and yes/no mass are stored per evaluation -- NO metrics,
weights, or aggregation here. All of that is offline in analyze.py, so new
reports never re-pay judge cost.

Question config (verifier.yaml):

    stride: 4               # eval positions every N env steps
    h_values: [4, 8, 16]    # window widths for pair questions, in env steps
    questions:
      - {name: grasp,   text: "Is the robot grasping the {top}?"}
      - {name: ontop,   text: "Is the {top} on top of the {bottom}?"}
      - {name: progress, pair: true,
         text: "The robot was instructed to: {instruction}. Is the robot
                closer to completing this task?"}

Placeholders {top}, {bottom}, {instruction} are filled from the config.
Single-frame questions are scored at each position p (frame[p]); pair
questions at each window (frame[p], frame[p+h]) for every h. Positions past
a rollout's termination reuse its last frame (the achieved state persists);
the alive mask in the manifest marks which positions are pre-termination.

Outputs in out_dir:
    scores.npz     -- "{q}/p" and "{q}/mass" [R, P] for single questions,
                      "{q}/h{h}/p" and "/mass" [R, W_h] for pair questions,
                      plus "alive" [R, P] and the axis arrays: "traj_ids"
                      [R] (bank group names), "positions" [P] (env-step of
                      each column), "{q}/h{h}/window_starts" [W_h] (env-step
                      each window STARTS at; the pair judged is
                      (start, start+h))
    manifest.json  -- rollout order + attrs, positions, windows, questions

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
    name: str
    seed: int
    k: int
    success: bool
    done_step: int
    frames: np.ndarray   # (S+1, H, W, 3)

    def frame_at(self, step: int) -> np.ndarray:
        return self.frames[min(step, len(self.frames) - 1)]

    @property
    def steps(self) -> int:
        return len(self.frames) - 1


def load_bank(path: str) -> list[Rollout]:
    rollouts = []
    with h5py.File(path, "r") as f:
        for name in sorted(f.keys()):
            g = f[name]
            if "frames" not in g:
                raise RuntimeError(f"{name} lacks per-step frames -- old bank; "
                                   "re-run verifier/capture.py")
            rollouts.append(Rollout(
                name=name, seed=int(g.attrs["seed"]), k=int(g.attrs["k"]),
                success=bool(g.attrs["success"]),
                done_step=int(g.attrs["done_step"]),
                frames=np.asarray(g["frames"])))
    return rollouts


def fill(text: str, cfg: dict) -> str:
    top, bottom = [b.replace("_", " ") for b in cfg["block_combo"]]
    return (text.replace("{top}", top).replace("{bottom}", bottom)
                .replace("{instruction}", f"Stack the {top} on top of the {bottom}"))


def score_all(judge, images: list, question: str, batch: int) -> tuple:
    """(p_yes, mass) arrays for one question over a list of images/pairs."""
    p = np.zeros(len(images))
    mass = np.zeros(len(images))
    for i in range(0, len(images), batch):
        chunk = images[i:i + batch]
        cp, cm = judge.p_yes(chunk, [question] * len(chunk))
        p[i:i + len(chunk)], mass[i:i + len(chunk)] = cp, cm
    return p, mass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    import sys
    sys.dont_write_bytecode = True
    from label_teacher import QwenJudge
    judge = QwenJudge(model_id=cfg["model_id"], device=cfg["device"])
    batch = int(cfg.get("batch", 8))

    rollouts = load_bank(os.path.join(cfg["out_dir"], "rollouts.h5"))
    stride = int(cfg.get("stride", 4))
    h_values = [int(h) for h in cfg.get("h_values", [4, 8, 16])]
    max_steps = max(r.steps for r in rollouts)
    positions = list(range(0, max_steps + 1, stride))

    # row r of every matrix = traj_ids[r]; column axes below
    arrays = {"traj_ids": np.array([r.name for r in rollouts]),
              "positions": np.array(positions),
              "alive": np.array([[p <= r.steps for p in positions]
                                 for r in rollouts])}
    R, P = len(rollouts), len(positions)

    for q in cfg["questions"]:
        name, text = q["name"], fill(q["text"], cfg)
        if not q.get("pair", False):
            images = [r.frame_at(p) for r in rollouts for p in positions]
            p_yes, mass = score_all(judge, images, text, batch)
            arrays[f"{name}/p"] = p_yes.reshape(R, P)
            arrays[f"{name}/mass"] = mass.reshape(R, P)
            print(f"[{name}] {R * P} single-frame evals done", flush=True)
        else:
            for h in h_values:
                starts = [p for p in positions if p + h <= max_steps]
                images = [(r.frame_at(s), r.frame_at(s + h))
                          for r in rollouts for s in starts]
                p_yes, mass = score_all(judge, images, text, batch)
                arrays[f"{name}/h{h}/p"] = p_yes.reshape(R, len(starts))
                arrays[f"{name}/h{h}/mass"] = mass.reshape(R, len(starts))
                arrays[f"{name}/h{h}/window_starts"] = np.array(starts)
                print(f"[{name}] h={h}: {R * len(starts)} window evals done",
                      flush=True)

    np.savez_compressed(os.path.join(cfg["out_dir"], "scores.npz"), **arrays)
    manifest = dict(
        config=cfg, positions=positions, h_values=h_values,
        window_starts={h: [p for p in positions if p + h <= max_steps]
                       for h in h_values},
        questions=[dict(q, text=fill(q["text"], cfg)) for q in cfg["questions"]],
        rollouts=[dict(name=r.name, seed=r.seed, k=r.k, success=r.success,
                       done_step=r.done_step, steps=r.steps) for r in rollouts])
    with open(os.path.join(cfg["out_dir"], "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"SWEEP_DONE: {cfg['out_dir']}/scores.npz", flush=True)


if __name__ == "__main__":
    main()
