"""Capture base-policy rollouts for the incremental verifier experiment.

For each seed, the base diffusion policy runs K times from an identical start
state (env resets are seed-deterministic; policy stochasticity varies by a
per-(seed, k) torch seed). Every rollout is banked to HDF5 with a frame at
each replan cycle, so verification experiments never re-pay rollout cost and
can score trajectories at any horizon.

Bank layout (one HDF5 group per rollout, named "s{seed}_k{k}"):

    cycle_frames  (C+1, H, W, 3) uint8 -- frame 0 is pre-action, then one
                                          frame per replan cycle (4 actions)
    state_start / state_end             -- pickled sim states
    attrs: seed, k, success, cycles, done_cycle (-1 if never), instruction

Rollouts terminate at oracle success, so a successful rollout has fewer
frames; consumers should treat its last frame as persisting afterward.

Usage (repo root on PYTHONPATH):

    PYTHONPATH=$PWD python swm-next/verifier/capture.py --config swm-next/configs/verifier.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass, field

import h5py
import numpy as np
import torch
import yaml

from swm.constants import ANSWER_OPTIONS
from swm.diffusion_policy import DiffusionPolicy
from swm.utils.envs import get_ogbench_env
from swm.utils.goal_generators import get_ogbench_goal


@dataclass
class Episode:
    seed: int
    k: int
    frames: list = field(default_factory=list)   # one per cycle, plus frame 0
    states: list = field(default_factory=list)   # sim state at start and end
    done_cycle: int = -1                         # first cycle with oracle done

    @property
    def success(self) -> bool:
        return self.done_cycle != -1

    @property
    def cycles(self) -> int:
        return len(self.frames) - 1


def run_episode(env, goal, cfg, seed: int, k: int) -> Episode:
    """One closed-loop rollout: replan every cycle, execute 4 actions."""
    torch.manual_seed(hash((seed, k)) % 2**31)
    np.random.seed(hash((k, seed)) % 2**31)

    frame = goal.reset_env(seed=seed)
    goal.reset_hook()
    policy = DiffusionPolicy.load(cfg["diffusion_path"], device=cfg["device"])
    policy.add_obs(frame)

    ep = Episode(seed=seed, k=k)
    ep.frames.append(np.asarray(frame, dtype=np.uint8))
    ep.states.append(env.get_state())

    for cycle in range(1, cfg["max_cycles"] + 1):
        done = False
        for action in policy.get_action()[: cfg["actions_per_cycle"]]:
            frame = env.step(action)
            policy.add_obs(frame)
            if goal.get_done():
                done = True
                break
        ep.frames.append(np.asarray(frame, dtype=np.uint8))
        if done:
            ep.done_cycle = cycle
            break

    ep.states.append(env.get_state())
    return ep


def save_episode(bank: h5py.File, ep: Episode, instruction: str):
    g = bank.create_group(f"s{ep.seed}_k{ep.k}")
    # gzip: successive cycle frames are near-duplicates, ~5x smaller.
    g.create_dataset("cycle_frames", data=np.stack(ep.frames),
                     compression="gzip", compression_opts=1)
    g.create_dataset("state_start",
                     data=np.frombuffer(pickle.dumps(ep.states[0]), dtype=np.uint8))
    g.create_dataset("state_end",
                     data=np.frombuffer(pickle.dumps(ep.states[-1]), dtype=np.uint8))
    g.attrs.update(seed=ep.seed, k=ep.k, success=ep.success, cycles=ep.cycles,
                   done_cycle=ep.done_cycle, instruction=instruction)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    import ogbench  # noqa: F401  (env registration)
    env = get_ogbench_env({"ood": False})
    goal = get_ogbench_goal("stack_blocks", env, None, ANSWER_OPTIONS,
                            {"block_combo": cfg["block_combo"]})
    top, bottom = [b.replace("_", " ") for b in cfg["block_combo"]]
    instruction = f"Stack the {top} on top of the {bottom}"

    os.makedirs(cfg["out_dir"], exist_ok=True)
    path = os.path.join(cfg["out_dir"], "rollouts.h5")
    successes = total = 0

    with h5py.File(path, "w") as bank:
        bank.attrs["block_combo"] = json.dumps(cfg["block_combo"])
        for seed in range(cfg["seed_start"], cfg["seed_start"] + cfg["num_seeds"]):
            for k in range(cfg["k_rollouts"]):
                ep = run_episode(env, goal, cfg, seed, k)
                save_episode(bank, ep, instruction)
                successes += ep.success
                total += 1
            print(f"seed {seed}: {total} rollouts banked "
                  f"(success rate {successes / total:.1%})", flush=True)

    print(f"CAPTURE_DONE: {path}", flush=True)


if __name__ == "__main__":
    main()
