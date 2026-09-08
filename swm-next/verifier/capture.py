"""Stage 1: capture base-policy rollouts for the incremental verifier.

For each seed the base diffusion policy runs K times from an identical start
state (env resets are seed-deterministic; policy sampling varies by a
per-(seed, k) torch seed). Every rollout is banked with a frame and oracle
predicates at EVERY env step, so downstream evals can score any frame pair
at any horizon or stride without touching the simulator again.

Bank layout (one HDF5 group per rollout, named by traj_id "s{seed}_k{k}"):

    frames      (S+1, H, W, 3) uint8  -- frame 0 is pre-action, then one per step
    gripper_contact (S+1,) bool       -- gripper pads touching top cube
    ontop           (S+1,) bool       -- top cube resting on bottom cube
    done            (S+1,) bool       -- oracle success (ontop and released)
    state_start / state_end           -- pickled sim states
    attrs: traj_id (= group name), seed, k, success, steps,
           done_step (-1 if never), instruction

Rollouts terminate at oracle success, so successful rollouts are shorter;
consumers should treat the last frame as persisting afterward.

`capture_workers` in the config parallelizes across rollouts (each worker
process owns its own env + policy; results stream back to one HDF5 writer).
Rollouts are seeded per (seed, k), so worker count never changes the data.

Usage (repo root on PYTHONPATH):

    PYTHONPATH=$PWD python swm-next/verifier/capture.py --config swm-next/configs/verifier.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import h5py
import numpy as np
import torch
import yaml

from swm.constants import ANSWER_OPTIONS
from swm.diffusion_policy import DiffusionPolicy
from swm.utils.envs import OGBenchEnv
from swm.utils.goal_generators import get_ogbench_goal

PAD_GEOMS = ("ur5e/robotiq/right_pad1", "ur5e/robotiq/right_pad2",
             "ur5e/robotiq/left_pad1", "ur5e/robotiq/left_pad2")


ARM_MATERIALS = ("ur5e/robotiq/metal", "ur5e/robotiq/silicone",
                 "ur5e/robotiq/gray", "ur5e/robotiq/black",
                 "ur5e/black", "ur5e/jointgray", "ur5e/linkgray",
                 "ur5e/lightblue")
PAD_MATERIAL = "ur5e/robotiq/pad_gray"


def make_env(cfg):
    """swm.utils.envs.get_ogbench_env with one addition: arm_alpha.

    OGBench's boolean pixel_transparent_arm renders the arm at alpha 0.1
    (gripper pads 0.5). Here the env is built opaque and the config's
    arm_alpha is written onto the compiled model's arm materials directly,
    so any value in (0, 1] works. Pads get min(1, 5 * arm_alpha), matching
    OGBench's 5x-more-visible-pads ratio: arm_alpha 0.1 reproduces the
    default rendering exactly; 1.0 is fully opaque. Local mirror because
    core swm code stays untouched."""
    import gymnasium
    import mujoco
    import ogbench  # noqa: F401  (env registration)
    env = gymnasium.make(
        "visual-cube-quadruple-v0",
        terminate_at_goal=False,
        visualize_info=False,
        mode="data_collection",
        max_episode_steps=1000,
        width=224,
        height=224,
        control_timestep=0.1,
        stack_goal=None,
        ood=False,
        pixel_transparent_arm=False,
    )
    alpha = float(cfg.get("arm_alpha", 0.1))
    model = env.unwrapped._model
    for name in ARM_MATERIALS:
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
        if mid >= 0:
            model.mat_rgba[mid, 3] = alpha
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, PAD_MATERIAL)
    if pid >= 0:
        model.mat_rgba[pid, 3] = min(1.0, 5 * alpha)
    return OGBenchEnv(env)


class OraclePredicates:
    """Per-step ground-truth predicates.

    MIRROR of StackBlocksGoal.get_done (swm/utils/goal_generators.py) with the
    intermediate predicates exposed; core swm code stays untouched by design
    (Kevin, 2026-09-07). If get_done's logic ever changes, update this to
    match. Sanity guard: capture asserts read()["done"] == goal.get_done()
    at every step, so any drift fails loudly instead of corrupting a bank.

    Reads goal.env (the gymnasium env the goal generator itself uses); geom
    ids resolve lazily on first read, after the env exists post-reset.
    """

    def __init__(self, goal):
        self.goal = goal
        self._ids = None

    def _bind(self):
        import mujoco
        u = self.goal.env.unwrapped
        top_num = self.goal.block_to_number[self.goal.block_combo[0]]
        bottom_num = self.goal.block_to_number[self.goal.block_combo[1]]
        self._ids = dict(
            top=u._cube_geom_ids_list[top_num][0],
            bottom=u._cube_geom_ids_list[bottom_num][0],
            pads=[mujoco.mj_name2id(u._model, mujoco.mjtObj.mjOBJ_GEOM, g)
                  for g in PAD_GEOMS])

    def read(self) -> dict:
        if self._ids is None:
            self._bind()
        ids, d = self._ids, self.goal.env.unwrapped._data
        contacts = d.contact.geom
        pair = np.array([ids["top"], ids["bottom"]])
        ontop = False
        if (np.any(np.all(pair == contacts, axis=1))
                or np.any(np.all(pair[::-1] == contacts, axis=1))):
            dz = d.geom(ids["top"]).xpos[2] - d.geom(ids["bottom"]).xpos[2]
            ontop = dz > 0.015
        gripper_contact = False
        for i in range(d.ncon):
            g = d.contact[i].geom
            if ids["top"] in g and any(p in g for p in ids["pads"]):
                gripper_contact = True
                break
        return dict(ontop=ontop, gripper_contact=gripper_contact,
                    done=ontop and not gripper_contact)


@dataclass
class Episode:
    seed: int
    k: int
    frames: list = field(default_factory=list)
    preds: list = field(default_factory=list)    # dicts from OraclePredicates.read()
    states: list = field(default_factory=list)   # sim state at start and end
    done_step: int = -1

    @property
    def success(self) -> bool:
        return self.done_step != -1

    @property
    def steps(self) -> int:
        return len(self.frames) - 1


def run_episode(env, goal, oracle, cfg, seed: int, k: int) -> Episode:
    """One closed-loop rollout: replan every cycle, execute 4 actions."""
    torch.manual_seed(hash((seed, k)) % 2**31)
    np.random.seed(hash((k, seed)) % 2**31)

    frame = goal.reset_env(seed=seed)
    goal.reset_hook()
    policy = DiffusionPolicy.load(cfg["diffusion_path"], device=cfg["device"])
    policy.add_obs(frame)

    ep = Episode(seed=seed, k=k)
    ep.frames.append(np.asarray(frame, dtype=np.uint8))
    ep.preds.append(oracle.read())
    ep.states.append(env.get_state())

    step = 0
    for _ in range(cfg["max_cycles"]):
        for action in policy.get_action()[: cfg["actions_per_cycle"]]:
            frame = env.step(action)
            policy.add_obs(frame)
            step += 1
            ep.frames.append(np.asarray(frame, dtype=np.uint8))
            preds = oracle.read()
            ep.preds.append(preds)
            assert preds["done"] == goal.get_done(), \
                "oracle mirror drifted from StackBlocksGoal.get_done"
            if preds["done"]:
                ep.done_step = step
                break
        if ep.success:
            break

    ep.states.append(env.get_state())
    return ep


def save_episode(bank: h5py.File, ep: Episode, instruction: str):
    traj_id = f"s{ep.seed}_k{ep.k}"
    g = bank.create_group(traj_id)
    # gzip: successive frames are near-duplicates, ~5x smaller.
    g.create_dataset("frames", data=np.stack(ep.frames),
                     compression="gzip", compression_opts=1)
    for key in ("gripper_contact", "ontop", "done"):
        g.create_dataset(key, data=np.array([p[key] for p in ep.preds], dtype=bool))
    g.create_dataset("state_start",
                     data=np.frombuffer(pickle.dumps(ep.states[0]), dtype=np.uint8))
    g.create_dataset("state_end",
                     data=np.frombuffer(pickle.dumps(ep.states[-1]), dtype=np.uint8))
    g.attrs.update(traj_id=traj_id, seed=ep.seed, k=ep.k, success=ep.success,
                   steps=ep.steps, done_step=ep.done_step, instruction=instruction)


# Per-worker sim stack, built once per process (see _init_worker).
_W: dict = {}


def _init_worker(cfg):
    env = make_env(cfg)
    goal = get_ogbench_goal("stack_blocks", env, None, ANSWER_OPTIONS,
                            {"block_combo": cfg["block_combo"]})
    _W.update(env=env, goal=goal, oracle=OraclePredicates(goal), cfg=cfg)


def _capture_one(job):
    seed, k = job
    return run_episode(_W["env"], _W["goal"], _W["oracle"], _W["cfg"], seed, k)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    top, bottom = [b.replace("_", " ") for b in cfg["block_combo"]]
    instruction = f"Stack the {top} on top of the {bottom}"
    jobs = [(seed, k)
            for seed in range(cfg["seed_start"], cfg["seed_start"] + cfg["num_seeds"])
            for k in range(cfg["k_rollouts"])]
    workers = int(cfg.get("capture_workers", 1))

    os.makedirs(cfg["out_dir"], exist_ok=True)
    path = os.path.join(cfg["out_dir"], "rollouts.h5")
    successes = total = 0

    with h5py.File(path, "w") as bank:
        bank.attrs["block_combo"] = json.dumps(cfg["block_combo"])
        bank.attrs["actions_per_cycle"] = cfg["actions_per_cycle"]

        if workers == 1:
            _init_worker(cfg)
            episodes = map(_capture_one, jobs)
            self_desc = "sequential"
        else:
            pool = ProcessPoolExecutor(max_workers=workers,
                                       initializer=_init_worker, initargs=(cfg,))
            episodes = pool.map(_capture_one, jobs)
            self_desc = f"{workers} workers"

        for ep in episodes:
            save_episode(bank, ep, instruction)
            successes += ep.success
            total += 1
            print(f"[{total}/{len(jobs)}] s{ep.seed}_k{ep.k}: "
                  f"{'success' if ep.success else 'fail'} @ {ep.steps} steps "
                  f"(running SR {successes / total:.1%}, {self_desc})", flush=True)

        if workers > 1:
            pool.shutdown()

    print(f"CAPTURE_DONE: {path}", flush=True)


if __name__ == "__main__":
    main()
