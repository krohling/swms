"""Check behind eval_closed_loop.py --log-cycles: does rolling all k proposals
out in the simulator and putting it back leave the episode untouched?

Simulator and base policy only (no SWM). Four passes over one seed with a
fixed pick rule (proposal cycle % k):
  plain     no rollouts — what an unlogged eval plays
  qposqvel  rollouts, restore by env.get_state()/set_state() (qpos + qvel),
            as verifier-MPC does
  full      rollouts, restore by copying the whole mjData, as --log-cycles does
  rerun     the plain pass a second time: does this host replay an episode
            at all?
and, per cycle, how far each pass is from the plain one. Prints
FULL_RESTORE_BIT_IDENTICAL when the full pass matches the plain one exactly in
proposals, joint positions and rendered frames. When it does not, the rerun
column says whether the rollouts are the cause (exit 1: the rerun is identical)
or the host (exit 2: the rerun differs as well), and the "first difference"
lines say what moved first. The last line counts the distinct frames among 200
renders of one unchanged state.

Usage (from the checkout root; a few seconds per cycle):
  PYTHONPATH=$PWD:$PWD/swm-next python \
    swm-next/swm_lora/check_log_cycles_restore.py \
    --config swm-next/configs/verifier_mpc.yaml --seed 90011 --cycles 20
"""
import argparse
import os
import sys

import mujoco
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "verifier"))

from swm.constants import ANSWER_OPTIONS
from swm.diffusion_policy import DiffusionPolicy
from swm.utils.goal_generators import get_ogbench_goal

from mpc import make_env, OracleProbe  # noqa: E402


def one_pass(cfg, seed, n_cycles, k, temps, mode):
    env = make_env(cfg)
    goal = get_ogbench_goal("stack_blocks", env, None, ANSWER_OPTIONS,
                            {"block_combo": cfg["block_combo"]})
    probe = OracleProbe(env, cfg)
    frame = goal.reset_env(seed=seed)
    goal.reset_hook()
    probe.reset()
    policy = DiffusionPolicy.load(cfg["diffusion_path"], device=cfg["device"])
    policy.add_obs(frame)
    sim = env.env.unwrapped
    snap = mujoco.MjData(sim._model)
    rec = []
    for cycle in range(n_cycles):
        torch.manual_seed(hash((seed, cycle)) % 2**31)
        candidates = np.asarray(policy.sample_trajs(k, temperatures=temps))
        lifts = None
        if mode != "plain":
            if mode == "full":
                mujoco.mj_copyData(snap, sim._model, sim._data)
            else:
                saved = env.get_state()
            lifts = 0
            for c in candidates:
                if mode == "full":
                    mujoco.mj_copyData(sim._data, sim._model, snap)
                else:
                    env.set_state(saved)
                for a in c:
                    env.step(np.asarray(a))
                lifts += probe.read()["lifted"]
            if mode == "full":
                mujoco.mj_copyData(sim._data, sim._model, snap)
            else:
                env.set_state(saved)
        done = False
        for a in candidates[cycle % k][:cfg["actions_per_cycle"]]:
            frame = env.step(np.asarray(a))
            policy.add_obs(frame)
            done = done or bool(goal.get_done())
        rec.append(dict(cands=candidates.copy(), qpos=sim._data.qpos.copy(),
                        frame=np.asarray(frame).astype(int), lifts=lifts,
                        lifted=probe.read()["lifted"], done=done))
    return rec


def render_repeats(cfg, seed, n=200):
    """Distinct frames among n renders of one unchanged simulator state."""
    env = make_env(cfg)
    goal = get_ogbench_goal("stack_blocks", env, None, ANSWER_OPTIONS,
                            {"block_combo": cfg["block_combo"]})
    goal.reset_env(seed=seed)
    sim = env.env.unwrapped
    frames = [np.asarray(sim.get_pixel_observation()).astype(int)
              for _ in range(n)]
    off = [np.abs(f - frames[0]) for f in frames[1:]]
    return (len({f.tobytes() for f in frames}),
            max(int(d.max()) for d in off),
            max(int((d > 0).sum()) for d in off))


def first_diff(plain, other):
    """(cycle, what differs there) of the first cycle that differs, or None."""
    for c, (p, r) in enumerate(zip(plain, other)):
        what = [n for n, f in (("proposals", "cands"), ("joint positions", "qpos"),
                               ("frame", "frame"))
                if not np.array_equal(p[f], r[f])]
        if what:
            px = int((p["frame"] != r["frame"]).sum())
            return c, f"{', '.join(what)} ({px} pixel values differ)"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=90011)
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    temps = [i * 2.0 / (args.k - 1) for i in range(args.k)]
    with torch.no_grad():
        plain, qq, full, rerun = (one_pass(cfg, args.seed, args.cycles, args.k,
                                           temps, m)
                                  for m in ("plain", "qposqvel", "full", "plain"))
        distinct, worst, worst_px = render_repeats(cfg, args.seed)

    print(f"task {'_'.join(cfg['block_combo'])} seed {args.seed}; per cycle, "
          "against the plain pass: same proposals, max |dqpos|, max |dframe|")
    print("cycle | qposqvel restore     | full restore         "
          "| plain rerun          "
          "| proposals lifting (qposqvel / full) | plain: lifted done")
    for c in range(args.cycles):
        cells = []
        for r in (qq[c], full[c], rerun[c]):
            cells.append(
                f"{str(np.array_equal(plain[c]['cands'], r['cands'])):5s} "
                f"{np.abs(plain[c]['qpos'] - r['qpos']).max():.2e} "
                f"{np.abs(plain[c]['frame'] - r['frame']).max():3d}")
        print(f"{c:5d} | {cells[0]} | {cells[1]} | {cells[2]} "
              f"| {qq[c]['lifts']} / {full[c]['lifts']} "
              f"| {int(plain[c]['lifted'])} {int(plain[c]['done'])}")
    d_full, d_rerun = first_diff(plain, full), first_diff(plain, rerun)
    for name, d in (("full restore", d_full), ("plain rerun", d_rerun)):
        print(f"first difference, {name}: "
              + ("none" if d is None else f"cycle {d[0]}: {d[1]}"))
    print(f"200 renders of one state: {distinct} distinct frame(s), "
          f"max |dframe| {worst}, at most {worst_px} pixel values off")
    print("PLAIN_RERUN_BIT_IDENTICAL" if d_rerun is None
          else "PLAIN_RERUN_DIFFERS")
    print("FULL_RESTORE_BIT_IDENTICAL" if d_full is None
          else "FULL_RESTORE_DIFFERS")
    sys.exit(0 if d_full is None else 1 if d_rerun is None else 2)


if __name__ == "__main__":
    main()
