"""SWM training-data generation: teacher-labeled candidate proposals from
verifier-MPC planning steps (sim as perfect world model).

Per planning step (cycle) this records:
  - frame_t                (committed env frame, JPEG)
  - per proposal i of k:   trajectory (16 x act_dim), end frame (JPEG),
                           p_approach, p_place, p_hold  (teacher p_yes on the
                           proposal's rollout end frame)
  - pick                   index of the executed proposal
Episode-level: seed, selection policy ('verifier' | 'random'), success, cycles.

Selection policy controls WHICH proposal is executed (state visitation):
  verifier: softgate_ema score from the labels themselves (gate = EMA of
            p_yes(frame_t, grasp), alpha 0.35) — deployment-like states.
  random:   uniform pick — adds failure/recovery states the verifier avoids.
Labels are identical in kind for both (teacher labels all k proposals).

Default questions (task blue->green, auto-filled from block_combo):
  approach = hold = "Is the robot grasping the {top}?"
  place    = "Is the {top} on top of the {bottom}?"
(hold duplicates approach by construction in the default softgate config; the
field is stored anyway so the schema is stable if the hold question changes.)

Output: one HDF5 per run, episodes as groups:
  ep_<seed>/frames_t   (C,)    vlen uint8 (JPEG)
  ep_<seed>/trajs      (C,k,H,A) f32
  ep_<seed>/end_frames (C,k)  vlen uint8 (JPEG)
  ep_<seed>/labels     (C,k,3) f32   [p_approach, p_place, p_hold]
  ep_<seed>/pick       (C,)   i8
  attrs: seed, policy, success, cycles, questions, temps, k, lookahead

Usage:
  PYTHONPATH=$PWD:$PWD/swm-next python swm-next/verifier/datagen.py \
    --config swm-next/configs/verifier_mpc.yaml --episodes 200 \
    --seed-start 10000 --frac-random 0.3 --out datasets_swm/bg_k16_L16.h5
Seeds: evaluation uses 6000-6099 — generation MUST use a disjoint range.
"""
import argparse
import io
import os

import h5py
import numpy as np
import torch
import yaml
from PIL import Image

from swm.constants import ANSWER_OPTIONS
from swm.diffusion_policy import DiffusionPolicy
from swm.utils.goal_generators import get_ogbench_goal

from mpc import make_env, build_questions  # noqa: E402  (same dir via PYTHONPATH)

GATE_ALPHA = 0.35
PLACE_W, HOLD_W = 0.6, 0.4


def jpeg(frame, quality=90):
    buf = io.BytesIO()
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
        buf, format="JPEG", quality=quality)
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


def ask(judge, images, question, batch):
    out = np.zeros(len(images), dtype=np.float32)
    for i in range(0, len(images), batch):
        p, _ = judge.p_yes(images[i:i + batch],
                           [question] * len(images[i:i + batch]))
        out[i:i + len(images[i:i + batch])] = p
    return out


def run_episode(h5, env, goal, judge, policy_path, cfg, questions, seed,
                policy_name, k, temps, lookahead, batch, exec_temp_max=None):
    torch.manual_seed(hash((seed, "mpc")) % 2**31)
    np.random.seed(hash(("mpc", seed)) % 2**31)
    rng = np.random.RandomState(seed)          # selection RNG (random policy)
    temp_rng = np.random.RandomState(hash((seed, "exectemp")) % 2**31)

    frame = goal.reset_env(seed=seed)
    goal.reset_hook()
    policy = DiffusionPolicy.load(policy_path, device=cfg["device"])
    policy.add_obs(frame)

    n_exec = cfg["actions_per_cycle"]
    q_app, q_place, q_hold = (questions["grasp"], questions["ontop"],
                              questions["grasp"])
    gate_ema = None
    rec = dict(frames_t=[], trajs=[], end_frames=[], labels=[], pick=[])
    done = False
    cycle = -1
    exec_temps = []
    for cycle in range(cfg["max_cycles"]):
        torch.manual_seed(hash((seed, cycle)) % 2**31)
        if exec_temp_max is not None:
            # noisy-behavior capture (robot-compatible): the executed chunk is
            # sampled at a fresh uniform temperatureevery cycle
            temps = [float(temp_rng.uniform(0.0, exec_temp_max))]
        exec_temps.append(list(temps))
        candidates = np.asarray(policy.sample_trajs(k, temperatures=temps))

        frame_t = np.asarray(frame, dtype=np.uint8)
        saved = env.get_state()
        end_frames = []
        for c in candidates:
            env.set_state(saved)
            f = frame
            for a in c[:lookahead]:
                f = env.step(np.asarray(a))
            end_frames.append(np.asarray(f, dtype=np.uint8))
        env.set_state(saved)

        # teacher labels for every proposal (hold == approach question by
        # default; asked once, stored twice for schema stability)
        p_app = ask(judge, end_frames, q_app, batch)
        p_place = ask(judge, end_frames, q_place, batch)
        p_hold = p_app if q_hold == q_app else ask(judge, end_frames, q_hold, batch)

        if policy_name == "verifier":
            rg = float(judge.p_yes([frame_t], [questions["grasp"]])[0][0])
            gate_ema = rg if gate_ema is None else \
                GATE_ALPHA * rg + (1 - GATE_ALPHA) * gate_ema
            g = gate_ema
            scores = (1.0 - g) * p_app + g * (PLACE_W * p_place + HOLD_W * p_hold)
            pick = int(np.argmax(scores))
        else:
            pick = int(rng.randint(k))

        rec["frames_t"].append(jpeg(frame_t))
        rec["trajs"].append(np.asarray(candidates, dtype=np.float32))
        rec["end_frames"].append([jpeg(f) for f in end_frames])
        rec["labels"].append(
            np.stack([p_app, p_place, p_hold], axis=-1).astype(np.float32))
        rec["pick"].append(pick)

        for a in candidates[pick][:n_exec]:
            frame = env.step(np.asarray(a))
            policy.add_obs(frame)
            if goal.get_done():
                done = True
                break
        if done:
            break

    C = len(rec["pick"])
    grp = h5.create_group(f"ep_{seed}")
    vlen = h5py.vlen_dtype(np.uint8)
    d = grp.create_dataset("frames_t", (C,), dtype=vlen)
    for i, v in enumerate(rec["frames_t"]):
        d[i] = v
    grp.create_dataset("trajs", data=np.stack(rec["trajs"]))
    e = grp.create_dataset("end_frames", (C, k), dtype=vlen)
    for i, row in enumerate(rec["end_frames"]):
        for j, v in enumerate(row):
            e[i, j] = v
    grp.create_dataset("labels", data=np.stack(rec["labels"]))
    grp.create_dataset("pick", data=np.asarray(rec["pick"], dtype=np.int8))
    grp.create_dataset("exec_temps", data=np.asarray(exec_temps, dtype=np.float32))
    grp.attrs.update(dict(
        seed=seed, policy=policy_name, success=bool(done),
        cycles=cycle + 1,
        q_approach=q_app, q_place=q_place, q_hold=q_hold))
    h5.flush()
    return done, cycle + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed-start", type=int, default=10000)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lookahead", type=int, default=16)
    ap.add_argument("--frac-random", type=float, default=0.3,
                    help="fraction of episodes advanced by random selection")
    ap.add_argument("--exec-temp-max", type=float, default=None,
                    dest="exec_temp_max",
                    help="k=1 noisy-behavior capture: executed chunk sampled "
                         "at t ~ U(0, this) per cycle (robot-compatible "
                         "diversity; records exec_temps)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    assert not (6000 <= args.seed_start <= 6100), \
        "seeds 6000-6099 are reserved for evaluation"
    temps = [i * 2.0 / (args.k - 1) for i in range(args.k)] if args.k > 1 else [0.0]
    batch = int(cfg.get("batch", 8))

    from label_teacher import QwenJudge
    judge = QwenJudge(model_id=cfg["model_id"], device=cfg["device"])
    env = make_env(cfg)
    goal = get_ogbench_goal("stack_blocks", env, None, ANSWER_OPTIONS,
                            {"block_combo": cfg["block_combo"]})
    questions = build_questions(cfg)

    # interleave policies deterministically: every round(1/frac)-th is random
    n_rand = int(round(args.episodes * args.frac_random))
    flags = (["random"] * n_rand + ["verifier"] * (args.episodes - n_rand))
    np.random.RandomState(0).shuffle(flags)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    mode = "a" if os.path.exists(args.out) else "w"
    with h5py.File(args.out, mode) as h5:
        h5.attrs.update(dict(
            task="_".join(cfg["block_combo"]), k=args.k,
            lookahead=args.lookahead, temps=temps,
            frac_random=args.frac_random, model_id=cfg["model_id"]))
        wins = 0
        for i in range(args.episodes):
            seed = args.seed_start + i
            if f"ep_{seed}" in h5:
                print(f"[{i+1}/{args.episodes}] ep_{seed} exists, skip", flush=True)
                continue
            success, cycles = run_episode(
                h5, env, goal, judge, cfg["diffusion_path"], cfg, questions,
                seed, flags[i], args.k, temps, args.lookahead, batch,
                exec_temp_max=args.exec_temp_max)
            wins += success
            print(f"[{i+1}/{args.episodes}] seed {seed} ({flags[i]}): "
                  f"{'success' if success else 'fail'} @ {cycles} "
                  f"(SR {wins/(i+1):.0%})", flush=True)
    print(f"DATAGEN_DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
