"""LangTable k=1 pointwise dataset generator: the v3b recipe of datagen_v3.py
ported to the LangTable push tasks (env_type: lang_table).

Per decision cycle the task's base diffusion policy proposes k chunks on the
eval's temperature ladder, ONE is picked uniformly at random, rolled out
`lookahead` steps from a saved simulator state, and the end state is labeled
by the Qwen judge with the teacher's two push questions
(mpc.build_lt_questions):

    "Is the X touching the Y?"            on the end frame            (0.8)
    "Are the X and Y closer together?"    on the (frame_t, end) pair  (0.2)

The simulator is then rewound and the first `actions_per_cycle` actions of
the same chunk are executed, as in the eval. One labeled trajectory per state
(k=1 in the stored labels); the other k-1 proposals are stored unlabeled.
Nothing here needs more than one rollout per state, so the recipe ports to a
real robot the way v3b does.

LangTable is single-phase (no grasp): there is no gate. The teacher's value
is score = 0.8 * p_touch + 0.2 * p_closer.

Rollouts use the Phase-1 validated restore (test_lt_set_state.py):
deterministicOverlappingPairs=1 plus set_state THEN restore_state_native.
Rollout steps skip the camera render (lookahead <= chunk length, so no
frame of the rollout is read by the policy); the judge gets its own
judge-scale render of the end state, exactly as the teacher does.

Schema (schema_version 3, variant "lt1") — the v3 layout, so the trainer's
loaders (swm_lora/data.py: frames_t, trajs, labels columns 0/1 with the
episode attrs q_approach / q_place, attr policy) read it unchanged:

  ep_<seed>/
    frames_t (C,) JPEG     judge-scale render of state t (the teacher's
                           anchor frame; the model's input frame)
    trajs (C,1,H,A)        the picked chunk
    end_frames (C,1) JPEG  judge-scale render of the rollout end state
    labels (C,1,3)         [p_touch, p_closer, p_touch] = the v3 columns
                           approach / place / hold (hold repeats approach,
                           as in v3 when q_hold == q_approach)
    label_mass (C,1,2)     judge's yes+no probability mass, both questions
    score (C,1)            0.8 * p_touch + 0.2 * p_closer
    pick (C,), propensity (C,) = 1/k
    proposal_trajs (C,k,H,A), proposal_temps (C,k), proposal_min_sep (C,)
    oracle_t (C,8), oracle_end (C,1,8)   simulator truth (never the judge):
                           [b1x, b1y, b2x, b2y, pegx, pegy, dist, dist_peg_b1]
    end_done (C,1)         the task's done test at the rollout end state
    state_t_pkl (C,), end_state_pkl (C,1)   zlib(pickle(env.get_state()))
                           (restorable with the same simulator checkout)
    attrs: seed, policy="random", arm, success, cycles, final_dist,
           q_touch, q_closer, q_approach (= q_touch), q_place (= q_closer),
           q_hold (= q_touch), block1, block2, secs_* (wall time split)

Frames are stored JPEG q90 while the labels come from the raw renders, as in
v3. A closed-loop eval of a model trained on these files should put its
input frame through the same JPEG round trip.

Usage (one task, one unit of episodes; the launcher runs many of these):
  PYTHONPATH=<language-table>:.:swm-next:swm-next/verifier \
  python swm-next/verifier/datagen_lt.py \
      --config swm-next/configs/verifier_mpc_lt_green_cube_blue_moon.yaml \
      --set device=cuda --set judge_url=http://127.0.0.1:8902 \
      --episodes 27 --seed-start 200000 --out <unit>.h5
"""
import argparse
import json
import os
import pickle
import time
import zlib

import h5py
import numpy as np
import torch
import yaml

from swm.constants import ANSWER_OPTIONS
from swm.diffusion_policy import DiffusionPolicy

from datagen import jpeg  # noqa: E402  (same dir via PYTHONPATH)
from mpc import build_lt_questions, make_lt_env  # noqa: E402

# 6000-6049 and 6100-6249 are the n=200 eval blocks, 6050-6099 is held out
EVAL_SEEDS = range(6000, 6250)
STATE_TOL = 1e-6           # restore check, as in test_lt_set_state.py


def ladder(k):
    assert k > 1, "the ladder needs k > 1 (k = 1 would be tau = 0 only)"
    return [i * 2.0 / (k - 1) for i in range(k)]


def oracle_vec(env, block1, block2):
    """mpc.LTProbe's quantities, unrounded (float64):
    [b1x, b1y, b2x, b2y, pegx, pegy, dist(b1, b2), dist(peg, b1)]."""
    st = env.env.get_block_states()

    def get(name):
        key = name if name in st else name.replace("_", " ")
        return np.asarray(st[key], dtype=np.float64)

    b1, b2 = get(block1), get(block2)
    peg = np.asarray(st["peg"], dtype=np.float64)
    return np.asarray([*b1, *b2, *peg, np.linalg.norm(b1 - b2),
                       np.linalg.norm(peg - b1)], dtype=np.float64)


def pack_state(state):
    return np.frombuffer(zlib.compress(pickle.dumps(state, protocol=4), 6),
                         dtype=np.uint8)


def unpack_state(blob):
    return pickle.loads(zlib.decompress(np.asarray(blob, np.uint8).tobytes()))


def run_episode(h5, env, goal, judge, cfg, seed, arm, k, temps, lookahead,
                check_cycles):
    torch.manual_seed((seed * 7919 + 17) % 2**31)
    np.random.seed((seed * 7919 + 29) % 2**31)
    rng = np.random.RandomState(seed)          # pick RNG (as in v3)
    # In-place reseed: the goal's reward function shares this RandomState.
    # LT's reset_env ignores its seed arg; the scene layout comes from it.
    env.env._rng.seed(seed)
    frame = goal.reset_env(seed=seed)
    policy = DiffusionPolicy.load(cfg["diffusion_path"], device=cfg["device"])
    policy.add_obs(frame)
    (q_touch, w_touch, _), (q_closer, w_closer, _) = build_lt_questions(goal)
    client = env.env._pybullet_client
    b1, b2 = goal.info.block1, goal.info.block2
    n_exec = cfg["actions_per_cycle"]
    ns = int(cfg.get("judge_native_scale", 1))

    rec = dict(frames_t=[], trajs=[], end_frames=[], labels=[], mass=[],
               score=[], pick=[], proposal_trajs=[], min_sep=[], oracle_t=[],
               oracle_end=[], end_done=[], state_t=[], end_state=[])
    secs = dict(policy=0.0, sim=0.0, judge=0.0, execute=0.0)
    check = dict(cycles=0, bad=0, max_state=0.0, max_px=0)
    done = False
    cycle = -1
    for cycle in range(cfg["max_cycles"]):
        torch.manual_seed(hash((seed, cycle)) % 2**31)
        t0 = time.perf_counter()
        candidates = np.asarray(policy.sample_trajs(k, temperatures=temps),
                                dtype=np.float32)
        secs["policy"] += time.perf_counter() - t0
        assert candidates.shape[1] >= lookahead, \
            (f"lookahead {lookahead} is longer than the policy's chunk "
             f"({candidates.shape[1]}): that needs the teacher's closed-loop "
             f"rollout, which this generator does not do")
        # pick FIRST: only the executed candidate is ever rolled out or labeled
        pick = int(rng.randint(k))
        chosen = candidates[pick]
        dd = np.sqrt(((candidates[:, None] - candidates[None, :]) ** 2
                      ).sum(-1)).mean(-1)
        np.fill_diagonal(dd, np.inf)
        rec["min_sep"].append(float(dd.min()))

        t0 = time.perf_counter()
        S = env.get_state()
        env.set_state(S)
        sid = env.save_state_native()

        def restore():
            env.set_state(S)                # python-side pose + ObjState
            env.restore_state_native(sid)   # engine snapshot

        anchor = env.get_judge_frame(ns)
        o_t = oracle_vec(env, b1, b2)

        def rollout():
            restore()
            for a in chosen[:lookahead]:
                env.step_norender(np.asarray(a))
            return (env.get_judge_frame(ns), oracle_vec(env, b1, b2),
                    bool(goal.get_done()), env.get_state())

        end_frame, o_end, end_done, S_end = rollout()
        if cycle < check_cycles:
            # the determinism gate on a real state of this episode
            f2, o2, _, _ = rollout()
            px = np.abs(end_frame.astype(np.int16) - f2.astype(np.int16))
            ds = float(np.abs(o_end - o2).max())
            ok = ds <= STATE_TOL and (
                px.max() == 0 or (px.max() <= 2 and (px > 0).sum() <= 200))
            check["cycles"] += 1
            check["bad"] += 0 if ok else 1
            check["max_state"] = max(check["max_state"], ds)
            check["max_px"] = max(check["max_px"], int(px.max()))
        restore()
        client.removeState(sid)
        secs["sim"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        if judge is None:                   # --no-judge dry run
            p_touch = p_closer = m_touch = m_closer = float("nan")
        else:
            # two single-item forwards, so no sequence in a batch is padded
            p, m = judge.p_yes([end_frame], [q_touch])
            p_touch, m_touch = float(p[0]), float(m[0])
            p, m = judge.p_yes([(anchor, end_frame)], [q_closer])
            p_closer, m_closer = float(p[0]), float(m[0])
        secs["judge"] += time.perf_counter() - t0

        rec["frames_t"].append(jpeg(anchor))
        rec["trajs"].append(chosen[None])
        rec["end_frames"].append(jpeg(end_frame))
        rec["labels"].append([[p_touch, p_closer, p_touch]])
        rec["mass"].append([[m_touch, m_closer]])
        rec["score"].append([w_touch * p_touch + w_closer * p_closer])
        rec["pick"].append(pick)
        rec["proposal_trajs"].append(candidates)
        rec["oracle_t"].append(o_t.astype(np.float32))
        rec["oracle_end"].append(o_end.astype(np.float32)[None])
        rec["end_done"].append([end_done])
        rec["state_t"].append(pack_state(S))
        rec["end_state"].append(pack_state(S_end))

        t0 = time.perf_counter()
        for a in chosen[:n_exec]:
            frame = env.step(np.asarray(a))
            policy.add_obs(frame)
            if goal.get_done():
                done = True
                break
        secs["execute"] += time.perf_counter() - t0
        if done:
            break

    C = len(rec["pick"])
    final_dist = float(oracle_vec(env, b1, b2)[6])
    # written under a temporary name: a run killed here leaves no ep_<seed>
    grp = h5.create_group(f"_tmp_ep_{seed}")
    vlen = h5py.vlen_dtype(np.uint8)
    for name, key in (("frames_t", "frames_t"), ("state_t_pkl", "state_t")):
        d = grp.create_dataset(name, (C,), dtype=vlen)
        for i, v in enumerate(rec[key]):
            d[i] = v
    for name, key in (("end_frames", "end_frames"),
                      ("end_state_pkl", "end_state")):
        d = grp.create_dataset(name, (C, 1), dtype=vlen)
        for i, v in enumerate(rec[key]):
            d[i, 0] = v
    grp.create_dataset("trajs", data=np.stack(rec["trajs"]))
    grp.create_dataset("labels", data=np.asarray(rec["labels"], np.float32))
    grp.create_dataset("label_mass", data=np.asarray(rec["mass"], np.float32))
    grp.create_dataset("score", data=np.asarray(rec["score"], np.float32))
    grp.create_dataset("pick", data=np.asarray(rec["pick"], dtype=np.int8))
    grp.create_dataset("propensity", data=np.full(C, 1.0 / k, np.float32))
    grp.create_dataset("proposal_trajs", data=np.stack(rec["proposal_trajs"]))
    grp.create_dataset("proposal_temps", data=np.tile(
        np.asarray(temps, np.float32), (C, 1)))
    grp.create_dataset("proposal_min_sep",
                       data=np.asarray(rec["min_sep"], dtype=np.float32))
    grp.create_dataset("oracle_t", data=np.stack(rec["oracle_t"]))
    grp.create_dataset("oracle_end", data=np.stack(rec["oracle_end"]))
    grp.create_dataset("end_done", data=np.asarray(rec["end_done"], bool))
    grp.attrs.update(dict(
        seed=seed, policy="random", arm=arm, success=bool(done), cycles=C,
        final_dist=final_dist,
        q_touch=q_touch, q_closer=q_closer,
        q_approach=q_touch, q_place=q_closer, q_hold=q_touch,
        block1=str(b1), block2=str(b2),
        **{f"secs_{n}": v for n, v in secs.items()},
        **{f"check_{n}": v for n, v in check.items()}))
    h5.move(f"_tmp_ep_{seed}", f"ep_{seed}")
    h5.flush()
    return done, C, secs, check


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--k", type=int, default=8,
                    help="proposals per state on the temperature ladder")
    ap.add_argument("--lookahead", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-judge", action="store_true", dest="no_judge",
                    help="dry run: labels are NaN and the file is marked "
                         "dry_run (the verifier rejects it)")
    ap.add_argument("--restore-check-cycles", type=int, default=0,
                    dest="check_cycles",
                    help="repeat the rollout of the first N cycles of every "
                         "episode and compare (the determinism gate on real "
                         "states); a mismatch ends the run")
    ap.add_argument("--code-state", default="", dest="code_state",
                    help="free text stored in the file attrs (commits)")
    ap.add_argument("--set", action="append", default=[], dest="overrides",
                    metavar="KEY=VALUE",
                    help="override a config key (yaml-parsed value)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    for kv in args.overrides:
        key, val = kv.split("=", 1)
        cfg[key] = yaml.safe_load(val)
    assert cfg.get("env_type") == "lang_table", "LangTable configs only"
    seeds = range(args.seed_start, args.seed_start + args.episodes)
    assert not set(seeds) & set(EVAL_SEEDS), \
        "seed range overlaps 6000-6249 (reserved for evaluation)"
    temps = ladder(args.k)
    arm = f"lt_v3_k{args.k}_ladder_uniform"

    import sys
    sys.dont_write_bytecode = True
    judge = None
    if not args.no_judge:
        if cfg.get("judge_url"):
            from http_judge import HTTPJudge
            judge = HTTPJudge(cfg["judge_url"],
                              upscale=int(cfg.get("judge_upscale", 1)))
        else:
            from label_teacher import QwenJudge
            judge = QwenJudge(model_id=cfg["model_id"], device=cfg["device"])

    from swm.utils.goal_generators import get_lang_table_goal
    env = make_lt_env(cfg)
    goal = get_lang_table_goal("block_to_block", env, None, ANSWER_OPTIONS,
                               {"block_combo": list(cfg["block_combo"]),
                                "ood": bool(cfg.get("ood", False))})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    mode = "a" if os.path.exists(args.out) else "w"
    if mode == "a":
        try:                        # a file left by a killed run may be broken
            with h5py.File(args.out, "r") as h5:
                kept = len(h5.keys())
            print(f"resuming {args.out}: {kept} episodes kept", flush=True)
        except OSError as e:
            bad = f"{args.out}.bad{int(time.time())}"
            os.replace(args.out, bad)
            print(f"unreadable {args.out} ({e}); moved to {bad}, starting "
                  f"the unit again", flush=True)
            mode = "w"
    with h5py.File(args.out, mode) as h5:
        for name in [n for n in h5 if n.startswith("_tmp_")]:
            del h5[name]            # an episode a killed run did not finish
        h5.attrs.update(dict(
            env="lang_table", task="_".join(cfg["block_combo"]), k=1,
            n_proposed=args.k, pick_rule="uniform", temp_schedule="ladder",
            lookahead=args.lookahead, temps=temps, arm=arm,
            model_id=cfg["model_id"], schema_version=3, schema_variant="lt1",
            frame_codec="jpeg_q90",
            judge_native_scale=int(cfg.get("judge_native_scale", 1)),
            dry_run=bool(args.no_judge), code_state=args.code_state,
            recipe=json.dumps(dict(
                config=cfg, k_proposed=args.k, lookahead=args.lookahead,
                temp_schedule="ladder", pick_rule="uniform",
                w_touch=0.8, w_closer=0.2,
                label_columns=["touching", "closer", "touching"]))))
        wins = n_new = 0
        for i, seed in enumerate(seeds):
            if f"ep_{seed}" in h5:
                print(f"[{i+1}/{args.episodes}] ep_{seed} exists, skip",
                      flush=True)
                continue
            t0 = time.perf_counter()
            success, cycles, secs, check = run_episode(
                h5, env, goal, judge, cfg, seed, arm, args.k, temps,
                args.lookahead, args.check_cycles)
            wall = time.perf_counter() - t0
            wins += success
            n_new += 1
            print(f"[{i+1}/{args.episodes}] seed {seed} ({arm}): "
                  f"{'success' if success else 'fail'} @ {cycles} "
                  f"(SR {wins/n_new:.0%}) {wall:.0f}s = "
                  f"{wall/cycles:.2f}s/cycle [policy {secs['policy']:.0f} "
                  f"sim {secs['sim']:.0f} judge {secs['judge']:.0f} "
                  f"execute {secs['execute']:.0f}]", flush=True)
            if check["cycles"]:
                print(f"    restore check: {check['cycles']} cycles, "
                      f"{check['bad']} bad, max state diff "
                      f"{check['max_state']:.1e}, max px diff "
                      f"{check['max_px']}", flush=True)
                if check["bad"]:
                    raise SystemExit("RESTORE_CHECK_FAILED")
    print(f"DATAGEN_DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
