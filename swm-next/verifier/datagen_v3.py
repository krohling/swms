"""v3 training-data generation (pointwise-k1 campaign): execute-and-label-ONE
collection, robot-faithful in label budget.

v1 (datagen.py) labels ALL k proposals per cycle — only a simulator can score
counterfactual rollouts. On a real robot you label what you executed. v3 keeps
the sim env but adopts the robot's information diet: per cycle, ONE candidate
is picked (uniformly — logged propensity), rolled out `lookahead` steps for its
judge label, then the env rewinds and the usual `actions_per_cycle` actions of
that same candidate are executed. One label per cycle (plus the frame_t gate
probe, which a robot labeler could also produce — frame_t is a real frame);
all judge probes of a cycle run as ONE batched forward (verified
bit-identical to sequential single-image calls, 2026-11-02 A/B).

Collection design = (k proposals, temperature schedule, pick rule):
  v3a  k=1, fixed tau=1.0   solo standard-sampling proposal (NOT the ladder's
                            degenerate tau=0 — see the 2026-10-29 base
                            re-anchor), pick trivially 0.
  v3b  k=8, ladder          deployment's proposal distribution (same temps as
                            eval_closed_loop; no min_sep rejection — eval has
                            none), uniform pick, propensity 1/k.
Both arms differ from v1 ONLY in labels/cycle (1 vs k) and selection
(uniform vs 50/50 verifier/random) — state-visitation under uniform pick over
a solo tau=1 draw (v3a) vs over the ladder (v3b) is the campaign's axis.

HDF5 schema (schema_version 3): the TRAINING VIEW is strictly k=1 so every
existing pointwise loader (ProposalDataset / CycleDataset / ExecutedChunk-
Dataset) reads the executed candidate with its own label — never a
mispaired sibling. The full proposal set lives under v3-only names for
off-policy work (IPS/DR, future listwise-on-logged-data):
  ep_<seed>/frames_t        (C,)        vlen uint8 (JPEG)   committed frame
  ep_<seed>/trajs           (C,1,H,A)   f32   EXECUTED candidate only
  ep_<seed>/end_frames      (C,1)       vlen uint8 (JPEG)   its rollout end
  ep_<seed>/labels          (C,1,3)     f32   [p_approach, p_place, p_hold]
  ep_<seed>/pick            (C,)        i8    index into proposal_trajs
  ep_<seed>/propensity      (C,)        f32   P(pick) = 1/k  (uniform logger)
  ep_<seed>/proposal_trajs  (C,k,H,A)   f32   ALL proposals (ladder order)
  ep_<seed>/proposal_temps  (C,k)       f32   their sampling temperatures
  ep_<seed>/proposal_min_sep(C,)        f32   k>1 only: min pairwise mean-L2
  ep_<seed>/gate_raw        (C,)        f32   judge p_yes(frame_t, q_hold)
  ep_<seed>/gate            (C,)        f32   EMA(gate_raw, alpha=0.35)
  ep_<seed>/score           (C,1)       f32   blended score of the executed
  ep_<seed>/state_t_qpos/qvel   (C,...)      sim state at frame_t
  ep_<seed>/end_qpos/qvel       (C,1,...)    sim state at executed rollout end
  attrs: policy="random" (loader-compat episode_split stratum), arm, seed,
         success, cycles, q_approach/q_place/q_hold
No planner_pick / spread_* — those need all-k labels, which v3 doesn't buy.

Usage (dataset-defining knobs come from the config's `generation:` section,
CLI overrides; --episodes/--seed-start/--out stay CLI-only, they vary per
shard of one recipe):
  PYTHONPATH=$PWD:$PWD/swm-next python swm-next/verifier/datagen_v3.py \
    --config swm-next/configs/datagen_v3a.yaml --episodes 1000 \
    --seed-start 40000 --out datasets_swm/bg_v3a_k1tau1_s40000.h5
Seeds — full ledger of taken ranges (verified on m5 + Vista datasets_swm,
2026-11-02): 6000-6099 eval / 10000-10799 bg_k16 / 12000-22799 bg_k1_exec
(run-3) / 30000-35399 v1 hybrid / 40000-40999 extra hybrid shard /
50000-56499 v2. v3a shards start at 42000, v3b at 60000, smoke at 90000+.
"""
import argparse
import os

import h5py
import numpy as np
import torch
import yaml

from swm.constants import ANSWER_OPTIONS
from swm.diffusion_policy import DiffusionPolicy
from swm.utils.goal_generators import get_ogbench_goal

from datagen import GATE_ALPHA, HOLD_W, PLACE_W, jpeg  # noqa: E402
from mpc import build_questions, make_env  # noqa: E402  (same dir)


def proposal_temps(schedule, k, fixed_temp):
    if schedule == "fixed":
        return [float(fixed_temp)] * k
    assert k > 1, ("ladder with k=1 degenerates to tau=0 (deterministic "
                   "mean path) — use temp_schedule: fixed")
    return [i * 2.0 / (k - 1) for i in range(k)]


def run_episode(h5, env, goal, judge, policy_path, cfg, q_app, q_place,
                q_hold, seed, arm, k, temps, lookahead):
    torch.manual_seed(hash((seed, "mpc")) % 2**31)
    np.random.seed(hash(("mpc", seed)) % 2**31)
    rng = np.random.RandomState(seed)          # pick RNG (same as v1 random)

    frame = goal.reset_env(seed=seed)
    goal.reset_hook()
    policy = DiffusionPolicy.load(policy_path, device=cfg["device"])
    policy.add_obs(frame)

    n_exec = cfg["actions_per_cycle"]

    gate_ema = None
    rec = dict(frames_t=[], trajs=[], end_frames=[], labels=[], pick=[],
               propensity=[], proposal_trajs=[], min_sep=[], gate_raw=[],
               gate=[], score=[], state_t=[], end_states=[])
    done = False
    cycle = -1
    for cycle in range(cfg["max_cycles"]):
        torch.manual_seed(hash((seed, cycle)) % 2**31)
        candidates = np.asarray(policy.sample_trajs(k, temperatures=temps))
        # pick FIRST — the whole budget story is that only the executed
        # candidate is ever rolled out or labeled
        pick = int(rng.randint(k))
        chosen = candidates[pick]
        if k > 1:
            dd = np.sqrt(((candidates[:, None] - candidates[None, :]) ** 2
                          ).sum(-1)).mean(-1)
            np.fill_diagonal(dd, np.inf)
            rec["min_sep"].append(float(dd.min()))

        frame_t = np.asarray(frame, dtype=np.uint8)
        saved = env.get_state()
        rec["state_t"].append({k2: np.array(v) for k2, v in saved.items()})

        env.set_state(saved)
        f = frame
        for a in chosen[:lookahead]:
            f = env.step(np.asarray(a))
        end_frame = np.asarray(f, dtype=np.uint8)
        rec["end_states"].append({k2: np.array(v)
                                  for k2, v in env.get_state().items()})
        env.set_state(saved)

        # all of the cycle's labels in ONE batched judge forward: approach +
        # place on the executed rollout end frame, gate probe (q_hold) on
        # frame_t (a real frame — robot-labelable). p_yes builds per-pair
        # prompts and reads each sequence's own last real token under right
        # padding — verified bit-identical to sequential single-image calls
        # (2026-11-02 A/B on smoke frames, max |dp| = 0.0).
        if q_hold == q_app:
            p, _ = judge.p_yes([end_frame, end_frame, frame_t],
                               [q_app, q_place, q_hold])
            p_app, p_place = p[0:1], p[1:2]
            p_hold, rg = p_app, float(p[2])
        else:
            p, _ = judge.p_yes([end_frame, end_frame, end_frame, frame_t],
                               [q_app, q_place, q_hold, q_hold])
            p_app, p_place, p_hold = p[0:1], p[1:2], p[2:3]
            rg = float(p[3])
        gate_ema = rg if gate_ema is None else \
            GATE_ALPHA * rg + (1 - GATE_ALPHA) * gate_ema
        g = gate_ema
        score = (1.0 - g) * p_app + g * (PLACE_W * p_place + HOLD_W * p_hold)

        rec["frames_t"].append(jpeg(frame_t))
        rec["trajs"].append(np.asarray(chosen, dtype=np.float32)[None])
        rec["end_frames"].append([jpeg(end_frame)])
        rec["labels"].append(np.stack(
            [p_app, p_place, p_hold], axis=-1).astype(np.float32))
        rec["pick"].append(pick)
        rec["propensity"].append(1.0 / k)
        rec["proposal_trajs"].append(np.asarray(candidates, dtype=np.float32))
        rec["gate_raw"].append(rg)
        rec["gate"].append(g)
        rec["score"].append(score.astype(np.float32))

        for a in chosen[:n_exec]:
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
    e = grp.create_dataset("end_frames", (C, 1), dtype=vlen)
    for i, row in enumerate(rec["end_frames"]):
        e[i, 0] = row[0]
    grp.create_dataset("labels", data=np.stack(rec["labels"]))
    grp.create_dataset("pick", data=np.asarray(rec["pick"], dtype=np.int8))
    grp.create_dataset("propensity",
                       data=np.asarray(rec["propensity"], np.float32))
    grp.create_dataset("proposal_trajs", data=np.stack(rec["proposal_trajs"]))
    grp.create_dataset("proposal_temps", data=np.tile(
        np.asarray(temps, np.float32), (C, 1)))
    grp.create_dataset("gate_raw", data=np.asarray(rec["gate_raw"], np.float32))
    grp.create_dataset("gate", data=np.asarray(rec["gate"], np.float32))
    grp.create_dataset("score", data=np.stack(rec["score"]))
    for key in ("qpos", "qvel"):
        grp.create_dataset(f"state_t_{key}",
                           data=np.stack([st[key] for st in rec["state_t"]]))
        grp.create_dataset(f"end_{key}", data=np.stack(
            [es[key] for es in rec["end_states"]])[:, None])
    if rec["min_sep"]:
        grp.create_dataset("proposal_min_sep",
                           data=np.asarray(rec["min_sep"], dtype=np.float32))
    grp.attrs.update(dict(
        seed=seed, policy="random", arm=arm, success=bool(done),
        cycles=cycle + 1,
        q_approach=q_app, q_place=q_place, q_hold=q_hold))
    h5.flush()
    return done, cycle + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--lookahead", type=int, default=None)
    ap.add_argument("--temp-schedule", choices=["fixed", "ladder"],
                    default=None, dest="temp_schedule")
    ap.add_argument("--fixed-temp", type=float, default=None,
                    dest="fixed_temp")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    gen = cfg.get("generation") or {}
    if args.k is None:
        args.k = int(gen["k"])
    if args.lookahead is None:
        args.lookahead = int(gen.get("lookahead", 16))
    if args.temp_schedule is None:
        args.temp_schedule = str(gen["temp_schedule"])
    if args.fixed_temp is None:
        args.fixed_temp = float(gen.get("fixed_temp", 1.0))
    assert args.seed_start + args.episodes - 1 < 6000 or args.seed_start > 6100, \
        "seed range overlaps 6000-6099 (reserved for evaluation)"
    temps = proposal_temps(args.temp_schedule, args.k, args.fixed_temp)
    arm = (f"v3_k{args.k}_" +
           (f"tau{args.fixed_temp:g}" if args.temp_schedule == "fixed"
            else "ladder") + "_uniform")

    from label_teacher import QwenJudge
    judge = QwenJudge(model_id=cfg["model_id"], device=cfg["device"])
    env = make_env(cfg)
    goal = get_ogbench_goal("stack_blocks", env, None, ANSWER_OPTIONS,
                            {"block_combo": cfg["block_combo"]})
    tmpl = build_questions(cfg)
    qcfg = cfg.get("questions") or {}
    q_app = qcfg.get("approach", tmpl["grasp"])
    q_place = qcfg.get("place", tmpl["ontop"])
    q_hold = qcfg.get("hold", q_app)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    mode = "a" if os.path.exists(args.out) else "w"
    with h5py.File(args.out, mode) as h5:
        import json as _json
        h5.attrs.update(dict(
            task="_".join(cfg["block_combo"]), k=1, n_proposed=args.k,
            pick_rule="uniform", temp_schedule=args.temp_schedule,
            lookahead=args.lookahead, temps=temps, arm=arm,
            model_id=cfg["model_id"], schema_version=3,
            recipe=_json.dumps(dict(
                config=cfg, k_proposed=args.k, lookahead=args.lookahead,
                temp_schedule=args.temp_schedule, fixed_temp=args.fixed_temp,
                pick_rule="uniform", gate_alpha=GATE_ALPHA, place_w=PLACE_W,
                hold_w=HOLD_W, q_approach=q_app, q_place=q_place,
                q_hold=q_hold))))
        wins = 0
        for i in range(args.episodes):
            seed = args.seed_start + i
            if f"ep_{seed}" in h5:
                print(f"[{i+1}/{args.episodes}] ep_{seed} exists, skip",
                      flush=True)
                continue
            success, cycles = run_episode(
                h5, env, goal, judge, cfg["diffusion_path"], cfg, q_app,
                q_place, q_hold, seed, arm, args.k, temps, args.lookahead)
            wins += success
            print(f"[{i+1}/{args.episodes}] seed {seed} ({arm}): "
                  f"{'success' if success else 'fail'} @ {cycles} "
                  f"(SR {wins/(i+1):.0%})", flush=True)
    print(f"DATAGEN_DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
