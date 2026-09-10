"""Verifier-guided MPC: Qwen as the value function in closed-loop control.

At every replan step, k candidate action chunks are sampled from the base
diffusion policy, each is rolled out in sim from a saved state (the sim
plays a PERFECT world model here -- this experiment isolates value quality
from dynamics quality), the judge scores each candidate's end frame, and
the winner's first actions_per_cycle actions are executed for real.

Scoring mirrors planning's objective exactly (StackBlocksGoal):
    phase 0:  score = p(grasp) at chunk end
    phase 1:  score = 0.6 * p(ontop) + 0.4 * p(grasp) at chunk end
    phase 0 -> 1 when p(grasp) on the CURRENT committed frame crosses 0.9
The alternative scorer "progress" asks the two-frame progress question
(current frame vs chunk end) instead. k=1 reduces to the base policy.

Result: task SR over seeds, directly comparable to base diffusion (52),
SWM gradient planning, and the published checkpoint. Per-cycle decisions
are logged for offline analysis.

Usage (repo root + swm-next on PYTHONPATH):

    PYTHONPATH=$PWD:$PWD/swm-next python swm-next/verifier/mpc.py \
        --config swm-next/configs/verifier.yaml [--k 8] [--scorer phase] \
        [--seeds 25] [--out outputs_verifier_mpc]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import yaml

from swm.constants import ANSWER_OPTIONS
from swm.diffusion_policy import DiffusionPolicy
from swm.utils.goal_generators import get_ogbench_goal

from capture import make_env

PHASE_THRESHOLD = 0.9


def build_questions(cfg):
    top, bottom = [b.replace("_", " ") for b in cfg["block_combo"]]
    return dict(
        grasp=f"Is the robot grasping the {top}?",
        ontop=f"Is the {top} on top of the {bottom}?",
        progress=(f"The robot was instructed to: Stack the {top} on top of "
                  f"the {bottom}. Is the robot closer to completing this "
                  "task than in the first image?"),
    )


def score_candidates(judge, scorer, phase, questions, current, end_frames, batch):
    """One score per candidate chunk."""
    def ask(images, q):
        out = np.zeros(len(images))
        for i in range(0, len(images), batch):
            p, _ = judge.p_yes(images[i:i + batch], [q] * len(images[i:i + batch]))
            out[i:i + len(images[i:i + batch])] = p
        return out

    if scorer == "progress":
        pairs = [(current, f) for f in end_frames]
        return ask(pairs, questions["progress"])
    if phase == 0:
        return ask(end_frames, questions["grasp"])
    return (0.6 * ask(end_frames, questions["ontop"])
            + 0.4 * ask(end_frames, questions["grasp"]))


def run_episode(env, goal, judge, cfg, args, questions, seed):
    torch.manual_seed(hash((seed, "mpc")) % 2**31)
    np.random.seed(hash(("mpc", seed)) % 2**31)

    frame = goal.reset_env(seed=seed)
    goal.reset_hook()
    policy = DiffusionPolicy.load(cfg["diffusion_path"], device=cfg["device"])
    policy.add_obs(frame)

    n_exec = cfg["actions_per_cycle"]
    phase, log = 0, []
    for cycle in range(cfg["max_cycles"]):
        # Phase transition, exactly like StackBlocksGoal.get_questions:
        # model-judged p(grasp) on the current committed frame.
        if phase == 0 and args.scorer == "phase":
            p_grasp, _ = judge.p_yes([frame], [questions["grasp"]])
            if float(p_grasp[0]) > PHASE_THRESHOLD:
                phase = 1

        torch.manual_seed(hash((seed, cycle)) % 2**31)
        candidates = np.asarray(policy.sample_trajs(args.k))

        if args.k == 1:
            pick, scores = 0, [0.0]
        else:
            saved = env.get_state()
            end_frames = []
            for ci, c in enumerate(candidates):
                env.set_state(saved)
                f = frame
                deep = args.lookahead > len(c)
                if deep:
                    from collections import deque
                    with policy._lock:
                        buf = deque(policy.obs_deque, maxlen=policy.obs_deque.maxlen)
                for a in c:
                    f = env.step(np.asarray(a))
                    if deep:
                        policy.add_obs(f)
                done_steps = len(c)
                while done_steps < args.lookahead:
                    torch.manual_seed(hash((seed, cycle, ci, done_steps)) % 2**31)
                    chunk = policy.get_action()
                    for a in chunk[: args.lookahead - done_steps]:
                        f = env.step(np.asarray(a))
                        policy.add_obs(f)
                    done_steps += min(len(chunk), args.lookahead - done_steps)
                if deep:
                    with policy._lock:
                        policy.obs_deque.clear()
                        policy.obs_deque.extend(buf)
                end_frames.append(np.asarray(f, dtype=np.uint8))
            env.set_state(saved)
            scores = score_candidates(judge, args.scorer, phase, questions,
                                      np.asarray(frame, dtype=np.uint8),
                                      end_frames, int(cfg.get("batch", 8)))
            pick = int(np.argmax(scores))

        done = False
        for a in candidates[pick][:n_exec]:
            frame = env.step(np.asarray(a))
            policy.add_obs(frame)
            if goal.get_done():
                done = True
                break
        log.append(dict(cycle=cycle, phase=phase, pick=pick,
                        scores=[round(float(s), 4) for s in scores]))
        if done:
            return True, cycle, log
    return False, cfg["max_cycles"], log


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--scorer", choices=["phase", "progress"], default="phase")
    ap.add_argument("--lookahead", type=int, default=16,
                    help="virtual rollout depth in env steps; beyond the "
                         "16-step candidate chunk the policy continues the "
                         "rollout closed-loop (resampled every 16 steps)")
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--out", default="outputs_verifier_mpc")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    import sys
    sys.dont_write_bytecode = True
    from label_teacher import QwenJudge
    judge = QwenJudge(model_id=cfg["model_id"], device=cfg["device"]) \
        if args.k > 1 or args.scorer == "phase" else None

    env = make_env(cfg)
    goal = get_ogbench_goal("stack_blocks", env, None, ANSWER_OPTIONS,
                            {"block_combo": cfg["block_combo"]})
    questions = build_questions(cfg)

    os.makedirs(args.out, exist_ok=True)
    results, wins = [], 0
    for i in range(args.seeds):
        seed = cfg["seed_start"] + i
        success, cycles, log = run_episode(env, goal, judge, cfg, args,
                                           questions, seed)
        wins += success
        results.append(dict(seed=seed, success=success, cycles=cycles, log=log))
        print(f"[{i+1}/{args.seeds}] seed {seed}: "
              f"{'success' if success else 'fail'} @ cycle {cycles} "
              f"(running SR {wins/(i+1):.0%})", flush=True)

    out = dict(k=args.k, scorer=args.scorer, lookahead=args.lookahead,
               n=args.seeds, sr=wins / args.seeds, episodes=results)
    path = os.path.join(args.out,
                        f"mpc_{args.scorer}_k{args.k}_L{args.lookahead}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"MPC_DONE: SR {wins}/{args.seeds} = {wins/args.seeds:.0%} -> {path}",
          flush=True)


if __name__ == "__main__":
    main()
