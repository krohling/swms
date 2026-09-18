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
from swm.utils.envs import OGBenchEnv
from swm.utils.goal_generators import get_ogbench_goal

ARM_MATERIALS = ("ur5e/robotiq/metal", "ur5e/robotiq/silicone",
                 "ur5e/robotiq/gray", "ur5e/robotiq/black",
                 "ur5e/black", "ur5e/jointgray", "ur5e/linkgray",
                 "ur5e/lightblue")
PAD_MATERIAL = "ur5e/robotiq/pad_gray"


def make_env(cfg):
    """swm.utils.envs.get_ogbench_env plus an arm_alpha rendering dial
    (0.1 = OGBench default, 1.0 = opaque). Local mirror; core swm stays
    untouched by design."""
    import gymnasium
    import mujoco
    import ogbench  # noqa: F401  (env registration)
    env = gymnasium.make(
        "visual-cube-quadruple-v0", terminate_at_goal=False,
        visualize_info=False, mode="data_collection", max_episode_steps=1000,
        stack_goal=None, width=224, height=224, control_timestep=0.1, ood=False,
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

PHASE_THRESHOLD = 0.9
PAD_GEOMS = ("ur5e/robotiq/right_pad1", "ur5e/robotiq/right_pad2",
             "ur5e/robotiq/left_pad1", "ur5e/robotiq/left_pad2")
BLOCK_NUM = {"red_cube": 0, "blue_cube": 1, "yellow_cube": 2, "green_cube": 3}
LIFT_EPS = 0.02   # meters above the cube's reset height


class OracleProbe:
    """Ground-truth grasp/lift state from the simulator (never the judge):
    gripper_contact = pads touching the top cube; lifted = cube center more
    than LIFT_EPS above its reset height."""

    def __init__(self, env, cfg):
        import mujoco
        u = env.env.unwrapped
        self.u = u
        self.num = BLOCK_NUM[cfg["block_combo"][0]]
        self.bottom_num = BLOCK_NUM[cfg["block_combo"][1]]
        self.geom = u._cube_geom_ids_list[self.num][0]
        self.pads = [mujoco.mj_name2id(u._model, mujoco.mjtObj.mjOBJ_GEOM, g)
                     for g in PAD_GEOMS]
        self.z0 = None

    def reset(self):
        self.z0 = float(self.u.get_block_and_eef_poses()[f"block_{self.num}_pos"][2])

    def read(self):
        d = self.u._data
        contact = False
        for i in range(d.ncon):
            g = d.contact[i].geom
            if self.geom in g and any(p in g for p in self.pads):
                contact = True
                break
        poses = self.u.get_block_and_eef_poses()
        eef = poses["eef_pos"]
        top = poses[f"block_{self.num}_pos"]
        bottom = poses[f"block_{self.bottom_num}_pos"]
        z = float(top[2])
        return dict(gripper_contact=bool(contact),
                    lifted=bool(z > self.z0 + LIFT_EPS),
                    dz=round(z - self.z0, 4),
                    eef=[round(float(x), 4) for x in eef[:3]],
                    top=[round(float(x), 4) for x in top[:3]],
                    bottom=[round(float(x), 4) for x in bottom[:3]])


def build_questions(cfg):
    top, bottom = [b.replace("_", " ") for b in cfg["block_combo"]]
    return dict(
        grasp=f"Is the robot grasping the {top}?",
        ontop=f"Is the {top} on top of the {bottom}?",
        progress=(f"The robot was instructed to: Stack the {top} on top of "
                  f"the {bottom}. Is the robot closer to completing this "
                  "task than in the first image?"),
        # shaped candidates (question-screen set)
        near=f"Is the gripper near the {top}?",
        above_cube=f"Is the gripper directly above the {top}?",
        touching=f"Is the gripper touching the {top}?",
        lifted=f"Is the {top} lifted off the table?",
        above_goal=f"Is the {top} directly above the {bottom}?",
        close_goal=f"Is the {top} close to the {bottom}?",
    )


def load_spec(path, cfg):
    """Phase-scorer spec: weighted question sets per phase plus the
    transition question/threshold. Placeholders {top}/{bottom} fill from
    the config's block_combo. Replicating planning's objective exactly is
    specs/planning.yaml; iterate on questions by writing a new spec."""
    top, bottom = [b.replace("_", " ") for b in cfg["block_combo"]]
    fill = lambda s: s.replace("{top}", top).replace("{bottom}", bottom)
    raw = yaml.safe_load(open(path))
    tr = raw["transition"]
    # transition: single question {text, threshold} or weighted set
    # {questions: [{text, weight}, ...], threshold}
    if "questions" in tr:
        tq = [(fill(q["text"]), float(q["weight"])) for q in tr["questions"]]
    else:
        tq = [(fill(tr["text"]), 1.0)]
    spec = dict(
        name=raw["name"],
        phase0=[(fill(q["text"]), float(q["weight"])) for q in raw["phase0"]],
        phase1=[(fill(q["text"]), float(q["weight"])) for q in raw["phase1"]],
        transition=tq,
        threshold=float(tr.get("threshold", 0.9)),
    )
    return spec


def score_candidates(judge, scorer, phase, questions, current, end_frames,
                     batch, spec=None):
    """One score per candidate chunk."""
    def ask(images, q):
        out = np.zeros(len(images))
        for i in range(0, len(images), batch):
            p, _ = judge.p_yes(images[i:i + batch], [q] * len(images[i:i + batch]))
            out[i:i + len(images[i:i + batch])] = p
        return out

    if spec is not None:
        total = np.zeros(len(end_frames))
        breakdown = {}
        for q, w in (spec["phase0"] if phase == 0 else spec["phase1"]):
            p = ask(end_frames, q)
            breakdown[q] = [round(float(v), 4) for v in p]
            total += w * p
        return total, breakdown
    if scorer == "progress":
        pairs = [(current, f) for f in end_frames]
        return ask(pairs, questions["progress"]), {}
    if scorer.startswith("q:"):
        # single fixed question, no phase machinery
        return ask(end_frames, questions[scorer[2:]]), {}
    if phase == 0:
        return ask(end_frames, questions["grasp"]), {}
    return (0.6 * ask(end_frames, questions["ontop"])
            + 0.4 * ask(end_frames, questions["grasp"])), {}


class JudgeView:
    """Re-render the env's CURRENT pose at a different arm alpha for the
    judge, leaving the env/policy rendering untouched. Same model/data,
    materials flipped around a second mujoco.Renderer pass."""

    def __init__(self, env, alpha):
        import mujoco
        self.mujoco = mujoco
        self.u = env.env.unwrapped
        self.alpha = float(alpha)
        m = self.u._model
        self.mat_ids = [i for i in (self.mujoco.mj_name2id(
            m, self.mujoco.mjtObj.mjOBJ_MATERIAL, n)
            for n in ARM_MATERIALS + (PAD_MATERIAL,)) if i >= 0]
        self.renderer = self.mujoco.Renderer(m, height=224, width=224)

    def render(self):
        m = self.u._model
        saved = [(i, float(m.mat_rgba[i, 3])) for i in self.mat_ids]
        for i in self.mat_ids:
            m.mat_rgba[i, 3] = self.alpha
        self.renderer.update_scene(self.u._data, camera="front_pixels")
        out = np.asarray(self.renderer.render(), dtype=np.uint8)
        for i, a in saved:
            m.mat_rgba[i, 3] = a
        return out


def save_png(arr, path):
    """PNG (lossless): grasp-boundary frames shift the judge's p(yes) by
    0.2+ under JPEG recompression, so saved frames must be bit-identical
    to what was scored."""
    from PIL import Image
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(path)


def save_mp4(frames, path, fps=10):
    """Encode a list of HWC uint8 frames via ffmpeg (rawvideo stdin)."""
    import subprocess
    arr = np.stack([np.asarray(f, dtype=np.uint8) for f in frames])
    h, w = arr.shape[1:3]
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28",
         "-movflags", "+faststart", path],
        input=arr.tobytes(), check=True)


def run_episode(env, goal, judge, cfg, args, questions, seed, spec=None,
                detail_dir=None, judge_view=None, probe=None, temps=None):
    torch.manual_seed(hash((seed, "mpc")) % 2**31)
    np.random.seed(hash(("mpc", seed)) % 2**31)

    frame = goal.reset_env(seed=seed)
    goal.reset_hook()
    if probe is not None:
        probe.reset()
    policy = DiffusionPolicy.load(cfg["diffusion_path"], device=cfg["device"])
    policy.add_obs(frame)

    n_exec = cfg["actions_per_cycle"]
    phase, log = 0, []
    traj_frames = [np.asarray(frame, dtype=np.uint8)] if detail_dir else None
    ep_dir = None
    if detail_dir is not None:
        ep_dir = os.path.join(detail_dir, f"s{seed}")
        os.makedirs(ep_dir, exist_ok=True)
    for cycle in range(cfg["max_cycles"]):
        # Phase transition: model-judged on the current committed frame
        # (question/threshold from the spec when one is loaded, else the
        # StackBlocksGoal defaults).
        trans_info = None
        if phase == 0 and (spec is not None or args.scorer == "phase"):
            tqs = spec["transition"] if spec else [(questions["grasp"], 1.0)]
            th = spec["threshold"] if spec else PHASE_THRESHOLD
            tframe = judge_view.render() if judge_view else frame
            total, parts = 0.0, {}
            for q, w in tqs:
                p_t, _ = judge.p_yes([tframe], [q])
                parts[q] = round(float(p_t[0]), 4)
                total += w * float(p_t[0])
            fired = total > th
            trans_info = dict(questions={q: dict(weight=w, p=parts[q])
                                         for q, w in tqs},
                              p=round(total, 4), threshold=th, fired=fired)
            if ep_dir is not None:
                save_png(tframe, os.path.join(ep_dir, f"c{cycle:02d}_transition.png"))
            if fired:
                phase = 1

        torch.manual_seed(hash((seed, cycle)) % 2**31)
        candidates = np.asarray(policy.sample_trajs(args.k, temperatures=temps))

        if args.k == 1:
            pick, scores = 0, [0.0]
        else:
            saved = env.get_state()
            end_frames = []
            all_cand_frames = []
            cand_oracle = []
            for ci, c in enumerate(candidates):
                env.set_state(saved)
                f = frame
                cand_frames = [np.asarray(frame, dtype=np.uint8)] \
                    if detail_dir is not None else None
                deep = args.lookahead > len(c)
                if deep:
                    from collections import deque
                    with policy._lock:
                        buf = deque(policy.obs_deque, maxlen=policy.obs_deque.maxlen)
                for a in c:
                    f = env.step(np.asarray(a))
                    if cand_frames is not None:
                        cand_frames.append(np.asarray(f, dtype=np.uint8))
                    if deep:
                        policy.add_obs(f)
                done_steps = len(c)
                while done_steps < args.lookahead:
                    torch.manual_seed(hash((seed, cycle, ci, done_steps)) % 2**31)
                    chunk = policy.get_action()
                    for a in chunk[: args.lookahead - done_steps]:
                        f = env.step(np.asarray(a))
                        if cand_frames is not None:
                            cand_frames.append(np.asarray(f, dtype=np.uint8))
                        policy.add_obs(f)
                    done_steps += min(len(chunk), args.lookahead - done_steps)
                if deep:
                    with policy._lock:
                        policy.obs_deque.clear()
                        policy.obs_deque.extend(buf)
                end_frames.append(judge_view.render() if judge_view
                                  else np.asarray(f, dtype=np.uint8))
                if probe is not None:
                    cand_oracle.append(probe.read())   # candidate's end state
                if cand_frames is not None:
                    all_cand_frames.append(cand_frames)
            env.set_state(saved)
            if args.scorer == "oracle_dist":
                # Transparent oracle proxy: before grasp, score each candidate
                # by -distance(gripper, top cube); after grasp, by
                # -distance(top cube, bottom cube). Phase from ground-truth
                # contact on the committed state (non-sticky). No judge.
                committed = probe.read()
                grasped = committed["gripper_contact"]
                def _d(a, b):
                    return float(np.linalg.norm(np.array(a) - np.array(b)))
                scores = np.array([
                    -_d(o["top"], o["bottom"]) if grasped else -_d(o["eef"], o["top"])
                    for o in cand_oracle])
                breakdown = {}
                phase = 1 if grasped else 0
            else:
                scores, breakdown = score_candidates(
                    judge, args.scorer, phase, questions,
                    np.asarray(frame, dtype=np.uint8),
                    end_frames, int(cfg.get("batch", 8)), spec=spec)
            pick = int(np.argmax(scores))
            if ep_dir is not None:
                d = ep_dir
                save_png(frame, os.path.join(d, f"c{cycle:02d}_committed.png"))
                for ci_, f_ in enumerate(end_frames):
                    save_png(f_, os.path.join(d, f"c{cycle:02d}_cand{ci_}.png"))
                for ci_, cf in enumerate(all_cand_frames):
                    save_mp4(cf, os.path.join(d, f"c{cycle:02d}_cand{ci_}.mp4"))

        done = False
        for a in candidates[pick][:n_exec]:
            frame = env.step(np.asarray(a))
            if traj_frames is not None:
                traj_frames.append(np.asarray(frame, dtype=np.uint8))
            policy.add_obs(frame)
            if goal.get_done():
                done = True
                break
        entry = dict(cycle=cycle, phase=phase, pick=pick,
                     scores=[round(float(s), 4) for s in scores])
        if temps is not None:
            entry["temps"] = temps
        if args.k > 1 and probe is not None and cand_oracle:
            entry["cand_oracle"] = cand_oracle
        if probe is not None:
            entry["oracle"] = probe.read()
        if trans_info is not None:
            entry["transition"] = trans_info
        if args.k > 1 and spec is not None:
            entry["breakdown"] = breakdown
        log.append(entry)
        if done:
            break
    if traj_frames is not None and len(traj_frames) > 1:
        save_mp4(traj_frames, os.path.join(ep_dir, "trajectory.mp4"))
    if done:
        return True, cycle, log
    return False, cfg["max_cycles"], log


_WCTX = {}


def _init_worker(cfg, args):
    """Build a per-process env/policy stack once. No judge is loaded for
    oracle_dist (judge=None); VLM scorers would load one per worker, which
    is why --workers should stay 1 for those."""
    import sys as _s
    _s.dont_write_bytecode = True
    env = make_env(cfg)
    goal = get_ogbench_goal("stack_blocks", env, None, ANSWER_OPTIONS,
                            {"block_combo": cfg["block_combo"]})
    judge = None
    if args.scorer != "oracle_dist" and (args.k > 1 or args.scorer == "phase" or args.spec):
        from label_teacher import QwenJudge
        judge = QwenJudge(model_id=cfg["model_id"], device=cfg["device"])
    spec = load_spec(args.spec, cfg) if (args.spec and args.scorer != "oracle_dist") else None
    temps = [float(x) for x in args.temps.split(",")] if args.temps else None
    jview = JudgeView(env, args.judge_alpha) if args.judge_alpha is not None else None
    tag = f"spec_{spec['name']}" if spec else args.scorer.replace(":", "_")
    if args.judge_alpha is not None:
        tag += f"_ja{args.judge_alpha:g}"
    if temps is not None:
        tag += "_templadder"
    detail_dir = os.path.join(args.out, f"detail_{tag}_k{args.k}_L{args.lookahead}") \
        if args.save_frames else None
    _WCTX.update(env=env, goal=goal, judge=judge, spec=spec, temps=temps,
                 jview=jview, detail_dir=detail_dir, cfg=cfg, args=args,
                 questions=build_questions(cfg), probe=OracleProbe(env, cfg))


def _worker_seed(seed):
    w = _WCTX
    return seed, run_episode(w["env"], w["goal"], w["judge"], w["cfg"], w["args"],
                             w["questions"], seed, spec=w["spec"],
                             detail_dir=w["detail_dir"], judge_view=w["jview"],
                             probe=w["probe"], temps=w["temps"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temps", default=None,
                    help="comma-separated per-candidate initial-noise "
                         "temperatures (diversity ladder); length overrides "
                         "--k. e.g. 0.0,0.66,1.33,2.0")
    ap.add_argument("--judge-alpha", type=float, default=None,
                    help="re-render frames for the JUDGE at this arm alpha "
                         "(policy/env keep cfg arm_alpha; e.g. 1.0 = judge "
                         "sees an opaque arm)")
    ap.add_argument("--save-frames", action="store_true",
                    help="persist committed + candidate end frames per "
                         "selection point (JPEG, ~100MB per n=50 run)")
    ap.add_argument("--spec", default=None,
                    help="path to a phase-scorer spec yaml (weighted question "
                         "sets per phase + transition); overrides --scorer")
    ap.add_argument("--scorer", default="phase",
                    help="phase | progress | q:<name> for a single fixed "
                         "question (e.g. q:above_cube), no phase machinery")
    ap.add_argument("--lookahead", type=int, default=16,
                    help="virtual rollout depth in env steps; beyond the "
                         "16-step candidate chunk the policy continues the "
                         "rollout closed-loop (resampled every 16 steps)")
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel episode workers (each builds its own env+"
                         "policy). Big speedup for oracle_dist (no judge); with "
                         "a VLM scorer every worker loads the judge, so keep 1.")
    ap.add_argument("--out", default="outputs_verifier_mpc")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    import sys
    sys.dont_write_bytecode = True
    needs_judge = args.scorer != "oracle_dist" and (
        args.k > 1 or args.scorer == "phase" or args.spec)
    judge = None
    if needs_judge:
        from label_teacher import QwenJudge
        judge = QwenJudge(model_id=cfg["model_id"], device=cfg["device"])
    spec = load_spec(args.spec, cfg) if (args.spec and args.scorer != "oracle_dist") else None
    temps = [float(x) for x in args.temps.split(",")] if args.temps else None
    if temps is not None:
        args.k = len(temps)

    env = make_env(cfg)
    goal = get_ogbench_goal("stack_blocks", env, None, ANSWER_OPTIONS,
                            {"block_combo": cfg["block_combo"]})
    questions = build_questions(cfg)
    judge_view = JudgeView(env, args.judge_alpha) \
        if args.judge_alpha is not None else None
    probe = OracleProbe(env, cfg)

    os.makedirs(args.out, exist_ok=True)
    tag = f"spec_{spec['name']}" if spec else args.scorer.replace(":", "_")
    if args.judge_alpha is not None:
        tag += f"_ja{args.judge_alpha:g}"
    if temps is not None:
        tag += "_templadder"
    detail_dir = os.path.join(args.out, f"detail_{tag}_k{args.k}_L{args.lookahead}") \
        if args.save_frames else None
    seeds = [cfg["seed_start"] + i for i in range(args.seeds)]
    results, wins = [], 0

    def one(seed, _env, _goal, _probe, _jview):
        # per-seed episode; deterministic in seed, so worker order is irrelevant
        return seed, run_episode(_env, _goal, judge, cfg, args, questions, seed,
                                 spec=spec, detail_dir=detail_dir,
                                 judge_view=_jview, probe=_probe, temps=temps)

    if args.workers <= 1:
        stream = (one(s, env, goal, probe, judge_view) for s in seeds)
    else:
        from concurrent.futures import ProcessPoolExecutor
        _WCTX["cfg"] = cfg; _WCTX["args"] = args
        pool = ProcessPoolExecutor(max_workers=args.workers,
                                   initializer=_init_worker,
                                   initargs=(cfg, args))
        stream = pool.map(_worker_seed, seeds)

    done = 0
    for seed, (success, cycles, log) in sorted(
            stream, key=lambda x: x[0]) if False else stream:
        done += 1
        wins += success
        results.append(dict(seed=seed, success=success, cycles=cycles, log=log))
        print(f"[{done}/{args.seeds}] seed {seed}: "
              f"{'success' if success else 'fail'} @ cycle {cycles} "
              f"(running SR {wins/done:.0%})", flush=True)
    if args.workers > 1:
        pool.shutdown()
    results.sort(key=lambda d: d["seed"])

    out = dict(k=args.k, scorer=args.scorer, judge_alpha=args.judge_alpha,
               temps=temps,
               spec=(dict(spec) if spec else None), lookahead=args.lookahead,
               n=args.seeds, sr=wins / args.seeds, episodes=results)
    path = os.path.join(args.out,
                        f"mpc_{tag}_k{args.k}_L{args.lookahead}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"MPC_DONE: SR {wins}/{args.seeds} = {wins/args.seeds:.0%} -> {path}",
          flush=True)


if __name__ == "__main__":
    main()
