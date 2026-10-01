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


def parse_wq(s):
    """Parse a weighted-question override: '0.7:Is X?;0.3:Is Y?' ->
    [('Is X?',0.7),('Is Y?',0.3)]; a bare 'Is X?' -> [('Is X?',1.0)]. None -> None."""
    if not s:
        return None
    out = []
    for part in s.split(";"):
        part = part.strip()
        head = part.split(":", 1)[0]
        if ":" in part and head.replace(".", "", 1).isdigit():
            w, q = part.split(":", 1)
            out.append((q.strip(), float(w)))
        else:
            out.append((part, 1.0))
    return out


def score_candidates(judge, scorer, phase, questions, current, end_frames,
                     batch, spec=None, gate=None):
    """One score per candidate chunk."""
    def ask(images, q):
        out = np.zeros(len(images))
        for i in range(0, len(images), batch):
            p, _ = judge.p_yes(images[i:i + batch], [q] * len(images[i:i + batch]))
            out[i:i + len(images[i:i + batch])] = p
        return out

    if scorer in ("softgate", "softgate_ema"):
        # Latch-free soft-gated objective (addresses the one-way phase latch).
        # gate = "am I holding it now?" = p(grasp) on the committed frame
        # (raw for `softgate`, EMA-smoothed across cycles for `softgate_ema`,
        # supplied by run_episode). When holding (gate high) weight placing;
        # when dropped (gate low) weight re-grasping. Self-corrects after a
        # drop; the EMA variant ignores single-cycle p(grasp) dips.
        # Each slot is a weighted question list (blends supported). Defaults:
        # approach = grasp, place = ontop, hold(drop-guard) = grasp. The gate
        # (g) is the hold question on the committed frame, computed in
        # run_episode; hold is decoupled from approach so overriding the
        # approach question never weakens the drop-guard / recovery.
        def wask(wqs):
            t = np.zeros(len(end_frames))
            for q, w in wqs:
                t += w * ask(end_frames, q)
            return t
        g = gate if gate is not None else float(ask([current], questions["grasp"])[0])
        appr = questions.get("_approach") or [(questions["grasp"], 1.0)]
        place = questions.get("_place") or [(questions["ontop"], 1.0)]
        hold = questions.get("_hold") or [(questions["grasp"], 1.0)]
        pg = wask(appr)
        ph = pg if hold == appr else wask(hold)
        po = wask(place)
        total = (1.0 - g) * pg + g * (0.6 * po + 0.4 * ph)
        return total, {"gate": round(float(g), 4),
                       "approach": [round(float(v), 4) for v in pg],
                       "place": [round(float(v), 4) for v in po]}
    if spec is not None:
        total = np.zeros(len(end_frames))
        breakdown = {}
        for q, w in (spec["phase0"] if phase == 0 else spec["phase1"]):
            p = ask(end_frames, q)
            breakdown[q] = [round(float(v), 4) for v in p]
            total += w * p
        return total, breakdown
    if scorer == "random":
        # Selection-ablated control: identical candidate distribution
        # (temperature ladder), uniform-random pick. Isolates the verifier's
        # discrimination from the modified proposal distribution. Seeded via
        # the global np.random stream (per-episode/cycle reseeded in callers).
        return np.random.rand(len(end_frames)), {}
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
    gate_ema = None                       # EMA holding-gate state (softgate_ema)
    # EMA smoothing constant(s). Symmetric by default (--gate-alpha, 0.35 =
    # ~5-cycle window, resists single-dip chatter). Asymmetric when
    # --gate-alpha-fall/-rise are set: a LARGER falling constant tracks a real
    # drop fast (gate swings toward re-grasping) while a SMALLER rising
    # constant stays smooth against single-frame false dips (e.g. fall=0.8,
    # rise=0.35).
    a_fall = args.gate_alpha_fall if args.gate_alpha_fall is not None else args.gate_alpha
    a_rise = args.gate_alpha_rise if args.gate_alpha_rise is not None else args.gate_alpha
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
                gate_val = None
                if args.scorer in ("softgate", "softgate_ema"):
                    gframe = (judge_view.render() if judge_view
                              else np.asarray(frame, dtype=np.uint8))
                    hold = questions.get("_hold") or [(questions["grasp"], 1.0)]
                    rg = 0.0
                    for q, w in hold:                    # gate = "am I holding it now?"
                        rg += w * float(judge.p_yes([gframe], [q])[0][0])
                    if args.scorer == "softgate_ema":
                        if gate_ema is None:
                            gate_ema = rg
                        else:
                            a = a_fall if rg < gate_ema else a_rise
                            gate_ema = a * rg + (1 - a) * gate_ema
                        gate_val = gate_ema
                    else:
                        gate_val = rg
                scores, breakdown = score_candidates(
                    judge, args.scorer, phase, questions,
                    np.asarray(frame, dtype=np.uint8),
                    end_frames, int(cfg.get("batch", 8)), spec=spec, gate=gate_val)
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


# ---------------------------------------------------------------------------
# LangTable branch (env_type: lang_table). Push task is SINGLE-PHASE (no
# grasp), so there is no gate/softgate here: the verifier value is the
# weighted push-question blend from the author framework
# (0.8 * "is X touching Y?" [end frame] + 0.2 * "are X and Y closer
# together?" [anchor,end frame PAIR]). Candidate rollouts use the Phase-1
# validated deterministic restore: deterministicOverlappingPairs=1 + combo
# restore (set_state THEN restore_state_native); see test_lt_set_state.py.
# Protocol mirrors configs/eval_base_diffusion_lt.yaml: 60 cycles x 2
# executed actions, horizon 16, action_dim 2.
# ---------------------------------------------------------------------------

def make_lt_env(cfg):
    from swm.utils.envs import get_lang_table_env
    env = get_lang_table_env({"ood": bool(cfg.get("ood", False)),
                              "block_combo": list(cfg["block_combo"])},
                             seed=int(cfg["seed_start"]))
    # Phase-1 determinism requirement (broadphase order independence)
    env.env._pybullet_client.setPhysicsEngineParameter(
        deterministicOverlappingPairs=1)
    return env


class LTProbe:
    """Ground-truth block/peg xy poses from the sim (never the judge)."""

    def __init__(self, env, goal):
        self.env, self.goal = env, goal

    def read(self):
        st = self.env.env.get_block_states()

        def get(name):
            key = name if name in st else name.replace("_", " ")
            return np.asarray(st[key], dtype=float)

        b1 = get(self.goal.info.block1)
        b2 = get(self.goal.info.block2)
        peg = np.asarray(st["peg"], dtype=float)
        return dict(
            b1=[round(float(x), 4) for x in b1],
            b2=[round(float(x), 4) for x in b2],
            peg=[round(float(x), 4) for x in peg],
            dist=round(float(np.linalg.norm(b1 - b2)), 4),
            dist_peg_b1=round(float(np.linalg.norm(peg - b1)), 4))


def build_lt_questions(goal):
    """[(question, weight, kind)] — kind 'pair' questions compare the anchor
    (committed) frame against the candidate end frame."""
    b1 = goal.info.block1.replace("_", " ")
    b2 = goal.info.block2.replace("_", " ")
    return [
        (f"Is the {b1} touching the {b2}?", 0.8, "single"),
        (f"Are the {b1} and {b2} closer together?", 0.2, "pair"),
    ]


def run_lt_episode(env, goal, judge, cfg, args, seed, detail_dir=None,
                   temps=None):
    torch.manual_seed(hash((seed, "mpc")) % 2**31)
    np.random.seed(hash(("mpc", seed)) % 2**31)
    # In-place reseed: goal.reward_function shares this RandomState object,
    # so .seed() (not reassignment) keeps both views consistent. LT's
    # reset_env ignores its seed arg; scene layout comes from this rng.
    env.env._rng.seed(seed)
    frame = goal.reset_env(seed=seed)
    policy = DiffusionPolicy.load(cfg["diffusion_path"], device=cfg["device"])
    policy.add_obs(frame)
    questions = build_lt_questions(goal)
    probe = LTProbe(env, goal)
    n_exec = cfg["actions_per_cycle"]
    log = []
    ep_dir = None
    if detail_dir is not None:
        ep_dir = os.path.join(detail_dir, f"seed_{seed}")
        os.makedirs(ep_dir, exist_ok=True)
    traj_frames = [np.asarray(frame, dtype=np.uint8)] if detail_dir else None
    done = False
    for cycle in range(cfg["max_cycles"]):
        torch.manual_seed(hash((seed, cycle)) % 2**31)
        candidates = np.asarray(policy.sample_trajs(args.k, temperatures=temps))
        scores, cand_oracle = [0.0], []
        pick = 0
        if args.k > 1:
            S = env.get_state()
            env.set_state(S)
            sid = env.save_state_native()

            def restore():
                env.set_state(S)            # python-side pose + ObjState
                env.restore_state_native(sid)   # engine snapshot

            # Judge frames are rendered natively at judge_native_scale x the
            # env resolution (same camera/FOV, real detail — the 180x320 obs
            # stream starves the VLM). Policy/obs pipeline stays at native res.
            ns = int(cfg.get("judge_native_scale", 1))
            anchor = env.get_judge_frame(ns)
            end_frames, all_cand_frames = [], []
            for ci, c in enumerate(candidates):
                restore()
                f = frame
                cand_frames = [np.asarray(frame, dtype=np.uint8)] \
                    if detail_dir is not None else None
                # deep lookahead: candidate differences at chunk scale are
                # ~1 cm (sub-perceptual for the judge); rolling the policy
                # closed-loop in sim past the sampled chunk amplifies the
                # separation before scoring. Policy obs history is snapshotted
                # and restored so rollouts don't pollute the real episode.
                deep = args.lookahead > len(c)
                if deep:
                    from collections import deque
                    with policy._lock:
                        buf = deque(policy.obs_deque,
                                    maxlen=policy.obs_deque.maxlen)
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
                end_frames.append(env.get_judge_frame(ns))
                cand_oracle.append(probe.read())
                if cand_frames is not None:
                    all_cand_frames.append(cand_frames)
            restore()
            if args.scorer == "random":
                # selection-ablated control: same ladder, uniform-random pick
                scores = np.random.rand(len(end_frames))
            elif args.scorer == "oracle_dist":
                # Transparent push oracle: primarily how close the two blocks
                # end up, small tiebreak for getting the peg to the push block.
                scores = np.array([-(o["dist"] + 0.25 * o["dist_peg_b1"])
                                   for o in cand_oracle])
            else:
                total = np.zeros(len(end_frames))
                for q, w, kind in questions:
                    imgs = ([(anchor, ef) for ef in end_frames]
                            if kind == "pair" else list(end_frames))
                    total += w * np.asarray(
                        judge.p_yes(imgs, [q] * len(end_frames))[0])
                scores = total
            pick = int(np.argmax(scores))
            if ep_dir is not None:
                save_png(frame, os.path.join(ep_dir, f"c{cycle:02d}_committed.png"))
                for ci_, f_ in enumerate(end_frames):
                    save_png(f_, os.path.join(ep_dir, f"c{cycle:02d}_cand{ci_}.png"))
                for ci_, cf in enumerate(all_cand_frames):
                    save_mp4(cf, os.path.join(ep_dir, f"c{cycle:02d}_cand{ci_}.mp4"))
        for a in candidates[pick][:n_exec]:
            frame = env.step(np.asarray(a))
            if traj_frames is not None:
                traj_frames.append(np.asarray(frame, dtype=np.uint8))
            policy.add_obs(frame)
            if goal.get_done():
                done = True
                break
        entry = dict(cycle=cycle, pick=pick,
                     scores=[round(float(s), 4) for s in scores],
                     oracle=probe.read())
        if temps is not None:
            entry["temps"] = temps
        if args.k > 1 and cand_oracle:
            entry["cand_oracle"] = cand_oracle
        log.append(entry)
        if done:
            break
    if traj_frames is not None and len(traj_frames) > 1:
        save_mp4(traj_frames, os.path.join(ep_dir, "trajectory.mp4"))
    if done:
        return True, cycle, log
    return False, cfg["max_cycles"], log


def main_lt(args, cfg):
    """LangTable eval loop: same output schema as the OGBench path so all
    downstream analysis/report tooling works unchanged."""
    from swm.utils.goal_generators import get_lang_table_goal

    needs_judge = args.scorer not in ("oracle_dist", "random") and args.k > 1
    judge = None
    if needs_judge:
        if cfg.get("judge_url"):
            from http_judge import HTTPJudge
            judge = HTTPJudge(cfg["judge_url"],
                              upscale=int(cfg.get("judge_upscale", 1)))
        else:
            from label_teacher import QwenJudge
            judge = QwenJudge(model_id=cfg["model_id"], device=cfg["device"])
    temps = [float(x) for x in args.temps.split(",")] if args.temps else None
    if temps is not None:
        args.k = len(temps)

    env = make_lt_env(cfg)
    goal = get_lang_table_goal("block_to_block", env, None, ANSWER_OPTIONS,
                               {"block_combo": list(cfg["block_combo"]),
                                "ood": bool(cfg.get("ood", False))})

    os.makedirs(args.out, exist_ok=True)
    tag = "lt_" + args.scorer.replace(":", "_")
    if temps is not None:
        tag += "_templadder"
    tag += f"_L{args.lookahead}"
    detail_dir = os.path.join(args.out, f"detail_{tag}_k{args.k}") \
        if args.save_frames else None
    seed0 = args.seed_start if args.seed_start is not None else cfg["seed_start"]
    seeds = [seed0 + i for i in range(args.seeds)]
    results, wins = [], 0
    for done_n, seed in enumerate(seeds, 1):
        success, cycles, log = run_lt_episode(
            env, goal, judge, cfg, args, seed,
            detail_dir=detail_dir, temps=temps)
        wins += success
        results.append(dict(seed=seed, success=bool(success), cycles=cycles,
                            log=log))
        print(f"[{done_n}/{args.seeds}] seed {seed}: "
              f"{'success' if success else 'fail'} @ cycle {cycles} "
              f"(running SR {wins/done_n:.0%})", flush=True)
    out = dict(k=args.k, scorer=args.scorer, env="lang_table",
               block_combo=list(cfg["block_combo"]), temps=temps,
               n=args.seeds, sr=wins / args.seeds, episodes=results)
    path = os.path.join(args.out, f"mpc_{tag}_k{args.k}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"MPC_DONE: SR {wins}/{args.seeds} = {wins/args.seeds:.0%} -> {path}",
          flush=True)


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
    ap.add_argument("--q-place", default=None, dest="q_place",
                    help="override the phase-1 PLACE question. Weighted blends: "
                         "'0.7:Is A?;0.3:Is B?' (default: ontop)")
    ap.add_argument("--q-approach", default=None, dest="q_approach",
                    help="override the phase-0 APPROACH question (softgate only; "
                         "weighted blends supported; gate + drop-guard stay the "
                         "hold question)")
    ap.add_argument("--q-hold", default=None, dest="q_hold",
                    help="override the softgate GATE + drop-guard 'holding' "
                         "question (default: grasp; weighted blends supported)")
    ap.add_argument("--seed-start", type=int, default=None, dest="seed_start",
                    help="override cfg seed_start (fresh-seed replays)")
    ap.add_argument("--gate-alpha", type=float, default=0.35, dest="gate_alpha",
                    help="softgate_ema smoothing constant (symmetric): "
                         "gate_ema = a*p(grasp) + (1-a)*gate_ema. Default 0.35.")
    ap.add_argument("--gate-alpha-fall", type=float, default=None, dest="gate_alpha_fall",
                    help="asymmetric EMA: constant used when the gate FALLS "
                         "(p(grasp) < gate_ema, i.e. a drop). Larger = faster "
                         "drop response. Defaults to --gate-alpha.")
    ap.add_argument("--gate-alpha-rise", type=float, default=None, dest="gate_alpha_rise",
                    help="asymmetric EMA: constant used when the gate RISES. "
                         "Smaller = smoother against false dips. Defaults to "
                         "--gate-alpha.")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    import sys
    sys.dont_write_bytecode = True
    if cfg.get("env_type") == "lang_table":
        return main_lt(args, cfg)
    needs_judge = args.scorer not in ("oracle_dist", "random") and (
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
    for key, val in (("_approach", args.q_approach), ("_place", args.q_place),
                     ("_hold", args.q_hold)):
        wq = parse_wq(val)
        if wq:
            questions[key] = wq
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
    seed0 = args.seed_start if args.seed_start is not None else cfg["seed_start"]
    seeds = [seed0 + i for i in range(args.seeds)]
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
