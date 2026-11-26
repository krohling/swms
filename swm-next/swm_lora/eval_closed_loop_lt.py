"""Closed-loop LangTable eval: a trained SWM checkpoint as the scorer.

The LangTable twin of eval_closed_loop.py. Replaces the teacher's (simulator
rollout + judge on the end frame) scoring of verifier-MPC (mpc.run_lt_episode)
with the SWM's direct estimate p_yes(frame_t, question, trajectory) per
proposal, the configuration the datagen_lt.py sets train it for:

  score_i = 0.8 * p_touch_i + 0.2 * p_closer_i      (mpc.build_lt_questions)

LangTable is single-phase, so there is no gate. Nothing here touches the
simulator state: no save, no rollout, no restore.

Same per cycle as the teacher and the generator: k proposals of the task's
base policy on the temperature ladder 0..2 (seeded per cycle exactly as
mpc.run_lt_episode seeds them), pick, execute the first actions_per_cycle
actions. The model's input is built the way training built it:

  frame    judge-scale render (judge_native_scale) -> JPEG q90 -> decode ->
           resize to the training config's image_size (data.decode_frame)
  actions  normalized with the training config's action range
           (data.action_range)

Eval seeds 6000+ (reserved; the generator refuses them).

--scorer random is the pick-blind control on the same harness: same proposals,
uniform-random pick, no model loaded.

Usage (LARG, from a full checkout; PYTHONPATH as for datagen_lt.py):
  PYTHONPATH=<language-table>:$PWD:$PWD/swm-next \
  python swm-next/swm_lora/eval_closed_loop_lt.py \
      --ckpt outputs_swm_lora/lt1_full/ckpt_step24000 \
      --config swm-next/configs/verifier_mpc_lt_green_cube_blue_moon.yaml \
      --train-config swm-next/swm_lora/train_lt1.yaml \
      --episodes 50 --seed-start 6000 --k 8 \
      --out experiments/lt1_larg/eval_cl_step24000_gcbm_k8.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "verifier"))

from swm.constants import ANSWER_OPTIONS
from swm.diffusion_policy import DiffusionPolicy

from datagen import jpeg                                   # noqa: E402
from mpc import LTProbe, build_lt_questions, make_lt_env   # noqa: E402
from data import action_range, decode_frame, normalize     # noqa: E402


def run_episode(env, goal, swm, cfg, seed, k, temps, amin, amax, image_size,
                scorer):
    # the per-cycle seed below is mpc.run_lt_episode's, so one seed gives the
    # teacher's proposals wherever the episodes have not yet diverged
    torch.manual_seed((seed * 7919 + 17) % 2**31)
    np.random.seed((seed * 7919 + 29) % 2**31)
    rng = np.random.RandomState(seed)          # --scorer random picks
    env.env._rng.seed(seed)                    # scene layout (see mpc.py)
    frame = goal.reset_env(seed=seed)
    policy = DiffusionPolicy.load(cfg["diffusion_path"], device=cfg["device"])
    policy.add_obs(frame)
    (q_touch, w_touch, _), (q_closer, w_closer, _) = build_lt_questions(goal)
    probe = LTProbe(env, goal)
    n_exec = cfg["actions_per_cycle"]
    ns = int(cfg.get("judge_native_scale", 1))
    H = swm.horizon if swm is not None else None

    secs = dict(policy=0.0, swm=0.0, sim=0.0)
    log = []
    done = False
    cycle = -1
    for cycle in range(cfg["max_cycles"]):
        torch.manual_seed(hash((seed, cycle)) % 2**31)
        t0 = time.perf_counter()
        candidates = np.asarray(policy.sample_trajs(k, temperatures=temps),
                                dtype=np.float32)
        secs["policy"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        entry = dict(cycle=cycle)
        if scorer == "random":
            pick = int(rng.randint(k))
        else:
            assert candidates.shape[1] >= H and candidates.shape[2] == len(amin), \
                (candidates.shape, H, len(amin))
            anchor = env.get_judge_frame(ns)
            img = decode_frame(jpeg(anchor).tobytes(), image_size)
            trajs = torch.from_numpy(normalize(candidates[:, :H], amin, amax))
            p_touch = swm.p_yes([img] * k, [q_touch] * k, trajs).cpu().numpy()
            p_closer = swm.p_yes([img] * k, [q_closer] * k, trajs).cpu().numpy()
            scores = w_touch * p_touch + w_closer * p_closer
            pick = int(np.argmax(scores))
            entry.update(
                scores=[round(float(s), 4) for s in scores],
                p_touch=[round(float(s), 4) for s in p_touch],
                p_closer=[round(float(s), 4) for s in p_closer])
        secs["swm"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        for a in candidates[pick][:n_exec]:
            frame = env.step(np.asarray(a))
            policy.add_obs(frame)
            if goal.get_done():
                done = True
                break
        secs["sim"] += time.perf_counter() - t0
        entry.update(pick=pick, oracle=probe.read())
        log.append(entry)
        if done:
            break
    return done, cycle + 1, log, secs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint dir (not needed with --scorer random)")
    ap.add_argument("--config", required=True,
                    help="LangTable env config (verifier_mpc_lt_<task>.yaml)")
    ap.add_argument("--train-config", required=True, dest="train_config",
                    help="training yaml (model dims, image_size, action_range)")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed-start", type=int, default=6000, dest="seed_start")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--scorer", default="swm", choices=["swm", "random"])
    ap.add_argument("--swm-device", default="cuda", dest="swm_device")
    ap.add_argument("--set", action="append", default=[], dest="overrides",
                    metavar="KEY=VALUE",
                    help="override an env config key (yaml-parsed value), "
                         "e.g. device=cuda for the base policy")
    ap.add_argument("--out", required=True, help="result json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    for kv in args.overrides:
        key, val = kv.split("=", 1)
        cfg[key] = yaml.safe_load(val)
    assert cfg.get("env_type") == "lang_table", "LangTable configs only"
    assert args.k > 1, "the ladder needs k > 1"
    tcfg = yaml.safe_load(open(args.train_config))
    assert int(tcfg["action_dim"]) == 2, "not a LangTable training config"
    amin, amax = action_range(tcfg)
    image_size = tuple(tcfg["image_size"]) if tcfg.get("image_size") else None
    temps = [i * 2.0 / (args.k - 1) for i in range(args.k)]

    swm = None
    if args.scorer == "swm":
        assert args.ckpt, "--ckpt is required with --scorer swm"
        from model import QwenSWM
        swm = QwenSWM(model_id=tcfg["model_id"], action_dim=tcfg["action_dim"],
                      horizon=tcfg["horizon"], lora_r=tcfg["lora_r"],
                      lora_alpha=tcfg["lora_alpha"], lora_dropout=0.0,
                      device=args.swm_device,
                      proj_init_scale=float(tcfg.get("proj_init_scale", 0.1)),
                      traj_cumsum=bool(tcfg.get("traj_cumsum", False)),
                      # must match the checkpoint: the LoRA state dict loads
                      # non-strict, so a target mismatch drops weights silently
                      lora_targets=tcfg.get("lora_targets", "all-linear"),
                      proj_hidden=tcfg.get("proj_hidden", 512),
                      proj_learnable_scale=bool(tcfg.get("learnable_scale", True)))
        swm.load_adapters(args.ckpt)
        swm.model.eval()

    from swm.utils.goal_generators import get_lang_table_goal
    env = make_lt_env(cfg)
    goal = get_lang_table_goal("block_to_block", env, None, ANSWER_OPTIONS,
                               {"block_combo": list(cfg["block_combo"]),
                                "ood": bool(cfg.get("ood", False))})

    meta = dict(ckpt=args.ckpt, scorer=args.scorer, env="lang_table",
                task="_".join(cfg["block_combo"]),
                block_combo=list(cfg["block_combo"]), k=args.k, temps=temps,
                image_size=list(image_size) if image_size else None,
                action_range=dict(min=[float(v) for v in amin],
                                  max=[float(v) for v in amax]),
                policy_device=str(cfg["device"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # a 50-episode block is long: keep each finished episode so a restarted
    # block picks up where it stopped (episodes reseed everything themselves)
    part = args.out + ".partial.jsonl"
    results = []
    if os.path.exists(part):
        lines = [json.loads(l) for l in open(part) if l.strip()]
        assert lines and lines[0].get("meta") == meta, \
            f"{part} belongs to another run: move it away first"
        results = lines[1:]
        print(f"resuming: {len(results)} episodes from {part}", flush=True)
    else:
        with open(part, "w") as fh:
            fh.write(json.dumps(dict(meta=meta)) + "\n")
    have = {r["seed"] for r in results}
    wins = sum(r["success"] for r in results)
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        if seed in have:
            continue
        t0 = time.perf_counter()
        with torch.no_grad():
            success, cycles, log, secs = run_episode(
                env, goal, swm, cfg, seed, args.k, temps, amin, amax,
                image_size, args.scorer)
        wall = time.perf_counter() - t0
        wins += success
        results.append(dict(seed=seed, success=bool(success), cycles=cycles,
                            log=log))
        with open(part, "a") as fh:
            fh.write(json.dumps(results[-1]) + "\n")
        print(f"[{len(results)}/{args.episodes}] seed {seed}: "
              f"{'success' if success else 'fail'} @ {cycles} "
              f"(SR {wins/len(results):.0%}) {wall:.0f}s [policy "
              f"{secs['policy']:.0f} swm {secs['swm']:.0f} sim "
              f"{secs['sim']:.0f}]", flush=True)

    results.sort(key=lambda r: r["seed"])
    summary = dict(meta, episodes=args.episodes, seed_start=args.seed_start,
                   sr=wins / args.episodes, episodes_detail=results)
    json.dump(summary, open(args.out, "w"), indent=1)
    os.remove(part)
    print(f"EVAL_DONE sr={summary['sr']:.2%}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
