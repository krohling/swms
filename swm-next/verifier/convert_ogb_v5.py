"""exp5: OGBench-oracle play episodes -> v3-compatible cycle HDF5 + judge labels.

Provenance (Jacob B., email 2026-08-04): the paper's OGBench world-model data
was NOT the public OGBench download; it was regenerated with his
`collect_demos.py` (OGBench planning oracles, `set_new_target` chaining,
p_stack ~ U(0.1, 0.5), play noise 0.1 / smoothing 0.5, 200-step episodes,
224 px, control_timestep 0.1, OGBench's default semi-transparent arm — the same
alphas our `make_env(arm_alpha=0.1)` reproduces). exp5 runs that script
UNCHANGED (krohling/ogbench fork; only episode count / seed offset / worker
count were exposed as flags) and this converter turns its per-episode pickles
into the cycle schema the pointwise trainer already reads (CycleDataset /
ProposalDataset / episode_split), labeled by OUR Qwen teacher with the same
three planner probes and the same batched `p_yes` call as datagen_v3.py.

Per pickle: frames (T=200, 224, 224, 3) uint8, actions (T-1=199, 5) clipped
oracle actions, qa_pairs (T, 32) ground-truth (question, answer, type) tuples
[grasp x4, gripper-touching x4, touching x12 ordered pairs, ontop x12 ordered
pairs], block_states (T) dicts of block/eef poses, color_lookup {cube idx:
color}. Frame i (i>=1) is the observation before action i (frame 0 is the
pre-settle reset frame, so windows start at t=1).

Window = cycle: t in {1, 5, ..., 181} (stride 4 = v3's actions_per_cycle, 46
windows/episode): frames_t = frame[t], trajs = actions[t:t+16], end_frames =
frame[t+16]. Labels exactly as v3: p_approach / p_place judged on the end
frame, the hold/gate probe on frame_t; gate = EMA(gate_raw, 0.35) over the
episode's windows in order; score = (1-g)*p_app + g*(0.6*p_place + 0.4*p_hold).

HDF5 (schema_version 5 — a strict superset of v3's TRAINING VIEW):
  ep_<seed>/frames_t, trajs (C,1,16,5), end_frames (C,1), labels (C,1,3),
            pick (=0), propensity (=1), proposal_trajs (=trajs),
            proposal_temps (C,1) NaN, gate_raw, gate, score (C,1)   [v3 names]
  ep_<seed>/window_start (C,) i16 ; state_t_poses (C,P) ; end_poses (C,1,P)
            [block_{0..3}_pos/quat + eef_pos/quat flattened exactly as the
             pickles carry them; attrs pose_layout = JSON {key: length} —
             the pickles carry poses, not qpos/qvel]
  ep_<seed>/gt_t (C,32) bool, gt_end (C,32) bool  simulator ground truth for
            all 32 questions at frame_t / end frame (types in root attrs)
  ep_<seed>/gt_place, gt_grasp, gt_place_t, gt_grasp_t (C,) bool  the task
            pair's ontop / grasp truth (end frame / frame_t)
  attrs: seed (env seed), policy="oracle_play" (new episode_split stratum),
         arm, success=False, cycles, q_approach/q_place/q_hold, color_lookup,
         source_pickle, episode_index
Root attrs mirror v3 (task, k=1, lookahead, model_id, recipe JSON) plus the
collect_demos parameters and the ogbench fork commit.

Usage (m5):
  PYTHONPATH=$PWD:$PWD/swm-next .venv/bin/python swm-next/verifier/convert_ogb_v5.py \
    --config swm-next/configs/datagen_v5_ogbplay.yaml \
    --pickles datasets_swm/exp5_raw/play/trajectories --seed-offset 70000 \
    --out datasets_swm/bg_v5_ogbplay_s70000.h5
  ... --verify datasets_swm/bg_v5_ogbplay_s70000.h5      (invariants + stats)
Resumable: existing ep_<seed> groups are skipped.
"""
import argparse
import glob
import json
import os
import pickle
import subprocess
import sys
import time

import h5py
import numpy as np
import yaml

from datagen import GATE_ALPHA, HOLD_W, PLACE_W, jpeg  # noqa: E402 (same dir)
from mpc import build_questions  # noqa: E402

EVAL_SEEDS = range(6000, 6100)
H, STRIDE, START = 16, 4, 1
N_CUBES = 4
# qa_pairs layout per step (collect_demos.py): grasp x4, gripper-touching x4,
# touching x12, ontop x12 — ordered pairs in nested (i, j != i) loop order.
OFF_GRASP, OFF_TOUCHING, OFF_ONTOP, N_GT = 0, 8, 20, 32


def pair_index(i, j):
    """Position of ordered pair (i, j), i != j, in the nested-loop order."""
    return i * (N_CUBES - 1) + (j if j < i else j - 1)


def color_index(lookup, color):
    hits = [int(k) for k, v in lookup.items() if color in str(v)]
    assert len(hits) == 1, f"color {color!r} not unique in lookup {lookup}"
    return hits[0]


def fork_commit(repo_hint):
    try:
        import ogbench
        repo = os.path.dirname(os.path.dirname(os.path.abspath(ogbench.__file__)))
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "--short",
                                        "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return repo_hint or "unknown"


def load_episode(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    frames = np.stack([np.asarray(x, dtype=np.uint8) for x in d["frames"]])
    actions = np.asarray(d["actions"], dtype=np.float32)
    assert frames.ndim == 4 and frames.shape[1:] == (224, 224, 3), frames.shape
    assert actions.shape == (frames.shape[0] - 1, 5), actions.shape
    qa = d["qa_pairs"]
    assert len(qa) == frames.shape[0] and all(len(q) == N_GT for q in qa), \
        (len(qa), {len(q) for q in qa})
    keys = list(d["block_states"][0].keys())
    layout = {k: int(np.asarray(d["block_states"][0][k]).size) for k in keys}
    poses = np.stack([np.concatenate([np.asarray(p[k], np.float32).ravel()
                                      for k in keys]) for p in d["block_states"]])
    assert poses.shape == (frames.shape[0], sum(layout.values())), poses.shape
    gt = np.array([[bool(a) for (_, a, _) in step] for step in qa], dtype=bool)
    types = [t for (_, _, t) in qa[0]]
    return dict(frames=frames, actions=actions, gt=gt, types=types, poses=poses,
                pose_layout=layout,
                lookup={int(k): str(v) for k, v in d["color_lookup"].items()})


def window_starts(T):
    # need actions[t:t+H] (t+H <= T-1) and frame[t+H] (t+H <= T-1)
    return [t for t in range(START, T - H, STRIDE)]


def label_windows(judge, frames, starts, q_app, q_place, q_hold, batch):
    """All of an episode's probes, batched like v3's per-cycle call
    ([end, end, frame_t] x [q_app, q_place, q_hold]) but across windows."""
    images, qs = [], []
    for t in starts:
        images += [frames[t + H], frames[t + H], frames[t]]
        qs += [q_app, q_place, q_hold]
    batch = max(3, (batch // 3) * 3)
    out = []
    for i in range(0, len(images), batch):
        p, _ = judge.p_yes(images[i:i + batch], qs[i:i + batch])
        out.append(np.asarray(p, dtype=np.float32))
    p = np.concatenate(out)
    return p.reshape(len(starts), 3)          # [p_app, p_place, hold-on-frame_t]


def write_episode(h5, ep, env_seed, ep_index, src, p3, starts, q_app, q_place,
                  q_hold, arm, blue, green):
    C = len(starts)
    frames, actions, gt, poses = ep["frames"], ep["actions"], ep["gt"], ep["poses"]
    p_app, p_place, rg = p3[:, 0], p3[:, 1], p3[:, 2]
    p_hold = p_app                                   # v3: q_hold == q_app
    gate = np.empty(C, np.float32)
    ema = None
    for i, r in enumerate(rg):
        ema = float(r) if ema is None else GATE_ALPHA * float(r) + (1 - GATE_ALPHA) * ema
        gate[i] = ema
    score = (1.0 - gate) * p_app + gate * (PLACE_W * p_place + HOLD_W * p_hold)

    grp = h5.create_group(f"ep_{env_seed}")
    vlen = h5py.vlen_dtype(np.uint8)
    d = grp.create_dataset("frames_t", (C,), dtype=vlen)
    e = grp.create_dataset("end_frames", (C, 1), dtype=vlen)
    for i, t in enumerate(starts):
        d[i] = jpeg(frames[t])
        e[i, 0] = jpeg(frames[t + H])
    trajs = np.stack([actions[t:t + H] for t in starts]).astype(np.float32)[:, None]
    grp.create_dataset("trajs", data=trajs)
    grp.create_dataset("labels", data=np.stack(
        [p_app, p_place, p_hold], axis=-1).astype(np.float32)[:, None])
    grp.create_dataset("pick", data=np.zeros(C, np.int8))
    grp.create_dataset("propensity", data=np.ones(C, np.float32))
    grp.create_dataset("proposal_trajs", data=trajs)
    grp.create_dataset("proposal_temps", data=np.full((C, 1), np.nan, np.float32))
    grp.create_dataset("gate_raw", data=rg.astype(np.float32))
    grp.create_dataset("gate", data=gate)
    grp.create_dataset("score", data=score.astype(np.float32)[:, None])
    st = np.asarray(starts)
    grp.create_dataset("window_start", data=st.astype(np.int16))
    grp.create_dataset("state_t_poses", data=poses[st])
    grp.create_dataset("end_poses", data=poses[st + H][:, None])
    grp.create_dataset("gt_t", data=gt[st])
    grp.create_dataset("gt_end", data=gt[st + H])
    k_place = OFF_ONTOP + pair_index(blue, green)
    grp.create_dataset("gt_place", data=gt[st + H, k_place])
    grp.create_dataset("gt_grasp", data=gt[st + H, OFF_GRASP + blue])
    grp.create_dataset("gt_place_t", data=gt[st, k_place])
    grp.create_dataset("gt_grasp_t", data=gt[st, OFF_GRASP + blue])
    grp.attrs.update(dict(
        seed=int(env_seed), policy="oracle_play", arm=arm, success=False,
        cycles=C, q_approach=q_app, q_place=q_place, q_hold=q_hold,
        color_lookup=json.dumps(ep["lookup"]), source_pickle=os.path.basename(src),
        episode_index=int(ep_index), gt_types=json.dumps(ep["types"]),
        pose_layout=json.dumps(ep["pose_layout"])))
    h5.flush()
    return dict(p_app=p_app, p_place=p_place, gt_place=gt[st + H, k_place],
                gt_grasp=gt[st + H, OFF_GRASP + blue])


def summarize(acc):
    pa, pp = np.concatenate(acc["p_app"]), np.concatenate(acc["p_place"])
    gp, gg = np.concatenate(acc["gt_place"]), np.concatenate(acc["gt_grasp"])
    n = len(pa)
    lines = [f"cycles {n:,}",
             f"approach: mean p {pa.mean():.3f} | frac p>0.5 {np.mean(pa > 0.5):.3%}"
             f" | GT blue grasped (end) {gg.mean():.3%}",
             f"place:    mean p {pp.mean():.3f} | frac p>0.5 {np.mean(pp > 0.5):.3%}"
             f" | GT blue-on-green (end) {gp.mean():.3%}"]
    for name, p, g in (("approach vs GT grasp", pa, gg), ("place vs GT ontop", pp, gp)):
        if 0 < g.sum() < len(g):
            # Mann-Whitney AUC of the judge probability against simulator truth
            r = p.argsort().argsort().astype(np.float64) + 1
            auc = (r[g].sum() - g.sum() * (g.sum() + 1) / 2) / (g.sum() * (~g).sum())
            lines.append(f"{name}: mean p | GT yes {p[g].mean():.3f}, | GT no "
                         f"{p[~g].mean():.3f}, AUC {auc:.3f} (n_yes {int(g.sum())})")
    return "\n".join(lines)


def verify(path):
    f = h5py.File(path, "r")
    assert int(f.attrs["schema_version"]) == 5, f.attrs.get("schema_version")
    rec = json.loads(f.attrs["recipe"])
    alpha, pw, hw = rec["gate_alpha"], rec["place_w"], rec["hold_w"]
    acc = dict(p_app=[], p_place=[], gt_place=[], gt_grasp=[])
    from PIL import Image
    import io
    n_eps = 0
    for ep in f:
        g = f[ep]
        C = g["labels"].shape[0]
        assert int(g.attrs["seed"]) not in EVAL_SEEDS, f"{ep}: eval seed"
        assert g.attrs["policy"] == "oracle_play"
        assert g["trajs"].shape == (C, 1, H, 5) and g["labels"].shape == (C, 1, 3)
        assert g["frames_t"].shape == (C,) and g["end_frames"].shape == (C, 1)
        assert g["proposal_trajs"].shape == (C, 1, H, 5)
        P = sum(json.loads(g.attrs["pose_layout"]).values())
        assert g["state_t_poses"].shape == (C, P) and g["end_poses"].shape == (C, 1, P)
        assert g["gt_t"].shape == (C, N_GT) and g["gt_end"].shape == (C, N_GT)
        lab, gate, score = g["labels"][:, 0], g["gate"][:], g["score"][:, 0]
        assert ((lab >= 0) & (lab <= 1)).all()
        assert np.allclose(lab[:, 0], lab[:, 2]), f"{ep}: p_hold != p_app"
        want = (1 - gate) * lab[:, 0] + gate * (pw * lab[:, 1] + hw * lab[:, 2])
        assert np.allclose(want, score, atol=1e-5), f"{ep}: score != blend"
        ema, chk = None, []
        for r in g["gate_raw"][:]:
            ema = float(r) if ema is None else alpha * float(r) + (1 - alpha) * ema
            chk.append(ema)
        assert np.allclose(chk, gate, atol=1e-6), f"{ep}: gate != EMA"
        ws = g["window_start"][:]
        assert (np.diff(ws) == STRIDE).all() and ws[0] == START
        assert np.abs(g["trajs"][:]).max() <= 1.0 + 1e-6
        im = Image.open(io.BytesIO(g["frames_t"][0].tobytes())).convert("RGB")
        assert im.size == (224, 224)
        im = Image.open(io.BytesIO(g["end_frames"][C - 1, 0].tobytes())).convert("RGB")
        assert im.size == (224, 224)
        acc["p_app"].append(lab[:, 0]); acc["p_place"].append(lab[:, 1])
        acc["gt_place"].append(g["gt_place"][:]); acc["gt_grasp"].append(g["gt_grasp"][:])
        n_eps += 1
    assert n_eps, f"{path}: no episodes"
    print(f"VERIFY_OK {path}: {n_eps} episodes\n" + summarize(acc), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="datagen_v5 yaml (model_id, device, block_combo, judge_batch)")
    ap.add_argument("--pickles", help="collect_demos.py <workdir>/trajectories dir")
    ap.add_argument("--seed-offset", type=int, default=None,
                    help="env seed = pickle index + offset (the --seed-offset given to collect_demos)")
    ap.add_argument("--out")
    ap.add_argument("--batch", type=int, default=None, help="judge probes per forward")
    ap.add_argument("--limit", type=int, default=0, help="smoke: stop after N episodes")
    ap.add_argument("--verify", default=None, metavar="H5", help="verify an existing file and exit")
    args = ap.parse_args()
    if args.verify:
        return verify(args.verify)
    assert args.config and args.pickles and args.out and args.seed_offset is not None

    cfg = yaml.safe_load(open(args.config))
    batch = args.batch or int(cfg.get("judge_batch", 24))
    tmpl = build_questions(cfg)
    qcfg = cfg.get("questions") or {}
    q_app = qcfg.get("approach", tmpl["grasp"])
    q_place = qcfg.get("place", tmpl["ontop"])
    q_hold = qcfg.get("hold", q_app)
    assert q_hold == q_app, "v5 mirrors v3a's 3-probe layout (q_hold == q_app)"
    top, bottom = cfg["block_combo"]                 # e.g. blue_cube, green_cube
    top_color, bottom_color = top.split("_")[0], bottom.split("_")[0]
    arm = "v5_ogbench_play"

    files = sorted(glob.glob(os.path.join(args.pickles, "*.pkl")),
                   key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    assert files, f"no pickles under {args.pickles}"
    if args.limit:
        files = files[:args.limit]
    for p in files:
        s = int(os.path.splitext(os.path.basename(p))[0]) + args.seed_offset
        assert s not in EVAL_SEEDS, f"{p}: env seed {s} in eval range"

    from label_teacher import QwenJudge
    judge = QwenJudge(model_id=cfg["model_id"], device=cfg["device"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    mode = "a" if os.path.exists(args.out) else "w"
    acc = dict(p_app=[], p_place=[], gt_place=[], gt_grasp=[])
    with h5py.File(args.out, mode) as h5:
        if mode == "w":
            h5.attrs.update(dict(
                task="_".join(cfg["block_combo"]), k=1, n_proposed=1,
                pick_rule="oracle", temp_schedule="none", lookahead=H,
                temps=[np.nan], arm=arm, model_id=cfg["model_id"], schema_version=5,
                recipe=json.dumps(dict(
                    config=cfg, source="ogbench/collect_demos.py --dataset-type play",
                    ogbench_fork_commit=fork_commit(cfg.get("ogbench_commit")),
                    collect=dict(env="visual-cube-quadruple-v0", max_episode_steps=200,
                                 noise=0.1, noise_smoothing=0.5, min_norm=0.4,
                                 p_random_action=0.0, p_stack="U(0.1,0.5)",
                                 control_timestep=0.1, width=224, height=224,
                                 pixel_transparent_arm="OGBench default (True)",
                                 seed_offset=args.seed_offset),
                    window=dict(H=H, stride=STRIDE, start=START),
                    gate_alpha=GATE_ALPHA, place_w=PLACE_W, hold_w=HOLD_W,
                    q_approach=q_app, q_place=q_place, q_hold=q_hold))))
        t0 = time.time()
        done = 0
        for n, path in enumerate(files):
            idx = int(os.path.splitext(os.path.basename(path))[0])
            env_seed = idx + args.seed_offset
            if f"ep_{env_seed}" in h5:
                print(f"[{n+1}/{len(files)}] ep_{env_seed} exists, skip", flush=True)
                continue
            ep = load_episode(path)
            blue = color_index(ep["lookup"], top_color)
            green = color_index(ep["lookup"], bottom_color)
            starts = window_starts(ep["frames"].shape[0])
            te = time.time()
            p3 = label_windows(judge, ep["frames"], starts, q_app, q_place, q_hold, batch)
            r = write_episode(h5, ep, env_seed, idx, path, p3, starts, q_app, q_place,
                              q_hold, arm, blue, green)
            for k2 in acc:
                acc[k2].append(r[k2])
            done += 1
            el = time.time() - t0
            print(f"[{n+1}/{len(files)}] ep_{env_seed}: {len(starts)} windows, "
                  f"{time.time()-te:.1f}s | app>0.5 {np.mean(r['p_app']>0.5):.0%} "
                  f"place>0.5 {np.mean(r['p_place']>0.5):.0%} | GT ontop "
                  f"{r['gt_place'].mean():.0%} grasp {r['gt_grasp'].mean():.0%} | "
                  f"{el/done:.1f}s/ep, ETA {(len(files)-n-1)*el/done/3600:.1f}h",
                  flush=True)
    if done:
        print("SUMMARY (this run)\n" + summarize(acc), flush=True)
    print(f"DATAGEN_DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
