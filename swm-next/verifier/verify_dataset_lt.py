"""Verifier for a datagen_lt.py unit file (schema_version 3, variant lt1).

Structure and label checks need only h5py. --replay N also needs the
simulator: for N cycles of every episode the stored state at t is restored,
the stored frame is re-rendered and the picked chunk is rolled out again,
then compared with what the file holds. That replay starts from set_state
alone (the engine snapshot of the generating process is gone), which is not
a complete restore, so it guards only against a stored state, chunk or frame
that belongs to another cycle; repeatability itself is the generator's own
--restore-check-cycles (a rollout repeated from one engine snapshot, held to
1e-4 m; see datagen_lt.py), whose per-episode result this verifier reads.

Prints one line per check, then a STATS json line; exit code 1 on any FAIL.

  python swm-next/verifier/verify_dataset_lt.py <unit>.h5 \
      --task green_cube_blue_moon --seed-start 200000 --episodes 54 --replay 2
"""
import argparse
import io
import json
import sys

import h5py
import numpy as np
from PIL import Image

EXPECT = dict(  # dataset -> trailing shape after the cycle axis (None = vlen)
    frames_t=None, state_t_pkl=None, end_frames=(1,), end_state_pkl=(1,),
    labels=(1, 3), label_mass=(1, 2), score=(1,), pick=(), propensity=(),
    proposal_min_sep=(), oracle_t=(8,), oracle_end=(1, 8), end_done=(1,))
DONE_DIST = 0.065          # the task's done test (block centers)
EDGE = 1e-4                # stored distances are float32: ignore ties at the line
FRAME_TOL = 4.0            # mean |re-render - stored JPEG| per channel, 0-255
# Replayed end state vs stored. A replay from set_state alone drifts (LARG,
# Nov 26, 288 replays against the snapshot rollout, 6 seeds, 5 tasks: median
# 5e-4 m, 96% within 5 mm, max 1.1 cm; per-seed medians 3e-6 to 9e-4), while
# another proposal's chunk ends a median 1.9 cm away and within 1 cm in only
# 23% of 36,729 cases. A unit of the smoke replays 8 cycles: resampled
# medians of 8 stay under 3.1 mm in 99.9% of draws (of 108, a batch unit,
# under 0.9 mm), and 6 of 8 within 1 cm is the worst draw. So: the median
# under 5 mm, and most cycles near.
END_MEDIAN_TOL = 5e-3      # m
END_TOL = 0.01             # m
END_NEAR_SHARE = 0.75


class Report:
    def __init__(self):
        self.bad = 0

    def __call__(self, ok, name, detail=""):
        self.bad += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'} {name}{': ' if detail else ''}"
              f"{detail}", flush=True)
        return ok


def decode(blob):
    return np.asarray(Image.open(io.BytesIO(
        np.asarray(blob, np.uint8).tobytes())).convert("RGB"), np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--replay", type=int, default=0,
                    help="cycles per episode to replay in the simulator")
    ap.add_argument("--allow-dry-run", action="store_true")
    args = ap.parse_args()
    rep = Report()

    with h5py.File(args.path, "r") as h5:
        a = dict(h5.attrs)
        recipe = json.loads(a.get("recipe", "{}"))
        k = int(a.get("n_proposed", 0))
        L = int(a.get("lookahead", 0))
        rep(a.get("schema_version") == 3 and a.get("schema_variant") == "lt1"
            and a.get("env") == "lang_table" and a.get("k") == 1,
            "file attrs schema")
        rep(a.get("task") == args.task, "file attrs task", str(a.get("task")))
        rep(bool(a.get("dry_run")) is False or args.allow_dry_run,
            "not a dry run")
        temps = np.asarray(a.get("temps", []), np.float32)
        rep(k > 1 and len(temps) == k and np.allclose(
            temps, [i * 2.0 / (k - 1) for i in range(k)]),
            "temperature ladder", f"k={k}")
        rep(a.get("pick_rule") == "uniform" and a.get("frame_codec") == "jpeg_q90",
            "pick rule and frame codec")
        cfg = recipe.get("config", {})
        max_cycles = int(cfg.get("max_cycles", 0))
        n_exec = int(cfg.get("actions_per_cycle", 0))

        want = [f"ep_{s}" for s in range(args.seed_start,
                                         args.seed_start + args.episodes)]
        have = sorted(h5.keys())
        rep(sorted(want) == have, "episode set",
            f"{len(have)} groups, expected {len(want)}")

        n_cyc = n_succ = n_chk = n_chk_bad = n_chk_eps = 0
        chk_pos = 0.0
        picks = np.zeros(max(k, 1), np.int64)
        P, M, O_t, O_end, DONE = [], [], [], [], []
        struct_bad, value_bad, chain_bad = [], [], []
        frame_bytes = 0
        questions = set()
        names_ok = [n for n in want if n in h5]
        for name in names_ok:
            g = h5[name]
            ga = dict(g.attrs)
            C = int(ga.get("cycles", -1))
            miss = [d for d in list(EXPECT) + ["trajs", "proposal_trajs",
                                               "proposal_temps"] if d not in g]
            if miss or C < 1:
                struct_bad.append(f"{name} missing {miss} cycles {C}")
                continue
            for d, tail in EXPECT.items():
                shp = g[d].shape
                if shp != (C,) + (tail or ()):
                    struct_bad.append(f"{name}/{d} shape {shp}")
            H, A = g["trajs"].shape[2:]
            if g["trajs"].shape != (C, 1, H, A) or H < L or \
                    g["proposal_trajs"].shape != (C, k, H, A) or \
                    g["proposal_temps"].shape != (C, k):
                struct_bad.append(f"{name} traj shapes")
                continue
            lab = g["labels"][:]
            mass = g["label_mass"][:]
            pick = g["pick"][:].astype(int)
            trajs, props = g["trajs"][:], g["proposal_trajs"][:]
            succ = bool(ga.get("success"))
            if not (C <= max_cycles and (succ or C == max_cycles)):
                value_bad.append(f"{name} cycles {C} success {succ}")
            fd = float(ga.get("final_dist", 9))
            if succ != (fd < DONE_DIST) and abs(fd - DONE_DIST) > EDGE:
                value_bad.append(f"{name} success vs final_dist")
            if int(ga.get("seed", -1)) != int(name[3:]) or ga.get("policy") != "random":
                value_bad.append(f"{name} seed/policy attr")
            if ga.get("q_approach") != ga.get("q_touch") or \
                    ga.get("q_place") != ga.get("q_closer"):
                value_bad.append(f"{name} question attrs")
            questions.add((ga.get("q_touch"), ga.get("q_closer")))
            if not a.get("dry_run"):
                if not (np.isfinite(lab).all() and lab.min() >= 0 and lab.max() <= 1):
                    value_bad.append(f"{name} labels outside [0, 1]")
                if not np.array_equal(lab[:, :, 2], lab[:, :, 0]):
                    value_bad.append(f"{name} hold column != touch column")
                if not (np.isfinite(mass).all() and mass.min() > 0.5
                        and mass.max() <= 1.001):
                    value_bad.append(f"{name} judge mass {mass.min():.3f}")
                sc = 0.8 * lab[:, 0, 0] + 0.2 * lab[:, 0, 1]
                if not np.allclose(g["score"][:, 0], sc, atol=1e-5):
                    value_bad.append(f"{name} score")
            if pick.min() < 0 or pick.max() >= k or not np.array_equal(
                    trajs[:, 0], props[np.arange(C), pick]):
                value_bad.append(f"{name} pick / picked chunk")
            if not np.allclose(g["proposal_temps"][:], temps[None]):
                value_bad.append(f"{name} proposal temps")
            if not np.allclose(g["propensity"][:], 1.0 / k):
                value_bad.append(f"{name} propensity")
            ot, oe = g["oracle_t"][:], g["oracle_end"][:, 0]
            if not (np.isfinite(ot).all() and np.isfinite(oe).all()):
                value_bad.append(f"{name} oracle not finite")
            # the first action of the rollout is the first executed action, so
            # the blocks cannot be further from their state-t place after the
            # executed steps than a table is wide: a coarse chain check that
            # catches cycles stored out of order
            if C > 1 and np.abs(np.diff(ot[:, :4], axis=0)).max() > 0.25:
                chain_bad.append(f"{name} block jump between cycles")
            if ((g["end_done"][:, 0] != (oe[:, 6] < DONE_DIST))
                    & (np.abs(oe[:, 6] - DONE_DIST) > EDGE)).any():
                value_bad.append(f"{name} end_done vs end distance")
            for d in ("frames_t", "end_frames"):
                frame_bytes += sum(len(x) for x in g[d][:].ravel())
            np.add.at(picks, pick, 1)
            n_cyc += C
            n_succ += succ
            n_chk += int(ga.get("check_cycles", 0))
            n_chk_bad += int(ga.get("check_bad", 0))
            n_chk_eps += int(ga.get("check_cycles", 0)) > 0
            chk_pos = max(chk_pos, float(ga.get("check_max_pos", 0.0)))
            P.append(lab[:, 0, :2]); M.append(mass[:, 0])
            O_t.append(ot); O_end.append(oe); DONE.append(g["end_done"][:, 0])

        rep(not struct_bad, "datasets and shapes", "; ".join(struct_bad[:3]))
        rep(not value_bad, "values", "; ".join(value_bad[:3]))
        rep(not chain_bad, "cycle order", "; ".join(chain_bad[:3]))
        rep(len(questions) == 1, "one question pair per file",
            str(sorted(questions))[:160])
        # every episode carries the generator's own repeatability gate
        rep((n_chk_eps == len(names_ok) or args.allow_dry_run)
            and n_chk_bad == 0,
            "generator's restore check",
            f"{n_chk} cycles in {n_chk_eps} of {len(names_ok)} episodes, "
            f"{n_chk_bad} over the limits, max any-body pos diff {chk_pos:.1e} m")
        if n_cyc:
            exp = n_cyc / k
            chi2 = float(((picks - exp) ** 2 / exp).sum())
            # chi-square with k-1 dof: 40 is p ~ 1e-6 at k = 8
            rep(chi2 < 40, "uniform pick", f"counts {picks.tolist()} chi2 {chi2:.1f}")

        # decode a sample of frames
        ns = int(a.get("judge_native_scale", 1))
        dec_bad = []
        names = [n for n in want if n in h5]
        for name in names[:: max(1, len(names) // 6)]:
            g = h5[name]
            for c in {0, int(g.attrs["cycles"]) - 1}:
                f0, f1 = decode(g["frames_t"][c]), decode(g["end_frames"][c, 0])
                if f0.shape != (180 * ns, 320 * ns, 3) or f1.shape != f0.shape:
                    dec_bad.append(f"{name} c{c} {f0.shape} {f1.shape}")
        rep(not dec_bad, "frames decode at judge scale", "; ".join(dec_bad[:3]))

        replay = None
        if args.replay > 0 and names and not struct_bad:
            replay = run_replay(h5, names, cfg, ns, L, args.replay, rep)

        stats = dict(path=args.path, task=args.task, episodes=len(have),
                     success_rate=round(n_succ / max(len(have), 1), 4),
                     cycles=n_cyc, examples=2 * n_cyc,
                     cycles_per_episode=round(n_cyc / max(len(have), 1), 2),
                     restore_checks=n_chk, restore_check_max_pos_m=chk_pos,
                     frame_kb_per_cycle=round(frame_bytes / max(n_cyc, 1) / 1024, 1))
        if n_cyc and not a.get("dry_run"):
            P, M = np.concatenate(P), np.concatenate(M)
            O_t, O_end = np.concatenate(O_t), np.concatenate(O_end)
            DONE = np.concatenate(DONE)
            closer = O_end[:, 6] < O_t[:, 6] - 1e-4
            stats.update(
                touch_pos_share=round(float((P[:, 0] > 0.5).mean()), 4),
                touch_mean=round(float(P[:, 0].mean()), 4),
                closer_pos_share=round(float((P[:, 1] > 0.5).mean()), 4),
                closer_mean=round(float(P[:, 1].mean()), 4),
                mass_mean=[round(float(x), 4) for x in M.mean(0)],
                sim_done_share=round(float(DONE.mean()), 4),
                sim_closer_share=round(float(closer.mean()), 4),
                # judge vs simulator truth (the judge never sees these)
                touch_agrees_done=round(float(((P[:, 0] > 0.5) == DONE).mean()), 4),
                touch_mean_when_done=round(float(P[DONE, 0].mean()), 4) if DONE.any() else None,
                touch_mean_when_not=round(float(P[~DONE, 0].mean()), 4) if (~DONE).any() else None,
                closer_agrees_sim=round(float(((P[:, 1] > 0.5) == closer).mean()), 4))
        if replay:
            stats.update(replay)
        print("STATS " + json.dumps(stats), flush=True)
    print("VERIFY_PASS" if rep.bad == 0 else f"VERIFY_FAIL ({rep.bad} checks)",
          flush=True)
    return 0 if rep.bad == 0 else 1


def run_replay(h5, names, cfg, ns, L, per_episode, rep):
    sys.dont_write_bytecode = True
    from swm.constants import ANSWER_OPTIONS
    from swm.utils.goal_generators import get_lang_table_goal
    from datagen_lt import oracle_vec, unpack_state
    from mpc import make_lt_env

    env = make_lt_env(cfg)
    goal = get_lang_table_goal("block_to_block", env, None, ANSWER_OPTIONS,
                               {"block_combo": list(cfg["block_combo"]),
                                "ood": bool(cfg.get("ood", False))})
    d_t, d_end, f_t, f_end, n = [], [], [], [], 0
    for name in names:
        g = h5[name]
        seed, C = int(g.attrs["seed"]), int(g.attrs["cycles"])
        b1, b2 = g.attrs["block1"], g.attrs["block2"]
        env.env._rng.seed(seed)
        goal.reset_env(seed=seed)
        rs = np.random.RandomState(seed)
        for c in sorted(rs.choice(C, size=min(per_episode, C), replace=False)):
            env.set_state(unpack_state(g["state_t_pkl"][c]))
            d_t.append(np.abs(oracle_vec(env, b1, b2)
                              - g["oracle_t"][c]).max())
            fr = env.get_judge_frame(ns).astype(np.float32)
            f_t.append(np.abs(fr - decode(g["frames_t"][c])).mean())
            for act in g["trajs"][c, 0, :L]:
                env.step_norender(np.asarray(act))
            d_end.append(np.abs(oracle_vec(env, b1, b2)
                                - g["oracle_end"][c, 0]).max())
            fr = env.get_judge_frame(ns).astype(np.float32)
            f_end.append(np.abs(fr - decode(g["end_frames"][c, 0])).mean())
            n += 1
    d_t, d_end, f_t, f_end = map(np.asarray, (d_t, d_end, f_t, f_end))
    rep(d_t.max() < 1e-5, "replay: stored state matches stored truth at t",
        f"max {d_t.max():.1e} m over {n} cycles")
    rep(f_t.max() < FRAME_TOL, "replay: stored frame at t is that state",
        f"mean abs diff max {f_t.max():.2f}")
    rep(np.median(d_end) < END_MEDIAN_TOL
        and np.mean(d_end < END_TOL) >= END_NEAR_SHARE,
        "replay: picked chunk reaches the stored end state",
        f"median {np.median(d_end):.1e} m, {np.mean(d_end < END_TOL):.0%} "
        f"within {END_TOL} m, max {d_end.max():.1e}")
    rep(np.mean(f_end < FRAME_TOL) >= 0.9, "replay: stored end frame",
        f"{np.mean(f_end < FRAME_TOL):.0%} under {FRAME_TOL}, "
        f"max {f_end.max():.2f}")
    return dict(replay_cycles=n, replay_end_median_m=float(np.median(d_end)),
                replay_end_near_share=float(np.mean(d_end < END_TOL)),
                replay_end_max_m=float(d_end.max()),
                replay_frame_t_max=float(f_t.max()))


if __name__ == "__main__":
    raise SystemExit(main())
