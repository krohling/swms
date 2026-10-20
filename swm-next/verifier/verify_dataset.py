"""Invariant checks for datagen.py v2 output files.

Verifies, per episode:
  - schema: all v2 datasets present with consistent shapes (C, k)
  - score reconstruction: score == (1-g)*p_app + g*(PLACE_W*p_place +
    HOLD_W*p_hold) from the stored labels and gate
  - gate reconstruction: gate == EMA(gate_raw, alpha) with the recipe's alpha
  - planner_pick == argmax(score); on verifier episodes, pick == planner_pick
  - spreads match std over candidates of the stored labels/score
  - sim-state snapshots: expected dims, and candidate end states differ
  - generation seeds are disjoint from the evaluation range (6000-6099)

Exits nonzero on the first violated invariant. Prints a summary (episodes by
policy, cycles, tie-list fraction at the JND) on success.

Usage:
  python swm-next/verifier/verify_dataset.py datasets_swm/bg_hybrid_v2_k8_*.h5
"""
import argparse
import json
import sys

import h5py
import numpy as np

EVAL_SEEDS = range(6000, 6100)


def verify_file(path, jnd=0.05):
    f = h5py.File(path, "r")
    assert f.attrs.get("schema_version") == 2, \
        f"{path}: schema_version != 2 (v1 files predate stored gate/score)"
    recipe = json.loads(f.attrs["recipe"])
    alpha = recipe["gate_alpha"]
    pw, hw = recipe["place_w"], recipe["hold_w"]
    stats = dict(eps=0, cycles=0, ties=0, by_policy={})
    for ep in f:
        g = f[ep]
        C, k, _ = g["labels"].shape
        seed = int(g.attrs["seed"])
        assert seed not in EVAL_SEEDS, f"{ep}: generation seed in eval range"
        pol = g.attrs["policy"]

        for name, shape in [("gate_raw", (C,)), ("gate", (C,)),
                            ("score", (C, k)), ("planner_pick", (C,)),
                            ("spread_approach", (C,)), ("spread_place", (C,)),
                            ("spread_blended", (C,)), ("pick", (C,))]:
            assert g[name].shape == shape, \
                f"{ep}/{name}: {g[name].shape} != {shape}"
        for key in ("qpos", "qvel"):
            assert g[f"state_t_{key}"].shape[0] == C
            assert g[f"end_{key}"].shape[:2] == (C, k)

        lab, gate, score = g["labels"][:], g["gate"][:], g["score"][:]
        want = (1 - gate[:, None]) * lab[:, :, 0] + \
            gate[:, None] * (pw * lab[:, :, 1] + hw * lab[:, :, 2])
        assert np.allclose(want, score, atol=1e-5), f"{ep}: score != blend"

        ema, rec = None, []
        for rg in g["gate_raw"][:]:
            ema = float(rg) if ema is None else alpha * float(rg) + (1 - alpha) * ema
            rec.append(ema)
        assert np.allclose(rec, gate, atol=1e-6), f"{ep}: gate != EMA(gate_raw)"

        assert (g["planner_pick"][:] == score.argmax(1)).all(), \
            f"{ep}: planner_pick != argmax(score)"
        if pol == "verifier":
            assert (g["pick"][:] == g["planner_pick"][:]).all(), \
                f"{ep}: executed pick != planner pick on a verifier episode"

        assert np.allclose(g["spread_approach"][:], lab[:, :, 0].std(1), atol=1e-6)
        assert np.allclose(g["spread_place"][:], lab[:, :, 1].std(1), atol=1e-6)
        assert np.allclose(g["spread_blended"][:], score.std(1), atol=1e-6)

        eq = g["end_qpos"][:]
        assert (np.abs(eq[:, 0] - eq[:, -1]).max(axis=1) > 1e-9).any(), \
            f"{ep}: candidate end states all identical"

        stats["eps"] += 1
        stats["cycles"] += C
        stats["ties"] += int((score.std(1) <= jnd).sum())
        stats["by_policy"][pol] = stats["by_policy"].get(pol, 0) + 1
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--jnd", type=float, default=0.05)
    args = ap.parse_args()
    tot = dict(eps=0, cycles=0, ties=0, by_policy={})
    for p in args.paths:
        s = verify_file(p, jnd=args.jnd)
        print(f"{p}: OK — {s['eps']} eps, {s['cycles']:,} cycles, "
              f"{s['by_policy']}", flush=True)
        tot["eps"] += s["eps"]; tot["cycles"] += s["cycles"]
        tot["ties"] += s["ties"]
        for k2, v in s["by_policy"].items():
            tot["by_policy"][k2] = tot["by_policy"].get(k2, 0) + v
    print(f"ALL_OK — {tot['eps']} episodes, {tot['cycles']:,} cycles, "
          f"policies {tot['by_policy']}, blended tie-lists "
          f"(spread<={args.jnd}): {tot['ties']/max(tot['cycles'],1):.1%}")


if __name__ == "__main__":
    sys.exit(main())
