"""V1/V5 scaffolds: predicates computed in code from grounded boxes.

V1 (contact/state types, PORTABLE): all quantities are relative -- distances
normalized by the median entity box size in that frame (image-derived scale),
overlap as fractions -- so nothing requires camera calibration or sim state.
Per-type thresholds are fit on a --calib-frac slice (disclosed; the real-world
analog is a small human-annotated calibration set) and applied to the rest.

  touching(a, b):   normalized box gap <= tau
  ontop(a, b):      a's center above b's, horizontal IoU-overlap >= tau_o,
                    vertical gap <= tau_v
  grasped(g, c):    cube center inside (padded) gripper box AND cube's
                    vertical position near gripper's lower half
  gripper_touch:    same gap rule as touching, gripper box vs cube box

V5 (closer types, DIAGNOSTIC -- flagged non-portable): 2D centroid distance
change between frames in cube-widths vs a fitted threshold. Depth-axis motion
is invisible to 2D centroids; this arm upper-bounds what flat geometry can do
and is excluded from production-candidate comparisons.

No VLM calls anywhere: boxes come from the on-disk grounding cache
(run_crop_reask must have populated it; questions with missing boxes are
scored by the majority-class fallback and reported separately).

    python swm-next/labeler_eval/run_geometry.py --artifact saqa_eval_12k_768.h5 \
        --cache labeler_eval_results_v2/ground_cache_sol.json [--limit 3000] \
        [--calib-frac 0.2]
"""
from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np

from run_crop_reask import CLOSER, parse_entities

CONTACT = ("block_touching_block", "block_ontop_block", "cube_grasped",
           "block_gripper_touching")


def box_gap_cw(a, b, cw):
    """Shortest edge-to-edge distance between two boxes, in cube-widths
    (0 when overlapping)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(dx, dy)) / cw


def h_overlap_frac(a, b):
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    denom = min(a[2] - a[0], b[2] - b[0])
    return inter / denom if denom > 0 else 0.0


def features(qtype, boxes, boxes_start=None):
    """Scalar decision feature per question; higher = more 'yes'."""
    cw = float(np.median([max(b[2] - b[0], b[3] - b[1]) for b in boxes]))
    if qtype == "block_touching_block" or qtype == "block_gripper_touching":
        return -box_gap_cw(boxes[0], boxes[1], cw)          # closer = more yes
    if qtype == "block_ontop_block":
        a, b = boxes[0], boxes[1]
        acy, bcy = (a[1] + a[3]) / 2, (b[1] + b[3]) / 2
        above = 1.0 if acy < bcy else -1.0
        vgap = max(0.0, b[1] - a[3]) / cw
        return above * h_overlap_frac(a, b) - vgap
    if qtype == "cube_grasped":
        g, c = boxes[1], boxes[0]
        ccx, ccy = (c[0] + c[2]) / 2, (c[1] + c[3]) / 2
        pad = 0.25 * cw
        inside = (g[0] - pad <= ccx <= g[2] + pad) and (g[1] - pad <= ccy <= g[3] + pad)
        return (1.0 if inside else -1.0) - box_gap_cw(g, c, cw)
    if qtype in CLOSER:   # V5 diagnostic
        def dist(bs):
            (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = bs
            return float(np.hypot((ax1 + ax2) / 2 - (bx1 + bx2) / 2,
                                  (ay1 + ay2) / 2 - (by1 + by2) / 2))
        return (dist(boxes_start) - dist(boxes)) / cw       # positive = got closer
    raise ValueError(qtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--calib-frac", type=float, default=0.2)
    ap.add_argument("--box-scale", type=float, default=1.0)
    ap.add_argument("--out", default="labeler_eval_results_v1")
    args = ap.parse_args()

    cache = json.load(open(args.cache))
    f = h5py.File(args.artifact, "r")
    n = min(args.limit, f["oracle_yes"].shape[0])

    rows = []
    for i in range(n):
        q = f["questions"][i].decode()
        qtype = f["qtypes"][i].decode()
        tname, i0, i1 = f["provenance"][i].decode().split(":")
        ents = parse_entities(q, qtype)
        def get(frame_idx, e):
            b = cache.get(f"{tname}:{frame_idx}:{e}")
            return [v / args.box_scale for v in b] if (b and args.box_scale != 1.0) else b
        be = [get(i1, e) for e in ents]
        ok = all(be)
        feat = None
        if ok and qtype in CLOSER:
            bs = [get(i0, e) for e in ents]
            ok = all(bs)
            if ok:
                feat = features(qtype, be, bs)
        elif ok:
            feat = features(qtype, be)
        rows.append(dict(i=i, qtype=qtype, ok=ok, feat=feat,
                         oracle=bool(f["oracle_yes"][i])))

    rng = np.random.default_rng(0)
    report = dict(variant="v1_geometry(+v5_closer_diagnostic)", n=n,
                  calib_frac=args.calib_frac, thresholds={}, per_type={},
                  portable={t: (t in CONTACT) for t in CONTACT + CLOSER},
                  boxes_missing_frac=float(np.mean([not r["ok"] for r in rows])))
    for qtype in CONTACT + CLOSER:
        sub = [r for r in rows if r["qtype"] == qtype and r["ok"]]
        if not sub:
            continue
        idx = rng.permutation(len(sub))
        k = max(1, int(len(sub) * args.calib_frac))
        cal, test = [sub[j] for j in idx[:k]], [sub[j] for j in idx[k:]]
        feats = np.array([r["feat"] for r in cal])
        ys = np.array([r["oracle"] for r in cal])
        cands = np.unique(feats)
        best_t, best_a = 0.0, -1
        for t in cands:
            a = float(((feats >= t) == ys).mean())
            if a > best_a:
                best_t, best_a = float(t), a
        tf = np.array([r["feat"] for r in test])
        ty = np.array([r["oracle"] for r in test])
        acc = float(((tf >= best_t) == ty).mean())
        report["thresholds"][qtype] = best_t
        report["per_type"][qtype] = dict(test_acc=acc, n_test=len(test),
                                         calib_acc=best_a)
    os.makedirs(args.out, exist_ok=True)
    json.dump(report, open(os.path.join(args.out, "v1_geometry.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
