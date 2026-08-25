"""Score one VLM backend against the frozen SAQA labeler-eval set.

Reports per-type balanced accuracy, overall accuracy, yes/no mass capture,
and margin-correlation (Spearman rho between the pose-derived margin and
per-question correctness) -- the two decision metrics from the failure-mode
analysis: distance to the 95% gate per type, and whether the model has any
signal on the 'closer' comparison task.

    python swm-next/run_saqa_eval.py --artifact saqa_eval_12k.h5 \
        --backend qwen3-vl-8b [--device mps] [--batch 8] [--limit 0] \
        [--out results/]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import h5py
import numpy as np

from labeler_backends import BACKENDS, CLOSER


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all questions")
    ap.add_argument("--out", default="labeler_eval_results")
    args = ap.parse_args()

    with h5py.File(args.artifact, "r") as f:
        n_all = f["oracle_yes"].shape[0]
        n = min(args.limit, n_all) if args.limit else n_all
        items = [dict(
            question=f["questions"][i].decode(),
            qtype=f["qtypes"][i].decode(),
            oracle=bool(f["oracle_yes"][i]),
            margin=float(f["margin_m"][i]),
            start=np.asarray(f["start_frames"][i]),
            end=np.asarray(f["end_frames"][i]),
        ) for i in range(n)]

    backend = BACKENDS[args.backend](args.device)
    p = np.zeros(n); mass = np.zeros(n)
    t0 = time.time()
    for b0 in range(0, n, args.batch):
        chunk = items[b0:b0 + args.batch]
        pp, mm = backend.score(chunk)
        p[b0:b0 + len(chunk)] = pp
        mass[b0:b0 + len(chunk)] = mm
        if (b0 // args.batch) % 25 == 0:
            rate = (b0 + len(chunk)) / max(time.time() - t0, 1e-9)
            print(f"[{args.backend}] {b0 + len(chunk)}/{n}  {rate:.1f} q/s", flush=True)
    dt = time.time() - t0

    oracle = np.array([it["oracle"] for it in items])
    qtypes = np.array([it["qtype"] for it in items])
    margins = np.array([it["margin"] for it in items])
    correct = (p >= 0.5) == oracle

    report = dict(backend=args.backend, n=n, seconds=round(dt, 1),
                  qps=round(n / dt, 2), overall_acc=float(correct.mean()),
                  mean_mass=float(mass.mean()), per_type={}, margin_corr={})
    for qt in sorted(set(qtypes)):
        m = qtypes == qt
        report["per_type"][qt] = float(correct[m].mean())
        if qt in CLOSER:
            v = m & ~np.isnan(margins)
            report["margin_corr"][qt] = spearman(margins[v], correct[v].astype(float))

    os.makedirs(args.out, exist_ok=True)
    base = os.path.join(args.out, args.backend.replace("/", "_"))
    with open(base + ".json", "w") as fh:
        json.dump(report, fh, indent=2)
    np.savez_compressed(base + "_scores.npz", p_yes=p, mass=mass)

    print(json.dumps({k: v for k, v in report.items() if k != "per_type"}, indent=2))
    for qt, acc in report["per_type"].items():
        extra = f"  rho={report['margin_corr'][qt]:+.3f}" if qt in report["margin_corr"] else ""
        print(f"  {qt:26s} {acc:.3f}{extra}")


if __name__ == "__main__":
    main()
