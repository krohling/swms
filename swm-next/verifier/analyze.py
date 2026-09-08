"""Stage 3: offline analysis of sweep scores. No GPU, re-runs in seconds.

Consumes scores.npz + manifest.json from sweep.py and the composites block
of the config. A composite turns raw per-question scores into one selection
signal, mirroring planning's weighted-question objective:

    composites:
      - name: phase1_planning       # planning's phase-1 objective
        aggregate: current          # rank rollouts by the latest value
        terms:
          - {q: ontop, weight: 0.6}
          - {q: grasp, weight: 0.4}
      - name: progress_h8
        aggregate: cumulative       # rank by mean of all scores up to t
        h: 8                        # pair questions need a window width
        terms:
          - {q: progress, weight: 1.0}

Aggregates: "current" = the signal's most recent value at t (a V(s_t)
reading); "cumulative" = mean of all its values up to t (accumulated
progress -- the right collapse for local pair judgments).

For each composite, at every eval position t, the K rollouts per seed are
ranked and we report selection-SR (argmax picks an eventually-successful
rollout) and rank-AUC vs eventual success. Output: printed table +
analysis.json. Phase context: manifest rollouts carry done_step, and the
bank's per-step grasped/ontop arrays support phase-conditioned analyses.

Usage:

    python swm-next/verifier/analyze.py --config swm-next/configs/verifier.yaml
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = labels.sum(), (1 - labels).sum()
    if not (pos and neg):
        return float("nan")
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def selection_sr(scores: np.ndarray, seeds: np.ndarray, success: np.ndarray) -> float:
    picks = []
    for s in np.unique(seeds):
        idx = np.where(seeds == s)[0]
        picks.append(success[idx[np.argmax(scores[idx])]])
    return float(np.mean(picks))


def composite_series(comp: dict, scores: np.lib.npyio.NpzFile,
                     manifest: dict) -> tuple[np.ndarray, list]:
    """Weighted sum of the composite's term signals -> [R, T] matrix plus the
    eval positions T refers to. Single questions live on `positions`; pair
    questions on `window_starts[h]` (a window's score is dated by its END,
    start + h, so no signal uses frames from the future of t)."""
    positions = manifest["positions"]
    total, times = None, None
    for term in comp["terms"]:
        q, w = term["q"], float(term.get("weight", 1.0))
        if f"{q}/p" in scores:
            sig, t = scores[f"{q}/p"], positions
        else:
            h = int(comp["h"])
            sig = scores[f"{q}/h{h}/p"]
            t = [s + h for s in manifest["window_starts"][str(h)]]
        if total is None:
            total, times = w * sig, t
        else:
            n = min(total.shape[1], sig.shape[1])   # align if grids differ
            total, times = total[:, :n] + w * sig[:, :n], times[:n]
    return total, times


def aggregate(series: np.ndarray, mode: str) -> np.ndarray:
    """[R, T] per-time signals -> [R, T] selection scores at each t."""
    if mode == "current":
        return series
    if mode == "cumulative":
        counts = np.arange(1, series.shape[1] + 1)
        return np.cumsum(series, axis=1) / counts
    raise ValueError(f"unknown aggregate: {mode}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out_dir = cfg["out_dir"]

    scores = np.load(os.path.join(out_dir, "scores.npz"))
    manifest = json.load(open(os.path.join(out_dir, "manifest.json")))
    rolls = manifest["rollouts"]
    seeds = np.array([r["seed"] for r in rolls])
    success = np.array([r["success"] for r in rolls], dtype=float)

    baseline = float(success.mean())
    ceiling = selection_sr(success, seeds, success)
    print(f"{len(rolls)} rollouts | random baseline {baseline:.1%} | "
          f"best-of-K ceiling {ceiling:.1%}\n")

    report = dict(random_baseline=baseline, best_of_k_ceiling=ceiling,
                  composites={})
    for comp in cfg["composites"]:
        series, times = composite_series(comp, scores, manifest)
        sel = aggregate(series, comp.get("aggregate", "current"))
        rows = {int(t): dict(
                    selection_sr=selection_sr(sel[:, i], seeds, success),
                    auc=rank_auc(sel[:, i], success))
                for i, t in enumerate(times)}
        report["composites"][comp["name"]] = dict(spec=comp, by_t=rows)
        shown = [t for t in rows if t in (4, 8, 16, 32, 64, 100, 150, 200)] \
            or list(rows)[:8]
        print(f"[{comp['name']}] ({comp.get('aggregate', 'current')})")
        for t in shown:
            r = rows[t]
            print(f"  t={t:3d}: SR={r['selection_sr']:.0%}  AUC={r['auc']:.2f}")
        print()

    out_path = os.path.join(out_dir, "analysis.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"ANALYSIS_DONE: {out_path}")


if __name__ == "__main__":
    main()
