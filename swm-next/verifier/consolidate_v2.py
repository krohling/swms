"""Consolidate datagen v2 shards into one dataset, split by the R10 filter.

R10 (2026-10-22): a cycle is REMOVED iff the max-min of its 8 candidate
scores is < 0.10 (2x the measured judge JND of ~0.05) on all three columns —
q_approach, q_place, and the gate-blended planner score. Equivalently, a
cycle is kept iff some pair of candidates differs by >= 0.10 on some
question. Since range < 0.10 implies std < 0.05, R10 removes a strict subset
of the std-based tie cycles: a list whose 8 scores are flat except one
>= 2-JND outlier (usually a winner) stays in.

Removed cycles are written verbatim to <out stem>_removed.h5 for provenance.
Both files keep the v2 per-episode schema plus, per episode: orig_cycle_idx
(each cycle's row in its source shard) and attrs cycles_orig / source_shard.
An episode appears in a file only if it has cycles on that side. File attrs
carry the shared recipe, the filter rule, and the source shard list, so
kept + removed reconstruct the unfiltered shards exactly. Verify the pair:

  python swm-next/verifier/verify_dataset.py <out>.h5 --removed <out>_removed.h5

Usage:
  python swm-next/verifier/consolidate_v2.py \
      --out datasets_swm/bg_hybrid_v2_k8_r10.h5 datasets_swm/bg_hybrid_v2_k8_s*.h5
"""
import argparse
import datetime
import json
import os

import h5py
import numpy as np

THR = 0.10


def copy_side(dst, src_grp, ep, mask, shard_name):
    """Copy the masked cycles of one episode group; returns cycles written."""
    if not mask.any():
        return 0
    g = dst.create_group(ep)
    for name, ds in src_grp.items():
        if h5py.check_vlen_dtype(ds.dtype) is not None:    # JPEG byte vectors
            data = ds[:][mask]
            d = g.create_dataset(name, data.shape, dtype=ds.dtype)
            for i in np.ndindex(data.shape):               # h5py can't
                d[i] = data[i]                             # bulk-write vlen
        else:
            g.create_dataset(name, data=ds[:][mask])
    g.create_dataset("orig_cycle_idx",
                     data=np.nonzero(mask)[0].astype(np.int32))
    for k, v in src_grp.attrs.items():
        g.attrs[k] = v
    g.attrs["cycles"] = int(mask.sum())
    g.attrs["cycles_orig"] = int(len(mask))
    g.attrs["source_shard"] = shard_name
    return int(mask.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=THR)
    args = ap.parse_args()
    out_rm = args.out.replace(".h5", "_removed.h5")

    recipe = None
    outs = {}
    for role, path in (("kept", args.out), ("removed", out_rm)):
        f = h5py.File(path, "w")
        f.attrs["schema_version"] = 2
        f.attrs["filter"] = "R10"
        f.attrs["filter_role"] = role
        f.attrs["filter_threshold"] = args.threshold
        f.attrs["filter_rule"] = (
            "cycle removed iff max-min of candidate scores < threshold on all "
            "of: labels[:,:,0] (approach), labels[:,:,1] (place), score "
            "(gate-blended)")
        f.attrs["source_files"] = json.dumps(
            [os.path.basename(p) for p in args.shards])
        f.attrs["created"] = datetime.date.today().isoformat()
        outs[role] = f

    tot = dict(eps=0, cycles=0, kept=0, removed=0, eps_kept=0, eps_removed=0)
    for path in args.shards:
        shard = os.path.basename(path)
        with h5py.File(path, "r") as src:
            assert src.attrs.get("schema_version") == 2, f"{path}: not v2"
            r = src.attrs["recipe"]
            assert recipe in (None, r), f"{path}: recipe differs across shards"
            recipe = r
            for ep in src:
                g = src[ep]
                lab, sc = g["labels"][:], g["score"][:]
                rng = [v.max(axis=1) - v.min(axis=1)
                       for v in (lab[:, :, 0], lab[:, :, 1], sc)]
                rm = ((rng[0] < args.threshold) & (rng[1] < args.threshold)
                      & (rng[2] < args.threshold))
                k = copy_side(outs["kept"], g, ep, ~rm, shard)
                d = copy_side(outs["removed"], g, ep, rm, shard)
                tot["eps"] += 1
                tot["cycles"] += len(rm)
                tot["kept"] += k
                tot["removed"] += d
                tot["eps_kept"] += int(k > 0)
                tot["eps_removed"] += int(d > 0)
        print(f"{shard}: done ({tot['eps']} eps so far)", flush=True)

    for role, f in outs.items():
        f.attrs["recipe"] = recipe
        f.attrs["episodes"] = tot[f"eps_{role}"]
        f.attrs["cycles"] = tot[role]
        f.close()
    print(f"CONSOLIDATE_DONE — {tot['eps']} eps, {tot['cycles']:,} cycles in; "
          f"kept {tot['kept']:,} ({tot['kept']/tot['cycles']:.1%}) in "
          f"{tot['eps_kept']} eps -> {args.out}; removed {tot['removed']:,} in "
          f"{tot['eps_removed']} eps -> {out_rm}")


if __name__ == "__main__":
    main()
