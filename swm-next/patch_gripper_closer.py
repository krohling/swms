"""Surgically recompute gripper_block_closer answers with the corrected sign.

The generator bug (krohling/ogbench@1bb5493): gripper_block_closer used
`old_dist < new_dist` (gripper moved AWAY => labeled True). Correct
convention, matching block_block_closer: `old_dist > new_dist` AND
|delta| > 0.015.

Answers cannot be fixed from the stored labels alone (a stored False is
ambiguous between "moved away" and "under threshold"), so this needs the
per-trajectory poses: eef_pos and block_{i}_pos at every frame, in HDF5
trajectory order. Accepted pose formats (--poses):
  - a directory of Jacob's raw trajectory .pkl files (sorted order must
    match the HDF5's traj_N order), each with data["block_states"][t]
    holding "eef_pos" and "block_{i}_pos"; or
  - a single .npz with arrays "traj_{N}_eef" (T,3) and "traj_{N}_block_{i}" (T,3).

Output: a COPY of the input HDF5 with only gripper_block_closer answers
rewritten. Everything else (frames, actions, questions, types, other
answers, group structure) is byte-preserved, so teacher sidecars and index
caches keyed on structure remain valid (index caches keyed on answers must
be rebuilt: pass the new file path so the cache stem differs).

Question-to-block matching: the block name inside the question text
("<color> cube") is mapped to a block index via each trajectory's color
lookup, which must also be supplied (in the pkls) or inferable; with the
npz format pass --colors <json> mapping traj -> [color0..color3].

    python patch_gripper_closer.py --hdf5 noisy.hdf5 --poses <dir|npz> \
        --out noisy-fixed.hdf5 [--colors colors.json]

Gates printed:
  G-F1  per-type answer-change counts (must be 0 for every type except
        gripper_block_closer)
  G-F2  gripper_block_closer flip fraction + resulting yes/no balance
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
from collections import defaultdict

import h5py
import numpy as np

THRESH = 0.015
QT = "gripper_block_closer"


def _s(x) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


def load_poses(poses_path: str, colors_path: str | None):
    """-> (eef[traj][t] -> (3,), block[traj][i][t] -> (3,), colors[traj] -> [names])"""
    eef, block, colors = {}, {}, {}
    if os.path.isdir(poses_path):
        files = sorted(f for f in os.listdir(poses_path) if f.endswith(".pkl"))
        for ti, fname in enumerate(files):
            data = pickle.load(open(os.path.join(poses_path, fname), "rb"))
            states = data["block_states"]
            eef[ti] = np.stack([s["eef_pos"] for s in states])
            block[ti] = {i: np.stack([s[f"block_{i}_pos"] for s in states])
                         for i in range(4)}
            lut = data.get("color_lookup", {})
            colors[ti] = [lut.get(i, f"block_{i}") for i in range(4)]
    else:
        z = np.load(poses_path)
        tids = sorted({int(k.split("_")[1]) for k in z.files if k.startswith("traj_")})
        for ti in tids:
            eef[ti] = z[f"traj_{ti}_eef"]
            block[ti] = {i: z[f"traj_{ti}_block_{i}"] for i in range(4)}
        if colors_path:
            cj = json.load(open(colors_path))
            colors = {int(k): v for k, v in cj.items()}
    return eef, block, colors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--colors", default=None)
    args = ap.parse_args()

    print(f"copying {args.hdf5} -> {args.out}")
    shutil.copyfile(args.hdf5, args.out)

    eef, block, colors = load_poses(args.poses, args.colors)

    changed = defaultdict(int)
    totals = defaultdict(int)
    flips_y2n = flips_n2y = 0
    n_yes_after = n_total_qt = 0

    with h5py.File(args.out, "r+") as f:
        names = sorted(f.keys(), key=lambda n: int(n.split("_")[-1]))
        for ti, tname in enumerate(names):
            cmap = {c: i for i, c in enumerate(colors.get(ti, []))}
            hs = f[tname]["horizon_start"]
            for start in hs:
                for hname in hs[start]:
                    d = hs[start][hname]
                    i0 = int(d["start_idx"][()])
                    i1 = int(d["end_idx"][()])
                    qs = d["questions"][()]
                    typs = [_s(t) for t in d["types"][()]]
                    ans = np.array(d["answers"][()], dtype=bool)
                    new_ans = ans.copy()
                    for qi, (q, t) in enumerate(zip(qs, typs)):
                        totals[t] += 1
                        if t != QT:
                            continue
                        n_total_qt += 1
                        m = re.search(r"(\w+) cube", _s(q))
                        assert m, f"no block color in question: {_s(q)}"
                        bi = cmap[m.group(1)]
                        old_d = np.linalg.norm(eef[ti][i0] - block[ti][bi][i0])
                        new_d = np.linalg.norm(eef[ti][i1] - block[ti][bi][i1])
                        correct = bool(old_d > new_d and abs(old_d - new_d) > THRESH)
                        new_ans[qi] = correct
                        n_yes_after += int(correct)
                        if correct != ans[qi]:
                            changed[t] += 1
                            if ans[qi]:
                                flips_y2n += 1
                            else:
                                flips_n2y += 1
                    if not np.array_equal(ans, new_ans):
                        del d["answers"]
                        d.create_dataset("answers", data=new_ans)

    print("=== G-F1: answer changes by type (only gripper_block_closer may be nonzero) ===")
    ok = True
    for t in sorted(totals):
        c = changed.get(t, 0)
        flag = ""
        if t != QT and c:
            ok = False
            flag = "  <-- UNEXPECTED"
        print(f"  {t:28s} changed {c:,} / {totals[t]:,}{flag}")
    print(f"=== G-F2: {QT} ===")
    print(f"  flipped {sum(changed.values()):,} / {n_total_qt:,} "
          f"({100*sum(changed.values())/max(1,n_total_qt):.1f}%)  "
          f"yes->no {flips_y2n:,}  no->yes {flips_n2y:,}  "
          f"yes-rate after fix: {100*n_yes_after/max(1,n_total_qt):.1f}%")
    print("PATCH_OK" if ok else "PATCH_GATE_FAILED")


if __name__ == "__main__":
    main()
