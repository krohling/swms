"""Build the frozen SAQA labeler-eval set from the sign-fix dataset.

Stratified reservoir sample: per (question type x oracle answer) cell, draw
--per-cell questions uniformly at random (seeded) from the full HDF5, then
extract each question's start/end frames and join the pose-derived metric
margin from the replay NPZ. The output is a single self-contained HDF5 the
eval driver (run_saqa_eval.py) — and any later round (API models, re-rendered
resolutions) — scores against, so every model sees the identical question set.

Margins (from replayed poses; NaN where entities can't be parsed):
  closer types:  |d(start) - d(end)| between the two named entities
  contact/state: entity separation at the END frame

    python swm-next/build_eval_set.py --hdf5 noisy-fixed.hdf5 \
        --teacher noisy-fixed.teacher.hdf5 --poses noisy-fixed.poses.npz \
        --colors noisy-fixed.poses.colors.json --out saqa_eval_12k.h5 \
        [--per-cell 1000] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import random
import re

import h5py
import numpy as np

CLOSER = ("block_block_closer", "gripper_block_closer")


def _s(x):
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--colors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cell", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    f = h5py.File(args.hdf5, "r")
    t = h5py.File(args.teacher, "r")
    poses = np.load(args.poses)
    colors = json.load(open(args.colors))

    # Pass 1: reservoir sample refs per (qtype, oracle-answer) cell.
    reservoirs: dict[tuple[str, bool], list] = {}
    counts: dict[tuple[str, bool], int] = {}
    trajs = sorted(f.keys(), key=lambda n: int(n.split("_")[-1]))
    for ti, tname in enumerate(trajs):
        g, tg = f[tname]["horizon_start"], t[tname]["horizon_start"]
        for start in g.keys():
            for hname in g[start].keys():
                d, td = g[start][hname], tg[start][hname]
                qs = d["questions"][()]
                ans = d["answers"][()].astype(bool)
                typs = d["types"][()]
                p = td["p_yes"][()].astype(np.float32)
                i0, i1 = int(d["start_idx"][()]), int(d["end_idx"][()])
                h = int(hname.split("_")[-1])
                for qi in range(len(qs)):
                    key = (_s(typs[qi]), bool(ans[qi]))
                    counts[key] = counts.get(key, 0) + 1
                    row = (tname, i0, i1, h, _s(qs[qi]), bool(ans[qi]),
                           _s(typs[qi]), float(p[qi]))
                    res = reservoirs.setdefault(key, [])
                    if len(res) < args.per_cell:
                        res.append(row)
                    else:
                        j = rng.randrange(counts[key])
                        if j < args.per_cell:
                            res[j] = row
        if (ti + 1) % 50 == 0:
            print(f"scanned {ti + 1}/{len(trajs)} trajectories", flush=True)

    rows = [r for key in sorted(reservoirs) for r in reservoirs[key]]
    rng.shuffle(rows)
    n = len(rows)
    print(f"sampled {n} questions across {len(reservoirs)} cells", flush=True)

    def margin(tname, i0, i1, qtype, q):
        ti = tname.split("_")[-1]
        cmap = {c: i for i, c in enumerate(colors[ti])}
        names = re.findall(r"(red|green|blue|yellow|orange|purple) cube", q.lower())
        try:
            eef = poses[f"{tname}_eef"]
            blk = [poses[f"{tname}_block_{i}"] for i in range(4)]
        except KeyError:
            return float("nan")
        if not names:
            return float("nan")
        b0 = blk[cmap[names[0]]]
        if qtype == "gripper_block_closer":
            return abs(float(np.linalg.norm(eef[i0] - b0[i0])) -
                       float(np.linalg.norm(eef[i1] - b0[i1])))
        if qtype == "block_block_closer":
            if len(names) < 2:
                return float("nan")
            b1 = blk[cmap[names[1]]]
            return abs(float(np.linalg.norm(b0[i0] - b1[i0])) -
                       float(np.linalg.norm(b0[i1] - b1[i1])))
        if qtype in ("cube_grasped", "block_gripper_touching"):
            return float(np.linalg.norm(eef[i1] - b0[i1]))
        if len(names) < 2:
            return float("nan")
        b1 = blk[cmap[names[1]]]
        return float(np.linalg.norm(b0[i1] - b1[i1]))

    with h5py.File(args.out, "w") as out:
        sf = out.create_dataset("start_frames", (n, 224, 224, 3), dtype=np.uint8)
        ef = out.create_dataset("end_frames", (n, 224, 224, 3), dtype=np.uint8)
        qs_, ty_, an_, pt_, mg_, hz_, ref_ = [], [], [], [], [], [], []
        cache_t, frames = None, None
        order = sorted(range(n), key=lambda i: rows[i][0])   # traj-locality for IO
        for j, i in enumerate(order):
            tname, i0, i1, h, q, ans, qtype, pt = rows[i]
            if tname != cache_t:
                cache_t, frames = tname, np.asarray(f[tname]["frames"][()], dtype=np.uint8)
            sf[i], ef[i] = frames[i0], frames[i1]
            if (j + 1) % 2000 == 0:
                print(f"extracted {j + 1}/{n}", flush=True)
        for i in range(n):
            tname, i0, i1, h, q, ans, qtype, pt = rows[i]
            qs_.append(q); ty_.append(qtype); an_.append(ans); pt_.append(pt)
            hz_.append(h); ref_.append(f"{tname}:{i0}:{i1}")
            mg_.append(margin(tname, i0, i1, qtype, q))
        S = h5py.string_dtype()
        out.create_dataset("questions", data=qs_, dtype=S)
        out.create_dataset("qtypes", data=ty_, dtype=S)
        out.create_dataset("oracle_yes", data=np.array(an_, dtype=bool))
        out.create_dataset("teacher_p_yes", data=np.array(pt_, dtype=np.float32))
        out.create_dataset("margin_m", data=np.array(mg_, dtype=np.float32))
        out.create_dataset("horizon", data=np.array(hz_, dtype=np.int32))
        out.create_dataset("provenance", data=ref_, dtype=S)
        out.attrs["source"] = args.hdf5
        out.attrs["seed"] = args.seed
        out.attrs["per_cell"] = args.per_cell

    for key in sorted(counts):
        got = len(reservoirs.get(key, []))
        print(f"{key[0]:26s} yes={key[1]!s:5s} pool={counts[key]:9,} sampled={got}")
    print("DONE:", args.out)


if __name__ == "__main__":
    main()
