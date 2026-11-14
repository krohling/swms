"""Copy a contiguous env-seed range of ep_<seed> groups into a new H5.

usage: subset_h5_episodes.py OUT.h5 SRC.h5 SEED_LO SEED_HI   (inclusive)
Root attrs are copied unchanged, plus subset_of / subset_seeds so the file
names its parent. Every seed in the range must exist; OUT must not exist yet.
Used for the paper-blend set's play half (2,100 of the 4,200 labeled v5 play
episodes; labels are reused as they are, nothing is re-judged).
"""
import os
import sys
import h5py

out, src, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with h5py.File(src, "r") as f, h5py.File(out, "w-") as o:
    names = [f"ep_{s}" for s in range(lo, hi + 1)]
    missing = [n for n in names if n not in f]
    assert not missing, f"{src}: {len(missing)} seeds absent, e.g. {missing[:3]}"
    for k, v in f.attrs.items():
        o.attrs[k] = v
    o.attrs["subset_of"] = os.path.basename(src)
    o.attrs["subset_seeds"] = [lo, hi]
    for i, n in enumerate(names):
        f.copy(f[n], o, name=n)
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(names)} groups copied", flush=True)
    cycles = sum(int(o[n].attrs["cycles"]) for n in o)
    policies = sorted({str(o[n].attrs["policy"]) for n in o})
    print(f"SUBSET_OK {out}: {len(o)} episodes, {cycles:,} cycles, seeds "
          f"{lo}..{hi}, policies {policies}", flush=True)
