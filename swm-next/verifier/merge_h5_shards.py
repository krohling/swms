"""Merge sharded convert_ogb_v5.py outputs into one H5.

usage: merge_v5_shards.py OUT.h5 SHARD0.h5 SHARD1.h5 [...]
Root attrs must be identical across shards (they are: same config/seed_offset);
ep_<seed> groups are copied in seed order; duplicate groups are an error; OUT
must not exist yet.
"""
import sys
import h5py
import numpy as np

out, shards = sys.argv[1], sys.argv[2:]
fs = [h5py.File(p, "r") for p in shards]
ref = dict(fs[0].attrs)


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    return a.shape == b.shape and np.array_equal(a, b)


for p, f in zip(shards[1:], fs[1:]):
    keys = set(f.attrs) | set(ref)
    diff = [k for k in sorted(keys)
            if k not in f.attrs or k not in ref or not same(ref[k], f.attrs[k])]
    assert not diff, f"root attrs differ in {p}: {diff}"

owner = {}
for p, f in zip(shards, fs):
    for n in f:
        assert n not in owner, f"duplicate group {n} in {p} and {owner[n]}"
        owner[n] = f
order = sorted(owner, key=lambda n: int(n.split("_")[1]))

with h5py.File(out, "w-") as o:
    for k, v in ref.items():
        o.attrs[k] = v
    for i, n in enumerate(order):
        src = owner[n]
        src.copy(src[n], o, name=n)
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(order)} groups copied", flush=True)
    n_eps = len(o)
    cycles = sum(int(o[n].attrs["cycles"]) for n in o)
    seeds = sorted(int(o[n].attrs["seed"]) for n in o)
    print(f"MERGE_OK {out}: {n_eps} episodes, {cycles:,} cycles, seeds "
          f"{seeds[0]}..{seeds[-1]} (contiguous: {seeds == list(range(seeds[0], seeds[-1] + 1))}), "
          f"root attrs {sorted(o.attrs)}", flush=True)
