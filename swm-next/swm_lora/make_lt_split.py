"""Fixed validation split for the LangTable k=1 sets (datagen_lt.py units).

Holds out the same number of whole episodes per task from the given unit
files and writes a split.json in data.save_split's layout, to be used as
`fixed_val_split` by the warm-up, the training run and any later run on a
set that grew: data.episode_split_fixed_val matches files by basename (unit
names are unique across batches) and trains on every other episode of every
file. data.episode_split stratifies by the policy flag only, which is
"random" for every LangTable episode, so it would not balance the tasks.

Usage (batch 0 = 50 units, 540 episodes per task; 54 per task held out):
  python swm-next/swm_lora/make_lt_split.py \
      --units 'datasets_swm/lt_v1/lt_v1_b0/units/b0_*_c[0-9].h5' \
      --val-per-task 54 --out swm-next/swm_lora/splits/lt_v1_b0_split.json
"""
import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data import expand_h5_paths, save_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--units", nargs="+", required=True,
                    help="unit files or glob patterns (quote the patterns)")
    ap.add_argument("--val-per-task", type=int, required=True, dest="n_val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = expand_h5_paths(args.units)
    by_task = {}
    for pi, path in enumerate(paths):
        with h5py.File(path, "r") as f:
            assert not f.attrs.get("dry_run", False), f"{path}: dry run"
            task = str(f.attrs["task"])
            by_task.setdefault(task, []).extend((pi, ep) for ep in f)
    rng = np.random.RandomState(args.seed)
    train, val = [], []
    for task in sorted(by_task):
        eps = sorted(by_task[task])
        assert len(eps) > args.n_val, f"{task}: only {len(eps)} episodes"
        rng.shuffle(eps)
        val += eps[:args.n_val]
        train += eps[args.n_val:]
        print(f"{task}: {len(eps)} episodes, {args.n_val} held out", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # basenames only: the split is matched by basename and travels between hosts
    save_split(args.out, sorted(train), sorted(val),
               [os.path.basename(p) for p in paths])
    print(f"LT_SPLIT_DONE {len(paths)} files, {len(train)} train / "
          f"{len(val)} val episodes -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
