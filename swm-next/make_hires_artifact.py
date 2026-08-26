"""Rebuild an eval artifact with frames swapped to a hi-res dataset.

Takes an existing frozen eval artifact (questions, labels, margins, provenance
all preserved byte-for-byte) and replaces start/end frames with renders from
the resolution-swapped dataset -- the identical question set at a different
resolution, for a controlled resolution arm.

    python swm-next/make_hires_artifact.py --artifact saqa_eval_12k.h5 \
        --frames-hdf5 noisy-fixed-768.hdf5 --out saqa_eval_12k_768.h5
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--frames-hdf5", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = h5py.File(args.artifact, "r")
    fr = h5py.File(args.frames_hdf5, "r")
    size = int(fr.attrs.get("frame_size", fr[list(fr.keys())[0]]["frames"].shape[1]))
    prov = [p.decode() for p in src["provenance"][()]]
    n = len(prov)

    with h5py.File(args.out, "w") as dst:
        for k in ("questions", "qtypes", "oracle_yes", "teacher_p_yes",
                  "margin_m", "horizon", "provenance"):
            src.copy(src[k], dst, k)
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        dst.attrs["frame_size"] = size
        dst.attrs["frames_source"] = args.frames_hdf5
        sf = dst.create_dataset("start_frames", (n, size, size, 3), dtype=np.uint8,
                                chunks=(1, size, size, 3), compression="gzip",
                                compression_opts=4)
        ef = dst.create_dataset("end_frames", (n, size, size, 3), dtype=np.uint8,
                                chunks=(1, size, size, 3), compression="gzip",
                                compression_opts=4)
        order = sorted(range(n), key=lambda i: prov[i].split(":")[0])
        cache_t, frames = None, None
        for j, i in enumerate(order):
            tname, i0, i1 = prov[i].split(":")
            if tname != cache_t:
                cache_t = tname
                frames = np.asarray(fr[tname]["frames"][()], dtype=np.uint8)
            sf[i] = frames[int(i0)]
            ef[i] = frames[int(i1)]
            if (j + 1) % 2000 == 0:
                print(f"swapped {j + 1}/{n}", flush=True)
    print("DONE:", args.out)


if __name__ == "__main__":
    main()
