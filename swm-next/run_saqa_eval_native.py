"""Score a PaliGemma-WM checkpoint on the frozen eval set in its NATIVE
action-conditioned format: current frame (window start) + action sequence +
future question -- exactly the SAQA training/planning distribution, via the
planner's own SWMGradModel. Actions are fetched from the source dataset using
each item's provenance (traj:i0:i1).

This is the control for the labeler-protocol runs (run_saqa_eval.py), which
feed the END frame with no actions: differences between the two isolate
protocol effects from capability.

    python swm-next/run_saqa_eval_native.py --artifact saqa_eval_12k.h5 \
        --hdf5 noisy-fixed.hdf5 --checkpoint ckpts/paligemma_wm_ogbench \
        --name pgwm-published [--device mps] [--batch 8] [--limit 0]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import h5py
import numpy as np
import torch

from swm.constants import ANSWER_OPTIONS
from swm.semantic_world_model import SWMGradModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--hdf5", required=True, help="source dataset (for actions)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="labeler_eval_results_native")
    args = ap.parse_args()

    src = h5py.File(args.hdf5, "r")
    with h5py.File(args.artifact, "r") as f:
        n_all = f["oracle_yes"].shape[0]
        n = min(args.limit, n_all) if args.limit else n_all
        items = []
        for i in range(n):
            tname, i0, i1 = f["provenance"][i].decode().split(":")
            items.append(dict(
                question=f["questions"][i].decode(),
                qtype=f["qtypes"][i].decode(),
                oracle=bool(f["oracle_yes"][i]),
                start=np.asarray(f["start_frames"][i]),
                actions=torch.as_tensor(
                    np.asarray(src[tname]["actions"][int(i0):int(i1)]),
                    dtype=torch.float32),
            ))

    model = SWMGradModel(checkpoint_path=args.checkpoint,
                         processor_path=args.checkpoint,
                         tokens=ANSWER_OPTIONS, precision=torch.bfloat16,
                         device=args.device)

    p = np.zeros(n)
    t0 = time.time()
    with torch.no_grad():
        for b0 in range(0, n, args.batch):
            chunk = items[b0:b0 + args.batch]
            probs = model.get_scores(
                images=[it["start"] for it in chunk],
                actions=[it["actions"] for it in chunk],
                questions=[it["question"] for it in chunk])
            yes, no = probs[0].float().cpu().numpy(), probs[1].float().cpu().numpy()
            p[b0:b0 + len(chunk)] = yes / np.clip(yes + no, 1e-12, None)
            if (b0 // args.batch) % 50 == 0:
                rate = (b0 + len(chunk)) / max(time.time() - t0, 1e-9)
                print(f"[{args.name}] {b0 + len(chunk)}/{n}  {rate:.1f} q/s", flush=True)

    oracle = np.array([it["oracle"] for it in items])
    qtypes = np.array([it["qtype"] for it in items])
    correct = (p >= 0.5) == oracle
    report = dict(backend=args.name, protocol="native_action_conditioned",
                  n=n, qps=round(n / (time.time() - t0), 2),
                  overall_acc=float(correct.mean()), per_type={})
    for qt in sorted(set(qtypes)):
        report["per_type"][qt] = float(correct[qtypes == qt].mean())

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.name}.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
