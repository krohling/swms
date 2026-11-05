"""Positional-bias probe for the selector (team-call item 2, 2026-10-30).

For each val cycle, runs the selection forward twice:
  stored:   candidates in stored ladder order (deployment-realistic)
  shuffled: candidates under a seeded random permutation (pick mapped to slot)

Reports pick-rate histograms over the k slots for both passes plus the
teacher's argmax slot distribution (stored order = temperature-ladder index),
and the candidate-consistency rate (same underlying candidate picked under
both orders — an order-invariance measure). Digit-token prior bias shows up
in BOTH passes; ladder-position correlation only in the stored pass.

Usage (m5):
  PYTHONPATH=$PWD:$PWD/swm-next .venv/bin/python \
    swm-next/swm_lora/eval_pos_bias.py \
    --ckpt outputs_swm_lora_local/exp2c_ckpt_step16000 \
    --train-config swm-next/swm_lora/train_exp2c.yaml \
    --h5 datasets_swm/bg_hybrid_v2_k8_r10.h5 \
    --max-cycles 2000 --device mps \
    --out experiments/exp2/pos_bias_exp2c_step16000.json
"""
import argparse
import io
import json
import os
import sys

import h5py
import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import QwenSWMSelector                              # noqa: E402
from data import action_stats, episode_split, normalize       # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train-config", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perm-seed", type=int, default=1234)
    ap.add_argument("--max-cycles", type=int, default=2000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tcfg = yaml.safe_load(open(args.train_config))
    device = args.device or tcfg.get("device", "cuda")
    k = int(tcfg.get("k", 8))
    swm = QwenSWMSelector(
        k=k, model_id=tcfg["model_id"], action_dim=tcfg["action_dim"],
        horizon=tcfg["horizon"], lora_r=tcfg["lora_r"],
        lora_alpha=tcfg["lora_alpha"], lora_dropout=0.0,
        device=device,
        proj_init_scale=float(tcfg.get("proj_init_scale", 0.1)),
        traj_cumsum=bool(tcfg.get("traj_cumsum", False)),
        lora_targets=tcfg.get("lora_targets", "attention"),
        proj_hidden=tcfg.get("proj_hidden", 512),
        proj_learnable_scale=bool(tcfg.get("learnable_scale", True)))
    swm.load_adapters(args.ckpt)
    swm.model.eval()

    amin, amax = action_stats(tcfg["diffusion_ckpt"])
    _, val_eps = episode_split([args.h5], val_frac=args.val_frac,
                               seed=args.seed)
    f = h5py.File(args.h5, "r")
    rng = np.random.default_rng(args.perm_seed)

    hist_stored = np.zeros(k, dtype=np.int64)   # model pick slot, ladder order
    hist_shuf = np.zeros(k, dtype=np.int64)     # model pick slot, random order
    hist_teacher = np.zeros(k, dtype=np.int64)  # teacher argmax, ladder order
    consistent = 0
    seen = 0
    with torch.no_grad():
        for _, ep in val_eps:
            if seen >= args.max_cycles:
                break
            g = f[ep]
            qa, qp = str(g.attrs["q_approach"]), str(g.attrs["q_place"])
            C, kk = g["score"].shape
            assert kk == k, f"H5 k={kk} != selector k={k}"
            for c in range(C):
                if seen >= args.max_cycles:
                    break
                img = Image.open(io.BytesIO(
                    g["frames_t"][c].tobytes())).convert("RGB")
                gate = [float(g["gate"][c])]
                T = normalize(g["trajs"][c].astype(np.float32), amin, amax)
                s = g["score"][c].astype(np.float64)

                trajs = torch.from_numpy(T).unsqueeze(0)       # (1, k, H, A)
                z = swm.selection_logits([img], [qa], [qp], gate, trajs)
                pick_stored = int(z[0].float().argmax())

                perm = rng.permutation(k)
                trajs_p = torch.from_numpy(T[perm]).unsqueeze(0)
                zp = swm.selection_logits([img], [qa], [qp], gate, trajs_p)
                pick_shuf_slot = int(zp[0].float().argmax())
                pick_shuf_cand = int(perm[pick_shuf_slot])

                hist_stored[pick_stored] += 1
                hist_shuf[pick_shuf_slot] += 1
                hist_teacher[int(s.argmax())] += 1
                consistent += int(pick_shuf_cand == pick_stored)
                seen += 1
                if seen % 250 == 0:
                    print(f"[{seen}] stored {hist_stored.tolist()} "
                          f"shuf {hist_shuf.tolist()} "
                          f"consist {consistent/seen:.3f}", flush=True)

    def pct(h):
        return [round(float(x) / seen * 100, 1) for x in h]

    summary = dict(
        ckpt=args.ckpt, h5=args.h5, cycles=seen, k=k,
        perm_seed=args.perm_seed,
        pick_rate_stored_pct=pct(hist_stored),
        pick_rate_shuffled_slot_pct=pct(hist_shuf),
        teacher_argmax_rate_pct=pct(hist_teacher),
        candidate_consistency=round(consistent / seen, 4),
        chance_pct=round(100.0 / k, 1))
    print("POS_BIAS_DONE " + json.dumps(summary), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(summary, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
