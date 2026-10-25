"""Compose baseline for the selector arm (exp2 step 1): how well do a
SCORING checkpoint's per-question predictions compose into the planner's
decision when blended with the STORED gate?

  s_hat_i = (1-g) * p_app_i + g * (0.6 * p_place_i + 0.4 * p_app_i)

evaluated against the stored teacher blend (v2 `score`) on held-out cycles.
This is the number exp2 has to beat: if composition already recovers the
teacher's pick, a trained selector has little headroom; if composition is
lossy, selection is the right place to spend capacity.

Uses the same episode split as exp1b (seed/val_frac from the train config),
so exp1 and exp1b checkpoints are compared on identical cycles. exp1 never
saw ANY v2 episode (disjoint seed ranges), so the split only matters for
exp1b. Gate is the stored EMA — deployment maintains it from the model's own
zero-traj grasp probe, but here we isolate composition quality from gate
drift.

Startup sanity: blending the stored LABELS with the stored gate must
reproduce the stored score exactly (validates columns + formula against the
datagen writer).

Usage (Vista gh-dev or m5):
  python swm-next/swm_lora/eval_compose.py \
    --ckpt outputs_swm_lora/exp1_full/ckpt_step84687 \
    --train-config swm-next/swm_lora/train_exp1.yaml \
    --h5 datasets_swm/bg_hybrid_v2_k8_r10.h5 \
    --max-cycles 4000 --out experiments/exp2/compose_exp1_final.json
"""
import argparse
import json
import os
import sys

import h5py
import io
import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import QwenSWM                                     # noqa: E402
from data import action_stats, episode_split, normalize       # noqa: E402

PLACE_W, HOLD_W = 0.6, 0.4


def blend(g, p_app, p_place):
    return (1.0 - g) * p_app + g * (PLACE_W * p_place + HOLD_W * p_app)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train-config", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cycles", type=int, default=4000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tcfg = yaml.safe_load(open(args.train_config))
    device = args.device or tcfg.get("device", "cuda")
    swm = QwenSWM(model_id=tcfg["model_id"], action_dim=tcfg["action_dim"],
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

    # sanity: stored labels + stored gate must reproduce the stored score
    pi, ep0 = val_eps[0]
    g0 = f[ep0]
    lab, gate, score = g0["labels"][:], g0["gate"][:], g0["score"][:]
    rec = blend(gate[:, None], lab[:, :, 0], lab[:, :, 1])
    err = float(np.abs(rec - score).max())
    assert err < 1e-4, f"teacher-compose sanity failed: max err {err}"
    print(f"teacher-compose sanity ok (max err {err:.2e})", flush=True)

    from scipy.stats import spearmanr
    rhos, rhos_f, top1, regret = [], [], [], []
    pair_hit = pair_n = 0
    seen = 0
    with torch.no_grad():
        for _, ep in val_eps:
            if seen >= args.max_cycles:
                break
            g = f[ep]
            qa, qp = str(g.attrs["q_approach"]), str(g.attrs["q_place"])
            C, k = g["score"].shape
            for c in range(C):
                if seen >= args.max_cycles:
                    break
                img = Image.open(io.BytesIO(
                    g["frames_t"][c].tobytes())).convert("RGB")
                trajs = torch.from_numpy(normalize(
                    g["trajs"][c].astype(np.float32), amin, amax))
                # one forward: k approach rows + k place rows
                p = swm.p_yes([img] * (2 * k), [qa] * k + [qp] * k,
                              torch.cat([trajs, trajs])).cpu().numpy()
                s_hat = blend(float(g["gate"][c]), p[:k], p[k:])
                s = g["score"][c].astype(np.float64)
                r = spearmanr(s_hat, s).statistic
                if np.isfinite(r):
                    rhos.append(r)
                if s.std() > 0.05:
                    if np.isfinite(r):
                        rhos_f.append(r)
                    pick = int(s_hat.argmax())
                    top1.append(float(pick == int(s.argmax())))
                    regret.append(float(s.max() - s[pick]))
                for a in range(k):
                    for b in range(a + 1, k):
                        if abs(s[a] - s[b]) >= 0.10:
                            pair_n += 1
                            pair_hit += int((s_hat[a] > s_hat[b]) ==
                                            (s[a] > s[b]))
                seen += 1
                if seen % 500 == 0:
                    print(f"[{seen}] rho_inf "
                          f"{np.mean(rhos_f) if rhos_f else 0:.3f} top1 "
                          f"{np.mean(top1) if top1 else 0:.3f} pair10 "
                          f"{pair_hit / pair_n if pair_n else 0:.3f}",
                          flush=True)

    summary = dict(
        ckpt=args.ckpt, h5=args.h5, cycles=seen,
        rho=float(np.mean(rhos)) if rhos else 0.0,
        rho_informative=float(np.mean(rhos_f)) if rhos_f else 0.0,
        rho_informative_n=len(rhos_f),
        top1_agree=float(np.mean(top1)) if top1 else 0.0,
        top1_regret=float(np.mean(regret)) if regret else 0.0,
        informative_n=len(top1),
        pair_acc10=(pair_hit / pair_n) if pair_n else 0.0, pair_n=pair_n)
    print("COMPOSE_EVAL_DONE " + json.dumps(summary), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(summary, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
