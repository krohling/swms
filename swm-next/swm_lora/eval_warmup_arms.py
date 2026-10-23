"""Deep offline eval of warm-up checkpoints: planner-aligned arm comparison.

Loads the frozen base once, then swaps projector checkpoints and scores each
on the SAME deterministic val subset (first --max-lists lists of the shared
episode_split val set, no shuffle), so numbers are comparable across arms.

Metrics (pooled and per question), chosen to separate calibration from the
quantities the planner actually consumes:
  - bce            uniform soft-target BCE (calibration + fit)
  - rho_inf        within-cycle Spearman on informative lists (teacher std
                   > 0.05), mean +/- s.e.
  - top1_regret    teacher_score[argmax model] vs teacher max — the planner
                   executes argmax, so this is the direct execution cost
                   (informative lists)
  - top1_agree     argmax match rate (informative lists)
  - pair_acc       pairwise order accuracy on pairs the teacher separates by
                   >= 0.10 (2x JND): ranking skill only, immune to tie noise
                   and calibration

Usage:
  python swm-next/swm_lora/eval_warmup_arms.py \
      --config swm-next/swm_lora/warmup_noscale.yaml \
      --ckpt noscale=outputs_swm_lora/warmup_noscale/ckpt_step6000 \
      --ckpt wns-spread=outputs_swm_lora/warmup_wns_spread/ckpt_step6000 \
      --max-lists 5000 --out eval_warmup_arms.json
"""
import argparse
import json
import os

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import QwenSWM                                     # noqa: E402
from data import (CycleDataset, action_stats, collate_cycles,  # noqa: E402
                  episode_split)

PAIR_SEP = 0.10          # 2x measured judge JND
INF_STD = 0.05


def eval_ckpt(model, dl, max_lists):
    stats = {q: dict(n=0, n_inf=0, bce_sum=0.0, bce_n=0, rhos=[],
                     regret=[], agree=[], pair_ok=0, pair_n=0)
             for q in ("approach", "place")}
    seen = 0
    with torch.no_grad():
        for b in dl:
            if seen >= max_lists:
                break
            logit = model.answer_logit(b["images"], b["questions"], b["trajs"])
            tt = b["targets"].to(logit.device, logit.dtype)
            per = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, tt, reduction="none").float().cpu().numpy()
            p = torch.sigmoid(logit).float().cpu().numpy().reshape(b["m"], b["k"])
            t = b["targets"].numpy().reshape(b["m"], b["k"])
            per = per.reshape(b["m"], b["k"])
            for mi in range(b["m"]):
                q = "approach" if "grasping" in b["questions"][mi * b["k"]] \
                    else "place"
                s = stats[q]
                s["n"] += 1
                s["bce_sum"] += float(per[mi].sum()); s["bce_n"] += b["k"]
                dt = t[mi][:, None] - t[mi][None, :]
                dp = p[mi][:, None] - p[mi][None, :]
                sep = np.triu(np.abs(dt) >= PAIR_SEP, 1)
                s["pair_n"] += int(sep.sum())
                s["pair_ok"] += int((np.sign(dp[sep]) == np.sign(dt[sep])).sum())
                if t[mi].std() > INF_STD:
                    s["n_inf"] += 1
                    r = spearmanr(p[mi], t[mi]).statistic
                    if np.isfinite(r):
                        s["rhos"].append(float(r))
                    j = int(p[mi].argmax())
                    s["regret"].append(float(t[mi].max() - t[mi][j]))
                    s["agree"].append(float(j == int(t[mi].argmax())))
            seen += b["m"]
    out = {}
    for q, s in stats.items():
        rh = np.array(s["rhos"]) if s["rhos"] else np.array([np.nan])
        out[q] = dict(
            n_lists=s["n"], n_informative=s["n_inf"],
            bce=s["bce_sum"] / max(s["bce_n"], 1),
            rho_inf=float(np.nanmean(rh)),
            rho_inf_se=float(np.nanstd(rh) / max(np.sqrt(len(rh)), 1)),
            top1_regret=float(np.mean(s["regret"])) if s["regret"] else None,
            top1_agree=float(np.mean(s["agree"])) if s["agree"] else None,
            pair_acc=s["pair_ok"] / max(s["pair_n"], 1), pair_n=s["pair_n"])
    tot = dict(
        n_lists=sum(s["n"] for s in stats.values()),
        bce=sum(s["bce_sum"] for s in stats.values())
        / max(sum(s["bce_n"] for s in stats.values()), 1),
        rho_inf=float(np.mean([r for s in stats.values() for r in s["rhos"]])),
        top1_regret=float(np.mean([r for s in stats.values()
                                   for r in s["regret"]])),
        top1_agree=float(np.mean([a for s in stats.values()
                                  for a in s["agree"]])),
        pair_acc=sum(s["pair_ok"] for s in stats.values())
        / max(sum(s["pair_n"] for s in stats.values()), 1))
    out["pooled"] = tot
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True,
                    help="any warm-up yaml; supplies data params")
    ap.add_argument("--ckpt", action="append", required=True,
                    metavar="NAME=DIR", help="repeatable")
    ap.add_argument("--max-lists", type=int, default=5000)
    ap.add_argument("--batch-lists", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    amin, amax = action_stats(cfg["diffusion_ckpt"])
    _, val_eps = episode_split(cfg["h5_paths"],
                               val_frac=cfg.get("val_frac", 0.05),
                               seed=cfg.get("seed", 0))
    val_ds = CycleDataset(cfg["h5_paths"], val_eps, amin, amax,
                          questions=cfg.get("questions", "planner"))
    print(f"val lists available: {len(val_ds):,}; evaluating first "
          f"{args.max_lists:,} (deterministic order)", flush=True)

    model = QwenSWM(model_id=cfg["model_id"], action_dim=cfg["action_dim"],
                    horizon=cfg["horizon"], device=cfg.get("device", "cuda"),
                    apply_lora=False,
                    proj_hidden=cfg.get("proj_hidden", 512),
                    traj_cumsum=bool(cfg.get("traj_cumsum", False)))
    model.model.eval()

    results = {}
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        sd = torch.load(f"{path}/projector.pt", map_location=model.device)
        model.projector.load_state_dict(sd)
        scale = float(model.projector.scale.detach())
        print(f"\n== {name} ({path}; scale={scale:.4f})", flush=True)
        dl = DataLoader(val_ds, batch_size=args.batch_lists, shuffle=False,
                        collate_fn=collate_cycles, num_workers=2)
        res = eval_ckpt(model, dl, args.max_lists)
        results[name] = dict(ckpt=path, **res)
        for q in ("approach", "place", "pooled"):
            r = res[q]
            print(f"  {q:>8}: bce {r['bce']:.4f}  rho_inf {r['rho_inf']:+.3f}"
                  + (f"±{r['rho_inf_se']:.3f}" if q != "pooled" else "")
                  + f"  regret {r['top1_regret']:.4f}"
                  f"  agree {r['top1_agree']:.3f}  pair_acc {r['pair_acc']:.3f}",
                  flush=True)
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}", flush=True)
    print("EVAL_ARMS_DONE", flush=True)


if __name__ == "__main__":
    main()
