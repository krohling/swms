"""Full held-out validation pass over saved checkpoints (Kevin 2026-11-10).

The trainer's val_bce scores only the first val_max_lists items of an
unshuffled loader -- at k=1 that is ~100 states from ~3 episodes. This
re-scores checkpoints on the WHOLE val split (the run's own split.json) with
the same soft-target BCE, and also reports the first-200-item subset, which
should reproduce the trainer's logged val_bce at that step (consistency
check). Per-item BCEs are saved alongside so checkpoints can be compared
paired, item for item.

Usage (from a full checkout; relative paths in the training config resolve
against --root, the checkout the run trained in):
  python swm-next/swm_lora/eval_val_full.py \
      --config swm-next/swm_lora/train_exp4a.yaml \
      --root $WORK/swm-followup/swms-exp4 --steps 16000 42000 64000 90000 \
      --out-dir experiments/exp4a
Writes <out-dir>/val_full_step<N>.json (+ .npz); existing JSONs are skipped.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import QwenSWM                                   # noqa: E402
from data import CycleDataset, action_stats, collate_cycles  # noqa: E402


def binary_entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def auc(score, y):
    from scipy.stats import rankdata
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    R = lambda p: p if os.path.isabs(p) else os.path.join(args.root, p)  # noqa: E731
    run_dir = R(cfg["out_dir"])
    os.makedirs(args.out_dir, exist_ok=True)

    split = json.load(open(f"{run_dir}/split.json"))
    assert [os.path.basename(p) for p in split["h5_paths"]] == \
        [os.path.basename(p) for p in cfg["h5_paths"]], "split/config h5 mismatch"
    h5 = [R(p) for p in cfg["h5_paths"]]
    val_eps = [tuple(e) for e in split["val"]]
    amin, amax = action_stats(R(cfg["diffusion_ckpt"]))
    ds = CycleDataset(h5, val_eps, amin, amax,
                      questions=cfg.get("questions", "planner"))
    ep_of = np.array([val_eps.index((pi, ep)) for pi, ep, *_ in ds.index])
    col_of = np.array([col for *_, col, _ in ds.index])
    print(f"val items: {len(ds):,} from {len(val_eps)} episodes", flush=True)

    model = QwenSWM(model_id=cfg["model_id"], action_dim=cfg["action_dim"],
                    horizon=cfg["horizon"], lora_r=cfg["lora_r"],
                    lora_alpha=cfg["lora_alpha"], lora_dropout=0.0,
                    device=cfg.get("device", "cuda"),
                    lora_targets=cfg.get("lora_targets", "all-linear"),
                    proj_hidden=cfg.get("proj_hidden", 512),
                    proj_init_scale=float(cfg.get("proj_init_scale", 0.1)),
                    proj_learnable_scale=bool(cfg.get("learnable_scale", True)),
                    traj_cumsum=bool(cfg.get("traj_cumsum", False)))

    for step in args.steps:
        out = f"{args.out_dir}/val_full_step{step}.json"
        if os.path.exists(out):
            print(f"SKIP {out}", flush=True)
            continue
        ckpt = f"{run_dir}/ckpt_step{step}"
        model.load_adapters(ckpt)
        model.model.eval()
        t0 = time.time()
        dl = DataLoader(ds, batch_size=args.batch, collate_fn=collate_cycles,
                        num_workers=4)
        bce, prob, tgt = [], [], []
        with torch.no_grad():
            for b in dl:
                lg = model.answer_logit(b["images"], b["questions"], b["trajs"])
                tt = b["targets"].to(lg.device)
                bce += torch.nn.functional.binary_cross_entropy_with_logits(
                    lg.float(), tt.float(), reduction="none").cpu().tolist()
                prob += torch.sigmoid(lg.float()).cpu().tolist()
                tgt += b["targets"].tolist()
                if len(bce) % 2000 < args.batch:
                    print(f"  step{step}: {len(bce):,}/{len(ds):,} "
                          f"({len(bce) / (time.time() - t0):.1f} items/s)", flush=True)
        bce, prob, tgt = np.array(bce), np.array(prob), np.array(tgt)
        pos = tgt > 0.5
        per_ep = np.array([bce[ep_of == e].mean() for e in np.unique(ep_of)])
        m = tgt.mean()
        res = dict(
            ckpt=ckpt, step=step, config=args.config, n_items=len(bce),
            n_episodes=len(per_ep),
            bce=float(bce.mean()),
            bce_se_episode=float(per_ep.std(ddof=1) / np.sqrt(len(per_ep))),
            bce_first200=float(bce[:200].mean()),
            bce_by_question={q: float(bce[col_of == c].mean())
                             for q, c in (("approach", 0), ("place", 1))},
            bce_pos=float(bce[pos].mean()), bce_neg=float(bce[~pos].mean()),
            bce_balanced=float(0.5 * (bce[pos].mean() + bce[~pos].mean())),
            pos_frac=float(pos.mean()), mae=float(np.abs(prob - tgt).mean()),
            auc=auc(prob, pos),
            entropy_floor=float(binary_entropy(tgt).mean()),
            prior_bce=float(-(tgt * np.log(m) + (1 - tgt) * np.log(1 - m)).mean()),
            seconds=round(time.time() - t0, 1))
        np.savez_compressed(out.replace(".json", ".npz"), bce=bce.astype(np.float32),
                            prob=prob.astype(np.float32), episode=ep_of, col=col_of)
        json.dump(res, open(out, "w"), indent=1)
        print(f"VAL_FULL step{step}: bce {res['bce']:.4f} (se {res['bce_se_episode']:.4f}) "
              f"first200 {res['bce_first200']:.4f} auc {res['auc']:.3f} "
              f"floor {res['entropy_floor']:.4f} prior {res['prior_bce']:.4f} "
              f"[{res['seconds']:.0f}s]", flush=True)
    print("VAL_FULL_DONE", flush=True)


if __name__ == "__main__":
    main()
