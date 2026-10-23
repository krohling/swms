"""Encoder warm-up: train the trajectory projector ALONE against frozen base
Qwen (no LoRA anywhere), pointwise soft-BCE on teacher scores.

Purpose (team call 2026-10-16): previous runs trained the action encoder and
the LM adapters simultaneously, so encoder quality was never observable in
isolation. This stage answers "can the encoder learn a useful action
representation at all?" and produces an initialization for full training.

Fixed learning rate, no schedule: watch wandb for the loss plateau and pick a
checkpoint (saved every ckpt_every steps); there is no automated stop.

Usage:
  python swm-next/swm_lora/warmup_encoder.py --config swm-next/swm_lora/warmup_encoder.yaml
"""
import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import QwenSWM                                    # noqa: E402
from data import (CycleDataset, action_stats, collate_cycles,  # noqa: E402
                  episode_split, save_split)
from train4 import within_cycle_rho                          # noqa: E402


def build_weight_fn(h5_paths, episodes, mode, bin_width=0.05):
    """Per-sample loss weights from train-split label statistics.

    Modes: "invbin" — each trajectory weighted inversely to how common its
    teacher-score bin is (bin_width resolution; 43% of labels sit in the
    first 0.05 bin); "spread" — every trajectory in a list weighted by the
    list's teacher-score std (sub-JND tie lists are 46% of lists but carry
    ~no ranking signal, yet kept ~35% of gradient mass on the run-4 ckpt);
    "invbin_spread" — product of the two. Every mode is normalized to mean
    1.0 over the train set so the effective LR matches the uniform baseline.
    The val pass stays uniform for all modes.

    Returns f(targets (m, k) ndarray) -> weights (m, k) ndarray.
    """
    import h5py
    nb = int(round(1.0 / bin_width))
    spreads, bins = [], []
    by_pi = {}
    for pi, ep in episodes:
        by_pi.setdefault(pi, []).append(ep)
    for pi, eps in by_pi.items():
        with h5py.File(h5_paths[pi], "r") as f:
            for ep in eps:
                lab = f[ep]["labels"][:]
                for col in (0, 1):                     # approach, place
                    t = lab[:, :, col].astype(np.float64)
                    spreads.append(t.std(axis=1))
                    bins.append(np.minimum((t / bin_width).astype(np.int64),
                                           nb - 1))
    spreads = np.concatenate(spreads)                  # (L,)   per list
    bins = np.concatenate(bins, axis=0)                # (L, k) per trajectory
    counts = np.bincount(bins.ravel(), minlength=nb).astype(np.float64)
    inv = 1.0 / np.maximum(counts, 1.0)
    inv /= inv[bins.ravel()].mean()                    # mean 1 over trajectories
    mean_spread = spreads.mean()
    comb_mean = (inv[bins] * (spreads / mean_spread)[:, None]).mean()
    print(f"weighting={mode}: {len(spreads):,} train lists, mean spread "
          f"{mean_spread:.4f}, bin counts {counts.astype(int).tolist()}",
          flush=True)

    def fn(targets):                                   # (m, k) -> (m, k)
        b = np.minimum((targets / bin_width).astype(np.int64), nb - 1)
        w_inv = inv[b]
        w_sp = (targets.std(axis=1) / mean_spread)[:, None]
        if mode == "invbin":
            return w_inv
        if mode == "spread":
            return np.broadcast_to(w_sp, targets.shape).copy()
        if mode == "invbin_spread":
            return w_inv * w_sp / comb_mean
        raise ValueError(f"unknown weighting mode {mode!r}")
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    device = cfg.get("device", "cuda")
    torch.manual_seed(cfg.get("seed", 0))
    np.random.seed(cfg.get("seed", 0))

    amin, amax = action_stats(cfg["diffusion_ckpt"])
    train_eps, val_eps = episode_split(cfg["h5_paths"],
                                       val_frac=cfg.get("val_frac", 0.05),
                                       seed=cfg.get("seed", 0))
    save_split(f"{out}/split.json", train_eps, val_eps, cfg["h5_paths"])
    qsel = cfg.get("questions", "planner")
    train_ds = CycleDataset(cfg["h5_paths"], train_eps, amin, amax, questions=qsel)
    val_ds = CycleDataset(cfg["h5_paths"], val_eps, amin, amax, questions=qsel)
    print(f"train lists: {len(train_ds):,}  val lists: {len(val_ds):,}", flush=True)

    model = QwenSWM(model_id=cfg["model_id"], action_dim=cfg["action_dim"],
                    horizon=cfg["horizon"], device=device,
                    apply_lora=False,                       # base LM, frozen
                    proj_init_scale=float(cfg.get("proj_init_scale", 0.1)),
                    proj_hidden=cfg.get("proj_hidden", 512),
                    proj_learnable_scale=bool(cfg.get("learnable_scale", True)),
                    traj_cumsum=bool(cfg.get("traj_cumsum", False)))
    params = model.trainable_parameters()
    print(f"trainable {sum(p.numel() for p in params)/1e6:.2f}M "
          f"(projector only)", flush=True)

    opt = torch.optim.AdamW(params, lr=float(cfg["learning_rate"]))

    wmode = cfg.get("weighting", "none")
    weight_fn = None
    if wmode != "none":
        weight_fn = build_weight_fn(cfg["h5_paths"], train_eps, wmode,
                                    bin_width=float(cfg.get("weight_bin", 0.05)))

    from wandb_safe import SafeRun
    run = SafeRun(project=cfg.get("wandb_project", "swm-lora"),
                  name=cfg.get("run_name"), config=cfg, resume="allow",
                  id=cfg.get("run_name"))

    mc = int(cfg["micro_cycles"])
    accum = int(cfg["accum_cycles"])
    max_steps = int(cfg.get("max_steps", 20000))
    val_every = int(cfg.get("val_every", 500))
    ckpt_every = int(cfg.get("ckpt_every", 500))
    clip = float(cfg.get("max_grad_norm", 1.0))

    def val_pass(max_lists=200):
        model.model.eval()
        dl = DataLoader(val_ds, batch_size=mc, collate_fn=collate_cycles,
                        num_workers=2)
        bces, rhos_f = [], []
        with torch.no_grad():
            seen = 0
            for b in dl:
                if seen >= max_lists:
                    break
                l = model.answer_logit(b["images"], b["questions"], b["trajs"])
                tt = b["targets"].to(l.device)
                bces.append(float(model.loss(l, tt)))
                p = torch.sigmoid(l).cpu().numpy()
                tt_np = b["targets"].numpy().reshape(b["m"], b["k"])
                for mi in range(b["m"]):
                    if tt_np[mi].std() > 0.05:
                        rhos_f += within_cycle_rho(
                            p.reshape(b["m"], b["k"])[mi], tt_np[mi], b["k"], 1)
                seen += b["m"]
        model.model.train()
        return dict(bce=float(np.mean(bces)),
                    within_cycle_rho_informative=(float(np.mean(rhos_f))
                                                  if rhos_f else 0.0),
                    rho_informative_n=len(rhos_f))

    model.model.train()
    g = torch.Generator().manual_seed(cfg.get("seed", 0))
    dl = DataLoader(train_ds, batch_size=mc, shuffle=True, generator=g,
                    collate_fn=collate_cycles, num_workers=4, drop_last=True)
    it = iter(dl)
    step = 0
    while step < max_steps:
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(accum):
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl); b = next(it)
            l = model.answer_logit(b["images"], b["questions"], b["trajs"])
            if weight_fn is None:
                bce = model.loss(l, b["targets"].to(l.device))
            else:
                w = torch.from_numpy(
                    weight_fn(b["targets"].numpy().reshape(b["m"], b["k"]))
                ).ravel().to(l.device, l.dtype)
                per = torch.nn.functional.binary_cross_entropy_with_logits(
                    l, b["targets"].to(l.device, l.dtype), reduction="none")
                bce = (w * per).mean()
            (bce / accum).backward()
            acc += float(bce) / accum
        gn = torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step(); step += 1
        if step % 20 == 0:
            run.log(dict(train_bce=acc, grad_norm=float(gn),
                         proj_scale=float(model.projector.scale.detach())),
                    step=step)
        if step % val_every == 0:
            m = val_pass()
            run.log({f"val_{k}": v for k, v in m.items()}, step=step)
            print(f"step {step}: val {m}", flush=True)
        if step % ckpt_every == 0:
            model.save_adapters(f"{out}/ckpt_step{step}")   # projector.pt only
    print("WARMUP_DONE", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
