"""Train the Qwen3-VL SWM LoRA (value distillation on teacher p_yes).

Recipe adopted from the validated qwen_oracle.yaml run: effective batch 96
(micro 48 x accum 2 on GH200; scale micro down elsewhere), lr 1e-4, cosine +
3% warmup, clip 1.0, bf16.

Logging (wandb): train/loss, train/grad_norm (pre-clip), train/lr,
train/proj_scale per step; val/loss, val/mae, val/spearman_{approach,place},
val/ece, val/base_drift every --val-every steps.

Checkpoints:
  tier-1 eval/export (adapters + projector) every --ckpt-every steps, kept
         forever, logged as wandb artifacts (m5 eval watcher pulls these);
  tier-2 full resume state (optimizer/scheduler/step/RNG) every
         --resume-every steps, atomic tmp+rename, keep latest 2.
--resume auto restarts from the newest valid tier-2 state.

Usage:
  python swm-next/swm_lora/train.py --config swm-next/swm_lora/train_bg.yaml
"""
import argparse
import glob
import os
import shutil

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import QwenSWM                                  # noqa: E402
from data import (ProposalDataset, ExecutedChunkDataset, CycleDataset,  # noqa: E402
                  action_stats, collate, collate_cycles, episode_split,
                  save_split)


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else 0.0


def ece(p, t, bins=10):
    p, t = np.asarray(p), np.asarray(t)
    e, idx = 0.0, np.clip((p * bins).astype(int), 0, bins - 1)
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - t[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="auto", choices=["auto", "none"])
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
    train_ds = ExecutedChunkDataset(cfg["h5_paths"], train_eps, amin, amax,
                                    use_aux=cfg.get("use_aux", True),
                                    binarize_aux=cfg.get("binarize_aux", False))
    val_ds = ExecutedChunkDataset(cfg["h5_paths"], val_eps, amin, amax,
                                  use_aux=cfg.get("use_aux", True),
                                  binarize_aux=cfg.get("binarize_aux", False))
    # cross-dataset ranking probe: branched files' held-out episodes give the
    # within-cycle siblings that k=1 data cannot (deployment metric)
    _, rank_eps = episode_split(cfg["rank_h5_paths"], val_frac=0.1,
                                seed=cfg.get("seed", 0))
    rank_ds = CycleDataset(cfg["rank_h5_paths"], rank_eps, amin, amax)
    print(f"train samples: {len(train_ds):,}  val: {len(val_ds):,}  "
          f"rank lists: {len(rank_ds):,}", flush=True)

    micro = int(cfg["micro_batch"])
    accum = int(round(cfg["effective_batch"] / micro))
    epochs = float(cfg.get("epochs", 4))
    steps_per_epoch = len(train_ds) // (micro * accum)
    total_steps = int(epochs * steps_per_epoch)
    warmup = int(cfg.get("warmup_frac", 0.03) * total_steps)

    model = QwenSWM(model_id=cfg["model_id"], action_dim=cfg["action_dim"],
                    horizon=cfg["horizon"], lora_r=cfg["lora_r"],
                    lora_alpha=cfg["lora_alpha"],
                    lora_dropout=cfg.get("lora_dropout", 0.05), device=device,
                    proj_init_scale=float(cfg.get("proj_init_scale", 0.1)))
    params = model.trainable_parameters()
    n_tr = sum(p.numel() for p in params)
    print(f"trainable params: {n_tr/1e6:.1f}M  total steps: {total_steps} "
          f"(eff batch {micro*accum})", flush=True)

    opt = torch.optim.AdamW(params, lr=float(cfg["learning_rate"]),
                            weight_decay=float(cfg.get("weight_decay", 0.0)))
    from torch.optim.lr_scheduler import LambdaLR
    def lr_fn(s):
        if s < warmup:
            return s / max(1, warmup)
        t = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * min(t, 1.0)))
    sched = LambdaLR(opt, lr_fn)

    step0 = 0
    if args.resume == "auto":
        states = sorted(glob.glob(f"{out}/resume_step*.pt"),
                        key=lambda p: int(p.split("step")[-1].split(".")[0]))
        for p in reversed(states):
            try:
                st = torch.load(p, map_location="cpu", weights_only=False)
                model.load_adapters(st["adapters_dir"])
                opt.load_state_dict(st["opt"])
                sched.load_state_dict(st["sched"])
                torch.set_rng_state(st["rng_torch"])
                if torch.cuda.is_available() and st.get("rng_cuda") is not None:
                    torch.cuda.set_rng_state_all(st["rng_cuda"])
                np.random.set_state(st["rng_np"])
                step0 = st["step"]
                print(f"resumed from {p} @ step {step0}", flush=True)
                break
            except Exception as e:
                print(f"resume candidate {p} unusable: {e}", flush=True)

    import wandb
    run = wandb.init(project=cfg.get("wandb_project", "swm-lora"),
                     name=cfg.get("run_name"), config=cfg,
                     resume="allow", id=cfg.get("run_name"))

    def val_pass(max_batches=None):
        model.model.eval()
        dl = DataLoader(val_ds, batch_size=micro, collate_fn=collate,
                        num_workers=2)
        losses, ps, ts, qs = [], [], [], []
        drift_ps = []
        with torch.no_grad():
            for bi, b in enumerate(dl):
                if max_batches and bi >= max_batches:
                    break
                l = model.answer_logit(b["images"], b["questions"], b["trajs"])
                losses.append(float(model.loss(l, b["targets"].to(l.device))))
                p = torch.sigmoid(l).cpu().numpy()
                ps += p.tolist(); ts += b["targets"].numpy().tolist()
                qs += b["questions"]
                if bi < 4:   # drift probe: empty trajectory vs cached base
                    l0 = model.answer_logit(b["images"], b["questions"],
                                            torch.zeros_like(b["trajs"]))
                    drift_ps += torch.sigmoid(l0).cpu().numpy().tolist()
        ps, ts = np.array(ps), np.array(ts)
        is_app = np.array(["grasping" in q for q in qs])
        base = np.load(f"{out}/drift_base.npy") if \
            os.path.exists(f"{out}/drift_base.npy") else None
        if base is None:
            np.save(f"{out}/drift_base.npy", np.array(drift_ps))
            drift = 0.0
        else:
            n = min(len(base), len(drift_ps))
            drift = float(np.abs(base[:n] - np.array(drift_ps[:n])).mean())
        # deployment metric: within-cycle rho on branched held-out cycles
        from scipy.stats import spearmanr as _sp
        rdl = DataLoader(rank_ds, batch_size=1, collate_fn=collate_cycles,
                         num_workers=2)
        rhos = []
        with torch.no_grad():
            for ri, rb in enumerate(rdl):
                if ri >= int(cfg.get("rank_val_lists", 120)):
                    break
                rl = model.answer_logit(rb["images"], rb["questions"],
                                        rb["trajs"])
                rp = torch.sigmoid(rl).cpu().numpy()
                rt = rb["targets"].numpy()
                if len(set(np.round(rt, 4))) > 1:
                    rr = _sp(rp, rt).statistic
                    if np.isfinite(rr):
                        rhos.append(rr)
        model.model.train()
        return dict(loss=float(np.mean(losses)),
                    mae=float(np.abs(ps - ts).mean()),
                    spearman_approach=spearman(ps[is_app], ts[is_app]),
                    spearman_place=spearman(ps[~is_app], ts[~is_app]),
                    ece=ece(ps, ts), base_drift=drift,
                    within_cycle_rho=float(np.mean(rhos)) if rhos else 0.0)

    clip = float(cfg.get("max_grad_norm", 1.0))
    ckpt_every = int(cfg.get("ckpt_every", 1000))
    resume_every = int(cfg.get("resume_every", 500))
    val_every = int(cfg.get("val_every", 500))
    model.model.train()
    step = step0
    g = torch.Generator().manual_seed(cfg.get("seed", 0) + step0)
    dl = DataLoader(train_ds, batch_size=micro, shuffle=True, generator=g,
                    collate_fn=collate, num_workers=4, drop_last=True)
    it = iter(dl)
    while step < total_steps:
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(accum):
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl)
                b = next(it)
            l = model.answer_logit(b["images"], b["questions"], b["trajs"])
            loss = model.loss(l, b["targets"].to(l.device)) / accum
            loss.backward()
            loss_acc += float(loss)
        gn = torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step(); sched.step(); step += 1
        if step % 20 == 0:
            run.log(dict(train_loss=loss_acc, grad_norm=float(gn),
                         lr=sched.get_last_lr()[0],
                         proj_scale=float(model.projector.scale.detach())),
                    step=step)
        if step % val_every == 0 or step == total_steps:
            m = val_pass(max_batches=int(cfg.get("val_max_batches", 60)))
            run.log({f"val_{k}": v for k, v in m.items()}, step=step)
            print(f"step {step}: val {m}", flush=True)
        if step % ckpt_every == 0 or step == total_steps:
            cdir = f"{out}/ckpt_step{step}"
            model.save_adapters(cdir)
            art = wandb.Artifact(f"swm-lora-ckpt", type="model",
                                 metadata=dict(step=step))
            art.add_dir(cdir)
            run.log_artifact(art, aliases=[f"step{step}"])
        if step % resume_every == 0 or step == total_steps:
            cdir = f"{out}/ckpt_step{step}"
            if not os.path.exists(cdir):
                model.save_adapters(cdir)
            tmp = f"{out}/resume_step{step}.pt.tmp"
            torch.save(dict(step=step, adapters_dir=cdir,
                            opt=opt.state_dict(), sched=sched.state_dict(),
                            rng_torch=torch.get_rng_state(),
                            rng_cuda=(torch.cuda.get_rng_state_all()
                                      if torch.cuda.is_available() else None),
                            rng_np=np.random.get_state()), tmp)
            os.replace(tmp, f"{out}/resume_step{step}.pt")
            old = sorted(glob.glob(f"{out}/resume_step*.pt"),
                         key=lambda p: int(p.split("step")[-1].split(".")[0]))
            for p in old[:-2]:
                os.remove(p)
                d = p.replace("resume_step", "ckpt_step").replace(".pt", "")
                if os.path.exists(d) and \
                        int(d.split("step")[-1]) % ckpt_every != 0:
                    shutil.rmtree(d, ignore_errors=True)
    print("TRAIN_DONE", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
