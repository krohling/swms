"""Train a Qwen3-VL SWM LoRA on branched proposal datasets (value
distillation on teacher p_yes).

Loss: pointwise soft-target BCE per proposal — for student logit l_i and
teacher prob p_i:
  L = mean_i [ softplus(l_i) - p_i * l_i ]        (BCEWithLogits, soft target)
Optionally, lambda_rank > 0 adds a listwise KL over each cycle's proposals:
  L_rank = KL( softmax(t/T) || softmax(l/T) ),  t_i = logit(clip(p_i)).

Batching: each batch element is one (cycle, question) list — the frame plus
all k sibling proposals forwarded together (micro_cycles lists per micro-step,
accumulated to the effective batch).

Data: questions are enumerated from the dataset files themselves; cfg
"questions" selects "planner" | "all" | [explicit list of question strings].

Validation: loss/MAE, within-cycle Spearman rho (rank correlation across each
held-out cycle's siblings), and a base-drift probe (empty-trajectory p_yes vs
step-0 outputs). Checkpoints: eval/export adapters every ckpt_every steps
(wandb artifacts) + atomic full resume states (optimizer/scheduler/RNG,
keep-2); --resume auto restarts from the newest valid state.

Usage:
  python swm-next/swm_lora/train4.py --config swm-next/swm_lora/train4_hybrid.yaml
"""
import argparse
import glob
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import QwenSWM                                   # noqa: E402
from data import (CycleDataset, action_stats, collate_cycles,  # noqa: E402
                  episode_split, save_split)


def rank_loss(logits, targets, k, m, T=1.0):
    """Listwise KL over each cycle's k proposals. logits/targets: (m*k,)."""
    l = logits.view(m, k) / T
    t = torch.logit(targets.view(m, k).clamp(1e-4, 1 - 1e-4)) / T
    return torch.nn.functional.kl_div(
        torch.log_softmax(l, dim=1), torch.softmax(t, dim=1),
        reduction="batchmean")


def within_cycle_rho(p, t, k, m):
    from scipy.stats import spearmanr
    rs = []
    p, t = p.reshape(m, k), t.reshape(m, k)
    for i in range(m):
        if len(set(np.round(t[i], 4))) > 1:
            r = spearmanr(p[i], t[i]).statistic
            if np.isfinite(r):
                rs.append(r)
    return rs


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
                                       val_frac=cfg.get("val_frac", 0.1),
                                       seed=cfg.get("seed", 0))
    save_split(f"{out}/split.json", train_eps, val_eps, cfg["h5_paths"])
    qsel = cfg.get("questions", "planner")
    train_ds = CycleDataset(cfg["h5_paths"], train_eps, amin, amax, questions=qsel)
    val_ds = CycleDataset(cfg["h5_paths"], val_eps, amin, amax, questions=qsel)
    print(f"train lists: {len(train_ds):,}  val lists: {len(val_ds):,}",
          flush=True)

    mc = int(cfg["micro_cycles"])                 # cycles per micro-step
    accum = int(cfg["accum_cycles"])              # micro-steps per opt step
    lam = float(cfg.get("lambda_rank", 0.0))
    T = float(cfg.get("rank_temperature", 1.0))
    epochs = float(cfg.get("epochs", 4))
    steps_per_epoch = len(train_ds) // (mc * accum)
    total_steps = int(epochs * steps_per_epoch)
    warmup = int(cfg.get("warmup_frac", 0.03) * total_steps)

    model = QwenSWM(model_id=cfg["model_id"], action_dim=cfg["action_dim"],
                    horizon=cfg["horizon"], lora_r=cfg["lora_r"],
                    lora_alpha=cfg["lora_alpha"],
                    lora_dropout=cfg.get("lora_dropout", 0.05), device=device,
                    proj_init_scale=float(cfg.get("proj_init_scale", 0.1)))
    params = model.trainable_parameters()
    print(f"trainable {sum(p.numel() for p in params)/1e6:.1f}M | "
          f"total steps {total_steps} (eff {mc*accum} lists/step)", flush=True)

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
        for p in reversed(sorted(glob.glob(f"{out}/resume_step*.pt"),
                key=lambda q: int(q.split("step")[-1].split(".")[0]))):
            try:
                st = torch.load(p, map_location="cpu", weights_only=False)
                model.load_adapters(st["adapters_dir"])
                opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
                torch.set_rng_state(st["rng_torch"])
                if torch.cuda.is_available() and st.get("rng_cuda") is not None:
                    torch.cuda.set_rng_state_all(st["rng_cuda"])
                np.random.set_state(st["rng_np"])
                step0 = st["step"]
                print(f"resumed @ {step0}", flush=True)
                break
            except Exception as e:
                print(f"skip resume {p}: {e}", flush=True)

    import wandb
    run = wandb.init(project=cfg.get("wandb_project", "swm-lora"),
                     name=cfg.get("run_name"), config=cfg, resume="allow",
                     id=cfg.get("run_name"))

    def val_pass(max_lists=200):
        model.model.eval()
        dl = DataLoader(val_ds, batch_size=mc, collate_fn=collate_cycles,
                        num_workers=2)
        bces, kls, rhos, ps, ts = [], [], [], [], []
        drift = []
        with torch.no_grad():
            seen = 0
            for b in dl:
                if seen >= max_lists:
                    break
                l = model.answer_logit(b["images"], b["questions"], b["trajs"])
                tt = b["targets"].to(l.device)
                bces.append(float(model.loss(l, tt)))
                kls.append(float(rank_loss(l, tt, b["k"], b["m"], T)))
                p = torch.sigmoid(l).cpu().numpy()
                rhos += within_cycle_rho(p, b["targets"].numpy(), b["k"], b["m"])
                ps += p.tolist(); ts += b["targets"].numpy().tolist()
                if seen < 8 * mc:
                    l0 = model.answer_logit(b["images"], b["questions"],
                                            torch.zeros_like(b["trajs"]))
                    drift += torch.sigmoid(l0).cpu().numpy().tolist()
                seen += b["m"]
        base_f = f"{out}/drift_base.npy"
        if not os.path.exists(base_f):
            np.save(base_f, np.array(drift)); bd = 0.0
        else:
            base = np.load(base_f); n = min(len(base), len(drift))
            bd = float(np.abs(base[:n] - np.array(drift[:n])).mean())
        ps, ts = np.array(ps), np.array(ts)
        model.model.train()
        return dict(bce=float(np.mean(bces)), rank_kl=float(np.mean(kls)),
                    within_cycle_rho=float(np.mean(rhos)) if rhos else 0.0,
                    rho_n=len(rhos), mae=float(np.abs(ps - ts).mean()),
                    base_drift=bd)

    clip = float(cfg.get("max_grad_norm", 1.0))
    ckpt_every = int(cfg.get("ckpt_every", 1000))
    resume_every = int(cfg.get("resume_every", 500))
    val_every = int(cfg.get("val_every", 500))
    model.model.train()
    step = step0
    g = torch.Generator().manual_seed(cfg.get("seed", 0) + step0)
    dl = DataLoader(train_ds, batch_size=mc, shuffle=True, generator=g,
                    collate_fn=collate_cycles, num_workers=4, drop_last=True)
    it = iter(dl)
    while step < total_steps:
        opt.zero_grad(set_to_none=True)
        acc_bce = acc_kl = 0.0
        for _ in range(accum):
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl); b = next(it)
            l = model.answer_logit(b["images"], b["questions"], b["trajs"])
            tt = b["targets"].to(l.device)
            bce = model.loss(l, tt)
            kl = rank_loss(l, tt, b["k"], b["m"], T) if lam > 0 else None
            loss = bce + lam * kl if kl is not None else bce
            (loss / accum).backward()
            acc_bce += float(bce) / accum
            acc_kl += (float(kl) / accum) if kl is not None else 0.0
        gn = torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step(); sched.step(); step += 1
        if step % 20 == 0:
            run.log(dict(train_bce=acc_bce, train_rank_kl=acc_kl,
                         grad_norm=float(gn), lr=sched.get_last_lr()[0],
                         proj_scale=float(model.projector.scale.detach())),
                    step=step)
        if step % val_every == 0 or step == total_steps:
            m = val_pass(max_lists=int(cfg.get("val_max_lists", 200)))
            run.log({f"val_{k}": v for k, v in m.items()}, step=step)
            print(f"step {step}: val {m}", flush=True)
        if step % ckpt_every == 0 or step == total_steps:
            cdir = f"{out}/ckpt_step{step}"
            model.save_adapters(cdir)
            import wandb as _w
            art = _w.Artifact("swm-lora-rank-ckpt", type="model",
                              metadata=dict(step=step))
            art.add_dir(cdir)
            run.log_artifact(art, aliases=[f"step{step}"])
        if step % resume_every == 0 or step == total_steps:
            cdir = f"{out}/ckpt_step{step}"
            if not os.path.exists(cdir):
                model.save_adapters(cdir)
            tmp = f"{out}/resume_step{step}.pt.tmp"
            torch.save(dict(step=step, adapters_dir=cdir, opt=opt.state_dict(),
                            sched=sched.state_dict(),
                            rng_torch=torch.get_rng_state(),
                            rng_cuda=(torch.cuda.get_rng_state_all()
                                      if torch.cuda.is_available() else None),
                            rng_np=np.random.get_state()), tmp)
            os.replace(tmp, f"{out}/resume_step{step}.pt")
            for p in sorted(glob.glob(f"{out}/resume_step*.pt"),
                    key=lambda q: int(q.split("step")[-1].split(".")[0]))[:-2]:
                os.remove(p)
    print("TRAIN_DONE", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
