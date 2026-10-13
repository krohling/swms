"""Train a Qwen3-VL SWM selector: all k candidate trajectories in ONE context,
k-way soft cross-entropy over digit answer tokens "1".."k".

Each sample is one (planning cycle, question): the frame plus all k sibling
trajectories as k blocks of placeholder tokens. The model answers which
candidate best achieves the question; the target is the teacher's softmax over
the k scores at temperature tau (~2x teacher JND), so near-ties train toward
uniform and clear winners toward sharp picks. Candidates are shuffled per
sample (ladder order carries label signal otherwise).

Validation: soft-CE, top-1 agreement and selection regret on teacher-rankable
lists, and within-cycle Spearman rho from the digit logits (directly
comparable to the scoring-arm metric).

Usage:
  python swm-next/swm_lora/train_sel.py --config swm-next/swm_lora/train_sel_hybrid.yaml
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
from model import QwenSWMSelector                            # noqa: E402
from data import (SelectionCycleDataset, action_stats,       # noqa: E402
                  collate_selection, episode_split, save_split)


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
    train_ds = SelectionCycleDataset(cfg["h5_paths"], train_eps, amin, amax,
                                     questions=qsel)
    val_ds = SelectionCycleDataset(cfg["h5_paths"], val_eps, amin, amax,
                                   questions=qsel, fixed_perm=True)
    print(f"train lists: {len(train_ds):,}  val lists: {len(val_ds):,}",
          flush=True)

    ml = int(cfg["micro_lists"])                  # lists per micro-step
    accum = int(cfg["accum_lists"])               # micro-steps per opt step
    tau = float(cfg.get("tau", 0.1))
    epochs = float(cfg.get("epochs", 3))
    steps_per_epoch = len(train_ds) // (ml * accum)
    total_steps = int(epochs * steps_per_epoch)
    warmup = int(cfg.get("warmup_frac", 0.03) * total_steps)

    model = QwenSWMSelector(
        k=int(cfg.get("k", 8)), model_id=cfg["model_id"],
        action_dim=cfg["action_dim"], horizon=cfg["horizon"],
        lora_r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg.get("lora_dropout", 0.05), device=device,
        proj_init_scale=float(cfg.get("proj_init_scale", 0.1)),
        traj_cumsum=bool(cfg.get("traj_cumsum", False)))
    params = model.trainable_parameters()
    print(f"trainable {sum(p.numel() for p in params)/1e6:.1f}M | "
          f"total steps {total_steps} (eff {ml*accum} lists/step)", flush=True)

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
    # warm start (fresh runs only; a resume state overrides it below): load
    # LoRA+projector from a scoring-arm checkpoint so trajectory-outcome
    # structure pre-exists and the digit readout only learns the argmax wiring
    # (cold selection-CE starts collapse into the uniform saddle — run5 leg 1)
    if cfg.get("init_adapters"):
        model.load_adapters(cfg["init_adapters"])
        print(f"warm-started from {cfg['init_adapters']}", flush=True)
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
        from scipy.stats import spearmanr
        model.model.eval()
        dl = DataLoader(val_ds, batch_size=ml, collate_fn=collate_selection,
                        num_workers=2)
        ces, rhos, rhos_f, top1, regret = [], [], [], [], []
        with torch.no_grad():
            seen = 0
            for b in dl:
                if seen >= max_lists:
                    break
                z = model.selection_logits(b["images"], b["questions"],
                                           b["trajs"])
                tt = b["targets"].to(z.device)
                ces.append(float(model.selection_loss(z, tt, tau=tau)))
                zn, tn = z.cpu().numpy(), b["targets"].numpy()
                for mi in range(b["m"]):
                    if len(set(np.round(tn[mi], 4))) > 1:
                        r = spearmanr(zn[mi], tn[mi]).statistic
                        if np.isfinite(r):
                            rhos.append(r)
                    if tn[mi].std() > 0.05:
                        r = spearmanr(zn[mi], tn[mi]).statistic
                        if np.isfinite(r):
                            rhos_f.append(r)
                        pick = int(zn[mi].argmax())
                        top1.append(float(pick == int(tn[mi].argmax())))
                        regret.append(float(tn[mi].max() - tn[mi][pick]))
                seen += b["m"]
        model.model.train()
        return dict(ce=float(np.mean(ces)),
                    within_cycle_rho=float(np.mean(rhos)) if rhos else 0.0,
                    within_cycle_rho_informative=(float(np.mean(rhos_f))
                                                  if rhos_f else 0.0),
                    rho_informative_n=len(rhos_f), rho_n=len(rhos),
                    top1_acc=float(np.mean(top1)) if top1 else 0.0,
                    regret=float(np.mean(regret)) if regret else 0.0)

    clip = float(cfg.get("max_grad_norm", 1.0))
    ckpt_every = int(cfg.get("ckpt_every", 1000))
    resume_every = int(cfg.get("resume_every", 500))
    val_every = int(cfg.get("val_every", 500))
    model.model.train()
    step = step0
    g = torch.Generator().manual_seed(cfg.get("seed", 0) + step0)
    dl = DataLoader(train_ds, batch_size=ml, shuffle=True, generator=g,
                    collate_fn=collate_selection, num_workers=4, drop_last=True)
    it = iter(dl)
    while step < total_steps:
        opt.zero_grad(set_to_none=True)
        acc_ce = 0.0
        for _ in range(accum):
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl); b = next(it)
            z = model.selection_logits(b["images"], b["questions"], b["trajs"])
            ce = model.selection_loss(z, b["targets"].to(z.device), tau=tau)
            (ce / accum).backward()
            acc_ce += float(ce) / accum
        gn = torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step(); sched.step(); step += 1
        if step % 20 == 0:
            run.log(dict(train_ce=acc_ce,
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
            art = _w.Artifact("swm-lora-sel-ckpt", type="model",
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
