"""Train a DinoWM-architecture world model under the SWM reproduction recipe.

This is train.py with ONE substitution: the model. Instead of fine-tuning
PaliGemma, a 64M OneShotPredictor is trained on top of a frozen Qwen3-VL ViT
encoder (architecture ported from krohling/dino_wm). Everything that is not
the model comes from the reproduction, not from the July predictor campaigns:

  dataset + sampling   saqa.py's balanced 12-stratum draw, verbatim
                       (PredictorSAQADataset only adds target/history frames)
  schedule             A.1: 64k optimizer steps, effective batch 96,
                       linear LR decay to 0, warmup 0
  optimizer            A.2: AdamW; max_grad_norm 1.0, weight_decay 0.0
  trainer              transformers.Trainer, same TrainingArguments shape
  evaluation           balanced future-QA probe on held-out trajectories,
                       scored through the frozen judge on PREDICTED latents
                       (comparable to the reproduction's balanced_acc)

Deliberate differences from train.py, all model-driven:
  - learning_rate 5e-4 (upstream DinoWM predictor default; the recipe's 1e-5
    is a full-FT VLM rate)
  - objective "cosine" (DinoWM-native latent regression) or "ce" (the
    recipe's paper-3.2 answer CE, routed through the frozen judge)
  - gradient_checkpointing off at the Trainer level: the trainable module is
    a plain nn.Module; the judge applies its own checkpointing internally
  - bf16 autocast as in the recipe; predictor params stay fp32

    python train_predictor.py --config configs/predictor_cosine.yaml [key=value ...]
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saqa import SAQAIndex  # noqa: E402
from dinowm.data import PredictorSAQADataset, predictor_collate  # noqa: E402
from dinowm.judge import StudentVQAHead  # noqa: E402
from dinowm.model import PredictorForSAQA  # noqa: E402
from dinowm.oneshot_world_model import OneShotWorldModel  # noqa: E402
from dinowm.qwen3_vl import Qwen3VLViTEncoder  # noqa: E402
from dinowm.vit_oneshot import OneShotPredictor  # noqa: E402


def load_config() -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="key=value")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    for ov in args.overrides:
        k, v = ov.split("=", 1)
        cur = cfg.get(k)
        cfg[k] = type(cur)(v) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else v
    return cfg


def build_model(cfg: dict, device) -> PredictorForSAQA:
    encoder = Qwen3VLViTEncoder(
        model_id=cfg["model_id"], image_size=int(cfg["image_size"]),
        freeze=True, output_stage=cfg.get("output_stage", "pre_merger"),
    ).to(device)
    predictor = OneShotPredictor(
        emb_dim=encoder.emb_dim, io_dim=None, action_dim=int(cfg["action_dim"]),
        num_patches=encoder.num_patches, obs_horizon=int(cfg["obs_horizon"]),
        max_action_horizon=int(cfg["max_horizon"]),
        depth=int(cfg["depth"]), heads=int(cfg["heads"]),
        mlp_dim=int(cfg["mlp_dim"]), dropout=float(cfg["dropout"]),
    ).to(device)
    wm = OneShotWorldModel(encoder=encoder, predictor=predictor).to(device)
    # The judge is required for the ce objective and used by evaluation for
    # both objectives (future-QA probe on predicted latents).
    judge = StudentVQAHead(cfg["model_id"], device=device,
                           image_size=int(cfg["image_size"]))
    model = PredictorForSAQA(wm, judge, objective=cfg["objective"],
                             cos_weight=float(cfg.get("cos_weight", 0.0)))
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n:,} (predictor only)", flush=True)
    return model


@torch.no_grad()
def evaluate(model: PredictorForSAQA, index: SAQAIndex, cfg: dict,
             n: int, batch_size: int, seed: int = 1234):
    """Future-QA accuracy on held-out trajectories, balanced over the strata,
    scored through the frozen judge on the PREDICTED future latent. Mirrors
    train.py's evaluate; also reports predicted-vs-target cosine metrics."""
    device = next(model.parameters()).device
    ds = PredictorSAQADataset(index, length=n, seed=seed,
                              obs_horizon=int(cfg["obs_horizon"]),
                              max_action_horizon=int(cfg["max_horizon"]))
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4,
                        collate_fn=predictor_collate)

    model.eval()
    per = collections.defaultdict(lambda: [0, 0])
    cos_sum = cos_id_sum = ce_sum = n_seen = 0.0
    for batch in loader:
        history = batch["history"].to(device)
        actions = batch["actions"].to(device)
        action_mask = batch["action_mask"].to(device)
        target = batch["target"].to(device)
        label_yes = batch["label_yes"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.wm(history, actions, action_mask, target)
            img_embeds = model.judge.merge(out["z_pred"])
            idx_by_q = collections.defaultdict(list)
            for i, q in enumerate(batch["question"]):
                idx_by_q[q].append(i)
            p_yes = torch.zeros(history.shape[0], device=device)
            for q, idxs in idx_by_q.items():
                pi = model.judge._prompt(q)
                p_yes[idxs] = model.judge.p_yes(pi, img_embeds[idxs]).float()
                # Val CE loss (the training objective of the ce arm) on the
                # same probe: accuracy alone can stay flat while CE rises,
                # which is the overfitting signature we watch for.
                ce_sum += float(model.judge.answer_ce(pi, img_embeds[idxs],
                                                      label_yes[idxs]))
        pred_yes = (p_yes >= 0.5).cpu().numpy()
        for qt, ans, p in zip(batch["qtype"], batch["label_yes"].numpy(), pred_yes):
            per[qt][1] += 1
            per[qt][0] += int(bool(p) == bool(ans >= 0.5))
        bs = history.shape[0]
        cos_sum += float(out["cos_sim"]) * bs
        cos_id_sum += float(out["cos_sim_identity"]) * bs
        n_seen += bs
    model.train()

    metrics = {f"eval/{t}": ok / n_ for t, (ok, n_) in per.items()}
    metrics["eval/balanced_acc"] = float(np.mean(list(metrics.values())))
    metrics["eval/cos_sim"] = cos_sum / max(1, n_seen)
    metrics["eval/cos_sim_identity"] = cos_id_sum / max(1, n_seen)
    metrics["eval/cos_sim_gain"] = metrics["eval/cos_sim"] - metrics["eval/cos_sim_identity"]
    metrics["eval/cos_loss"] = 1.0 - metrics["eval/cos_sim"]
    metrics["eval/ce_loss"] = ce_sum / max(1, n_seen)
    return metrics


def main() -> None:
    from transformers import Trainer, TrainerCallback, TrainingArguments

    cfg = load_config()
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    # Trainer reads the wandb project from the environment; without this it
    # silently logs to wandb's default project rather than ours.
    if cfg.get("wandb_project"):
        os.environ.setdefault("WANDB_PROJECT", cfg["wandb_project"])

    files = [cfg["noisy_hdf5"], cfg["play_hdf5"]]
    train_idx = SAQAIndex.build(files, split="train", val_frac=cfg["val_frac"])
    val_idx = SAQAIndex.build(files, split="val", val_frac=cfg["val_frac"])

    steps = int(cfg["max_steps"])
    eff_batch = int(cfg["effective_batch"])
    micro = int(cfg["micro_batch"])
    assert eff_batch % micro == 0, "effective batch must be a multiple of micro batch"
    accum = eff_batch // micro

    print("=== G2: sampler balance (train split) ===", flush=True)
    train_idx.report(draws=steps * eff_batch)

    device = torch.device("cuda")
    model = build_model(cfg, device)

    train_ds = PredictorSAQADataset(train_idx, length=steps * eff_batch,
                                    seed=cfg["seed"],
                                    obs_horizon=int(cfg["obs_horizon"]),
                                    max_action_horizon=int(cfg["max_horizon"]))

    class EvalCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if state.global_step and state.global_step % int(cfg["eval_every"]) == 0:
                m = evaluate(model, val_idx, cfg, int(cfg["eval_samples"]), micro)
                m["step"] = state.global_step
                print(f"eval @ {state.global_step}: {m}", flush=True)
                if "wandb" in (args.report_to or []):
                    import wandb
                    wandb.log(m, step=state.global_step)

    class TrainMetricsCallback(TrainerCallback):
        """Surfaces the model's extra loss terms (cos_sim etc.) alongside
        Trainer's own loss logging."""
        def on_log(self, args, state, control, logs=None, **kw):
            extra = getattr(model, "_last_metrics", None)
            if logs is not None and extra:
                logs.update(extra)

    targs = TrainingArguments(
        output_dir=out_dir,
        max_steps=steps,
        per_device_train_batch_size=micro,
        gradient_accumulation_steps=accum,
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type="linear",                  # A.1: linear decay
        warmup_steps=int(cfg.get("warmup_steps", 0)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        optim="adamw_torch",                         # A.2
        # Precision follows the ARCHITECTURE, not the VLM recipe: the cosine
        # objective diverged under bf16 autocast at 5e-4 (grad_norm ~1e10 by
        # step ~7k), while this predictor + lr has only ever been stable in
        # fp32 (July arms, upstream DinoWM). The ce objective needs autocast
        # for the judge path and trains stably under it.
        bf16=bool(cfg.get("bf16", True)),
        gradient_checkpointing=False,                # judge checkpoints itself
        dataloader_num_workers=int(cfg["workers"]),
        logging_steps=int(cfg["logging_steps"]),
        save_steps=int(cfg["save_steps"]),
        save_total_limit=int(cfg.get("save_total_limit", 16)),
        report_to=["wandb"] if cfg.get("wandb_project") else [],
        run_name=cfg.get("run_name"),
        remove_unused_columns=False,
        seed=int(cfg["seed"]),
    )

    resume = cfg.get("resume")
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      data_collator=predictor_collate,
                      callbacks=[EvalCallback(), TrainMetricsCallback()])
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(os.path.join(out_dir, "final"))
    print("TRAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
