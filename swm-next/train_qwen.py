"""Train a Qwen3-VL + LoRA Semantic World Model on the SAQA dataset.

A VARIANT of train.py, not a reproduction: data, sampler, objective
(cross-entropy on the answer tokens) and evaluation are identical; only the
architecture and tuning method differ. This file deliberately mirrors train.py
section for section so `diff train.py train_qwen.py` reads as the model swap
and nothing else.

The model, prompt construction and action packing live in swm.qwen_backend
(shared with the planning adapter swm/qwen_wm.py), so run this from the repo
root with the repo on PYTHONPATH:

    python swm-next/train_qwen.py --config swm-next/configs/qwen_oracle.yaml [key=value ...]

What differs from train.py, exhaustively:

  - swm.qwen_backend.build_model(): Qwen3-VL-8B + LoRA adapters (LLM decoder
    only) + a trained action projection, instead of the authors' PaliGemmaWM
    full fine-tune. Actions enter by reserving one placeholder token per action
    step and swapping projected actions into those embedding positions, leaving
    Qwen's image splicing, DeepStack features and M-RoPE positions stock.
  - swm.qwen_backend.make_collator(): builds Qwen chat-template prompts and
    supervises exactly the answer token, matching PaliGemma's `suffix=` masking.
  - evaluate(): same balanced probe; yes/no read from the last position's
    logits (Qwen right-pads, so the last *unpadded* position is used).
"""
from __future__ import annotations

import argparse
import collections
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from saqa import SAQADataset, SAQAIndex

from swm.constants import ANSWER_OPTIONS
from swm.qwen_backend import build_model, encode, make_collator, single_token


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


@torch.no_grad()
def evaluate(model, processor, placeholder: str, cfg: dict,
             index: SAQAIndex, n: int, batch_size: int, seed: int = 1234):
    """Future-QA accuracy on held-out trajectories, balanced over the strata.

    Identical probe to train.py's evaluate; only the readout differs (Qwen
    right-pads, so the last unpadded position carries the next-token logits).
    """
    tok = processor.tokenizer
    yes_id, no_id = (single_token(tok, a) for a in ANSWER_OPTIONS)
    ds = SAQADataset(index, length=n, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4,
                        collate_fn=lambda b: b)  # keep dicts; we batch by hand

    model.eval()
    per = collections.defaultdict(lambda: [0, 0])
    for raw in loader:
        inputs = encode(processor, placeholder, int(cfg["action_dim"]),
                        raw, with_answer=False)
        dev = next(model.parameters()).device
        inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
        out = model(**inputs)
        last = inputs["attention_mask"].sum(dim=1) - 1
        logits = out.logits[torch.arange(len(raw)), last].float()
        pred_yes = (logits[:, yes_id] > logits[:, no_id]).cpu().numpy()
        for b, p in zip(raw, pred_yes):
            per[b["qtype"]][1] += 1
            per[b["qtype"]][0] += int(bool(p) == (b["answer"] == "yes"))
    model.train()

    metrics = {f"eval/{t}": ok / n_ for t, (ok, n_) in per.items()}
    metrics["eval/balanced_acc"] = float(np.mean(list(metrics.values())))
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
    # Teacher mode (label_teacher.py sidecars): training strata and answers
    # come from the teacher's argmax; the val index/probe stays ORACLE-scored
    # so eval/balanced_acc measures truth and stays comparable to the repro.
    teacher = ([cfg["noisy_teacher"], cfg["play_teacher"]]
               if cfg.get("noisy_teacher") else None)
    train_idx = SAQAIndex.build(files, split="train", val_frac=cfg["val_frac"],
                                teacher_files=teacher)
    val_idx = SAQAIndex.build(files, split="val", val_frac=cfg["val_frac"])

    steps = int(cfg["max_steps"])
    eff_batch = int(cfg["effective_batch"])
    micro = int(cfg["micro_batch"])
    assert eff_batch % micro == 0, "effective batch must be a multiple of micro batch"
    accum = eff_batch // micro

    print("=== G2: sampler balance (train split) ===", flush=True)
    train_idx.report(draws=steps * eff_batch)

    # NOTE: unlike train.py there is no from_pretrained resume path here —
    # QwenWithActions is a plain nn.Module, so Trainer's state_dict load is the
    # only mechanism. It has NOT been verified by an A/B gate the way train.py's
    # was; run scripts/verify_resume-style smoke before relying on it.
    resume = cfg.get("resume")
    model, processor, placeholder = build_model(cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {n_train:,} of {n_total:,} "
          f"(LoRA r={cfg['lora_r']} + action projection)", flush=True)

    train_ds = SAQADataset(train_idx, length=steps * eff_batch, seed=cfg["seed"],
                           teacher_files=teacher)

    class EvalCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if state.global_step and state.global_step % int(cfg["eval_every"]) == 0:
                m = evaluate(model, processor, placeholder, cfg,
                             val_idx, int(cfg["eval_samples"]), micro)
                m["step"] = state.global_step
                print(f"eval @ {state.global_step}: {m}", flush=True)
                if "wandb" in (args.report_to or []):
                    import wandb
                    wandb.log(m, step=state.global_step)

    targs = TrainingArguments(
        output_dir=out_dir,
        max_steps=steps,
        per_device_train_batch_size=micro,
        gradient_accumulation_steps=accum,
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type="linear",                  # matches train.py (A.1)
        # Trainer defaults made explicit, mirroring train.py.
        warmup_steps=int(cfg.get("warmup_steps", 0)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        optim="adamw_torch",                         # matches train.py (A.2)
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=int(cfg["workers"]),
        logging_steps=int(cfg["logging_steps"]),
        save_steps=int(cfg["save_steps"]),
        save_total_limit=int(cfg.get("save_total_limit", 3)),
        report_to=["wandb"] if cfg.get("wandb_project") else [],
        run_name=cfg.get("run_name"),
        remove_unused_columns=False,
        seed=int(cfg["seed"]),
    )

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      data_collator=make_collator(processor, placeholder,
                                                  int(cfg["action_dim"])),
                      callbacks=[EvalCallback()])
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(os.path.join(out_dir, "final"))
    print("TRAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
