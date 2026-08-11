"""Train the Semantic World Model on the SAQA dataset.

Model, processor and loss are the authors' own (`swm.paligemma_wm`); the answer
is supervised with cross-entropy via the processor's `suffix=` argument, which
masks the prefix (paper 3.2). Training uses `transformers.Trainer`, per the
author, with the hyperparameters from the paper rather than Trainer defaults.

Hyperparameters and their sources:

  A.1    full-weight fine-tuning of all parameters
  A.1    linear LR decay starting at 1e-5
  A.1    ~64,000 gradient steps for OGBench
  A.1    effective batch size 96
  A.2    AdamW
  3.2    cross-entropy on the answer tokens

    python train.py --config configs/repro.yaml [key=value ...]
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
from saqa import SAQADataset, SAQAIndex, make_collator  # noqa: E402


def build_model(processor_path: str, base_model: str):
    from swm.paligemma_wm.configuration_paligemma_wm import PaliGemmaWMConfig
    from swm.paligemma_wm.modeling_paligemma_wm import PaliGemmaWMForConditionalGeneration
    from swm.paligemma_wm.processing_paligemma_wm import PaliGemmaWMProcessor

    processor = PaliGemmaWMProcessor.from_pretrained(processor_path)
    config = PaliGemmaWMConfig.from_pretrained(processor_path)
    # Base PaliGemma weights; the action projection P is new and initialises
    # fresh (reported in missing_keys).
    model = PaliGemmaWMForConditionalGeneration.from_pretrained(
        base_model, config=config, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
    model.config.use_cache = False
    return model, processor


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
def evaluate(model, processor, index: SAQAIndex, n: int, batch_size: int, seed: int = 1234):
    """Future-QA accuracy on held-out trajectories, balanced over the strata.

    Reported per question type: accuracy saturates early while planning keeps
    improving, and the per-type split is what localises a regression.
    """
    from swm.constants import ANSWER_OPTIONS

    tok = processor.tokenizer
    yes_id, no_id = (tok.encode(a)[-1] for a in ANSWER_OPTIONS)
    ds = SAQADataset(index, length=n, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4,
                        collate_fn=lambda b: b)  # keep dicts; we batch by hand

    model.eval()
    per = collections.defaultdict(lambda: [0, 0])
    for raw in loader:
        inputs = processor(text=[b["question"] for b in raw],
                           images=[b["image"] for b in raw],
                           actions=[b["actions"] for b in raw],
                           return_tensors="pt", padding="longest")
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
        if "action_values" in inputs:
            inputs["action_values"] = inputs["action_values"].to(torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**inputs).logits[:, -1, :].float()
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

    # On resume, load the weights from the checkpoint rather than the base model.
    # The authors' model declares `_checkpoint_conversion_mapping`, so the names
    # on disk ("vision_tower.*") differ from the module paths ("model.vision_tower.*").
    # `from_pretrained` applies that mapping; Trainer's own resume path does not,
    # and matches none of the 606 keys -- silently leaving the model untrained.
    # Trainer still runs its own load afterwards, but every key it fails to match
    # is left at the value we loaded here, which is the correct one.
    resume = cfg.get("resume")
    model, processor = build_model(cfg["processor_path"], resume or cfg["base_model"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_params:,} (full fine-tune)", flush=True)

    train_ds = SAQADataset(train_idx, length=steps * eff_batch, seed=cfg["seed"],
                           teacher_files=teacher)

    class EvalCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if state.global_step and state.global_step % int(cfg["eval_every"]) == 0:
                m = evaluate(model, processor, val_idx, int(cfg["eval_samples"]), micro)
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
        learning_rate=float(cfg["learning_rate"]),   # A.1: 1e-5
        lr_scheduler_type="linear",                  # A.1: linear decay
        # Not specified by the paper -- Trainer defaults, made explicit in the
        # config so they are auditable. max_grad_norm in particular is active:
        # observed grad norms are 3.4-22.3, so nearly every step gets clipped.
        warmup_steps=int(cfg.get("warmup_steps", 0)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        optim="adamw_torch",                         # A.2
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
                      data_collator=make_collator(processor),
                      callbacks=[EvalCallback()])
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(os.path.join(out_dir, "final"))
    print("TRAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
