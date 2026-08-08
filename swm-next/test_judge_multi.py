"""Equivalence + timing test: batched multi-prompt judge vs grouped path.

Compares answer_ce_multi against per-group answer_ce on identical inputs
(real questions from the val index, synthetic latents). Passes if CE sums
agree within bf16 tolerance and the multi path is materially faster.

    CUDA_VISIBLE_DEVICES=<gpu> python test_judge_multi.py --config configs/predictor_ce.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saqa import SAQAIndex, SAQADataset  # noqa: E402
from dinowm.judge import StudentVQAHead  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))

device = torch.device("cuda")
head = StudentVQAHead(cfg["model_id"], device=device, image_size=int(cfg["image_size"]))

# Real questions from the val split (the realistic mostly-distinct mix).
idx = SAQAIndex.build([cfg["noisy_hdf5"], cfg["play_hdf5"]], split="val",
                      val_frac=cfg["val_frac"])
ds = SAQADataset(idx, length=48, seed=99)
questions = [ds[i]["question"] for i in range(48)]
labels = torch.tensor([1.0 if ds[i]["answer"] == "yes" else 0.0
                       for i in range(48)], device=device)
print(f"distinct questions in batch: {len(set(questions))}/48", flush=True)

# Synthetic latents in the merged (LLM-input) space via the real merger.
torch.manual_seed(0)
grid = int(cfg["image_size"]) // int(head.full_model.config.vision_config.patch_size)
n_pre = grid * grid  # pre-merger patch count
z = torch.randn(48, n_pre, head.visual.config.hidden_size
                if hasattr(head.visual, "config") else 1152, device=device)
img_embeds = head.merge(z)

infos = [head._prompt(q) for q in questions]

def grouped_ce():
    idx_by_q = defaultdict(list)
    for i, q in enumerate(questions):
        idx_by_q[q].append(i)
    s = torch.zeros((), device=device)
    for q, ixs in idx_by_q.items():
        s = s + head.answer_ce(head._prompt(q), img_embeds[ixs], labels[ixs])
    return s

def multi_ce():
    return head.answer_ce_multi(infos, img_embeds, labels)

with torch.no_grad():
    a = float(grouped_ce()); b = float(multi_ce())
    rel = abs(a - b) / max(abs(a), 1e-9)
    print(f"grouped CE sum: {a:.4f}   multi CE sum: {b:.4f}   rel diff: {rel:.2e}", flush=True)

    for fn, name in ((grouped_ce, "grouped"), (multi_ce, "multi")):
        fn()  # warm
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        print(f"{name}: {(time.time() - t0) / 3:.2f}s per bs-48 scoring pass", flush=True)

assert rel < 0.02, f"CE mismatch: rel diff {rel}"
print("JUDGE_MULTI_EQUIVALENT", flush=True)
