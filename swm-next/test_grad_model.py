"""Adapter validation: rehydrate a Trainer checkpoint into DinoWMGradModel
and exercise the two harness entry points, including gradient flow to the
action tensors (what the gradient planner depends on).

    CUDA_VISIBLE_DEVICES=<gpu> python test_grad_model.py \
        --ckpt <.../checkpoint-4000> --config configs/predictor_cosine.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)                       # dinowm package
sys.path.insert(0, os.path.dirname(_here))      # swm package (repo root)

from dinowm.grad_model import DinoWMGradModel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--config", required=True)
args = ap.parse_args()

device = torch.device("cuda")
m = DinoWMGradModel(checkpoint_path=args.ckpt, config_path=args.config, device=device)
print("model loaded", flush=True)

img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
questions = [
    "Is the blue cube on top of the green cube?",
    "Is the robot gripper touching the blue cube?",
    "Is the blue cube grasped by the robot?",
    "Is the yellow cube closer to the red cube than to the blue cube?",
]

# --- get_scores: shapes + gradient flow through padded variable-length actions
actions = [torch.randn(h, 5, device=device, requires_grad=True) for h in (8, 16, 8, 16)]
with torch.enable_grad():
    p_yes, p_no = m.get_scores([img] * 4, actions, questions)
    assert p_yes.shape == p_no.shape == (4,), (p_yes.shape, p_no.shape)
    assert float(p_yes.min()) >= 0 and float((p_yes + p_no).max()) <= 1.001
    p_yes.sum().backward()
g0 = actions[0].grad
assert g0 is not None and torch.isfinite(g0).all() and float(g0.abs().sum()) > 0, "no gradient to actions"
print(f"get_scores OK  p_yes={p_yes.detach().cpu().numpy().round(4)}  "
      f"|dL/da|={float(g0.abs().mean()):.2e}", flush=True)

# --- get_probabilistic_rewards_wm: the harness reward loop end to end
action_seq = torch.randn(2, 16, 5, device=device, requires_grad=True)
qs = [(questions[0], "yes", 1.0), (questions[1], "no", 1.0)]
rewards, weighted, grad_sum = m.get_probabilistic_rewards_wm(
    action_seq=action_seq, image=img, pred_horizon=16, questions=qs,
    batch_size=8, action_skip=8, gradient=True)
assert rewards.shape == (2, 2, 16), rewards.shape
grad_sum.backward()
assert action_seq.grad is not None and torch.isfinite(action_seq.grad).all()
print(f"rewards loop OK  rewards[:, :, 7]={rewards[:, :, 7].round(4).tolist()}  "
      f"|dR/da|={float(action_seq.grad.abs().mean()):.2e}", flush=True)
print("GRAD_MODEL_OK", flush=True)
