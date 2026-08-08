"""Predictor-side view of the SAQA dataset.

Sampling is inherited VERBATIM from saqa.SAQADataset: same balanced
12-stratum draw, same (seed, i) rng stream, same index rows — this class
only reads MORE fields out of each drawn row (the target future frame and a
second history frame), so the predictor trains on exactly the sample stream
the PaliGemma reproduction trained on. It must never override _draw.

Per-row semantics (matching the reproduction):
  - current frame  = frames[i0]; history = frames[i0-1], frames[i0]
    (clamped at the trajectory start: row 0 repeats frame 0)
  - actions        = actions[i0:i1], RAW (unnormalized), zero-padded to
    max_action_horizon with a validity mask — the recipe feeds raw actions
  - target frame   = frames[i1]
  - h = i1 - i0 may be 0 (the recipe samples horizon 0); the predictor then
    sees an all-False action mask and target == current frame
"""
from __future__ import annotations

import numpy as np
import torch

from saqa import SAQADataset, SAQAIndex, _s  # noqa: F401  (SAQAIndex re-exported)


class PredictorSAQADataset(SAQADataset):
    def __init__(self, index, length: int, seed: int = 0,
                 obs_horizon: int = 2, max_action_horizon: int = 20):
        super().__init__(index, length, seed)
        self.obs_horizon = int(obs_horizon)
        self.max_action_horizon = int(max_action_horizon)

    def __getitem__(self, i: int) -> dict:
        g, d, i0, i1, qi, stratum = self._draw(i)
        frames = g["frames"]

        hist_idx = [max(t, 0) for t in range(i0 - self.obs_horizon + 1, i0 + 1)]
        history = np.stack([np.asarray(frames[t], dtype=np.uint8) for t in hist_idx])

        acts = np.asarray(g["actions"][i0:i1], dtype=np.float32)  # (h, A), h may be 0
        h = i1 - i0
        assert acts.shape[0] == h, (acts.shape, i0, i1)
        assert 0 <= h <= self.max_action_horizon, f"horizon {h} outside [0, {self.max_action_horizon}]"
        actions = np.zeros((self.max_action_horizon, acts.shape[1]), dtype=np.float32)
        actions[:h] = acts
        mask = np.zeros(self.max_action_horizon, dtype=bool)
        mask[:h] = True

        return {
            "history": torch.from_numpy(history).permute(0, 3, 1, 2).float() / 255.0,
            "actions": torch.from_numpy(actions),
            "action_mask": torch.from_numpy(mask),
            "target": torch.from_numpy(
                np.asarray(frames[i1], dtype=np.uint8)).permute(2, 0, 1).float() / 255.0,
            "question": _s(d["questions"][()][qi]),
            "answer_yes": bool(d["answers"][()][qi]),
            "qtype": stratum[0],
        }


def predictor_collate(batch: list[dict]) -> dict:
    """Tensors stacked; questions/qtypes stay as lists (Trainer passes
    non-tensor values through untouched with remove_unused_columns=False)."""
    return {
        "history": torch.stack([b["history"] for b in batch]),
        "actions": torch.stack([b["actions"] for b in batch]),
        "action_mask": torch.stack([b["action_mask"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "label_yes": torch.tensor([float(b["answer_yes"]) for b in batch]),
        "question": [b["question"] for b in batch],
        "qtype": [b["qtype"] for b in batch],
    }
