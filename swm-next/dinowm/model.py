"""Trainable wrapper: DinoWM predictor + frozen encoder/judge, Trainer-ready.

Only the OneShotPredictor is a registered submodule, so state_dict() (and
therefore every Trainer checkpoint) holds just the ~64M-param predictor —
not the frozen Qwen ViT encoder or the frozen 8B judge. The frozen parts are
attached through a plain tuple, which nn.Module.__setattr__ does not
register; they must be moved to the device by the caller (Trainer's
model.to() will not see them).

Objectives:
  - "cosine": DinoWM-native latent regression, loss = 1 - mean cosine sim.
  - "ce":     the SWM recipe objective (paper 3.2) routed through the frozen
              judge — full-vocab cross-entropy on the yes/no answer token
              about the PREDICTED future latent. cos_weight (default 0)
              optionally mixes the cosine term back in.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .judge import StudentVQAHead
from .oneshot_world_model import OneShotWorldModel


class PredictorForSAQA(nn.Module):
    def __init__(self, wm: OneShotWorldModel, judge: StudentVQAHead | None,
                 objective: str = "cosine", cos_weight: float = 0.0):
        super().__init__()
        assert objective in ("cosine", "ce"), objective
        if objective == "ce":
            assert judge is not None, "ce objective requires the judge"
        self.predictor = wm.predictor          # the ONLY registered submodule
        self._frozen = (wm, judge)             # tuple: hidden from state_dict/.to()
        self.objective = objective
        self.cos_weight = float(cos_weight)

    @property
    def wm(self) -> OneShotWorldModel:
        return self._frozen[0]

    @property
    def judge(self) -> StudentVQAHead | None:
        return self._frozen[1]

    def forward(self, history, actions, action_mask, target,
                label_yes=None, question=None, qtype=None):
        out = self.wm(history, actions, action_mask, target)

        if self.objective == "cosine":
            loss = out["loss"]
            ce_mean = torch.zeros((), device=loss.device)
        else:
            # One batched multi-prompt judge forward for the whole micro-batch
            # (numerically equivalent to grouping by question; ~5x faster on
            # mostly-distinct questions, which is what balanced SAQA draws give).
            img_embeds = self.judge.merge(out["z_pred"])
            prompt_infos = [self.judge._prompt(q) for q in question]
            ce_sum = self.judge.answer_ce_multi(prompt_infos, img_embeds, label_yes)
            ce_mean = ce_sum / max(1, len(question))
            loss = ce_mean + self.cos_weight * out["loss"]

        # Scalars for the trainer's logging callback (Trainer only logs
        # outputs["loss"] on its own).
        self._last_metrics = {
            "train/cos_sim": float(out["cos_sim"]),
            "train/cos_sim_identity": float(out["cos_sim_identity"]),
            "train/ce_loss": float(ce_mean),
            "train/cos_loss": float(out["loss"]),
        }
        return {
            "loss": loss,
            "ce_loss": ce_mean.detach(),
            "cos_loss": out["loss"].detach(),
            "cos_sim": out["cos_sim"],
            "cos_sim_identity": out["cos_sim_identity"],
            "z_pred": out["z_pred"],
        }
