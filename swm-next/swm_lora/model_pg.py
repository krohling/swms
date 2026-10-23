"""PaliGemma-3B SWM for score regression — the full-weight arm.

Same recipe surface as QwenSWM (answer_logit / loss / save_adapters /
trainable_parameters contract, CycleDataset planner lists, soft-target BCE on
the two-class yes/no logit) with the backbone swapped for the authors'
PaliGemmaWM model and the capacity ceiling removed: full_finetune=True
unfreezes every weight, trained in the repro's proven A.1 regime (fp32
params, bf16 autocast, AdamW, linear LR decay from 1e-5, clip 1.0, effective
batch 96 sequences = 12 lists/step).

Learnings carried over from the Qwen arms (deep eval 2026-10-23):
  - TrajectoryProjector (MLP-512, cumsum channels) replaces the WM's single
    linear action projector; fresh init, trained in a projector-only warm-up
    first (uniform weighting, run to step 6000, then the final checkpoint).
  - NO scalar gain: tokens enter at fixed 1x (learnable_scale=False).
  - v2 R10-filtered dataset.

Trajectory injection is the WM's native mechanism: the processor plants
horizon-many <action> placeholders in the prefix and the model scatters
action_projector(action_values) into those positions — the pattern QwenSWM
reimplements with an embedding hook, used here directly. The yes/no readout
follows swm-next/train.py's evaluate(): single token ids from
tokenizer.encode(ANSWER_OPTIONS)[-1], logit l = z_yes - z_no at the last
prefix position.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import TrajectoryProjector


class PaliGemmaSWM(nn.Module):
    def __init__(self, base_model="google/paligemma-3b-pt-224",
                 processor_path="ckpts/paligemma_wm_ogbench",
                 action_dim=5, horizon=16, device="cuda",
                 proj_hidden=512, traj_cumsum=True, full_finetune=False):
        super().__init__()
        from swm.paligemma_wm.configuration_paligemma_wm import PaliGemmaWMConfig
        from swm.paligemma_wm.modeling_paligemma_wm import (
            PaliGemmaWMForConditionalGeneration)
        from swm.paligemma_wm.processing_paligemma_wm import PaliGemmaWMProcessor

        self.device = device
        self.horizon = horizon
        self.full_finetune = full_finetune
        self.has_lora = False
        self.processor = PaliGemmaWMProcessor.from_pretrained(processor_path)
        config = PaliGemmaWMConfig.from_pretrained(processor_path)
        # Base PaliGemma weights; the action projector initialises fresh
        # (reported in missing_keys) — exactly build_model() in swm-next/train.py.
        self.model = PaliGemmaWMForConditionalGeneration.from_pretrained(
            base_model, config=config, torch_dtype=torch.float32,
            low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
        self.model.config.use_cache = False
        d_model = self.model.config.vision_config.projection_dim
        self.projector = TrajectoryProjector(
            action_dim, d_model, hidden=proj_hidden, init_scale=0.0,
            cumsum=traj_cumsum, learnable_scale=False).to(torch.float32)
        # Swap the WM's single-linear action projector for ours; the model's
        # placeholder scatter then consumes our MLP's output unchanged.
        self.model.model.action_projector = self.projector
        self.model.to(device)
        if not full_finetune:
            for p in self.model.parameters():
                p.requires_grad_(False)
            for p in self.projector.parameters():
                p.requires_grad_(True)
        tok = self.processor.tokenizer
        from swm.constants import ANSWER_OPTIONS
        self.yes_id = tok.encode(ANSWER_OPTIONS[0])[-1]
        self.no_id = tok.encode(ANSWER_OPTIONS[1])[-1]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def answer_logit(self, images, questions, trajs):
        """l = z_yes - z_no at the final prefix position. trajs: (B,H,A)
        normalized; pass zeros for the empty-trajectory/drift probe."""
        trajs = trajs.to(torch.float32)
        inputs = self.processor(text=list(questions), images=list(images),
                                actions=[t for t in trajs],
                                return_tensors="pt", padding="longest")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
        inputs["action_values"] = inputs["action_values"].to(torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(self.device == "cuda")):
            out = self.model(**inputs)
        mask = inputs["attention_mask"]
        # index of the last attended position per row — correct under either
        # padding side, unlike mask.sum()-1 (right-pad only) or -1 (left-pad only)
        pos = torch.arange(mask.shape[1], device=mask.device)
        last = (mask * pos).argmax(dim=1)
        idx = torch.arange(out.logits.shape[0], device=out.logits.device)
        z = out.logits[idx, last, :].float()
        return z[:, self.yes_id] - z[:, self.no_id]

    @staticmethod
    def loss(logit, target_p):
        return F.binary_cross_entropy_with_logits(logit, target_p)

    @torch.no_grad()
    def p_yes(self, images, questions, trajs):
        return torch.sigmoid(self.answer_logit(images, questions, trajs))

    def save_adapters(self, path):
        """Projector only — the warm-up checkpoint contract."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.projector.state_dict(), f"{path}/projector.pt")

    def load_adapters(self, path):
        self.projector.load_state_dict(
            torch.load(f"{path}/projector.pt", map_location=self.device))

    def save_full(self, path):
        """Full model (incl. projector) + processor, reloadable standalone."""
        self.model.save_pretrained(path, safe_serialization=True)
        self.processor.save_pretrained(path)
        torch.save(self.projector.state_dict(), f"{path}/projector.pt")
