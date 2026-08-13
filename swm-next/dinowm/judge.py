"""Frozen Qwen3-VL judge for scoring predicted latents, with gradients.

Ported from dino_wm's train_oneshot_distill.StudentVQAHead (the July arms'
judge-in-the-loop head, including the do_resize fix and the full-vocab
answer_ce objective). _PromptInfo is inlined from dino_wm's
planning/qwen_wm_model.py so this file has no dino_wm dependency.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class _PromptInfo:
    """Cached per-question prompt tokens to avoid re-tokenizing in the inner loop."""
    text: str
    input_ids: torch.Tensor          # (L,)
    attention_mask: torch.Tensor     # (L,)
    image_token_positions: torch.Tensor  # (n_image_tokens,) long
    desired_token_ids: list          # all single-token variants of the desired answer (e.g. yes / Yes / " yes" / " Yes")
    other_token_ids: list            # all single-token variants of the other answer
    weight: float
    mm_token_type_ids: torch.Tensor | None = None  # (L,) -- required by transformers>=5 for M-RoPE


class StudentVQAHead:
    """Wraps the frozen Qwen3-VL merger+LLM for scoring predicted latents
    with gradients enabled (for training through the judge)."""

    def __init__(self, model_id: str, device, precision=torch.bfloat16, image_size=448):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        self.device = device
        self.precision = precision
        self.image_size = image_size
        self.full_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=precision, low_cpu_mem_usage=True
        ).to(device).eval()
        for p in self.full_model.parameters():
            p.requires_grad = False
        # Gradient checkpointing keeps activation memory manageable when we
        # backprop through the (frozen) LLM into the predictor.
        self.full_model.gradient_checkpointing_enable()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        _m = self.full_model
        self.visual = _m.model.visual if hasattr(_m, "model") and hasattr(_m.model, "visual") else _m.visual
        self.merger = self.visual.merger
        self.image_token_id = self.full_model.config.image_token_id

        self._prompt_cache: dict[str, object] = {}

    def _prompt(self, question: str) -> _PromptInfo:
        if question in self._prompt_cache:
            return self._prompt_cache[question]
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{question} Answer with one word: yes or no."},
            ],
        }]
        dummy_img = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))
        chat_str = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # do_resize=False: the processor smart-resizes small images (224 -> 256,
        # grid 16^2 = 64 tokens) which mismatches the encoder-side merged latent
        # count (224 -> grid 14^2 = 49). Disabling keeps prompt tokens == merged
        # features at any image_size that is a multiple of 32 (224 and 448 both).
        try:
            inputs = self.processor(text=[chat_str], images=[dummy_img], return_tensors="pt",
                                    padding=False, do_resize=False, return_mm_token_type_ids=True)
        except TypeError:
            inputs = self.processor(text=[chat_str], images=[dummy_img], return_tensors="pt",
                                    padding=False, do_resize=False)
        input_ids = inputs["input_ids"][0]
        mm_tti = inputs.get("mm_token_type_ids")

        def singles(cands):
            out = []
            for c in cands:
                ids = self.tokenizer(c, add_special_tokens=False).input_ids
                if len(ids) == 1:
                    out.append(ids[0])
            return out

        info = _PromptInfo(
            text=question,
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"][0],
            image_token_positions=(input_ids == self.image_token_id).nonzero(as_tuple=False).squeeze(-1),
            desired_token_ids=singles([" Yes", " yes", "Yes", "yes"]),
            other_token_ids=singles([" No", " no", "No", "no"]),
            weight=1.0,
            mm_token_type_ids=mm_tti[0] if mm_tti is not None else None,
        )
        self._prompt_cache[question] = info
        return info

    def merge(self, z_pred: torch.Tensor) -> torch.Tensor:
        """(B, P_in, D_in) fp32 -> (B, P_out, D_out) via frozen merger (bf16)."""
        B, P_in, D_in = z_pred.shape
        flat = z_pred.to(self.precision).reshape(B * P_in, D_in)
        out = self.merger(flat)
        P_out = out.shape[0] // B
        return out.reshape(B, P_out, out.shape[-1])

    def _llm_answer_logits(self, prompt_info: _PromptInfo, img_embeds: torch.Tensor) -> torch.Tensor:
        """Batched LLM forward (with grad) -> (B, vocab) logits at the answer
        position. Predicted latents are injected by patching get_image_features;
        no deepstack for predicted latents (predictor has no deepstack
        equivalent), matching what the planner does when deepstack is absent."""
        B = img_embeds.shape[0]
        device = self.device
        input_ids = prompt_info.input_ids.unsqueeze(0).expand(B, -1).contiguous().to(device)
        attn = prompt_info.attention_mask.unsqueeze(0).expand(B, -1).contiguous().to(device)
        per_image = [img_embeds[b] for b in range(B)]
        ps = int(self.full_model.config.vision_config.patch_size)
        grid = self.image_size // ps
        image_grid_thw = torch.tensor([[1, grid, grid]], dtype=torch.long, device=device).expand(B, 3).contiguous()
        dummy_pixels = torch.zeros(1, dtype=self.precision, device=device)
        extra = {}
        if prompt_info.mm_token_type_ids is not None:
            extra["mm_token_type_ids"] = prompt_info.mm_token_type_ids.unsqueeze(0).expand(B, -1).contiguous().to(device)

        def patched(*args, **kwargs):
            if kwargs.get("return_dict", False):
                import types
                return types.SimpleNamespace(pooler_output=tuple(per_image), deepstack_features=None)
            return tuple(per_image), None

        orig = self.full_model.model.get_image_features
        self.full_model.model.get_image_features = patched
        try:
            outputs = self.full_model.model(
                input_ids=input_ids, attention_mask=attn,
                pixel_values=dummy_pixels, image_grid_thw=image_grid_thw, **extra,
            )
            hidden = outputs.last_hidden_state
            logits = self.full_model.lm_head(hidden[:, -1, :]).float()
        finally:
            self.full_model.model.get_image_features = orig
        return logits

    def _llm_answer_logits_multi(self, prompt_infos: list, img_embeds: torch.Tensor) -> torch.Tensor:
        """One LLM forward for B samples with per-sample (possibly different)
        prompts. Right-pads input_ids to the batch max length and reads the
        answer logits at each sample's last REAL token. Numerically equivalent
        to grouping by question and calling _llm_answer_logits per group, but
        ~5x faster on batches of mostly-distinct questions (one big forward
        instead of many size-1 forwards through the 8B judge).

        Every prompt embeds the same number of image tokens (fixed grid), so
        the patched get_image_features splices per-sample latents correctly."""
        B = img_embeds.shape[0]
        assert len(prompt_infos) == B
        device = self.device
        lens = [int(pi.input_ids.shape[0]) for pi in prompt_infos]
        L = max(lens)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        input_ids = torch.full((B, L), pad_id, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=prompt_infos[0].attention_mask.dtype)
        mm = None
        if prompt_infos[0].mm_token_type_ids is not None:
            mm = torch.zeros((B, L), dtype=prompt_infos[0].mm_token_type_ids.dtype)
        for b, pi in enumerate(prompt_infos):
            input_ids[b, : lens[b]] = pi.input_ids
            attn[b, : lens[b]] = pi.attention_mask
            if mm is not None:
                mm[b, : lens[b]] = pi.mm_token_type_ids
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        per_image = [img_embeds[b] for b in range(B)]
        ps = int(self.full_model.config.vision_config.patch_size)
        grid = self.image_size // ps
        image_grid_thw = torch.tensor([[1, grid, grid]], dtype=torch.long, device=device).expand(B, 3).contiguous()
        dummy_pixels = torch.zeros(1, dtype=self.precision, device=device)
        extra = {}
        if mm is not None:
            extra["mm_token_type_ids"] = mm.to(device)

        def patched(*args, **kwargs):
            if kwargs.get("return_dict", False):
                import types
                return types.SimpleNamespace(pooler_output=tuple(per_image), deepstack_features=None)
            return tuple(per_image), None

        orig = self.full_model.model.get_image_features
        self.full_model.model.get_image_features = patched
        try:
            outputs = self.full_model.model(
                input_ids=input_ids, attention_mask=attn,
                pixel_values=dummy_pixels, image_grid_thw=image_grid_thw, **extra,
            )
            hidden = outputs.last_hidden_state
            last = torch.tensor(lens, device=device) - 1
            logits = self.full_model.lm_head(
                hidden[torch.arange(B, device=device), last, :]).float()
        finally:
            self.full_model.model.get_image_features = orig
        return logits

    def answer_ce_multi(self, prompt_infos: list, img_embeds: torch.Tensor,
                        label_yes: torch.Tensor) -> torch.Tensor:
        """Batched multi-prompt version of answer_ce (sum-reduced, same math).
        yes/no token ids are prompt-independent (same answer wording), taken
        from the first prompt."""
        B = img_embeds.shape[0]
        logits = self._llm_answer_logits_multi(prompt_infos, img_embeds)
        yes_id = prompt_infos[0].desired_token_ids[0]
        no_id = prompt_infos[0].other_token_ids[0]
        targets = torch.where(label_yes >= 0.5,
                              torch.full((B,), yes_id, device=self.device),
                              torch.full((B,), no_id, device=self.device))
        return F.cross_entropy(logits, targets, reduction="sum")

    def answer_bce_multi(self, prompt_infos: list, img_embeds: torch.Tensor,
                         label_yes: torch.Tensor) -> torch.Tensor:
        """Batched multi-prompt BCE on the renormalized pair probability
        (sum-reduced, mirroring answer_ce_multi). The logit is
        logsumexp(yes variants) - logsumexp(no variants), so sigmoid(logit)
        equals p_yes() exactly; unlike full-vocab CE, no gradient reaches the
        other vocab entries."""
        logits = self._llm_answer_logits_multi(prompt_infos, img_embeds)
        yes = torch.logsumexp(logits[:, prompt_infos[0].desired_token_ids], dim=-1)
        no = torch.logsumexp(logits[:, prompt_infos[0].other_token_ids], dim=-1)
        return F.binary_cross_entropy_with_logits(
            yes - no, (label_yes >= 0.5).float(), reduction="sum")

    def p_yes(self, prompt_info: _PromptInfo, img_embeds: torch.Tensor) -> torch.Tensor:
        """(B,) P(yes) over the yes/no token variants."""
        logits = self._llm_answer_logits(prompt_info, img_embeds)
        probs = torch.softmax(logits, dim=-1)
        yes = probs[:, prompt_info.desired_token_ids].sum(dim=-1)
        no = probs[:, prompt_info.other_token_ids].sum(dim=-1)
        return yes / (yes + no).clamp_min(1e-12)

    def answer_ce(self, prompt_info: _PromptInfo, img_embeds: torch.Tensor,
                  label_yes: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary cross-entropy on the answer token (SWM's objective,
        paper 3.2) against hard oracle labels, summed over the batch. The
        target is the primary yes/no token id and the softmax runs over the
        whole vocab rather than the {yes, no} pair."""
        B = img_embeds.shape[0]
        logits = self._llm_answer_logits(prompt_info, img_embeds)
        yes_id = prompt_info.desired_token_ids[0]
        no_id = prompt_info.other_token_ids[0]
        targets = torch.where(label_yes >= 0.5,
                              torch.full((B,), yes_id, device=self.device),
                              torch.full((B,), no_id, device=self.device))
        return F.cross_entropy(logits, targets, reduction="sum")
