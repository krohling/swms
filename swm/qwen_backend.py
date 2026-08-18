"""Qwen3-VL + LoRA SWM: model construction and prompt/action encoding.

Shared by training (swm-next/train_qwen.py) and planning evaluation
(swm/qwen_wm.py) so both sides see byte-identical prompt construction and
action packing. Mirrors how the PaliGemma WM classes live in this package.
"""
from __future__ import annotations

import torch

from swm.constants import ANSWER_OPTIONS


class QwenWithActions(torch.nn.Module):
    """Qwen3-VL plus a trained action projection.

    `forward` accepts an extra `action_values` tensor, projects it, and swaps it
    into the placeholder-token positions via a hook on the embedding layer. The
    projection lives here rather than in the collator so gradients reach it --
    collation runs in DataLoader worker processes, outside autograd.
    """

    def __init__(self, inner, action_dim: int, max_horizon: int, placeholder_id: int):
        super().__init__()
        self.inner = inner
        self.placeholder_id = placeholder_id
        d_llm = self._llm_width(inner)
        self.action_proj = torch.nn.Sequential(
            torch.nn.Linear(action_dim, d_llm), torch.nn.LayerNorm(d_llm))
        self.action_pos = torch.nn.Parameter(torch.zeros(max_horizon, d_llm))
        torch.nn.init.trunc_normal_(self.action_pos, std=0.02)

        self._pending: torch.Tensor | None = None
        self._pending_ids: torch.Tensor | None = None
        base = inner.get_base_model() if hasattr(inner, "get_base_model") else inner
        base.get_input_embeddings().register_forward_hook(self._inject)

    @staticmethod
    def _llm_width(inner) -> int:
        base = inner.get_base_model() if hasattr(inner, "get_base_model") else inner
        return base.config.text_config.hidden_size

    def _inject(self, module, inputs, output):
        if self._pending is None:
            return output
        mask = self._pending_ids == self.placeholder_id
        output = output.clone()
        acts = self._pending.to(output.dtype)
        for row in range(mask.shape[0]):
            pos = mask[row].nonzero(as_tuple=False).squeeze(-1)
            if pos.numel():
                output[row, pos] = acts[row, : pos.numel()]
        return output

    def forward(self, action_values=None, **kw):
        if action_values is not None:
            proj = self.action_proj(action_values.to(self.action_pos.dtype))
            self._pending = proj + self.action_pos[: proj.shape[1]].unsqueeze(0)
            self._pending_ids = kw.get("input_ids")
        try:
            return self.inner(**kw)
        finally:
            self._pending = self._pending_ids = None

    def gradient_checkpointing_enable(self, **kw):
        self.inner.gradient_checkpointing_enable(**kw)


def single_token(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    assert len(ids) == 1, f"{text!r} is not a single token: {ids}"
    return ids[0]


def _reserved_token(tokenizer) -> tuple[int, str]:
    for cand in ("<|fim_pad|>", "<|fim_prefix|>", "<|box_start|>"):
        ids = tokenizer(cand, add_special_tokens=False).input_ids
        if len(ids) == 1:
            return ids[0], cand
    raise RuntimeError("no single-token placeholder available")


def build_model(cfg: dict):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(cfg["base_model"])
    inner = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["base_model"], dtype=torch.bfloat16, low_cpu_mem_usage=True)
    inner.config.use_cache = False

    lora = LoraConfig(
        r=int(cfg["lora_r"]), lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]), bias="none",
        # LLM decoder layers only; the vision tower stays frozen.
        target_modules=r"model\.language_model\.layers\.\d+\."
                       r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)")
    inner = get_peft_model(inner, lora)

    placeholder_id, placeholder = _reserved_token(processor.tokenizer)
    model = QwenWithActions(inner, int(cfg["action_dim"]),
                            int(cfg["max_horizon"]), placeholder_id)
    return model, processor, placeholder


def _prompt(processor, placeholder: str, question: str, n_actions: int) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text",
         "text": f"Actions: {placeholder * n_actions}\n{question}"}]}]
    return processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)


def encode(processor, placeholder: str, action_dim: int,
           batch: list[dict], with_answer: bool) -> dict:
    texts = [_prompt(processor, placeholder, b["question"], len(b["actions"]))
             for b in batch]
    if with_answer:
        texts = [t + b["answer"] for t, b in zip(texts, batch)]
    inputs = dict(processor(text=texts, images=[[b["image"]] for b in batch],
                            return_tensors="pt", padding=True))
    hmax = max(1, max(len(b["actions"]) for b in batch))
    acts = torch.zeros(len(batch), hmax, action_dim)
    for i, b in enumerate(batch):
        n = len(b["actions"])
        if n:
            acts[i, :n] = b["actions"].float()
    inputs["action_values"] = acts              # projected inside forward
    return inputs


def make_collator(processor, placeholder: str, action_dim: int):
    """Supervise exactly the answer token, matching PaliGemma's `suffix=`
    masking in train.py (paper 3.2)."""

    def collate(batch: list[dict]) -> dict:
        inputs = encode(processor, placeholder, action_dim, batch, with_answer=True)
        ids, am = inputs["input_ids"], inputs["attention_mask"]
        labels = torch.full_like(ids, -100)
        last = am.sum(dim=1) - 1
        rows = torch.arange(len(batch))
        labels[rows, last] = ids[rows, last]
        inputs["labels"] = labels
        return inputs

    return collate
