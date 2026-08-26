"""VLM backends for the SAQA labeler eval (run_saqa_eval.py).

Every backend exposes score(items) -> (p_yes, mass) over a batch of eval items
under the PRODUCTION labeling protocol: state questions see the END frame;
temporal 'closer' questions see (start, end). The pooled yes/no readout is
deliberately duplicated from swm-next/label_teacher.py (source of truth) so
this module has no cross-branch dependency.

Single-image models (PaliGemma family) cannot take a two-image closer prompt;
they get a side-by-side composite (start | end) with the pair preamble --
flagged in results as a protocol substitution.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

PAIR_PREAMBLE = ("The first image shows the scene at an earlier time; the "
                 "second image shows the scene now. Compared to the earlier time: ")
COMPOSITE_PREAMBLE = ("The left half shows the scene at an earlier time; the "
                      "right half shows the scene now. Compared to the earlier time: ")
CLOSER = ("block_block_closer", "gripper_block_closer")


def _composite(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    pad = np.full((a.shape[0], 4, 3), 255, dtype=np.uint8)
    return np.concatenate([a, pad, b], axis=1)


def _singles(tok, cands):
    return [tok(c, add_special_tokens=False).input_ids[0]
            for c in cands if len(tok(c, add_special_tokens=False).input_ids) == 1]


class ChatVLBackend:
    """Qwen3-VL-8B, Cosmos-Reason2-8B, Qwen2.5-VL — chat-template models with
    multi-image support, loaded via AutoModelForImageTextToText."""

    def __init__(self, model_id: str, device: str = "mps", dtype=torch.bfloat16):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = device
        tok = self.processor.tokenizer
        self.yes_ids = _singles(tok, [" Yes", " yes", "Yes", "yes"])
        self.no_ids = _singles(tok, [" No", " no", "No", "no"])

    @torch.no_grad()
    def score(self, items):
        texts, image_lists = [], []
        for it in items:
            pair = it["qtype"] in CLOSER
            q = it["question"]
            if pair:
                content = [{"type": "image"}, {"type": "image"},
                           {"type": "text", "text": f"{PAIR_PREAMBLE}{q} "
                            "Answer with one word: yes or no."}]
                image_lists.append([it["start"], it["end"]])
            else:
                content = [{"type": "image"},
                           {"type": "text", "text": f"{q} Answer with one word: yes or no."}]
                image_lists.append([it["end"]])
            texts.append(self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True, tokenize=False))
        inputs = self.processor(text=texts, images=image_lists,
                                return_tensors="pt", padding=True).to(self.device)
        out = self.model(**inputs)
        mask = inputs["attention_mask"]
        last = mask.sum(dim=1) - 1
        idx = torch.arange(out.logits.shape[0], device=out.logits.device)
        logits = out.logits[idx, last, :].float()
        probs = torch.softmax(logits, dim=-1)
        yes = probs[:, self.yes_ids].sum(dim=-1)
        no = probs[:, self.no_ids].sum(dim=-1)
        mass = yes + no
        return ((yes / mass.clamp_min(1e-12)).cpu().numpy(), mass.cpu().numpy())


class PaliGemmaBaseBackend:
    """google/paligemma-3b-pt-224 — single-image VQA prompt format."""

    def __init__(self, model_id: str = "google/paligemma-3b-pt-224",
                 device: str = "mps", dtype=torch.bfloat16):
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = device
        tok = self.processor.tokenizer
        self.yes_ids = _singles(tok, ["yes", " yes", "Yes", " Yes"])
        self.no_ids = _singles(tok, ["no", " no", "No", " No"])

    @torch.no_grad()
    def score(self, items):
        texts, images = [], []
        for it in items:
            if it["qtype"] in CLOSER:
                images.append(Image.fromarray(_composite(it["start"], it["end"])))
                texts.append(f"answer en {COMPOSITE_PREAMBLE}{it['question']}")
            else:
                images.append(Image.fromarray(it["end"]))
                texts.append(f"answer en {it['question']}")
        inputs = self.processor(text=texts, images=images,
                                return_tensors="pt", padding=True).to(self.device)
        out = self.model(**inputs)
        mask = inputs["attention_mask"]
        last = mask.sum(dim=1) - 1
        idx = torch.arange(out.logits.shape[0], device=out.logits.device)
        logits = out.logits[idx, last, :].float()
        probs = torch.softmax(logits, dim=-1)
        yes = probs[:, self.yes_ids].sum(dim=-1)
        no = probs[:, self.no_ids].sum(dim=-1)
        mass = yes + no
        return ((yes / mass.clamp_min(1e-12)).cpu().numpy(), mass.cpu().numpy())


class PaliGemmaWMBackend:
    """The published SWM fine-tune, scored through swm's own classes.
    Single-image; closer questions get the composite substitution."""

    def __init__(self, checkpoint_path: str, device: str = "mps",
                 dtype=torch.bfloat16):
        from swm.paligemma_wm import (PaliGemmaWMForConditionalGeneration,
                                      PaliGemmaWMProcessor)
        self.model = PaliGemmaWMForConditionalGeneration.from_pretrained(
            checkpoint_path, torch_dtype=dtype).to(device).eval()
        self.processor = PaliGemmaWMProcessor.from_pretrained(checkpoint_path)
        self.device = device
        tok = self.processor.tokenizer
        self.yes_ids = [tok.encode("yes")[0] if isinstance(tok.encode("yes"), list)
                        else tok.encode("yes")]
        self.no_ids = [tok.encode("no")[0] if isinstance(tok.encode("no"), list)
                       else tok.encode("no")]

    @torch.no_grad()
    def score(self, items):
        texts, images = [], []
        for it in items:
            if it["qtype"] in CLOSER:
                images.append(Image.fromarray(_composite(it["start"], it["end"])))
                texts.append(f"{COMPOSITE_PREAMBLE}{it['question']}")
            else:
                images.append(Image.fromarray(it["end"]))
                texts.append(it["question"])
        inputs = self.processor(text=texts, images=images, actions=None,
                                return_tensors="pt", padding="longest",
                                tokenize_newline_separately=False).to(self.device)
        out = self.model(**{k: (v.to(self.model.dtype)
                                if v.dtype.is_floating_point else v)
                            if hasattr(v, "dtype") else v
                            for k, v in inputs.items()})
        logits = out.logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        yes = probs[:, self.yes_ids].sum(dim=-1)
        no = probs[:, self.no_ids].sum(dim=-1)
        mass = yes + no
        return ((yes / mass.clamp_min(1e-12)).cpu().numpy(), mass.cpu().numpy())


BACKENDS = {
    "qwen3-vl-8b": lambda dev: ChatVLBackend("Qwen/Qwen3-VL-8B-Instruct", dev),
    "cosmos-reason2-8b": lambda dev: ChatVLBackend("nvidia/Cosmos-Reason2-8B", dev),
    "qwen2.5-vl-3b": lambda dev: ChatVLBackend("Qwen/Qwen2.5-VL-3B-Instruct", dev),
    "paligemma-3b-base": lambda dev: PaliGemmaBaseBackend(device=dev),
    "paligemma-wm": lambda dev: PaliGemmaWMBackend("ckpts/paligemma_wm_ogbench", device=dev),
}


class OpenAIBackend:
    """API models (GPT-5.6 Luna etc.) via chat completions with logprobs.
    p_yes pooled over yes/no variants found in top_logprobs of the answer
    token, renormalized; mass = combined yes+no probability found there.
    Images sent as base64 data URLs at detail:'high' (224px = 1 tile)."""

    VARIANTS_YES = ("yes", " yes", "Yes", " Yes", "YES")
    VARIANTS_NO = ("no", " no", "No", " No", "NO")

    def __init__(self, model_id: str, concurrency: int = 8):
        import os
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model_id = model_id
        self.concurrency = concurrency

    @staticmethod
    def _data_url(arr):
        import base64
        import io
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def _one(self, it):
        pair = it["qtype"] in CLOSER
        content = []
        if pair:
            content += [{"type": "image_url", "image_url": {"url": self._data_url(it["start"]), "detail": "high"}},
                        {"type": "image_url", "image_url": {"url": self._data_url(it["end"]), "detail": "high"}}]
            text = f"{PAIR_PREAMBLE}{it['question']} Answer with one word: yes or no."
        else:
            content += [{"type": "image_url", "image_url": {"url": self._data_url(it["end"]), "detail": "high"}}]
            text = f"{it['question']} Answer with one word: yes or no."
        content.append({"type": "text", "text": text})
        # Luna (gpt-5.x reasoning family) supports neither temperature nor
        # logprobs: parse the generated word instead. Hard 0/1 p_yes; mass=1
        # when an answer parses, 0 otherwise (those score as abstentions).
        r = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=16, reasoning_effort="none")
        text = (r.choices[0].message.content or "").strip().lower()
        if text.startswith("yes"):
            return 1.0, 1.0
        if text.startswith("no"):
            return 0.0, 1.0
        return 0.5, 0.0

    def score(self, items):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            out = list(ex.map(self._one, items))
        return (np.array([o[0] for o in out]), np.array([o[1] for o in out]))


BACKENDS["gpt-5.6-luna"] = lambda dev: OpenAIBackend("gpt-5.6-luna")
