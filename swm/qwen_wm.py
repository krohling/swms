"""Planning adapter for the Qwen3-VL + LoRA SWM (swm-next/train_qwen.py).

Subclasses SWMGradModel so the reward loop, objective handling and planner
interface are inherited unchanged; only construction and scoring differ.
Model construction and prompt/action encoding come from swm.qwen_backend
(shared with training), so the planner sees exactly the distribution the
model was trained on. Actions reach the model through QwenWithActions'
embedding hook, which keeps them in the autograd graph for the gradient
planner.
"""
import json
import os

import torch
import yaml
from safetensors.torch import load_file

from swm.constants import ANSWER_OPTIONS
from swm.qwen_backend import build_model, encode, single_token
from swm.semantic_world_model import SWMGradModel


class QwenSWMGradModel(SWMGradModel):
    def __init__(self, checkpoint_path, qwen_config,
                 tokens=ANSWER_OPTIONS, precision=torch.bfloat16,
                 device="cuda", objective="sigmoid"):
        if objective not in self.OBJECTIVES:
            raise ValueError(f"objective must be one of {self.OBJECTIVES}, got {objective!r}")
        self.objective = objective
        self.precision = precision
        self.device = device
        self.tokens = tokens

        cfg = yaml.safe_load(open(qwen_config))
        self.action_dim = int(cfg["action_dim"])
        model, self.processor, self.placeholder = build_model(cfg)
        self._load_trainer_state(model, checkpoint_path)
        model.to(device=device, dtype=precision)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        self.model = model

        tok = self.processor.tokenizer
        # Same single-token yes/no the training loss supervised.
        self.token_to_id = {t: single_token(tok, t) for t in tokens}

    @staticmethod
    def _load_trainer_state(model, ckpt_dir):
        index = os.path.join(ckpt_dir, "model.safetensors.index.json")
        single = os.path.join(ckpt_dir, "model.safetensors")
        if os.path.exists(index):
            state = {}
            for shard in sorted(set(json.load(open(index))["weight_map"].values())):
                state.update(load_file(os.path.join(ckpt_dir, shard)))
        elif os.path.exists(single):
            state = load_file(single)
        else:
            state = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"),
                               map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # Base weights may legitimately be absent (they come from from_pretrained);
        # anything that was trained -- LoRA + action modules -- must load.
        trained = {n for n, p in model.named_parameters() if p.requires_grad}
        lost = trained & set(missing)
        assert not lost, f"checkpoint is missing trained parameters: {sorted(lost)[:8]}"
        assert not unexpected, f"unexpected checkpoint keys: {sorted(unexpected)[:8]}"

    def get_scores(self, images, actions, questions, return_logits=False):
        if isinstance(questions, str):
            questions = [questions] * len(images)
        batch = [{"question": q, "actions": a, "image": im}
                 for q, a, im in zip(questions, actions, images)]
        inputs = encode(self.processor, self.placeholder, self.action_dim,
                        batch, with_answer=False)
        acts = inputs.pop("action_values")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        inputs["action_values"] = acts.to(self.device)

        outputs = self.model(**inputs)
        last = inputs["attention_mask"].sum(dim=1) - 1
        logits = outputs.logits[torch.arange(len(batch)), last].to(torch.float32)
        probs = torch.softmax(logits, dim=1)
        results = tuple(probs[:, self.token_to_id[t]] for t in self.tokens)
        if return_logits:
            token_logits = tuple(logits[:, self.token_to_id[t]] for t in self.tokens)
            return results, token_logits
        return results
