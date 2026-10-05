"""Qwen3-VL SWM: frozen Qwen3-VL-8B + trajectory projector + LM LoRA.

Architecture (option 1, token-append):
  [image][question + "Answer with one word: yes or no."][Trajectory:][16 traj
  tokens] -> yes/no logits at the final position.
Trajectory tokens enter by PLACEHOLDER SCATTER: the prompt contains 16 copies
of a reserved special token; a forward hook on the input embedding layer
replaces those positions with projector(actions). No Qwen modeling code is
copied or modified — the image merge, mrope, etc. all run untouched on normal
input_ids. Vision tower / embeddings / LM head frozen; LoRA on LM q/k/v/o.

p_yes readout matches label_teacher.QwenJudge: two-class renormalization over
yes/no token variants, which collapses to a single logit
  l = LSE(z_YES) - LSE(z_NO),  q = sigmoid(l)
so training uses BCEWithLogits(l, teacher_p) — see loss() here.

Zero-init: projector output is multiplied by a learnable scalar initialized
to 0, so at step 0 traj tokens contribute the placeholder embedding only and
(with LoRA B=0) the model is EXACTLY base Qwen.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

TRAJ_PLACEHOLDER = "<|fim_middle|>"   # reserved Qwen special, unused in prompts
PROMPT_SUFFIX = " Answer with one word: yes or no."
YES_VARIANTS = [" Yes", " yes", "Yes", "yes"]
NO_VARIANTS = [" No", " no", "No", "no"]


class TrajectoryProjector(nn.Module):
    def __init__(self, action_dim, d_model, hidden=512, init_scale=0.0):
        """init_scale=0 -> exact-base at step 0 (run-1 behavior; shown to let
        the frame-mean shortcut win while the gate stays shut). init_scale>0
        (run-2: 0.1) opens the trajectory channel from the start; LoRA B=0
        still keeps the LM itself at base behavior at init."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden), nn.GELU(), nn.Linear(hidden, d_model))
        self.scale = nn.Parameter(torch.tensor([float(init_scale)]))

    def forward(self, actions):                     # (B, H, A) -> (B, H, D)
        return self.net(actions) * self.scale


class QwenSWM(nn.Module):
    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Instruct", action_dim=5,
                 horizon=16, lora_r=16, lora_alpha=32, lora_dropout=0.05,
                 device="cuda", dtype=torch.bfloat16, proj_init_scale=0.0):
        super().__init__()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from peft import LoraConfig, get_peft_model

        self.processor = AutoProcessor.from_pretrained(model_id)
        tok = self.processor.tokenizer
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True)
        for p in base.parameters():
            p.requires_grad = False

        lcfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                          lora_dropout=lora_dropout, bias="none",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          exclude_modules=r".*visual.*")
        self.model = get_peft_model(base, lcfg)

        d_model = base.config.text_config.hidden_size
        self.projector = TrajectoryProjector(action_dim, d_model,
                                             init_scale=proj_init_scale)
        # projector stays fp32 (stable AdamW on the zero-init scale);
        # the scatter hook casts its output to the model dtype.
        self.horizon = horizon

        self.traj_id = tok.convert_tokens_to_ids(TRAJ_PLACEHOLDER)
        assert isinstance(self.traj_id, int) and self.traj_id >= 0 and \
            self.traj_id != tok.unk_token_id, "placeholder must be a real token"

        def singles(cands):
            return [tok(c, add_special_tokens=False).input_ids[0]
                    for c in cands
                    if len(tok(c, add_special_tokens=False).input_ids) == 1]
        self.yes_ids = singles(YES_VARIANTS)
        self.no_ids = singles(NO_VARIANTS)

        # scatter hook: replaces placeholder-position embeddings with
        # projected actions staged in self._pending (set per forward)
        self._pending = None

        def hook(module, inputs, output):
            if self._pending is None:
                return output
            ids, traj_embeds = self._pending
            out = output.clone()
            mask = ids == self.traj_id                     # (B, T)
            # each row has exactly `horizon` placeholders, in order
            out[mask] = traj_embeds.reshape(-1, traj_embeds.shape[-1]) \
                .to(out.dtype)
            return out

        self.model.get_input_embeddings().register_forward_hook(hook)
        self.device = device
        self.to_device()

    def to_device(self):
        self.model.to(self.device)
        self.projector.to(self.device)

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad] + \
               list(self.projector.parameters())

    def build_inputs(self, images, questions):
        """Chat-templated batch; trajectory placeholders embedded in the user
        text. images: list of PIL/np (or None for empty-trajectory probes)."""
        texts = []
        for q in questions:
            content = [{"type": "image"},
                       {"type": "text",
                        "text": f"{q}{PROMPT_SUFFIX} Trajectory: "
                                + TRAJ_PLACEHOLDER * self.horizon}]
            msgs = [{"role": "user", "content": content}]
            texts.append(self.processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False))
        return self.processor(text=texts, images=[[im] for im in images],
                              return_tensors="pt", padding=True)

    def answer_logit(self, images, questions, trajs):
        """l = LSE(z_YES) - LSE(z_NO) at the final position. trajs: (B,H,A)
        normalized; pass zeros for the empty-trajectory/drift probe."""
        inputs = self.build_inputs(images, questions).to(self.device)
        traj_embeds = self.projector(trajs.to(self.device, dtype=torch.float32))
        self._pending = (inputs["input_ids"], traj_embeds)
        try:
            out = self.model(**inputs)
        finally:
            self._pending = None
        mask = inputs["attention_mask"]
        last = mask.sum(dim=1) - 1
        idx = torch.arange(out.logits.shape[0], device=out.logits.device)
        z = out.logits[idx, last, :].float()
        l_yes = torch.logsumexp(z[:, self.yes_ids], dim=-1)
        l_no = torch.logsumexp(z[:, self.no_ids], dim=-1)
        return l_yes - l_no

    @staticmethod
    def loss(logit, target_p):
        """Soft-target BCE on the aggregated two-class logit (== KL to the
        teacher's {yes,no} distribution up to a constant)."""
        return F.binary_cross_entropy_with_logits(logit, target_p)

    @torch.no_grad()
    def p_yes(self, images, questions, trajs):
        return torch.sigmoid(self.answer_logit(images, questions, trajs))

    def save_adapters(self, path):
        import os
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)                   # LoRA only
        torch.save(self.projector.state_dict(), f"{path}/projector.pt")

    def load_adapters(self, path):
        from peft import set_peft_model_state_dict
        import safetensors.torch as st
        import os
        f = f"{path}/adapter_model.safetensors"
        set_peft_model_state_dict(self.model, st.load_file(f))
        self.projector.load_state_dict(
            torch.load(f"{path}/projector.pt", map_location=self.device))
