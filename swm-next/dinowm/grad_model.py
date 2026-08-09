"""DinoWM predictor as an SWMModel: plugs into the swms eval harness.

Subclasses SWMGradModel to inherit its get_probabilistic_rewards_wm reward
loop verbatim (it only touches the model through get_scores), substituting
the model stack: frozen Qwen3-VL ViT encoder -> OneShotPredictor (loaded
from a train_predictor.py Trainer checkpoint) -> frozen Qwen judge scoring
yes/no on the PREDICTED future latent.

Semantics vs the PaliGemma SWMGradModel, documented for review:
  - History: the harness supplies one current frame; the predictor's 2-frame
    history is the frame duplicated (identical to the trajectory-start clamp
    the model saw in training).
  - Scores: probability mass summed over the single-token yes/no variants
    (" Yes"/" yes"/"Yes"/"yes" etc.) instead of one token id; same convention
    the training-time judge uses.
  - Actions arrive as variable-length gradient-carrying tensors; they are
    zero-padded to max_horizon with a validity mask WITHOUT detaching, so the
    gradient planner's chain through the action sequence is intact.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from swm.constants import ANSWER_OPTIONS
from swm.semantic_world_model import SWMGradModel

from .judge import StudentVQAHead
from .oneshot_world_model import OneShotWorldModel
from .qwen3_vl import Qwen3VLViTEncoder
from .vit_oneshot import OneShotPredictor


class DinoWMGradModel(SWMGradModel):
    def __init__(self, checkpoint_path: str, config_path: str,
                 tokens=ANSWER_OPTIONS, precision=torch.bfloat16, device="cuda"):
        # Deliberately NOT calling super().__init__ (it loads PaliGemma);
        # get_probabilistic_rewards_wm only uses self.get_scores/self.tokens.
        self.tokens = tokens
        self.precision = precision
        self.device = device

        cfg = yaml.safe_load(open(config_path))
        self.image_size = int(cfg["image_size"])
        self.obs_horizon = int(cfg["obs_horizon"])
        self.max_horizon = int(cfg["max_horizon"])
        self.action_dim = int(cfg["action_dim"])

        encoder = Qwen3VLViTEncoder(
            model_id=cfg["model_id"], image_size=self.image_size,
            freeze=True, output_stage=cfg.get("output_stage", "pre_merger"),
        ).to(device)
        predictor = OneShotPredictor(
            emb_dim=encoder.emb_dim, io_dim=None, action_dim=self.action_dim,
            num_patches=encoder.num_patches, obs_horizon=self.obs_horizon,
            max_action_horizon=self.max_horizon, depth=int(cfg["depth"]),
            heads=int(cfg["heads"]), mlp_dim=int(cfg["mlp_dim"]),
            dropout=float(cfg["dropout"]),
        ).to(device)

        state = self._load_predictor_state(checkpoint_path)
        predictor.load_state_dict(state, strict=True)
        predictor.eval()
        for p in predictor.parameters():
            p.requires_grad = False

        self.wm = OneShotWorldModel(encoder=encoder, predictor=predictor).to(device).eval()
        self.judge = StudentVQAHead(cfg["model_id"], device=device,
                                    precision=precision, image_size=self.image_size)
        self.model = self.judge.full_model  # parity with SWMGradModel attribute

    @staticmethod
    def _load_predictor_state(checkpoint_path: str) -> dict:
        """Load 'predictor.*' keys from a Trainer checkpoint dir (safetensors
        or bin) or a raw .pt state dict, stripping the wrapper prefix."""
        if os.path.isdir(checkpoint_path):
            st_path = os.path.join(checkpoint_path, "model.safetensors")
            if os.path.exists(st_path):
                from safetensors.torch import load_file
                state = load_file(st_path)
            else:
                state = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"),
                                   map_location="cpu")
        else:
            state = torch.load(checkpoint_path, map_location="cpu")
            if "predictor" in state and isinstance(state["predictor"], dict):
                state = state["predictor"]
        out = {}
        for k, v in state.items():
            out[k[len("predictor."):] if k.startswith("predictor.") else k] = v
        return out

    # ------------------------------------------------------------- scoring
    def _frame_tensor(self, image) -> torch.Tensor:
        """Env frame (np uint8 HWC or PIL) -> (3, S, S) float in [0, 1]."""
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8)
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

    def get_scores(self, images, actions, questions):
        if isinstance(questions, str):
            questions = [questions] * len(images)
        B = len(images)
        device = self.device

        # actions=None: current-state VQA (the goal generator probes e.g.
        # "is the robot grasping X?" before planning). Map to h=0 — an empty
        # action window with an all-False mask — the identity-prediction case
        # the predictor trained on (target == current frame at horizon 0).
        if actions is None:
            actions = [torch.zeros(0, self.action_dim, device=device)] * B

        # The planner scores one current frame against many action candidates;
        # encode it once and expand. Fall back to per-image encoding if the
        # batch ever mixes frames.
        same = all(im is images[0] for im in images)
        frames = [self._frame_tensor(images[0])] if same else [self._frame_tensor(im) for im in images]
        stack = torch.stack(frames).to(device)                       # (1|B, 3, S, S)
        hist = stack.unsqueeze(1).expand(-1, self.obs_horizon, -1, -1, -1)
        z_obs = self.wm.encode(hist)                                 # (1|B, T, P, D)
        if same:
            z_obs = z_obs.expand(B, -1, -1, -1)

        # Pad variable-length gradient-carrying action tensors; no detach.
        padded, mask = [], torch.zeros(B, self.max_horizon, dtype=torch.bool, device=device)
        for i, a in enumerate(actions):
            a = a.to(device=device, dtype=torch.float32)
            h = a.shape[0]
            assert h <= self.max_horizon, (h, self.max_horizon)
            padded.append(F.pad(a, (0, 0, 0, self.max_horizon - h)))
            mask[i, :h] = True
        acts = torch.stack(padded)                                   # (B, H, A)

        z_pred = self.wm.predictor(z_obs, acts, action_mask=mask)    # (B, P, D)
        img_embeds = self.judge.merge(z_pred)
        infos = [self.judge._prompt(q) for q in questions]
        logits = self.judge._llm_answer_logits_multi(infos, img_embeds)
        probs = torch.softmax(logits, dim=-1)
        p_yes = probs[:, infos[0].desired_token_ids].sum(dim=-1)
        p_no = probs[:, infos[0].other_token_ids].sum(dim=-1)
        return p_yes, p_no
