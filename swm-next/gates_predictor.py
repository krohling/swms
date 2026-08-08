"""Pre-launch gates for the DinoWM predictor arms (run on the cluster).

G-DIN-1  sampling equivalence: for the same (index, seed, i), the predictor
         dataset returns THE SAME drawn row as the reproduction's SAQADataset
         (question/answer/qtype identical, current frame == last history
         frame, raw actions equal), plus correct target frame, h=0 handling
         and history clamping at trajectory start.
G-DIN-2  checkpoint hygiene: the wrapper's state_dict contains ONLY
         predictor.* keys (no encoder, no judge), and a save->load round trip
         reproduces identical predictor outputs.

    python gates_predictor.py --config configs/predictor_cosine.yaml
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saqa import SAQADataset, SAQAIndex  # noqa: E402
from dinowm.data import PredictorSAQADataset  # noqa: E402


def gate_din1(cfg: dict, n_checks: int = 500) -> None:
    files = [cfg["noisy_hdf5"], cfg["play_hdf5"]]
    idx = SAQAIndex.build(files, split="train", val_frac=cfg["val_frac"])
    ref = SAQADataset(idx, length=n_checks, seed=cfg["seed"])
    ours = PredictorSAQADataset(idx, length=n_checks, seed=cfg["seed"],
                                obs_horizon=int(cfg["obs_horizon"]),
                                max_action_horizon=int(cfg["max_horizon"]))
    n_h0 = n_clamp = 0
    for i in range(n_checks):
        a, b = ref[i], ours[i]
        assert a["question"] == b["question"], i
        assert (a["answer"] == "yes") == b["answer_yes"], i
        assert a["qtype"] == b["qtype"], i
        # current frame == last history frame
        cur = torch.from_numpy(a["image"]).permute(2, 0, 1).float() / 255.0
        assert torch.equal(cur, b["history"][-1]), f"frame mismatch at draw {i}"
        # raw actions equal on the valid prefix, zero beyond, mask consistent
        h = int(b["action_mask"].sum())
        assert a["actions"].shape[0] == h, (i, a["actions"].shape, h)
        if h:
            assert torch.allclose(a["actions"], b["actions"][:h]), i
        assert torch.count_nonzero(b["actions"][h:]) == 0, i
        if h == 0:
            n_h0 += 1
            assert torch.equal(b["history"][-1], b["target"]), f"h=0 target at {i}"
        # verify target against the file directly
        g, d, i0, i1, qi, stratum = ours._draw(i)
        tgt = torch.from_numpy(np.asarray(g["frames"][i1], dtype=np.uint8)) \
                   .permute(2, 0, 1).float() / 255.0
        assert torch.equal(tgt, b["target"]), i
        if i0 - int(cfg["obs_horizon"]) + 1 < 0:
            n_clamp += 1
            assert torch.equal(b["history"][0], b["history"][-1]) or i0 > 0
    print(f"G-DIN-1 PASS: {n_checks} draws identical to SAQADataset "
          f"({n_h0} with h=0, {n_clamp} history-clamped)", flush=True)


def gate_din2(cfg: dict) -> None:
    from dinowm.judge import StudentVQAHead
    from dinowm.model import PredictorForSAQA
    from dinowm.oneshot_world_model import OneShotWorldModel
    from dinowm.qwen3_vl import Qwen3VLViTEncoder
    from dinowm.vit_oneshot import OneShotPredictor

    device = torch.device("cuda")
    encoder = Qwen3VLViTEncoder(model_id=cfg["model_id"],
                                image_size=int(cfg["image_size"]),
                                freeze=True,
                                output_stage=cfg.get("output_stage", "pre_merger")).to(device)
    def make_predictor():
        return OneShotPredictor(
            emb_dim=encoder.emb_dim, io_dim=None, action_dim=int(cfg["action_dim"]),
            num_patches=encoder.num_patches, obs_horizon=int(cfg["obs_horizon"]),
            max_action_horizon=int(cfg["max_horizon"]), depth=int(cfg["depth"]),
            heads=int(cfg["heads"]), mlp_dim=int(cfg["mlp_dim"]),
            dropout=float(cfg["dropout"])).to(device)

    wm = OneShotWorldModel(encoder=encoder, predictor=make_predictor()).to(device)
    judge = StudentVQAHead(cfg["model_id"], device=device,
                           image_size=int(cfg["image_size"]))
    model = PredictorForSAQA(wm, judge, objective=cfg["objective"],
                             cos_weight=float(cfg.get("cos_weight", 0.0)))

    keys = list(model.state_dict().keys())
    bad = [k for k in keys if not k.startswith("predictor.")]
    assert not bad, f"non-predictor keys in state_dict: {bad[:5]}"
    print(f"G-DIN-2a PASS: state_dict has {len(keys)} keys, all predictor.*", flush=True)

    # save -> load into a fresh predictor -> identical outputs
    S = int(cfg["image_size"])
    B, T, A, H = 2, int(cfg["obs_horizon"]), int(cfg["action_dim"]), int(cfg["max_horizon"])
    hist = torch.rand(B, T, 3, S, S, device=device)
    acts = torch.randn(B, H, A, device=device)
    mask = torch.zeros(B, H, dtype=torch.bool, device=device); mask[:, :4] = True
    model.eval()
    with torch.no_grad():
        z1 = wm.predictor(wm.encode(hist), acts, action_mask=mask)
    path = "/tmp/gate_din2_predictor.pt"
    torch.save(model.state_dict(), path)
    wm2 = OneShotWorldModel(encoder=encoder, predictor=make_predictor()).to(device)
    model2 = PredictorForSAQA(wm2, judge, objective=cfg["objective"])
    model2.load_state_dict(torch.load(path, map_location=device))
    model2.eval()
    with torch.no_grad():
        z2 = wm2.predictor(wm2.encode(hist), acts, action_mask=mask)
    assert torch.equal(z1, z2), f"max diff {(z1 - z2).abs().max().item()}"
    os.remove(path)
    print("G-DIN-2b PASS: save->load round trip bit-identical", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-gpu", action="store_true", help="run only G-DIN-1")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    gate_din1(cfg)
    if not args.skip_gpu:
        gate_din2(cfg)
    print("ALL_GATES_PASS", flush=True)
