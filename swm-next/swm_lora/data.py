"""Dataset over datagen.py HDF5(s): samples = (frame_t, question, trajectory,
teacher p). Two samples per proposal (approach, place); hold is dropped in
training (duplicate of approach by construction — reused at inference).

Split is by WHOLE EPISODES (no frame leakage across train/val), stratified by
selection policy. Trajectories are normalized with the diffusion checkpoint's
own stats so normalization is frozen with the data, not the code.
"""
import io
import json

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

QUESTION_KEYS = [("q_approach", 0), ("q_place", 1)]   # (episode attr, label col)


def action_stats(diffusion_ckpt_path):
    """min/max action stats from the diffusion checkpoint (its own
    normalization convention: x -> 2*(x-min)/(max-min) - 1)."""
    ck = torch.load(diffusion_ckpt_path, map_location="cpu", weights_only=False)
    stats = ck["stats"]["action"]
    return np.asarray(stats["min"], np.float32), np.asarray(stats["max"], np.float32)


def normalize(traj, amin, amax):
    return 2.0 * (traj - amin) / (amax - amin + 1e-8) - 1.0


class ProposalDataset(Dataset):
    def __init__(self, h5_paths, episodes, amin, amax):
        """episodes: list of (path_idx, ep_name). Index maps are built once;
        h5 files are opened lazily per worker (h5py is not fork-safe)."""
        self.paths = list(h5_paths)
        self.amin, self.amax = amin, amax
        self._files = None
        self.index = []          # (path_idx, ep, cycle, proposal, label_col, q)
        for pi, ep in episodes:
            with h5py.File(self.paths[pi], "r") as f:
                g = f[ep]
                C, k, _ = g["labels"].shape
                qs = {a: g.attrs[a] for a, _ in QUESTION_KEYS}
            for c in range(C):
                for p in range(k):
                    for attr, col in QUESTION_KEYS:
                        self.index.append((pi, ep, c, p, col, qs[attr]))

    def _f(self, pi):
        if self._files is None:
            self._files = [None] * len(self.paths)
        if self._files[pi] is None:
            self._files[pi] = h5py.File(self.paths[pi], "r")
        return self._files[pi]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        pi, ep, c, p, col, q = self.index[i]
        g = self._f(pi)[ep]
        img = Image.open(io.BytesIO(g["frames_t"][c].tobytes())).convert("RGB")
        traj = normalize(g["trajs"][c, p].astype(np.float32), self.amin, self.amax)
        target = np.float32(g["labels"][c, p, col])
        return dict(image=img, question=str(q), traj=torch.from_numpy(traj),
                    target=torch.tensor(target))


def episode_split(h5_paths, val_frac=0.1, seed=0):
    """Whole-episode split, stratified by policy flag. Returns (train, val)
    lists of (path_idx, ep_name)."""
    by_policy = {"verifier": [], "random": []}
    for pi, path in enumerate(h5_paths):
        with h5py.File(path, "r") as f:
            for ep in f:
                by_policy[f[ep].attrs["policy"]].append((pi, ep))
    rng = np.random.RandomState(seed)
    train, val = [], []
    for pol, eps in by_policy.items():
        eps = sorted(eps)
        rng.shuffle(eps)
        n_val = max(1, int(round(len(eps) * val_frac)))
        val += eps[:n_val]
        train += eps[n_val:]
    return sorted(train), sorted(val)


def collate(batch):
    return dict(images=[b["image"] for b in batch],
                questions=[b["question"] for b in batch],
                trajs=torch.stack([b["traj"] for b in batch]),
                targets=torch.stack([b["target"] for b in batch]))


def save_split(path, train, val, h5_paths):
    json.dump(dict(h5_paths=[str(p) for p in h5_paths],
                   train=train, val=val), open(path, "w"), indent=1)
