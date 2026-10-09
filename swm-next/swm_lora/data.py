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


class CycleDataset(Dataset):
    """Run-2 unit: one (planning step, question) = the frame + all k sibling
    proposals + their teacher p's. The within-cycle ranking loss needs the
    siblings together; frame-conditional shortcuts contribute exactly zero
    to that term because everything but the trajectories is shared."""

    def __init__(self, h5_paths, episodes, amin, amax, questions="planner"):
        """questions: "planner" (approach+place from labels), "all" (also every
        aux question stored in the file), or an explicit list of question
        strings to include. Question texts come from file attrs — the trainer
        stays agnostic to what the dataset contains."""
        self.paths = list(h5_paths)
        self.amin, self.amax = amin, amax
        self._files = None
        self.index = []          # (path_idx, ep, cycle, source, col, q)
        for pi, ep in episodes:
            with h5py.File(self.paths[pi], "r") as f:
                g = f[ep]
                C = g["labels"].shape[0]
                cand = [("labels", col, str(g.attrs[a]))
                        for a, col in QUESTION_KEYS]
                if questions != "planner" and "aux_labels" in g:
                    for col, q in enumerate(g.attrs.get("aux_questions", [])):
                        cand.append(("aux_labels", col, str(q)))
                if isinstance(questions, (list, tuple)):
                    cand = [c for c in cand if c[2] in set(questions)]
            for c in range(C):
                for src, col, q in cand:
                    self.index.append((pi, ep, c, src, col, q))

    def _f(self, pi):
        if self._files is None:
            self._files = [None] * len(self.paths)
        if self._files[pi] is None:
            self._files[pi] = h5py.File(self.paths[pi], "r")
        return self._files[pi]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        pi, ep, c, src, col, q = self.index[i]
        g = self._f(pi)[ep]
        img = Image.open(io.BytesIO(g["frames_t"][c].tobytes())).convert("RGB")
        trajs = normalize(g["trajs"][c].astype(np.float32), self.amin, self.amax)
        targets = g[src][c, :, col].astype(np.float32)
        return dict(image=img, question=str(q),
                    trajs=torch.from_numpy(trajs),
                    targets=torch.from_numpy(targets))


def collate_cycles(batch):
    """Flatten M cycles x k proposals into one forward batch; keeps cycle
    boundaries for the listwise term."""
    k = batch[0]["trajs"].shape[0]
    images, questions = [], []
    for b in batch:
        images += [b["image"]] * k
        questions += [b["question"]] * k
    return dict(images=images, questions=questions, k=k, m=len(batch),
                trajs=torch.cat([b["trajs"] for b in batch]),
                targets=torch.cat([b["targets"] for b in batch]))


class ExecutedChunkDataset(Dataset):
    """Run-3 unit: one (executed chunk, question) sample from k=1 noisy
    capture files. Questions = 2 planner labels + the 6 aux questions stored
    by datagen --aux-labels. binarize_aux thresholds aux targets at 0.5
    (oracle-style hard targets ablation); planner targets stay soft."""

    def __init__(self, h5_paths, episodes, amin, amax, use_aux=True,
                 binarize_aux=False):
        self.paths = list(h5_paths)
        self.amin, self.amax = amin, amax
        self.use_aux, self.binarize_aux = use_aux, binarize_aux
        self._files = None
        self.index = []    # (pi, ep, cycle, source, qi)  source: 0=labels 1=aux
        for pi, ep in episodes:
            with h5py.File(self.paths[pi], "r") as f:
                g = f[ep]
                C = g["labels"].shape[0]
                qs = [str(g.attrs["q_approach"]), str(g.attrs["q_place"])]
                aux_qs = [str(q) for q in g.attrs.get("aux_questions", [])] \
                    if use_aux else []
            for c in range(C):
                for qi in range(2):
                    self.index.append((pi, ep, c, 0, qi, qs[qi]))
                for qi in range(len(aux_qs)):
                    self.index.append((pi, ep, c, 1, qi, aux_qs[qi]))

    def _f(self, pi):
        if self._files is None:
            self._files = [None] * len(self.paths)
        if self._files[pi] is None:
            self._files[pi] = h5py.File(self.paths[pi], "r")
        return self._files[pi]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        pi, ep, c, src, qi, q = self.index[i]
        g = self._f(pi)[ep]
        img = Image.open(io.BytesIO(g["frames_t"][c].tobytes())).convert("RGB")
        traj = normalize(g["trajs"][c, 0].astype(np.float32), self.amin, self.amax)
        if src == 0:
            t = np.float32(g["labels"][c, 0, qi])        # 0=approach 1=place
        else:
            t = np.float32(g["aux_labels"][c, 0, qi])
            if self.binarize_aux:
                t = np.float32(t > 0.5)
        return dict(image=img, question=q, traj=torch.from_numpy(traj),
                    target=torch.tensor(t))
