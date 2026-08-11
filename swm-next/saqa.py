"""SAQA dataset: balanced sampling over the HDF5 built by add_relative_questions.py.

Paper 4.1: "the dataset was balanced in both the number of each possible
question type and the answer distribution for each respective question ...
addressed during training by oversampling tuples such that there is a balanced
amount of question types and answer distributions."

We implement that as uniform sampling over the 12 strata (6 question types x
yes/no). The natural distribution is severely skewed — `block_ontop_block`=yes
is 0.2% of all question instances — so the rare strata are oversampled heavily.
`report_index()` prints the per-stratum pool sizes and the resulting repetition
factor, which is gate G2.

The index is a list, per stratum, of (file, traj, start, horizon, q_idx) rows.
Building it scans every question in the HDF5 once (~15M rows, a few minutes),
so it is cached next to the data.

Train/val split is by trajectory: the last `val_frac` of each file's
trajectories are held out and never trained on.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

QUESTION_TYPES = ("cube_grasped", "block_gripper_touching", "block_touching_block",
                  "block_ontop_block", "gripper_block_closer", "block_block_closer")
STRATA = [(t, a) for t in QUESTION_TYPES for a in (False, True)]


def _s(x) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


@dataclass
class SAQAIndex:
    """Per-stratum row lists. Columns: file, traj, start, horizon, q_idx."""
    rows: dict[tuple[str, bool], np.ndarray]
    files: list[str]
    trajs: list[list[str]]

    @staticmethod
    def _cache_path(files: list[str], split: str, teacher: bool = False) -> str:
        stem = "_".join(os.path.basename(f).split(".")[0] for f in files)
        tag = "_teacher" if teacher else ""
        return os.path.join(os.path.dirname(files[0]), f".saqa_index_{stem}{tag}_{split}.npz")

    @classmethod
    def build(cls, files: list[str], split: str, val_frac: float = 0.1,
              cache: bool = True, teacher_files: list[str] | None = None) -> "SAQAIndex":
        """teacher_files: optional sidecar HDF5s (from label_teacher.py), one
        per input file. When given, STRATA MEMBERSHIP uses the teacher's
        argmax answer (p_yes >= 0.5) instead of the oracle answer, so the
        balanced sampler balances what the model will actually be trained on.
        Rows keep the same (file, traj, start, horizon, q_idx) shape."""
        import h5py

        path = cls._cache_path(files, split, teacher=teacher_files is not None)
        if cache and os.path.exists(path):
            z = np.load(path, allow_pickle=True)
            rows = {STRATA[i]: z[f"s{i}"] for i in range(len(STRATA))}
            return cls(rows, list(z["files"]), [list(t) for t in z["trajs"]])

        acc: dict[tuple[str, bool], list] = {s: [] for s in STRATA}
        kept_trajs: list[list[str]] = []
        for fi, fpath in enumerate(files):
            tf = h5py.File(teacher_files[fi], "r") if teacher_files else None
            with h5py.File(fpath, "r") as f:
                names = sorted(f.keys(), key=lambda n: int(n.split("_")[-1]))
                cut = int(round(len(names) * (1 - val_frac)))
                names = names[:cut] if split == "train" else names[cut:]
                kept_trajs.append(names)
                for ti, tname in enumerate(names):
                    hs = f[tname]["horizon_start"]
                    for start in hs:
                        sg = hs[start]
                        for hname in sg:
                            h = int(hname.split("_")[-1])
                            d = sg[hname]
                            types = d["types"][()]
                            answers = d["answers"][()]
                            if tf is not None:
                                p = tf[tname]["horizon_start"][start][hname]["p_yes"][()]
                                answers = p >= 0.5
                            for qi, (t, a) in enumerate(zip(types, answers)):
                                acc[(_s(t), bool(a))].append(
                                    (fi, ti, int(start), h, qi))
            if tf is not None:
                tf.close()

        rows = {k: np.asarray(v, dtype=np.int32).reshape(-1, 5) for k, v in acc.items()}
        idx = cls(rows, files, kept_trajs)
        if cache:
            np.savez_compressed(path, files=np.array(files),
                                trajs=np.array(kept_trajs, dtype=object),
                                **{f"s{i}": rows[STRATA[i]] for i in range(len(STRATA))})
        return idx

    def report(self, draws: int) -> bool:
        """Gate G2: pool sizes and oversampling factors for a run of `draws`."""
        per = draws / len(STRATA)
        total = sum(len(v) for v in self.rows.values())
        print(f"  {'stratum':34s} {'pool':>12s} {'natural':>9s} {'repeats':>9s}")
        ok = True
        for s in STRATA:
            n = len(self.rows[s])
            rep = per / max(n, 1)
            flag = ""
            if n == 0:
                ok = False
                flag = "   <-- EMPTY"
            elif rep > 50:
                flag = "   <-- heavy"
            print(f"  {s[0] + '=' + ('yes' if s[1] else 'no'):34s} {n:12,d} "
                  f"{100*n/total:8.2f}% {rep:8.1f}x{flag}")
        print(f"  total question instances: {total:,}  |  draws/stratum: {per:,.0f}")
        return ok


class SAQADataset(Dataset):
    """Balanced draws. `__len__` is the training budget, not the corpus size.

    Each index deterministically seeds a draw: pick a stratum uniformly, then a
    row uniformly within it. Deterministic in the index so a resumed run repeats
    the same stream.
    """

    def __init__(self, index: SAQAIndex, length: int, seed: int = 0,
                 teacher_files: list[str] | None = None):
        """teacher_files: optional label_teacher.py sidecars (one per index
        file). When given, `answer` comes from the teacher's argmax
        (p_yes >= 0.5) instead of the oracle — the ONLY behavioral change;
        draws, images and actions are identical to oracle mode."""
        self.index = index
        self.length = int(length)
        self.seed = seed
        self.teacher_files = teacher_files
        self._files: dict[int, object] = {}      # opened lazily, per worker
        self._tfiles: dict[int, object] = {}

    def __len__(self) -> int:
        return self.length

    def _h5(self, fi: int):
        import h5py
        if fi not in self._files:
            self._files[fi] = h5py.File(self.index.files[fi], "r")
        return self._files[fi]

    def _t5(self, fi: int):
        import h5py
        if fi not in self._tfiles:
            self._tfiles[fi] = h5py.File(self.teacher_files[fi], "r")
        return self._tfiles[fi]

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng((self.seed, i))
        stratum = STRATA[rng.integers(len(STRATA))]
        pool = self.index.rows[stratum]
        fi, ti, start, h, qi = pool[rng.integers(len(pool))]

        f = self._h5(int(fi))
        tname = self.index.trajs[int(fi)][int(ti)]
        g = f[tname]
        d = g["horizon_start"][str(int(start))][f"horizon_len_{int(h)}"]
        i0, i1 = int(d["start_idx"][()]), int(d["end_idx"][()])
        if self.teacher_files is not None:
            p = self._t5(int(fi))[tname]["horizon_start"][str(int(start))][
                f"horizon_len_{int(h)}"]["p_yes"][int(qi)]
            answer_yes = bool(p >= 0.5)
        else:
            answer_yes = bool(d["answers"][()][int(qi)])
        return {
            "image": np.asarray(g["frames"][i0], dtype=np.uint8),
            "actions": torch.as_tensor(np.asarray(g["actions"][i0:i1]), dtype=torch.float32),
            "question": _s(d["questions"][()][int(qi)]),
            "answer": "yes" if answer_yes else "no",
            "qtype": stratum[0],
        }


def make_collator(processor):
    """Batch via their processor. `suffix=` masks the prefix so cross-entropy
    lands on the answer tokens only (paper 3.2)."""

    def collate(batch: list[dict]) -> dict:
        inputs = processor(
            text=[b["question"] for b in batch],
            images=[b["image"] for b in batch],
            actions=[b["actions"] for b in batch],
            suffix=[b["answer"] for b in batch],
            return_tensors="pt", padding="longest",
        )
        out = dict(inputs)
        out["pixel_values"] = out["pixel_values"].to(torch.float32)
        if "action_values" in out:
            out["action_values"] = out["action_values"].to(torch.float32)
        return out

    return collate
