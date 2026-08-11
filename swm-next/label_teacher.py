"""Teacher labeling pass: frozen Qwen3-VL answers every SAQA question.

For each (traj, start, horizon) group in the corrected HDF5s, the judge sees
the REAL future frame (frames[end_idx]) and each question, and emits
P(yes) — the softmax mass on yes-token variants renormalized over {yes, no}
(the July judge convention, 86.6-86.8% oracle agreement on the old data).

Output: a sidecar HDF5 per input file (<stem>.teacher.hdf5) mirroring the
group structure, with one float16 dataset `p_yes` per horizon group, aligned
index-for-index with `questions`. Soft values are stored even though the
first training arm uses argmax — this enables soft-label arms later without
relabeling.

Per-trajectory atomic writes; rerunning skips completed trajectories, so a
killed job resumes for free. Shardable by trajectory for multi-node runs.

    python label_teacher.py --hdf5 noisy.hdf5 --out noisy.teacher.hdf5 \
        [--shard 0/4] [--batch 96] [--limit-trajs 2]

Prints running oracle-agreement (overall and per question type) — gate G-T1.
"""
from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict

import h5py
import numpy as np
import torch


def _s(x) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


class QwenJudge:
    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Instruct", device="cuda"):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.device = device
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.processor = AutoProcessor.from_pretrained(model_id)
        tok = self.processor.tokenizer
        def singles(cands):
            return [tok(c, add_special_tokens=False).input_ids[0]
                    for c in cands
                    if len(tok(c, add_special_tokens=False).input_ids) == 1]
        self.yes_ids = singles([" Yes", " yes", "Yes", "yes"])
        self.no_ids = singles([" No", " no", "No", "no"])

    @torch.no_grad()
    def p_yes(self, images, questions):
        """images: list of np uint8 HWC frames; questions: list[str] (same
        length). One forward, batched. Returns (p_yes, mass), both (B,)
        float32: p_yes is P(yes) RENORMALIZED over {yes, no} (the training
        label); mass is the UNNORMALIZED P(yes)+P(no) — how much of the full
        vocab distribution the two answer classes actually capture
        (visibility only; ~1.0 means the prompt fully constrained the
        model, low values flag answers leaking into other tokens)."""
        texts = []
        for q in questions:
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f"{q} Answer with one word: yes or no."},
            ]}]
            texts.append(self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False))
        inputs = self.processor(text=texts, images=list(images),
                                return_tensors="pt", padding=True).to(self.device)
        out = self.model(**inputs)
        # Answer position = each sequence's last real token (left/right padding
        # both handled by attention_mask).
        mask = inputs["attention_mask"]
        last = mask.sum(dim=1) - 1
        idx = torch.arange(out.logits.shape[0], device=self.device)
        logits = out.logits[idx, last, :].float()
        probs = torch.softmax(logits, dim=-1)
        yes = probs[:, self.yes_ids].sum(dim=-1)
        no = probs[:, self.no_ids].sum(dim=-1)
        mass = yes + no
        return ((yes / mass.clamp_min(1e-12)).cpu().numpy(),
                mass.cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--shard", default="0/1", help="i/n over sorted trajectories")
    ap.add_argument("--limit-trajs", type=int, default=0, help="smoke: stop after N trajs")
    ap.add_argument("--device", default="cuda", help="cuda | mps | cpu (mps/cpu for local stepping)")
    args = ap.parse_args()

    si, sn = (int(x) for x in args.shard.split("/"))
    judge = QwenJudge(model_id=args.model, device=args.device)

    agree = defaultdict(lambda: [0, 0])   # qtype -> [agree, total]
    mass_all: list[np.ndarray] = []       # unnormalized P(yes)+P(no) per question
    n_q = 0
    t0 = time.time()

    with h5py.File(args.hdf5, "r") as f, h5py.File(args.out, "a") as out:
        names = sorted(f.keys(), key=lambda n: int(n.split("_")[-1]))
        names = [n for i, n in enumerate(names) if i % sn == si]
        if args.limit_trajs:
            names = names[: args.limit_trajs]
        for done_i, tname in enumerate(names):
            if tname in out and out[tname].attrs.get("complete", False):
                continue
            g = f[tname]
            frames = g["frames"]
            # Gather this trajectory's full question list, batched across
            # horizon groups but reusing each future frame for all its
            # questions (the processor re-encodes per row; frame reuse still
            # saves HDF5 reads).
            rows = []      # (start, hname, qi, frame_idx, question, oracle, qtype)
            hs = g["horizon_start"]
            for start in hs:
                sg = hs[start]
                for hname in sg:
                    d = sg[hname]
                    i1 = int(d["end_idx"][()])
                    qs = d["questions"][()]
                    ans = d["answers"][()]
                    typs = d["types"][()]
                    for qi in range(len(qs)):
                        rows.append((start, hname, qi, i1, _s(qs[qi]),
                                     bool(ans[qi]), _s(typs[qi])))
            frame_cache: dict[int, np.ndarray] = {}
            preds = np.zeros(len(rows), dtype=np.float16)
            masses = np.zeros(len(rows), dtype=np.float16)
            for b0 in range(0, len(rows), args.batch):
                chunk = rows[b0 : b0 + args.batch]
                imgs = []
                for _, _, _, fi_, _, _, _ in chunk:
                    if fi_ not in frame_cache:
                        frame_cache[fi_] = np.asarray(frames[fi_], dtype=np.uint8)
                    imgs.append(frame_cache[fi_])
                p, m = judge.p_yes(imgs, [c[4] for c in chunk])
                preds[b0 : b0 + len(chunk)] = p.astype(np.float16)
                masses[b0 : b0 + len(chunk)] = m.astype(np.float16)
                for c, pv in zip(chunk, p):
                    agree[c[6]][1] += 1
                    agree[c[6]][0] += int((pv >= 0.5) == c[5])
                n_q += len(chunk)
                mass_all.append(m)

            # Atomic-ish per-traj write: fill all horizon groups, then mark complete.
            tgrp = out.require_group(tname)
            by_group = defaultdict(list)
            for r_i, (start, hname, qi, _, _, _, _) in enumerate(rows):
                by_group[(start, hname)].append((qi, preds[r_i], masses[r_i]))
            for (start, hname), vals in by_group.items():
                n_vals = max(v[0] for v in vals) + 1
                arr = np.zeros(n_vals, dtype=np.float16)
                marr = np.zeros(n_vals, dtype=np.float16)
                for qi, pv, mv in vals:
                    arr[qi] = pv
                    marr[qi] = mv
                dgrp = tgrp.require_group(f"horizon_start/{start}/{hname}")
                for name, data in (("p_yes", arr), ("yn_mass", marr)):
                    if name in dgrp:
                        del dgrp[name]
                    dgrp.create_dataset(name, data=data)
            tgrp.attrs["complete"] = True
            out.flush()

            if (done_i + 1) % 5 == 0 or done_i == len(names) - 1:
                rate = n_q / max(time.time() - t0, 1e-9)
                tot_a = sum(v[0] for v in agree.values())
                tot_n = sum(v[1] for v in agree.values())
                print(f"[{done_i+1}/{len(names)} trajs] {n_q:,} q  {rate:.0f} q/s  "
                      f"agreement {100*tot_a/max(1,tot_n):.2f}%", flush=True)

    print("=== G-T1: teacher-vs-oracle agreement (this shard) ===")
    for t, (a, n) in sorted(agree.items()):
        print(f"  {t:28s} {100*a/max(1,n):6.2f}%  (n={n:,})")
    tot_a = sum(v[0] for v in agree.values())
    tot_n = sum(v[1] for v in agree.values())
    print(f"  {'OVERALL':28s} {100*tot_a/max(1,tot_n):6.2f}%  (n={tot_n:,})")
    if mass_all:
        m = np.concatenate(mass_all)
        print(f"=== G-T2: yes/no mass capture (unnormalized P(yes)+P(no)) ===")
        print(f"  mean {m.mean():.4f}  q01/q10/q50 "
              f"{np.quantile(m, .01):.4f}/{np.quantile(m, .10):.4f}/{np.quantile(m, .50):.4f}  "
              f"min {m.min():.4f}  frac<0.9: {100*(m < 0.9).mean():.2f}%")
    print("LABELING_SHARD_DONE", flush=True)


if __name__ == "__main__":
    main()
