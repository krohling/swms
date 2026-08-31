"""V2 scaffold: ground -> crop -> re-ask (the zoom hypothesis, portable).

Pipeline per question, strictly image-based (no sim state anywhere in the
label path; poses are used only by the offline audits):

  1. GROUND: a grounding provider returns a box per named entity on the
     question's frame(s). Provider is pluggable; the audited reference is
     GPT-5.6 Sol on 768 frames (99% correct-box). Boxes are cached to disk
     keyed by (traj, frame, entity) so repeated questions/entities are free
     and reruns never re-pay grounding.
  2. CROP: union of the entity boxes, padded by --margin cube-widths (scale
     taken from the median entity box size in the image itself -- relative
     units, no metric calibration), squared, then upscaled to --crop-size.
     Closer questions crop BOTH frames with the SAME union region (union
     across frames) so the pair stays spatially comparable.
  3. RE-ASK: the original question on the crop(s), answered by a standard
     labeler backend (pooled p_yes readout). Answerer is independent of the
     grounding provider.

Failure handling: if any entity fails to ground, fall back to the full frame
and flag the question (reported separately -- fallback accuracy is the
no-scaffold baseline, so this degrades gracefully rather than erroring).

    python swm-next/labeler_eval/run_crop_reask.py \
        --artifact saqa_eval_12k_768.h5 --grounder sol --answerer cosmos-reason2-8b \
        [--limit 0] [--margin 1.0] [--crop-size 448] [--preview N]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time

import h5py
import numpy as np
from PIL import Image

from labeler_backends import BACKENDS, CLOSER, PAIR_PREAMBLE

ENTITY_RE = re.compile(r"(red|green|blue|yellow|orange|purple) cube|robot(?:ic)? (?:gripper|peg)|gripper")


def parse_entities(question: str, qtype: str) -> list[str]:
    """Named entities the crop must contain, in question order."""
    ents = []
    for m in ENTITY_RE.finditer(question.lower()):
        ent = m.group(0)
        ent = "gripper" if "gripper" in ent or "peg" in ent else ent
        if ent not in ents:
            ents.append(ent)
    if qtype in ("cube_grasped", "gripper_block_closer", "block_gripper_touching") \
            and "gripper" not in ents:
        ents.append("gripper")   # implied by the predicate even if unnamed
    return ents


class SolGrounder:
    """GPT-5.6 Sol boxes with a persistent on-disk cache."""

    def __init__(self, cache_path: str, model_id: str = "gpt-5.6-sol"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model_id = model_id
        self.cache_path = cache_path
        self.cache = {}
        if os.path.exists(cache_path):
            self.cache = json.load(open(cache_path))
        self._dirty = 0

    @staticmethod
    def _data_url(arr):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def box(self, frame: np.ndarray, entity: str, key: str):
        if key in self.cache:
            return self.cache[key]
        H = frame.shape[0]
        target = f"the {entity}" if entity != "gripper" else \
            "the robot gripper (the purple/magenta end-effector)"
        r = self.client.chat.completions.create(
            model=self.model_id, messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": self._data_url(frame), "detail": "high"}},
                {"type": "text", "text":
                 f"The image is {H}x{H} pixels. Locate {target}. Respond with "
                 "only JSON, pixel coordinates: {\"bbox_2d\": [x1, y1, x2, y2]}"}]}],
            max_completion_tokens=64, reasoning_effort="none")
        m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
                      r.choices[0].message.content or "")
        box = None
        if m:
            box = [int(x) for x in m.groups()]
            if max(box) > H:
                box = [b * H / 1000.0 for b in box]
        self.cache[key] = box
        self._dirty += 1
        if self._dirty >= 50:
            self.flush()
        return box

    def flush(self):
        json.dump(self.cache, open(self.cache_path, "w"))
        self._dirty = 0


GROUNDERS = {"sol": SolGrounder}


def union_crop(frame: np.ndarray, boxes: list, margin_cw: float, crop_size: int):
    """Square crop covering all boxes, padded by margin in cube-widths
    (cube width estimated from the median box size -- image-derived scale)."""
    H = frame.shape[0]
    x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
    sizes = [max(b[2] - b[0], b[3] - b[1]) for b in boxes]
    cw = float(np.median(sizes))
    pad = margin_cw * cw
    x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x1, y1 = cx - side / 2, cy - side / 2
    x1 = max(0.0, min(x1, H - side)); y1 = max(0.0, min(y1, H - side))
    side = min(side, H)
    crop = frame[int(y1):int(y1 + side), int(x1):int(x1 + side)]
    return np.asarray(Image.fromarray(crop).resize((crop_size, crop_size),
                                                   Image.BICUBIC))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--grounder", default="sol", choices=sorted(GROUNDERS))
    ap.add_argument("--answerer", required=True, choices=sorted(BACKENDS))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--crop-size", type=int, default=448)
    ap.add_argument("--preview", type=int, default=0,
                    help="ground+crop N questions, write crop cards, answer NOTHING")
    ap.add_argument("--out", default="labeler_eval_results_v2")
    args = ap.parse_args()

    f = h5py.File(args.artifact, "r")
    n_all = f["oracle_yes"].shape[0]
    n = min(args.limit, n_all) if args.limit else n_all
    if args.preview:
        n = min(args.preview, n_all)

    os.makedirs(args.out, exist_ok=True)
    grounder = GROUNDERS[args.grounder](
        os.path.join(args.out, f"ground_cache_{args.grounder}.json"))

    items, flags = [], []
    for i in range(n):
        q = f["questions"][i].decode()
        qtype = f["qtypes"][i].decode()
        prov = f["provenance"][i].decode()
        tname, i0, i1 = prov.split(":")
        ents = parse_entities(q, qtype)
        start = np.asarray(f["start_frames"][i])
        end = np.asarray(f["end_frames"][i])
        pair = qtype in CLOSER
        boxes_end = [grounder.box(end, e, f"{tname}:{i1}:{e}") for e in ents]
        boxes = [b for b in boxes_end if b]
        if pair:
            boxes_start = [grounder.box(start, e, f"{tname}:{i0}:{e}") for e in ents]
            boxes += [b for b in boxes_start if b]
        ok = len([b for b in boxes_end if b]) == len(ents) and \
            (not pair or len([b for b in boxes_start if b]) == len(ents))
        if ok:
            crop_end = union_crop(end, boxes, args.margin, args.crop_size)
            crop_start = union_crop(start, boxes, args.margin, args.crop_size) if pair else crop_end
        else:
            crop_end, crop_start = end, start   # full-frame fallback, flagged
        flags.append(ok)
        items.append(dict(question=q, qtype=qtype, oracle=bool(f["oracle_yes"][i]),
                          start=crop_start, end=crop_end))
        if (i + 1) % 200 == 0:
            grounder.flush()
            print(f"grounded {i + 1}/{n} (fallbacks so far: {flags.count(False)})",
                  flush=True)
    grounder.flush()

    if args.preview:
        for k, it in enumerate(items):
            a = Image.fromarray(it["start"]); b = Image.fromarray(it["end"])
            card = Image.new("RGB", (a.width + b.width + 12, a.height + 40), "white")
            card.paste(a, (4, 4)); card.paste(b, (a.width + 8, 4))
            from PIL import ImageDraw
            dr = ImageDraw.Draw(card)
            dr.text((4, a.height + 8),
                    f"{it['qtype']} | {it['question']} | grounded={flags[k]}",
                    fill="black")
            card.save(os.path.join(args.out, f"preview_{k:02d}.png"))
        print(f"PREVIEW_DONE: {sum(flags)}/{len(flags)} grounded, cards in {args.out}")
        return

    backend = BACKENDS[args.answerer](args.device)
    p = np.zeros(n); mass = np.zeros(n)
    t0 = time.time()
    for b0 in range(0, n, args.batch):
        chunk = items[b0:b0 + args.batch]
        pp, mm = backend.score(chunk)
        p[b0:b0 + len(chunk)] = pp
        mass[b0:b0 + len(chunk)] = mm
        if (b0 // args.batch) % 25 == 0:
            print(f"[v2:{args.answerer}] {b0 + len(chunk)}/{n} "
                  f"{(b0 + len(chunk)) / max(time.time() - t0, 1e-9):.1f} q/s", flush=True)

    oracle = np.array([it["oracle"] for it in items])
    qtypes = np.array([it["qtype"] for it in items])
    ok = np.array(flags)
    correct = (p >= 0.5) == oracle
    report = dict(variant="v2_crop_reask", grounder=args.grounder,
                  answerer=args.answerer, portable=True, n=n,
                  margin_cw=args.margin, crop_size=args.crop_size,
                  grounded_frac=float(ok.mean()),
                  overall_acc=float(correct.mean()),
                  overall_acc_grounded_only=float(correct[ok].mean()) if ok.any() else None,
                  mean_mass=float(mass.mean()), per_type={}, per_type_grounded={})
    for qt in sorted(set(qtypes)):
        m = qtypes == qt
        report["per_type"][qt] = float(correct[m].mean())
        if (m & ok).any():
            report["per_type_grounded"][qt] = float(correct[m & ok].mean())
    base = os.path.join(args.out, f"v2_{args.grounder}_{args.answerer}")
    json.dump(report, open(base + ".json", "w"), indent=2)
    np.savez_compressed(base + "_scores.npz", p_yes=p, mass=mass, grounded=ok)
    print(json.dumps({k: v for k, v in report.items() if "per_type" not in k}, indent=2))
    for qt, a in report["per_type"].items():
        print(f"  {qt:26s} {a:.3f}  (grounded-only: {report['per_type_grounded'].get(qt, float('nan')):.3f})")


if __name__ == "__main__":
    main()
