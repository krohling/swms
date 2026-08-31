"""Grounding audit: can Qwen3-VL localize the cubes well enough to support
geometry-in-code labeling scaffolds?

Ground truth comes from HSV color segmentation of the rendered frames (cube
colors are saturated and unique per scene; the largest connected blob per
color is the cube). Qwen3-VL is prompted for a bounding box per named cube;
we score detection rate, center-in-blob rate, center offset, and IoU vs the
HSV bbox, at both 224 and 768 for the same items.

    python swm-next/grounding_audit.py --artifact224 saqa_eval_12k.h5 \
        --artifact768 saqa_eval_12k_768.h5 --colors noisy-fixed.poses.colors.json \
        --n 100 [--device mps]
"""
from __future__ import annotations

import argparse
import json
import re

import h5py
import numpy as np
import torch

# HSV hue centers (OpenCV 0-179 scale) for the cube palette.
HUE = {"red": 0, "orange": 12, "green": 60, "blue": 105}
SAT_MIN, VAL_MIN = 90, 60


def hsv_gt(frame, color):
    import cv2
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1], hsv[..., 2]
    hue = HUE[color]
    dh = np.minimum(np.abs(h - hue), 180 - np.abs(h - hue))
    tol = 8 if color in ("red", "orange") else 15
    mask = ((dh <= tol) & (s >= SAT_MIN) & (v >= VAL_MIN)).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    i = 1 + int(np.argmax(areas))
    if stats[i, cv2.CC_STAT_AREA] < (frame.shape[0] / 45) ** 2:
        return None
    x, y, w, hgt = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    return dict(bbox=(x, y, x + w, y + hgt), centroid=tuple(cents[i]),
                mask=labels == i)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact224", required=True)
    ap.add_argument("--artifact768", required=True)
    ap.add_argument("--colors", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="grounding_audit.json")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16,
        low_cpu_mem_usage=True).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    colors_map = json.load(open(args.colors))

    f224 = h5py.File(args.artifact224, "r")
    f768 = h5py.File(args.artifact768, "r")
    # distinct (traj, end-frame) items, deterministic sample
    seen, picks = set(), []
    for i in range(f224["provenance"].shape[0]):
        key = f224["provenance"][i].decode().rsplit(":", 2)[0], \
              f224["provenance"][i].decode().rsplit(":", 1)[-1]
        if key not in seen:
            seen.add(key)
            picks.append(i)
        if len(picks) >= args.n:
            break

    def ask_box(frame, color):
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text":
             f"Locate the {color} cube in the image. Respond with only its "
             "bounding box as JSON: {\"bbox_2d\": [x1, y1, x2, y2]}"}]}]
        text = processor.apply_chat_template(msgs, add_generation_prompt=True,
                                             tokenize=False)
        inputs = processor(text=[text], images=[[frame]],
                           return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
        resp = processor.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", resp)
        if not m:
            return None
        box = [int(x) for x in m.groups()]
        # Qwen3-VL grounding coordinates are normalized to 0-1000; rescale to
        # pixels (heuristic keeps compatibility if a model emits absolute).
        H = frame.shape[0]
        if max(box) > H:
            box = [b * H / 1000.0 for b in box]
        return tuple(box)

    results = {"224": [], "768": []}
    for j, i in enumerate(picks):
        tname = f224["provenance"][i].decode().split(":")[0]
        ti = tname.split("_")[-1]
        present = [c for c in colors_map[ti] if c in HUE]
        for res, fh in (("224", f224), ("768", f768)):
            frame = np.asarray(fh["end_frames"][i])
            for color in present:
                gt = hsv_gt(frame, color)
                if gt is None:
                    continue
                box = ask_box(frame, color)
                rec = dict(idx=int(i), color=color, res=res, detected=box is not None, box=box)
                if box is not None:
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    H = frame.shape[0]
                    inb = (0 <= int(cy) < H and 0 <= int(cx) < H
                           and bool(gt["mask"][int(cy), int(cx)]))
                    rec.update(center_in_blob=inb,
                               center_off_px=float(np.hypot(cx - gt["centroid"][0],
                                                            cy - gt["centroid"][1])),
                               iou=iou(box, gt["bbox"]))
                results[res].append(rec)
        if (j + 1) % 20 == 0:
            print(f"audited {j + 1}/{len(picks)} frames", flush=True)

    report = {}
    for res, recs in results.items():
        det = [r for r in recs if r["detected"]]
        report[res] = dict(
            n=len(recs), detect_rate=len(det) / max(len(recs), 1),
            center_in_blob=float(np.mean([r["center_in_blob"] for r in det])) if det else 0,
            median_center_off_px=float(np.median([r["center_off_px"] for r in det])) if det else -1,
            mean_iou=float(np.mean([r["iou"] for r in det])) if det else 0)
    json.dump(dict(summary=report, records=results), open(args.out, "w"), indent=2)
    print(json.dumps(report, indent=2))
    print("AUDIT_DONE")


if __name__ == "__main__":
    main()
