"""Gemini Batch API driver for the labeler-eval protocols.

Interactive Tier-1 quota for gemini-3.1-pro-preview is 250 requests/day;
the Batch API has a separate quota (5M enqueued tokens) and 50% pricing,
so every Gemini protocol run goes through here instead:

  audit    -- bounding-box accuracy vs pose-projected GT (n frames, 224+768)
  crops    -- crop->answer on cached Sol boxes (balanced 3k)
  zeroshot -- full-frame 0-shot answering (balanced or natural artifact)

Requests are built locally into JSONL waves (<= --wave requests each, well
under the enqueue cap), uploaded, polled, and scored with the same rules as
the interactive harnesses (run_crop_reask / run_saqa_eval /
grounding_audit_openai). Parsed-answer scoring: p_yes in {0,1}, mass=1 when
a yes/no parses, 0 otherwise. State is persisted after each submit so a
killed driver resumes server-side jobs instead of re-paying for them.

    python swm-next/labeler_eval/gemini_batch.py --stage crops \
        --artifact ../data/saqa_eval_12k.h5 \
        --cache labeler_eval_results_v2_cosmos224/ground_cache_sol.json \
        --box-scale 3.4285714 --limit 3000 --out gemini_batch_crops

Gemini box convention (native, requested explicitly): [ymin, xmin, ymax,
xmax] normalized 0-1000; asking for x1y1x2y2 pixels gets silently answered
in the native convention anyway, so we embrace it and convert ourselves.
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

from labeler_backends import PAIR_PREAMBLE, CLOSER
from run_crop_reask import parse_entities, union_crop

MODEL = "gemini-3.1-pro-preview"
GEN_CONFIG = {"maxOutputTokens": 256,
              "thinkingConfig": {"thinkingBudget": 128, "includeThoughts": False}}
BOX_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")


def b64_png(arr) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def req(key: str, images: list, text: str) -> dict:
    parts = [{"inlineData": {"mimeType": "image/png", "data": b64_png(im)}}
             for im in images]
    parts.append({"text": text})
    return {"key": key, "request": {"contents": [{"parts": parts}],
                                    "generationConfig": GEN_CONFIG}}


# ---------------------------------------------------------------- builders

def build_audit(args):
    from grounding_audit_openai import make_env  # noqa: F401 (env for scoring)
    f224 = h5py.File(args.artifact, "r")
    f768 = h5py.File(args.artifact768, "r")
    colors_map = json.load(open(args.colors))
    seen, picks = set(), []
    for i in range(f224["provenance"].shape[0]):
        pr = f224["provenance"][i].decode()
        key = (pr.split(":")[0], pr.split(":")[-1])
        if key not in seen:
            seen.add(key); picks.append(i)
        if len(picks) >= args.n:
            break
    out = []
    for i in picks:
        pr = f224["provenance"][i].decode()
        tname, i1 = pr.split(":")[0], int(pr.split(":")[-1])
        ti = tname.split("_")[-1]
        for res, fh in (("224", f224), ("768", f768)):
            frame = np.asarray(fh["end_frames"][i])
            for color in colors_map[ti]:
                text = (f"Locate the {color} cube. Respond with only JSON: "
                        "{\"box_2d\": [ymin, xmin, ymax, xmax]} "
                        "with coordinates normalized to 0-1000.")
                out.append(req(f"{res}|{color}|{tname}|{i1}", [frame], text))
    return out


def score_audit(args, responses: dict):
    from grounding_audit_openai import make_env
    env = make_env(); env.reset(seed=0)
    P = env.unwrapped.get_camera_matrices()
    poses = np.load(args.poses)
    colors_map = json.load(open(args.colors))
    out, records = {}, {"224": [], "768": []}
    for res in ("224", "768"):
        H = int(res)
        scale = H / 224.0
        n = det = hit = 0
        offs = []
        for key, text in responses.items():
            r_res, color, tname, i1 = key.split("|")
            if r_res != res:
                continue
            n += 1
            m = BOX_RE.search(text or "")
            box = None
            if m:
                ymin, xmin, ymax, xmax = [int(v) * H / 1000.0 for v in m.groups()]
                box = [xmin, ymin, xmax, ymax]
            records[res].append(dict(color=color, tname=tname, i1=int(i1), box=box))
            if box is None:
                continue
            det += 1
            ti = tname.split("_")[-1]
            b = colors_map[ti].index(color)
            p3 = np.append(poses[f"{tname}_block_{b}"][int(i1)], 1.0)
            clip = P @ p3
            gx, gy = float(clip[0] / clip[2]) * scale, float(clip[1] / clip[2]) * scale
            x1, y1, x2, y2 = box
            offs.append(float(np.hypot((x1 + x2) / 2 - gx, (y1 + y2) / 2 - gy)))
            hit += int((x1 - 2 <= gx <= x2 + 2) and (y1 - 2 <= gy <= y2 + 2))
        out[res] = dict(detect=round(det / max(n, 1), 3),
                        contain=round(hit / max(det, 1), 3),
                        median_off_cubewidths=round(float(np.median(offs)) / (15 * scale), 2) if offs else -1)
    json.dump(dict(summary=out, records=records),
              open(os.path.join(args.out, "grounding_audit.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def _answer_items(args, cropped: bool):
    """Yield (key, images, text) for answering stages."""
    f = h5py.File(args.artifact, "r")
    n = min(args.limit or f["oracle_yes"].shape[0], f["oracle_yes"].shape[0])
    cache = json.load(open(args.cache)) if cropped else None
    out, skipped = [], 0
    for i in range(n):
        q = f["questions"][i].decode()
        qtype = f["qtypes"][i].decode()
        pair = qtype in CLOSER
        start, end = np.asarray(f["start_frames"][i]), np.asarray(f["end_frames"][i])
        if cropped:
            tname, i0, i1 = f["provenance"][i].decode().split(":")
            ents = parse_entities(q, qtype)
            def get(fr, e):
                b = cache.get(f"{tname}:{fr}:{e}")
                return [v / args.box_scale for v in b] if (b and args.box_scale != 1.0) else b
            boxes = [get(i1, e) for e in ents]
            if not all(boxes):
                skipped += 1
                continue
            end = union_crop(end, boxes, args.margin, args.crop_size)
            if pair:
                bs = [get(i0, e) for e in ents]
                if not all(bs):
                    skipped += 1
                    continue
                start = union_crop(np.asarray(f["start_frames"][i]), bs,
                                   args.margin, args.crop_size)
        if pair:
            text = f"{PAIR_PREAMBLE}{q} Answer with one word: yes or no."
            images = [start, end]
        else:
            text = f"{q} Answer with one word: yes or no."
            images = [end]
        out.append(req(str(i), images, text))
    if skipped:
        print(f"[build] {skipped} questions skipped (missing boxes)", flush=True)
    return out


def score_answers(args, responses: dict):
    f = h5py.File(args.artifact, "r")
    per = {}
    p_yes = {}
    for key, text in responses.items():
        i = int(key)
        t = (text or "").strip().lower()
        if t.startswith("yes"):
            p = 1.0
        elif t.startswith("no"):
            p = 0.0
        else:
            p = None   # unparsed: excluded, reported
        p_yes[i] = p
        qtype = f["qtypes"][i].decode()
        oracle = bool(f["oracle_yes"][i])
        d = per.setdefault(qtype, dict(n=0, correct=0, unparsed=0))
        d["n"] += 1
        if p is None:
            d["unparsed"] += 1
        else:
            d["correct"] += int((p >= 0.5) == oracle)
    total_n = sum(d["n"] for d in per.values())
    total_c = sum(d["correct"] for d in per.values())
    total_u = sum(d["unparsed"] for d in per.values())
    report = dict(model=MODEL, mode=args.stage, n=total_n,
                  overall_acc=round(total_c / max(total_n, 1), 4),
                  unparsed_frac=round(total_u / max(total_n, 1), 4),
                  per_type={t: round(d["correct"] / max(d["n"], 1), 4)
                            for t, d in sorted(per.items())})
    json.dump(dict(report=report, p_yes=p_yes),
              open(os.path.join(args.out, "scores.json"), "w"))
    print(json.dumps(report, indent=1))


# ---------------------------------------------------------------- driver

def run_waves(args, requests: list):
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    state_path = os.path.join(args.out, "batch_state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"jobs": {}}
    waves = [requests[i:i + args.wave] for i in range(0, len(requests), args.wave)]
    print(f"[batch] {len(requests)} requests in {len(waves)} wave(s)", flush=True)
    responses = {}
    for wi, wave in enumerate(waves):
        wname = f"wave{wi}"
        if state["jobs"].get(wname, {}).get("done"):
            responses.update(state["jobs"][wname]["responses"])
            continue
        job_name = state["jobs"].get(wname, {}).get("job")
        if job_name is None:
            jsonl = os.path.join(args.out, f"{wname}.jsonl")
            with open(jsonl, "w") as fh:
                for r in wave:
                    fh.write(json.dumps(r) + "\n")
            up = client.files.upload(
                file=jsonl, config={"mime_type": "application/jsonl",
                                    "display_name": f"{args.stage}-{wname}"})
            job = client.batches.create(
                model=MODEL, src=up.name,
                config={"display_name": f"labeler-{args.stage}-{wname}"})
            job_name = job.name
            state["jobs"][wname] = {"job": job_name}
            json.dump(state, open(state_path, "w"))
            print(f"[batch] {wname}: submitted {len(wave)} reqs as {job_name}", flush=True)
        while True:
            job = client.batches.get(name=job_name)
            s = str(job.state)
            if "SUCCEEDED" in s:
                break
            if any(t in s for t in ("FAILED", "CANCELLED", "EXPIRED")):
                raise RuntimeError(f"{wname} ended {s}: {getattr(job, 'error', '')}")
            print(f"[batch] {wname}: {s}", flush=True)
            time.sleep(args.poll)
        blob = client.files.download(file=job.dest.file_name)
        wave_resp = {}
        for line in blob.decode().splitlines():
            d = json.loads(line)
            try:
                text = d["response"]["candidates"][0]["content"]["parts"][-1]["text"]
            except (KeyError, IndexError, TypeError):
                text = ""
            wave_resp[d["key"]] = text
        state["jobs"][wname] = {"job": job_name, "done": True, "responses": wave_resp}
        json.dump(state, open(state_path, "w"))
        responses.update(wave_resp)
        print(f"[batch] {wname}: done ({len(wave_resp)} responses)", flush=True)
    return responses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["audit", "crops", "zeroshot"])
    ap.add_argument("--artifact", required=True, help="224 artifact")
    ap.add_argument("--artifact768", help="768 artifact (audit only)")
    ap.add_argument("--colors", help="colors json (audit only)")
    ap.add_argument("--poses", help="poses npz (audit only)")
    ap.add_argument("--n", type=int, default=100, help="audit frames")
    ap.add_argument("--cache", help="grounding cache (crops only)")
    ap.add_argument("--box-scale", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--crop-size", type=int, default=448)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--wave", type=int, default=1500)
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.stage == "audit":
        requests = build_audit(args)
    else:
        requests = _answer_items(args, cropped=(args.stage == "crops"))
    responses = run_waves(args, requests)
    if args.stage == "audit":
        score_audit(args, responses)
    else:
        score_answers(args, responses)
    print(f"{args.stage.upper()}_DONE", flush=True)


if __name__ == "__main__":
    main()
