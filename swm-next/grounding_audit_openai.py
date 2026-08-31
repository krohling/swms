"""Sol/OpenAI variant of the grounding audit: same frames, boxes via API,
scored against pose-projected ground truth inline."""
import argparse, base64, io, json, os, re, sys
import h5py
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "ogbench"))


def data_url(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact224", required=True)
    ap.add_argument("--artifact768", required=True)
    ap.add_argument("--colors", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--out", default="grounding_audit_sol.json")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    from replay_extract_poses import make_env
    env = make_env(); env.reset(seed=0)
    P = env.unwrapped.get_camera_matrices()
    poses = np.load(args.poses)
    colors_map = json.load(open(args.colors))

    f224 = h5py.File(args.artifact224, "r")
    f768 = h5py.File(args.artifact768, "r")
    seen, picks = set(), []
    for i in range(f224["provenance"].shape[0]):
        pr = f224["provenance"][i].decode()
        key = (pr.split(":")[0], pr.split(":")[-1])
        if key not in seen:
            seen.add(key); picks.append(i)
        if len(picks) >= args.n:
            break

    def ask(frame, color):
        H = frame.shape[0]
        r = client.chat.completions.create(model=args.model, messages=[{
            "role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url(frame), "detail": "high"}},
                {"type": "text", "text":
                 f"The image is {H}x{H} pixels. Locate the {color} cube. "
                 "Respond with only JSON, pixel coordinates: "
                 "{\"bbox_2d\": [x1, y1, x2, y2]}"}]},
        ], max_completion_tokens=64, reasoning_effort="none")
        m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
                      r.choices[0].message.content or "")
        if not m:
            return None
        box = [int(x) for x in m.groups()]
        if max(box) > H:
            box = [b * H / 1000.0 for b in box]
        return box

    jobs = []
    for i in picks:
        pr = f224["provenance"][i].decode()
        tname = pr.split(":")[0]
        i1 = int(pr.split(":")[-1])
        ti = tname.split("_")[-1]
        for res, fh in (("224", f224), ("768", f768)):
            frame = np.asarray(fh["end_frames"][i])
            for color in colors_map[ti]:
                jobs.append((res, frame, color, tname, i1))

    def run(job):
        res, frame, color, tname, i1 = job
        try:
            box = ask(frame, color)
        except Exception as e:
            return (res, color, tname, i1, None, str(e)[:60])
        return (res, color, tname, i1, box, None)

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(run, jobs))
    print(f"api calls done: {len(results)}", flush=True)

    out = {}
    records = {"224": [], "768": []}
    for res in ("224", "768"):
        scale = int(res) / 224.0
        n = det = hit = 0
        offs = []
        for r_res, color, tname, i1, box, err in results:
            if r_res != res:
                continue
            n += 1
            ti = tname.split("_")[-1]
            records[res].append(dict(color=color, tname=tname, i1=i1, box=box, err=err))
            if box is None:
                continue
            det += 1
            b = colors_map[ti].index(color)
            p3 = np.append(poses[f"{tname}_block_{b}"][i1], 1.0)
            clip = P @ p3
            gx, gy = float(clip[0]/clip[2]) * scale, float(clip[1]/clip[2]) * scale
            x1, y1, x2, y2 = box
            offs.append(float(np.hypot((x1+x2)/2-gx, (y1+y2)/2-gy)))
            hit += int((x1-2 <= gx <= x2+2) and (y1-2 <= gy <= y2+2))
        out[res] = dict(detect=round(det/max(n,1), 3), contain=round(hit/max(det,1), 3),
                        median_off_cubewidths=round(float(np.median(offs))/(15*scale), 2) if offs else -1)
    json.dump(dict(summary=out, records=records), open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("AUDIT_DONE")


if __name__ == "__main__":
    main()
