"""Label identity check for judge_server_mb.py: does an item get the same
p(yes) when it shares a forward with other items as when it is alone?

Reads frames from a datagen_lt.py file (a dry run is fine: only the frames
and the two question texts are used) and sends each item three times to a
server that NOTHING ELSE is using:

  pass A  one request at a time           -> every forward holds 1 item
  pass B  all items at once, from threads -> forwards hold up to --batch items
  pass C  one request at a time again     -> the run-to-run noise at 1 item

Prints the A-vs-B and A-vs-C differences per question kind and the server's
mean batch in pass B (from GET /stats).

What to expect (measured Nov 26 on an A40, 370 items of three smoke units):
the judge runs in bfloat16, so its yes/no logits come in steps of 1/16 to
1/8, and a forward of 8 items rounds some of them one step differently than
a forward of 1. Alone vs in a batch: logit difference mean abs 0.10-0.14,
max 0.5-0.6, signed mean within +-0.035 (no bias). In p that is 0.004-0.012
on average, up to 0.12 for an item near 0.5, and 1-3 items of 185 that sit
next to 0.5 change sides. The size of the forward matters, the companions
do not (batch of 8 vs another batch of 8: mean abs 0.02). An item alone
gives the same p every time.

So the rule is on the logit differences between A and B: exit code 1 if the
signed mean exceeds --bias-tol (a batch-size bias in the labels), the mean
abs exceeds --mean-tol, or the largest exceeds --max-tol (more than rounding).

  python swm-next/verifier/judge_mb_check.py --url http://127.0.0.1:8902 \
      --h5 <unit>.h5 --cycles 48
"""
import argparse
import io
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
from PIL import Image

from http_judge import HTTPJudge


def decode(blob):
    return np.asarray(Image.open(io.BytesIO(
        np.asarray(blob, np.uint8).tobytes())).convert("RGB"), np.uint8)


def stats(url):
    return json.loads(urllib.request.urlopen(
        url.rstrip("/") + "/stats", timeout=30).read())


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--cycles", type=int, default=48)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--bias-tol", type=float, default=0.06, dest="bias_tol")
    ap.add_argument("--mean-tol", type=float, default=0.25, dest="mean_tol")
    ap.add_argument("--max-tol", type=float, default=1.0, dest="max_tol")
    args = ap.parse_args()

    items = []                                   # (kind, image(s), question)
    with h5py.File(args.h5, "r") as h5:
        eps = sorted(h5.keys())
        per = -(-args.cycles // len(eps))
        for name in eps:
            g = h5[name]
            C = int(g.attrs["cycles"])
            for c in np.linspace(0, C - 1, min(per, C)).astype(int):
                a, e = decode(g["frames_t"][c]), decode(g["end_frames"][c, 0])
                items.append(("touch", e, g.attrs["q_touch"]))
                items.append(("closer", (a, e), g.attrs["q_closer"]))
    judge = HTTPJudge(args.url)

    def one(it):
        p, m = judge.p_yes([it[1]], [it[2]])
        return float(p[0]), float(m[0])

    def serial():
        t0 = time.perf_counter()
        out = [one(it) for it in items]
        return np.asarray(out), time.perf_counter() - t0

    one(items[0]); one(items[1])                 # warm-up, both kinds
    A, tA = serial()
    s0 = stats(args.url)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.threads) as ex:
        B = np.asarray(list(ex.map(one, items)))
    tB = time.perf_counter() - t0
    s1 = stats(args.url)
    C_, tC = serial()

    n = len(items)
    fw = s1["forwards"] - s0["forwards"]
    print(f"{n} items | pass A {tA:.1f}s = {n / tA:.1f} items/s at 1 item per "
          f"forward | pass B {tB:.1f}s = {n / tB:.1f} items/s, {fw} forwards, "
          f"mean batch {(s1['items'] - s0['items']) / max(fw, 1):.2f}, GPU "
          f"thread busy {(s1['busy_s'] - s0['busy_s']) / tB:.0%} | pass C "
          f"{tC:.1f}s", flush=True)
    bad = False
    for kind in ("touch", "closer"):
        ix = np.asarray([i for i, it in enumerate(items) if it[0] == kind])
        a, b, c = A[ix, 0], B[ix, 0], C_[ix, 0]
        dab, dac = np.abs(a - b), np.abs(a - c)
        flip = (a > 0.5) != (b > 0.5)
        dl = logit(b) - logit(a)
        print(f"{kind}: n {len(ix)}, p mean {a.mean():.3f}, share > 0.5 "
              f"{(a > 0.5).mean():.2f} | batched vs alone: p max {dab.max():.4f}"
              f", mean {dab.mean():.5f}; logit signed mean {dl.mean():+.4f}, "
              f"mean abs {np.abs(dl).mean():.4f}, max {np.abs(dl).max():.3f}; "
              f"flips across 0.5: {int(flip.sum())}"
              f"{' at p alone ' + ', '.join(f'{v:.3f}' for v in a[flip]) if flip.any() else ''}"
              f" | alone vs alone again: max {dac.max():.4f}, mean "
              f"{dac.mean():.5f} | mass min {A[ix, 1].min():.3f}", flush=True)
        bad |= (abs(dl.mean()) > args.bias_tol
                or np.abs(dl).mean() > args.mean_tol
                or np.abs(dl).max() > args.max_tol)
    print("MB_CHECK_FAIL" if bad else "MB_CHECK_PASS", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
