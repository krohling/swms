"""Micro-batching judge server: the HTTP contract of judge_server.py
(counterpart http_judge.HTTPJudge), for MANY client processes that each send
one or two items per request (datagen_lt.py: one labeled rollout per state).

  POST /p_yes   {"images": [img | [img, img], ...], "questions": [str, ...]}
                 img = base64 PNG. A 2-list is a temporal PAIR (earlier, now).
  -> {"p": [...], "mass": [...]}
  GET  /health  200
  GET  /stats   {"forwards": n, "items": n, "busy_s": s, "up_s": s}

judge_server.py serves one request at a time, one forward per request, so N
single-item clients cost N forwards. Here requests are accepted in threads
and one GPU thread drains a queue: whatever arrived while the previous
forward ran is batched into the next one (up to --batch). Nothing waits for
a batch to fill, so an idle server answers as fast as judge_server.py does.

A batch only ever holds items with the SAME question text, the same kind
(single frame or pair) and the same frame shapes, so every sequence in a
forward has the same length and nothing is padded — the batches the teacher
(mpc.run_lt_episode: k end frames, one question) sends. The oldest waiting
item picks the next group.

Run:  python swm-next/verifier/judge_server_mb.py \
          --model Qwen/Qwen3-VL-8B-Instruct --device cuda --port 8902
Listens on 127.0.0.1 only by default.
"""
import argparse
import base64
import io
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image


def _decode(b64):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"),
                      dtype=np.uint8)


class Item:
    __slots__ = ("image", "question", "key", "t", "p", "mass", "err", "left",
                 "event")

    def __init__(self, image, question, left, event):
        self.image, self.question = image, question
        pair = isinstance(image, tuple)
        shapes = tuple(f.shape for f in image) if pair else image.shape
        self.key = (pair, question, shapes)
        self.t = time.monotonic()
        self.p = self.mass = self.err = None
        self.left, self.event = left, event      # per-request countdown


def gpu_loop(judge, q, max_batch, stats):
    groups = {}                                  # key -> [Item] (FIFO)
    while True:
        if not groups:
            it = q.get()
            groups.setdefault(it.key, []).append(it)
        while True:                              # everything that has arrived
            try:
                it = q.get_nowait()
            except queue.Empty:
                break
            groups.setdefault(it.key, []).append(it)
        key = min(groups, key=lambda k: groups[k][0].t)
        batch, rest = groups[key][:max_batch], groups[key][max_batch:]
        if rest:
            groups[key] = rest
        else:
            del groups[key]
        t0 = time.monotonic()
        try:
            p, m = judge.p_yes([b.image for b in batch],
                               [b.question for b in batch])
            for b, pi, mi in zip(batch, p, m):
                b.p, b.mass = float(pi), float(mi)
        except Exception as e:                   # the clients retry, then fail
            for b in batch:
                b.err = repr(e)
            print(f"forward failed ({len(batch)} items): {e!r}", flush=True)
        stats["busy_s"] += time.monotonic() - t0
        stats["forwards"] += 1
        stats["items"] += len(batch)
        for b in batch:
            with b.left[1]:
                b.left[0] -= 1
                if b.left[0] == 0:
                    b.event.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8902)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--log-every", type=float, default=300.0, dest="log_every")
    args = ap.parse_args()

    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))          # swm-next/
    from label_teacher import QwenJudge

    print(f"loading {args.model} on {args.device} ...", flush=True)
    judge = QwenJudge(model_id=args.model, device=args.device)
    print("judge ready", flush=True)

    q = queue.Queue()
    t_up = time.monotonic()
    stats = dict(forwards=0, items=0, busy_s=0.0)
    threading.Thread(target=gpu_loop, args=(judge, q, args.batch, stats),
                     daemon=True).start()

    def log_loop():
        last = dict(stats)
        while True:
            time.sleep(args.log_every)
            now = dict(stats)
            f = now["forwards"] - last["forwards"]
            n = now["items"] - last["items"]
            if f:
                print(f"{time.strftime('%m-%d %H:%M:%S')} {n} items in {f} "
                      f"forwards (mean batch {n / f:.2f}), "
                      f"{n / args.log_every:.1f} items/s, GPU thread busy "
                      f"{(now['busy_s'] - last['busy_s']) / args.log_every:.0%}",
                      flush=True)
            last = now

    threading.Thread(target=log_loop, daemon=True).start()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):                     # quiet access log
            pass

        def _send(self, code, obj=None):
            out = b"" if obj is None else json.dumps(obj).encode()
            self.send_response(code)
            if out:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_POST(self):
            if self.path != "/p_yes":
                return self._send(404)
            body = json.loads(self.rfile.read(
                int(self.headers["Content-Length"])))
            images = [
                (_decode(im[0]), _decode(im[1])) if isinstance(im, list)
                else _decode(im)
                for im in body["images"]]
            questions = body["questions"]
            if len(images) != len(questions) or not images:
                return self._send(400, {"error": "images/questions mismatch"})
            event, left = threading.Event(), [len(images), threading.Lock()]
            items = [Item(im, qu, left, event)
                     for im, qu in zip(images, questions)]
            for it in items:
                q.put(it)
            event.wait()
            errs = [it.err for it in items if it.err]
            if errs:
                return self._send(500, {"error": errs[0]})
            self._send(200, {"p": [it.p for it in items],
                             "mass": [it.mass for it in items]})

        def do_GET(self):
            if self.path == "/health":
                return self._send(200)
            if self.path == "/stats":
                return self._send(200, dict(
                    stats, up_s=time.monotonic() - t_up))
            self._send(404)

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 128

    print(f"serving on {args.host}:{args.port} (micro-batch {args.batch})",
          flush=True)
    Server((args.host, args.port), H).serve_forever()


if __name__ == "__main__":
    main()
