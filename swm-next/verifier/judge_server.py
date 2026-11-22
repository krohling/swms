"""Host-side judge server: wraps label_teacher.QwenJudge (native, e.g. MPS on
macOS) behind a minimal HTTP endpoint so the LangTable container — which
cannot load the VLM — can score frames. Counterpart: http_judge.HTTPJudge.

  POST /p_yes   {"images": [img | [img, img], ...], "questions": [str, ...]}
                 img = base64 PNG. A 2-list is a temporal PAIR (earlier, now).
  -> {"p": [...], "mass": [...]}

Run (host):  python swm-next/verifier/judge_server.py \
                 --model Qwen/Qwen3-VL-8B-Instruct --device mps --port 8901
The container reaches it at http://host.docker.internal:8901.
Batches are chunked to --batch internally (single forward per chunk).
"""
import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from PIL import Image


def _decode(b64):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"),
                      dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))          # swm-next/
    from label_teacher import QwenJudge

    print(f"loading {args.model} on {args.device} ...", flush=True)
    judge = QwenJudge(model_id=args.model, device=args.device)
    print("judge ready", flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):                     # quiet access log
            pass

        def do_POST(self):
            if self.path != "/p_yes":
                self.send_response(404), self.end_headers()
                return
            body = json.loads(self.rfile.read(
                int(self.headers["Content-Length"])))
            images = [
                (_decode(im[0]), _decode(im[1])) if isinstance(im, list)
                else _decode(im)
                for im in body["images"]]
            questions = body["questions"]
            ps, ms = [], []
            for i in range(0, len(images), args.batch):
                p, m = judge.p_yes(images[i:i + args.batch],
                                   questions[i:i + args.batch])
                ps.extend(float(x) for x in p)
                ms.extend(float(x) for x in m)
            out = json.dumps({"p": ps, "mass": ms}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_GET(self):                              # health check
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()

    print(f"serving on :{args.port}", flush=True)
    HTTPServer(("0.0.0.0", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
