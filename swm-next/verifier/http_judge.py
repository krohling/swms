"""Container-side judge client: duck-types label_teacher.QwenJudge.p_yes over
HTTP to judge_server.py running on the host (or any GPU box). Same contract:
p_yes(images, questions) -> (p_yes, mass) float32 arrays; an images element
may be a single HWC uint8 frame or an (earlier, now) pair for temporal
questions. cfg key: judge_url (e.g. http://host.docker.internal:8901)."""
import base64
import io
import json
import time
import urllib.request

import numpy as np
from PIL import Image


def _b64(frame, upscale=1):
    im = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    if upscale > 1:
        # LangTable's native 180x320 render starves the VLM of visual tokens:
        # measured on "is X touching Y?" (apart vs touching ground truth),
        # 1x gives p_yes 0.053/0.485 (uncertain), 2x gives 0.001/0.992.
        im = im.resize((im.width * upscale, im.height * upscale),
                       Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class HTTPJudge:
    def __init__(self, url, timeout=600, retries=3, upscale=1):
        self.url = url.rstrip("/") + "/p_yes"
        self.timeout = timeout
        self.retries = retries
        self.upscale = upscale

    def p_yes(self, images, questions):
        u = self.upscale
        payload = {"images": [
            [_b64(im[0], u), _b64(im[1], u)] if isinstance(im, (tuple, list))
            else _b64(im, u)
            for im in images], "questions": list(questions)}
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        last = None
        for attempt in range(self.retries):
            try:
                r = json.loads(urllib.request.urlopen(
                    req, timeout=self.timeout).read())
                return (np.asarray(r["p"], dtype=np.float32),
                        np.asarray(r["mass"], dtype=np.float32))
            except Exception as e:                     # transient net/judge hiccup
                last = e
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"judge server unreachable after "
                           f"{self.retries} tries: {last}")
