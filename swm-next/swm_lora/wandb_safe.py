"""Crash-proof wandb: telemetry must never kill a training run.

Vista exp1 leg 1 (job 1085550) died at step ~29.5k, 21.5h in, when the wandb
SDK<->service socket reset (ConnectionResetError raised out of run.log) while
training itself was healthy. This wrapper treats wandb as expendable: every
call is caught; on failure it reattaches to the SAME run (the init kwargs
must carry id=<run_name>, resume="allow") at most once per RETRY_SEC; while
wandb is down, training continues with stdout logs and checkpoints as the
source of truth. Outages appear as gaps in the curves, nothing more.
"""
import time

RETRY_SEC = 300


class SafeRun:
    def __init__(self, **init_kw):
        self._kw = init_kw
        self._run = None
        self._next_try = 0.0
        self._connect(first=True)

    def _connect(self, first=False):
        self._next_try = time.time() + RETRY_SEC
        try:
            import wandb
            if not first:
                try:
                    wandb.teardown()      # reap the dead service process
                except Exception:
                    pass
            self._run = wandb.init(**self._kw)
            if not first:
                print("[wandb] reconnected; telemetry resumed", flush=True)
        except Exception as e:
            self._run = None
            print(f"[wandb] {'init' if first else 'reconnect'} failed "
                  f"({type(e).__name__}: {e}); training continues without "
                  f"telemetry, retry in {RETRY_SEC}s", flush=True)

    def _call(self, method, *a, **kw):
        if self._run is None and time.time() >= self._next_try:
            self._connect()
        if self._run is None:
            return
        try:
            getattr(self._run, method)(*a, **kw)
        except Exception as e:
            print(f"[wandb] {method} failed ({type(e).__name__}: {e}); "
                  f"dropping telemetry until reconnect", flush=True)
            self._run = None
            self._next_try = time.time() + RETRY_SEC

    def log(self, *a, **kw):
        self._call("log", *a, **kw)

    def log_artifact(self, *a, **kw):
        self._call("log_artifact", *a, **kw)

    def finish(self):
        self._call("finish")
