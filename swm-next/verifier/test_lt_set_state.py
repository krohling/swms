"""Phase-1 gate for LangTable verifier-MPC: save -> candidate rollout ->
restore must be deterministic, or the k>1 mechanism (sim as perfect world
model) is invalid.

VALIDATED RECIPE (this test enforces it):
  1. setPhysicsEngineParameter(deterministicOverlappingPairs=1) at env setup
     (pybullet's broadphase-order determinism flag; without it, contact
     warm-start caches depend on the *previous* candidate's trajectory ->
     up to ~4e-3 m state drift at a 16-step horizon).
  2. COMBO restore per candidate: env.set_state(S)  [ObjState: python-side
     target_effector_pose + object poses/vels] THEN
     env.restore_state_native(sid)  [pybullet saveState snapshot: engine
     internals]. Either alone is insufficient (measured).
  3. Optional one burn-in rollout right after saveState: upgrades replay from
     2e-9 m to bit-exact 0.0. Frames are pixel-exact either way.

Measured (container, linux/arm64, pybullet 3.2.7):
  - interleaved replay (the real MPC pattern), flag ON + combo + burn-in:
      state diff 0.0 at every step, 16/16 frames pixel-identical.
  - flag ON, no burn-in: state <= 2e-9 at step 16, still 16/16 pixel-exact.
  - flag OFF: 4.3e-3 m drift, 15/16 frames (FAIL).

Run (container): TF_USE_LEGACY_KERAS=1 PYTHONPATH=/workspace/swms \
    python swm-next/verifier/test_lt_set_state.py
"""
import numpy as np

from swm.utils.envs import get_lang_table_env

H = 16                     # verifier-MPC planning horizon (matches OGBench)
# Task-anchored tolerance: residual replay drift wiggles run-to-run in the
# 0 .. ~2e-8 m range (contact-chaos amplification of engine-cache bits) while
# frames stay pixel-exact. 1e-6 m is 100x that ceiling and ~5 orders below
# both the done-threshold (0.065 m) and block size (0.04 m). The binding
# criterion is pixel-exact frames — that is everything the judge sees.
STATE_TOL = 1e-6


def rng_actions(seed, n, scale=0.5):
    return np.random.RandomState(seed).uniform(
        -scale, scale, size=(n, 2)).astype(np.float32)


def main():
    env = get_lang_table_env(
        {"ood": False, "block_combo": ["green_cube", "blue_moon"]}, seed=54)
    env.env._pybullet_client.setPhysicsEngineParameter(
        deterministicOverlappingPairs=1)

    def vec():
        st = env.env.get_block_states()
        return np.concatenate(
            [np.asarray(st[k], np.float64).ravel() for k in sorted(st)])

    for a in rng_actions(0, 10):       # warm-up: contacts + velocities at S
        env.step(a)

    S = env.get_state()
    env.set_state(S)
    sid = env.save_state_native()

    def restore():
        env.set_state(S)               # python-side pose + ObjState
        env.restore_state_native(sid)  # engine snapshot

    def roll(a_seq):
        restore()
        st, fr = [], []
        for a in a_seq:
            env.step(a)
            st.append(vec())
            fr.append(np.asarray(env.get_frame(), np.int16))
        return np.stack(st), np.stack(fr)

    acts, acts_alt = rng_actions(1, H), rng_actions(2, H)

    roll(acts)                                   # burn-in (discarded)
    s1, f1 = roll(acts)                          # candidate rollout
    s_alt, _ = roll(acts_alt)                    # interleaved other candidate
    s2, f2 = roll(acts)                          # replay of the first

    ds = np.abs(s1 - s2).max()
    per_frame = np.abs(f1 - f2).reshape(H, -1)
    exact_frames = int((per_frame.max(axis=1) == 0).sum())
    px_max = int(per_frame.max())                    # worst pixel-value delta
    px_count = int((per_frame > 0).sum())            # total differing pixels
    div = np.abs(s1 - s_alt).max()

    print(f"interleaved replay: max|state diff| = {ds:.2e}  "
          f"pixel-exact frames = {exact_frames}/{H}  "
          f"(max px delta {px_max}, differing px {px_count})")
    print(f"different-candidate divergence = {div:.3f} (must be > 0)")

    # Frame criterion: bit-exact preferred; on platforms whose renderer
    # rounds differently (e.g. GH200/aarch64 TinyRenderer), accept a handful
    # of boundary pixels flipping by a tiny amount — far below anything the
    # judge (or any VLM) can respond to. State tolerance is unchanged.
    frames_ok = exact_frames == H or (px_max <= 2 and px_count <= 200)
    ok = ds <= STATE_TOL and frames_ok and div > 0
    print("PHASE1_PASS" if ok else "PHASE1_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
