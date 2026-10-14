# Run 4 — Qwen SWM on hybrid branched data (pointwise BCE + cumsum adapter)

**Verdict: offline learning succeeded, closed-loop selection did not.** The SWM
reached its best-ever offline ranking (within-cycle ρ ≈ 0.30 on informative
lists, vs 0.15–0.17 for run-3) but closed-loop SWM-MPC ends at base-policy
parity — none of the judge's +17pp selection lift transfers.

## Recipe

- Data: 5,400 hybrid episodes (k=8 branched proposals/cycle, min_sep 0.05
  rejection sampling, 50/50 verifier/random execution), task blue→green only.
  Shards: m5 s30000 (1,000) + Vista s31000/32100/33200/34300 (1,100 ea).
- Model: frozen Qwen3-VL-8B + LoRA r16 (LM q/k/v/o) + per-step trajectory
  projector with **cumsum channels** ([a_t, cumsum/H] → 16×10 input, new this
  run), open-gate init 0.1.
- Loss: pointwise soft-BCE on teacher p_yes; cycle-grouped batching
  (12 lists/step eff). 3 epochs = 84,687 steps, fully trained (leg A TIMEOUT
  at 84,000; leg B finished the tail — TRAIN_DONE 2026-10-13). The "84k" eval
  checkpoint is 687 steps before the true final; rho was flat across the gap
  (0.303 @ 84.5k). ~1,740 steps/h on GH200.
- wandb: `kevin_ai/swm-lora/run4-hybrid-k8-pointwise`.

## Offline metrics (val, 160 informative lists)

ρ_informative: 0.19–0.22 by 8k steps, 0.24 @ 23k, 0.26 @ 42k, 0.28 @ 56k,
~0.30 plateau 66k→84k. BCE 0.59 → 0.50. Base-drift stable. (Not comparable to
the pilot's val set — different shard composition, 37 vs 160 informative lists.)

## Closed-loop SR (n=50/task, k=8, seeds 6000–6049, eval_closed_loop.py)

SWM replaces sim+judge in the MPC loop: scores (frame_t, trajectory) directly,
zero-trajectory probe supplies the frame-only gate.

| Task | Base k=1 | Random k8 | SWM 23k | 42k | 56k | **84k** | Judge k8 | Oracle k8 |
|---|---|---|---|---|---|---|---|---|
| bg (in-dist) | 56 | 48 | 42 | 50 | 54 | **52** | 64 | — |
| by (OOD) | 48 | 56 | — | — | 52 | **54** | 68 | 84 |
| yr (OOD) | 38 | 38 | — | — | 42 | **44** | 62 | 80 |
| aggregate | 47.3 | 47.3 | — | — | 49.3 | **50.0** | 64.7 | — |

- In-dist SR climbed with training (42→50→54) then flattened; 56k↔84k is noise.
- OOD: no collapse (interface + questions transfer mechanically), no lift.
- Judge-MPC captures +17.4pp aggregate over base on identical candidates;
  SWM-MPC captures +2.7pp (within n=150 noise of zero).

## Reading

ρ ≈ 0.30 ranking is real but far below what argmax top-1 selection needs: the
SWM orders siblings better than chance while still being wrong at the top of
the list most of the time, and the gate-blended argmax amplifies top-1 errors.
Offline ρ and closed-loop SR decoupled in both directions during the run
(SR rose 8pp while ρ was flat 23k→42k; ρ rose 0.26→0.30 while SR was flat
56k→84k) — reinforcing the no-offline-gating policy.

Distillation-fidelity framing: the teacher's own closed-loop value (judge k8,
+17pp) needs top-1 fidelity that pointwise BCE on a ±0.05-JND teacher did not
deliver at this scale. Structural candidates, not more steps: training signal
that targets the decision (selection/margin losses), or comparison-capable
inputs (all-k-in-context).

## Run 5 note (selection arm, first attempt)

The all-k-in-context selection arm (digit-token soft CE, τ=0.1, shuffled
candidates; branch `selection-arm`, run `run5-selection-k8`) collapsed into
the uniform-output saddle within 500 steps (train CE pinned at ln 8 = 2.079,
grad norm 32→0.02, 17k+ steps flat). Cause: with shuffled candidates the
marginal best-index distribution is uniform, and the loss has no pointwise
anchor to bootstrap trajectory-differentiation — gradients flow only through
the (quiet, scale≈0.1) trajectory tokens. Candidate fix: warm-start from the
run-4 84k adapters so trajectory-outcome structure pre-exists and the digit
readout only learns the argmax wiring. Decision pending.

## Artifacts

- Eval JSONs: `experiments/run4/eval_cl_{23k,42k,56k_*,84k_*}*.json`
- Checkpoints: Vista `$WORK/swm-followup/swms-datagen/outputs_swm_lora/run4_hybrid/`
  (every 1k steps); 23k/42k/56k/84k mirrored on m5 `outputs_swm_lora_local/`.
- Baselines provenance: `outputs_base/`, `outputs_random/`, `outputs_multitask/`
  (seed-matched n=50; the n=800 bg base 43.4% is generation-seed, do not mix).
