# Exp1 closed-loop SR curve (blue→green)

**2026-10-25.** Closed-loop success rate of the exp1 score-regression SWM
(`exp1-allinear-warmstart`: run4 recipe + all-linear LoRA r16 + warm-started
encoder, v1 data) used as the MPC scorer, across the training trajectory.

## Setup

- Harness: `swm-next/swm_lora/eval_closed_loop.py` @ `1eefec5` on m5 (MPS).
  The fix in `1eefec5` matters: before it, the harness rebuilt the model with
  attention-only LoRA and peft's non-strict loading silently dropped the MLP
  adapter weights of all-linear checkpoints. Verified post-fix: 504/504
  adapter tensors load (attention-only loads 288).
- Deployment-faithful scoring: zero-traj probe as frame-only gate,
  EMA α=0.35, score = (1−g)·p_app + g·(0.6·p_place + 0.4·p_hold), k=8
  temperature ladder 0→2, L16, 4 actions/cycle, max 50 cycles.
- 50 episodes/checkpoint, eval seeds 6000–6049 — the same seed set as every
  anchor below, so columns are directly comparable and pairable.
- Checkpoints pulled from Vista `outputs_swm_lora/exp1_full` (training
  completed cleanly at step 84,687; ~3 epochs).

## Results

| scorer | SR (n=50) |
|---|---|
| exp1 @16k | 54% |
| exp1 @42k | 54% |
| exp1 @64k | 54% |
| exp1 @84,687 (final) | **56%** |
| base diffusion k=1 (`outputs_base/blue_green_k1`) | 56% |
| run4 @23k (`experiments/run4/eval_cl_23k_bg_k8.json`) | 42% |
| run4 @84k final (run4 REPORT) | 52% |
| teacher judge-MPC soft-gate k=8 (`outputs_verifier_mpc_softgate_ema_k8_n50`) | 64% |

Per-episode JSONs: `experiments/exp1/eval_cl_step{16000,42000,64000,84687}_bg_k8.json`.

Paired sign tests (same seeds):

- exp1-final vs base-k1: 9 discordant each way, p = 1.00 — **exact parity**.
- exp1-final vs teacher-k8: teacher wins 7 of 10 discordant, p = 0.34
  (underpowered at n=50, direction consistent with the 8-pt gap).
- exp1-final vs run4@23k: exp1 wins 10 of 13 discordant, p = 0.09.

Mean cycles-to-success ≈ 19 of 50 at all points. Checkpoint-to-checkpoint
per-seed churn is high (~half of the union of successes flip between any two
checkpoints) — consistent with selection noise, not an accumulating skill.

## Reading

1. **Exp1 is base parity closed-loop, end to end.** The curve is flat
   (54/54/54/56) and the final point is seed-for-seed even with running the
   diffusion policy alone. The teacher's +8 over base (64 vs 56) is not
   captured at any checkpoint.
2. **Offline gains again failed to convert.** Exp1 set project-best offline
   numbers (val BCE 0.490, ρ_inf 0.298) — closed-loop didn't move. Third
   consecutive arm (run4, run5b, exp1) where offline fidelity and SR
   decouple; reaffirms the standing lesson to never gate on offline ρ alone.
3. **What all-linear + warm-up did buy:** run4 passed through an actively
   harmful phase (42% @23k, 10 pts below base); exp1 is never below base at
   any measured point and reaches base parity by 16k instead of 84k. The
   extra capacity/warm-up made the model harmless faster; it did not make it
   useful.
4. **Open hypotheses, already in flight:** teacher signal density (exp1b:
   R10 tie-filtered v2 data + no-scale warm-up, training now) and backbone/
   capacity ceiling (pg-fullft: PaliGemma-3B full-weight, queued on Vista).
   Their SR curves go through this same harness and seed set.

## Full 3-task suite (added 2026-10-25)

Same harness/seeds, by/yr via the teacher-suite configs (template questions
in each task's colors — the run4 OOD protocol; exp1 trained on bg only).

| scorer | bg | by | yr | suite |
|---|---|---|---|---|
| exp1 @16k | 54 | 54 | 48 | 52.0 |
| exp1 @42k | 54 | 58 | 52 | **54.7** |
| exp1 @64k | 54 | 58 | 42 | 51.3 |
| exp1 @84,687 final | 56 | 60 | 44 | 53.3 |
| base diffusion k=1 | 56 | 48 | 38 | 47.3 |
| random-pick k=8 (matched-k) | 48 | 56 | 38 | 47.3 |
| run4 final | 52 | 54 | 44 | 50.0 |
| teacher judge-MPC k=8 | 64 | 68 | 62 | 64.7 |

Suite read: exp1-final sits ~+6 over both no-knowledge controls, capturing
~1/3 of the teacher's +17.4 lift over random (run4 captured ~15%). Pooled
n=150/row → s.e. ~4 pts: directionally the best closed-loop result from a
trained SWM so far, not yet significance. All suite-level gain is present by
16k; OOD holds (by strongest at 60); bg remains at base parity.

Per-episode JSONs: `experiments/exp1/eval_cl_step*_{bg,by,yr}_k8.json`;
exp1b's ladder accumulates in `experiments/exp1b/` as training progresses
(16k: bg 52 / by 58 / yr 40, agg 50.0 — base-level at 16k, like run4;
42k: bg 58 / by 52 / yr 44, agg 51.3 — bg 58 is the first trained-arm point
above base on bg, the training task; suite still ~3 behind exp1@42k, within
noise; 64k: bg 56 / by 54 / yr 54, agg 54.7 — ties exp1's best suite point,
with the curve still rising at 64k where exp1's had rolled over, and yr 54
the best yr from any trained arm; final 80,367: bg 52 / by 70 / yr 42,
agg 54.7).

exp1b curve complete: 50.0 → 51.3 → 54.7 → 54.7. Where exp1 peaked at 42k
and rolled over, exp1b plateaus at the same 54.7 ceiling — R10 filtering
bought a later, flatter peak, not a higher one. The final point's
composition is very different from 64k's balanced 56/54/54, though:
**by 70 beats the teacher's 68 on that task** — the best single-task
number from any trained arm this campaign, +14 over random's 56 (n=50,
s.e. ~7, so directionally strong) — while yr gives back its 64k gain
(54 → 42, back near controls) and bg slips to 52, below base 56. Same
suite, opposite shape: late training concentrated the model's skill on
one OOD task at the expense of the other two. Checkpoint selection
matters again (64k is the more deployable point); the teacher's 64.7
stays well clear of every trained arm.

**Compose baseline** (exp2 step 1, `experiments/exp2/compose_exp1_final.json`,
4,000 held-out v2-R10 cycles, stored-gate blend): exp1-final's per-question
predictions recover the planner's pick only 23.3% of the time (chance 12.5%),
top-1 regret 0.186, pair-acc(≥0.10) 0.613. Composition is lossy — the
selector arm's headroom argument, and the bar exp2 must beat.
