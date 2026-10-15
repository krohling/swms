# Run 5/5b — All-k-in-context selection arm (digit-token CE, warm-started)

**Verdict: same closed-loop conclusion as run4 from the opposite direction.**
The selector reached the project's best offline numbers (ρ_informative 0.357,
top-1 agreement 0.288 = 2.3× chance) yet the closed-loop suite aggregate is
flat at the no-signal line at every checkpoint (48.0 / 47.3 / 48.0 vs
base=random=47.3). Reframing scoring as selection did not convert offline
fidelity into deployment picks.

## Architecture (branch `selection-arm`)

One forward sees the frame + ALL k=8 candidate trajectories (8 blocks × 16
placeholder tokens, shared cumsum projector) and answers with a digit token
"1"–"8"; loss = soft CE toward the teacher's softmax over the k scores at
τ=0.1 (~2× teacher JND). Candidates shuffled per sample; zero-init per-block
candidate-index embedding (8×4096) binds block identity. No new head — the
digit rows of the frozen LM head are the readout. ~3× cheaper per list than
the scoring arm (one image encode per 8 candidates).

## Run 5 (cold start): uniform-saddle collapse

With shuffled candidates the marginal best-index distribution is uniform, and
the loss has no pointwise anchor: the fastest early descent erases the base
model's digit prior and lands at exactly uniform output (train CE = ln 8 =
2.0794 for 17k+ steps, grad norm 32 → 0.02, proj_scale frozen). Train = val =
ln 8 on both splits — the signature of a constant predictor, an optimization
saddle rather than a capacity limit. Killed at 18k steps.

## Run 5b: warm start from run4's final adapters

Loading run4's 84k LoRA+projector breaks the symmetry at init (digit-logit
spread ~3.3 before any selection training). Trained 63,513 steps (eff 16
lists/step, 3 epochs over the same 5,400-ep hybrid data as run4; ~2,700
steps/h on GH200; one restart at 53k — /home1 quota blown by wandb artifact
staging, fixed via WANDB_DATA_DIR=$WORK, resumed from 52.5k).

Offline (val, 160 informative lists): CE 2.12 → 1.986, ρ_informative 0.325 @
35k → **0.357** final, top1 0.288, proj_scale grew 0.095 → 0.24+. All
project-best; the selection objective does fit the teacher's rankings better
than pointwise BCE did.

## Closed-loop SR (n=50/task, k=8, seeds 6000–6049, eval_closed_loop_sel.py)

Picker = gate-blend of per-question digit softmaxes; gate = inherited yes/no
readout on a zero trajectory (frame-only p(grasp), EMA α=0.35).

| ckpt | bg | by (OOD) | yr (OOD) | aggregate |
|---|---|---|---|---|
| 21k | 48 | 56 | 40 | 48.0 |
| 42k | 50 | 54 | 38 | 47.3 |
| 63.5k (final) | 54 | 62 | 28 | **48.0** |
| base / random k8 | 56 / 48 | 48 / 56 | 38 / 38 | **47.3 / 47.3** |
| judge k8 | 64 | 68 | 62 | **64.7** |

Final-row per-task moves (by +6 over random; yr −10 under base) are
individually inside n=50 noise and cancel in aggregate. No checkpoint
separates from 47.3.

## Reading (run4 + run5b jointly)

Two architecturally distinct arms — absolute scoring and relative selection —
now agree: offline agreement with this teacher's within-cycle rankings, up to
ρ≈0.36 / top1≈0.29, does not move closed-loop SR, while the teacher itself
(judge-MPC) gains +17pp from the same candidates. The remaining suspects are
upstream of architecture/loss:

1. **Teacher signal density**: top1 0.29 vs the ~0.6+ a useful argmax likely
   needs; most lists are still JND-ties where the teacher itself can't rank —
   the learnable signal may cap near where both arms landed.
2. **Data distribution**: 5,400 episodes of one task; run4 showed
   generalization-limited behavior (train/val BCE 0.36/0.50). The +19% s40000
   insurance shard is staged but untested; multi-task data would break the
   OOD protocol — team decision.
3. **Closed-loop mismatch**: training states come from softgate/random
   execution, eval states from the student's own picks — compounding drift
   that offline metrics never see.

## Artifacts

- Eval JSONs: `experiments/run5b/eval_cl_{21000,21k,42000,63513}_*.json`
  (21k bg exists twice — filename-mismatch rerun, identical 48%: determinism check)
- Checkpoints: Vista `$WORK/swm-followup/swms-sel/outputs_swm_lora/run5b_selection/`;
  21k/42k/63513 mirrored on m5 `outputs_swm_lora_local/`.
- wandb: `run5-selection-k8` (saddle), `run5b-selection-warmstart`.
- Run 4 companion report: `experiments/run4/REPORT.md`.
