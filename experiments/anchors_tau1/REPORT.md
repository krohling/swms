# Base-anchor re-audit (2026-10-29)

Trigger: cross-campaign discrepancy in the base diffusion k=1 anchor — current
campaign 56/48/38 (bg/by/yr, suite 47.3) vs repro-era 52/64/50 (55.3), same
provided checkpoints. Investigated per Kevin's 4-step plan.

## Findings

1. **τ=0 artifact in the current anchors.** All three base k=1 anchor runs
   (`outputs_base/`, `outputs_multitask/*_base_k1/`) were invoked through the
   temperature-ladder path with `temps=[0.0]` — the k=1 degenerate of the
   0→2 ladder. τ scales both DDPM init noise and per-step noise, so τ=0 is the
   deterministic mean-path variant, not the base policy as published. Paired
   effect on seeds 6000–6049 (n=50/task): bg 56→48 (mean-path flatters the
   trained task), by 48→50, yr 38→44. Suite unchanged: 47.3 both ways.
2. **Seeds ruled out.** Both campaigns evaluate seeds 6000–6049
   (VALIDATION.md §A.3.1 and `eval_ckpt*.sbatch` on Vista confirm the
   repro era used the same range).
3. **Budget/cadence ruled out.** Both protocols execute 50 cycles × 4 actions
   = 200 env steps (the authors' OGBench config overrides their 5-action
   default to 4; our `max_cycles`/`actions_per_cycle` were written to match).
4. **Cross-validation with the authors' own harness, same machine, same
   seeds** (`scripts/evaluate_swm_hydra.py --config-name eval_base_ogbench
   device=mps`, expert_diffusion): bg 48 / by 54 / yr 44 (suite 48.7) —
   agrees with our mpc-harness τ=1 numbers within noise.
5. **Both agree with the paper.** arXiv:2510.19818v2 Fig. 5: bg 44 / by 52 /
   yr 40, avg 45.33 (n=50, 95% CI ±~14pp/task). Every task within 1σ of both
   of today's measurements.

## Verdict

| task | paper (Fig. 5) | mpc harness τ=1 | authors' harness (m5) | τ=0 (deprecated) | repro-era (Jul/Aug) |
|---|---|---|---|---|---|
| blue→green | 44 | 48 | 48 | 56 | 52 |
| blue→yellow | 52 | 50 | 54 | 48 | 64 |
| yellow→red | 40 | 44 | 44 | 38 | 50 |
| suite | 45.33 | 47.3 | 48.7 | 47.3 | 55.3 |

The current harness measures the base policy correctly. The repro-era
52/64/50 (+10 over the paper, flagged at the time in VALIDATION.md as "our
base runs ~10 pp above the paper's", reproduced bit-identically on LARG-July
and Vista-Aug) is era/stack-specific inflation; the responsible dependency was
not isolated and does not need to be for current claims.

## Policy going forward

- The campaign's base k=1 anchor is the **τ=1 row: bg 48 / by 50 / yr 44
  (suite 47.3)**. The old 56/48/38 row is deprecated (τ=0 mean-path); suite
  level is coincidentally identical, per-task comparisons must use τ=1.
- Narrative corrections that follow: "bg at base parity" was a τ=0 artifact —
  vs the correct base (and vs random-k8's bg 48), exp1b@64k bg 56 is +8; by
  shows ~0 vs random-k8 at 64k (54 vs 56); yr remains the largest gap
  (+10 vs base, +16 vs random at 64k).
- Never mix eras or protocols in one table; in-table matched-seed controls
  only (random-k8 remains the preferred matched-k control).
- k=8 arms are unaffected: all share identical ladder-generated candidate
  pools (random-k8 JSON records temps 0→2, seeds 6000+).

## Files

- `eval_base_k1_tau1_{bg,by,yr}.json` — mpc harness, `--k 1 --temps 1.0
  --scorer softgate_ema --lookahead 16 --seeds 50 --seed-start 6000`,
  per-episode results.
- `authors_harness_m5_results.txt` — authors' eval, per-seed results, all
  three tasks.
- τ=0 originals remain in `outputs_base/` and `outputs_multitask/` (not
  committed; deprecated).
