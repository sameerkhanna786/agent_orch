# Phase-planner A/B — pre-registered evaluation protocol

Written BEFORE the powered run, so the analysis is hypothesis-driven, not post-hoc. Holds the
experiment to the standard of `APEX_COMMIT0_REPORT.md` (n≥3, exclude infra non-results, separate
verified effects from variance, report cost alongside rate).

## Question
Does the Claude-Code-style **phase planner** (`hybrid`) improve the orchestrator over the proven
**converge** incumbent on hard commit0 repos, at acceptable cost — and does the adversarial
**goal-alignment gate** earn its keep?

## Arms (paired — same seeds across arms within each repo)
| arm | flags | role |
|---|---|---|
| `converge` | `APEX_OMEGA_ORCHESTRATION=converge` | **control** (incumbent baseline) |
| `hybrid` | `=hybrid APEX_OMEGA_PHASE_PLANNER=1` (gate ON) | **treatment** |
| `hybrid-nogate` | `=hybrid APEX_OMEGA_PHASE_PLANNER=1 APEX_OMEGA_GOAL_GATE=0` | **ablation** (isolates the gate) |

All three carry identical repair flips (`REPAIR_ITERS=2`, `REPAIR_EXCERPTS=1`), so the **only**
variable is the orchestration shape. `APEX_OMEGA_SKIP_AUTH_PREFLIGHT=1` (the pilot's auth-preflight
race must not contaminate cells).

## Repos
- **Hard discriminators (where the hypothesis lives):** `mimesis` (6159-test near-solve repo, the
  run-4 lost-solve victim), `babel` (4598/4607 near-solve).
- **Controls:** `jinja` (medium; the historical single discriminator), `voluptuous` (easy;
  no-regression — `hybrid` MUST hit the skip-gate and run the identical cheap path as `converge`).

## Replication & budget
- **n = 3 seeds** per (arm × repo) → 36 cells. (Report's stated minimum for a claim; n=1 cannot rank
  arms. n=5 on the discriminators is the pre-registered extension if the n=3 direction is positive.)
- Wall `LADDER_CELL_TIMEOUT=10800` (3h/cell), `LADDER_MAX_RELAUNCH=1`, concurrency 2. Safe to clip:
  acceptance-checkpointing banks any verified solve the instant it passes; partial frontiers are
  recovered. Paired, seed-major; resumable.

## Outcomes
- **Primary (binary):** per-cell SOLVE = execution-accepted full-suite pass (C7: a winner with
  `accepted==True` AND non-zero usage). Excludes false/fetch solves.
- **Secondary (continuous):** best banked **gold frontier** = `gold_passed / gold_total` from
  `phase_checkpoint.json` (graded progress; has power even when full solves are rare on mimesis).
- **Cost:** `agents_used` per cell (median + IQR), compared on shared-outcome cells.
- **No-veer / gate value:** frontier(`hybrid`) − frontier(`hybrid-nogate`); recorded `defer`
  (merge-conflict / fetch) counts; integrity (escape/cheat) attempts (telemetry, never penalized).

## Exclusions (NON-RESULTS — excluded from denominators, listed to re-run)
`timeout` (wall ≥ 0.97·cap before budget used), `infra` (harness/auth/os-error crash), `token`
(ceiling). Classified by `scripts/reclassify_grid.py` (C7). Reported separately; never counted as a
failure.

## Statistical analysis (pre-registered; `scripts/analyze_phase_ab.py`)
1. **Per-arm solve-rate** over genuine cells (solved + honest-fails) with **Wilson 95% CI**.
2. **Paired McNemar exact test** (two-sided binomial on discordant pairs) for `hybrid` vs
   `converge` and `hybrid` vs `hybrid-nogate`, matched on (repo, seed), per-repo and pooled.
3. **Paired sign test** on the continuous frontier differences (same pairings) + median frontiers.
4. **pass@k** per (arm, repo); **agents/solve** distribution.
5. **Power/limitations** statement: number of discordant pairs and what the data can/cannot support.

## Decision rules (pre-registered)
- **Promote `hybrid` to default** ONLY if: (H1) it banks ≥1 verified solve `converge` misses on
  mimesis/babel with **no control regression** (voluptuous/jinja solve-rate unchanged within CI),
  AND (H2) no cost blowup (agents/wall within the governor window of `converge`; no A-solve turned
  into a B-timeout). If frontier (secondary) rises significantly but no net solve, keep `hybrid`
  behind its flag and report the graded gain honestly.
- **Gate (H3):** ship `hybrid` over `hybrid-nogate` only if the gate shows a solve OR frontier
  benefit; if they tie, ship the cheaper gate-off arm.
- **Null:** if `hybrid` ≈ `converge` on both solve and frontier at higher cost, the phase layer is
  pure cost → stays flag-gated, default OFF (honest negative result).

## Threats to validity (acknowledged up front)
- **Low power:** with n=3 and rare full solves, McNemar on binary solves may have few discordant
  pairs → underpowered; the continuous frontier endpoint mitigates this.
- **Stochasticity:** agent runs are non-deterministic; seeds capture it but n=3 is small.
- **Single environment / vendor (codex):** results may not transfer to other CLIs.
- **mimesis fetch-monoculture prior** can suppress all arms equally (a shared confound, not an
  arm effect).
