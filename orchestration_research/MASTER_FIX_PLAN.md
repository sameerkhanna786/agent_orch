# Master Fix Plan — address every issue before the next run

Synthesis of two planning workflows: `wgc4dhz2x` (O1/O5 + completeness sweep + sequencing) and
`ws5veaolg` (O2/O3/O4 redesign), with adversarial corrections folded in. Branch
`feat/phase-planner-hybrid`. Goal: after implementation, be confident EVERY identified issue is
addressed so the next experiment is much better.

## Batch 1 — P0 lossless-resume foundation (MUST land first; the 2220→13 headline loss)
The completeness sweep showed O1 is a CLUSTER of resume-robustness bugs, not one:
- **NEW-I1 (worktree.py `WorktreeProvider.acquire`)**: stale-worktree removal "continues even if removal
  fails" → "worktree already exists" on resume → cascading infra. **Fix:** hard-clean stale `wt_<rid>`
  (git worktree prune + rmtree) before acquire; fail loud if still present.
- **NEW-I4**: runtime/worktree dirs not cleared between resume cycles → floor-probe acquire failures.
  **Fix:** on cell resume, explicitly clear all stale `worktrees/` before re-prep.
- **NEW-I8 (worktree.py fcntl lock)**: stale lock files from killed procs block acquire. **Fix:**
  dead-PID lock reclamation.
- **O1 / NEW-I5 (wal.py + journal)**: a scored candidate's diff blob can be unlinked
  (`fs_diff_ref=False`) → `select_best`/`carry_best` skip diff-less candidates → best frontier lost
  (networkx 2220→13). **Fix:** `Journal.ensure_diff_linked(input_hash, fs_diff)` — idempotent
  bank+relink, called from `_scored()` + `_attempt()`; `commit()` always stores the blob before
  updating `_index`; persist `gold_passed/pass_rate/indeterminate/content_sha` into
  `structured_result`.
- **O1 / NEW-I2 (context.py)**: `_all_candidates` starts empty on resume → prior candidates not
  restored → carry/select see a partial set. **Fix:** `_restore_candidates_from_journal()` in
  `OrchestrationContext.__init__` (lazy diff-load to bound init cost).

## Batch 2 — P1 (independent; can parallelize with Batch 1 except the context.py items)
- **O5 / NEW-I10 (context.py `_scored` cache key, journal/key.py)**: the `--binary` change busted the
  `sha256(fs_diff)` score-cache key → resumed cells re-score from scratch. **Fix:** key on CODE
  identity (candidate `content_sha` / write-tree of the applied diff) + a `diff_format_version`, not
  the raw diff text; keep fs_diff out of the canonical key.
- **O4 / NEW-I7 (apex/evaluation/commit0_benchmark.py)**: `--memray` addopts not loaded under
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD` → `unrecognized arguments`. **Fix:** at pytest-command build, parse
  the repo addopts and STRIP any plugin option whose plugin isn't importable in the scoring venv (log
  every strip; check immediately before invocation). **Verify first** (corr.): confirm the gold
  *scoring* pytest actually applies the repo addopts — else demote to telemetry.
- **F1 / CONFIRM-F1**: pin `pydantic-core==2.46.1` (and other fragile deps) in the eval venv setup +
  a loud pre-run version check, so the prep-crash can't recur on a venv rebuild.

## Batch 3 — P2 (hardening + the new partial-loss guards)
- **NEW-I3 / NEW-I9 (worktree.py `apply_diff`)**: a `--3way` apply can leave conflict markers →
  silent scoring corruption. **Fix:** after apply, scan the modified files for the full `<<<<<<<`/
  `=======`/`>>>>>>>` triplet; on detection return failure (indeterminate), don't score a poisoned
  tree.
- **NEW-I6 (context.py)**: governor cut not synchronized with banking → recovery opportunities missed.
  **Fix:** a best-partial checkpoint (sibling of phase_checkpoint) banked BEFORE a governor cut.
- **F2–F6 regression confirmation**: add the missing regression guards/tests (binary-carry, agent
  timeout, integrator, run_phase routing, concurrency) so they can't silently regress.

## The O2/O3/O4 redesign (from ws5veaolg — merge AFTER Batch 1, with corrections)
New orchestration ("hybrid-diag"), default-on but ablatable (`APEX_OMEGA_DIAG`, `APEX_OMEGA_PLAN_REVIEW`,
`APEX_OMEGA_PHASE0`); all OFF == hybrid-nogate byte-identical:
- **`ctx.diagnose()` (wave −1):** STAGE 1 zero-token AST import-graph pre-pass in `build_repo_map`
  (walk conftest/`__init__`/test bootstrap, resolve imports vs source, emit unresolved closure +
  `collects_cleanly` + addopts → tokenlessly catches pydantic `GenerateSchema` + `--memray`); STAGE 2
  1–3 read-only scouts classify the blocker into `DIAGNOSIS_SCHEMA {blocker_class, import_chain,
  must_implement_modules, suggested_first_fix, evidence}`, fact-checked vs the AST edges.
- **`review_plan()` (repurpose `goal_align_gate`):** fires at EVERY plan seam (decompose / phase-plan /
  rephase / repair-plan), grounded in the diagnosis (not just failing ids), **advisory + bounded:
  iters=1 per seam via a host-side per-seam counter, may rank/repair-plan-pre-execution, NEVER abort**;
  keep grounded-majority + ungrounded-downgrade + fail-open.
- **Synthetic Phase 0** = make-it-import/collect, with an **explicit execution-grounded EXIT**
  (`collects_cleanly` flips / errors==0 on a valid measurement), a **distinct sentinel** (not empty
  `acceptance_gold_ids`), its **own agent sub-budget**, the **coherent integrator enabled** (relax
  `coupled_plateau` gold≤0 guard using the import-depth secondary frontier), and
  `must_implement_modules` reconciled to decompose's namespace **by file overlap**.
- **Cardinal/replay:** diagnosis/review/Phase-0 are signals; only `ctx.select` on the full gold suite
  ACCEPTS; the Phase-0 pass branch must NOT touch `best_gold_passed`/Candidate/select; all LLM in
  `ctx.ask/ctx.signals`.

## Sequencing
1. Batch 1 (worktree.py + wal.py + context.py reconstruction) — interlocking, land together.
2. Batch 2 (O5, O4, F1) — mostly disjoint files, parallel-safe.
3. Batch 3 (apply-validation, best-partial checkpoint, F-regression tests).
4. Merge the ws5veaolg redesign onto context.py (serialize after Batch 1 — same planning seams,
   merge-conflict risk), re-run the full suite, re-confirm the babel/mimesis baseline.
5. Then ONE clean n=3 run (no mid-cell kills).

## Definition of Done (every issue addressed)
- [ ] Full suite green; new regression tests for O1/O5/O4/F1-F6/NEW-I1..I10 pass.
- [ ] Induced mid-cell-kill repro: resume → `carry_best()` returns the high candidate's diff;
      `partial_frontier == pre-kill best` (no 2220→13).
- [ ] No unlinked-high-score WAL entries reproducible; resume reuses prior scores (cache stable).
- [ ] `--memray`-class addopts stripped only when the plugin is absent (logged).
- [ ] pydantic-core pin persisted; fresh-venv bootstrap asserts the version.
- [ ] Diagnosis OFF ⇒ hybrid-nogate byte-identical (babel 5663 / mimesis 6156 reproduce); Diagnosis
      ON ⇒ pydantic synthesizes Phase 0 and makes collection progress.
- [ ] Cardinal Contract + journal-replay determinism preserved (no LLM in reduce; only select accepts).
