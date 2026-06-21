# Run Issues Ledger — comprehensive commit0 re-run (/tmp/omega_comprehensive)

Live ledger of errors/issues surfaced during the eval, with status. Appended as new ones appear.
Branch `feat/phase-planner-hybrid`. Run started 2026-06-20.

## FIXED during the run (committed + pushed)

| # | Issue | Symptom | Fix | Commit |
|---|---|---|---|---|
| F1 | apex-venv `pydantic-core` mismatch | every cell crashed at prep (`import datasets` → SystemError 2.20.1 vs 2.46.1) | `uv pip install pydantic-core==2.46.1` (env, not code) | — (env) |
| F2 | **Binary-carry not re-applyable** | babel agents emit binary `.dat`; plain `git diff` records "Binary files differ" (no content) → `git apply` rejects → every carry re-apply CONFLICTS → ralph/converge collapse (converge/babel 2, ralph 1165) | `git diff --binary` at `_git_diff` + `_merged_diff` | `7c9ba08` |
| F3 | **Per-agent timeout abandons productive agents** | pydantic/networkx (API/reasoning-bound) agents need >50min; the 3000s hard cap abandoned them (heartbeat_timeout, work lost) | env-overridable `APEX_OMEGA_AGENT_TIMEOUT_{HARD,…}`; run uses HARD=7200 | `15f023b` |
| F4 | merge/reduce sheds work on coupled repos | converge/babel plateaus at 937 (~50 hunks/module rejected) ≪ ralph | v2 coupling-triggered coherent integrator (coupled_plateau → ralph_loop seeded by carry_best) | `7f2741e` |
| F5 | v2 integrator not in the phased path | hybrid arms (run_phase) ran converge-v1-per-phase (0 integrator switches) | wire the router into `run_phase` too | `0afde36` |
| F6 | Concurrency thrash | C=12 → load ~28 → eval-burst slowdown; (the agent heartbeats were actually F3, not thrash) | config: C=8 × within-cell 6 | — (config) |

## OPEN / to address (priority order)

- **O1 — SILENT CANDIDATE LOSS on resume (robustness, HIGH before the final run).**
  `converge/networkx` had a candidate scoring **2220** but its diff was unlinked (`fs_diff_ref=False`,
  likely a floor-probe candidate) → `select`/`carry_best` skip diff-less candidates → cell reported
  **13**. Aggravated by my ~4 mid-cell kill/restart cycles (resume corrupts the journal↔diff link on
  big cells). Action: (a) make diff-banking/resume robust so a high candidate's diff is never lost;
  (b) for the FINAL paper run, do ONE clean run with NO mid-cell kills. Until then, distrust per-cell
  frontiers on heavily-restarted big cells (trust SOLVES + unambiguous results).

- **O2 — Goal gate is net-negative (decision: DROP it).**
  hybrid-WITH-gate babel 2486 (64 agents) vs hybrid-nogate babel SOLVED (26 agents). The gate's
  adversarial review mis-steers + burns budget. Action: default `APEX_OMEGA_GOAL_GATE=0` (or remove the
  gate from the recommended config). It's a "bad path" — flag, don't patch.

- **O3 — pydantic collection-collapse (fundamental; candidate fix = FM-6).**
  `conftest.py: from pydantic import GenerateSchema` → nothing collects until a huge fraction is
  implemented; ~0 on all arms. Candidate (speculative): FM-6 collection-gate phase (a phase dedicated
  to making the suite import/collect before targeting passes). Defer until after the directional run.

- **O4 — pydantic `--memray` addopts (minor harness).**
  pydantic's pyproject addopts pass `--memray`; pytest-memray isn't loaded under
  PYTEST_DISABLE_PLUGIN_AUTOLOAD → `unrecognized arguments: --memray` on some paths. Secondary to O3
  (conftest fails first). Action: strip uninstallable plugin addopts (or load the plugin) in the eval
  pytest command. Existing `_COMMIT0_PYTEST_OPTION_PLUGIN_PACKAGE_MODULES` map is present but not
  taking effect on this path.

- **O5 — Resume doesn't preserve in-cell progress (related to O1).**
  The `--binary` change busts per-candidate score-cache keys, so resumed cells re-run from floor-probe
  (lost prior frontiers). Acceptable for iteration; for the final run, either accept fresh-run cost or
  fix cache-key stability + diff-banking so resume is lossless.

## Operational notes
- Do NOT mid-cell kill/restart the run (causes O1 corruption). All fixes are landed + stable now.
- Monitor: `/tmp/omega_monitor.py` (delta-based: prep-crash / carry-conflict / infra-spike / load /
  disk / runner-death / catastrophic-collapse / per-cell completion).

## ALL OPEN ITEMS ADDRESSED 2026-06-21 (branch feat/phase-planner-hybrid; suite 445 green)
Implementation of every O1-O5 + NEW-I1..I10 + F1/O4 item from the comprehensive fix workflow +
the O2/O3/O4 redesign is committed + pushed:
- O1/NEW-I2/NEW-I5 (9880b90): durable candidate banking (kind="candidate", content-addressed diff)
  + _restore_candidates_from_journal on resume -> carry_best/select/reduce see the full frontier
  (the 2220->13 loss). +Journal.committed_entries. test_candidate_resume.
- NEW-I1/I4/I8/I9 (8f6542d): worktree resume-hardening (dead-pid lock reclaim, hard stale-worktree
  reclaim/fail-loud) + apply conflict-marker guard (apply_diff/apply_diff_partial). test_worktree_resume.
- NEW-I6 (e6eed52): bank best-partial before a governor cut (synchronize cut with banking).
- O4 + F1 + O1-journal-banking (c00c1ef): --memray-class addopts strip when plugin absent;
  pydantic-core==2.46.1 pin + loud preflight; Journal.ensure_diff_linked + commit-stores-blob-first.
- O2/O3/O4 REDESIGN (3929acd): ctx.diagnose() (zero-token AST collection pre-pass + fact-checked
  scouts), ctx.review_plan() (advisory, bounded, diagnosis-grounded plan review at every seam, never
  abort), synthetic Phase 0 (make-it-collect). Gated APEX_OMEGA_DIAG/PLAN_REVIEW/PHASE0; ALL OFF ==
  hybrid-nogate byte-identical. test_diagnose_ast + test_diagnose_redesign.
- O5 (score-cache key on code/write-tree identity): DEFERRED — resume-efficiency only; the planned
  clean run has no mid-cell kills so cross-format cache staleness does not occur. O1 makes resume CORRECT.
- O2 (drop goal gate): the "old best" arm runs APEX_OMEGA_GOAL_GATE=0 (hybrid-nogate); the new arm
  runs the redesign gates on.
NEXT: 2-arm clean eval — hybrid-diag (DIAG+PLAN_REVIEW+PHASE0 on, gate off) vs hybrid-nogate (all off).

## RUN STOPPED 2026-06-21 at 20/105 (15 scored + 5 v1-ref) — enough signal for next-phase dev
Archived (lightweight): runs/comprehensive_v2/ (progress + per-cell reports/checkpoints/narration/wal).
VERDICT (seed-0): hybrid-nogate (phase planner, gate OFF) is the strongest arm — SOLVED babel
(5663/5663, 26 agents) + near-solved mimesis (6156/6159); beats flat converge (babel 937, mimesis
3557) and ralph (babel 5648 near but mimesis 0). Goal gate net-negative (O2). pydantic fundamental
(O3). converge/networkx=13 is the O1 resume-corruption artifact (real ~2220). Open items O1-O5 are the
next-phase work (esp. O1 diff-banking/resume robustness + O2 drop-gate before a clean n=3).
