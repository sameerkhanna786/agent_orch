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
