# Why Large Repos Fail — Definitive, Research-Grounded RCA

**Deciding analyst report.** Question: APEX-Ω solves EASY/MEDIUM commit0 repos (voluptuous, jinja)
but solved ZERO large repos in the n=3 A/B (63 cells: mimesis ~6159 tests, networkx ~5436,
pydantic ~5091, babel ~5663, minitorch ~230). This document enumerates EVERY failure mode, grounds
each root cause + fix in evidence and the external research, and separates FIXABLE GAPS (address
before the re-run) from FUNDAMENTAL LIMITS (a ~95-file from-scratch reimpl may exceed the budget).

---

## 0. Ground truth from the live run (`/tmp/omega_phase_ab_n3max`, 86 progress rows, verified)

| repo | difficulty | cells | solved | max pass% (progress) | max wall/cell |
|---|---|---|---|---|---|
| voluptuous | easy | 9 | **6** | 100% | 1,254 s |
| jinja | medium | 10 | **5** | 100% | 11,415 s |
| babel | medium\* | 10 | **0** | 0% | 16,247 s |
| minitorch | medium\* | 10 | **0** | 0% | 11,415 s |
| mimesis | medium\* | 13 | **0** | 0% | 20,092 s |
| networkx | hard | 20 | **0** | 0% | 22,938 s |
| pydantic | hard | 14 | **0** | 0% | 17,663 s |

**Total: 11/86 solved, ALL on the two small repos.** Every large repo = 0 solved despite 3-6 h
of wall per cell. (\* babel/minitorch/mimesis are labeled "medium" by the harness but are large
multi-thousand-test libraries; they behave as large repos here.)

The CONFOUND that dominates this run: it ran on the PRE-fix code AND under `/private/tmp/...`. The
two committed fixes (64a69c5 guard allow-list, 8e2b032 secondary frontier) are REAL (verified in
git) but do NOT cover this run's filesystem layout — see FM-1. So this run's zero-solve is
**confounded**, not a clean test of the orchestrator's large-repo ceiling.

---

## 1. RANKED, DE-DUPED FAILURE MODES (every problem, each tagged)

Ranking = (impact on large-repo solve rate) × (fraction of rollouts/cells affected), fixable gaps
above already-fixed and fundamental limits where impact is comparable.

### FM-1 — Guard allow-list still leaves ~91% of policy aborts FATAL on this run's `/private/tmp` layout  ·  **status: fixable_gap (partially fixed)**  ·  RANK 1
- **Evidence (verified):** 442 policy_violation agent rows across the 63 cells (re-counted live,
  exact match). All are `tokens=0` (whole rollout discarded). The committed fix (64a69c5) keys its
  SOFT downgrade on env-infra roots + `_WORKSPACE_POLICY_SYSTEM_READONLY_ROOTS`, which **explicitly
  excludes `/tmp`, `/private/tmp`, `/var/folders`** (verified at cli_backend.py:1289-1294). The live
  ladder ran under `/private/tmp/omega_phase_ab_n3max` — I confirmed
  `('/private/tmp/...').startswith(READONLY_ROOTS) == False`. So the dominant FP classes (sibling
  best-of-N worktrees, cross-cell, ladder-root traversal, cell-scoped codex venv/uv-cache) all live
  in the excluded region and **stay FATAL**. Quantified across the 442: ~401 (91%) stay fatal,
  only ~29-39 (7-9%) become SOFT.
- **Root cause:** the fix's allow-list is structurally misaligned with the FP population. It keeps
  sibling worktrees fatal "by construction" to preserve anti-cheat — but on best-of-N the SAME-task
  sibling worktrees ARE the bulk of benign FPs. The allow-list and the FP set are misaligned.
- **Fix:** re-scope from "env-infra roots" to "THIS cell's own roots are SOFT; only OTHER cells /
  planted upstream copies are FATAL." Pass the cell-root + this cell's worktree roots as soft-roots;
  reads under the cell's own `worktrees/`, `cells/`, `journal/`, `orchestrator/` become SOFT (they
  cannot false-solve: shadow + no-network + gold-id acceptance all hold). Keep FATAL only: (a) a
  DIFFERENT cell's root, (b) `_upstream/_wheel/_restore/_spec` markers (G1), (c) arbitrary /tmp
  source copies outside any cell. Converts ~305 of 401 without weakening anti-cheat.
- **Expected impact:** recovers ~33% of pydantic rollouts, 5-26% networkx, 9-21% minitorch — the
  single largest lever on large repos, because every recovered rollout is a whole repair wave that
  is currently discarded at tokens=0.
- **Research basis:** SANDBOX-NOT-PROMPT deny-don't-kill (context.py:170-178); ProgramBench's
  structural-isolation philosophy (arXiv:2605.03546 §2.2) — isolation should be location-based, not
  prompt/marker-based; the shadow + no-network make a benign read incapable of a false solve.

### FM-2 — COLLECTION-COLLAPSE: a single load-bearing import bug errors the ENTIRE gold suite → frozen frontier, no gradient  ·  **status: fixable_gap (signal) + fundamental (the work itself)**  ·  RANK 2
- **Evidence (verified live):** `hybrid-nogate__pydantic__s0` eval records: `errors=5091 passed=0
  collected=None rc=4`, repeated x9 — the suite never collects a single test; the error count is
  flat 5091→5091. babel similarly cannot collect until a large fraction is implemented.
- **Root cause:** pydantic/babel have ONE import that gates all ~5091/5663 tests at collection; the
  gold-pass COUNT frontier stays pinned at 0 until that fix lands, so an arm doing real
  module-implementation work is indistinguishable from a dead arm. The governor audit confirms a
  collection collapse is a VALID `gold_passed=0`, not indeterminate — so it earns zero credit.
- **Fix (signal half, already landed 8e2b032):** secondary frontier on min-collection-errors resets
  patience clocks when more of the suite begins to collect — DOES NOT bank a gold solve. Verified in
  git. **Gap that remains:** on pydantic the error count was genuinely FLAT (5091→5091) because the
  agents were largely sandbox-BLOCKED (FM-1), so the secondary frontier had nothing to credit. Until
  FM-1 is fixed, the secondary frontier is untested on the repo it was built for. Additional fix:
  add a "collection-first" phase that targets the load-bearing import chain BEFORE fanning out
  modules (decomposition gap, FM-6).
- **Expected impact:** keeps a genuinely-progressing collection-collapse repo alive instead of
  cutting it; necessary-but-not-sufficient (the work still has to land the import fix).
- **Research basis:** GOVERNOR_AUDIT.md "precise missing progress signal"; ProgramBench notes
  large libs "remain out of reach" partly because evaluation gives no partial-credit gradient until
  a behavioral threshold is crossed.

### FM-3 — BABEL DEGENERATE PLAN: hybrid arm scored real work (passed=5024) against a `gold_total=10` inventory  ·  **status: fixable_gap**  ·  RANK 3
- **Evidence (verified live, a NEW finding this audit surfaced):** `hybrid__babel__s0`
  `phase_checkpoint.json` has `gold_total=10` while `converge__babel__s0` and
  `hybrid-nogate__babel__s0` both have the correct `gold_total=5663`. Critically, the babel eval
  artifacts show `hybrid__babel__s0` reached **passed=5024** and `converge__babel__s0` reached
  passed=1291 — i.e. babel is COLLECTABLE and the orchestrator did large amounts of real work, but
  the hybrid arm's degenerate `gold_total=10` inventory mis-scored it. Only s0 of the hybrid arm is
  affected (s1 = 5663 correct); it is an intermittent plan-construction edge case, NOT every cell.
- **Root cause:** a `plan_phases` / expected-id construction path in the phase-planner (hybrid) arm
  can produce a tiny degenerate gold inventory (10 ids) instead of resolving the full 5663 expected
  gold ids — likely an empty/partial expected-id resolution that falls back to a stub count.
- **Fix:** assert `gold_total` matches the repo_map's expected-id count at plan-construction time;
  fail-loud (do not silently proceed with a degenerate inventory). Resolve expected ids from the
  same source for all arms (the converge path already gets 5663). Add a regression test that a
  plan's gold_total equals the gold scoring contract's id count.
- **Expected impact:** babel is the closest large repo to a solve (5024/5663 in one arm). A correct
  inventory + acceptance-checkpointing could convert babel from "0 solved (mis-scored)" to a real
  partial/full solve — potentially the FIRST large-repo solve.
- **Research basis:** COMMIT0_GOLD_EVALUATION_CONTRACT (exact-id scoring); SWE-bench-style
  expected-id resolution must be the single source of truth across arms.

### FM-4 — HUGE-CONTEXT REPAIR: loop-until-dry pastes a 100KB+ carry diff + thousands of node-ids → hits the hard timeout, diff never committed (tokens=0)  ·  **status: fixable_gap**  ·  RANK 4
- **Evidence:** networkx s1 WAL: 13 `timeout` finalizations + 58 policy_violations + 71 `tokens=0`
  failures in a single cell (393 rows). The MEMORY/brief documents the 112KB carry diff + 263KB
  diff that hit the 3000s hard timeout at xhigh effort with tokens=0 (diff never committed). The
  repair prompt scales with the failing-suite size, so on a 5000-test repo it is enormous.
- **Root cause:** the repair stage feeds the full carry diff + the full failing-node-id list into a
  single agent turn; on large repos this blows the context/time budget before any output is
  produced, and a timeout discards the in-progress diff (no partial commit).
- **Fix:** (a) cap/summarize the repair context — feed a bounded, prioritized slice of failing ids
  (e.g. top-N by module, collection errors first), not the whole list; (b) chunk the carry diff by
  module rather than one monolithic paste; (c) commit-on-progress so a subsequent timeout does not
  zero the work (acceptance-checkpointing exists for SOLVES; extend to in-progress diffs);
  (d) lower effort/turn budget for the repair stage on large repos so a turn finishes.
- **Expected impact:** stops the largest single source of wasted wall on networkx/mimesis (multi-
  hour cells that timeout with tokens=0); turns dead repair waves into committed incremental progress.
- **Research basis:** long-context degradation / "lost in the middle"; agent-loop best practice of
  bounded, prioritized context windows; ProgramBench's per-task budget framing.

### FM-5 — CROSS-CELL PROCESS-TREE KILL: a concurrent OTHER cell's find/grep is swept into THIS cell's ps-tree audit → fatal abort  ·  **status: fixable_gap**  ·  RANK 5
- **Evidence:** 63 of 442 aborts (14.3%) name a DIFFERENT cell in the error (e.g.
  `converge__minitorch__s1` killed for a `find` running from `hybrid__networkx__s1/cells/...`).
  100% of 62 done-cells temporally overlap ≥1 other cell. `_collect_process_tree_entries`
  (cli_backend.py:9526) runs host-wide `ps -axo pid,ppid,command` and walks PPID.
- **Root cause:** host-wide ps + PPID walking cannot isolate one cell's subprocess tree from a
  concurrent cell's: orphaned grandchildren reparent (PPID→1/subreaper), PIDs are reused, the scan
  is non-atomic. No per-cell process-group/cgroup/env-tag boundary.
- **Fix:** tag every cell's agent subprocess (env var `APEX_CELL_ID` read from `/proc/<pid>/environ`
  on Linux; `os.setpgrp` at launch + filter by PGID on macOS) and filter the audit to ONLY the
  cell's own PGID/CELL_ID. A foreign cell's find can then never enter this cell's audit.
- **Expected impact:** eliminates all 63 cross-cell kills (~14% of aborts) — pure recovery, no
  anti-cheat cost.
- **Research basis:** standard "use PGID/cgroups, not reconstructed PPID trees" process-isolation
  guidance; lwn.net subreaper / Linux orphan-reparenting semantics ("ps trees are a snapshot of a
  moving target").

### FM-6 — NO TRUE DECOMPOSITION FOR COLLECTION-COLLAPSE REPOS: fan-out implements modules before the suite can collect  ·  **status: fixable_gap**  ·  RANK 6
- **Evidence:** dynamic-workflow-rebuild MEMORY note (3 verbatim defects: no decomposition, near-
  solves discarded, repair_iters=0 dead); pydantic fanout produced 5 module diffs (~112KB) but the
  conftest ImportError only advanced ~4 import layers — never reached collection.
- **Root cause:** the decompose→fan-out→reduce plan treats a collection-collapse repo like an
  independent-module repo. On pydantic the FIRST necessary milestone is "make the suite collect"
  (fix the load-bearing import chain), which is a serial dependency, not a parallel fan-out.
- **Fix:** add a "collection-gate" phase: before module fan-out, run a focused agent whose ONLY goal
  is to make `pytest --collect-only` succeed (drive collection-errors→0), then fan out. Order
  modules by the import dependency graph so load-bearing modules land first.
- **Expected impact:** converts pydantic/babel from "frozen frontier" to "collecting, then
  gradiented." Combined with FM-1 (so agents can actually run) this is what could yield the first
  pydantic gradient.
- **Research basis:** Agentless / structured-decomposition literature (localize→repair); ProgramBench
  behavioral-threshold framing; the dependency-ordered build is standard for monolithic packages.

### FM-7 — DISABLED OS SANDBOX makes the strategy-side guard the SOLE isolation layer, so every guard FP = total rollout kill (tokens=0)  ·  **status: fixable_gap (severity policy)**  ·  RANK 7
- **Evidence:** codex retry-diagnostic stderr: `--dangerously-disable-osx-sandbox flag is enabled`.
  All 442 aborts are tokens=0 (whole rollout discarded; in-progress diff lost). Per-cell pv%:
  converge__pydantic__s2 40%, hybrid__pydantic__s1 33%, hybrid-nogate__networkx__s1 26%.
- **Root cause:** OS sandbox off (Meta gateway / macOS TCC) ⇒ the guard is load-bearing AND its
  failure mode is a hard fatal raise that zeroes the rollout, contradicting the stated deny-don't-
  kill policy for the 91% of FP classes not on the allow-list.
- **Fix:** make the DEFAULT severity for the read-only discovery class (find/grep/rg/ls/cat/head/
  tail/du/tree/sed/awk) SOFT (deny-the-read, log telemetry, CONTINUE), reserving FATAL for the
  narrow structural cheat set (different-cell roots + G1 markers + mutating writes into the
  worktree). Invert the allow-list into a deny-list for the read-only class. (This subsumes FM-1 and
  FM-8 cleanly; the shadow + no-network + gold-id acceptance keep false-solves impossible.)
- **Expected impact:** turns the dominant tokens=0 loss into deny-and-continue across the suite;
  preserves the T1 telemetry already wired in v1_executor.
- **Research basis:** SANDBOX-NOT-PROMPT (context.py); ProgramBench no-internet + shadow as the
  structural guarantee (arXiv:2605.03546) — the kill adds zero marginal anti-cheat benefit because
  telemetry already fires on every escape.

### FM-8 — sed/awk SCRIPT arg mis-parsed as a path operand → resolves outside workspace → FATAL (STILL PRESENT)  ·  **status: fixable_gap**  ·  RANK 8
- **Evidence (verified by executing CURRENT code):** `_command_path_operands('sed',['sed','-n',
  '/^$/d','file.txt'])` → `['/^$/d','file.txt']`; `('awk',['awk','/pattern/{print}','f'])` →
  `['/pattern/{print}','f']`; `('awk',['awk','/^tests//','x'])` → `['/^tests//','x']`. The program
  text is returned as a path operand. `_command_path_operands` (cli_backend.py:9663) has branches
  for find and rg/grep/ag/fd but NO sed/awk branch. A parallel bash-payload handler at line 10129
  DOES handle awk/sed pattern-skipping (`options_with_values={'awk':{'-F','-v'},'sed':{'-e'}}`,
  verified), but the process-tree direct-monitored path at 11248 uses `_command_path_operands`,
  which does not. Live integrity logs show 20 awk + 6 sed escapes (e.g. `awk targeted /`).
- **Root cause:** missing sed/awk handling in `_command_path_operands` — the mandatory first
  positional (the program text, when no -e/-f) is treated as a path. POSIX: awk's/sed's first
  non-option arg is the program, never a path.
- **Fix:** add a sed/awk branch to `_command_path_operands` mirroring the 10129-10194 logic: when no
  -f progfile / -e script, skip the first non-option positional; treat the rest as files; for awk
  treat name=value as assignment, not a path. Reuse the existing maps.
- **Expected impact:** removes 26-31 sed/awk FP aborts (~7% of aborts).
- **Research basis:** POSIX man awk.1p / sed; WORKSPACE_GUARD_ANALYSIS.md §4 explicitly lists this
  as out-of-scope/unfixed.

### FM-9 — G1 upstream markers require a TRAILING SLASH → bare-dir operand (`find /tmp/mimesis_wheel`) misses the cheat marker  ·  **status: fixable_gap (latent, load-bearing AFTER FM-1)**  ·  RANK 9
- **Evidence:** `_WORKSPACE_POLICY_UPSTREAM_REFERENCE_MARKERS = ('_upstream/','_upstream.',
  '_wheel/','_restore/','_spec.txt')` (cli_backend.py:1297, verified). `_path_resolves_to_upstream_
  reference_copy('pydantic_spec.txt')→True`, `('mimesis_wheel')→False` (no trailing slash). Today
  masked because /private/tmp stays fatal anyway; becomes a real anti-cheat HOLE once /tmp-region
  reads are softened by FM-1/FM-7.
- **Root cause:** marker matching requires `/`-suffixed forms, so the bare directory operand evades.
- **Fix:** also match the bare segment form (a path segment equal to or ending with `_upstream`/
  `_wheel`/`_restore`) plus the existing `/`-suffixed forms and `_spec.txt`. Add a bare-dir unit
  test. **Do this BEFORE/with FM-1/FM-7** so genuine reference copies stay fatal once /tmp FPs soften.
- **Expected impact:** zero solve-rate change now; PREREQUISITE for safely landing FM-1/FM-7 without
  opening a cheat hole.
- **Research basis:** SWE-bench #465 (reference-leak hardening); ProgramBench no-build-artifact
  isolation — the anti-cheat carve-out must be robust to operand form.

### FM-10 — RELAUNCH COST: hard cells relaunch and run 5-6 h each, frontier slowly climbing, never solving  ·  **status: fixable_gap (efficiency) / fundamental (if budget < repo)**  ·  RANK 10
- **Evidence:** max wall/cell: networkx 22,938 s (6.4 h), mimesis 20,092 s, pydantic 17,663 s,
  babel 16,247 s. networkx frontier rises 780→1763 (GOVERNOR_AUDIT) but never reaches the full
  5436. The per-cell wall was raised to 86400 s (24 h) (commit 5677421) specifically so heavy cells
  aren't truncated — so cells run very long without solving.
- **Root cause:** large repos consume the entire budget climbing a frontier that asymptotes below
  100%. Partly efficiency (wasted on FP aborts FM-1/5/8, dead repair FM-4) and partly the
  fundamental from-scratch-at-scale ceiling (FM-12).
- **Fix:** after the efficiency fixes (FM-1/4/5/7/8) recover the wasted rollouts, set a
  progress-aware relaunch policy: relaunch ONLY while the (gold OR secondary collection) frontier is
  still rising at a minimum rate; stop a cell whose frontier has truly asymptoted (per the
  progress-only stop policy, task #10). This re-allocates compute from asymptoted cells to ones
  still climbing.
- **Expected impact:** large compute-efficiency gain; better expected solves per wall-hour. Does
  NOT by itself solve a repo whose true requirement exceeds any feasible budget (FM-12).
- **Research basis:** SPFG+ progress-frontier governor (d60ad02 / GOVERNOR_AUDIT); ProgramBench
  per-task budget findings.

### FM-11 — FETCH-MONOCULTURE hypothesis FALSIFIED: ~0 genuine fetch attempts; the "escape on every pydantic agent" signal is the FP guard, not real fetching  ·  **status: already_fixed / non-issue (re-label only)**  ·  RANK 11
- **Evidence (verified live):** integrity signals across 39 logs = `sandbox_escape 410,
  test_modification 340, fetch_attempt 2`. sandbox_escape command histogram = `find 373, awk 20,
  rg 8, sed 6, tee 2, tar 1` — ALL repository-discovery commands, ZERO pip/curl/wget/git-clone/uv.
  Grepping every signal detail for pip/curl/clone/uv = 0 genuine fetch substrings. By repo,
  escapes: networkx 159, pydantic 150, minitorch 48, mimesis 34 — i.e. they track suite SIZE /
  discovery volume, not "famous-package fetch instinct." The brief's "sandbox_escape on every
  pydantic agent" is true by count but is the workspace-discovery FP (FM-1), not real fetches.
- **Root cause:** the "fetch-monoculture" framing conflated the integrity `sandbox_escape` signal
  (fires for ANY policy_violation / out-of-root text) with actual fetch behavior.
- **Fix:** no fetch-specific guard work needed. Re-label: distinguish a `discovery_fp` signal class
  from `fetch_attempt` so future runs don't mistake guard FPs for cheating. Add action-grounded
  fetch telemetry (parse the codex command/tool-call stream for git clone / pip install / curl /
  site-packages reads) so the ledger is grounded in actions, not error text (a no-network sandbox
  blocks a fetch SILENTLY, so the current error-text classifier is structurally blind).
- **Expected impact:** none on solve rate; cleans up the diagnosis so the re-run is apples-to-apples.
- **Research basis:** ProgramBench open-internet ablation (cheating 20-36%, source-lookup 79-95%)
  shows fetch WOULD dominate IF network were on — but the agent egress is OFF here, so it's a
  non-issue; ProgramBench action-classification taxonomy (arXiv:2605.03546 §4.1).

### FM-12 — FUNDAMENTAL: from-scratch reimplementation of a ~95-file library may exceed any feasible per-cell budget  ·  **status: fundamental_limit**  ·  RANK 12
- **Evidence:** ProgramBench (the direct from-scratch-reimpl successor to commit0) reports frontier
  models solve essentially 0% of large from-scratch reconstructions; best is ~3% of tasks at ≥95%
  tests, and large libs (FFmpeg/php-src) "remain out of reach." Our networkx frontier asymptotes at
  1763/5436 even when the suite collects and the governor keeps it alive — a ceiling, not a cut.
- **Root cause:** the task is to reconstruct thousands of interdependent behaviors from tests alone;
  the joint probability of getting all load-bearing modules + the import graph + edge cases right in
  one budget is very low at current capability, independent of orchestration.
- **Fix:** NONE that makes a 95-file repo solvable within budget by orchestration alone. Mitigations
  (FM-2/4/6/10) improve the gradient and partial credit and CAN cross the threshold for the
  borderline repos (babel at 5024/5663, networkx climbing). Honest framing: report partial-credit
  (max gold-pass frontier) for large repos, not just binary solved; do not expect pydantic/mimesis/
  networkx full solves from this orchestrator at this capability tier.
- **Research basis:** ProgramBench (arXiv:2605.03546) §5.1; Commit0 paper (body-stripped from-scratch
  is the hardest commit0 regime); memorization is real-but-small at low pass rates so it does NOT
  rescue a 95-file repo.

### FM-13 — test_modification FP flags legit impl edits to SHIPPED test-helper modules (minitorch/testing.py)  ·  **status: fixable_gap (telemetry FP, low impact)**  ·  RANK 13
- **Evidence:** 340 test_modification signals: minitorch 139, mimesis 84, networkx 59, jinja 55,
  pydantic 2, babel 1. Samples repeatedly flag `minitorch/testing.py` and
  `pytest_factoryboy/__init__.py` — these are SHIPPED library modules the agent must implement, not
  hidden gold test files.
- **Root cause:** the test_modification classifier flags edits to any path matching a test-ish name,
  including shipped testing-utility modules that are part of the package under implementation.
- **Fix:** scope the test_modification signal to the GOLD test files (the hidden expected-id suite
  paths), not to package source modules that merely contain "test" in the name. This is telemetry-
  only (does not gate acceptance) so impact on solve rate is low, but it pollutes the integrity
  ledger and could mislead a future gate.
- **Expected impact:** cleaner integrity telemetry; no direct solve-rate change.
- **Research basis:** ProgramBench tests-never-revealed / implementation-agnostic acceptance — only
  the hidden gold suite is the protected surface.

---

## 2. FIXABLE GAPS vs FUNDAMENTAL LIMITS (the honest split)

**FIXABLE before the re-run (the zero-solve is dominated by these, not by capability):**
FM-1, FM-2(signal), FM-3, FM-4, FM-5, FM-6, FM-7, FM-8, FM-9, FM-10(efficiency), FM-11(relabel),
FM-13. Together these are recovering wasted rollouts (FM-1/5/8: ~91%+14%+7% of 442 aborts),
un-masking real progress (FM-3 babel 5024/5663 mis-scored; FM-2 frozen frontier), and stopping
budget waste (FM-4/10). The run did NOT cleanly test capability because the agents were largely
blocked or mis-scored.

**FUNDAMENTAL (not a bug — say so honestly):** FM-12. A ~95-file library reconstructed from tests
alone is at/near the frontier ceiling (ProgramBench ~0% on large from-scratch). Babel (5024/5663),
networkx (climbing), and minitorch (small at 230 tests) are the realistic large-repo solve
candidates AFTER the fixes; pydantic/mimesis full solves should be treated as out-of-reach and
reported as partial-credit frontiers.

---

## 3. REMAINING FIXABLE GAPS — ordered for an implementation workflow (priority 1 = highest)

1. **Cell-scoped guard severity (FM-1 + FM-7 + FM-9 together).** Make read-only discovery SOFT by
   default; FATAL only for a DIFFERENT cell's root, G1 upstream markers (also matching bare-dir
   form), and mutating writes into the worktree. Files: `apex/core/cli_backend.py`
   (`_path_is_agent_runtime_infra`, `_process_tree_workspace_policy_violation`, the readonly-roots /
   upstream-markers constants ~1289-1299, dispatcher ~8996/9180), `apex_omega/executor/v1_executor.py`
   (thread cell-root). Effort: M. Impact: recovers ~91% of 442 aborts. **Land FM-9 in the SAME diff.**
2. **sed/awk operand parsing (FM-8).** Add a sed/awk branch to `_command_path_operands`
   (cli_backend.py:9663) mirroring the 10129-10194 logic. Files: `apex/core/cli_backend.py` +
   `tests/test_workspace_policy_classification.py`. Effort: S. Impact: ~7% of aborts.
3. **Babel degenerate-plan assertion (FM-3).** Assert `gold_total == repo_map expected-id count` at
   plan construction; resolve expected ids from one source for all arms; fail-loud on a degenerate
   inventory. Files: `apex_omega/autogen/architect.py` (plan_phases / expected-id resolution),
   `apex_omega/eval/*` (gold contract), + regression test. Effort: M. Impact: un-masks babel
   5024/5663 — the closest large-repo solve.
4. **Cross-cell process-tree isolation (FM-5).** Tag agent subprocesses per cell (PGID via
   `os.setpgrp` on macOS / `APEX_CELL_ID` env on Linux) and filter `_collect_process_tree_entries`
   to the cell's own group. Files: `apex/core/cli_backend.py` (`_collect_process_tree_entries` ~9526
   + launcher), `apex_omega/executor/v1_executor.py` (set PGID/env at launch). Effort: M. Impact:
   eliminates 63 cross-cell kills (~14%).
5. **Bounded repair context + commit-on-progress (FM-4).** Cap/prioritize the failing-id list and
   carry diff fed to the repair agent; chunk by module; commit in-progress diffs so a timeout does
   not zero work; lower repair effort/turn budget on large repos. Files:
   `apex_omega/autogen/templates.py` (repair/loop-until-dry prompt), `apex_omega/autogen/context.py`
   (carry-diff assembly, checkpointing), `apex_omega/engine/*` (budget per stage). Effort: M-L.
   Impact: stops multi-hour tokens=0 timeouts on networkx/mimesis.
6. **Collection-gate decompose phase (FM-6 + FM-2 work-half).** Add a phase whose sole goal is to
   make `pytest --collect-only` succeed before module fan-out; order modules by import-dependency.
   Files: `apex_omega/autogen/architect.py` (phase plan), `apex_omega/autogen/context.py`
   (phase sequencing). Effort: L. Impact: gives pydantic/babel a gradient (depends on #1 so agents
   can run).
7. **Progress-aware relaunch / stop policy (FM-10, ties to task #10).** Relaunch only while a
   frontier (gold OR secondary collection-error) is still rising at a min rate; stop asymptoted
   cells. Files: `apex_omega/engine/governor.py`, `apex_omega/engine/frontier.py` (relaunch gate).
   Effort: M. Impact: compute efficiency / more solves per wall-hour.
8. **Integrity-telemetry cleanups (FM-11 relabel + FM-13 scoping).** Add a `discovery_fp` class and
   action-grounded fetch telemetry; scope `test_modification` to the gold suite paths only. Files:
   `apex_omega/autogen/context.py` (`classify_attempt_integrity` ~181-204). Effort: S. Impact:
   clean apples-to-apples diagnosis on the re-run (no solve-rate change).

**Re-run gate:** land #1+#2+#9(in #1) FIRST (they unblock the agents), then #3 (un-masks babel),
then re-run the large repos. #4-#8 improve efficiency/gradient and can land in parallel. Report
large repos by partial-credit frontier (max gold-pass), not binary solved, per FM-12.
