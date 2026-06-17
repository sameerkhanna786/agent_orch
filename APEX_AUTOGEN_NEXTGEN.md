# APEX-Ω Autogen Next-Gen — Design, Plan & Honest Verdict

> Produced by a 44-agent workflow (map x7 + forensics x5 + research x8 -> 4 design proposals -> judged -> synthesized -> 12 adversarial validations -> impl-plan + completeness critic + calibrated final). Every load-bearing claim was ground-truth-verified against the source.

## Executive summary

DIAGNOSIS (ground-truth verified in code). The autogen arm's clean-run failures are dominated by TWO harness bugs that make the execution gate score correct code as zero, plus a fetch-shortcut monoculture and ranking variance. I verified every load-bearing claim against the actual source and CONFIRM the validators' critiques over the original design's prose:

- P0.1 "evals are serialized" is FALSE: score_fn runs inside solve_attempt thunks on a ThreadPoolExecutor(max_workers=min(16,cpu-2)) (runtime.py:59,180) and reuses a SHARED env dict (commit0_autogen.py:249). A shared-.pth mutation is a race -> the original design's marquee fix is concurrency-unsafe.
- §4.1 anti-fetch premise is FALSE: ScopedTask.internet already defaults False (types.py:126), solve_attempt never sets it, and _internet_launcher_args returns [] for internet=False (cli_backend.py:11675) — it only OMITS the internet flag, never adds a network DENY. The seatbelt profile is "(allow default)" with file-only denies (cli_backend.py:3634), and in host_cli_read_jail mode codex launches with --dangerously-bypass-approvals-and-sandbox (11577), disabling its own network-enforcing sandbox. So pip-install-the-package is physically unblocked today and the §4.5 "jail fetch-attempt flag" DOES NOT EXIST.
- apply_diff is bare `git apply` with NO --3way and returns False silently (worktree.py:129).
- The codebase ALREADY has the proven per-tree editable-reinstall path the design ignored: _verify_editable_target_inside_repo (commit0_benchmark.py:14033).
- Budget.total=None -> can_start() always True (budget.py:53): the agent cap is the SOLE spend brake, so dropping the max-agents pin is a runaway risk on pydantic.
- scout overrides static difficulty (architect.py:35 "scout overrides the static proxy") — the exact cause of voluptuous's 4x cost.

THE DESIGN (revised: Ladder-of-Lineages, gaps closed). Keep the spine's load-bearing primitives (failing-test surfacing, repair forking parent diff, difficulty-gated escalation, structural in-workspace floor lineage, monotone downgrade-only refuters) but: (1) make P0.1 race-free via per-call process-local PYTHONPATH for src-layout, gated behind a flat-layout no-op fast path, with a loud worktree-resolution assertion AND a fallback to the existing _verify_editable_target_inside_repo reinstall; land P0.1 STANDALONE and re-run jinja under today's orchestration first to prove the gate fix alone flips it; (2) re-scope anti-fetch to the genuinely-buildable teeth — a NEW jail PyPI/clone command detector + post-session dist-info provenance check + a concrete allowlist that PERMITS the agent's own pip install -e . / pytest / compileall and the legit pydantic-core dep while denying target-package acquisition by name; correct the false internet premise; (3) add a collection-error-count climbing metric + pre-eval compileall/import/collect-only smoke gate so collection-blocked pydantic isn't abandoned by a pass_rate plateau; add a broad-deficit detector routing low-pass-rate/high-error-diversity (mimesis) to re-implement/decompose, not 3-iter in-place repair; (4) size per-attempt budget against observed single-solve cost (~1921s) and decouple from the cell cap; (5) add --3way + indeterminate-on-apply-failure; (6) restore a hard Budget.total token ceiling and specify raise_cap delta/strike thresholds; (7) gate the difficulty decision on STATIC build_repo_map difficulty for R0-R2 so voluptuous routes to 1 agent; (8) gate the entire anti-shortcut layer behind a no-op fast path for easy/flat repos with a CI floor-protection blocker (voluptuous: agents<=1 AND solved).

CONFIDENCE VERDICT (calibrated, honest). The harness diagnosis is forensically exact and the structural floor is sound, but this is a DESIGN+PLAN deliverable, NOT a validated run — no code is written and no eval has been re-run. The honest near-term outcome on pydantic and mimesis is most likely correct-ABSTAIN (gate honest, fetch blocked, but genuine from-scratch reimplementation unproven), NOT solve. voluptuous is preserve+cheapen (high). jinja is one-fix-from-solved IF P0.1 is made race-safe. Overall confidence that this SOLVES all four runnable repos reliably: LOW (~30%). Confidence it makes every repo honestly-measured and preserves/improves the voluptuous solve while not regressing: high.

## Confidence verdict (calibrated, honest)

**Overall confidence it RELIABLY solves all 4 runnable repos: 30%**

| repo | confidence | one-line |
|---|---|---|
| voluptuous | 88% | Already solved on the autogen arm today (winner a0, score 1.0); flat-layout so P0.1 is a verified NO-OP via the _detect_src_pkg fast path; gate has no --memray so P0.2 inapplicable. |
| jinja | 60% | The strongest case in principle: failure is a pure harness bug (base-stub editable shadow), and 3 of 5 attempts already produced near-byte-identical-to-baseline correct code; the gate only needs to import from the worktree. |
| mimesis | 35% | Flat-layout so P0.1 is a no-op; the mimesis abstain was a TRUE 5% score, not a false-zero. |
| pydantic | 25% | P0.2 (memray) is forensically exact and decisive for making the gate honest (flips 3 autogen + 2 template clean-collection attempts from false-zero). |

CALIBRATED AND HONEST. This is a design + file-level plan, fully ground-truth-verified against the codebase, but NO code is written and NO eval has been re-run. The 30% overall is my honest confidence that, once implemented, this RELIABLY SOLVES ALL FOUR runnable repos. It is deliberately low because the dominant evidence says the honest near-term outcome on pydantic and mimesis is correct-ABSTAIN, not solve: the only paths that ever collected those suites were the fetch the design exists to forbid, and there is no precedent for the system reimplementing pydantic (5091 tests, ~95-100 files) or mimesis (~240 stubs) from scratch in one session.

What I AM highly confident about (and what the deliverable genuinely achieves): (1) the harness diagnosis is forensically exact — P0.1 root cause and P0.2 memray are confirmed in code, and fixing them stops the gate from scoring correct code as zero; (2) the revised P0.1 is race-safe (process-local PYTHONPATH behind a flat-layout no-op fast path + loud assertion + the existing _verify_editable_target_inside_repo reinstall fallback), closing the concurrency bug that was fatal in the original; (3) voluptuous is preserved and cheapened (the 4x->1 win is structural via fixing the scout-overrides-static-difficulty bug at architect.py:35); (4) the design corrects the original's three load-bearing factual errors (internet already False / never blocks egress; score_fn is concurrent with a shared env; the jail fetch-flag does not exist and must be built); (5) it makes pydantic/mimesis honestly measurable and maximizes the chance of catching a good in-workspace draw via the structural floor lineage (which B0 evidence shows can solve mimesis).

REQUIRED EMPIRICAL VALIDATION before any solve claim: (a) re-confirm the real pytest baseline on the apex venv (static grep shows ~78 def test_, plan claimed 79 — must be verified); (b) land P0.1 STANDALONE and re-run jinja under today's orchestration to prove the gate fix alone flips jinja fail->solve (highest-value single check); (c) prove the editable artifact form on jinja and that PYTHONPATH-prepend (or the reinstall fallback) wins; (d) prove the NEW jail command detector + dist-info provenance check actually block pip install <target> while permitting pip install -e . and pydantic-core (these mechanisms do NOT exist yet); (e) run n>=3 on each hard repo and report pass@k / agents-per-solve / tokens-per-solve — n=1 cannot rank and the same mimesis cell has both solved and failed under identical conditions. Until (b)-(e) are done on a real run, every per-repo solve number above is a calibrated prior, not a measurement.

### Per-repo detail

#### voluptuous — 88%
Already solved on the autogen arm today (winner a0, score 1.0); flat-layout so P0.1 is a verified NO-OP via the _detect_src_pkg fast path; gate has no --memray so P0.2 inapplicable. The design's net effect is preserve + cheapen (4 agents -> 1) because R0-R2 read STATIC difficulty (fixing architect.py:35 where the scout overrode easy->medium) so it routes to a 1-agent in-workspace probe behaviorally equivalent to the template that already solves it. The win is structural and well-grounded.

Residual gaps:
- No empirical re-run yet proves the exact R0 sanitized prompt (no scout approach + anti-fetch suffix) solves voluptuous at agents_used==1; the today-win used an authored 432-char prompt. Small risk it needs one repair iteration (still solves, not at 1 agent).
- New-code regression surface onto a currently-green path: the P0.1 flat fast-path, the jail command-deny allowlist (must not block the agent's own pip install -e . / pytest self-verify), the ANTI_FETCH_POLICY suffix (must permit urllib.parse), and the lint regex (must not match 'restoration'). All are addressed in the revised design but UNTESTED until the code is written.
- The floor-protection CI gate (voluptuous agents<=1 AND solved) is specified but not yet implemented/run.

#### jinja — 60%
The strongest case in principle: failure is a pure harness bug (base-stub editable shadow), and 3 of 5 attempts already produced near-byte-identical-to-baseline correct code; the gate only needs to import from the worktree. The revised P0.1 (process-local PYTHONPATH, race-free, with the loud worktree-resolution assertion and the _verify_editable_target_inside_repo reinstall fallback) is the correct fix and the design now lands it standalone + re-runs jinja under today's orchestration to prove it before the ladder.

Residual gaps:
- The exact editable artifact form on jinja (uv path-entry _editable_impl_jinja2.pth vs PEP-660 finder) could not be confirmed in this sandbox (cell venvs are cleaned); if PYTHONPATH-prepend does not win for the real artifact, the reinstall fallback must engage — adds a code path that needs the real venv to validate.
- Concurrency: the revised process-local-env approach is race-free by construction, but this must be proven by the new test_score_fn_concurrent_no_crosstalk and by a real n>=3 jinja run (the original shared-.pth design was concurrency-fatal).
- No empirical run yet confirms the fix actually flips jinja fail->solve; this is the single highest-value validation to do first.
- git apply round-trip of jinja's ~500KB/25-file cumulative diff for repair forking is plausible but unverified (jinja likely solves at R0 without repair, so this is secondary).

#### mimesis — 35%
Flat-layout so P0.1 is a no-op; the mimesis abstain was a TRUE 5% score, not a false-zero. The structural floor lineage (the strongest part of the design) genuinely closes the fetch monoculture that zeroed the cell, and B0 proved one good in-workspace implementation CAN solve it. So the design materially raises the probability of catching a solving draw. But the honest near-term outcome is most likely correct-abstain unless a strong full implementation lands: the genuine attempts were ~5% (318/6159), a near-total reimplementation, NOT a near-miss the 3-iter repair loop is sized for.

Residual gaps:
- The pip-install-the-answer shortcut (mimesis-17.0.0.dist-info present in the venv) is only closed by the NEW jail command detector + dist-info provenance check — both must be BUILT (do not exist today); until built and tested, the cheat could be scored as a solve or the genuine attempt blocked.
- The win path is the floor lineage producing one good full implementation, NOT the celebrated repair/SBFL machinery; the broad-deficit detector (route to decompose, not repair) is new and unvalidated on mimesis.
- Abnormal termination (usage:{} / infra_nonresult) on the genuine attempts is relabeled but the root cause (likely per-attempt budget exhaustion under the shared cell cap; B0 needed ~1921s) needs the per-attempt budget sizing + decoupling that is specified but unimplemented.
- n>=3 production-cell variance: the same cell solved (535s) and failed (875s) under identical conditions; a single K=8 cell remains a coin-flip unless the floor attempt is both guaranteed AND given solve-sufficient budget.

#### pydantic — 25%
P0.2 (memray) is forensically exact and decisive for making the gate honest (flips 3 autogen + 2 template clean-collection attempts from false-zero). But every path that EVER collected the suite was a pip install pydantic==2.8.2 cheat; the genuine attempts collection-blocked at errors=5091. So a correct fix exposes that the residual is an unprecedented from-scratch reimplementation of ~95-100 files / 5091 tests. Honest near-term outcome: correct-abstain, not solve.

Residual gaps:
- The anti-cheat depends entirely on the NEW jail command detector + dist-info provenance check (both unbuilt) AND a brittle target-vs-dep discrimination (pydantic vs pydantic-core differ only by -core); if imperfect it either scores the cheat green or blocks the legit dep.
- Collection-blocked failures pin pass_rate at 0, so the original plateau guard + SBFL would ABANDON a progressing import-repair lineage; the design now adds a collection-error-count climbing metric + pre-eval compileall/import/collect-only smoke gate, but these are new and unvalidated.
- Budget runaway: Budget.total=None makes the agent cap the sole brake; the design restores a hard token ceiling but pydantic already cost 5.6M tokens at 8 agents and the raise_cap delta/strike thresholds + token-headroom check are new.
- No evidence ANY arm produced an importable from-scratch pydantic; the repair loop is unproven to close a defect of this blast radius. apply_diff on 0.5-1.5MB diffs needs the --3way fallback the design adds.
- SIGNALS_LEDGER notes a disk ENOSPC event in the pydantic cell; the baseline is partially contaminated and must be re-established with n>=3.

## Residual risks
- NO RUN YET: the entire package is a spec; the real pytest baseline (~78 vs claimed 79), the jinja flip, the anti-fetch teeth, and n>=3 variance are all unvalidated empirically.
- The two genuinely-new anti-fetch mechanisms (jail PyPI/clone command detector + post-session dist-info provenance check) DO NOT EXIST in the codebase and must be built from scratch; in host_cli_read_jail mode codex's own sandbox is bypassed (cli_backend.py:11577) so this jail-layer detector is the ONLY enforcement point and must be the real boundary.
- Target-vs-dep discrimination is brittle: pydantic vs pydantic-core differ only by '-core'; an over-broad deny breaks the legit dep, an under-broad one lets the cheat through. Must match the target package NAME exactly.
- Allowlist regression onto green paths: the command deny must permit the agent's own pip install -e . / pytest / compileall self-verify, or it regresses voluptuous (the universal floor) multiplied across n>=3 seeds.
- P0.1 fallback path: if jinja's real editable artifact is a PEP-660 finder rather than a uv path-entry, PYTHONPATH-prepend may not win and the reinstall fallback must engage — unvalidated without the real venv.
- Budget runaway on pydantic: Budget.total=None makes the agent cap the sole brake; the restored hard token ceiling + raise_cap delta/strike thresholds are new and must be tuned so a flickering pass_rate (from any P0.1 imperfection) cannot drive spend toward 30-50M tokens.
- Collection-blocked pydantic: the new collection-error-count climbing metric + pre-eval smoke gate are essential (a pass_rate plateau would abandon a progressing lineage) but unvalidated; SBFL is inert without a passing/failing contrast.
- mimesis abnormal termination (usage:{}/infra_nonresult) is relabeled but its root cause (likely per-attempt budget exhaustion vs the ~1921s single-solve cost under a 2400s cell cap) needs the per-attempt budget sizing/decoupling that is specified but unimplemented.
- apply_diff on 0.5-1.5MB pydantic/jinja diffs may fail context matching; the --3way fallback + indeterminate-on-failure is specified but unverified on real large diffs.
- Honest-solve ceiling: even with everything working, no arm has ever reimplemented pydantic/mimesis from scratch; the most probable outcome on those two is correct-abstain, and 'AUTOGEN_WON' may stay 0 even after all fixes.

## Validation run command

```bash
APEX_OMEGA_PYTHON=/Users/sameertkhanna/Documents/agent_orch/.venv/bin/python LADDER_CONCURRENCY=3 PYTHONPATH=/Users/sameertkhanna/Documents/agent_orch /Users/sameertkhanna/Documents/agent_orch/.venv/bin/python /Users/sameertkhanna/Documents/agent_orch/scripts/run_ladder.py   # PREREQ 1: confirm unit baseline -> PYTHONPATH=/Users/sameertkhanna/Documents/agent_orch <apex-venv>/bin/python -m pytest /Users/sameertkhanna/Documents/agent_orch/tests/ -q  (record the real passing count; plan claims 79, static grep shows ~78).  PREREQ 2 (highest value): land P0.1 + P0.2 ONLY, then run a single jinja cell under TODAY's unchanged orchestration to prove the gate fix alone flips jinja fail->solve before enabling the ladder.  Then run the full command above with --seeds 3 for jinja/mimesis/pydantic; completed cells are skipped (cell_done), progress at runs/ladder/progress.jsonl; substitute the real apex venv python that has the apex package importable.
```

---

# APEX-Ω AUTOGEN — MASTER DESIGN: Ladder-of-Lineages (LoL), Revised & Gap-Closed

## 0. Status and honesty preamble

This is a **design + file-level plan**, ground-truth-verified against the codebase. **No code is yet written and no eval has been re-run.** Every confidence number below is calibrated to that fact. The single most important honest statement, which the original design refused to make and which all hard-repo validators converged on:

> **After the harness is fixed and the fetch is genuinely blocked, the most likely near-term outcome on pydantic and mimesis is a CORRECT ABSTAIN, not a solve.** The design's value on those repos is making the result *honest and measurable* (no false-zero, no fetch-cheat-as-solve), plus maximizing the probability of catching a good in-workspace draw — not a guaranteed solve.

## 1. Verified ground-truth (the facts the design is built on)

All confirmed by reading the actual source in this session:

| Claim | Verdict | Evidence |
|---|---|---|
| `ScopedTask.internet` defaults False; `solve_attempt` never sets it | TRUE (design §4.1 premise was WRONG) | `types.py:126`; `context.py:128-132` |
| `internet=False` adds NO network deny (only omits the internet flag) | TRUE | `_internet_launcher_args` returns `[]` for False, `cli_backend.py:11675` |
| Seatbelt profile is `(allow default)` with file-only denies; no network egress deny | TRUE | `cli_backend.py:3634` |
| In host_cli_read_jail mode codex runs `--dangerously-bypass-approvals-and-sandbox` (own sandbox off) | TRUE | `cli_backend.py:11577` |
| codex `workspace-write` (which would enforce network-off) is NOT reachable when host_cli_read_jail is active | TRUE | else-branch at `cli_backend.py:11586`; only taken when NOT read-jail |
| The §4.5 "jail fetch-attempt flag" exists | FALSE — it must be BUILT | no such flag in `cli_backend.py`; jail classifies file reads only |
| `score_fn` runs concurrently (ThreadPoolExecutor) and reuses a SHARED `env` | TRUE (design's "evals serialized" was WRONG) | `runtime.py:59,180`; `commit0_autogen.py:249` |
| `apply_diff` is bare `git apply` (no `--3way`), returns False silently | TRUE | `worktree.py:129` |
| `acquire` force-removes + re-adds the worktree off `base_commit` each call | TRUE | `worktree.py:81-88` |
| Proven per-tree editable reinstall already exists | TRUE — design ignored it | `_verify_editable_target_inside_repo`, `commit0_benchmark.py:14033` |
| `Budget.total=None` -> `can_start()` always True; agent cap is the sole spend brake | TRUE | `budget.py:53` |
| Scout OVERRIDES static difficulty (cause of voluptuous 4x cost) | TRUE | `architect.py:35` "scout overrides the static proxy" |
| memray plugin table has only `pytest-cov` | TRUE | `_COMMIT0_PYTEST_OPTION_PLUGIN_PACKAGE_MODULES`, `commit0_benchmark.py:1588` |
| `apply_evidence_bound_review` is monotone (downgrade-only) | TRUE — §4.5 contract holds | `select.py:85-97` |
| `interaction_required` -> `policy_violation` collapse | TRUE | `v1_executor.py:35` |
| difficulty thresholds: easy<15, medium<80, hard>=80 src modules | TRUE | `architect.py:105` |
| issue seed is `Task objective: {self.specification}` | TRUE | `commit0_benchmark.py:3882` |

Test baseline: 10 test files, ~78 `def test_` (one fewer than the plan's "79" claim — likely a parametrize/class case). **The exact passing count must be re-confirmed on the real venv before landing** (no pytest-capable venv was accessible in this analysis sandbox).

## 2. PRIORITY 0 — harness correctness (must land + be proven green first)

### P0.1 (REVISED, race-safe) — per-worktree editable resolution. `eval/commit0_autogen.py::score_fn`

Root cause confirmed: `_prepare_repo` (`commit0_autogen.py:199`) installs `-e .` against the **base** repo_dir, pinning the editable finder at `repo_dir/src/<pkg>`; `score_fn` runs pytest in the worktree but reuses the same base `env` -> imports the base stub -> false-zero (jinja, 851 collection errors).

**The original design's fix (shared-`.pth` rewrite + restore, "evals serialized") is concurrency-UNSAFE.** `score_fn` runs on a ThreadPoolExecutor with a shared `env`. The revised fix:

1. **Flat-layout fast-path NO-OP:** `_detect_src_pkg(worktree)` returns None when there is no `<worktree>/src/<pkg>/__init__.py`. voluptuous/mimesis/pydantic are flat -> the entire P0.1 branch is skipped; the currently-green flat eval is untouched. **This is the floor-protection the voluptuous validators demanded.**
2. **Process-local PYTHONPATH (race-free):** for src-layout repos, build a **per-call** `call_env = dict(env)` and prepend `worktree/src` to its `PYTHONPATH`. Never mutate the shared `env` and never mutate a shared `.pth`. Because `env` is passed into the `evaluate_repo` subprocess per-call, this is process-local and race-free across the 8-wide jinja wave.
3. **Loud worktree-resolution assertion:** resolve `<pkg>.__file__` under the venv + `call_env` (subprocess, no in-process import pollution); if it does not live under THIS worktree, return `VerificationResult(indeterminate=True, accepted=False)` — never a silent false-accept (Cardinal Contract preserved).
4. **Fallback to the proven path:** if PYTHONPATH-prepend does not win for a repo whose editable is a true PEP-660 finder (not a plain path-entry), fall back to the existing `_verify_editable_target_inside_repo` per-tree reinstall (`commit0_benchmark.py:14033`) under a per-venv lock. This is the codebase's own robust mechanism for all artifact forms.
5. **De-risk by sequencing:** land P0.1 STANDALONE and re-run jinja under TODAY's unchanged orchestration to prove the gate fix alone flips jinja fail->solve, BEFORE layering the ladder.

### P0.2 — gate/self-verify parity (the pydantic `--memray` bug). `apex/evaluation/commit0_benchmark.py` + `commit0_autogen.py`

Confirmed: the gate adds `--memray` under `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` but never `-p pytest_memray` -> rc=4 pre-collection. `infer_additional_pytest_packages` already maps `--memray`->`pytest-memray`; only the `-p` emission table is missing it. Add `"pytest-memray": "pytest_memray"` to `_COMMIT0_PYTEST_OPTION_PLUGIN_PACKAGE_MODULES` (`commit0_benchmark.py:1588`) PLUS a generic guard: for any package returned by `infer_additional_pytest_packages` whose option flag is in the command, emit `-p <module>` (prevents `--timeout`/`--reruns` recurrence). Surface the **exact authoritative gate command** into the worker prompt so an attempt cannot self-certify on a weaker command.

### P0.3 — reflog + non-base-branch scrub. `eval/repo_sanitize.py::scrub_upstream_identifiers`

`git reflog expire --all --expire=now` + `git gc --prune=now` + delete every branch != `apex-base` (SWE-bench #465 leak vector).

### P0.4 — de-seed the issue text. `apex/evaluation/commit0_benchmark.py::build_issue_description`

Neutralize URL/version literals in `self.specification` (`commit0_benchmark.py:3882`) before it renders as `Task objective: ...`. This removes the seed BOTH arms inherit.

## 3. ENABLING CHANGE — surface failing-test detail (advisory, never a gate input)

Add `failing_nodeids`, `failure_excerpts`, `finalization_status` to `VerificationResult` + `to_dict` (`kernel/verify.py`); populate from the pytest-json the backend already writes (`eval/scoring.py`); thread into `Candidate.meta`. `ranking_key` is unchanged — these never influence acceptance.

## 4. THE LINEAGE PRIMITIVE — `repair_attempt` + `solve_and_repair` (`autogen/context.py`)

The atomic unit is a **lineage**: implement -> run real tests -> read failures -> targeted fix -> repeat, reconciled with the confirmed worktree mechanic (each iteration is a fresh `acquire` + `apply_diff(parent.diff)`, like journal replay, because `acquire` force-rebuilds off base).

- **`repair_attempt(parent, ...)`**: fork the parent's cumulative diff into a fresh worktree, seed the agent with `parent.meta["failing_nodeids"]` + `failure_excerpts`, run ONE journaled `engine.agent`, re-score. **REVISED:** if `apply_diff` returns False, first retry with `git apply --3way`; on continued failure return `indeterminate` (never silently repair from base). `scoped_inputs` includes `parent_diff_sha` + the failing set so the journal key changes with the repair context. Cannot set `accepted`.
- **`solve_and_repair(...)`**: base attempt then up to `max_iters` repairs under the stopping criteria (§7). `solve_attempt` stays the `max_iters=1` floor (provably never worse than verified best-of-N).

## 5. ANTI-SHORTCUT / GROUNDED ACCEPTANCE (re-scoped to what is actually buildable)

The original §4.1 "internet=False makes fetch impossible" is a **no-op** (verified). The genuinely-new, buildable teeth:

### 5.1 NEW jail command detector (the real capability block). `apex/core/cli_backend.py` + `executor/v1_executor.py`
Build (it does not exist) a command-path detector that, when `internet=False` and a target package is known, flags/denies: `pip install <target>`, `uv pip install <target>`, `pip download <target>`, `curl|wget` of PyPI, `git clone <upstream>`. **Concrete allowlist (the voluptuous floor-protection the validators demanded):** ALWAYS permit `pip install -e .`, `pytest`, `python -m pytest`, `python -m compileall`, and the legit transitive dep (e.g. `pydantic-core` — note this differs from `pydantic` only by `-core`, so the discriminator matches the target package NAME exactly, not a prefix). Surface a `fetch_attempted` flag through `ExecResult.meta` into `Candidate.meta`. Enforcement lives at the vendor-neutral process-tree jail so it is vendor-agnostic.
> **Honest caveat:** in host_cli_read_jail mode codex's own sandbox is bypassed, so this jail-layer detector is the ONLY enforcement point — it must be the real boundary, not a flag.

### 5.2 Un-overridable anti-fetch suffix. `context.py:128`
`task_prompt = (prompt or self._prompt_builder(...)) + ANTI_FETCH_POLICY` — always concatenated, never replaceable. **REVISED wording:** the suffix bans *external-package acquisition* only; it explicitly PERMITS stdlib imports (`urllib.parse` for voluptuous's Url validator) and `pip install -e .` self-verify. Split the `interaction_required->policy_violation` collapse so a genuine-but-incomplete attempt (real code, pass_rate>0) is a repair target, not a dead policy_violation.

### 5.3 Structural in-workspace floor lineage (draw-independent invariant). `context.py` + `templates.py`
Lineage 0 is ALWAYS a sanitized in-workspace `test_driven` lineage built WITHOUT the scout approach (`suppress_approach=True`). A poisoned shared prefix can never route 100% of the fan through fetch because at least one lineage physically cannot. **This is the strongest part of the design for mimesis** (B0 proved one good in-workspace implementation solves it).

### 5.4 Grounded acceptance (monotone downgrade-only). `context.py::select`
- **Provenance refute:** refute (True->False only, via `apply_evidence_bound_review`) any candidate with `meta["fetch_attempted"]` OR a **post-session dist-info provenance hit** (target `<pkg>==<ver>.dist-info` present in the venv that was NOT in the prepared base) — this catches the pip-install-the-answer vector that leaves no diff trace (the pydantic seq-12/16 cheat) and that the jail flag alone misses.
- **Test-file immutability:** refute any candidate whose diff modifies `test_*.py`/`*_test.py`/`conftest.py`.

### 5.5 Authoring-time guards. `architect.py::INVARIANTS` + `sandbox.py::lint_source`
INVARIANT 6 (>=1 from-scratch in-workspace lineage). `lint_source` bans fetch-ACQUISITION literals only — regex matches `git clone`, `pip install <pkg>` (negative-lookahead on `-e`), `download the official`, `clone the upstream`, but NOT `pip install -e .`, NOT `minimal verified restoration`, NOT `restore the implementation from the visible tests`.

## 6. LOCALIZATION + collection-progress signal (closes the pydantic/mimesis sabotage)

- **SBFL between waves (Ochiai):** rank stub functions by suspiciousness from the real gate's per-test outcomes; surface into the next wave's prompt. Advisory only.
- **NEW collection-error climbing metric (critical for pydantic):** when `pass_rate==0` because the import chain does not collect (pydantic), SBFL has no contrast and a pass_rate plateau would ABANDON a genuinely-progressing lineage. So add a **pre-eval smoke gate** (`compileall` + `import <pkg>` + `pytest --collect-only`) and let the repair loop climb on **decreasing collection-error count** while pass_rate is still 0. Without this, the stopping criteria sabotage pydantic.
- **NEW broad-deficit detector (critical for mimesis):** when best genuine pass_rate is very low (<20%) AND failures span many modules (high distinct-error diversity), route to RE-IMPLEMENT / decompose-by-subsystem (the win path B0 used) rather than 3-iter in-place repair on a 5% stub.

## 7. THE ESCALATION LADDER — difficulty-gated, signal-conditioned (`templates.py` + `architect.py`)

| Rung | What runs | Cost | Gate to ENTER |
|---|---|---|---|
| R0 Lean probe | 1 in-workspace floor lineage (suppress_approach), anti-fetch | 1 agent | always |
| R1 Repair the probe | `solve_and_repair` seeded by R0 failures (climb on pass_rate OR collection-error count) | <= D agents | R0 not accepted AND genuine |
| R2 Diverse fan | decorrelated best-of-N from STATIC difficulty, repair-capable, >=1 pinned no-lookup | static-difficulty wave | R0+R1 abstained |
| R3 Scout + author | NOW pay scout (as localization) + architect; broad-deficit -> decompose | scouts + author + plan | R2 abstained AND (static==hard OR partial-progress floor) |
| R4 Pooled select | `select_best` over all candidates | 0 new agents | always, at end |

**Difficulty gate (REVISED):** R0-R2 read STATIC `build_repo_map` difficulty (NOT scout-overridden — fixing `architect.py:35`). voluptuous reads `easy` -> R0=1 agent -> matches the template, kills the 678K scout tax and the easy->medium bump. Scout/author run ONLY at R3 (static==hard).

**Stopping criteria:** stop on first of (1) accepted (real gate); (2) plateau — pass_rate (or collection-error count, for collection-blocked repos) not strictly improving for `plateau_patience` iters; (3) indeterminate/infra -> do NOT repair in place; (4) policy_violation/fetch_attempted -> pivot to fresh no-lookup lineage; (5) ceiling/budget.

**Cap growth (REVISED for runaway risk):** Keep difficulty as the INITIAL `effective_max`. `raise_cap` bumps toward the ceiling ONLY while best pass_rate strictly improves by `>= min_delta=0.02` across waves, with a 3-strike futility stop. **RESTORE a hard `Budget.total` token ceiling** (since `Budget.total=None` makes the agent cap the sole brake, and pydantic already cost 5.6M tokens at 8 agents) tied to a token-headroom check, not just agent count. The 1000 backstop (`runtime.py:141`) stays immovable.

**Per-attempt budget (REVISED, mimesis):** size the floor-lineage per-attempt budget against the OBSERVED single-solve cost (B0 needed ~1921s wall for mimesis) and DECOUPLE it from the shared cell cap, so the ladder cannot starve the one attempt that can solve. Diagnose the `usage:{}`/`infra_nonresult` abnormal-termination as a distinct `session_truncated` class -> restart a fresh full attempt rather than feed a truncated 5% stub into repair.

## 8. VARIANCE / HONEST RANKING (`eval/commit0_driver.py`)

n>=3 cells per (arm, hard-repo); report pass@k / solve@R distribution; deterministic attempt/repair ids; structural floor lineage 0; vendor cycling on repairs. Report `autogen_solved` (autogen-only — the honest arm number, preserving the architect.py:387-393 invariant) AND `pooled_solved` (cross-arm) as DISTINCT metrics. Report agents/solve + tokens/solve so "complexity pays off" is falsifiable.

## 9. INVARIANTS PRESERVED

- Execution-authoritative: `select`->`select_best` is the sole winner producer; `accepted` only from `score_fn`; `repair_attempt` cannot set accepted; advisory fields never gate; §5.4 only downgrades via the proven monotone review.
- Env sanitizer kept (extended). Process-tree jail kept (extended with the NEW command detector).
- Vendor-agnostic: every new primitive operates on `WorkerSpec.vendor`/`Candidate`/`ScopedTask`; the one vendor-sensitive piece (jail command detector) lives at the vendor-neutral jail.
- Determinism/replay: repairs journaled with context-sensitive keys; lint extended; `engine/{runtime,pipeline,budget}` and `kernel/select` ranking unchanged.

## 10. Bottom line

Fix the gate first (P0; land P0.1 standalone and prove jinja flips). Make P0.1 race-safe (flat no-op fast path + process-local PYTHONPATH + reinstall fallback). Surface real failing tests. Make the unit a difficulty-gated ladder of repair lineages with a structural in-workspace floor. Build the REAL anti-fetch teeth (jail command detector + dist-info provenance + concrete allowlist) — not the no-op internet flag. Add a collection-progress metric and broad-deficit detector so the ladder doesn't sabotage pydantic/mimesis. Restore a hard token ceiling. Rank honestly with n>=3. And state plainly: on pydantic/mimesis the honest near-term outcome is most likely correct-abstain, not solve.