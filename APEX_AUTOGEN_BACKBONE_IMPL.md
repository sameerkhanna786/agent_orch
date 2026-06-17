# APEX-Ω Autogen Backbone — File-Level Implementation Plan

# FILE-LEVEL IMPLEMENTATION PLAN (phased, executable)

Baseline: 94 test functions across 12 files in tests/ (the "92 green" baseline). All new tests are additive. Each anchor verified against the live code.

## PHASE 0 — Backbone safety floor (close runaway + discard holes; smallest blast radius)

### 0.1 Opt-in token budget (remove fragile `budget or` truthiness)
File: apex_omega/eval/commit0_driver.py:109-113. Replace `token_ceiling = int(os.environ.get("APEX_OMEGA_TOKEN_CEILING", str(40_000_000)))` + `budget=budget or Budget(total=token_ceiling)` with:
```python
_env = os.environ.get("APEX_OMEGA_TOKEN_CEILING")
_ceiling = int(_env) if (_env and _env.strip()) else None   # default UNBOUNDED
if budget is not None: _budget = budget
elif _ceiling is not None: _budget = Budget(total=_ceiling)
else: _budget = Budget()                                     # total=None
self.engine = engine or Engine(..., budget=_budget, ...)
```
Test: tests/test_mode_c.py::test_budget_default_unbounded — assert driver.engine.budget.total is None with no env/arg; ==40_000_000 only when env set.

### 0.2 Always-on FINITE per-agent wall (the real runaway guarantee)
Files: configs/base_commit0_local.json; apex_omega/autogen/context.py:175,248; apex_omega/types.py:128; apex_omega/engine/runtime.py:135-155; apex_omega/journal/resume.py.
1. Config (the bound that BITES) — add to llm_configs[0]: `"cli_strict_hard_timeout": true` (cli_hard_timeout_seconds:2400 already present). cli_backend.py:6699 now returns a real cap; :8536 enforces a Popen kill via terminate_for_rollout.
2. Decouple value: context.py:175,248 stop passing `timeout_seconds=self.timeout_seconds`; pass `self.per_agent_timeout_seconds` (new __init__ field, derived from difficulty: easy/medium/hard → 1800/2400/3000, all < cell_timeout). Add `heartbeat_timeout_seconds` to ScopedTask (types.py:128).
3. Engine watchdog (DEMOTED, honest scope): in runtime.py _safe_runner, wrap runner(task) with a thread+join(wall+120). On timeout return `ExecResult(ok=False, finalization_status="heartbeat_timeout")`. EXPLICIT: this only abandons the wait for SELECTION purposes; the vendor-native cli_strict_hard_timeout kills the process.
```python
def _safe_runner() -> ExecResult:
    with self._total_lock:
        self._total_agents += 1; n = self._total_agents
    if self.max_total_agents and n > self.max_total_agents:
        return ExecResult(ok=False, finalization_status="infra_nonresult",
                          error=f"max_total_agents ({self.max_total_agents}) exceeded")
    self._sem.acquire()
    try:
        wall = getattr(task, "timeout_seconds", None)
        if not wall or wall <= 0:
            res = _call(runner, task)              # vendor-native bound governs
        else:
            box: dict = {}
            th = threading.Thread(target=lambda: box.__setitem__("r", _call(runner, task)), daemon=True)
            th.start(); th.join(wall + 120)
            if th.is_alive():
                return ExecResult(ok=False, finalization_status="heartbeat_timeout",
                                  error=f"per-agent watchdog {wall}s exceeded (process killed by vendor cap)")
            res = box.get("r")
        if not isinstance(res, ExecResult):
            return ExecResult(ok=False, finalization_status="infra_nonresult",
                              error=f"runner returned {type(res).__name__}")
        return res
    except Exception as exc:
        return ExecResult(ok=False, finalization_status="infra_nonresult", error=f"{type(exc).__name__}: {exc}")
    finally:
        self._sem.release()
```
(`_call` runs runner(task), returns infra_nonresult ExecResult on exception so the thread never raises.)
4. resume.py _serialize_exec_result already maps `not result.ok`→RESULT_INFRA_NONRESULT (verified line 16), so heartbeat_timeout is never a hit and re-runs.
Test: tests/test_engine_journal.py::test_per_agent_watchdog_excludes_and_reruns (FakeExecutor sleeps > wall → heartbeat_timeout, not a hit).

### 0.3 Per-RUN backstop (cross-relaunch), default 2048→1000
Files: apex_omega/engine/runtime.py:51,66-69,138-143; apex_omega/journal/wal.py.
1. max_total_agents: int = 1000.
2. Add Journal.fresh_agent_count() = count of committed kind=="agent" entries with result_status==RESULT_OK (a HIT replays, never re-begins, so committed-OK is the true cross-process fresh tally). In Engine.__init__: `self._total_agents = self.journal.fresh_agent_count()`.
Test: tests/test_journal_crash_resume.py::test_backstop_is_per_run (journal with 999 committed agents → fresh Engine admits 1 more then backstops).

### 0.4 parallel ≤4096 guard
File: apex_omega/engine/runtime.py:171-192. After `thunks = list(thunks)`: `if len(thunks) > 4096: raise FailLoud(...)`.
Test: tests/test_engine_journal.py::test_parallel_4096_cap.

### 0.5 Structural template-FLOOR FIRST (host-side, not authored discipline)
File: apex_omega/autogen/architect.py autosolve (~line 376, before run_orchestration).
```python
engine.phase("floor-probe")
floor_cand = None
try:
    floor_cand = ctx.solve_attempt(attempt_id=0, strategy="minimal")  # wave-0=1, journaled+checkpointed
except Exception as exc:
    engine.log(f"floor-probe failed: {type(exc).__name__}: {exc}")
```
Thread a base_attempt_id=1 offset into OrchestrationContext so authored ids never collide with the floor's a0 (replayed as HIT). After run_orchestration returns:
```python
winner = run_orchestration(frozen.source, ctx)
if winner is None and floor_cand is not None and floor_cand.accepted:
    winner = ctx.select([floor_cand, winner])   # rescue clean-abstain ONLY with verified floor
```
Gate behind ctx.abl (default ON for headline; `honest_no_floor_rescue` ablation disables for strict autogen-stands-alone). Keep crash/malformed _floor() unchanged.
Tests: tests/test_autogen.py::test_floor_first_banked_before_authored; ::test_authored_abstain_rescued_by_verified_floor.

### 0.6 Fix frozen-.py strip bug (VERIFIED LIVE — resume prerequisite)
File: scripts/run_ladder.py:42. `_KEEP_SUFFIXES = (".json", ".jsonl", ".md", ".diff", ".patch", ".txt", ".log", ".py")`. Confirmed: the run4 archive's autogen jinja frozen.json references bb4ba79...py which is DELETED while frozen.json survives; load_frozen (architect.py:159) would raise FileNotFoundError on resume. .py artifacts are tiny.
Test: tests/test_strip_checkout.py::test_strip_keeps_frozen_py.

**GO/NO-GO Phase 0:** all 94 baseline + new tests green; budget.total is None by default; a slow FakeExecutor agent is excluded+re-runs without killing the cell; 999-agent journal admits exactly 1 more across a fresh Engine; floor candidate present in journal before authored script. NO-GO if any baseline regresses or budget default ≠ None.

## PHASE 1 — Remove guillotine + full mid-run RESUME

### 1.1 Journal the `score` step (resume-cheapness + score-gap close)
Files: apex_omega/kernel/verify.py (from_dict); apex_omega/autogen/context.py:182,258; apex_omega/eval/commit0_autogen.py (thread expected_ids_sha + scoring_env_sha).
1. VerificationResult.from_dict (lossless for ACCEPTANCE fields; advisory failing_nodeids[:50]/failure_excerpts[:3000] knowingly lossy — soften any "lossless" claim):
```python
@classmethod
def from_dict(cls, d):
    d = d or {}
    return cls(accepted=bool(d.get("accepted", False)), score=float(d.get("score", 0.0)),
               reason=d.get("reason"), passed=int(d.get("passed", 0)), failed=int(d.get("failed", 0)),
               errors=int(d.get("errors", 0)), total=int(d.get("total", 0)),
               missing_expected=int(d.get("missing_expected", 0)), pass_rate=float(d.get("pass_rate", 0.0)),
               taxonomy=d.get("taxonomy"), indeterminate=bool(d.get("indeterminate", False)),
               failing_nodeids=list(d.get("failing_nodeids", [])), failure_excerpts=d.get("failure_excerpts", ""))
```
2. _scored() in context.py — hoisted to where res.fs_diff is in scope (NOT inside score_fn, which only receives the path — VERIFIED commit0_autogen.py:286). Computes diff_sha from the JOURNALED blob, not a worktree re-read:
```python
def _scored(self, wt, res):
    from ..journal.key import sha256_hex
    from ..journal.resume import resume_or_run_json
    from ..journal.wal import RESULT_OK, RESULT_INFRA_NONRESULT
    components = {"kind": "score", "scoped_inputs": {
        "diff_sha": sha256_hex(res.fs_diff or ""),
        "repo_snapshot_sha": self._provider.base_commit,
        "expected_ids_sha": self._expected_ids_sha,
        "scoring_env_sha": self._scoring_env_sha}}
    def _run(): return self._score_fn(wt).to_dict()
    def _status(v): return RESULT_INFRA_NONRESULT if (v or {}).get("indeterminate") else RESULT_OK
    d, _hit = resume_or_run_json(self._engine.journal, components, _run, kind="score",
                                 node_id="score", status_fn=_status)
    return VerificationResult.from_dict(d)
```
Replace `vr = self._score_fn(wt)` at 182 and 258 with `vr = self._scored(wt, res)`.
3. diff_sha stability (debug-gated assert): on an agent HIT the diff is materialized from journal/diffs/<sha>.diff (content-addressed by exactly that sha, wal.py:185-194) so res.fs_diff on replay is byte-identical → score key is a HIT. On mismatch treat as MISS (re-run pytest) — converts the under-specified coupling into a checked contract.
4. scoring_env_sha = hash of {venv_python version, resolved-dependency set, src-layout PYTHONPATH decision}, computed once after _prepare_repo (commit0_autogen.py:243) and threaded into the context — so a re-prepped venv with drifted deps cannot HIT a stale score.
5. indeterminate→RESULT_INFRA_NONRESULT is never a phantom-accept (wal.py only indexes RESULT_OK).
Tests: tests/test_journal_crash_resume.py::test_score_hit_no_pytest_rerun; ::test_from_dict_roundtrip_rederives_accepted; ::test_score_key_changes_on_env_drift; ::test_kill_after_green_recovers_from_counts.

### 1.2 Journal `prep` and `author`
Files: apex_omega/eval/commit0_autogen.py:243; apex_omega/autogen/architect.py:202-209.
1. prep: wrap `env = runner._prepare_repo(...)` in resume_or_run_json(kind="prep", key={repo, base_commit, config_sha}), store env DESCRIPTION dict. Honest: cold resume after strip re-materializes the venv; relaunch must NOT strip between attempts (enforced 1.3).
2. author: reroute _author_via_llm's direct `session.run(...)` (architect.py:207 — VERIFIED the one un-journaled solve-path call) through engine.agent/resume_or_run_exec with components={kind:"author", prompt, model, vendor, repo_map_sha}. On resume load_frozen replays the frozen script AND the author LLM call is a HIT.
3. structured_output round-trip: VERIFIED ExecResult.to_dict/from_dict round-trip structured_output (types.py:177,192). Add a regression test so the scout→author replay chain stays deterministic.
Tests: tests/test_mode_c.py::test_prep_journaled_skips_reclone; tests/test_scout.py::test_exec_result_structured_output_roundtrips.

### 1.3 Relaunch-on-kill loop (FINITE + progress-gated)
File: scripts/run_ladder.py:166-207, :36.
```python
LADDER_MAX_RELAUNCH = int(os.environ.get("LADDER_MAX_RELAUNCH", "8"))
last_prog = None
for attempt in range(1 + LADDER_MAX_RELAUNCH):
    try:
        subprocess.run(cmd, ..., timeout=CELL_TIMEOUT + 600); break
    except Exception as exc:
        if not _has_journal(rundir):
            _emit(label, repo, "error", {"error": f"{type(exc).__name__}: {exc}"[:200]}); return
        prog = _journal_progress(rundir)   # (committed_ok, fresh_agents, last_accept_seq)
        if attempt >= LADDER_MAX_RELAUNCH or (last_prog is not None and prog <= last_prog):
            ckpt = _recover_checkpoint(rundir)   # diff_ref→journal/diffs/<sha>.diff join
            ... recover banked accept, else error
            return
        last_prog = prog
        # NOTE: do NOT _strip_checkout between attempts (keep venv warm)
```
Soften outer timer: CELL_TIMEOUT becomes the child's soft --cell-timeout; outer subprocess.run timeout becomes a pause trigger, not a discard. Extend _recover_checkpoint to read accepted_checkpoint.json's content_sha and join the journaled score/diff so recovery yields the patch.
Tests: tests/test_journal_crash_resume.py::test_relaunch_rides_prefix_no_redispatch; ::test_relaunch_stops_on_no_progress.

### 1.4 `accept` as first-class WAL event (evidence, not authority)
Files: apex_omega/autogen/context.py:116-136; apex_omega/journal/wal.py; apex_omega/journal/resume.py; scripts/run_ladder.py.
In _checkpoint_accepted, additionally append `commit(kind="accept", structured_result={candidate_id, content_sha, pass_rate, vr_counts, diff_ref})`. Add resume.py::last_accept(journal). Ordering: score committed FIRST, accept SECOND — a kill between them recovers from the score event alone. Cutover: journal authoritative on read; JSON mirror fallback only; on disagreement journal wins. Recovery re-derives accepted via the score HIT through candidate_from_verification — never the stored boolean.
Tests: tests/test_journal_crash_resume.py::test_accept_event_rederived_from_score; ::test_kill_between_score_and_accept_recovers.

**GO/NO-GO Phase 1:** empirical — kill a real voluptuous autogen cell mid-orchestrate; relaunch; narration shows cache_hit:true for the entire prior prefix, zero re-dispatched agents for cached work, venv not re-materialized, final report identical. A SIGKILL after pytest-green but before accept-fsync, then relaunch with a warm venv → solve recovered from journaled COUNTS (no pytest re-run). No baseline regression. NO-GO if resume re-runs pytest for an unchanged diff, any verified solve is lost, or relaunch loops without progress.

## PHASE 2 — Composable architect vocabulary

### 2.0 Determinism-under-resume hardening (PREREQUISITE — the real erosion)
Files: apex_omega/engine/runtime.py; apex_omega/autogen/context.py; apex_omega/journal/resume.py.
VERIFIED: agents_used()/budget.spent() are miss-only (runtime.py:75-78,138,160-161); on resume prior agents are HITs → they read 0 while the frozen script replays → any control-flow branch diverges. FIX (b): journal each wave's decision as resume_or_run_json(kind="wave", key={frozen_content_sha, wave_index}) so plan_waves/loop_until_dry replay the SAME branch. Provide ctx.should_continue_waves(frontier) reading the journaled wave decision on replay. Keep agents_used()/budget as narration only.
Test: tests/test_autogen.py::test_escalating_plan_replays_same_branch (plan branching on agents_used, killed mid-wave, replays identical wave sequence).

### 2.1 RunGovernor + always-on plateau
File: NEW apex_omega/engine/governor.py; wired in OrchestrationContext.
```python
class RunGovernor:
    def __init__(self, *, engine, agent_ceiling=1000, token_budget=None, agent_budget=None, plateau_k_dry=2):
        self._engine = engine; self.agent_ceiling = agent_ceiling
        self.token_budget = token_budget; self.agent_budget = agent_budget; self.plateau_k_dry = plateau_k_dry
    def can_start(self, *, reserve=1) -> bool:
        if not self._engine.budget.can_start(reserve=reserve): return False
        if self.agent_budget is not None and self._engine.agents_used() >= self.agent_budget: return False
        return self._engine.agents_used() < self.agent_ceiling
    def should_continue_waves(self, *, dry_rounds, best_pass_rate, prev_best) -> bool:
        if not self.can_start(): return False                  # NO CLOCK
        if dry_rounds >= self.plateau_k_dry and best_pass_rate <= prev_best + 1e-9: return False
        return True
```
Always-on plateau: bake k-dry/no-improvement into BOTH DEFAULT_ORCHESTRATION's wave loop AND a host-side check (OrchestrationContext.parallel increments a host dry-round counter; when should_continue_waves is False the next parallel/solve_attempt short-circuits with narrated plateau_stop). agent_budget defaults to the difficulty soft_cap (8/24/64) so fan-out patterns count against the run-4 work brake.
Tests: tests/test_autogen.py::test_governor_plateau_stops_pathological_while_true; ::test_governor_unbounded_runs_forever_when_nothing_set_except_plateau.

### 2.2 Atoms (ctx.ask) + structural soft-write seam
Files: apex_omega/autogen/context.py; apex_omega/kernel/select.py; NEW apex_omega/patterns/.
- ctx.ask(prompt, *, schema=None, vendor=None, model=None, agent_id=None) → read-only schema'd subagent (generalizes agent_scout's engine.agent(agent_type="scout")), forced sandbox="read-only", deterministic caller-assigned id. Produces signals never Candidates.
- Soft-write seam made STRUCTURAL (patterns run host-side, OUTSIDE the sandbox AST guard — VERIFIED setattr only forbidden in lint at sandbox.py:22-28): add Candidate.set_soft(perspective=None, eg_critic=None) (can ONLY touch those two fields) and Candidate.refute() (can ONLY flip accepted True→False). Patterns call those, never bare setattr. ctx.review→apply_evidence_bound_review; ctx.set_tiebreak→set_soft.
- Extract shared _attempt(prompt, *, node_prefix, scoped_extra) from solve_attempt+repair_attempt.
Tests: tests/test_safety_and_ablation.py::test_ask_cannot_produce_candidate; ::test_ranking_key_order_pinned; ::test_set_soft_cannot_touch_accepted.

### 2.3 The six patterns
File: apex_omega/patterns/{verify,judge,synthesize,loop,critic}.py + thin ctx.* wrappers.
- ctx.adversarial_verify(cand,n,perspectives=,vendor_diverse=True) → N skeptic ctx.asks; aggregation OR-over-refute (no quorum vetoes a refute); default informs set_tiebreak; hard downgrade ablation-gated OFF in code.
- ctx.judge_panel(cands,n_judges) → PoLL vendor-diverse → median → set_tiebreak; winner still ctx.select.
- ctx.synthesize(cands,…) → literal _attempt(prompt=merge_top_K) re-scored by pytest; deterministic top-K by ranking_key with content_sha terminal tiebreak; value-strip merged diffs/excerpts via design_contract.redact (highest leak surface).
- ctx.loop_until_dry(spawn_fn,k_dry=2) → wave loop stopping on K dry OR not governor.should_continue_waves(); lifts plateau out of solve_and_repair (context.py:307-318).
- ctx.completeness_gaps(cand)/completeness_critic(cand) → surface cand.meta["failing_nodeids"] (ids not values) + optional critic lens; downgrade-only.
- Read-accessors for escalation: ctx.best_pass_rate, ctx.residual_failures, tiered ctx.worker_specs. All knobs zero → flat best-of-N.
Tests: tests/test_patterns.py (each degrades to best-of-N at zero knobs; OR-over-refute; synthesize cannot self-accept; pattern exemplar lints+runs).

### 2.4 Teaching deliverable
Files: apex_omega/autogen/architect.py:32-75 (API_REFERENCE QUALITY-PATTERNS + INVARIANTS rule 6); apex_omega/autogen/templates.py (NEW PATTERN_EXEMPLAR; DEFAULT_ORCHESTRATION unchanged except plateau).
Test: tests/test_autogen.py::test_author_prompt_includes_patterns.

**GO/NO-GO Phase 2:** every pattern with zero knobs byte-identical to flat best-of-N; no pattern can set accepted; ranking_key order test green; OR-over-refute green; escalating plan replays same branch after kill; pathological while True terminates on plateau. NO-GO if any pattern promotes acceptance or determinism-under-resume fails.

## PHASE 3 — Features re-layer + design_contract wiring
- Wire design_contract.render/redact into commit0_autogen.prompt_builder (line 255) + architect.build_author_prompt (line 127). Assert accept path byte-identical with/without contract; assert rendered contract is leak-safe.
Tests: tests/test_design_contract_wiring.py::test_contract_enriches_prompt_not_accept_path; ::test_rendered_contract_is_leak_safe.

**GO/NO-GO Phase 3:** contract present vs absent → identical accept decisions; leak-safe audit passes.

## TEST PLAN SUMMARY
Existing 94 functions are the regression baseline. New additive tests in test_mode_c, test_engine_journal, test_journal_crash_resume, test_strip_checkout, test_autogen, test_scout, test_safety_and_ablation, plus NEW test_patterns.py + test_design_contract_wiring.py. Property test: vr→to_dict→from_dict→candidate_from_verification preserves accepted/indeterminate/missing_expected/pass_rate.

## VALIDATION-RUN PLAN (equal-budget, n≥3 seeds; avoid the run-4 multi-knob confound)
Repos: voluptuous (floor/easy), jinja (run-4 SOLVE→TIMEOUT + src-layout P0.1), mimesis (verified-discard witness), pydantic (cost-pathology). All changes behind ablation flags; A/B at EQUAL agent+token+wall budget across arms.
- R0 baseline parity: current code, confirm 94 green + ladder baseline reproduces.
- R1 Phase-0+1 backbone only: assert (a) no discard — kill jinja/mimesis mid-flight, relaunch, verified solve survives; (b) resume works — full-prefix cache hits, no pytest re-run on unchanged diffs, venv not re-stripped; (c) no regression vs R0 on all four; (d) budget default unbounded; (e) per-agent watchdog fires on a synthetic slow agent without killing the cell.
- R2 patterns ablation: each pattern toggled independently at equal budget; require no solve-rate/banked-solve regression vs R1. Replay the run-4 jinja scenario (SOLVE 6ag/607s vs abstain 16ag/4000s): assert floor-first banks the cheap solve and resume prevents discard.
- R3 full stack + design_contract; n≥3 seeds; compare to R1.
Acceptance: R1 demonstrates no-discard + working resume + no regression; R2 shows no single pattern regresses at equal budget; the mimesis 6052/6052 verified-discard mode is provably closed (kill-after-green test + empirical kill).

## LOWEST-RISK FIRST CUT (kills run-4, reuses WAL): 0.1 + 0.5 + 0.6 + 1.3. BIGGEST CORRECTNESS WIN: 0.2 (cli_strict_hard_timeout is what bites). BIGGEST RESUME-COST WIN: 1.1 (journal score COUNTS). BIGGEST DETERMINISM WIN: 2.0 (journaled wave decisions).
