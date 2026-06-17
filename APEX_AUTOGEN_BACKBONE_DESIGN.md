# APEX-Ω Autogen Backbone — "Replayed-Workflow Cell": Design

> 29-agent backbone design workflow (map+research → 4 proposals → judged → synthesized → 7 adversarial validations → phased plan + critic + verified final). Every load-bearing anchor verified against the live code.

## Executive summary

APEX-Ω BACKBONE — "Replayed-Workflow Cell": the eval cell becomes a deterministic workflow replayed against a warm journal. The wall-clock guillotine becomes pause+relaunch; acceptance becomes a durable journaled event; the architect gets a downgrade-only composable pattern vocabulary that rides existing seams for free. Default is UNBOUNDED (no cost/time budget); the only always-on guards are a PER-RUN 1000-agent backstop, an always-on plateau-stop, a generous-but-finite per-agent wall, and the resumable journal — so no timer ever discards verified or in-flight work, yet the adversarial/hopeless case still terminates.

VERDICT after verifying every load-bearing anchor against the live code: the prior design+plan close run-4's SPECIFIC discard (the barrier+guillotine SIGKILL), but six adversarial validations + the completeness critique surface real residual holes that I have now folded into HARD requirements rather than honest-notes. The decisive ones, all code-confirmed:
1. SCORE-GAP (A2): pytest-green is computed OUTSIDE the journal (context.py:182); a kill in [green..accept-fsync] loses the fact and forces a cold-venv re-score that can flip to indeterminate → discard. FIX: journal verification COUNTS the instant pytest returns; re-derive accepted from committed counts (no re-run, no venv).
2. FROZEN .py STRIP BUG (B1): VERIFIED LIVE — `_KEEP_SUFFIXES` (run_ladder.py:42) lacks `.py`; the run4 archive shows the autogen jinja cell's `frozen.json` references a `<sha>.py` that is GONE while `frozen.json` survives → `load_frozen` (architect.py:159) raises on resume. FIX: add `.py` to keep-suffixes.
3. PER-RUN BACKSTOP RESET (A1): `_total_agents` is a per-Engine miss-only counter (runtime.py:68,138); relaunch resets it → R×1000 ceiling. FIX: rehydrate from journal + progress-gate relaunch.
4. DETERMINISM EROSION (C1): `agents_used()`/`budget.spent()` are miss-only; on resume they read 0 while the frozen script replays → any control-flow branch on them diverges and breaks the prefix-cache the whole resume rests on. FIX: journal wave decisions; counters become narration-only.
5. PER-AGENT WALL INERT (calibration): `cli_strict_hard_timeout` absent from config → cli_backend.py:6699 returns None → `ScopedTask.timeout_seconds` is inert; the engine watchdog can only ABANDON the wait, not kill the Popen. FIX: enable `cli_strict_hard_timeout` (the SOLE real kill); demote engine watchdog to a selection-exclusion non-result.
6. WORK-CEILING REMOVED (D1): the design drops the 40M default that was the runaway brake AND adds fan-out vocabulary; FIX: keep a default-ON soft per-cell work ceiling (difficulty soft_cap 8/24/64) + always-on plateau.

Phasing is backbone-first (Phase 0 safety floor → Phase 1 guillotine-removal+resume → Phase 2 patterns → Phase 3 feature re-layer) with a go/no-go gate per phase and an equal-budget n≥3-seed A/B validation plan to avoid the run-4 multi-knob confound.

## Confidence: overall 80%

The backbone design is sound, code-grounded, and closes run-4's specific discard — but it only achieves the FULL "never discard verified work / faithful resume / no regression" bar after a specific, ordered set of fixes that the prior design+plan listed mostly as honest-notes rather than hard requirements. I have promoted all of them to Phase-0/Phase-1 hard gates.

What is verified-true and load-bearing: the journal/sandbox/budget/select substrate is correctly built and reusable verbatim; the Cardinal acceptance gate is structurally preserved; the worktree-per-attempt model means no double-apply today; _checkpoint_accepted already banks pre-barrier so run-4's specific barrier window is closed even now; cli_strict_hard_timeout is genuinely absent from the config so the per-agent timeout is genuinely inert (the design's central calibration is correct).

What MUST be true for the headline claims, and is NOT yet (all verified against code): (1) the frozen-.py strip bug is LIVE — the run4 archive shows an authored cell's frozen.json pointing at a deleted <sha>.py — and breaks resume outright until .py is kept; (2) the score-gap loses a verified solve on a mid-gap kill unless verification COUNTS are journaled before Candidate construction; (3) the 1000 backstop is per-process not per-run until rehydrated from the journal, and relaunch must be finite+progress-gated; (4) determinism-under-resume is eroded by miss-only counters and must be fixed (journaled wave decisions) BEFORE patterns expand the counter-reading surface; (5) the per-agent kill must be the vendor-native flag, not the engine watchdog (which cannot kill the Popen); (6) a default-ON per-cell work ceiling + always-on plateau must be retained or the run-4 blowup window reopens wider.

Calibrated bottom line: with Phase 0 + Phase 1 + the determinism prerequisite (2.0) shipped and gated, the backbone meets every requirement. The residual uncertainty is empirical, not architectural — it lives in (a) confirming the resume cost numbers and the kill-after-green recovery on real repos, and (b) confirming no single pattern regresses banked-solves at equal budget. Both are addressed by the R1/R2 validation runs with explicit go/no-go gates.

| concern | conf | note |
|---|---|---|
| Run-to-completion: no wall-clock discards verified or in-flight work | 86% | G2 (relaunch over same run-dir) + score-COUNTS journaling close both the barrier-discard (run-4, already closed by _checkpoint_accepted pre-barrier banking, verified context.py:191) and the score-gap (A2, the residual I fold in: journal counts BEFORE Candidate construction so a mid-gap kill recovers without re-running pytest or needing the venv). Empirical kill-after-green test required to confirm; the mechanism is sound and code-anchored. |
| Budget+ceiling-bounded, default unbounded | 90% | Budget primitive defaults unbounded (budget.py:24); can_start() True forever when None (verified). Opt-in path clean once the fragile `budget or Budget(total=token_ceiling)` (commit0_driver.py:111) is removed (#0.1). The 1000 backstop is per-RUN only after journal-rehydration (#0.3) — without it, relaunch resets it to R×1000. High confidence given the regression test asserting budget.total is None by default. |
| Resumable/anytime-checkpointed; mid-run resume re-derives without re-running completed work | 74% | The agent-grain resume substrate is verified-correct (content-keyed, in_flight≠hit, fsync-write-ahead, no double-apply because acquire mints a fresh worktree). But 'resume is free' is FALSE until score(#1.1)+prep(#1.2)+author(#1.2) journaling lands, and the frozen-.py strip bug (#0.6, VERIFIED live in the archive) breaks load_frozen on resume TODAY. Confidence reflects that these are unbuilt; the score/worktree coupling needs the diff_sha-from-journaled-blob + scoring_env_sha guards to be correct, not just present. |
| Rich composable architect patterns | 82% | Patterns are pure ctx.ask/parallel/select compositions that pass the sandbox lint unchanged (no new builtins/dunders) and win only through ctx.select. The design law is enforceable. Confidence reserved because the soft-write seam runs host-side OUTSIDE the sandbox AST guard, so no-self-accept must be re-established at the Candidate API level (set_soft/refute + ranking_key order test) — a convention-to-structural upgrade that must actually be coded, and an exemplar must be shown to lint+run. |
| Features re-layer on top without backbone change | 78% | Verified clean for verifier-guided selection (select.py/verify.py untouched) and repair lineages (already a journaled agent). design_contract.py is imported NOWHERE today — prompt-enrichment is zero-backbone but the resume-faithful contribution requires journaled author(#8)+score(#6). The honest fix is to STATE the boundary in two halves rather than file resume-faithfulness under 'no backbone change'. |
| All invariants preserved | 80% | Gate invariants (exec-authoritative, no-self-accept, monotone-refute, soft-below-execution, abstain-on-no-accept, no-answer-leak) are verified structural and preserved. The GENUINE erosion is determinism-under-resume (C1): miss-only counters read 0 on replay, breaking the prefix-cache for any counter-branching plan. This is OPEN in the prior design and CLOSED here only by journaling wave decisions (#2.0) — a must-fix prerequisite before patterns, not a nice-to-have. |
| Vendor-agnostic | 92% | Verified: vendor is a WorkerSpec field; kernel/scoring/sandbox/journal-keys never branch on vendor; the real per-agent kill is per-rollout (terminate_for_rollout) not vendor logic. The one over-claim corrected: the engine _safe_runner watchdog cannot kill the in-flight CLI subprocess (it blocks in Popen) — demoted to a selection-exclusion non-result; the vendor-native cli_strict_hard_timeout is the SOLE real kill. That path is itself vendor-agnostic (config flag, not code branch). |
| No-regression / no run-4-style cost blowup | 70% | Lowest because the design's instinct ('never optimize for cost / default unbounded') drops the exact time/work brake run-4's post-mortem demanded AND adds fan-out vocabulary. I close it by keeping a default-ON soft per-cell agent_budget (difficulty soft_cap 8/24/64) + always-on plateau, but this is the most empirically-uncertain piece: only the equal-budget n>=3-seed A/B (R2) on jinja/mimesis can confirm no single pattern regresses banked-solves at equal budget. |

## Residual risks
- EMPIRICAL, NOT YET RUN: 'resume is near-free' and 'kill-after-green recovers from journaled counts' are mechanism-sound but unvalidated on real repos. R1 (kill voluptuous/jinja/mimesis mid-orchestrate, relaunch, assert full-prefix cache hits + no pytest re-run + venv not re-materialized + identical report) is the required proof. Until R1 passes, treat the cheap-resume language as conditional.
- COLD-VENV RE-SCORE DETERMINISM: even with score-COUNTS journaling, a kill in the narrow sub-second window before the score event fsyncs forces a re-score; if the re-prepped venv resolves drifted deps the scoring_env_sha key correctly MISSES and re-runs (correct, not free) — but if a transient eval_cap/editable-shadow indeterminate fires differently, an already-verified solve could still be excluded. The scoring_env_sha guard narrows this but does not eliminate the dependency on score_fn re-run determinism; flagged as the deepest residual.
- DETERMINISM-UNDER-RESUME is the single biggest invariant risk and the fix (journaled wave decisions, #2.0) is net-new and must land BEFORE any counter-branching pattern; if an authored plan branches on a live counter through a path the wave-journaling does not cover, replay diverges and the prefix-cache breaks silently (wrong-resume still ships only verified passes, but the resume guarantee is defeated). Needs a dedicated test for every counter-reading accessor.
- FLOOR-UNDER-RESUME for AUTHORED arms is structural only for clean-abstain rescue (verified-floor) and crash/malformed; a wave-0=8 diverse authored plan killed mid-wave before the host floor-probe's accept fsyncs still has the floor banked (probe runs first), but the rescue re-introduces a template solve into the autogen number — gated behind an ablation for the honest autogen-stands-alone comparison, which must be reported separately.
- WORK-CEILING / COST: keeping default-ON agent_budget (difficulty soft_cap) + always-on plateau is my chosen brake against run-4-style blowup, but it is a JUDGMENT that trades some of the 'unbounded run-to-completion' purity for terminating-in-practice. R2 (equal-budget A/B) must confirm the brake does not itself suppress a banked solve a longer run would have found.
- JOURNAL/DIFF-BLOB GROWTH across relaunches is bounded by content-dedup (diffs are content-addressed, identical replayed diffs do not multiply) but worktrees are released per-attempt while diff blobs accumulate; on a near-full disk the only guard is MIN_FREE_MB which refuses to START, never stops a growing run. Documented worst-case footprint ~ unique-fresh-agent diffs; acceptable but named.
- HOST-SIDE SOFT-WRITE SEAM: patterns execute outside the sandbox AST guard, so no-self-accept is convention until Candidate.set_soft/refute + the ranking_key-order pin test are actually coded; a future edit inserting a soft key above int(accepted), or a pattern using bare setattr, would silently erode the gate. The regression tests are the only structural backstop here.
- VENDOR-NATIVE KILL ORPHAN: if cli_strict_hard_timeout's terminate_for_rollout fails to reap a child, the engine watchdog returns a non-result but the orphaned process could still write into a released worktree; needs a PID-dead assertion in the validation, not just a returned-non-result assertion.

## Primitives API

## PROPOSED ctx / architect API SURFACE (signatures + 1-line semantics)

### Atoms (the only foundations; everything else composes these)
- `ctx.ask(prompt, *, schema=None, vendor=None, model=None, agent_id=None) -> dict|str|None` — read-only schema'd subagent run to completion; produces SIGNALS never Candidates (cannot self-accept); journaled/resumable for free.
- `ctx.solve_attempt(*, strategy=None, vendor=None, model=None, prompt=None, attempt_id=None) -> Candidate|None` — one isolated coding-agent attempt, scored by real pytest (EXISTING; refactored onto shared `_attempt`).
- `ctx.select(candidates) -> Candidate|None` — the ONLY producer of a winner; execution-authoritative; abstains (None) if none accepted (EXISTING, unchanged).

### Soft-write seam (host-side; the ONLY soft→kernel writes; structurally narrowed)
- `ctx.review(cand, reviews) -> Candidate` — applies SoftReviews via apply_evidence_bound_review (downgrade-only; can ONLY flip accepted True→False).
- `ctx.set_tiebreak(cand, *, perspective=None, eg_critic=None) -> Candidate` — writes ONLY perspective_score/eg_critic_tiebreak (strictly below execution keys) via the new `Candidate.set_soft`.
- `Candidate.set_soft(perspective=None, eg_critic=None)` — structural: can touch ONLY those two fields. `Candidate.refute()` — structural: can ONLY flip accepted True→False. (No setter for accepted=True anywhere.)

### Patterns (each a few lines over the atoms + existing primitives)
- `ctx.adversarial_verify(cand, n, *, perspectives=None, vendor_diverse=True) -> list[SoftReview]` — N skeptics per finding; OR-over-refute aggregation; default informs set_tiebreak; hard downgrade ablation-gated OFF.
- `ctx.judge_panel(cands, n_judges) -> list[Candidate]` — PoLL vendor-diverse scorers → median → set_tiebreak; winner is STILL ctx.select.
- `ctx.synthesize(cands, *, k=3) -> Candidate|None` — literal solve_attempt(prompt=merge_top_K) re-scored by pytest; deterministic top-K by ranking_key+content_sha; value-stripped merge prompt; cannot self-accept.
- `ctx.loop_until_dry(spawn_fn, *, k_dry=2) -> list[Candidate]` — wave loop stopping on K dry rounds OR not governor.should_continue_waves(); the run-to-completion replacement for the guillotine.
- `ctx.completeness_gaps(cand) -> list[str]` — surfaces cand.meta["failing_nodeids"] (ids, not values) to authored code for next-wave routing.
- `ctx.completeness_critic(cand) -> list[SoftReview]` — optional critic lens; downgrade-only.

### Governor / read-accessors (narration-only; never drive replayed control flow directly)
- `ctx.should_continue_waves(frontier) -> bool` — host-side RunGovernor hook; on replay reads the JOURNALED wave decision, not the live miss-only counter (determinism).
- `ctx.best_pass_rate(cands) -> float`, `ctx.residual_failures(cands) -> list[str]` — for architect-composed Snell/FrugalGPT/ESC escalation; all knobs zero → flat best-of-N.
- `ctx.agents_used() -> int`, `ctx.budget` — EXISTING; now narration-only (authored control flow must key off journaled wave state, not these).
- `ctx.plan_waves(...) -> list[int]`, `ctx.parallel(thunks)`, `ctx.pipeline(items,*stages)`, `ctx.phase/log` — EXISTING, unchanged.

### Engine / host (not exposed to the sandbox)
- `RunGovernor(*, engine, agent_ceiling=1000, token_budget=None, agent_budget=None, plateau_k_dry=2)` — `.can_start(*, reserve=1) -> bool`; `.should_continue_waves(*, dry_rounds, best_pass_rate, prev_best) -> bool` (NO clock).
- `Journal.fresh_agent_count() -> int` — committed-OK kind=="agent" tally for per-run backstop rehydration.
- `Journal.last_accept() -> dict|None`; `resume_or_run_json(..., kind in {"score","prep","author","accept","wave"}, status_fn=...)` — EXISTING signature, new kinds.
- `VerificationResult.from_dict(d) -> VerificationResult` — NEW; lossless for acceptance fields (advisory truncated fields knowingly lossy).
- `ScopedTask.heartbeat_timeout_seconds: Optional[int]` — NEW; `timeout_seconds` decoupled from cell wall (per-agent).

### Design law (hard constraint on all of the above)
Every primitive is pure composition of `engine.agent()` + `engine.parallel()` + `select_best()` + existing kernel seams. NO new acceptance authority, persistence, or control-flow runtime ⇒ all inherit journaling/resume/concurrency/budget/determinism for free; all still WIN only through `ctx.select`.

## Phasing

BACKBONE FIRST, then features — strict ordering with a go/no-go gate per phase.

PHASE 0 — Backbone safety floor (runaway + discard holes). Steps 0.1 budget opt-in, 0.2 always-on finite per-agent wall (cli_strict_hard_timeout + decouple + demoted watchdog), 0.3 per-RUN backstop 1000 + journal rehydration, 0.4 parallel 4096 guard, 0.5 host-side floor-probe, 0.6 frozen-.py strip fix.
GO/NO-GO: all 94 baseline + new tests green; budget.total is None by default; slow FakeExecutor agent excluded+re-runs without killing the cell; 999-agent journal admits exactly 1 fresh across a new Engine; floor candidate banked before the authored script. NO-GO if any baseline regresses or budget default != None.

PHASE 1 — Remove guillotine + full mid-run resume. Steps 1.1 journal score COUNTS (+from_dict, diff_sha-from-blob, scoring_env_sha), 1.2 journal prep+author (reroute direct session.run), 1.3 finite progress-gated relaunch loop (no inter-attempt strip; diff_ref join), 1.4 accept WAL event (evidence not authority; score-then-accept ordering).
GO/NO-GO: empirically kill a real autogen cell mid-orchestrate → relaunch shows full-prefix cache_hit:true, zero re-dispatched cached agents, venv not re-materialized, identical report; SIGKILL after green before accept-fsync → solve recovered from journaled counts (no pytest re-run); no baseline regression. NO-GO if resume re-runs pytest for an unchanged diff, any verified solve is lost, or relaunch loops without progress.

PHASE 2 — Composable architect vocabulary. Step 2.0 (PREREQUISITE) determinism-under-resume via journaled wave decisions + counters→narration-only, then 2.1 RunGovernor + always-on plateau, 2.2 atoms + structural soft-write seam (set_soft/refute + ranking_key-order pin), 2.3 the six patterns (OR-over-refute, value-strip synthesize), 2.4 teaching deliverable.
GO/NO-GO: every pattern with zero knobs is byte-identical to flat best-of-N; no pattern can set accepted; ranking_key-order + OR-over-refute tests green; an escalating plan replays the same branch after a kill; a pathological while True terminates on plateau. NO-GO if any pattern promotes acceptance or the determinism-under-resume test fails.

PHASE 3 — Features re-layer. Wire design_contract into the prompt builders only.
GO/NO-GO: contract present vs absent → identical accept decisions on the validation repos; leak-safe audit passes.

CROSS-PHASE VALIDATION GATE (must pass before declaring done): R0 baseline parity → R1 (Phase 0+1 backbone only: no-discard + working resume + no regression on voluptuous/jinja/mimesis/pydantic, n>=3 seeds, equal budget) → R2 (patterns ablation, each toggled independently at equal budget, no banked-solve regression; replay the run-4 jinja SOLVE→TIMEOUT scenario and assert floor-first banks the cheap solve + resume prevents discard) → R3 (full stack + design_contract). Acceptance: the mimesis 6052/6052 verified-discard mode is provably closed (kill-after-green test + empirical kill) and no single pattern regresses at equal budget.

---

# APEX-Ω UNIFIED BACKBONE DESIGN — "Replayed-Workflow Cell" (FINAL)

All anchors below are verified against the live tree at /Users/sameertkhanna/Documents/agent_orch.

## 0. The model
The eval cell is a deterministic `orchestrate(ctx)` workflow (already authored, frozen, lint-checked, journaled). We re-align the RUNTIME GUARANTEES to the dynamic-workflow reference model: agents run to completion; default UNBOUNDED; bounds are budget/ceiling not time; resume from a runId returns the longest-unchanged prefix as cached hits. The single deepest invariant resolution: **the journal persists replayable EXECUTION EVIDENCE (verification counts + diff); the host RE-DERIVES `accepted` through the existing candidate_from_verification/select path; all time/budget/counter state stays in the host runtime, invisible to the deterministic sandbox.**

## 1. RUNTIME MODEL (verified call chain)
```
run_ladder.run_cell (Tier-0: relaunch loop, FINITE+progress-gated, NOT guillotine)
  └─ subprocess: apex_omega eval --run-dir <SAME> --cell-timeout <soft>
       └─ commit0_driver.run_cell  [kind=commit0_cell, journaled]
            └─ commit0_autogen.run_autogen_cell
                 ├─ prep            [NEW kind=prep — env DESCRIPTION, not bytes]
                 └─ autosolve(engine, ctx)
                      ├─ agent_scout         [kind=agent type=scout — journaled today]
                      ├─ FLOOR-PROBE (host)  [NEW: 1 cheap template cand banked FIRST]
                      ├─ author_orchestration[kind=author — NEW journaled]  → freeze(<sha>.py KEPT)
                      └─ run_orchestration(frozen.source, ctx)   ← REPLAYED top-to-bottom
                           orchestrate(ctx): patterns over atoms
                             └─ engine.agent [kind=agent — journaled; HIT=free]
                                  └─ _scored  [NEW kind=score — COUNTS journaled BEFORE Candidate]
                                       └─ accept [NEW kind=accept — evidence, not authority]
```

### 1.1 Bounds — the `RunGovernor` (engine/governor.py, NEW)
The ONE "may we continue" object. It is NOT a timer.
- **Always-on (cannot disable):** PER-RUN `agent_ceiling=1000` (runtime.py:51 default flipped 2048→1000, rehydrated from journal so it bounds the whole resumable run); always-on plateau-stop (k-dry / no-pass-rate-improvement, host-enforced regardless of authored code); a generous-but-finite per-agent wall (`cli_strict_hard_timeout` enabled in config); the resumable journal.
- **Opt-in (default None=unbounded):** `token_budget` (Budget), `agent_budget` (soft per-cell cap < ceiling; default = difficulty soft_cap 8/24/64 to preserve the run-4 work brake), `start_to_close` per-agent override.
- `can_start()` delegates to `Budget.can_start` (budget.py:49-55, gate-on-start only, never aborts in-flight) + agent_budget + per-run ceiling.
- `should_continue_waves()` is pure budget + plateau, NO clock; host short-circuits the next `parallel`/`solve_attempt` to `[]`/`None` with a narrated `plateau_stop` when it returns False — so even `while True: ctx.parallel([...])` terminates.

### 1.2 Two-level liveness/relaunch (closes run-4)
1. **CELL → pause+relaunch (load-bearing, G2).** run_ladder.run_cell replaces the `except`-marks-error guillotine with a FINITE, PROGRESS-GATED relaunch loop over the same --run-dir. Relaunch only if the journal shows new committed accepts/score-improvements/fresh agents since the last attempt; cap default LADDER_MAX_RELAUNCH=8. The journal is the true bound; the cap+gate keep the adversarial case terminating. Do NOT `_strip_checkout` between attempts (keep venv warm).
2. **AGENT → real finite wall (supporting, G1).** Decouple per-agent timeout from the cell wall (context.py:175,248 stop passing `self.timeout_seconds`); add `heartbeat_timeout_seconds` to ScopedTask. The REAL bound is vendor-native `cli_strict_hard_timeout=true` + `cli_hard_timeout_seconds=2400` in configs/base_commit0_local.json (cli_backend.py:6699/8536 enforce a real Popen kill via terminate_for_rollout). The engine `_safe_runner` watchdog is DEMOTED to a selection-exclusion non-result (it abandons the wait, returns `heartbeat_timeout`→RESULT_INFRA_NONRESULT; it does NOT kill the subprocess — verified: `_safe_runner` calls runner synchronously into a blocking Popen).

### 1.3 Resume (Detect/Replay/Resume)
On relaunch the child reattaches the warm WAL (Engine.__init__→Journal._recover, wal.py:138), `load_frozen` skips re-authoring (architect.py:153 — REQUIRES the .py-strip fix), and `run_orchestration(frozen.source, ctx)` re-executes. Every prior agent+score+author+scout+prep is a content-keyed HIT (free, no budget charge, runtime.py:160-161). Determinism holds because the sandbox forbids clock/RNG/counter (sandbox.py:22-28) AND — the new requirement — authored control flow keys off journaled wave decisions, not miss-only live counters.

Honest calibration: resume is near-free only after the score+prep keys land. `prep` journals the env DESCRIPTION, not venv bytes — a cold resume after `_strip_checkout` still re-materializes the venv (the one unavoidable cost). The score-COUNTS journaling removes the venv from the recovery path entirely for an already-verified solve.

## 2. INVARIANTS PRESERVED (all verified structural, with the new hardening)
- **Execution-authoritative acceptance:** select_best returns only accepted (select.py:114-116); accepted comes only from candidate_from_verification(vr.accepted) (verify.py:69). `accept` event stores EVIDENCE (counts+diff_ref), never authority; recovery re-derives via the score HIT. Atoms (`ctx.ask`) produce signals never Candidates; `ctx.synthesize` is a literal solve_attempt re-scored by pytest → cannot self-accept.
- **Monotone downgrade-only:** apply_evidence_bound_review (select.py:91-95) has no True-setting branch (verified). Panels are OR-over-refute (any refute downgrades, no quorum vetoes), enforced in code+test.
- **Soft strictly below execution:** ranking_key (select.py:63-75) puts combined_score/accepted/public_signal_score/critic_score ABOVE eg_critic_tiebreak/perspective_score. NEW: host-side patterns sit OUTSIDE the sandbox AST guard, so the no-self-accept guarantee is re-established at the Candidate API level (`Candidate.set_soft`/`refute` — the only soft-write methods; bare setattr never used) + a regression test pinning the ranking_key tuple order.
- **Determinism/replayability:** sandbox unchanged; counters→narration-only; wave decisions journaled.
- **Vendor-agnostic:** vendor is a WorkerSpec field; kernel/scoring/sandbox/journal-keys never branch on vendor; the real per-agent kill is per-rollout (terminate_for_rollout), not vendor-specific logic.
- **Fewest-agents-first + best-of-N floor:** plan_waves doubling preserved (context.py:340-380); floor made STRUCTURAL via host-side floor-probe banked before the authored script + `_floor()` on crash/malformed unchanged (architect.py:379-385).
- **No answer-leak:** select.py/verify.py untouched; full-suite EXACT-id re-run gates acceptance; synthesize value-strips merged diffs/excerpts via design_contract.redact before they enter prompts.

## 3. FEATURES RE-LAYER (boundary stated honestly in two halves)
- (A) PROMPT-ENRICHMENT (zero backbone): wire design_contract.render/redact into commit0_autogen.prompt_builder (line 255) and architect.build_author_prompt (line 127). Verified: design_contract.py is imported NOWHERE today — this is "designed, integrate now," and a test must assert the accept path (scoring.decide_*) is byte-identical with/without the contract.
- (B) RESUME-FAITHFUL (requires backbone): the contract's contribution to the replayed input-hash requires the journaled `author` step (#8); accept-as-evidence requires the journaled `score` step (#6). These are NOT "no backbone change" and are filed separately.
- Repair lineages (context.py:206-329): already a journaled agent(); `loop_until_dry` can drive them; repair_iters ceiling (default 0) clamps; provably never worse than flat best-of-N at 0.
- Verifier-guided selection / Cardinal Contract: untouched; the accept event records what select decided, it cannot make the decision.

## 4. COMPONENT→FILE MAP
- Journal/resume substrate (REUSE VERBATIM, verified correct): apex_omega/journal/{wal.py,resume.py,key.py} — content-keyed, fsync-write-ahead, in_flight≠hit, RESULT_INFRA_NONRESULT not indexed (wal.py:244), indeterminate→non-result.
- Engine: apex_omega/engine/{runtime.py(governor wiring, counters), budget.py(verbatim), governor.py(NEW)}.
- Autogen: apex_omega/autogen/{context.py(score/atoms/patterns), architect.py(floor-probe, author journaling), templates.py(plateau+PATTERN_EXEMPLAR), sandbox.py(verbatim)}.
- Patterns: apex_omega/patterns/* (NEW).
- Eval: apex_omega/eval/{commit0_autogen.py(prep,score key threading), commit0_driver.py(budget opt-in), scoring.py, design_contract.py(wire)}.
- Kernel: apex_omega/kernel/{select.py(Candidate.set_soft/refute), verify.py(from_dict)}.
- Types: apex_omega/types.py(heartbeat_timeout_seconds).
- Vendor: apex/core/cli_backend.py:6699/8536 (already supports the path; config enables it).
- Harness: scripts/run_ladder.py(.py keep-suffix, relaunch loop, diff_ref join).
- Config: configs/base_commit0_local.json (cli_strict_hard_timeout).

## 5. HONEST CALIBRATION (carried into hard guarantees)
- Per-agent decouple is INERT alone; the real bound is `cli_strict_hard_timeout` (config); the engine watchdog only records a non-result.
- Resume is near-free only after #6(score)+#7(prep); prep journals env description not bytes.
- `accept` stores evidence; acceptance re-derived through the score HIT, never a stored boolean.
- The 1000 backstop is per-RUN only after journal-rehydration; relaunch is finite+progress-gated so the adversarial case terminates.
- Determinism under resume REQUIRES journaled wave decisions; miss-only counters must not drive authored control flow.
- The template floor is structural under resume/clean-abstain only after the host-side floor-probe; the verified-floor rescue is ablation-gated for the honest autogen-stands-alone comparison.
