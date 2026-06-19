# APEX-Ω Phased-Planner Research Context

Single grounded synthesis for building a Claude-Code-style PHASED planner + adversarial
goal-review on top of the existing APEX-Ω dynamic-workflow engine, tailored to the hard
commit0 discriminators (mimesis, babel). Every claim below is grounded in code I read or in
the fact-checked external research bundle; source families/files are cited inline.

Sources of ground-truth:
- Code: `apex_omega/autogen/{architect.py, templates.py, context.py}`,
  `apex_omega/eval/{commit0_driver.py, commit0_autogen.py}`, `scripts/run_ladder.py`.
- Empirical: `APEX_COMMIT0_REPORT.md` (runs 1-4, re-verified against run archives).
- Paradigm: `Anthropic_Dynamic_Workflows_Authoritative_Guide_Expanded.md`.
- External (fact-checked): family `claude-code-dynamic` — Anthropic multi-agent-research,
  building-effective-agents, swe-bench-sonnet, effective-context-engineering; Claude Code
  `/goal` docs + VentureBeat; DataCamp Opus-4.5 + best-practices; arXiv:2603.05344 (OpenDev,
  independent — NOT Anthropic). Refuted/uncertain items flagged in §3.

---

## 1. Current orchestrator: capabilities + GAPS vs a Claude-Code-style phased planner

### 1a. What exists today (the single-shot, single-plan architecture)

The planner is **scout -> author ONE script -> freeze -> execute once -> fail-open**
(`architect.py:autosolve` 454-683). Concretely:

- **SCOUT** — `build_repo_map` (`architect.py:183-216`) enumerates source/test files and a
  crude file-count difficulty proxy (easy<15, medium<80, hard>=80 source files). Optional
  `agent_scout` (`architect.py:405-451`) runs a 3-lens READ-ONLY fan-out (implementation
  surface / test topology / edge cases), median-aggregated, returning `SCOUT_SCHEMA
  {difficulty, approach, key_files, risks}`. `autosolve` caps scouts to `budget//6`
  (`architect.py:498`) and takes the LOWER of static-proxy vs scout difficulty
  (`architect.py:511-513`) — anti-inflation, never escalates budget up.
- **AUTHOR** — `author_orchestration` (`architect.py:297-347`) resumes a prior frozen script,
  or freezes a FIXED template (`ralph`/`converge`/`rebuild` selectors, or `author=False`), or
  calls `_author_via_llm` (`architect.py:350-357`): ONE read-only architect agent given
  `build_author_prompt` (`architect.py:232-270`) = API_REFERENCE + INVARIANTS +
  difficulty-gated PRIMARY exemplar (CONVERGE_EXEMPLAR for medium/hard, BEST_OF_N for easy) +
  PATTERN_EXEMPLAR + the repo map + binding task_framing. Authored source is `extract_code`'d,
  `lint_source`-checked, frozen only if lint passes; lint-fail -> DEFAULT_ORCHESTRATION.
- **FREEZE** — `_freeze` (`architect.py:273-283`) content-hashes the source (sha256), writes
  `orchestrator/<sha>.py` + `frozen.json` with an origin tag. `load_frozen` reuses it on
  resume so a run never re-authors.
- **EXECUTE + FAIL-OPEN** — `autosolve` (`architect.py:580-628`) runs a host-side floor-probe
  (banked best-of-N wave-0, rescue OFF by default), then `run_orchestration(frozen.source,
  ctx)`. `PlateauStop` -> select best banked authored candidate; `FailLoud`/any Exception ->
  `_floor()` runs BEST_OF_N directly. review-fix #8 (`architect.py:615-619`) reconciles a
  banked verified candidate if the run abstained. There is **deliberately no
  no-winner-fall-open-to-template** (`architect.py:601-607`): a clean abstain is autogen's
  real result.
- **THE "CONVERGE" DEFAULT** (`templates.py:71-158`) is a FIXED 5-stage script:
  (0) gate `ctx.decompose()` to non-easy >=2-module repos (skip-gate `templates.py:83-88`
  prevents 5-6x over-spawn on voluptuous/jinja); (1) `carry_best()` + `fanout_modules` per
  module; (2) `reduce_residuals` merges diffs + scores the full suite ONCE, with a
  no-silent-loss collapse fallback (`templates.py:109-126`); (3) loop-until-dry repairing the
  residual on the live merged tree; (4) verify (medium/hard): `adversarial_verify(n=3)` +
  `completeness_critic`, re-select. **`ctx.phase()` labels are narration ONLY.**

The `ctx` API (`context.py`, 1515 LOC) is a rich capability-restricted surface that ALREADY
contains the substrate a phased planner needs: `decompose` (1155-1200, returns
modules{module, gold_test_ids, depends_on, files}+order), `carry_best`/`reduce_residuals`
(cross-phase state carry), `ask`/`signals`/`quarantined_ask` (read-only schema'd sub-questions
= the objective/review substrate), `defer`/`blocked` (712-723, IOU between phases),
`adversarial_verify`/`adversarial_filter`/`completeness_critic`/`judge_select` (quality gates),
`workflow` (639-652, ONE-level nested composition via a namespaced child journal sharing the
engine/budget), `args` (614-618, the launch payload), and `should_continue_waves` +
the SPFG+ governor (579-587, 430-577, resume-deterministic per-phase stop authority).

### 1b. Precise GAPS vs a Claude-Code-style phased planner

1. **No explicit ordered PHASE list with objectives + per-phase acceptance.** The only
   acceptance in the entire system is the engine-owned WHOLE-SUITE green via `ctx.select`
   (`context.py:1467-1469`). "Phases" are `ctx.phase(title)` narration strings
   (`templates.py:77,91,95,134,151`) with no objective, no acceptance predicate, no gating.
   There is no "phase-1 objective met -> proceed to phase-2".
2. **Authoring is ONE-SHOT for the whole repo, not per-phase.** `_author_via_llm` is called
   exactly once (`architect.py:530`) producing ONE frozen `orchestrate(ctx)` BEFORE any agent
   runs. No loop regenerates orchestration code for phase N after observing phase N-1's
   results; the architect never sees execution feedback (the prompt has only the static repo
   map + scout notes, `architect.py:232-270`).
3. **No adversarial GOAL-ALIGNMENT review gating phase progression.** Adversarial primitives
   operate on Candidates (`adversarial_verify`) or plain-data findings
   (`adversarial_filter`) — none reviews a PLAN/objective against the goal, and none gates
   whether to advance. The authored plan itself is never adversarially reviewed (only
   lint-checked, `architect.py:342`). There is no "does this phase plan still serve goal G?
   proceed/revise/abort" gate.
4. **`decompose` is a one-time WAVE-0 spatial module split, not iterative phase planning.**
   It is called once (`templates.py:84`) and returns independent MODULES (spatial slices),
   carrying `depends_on` + a topological `order` — but the converge script **ignores the
   order** and fans out ALL modules in parallel (`fanout_modules`). It is not ordered
   temporal/dependency PHASES with objectives.
5. **No per-phase PARTIAL acceptance / checkpoint of phase objectives.**
   `_checkpoint_accepted` (`context.py:372-392`) only banks a FULL verified solve;
   `carry_best` carries the highest-gold-passed partial diff but nothing records "phase-2
   objective (these gold ids) is satisfied, lock it and move on." Every `reduce` re-scores
   the FULL suite; the loop stops only on full acceptance or governor cut.
6. **The convergence shape is HARD-CODED, not planned.** An author can produce a variant but
   the SAME single-pass structure gets frozen. No planner chooses a DIFFERENT phase sequence
   per repo (e.g. "phase 1: data models -> phase 2: parsers -> phase 3: API") with code per
   phase.
7. **Scout/decompose run at SHALLOW depth.** `build_repo_map` caps at 50 modules / 40 sample
   files (`architect.py:209-212`); `agent_scout` uses 3 fixed lenses with no recursive
   drill-down. No progressive task breakdown into manageable chunks-with-plans like Claude
   Code's Plan Mode.
8. **`workflow()` nesting is capped at ONE level** (`context.py:646-647` raises FailLoud), so
   phase-orchestrators each composing sub-workflows are structurally limited to depth 1 —
   a phased planner should use a HOST-SIDE phase loop, not deep nesting.

---

## 2. The eval-harness acceptance-loss seam + what checkpointing MUST do

### 2a. How one Mode-C cell runs, and where a verified solve is lost

1. `scripts/run_ladder.py:run_cell` (the OUTER process) builds `cmd = [VENV, -m, apex_omega,
   eval, ..., --run-dir, --cell-timeout]` and launches `subprocess.run(..., timeout=
   CELL_TIMEOUT+600)` (`run_ladder.py:421,452`). `CELL_TIMEOUT` now defaults to 86400s
   (`run_ladder.py:64`); in run-4 it was 3600s — **this outer subprocess wall is the
   guillotine that killed the mimesis solve.**
2. The child `commit0_driver.py:Commit0EvalDriver.run_cell` (150) journals the cell and routes
   autogen to `_invoke_autogen` (271-293) -> `commit0_autogen.run_autogen_cell`. The driver's
   OWN `_run_mode_a` subprocess also has a hard `timeout=self.cell_timeout_seconds`
   (`commit0_driver.py:255-263`) that returns an infra non-result on `TimeoutExpired`.
3. Inside, `autosolve` runs the orchestrator; a winner is banked ONLY by `ctx.select`
   returning an `.accepted` candidate. **The loss (run-4, `APEX_COMMIT0_REPORT.md` §6):** in
   BOTH orchestrated mimesis cells an attempt computed a verified `6052/6052` full pass
   (per-eval `report.json` exitcode:0, e.g. `autogen_mimesis_2/test_output.txt = 6052 passed`),
   but the cell ERRORed at the outer wall (subprocess `TimeoutExpired`) BEFORE `ctx.select`
   banked the cell-level winner — so no finalized cell result was written and the per-eval
   pass was thrown away. The per-eval report proves the solve was computed; the **bank** is
   what was absent. This is the #1 unfixed bottleneck and is arm-GENERAL (template lost two
   passes the same way).

### 2b. What is ALREADY partially built (do not rebuild — extend)

A checkpoint mechanism now EXISTS and is partly wired:
- `context.py:_checkpoint_accepted` (372-392) atomically writes
  `accepted_checkpoint.json` (candidate_id, content_sha, pass_rate, score, repo) the instant a
  candidate is accepted; idempotent (first accept wins). It is called from the per-attempt
  body (`context.py:842`, gated on `checkpoint=` so the floor-probe doesn't fake a solve) and
  from `reduce_residuals` (`context.py:1344`, only when `cand.accepted`).
- `run_ladder.py:_recover_checkpoint` (267-279) rglobs `accepted_checkpoint.json` and, on an
  outer kill, the relaunch path (`run_ladder.py:470-501`) prefers it and emits `done`+solved
  instead of ERR. `scripts/validate_checkpoint.py` exists as a regression test.

### 2c. What acceptance-checkpointing MUST do (the remaining gaps to close)

The execution-grounded Stop-hook (external family `claude-code-dynamic`, novel idea #1 —
copy `/goal`'s loop shape but ground it in real pytest, which is provably STRONGER than
`/goal`'s transcript-only verifier) must guarantee:

1. **Bank the INSTANT a verified pass appears, before any barrier.** The `ctx.select`/
   `ctx.parallel` barrier must NOT be the point of banking — the per-attempt body already
   does this (`context.py:842`), but the floor-probe path uses `checkpoint=False`
   (`architect.py:566`) by design and the `reduce_residuals` path banks only whole-suite
   accepts (`context.py:1344`). A per-PHASE pass (a subset of gold ids green) currently has
   NO checkpoint — close this for partial/phase acceptance.
2. **Survive an outer subprocess kill.** Atomic temp-write + `replace` (already done,
   `context.py:388-390`); recovery already prefers it (`run_ladder.py:470`). Ensure the child
   driver's own `TimeoutExpired` path (`commit0_driver.py:259`) ALSO consults the checkpoint
   before returning the infra non-result (Tier-1.1 in `APEX_COMMIT0_REPORT.md:190` names
   `commit0_autogen.py` + `run_ladder.py:169-174` as the wiring points).
3. **Survive a budget-aware eval timeout.** Tier-1.2 (`APEX_COMMIT0_REPORT.md:191`):
   `evaluate_repo` accepts `timeout_seconds` but it is never passed; pass
   `min(1800, remaining_cell_budget*0.4)` so one candidate can't eat 1800s of a 3600s cell.
   A scoring timeout already maps to `indeterminate` (safe). Per-repo override
   `mimesis: {evaluation_timeout_seconds: 2700}` (Tier-1.4, config-only).
4. **Never score a timeout/infra-kill as `solved:0`.** Tier-1.3: any cell with
   `wall_s>=CELL_TIMEOUT` or a null/partial result -> `status:"timeout"`/infra_nonresult,
   EXCLUDED from denominators with a `nonresult_reason`.
5. **Stay execution-authoritative.** The checkpoint records a candidate that REAL pytest
   accepted; it never lets the orchestrator self-declare. This preserves the Cardinal Contract
   and is the engine's moat over `/goal` (whose evaluator "runs nothing... a confident
   summary of broken work reads as 'fine'" — fact-checked corroborated, `code.claude.com/goal`).

---

## 3. Fact-checked catalog of external approaches (family `claude-code-dynamic`)

Only SURVIVING claims; refuted/uncertain flagged. Credibility per the verify pass.

### CORROBORATED (high confidence) — safe to design on

- **Orchestrator-worker multi-agent w/ durable plan** (`anthropic.com/engineering/
  multi-agent-research-system`). LeadResearcher analyzes the query, **SAVES ITS PLAN TO
  MEMORY** (durability across context reset), spawns 3-5 parallel subagents each with
  *an objective, an output format, tool/source guidance, and clear task boundaries*,
  synthesizes, decides if more is needed, then a separate CitationAgent runs. **Decomposition
  mechanism:** dedicated lead role decomposes into parallel subtasks each with explicit
  objective+boundaries; plan persisted before work. **Reported result:** +90.2% over
  single-agent Opus 4 on Anthropic's INTERNAL research eval (caveat: internal, not commit0,
  not independently reproduced).
- **Vague delegation is THE documented failure** (same source, corroborated verbatim incl.
  the 2021-chip-crisis-vs-2025-supply-chain duplication anecdote and the "50 subagents for a
  simple query" over-spawn bug). **Plan-adherence mechanism:** DETAILED DELEGATION CONTRACTS
  (objective + output format + tool guidance + EXPLICIT BOUNDARIES) cure duplicate-work/gaps.
  **Scaling heuristic** (corroborated): simple=1 agent/3-10 calls; comparison=2-4
  subagents/10-15 calls; complex=10+. Directly validates APEX-Ω's decompose-gating-on-easy.
- **`/goal` separates the worker model from the evaluator model** (`code.claude.com/docs/en/
  goal` + VentureBeat, corroborated, v2.1.139). Session-scoped prompt-based Stop hook: after
  every turn, condition + transcript -> a small fast model (default Haiku) -> yes/no + reason;
  "no" feeds the reason back as next-turn guidance. **CRITICAL CAVEAT (corroborated):** the
  evaluator "only reads the conversation... runs nothing... a confident summary of broken work
  reads as 'fine'." => APEX-Ω's execution-authoritative accept is STRICTLY STRONGER; keep
  `ctx.select`-may-abstain, import only the LOOP SHAPE + durable plan, NOT the weak verifier.
- **Context engineering for long-horizon** (`anthropic.com/engineering/effective-context-
  engineering`). Three matched techniques: compaction (preserve architectural decisions /
  unresolved bugs / 5 most-recent files), **sub-agent isolation** (a subagent burns 10k+
  tokens but returns a 1-2k-token distilled summary to the lead), and note-taking/persistent
  memory outside context (Claude Plays Pokemon reads its own NOTES.md after a context reset).
  Match technique to task: compaction for back-and-forth, note-taking for "iterative
  development with clear milestones", multi-agent for parallel exploration.
- **Minimal-scaffolding + error-proof tools + TDD beats orchestration depth**
  (`anthropic.com/engineering/swe-bench-sonnet`, corroborated). 2 tools (bash +
  str_replace_editor); reproduce->fix->verify baked in. Documented failure = APEX-Ω lesson #1:
  with HIDDEN tests the model "cannot see the tests it's being graded against" and "'thinks'
  it has succeeded when the task is a failure" => self-assessment is unreliable, only real
  execution counts. **Reported:** 49% SWE-bench Verified (Claude 3.5 Sonnet new, vs 45% prior
  SOTA).
- **Plan Mode persists plans to `~/.claude/plans/` surviving compaction/restarts** (DataCamp,
  corroborated). Maps to a durable phase-plan + residuals ledger on disk.
- **Cost / token-variance** (multi-agent-research-system, corroborated): agents ~4x chat
  tokens, multi-agent ~15x; on BrowseComp **token usage alone explains 80% of performance
  variance** (3 factors -> 95%). => more parallel agents help mainly by spending more tokens,
  so a win MUST be banked or it is pure waste (exactly run-4's discarded mimesis solve).
- **Budget-aware effort lever** (DataCamp Opus-4.5, Anthropic-reported): ~76% fewer output
  tokens at MEDIUM effort while matching prior best SWE-bench. (Caveat: Anthropic marketing
  numbers; SWE-bench 80.9%, Terminal-bench 59.3% — not commit0, not reproduced.)

### UNCERTAIN — design tolerant of these being wrong

- **arXiv:2603.05344 "7-section plan" + `spawn_subagent(type='Planner')`.** The paper is REAL
  (OpenDev, Rust, an INDEPENDENT open-source agent — NOT Anthropic/Claude Code). Plan Mode vs
  Normal Mode, a read-only Planner subagent, system reminders countering "instruction
  fade-out", doom-loop detection, iteration cap, and a completion signal are CONFIRMED. The
  literal 7 named sections and the exact `spawn_subagent` signature did NOT surface in
  retrieved excerpts — treat the specific schema as not-yet-verified. **Transferable shape**
  (a read-only planner emitting a structured plan with explicit verification criteria) is
  sound; do not over-commit to the exact 7 fields.

### Plan-adherence: the FOUR stacked no-veer mechanisms (all transferable)

1. Detailed delegation contracts per subtask (objective + output format + tool guidance +
   explicit boundaries) — corroborated cure for duplicate-work/gaps.
2. Exactly-one-`in_progress` TODO discipline + a nag reminder if the list goes untouched
   (>=3 rounds) — forces sequential focus. (TodoWrite mechanics from secondary source
   shareAI-lab; the nag-reminder + exactly-one discipline is the transferable idea.)
3. Event-driven SYSTEM REMINDERS re-injecting the goal "at the point of decision" to fight
   instruction fade-out + doom-loop detection + iteration caps (arXiv:2603.05344, confirmed
   shape).
4. A SEPARATE VERIFIER deciding completion (`/goal` Stop-hook). Deepest lesson: the verifier
   MUST be grounded in reality — `/goal`'s blind spot is precisely the failure APEX-Ω already
   avoids by accepting only a real gold-pytest pass.

---

## 4. Hard empirical constraints any design MUST respect

From `APEX_COMMIT0_REPORT.md` + project memory + the fact-checked cost lessons:

C1. **ACCEPTANCE-CHECKPOINTING is the #1 bottleneck** (report §1, §6, Tier-1.1). A verified
    solve must be banked to disk the instant it passes, survive an outer subprocess kill AND a
    budget-aware eval timeout. Partly built (§2b); the partial/phase-acceptance case is still
    open.
C2. **More compute != more solves if it blows the wall** (report §5 Layer-2). Repair-ON +
    cap 8->16 produced ZERO new solves, turned a verified jinja solve into a TIMEOUT, errored
    3 cells. Corroborated by the 80%-token-variance finding: compute only helps if banked.
C3. **Decomposition OVER-SPAWN is the cost pathology on easy repos** (voluptuous/jinja).
    `decompose` is GATED to medium/hard >=2-module repos (`templates.py:83-88`) — a phased
    planner MUST preserve this gate and size fan-out to complexity (external scaling
    heuristic).
C4. **FETCH-MONOCULTURE** (report §6). The architect once baked "restore official upstream"
    into the shared prefix -> every variant tripped the fetch-jail, scored 0 tokens. The
    structural worktree-shadow makes a fetch-cheat impossible-to-false-solve, but the prior
    still wastes attempts; the sanitizer is necessary-not-sufficient. mimesis is the canonical
    victim.
C5. **The matrix is statistically weak at n=1** (report §7). voluptuous always-solves,
    pydantic never, mimesis is a coin-flip (several non-solves are timeout-CLIPS, not honest
    fails), jinja is the only discriminator. **babel + mimesis are the intended HARD
    discriminators** (babel near-solve 4598/4607; mimesis 6044/6052). Any win claim needs
    n>=3-5 seeds with CIs (Tier-2).
C6. **Authored orchestration must EARN its cost** (report §4, AUTOGEN_WON=0). Autogen never
    beat the fixed template on any repo; at the one shared solve it cost 4x the agents. A
    phased planner adds architect+review agents — it MUST show a win on mimesis/babel to
    justify them.
C7. **Acceptance stays engine-owned** (Cardinal Contract, `INVARIANTS` `architect.py:122-139`).
    `ctx.select` may ABSTAIN; patterns can only downgrade/re-rank, never promote. No phase
    objective, plan-review, or TODO state may set `.accepted`.

---

## 5. Recommended design DIRECTIONS for a phased planner + adversarial goal review

Tailored to mimesis/babel (modular, near-solve repos where the residual-tail is the killer).
Each preserves C1-C7. Build on the existing seams (§1a) — do NOT tear down the converge shape.

### D1. Acceptance-checkpoint as an execution-grounded Stop-hook (FOUNDATION — do first)
Extend the EXISTING `_checkpoint_accepted` (`context.py:372`) to also bank **partial/phase**
acceptance (a strict gold-pass-COUNT improvement, not just whole-suite), wire the child
driver's `TimeoutExpired` path (`commit0_driver.py:259`) to consult the checkpoint, and pass
the budget-aware `evaluate_repo` timeout (Tier-1.2). This is `/goal`'s loop shape grounded in
real pytest (external novel-idea #1, provably stronger than transcript verification).
- **ROI: very high.** Directly recovers the run-4 mimesis 6052/6052 losses on BOTH arms;
  prerequisite for measuring any phased-planner win (a banked solve is the unit of credit).
- **Cost risk: low.** ~1-1.5d, mostly wiring; the mechanism is half-built and validated by
  `validate_checkpoint.py`. No new model capability, no extra agents.

### D2. Per-phase Planner subagent emitting a structured plan + per-phase acceptance ids
Replace the one-shot whole-repo author with a host-side phase loop: a read-only Planner
(`ctx.ask` with a schema, `context.py:870`) emits ordered phases `{objective, files_owned,
acceptance_gold_ids, depends_on, risks}` derived from `ctx.decompose`'s `order`+`gold_test_ids`
(currently discarded). Execute phases in dependency order; each phase's `acceptance_gold_ids`
is its acceptance predicate (score the SUBSET, not the whole suite); persist the plan + a
residuals ledger to disk (durable like `~/.claude/plans` + Pokemon NOTES.md). Use the SPFG+
governor `should_continue_waves` as the per-phase stop authority. Keep `workflow()` at depth 1
by looping host-side (avoids the C8/nesting cap).
- **ROI: high on mimesis/babel.** Per-phase partial acceptance banks the near-solve tail (the
  exact 4598/4607, 6044/6052 class) instead of abstaining on the whole suite; ordered phases
  beat the current parallel-ignore-order fan-out on dependency-coupled modules.
- **Cost risk: medium.** Adds 1 planner agent/phase. MUST keep the easy-repo skip-gate (C3)
  and size phases to the external scaling heuristic (1 / 2-4 / more). Risk = over-planning a
  shallow repo; mitigate by reusing the existing difficulty gate to bound phase count.

### D3. Grounded adversarial GOAL-ALIGNMENT review gating phase progression
Before codegen for phase N (and before advancing past it), run N skeptics via `ctx.ask` /
`ctx.signals` asking "does this phase plan still serve goal G, given residual failing ids
R? verdict proceed/revise/abort + reason", admit-gated through the existing
`adversarial_filter` (`context.py:664`). CRITICAL: ground the "revise/abort" reason in REAL
failing-test output (the residual node-ids from `reduce_residuals`), not the transcript —
this closes `/goal`'s "confident summary of broken work" blind spot while reusing its loop
shape. The gate is a read-only SIGNAL: it can re-plan/re-target but can NEVER set acceptance
(C7).
- **ROI: high for no-veer on big comprehensive tasks** (the user's explicit goal). Prevents
  the fetch-monoculture drift (C4) and the duplicate-work failure on multi-module repos.
- **Cost risk: medium.** N review agents per phase boundary. Gate the review to medium/hard
  only (mirror the verify-phase gate `templates.py:153`) so easy repos pay nothing. Risk =
  review agents become pure cost (C6) — measure the review against an ablation with it OFF.

### D4. Detailed delegation contracts with explicit file-ownership boundaries per fan-out module
Augment `fanout_modules`/`solve_module` (`context.py:1247,1202`) so each module agent's prompt
carries the external delegation-contract shape: objective + output format + the EXACT
files/symbols it OWNS (from the `files` field already in `DECOMPOSE_SCHEMA`,
`context.py:108-111`) + boundaries it must not cross. Pairs with sub-agent context isolation:
each solver returns a 1-2k-token distilled summary (passed / residual gold-ids / files
touched) to the reducer, keeping `reduce_residuals` cheap on hard repos.
- **ROI: high on babel/mimesis specifically** — kills the duplicate-work/gap failure
  (corroborated as THE multi-agent failure mode) and the merge-collapse the converge default
  already fights (`templates.py:109-126`); cheaper reduces on-budget (C2).
- **Cost risk: low.** Prompt-shape change + a summary return; no extra agents. Depends on
  `decompose` populating `files` (advisory today) — fail-open if absent.

### D5. Complexity-matched fan-out + budget-aware effort lever (efficiency guard)
Size best-of-N / subagent count to module count x difficulty (external scaling heuristic:
1 trivial / 2-4 medium / more hard) and expose a per-phase effort/token knob (Opus-4.5
medium-effort ~76% output-token savings) so heavy effort is spent ONLY on the hard residual
modules AFTER the easy ones are banked. This is the structural answer to C2/C6 — bank cheap
wins first, escalate effort on the residual tail.
- **ROI: medium-high.** Directly attacks the "compute-without-banking blows the wall" failure
  (run-4) and the AUTOGEN_WON=0 cost problem; lets the planner spend its budget where the
  near-solve tail actually is.
- **Cost risk: low.** Reuses `plan_waves` + `difficulty_profile` (`architect.py:373-380`) and
  the SPFG+ governor; mostly a scheduling/knob change. Risk = under-spending a genuinely hard
  module; the rising-frontier governor (`run_ladder.py:393`) already protects a progressing
  arm from premature cut.

**Sequencing:** D1 (foundation, recover lost solves) -> D2 + D4 (phased plan + ownership
contracts, the core capability) -> D3 (adversarial no-veer gate) -> D5 (efficiency). Gate the
whole stack behind an ablation flag, A/B against the frozen converge template on mimesis +
babel with n>=3-5 seeds (C5), and accept it ONLY if it banks a solve the template misses (C6).
