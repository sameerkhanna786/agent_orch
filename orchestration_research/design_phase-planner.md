# Design: Faithful Claude-Code-style Phase Planner (`phase-planner`)

Author: principal-orchestration-designer subagent
Target engine: APEX-Ω (`apex_omega/`)
Grounding: `orchestration_research/CONTEXT.md`, `apex_omega/autogen/{architect,templates,context}.py`,
`apex_omega/eval/{commit0_autogen,commit0_driver}.py`, `scripts/run_ladder.py`, `APEX_COMMIT0_REPORT.md`.

---

## 0. One-liner

Replace the single-shot whole-repo author with a **host-side phase loop**: a read-only Planner
agent emits an ordered list of phases (objective + per-phase acceptance gold-id subset + scope +
deps); each phase runs a tailored sub-workflow (authored snippet OR `ctx.workflow`), banks an
execution-verified **phase checkpoint** the instant its gold subset goes green, and passes through
an adversarial goal-alignment gate (grounded in real residual failing node-ids) that can replan but
never accept. The running best diff carries forward across phases. The whole-suite accept and the
abstain-or-floor degrade stay exactly where they are today — engine-owned.

---

## 1. Why phases, and what stays unchanged

### What it fixes (mapped to CONTEXT.md gaps §1b)

- Gap 1/6 (no ordered objectives, hard-coded shape) → an explicit ordered phase list per repo.
- Gap 2 (one-shot authoring) → orchestration is generated/selected **per phase, after observing
  phase N-1's residual** — the architect finally sees execution feedback.
- Gap 3 (no goal-alignment review) → an adversarial gate at each phase boundary, grounded in real
  pytest residuals, that gates progression / triggers replan.
- Gap 4 (decompose order discarded) → phases ARE the topological order; `decompose`'s `order` +
  `gold_test_ids` (today discarded by the parallel-everything fan-out) become the phase spine.
- Gap 5 (no partial/phase acceptance) → a **phase checkpoint** banks a verified gold-subset pass,
  durable across an outer kill (the run-4 data-loss class for partial progress).

### What stays UNCHANGED (the moat — do not touch)

- Acceptance is engine-owned. The ONLY producer of a winner is `ctx.select` returning an
  `.accepted` Candidate (`context.py:1467`). No phase objective, plan, gate, or checkpoint may set
  `.accepted` (Cardinal Contract, INVARIANTS `architect.py:122-139`). [C7]
- The verified best-of-N floor (`architect.py:570 _floor`, `BEST_OF_N_ORCHESTRATION`) is the
  degrade target. A phase planner that fails to plan → fall through to today's converge/best-of-N.
- The SPFG+ governor (`context.py:430-587`, `engine/governor.py`) remains the per-wave/per-phase
  stop authority. We add NO new stop logic.
- The easy-repo decomposition skip-gate (`templates.py:83-88`) is preserved verbatim — easy/
  single-module repos never enter the phase loop (cost guard, C3).

---

## 2. Architecture: the host-side phase loop

The phase planner is a **new authored orchestration string** (`PHASE_PLANNER_ORCHESTRATION` in
`templates.py`) selected by `APEX_OMEGA_ORCHESTRATION=phase-planner`, frozen exactly like
`converge`/`ralph` (`architect.py:311-322`). It is plain `orchestrate(ctx)` code using only the
`ctx` API plus a small set of NEW `ctx` methods (§4). It loops **host-side at depth 0** — no deep
`workflow()` nesting (respects the depth-1 cap, gap 8). Pseudocode:

```python
def orchestrate(ctx):
    ctx.phase("scope")
    difficulty = str(ctx.repo_map.get("difficulty") or "").lower()

    # COST GUARD #1 (C3): easy / single-module repos NEVER plan phases.
    plan = ctx.decompose() if difficulty != "easy" else None
    modules = (plan or {}).get("modules") or []
    if not plan or len(modules) <= 1 or difficulty == "easy":
        return ctx.workflow("default-best-of-n")     # cheap path, unchanged

    # (P0) PLAN: a read-only Planner emits the ordered phase list (objective + acceptance ids
    # + scope + deps), derived from decompose's order + gold_test_ids (NEW ctx.plan_phases).
    phases = ctx.plan_phases(plan)                    # durable: persisted to disk (§3)
    if not phases:                                    # planner abstained -> degrade to converge
        return ctx.workflow("converge")

    carry = ctx.carry_best()
    for ph in phases:                                 # phases run in dependency order
        if not ctx.should_continue_waves():
            break                                     # governor cut / ceiling / budget

        # (P1) GOAL-ALIGNMENT GATE (pre): does this phase still serve the goal given the
        # residual? proceed / revise / skip. Read-only signal; can re-target, never accept.
        gate = ctx.goal_gate(ph, carry_diff=carry, when="pre")
        if gate["verdict"] == "skip":
            ctx.defer("phase_skipped", ph["id"], gate["reason"]); continue
        if gate["verdict"] == "revise":
            ph = ctx.replan_phase(ph, gate, carry_diff=carry) or ph

        # (P2) EXECUTE the phase sub-workflow (per-phase generated/selected; §5).
        carry = ctx.run_phase(ph, carry_diff=carry)   # banks phase checkpoint on subset-green

        # (P3) VERIFY the phase OBJECTIVE by EXECUTION (subset score), bank phase checkpoint.
        pr = ctx.phase_accept(ph, carry_diff=carry)   # NEW: scores the gold SUBSET, checkpoints
        if pr["whole_suite_accepted"]:                # a phase incidentally solved everything
            return pr["candidate"]                    # engine-owned accept; done
        carry = pr["carry_diff"]                      # monotone best partial forward

        # (P4) GOAL-ALIGNMENT GATE (post): did we veer? if a phase made the global residual
        # WORSE or drifted off-scope, abort the plan and fall to convergence on the residual.
        gate2 = ctx.goal_gate(ph, carry_diff=carry, when="post")
        if gate2["verdict"] == "abort":
            ctx.defer("plan_aborted", ph["id"], gate2["reason"]); break

    # (P5) CONVERGE the residual tail on the live merged carry, then SELECT (engine-owned).
    # This is the existing loop-until-dry on residual ids, seeded by the phase carry.
    return ctx.converge_tail(carry_diff=carry, modules=modules)
```

`converge_tail` is a thin wrapper around the EXISTING reduce → loop-until-dry → verify body of
`DEFAULT_ORCHESTRATION` (`templates.py:128-157`) seeded with the phase carry, so the planner always
ends in the proven residual-closing path and `ctx.select` (never a fake accept).

---

## 3. The durable phase plan + residuals ledger (Pokemon NOTES.md / Plan Mode)

External corroborated: LeadResearcher **saves its plan to memory** before work (multi-agent-
research-system); Plan Mode persists plans to `~/.claude/plans/` surviving restarts (DataCamp);
Claude-Plays-Pokemon reads its own NOTES.md after a context reset (effective-context-engineering).

Mechanism: `ctx.plan_phases` writes the plan once, atomically, to
`<run_dir>/phase_plan.json` and appends per-phase outcomes to `<run_dir>/phase_ledger.jsonl`. On
resume the plan is re-read (NOT re-planned) — identical to how `load_frozen` (`architect.py:286`)
re-uses the frozen script. Every plan/gate/replan call is also a journaled `ctx.ask` (read-only,
replay-deterministic), so the host phase loop replays identically.

`phase_plan.json` schema:

```json
{ "phases": [
    {"id": "p1", "objective": "implement <module> data models",
     "acceptance_gold_ids": ["tests/test_x.py::test_a", "..."],
     "scope_files": ["pkg/models.py"], "depends_on": [],
     "risks": "shared symbol Y needed by p2"} ],
  "order": ["p1","p2","p3"], "source_decomposition_sha": "<sha>" }
```

`phase_ledger.jsonl` (one line per phase event): `{phase_id, event, gold_passed_subset,
gold_total_subset, whole_suite_pass_rate, gate_verdict, gate_reason, agents_used, ts}`.
This is the residuals ledger the post-gate reads to detect veering (a phase that did not raise the
global frontier, or lowered it).

---

## 4. ctx API mapping (existing + NEW)

### Reused as-is

| Need | Existing ctx |
|---|---|
| Module breakdown + order + gold ids | `ctx.decompose()` (`context.py:1155`) — already returns `order` + per-module `gold_test_ids` |
| Read-only schema'd planner/gate signal | `ctx.ask(prompt, schema=…)` (`context.py:870`) — journaled, nudged, null-terminal |
| Read-only fan-out of gate skeptics | `ctx.signals(thunks)` (no plateau accounting, `context.py:604`) |
| Admit-gate gate verdicts (drop false alarms) | `ctx.adversarial_filter(items, votes=3)` (`context.py:664`) |
| Per-module solve, carry-seeded | `ctx.solve_module` / `ctx.fanout_modules` (`context.py:1202/1247`) |
| Merge + full-suite score (0 tokens) | `ctx.reduce_residuals` (`context.py:1274`) |
| Residual repair on live tree | `ctx.repair_residual` (`context.py:1368`) |
| Running best partial diff | `ctx.carry_best()` (`context.py:1097`) |
| Per-phase stop authority | `ctx.should_continue_waves()` (`context.py:579`) |
| Whole-suite accept (ONLY winner producer) | `ctx.select` (`context.py:1467`) |
| Compose a named sub-workflow | `ctx.workflow(name, args)` (`context.py:639`, depth-1) |
| IOU / deferral between phases | `ctx.defer` / `ctx.blocked` (`context.py:712`) |
| Instant accept checkpoint | `ctx._checkpoint_accepted` (`context.py:372`) |

### NEW ctx methods (thin host-side wrappers; each degrades to a no-op/existing path)

All are added to `OrchestrationContext`. None can set `.accepted`. All journaled via `ctx.ask` /
the existing `_attempt`/`reduce_residuals` seams, so replay-deterministic.

**1. `plan_phases(self, decomposition: dict, *, vendor=None, model=None, agent_id=700200) -> list[dict] | None`**
- ONE read-only `ctx.ask` (sandbox=read-only, FIXED agent_id disjoint from decompose 700100 /
  module 730000 / repair 71000x / pattern 9xxxxx namespaces) with `PHASE_PLAN_SCHEMA`. Prompt:
  "Given this module decomposition (order + gold ids), produce an ORDERED list of implementation
  PHASES. Each phase = a coherent objective + the EXACT gold node-ids it must turn green
  (acceptance_gold_ids ⊆ the union of module gold ids) + scope_files it owns + depends_on. Respect
  the topological order; group tightly-coupled modules into one phase; do NOT exceed N phases."
- **COST GUARD #2 (C3, the run-4 over-spawn fix):** the max phase count `N` is bound by difficulty
  via `difficulty_profile` (`architect.py:378`): easy=skip (never reached), medium≤3, hard≤5. This
  is the structural analogue of the external scaling heuristic (1 / 2-4 / more). Prompt states the
  cap; post-validation truncates to `N` (extra phases folded into `converge_tail`).
- Validates `acceptance_gold_ids` against the real gold-id inventory on `repo_map` (drop ids not in
  the suite — anti-hallucination). Persists `phase_plan.json` (§3). FAIL-OPEN: schema-miss / null /
  zero valid phases → return `None` (caller degrades to `ctx.workflow("converge")`).

**2. `goal_gate(self, phase: dict, *, carry_diff: str, when: str, votes=3) -> dict`**
- The adversarial GOAL-ALIGNMENT review (§6). Runs `votes` read-only skeptics via `ctx.signals`,
  each asked: GOAL G (the binding `task_framing` + the phase objective) + the REAL residual failing
  node-ids (from a cheap `reduce_residuals([], carry_diff=carry)` full-suite re-score, or the
  ledger's last residual) → returns `{verdict ∈ {proceed,revise,skip,abort}, reason, evidence_ids}`
  per `GATE_SCHEMA`. Aggregated by `adversarial_filter`-style majority (default proceed on a tie —
  the gate is conservative: it never blocks unless skeptics agree on a veer). Returns the merged
  `{verdict, reason}`. **Grounded-not-transcript:** every `revise/skip/abort` MUST cite
  `evidence_ids` that are real failing node-ids; a verdict with no grounded evidence is downgraded
  to `proceed` (this is the closure of `/goal`'s "confident summary of broken work" blind spot,
  CONTEXT.md §3). Read-only SIGNAL only — cannot accept (C7).
- **COST GUARD #3:** the gate runs ONLY for medium/hard (mirrors `templates.py:153`); `votes`
  scales 1 (medium) / 3 (hard). Easy never reaches it.

**3. `replan_phase(self, phase, gate, *, carry_diff) -> dict | None`**
- ONE read-only `ctx.ask` that revises the phase (re-scope files / re-target acceptance ids /
  reorder) given the gate's grounded reason. Returns a revised phase dict or `None` (keep original).
  Persists the revision to the ledger. Read-only.

**4. `run_phase(self, phase, *, carry_diff: str) -> str`**
- Executes the phase's sub-workflow and returns the new carry diff. Selection (§5):
  - If `phase.get("workflow_ref")` is set (per-phase authored snippet, §5b) → run via a sandboxed
    `run_orchestration` in a CHILD context seeded with `args={"phase": phase, "carry_diff": carry}`
    (depth-1 `ctx.workflow` with a by-ref source).
  - Else (default) → fan out the phase's modules with `fanout_modules([phase-as-module],
    carry_diff=carry)` then `reduce_residuals(..., carry_diff=carry)`, scoping each agent to
    `acceptance_gold_ids` (delegation contract, §7). Returns `reduce_residuals["merged_diff"]` or
    the prior carry on collapse (never erases carry).

**5. `phase_accept(self, phase, *, carry_diff: str) -> dict`**  — THE PHASE CHECKPOINT (D1 extension)
- Scores the phase by EXECUTION against its `acceptance_gold_ids` SUBSET (a new `score_fn` mode, §8)
  AND the full suite once (reuse `reduce_residuals` for the full-suite number). Returns
  `{subset_passed, subset_total, whole_suite_pass_rate, whole_suite_accepted, candidate,
  carry_diff, phase_satisfied}`.
- **Banks a phase checkpoint** the instant the subset is fully green OR the global gold frontier
  strictly rose: calls `_checkpoint_phase` (NEW, §8) writing `<run_dir>/phase_checkpoint.json`
  (best partial diff + subset/global counts). Mirrors `_checkpoint_accepted` atomicity (temp-write
  + `replace`). The WHOLE-suite accept path is unchanged: if the full suite is green here,
  `reduce_residuals` already calls `_checkpoint_accepted` (`context.py:1344`) and `phase_satisfied`
  routes to `ctx.select`.

**6. `converge_tail(self, *, carry_diff, modules) -> Candidate | None`**
- The existing reduce → loop-until-dry → verify body (`templates.py:128-157`) factored into a ctx
  method, seeded with the phase carry. Ends in `ctx.select(ctx.all_candidates())` (engine-owned
  accept; may abstain).

### NEW engine/harness hooks

- `context.py:_checkpoint_phase(cand, *, subset_passed, subset_total)` — sibling of
  `_checkpoint_accepted`; writes `phase_checkpoint.json` (NOT `accepted_checkpoint.json`, which
  must remain whole-suite-only so the harness never reports a partial as a solve). Idempotent on
  best-so-far (overwrite only when subset/global count strictly rises).
- `scripts/run_ladder.py:_recover_checkpoint` — extend to ALSO recover the warm partial carry on a
  kill so a relaunch resumes from the banked phase diff (today it only recovers a whole-suite
  accept → `solved:1`). The partial recovery feeds the warm `--run-dir` resume (it does NOT emit
  `solved:1` — a partial is never a solve). This closes the partial-progress half of C1.
- `commit0_driver.py:_run_mode_a` TimeoutExpired path (line 259) — already returns an infra
  non-result; Mode-C runs in-process and is covered by `run_ladder`'s recovery. The budget-aware
  eval timeout (Tier-1.2) is ALREADY wired (`commit0_autogen.py:392 eval_cap`), so no change
  needed there — verified during this design.

---

## 5. Per-phase orchestration: generate-vs-reuse (the user's "generate code per phase")

Two modes, picked by a cheap classifier so we do not pay an architect agent per phase on simple
phases (C6 cost discipline):

### 5a. Reuse (default, cheapest) — `ctx.workflow` / built-in module fan-out
Most phases are "implement module(s) M to pass ids I." `run_phase` handles these with the existing
`fanout_modules` + `reduce_residuals`, scoped to the phase's ids. Zero architect agents. This is
the path easy-after-the-gate phases take.

### 5b. Generate (gated to hard/flagged phases) — per-phase authored snippet
For a phase flagged `needs_custom_orchestration` by the planner (e.g. "this phase needs a
cross-module synthesis" / "route the hard parser to a stronger vendor"), call a NEW
`ctx.author_phase(phase, carry_diff) -> str|None`: ONE read-only architect `ctx.ask` returning an
`orchestrate(ctx)` snippet, lint-checked via the EXISTING `lint_source` (`sandbox.py`), frozen to
`<run_dir>/orchestrator/phase_<id>.py`, run via `ctx.workflow({"scriptPath": …}, args=phase)`.
Lint-fail → fall back to 5a. This is the literal "generate orchestration code per phase," reusing
the entire existing author/lint/freeze machinery (`architect.py:339-347`), now driven PER PHASE
with execution feedback (the carry + residual) — the thing one-shot authoring could never do
(gap 2). **COST GUARD #4:** `author_phase` is gated to hard repos AND a planner flag, and bounded
to ≤1 per phase; on a budget pinch it is skipped (5a). This prevents the run-4 architect-cost
blowup (C6) — most phases pay zero authoring cost.

---

## 6. Adversarial goal-alignment review (no-veer) — the design

The user's explicit goal: "use ADVERSARIAL REVIEW so the run never veers from the goal." Four
stacked no-veer mechanisms from CONTEXT.md §3, all grounded in EXECUTION (the moat over `/goal`):

1. **Detailed delegation contracts** (§7) per phase/module — objective + scope_files + acceptance
   ids + explicit boundaries. Corroborated cure for duplicate-work/gaps.
2. **The goal_gate skeptics** (`ctx.goal_gate`, §4) at each phase boundary — N read-only reviewers
   judge "does this phase still serve goal G given residual R?" Their verdict is admit-gated
   (`adversarial_filter`) and MUST cite real failing node-ids (`evidence_ids`); an ungrounded
   verdict is downgraded to `proceed`. This is the execution-grounded version of `/goal`'s
   Stop-hook loop shape (CONTEXT.md §3 novel-idea #1) — provably stronger than transcript review.
3. **System-reminder re-injection of the goal at the decision point** (arXiv:2603.05344 confirmed
   shape) — the binding `task_framing` (already on `repo_map`, threaded to workers via
   `build_issue_description`) is re-stated in EVERY phase/module/gate prompt, fighting instruction
   fade-out. Mechanism: `run_phase`/`goal_gate` prepend the framing block (the same one
   `build_author_prompt` uses, `architect.py:240`).
4. **Doom-loop / no-progress detection** — already provided by SPFG+ (`should_continue_waves`):
   a phase that produces no frontier rise across its waves is cut, the gate books the veer reason
   to the ledger, and `converge_tail` takes over. No new stop logic (C-respecting).

The gate is a pure SIGNAL: it can `revise`/`skip`/`abort` the PLAN and re-target compute, but it
can NEVER set `.accepted` (C7). The plan-abort path falls to `converge_tail` → `ctx.select`, so a
veering plan degrades to the proven residual-closing path, never to a fake pass.

---

## 7. Delegation contracts with explicit boundaries (D4)

`run_phase`'s module agents and `author_phase`'s snippet inherit the external delegation-contract
shape (CONTEXT.md §3, the corroborated cure for the multi-agent duplicate-work/gap failure):
- **objective**: the phase objective string.
- **output format**: a diff scoped to `scope_files`.
- **boundaries**: "edit ONLY these files: {scope_files}; do NOT edit tests; do NOT reimplement
  other phases' modules; note a genuinely-missing shared symbol via the residual instead of forking
  it." (The `solve_module` brief, `context.py:1228-1239`, already has this shape — `run_phase`
  passes `scope_files` as the explicit file-ownership set.)
- **acceptance**: the `acceptance_gold_ids` the agent must turn green.
- Sub-agent context isolation: each phase agent returns a distilled 1-2k-token summary
  (passed/residual ids + files touched) — already the shape `fanout_modules` collects (it forwards
  candidate-ids and re-collects diffs, `context.py:1265-1272`), keeping `reduce_residuals` cheap on
  hard repos (C2).

---

## 8. Acceptance-checkpointing (C1) — the foundation, extended for phases

The whole-suite path is already solid (verified during this design):
`_checkpoint_accepted` (`context.py:372`) atomic-writes `accepted_checkpoint.json` the instant a
candidate accepts; `run_ladder._recover_checkpoint` (`run_ladder.py:267`) recovers it on a kill and
on clean completion (review-fix #8); the budget-aware `eval_cap` is wired
(`commit0_autogen.py:392`). **Do not rebuild — extend for the PARTIAL/phase case** (the open gap,
CONTEXT.md §2c.1):

1. **Phase/partial checkpoint** — `_checkpoint_phase` writes `phase_checkpoint.json` (best partial
   diff + subset_passed/total + whole_suite gold count). Banked the instant a phase subset goes
   green OR the global gold frontier strictly rises. Atomic temp-write + `replace`, overwrite only
   on a strict count rise. Kept SEPARATE from `accepted_checkpoint.json` so a partial can NEVER be
   reported as a solve (C7).
2. **Subset scoring** — a NEW `score_subset(worktree, gold_ids)` mode on the score_fn (the eval
   harness's `score_fn` closure in `commit0_autogen.py:386-433` gains an optional
   `subset_ids=` param that runs pytest over the id subset only). Cheaper than the full suite →
   phase acceptance is cheap. A subset-scoring timeout maps to `indeterminate` (safe), exactly like
   the full score.
3. **Warm partial recovery** — `run_ladder._recover_checkpoint` also returns the partial carry so a
   relaunch resumes from the banked phase diff (warm `--run-dir`). Emits `relaunch` (not `solved`).
4. **Stays execution-authoritative** — every checkpoint records a candidate REAL pytest produced;
   the orchestrator never self-declares (the moat over `/goal`, CONTEXT.md §2c.5).

A verified WHOLE-suite solve inside any phase therefore survives an outer kill (whole-suite
checkpoint) AND a partial near-solve survives it (phase checkpoint → warm resume continues from the
banked diff). This is the direct recovery of the run-4 mimesis-class loss for BOTH full and partial
progress.

---

## 9. Cost guards summary (the run-4 regression must NOT recur) [C2, C3, C6]

run-4 lesson: repair-ON + agent-cap 8→16 produced ZERO new solves, turned a verified jinja solve
into a TIMEOUT, errored 3 cells (CONTEXT.md C2). The phase planner adds Planner + gate + (gated)
per-phase architect agents — these MUST earn their cost. Guards:

| Guard | Mechanism | Defends |
|---|---|---|
| #1 Easy/single-module skip | `difficulty=="easy"` or ≤1 module → `default-best-of-n`, never the phase loop (`templates.py:83-88` preserved) | C3 over-spawn on voluptuous/jinja |
| #2 Phase-count cap | `plan_phases` N bound by `difficulty_profile`: medium≤3, hard≤5 (external 1/2-4/more heuristic) | C3 over-planning |
| #3 Gate gated to medium/hard, votes 1/3 | `goal_gate` only for medium/hard; mirrors verify-gate `templates.py:153` | C6 review-agent cost |
| #4 Per-phase author gated + ≤1/phase + budget-skip | `author_phase` only on hard + planner flag; falls to 5a on budget pinch | C6 architect-cost blowup |
| #5 SPFG+ owns all stops | `should_continue_waves` per phase; no new stop logic; per-agent wall decoupled from cell wall (`context.py:235`) | C2 budget blowup |
| #6 Bank cheap wins first, escalate effort on tail | phases run in order, easy modules banked (phase checkpoint) before hard residual gets effort | C2 compute-without-banking |

Net additional agents on a HARD repo: 1 planner + ≤5 phases × (≤3 gate-skeptics + ≤1 author) +
the same module/repair agents the converge default already spends. The gate/author agents are
read-only and cheap; the gate count is bounded and ablatable. On medium: 1 planner + ≤3 phases ×
1 gate-skeptic. On easy: ZERO added agents (skip-gate). The A/B (§10) measures whether this earns
its cost (C6: it is accepted ONLY if it banks a solve the converge template misses).

---

## 10. A/B eval plan (mimesis + babel + controls)

### Arms (all under `scripts/run_ladder.py`, `--cell-timeout 86400`)

- **Arm CONVERGE (control):** `APEX_OMEGA_ORCHESTRATION=converge` — the current frozen
  `DEFAULT_ORCHESTRATION`. The bar to beat (C6).
- **Arm PHASE (treatment):** `APEX_OMEGA_ORCHESTRATION=phase-planner` — the full stack
  (plan_phases + goal_gate + phase_accept + converge_tail; per-phase author ON for hard).
- **Arm PHASE-NOGATE (ablation):** phase planner with `goal_gate` OFF (env
  `APEX_PHASE_GOAL_GATE=0`) — isolates the adversarial-review contribution (C6 / CONTEXT.md §5 D3:
  "measure the review against an ablation with it OFF").
- **Arm PHASE-NOAUTHOR (ablation):** per-phase `author_phase` OFF (only 5a reuse) — isolates the
  per-phase codegen contribution.
- **Arm RALPH (baseline):** `APEX_OMEGA_ORCHESTRATION=ralph` — the vanilla persistence control.

### Repos

- **HARD discriminators (the verdict):** `mimesis` (6044/6052 near-solve, the run-4 loss),
  `babel` (4598/4607 near-solve). These are modular, near-solve repos where the residual tail is
  the killer — exactly the phase-checkpoint + ordered-phase sweet spot (CONTEXT.md C5).
- **Controls (no-regression):** `voluptuous` (always-solves), `jinja` (the one historical
  discriminator). PHASE must NOT regress these — they hit the skip-gate (easy/single-module) and
  run the IDENTICAL cheap path as CONVERGE, so a regression here is a bug, not a strategy outcome.

### Protocol

- **n ≥ 3-5 seeds per (arm × repo)** with CIs (CONTEXT.md C5: the matrix is statistically weak at
  n=1; mimesis is a coin-flip). Report solve-rate ± Wilson CI per cell.
- **Exclude infra non-results from denominators** (Tier-1.3): any cell with `wall_s ≥ CELL_TIMEOUT`
  or null/partial result → `status:timeout`/`indeterminate`, excluded with a `nonresult_reason`
  (already implemented in `run_ladder` relaunch_decision / outcome taxonomy).
- **Checkpoint-loss audit:** for every cell, assert that if any per-eval `report.json` shows a full
  pass, the cell reports `solved:1` (directly tests the run-4 data-loss class is closed). Reuse /
  extend `scripts/validate_checkpoint.py`.

### Acceptance criteria (when is PHASE adopted?)

- **PRIMARY:** PHASE banks ≥1 verified solve on mimesis OR babel that CONVERGE misses, across the
  seed set, with no control regression (voluptuous/jinja solve-rate unchanged within CI). [C6]
- **SECONDARY (no-veer evidence):** PHASE-NOGATE shows MORE off-scope drift (more `defer`
  merge-conflict / a lower gold frontier on a multi-module repo) than PHASE — i.e. the gate
  measurably reduces veering. If PHASE ≈ PHASE-NOGATE, the gate is pure cost → ship PHASE-NOGATE.
- **COST:** PHASE's agents_used at the shared solve is within ~1.5× CONVERGE (the gate/planner
  overhead is justified by a solve CONVERGE misses; if it costs ≥4× like run-4 autogen with no new
  solve → reject, per C6/C2).
- **CHECKPOINT:** zero checkpoint-loss audit failures on all arms (C1 closed).

### Sequencing (CONTEXT.md §5 ordering)

D1 (checkpoint extension §8 — partial/phase banking + warm partial recovery) FIRST — it is the unit
of credit and is mostly wiring on a half-built mechanism. Then D2+D4 (plan_phases + delegation
contracts §4/§7 — the core capability). Then D3 (goal_gate §6 — the no-veer gate). Then D5 (the
per-phase author + complexity-matched effort §5b/§9). Gate the whole stack behind
`APEX_OMEGA_ORCHESTRATION=phase-planner` + the per-feature env flags so each layer is independently
ablatable, and accept ONLY on the §10 criteria.

---

## 11. Risks and mitigations

- **Planner hallucinates acceptance ids** → validated against the real gold inventory; invalid ids
  dropped; zero valid phases → degrade to `converge`.
- **Gate becomes pure cost** (C6) → PHASE-NOGATE ablation is a first-class arm; ship the cheaper arm
  if the gate shows no no-veer benefit.
- **Phase-count over-plan** (C3 run-4 class) → hard cap by `difficulty_profile`; easy never enters.
- **Per-phase authoring blows the budget** (C6) → gated to hard + flag, ≤1/phase, budget-skip to 5a.
- **Partial checkpoint mistaken for a solve** (C7) → `phase_checkpoint.json` is a SEPARATE file from
  `accepted_checkpoint.json`; only the latter ever yields `solved:1`; partial only warm-resumes.
- **Replan thrash / instruction fade** → SPFG+ `should_continue_waves` caps the phase loop; the
  gate is conservative (default proceed on tie, ungrounded verdict downgraded to proceed).
- **Nesting cap** (gap 8) → the phase loop runs host-side at depth 0; only per-phase snippets use
  one `ctx.workflow` level (depth-1, legal).
