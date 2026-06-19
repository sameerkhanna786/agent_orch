# DECISION: the orchestration design to implement

**Deciding architect** — grounded in re-reading the real code (`architect.py`, `templates.py`,
`context.py`, `commit0_driver.py`, `commit0_autogen.py`, `run_ladder.py`), `CONTEXT.md`, all four
candidate design docs, and the three adversarial judge rankings.

---

## 0. Verdict

**WINNER: `hybrid-converge-plus`** — a thin Claude-Code-style PHASE PLANNER layered ON TOP of the
proven converge engine, **grafting two ideas from the runners-up**:

1. From `novel-best`: the `scope_ids=` optional param on `reduce_residuals` (back-compatible, zero
   extra pytest run) so a phase pass is a pure SET TEST over the full-suite `residual_failing_ids`
   the engine **already computes** — NOT a second subset pytest run. This is strictly cheaper than
   `phase-planner`/`agentless`'s `score_subset` mode and avoids new harness cost (C2/C6).
2. From `phase-planner`/`agentless`: the **explicit delegation-contract scope** (objective +
   `files_owned` boundaries) threaded into the per-phase module agents, reusing the
   `solve_module` brief that already exists (`commit0_autogen.py:345-376`).

The full per-phase Tier-B codegen (`novel-best` §2, `phase-planner` §5b `author_phase`) is
**explicitly deferred behind a flag, default OFF**, to be enabled only if the A/B shows the cheaper
arms plateau. This realizes the user's "generate orchestration code per phase" goal as a *seam that
exists and is one flag away*, while refusing to pay run-4's authoring-cost blowup speculatively.

This decision **overrides the judges' headline picks** (solve-lift judge → hybrid at 58;
cost-safety judge → agentless at 86; the third lens not shown). It agrees with the solve-lift
judge that hybrid is the safest carrier of the already-banked lift, and it **rejects agentless's
"eliminate autonomy via sampling" thesis** as the least-grounded transfer to commit0 (the
cost-safety judge's own R5, and the solve-lift judge scored it last at 41).

---

## 1. Why hybrid-converge-plus wins (rationale vs the judges and the other three)

### 1.1 The single most important verified fact: the #1 lift is ALREADY banked for THIS arm

I confirmed in code that the autogen arm (the target of this research) runs **Mode C =
`_invoke_autogen` IN-PROCESS** (`commit0_driver.py:271-293`), NOT via the `_run_mode_a` subprocess
whose `TimeoutExpired` branch is at `:259`. The run-4 mimesis `6052/6052` loss was an **outer-kill**
loss, and `run_ladder._recover_checkpoint` (`run_ladder.py:267`) **already** rglobs
`accepted_checkpoint.json` and recovers it on the outer kill (`:470`, `:501`). `_checkpoint_accepted`
(`context.py:372`) already banks atomically (temp-write + `replace`, `:388-390`) the instant a
candidate accepts, called from `_attempt` (`:842`) and `reduce_residuals` (`:1344`). The
budget-aware `eval_cap` (Tier-1.2) is already wired (`commit0_autogen.py:392`, passed at `:431`).

**Consequence:** the highest-confidence solve-recovery (a verified WHOLE-suite pass surviving the
wall) is already closed for Mode C. This deflates the headline lift claims of *every* design — and it
means the right move is the SMALLEST delta on the proven converge body, because the marginal
remaining lift is uncertain (the off-by-K tail) while the marginal regression risk from more moving
parts is real and proven (run-4: C2). That is exactly hybrid's stance. The solve-lift judge reached
the same conclusion (58, #1): "most plausible to NOT lose existing lift."

The genuinely-OPEN C1 gap is only the **partial/phase** checkpoint (`phase_checkpoint.json` does not
exist) and the **child-driver Mode-A seam** (a hardening assertion, not load-bearing for Mode C).
All four designs target the partial case; hybrid does it with the least new surface.

### 1.2 Why NOT the others

- **agentless-structured (cost-safety judge's pick, 86; solve-lift judge's last, 41).** Its
  cost-safety crown is real BUT its lift thesis is the weakest-grounded: Agentless's
  ~10x cost win is SWE-bench-Lite *single-file bug-fixes*; commit0 is whole-repo completion (its own
  R5). Its lift mechanism — sample `k_P` patches per module behind an execution filter — is
  **best-of-N-per-module, which APEX-Ω already does** via `fanout_modules` + the floor-probe; the
  report shows best-of-N already banks the floor and yields zero new solves when it blows the wall
  (C2). Adopting agentless trades a *proven* converge body (which closes the near-solve tail via
  carry-forward — `templates.py:59-61`, `DEFAULT_ORCHESTRATION` docstring) for an unproven sampling
  spine. It also over-claims the driver-`TimeoutExpired` seam as load-bearing (it is not, for Mode C).
  **Reject as the base; KEEP its `scope_ids` idea and its discipline ethos.**

- **novel-best (PSP-Ω, solve-lift judge 52).** The tightest single design and the source of two
  grafts (`scope_ids`; `_checkpoint_frontier` from `_observe`). But it adds the **most new control
  logic**: a budget PARTITION across phases (`w_i` weighting) layered on top of the SPFG+ governor.
  The cost-safety judge and the solve-lift judge both flagged this as a *new* over-spend surface that
  can starve a hard early phase (its own R4) — i.e. it risks re-introducing the very blowup it claims
  to prevent, on top of a governor that already owns the stop decision. Its Tier-B per-phase codegen
  is ON in the headline arm — the single riskiest lever, and run-4 was itself a codegen regression.
  **Reject as the base (too much new stop/budget logic); GRAFT `scope_ids` and the frontier-rise
  checkpoint hook.**

- **phase-planner (solve-lift judge 47).** The most Claude-Code-faithful and the strongest no-veer
  story, but it stacks the MOST new agents (planner + per-phase gate skeptics + per-phase
  `author_phase` ON for hard) — the highest bar to clear under C6 (autogen NEVER beat the template
  and cost 4x at the one shared solve). Its `phase_accept` adds a `score_subset` pytest mode (the
  cost hybrid avoids), and its own SECONDARY criterion ("ship PHASE-NOGATE if equal") concedes the
  headline gate may not earn its cost. Its phase-loop *shape* is sound and nearly identical to
  hybrid's; the difference is hybrid pays less. **Reject as the base; its delegation-contract scope
  and durable-plan shape are already in hybrid.**

### 1.3 The decision in one line

hybrid-converge-plus is the **lowest-regression-risk carrier of the (mostly already-banked) lift**,
the **cheapest** way to add ordered phases + partial banking + a grounded no-veer gate, and it
**fully realizes the user's goal** (phases-with-objectives + per-phase orchestration via
`ctx.workflow`/scoped converge + adversarial goal-alignment review) — with per-phase *codegen* as a
flag-gated seam rather than a speculative default. It changes NO proven seam
(`decompose`/`fanout`/`reduce`/`loop-until-dry`, the SPFG+ governor, the fail-open chain,
`ctx.select`-may-abstain).

---

## 2. Implementation plan (ordered, keyed to real files)

Build order is lowest-risk-first; each step is independently shippable and revertible. Everything is
gated behind `APEX_OMEGA_ORCHESTRATION=hybrid` (a new selector) + per-mechanism env flags so each
layer is independently ablatable and the A/B is apples-to-apples.

### STEP 1 — Acceptance-checkpointing: close the PARTIAL/phase gap (FOUNDATION, C1)

This is the unit of credit; do it first. The whole-suite path is already solid (verified §1.1).

**1a. `context.py` — new `_checkpoint_phase` (sibling of `_checkpoint_accepted` at `:372`).**
```python
def _checkpoint_phase(self, cand, *, subset_passed: int, subset_total: int,
                      phase_id: str = "") -> None:
    """Bank a PARTIAL/phase frontier gain to disk immediately, surviving an outer kill.
    Writes <run_dir>/phase_checkpoint.json (SEPARATE from accepted_checkpoint.json so a
    partial is NEVER reported solved:1 — C7). Atomic temp-write + replace (mirror
    _checkpoint_accepted:388-390). MONOTONE: overwrite only on a strict gold-pass-COUNT
    rise. Records {accepted: False, gold_passed, gold_total, content_sha, candidate_id,
    pass_rate, phase_id, repo}. NEVER sets .accepted. Best-effort, never fatal."""
```
Call it from `reduce_residuals` (`context.py:1342`, in the `_observe` branch) AND from the new
`run_phase` (Step 3) the instant a phase subset goes green or the global gold frontier strictly
rises. Graft from `novel-best` §4.1: the natural single call site is the `_observe` frontier-rise
branch (`context.py:494-505`), so EVERY frontier rise across fan-out/reduce/repair banks the partial
automatically — implement it as a call inside `_observe`'s `if improved:` block guarded on
`round_gold > self._best_gold_passed`, passing the candidate being observed.

**1b. `scripts/run_ladder.py` — extend `_recover_checkpoint` (`:267`)** to ALSO rglob
`phase_checkpoint.json` and return the partial frontier in telemetry ONLY (a separate return field,
e.g. `partial_frontier`). The whole-suite `accepted_checkpoint.json` stays the ONLY source that
emits `solved:1` (`:470`). A partial NEVER becomes a solve; it only (a) surfaces the recovered
frontier for the §3 audit and (b) lets a relaunch warm-resume read the strongest partial diff.

**1c. `apex_omega/eval/commit0_driver.py` — harden the Mode-A `TimeoutExpired` branch (`:259`)** to
consult `accepted_checkpoint.json` before returning the infra non-result (closes CONTEXT.md §2c.2 for
the Mode-A path). Mode C is already covered by 1b — document this so we do not over-claim it.

**1d. `scripts/validate_checkpoint.py` — extend** with (i) a partial-checkpoint case (assert a
phase-subset-green writes `phase_checkpoint.json` with `accepted:false` and NEVER yields `solved:1`)
and (ii) the existing whole-suite recovery case. This is the C1 regression gate.

### STEP 2 — `scope_ids` on `reduce_residuals` (graft from novel-best; the cheap subset predicate)

**`context.py` — extend `reduce_residuals` (`:1274`) signature** with an optional
`scope_ids: Optional[Sequence[str]] = None`:
```python
def reduce_residuals(self, candidates, *, carry_diff: str = "",
                     scope_ids: Optional[Sequence[str]] = None) -> dict:
```
- Default `None` == today's behaviour (the converge arm is byte-for-byte unchanged — verified the
  whole-suite accept still drives `_checkpoint_accepted` at `:1344`).
- When given: ALSO compute `phase_passed = scope_ids ⊆ (gold ids green in the merged tree)` as a
  **pure set test** over the full-suite score the merge already ran — NO second pytest call (the
  cost the `score_subset` mode in phase-planner/agentless would add). Return new keys
  `{phase_passed: bool, phase_pass_count: int, phase_total: int}` alongside the existing dict.
- The WHOLE-suite `accepted` field and its checkpoint are unchanged. This keeps the engine-owned
  whole-suite accept the only `solved:1` path (C7).

### STEP 3 — The host-side phase loop + `plan_phases` + `run_phase` (the core capability)

The phase loop lives **HOST-SIDE in `architect.py`** (not in the frozen script), sidestepping the
`workflow()` depth-1 cap (`context.py:646`). New selector + a new top layer in `autosolve`.

**3a. `architect.py` — new selector** in `author_orchestration` (`:311-326`): add
`if _orch_selector == "hybrid": return _freeze(engine, DEFAULT_ORCHESTRATION, "hybrid", ...)` — the
frozen script is the converge default; the HYBRID layer is host-side around it (so the frozen-script
fall-through is always today's proven path).

**3b. `architect.py` — new `phase_planned_solve(ctx, repo_map)`** called from `autosolve`
(`:580` region, after the floor-probe, before `run_orchestration`), gated by
`APEX_OMEGA_PHASE_PLANNER=1` (and only when `difficulty in {medium,hard}` AND `decompose()` ≥ 2
modules — the EXACT skip-gate as `templates.py:83-88`, C3). Pseudocode:
```python
def phase_planned_solve(ctx, repo_map):
    difficulty = str(repo_map.get("difficulty") or "").lower()
    if difficulty == "easy":
        return None  # caller runs the frozen converge script (which itself skip-gates)
    plan = ctx.decompose()
    modules = (plan or {}).get("modules") or []
    if not plan or len(modules) <= 1:
        return None
    phases = ctx.plan_phases(plan=plan, max_phases={"medium": 3, "hard": 4}[difficulty])
    if not phases or len(phases) <= 1:
        return None  # degenerate -> caller falls through to whole-repo converge
    carry = ctx.carry_best()
    for ph in phases:                                  # dependency order
        if not ctx.should_continue_waves():
            break
        g = ctx.goal_align_gate(plan, ph, residual_ids=ctx.last_residual(), stage="pre")
        if g["verdict"] == "abort":
            ctx.defer("plan_abort", ph["name"], g["reason"]); break
        if g["verdict"] == "revise":
            ph = _apply_retarget(ph, g)                # re-scope acceptance ids; never accept
        red = ctx.run_phase(ph, carry_diff=carry)      # scoped converge body (Step 4)
        if red.get("accepted_full"):
            return red["candidate"]                    # whole suite green in a phase -> done
        if red.get("phase_passed"):
            ctx._checkpoint_phase(red["candidate"],
                                  subset_passed=red["phase_pass_count"],
                                  subset_total=red["phase_total"], phase_id=ph["name"])
        carry = red.get("merged_diff") or carry
        g2 = ctx.goal_align_gate(plan, ph, residual_ids=red.get("residual"), stage="post")
        if g2["verdict"] == "abort":
            ctx.defer("plan_abort", ph["name"], g2["reason"]); break
    winner = ctx.select(ctx.all_candidates())          # engine-owned accept over ALL banked
    return winner                                       # may be None -> caller falls through
```
`autosolve` wiring: `winner = phase_planned_solve(...)`; if `None`, `winner =
run_orchestration(frozen.source, ctx)` (today's whole-repo converge); on any exception the existing
`_floor()` best-of-N runs (`architect.py:570`). Every degenerate path degrades to the proven floor.

**3c. `context.py` — new `plan_phases`** (read-only planner subagent; signature):
```python
PHASE_PLAN_SCHEMA = {  # added next to DECOMPOSE_SCHEMA (~context.py:94)
  "type": "object", "required": ["phases"], "properties": {"phases": {"type": "array",
    "items": {"type": "object", "required": ["objective", "acceptance_gold_ids"],
      "properties": {"name": {"type": "string"}, "objective": {"type": "string"},
        "acceptance_gold_ids": {"type": "array", "items": {"type": "string"}},
        "files_owned": {"type": "array", "items": {"type": "string"}},
        "depends_on": {"type": "array", "items": {"type": "string"}},
        "modules": {"type": "array", "items": {"type": "string"}}}}}}}

def plan_phases(self, *, plan: dict, max_phases: int, vendor=None, model=None,
                agent_id: int = 700200) -> Optional[list[dict]]:
    """ONE read-only ctx.ask (FIXED agent_id 700200, disjoint from decompose 700100),
    PHASE_PLAN_SCHEMA. Given decompose's modules + order + per-module gold_test_ids,
    emit <= max_phases ORDERED phases grouping dependency-coupled modules; each phase's
    acceptance_gold_ids = union of its modules' gold ids, VALIDATED against the real gold
    inventory on repo_map (drop hallucinated ids). Persist phase_plan.json (durable, like
    ~/.claude/plans + Pokemon NOTES.md). Re-read (not re-planned) on resume — mirror
    load_frozen (architect.py:286). FAIL-OPEN: schema-miss / <2 valid phases -> None."""
```

**3d. `context.py` — new `run_phase`** (the scoped converge body; signature):
```python
def run_phase(self, phase: dict, *, carry_diff: str = "") -> dict:
    """Run the proven converge inner loop (templates.py:90-148: fanout_modules ->
    reduce_residuals -> loop-until-dry) for ONE phase, SCOPED to phase['acceptance_gold_ids']
    via reduce_residuals(scope_ids=...). Each module agent is given the delegation contract
    (objective + files_owned boundaries + acceptance ids) reusing the solve_module brief
    (commit0_autogen.py:345-376). Returns {merged_diff, residual, phase_passed,
    phase_pass_count, phase_total, accepted_full, candidate, conflicts}. Stop authority is
    should_continue_waves() (SPFG+ governor) — NO new stop logic. ~30 lines, EXISTING seams
    only; a literal lift of the converge stages with residual target = the phase subset."""
```
Plus a tiny `last_residual()` accessor (returns the most recent `residual_failing_ids` for the
pre-gate grounding) — read-only, no agent.

### STEP 4 — Delegation-contract scope on the per-phase module agents (graft from phase-planner/D4)

`run_phase`'s `fanout_modules` call passes each module agent its `objective` + `files_owned`
boundaries + `acceptance_gold_ids`. This REUSES the `solve_module` brief shape that **already
scopes one agent to a module + its gold subset and tells it "Do NOT edit tests; re-run only the
failing subset"** (`commit0_autogen.py:345-376`) — so this is a prompt-augmentation, not a new
agent. Fail-open if `files_owned` is absent (advisory). No extra cost (CONTEXT.md D4, "cost risk:
low").

### STEP 5 — Grounded adversarial goal-alignment gate (the no-veer review, C6-gated)

**`context.py` — new `goal_align_gate`** (signature):
```python
GATE_SCHEMA = {"type": "object", "required": ["verdict"], "properties": {
  "verdict": {"type": "string", "enum": ["proceed", "revise", "abort"]},
  "reason": {"type": "string"}, "evidence_ids": {"type": "array", "items": {"type":"string"}},
  "retarget_gold_ids": {"type": "array", "items": {"type": "string"}}}}

def goal_align_gate(self, plan: dict, phase: dict, *, residual_ids: Sequence[str],
                    stage: str, n: int = 3) -> dict:
    """N read-only skeptics via ctx.signals (no plateau accounting, context.py:604), each
    asked: GOAL G (the binding task_framing re-injected at the decision point to fight
    instruction fade-out) + the phase objective + the REAL residual failing node-ids R (from
    reduce_residuals / last_residual, NOT the transcript) -> {verdict, reason, evidence_ids}.
    Admit-gated through adversarial_filter (context.py:664). GROUNDING (the moat over /goal):
    every revise/abort MUST cite evidence_ids that are real failing node-ids; an ungrounded
    verdict is DOWNGRADED to proceed. Ties / no-majority -> proceed (fail-open; the gate can
    only STOP a veer, never stall a progressing run). Read-only SIGNAL: can re-target (revise)
    or stop the phase loop (abort -> keep banked work + fall through to converge_tail), NEVER
    sets .accepted (C7). Gated to medium/hard by the caller; flag APEX_OMEGA_GOAL_GATE=0 OFF."""
```
Cost guards: gated to medium/hard only (easy never reaches it); `n` = 1 (medium) / 3 (hard); these
are CHEAP read-only `ctx.ask`/`ctx.signals` (no worktree, no scoring) — the solve-agent budget that
blew run-4's wall is UNCHANGED in count. Ablation arm `hybrid-nogate` measures whether it earns its
agents (C6).

### STEP 6 — Per-phase CODEGEN seam (the user's "generate code per phase"), flag-gated, DEFAULT OFF

Leave `run_phase` an optional `script_ref=` extension point: when a phase is flagged
`needs_custom_orchestration` by the planner AND `APEX_OMEGA_PHASE_CODEGEN=1` (default OFF), call the
existing `_author_via_llm` (`architect.py:350`) with a PHASE-SCOPED prompt (objective + acceptance
ids + `files_owned` + the real residual node-ids + the converge exemplar), `extract_code` +
`lint_source` it, `_freeze` to `orchestrator/phase_<id>_<sha>.py`, run via
`ctx.workflow({"scriptPath": ...})` (depth-1, legal). Lint-fail -> Tier-A scoped converge. This is
the literal "generate orchestration code per phase," reusing the ENTIRE existing author/lint/freeze
machinery, now driven per-phase WITH execution feedback (the gap §1b.2). It is OFF by default and
only enabled in a dedicated A/B arm if the cheaper arms plateau — refusing to pay run-4's
authoring-cost blowup speculatively (C6/C2).

---

## 3. A/B eval plan (exact arms + env + repos)

Run via `scripts/run_ladder.py` on the MAIN editable install (project memory: test on main, not a
worktree), `CELL_TIMEOUT=86400` (the 24h fair wall, already the default `run_ladder.py:64`),
checkpoint recovery ON.

### Arms
| Arm | Selector / flags | Isolates |
|---|---|---|
| **A converge** (control, the bar) | `APEX_OMEGA_ORCHESTRATION=converge` | incumbent |
| **B hybrid** (full) | `APEX_OMEGA_ORCHESTRATION=hybrid APEX_OMEGA_PHASE_PLANNER=1` (gate ON, codegen OFF) | the headline treatment |
| **C hybrid-nogate** (ablation) | `=hybrid APEX_OMEGA_PHASE_PLANNER=1 APEX_OMEGA_GOAL_GATE=0` | does the adversarial review EARN its agents? (C6) |
| **D hybrid-codegen** (ablation, optional) | `=hybrid APEX_OMEGA_PHASE_PLANNER=1 APEX_OMEGA_PHASE_CODEGEN=1` | does per-phase codegen earn its cost? (run ONLY if B/C plateau) |
| **E ralph** (floor) | `APEX_OMEGA_ORCHESTRATION=ralph` | vanilla persistence baseline |

### Repos (env)
```
LADDER_REPOS=mimesis,babel,voluptuous,jinja
LADDER_ARMS=converge,hybrid,hybrid-nogate,ralph
# (add hybrid-codegen to LADDER_ARMS only for the optional Arm D follow-up)
CELL_TIMEOUT=86400
```
- **HARD discriminators (the verdict):** `mimesis` (6044/6052 near-solve, the run-4 lost-solve repo,
  fetch-cheat canonical victim C4) + `babel` (4598/4607 near-solve) — modular, dependency-coupled,
  near-solve tails: exactly the ordered-phase + partial-banking sweet spot.
- **Controls (no-regression):** `voluptuous` (always-solves easy) + `jinja` (the one historical
  discriminator). B/C MUST hit the skip-gate and run the IDENTICAL cheap path as A — a regression
  here is a bug, not a strategy outcome; assert `agents_used` parity on voluptuous (the C3 guard).

### Protocol (C5)
- **n ≥ 3 seeds** per (arm × repo), n = 5 on mimesis/babel if budget allows; report solve-rate with
  Wilson CIs and median `agents_used`.
- **Exclude infra non-results from denominators** (Tier-1.3): `wall_s ≥ CELL_TIMEOUT` or null/partial
  → `status:timeout` with `nonresult_reason`, excluded (already in the run_ladder outcome taxonomy).
- **Checkpoint-loss audit** (the direct C1 regression test, Step 1d): for every cell assert that if
  any per-eval `report.json` shows a full pass, the cell reports `solved:1`; and that any
  `phase_checkpoint.json` has `accepted:false` and NEVER produces `solved:1`. Inject a synthetic
  outer kill on one pilot mimesis cell AFTER a known pass and assert recovery → `solved:1`.

### Acceptance criteria (when does B land? — C6 is the gate)
1. **PRIMARY (win):** B banks ≥ 1 verified solve on mimesis OR babel that A misses, across the seed
   set, with no control regression (voluptuous/jinja solve-rate unchanged within CI). If B never
   out-solves A, the phase layer is pure cost → keep it behind the flag, default OFF.
2. **NO run-4 blowup (C2):** B's solve-agent count + wall on mimesis/babel are within the governor's
   window of A's; NO cell converts an A-solve into a B-timeout (track `agents_used`, `wall_s`,
   `cut_losses.outcome`).
3. **GATE VALUE (secondary):** B beats C (gate-OFF) on a no-veer metric (fewer `defer` merge-conflict
   / fetch-monoculture attempts, higher median `best_gold_passed` on multi-module repos). If B ≈ C,
   the gate is pure cost → ship C (the cheaper arm).
4. **CHECKPOINT:** zero checkpoint-loss audit failures on all arms (C1 closed).
5. **CODEGEN (Arm D, only if run):** lands ONLY if D banks a solve B/C miss within ~1.5× B's agents;
   else codegen stays OFF.

### Sequencing
Step 1 (checkpoint) → Step 2 (`scope_ids`) → Steps 3+4 (phase loop + delegation contracts) →
Step 5 (gate) → Step 6 (codegen, flag-OFF). A/B after Steps 1-5 (Arms A/B/C/E); Arm D only if B/C
plateau.

---

## 4. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Phase layer adds cost without a win (C6) | Gated to medium/hard + ≥2 modules (skip-gate verbatim); only CHEAP read-only asks added (planner + gate); solve-agent budget UNCHANGED; A/B criterion 3.1 blocks landing; flag default OFF. |
| Subset banking masks a regression a later phase causes (novel-best R1) | The WHOLE-suite engine-owned `ctx.select` re-scores the merged tree and is the ONLY winner; `phase_checkpoint.json` is a recoverable PARTIAL, never a solve (C7). |
| Planner hallucinates acceptance gold ids | `plan_phases` validates against the real gold inventory on `repo_map`, drops invalid ids; <2 valid phases → `None` → fall through to whole-repo converge. |
| Gate becomes pure cost (C6) | Arm C (`hybrid-nogate`) is first-class; ship the cheaper arm if the gate shows no no-veer benefit; gate is fail-open (ties → proceed). |
| Gate false-abort veers the run wrongly | Admit-gated by `adversarial_filter`; ungrounded verdict downgraded to proceed; abort only STOPS the phase loop + KEEPS banked work + falls through to converge — never a fake pass. |
| Partial checkpoint misread as a solve (C7) | `phase_checkpoint.json` is a SEPARATE file; `_recover_checkpoint` returns it as `partial_frontier` telemetry only; only `accepted_checkpoint.json` yields `solved:1`. |
| Over-spawn / over-plan a shallow repo (C3) | Skip-gate verbatim (`templates.py:83-88`) + `max_phases` cap by difficulty (medium 3 / hard 4) + degenerate-plan fall-through; the planner is instructed to MERGE thin modules, not split. |
| Per-phase codegen re-incurs run-4's authoring blowup (C2) | Step 6 is flag-gated DEFAULT OFF; lint-checked; falls back to Tier-A; only enabled in Arm D if B/C plateau; never on easy. |
| `workflow()` depth-1 cap (gap 8) | Phase loop runs HOST-SIDE in `architect.py` at depth 0; only the optional per-phase snippet uses one `ctx.workflow` level (depth-1, legal). |
| Replan thrash / instruction fade-out | SPFG+ `should_continue_waves` caps the phase loop (no new stop logic); goal G re-injected at every phase/gate prompt (fights fade-out); gate conservative (default proceed). |
| Statistically weak matrix at n=1 (C5) | n ≥ 3-5 seeds with Wilson CIs; timeout-clips excluded from denominators. |
| Mode-A driver seam over-claimed | Documented: Mode C (the autogen arm) is in-process, covered by `run_ladder._recover_checkpoint`; the Mode-A `TimeoutExpired` consult (Step 1c) is a hardening, not load-bearing for this research. |
