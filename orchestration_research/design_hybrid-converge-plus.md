# Design: `hybrid-converge-plus`

A thin Claude-Code-style PHASE PLANNER layered ON TOP of the proven converge engine
(`decompose -> fanout -> reduce -> loop-until-dry`), plus an adversarial goal-alignment
gate between phases and full acceptance-checkpointing (including the partial/phase case).

**Design stance (the angle brief, verbatim):** keep the per-PHASE engine exactly as it is
(it is proven, lands the near-solve tail, and is replay-safe). Add ONLY (a) a thin host-side
top-level phase planner that emits objectives + ordering, (b) a grounded adversarial
goal-alignment gate at each phase boundary, and (c) the remaining acceptance-checkpointing
(partial/phase banking + the one un-wired recovery seam). This is the LOWEST-RISK evolution
that directly targets the run-4 regression (`C2`) and the near-solve loss (`C1`). Everything
degrades to the verified best-of-N floor; acceptance stays engine-owned (`C7`).

Grounded in: `apex_omega/autogen/{architect.py, context.py, templates.py}`,
`apex_omega/eval/{commit0_driver.py, commit0_autogen.py}`, `scripts/run_ladder.py`,
`orchestration_research/CONTEXT.md`, `APEX_COMMIT0_REPORT.md`.

---

## 0. TL;DR — what changes, in one paragraph

Today `autosolve` authors ONE whole-repo `orchestrate(ctx)` and runs it once. The convergence
default decomposes into spatial MODULES and fans them ALL out in parallel, ignoring
`decompose()`'s `order`/`depends_on`. We add a HOST-SIDE phase loop (in `architect.py`, NOT in
authored script) that turns `decompose()`'s already-computed `order`+`gold_test_ids` into ordered
PHASES, each with an explicit objective + an acceptance gold-id SUBSET. For each phase we RUN
THE EXISTING CONVERGE ENGINE scoped to that phase's gold ids (via `ctx.workflow("converge")`
plus a new lightweight phase scope), bank a PARTIAL checkpoint the instant a phase's subset goes
green, and run a grounded adversarial goal-alignment gate (proceed/revise/abort) before and after
each phase. If anything fails — no plan, single phase, planner abstains, gate aborts, engine
crashes — we fall straight through to today's behaviour (the whole-repo converge default, then the
best-of-N floor). Cost is bounded by reusing the existing easy-repo skip-gate + difficulty caps.

---

## 1. Architecture: where the phase loop lives

```
autosolve(engine, ...)                                  [architect.py — MODIFIED]
  scout -> repo_map (difficulty, modules, approach)
  author/freeze (unchanged; the phase plan is HOST-SIDE, not in the frozen script)
  floor-probe (unchanged: banks a verified wave-0 best-of-N, checkpoint gated)
  +-- NEW: phase_planned_solve(ctx, repo_map)           [the hybrid top layer]
  |     if not eligible (easy / <2 modules / flag off):  -> run_orchestration(frozen.source, ctx)   # today's path
  |     plan = ctx.plan_phases()                          # NEW ctx method: planner subagent
  |     if plan is None or len(plan.phases) <= 1:         -> run_orchestration(frozen.source, ctx)   # fall through
  |     persist plan to disk (durable; survives kill)
  |     carry = ""
  |     for phase in plan.phases (dependency order):
  |        gate = ctx.goal_align_gate(plan, phase, residual_ids=..., stage="pre")   # NEW: adversarial
  |        if gate.verdict == "abort": break              # honest stop, keep banked work
  |        if gate.verdict == "revise": phase = apply_revision(phase, gate)         # re-target, never accept
  |        red = ctx.run_phase(phase, carry_diff=carry)   # NEW thin wrapper -> existing converge engine, scoped
  |        if red.accepted_full: return red.candidate     # whole gold suite green inside a phase -> done
  |        if red.phase_accepted: ctx.checkpoint_phase(phase, red)   # NEW: PARTIAL checkpoint banked instantly
  |        carry = red.merged_diff or carry               # carry the best partial forward across phases
  |        gate = ctx.goal_align_gate(plan, phase, residual_ids=red.residual, stage="post")
  |        if gate.verdict == "abort": break
  |     winner = ctx.select(ctx.all_candidates())         # engine-owned accept over EVERYTHING banked
  |     if winner is None: winner = run_orchestration(frozen.source, ctx)   # fall through to whole-repo converge
  |     return winner
  +-- fail-open to _floor() best-of-N on any exception (unchanged)
```

Key invariants the placement preserves:
- **The phase loop is HOST-SIDE Python in `architect.py`**, not authored script. This sidesteps
  the `workflow()` depth-1 cap (`context.py:646`, gap #8) — each phase composes ONE level
  (`ctx.workflow("converge")`-shaped), never deeper.
- **The per-phase engine is the UNCHANGED converge shape.** We do not rewrite
  `decompose/fanout_modules/reduce_residuals/loop-until-dry`. We scope them to a phase's gold-id
  subset and feed the cross-phase `carry`.
- **Acceptance is still ONLY `ctx.select` over real candidates** (`C7`). A phase objective being
  "met" is recorded as a checkpoint + a `defer`-style ledger entry; it never sets `.accepted`.

---

## 2. ctx API mapping (existing methods reused + NEW methods/hooks)

### 2a. EXISTING ctx methods reused verbatim (no change)
| Need | Existing method | Where |
|---|---|---|
| spatial breakdown + `order` + `depends_on` + `gold_test_ids` | `ctx.decompose()` | context.py:1155 |
| per-module fan-out (no barrier), carry-seeded | `ctx.fanout_modules(modules, carry_diff=)` | context.py:1247 |
| merge + ONE full-suite score, conflict-safe | `ctx.reduce_residuals(cands, carry_diff=)` | context.py:1274 |
| loop-until-dry repair on live tree | `ctx.repair_residual(ids, carry_diff=, round=)` | context.py:1368 |
| union of gold ids for collection-error repair | `ctx.module_gold_ids(modules)` | context.py:1117 |
| running best partial diff (cross-phase carry) | `ctx.carry_best()` | context.py:1097 |
| resume-safe wave stop authority (per-phase) | `ctx.should_continue_waves()` | context.py:579 |
| read-only schema'd subagent (planner + gate substrate) | `ctx.ask(prompt, schema=, agent_id=, strict=)` | context.py:870 |
| read-only fan-out without plateau accounting | `ctx.signals(thunks)` | context.py:604 |
| admit-gate plain-data findings (skeptic survivors) | `ctx.adversarial_filter(items, votes=)` | context.py:664 |
| execution-authoritative winner / abstain | `ctx.select(candidates)` | context.py:1467 |
| structured IOU / blocked-on ledger | `ctx.defer(scope, item, reason)` / `ctx.blocked()` | context.py:712 |
| partial whole-suite checkpoint (already half-built) | `ctx._checkpoint_accepted(cand)` | context.py:372 |
| one-level nested composition | `ctx.workflow("converge")` | context.py:639 |

### 2b. NEW ctx methods (thin, journaled, never set acceptance)

All four are thin host-side wrappers that reuse `ctx.ask`/`ctx.signals`/`reduce_residuals` and the
existing journal — so they are replay-deterministic and cannot promote an unverified solve.

```python
# --- 1) PHASE PLANNER: one read-only subagent emitting an ORDERED plan ---
PHASE_PLAN_SCHEMA = {
  "type": "object", "required": ["phases"],
  "properties": {"phases": {"type": "array", "items": {
      "type": "object", "required": ["objective", "acceptance_gold_ids"],
      "properties": {
        "name": {"type": "string"},
        "objective": {"type": "string"},          # 1-2 sentence delegation contract
        "acceptance_gold_ids": {"type": "array", "items": {"type": "string"}},
        "files_owned": {"type": "array", "items": {"type": "string"}},   # boundary (advisory)
        "depends_on": {"type": "array", "items": {"type": "string"}},
        "modules": {"type": "array", "items": {"type": "string"}}}}},     # which decompose modules
   "order": {"type": "array", "items": {"type": "string"}}}}

def plan_phases(self, *, plan=None, max_phases=None, vendor=None, model=None,
                agent_id=700200) -> Optional[dict]:
    """Read-only Planner subagent (sandbox=read-only, FIXED agent_id 700200, disjoint from
    decompose 700100). Given the decompose() plan (or compute it), emit ORDERED phases each
    with an objective + acceptance_gold_ids SUBSET + depends_on, derived from the modules'
    gold_test_ids + topological order. Persists to <run_dir>/phase_plan.json (durable like
    ~/.claude/plans). FAIL-OPEN: schema-miss / <2 phases -> None -> caller falls through to
    the whole-repo converge default. max_phases bounds over-planning (see cost guard §5)."""
```

```python
# --- 2) RUN ONE PHASE: the EXISTING converge engine scoped to a gold-id subset ---
def run_phase(self, phase: dict, *, carry_diff: str = "") -> dict:
    """Run the proven converge shape for ONE phase, scoped to phase['acceptance_gold_ids'].
    Internally: select the phase's modules from the plan, fanout_modules(carry_diff=carry),
    reduce_residuals(carry_diff=carry), then loop-until-dry on the PHASE residual until the
    phase subset is green OR should_continue_waves() halts. Returns:
      {merged_diff, residual, phase_passed (subset green), phase_pass_count, gold_total_subset,
       accepted_full (WHOLE gold suite green -> bank winner + return now), candidate, conflicts}.
    Reuses reduce_residuals' full-suite score (zero extra tokens); phase_passed is computed from
    that score restricted to the phase's gold-id subset (a pure set test, no extra pytest run)."""
```
`run_phase` is ~30 lines and is a *literal* lift of the converge default's stages (1)-(3)
(`templates.py:90-148`) with two changes: the residual target is `phase['acceptance_gold_ids']`
(not the whole suite), and the loop stop adds `phase_passed`. It calls EXISTING seams only.

```python
# --- 3) PARTIAL / PHASE CHECKPOINT: bank a phase subset green to disk instantly ---
def checkpoint_phase(self, phase: dict, red: dict) -> None:
    """Atomically append a PARTIAL acceptance record to <run_dir>/phase_checkpoints.jsonl the
    instant a phase's gold-id SUBSET goes green (and bank the merged_diff candidate). This is the
    missing partial case of _checkpoint_accepted (CONTEXT.md §2c.1). It records progress
    (best gold-pass count + merged content_sha), NOT a whole-suite accept — so it can NEVER be
    read as solved:1 by the harness; it only (a) lets a relaunch warm-resume from the strongest
    partial and (b) feeds the cut-losses ledger a real frontier. Idempotent per phase; best-effort.
    Also records ctx.defer('phase_done', phase['name'], '<n>/<m> gold ids green')."""
```

```python
# --- 4) GROUNDED ADVERSARIAL GOAL-ALIGNMENT GATE between phases ---
GATE_SCHEMA = {"type": "object", "required": ["verdict"],
  "properties": {"verdict": {"type": "string", "enum": ["proceed", "revise", "abort"]},
                 "reason": {"type": "string"},
                 "retarget_gold_ids": {"type": "array", "items": {"type": "string"}}}}

def goal_align_gate(self, plan: dict, phase: dict, *, residual_ids: list, stage: str,
                    n: int = 3) -> dict:
    """N read-only skeptics (ctx.signals fan-out of ctx.ask, agent_ids 700210+i) judge:
    'Given goal G (make the gold suite pass), the ordered phase plan, and the REAL residual
    failing node-ids R, does proceeding with THIS phase still serve G? verdict
    proceed/revise/abort + reason + optional retarget_gold_ids.' The verdicts are then passed
    through ctx.adversarial_filter so only a verdict that SURVIVES the skeptics is admitted;
    ties / no-majority default to 'proceed' (fail-open, never blocks progress). CRITICAL: R is
    the real failing-test output from reduce_residuals, NOT the transcript — closing /goal's
    'confident summary of broken work' blind spot (CONTEXT.md §3, D3). The gate is a read-only
    SIGNAL: 'revise' RE-TARGETS the phase's gold ids; 'abort' STOPS the phase loop (host keeps
    all banked work + falls through to whole-repo converge). It can NEVER set acceptance (C7).
    Gated to medium/hard via the caller (easy repos pay nothing)."""
```

### 2c. NEW engine/harness hooks (the remaining checkpoint wiring)

The outer-kill recovery (`run_ladder.py:_recover_checkpoint` + the clean-completion review-fix #8)
and the budget-aware eval cap (`commit0_autogen.py:392` `eval_cap = max(300, min(1800,
cell_timeout//3))`) are ALREADY built. Two seams remain:

1. **`run_ladder._recover_checkpoint` also reads `phase_checkpoints.jsonl`** — but ONLY to surface
   the partial frontier in telemetry, NEVER to emit `solved:1` (a partial is not a solve). The
   whole-suite `accepted_checkpoint.json` stays the ONLY solve-recovery source. (5-line change.)
2. **The child driver's own `TimeoutExpired` path consults the checkpoint** before returning the
   infra non-result. `commit0_driver._run_mode_a` (line 259) and the autogen-cell return in
   `commit0_autogen.run_autogen_cell` both must call a shared `_recover_checkpoint(run_dir)` and,
   on a whole-suite `accepted_checkpoint.json`, return `solved:1` instead of the non-result. This
   is the one un-wired recovery seam named in CONTEXT.md §2c.2 (Mode-A driver path). (~15 lines.)

No change to `_checkpoint_accepted`'s whole-suite write, the SPFG+ governor, or `ctx.select`.

---

## 3. How each MANDATORY requirement is met

**(1) Split into phases/chunks with clear objectives (like Claude Code).**
`ctx.plan_phases()` emits an ordered list of `{name, objective, acceptance_gold_ids, files_owned,
depends_on}` derived from `decompose()`'s `order` + per-module `gold_test_ids` (currently
discarded, gap #4). The `objective` + `files_owned` ARE the external "detailed delegation
contract" (CONTEXT.md §3, plan-adherence mechanism #1: objective + boundaries cure
duplicate-work/gaps). The plan is persisted to `phase_plan.json` (durable like `~/.claude/plans`,
CONTEXT.md §3 corroborated).

**(2) Generate orchestration code per phase OR reuse `ctx.workflow`.**
We REUSE the proven converge engine per phase via `ctx.run_phase` (which is the converge stages
scoped to the phase). Equivalent: a phase MAY instead compose `ctx.workflow("converge")` for the
whole-repo shape — both honored. We deliberately do NOT author NEW Python per phase: that is the
higher-risk path the angle brief rejects, and `_author_via_llm` running per-phase would re-incur
the run-4 authoring cost. (If a future iteration wants per-phase codegen, the seam is
`ctx.run_phase` accepting an optional `script=` ref — left as a one-line extension point.)

**(3) Adversarial goal-alignment review preventing veering.**
`ctx.goal_align_gate` runs N skeptics at every phase boundary, GROUNDED in the real residual
failing node-ids (not the transcript), admit-gated through `adversarial_filter`. It implements
the four stacked no-veer mechanisms (CONTEXT.md §3): delegation contracts (the phase objective),
exactly-one-in-progress focus (one phase active at a time + the `defer` ledger), goal re-injection
at the point of decision (the gate re-states G each boundary), and a separate verifier (the gate)
— but the verifier is execution-grounded, the strict improvement over `/goal`.

**(4) Acceptance-checkpointing (verified solve banked instantly, survives outer kill).**
Whole-suite: `_checkpoint_accepted` already banks the instant a candidate is `.accepted`
(context.py:842, 1344); recovered on outer kill (run_ladder:470) and clean completion (review-fix
#8). NEW: (a) `checkpoint_phase` banks the PARTIAL case the instant a phase subset is green
(closes CONTEXT.md §2c.1); (b) the child-driver `TimeoutExpired` path consults the checkpoint
(closes §2c.2); (c) the budget-aware eval cap is already wired (commit0_autogen:392). All writes
are atomic temp-write + `replace` (context.py:388) so an outer kill mid-write cannot corrupt them.

**(5) No run-4 over-spawn/budget-blowup regression — the cost guard (§5 below).**

**(6) Degrade to verified best-of-N floor; acceptance engine-owned.**
Every failure mode (no plan / 1 phase / planner abstain / gate abort / phase engine crash) falls
through to `run_orchestration(frozen.source, ctx)` (whole-repo converge), then to `_floor()`
best-of-N on any exception — the UNCHANGED fail-open chain (architect.py:580-628). `ctx.select`
remains the only producer of `.accepted` (C7); no phase objective, gate verdict, or checkpoint
ever sets it.

---

## 4. Phase planner design (detail)

- **Eligibility:** only when `difficulty in {medium, hard}` AND `decompose()` returns >=2 modules
  AND `APEX_OMEGA_PHASE_PLANNER=1` (or the `phase_planner` ablation flag). This mirrors the
  converge skip-gate (templates.py:83-88) EXACTLY, so easy/voluptuous/jinja never enter the phase
  layer (C3).
- **Plan derivation:** the Planner subagent is GIVEN `decompose()`'s output (modules + order +
  per-module gold ids + scout `approach`). Its job is to GROUP modules into ordered phases honoring
  `depends_on` (e.g. phase 1 = data models, phase 2 = parsers depending on models, phase 3 = API).
  Each phase's `acceptance_gold_ids = union(gold_test_ids of its modules)`. If the model returns
  a degenerate 1-phase plan, we fall through (no benefit, no cost beyond one read-only ask).
- **Execution order:** phases run sequentially in dependency order (the thing the current
  parallel-ignore-order fanout gets wrong, gap #4). WITHIN a phase, `fanout_modules` still fans the
  phase's modules in parallel (cheap, no barrier). So we get ordered PHASES of parallel MODULES.
- **Cross-phase carry:** `carry` (the best merged partial diff) threads through every phase via the
  existing `carry_diff=` parameter, so phase N edits the accumulated work of phases 1..N-1 — the
  off-by-K near-solve closer, now across temporal phases.
- **Durability:** `phase_plan.json` + `phase_checkpoints.jsonl` on disk; on relaunch the host reads
  them and skips already-banked phases (warm resume = the Pokemon NOTES.md pattern, CONTEXT.md §3).
  The journal already replays the agent calls; the plan file replays the PHASE control flow.

---

## 5. Cost guard — explicitly NOT the run-4 regression

Run-4 regressed because repair-lineages-ON + agent-cap 8->16 spent more compute with ZERO new
solves and turned a verified solve into a TIMEOUT (C2). The phase layer's guards:

1. **Same easy-repo skip-gate (C3).** The phase layer is gated to medium/hard >=2-module repos —
   identical to the converge gate. voluptuous/jinja NEVER pay the planner/gate cost.
2. **Bounded extra agents.** The ONLY new agents are: 1 planner (per cell) + N=3 gate skeptics per
   phase boundary, gated to medium/hard. With `max_phases` bounded (default 4, see #3) that is
   `1 + 3*(phases+1) <= ~16` read-only asks per cell — but these are `ctx.ask`/`ctx.signals`
   (read-only, no worktree, no scoring), the CHEAP agent class. The expensive SOLVE agents are
   UNCHANGED in count (the same fanout_modules + repair_residual the converge default already
   spends). So the phase layer does not increase the solve-agent budget that blew the run-4 wall.
3. **`max_phases` cap sized to difficulty** (external scaling heuristic, CONTEXT.md §3): medium=3,
   hard=4. Over-planning a shallow repo is structurally impossible (capped + degenerate-plan
   fall-through). The Planner is INSTRUCTED to merge thin modules into a phase, not split.
4. **Bank-cheap-wins-first.** Because phase 1 runs first and its partial checkpoint banks the
   instant its subset is green, a cell killed at the wall has ALREADY banked phases 1..k — the
   exact thing run-4 failed to do (it computed a full solve and banked nothing). More compute now
   ALWAYS leaves a recoverable artifact.
5. **The SPFG+ governor is the per-phase stop authority (unchanged).** `should_continue_waves()`
   cuts a true plateau within a phase; a climbing frontier keeps going. No new stop logic, no new
   way to overspend. The agent ceiling + token budget are shared across phases via the same engine.
6. **Gate is fail-open + medium/hard-only.** If the gate skeptics error or tie, the verdict
   defaults to `proceed` — the gate can only STOP veering, never stall a progressing run. Its cost
   is measured against an ablation with the gate OFF (§6) so it must EARN its agents (C6).

Net: the phase layer adds only cheap read-only asks, reuses the exact solve-agent budget the
converge default already governs, and makes every wall-killed cell leave a recoverable partial.
That is the structural inverse of the run-4 blowup.

---

## 6. A/B eval plan (mimesis + babel + controls)

**Arms (each a frozen, resumable selector, like the existing `converge`/`ralph` arms):**
- `A. converge` (control) — `APEX_OMEGA_ORCHESTRATION=converge`, today's whole-repo converge
  default. The incumbent the hybrid must beat (C6).
- `B. hybrid-converge-plus` (full) — `APEX_OMEGA_PHASE_PLANNER=1` (planner + per-phase converge +
  goal-align gate + partial checkpoint).
- `C. hybrid-no-gate` (ablation) — phase planner + per-phase converge + partial checkpoint, gate
  OFF. Isolates the gate's contribution (does the no-veer review EARN its skeptics? C6).
- `D. ralph` (vanilla control) — naive persistence, already exists.

**Repos:**
- HARD discriminators (the win target): **mimesis** (near-solve 6044/6052), **babel**
  (near-solve 4598/4607). These are modular, dependency-coupled, near-solve-tail repos — exactly
  what ordered phases + partial banking target (CONTEXT.md C5, D2).
- Controls: **voluptuous** (always-solves, easy — must STAY solved at NO extra cost: confirms the
  skip-gate holds), **jinja** (the only n=1 discriminator — must not regress).

**Protocol:**
- `n>=3` seeds per (arm x repo) cell (n=5 if budget allows), per C5 (n=1 is statistically void).
  Report solve-rate with CIs.
- Cell wall `CELL_TIMEOUT=86400` (the current 24h fair wall) so nothing is truncated mid-work; the
  partial/whole checkpoint makes any earlier wall safe regardless.
- Run via `scripts/run_ladder.py` (REPOS/ARMS env-overridable) on the MAIN editable install (per
  project memory: test on main, not a worktree).

**Primary acceptance criteria (the gate to land):**
1. **Banks a solve the control misses (C6).** B solves >=1 of {mimesis, babel} on >=1 seed where
   A abstains/times-out — the concrete near-solve-tail recovery claim. If B never out-solves A,
   the phase layer is pure cost and is NOT landed (kept behind the flag, OFF).
2. **No control regression (C3).** B and C solve voluptuous + jinja at the SAME rate as A, with
   agents_used within +-15% on the easy controls (the skip-gate must make the phase layer free
   there).
3. **No run-4-style blowup (C2).** B's solve-agent count and wall on mimesis/babel are within the
   governor's window of A's; NO cell converts an A-solve into a B-timeout. Track agents_used,
   wall_s, and `cut_losses.outcome` per cell.
4. **Checkpoint correctness (C1).** Inject a synthetic outer kill (lower `CELL_TIMEOUT` for one
   pilot cell) AFTER a known mimesis full-solve and AFTER a phase-1 subset-green; assert the
   whole-suite checkpoint recovers `solved:1` and the partial checkpoint surfaces the frontier
   WITHOUT faking a solve. Extend `scripts/validate_checkpoint.py` with a partial-checkpoint case.

**Secondary metrics:** per-phase frontier history (`_frontier_history`), gate verdict distribution
(proceed/revise/abort counts), `defer('phase_done')` ledger, and the integrity log (fetch
monoculture, C4). Compare B vs C to attribute any win to the gate vs the phase structure alone.

---

## 7. Implementation order (lowest-risk first)

1. **Checkpoint wiring (D1 remainder).** Add the shared `_recover_checkpoint` consult to the child
   driver `TimeoutExpired` path (commit0_driver:259 + commit0_autogen return). Add
   `checkpoint_phase` + the partial `phase_checkpoints.jsonl` write. Extend
   `validate_checkpoint.py`. (Lowest risk, recovers existing losses, prerequisite to measuring any
   phase win.)
2. **`run_phase`** — the scoped converge wrapper (pure lift of converge stages; the engine seams
   are unchanged). Unit-test it solves a 2-module fixture scoped to a subset.
3. **`plan_phases`** + `PHASE_PLAN_SCHEMA` + `phase_planned_solve` host loop in `architect.py`,
   behind `APEX_OMEGA_PHASE_PLANNER`. Fall-through on every degenerate case.
4. **`goal_align_gate`** + `GATE_SCHEMA`, gated medium/hard, fail-open to `proceed`.
5. **A/B run** (arms A-D, repos above, n>=3) — the gate to land. Accept ONLY if criterion 6.1 holds.

Each step is independently shippable and independently revertible; steps 1-2 have value even if the
phase planner (3-4) never lands.

---

## 8. Risks + mitigations

| Risk | Mitigation |
|---|---|
| Planner emits a bad/degenerate phase plan | Schema-validated `ctx.ask` + fall-through on <2 phases; the per-phase engine is the proven converge shape regardless. |
| Phase layer adds cost without a win (C6) | Gated to medium/hard; only cheap read-only asks added; A/B criterion 6.1 blocks landing if no out-solve; flag default OFF. |
| Gate veers the run wrongly (false abort) | Gate is fail-open (ties->proceed), admit-gated by `adversarial_filter`, grounded in REAL residual ids; abort only KEEPS banked work + falls through. Ablation C measures its value. |
| Partial checkpoint misread as a solve | `phase_checkpoints.jsonl` is a SEPARATE file from `accepted_checkpoint.json`; only the whole-suite file ever yields `solved:1`; partial is telemetry/warm-resume only. |
| `workflow()` depth-1 cap | Host-side phase loop composes at most one level per phase; never nests. |
| Over-spawn on a shallow repo | Same skip-gate as converge + `max_phases` cap + degenerate-plan fall-through; structurally cannot over-plan easy repos. |
| Cross-phase carry conflict | Reuses the existing carry-conflict handling (`apply_diff` False -> indeterminate candidate, carry kept) in `_attempt`/`reduce_residuals`; a failed carry never erases prior phases. |

---

## 9. Why this is the right (low-risk) evolution

It changes NO proven seam: `decompose/fanout/reduce/loop-until-dry`, the SPFG+ governor, the
fail-open chain, and `ctx.select`-may-abstain are all untouched. It adds a thin host-side ordering
layer that uses data `decompose()` ALREADY computes but throws away (`order`, per-module
`gold_test_ids`), a grounded review that reuses `adversarial_filter`, and the partial-checkpoint
case of a mechanism that is already half-built and validated. It is the minimal set of additions
that makes the engine plan like Claude Code (ordered objectives + no-veer review + durable plan)
while structurally preventing the run-4 regression and recovering the near-solve loss — with a hard
A/B gate (out-solve the control or stay OFF) so the new agents must earn their cost.
