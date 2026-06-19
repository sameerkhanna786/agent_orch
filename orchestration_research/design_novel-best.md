# Design: PSP-Ω — Phase-Scheduled Planner with Frontier-Banked Acceptance

`name: novel-best`

A single, concrete, implementable design for APEX-Ω that makes the planner behave like
Claude Code on large comprehensive tasks: split into ordered PHASES with explicit objectives,
generate orchestration code PER PHASE (or reuse `ctx.workflow`), run a grounded ADVERSARIAL
goal-alignment review that gates phase progression, and bank a verified solve THE INSTANT it
passes so it survives an outer kill and a budget-aware eval timeout — never repeating run-4's
over-spawn/budget-blowup, never faking a pass.

Every choice below is justified against the empirical constraints C1–C7 (CONTEXT.md §4) and
mapped precisely onto the real `ctx` API I read in `context.py`/`architect.py`/`templates.py`.

---

## 0. The one-liner and the core thesis

**One-liner:** A host-side PHASE SCHEDULER that turns `ctx.decompose`'s *discarded* topological
order + per-module gold ids into ordered phases, runs each phase as the existing
fan-out→reduce→loop-until-dry converge body bounded to that phase's gold subset, banks each
phase's strict gold-frontier gain to disk immediately (partial-acceptance checkpoint), gates
progression with a residual-grounded adversarial goal-review, and degrades to the verified
best-of-N floor — with a strict budget partition so phases can never collectively over-spend.

**Thesis (why this is the best single design):** The hard discriminators (mimesis 6044/6052,
babel 4598/4607) are *near-solves where the residual tail is the killer*. The current converge
default already carries the best partial forward and loops on residuals — but it (a) ignores
`decompose`'s `order`, fanning ALL modules in parallel even when they are dependency-coupled
(CONTEXT.md §1b.4); (b) scores the WHOLE suite every round so a finished early module earns no
banked credit and a late-phase regression can erase it (§1b.5); and (c) has no mechanism to
detect "this attempt is drifting from the goal" (§1b.3). The single highest-leverage change is
therefore not a new agent loop but a **scheduler that imposes order + per-phase banking + a
grounded no-veer gate on the converge body we already trust** — reusing the engine-owned accept
gate verbatim so acceptance stays honest (C7).

This is `/goal`'s loop shape (durable plan + per-step verifier-gated progression) grounded in
real pytest instead of a transcript — provably stronger than `/goal`'s blind evaluator
(CONTEXT.md §3, novel-idea #1), and it is built on seams that already exist.

---

## 1. The phase model (Claude-Code-style split with objectives)

### 1.1 What a PHASE is

A phase is a **delegation contract** (the corroborated cure for vague-delegation veering,
CONTEXT.md §3). Derived from `ctx.decompose()` (which already returns
`{modules:[{module, gold_test_ids, depends_on, files}], order}`) but assembled into a *temporal*
plan the current code throws away:

```
Phase = {
  "id":               "p3",
  "objective":        "implement the parsers so their gold ids go green",   # human-readable goal
  "modules":          [<module dicts from decompose, grouped by dependency level>],
  "acceptance_ids":   [<union of gold_test_ids the phase OWNS>],            # per-phase predicate
  "files_owned":      [<union of module.files; advisory boundary>],
  "depends_on":       ["p1", "p2"],                                          # from decompose order
  "agent_budget":     <int>,        # this phase's slice of the cell budget (see §5)
  "effort":           "low|high",   # budget-aware effort lever (§5)
}
```

Phases are formed by **topological-leveling** `decompose`'s `order`/`depends_on`: modules with
no unmet dependency form phase 1, the next dependency level forms phase 2, etc. Modules within a
phase are independent (safe to fan out in parallel); phases run sequentially (the dependency
respect the current default ignores). This directly closes §1b.4 and the duplicate-work failure
mode on babel/mimesis (CONTEXT.md §3, "vague delegation is THE documented failure").

### 1.2 Phase-count discipline (Claude Code "manageable chunks", and the C3 cost guard)

Phase count is bounded by difficulty, NOT by module count, so a shallow repo never gets
over-planned (C3, the voluptuous/jinja over-spawn pathology):

- **easy / `<=1` module / undecomposable** → **ZERO phases.** The scheduler does NOT plan; it
  delegates straight to `ctx.workflow("default-best-of-n")`. This is the exact existing skip-gate
  (`templates.py:83-88`) and is load-bearing for C3 — the planner pays *nothing* on easy repos.
- **medium** → up to **3 phases** (matches the external scaling heuristic: comparison/medium =
  2–4 subagents).
- **hard** → up to **5 phases**; deeper dependency chains are merged into the last phase rather
  than spawning more phases (cap the *plan*, not the *work*).

If leveling produces more groups than the cap, adjacent low-fan-out levels are merged so the cap
holds. This is a pure host-side function over the journaled plan — deterministic, zero tokens.

### 1.3 Durability (survive context reset / restart — the §3 "save plan to memory" lesson)

The plan + a residuals ledger are persisted to disk the moment they are computed, mirroring
Anthropic's LeadResearcher "save plan to memory before work" and Plan Mode's `~/.claude/plans/`
(CONTEXT.md §3, both corroborated):

```
<run_dir>/phase_plan.json        # the ordered phases (immutable once frozen)
<run_dir>/phase_ledger.json      # mutable: per-phase {status, best_gold_passed, banked_diff_sha, review_verdict}
```

On resume the scheduler reads these instead of re-planning (same pattern as
`load_frozen`). The plan is journaled via `ctx.ask` so it is *also* replay-deterministic; the
JSON files are the human-readable + cross-process-recoverable mirror.

---

## 2. Per-phase orchestration code (generate OR reuse `ctx.workflow`)

The requirement "generate orchestration code per phase OR reuse ctx.workflow" is satisfied with
a **two-tier** approach that earns its cost (C6):

### Tier A (default, cheap): reuse the trusted converge body, bounded to the phase

Each phase runs the EXISTING converge inner loop — fan-out the phase's modules → reduce →
loop-until-dry — but scored against the **phase's `acceptance_ids` subset**, not the whole suite.
This is the converge default we already trust (it banks the near-solve tail), now *scoped*. No
new codegen, no new agent, zero added authoring cost on top of the converge arm. This is the
A-tier because C6 says authored orchestration must EARN its cost: most phases don't need bespoke
code.

### Tier B (gated, per-phase codegen): author a phase-specific `orchestrate_phase(ctx)`

For a phase the scheduler flags as *structurally unusual* (heuristic, §6: a phase whose
adversarial review (§3) returned `revise`, OR a hard-difficulty phase with a known residual class
after Tier-A failed once), call the existing `_author_via_llm` path — but with a **phase-scoped
author prompt** carrying: the phase objective, `acceptance_ids`, `files_owned` boundaries, the
real residual node-ids from the prior reduce, and the converge exemplar. The authored source is
`extract_code` + `lint_source`'d and frozen as `orchestrator/<phase_id>_<sha>.py` (reusing
`_freeze`'s machinery, namespaced by phase). It runs via `ctx.workflow({"scriptPath": ...})` in a
child context (one level deep — within the existing nesting cap, CONTEXT.md §1b.8). Lint-fail →
fall back to Tier A. This is per-phase codegen that observes phase N-1's execution feedback —
the exact gap §1b.2 identifies (the current author never sees execution feedback).

**Why two tiers:** Tier A guarantees we never do worse than the converge arm (the floor of the
floor). Tier B is the Claude-Code-style "generate code per phase" capability, but *gated* so it
only fires where it can plausibly pay for itself — directly answering C6.

---

## 3. Grounded adversarial goal-alignment review (the no-veer gate)

A read-only review gates phase progression at TWO points: **before** codegen/execution of phase N
(is the plan still right?) and **after** phase N (did we drift, should we proceed/revise/abort?).

### 3.1 Mechanism

`N` skeptics via `ctx.signals([...ctx.ask...])` (read-only fan-out, NOT a solve wave so it never
advances the plateau counter — `context.py:604`), each asked a schema'd question:

```
GOAL_REVIEW_SCHEMA = {verdict: "proceed|revise|abort", reason: str, retarget_ids: [str]}
```

Prompt (the load-bearing grounding): the phase objective + goal G + **the REAL residual failing
node-ids and sanitized failure excerpts from the last `reduce_residuals`** (not the agent's
transcript). Verdicts are admitted through the existing `ctx.adversarial_filter` (majority vote,
`context.py:664`) so a single hallucinating skeptic cannot stall the run.

- **proceed** → advance to the next phase.
- **revise** → re-plan THIS phase (re-target `acceptance_ids` to `retarget_ids`, or flag Tier-B
  codegen) and re-run it once; if still no frontier gain, treat as abort.
- **abort** → stop spending on this phase, `ctx.defer("phase_abort", phase_id, reason)` (the IOU
  ledger, `context.py:712`), and proceed to the next independent phase (its work is not wasted —
  the banked partial carries forward).

### 3.2 Why grounding in real pytest is the moat

This is the precise place CONTEXT.md §3 says `/goal` fails: its evaluator "only reads the
conversation... runs nothing... a confident summary of broken work reads as 'fine'." By feeding
the review the **real residual node-ids from `reduce_residuals`** (execution evidence, not the
agent's self-report), the gate cannot be fooled by a confident-but-broken summary. It is `/goal`'s
loop shape with a strictly stronger verifier — the engine's documented moat (CONTEXT.md §2c.5).

### 3.3 Cost guard (C6): the review is gated and engine-subordinate

- The review runs ONLY on medium/hard repos (mirror the verify-phase gate `templates.py:153`);
  easy repos pay zero.
- It is a **read-only SIGNAL**: it can re-plan / re-target / abort but can NEVER set `.accepted`
  (C7). Acceptance is engine-owned via `ctx.select` exactly as today.
- It is **ablatable** (`PSP_GOAL_REVIEW=0`) so the A/B can measure whether the review earns its
  agents vs. a review-OFF arm (C6 demands authored/extra agents show a win).

---

## 4. Acceptance-checkpointing: frontier-banked, partial-aware (FOUNDATION — C1)

C1 is the #1 unfixed bottleneck and the prerequisite for measuring any win. The mechanism is
**half-built and validated** (`_checkpoint_accepted` `context.py:372`; `_recover_checkpoint`
`run_ladder.py:267`; budget-aware `eval_cap` already wired `commit0_autogen.py:392`;
`validate_checkpoint.py` regression test). This design EXTENDS it for the phase model — it does
NOT rebuild it.

### 4.1 New: partial / phase-frontier checkpoint (the open gap, CONTEXT.md §2c.1)

Today only a WHOLE-SUITE accept is banked (`context.py:842`, `1344`). A per-phase pass (a strict
gold-COUNT improvement on a subset) has no checkpoint. New method:

```python
def _checkpoint_frontier(self, cand, *, phase_id: str = "") -> None:
    """Bank a strict gold-frontier GAIN (not a whole-suite accept) to disk immediately,
    so the best partial survives an outer kill / eval timeout. Atomic temp-write+replace
    (same as _checkpoint_accepted). Monotone: only overwrites if cand's gold_passed strictly
    exceeds the banked record's. NEVER sets accepted (a partial is not a solve) — it records
    the recoverable best-effort diff + its verified gold-pass count + the failing residual."""
```

Writes `<run_dir>/frontier_checkpoint.json`
`{accepted: False, gold_passed, gold_total, content_sha, banked_diff_path, residual_ids, phase_id}`.
It is the partial twin of `accepted_checkpoint.json`. Called from `_observe` (so EVERY frontier
rise across fan-out/reduce/repair banks, `context.py:494-505` is exactly where the frontier rise
is already detected) and from `reduce_residuals` (`context.py:1342`). The full-suite
`accepted_checkpoint.json` still wins outright on recovery.

### 4.2 New: wire the child driver's TimeoutExpired to consult the checkpoint

The child driver's own `TimeoutExpired` path (`commit0_driver.py:259-263`) currently returns a
bare infra non-result, ignoring any banked accept (CONTEXT.md §2c.2, Tier-1.1). Fix: before
returning, rglob `accepted_checkpoint.json` under `output_dir`; if an accepted record exists,
return a SOLVED result built from it. (`run_ladder.py` already does this on the OUTER kill at
`:470`; this closes the inner driver seam so a Mode-A-style subprocess kill is also covered.)

### 4.3 Never score a timeout/infra-kill as `solved:0` (Tier-1.3)

Any cell with `wall_s >= CELL_TIMEOUT` or a null/partial result is tagged
`status:"timeout"`/`nonresult_reason` and EXCLUDED from solve-rate denominators (not booked as a
fail). This already exists for the outer path; assert it for the inner driver path too. This is
what makes the A/B statistically honest (C5: several mimesis "non-solves" are timeout-clips, not
honest fails).

### 4.4 Stays execution-authoritative (C7)

Every checkpoint records a candidate REAL pytest accepted (or a real gold-count it measured). The
checkpoint never lets the orchestrator self-declare. The phase objective, plan-review, and ledger
are all read-only signals; only `ctx.select` (engine-owned) produces a winner. This is the
Cardinal Contract preserved verbatim.

---

## 5. Cost guard: strict budget partition (NO run-4 over-spawn/blowup — C2, C3)

Run-4 blew the wall because repair-ON + cap-16 spent compute without banking, turning a verified
jinja solve into a TIMEOUT (C2). This design makes over-spend **structurally impossible** at the
phase layer:

### 5.1 Phase budget partition

The cell's agent budget (`effective_max`, `architect.py:524`) is partitioned across phases up
front: each phase gets `agent_budget = floor(effective_max * w_i / Σw)` where `w_i` is the phase's
module count × difficulty weight. A phase that finishes under budget DONATES its remainder to a
shared **residual pool** the LAST/hardest phase draws from (bank cheap wins first, escalate on
the tail — CONTEXT.md D5). A phase can NEVER exceed its slice + the pool, so the sum across phases
≤ `effective_max`. This is enforced by passing each phase a per-phase soft cap the existing
governor honors (`should_continue_waves` already stops at the agent ceiling).

### 5.2 Budget-aware effort lever (C2)

Phases default to `effort: "low"` (the Opus-4.5 medium-effort lever — ~76% fewer output tokens,
CONTEXT.md §3, corroborated-with-caveat). Effort escalates to `"high"` ONLY for the residual-tail
phase AFTER cheaper phases have banked their frontier. This is "spend heavy compute only where
the near-solve tail actually is" — the structural answer to C2 and the AUTOGEN_WON=0 cost problem
(C6). Implementation: `effort` maps to `per_agent_timeout_seconds` and the repair `max_iters`
clamp the phase passes to the converge body.

### 5.3 The skip-gate is preserved verbatim (C3)

Easy / `<=1`-module / undecomposable → zero phases → `ctx.workflow("default-best-of-n")`. The
decomposition over-spawn that bites voluptuous/jinja never fires because decomposition (and
therefore phasing) is gated to medium/hard `>=2`-module repos exactly as `templates.py:83-88`.

### 5.4 The governor is unchanged and remains the halt authority

Each phase's loop uses `ctx.should_continue_waves()` — the SPFG+ governor (`governor.py:93`,
`frontier.py`) decides when a phase has plateaued. A climbing frontier keeps a phase going; a
true plateau cuts it. This is the run-4 budget-blowup fix and it is reused, not modified. No new
stop logic is introduced (CONTEXT.md §2c, "do not rebuild — extend").

---

## 6. Degradation to the verified best-of-N floor (C6, C7)

The whole stack degrades gracefully, at every layer:

1. `decompose` returns None / 1 module / easy → no phases → `ctx.workflow("default-best-of-n")`.
2. Phase plan empty after leveling → converge default whole-repo (current behaviour).
3. Tier-B codegen lint-fails → Tier-A (scoped converge body).
4. A phase aborts (review) → `ctx.defer` + proceed; banked partial carries forward.
5. The whole PSP orchestration crashes → `autosolve`'s existing `_floor()` runs BEST_OF_N
   directly (`architect.py:570-578`). The host-side floor-probe (`architect.py:566`) still banks
   a wave-0 candidate first for resilience.
6. No accepted winner anywhere → honest ABSTAIN (no fake pass; the existing review-fix #8
   reconciliation `architect.py:615` recovers any banked accept).

Acceptance is engine-owned at every step (C7). The planner only decides *where compute goes*; the
kernel decides *what passes* — the existing split (`context.py` docstring) is preserved exactly.

---

## 7. Precise mapping onto the ctx API

### 7.1 Existing methods reused verbatim (no change)

| Need | Existing ctx/engine seam |
|---|---|
| Module breakdown + topological order + per-module gold ids | `ctx.decompose()` → `{modules, order}` (`context.py:1155`) |
| Per-phase fan-out (no barrier) | `ctx.fanout_modules(phase.modules, carry_diff)` (`context.py:1247`) |
| Merge + score subset | `ctx.reduce_residuals(cands, carry_diff)` (`context.py:1274`) |
| Loop-until-dry on residual | `ctx.repair_residual(ids, carry_diff, round)` + `ctx.should_continue_waves()` |
| Carry best partial forward | `ctx.carry_best()` (`context.py:1097`) |
| Phase acceptance ids when collection errors | `ctx.module_gold_ids(phase.modules)` (`context.py:1117`) |
| Read-only planner & review (signals) | `ctx.ask(schema=...)` / `ctx.signals([...])` (`context.py:870`, `604`) |
| Admit-gate review verdicts | `ctx.adversarial_filter(verdicts, votes=3)` (`context.py:664`) |
| IOU on phase abort | `ctx.defer(scope, item, reason)` (`context.py:712`) |
| Engine-owned accept | `ctx.select(candidates)` (`context.py:1467`) |
| Per-phase codegen compose | `ctx.workflow({"scriptPath": ...})` (`context.py:639`, one level deep) |
| Whole-suite verify/harden | `ctx.adversarial_verify` / `ctx.completeness_critic` (`context.py:660`, `707`) |

### 7.2 NEW ctx methods / engine hooks (with signatures)

```python
# context.py — partial-acceptance checkpoint (the open C1 gap, §4.1)
def _checkpoint_frontier(self, cand, *, phase_id: str = "") -> None: ...
    # called from _observe (the frontier-rise branch, ~context.py:494) + reduce_residuals (:1342)

# context.py — score against a phase's gold SUBSET (not the whole suite), reusing the journaled
# score path. A thin wrapper so reduce/repair can be phase-bounded without forking _scored.
def reduce_residuals(self, candidates, *, carry_diff="", scope_ids: Optional[Sequence[str]] = None) -> dict: ...
    # ADD optional scope_ids: when given, "accepted" means the SUBSET is green (a phase pass);
    # the WHOLE-suite accept still drives _checkpoint_accepted. Default None == current behaviour
    # (fully back-compatible — the converge arm is unchanged).

# context.py — the phase scheduler entry the new template calls (host-side phase loop, depth-1 safe)
def run_phases(self, phases: Sequence[dict], *, goal: str = "") -> Optional[Candidate]: ...
    # executes phases in dependency order: per-phase carry-seeded fan-out -> reduce(scope_ids) ->
    # loop-until-dry(scope_ids) -> _checkpoint_frontier; runs the §3 goal-review gate between
    # phases; partitions budget (§5); returns ctx.select(all_candidates) (may abstain).

# context.py — build the ordered plan from decompose (pure, journaled, zero extra tokens beyond
# the one decompose ask + the one planner ask)
def plan_phases(self, plan: dict, *, max_phases: int) -> list[dict]: ...
    # topological-level decompose's order/depends_on into <=max_phases phases; persist phase_plan.json.

# context.py — the grounded goal-alignment review (read-only signal, §3)
def goal_review(self, *, phase: dict, goal: str, residual_ids: Sequence[str],
                excerpts: str = "", n: int = 3) -> dict: ...
    # returns {verdict, reason, retarget_ids} after adversarial_filter majority over n skeptics.

# commit0_driver.py — consult the checkpoint on the child driver's TimeoutExpired (§4.2)
#   (no new ctx method; a ~6-line edit to _run_mode_a's except branch to rglob accepted_checkpoint.json)
```

All new methods are journaled/replay-safe (they reuse `_attempt`/`ask`/`_provider`/`apply_diff`)
and NONE can set `.accepted` — the Cardinal Contract is structurally preserved (C7).

### 7.3 The new orchestration template (`PSP_ORCHESTRATION`, added to templates.py)

```python
def orchestrate(ctx):
    ctx.phase("scope")
    difficulty = str((ctx.repo_map.get("difficulty") or "")).lower()
    # C3 SKIP-GATE (verbatim): easy/<=1-module -> cheap best-of-N, pay nothing for phasing.
    plan = ctx.decompose() if difficulty != "easy" else None
    modules = (plan or {}).get("modules") or []
    if not plan or len(modules) <= 1 or difficulty == "easy":
        return ctx.workflow("default-best-of-n")
    max_phases = {"medium": 3, "hard": 5}.get(difficulty, 3)
    phases = ctx.plan_phases(plan, max_phases=max_phases)      # ordered, budget-partitioned, persisted
    goal = str(ctx.repo_map.get("task_framing") or "make the gold suite pass")
    winner = ctx.run_phases(phases, goal=goal)                 # per-phase converge + grounded review gate
    # whole-suite harden (medium/hard) — can only downgrade a cheat/incomplete, never promote (C7).
    if winner is not None and difficulty in ("medium", "hard"):
        ctx.adversarial_verify(winner, n=3)
        ctx.completeness_critic(winner)
        winner = ctx.select(ctx.all_candidates())
    return winner   # may abstain (never fake a pass)
```

Selected as a new arm via `APEX_OMEGA_ORCHESTRATION=psp` (mirror the `ralph`/`converge` selectors
in `author_orchestration` `architect.py:311-322`), frozen directly so it resumes
deterministically. `run_phases`/`plan_phases`/`goal_review` live in `context.py` (not in the
authored string) so the template stays lint-clean and short — the heavy logic is host-side Python
the lint allows, exactly like `reduce_residuals` today.

---

## 8. A/B eval plan (mimesis + babel + controls)

### 8.1 Arms

| Arm | Selector | Purpose |
|---|---|---|
| **A: converge** (control) | `APEX_OMEGA_ORCHESTRATION=converge` | the current trusted default (the bar to beat) |
| **B: psp-full** | `APEX_OMEGA_ORCHESTRATION=psp` | phases + per-phase banking + grounded review + budget partition |
| **C: psp-no-review** | `psp` + `PSP_GOAL_REVIEW=0` | isolates whether the adversarial review earns its agents (C6) |
| **D: psp-no-codegen** | `psp` + `PSP_TIER_B=0` | isolates Tier-A-only (does per-phase codegen earn its cost?) |
| **E: ralph** (floor) | `APEX_OMEGA_ORCHESTRATION=ralph` | vanilla persistence baseline (already exists) |

A vs B is the headline gate. C and D are ablations that satisfy C6 ("authored/extra agents must
earn their cost") by measuring each added mechanism against itself OFF.

### 8.2 Repos

- **mimesis** (HARD discriminator, 6044/6052 near-solve, the run-4 lost-solve repo, fetch-cheat
  canonical victim C4) — the primary target.
- **babel** (HARD discriminator, 4598/4607 near-solve) — the second target; modular,
  dependency-coupled → exactly where ordered phases should beat parallel-ignore-order.
- **jinja** (the only n=1 discriminator) — guards against regression on the repo the converge
  arm wins.
- **voluptuous** (always-solves easy) — the C3 over-spawn guard: PSP must take the skip-gate and
  cost NO more agents than converge (assert `agents_used` parity).

### 8.3 Protocol (respects C5: n=1 is statistically weak)

- **n >= 3 seeds per (arm × repo)** (Tier-2), 5 on mimesis/babel if budget allows; report solve
  rate with Wilson CIs and median `agents_used`.
- Per-cell wall = the current 24h default (`run_ladder.py:64`) so a heavy phase is never
  truncated mid-work (the run-4 guillotine).
- **Exclude timeout-clips from denominators** (§4.3): a cell with `wall_s >= CELL_TIMEOUT` or a
  null result is `status:timeout`, not a fail (C5 — several mimesis non-solves are clips).
- **Checkpoint-recovery assertion:** kill a mimesis cell at +60s after a known pass and assert
  `_recover_checkpoint` (outer) AND the new driver-TimeoutExpired path (inner) both report SOLVED.
  This is the direct C1 regression test, extending `validate_checkpoint.py`.

### 8.4 Acceptance criteria for shipping PSP (C6 is the gate)

PSP (Arm B) ships ONLY if BOTH hold:
1. **It banks a solve the converge control misses** on mimesis OR babel (a real, recovered,
   gold-verified pass — the unit of credit, C1/C6), with non-overlapping CIs at n>=3.
2. **It does NOT regress** jinja or voluptuous, and on voluptuous spends `agents_used` within
   parity of converge (the C3 guard).

If only Arm C/D (the cheaper ablations) match Arm B, ship the cheaper one — extra agents that
don't earn a win are pure cost (C2/C6).

### 8.5 Sequencing (build order)

1. **§4 (checkpointing)** first — the foundation; recovers the run-4 lost solves and is the unit
   of credit for everything else. Mostly wiring; validated by an extended `validate_checkpoint.py`.
2. **§1–2 + §5 (phase scheduler, Tier-A, budget partition)** — the core capability.
3. **§3 (grounded review gate)** — the no-veer mechanism.
4. **§2 Tier-B (per-phase codegen)** — last, gated, measured by Arm D.

Gate the whole stack behind `APEX_OMEGA_ORCHESTRATION=psp` + per-mechanism env flags so the A/B
is apples-to-apples and each layer is independently ablatable.

---

## 9. Risk register

- **R1 — phase scoring drift:** scoring a SUBSET (`scope_ids`) could bank a "phase pass" that a
  later phase silently breaks. *Mitigation:* the WHOLE-suite accept (engine-owned `ctx.select`)
  remains the only winner; subset banking is a *recoverable partial*, never a winner. The final
  `ctx.select` always re-scores the merged whole-tree. C7 holds.
- **R2 — over-planning a deceptively-modular repo:** *Mitigation:* phase cap by difficulty (§1.2)
  + skip-gate (§5.3); the review's `abort` verdict collapses a bad plan back toward the converge
  whole-repo path.
- **R3 — review agents become pure cost (C6):** *Mitigation:* Arm C (review-OFF) measures it; gate
  to medium/hard; admit-gate via majority so it can't stall.
- **R4 — budget partition starves a genuinely-hard early phase:** *Mitigation:* the residual pool
  (§5.1) lets the hard tail draw donated budget; the SPFG+ governor protects a still-progressing
  phase from premature cut (it never cuts a rising frontier).
- **R5 — Tier-B per-phase codegen regresses (run-4 was a codegen regression):** *Mitigation:*
  Tier-B is gated, lint-checked, and falls back to Tier-A; Arm D measures it; it never runs on
  easy repos.
- **R6 — fetch-monoculture drift on mimesis (C4):** *Mitigation:* the worktree-shadow makes a
  fetch-cheat structurally unable to false-solve; the grounded review (§3) flags an attempt whose
  residual isn't shrinking (a fetch-cheat shows zero gold progress) and re-targets it.

---

## 10. Why this is the SINGLE best design (summary justification)

- It is **Claude-Code-shaped**: ordered phases with explicit objectives + boundaries (delegation
  contracts), per-phase plans, durable plan-to-disk, a verifier-gated progression loop — the
  corroborated multi-agent-research + `/goal` + Plan Mode shape (CONTEXT.md §3).
- It **builds on seams that already exist** (`decompose`/`fanout_modules`/`reduce_residuals`/
  `repair_residual`/`carry_best`/`ask`/`adversarial_filter`/`defer`/`should_continue_waves` +
  the half-built checkpoint), so it is implementable in days, not weeks, and reuses the trusted
  converge body — the lowest-risk path to the user's goal.
- It is **provably stronger than `/goal`**: the no-veer gate is grounded in real pytest residuals,
  closing `/goal`'s blind-evaluator hole (CONTEXT.md §2c.5, §3).
- It **respects every empirical constraint**: C1 (partial+full checkpoint, survives outer kill +
  eval timeout), C2 (budget partition + effort lever, never blows the wall), C3 (skip-gate
  verbatim, phase cap by difficulty), C4 (review re-targets a non-shrinking residual), C5 (n>=3
  with CIs, timeout-clips excluded), C6 (ablation arms C/D gate whether extra agents earn it),
  C7 (acceptance engine-owned at every layer, abstain over fake pass).
- It targets the **actual killer** on the hard discriminators — the residual near-solve tail —
  by ordering dependency-coupled work and banking each phase's frontier instead of abstaining on
  the whole suite.
</content>
</invoke>
