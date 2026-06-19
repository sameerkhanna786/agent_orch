# Design: Agentless-Structured Deterministic Pipeline for APEX-Ω

**Name:** `agentless-structured`
**One-liner:** Minimize agent autonomy — a deterministic, host-owned LOCALIZE → PLAN → REPAIR(sampled) → VALIDATE pipeline (one phase per dependency-ordered module), spending compute on SAMPLING at execution-filtered stages, with an adversarial goal-alignment gate, instant acceptance-checkpointing, and a guaranteed verified best-of-N floor.

---

## 0. The thesis (why Agentless, mapped to this engine)

Agentless (Xia, Deng, Dunn, Zhang, 2024, *"Agentless: Demystifying LLM-based Software Engineering Agents"*, **arXiv:2407.01489**) is the canonical evidence that a **fixed deterministic pipeline with no autonomous control flow** can beat autonomous agents on SWE-bench at a fraction of the cost. Its published results (verified against the paper; cited as the design's empirical anchor):

- **SWE-bench Lite:** v1 = **27.33% at $0.34/instance**; v2 = **32.00% (96/300) at ~$0.70/instance** — the highest performance AND lowest cost among open-source approaches at publication, vs agent-based approaches costing up to **~$3.34/instance** (e.g. the paper's quoted $0.34 vs $3.34 contrast — a **~10×** cost gap at higher solve rate). On **SWE-bench Verified, Agentless reaches >50% with Claude 3.5 Sonnet**. Subsequently adopted by OpenAI (GPT-4o/o1 showcase) and DeepSeek (V3/R1 eval) as the go-to non-agentic scaffold — strong external corroboration of the cost/structure thesis.
- The pipeline is **three deterministic phases**: (1) **hierarchical localization** (repo tree → suspicious files → suspicious classes/functions → edit locations), (2) **repair by SAMPLING multiple candidate patches** at a fixed temperature, (3) **patch validation** by regression + reproduction tests, then **majority/normalized selection** among the patches that pass.
- The two load-bearing findings that transfer directly:
  1. **No agentic autonomy is needed** — the LLM is called as a *stateless function* at each stage; the host owns all control flow. This removes the "vague delegation / over-spawn / doom-loop" failure class entirely (the same class APEX-Ω fights with the SPFG+ governor and the decompose skip-gate).
  2. **Spend compute on SAMPLING + EXECUTION FILTERING, not on more autonomy.** Multiple patches are generated per location; tests filter them; selection picks among *validated* survivors. Compute that is not behind an execution filter is waste.

These two findings are a near-perfect fit for APEX-Ω's constraints C1–C7 (CONTEXT.md §4): execution-authoritative accept (C7), compute-must-be-banked (C1/C2), earn-your-cost (C6). The novelty here is *not* "be Agentless" — it is **fusing Agentless's deterministic localize→sample→validate spine with APEX-Ω's per-phase dependency ordering, instant checkpointing, and an adversarial goal gate**, so the engine gets Agentless's cost discipline AND a Claude-Code-style phased plan AND the no-veer guarantee.

**Crucial honesty about transfer:** Agentless validates patches against *reproduction tests it writes itself* plus visible regression tests, because real SWE-bench hides the gold suite. APEX-Ω is **strictly stronger** here: acceptance is the real gold suite via `ctx.select` (execution-authoritative). So we keep Agentless's *sampling-and-filtering structure* but replace its self-written-test validation with APEX-Ω's gold-grounded `reduce_residuals` / `select`. This is the single most important adaptation and it is what makes the design safe under C7.

---

## 1. Where this lands vs the current converge default

The current `DEFAULT_ORCHESTRATION` (templates.py:71-158) is already a decompose→fan-out→reduce→loop-until-dry→verify shape. The gaps (CONTEXT.md §1b) this design closes:

| Gap (CONTEXT.md §1b) | Agentless-structured fix |
|---|---|
| #1 no ordered phases w/ objectives + per-phase acceptance | Host-side phase loop over `decompose().order`; each phase has `acceptance_gold_ids`; partial acceptance banked. |
| #2 authoring is one-shot whole-repo | No LLM author at all in this arm — the pipeline IS the orchestration (deterministic host code). Agent autonomy minimized to stateless stage calls. |
| #3 no adversarial goal-alignment gate | `goal_gate()` per phase boundary, grounded in real residual node-ids. |
| #4 decompose ignores `order` | Phases executed in topological `order` (dependency-respecting). |
| #5 no per-phase partial acceptance | New `_checkpoint_partial()` + `phase_locked` ledger. |
| #6 convergence shape hard-coded | Still structured (that's the point — Agentless says structure beats autonomy) but now PHASED + sampled, not single-pass. |

It does **not** tear down converge — it is a *new selectable arm* (`APEX_OMEGA_ORCHESTRATION=agentless`) frozen directly like `ralph`/`converge` (architect.py:311-322), A/B'd against converge. The converge default and the best-of-N floor remain the fall-open path.

---

## 2. The pipeline (phases/chunks with objectives, like Claude Code)

Five host-owned stages. Stages 2–4 run **per phase**, where a *phase* = one module in dependency `order`. The whole thing is one `orchestrate(ctx)` (frozen as a template string), but the LLM is only ever called as a stateless stage function — there is no agent that "decides what to do next."

```
LOCALIZE  (stage 1, once)   — deterministic + read-only ask: hierarchical localization
   |        decompose() -> modules{module, gold_test_ids, depends_on, files, order}
   |        => the PHASE LIST (one phase per module, topologically ordered)
   v
for each phase P in order:                                    # Claude-Code-style chunking
   PLAN     (stage 2)  — read-only: objective + edit-locations + boundaries for P
   GOAL-GATE (stage 3) — adversarial: "does P's plan still serve goal G given residual R?"
   GENERATE (stage 4)  — SAMPLE N module-scoped patches for P (execution-scored), carry-seeded
   VALIDATE (stage 5)  — reduce_residuals: merge + full-gold-score ONCE; bank partial accept
   CHECKPOINT          — if phase acceptance ids now green -> lock phase, checkpoint, advance
   |  (governor decides whether to keep sampling this phase or move on)
   v
GLOBAL VALIDATE+REPAIR — loop_until_dry on the exact residual ids across all phases
   v
HARDEN (medium/hard)  — adversarial_verify + completeness_critic, re-select
   v
SELECT (engine-owned) — ctx.select over all banked candidates (may ABSTAIN; never fakes)
```

### Stage 1 — LOCALIZE (objective: turn the repo into an ordered phase list)
- `plan = ctx.decompose()` (context.py:1155) returns `modules` + `order`. This *is* Agentless's hierarchical localization, already read-only and schema-validated. We additionally consume the `order` field (currently discarded — gap #4) and the optional `files` field (gap, delegation contract D4 in CONTEXT.md).
- **Cost guard (C3):** the easy-repo skip-gate from converge is preserved verbatim — `if difficulty == "easy" or not plan or len(modules) <= 1: return ctx.workflow("default-best-of-n")`. Easy/single-module repos never enter the phased pipeline. This is the run-4 over-spawn guard (CONTEXT.md C3) and it is mandatory.

### Stage 2 — PLAN (objective: per-phase edit locations + boundaries)
- Read-only `ctx.ask` with a `PHASE_PLAN_SCHEMA` → `{objective, edit_locations:[file:symbol], boundaries, acceptance_gold_ids}`. `acceptance_gold_ids` defaults to the module's `gold_test_ids` from decompose (so PLAN cannot *invent* acceptance — it can only restate/refine the gold subset). This mirrors Agentless's class/function localization step.
- This is a SIGNAL (`ctx.ask`), never a Candidate — it cannot set acceptance (C7).

### Stage 3 — GOAL-GATE (adversarial, no-veer)
- New `ctx.goal_gate(phase, residual_ids)` (see §4) runs N skeptics via `ctx.signals` asking: *"Given goal G (make the full gold suite pass) and the EXACT current residual failing node-ids R, does this phase plan still serve G? verdict ∈ {proceed, revise, abort_phase} + reason."* Admit-gated via existing `adversarial_filter` (context.py:664).
- **Grounded in reality, not transcript** (the `/goal` blind-spot fix, CONTEXT.md §3): the reason MUST cite residual node-ids from `reduce_residuals`. The gate can re-target the PLAN (feed the reason back into Stage 2's prompt) or skip a now-redundant phase (its ids already green) — but it can NEVER set `.accepted` (C7).
- **Cost guard (C6):** gated to medium/hard only (mirror templates.py:153). Easy repos already exited at Stage 1; medium repos with few phases get a cheap 3-skeptic vote; hard repos get up to 5. Ablation flag `APEX_OMEGA_GOAL_GATE=0` turns it fully off so the gate's marginal value is measurable.

### Stage 4 — GENERATE (the SAMPLING stage — where compute goes)
- This is the Agentless "spend compute on sampling at a validated stage" core. For phase P, sample `k_P` module-scoped patches via `ctx.fanout_modules([P]*k_P, carry_diff=carry)` — i.e. **N independent samples of the SAME module**, each seeded with the running carry diff, each scored on the full gold suite by execution. (`solve_module` already exists, context.py:1202; fanout uses the no-barrier pipeline.)
- `k_P` is **complexity-matched** (C2/C3, external scaling heuristic): `k_P = sample_budget(difficulty, |gold_test_ids|)` → easy-tail module 1, medium 2-3, hard 4-5. Implemented via `ctx.plan_waves(start=...)`-bounded counts, never a flat large N. **This is the explicit answer to the run-4 budget-blowup (C2): samples are bounded per phase AND only spent while the SPFG+ frontier is still rising.**

### Stage 5 — VALIDATE (execution filter + bank)
- `red = ctx.reduce_residuals(samples, carry_diff=carry)` (context.py:1274): zero-token plain-Python merge + ONE full-gold score. This is Agentless's regression+reproduction filter, but grounded in the *real* gold suite (stronger than self-written tests).
- **Selection among validated survivors** = the Agentless majority/normalized step, but execution-authoritative: `ctx.judge_select` only breaks execution-equal ties; `ctx.select` is the sole producer of an accept (C7).
- If `red["accepted"]` → solved, return. Else `carry = red["merged_diff"]` (carry the best partial forward — closes the off-by-K near-solve class, CONTEXT.md). Conflicts → `ctx.defer` (re-solved clean next round; progress never erased).

### CHECKPOINT + advance
- After Stage 5, compute the phase's acceptance: are `phase.acceptance_gold_ids` ⊆ (gold ids green in the merged tree)? If yes → `ctx.checkpoint_partial(red["candidate"], phase)` (see §4), mark the phase `locked` in the host-side `phase_ledger`, and advance. The governor (`ctx.should_continue_waves()`) decides whether to keep sampling this phase or move on — a rising per-phase frontier keeps sampling; a plateau cuts to the next phase (never the whole cell).

### Global validate+repair, harden, select
- After all phases: `loop_until_dry` repairing the union residual on the live merged tree (reuse converge Stage 3 verbatim, templates.py:128-147). Then medium/hard `adversarial_verify` + `completeness_critic`. Then engine-owned `ctx.select` — **may abstain (C7), never fakes a pass.**

---

## 3. Acceptance-checkpointing (C1 — the #1 bottleneck)

Two-tier banking, both surviving an outer subprocess kill and a budget-aware eval timeout:

1. **Whole-suite accept (already built):** `_checkpoint_accepted` (context.py:372) writes `accepted_checkpoint.json` atomically the instant any candidate is `.accepted`. Called from `_attempt` (context.py:842) and `reduce_residuals` (context.py:1344). `run_ladder._recover_checkpoint` (run_ladder.py:267) recovers it on an outer kill. **No change needed — reuse.**

2. **Per-PHASE partial accept (NEW — closes gap #5 / CONTEXT.md §2c.1):** when a phase's `acceptance_gold_ids` go green (a strict gold-pass-COUNT improvement) but the whole suite isn't, bank a *phase checkpoint* so a kill mid-pipeline doesn't lose locked phases. New `ctx.checkpoint_partial(cand, phase)` writes `phase_checkpoint.json` = `{best_gold_passed, best_merged_diff_sha, locked_phases:[...], pass_rate}` atomically + idempotently-monotone (only overwrites on a strict gold-count rise). On resume, the host phase loop reads it and skips already-locked phases (their gold ids are re-validated cheaply, never re-solved). This is the durable plan + residuals ledger (CONTEXT.md D2; Claude-Code `~/.claude/plans` + Pokemon NOTES.md analogue).

3. **Survive the budget-aware eval timeout (Tier-1.2, already wired):** `commit0_autogen.py:392` already computes `eval_cap = max(300, min(1800, cell_timeout//3))` and passes `timeout_seconds=eval_cap` to `evaluate_repo` (commit0_autogen.py:422-431). A scoring timeout maps to `indeterminate` → neutral to the frontier (safe). **No change needed — confirm it stays wired for this arm (it routes through the same score_fn).**

4. **Never score a timeout as solved:0 (Tier-1.3):** unchanged — the driver's `TimeoutExpired` path (commit0_driver.py:259) returns an infra non-result; combined with checkpoint recovery this becomes a `done`+solved if a phase/whole checkpoint exists. The design adds ONE wiring assertion: the driver's `TimeoutExpired` branch must consult **both** `accepted_checkpoint.json` AND `phase_checkpoint.json` before returning the infra non-result (the whole-suite one wins if present; the phase one is recorded as a partial for the ledger, not a solve).

**Acceptance stays engine-owned (C7):** `checkpoint_partial` records *what real pytest measured* (gold ids green in a merged tree the engine scored). It never sets `.accepted`; it is a durability + resume-skip record only. A phase being "locked" never makes the cell "solved" — only `ctx.select` over an `.accepted` candidate does.

---

## 4. Mapping onto the ctx API (existing + NEW)

### Reused verbatim (no change)
- `ctx.decompose()` → phase list + `order` (now consumed). `ctx.carry_best()`, `ctx.fanout_modules`, `ctx.solve_module`, `ctx.reduce_residuals`, `ctx.repair_residual`, `ctx.module_gold_ids`, `ctx.modules_overlap`, `ctx.loop_until_dry`, `ctx.should_continue_waves`, `ctx.select`, `ctx.judge_select`, `ctx.adversarial_verify`, `ctx.adversarial_filter`, `ctx.completeness_critic`, `ctx.ask`, `ctx.signals`, `ctx.defer`, `ctx.plan_waves`, `ctx.repo_map`, `ctx.budget`, `ctx.agents_used`, `ctx.max_agents`, `ctx.phase`, `ctx.log`. `_checkpoint_accepted` (context.py:372).

### NEW ctx methods (thin host-side wrappers; all read-only or execution-scored; none can accept)

```python
# context.py — three new methods on OrchestrationContext.

def plan_phase(self, module: dict, *, residual_ids=None, vendor=None, model=None,
               agent_id: int = 740000) -> dict:
    """READ-ONLY per-phase plan (Agentless class/function localization step). Returns
    {objective, edit_locations:[str], boundaries:str, acceptance_gold_ids:[str]} validated
    against PHASE_PLAN_SCHEMA. acceptance_gold_ids is CLAMPED to the module's gold_test_ids
    (the model may restate/order them but cannot invent acceptance). Pure SIGNAL via ctx.ask;
    NEVER a Candidate. Fail-open: schema-miss -> {objective:"", edit_locations:[],
    acceptance_gold_ids: module['gold_test_ids']}."""
    # builds an ask() prompt seeded with module + residual_ids; schema=PHASE_PLAN_SCHEMA;
    # intersect returned acceptance ids with set(module['gold_test_ids']) (clamp).

def goal_gate(self, phase_plan: dict, residual_ids, *, votes: int = 3,
              vendor=None, model=None, id_base: int = 750000) -> dict:
    """ADVERSARIAL goal-alignment gate (no-veer). N read-only skeptics vote, each given goal G
    ('make the full gold suite pass'), the phase objective, and the EXACT residual failing
    node-ids R. Returns {verdict: 'proceed'|'revise'|'abort_phase', reason:str, votes:[...]}.
    Grounded in R (real test output), not the transcript -> closes /goal's blind spot. Uses
    ctx.signals (no plateau accounting) + adversarial_filter to admit only reasons that cite a
    real residual id. A SIGNAL only: can re-target/skip a phase, can NEVER set acceptance (C7).
    Fail-open: any infra failure -> {verdict:'proceed'} (never blocks progress on a flaky vote)."""

def checkpoint_partial(self, cand, phase: dict) -> None:
    """Bank a PER-PHASE partial accept (gap #5 / CONTEXT.md §2c.1). Atomically + monotonically
    (strict gold-count rise only) writes phase_checkpoint.json = {best_gold_passed,
    merged_diff_sha, pass_rate, locked_phases:[module names], repo}. Survives an outer kill so a
    resumed run skips already-green phases. Records what REAL pytest measured; NEVER sets
    .accepted (durability/resume-skip only). Best-effort, never fatal (mirror _checkpoint_accepted
    context.py:372-392)."""
```

`PHASE_PLAN_SCHEMA` and `GOAL_GATE_SCHEMA` added next to `DECOMPOSE_SCHEMA` (context.py:94). `plan_phase`/`goal_gate` are exposed in `API_REFERENCE` (architect.py:38) so a future author arm can use them too, but **this arm is template-frozen, not authored.**

### NEW engine/harness hook (one assertion, not new capability)
- `commit0_driver._run_mode_a` `TimeoutExpired` branch (commit0_driver.py:259) and `run_ladder._recover_checkpoint` (run_ladder.py:267): extend the rglob to also match `phase_checkpoint.json`. `_recover_checkpoint` returns the whole-suite accept if present (a real solve); otherwise it returns the phase record tagged `partial:true` so the reclassifier books it as a near-solve (excluded from solve numerator, recorded for the frontier ledger) — never as solved, never as a clean fail. ~15 LOC, no new model capability.

### The frozen template (templates.py — new `AGENTLESS_ORCHESTRATION`)
```python
def orchestrate(ctx):
    ctx.phase("localize")
    difficulty = str(ctx.repo_map.get("difficulty") or "").lower()
    plan = ctx.decompose() if difficulty != "easy" else None
    modules = (plan or {}).get("modules") or []
    if not plan or len(modules) <= 1 or difficulty == "easy":
        return ctx.workflow("default-best-of-n")          # C3 over-spawn guard (mandatory)
    order = (plan.get("order") or [m["module"] for m in modules])
    by_name = {m["module"]: m for m in modules}
    carry = ctx.carry_best()
    red = {"accepted": False, "residual_failing_ids": [], "merged_diff": carry}
    gate_on = difficulty in ("medium", "hard")
    for name in order:                                     # PHASES in dependency order
        if not ctx.should_continue_waves() or red["accepted"]:
            break
        m = by_name.get(name)
        if m is None:
            continue
        ctx.phase("phase:" + str(name))
        residual = red["residual_failing_ids"]
        pp = ctx.plan_phase(m, residual_ids=residual)     # Agentless localize-to-functions
        if gate_on:
            g = ctx.goal_gate(pp, residual)               # no-veer adversarial gate
            if g["verdict"] == "abort_phase":
                ctx.defer("goal_gate", name, g["reason"]); continue
            if g["verdict"] == "revise":
                pp = ctx.plan_phase(m, residual_ids=residual, vendor=None)  # one re-plan, grounded
        # GENERATE: SAMPLE k module-scoped patches (compute at a validated stage).
        k = max(1, min(len(pp.get("acceptance_gold_ids") or m.get("gold_test_ids") or [1]),
                       {"medium": 3, "hard": 5}.get(difficulty, 2)))
        samples = ctx.fanout_modules([m] * k, carry_diff=carry)
        # VALIDATE: merge + ONE full-gold score; bank.
        red = ctx.reduce_residuals(samples, carry_diff=carry)
        if red["accepted"]:
            return red["candidate"]
        if red["merged_diff"]:
            carry = red["merged_diff"]
        # phase partial acceptance -> checkpoint + lock (survives outer kill).
        acc = set(pp.get("acceptance_gold_ids") or m.get("gold_test_ids") or [])
        still_failing = set(red["residual_failing_ids"])
        if acc and acc.isdisjoint(still_failing):
            ctx.checkpoint_partial(red["candidate"], m)
    # GLOBAL validate + repair on the exact residual (reuse converge loop-until-dry).
    ctx.phase("converge")
    rnd = 0
    while ctx.should_continue_waves() and not red["accepted"]:
        targets = red["residual_failing_ids"] or ctx.module_gold_ids(modules)
        c = ctx.repair_residual(targets, carry_diff=carry, round=rnd); rnd += 1
        red = ctx.reduce_residuals([c], carry_diff=carry)
        if red["accepted"]:
            return red["candidate"]
        if red["merged_diff"]:
            carry = red["merged_diff"]
    # HARDEN + engine-owned SELECT (may abstain; never fakes).
    winner = ctx.select(ctx.all_candidates())
    if winner is not None and difficulty in ("medium", "hard"):
        ctx.adversarial_verify(winner, n=3); ctx.completeness_critic(winner)
        winner = ctx.select(ctx.all_candidates())
    return winner
```
Frozen directly in `author_orchestration` (architect.py:320-322 pattern): add `if _orch_selector == "agentless": return _freeze(engine, AGENTLESS_ORCHESTRATION, "agentless", ...)`.

---

## 5. Cost-safety: how this avoids the run-4 over-spawn/budget-blowup regression (C2/C3/C6)

Explicit guards, each tied to a named lesson:

1. **Easy-repo skip-gate (C3):** Stage 1 returns `ctx.workflow("default-best-of-n")` for easy/≤1-module repos. voluptuous/jinja never enter the phased pipeline — zero over-spawn. (Verbatim from converge templates.py:83-88.)
2. **Per-phase bounded sampling (C2):** `k_P ∈ {2,3,5}` capped, NOT a flat large N. Total samples ≤ `Σ k_P` ≤ `max_agents`; `plan_waves`/`should_continue_waves` enforce the ceiling. The run-4 blowup was repair-ON + cap 8→16 spending unconditionally; here sampling is *behind the SPFG+ frontier* (`should_continue_waves` cuts a plateaued phase, advances to the next — never the whole cell).
3. **Adversarial gate gated to medium/hard (C6):** easy already exited; goal_gate costs 0 there. Medium gets 3 votes, hard 5. Ablation flag isolates its cost.
4. **Reduce is zero-token (C2):** `reduce_residuals` is plain-Python merge + ONE pytest run per round — the validation filter is cheap; only generation samples cost agents.
5. **Bank-first (C1):** every win (whole or phase) is checkpointed the instant it's measured, so the 80%-token-variance/compute-only-helps-if-banked finding is honored — no run-4-style discarded solve.
6. **Earn-your-cost gate (C6):** this arm is accepted ONLY if it banks a solve converge misses on mimesis/babel (§7 acceptance criteria). If it merely matches converge at higher agent cost, it is rejected.

---

## 6. Degrade-to-floor + acceptance engine-owned (C6/C7)

- **Floor:** unchanged. The host-side floor-probe (architect.py:557-578) banks a verified best-of-N wave-0; on any crash/lint-reject the arm falls open to `BEST_OF_N_ORCHESTRATION` via `_floor()`. The phased pipeline can only *add* to the banked set; it can never do worse than the floor.
- **Abstain, never fake:** the final `ctx.select` may return None (honest abstain) — preserved verbatim. No phase lock, plan, gate verdict, or checkpoint sets `.accepted`. Acceptance is the gold suite via `select_best` (context.py:1467-1469).
- **Stateless-function discipline (Agentless core):** every LLM call is read-only (`ask`/`signals`/`decompose`/`plan_phase`/`goal_gate`) OR an execution-scored attempt (`solve_module`/`repair_residual`). No agent owns control flow → no autonomy to veer, over-spawn, or doom-loop. This is the structural reason the arm is safe.

---

## 7. A/B eval plan (mimesis + babel discriminators; voluptuous/jinja controls)

**Arms (apples-to-apples, same K, same seeds):**
- **A (control):** `APEX_OMEGA_ORCHESTRATION=converge` — the current frozen default.
- **B (treatment):** `APEX_OMEGA_ORCHESTRATION=agentless` — this design.
- **B′ (ablation, gate off):** `agentless` + `APEX_OMEGA_GOAL_GATE=0` — isolates the adversarial gate's marginal value (C6 earn-your-cost).
- (optional) **B″ (ablation, sampling=1):** `k_P=1` everywhere — isolates the sampling stage's value vs phasing alone.

**Repos:** mimesis + babel (HARD discriminators, near-solve tails 6044/6052 and 4598/4607 — CONTEXT.md C5) are the win-gate. voluptuous (always-solves) + jinja (the one historical discriminator) are CONTROLS — B must NOT regress them (must hit the skip-gate / match converge).

**Protocol (CONTEXT.md C5, statistical):**
- n ≥ 3–5 seeds per (arm × repo) cell; report solve-rate with bootstrap CIs.
- **Exclude timeout-clips from denominators** (Tier-1.3): any cell with `wall_s ≥ CELL_TIMEOUT` or a null/partial result → `status:timeout`, excluded with a `nonresult_reason`. (This is mandatory or the matrix lies — CONTEXT.md C5/§2c.4.)
- Run via `scripts/run_ladder.py` with `REPOS=mimesis,babel,voluptuous,jinja`, `ARMS=converge,agentless,agentless-nogate`, `CELL_TIMEOUT=86400` (the 24h fair wall, already default run_ladder.py:64), checkpoint recovery ON.

**Metrics per cell:** solved (banked verified pass), best_gold_passed (frontier reached — the near-solve credit), agents_used (cost), tokens (cost), phase_checkpoints written (durability evidence), goal_gate verdicts (no-veer telemetry), wall_s, nonresult_reason.

**Acceptance criteria (C6 — earn-your-cost):**
1. **WIN:** B banks ≥1 verified solve on mimesis OR babel that A misses, across the seed set, with overlapping-or-better cost CI. This is the primary gate.
2. **NO-HARM:** B's solve-rate on voluptuous/jinja ≥ A's (must not regress controls).
3. **FRONTIER:** even without a full solve, B's median `best_gold_passed` on babel/mimesis > A's (banks more of the near-solve tail — the checkpoint_partial payoff).
4. **COST:** B's agents_used per solve ≤ A's (Agentless's cost thesis must hold; if B solves at higher cost it is rejected per C6 even if it solves more).
5. **GATE VALUE:** B beats B′ on a no-veer-sensitive metric (fewer fetch-monoculture/duplicate-work attempts; lower wasted-agent count) — else the gate is dropped as pure cost.

---

## 8. Implementation checkpointing (build order)

1. **D1 foundation (½–1d):** add `checkpoint_partial` (context.py) + `phase_checkpoint.json` recovery in `_recover_checkpoint`/driver `TimeoutExpired` (run_ladder.py:267, commit0_driver.py:259). Regression-test like `validate_checkpoint.py`. *(Recovers near-solve durability; prerequisite for measuring §7 #3.)*
2. **Phase loop + sampling (1d):** add `AGENTLESS_ORCHESTRATION` (templates.py) + `plan_phase` + `PHASE_PLAN_SCHEMA` (context.py); wire the `agentless` selector (architect.py:320). Unit-test the phase loop on a stub decompose plan (no LLM): assert dependency-order execution, skip-gate on easy, partial-checkpoint on a phase whose ids go green.
3. **Goal gate (½d):** add `goal_gate` + `GOAL_GATE_SCHEMA` + `APEX_OMEGA_GOAL_GATE` flag. Unit-test fail-open (infra → proceed) and grounding (a reason not citing a residual id is filtered out by `adversarial_filter`).
4. **A/B (eval, Workflow 2):** run §7 on the 4 repos × 3 arms × n≥3 seeds; reclassify timeout-clips; report.

**Adversarial-review-of-the-design checkpoint (no-veer on the design itself):** before eval, a skeptic pass must confirm (a) no path sets `.accepted` outside `select` (grep `cand.accepted =` / `.accepted = True` → must be 0 in new code), (b) the easy skip-gate is unconditional, (c) `k_P` is bounded by `max_agents`, (d) `goal_gate` is fail-open. These are the C7/C2/C3 invariants; violating any one rejects the build before it burns eval budget.

---

## 9. Risks

- **R1 — decompose quality is the ceiling.** If `decompose()` returns a poor module split / wrong `order`, the phasing helps nothing. Mitigation: fail-open to converge's parallel fan-out if `order` is degenerate (single module / empty deps); the global loop-until-dry recovers cross-module residuals regardless.
- **R2 — sampling N may still over-spend on a genuinely-hard phase.** Mitigation: `should_continue_waves` cuts a plateaued phase and advances; total bounded by `max_agents`; B″ ablation measures whether sampling earns its cost.
- **R3 — goal_gate becomes pure cost (C6).** Mitigation: B′ ablation; drop the gate if it doesn't beat B′ on a no-veer metric.
- **R4 — phase partial-acceptance is a near-solve, not a solve.** It must NEVER count toward solve-rate (C7). Mitigation: `_recover_checkpoint` tags it `partial:true`; the reclassifier excludes partials from the numerator. Enforced in the §8 adversarial-review checkpoint.
- **R5 — Agentless's cost win may not transfer** (their numbers are SWE-bench Lite single-file bug-fixes; commit0 is whole-repo completion). Mitigation: this is exactly what §7 measures; the design degrades to the verified floor either way, so a non-transfer is a clean negative result, not a regression.

---

## 10. Citations (Agentless cost/score evidence)

- **Agentless** (Xia, Deng, Dunn, Zhang, 2024, **arXiv:2407.01489**; FSE 2025 / PACMSE): three-phase deterministic LOCALIZE→REPAIR(sampled)→VALIDATE pipeline. **SWE-bench Lite: v1 27.33% @ $0.34/inst, v2 32.00% @ ~$0.70/inst** (highest perf + lowest cost among open-source at publication); **>50% on SWE-bench Verified with Claude 3.5 Sonnet**; agent-based baselines quoted up to **~$3.34/inst** (~10× costlier). Adopted by OpenAI (GPT-4o/o1) and DeepSeek (V3/R1) as the go-to non-agentic eval scaffold. Core findings used: (1) no agentic autonomy required (LLM as stateless stage function); (2) spend compute on SAMPLING multiple patches behind an EXECUTION filter, then select among validated survivors. *(External, independent — NOT Anthropic; transferable STRUCTURE, with the cost win validated empirically in §7 since commit0 ≠ SWE-bench.)* Source: https://arxiv.org/abs/2407.01489
- **APEX-Ω empirical anchors** (CONTEXT.md §4 / APEX_COMMIT0_REPORT.md): C1 (checkpointing #1 bottleneck), C2 (compute≠solves if it blows the wall), C3 (decompose over-spawn on easy), C6 (earn-your-cost), C7 (engine-owned accept). All five are honored by design (§3, §5, §6).
- **`/goal` blind-spot** (CONTEXT.md §3, code.claude.com/docs/goal): transcript-only verifier "runs nothing" → APEX-Ω's execution-grounded accept is strictly stronger; goal_gate is grounded in real residual node-ids (§4).
```
