# Diagnose-Directed Re-Aim — Cadence Research (does earlier help?)
Source: workflow wf_2cc15d54-824 (9 agents). DESIGN-ONLY. Question: is diagnose-directed re-aim only useful at near-solve, or does applying it earlier help the code-gen orchestrator reach the goal faster/cheaper?

## ANSWER: GRADUATED CADENCE
Split into two ingredients with OPPOSITE optimal cadences (the codebase fuses both behind APEX_OMEGA_SARP, off by default):
- **EXCERPTS (the ~free WHY = pytest assertion tails): thread EARLIER, EVERYWHERE a repair runs.** Produced at 0 tokens by every reduce (context.py:2415/2432) but DISCARDED by the base loop — the 4 repair sites (templates.py:139,:314; context.py:1883,:1892) call repair_residual WITHOUT excerpts=, so every base-loop repair wave (BOTH live arms) re-rolls near-misses blind to WHY. Fix = pass the kwarg (data already in `red`). +0 agents.
- **SCOUT diagnosis (diagnose_residual, N=2 read-only fan-out): near-solve-only, but frontier-height-aware** — fire on ANY sterile plateau (not just terminal rescue), bounded by the SARP rung budget (value-of-information gated). Early = bulk-implement (obvious 'write the module'), scouts add noise.

## EVIDENCE (archived runs/diag_ab_s0; the live run was the collision-broken one)
verdict_signal: near_solve_only
- Frontier is FAST-RISE-THEN-PLATEAU: mimesis-diag frontier_history [[1,592],[2,4618],[3,5913],[4,5930],[7,5935],[8,6110]]/6159 — 75% by measurement 2.
- Budget LOST on PLATEAU CHURN at near-solve, NOT unimplemented code: 60-84% of dispatched agents produced NO valid measurement (sterile re-attempts/dup diffs); mimesis-diag 101 agents/16 valid meas, ~8-12 sterile waves after the last rise before the cut.
- FAST-RISE-THEN-PLATEAU, decisively. The authoritative data is the archived diag_ab_s0 run (the live /tmp/omega_sarp_ab A/B is NOT usable: every cell is at preflight — 2-line WALs, only "froze orchestration script" in narration.jsonl; mimesis s1 crashed with FileNotFoundError on journal/calls_wal.jsonl). frontier_history is [valid_measurement_idx, gold_count] (apex_omega/engine/frontier.py:13-17,216-221).

mimesis hybrid-diag (runs/diag_ab_s0/hybrid-diag__mimesis__s0/autogen_cell_report.json): [[1,592],[2,4618],[3,5913],[4,5930],[7,5935],[8,6110]]/6159. Meas2 already at 75% of final frontier; plateaus at 6110/6159 = 99.20%, residual 49 tests, then cut:sterile-diff-streak after 101 agents.

ba
- regime: Two distinct regimes, splitting cleanly by wave-phase within each cell:

(1) EARLY waves = BULK-IMPLEMENT, huge residual, obvious direction. Meas 1-3 climb from ~5-10% to 75% (mimesis-diag 592→4618→5913) or to ~99% (babel-diag by meas7). Residual is "implement the whole module set" — phase_plan.json enumerates files_owned/modules per phase (Core Foundations, Providers And Builtins, Schema And Plugins). Direction is the obvious "write minifier.py / providers/*.py"; failure EXCERPTS add little because nothing is implemented yet to diagnose. Direction-finding value here is LOW.

(2) LATE waves = 

## DESIGN
excerpts_policy: Thread red['failure_excerpts'] (already returned at context.py:2432) as excerpts= into EVERY repair_residual call on the base loop, plus the fan-out residual-repair brief — i.e. exactly the 4 currently-blind sites: templates.py:139 (DEFAULT_ORCHESTRATION loop-until-dry), templates.py:314 (CONVERGE_EXEMPLAR), context.py:1883 and :1892 (run_phase hybrid path). The data is sitting in the loop variable red on every iteration; the only change is passing the kwarg, so CONTRACT 2's evidence block (commit0_autogen.py:493) renders the assertion tails instead of nothing. Do NOT add excerpts to: decompose/plan_phases (no execution yet) or fan-out wave 0 / CONTRACT 1 solve briefs (first implementation pass, no failures to diagnose — commit0_autogen.py:466 correctly has no param). Rationale: excerpts are ~0 token-cost, are the proven active ingredient (sarp-last-mile-fix.md:35-36), and every wave from the 2nd on has a real residual whose WHY is free; gating them to rescue-only is the current defect that makes the whole base loop re-roll blind. Decouple this from APEX_OMEGA_SARP — excerpts should ride the default loop, not the gated scout path.

scout_policy: Spend scout direction-finding (diagnose_residual, the N=2 read-only fan-out) ONLY on a sterile round (red['advanced']==False at a non-trivial frontier) — the existing _sarp_observe trigger (context.py:2702-2708) — NOT every wave. Make the gate FRONTIER-HEIGHT-AWARE rather than terminal-only: (1) fire on ANY intermediate plateau, not just pre-abstain — sarp_step already lives in the loop (templates.py:159, context.py:1920); the fix is to let _sarp_holds defer the soft cut mid-run (governor.py:126-128,:143-144 already do, gated on sarp_enabled), so a plateau at meas 7 gets a scout, not just the final one; (2) escalate cadence as the gap shrinks: scale the sterile patience inversely with residual size so a 8-test residual (babel-diag) earns a scout after 1-2 sterile waves while a 970-test residual (babel-nogate) waits longer (its direction is 'more code', not a subtle cause). HARD BOUNDS, reuse as-is: APEX_OMEGA_SARP_TOTAL_RUNG_BUDGET=12 per run (context.py:2716), APEX_OMEGA_SARP_RUNGS_PER_EPISODE=3 (:2712), MAX_DISTINCT_RESIDUALS=4 thrash cap (:2721), per-sha diagnosis cache (:2598), SARP_TARGETED_MAX=2/sha (:2770), and the anti-hallucination fact-check that downgrades ungrounded coupling/unsolvable to semantic_logic_bug (:2642). Diagnosis is STEERING only — never an accept; only ctx.select on the full gold suite accepts (LLM-Modulo discipline, context.py:2819-2820).

cost_model: EXCERPTS-EVERYWHERE: +0 agents, +~3KB prompt/repair-turn (excerpts hard-capped at [:3000], commit0_autogen.py:493; diagnose at context.py:2612). Net FASTER: the wasted-budget is plateau churn, not unimplemented code — in the 3 near-solve cells 60-84% of dispatched agents produced no frontier rise (mimesis-diag 101 agents/16 valid meas; ~8-12 sterile waves after the last rise at meas 8 before cut at sterile_streak=8, governor.py:125-128). Each of those blind re-rolls re-passes the same ~6100 tests and re-fails the same ~49 because it never saw the assertion tail. Converting even a fraction of those sterile waves to closes is pure savings — the agent was already being dispatched.

SCOUT DIAGNOSIS: +N agents/episode (APEX_OMEGA_SARP_DIAG_N=2, via ctx.signals = NOT counted toward the plateau, context.py:2589) hard-bounded by APEX_OMEGA_SARP_TOTAL_RUNG_BUDGET=12 (context.py:2716) and MAX_DISTINCT_RESIDUALS=4 (:2721), cached per residual-sha (:2598) so an unchanged residual is never re-diagnosed. Worst case +~12 diagnosis fan-outs/run = ~24 read-only agent-asks, vs the 47-101 agents a single cell already spends. It nets cheaper ONLY if a diagnosed direction closes the run before the agent ceiling; on a genuinely-unimplemented bulk plateau (babel-nogate) it is near-break-even (the rung budget caps the downside). Where it does NOT pay: a large diffuse residual with no dominant cluster — re-decompose is correctly gated to len(clusters)==1 (context.py:2757) so it can't over-spawn there.

experiment: 2x2 factorial, hard coupled repos mimesis + babel (the near-solve and the bulk-plateau exemplars), n>=3 seeds, fair 24h/cell wall. FACTOR 1 EXCERPTS: OFF (today, repair_residual blind) vs EVERYWHERE (thread excerpts= at templates.py:139,:314 + context.py:1883,:1892). FACTOR 2 SCOUT: OFF (APEX_OMEGA_SARP=0) vs RESCUE-ONLY (sarp_rescue at abstain only) vs GRADUATED (frontier-height-aware in-loop, height-scaled sterile patience). Arms: A excerpts-OFF/scout-OFF (control = current default); B excerpts-EVERYWHERE/scout-OFF (isolates the free part); C excerpts-EVERYWHERE/scout-RESCUE (= today's SARP intent); D excerpts-EVERYWHERE/scout-GRADUATED (the proposal). PRIMARY METRIC: agents-to-frontier-99% and agents-to-last-frontier-rise (from cut_losses.frontier_history, e.g. mimesis-diag last rose at meas 8/agent~ — measure if B/D shrink the post-rise sterile tail). SECONDARY: solve-rate + best_gold_passed + total agents_used. FALSIFIABLE PREDICTIONS: (i) B beats A on agents-to-frontier with equal-or-better best (excerpts are free and convert sterile waves) — if B==A, excerpts-everywhere is inert, reject the cheap-part claim; (ii) D beats C on near-solve cells (mid-run plateau scouts close earlier) but D~=C on babel-nogate bulk plateau — if D burns its rung budget with no extra closes vs C, the 'any plateau' escalation is over-spend, fall back to rescue-only. GATED KNOBS to vary: APEX_OMEGA_SARP, APEX_OMEGA_SARP_TOTAL_RUNG_BUDGET (12 vs 6 vs 18), APEX_OMEGA_SARP_FLOOR_FRAC (0.50 height gate), APEX_OMEGA_SARP_DIAG_N (2 vs 1). PRECONDITION: fix the live harness (the /tmp/omega_sarp_ab cells are at agents=1 with FileNotFoundError on journal/calls_wal.jsonl — runs unusable) before trusting any A/B; only diag_ab_s0 is currently authoritative.

## REVIEW (folded)
- Cost bounds, VOI gating, and unbounded-diagnosis risk of the graduated-cadence scout design => sound
- Early-phase value of direction-finding (scout diagnosis) vs confirm-excerpts; separating real signal from theater => concern
  - Decouple and ship the excerpts fix on its own, exactly as scoped: thread red['failure_excerpts'] as excerpts= into the 4 blind sites (templates.py:139, :314; context.py:1883, :1892). This is the high-confidence, ~0-cost win. Do NOT bundle it with the scout changes -- it should ride the default loop 
  - Stop claiming scout diagnosis is empirically validated. The only evidence offered (babel diag>nogate) is confounded: the residual scout never fired (SARP off in both arms), winners are the integrator (a812xxx), 3 gates are bundled, n=1. Re-state scout value as a HYPOTHESIS pending the SARP A/B, not 
  - Separate the two diagnosis components in the writeup and in the cost model: (1) zero-token AST/collection pre-pass at plan time (APEX_OMEGA_DIAG) -- cheap, defensible early; (2) the N=2 residual scout (diagnose_residual, SARP) -- costly, near-solve-only. Do not present them as one 'direction-finding
  - Drop the over-claim that 'every repair wave re-rolls blind.' repair_attempt/solve_and_repair already inject redacted excerpts by default (context.py:1449, APEX_OMEGA_REPAIR_EXCERPTS=1). Scope the blindness claim to the loop-until-dry/run_phase residual-repair seams only.
  - Acknowledge that the 'redefine near-solve / fire on intermediate plateaus' behavior already exists (_sarp_frontier_nontrivial ratio floor 0.50, context.py:2524; governor _sarp_holds defers cut mid-run). Limit the new scout proposal to the genuinely-novel piece: inverse-patience scaling by residual s
  - HONEST MINIMAL VERSION: (a) excerpts-everywhere on the 4 residual-repair seams (ship now, decoupled); (b) keep the zero-token AST pre-pass at plan time if it is cheap; (c) hold the paid residual scout behind SARP, fire it ONLY on a sterile near-solve tail, and require it to BEAT the excerpts-only lo
- Invariant compliance: ENV-gating/ablatability, journal-replay determinism, Cardinal-safety of the graduated_cadence design => concern
  - Put excerpts-everywhere behind its own ablatable env flag (e.g. APEX_OMEGA_REPAIR_EXCERPTS, default '0' = off) rather than decoupling it from all gating. OFF must reproduce today's behavior: the 4 default-path repair_residual calls (templates.py:139,:314; context.py:1883,:1892) pass excerpts only wh
  - Before excerpts enter ANY prompt that becomes a journal key, exclude the volatile span from the key. Either wrap the assertion-tail in VOLATILE_OPEN/VOLATILE_CLOSE inside residual_repair_brief (commit0_autogen.py:493) and the built-in residual brief (context.py:2939) so canonicalize_prompt's _VOLATI
  - Make the height-aware sterile-patience gate read frontier height ONLY from the journaled frontier (_best_gold_passed / red['gold_total'] at context.py:2696), never from a live measurement, and route the deferral exclusively through governor._sarp_holds (governor.py:93-104) so the continue/halt verdi

## SEAM MAP (where excerpts/scouts apply today)
- **build_repo_map -> author prompt (decompose/scout input)**: excerpts=no — the author/scout sees only the static repo map (files/m | scouts=partial — the zero-token AST collection pre-pass i | cost=~0 (AST pre-pass is zero-token, pure sta
- **decompose() — convergence wave 0 scoping agent**: excerpts=no — runs before any execution; prompt carries only the sour | scouts=no — decompose is itself ONE read-only ask (not a  | cost=1 agent (single ctx.ask, fixed id 700100
- **diagnose() — STAGE-2 first-blocker classifier (import/collec**: excerpts=n/a — it grounds on the STATIC AST pre-pass (must_implement/ | scouts=yes — N read-only scouts via ctx.signals classify  | cost=N agents per call (default n=2, APEX_OME
- **review_plan() — advisory plan review at decompose/phase-plan**: excerpts=no — skeptics get the diagnosis (blocker_class/must_implemen | scouts=yes — it IS a scout fan-out (N skeptics), but it c | cost=N agents per seam (default n=2, ctx.sign
- **plan_phases() — group modules into ordered phases**: excerpts=no — planner prompt carries module briefs + topo order + ove | scouts=no — ONE read-only planner ask (id 700200), not a  | cost=1 agent (single ctx.ask)
- **goal_align_gate() — pre/post per-phase no-veer guard**: excerpts=no — N skeptics get the REAL residual failing node-id LIST ( | scouts=partial — it is an adversarial scout fan-out, but  | cost=N agents per gate (medium n=1, hard n=3,
- **fanout_modules() -> solve_module() — per-module solve briefs**: excerpts=no — and correctly so: this is the FIRST implementation pass | scouts=no — pure solve fan-out (one agent per module via  | cost=1 solve agent per module (id_base+index)
- **reduce_residuals() — merge + ONE full-suite score (the excer**: excerpts=yes (PRODUCES them) — zero tokens. The merged-tree score yie | scouts=no — plain-Python merge + score, no LLM (context.p | cost=0 agents (pure Python apply+score, zero 
- **loop-until-dry -> repair_residual() (the BLIND seam)**: excerpts=no — THE GAP. red['failure_excerpts'] is sitting right there | scouts=no (unless SARP) — the bare loop runs no diagnosis | cost=1 repair agent per round. Threading exce
- **run_phase() inner loop -> repair_residual() (hybrid path, sa**: excerpts=no — same defect as templates: reduce_residuals(scope_ids=.. | scouts=no in the base loop; sarp_step at :1920 adds it on | cost=1 repair agent per round; excerpt thread
- **coupled_plateau() -> ralph_loop integrator (coupled-repo fin**: excerpts=no — integrator_brief gets the residual gold-id LIST (contex | scouts=no — the detector is pure-Python signal latching ( | cost=0 for the detector; the integrator is 1 
- **sarp_step() / _sarp_observe() in-loop adaptation (gated APEX**: excerpts=yes — SARP is the ONLY in-loop seam that passes excerpts=. I | scouts=yes — diagnose_residual() = N read-only scouts cla | cost=RUNG-0 diagnosis = N agents/episode (APE
- **sarp_rescue() — explicit post-loop last-mile (governor-timin**: excerpts=yes — sources excerpts from _sarp_last['excerpts'] or best.m | scouts=yes — same diagnose_residual scout fan-out per rou | cost=N diagnosis scouts + 1 repair agent per 
- **general repair_attempt() (solve_and_repair lineage; best-of-**: excerpts=partial — it injects excerpts but ONLY after redact_excerpts | scouts=no — Reflexion-style self-repair on parent diff +  | cost=1 repair agent per lineage iteration (ma
- **decompose (architect plan)**: excerpts=no — decompose runs at t=0 before any pytest, so there are n | scouts=partial — gated by APEX_OMEGA_DIAG via ctx.diagnos | cost=diagnose() = nn scouts, default n=2 (con
- **plan_phases / plan_review (phase planner)**: excerpts=no — review_plan takes an optional diagnosis dict, not failu | scouts=yes — review_plan fans out n=2 read-only scouts (c | cost=2 agents/call via ctx.signals; runs per 
- **fanout_modules (per-module solve)**: excerpts=no — first fan-out is pre-residual; later SARP re-decompose  | scouts=no — pure solve, no diagnosis at this seam (diagno | cost=1 solve agent per module (context.py:228
- **solve_module brief**: excerpts=no — initial solve brief has no failures yet | scouts=no | cost=0 direction-finding agents (it is the so
- **repair_residual / loop-until-dry (inner repair round)**: excerpts=yes — excerpts threaded into the prompt at context.py:2939/2 | scouts=no — repair_residual itself runs no scout; it just | cost=excerpts plumbing = ~0 agents (free). Th
- **reduce_residuals (the REDUCE / scoring step)**: excerpts=yes — it PRODUCES failure_excerpts from vr.failure_excerpts  | scouts=no — and it must not (zero-LLM, replay-determinist | cost=0 agents. The excerpts are a byproduct o
- **repair_residual loop diagnosis (SARP in-loop sarp_step -> _s**: excerpts=yes — pulls red['failure_excerpts'] (context.py:2724) into d | scouts=yes — diagnose_residual fans out SARP_DIAG_N scout | cost=per RUNG: diagnose = N scouts (default A
- **coupled_plateau integrator rung (SARP rung c)**: excerpts=yes — residual+excerpts feed the diagnosis that selects this | scouts=yes (the same diagnose_residual that routed here) | cost=1 rung (counts vs SARP_TOTAL_RUNG_BUDGET
- **near-solve sarp_rescue (explicit post-loop last-mile)**: excerpts=yes — last.get('excerpts')/bm['failure_excerpts'] (context.p | scouts=yes — diagnose_residual per round (context.py:2854 | cost=while-loop bounded by SARP_TOTAL_RUNG_BU
- **collection diagnose() (pre-solve blocker classification)**: excerpts=no — pre-execution; uses AST static evidence (ast_must/ext/a | scouts=yes — nn scouts (default n=2, context.py:1973) via | cost=2 agents per RUN (cached once/cell, cont

## FULL RECOMMENDATION
All frontier_history data is confirmed from the authoritative run (runs/diag_ab_s0). The numbers in the design are accurate:
- mimesis-diag: last rise meas 8 → 6110/6159, 101 agents
- babel-diag: last rise meas 7 → 5655/5663, 47 agents
- babel-nogate: bulk plateau [[1,4689],[12,4693]] = 4693/5663 (82.87%), 58 agents — diag BEAT nogate (5655 vs 4693)
- mimesis-nogate: last rise meas 9 → 6146/6159, 91 agents

The live `/tmp/omega_sarp_ab` confirmed broken (agents=1, the design's note). All seams grounded. Writing the final recommendation.

---

# Diagnose-Directed Re-Aim: Final Recommendation

## 1. The answer

**Graduated cadence — not near-solve-only, not earlier-everywhere.** The question conflates two ingredients that the codebase fuses (both gated behind `APEX_OMEGA_SARP`, off by default) but that have **opposite** optimal cadences: cheap **failure excerpts** (the ~free WHY) should run *earlier and everywhere a repair runs*; the costly **scout diagnosis** (the N=2 read-only direction-find) should fire *only on a sterile plateau, on a bounded, frontier-height-aware cadence* — which includes intermediate plateaus, not just the terminal rescue.

## 2. The cheap-vs-expensive split

### (A) EXCERPTS — thread them everywhere a repair runs, now. Decouple from SARP.
The assertion tails are computed at **zero extra cost** by every reduce (`context.py:2415` `out_excerpts = getattr(vr,"failure_excerpts","")`) and banked **unconditionally** into `result["failure_excerpts"]` (`context.py:2432`, *not* behind the `_sarp_on()` gate at `:2453`). They are the proven active ingredient (sterile 2/3 → excerpts+re-aim → 3/3, per sarp-last-mile-fix.md; the failing-id-LIST-only loop stalled, per ADAPTIVE_REPLANNING_DESIGN.md).

But the base loop is **blind**: the 4 repair sites call `repair_residual(...)` with **no `excerpts=`** kwarg, so it defaults to `""`. Confirmed `repair_residual` signature: `excerpts: str = ""` (`context.py:2903`). When the harness wired the CONTRACT-2 builder (`brief_builders["residual_repair"]`, `context.py:2932`), an empty kwarg makes its evidence block render nothing (`commit0_autogen.py:493`, hard-capped `[:3000]`). Only the 3 SARP sites pass excerpts (`context.py:2788, 2799, 2887`). **Net effect today: every base-loop repair wave in both live arms re-rolls near-misses blind to WHY.** Cost to fix: +0 agents, +~3KB/repair-turn. Net *faster* (see §cost).

### (B) SCOUT DIAGNOSIS — bounded, graduated, plateau-only.
`diagnose_residual` (`context.py:2584`, RUNG 0, read-only, `APEX_OMEGA_SARP_DIAG_N=2` at `:2602`) is `near_solve_only` by the run data — **but "near-solve" must mean "any sterile plateau at a non-trivial frontier," not only the terminal one.** Authoritative data (runs/diag_ab_s0, frontier_history confirmed from matrix_report.json; the live `/tmp/omega_sarp_ab` is broken — agents=1):

| cell | best | % | last frontier rise | agents | cut |
|---|---|---|---|---|---|
| mimesis-diag | 6110/6159 | 99.2% | meas 8 (of 16) | 101 | sterile-diff-streak |
| babel-diag | 5655/5663 | 99.9% | meas 7 (of 19) | 47 | sterile-diff-streak |
| mimesis-nogate | 6146/6159 | 99.8% | meas 9 | 91 | sterile-diff-streak |
| babel-nogate | 4693/5663 | **82.9%** | meas 12 ([[1,4689],[12,4693]]) | 58 | sterile-diff-streak |

All 4 cut by `cut:sterile-diff-streak` — the exact soft cut `_sarp_holds` defers (`governor.py:125-128`). Early waves (5%→75%) have obvious direction; excerpts suffice, scouts add little. Scouts pay off **as the gap shrinks AND on the bulk→near-solve transition**: on the one bulk plateau (babel-nogate), **diag BEAT nogate (5655 vs 4693)** — diagnosis helped cross the bulk→near-solve barrier. So gate scouts on *any* sterile plateau, with patience that **escalates as the frontier rises** (diagnose sooner when 8 tests remain than when 970 do).

## 3. Exact seams + env gates to change — smallest first

1. **Excerpts pass-through (the whole win, ~4 one-line kwarg adds).** Thread the already-in-hand `red["failure_excerpts"]` (a.k.a. `red.get(...)` in the loop) as `excerpts=` into the 4 blind sites:
   - `templates.py:139` (DEFAULT_ORCHESTRATION loop-until-dry)
   - `templates.py:314` (CONVERGE_EXEMPLAR)
   - `context.py:1883` and `context.py:1892` (run_phase hybrid path — note the loop var here is `red`, residual at `:1891`)
   
   Do **not** add excerpts to decompose/plan_phases or fan-out wave-0 / CONTRACT-1 solve briefs (`commit0_autogen.py:466` correctly has no param — no execution yet, nothing to diagnose). This change must ride the default loop, **independent of `APEX_OMEGA_SARP`**.

2. **Frontier-height-aware scout cadence (the graduated lever).** `sterile_streak_cut` is currently a fixed `8` (governor.py constant). Make the SARP-deferred patience scale inversely with residual size, so an 8-test residual earns a scout after 1-2 sterile waves while a 970-test residual waits longer. This rides the **existing** machinery: `_sarp_observe` already fires only on a sterile round at a non-trivial frontier (`context.py:2702-2708`), and `_sarp_holds` already defers the soft cut mid-run (`governor.py:126, 143, 150`), so a plateau at meas 7 gets a scout, not just the final one. Keep all hard bounds as-is.

3. **Keep the rescue ladder unchanged.** `sarp_rescue` (`context.py:2820+`) already threads excerpts (`:2852, 2887`); leave it.

## 4. The falsifiable experiment (run after the current SARP A/B)

- **Arms (3):** `excerpts-only` (fix §3.1, scouts OFF) | `graduated` (fix §3.1 + §3.2) | `nogate` baseline (current default, both off). This isolates the cheap ingredient from the costly one — the current diag-vs-nogate A/B cannot.
- **Repos:** mimesis, babel (both near-solve plateaus in-hand), + 1 bulk-plateau repo and pydantic (a from-zero repo) to test the bulk→near-solve claim and confirm no regression where excerpts can't help.
- **Seeds:** n≥3.
- **Primary metric:** sterile-wave→close conversion = (frontier rises after the last pre-fix plateau) and **agents-to-frontier-ceiling** (the 60-84% post-rise sterile churn is the waste to kill). **Secondary:** final gold-pass %, solved count, total scout agents/run.
- **Pre-registered prediction:** `excerpts-only` ≥ `nogate` on near-solve cells at equal/lower agent cost; `graduated` ≥ `excerpts-only` specifically on the bulk-plateau cell (the babel-nogate-style barrier) where direction-finding pays.

## 5. Risks + what would falsify

- **Excerpts add prompt noise without lift.** Falsifier: `excerpts-only` does **not** beat `nogate` on near-solve cells. Mitigant: cost is +0 agents and the lift is already shown in the design's repro; low risk.
- **Scouts over-spend / thrash.** Bounded by construction — confirmed: total distinct diagnoses = distinct residual-shas (per-sha cache `context.py:2598-2599`, written `:2668`), capped by `MAX_DISTINCT_RESIDUALS=4` in-loop (`context.py:2721`) and the per-run `APEX_OMEGA_SARP_TOTAL_RUNG_BUDGET=12` (`:2716`, single never-decremented counter). Worst case ~12 shas × DIAG_N=2 = ~24 read-only asks vs the 47-101 agents a cell already spends. Firing "earlier" only changes *when* a sha is first diagnosed, never *how many* run.
- **"Fire earlier" blows the budget.** Cannot: `_sarp_holds` returns True only with rung+budget headroom (`governor.py:100-104`), and the **agent ceiling is an independent inviolable backstop** — `governor.verdict` reaches `stop:agent-ceiling` with **no** SARP override (`governor.py:158-159`), `max_total_agents=1000` (`runtime.py`).
- **Diagnosis steers wrong (hallucinated direction).** Already mitigated: re-decompose gated to `len(clusters)==1` (`context.py:2757`), targeted cap 2/sha (`:2770`), fact-check downgrades ungrounded coupling/unsolvable to `semantic_logic_bug` (`:2642`). Diagnosis is **steering only** — only `ctx.select` on the full gold suite accepts (LLM-Modulo). Falsifier: `graduated` regresses vs `excerpts-only` on near-solve cells (scouts steering into worse residuals).
- **Overall falsifier for the recommendation:** if `excerpts-only` already captures ~all the lift and `graduated` adds none on the bulk-plateau cell, collapse to "excerpts everywhere, scouts rescue-only" (the cheap half alone).

**Relevant files:** `/Users/sameertkhanna/Documents/agent_orch/apex_omega/autogen/context.py`, `/Users/sameertkhanna/Documents/agent_orch/apex_omega/autogen/templates.py`, `/Users/sameertkhanna/Documents/agent_orch/apex_omega/engine/governor.py`, `/Users/sameertkhanna/Documents/agent_orch/apex_omega/engine/runtime.py`, `/Users/sameertkhanna/Documents/agent_orch/apex_omega/eval/commit0_autogen.py`; data: `/Users/sameertkhanna/Documents/agent_orch/runs/diag_ab_s0/` (authoritative), `/tmp/omega_sarp_ab/` (broken, agents=1 — do not use).