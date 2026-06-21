"""The default orchestration script.

Serves two roles: (1) the fail-open fallback when authoring is off/fails/lints
bad — guaranteeing the verified best-of-N floor; (2) the exemplar handed to the
architect so generated strategies inherit the invariants (completion-first,
fewest-agents-first, deterministic attempt ids, execution-authoritative select).
"""

from __future__ import annotations

# NOTE: attempt ids are assigned at thunk-CREATION (base + j), not at call time,
# so journal keys are a pure function of the logical attempt index (deterministic
# under concurrency -> faithful replay), never of thread-scheduling order.

# The CHEAP escalating best-of-N + test-driven repair path. This was the OLD default; it is
# preserved verbatim as a named workflow because the convergence default falls THROUGH to it for
# easy / single-module repos (the SKIP-DECOMPOSITION gate that prevents the 5-6x over-spawn cost
# pathology on voluptuous/jinja), and ctx.workflow("default-best-of-n") still resolves to it.
BEST_OF_N_ORCHESTRATION = '''
def orchestrate(ctx):
    """Completion-first, fewest-agents-first verified best-of-N with TEST-DRIVEN
    REPAIR lineages and escalation.

    Each unit of work is a repair LINEAGE (implement -> run the real tests -> read
    the failing output -> fix -> repeat), not a one-shot rollout: a genuine-but-
    incomplete attempt earns additional work instead of being discarded. Start with
    the fewest agents; widen only while unsolved and within budget / the agent
    ceiling. A winner is returned ONLY when execution evidence accepts it
    (ctx.select); otherwise we abstain rather than ship an unverified patch.
    """
    ctx.phase("autosolve")
    candidates = []
    base = 0
    for wave, k in enumerate(ctx.plan_waves()):
        if not ctx.budget.can_start():
            ctx.log("budget exhausted; stopping escalation")
            break
        thunks = [ctx.make_repairing_attempt(base + j) for j in range(k)]
        base = base + k
        batch = ctx.parallel(thunks)
        candidates = candidates + [c for c in batch if c is not None]
        winner = ctx.select(candidates)
        if winner is not None:
            ctx.log("SOLVED at wave " + str(wave) + " using " + str(ctx.agents_used()) + " agents")
            return winner
        ctx.log("wave " + str(wave) + " unsolved; " + str(ctx.agents_used()) + " agents used so far")
        if ctx.agents_used() >= ctx.max_agents:
            ctx.log("agent ceiling reached; stopping")
            break
    return ctx.select(candidates)  # may abstain (never fake a pass)
'''


# THE DEFAULT: the dynamic-workflow CONVERGENCE shape (guide §4 + §4.3) =
#   DECOMPOSE -> FAN-OUT -> REDUCE -> LOOP-UNTIL-DRY -> VERIFY, with a progress-based stop.
# Layered on the faithful per-agent codex-exec loop + the execution-authoritative accept gate +
# the SPFG+ governor. The missing CONVERGENCE STRUCTURE the old flat best-of-N lacked: it sprayed
# whole-repo rollouts and ABSTAINED on near-solves (babel 4598/4607, mimesis 6044/6052 thrown
# away). This default CLOSES the off-by-K near-solve class by carrying the running best partial
# diff forward into every fresh worktree and iterating on the EXACT residual failing node-ids on
# the live merged tree.
#
# COST-SAFETY (load-bearing): decomposition OVER-SPAWN bites exactly the cost-pathology repos
# (voluptuous/jinja). So the default GATES decomposition to medium/hard repos with >=2 modules;
# easy / <=1-module repos SKIP decomposition and stay on the cheap escalating best-of-N+repair
# path (preserving those wins, NO over-spawn). repair_iters=2 is SAFE only because the SPFG+
# governor stops a TRUE plateau (no valid-measurement improvement + no frontier rise) while
# letting a climbing frontier keep going (the run-4 budget-blowup fix); the loop terminates on
# accept / K-dry / governor cut / budget. Acceptance stays engine-owned: ctx.select may ABSTAIN
# (never fake a pass).
DEFAULT_ORCHESTRATION = '''
def orchestrate(ctx):
    """Dynamic-workflow convergence: decompose -> fan-out per module -> reduce ->
    loop-until-dry on the exact residual failing tests (carrying the best partial
    diff forward) -> verify. Easy/single-module repos skip decomposition and fall
    through to the cheap escalating best-of-N + repair path (no over-spawn)."""
    ctx.phase("scope")
    # (0) DECOMPOSE. SKIP-GATE: only decompose a medium/hard repo into >=2 INDEPENDENT modules.
    # An easy / undecomposable / single-module repo stays on the cheap best-of-N path so the
    # decomposition over-spawn never bites the cost-pathology repos (voluptuous/jinja).
    difficulty = str((ctx.repo_map.get("difficulty") or "")).lower()
    plan = None
    if difficulty != "easy":
        plan = ctx.decompose()
    modules = (plan or {}).get("modules") or []
    if not plan or len(modules) <= 1 or difficulty == "easy":
        ctx.log("skip-decomposition (easy/<=1 module); falling through to best-of-N + repair")
        return ctx.workflow("default-best-of-n")

    # (1) FAN-OUT per module (no barrier), each agent seeded with the running carry diff.
    ctx.phase("fanout")
    carry = ctx.carry_best()
    cands = ctx.fanout_modules(modules, carry_diff=carry)

    # (2) REDUCE: merge the per-module diffs into ONE tree, run the FULL gold suite once.
    ctx.phase("reduce")
    red = ctx.reduce_residuals(cands, carry_diff=carry)
    if red["accepted"]:
        ctx.log("SOLVED after fan-out + reduce")
        return red["candidate"]
    if red["merged_diff"]:
        carry = red["merged_diff"]          # carry the best merged partial forward
    # a conflicting module is logged (telemetry); its banked candidate is consumed below by the
    # collapse SELECT / the final ctx.select over all_candidates -- never silently orphaned.
    if red["conflicts"]:
        for name in red["conflicts"]:
            ctx.defer("merge_conflict", name, "module diff conflicted on the merge tree")

    # (2b) NO-SILENT-LOSS COLLAPSE FALLBACK. When the fan-out emitted COMPETING WHOLE-REPO
    # candidates (not disjoint slices), the textual merge collapses: most/all module diffs conflict
    # AND the merged tree makes zero gold progress. A textual 3-way merge of overlapping FULL
    # artifacts is exactly what the dynamic-workflow paradigm forbids -> SELECT among the competing
    # fulls, then fall back to the cheap whole-repo best-of-N verify-and-accept path (the path the
    # flat default solves by). Gated to a GENUINE collapse (zero gold progress) so a climbing
    # partial still enters loop-until-dry below.
    n = len([c for c in cands if c is not None])
    majority_conflict = len(red["conflicts"]) >= max(2, (n + 1) // 2)
    collapsed = int(red.get("gold_passed", 0) or 0) == 0 and (
        not (red["merged_diff"] or "").strip() or majority_conflict)
    if collapsed:
        ctx.log("fan-out collapsed (competing whole-repo candidates; overlap="
                + str(ctx.modules_overlap(cands)) + "); SELECT then best-of-N fallback")
        w = ctx.judge_select(ctx.all_candidates())
        if w is not None:
            return w
        return ctx.workflow("default-best-of-n")

    # (3) LOOP-UNTIL-DRY on the live merged tree until the merge ACCEPTS or the governor cuts.
    # The guard is `not red["accepted"]` (NOT `and residual`): a merged tree that ERRORS at
    # COLLECTION has an EMPTY failing_nodeids yet is unsolved, so keying the loop on residual ran
    # ZERO recovery rounds and discarded the whole fan-out (the jinja abstain). When residual is
    # empty we repair against the union of module gold ids. should_continue_waves() stays the halt
    # authority: a true plateau / agent ceiling / budget stops it (honest abstain, never fake pass).
    ctx.phase("loop-until-dry")
    residual = red["residual_failing_ids"]
    rnd = 0
    while ctx.should_continue_waves() and not red["accepted"]:
        targets = residual or ctx.module_gold_ids(modules)
        c = ctx.repair_residual(targets, carry_diff=carry, round=rnd)
        rnd = rnd + 1
        red = ctx.reduce_residuals([c], carry_diff=carry)
        if red["accepted"]:
            ctx.log("SOLVED in loop-until-dry round " + str(rnd))
            return red["candidate"]
        # (3b) COUPLED-REPO ROUTER (merge-reduce v2): when the textual merge keeps SHEDDING on a
        # coupled plateau (overlapping modules reject ~50 hunks each, frontier flat >0 — the babel
        # 937 trap the gold==0 collapse fallback never catches), abandon the lossy per-module merge
        # loop for a COHERENT INTEGRATOR lineage seeded by the best coherent tree (carry_best), which
        # reconciles the overlapping modules in ONE tree (ralph-on-the-carry, carry-LAST, governed).
        if ctx.coupled_plateau(red, cands):
            ctx.log("coupled plateau (overlapping modules shedding hunks) -> coherent integrator")
            w = ctx.ralph_loop(id_base=812000, seed_carry=ctx.carry_best(),
                               brief=ctx.integrator_brief(modules, residual))
            if w is not None:
                return w
        if red["merged_diff"]:
            carry = red["merged_diff"]
        residual = red["residual_failing_ids"]

    # (4) VERIFY/HARDEN (medium/hard only): skeptics try to REFUTE the leader; a completeness
    # critic flags gaps. These can only DOWNGRADE a cheat/incomplete pass, never promote one.
    ctx.phase("verify")
    winner = ctx.select(ctx.all_candidates())
    if winner is not None and difficulty in ("medium", "hard"):
        ctx.adversarial_verify(winner, n=3)
        ctx.completeness_critic(winner)
        winner = ctx.select(ctx.all_candidates())   # re-rank after any downgrade
    return winner  # may abstain (never fake a pass)
'''


# The RALPH-WIGGUM baseline workflow: a vanilla CLI in a dumb iterate-until-done loop.
# Frozen directly (never authored, never scouted) when APEX_OMEGA_ORCHESTRATION=ralph. It
# is a CONTROL arm — naive single-lineage persistence with test feedback, governed by the
# same cut-losses detector as omega — to measure how a "vanilla" CLI with a large budget
# fares against the orchestrated arms.
RALPH_ORCHESTRATION = '''
def orchestrate(ctx):
    """Ralph-wiggum baseline: ONE sequential lineage, fed the failing tests each turn,
    iterate-until-done. No scout, no author, no patterns, no parallel waves — pure naive
    persistence. The governor cut-losses detector decides when persistence has stopped
    making progress and the loss is cut."""
    ctx.phase("ralph")
    return ctx.ralph_loop()
'''


# Codebase-audit blueprint (dynamic-workflows Pattern B, guide §4.4): fan out a read-only
# audit across the repo's modules, adversarially FILTER the findings to drop false positives,
# synthesize the survivors into one brief, then SOLVE from it with escalating verified waves.
# Unlike the guide's report-producing audit, this ends in an EXECUTION-SCORED winner (so it is
# valid both as a standalone orchestration AND when composed via ctx.workflow("audit")).
AUDIT_ORCHESTRATION = '''
def orchestrate(ctx):
    """Codebase-audit: discover -> fan-out findings -> adversarial_filter -> synthesize -> solve."""
    ctx.phase("audit-discover")
    modules = (ctx.repo_map.get("modules") or [])[:24]
    targets = modules or ["the repository"]
    finding_schema = {"type": "object", "additionalProperties": True, "required": ["finding"],
                      "properties": {"finding": {"type": "string"}, "file": {"type": "string"}}}
    # 1) FAN OUT one read-only finding per module (ctx.signals: no plateau accounting).
    ctx.phase("audit-findings")
    asks = []
    for i, m in enumerate(targets):
        asks.append(lambda m=m, i=i: ctx.ask(
            "Audit the module '" + str(m) + "' for what is MISSING or incorrect to make the "
            "visible test suite pass. Return JSON {finding, file}.",
            schema=finding_schema, agent_id=970000 + i,
            label="audit:" + str(m), phase="audit-findings"))
    finds = ctx.signals(asks)
    findings = [f.get("finding") for f in finds if isinstance(f, dict) and f.get("finding")]
    # 2) Adversarially FILTER (drop false positives; admit only survivors).
    ctx.phase("audit-verify")
    kept = ctx.adversarial_filter(findings, votes=3)
    brief = None
    if kept:
        brief = ("Implement the repository task. A codebase audit found these concrete gaps "
                 "(address each):\\n- " + "\\n- ".join([str(x) for x in kept[:30]]))
    # 3) SOLVE from the synthesized brief with escalating verified best-of-N waves.
    ctx.phase("audit-solve")
    cands = []
    base = 0
    for wave, k in enumerate(ctx.plan_waves()):
        if not ctx.budget.can_start() or ctx.agents_used() >= ctx.max_agents:
            break
        thunks = [(lambda j=base + j: ctx.solve_attempt(attempt_id=j, prompt=brief,
                  phase="audit-solve")) for j in range(k)]
        base = base + k
        cands = cands + [c for c in ctx.parallel(thunks) if c is not None]
        w = ctx.select(cands)
        if w is not None:
            return w
    return ctx.select(cands)
'''


# A second exemplar: decompose a large modular repo and pipeline per-module, then
# a global verified select. Shows the architect the decomposition pattern.
DECOMPOSE_EXEMPLAR = '''
def orchestrate(ctx):
    ctx.phase("autosolve-decompose")
    modules = ctx.repo_map.get("modules") or []
    candidates = []
    base = 0
    # First: a cheap single speculative attempt (fewest agents on easy tasks).
    first = ctx.solve_attempt(attempt_id=base); base = base + 1
    if first is not None:
        candidates.append(first)
        w = ctx.select(candidates)
        if w is not None:
            return w
    # Escalate: a small wave, diversified, while budget/ceiling allow.
    for wave, k in enumerate(ctx.plan_waves(start=2)):
        if not ctx.budget.can_start() or ctx.agents_used() >= ctx.max_agents:
            break
        thunks = [ctx.make_attempt(base + j) for j in range(k)]
        base = base + k
        candidates = candidates + [c for c in ctx.parallel(thunks) if c is not None]
        w = ctx.select(candidates)
        if w is not None:
            return w
    return ctx.select(candidates)
'''


# The DECOMPOSE-THEN-CONVERGE exemplar handed to the architect for medium/hard repos. It teaches
# the convergence shape (the new DEFAULT) so an enabled author inherits it: decompose -> fan-out
# per module (carry-seeded) -> reduce -> loop-until-dry on the EXACT residual ids carrying the
# best partial forward -> harden. Easy/single-module repos compose the cheap best-of-N path.
CONVERGE_EXEMPLAR = '''
def orchestrate(ctx):
    ctx.phase("scope")
    # decompose ONLY a medium/hard repo into >=2 modules; otherwise compose the cheap best-of-N.
    difficulty = str((ctx.repo_map.get("difficulty") or "")).lower()
    plan = ctx.decompose() if difficulty != "easy" else None
    modules = (plan or {}).get("modules") or []
    if not plan or len(modules) <= 1 or difficulty == "easy":
        return ctx.workflow("default-best-of-n")
    carry = ctx.carry_best()
    cands = ctx.fanout_modules(modules, carry_diff=carry)         # per-module fan-out (no barrier)
    red = ctx.reduce_residuals(cands, carry_diff=carry)          # merge + full-suite score once
    if red["accepted"]:
        return red["candidate"]
    if red["merged_diff"]:
        carry = red["merged_diff"]
    # no-silent-loss collapse: competing whole-repo candidates -> SELECT then best-of-N fallback.
    n = len([c for c in cands if c is not None])
    if int(red.get("gold_passed", 0) or 0) == 0 and (
            not (red["merged_diff"] or "").strip()
            or len(red["conflicts"]) >= max(2, (n + 1) // 2)):
        w = ctx.judge_select(ctx.all_candidates())
        if w is not None:
            return w
        return ctx.workflow("default-best-of-n")
    residual = red["residual_failing_ids"]
    rnd = 0
    while ctx.should_continue_waves() and not red["accepted"]:   # loop-until-dry on the live tree
        targets = residual or ctx.module_gold_ids(modules)
        c = ctx.repair_residual(targets, carry_diff=carry, round=rnd)
        rnd = rnd + 1
        red = ctx.reduce_residuals([c], carry_diff=carry)
        if red["accepted"]:
            return red["candidate"]
        if ctx.coupled_plateau(red, cands):   # coupled plateau -> coherent integrator (merge-reduce v2)
            w = ctx.ralph_loop(id_base=812000, seed_carry=ctx.carry_best(),
                               brief=ctx.integrator_brief(modules, residual))
            if w is not None:
                return w
        if red["merged_diff"]:
            carry = red["merged_diff"]
        residual = red["residual_failing_ids"]
    winner = ctx.select(ctx.all_candidates())
    if winner is not None and difficulty in ("medium", "hard"):
        ctx.adversarial_verify(winner, n=3)
        winner = ctx.select(ctx.all_candidates())
    return winner
'''
