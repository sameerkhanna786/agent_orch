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
DEFAULT_ORCHESTRATION = '''
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
