"""synthesize — combine several partial solutions into one new attempt (Backbone 2.3).

A read-only ``ctx.ask`` produces a synthesis PLAN that grafts the correct parts of the
best partial candidates and targets their shared remaining failures; that plan then
seeds ONE ordinary ``ctx.solve_attempt`` — so the synthesized candidate is
EXECUTION-SCORED (the legitimate accept path), never a soft write. Degrades: with a
single candidate (or one already accepted) it returns that candidate with no extra agent.
"""

from __future__ import annotations

SYNTH_SCHEMA = {
    "type": "object", "additionalProperties": True, "required": ["plan"],
    "properties": {"plan": {"type": "string"},
                   "keep": {"type": "array", "items": {"type": "string"}}},
}


def synthesize(ctx, candidates, *, attempt_id, top_k: int = 3, vendor=None, model=None,
               ask_id: int = 920000):
    cands = [c for c in candidates if c is not None]
    if not cands:
        return None
    ranked = sorted(cands, key=lambda c: float(c.public_signal_score or 0.0), reverse=True)
    for c in ranked:
        if getattr(c, "accepted", False):
            return c  # already solved -> synthesis is unnecessary (completion-first)
    top = ranked[: max(1, top_k)]
    if len(top) == 1:
        return top[0]  # nothing to combine -> the single best partial (no extra agent)
    blocks = []
    for i, c in enumerate(top):
        fails = ", ".join(map(str, (c.meta or {}).get("failing_nodeids", [])[:15]))
        blocks.append(
            f"--- CANDIDATE {i} (pass_rate={float(c.public_signal_score or 0.0):.2f}) ---\n"
            "still failing: " + (fails or "(unknown)") + "\n```diff\n"
            + (c.diff or "")[:3500] + "\n```")
    plan = ctx.ask(
        "Several partial solutions to the SAME repository task each pass most of the "
        "visible test suite but none is fully green. Synthesize ONE concrete plan that "
        "COMBINES their correct parts and fixes the remaining failures.\n\n"
        + "\n\n".join(blocks)
        + "\n\nReturn JSON {plan: a concrete implementation plan, keep: which ideas to keep}.",
        schema=SYNTH_SCHEMA, vendor=vendor, model=model, agent_id=ask_id)
    if not isinstance(plan, dict) or not plan.get("plan"):
        return top[0]  # synthesis signal unavailable -> best partial (never worse)
    synth_prompt = (
        "Implement this repository task by following this synthesis plan, derived from "
        "several partial solutions that each passed most of the visible suite:\n\n"
        + str(plan["plan"])[:6000])
    # EXECUTION-SCORED fresh attempt -> the legitimate accept path.
    return ctx.solve_attempt(attempt_id=attempt_id, strategy="synthesize",
                             prompt=synth_prompt, vendor=vendor, model=model)
