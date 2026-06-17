"""completeness_critic — a final read-only agent that asks "what's missing?" so the
orchestrator can decide whether to keep escalating (Backbone 2.3).

It is a pure SIGNAL: a ``ctx.ask`` over the best not-yet-accepted candidate and the
tests it still fails, returning ``{complete, gaps, recommendation}``. It never returns
a Candidate and never touches acceptance. Degrades to ``{'complete': True, 'gaps': []}``
when there is no candidate or the ask fails (so a caller can always read ``.gaps``).
"""

from __future__ import annotations

CRITIC_SCHEMA = {
    "type": "object", "additionalProperties": True, "required": ["complete"],
    "properties": {
        "complete": {"type": "boolean"},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "recommendation": {"type": "string"},
    },
}

_EMPTY = {"complete": True, "gaps": [], "recommendation": ""}


def completeness_critic(ctx, candidate, *, agent_id: int = 930000, vendor=None, model=None):
    if candidate is None:
        return dict(_EMPTY)
    fails = ", ".join(map(str, (candidate.meta or {}).get("failing_nodeids", [])[:25]))
    out = ctx.ask(
        "Review this in-progress solution to a repository task. Its visible test suite is "
        "NOT fully green. Identify concretely what is MISSING or unhandled, and whether "
        "another targeted attempt is worthwhile.\n\nStill-failing tests: "
        + (fails or "(unknown)") + "\n```diff\n" + (candidate.diff or "")[:6000]
        + "\n```\n\nReturn JSON {complete (true only if essentially done), gaps (the "
        "concrete missing pieces), recommendation (what the next attempt should focus on)}.",
        schema=CRITIC_SCHEMA, vendor=vendor, model=model, agent_id=agent_id)
    return out if isinstance(out, dict) else dict(_EMPTY)
