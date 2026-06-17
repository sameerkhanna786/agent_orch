"""judge_panel — score candidates with read-only judges along diverse lenses and
write the aggregate into the SOFT ``perspective`` tiebreak (Backbone 2.3).

Safety: judges are ``ctx.ask`` signals; the only write is ``set_soft`` (strictly
below every execution key in ``ranking_key``), so a judge can only break ties among
candidates the EXECUTION layer already ranks equal — never promote an unaccepted one
over an accepted one. With a single lens it degrades to one judge per candidate.
"""

from __future__ import annotations

JUDGE_SCHEMA = {
    "type": "object", "additionalProperties": True, "required": ["score"],
    "properties": {"score": {"type": "number"}, "reason": {"type": "string"}},
}

_DEFAULT_LENS = "overall correctness, robustness, and code quality of the patch"


def _judge_prompt(cand, lens: str) -> str:
    diff = (cand.diff or "")[:6000]
    return (
        f"Score this candidate patch for a repository task on: {lens}. Return JSON "
        "{score, reason} where score is in [0,1] (higher is better).\n\n```diff\n"
        + diff + "\n```"
    )


def judge_panel(ctx, candidates, *, lenses=None, base_id: int = 910000, vendor=None, model=None):
    """Attach a soft ``perspective`` score (mean over lenses/judges) to each candidate.
    Returns the same candidates (re-rank later with ctx.select). Accept is untouched."""
    cands = [c for c in candidates if c is not None]
    if not cands:
        return []
    lenses = tuple(lenses) if lenses else (_DEFAULT_LENS,)
    thunks, owner = [], []
    for ci, c in enumerate(cands):
        for li, lens in enumerate(lenses):
            aid = base_id + ci * 100 + li
            thunks.append(lambda c=c, lens=lens, aid=aid:
                          ctx.ask(_judge_prompt(c, lens), schema=JUDGE_SCHEMA,
                                  vendor=vendor, model=model, agent_id=aid))
            owner.append(ci)
    results = ctx.signals(thunks)
    agg: dict = {}
    for ci, r in zip(owner, results):
        if isinstance(r, dict) and isinstance(r.get("score"), (int, float)) and not isinstance(r.get("score"), bool):
            agg.setdefault(ci, []).append(float(r["score"]))
    for ci, c in enumerate(cands):
        scores = agg.get(ci)
        if scores:
            c.set_soft(perspective=sum(scores) / len(scores))
    return cands


def judge_select(ctx, candidates, *, lenses=None, base_id: int = 910000, vendor=None, model=None):
    """Judge-panel-then-SELECT (the guide's "score with parallel judges, synthesize from the
    winner" shape): attach the soft judge score, then return the EXECUTION-AUTHORITATIVE winner
    via ``ctx.select``. The promotion guard is ``select_best`` (kernel/select.py): it returns the
    FIRST ACCEPTED candidate and abstains if none is accepted, so the soft judge score (which also
    sits below every execution key in ``ranking_key``) only re-orders candidates WITHIN the accepted
    set — it can never promote an unaccepted one. Returns the selected Candidate or None (abstain).
    Degrades to plain ``ctx.select`` with a single lens / no candidates."""
    scored = judge_panel(ctx, candidates, lenses=lenses, base_id=base_id, vendor=vendor, model=model)
    return ctx.select(scored)
