"""tournament — pairwise round-robin judging to rank candidates (Backbone 2.3 / guide §4.2).

"Tournament" is one of Anthropic's six officially-named harness patterns (per the "A harness for
every task" blog): PAIRWISE judging until a winner — deliberately distinct from ``judge_panel``,
which scores each candidate in isolation.

A richer alternative to ``judge_panel``'s absolute scoring: every unordered PAIR of candidates
is judged head-to-head ("which patch is better?") by an independent read-only ``ctx.ask``, and
each candidate's WIN RATE becomes its SOFT ``perspective`` tiebreak. Pairwise comparison is often
more reliable than absolute 0..1 scoring, but it is still a SOFT signal and cannot promote an
unverified candidate: the actual promotion guard is ``select_best`` (kernel/select.py), which
returns the FIRST ACCEPTED candidate in rank order and abstains if none is accepted — so the soft
win-rate (which also sits below every execution key in ``ranking_key``) only re-orders candidates
WITHIN the accepted set (Cardinal Contract preserved). Degrades: <2 candidates -> no-op (no agent).
"""

from __future__ import annotations

TOURNEY_SCHEMA = {
    "type": "object", "additionalProperties": True, "required": ["winner"],
    "properties": {
        "winner": {"type": "integer", "enum": [0, 1]},   # 0 => A is better, 1 => B is better
        "reason": {"type": "string"},
    },
}

_DEFAULT_LENS = "overall correctness, robustness, and code quality of the patch"


def _pair_prompt(a, b, lens: str) -> str:
    da = (a.diff or "")[:4000]
    db = (b.diff or "")[:4000]
    return (
        f"Two candidate patches (A and B) solve the SAME repository task. Judge head-to-head on: "
        f"{lens}. Return JSON {{winner, reason}} where winner=0 if A is better, winner=1 if B is "
        "better.\n\n--- PATCH A ---\n```diff\n" + da + "\n```\n\n--- PATCH B ---\n```diff\n" + db + "\n```")


def tournament(ctx, candidates, *, lens=None, base_id: int = 940000, vendor=None, model=None):
    """Pairwise round-robin: write each candidate's win-rate into the soft ``perspective`` tiebreak
    and return the candidates (re-rank later with ``ctx.select`` / ``judge_select``). A pair with no
    usable verdict splits the point (fail-open — never silently drops a candidate)."""
    cands = [c for c in candidates if c is not None]
    if len(cands) < 2:
        return cands
    lens = lens or _DEFAULT_LENS
    pairs = [(i, j) for i in range(len(cands)) for j in range(i + 1, len(cands))]
    thunks, owners = [], []
    for pi, (i, j) in enumerate(pairs):
        aid = base_id + pi
        thunks.append(lambda i=i, j=j, aid=aid: ctx.ask(
            _pair_prompt(cands[i], cands[j], lens), schema=TOURNEY_SCHEMA,
            vendor=vendor, model=model, agent_id=aid))
        owners.append((i, j))
    results = ctx.signals(thunks)
    wins = [0.0] * len(cands)
    games = [0.0] * len(cands)
    for (i, j), r in zip(owners, results):
        games[i] += 1.0
        games[j] += 1.0
        w = r.get("winner") if isinstance(r, dict) else None
        if w == 0:
            wins[i] += 1.0
        elif w == 1:
            wins[j] += 1.0
        else:  # tie / no usable verdict -> split the point (fail-open)
            wins[i] += 0.5
            wins[j] += 0.5
    for idx, c in enumerate(cands):
        if games[idx] > 0:
            c.set_soft(perspective=wins[idx] / games[idx])
    return cands
