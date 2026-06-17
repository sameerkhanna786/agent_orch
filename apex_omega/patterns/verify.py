"""adversarial_verify — independent read-only skeptics try to REFUTE an
execution-accepted candidate (Backbone 2.3).

Safety: every verifier is a ``ctx.ask`` (a forced read-only SIGNAL). The only
mutations are ``candidate.refute()`` (monotone True->False) and a ``set_soft``
tiebreak (strictly below every execution key). A verifier can therefore NEVER
promote an unverified candidate — at worst it DROPS a suspicious solve. With
``n<=0`` it is a no-op, so the pattern degrades to plain best-of-N.
"""

from __future__ import annotations

VERDICT_SCHEMA = {
    "type": "object", "additionalProperties": True, "required": ["refuted"],
    "properties": {
        "refuted": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
}

# Distinct refutation grounds — diversity catches failure modes redundancy can't.
_DEFAULT_LENSES = (
    "the solution CHEATED the visible suite: deleted/weakened/edited tests, hardcoded "
    "test-specific return values, or stubbed the function under test to echo fixtures",
    "the implementation is INCORRECT in general despite passing: a wrong algorithm that "
    "only fits the visible cases, or missing edge handling the suite clearly implies",
    "the change is INCOMPLETE: it left TODO/NotImplementedError/bare-pass bodies in the "
    "very functionality the visible tests exercise",
)


def _verdict_prompt(cand, lens: str) -> str:
    diff = (cand.diff or "")[:6000]
    return (
        "You are an ADVERSARIAL verifier. A coding agent claims to have solved a "
        "repository task and its visible pytest suite passes. Examine its patch and try "
        f"to REFUTE the claim specifically on this ground:\n  {lens}\n\nPATCH:\n```diff\n"
        + diff + "\n```\n\nThe visible suite ALREADY passes, so the bar for refuting is "
        "HIGH: refute ONLY with concrete evidence visible in the patch itself. If you have "
        "no concrete evidence, return refuted=false. Return JSON {refuted, confidence, reason}."
    )


def adversarial_verify(ctx, candidate, *, n: int = 1, lenses=None, refute_if="majority",
                       base_id: int = 900000, vendor=None, model=None):
    """Refute-test an execution-accepted ``candidate`` with ``n`` independent read-only
    skeptics. ``refute_if`` is ``"majority"`` (default), ``"any"``, ``"all"``, or an int
    vote threshold. Returns the (possibly refuted) candidate; accept can only go down."""
    if candidate is None or not getattr(candidate, "accepted", False) or n <= 0:
        return candidate  # nothing verified to refute (or disabled) -> unchanged
    lenses = tuple(lenses) if lenses else _DEFAULT_LENSES
    # review-fix #9: bind candidate identity into the verifier's journal key-space. The ask
    # prompt only embeds diff[:6000], so two distinct candidates whose diffs match in the
    # first 6000 bytes would share keys at the default base_id and replay each other's
    # verdicts. content_sha is the sha1 of the FULL diff, so this disjoins them.
    if base_id == 900000:
        try:
            base_id = 900000 + (int((candidate.content_sha or "0")[:8], 16) % 1_000_000)
        except (ValueError, TypeError):
            base_id = 900000

    def _ask(i):
        return lambda: ctx.ask(_verdict_prompt(candidate, lenses[i % len(lenses)]),
                               schema=VERDICT_SCHEMA, vendor=vendor, model=model,
                               agent_id=base_id + i)

    verdicts = [v for v in ctx.signals([_ask(i) for i in range(n)]) if isinstance(v, dict)]
    if not verdicts:
        return candidate  # every skeptic failed -> keep the execution verdict (fail-open)
    refutes = sum(1 for v in verdicts if v.get("refuted") is True)
    m = len(verdicts)
    # review-fix #10: compute the threshold against the REQUESTED n, not the count of
    # responders — a failed/None verifier counts as non-refuting (it is dropped), so partial
    # verifier failure can never LOWER the bar and wrongly discard a verified solve.
    if isinstance(refute_if, int) and not isinstance(refute_if, bool):
        need = max(1, refute_if)
    elif refute_if == "any":
        need = 1
    elif refute_if == "all":
        need = n                      # unanimity over the requested n
    else:  # majority
        need = n // 2 + 1
    # agreement recorded as a SOFT (sub-execution) tiebreak; never an accept input.
    candidate.set_soft(perspective=(m - refutes) / m)
    if refutes >= need:
        ctx.log(f"adversarial_verify: {refutes}/{m} refuted {candidate.candidate_id} -> downgraded")
        candidate.refute()
    return candidate


def _need(refute_if, n: int) -> int:
    if isinstance(refute_if, int) and not isinstance(refute_if, bool):
        return max(1, refute_if)
    if refute_if == "any":
        return 1
    if refute_if == "all":
        return n
    return n // 2 + 1                     # majority (default)


def adversarial_filter(ctx, items, *, votes: int = 3, to_text=None, refute_if="majority",
                       base_id: int = 950000, vendor=None, model=None):
    """ADMIT-gate for PLAIN-DATA items (audit findings, extracted claims, ...): each item faces
    ``votes`` independent read-only skeptics prompted to REFUTE it as a false positive; an item is
    ADMITTED (kept, in order) only if it SURVIVES (refutations below the threshold). Distinct from
    ``adversarial_verify`` (which downgrades an execution-accepted Candidate) — this filters
    arbitrary items and NEVER touches Candidate.accepted, so the Cardinal Contract is preserved.
    Uses ``ctx.signals`` (read-only fan-out; does not advance the plateau). ``votes<=0`` -> identity;
    a finding with NO usable verdicts is kept (fail-open — never silently dropped)."""
    items = list(items or [])
    if votes <= 0 or not items:
        return items
    to_text = to_text or (lambda x: x if isinstance(x, str) else repr(x))
    survivors: list = []
    for idx, item in enumerate(items):
        text = str(to_text(item))[:6000]

        def _ask(j, _t=text, _idx=idx):
            return lambda: ctx.ask(
                "You are an ADVERSARIAL reviewer. Try to REFUTE the following finding/claim as a "
                "FALSE POSITIVE. Refute ONLY with concrete reasoning; if you cannot, return "
                f"refuted=false.\n\nFINDING:\n{_t}\n\nReturn JSON {{refuted, confidence, reason}}.",
                schema=VERDICT_SCHEMA, vendor=vendor, model=model,
                agent_id=base_id + _idx * 100 + j)

        verdicts = [v for v in ctx.signals([_ask(j) for j in range(votes)]) if isinstance(v, dict)]
        if not verdicts:
            survivors.append(item)               # fail-open: keep when no usable verdict
            continue
        refutes = sum(1 for v in verdicts if v.get("refuted") is True)
        if refutes < _need(refute_if, votes):
            survivors.append(item)
    return survivors
