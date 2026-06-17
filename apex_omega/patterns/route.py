"""classify_and_route — classify each item then dispatch it to the right handler.

This is Anthropic's officially-named "Classify-and-act" harness pattern (one of the six in the
"A harness for every task" blog): a classifier labels each item, and the label routes it to a
different agent/behavior. See also guide §4.3 (the deep-research "scope -> route" shape) and §3.2
(route cheap work to a smaller model).

A read-only ``ctx.ask`` labels each item with a category (a SIGNAL); the orchestrator's
``routes`` map then dispatches each item to the handler registered for its category — e.g. route
easy items to a cheap model and hard items to a stronger vendor, or route by file type to a
specialized prompt. Classification is a pure signal (it can never produce a Candidate or accept
anything); the HANDLERS do the real work (typically a ``ctx.solve_attempt`` with a chosen
vendor/model). Returns the per-item handler results in INPUT ORDER.
"""

from __future__ import annotations

CLASSIFY_SCHEMA = {
    "type": "object", "additionalProperties": True, "required": ["category"],
    "properties": {"category": {"type": "string"}, "reason": {"type": "string"}},
}


def classify_and_route(ctx, items, *, classify, routes, default=None,
                       base_id: int = 960000, vendor=None, model=None):
    """Classify each item via a read-only ``ctx.ask`` (fanned out through ``ctx.signals`` so it
    never advances the solve plateau), then dispatch it to ``routes[category]``.

      classify : callable item -> classification-prompt str (or a fixed prompt str applied to all)
      routes   : dict category(str) -> handler callable item -> result
      default  : handler for an unmatched/missing category (None -> that item yields None)

    A handler that raises is caught (logged, yields None) so one bad route can never crash the
    cell. Returns a list aligned with ``items``."""
    items = list(items or [])
    if not items:
        return []
    routes = dict(routes or {})

    def _mk(idx, it):
        prompt = classify(it) if callable(classify) else str(classify)
        return lambda p=prompt, aid=base_id + idx: ctx.ask(
            str(p) + "\n\nReturn JSON {category, reason}.", schema=CLASSIFY_SCHEMA,
            vendor=vendor, model=model, agent_id=aid, agent_type="classify")

    verdicts = ctx.signals([_mk(i, it) for i, it in enumerate(items)])
    out = []
    for it, v in zip(items, verdicts):
        cat = v.get("category") if isinstance(v, dict) else None
        handler = routes.get(cat) if cat is not None else None
        if handler is None:
            handler = default
        if handler is None:
            ctx.log(f"classify_and_route: no route for category {cat!r}; skipped")
            out.append(None)
            continue
        try:
            out.append(handler(it))
        except Exception as exc:
            ctx.log(f"classify_and_route: handler for {cat!r} raised {type(exc).__name__}: {exc}")
            out.append(None)
    return out
