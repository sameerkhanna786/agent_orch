"""Strategy-escalation helpers for the dynamic-transition loop.

Phase 3.2: extracted from the monolithic ``ApexOrchestrator``. The
core dynamic-transition loop lives in :mod:`apex.orchestration.waves`;
this module owns just the small, pure helpers that drive the planner
escalation comparisons.
"""

from __future__ import annotations

from typing import Any, Optional


def strategy_identity_for_loop_guard(strategy: Any) -> Optional[Any]:
    """Phase 2C 2.9: derive a hashable identity for a strategy object.

    We compare ``next_strategy`` to the previously-seen one via this
    identity so a planner that returns the same strategy in
    consecutive iterations is detected and the loop short-circuits
    with ``strategy_stuck`` diagnostics.

    Strategy objects can vary in shape (PlanningDecision in normal
    operation, mocks in tests). We try common identity-bearing
    attributes; if none are present, fall back to the object's
    ``repr`` (which is hashable but order-stable enough for back-to-
    back comparison).
    """
    if strategy is None:
        return None
    primitives = getattr(strategy, "primitives", None)
    if primitives is not None:
        try:
            primitive_keys = tuple(getattr(p, "value", p) for p in primitives)
            rollout_count = getattr(strategy, "rollout_count", None)
            difficulty = getattr(strategy, "difficulty_estimate", None)
            return (primitive_keys, rollout_count, difficulty)
        except TypeError:
            pass
    try:
        return repr(strategy)
    except Exception:  # noqa: BLE001
        return None


__all__ = ["strategy_identity_for_loop_guard"]
