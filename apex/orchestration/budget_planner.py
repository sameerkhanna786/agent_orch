"""Phase 6.5 — Hierarchical agent budget management.

Today the V5 ``InContainerAgent`` gets a single ``max_turns`` cap (default
8) for the whole problem. For decomposable problems that's wasteful: a
trivial subtask burns one turn but is allocated 8; a complex subtask
exhausts 8 turns but the budget that ``trivial subtask`` left on the
table is never reclaimed.

:class:`BudgetPlanner` sits *above* the V5 agent. It:

  1. Receives a ``TurnBudget`` with a total turn cap (and optional token
     cap).
  2. Splits the budget across N sub-tasks (initially equal, optionally
     priority-weighted).
  3. Receives ``report_actual()`` callbacks after each sub-task and
     reallocates the remaining budget across the *pending* sub-tasks.
  4. Enforces a hard cap (no sub-task gets more than 2x its initial
     allocation) so a single runaway sub-task can't starve the rest.

The planner is deliberately mechanism-only — it doesn't know what a
"subtask" is, doesn't know how the agent reports turns, and doesn't
know the LLM. :class:`apex.orchestration.hierarchical_agent.HierarchicalAgent`
wires it to the V5 agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

__all__ = [
    "TurnBudget",
    "BudgetPlanner",
    "DEFAULT_PER_SUBTASK_TURNS",
    "DEFAULT_REBALANCE_STRATEGY",
    "VALID_REBALANCE_STRATEGIES",
]


DEFAULT_PER_SUBTASK_TURNS: int = 8
DEFAULT_REBALANCE_STRATEGY: str = "feedback"
VALID_REBALANCE_STRATEGIES: tuple[str, ...] = ("feedback", "static")
HARD_CAP_MULTIPLIER: int = 2


@dataclass
class TurnBudget:
    """Mutable turn-budget container the planner reads/writes.

    ``per_subtask_turns`` is the *current* allocation; the planner
    overwrites it on each ``reallocate()``.
    """

    total_turns: int
    per_subtask_turns: dict[str, int] = field(default_factory=dict)
    tokens_used: int = 0
    tokens_remaining: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "total_turns": int(self.total_turns),
            "per_subtask_turns": {str(k): int(v) for k, v in self.per_subtask_turns.items()},
            "tokens_used": int(self.tokens_used),
            "tokens_remaining": int(self.tokens_remaining),
        }


def _coerce_strict_positive_int(value: int, name: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive int") from exc
    if out < 1:
        raise ValueError(f"{name} must be >= 1 (got {out})")
    return out


class BudgetPlanner:
    """Allocate, observe, and reallocate the turn budget across subtasks.

    Lifecycle:

    >>> planner = BudgetPlanner(TurnBudget(total_turns=24))
    >>> planner.allocate_initial(["repro", "fix", "validate"])
    {'repro': 8, 'fix': 8, 'validate': 8}
    >>> planner.report_actual("repro", turns_used=3, tokens_used=1500)
    >>> planner.reallocate()                  # remaining 21 over [fix, validate]
    {'fix': 10, 'validate': 11}
    >>> planner.report_actual("fix", turns_used=10, tokens_used=8000)
    >>> planner.reallocate()
    {'validate': 11}
    """

    def __init__(
        self,
        total_budget: TurnBudget,
        n_subtasks_estimate: int = 3,
        rebalance_strategy: str = DEFAULT_REBALANCE_STRATEGY,
    ) -> None:
        if not isinstance(total_budget, TurnBudget):
            raise TypeError("total_budget must be a TurnBudget instance")
        if total_budget.total_turns < 1:
            raise ValueError(
                f"TurnBudget.total_turns must be >= 1 (got {total_budget.total_turns})"
            )
        self.budget = total_budget
        self.n_subtasks_estimate = _coerce_strict_positive_int(
            n_subtasks_estimate, "n_subtasks_estimate"
        )
        if rebalance_strategy not in VALID_REBALANCE_STRATEGIES:
            raise ValueError(
                f"rebalance_strategy must be one of {VALID_REBALANCE_STRATEGIES}; "
                f"got {rebalance_strategy!r}"
            )
        self.rebalance_strategy = rebalance_strategy
        # State updated as the agent reports actuals back.
        self._subtasks: list[str] = []
        self._initial_allocation: dict[str, int] = {}
        self._weights: dict[str, float] = {}
        self._completed: dict[str, int] = {}
        self._completed_tokens: dict[str, int] = {}
        self._completed_order: list[str] = []

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def allocate_initial(
        self,
        subtasks: Iterable[str],
        *,
        weights: Optional[Mapping[str, float]] = None,
    ) -> dict[str, int]:
        """Divide ``total_turns`` across the given subtasks.

        Defaults to equal allocation. Pass ``weights`` (positive floats,
        keyed by subtask name) for priority-weighted division. The sum
        of allocations is exactly ``total_turns`` (modulo a remainder
        that is distributed first-come/first-served).
        """
        ordered = [str(s) for s in subtasks]
        if not ordered:
            raise ValueError("subtasks must be non-empty")
        if len(set(ordered)) != len(ordered):
            raise ValueError("subtask names must be unique")
        if weights is not None:
            for k in weights:
                if k not in ordered:
                    raise ValueError(f"weight key {k!r} is not in subtasks {ordered}")
            for v in weights.values():
                if float(v) < 0.0:
                    raise ValueError("weights must be non-negative")
        self._subtasks = list(ordered)
        if weights is None or not weights:
            self._weights = {name: 1.0 for name in ordered}
        else:
            # Default missing keys to 1.0 — caller may pass partial weights.
            self._weights = {name: float(weights.get(name, 1.0)) for name in ordered}
            # Guard against all-zero weights collapsing the planner.
            if sum(self._weights.values()) <= 0.0:
                self._weights = {name: 1.0 for name in ordered}

        allocation = self._proportional_split(
            ordered,
            total=int(self.budget.total_turns),
            weights=self._weights,
        )
        self._initial_allocation = dict(allocation)
        self.budget.per_subtask_turns = dict(allocation)
        # Reset state — allocate_initial restarts the planner.
        self._completed = {}
        self._completed_tokens = {}
        self._completed_order = []
        return dict(allocation)

    def report_actual(
        self,
        subtask: str,
        turns_used: int,
        tokens_used: int = 0,
    ) -> None:
        """Record that ``subtask`` finished after using ``turns_used`` turns
        and ``tokens_used`` tokens. Idempotent re-report of the same name
        OVERWRITES (so retries with corrections are supported)."""
        if subtask not in self._initial_allocation:
            raise KeyError(
                f"subtask {subtask!r} was not in the initial allocation; "
                f"known = {list(self._initial_allocation)}"
            )
        if turns_used < 0:
            raise ValueError("turns_used must be non-negative")
        if tokens_used < 0:
            raise ValueError("tokens_used must be non-negative")
        already_seen = subtask in self._completed
        self._completed[subtask] = int(turns_used)
        self._completed_tokens[subtask] = int(tokens_used)
        if not already_seen:
            self._completed_order.append(subtask)
        self.budget.tokens_used = sum(self._completed_tokens.values())
        if self.budget.tokens_remaining > 0:
            # If the caller seeded a token cap, decrement it.
            self.budget.tokens_remaining = max(
                0,
                self.budget.tokens_remaining - int(tokens_used),
            )

    def reallocate(self) -> dict[str, int]:
        """Rebalance the *remaining* turn budget across the *pending* subtasks.

        Strategy:

          * In ``"static"`` mode this is a no-op — initial allocation
            stands. We still update ``budget.per_subtask_turns`` to drop
            completed subtasks, so callers get a clean view.
          * In ``"feedback"`` mode the remaining budget (total - turns
            already used) is divided across pending subtasks weighted
            by their ORIGINAL allocation. A pending subtask never
            receives more than ``HARD_CAP_MULTIPLIER × initial``.

        Returns the new ``{pending_subtask: turns}`` map. If everything
        is complete the result is ``{}``.
        """
        pending = [s for s in self._subtasks if s not in self._completed]
        if not pending:
            self.budget.per_subtask_turns = {}
            return {}

        if self.rebalance_strategy == "static":
            new_alloc = {s: self._initial_allocation[s] for s in pending}
            self.budget.per_subtask_turns = dict(new_alloc)
            return dict(new_alloc)

        used = sum(self._completed.values())
        remaining = max(0, int(self.budget.total_turns) - used)
        # Each pending subtask is guaranteed at least 1 turn if any budget
        # remains, but never more than the hard cap (2x initial).
        if remaining <= 0:
            new_alloc = {s: 0 for s in pending}
            self.budget.per_subtask_turns = dict(new_alloc)
            return dict(new_alloc)

        weights = {s: float(self._initial_allocation[s]) for s in pending}
        if sum(weights.values()) <= 0.0:
            weights = {s: 1.0 for s in pending}
        proportional = self._proportional_split(
            pending,
            total=int(remaining),
            weights=weights,
        )

        # Apply the hard cap (2 × initial) and redistribute the
        # surplus turns among the still-uncapped subtasks.
        caps = {s: int(HARD_CAP_MULTIPLIER * self._initial_allocation[s]) for s in pending}
        new_alloc = dict(proportional)
        for _ in range(len(pending)):  # bounded redistribution rounds
            surplus = 0
            for s in pending:
                if new_alloc[s] > caps[s]:
                    surplus += new_alloc[s] - caps[s]
                    new_alloc[s] = caps[s]
            if surplus <= 0:
                break
            uncapped = [s for s in pending if new_alloc[s] < caps[s]]
            if not uncapped:
                break
            uncapped_weights = {s: weights[s] for s in uncapped}
            extra = self._proportional_split(
                uncapped,
                total=int(surplus),
                weights=uncapped_weights,
            )
            for s, add in extra.items():
                new_alloc[s] += int(add)

        # Final clamp — never exceed the hard cap even after redistribution.
        for s in pending:
            if new_alloc[s] > caps[s]:
                new_alloc[s] = caps[s]
        self.budget.per_subtask_turns = dict(new_alloc)
        return dict(new_alloc)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def initial_allocation(self) -> dict[str, int]:
        return dict(self._initial_allocation)

    def completed(self) -> dict[str, int]:
        return dict(self._completed)

    def total_turns_used(self) -> int:
        return sum(self._completed.values())

    def total_tokens_used(self) -> int:
        return sum(self._completed_tokens.values())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _proportional_split(
        keys: Iterable[str],
        *,
        total: int,
        weights: Mapping[str, float],
    ) -> dict[str, int]:
        """Split ``total`` ints across ``keys`` proportional to ``weights``.

        Each key receives at least 1 (unless ``total`` < n_keys, in which
        case the lowest-weight keys round to 0). Remainder is allocated
        first-come/first-served by largest fractional part.
        """
        ordered = list(keys)
        if not ordered:
            return {}
        n = len(ordered)
        total = max(0, int(total))
        if total == 0:
            return {k: 0 for k in ordered}

        weight_sum = float(sum(weights.get(k, 1.0) for k in ordered)) or float(n)
        floats = {k: total * (float(weights.get(k, 1.0)) / weight_sum) for k in ordered}
        floor = {k: int(v) for k, v in floats.items()}
        remainder = total - sum(floor.values())
        if remainder > 0:
            # Distribute the remainder by largest fractional part, ties
            # broken by ordered position (stable / deterministic).
            fracs = sorted(
                ordered,
                key=lambda k: (-(floats[k] - floor[k]), ordered.index(k)),
            )
            for k in fracs[: max(0, remainder)]:
                floor[k] += 1
        # Guarantee at least 1 for as many keys as possible (we already
        # reserved budget; this only matters when total >= n).
        if total >= n:
            zeros = [k for k in ordered if floor[k] == 0]
            donors = sorted(ordered, key=lambda k: -floor[k])
            for z in zeros:
                for d in donors:
                    if floor[d] > 1 and d != z:
                        floor[d] -= 1
                        floor[z] = 1
                        break
        return floor
