"""Phase 6.5 — Hierarchical agent that wraps the V5 InContainerAgent.

The :class:`HierarchicalAgent` is a planner-above-agent: it receives the
full problem, decomposes it into sub-tasks, asks a
:class:`apex.orchestration.budget_planner.BudgetPlanner` for an initial
turn allocation, then runs the V5 in-container agent once per sub-task
inside a single workspace. After each sub-task finishes the planner
re-balances the remaining budget so the next sub-task can borrow turns
that earlier sub-tasks left on the table.

Decomposition fallback
----------------------

The agent accepts an optional ``decomposer`` callable that maps a
problem statement to a list of sub-task strings. If none is supplied
we use :func:`_default_decompose` which:

  * splits on numbered/bulleted list items at the start of a line, then
  * splits on blank-line paragraphs,
  * collapses whitespace,
  * caps the result at ``max_subtasks`` items,
  * falls back to ``[problem_statement]`` (single sub-task) when no
    structure is found.

This keeps the V5 agent reachable from a single ``solve()`` even on
problem statements that aren't list-shaped (the budget then degrades
gracefully to the V5 default).

Aggregation
-----------

Sub-task ``AgentRunSummary`` objects are concatenated:

  * patches: only sub-tasks that emitted a real diff contribute. Patches
    are joined with newlines preserving order. (V5 patches are unified
    diffs — applying them in order onto the same workspace is the
    intended use.)
  * turns: full per-turn telemetry from every sub-task
  * total_elapsed_seconds: sum
  * terminated_reason: ``"submit_patch"`` if any sub-task submitted, else
    the last sub-task's terminated_reason.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .budget_planner import DEFAULT_PER_SUBTASK_TURNS, BudgetPlanner, TurnBudget

__all__ = [
    "HierarchicalAgent",
    "HierarchicalRunSummary",
    "default_decompose",
    "DEFAULT_MAX_SUBTASKS",
]


logger = logging.getLogger("apex.orchestration.hierarchical_agent")

DEFAULT_MAX_SUBTASKS: int = 3

_NUMBERED_LINE_RE = re.compile(r"^\s*(?:\d+[\.)]|[-*•])\s+(?P<body>.+?)\s*$")


def default_decompose(
    problem_statement: str,
    *,
    max_subtasks: int = DEFAULT_MAX_SUBTASKS,
) -> list[str]:
    """Cheap structural decomposition of a problem statement.

    Strategy:
      1. If the statement contains numbered or bulleted list items,
         use those as sub-tasks.
      2. Otherwise split on blank-line paragraphs.
      3. If neither produces 2+ items, return ``[problem_statement]``.
    """
    if not isinstance(problem_statement, str):
        raise TypeError("problem_statement must be a string")
    text = problem_statement.strip()
    if not text:
        raise ValueError("problem_statement must be non-empty")

    bullets: list[str] = []
    for line in text.splitlines():
        match = _NUMBERED_LINE_RE.match(line)
        if match:
            body = match.group("body").strip()
            if body:
                bullets.append(body)
    if len(bullets) >= 2:
        return bullets[: max(1, max_subtasks)]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) >= 2:
        return paragraphs[: max(1, max_subtasks)]

    return [text]


@dataclass
class _SubtaskRecord:
    name: str
    statement: str
    allocated_turns: int
    actual_turns: int = 0
    actual_tokens: int = 0
    summary: Optional[Any] = None  # AgentRunSummary
    error: Optional[str] = None


@dataclass
class HierarchicalRunSummary:
    """Aggregate result returned by ``HierarchicalAgent.solve``.

    Mirrors the shape of ``InContainerAgent.AgentRunSummary`` for the
    aggregate-level fields, plus per-subtask records for introspection.
    """

    final_patch: Optional[str]
    terminated_reason: str
    give_up_reason: Optional[str] = None
    turns: list[Any] = field(default_factory=list)
    total_elapsed_seconds: float = 0.0
    subtasks: list[dict[str, Any]] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_patch_chars": len(self.final_patch or ""),
            "terminated_reason": self.terminated_reason,
            "give_up_reason": self.give_up_reason,
            "turn_count": len(self.turns),
            "total_elapsed_seconds": round(self.total_elapsed_seconds, 4),
            "turns": [t.to_dict() if hasattr(t, "to_dict") else t for t in self.turns],
            "subtasks": list(self.subtasks),
            "budget": dict(self.budget),
        }


def _safe_turns_from_summary(summary: Any) -> int:
    """Number of turns the in-container agent actually used."""
    if summary is None:
        return 0
    return len(getattr(summary, "turns", []) or [])


def _safe_tokens_from_summary(summary: Any) -> int:
    """Token usage estimate. The V5 summary doesn't track tokens directly,
    so we expose an attribute hook (``total_tokens``) and degrade to 0
    when absent. HierarchicalAgent callers can drive token cost through a
    decorator on the ``in_container_agent`` if they need precise tracking.
    """
    if summary is None:
        return 0
    return int(getattr(summary, "total_tokens", 0) or 0)


def _build_planner_for_problem(
    *,
    total_turns: int,
    n_subtasks: int,
    rebalance_strategy: str,
    tokens_remaining: int,
) -> BudgetPlanner:
    budget = TurnBudget(
        total_turns=int(total_turns),
        tokens_remaining=int(tokens_remaining),
    )
    return BudgetPlanner(
        total_budget=budget,
        n_subtasks_estimate=max(1, int(n_subtasks)),
        rebalance_strategy=rebalance_strategy,
    )


class HierarchicalAgent:
    """Plan-then-act agent that wraps :class:`InContainerAgent`.

    The wrapped agent is mutated in-place: each sub-task overrides the
    in-container agent's ``max_turns`` for the duration of that sub-task,
    runs ``solve_with_summary``, then the planner records actuals.

    The wrapped agent must expose:

      * ``solve_with_summary(problem_statement: str) -> AgentRunSummary``
      * ``max_turns: int`` (writable)

    If the wrapped agent raises during a sub-task we record the error
    on the sub-task record and continue to the next one — partial
    progress is preferable to discarding everything.
    """

    def __init__(
        self,
        planner: BudgetPlanner,
        in_container_agent: Any,
        decomposer: Optional[Callable[[str], list[str]]] = None,
        max_subtasks: int = DEFAULT_MAX_SUBTASKS,
    ) -> None:
        if not isinstance(planner, BudgetPlanner):
            raise TypeError("planner must be a BudgetPlanner instance")
        if in_container_agent is None:
            raise ValueError("in_container_agent is required")
        if not hasattr(in_container_agent, "solve_with_summary"):
            raise TypeError("in_container_agent must expose solve_with_summary(problem_statement)")
        if not hasattr(in_container_agent, "max_turns"):
            raise TypeError("in_container_agent must expose a writable 'max_turns' attribute")
        self.planner = planner
        self.in_container_agent = in_container_agent
        self.decomposer = decomposer or (
            lambda ps: default_decompose(ps, max_subtasks=max_subtasks)
        )
        self.max_subtasks = int(max_subtasks)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, problem_statement: str) -> list[str]:
        """Return sub-task statements for ``problem_statement``."""
        subtasks = list(self.decomposer(problem_statement))
        if not subtasks:
            return [problem_statement]
        # Hard-cap at max_subtasks; degrade gracefully on long lists.
        if len(subtasks) > self.max_subtasks:
            subtasks = subtasks[: self.max_subtasks]
        return [str(s).strip() for s in subtasks if str(s).strip()]

    def solve(self, problem_statement: str) -> HierarchicalRunSummary:
        """Decompose, allocate, run, and aggregate.

        See module docstring for the aggregation semantics.
        """
        if not isinstance(problem_statement, str) or not problem_statement.strip():
            raise ValueError("problem_statement must be a non-empty string")

        subtasks = self.decompose(problem_statement)
        names = [self._name_for(i, statement) for i, statement in enumerate(subtasks)]
        statements = dict(zip(names, subtasks))

        # Re-allocate against the actual number of subtasks (the planner
        # was constructed with an estimate; correct it here).
        initial = self.planner.allocate_initial(names)

        records: dict[str, _SubtaskRecord] = {
            name: _SubtaskRecord(
                name=name,
                statement=statements[name],
                allocated_turns=int(initial[name]),
            )
            for name in names
        }

        total_elapsed = 0.0
        aggregate_turns: list[Any] = []
        any_submit = False
        last_terminated_reason = "max_turns"
        last_give_up_reason: Optional[str] = None

        for index, name in enumerate(names):
            allocation_map = initial if index == 0 else self.planner.reallocate()
            allocated = max(1, int(allocation_map.get(name, 0)))
            records[name].allocated_turns = allocated
            self.in_container_agent.max_turns = allocated
            # 3G: isolate per-subtask state so one subtask's turns / stall counter
            # / verify-gate reject count don't bleed into the next (the agent is
            # re-used across decomposed subtasks).
            reset = getattr(self.in_container_agent, "reset_run_state", None)
            if callable(reset):
                reset()
            try:
                summary = self.in_container_agent.solve_with_summary(records[name].statement)
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    "HierarchicalAgent: subtask %r raised %s: %s; recording error and continuing.",
                    name,
                    type(exc).__name__,
                    exc,
                )
                records[name].error = f"{type(exc).__name__}: {exc}"
                # Still bill the planner the full allocation so the rest
                # of the budget reflects the cost of the failure.
                self.planner.report_actual(
                    name,
                    turns_used=allocated,
                    tokens_used=0,
                )
                continue
            records[name].summary = summary
            records[name].actual_turns = _safe_turns_from_summary(summary)
            records[name].actual_tokens = _safe_tokens_from_summary(summary)
            total_elapsed += float(getattr(summary, "total_elapsed_seconds", 0.0) or 0.0)
            aggregate_turns.extend(getattr(summary, "turns", []) or [])
            terminated_reason = str(getattr(summary, "terminated_reason", "") or "")
            if terminated_reason in ("submit_patch", "submit_patch_verified"):
                any_submit = True
            if terminated_reason:
                last_terminated_reason = terminated_reason
            give_up = getattr(summary, "give_up_reason", None)
            if give_up:
                last_give_up_reason = str(give_up)
            self.planner.report_actual(
                name,
                turns_used=records[name].actual_turns,
                tokens_used=records[name].actual_tokens,
            )

        patches: list[str] = []
        for name in names:
            summary = records[name].summary
            patch = getattr(summary, "final_patch", None) if summary else None
            if patch and isinstance(patch, str) and patch.strip():
                patches.append(patch)
        final_patch = "\n".join(patches) if patches else None

        terminated = "submit_patch" if any_submit else last_terminated_reason

        subtask_records = [
            {
                "name": records[name].name,
                "statement": records[name].statement,
                "allocated_turns": records[name].allocated_turns,
                "actual_turns": records[name].actual_turns,
                "actual_tokens": records[name].actual_tokens,
                "terminated_reason": (
                    str(getattr(records[name].summary, "terminated_reason", "") or "")
                    if records[name].summary
                    else None
                ),
                "produced_patch": bool(
                    records[name].summary and getattr(records[name].summary, "final_patch", None)
                ),
                "error": records[name].error,
            }
            for name in names
        ]
        budget_view = {
            "total_turns": int(self.planner.budget.total_turns),
            "turns_used": int(self.planner.total_turns_used()),
            "tokens_used": int(self.planner.total_tokens_used()),
            "initial_allocation": self.planner.initial_allocation(),
            "final_allocation_view": dict(self.planner.budget.per_subtask_turns),
            "rebalance_strategy": self.planner.rebalance_strategy,
        }

        return HierarchicalRunSummary(
            final_patch=final_patch,
            terminated_reason=terminated,
            give_up_reason=last_give_up_reason if not any_submit else None,
            turns=aggregate_turns,
            total_elapsed_seconds=round(total_elapsed, 4),
            subtasks=subtask_records,
            budget=budget_view,
        )

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def with_default_planner(
        cls,
        in_container_agent: Any,
        *,
        total_turns: int = DEFAULT_PER_SUBTASK_TURNS * DEFAULT_MAX_SUBTASKS,
        n_subtasks: int = DEFAULT_MAX_SUBTASKS,
        rebalance_strategy: str = "feedback",
        decomposer: Optional[Callable[[str], list[str]]] = None,
        max_subtasks: int = DEFAULT_MAX_SUBTASKS,
        tokens_remaining: int = 0,
    ) -> "HierarchicalAgent":
        planner = _build_planner_for_problem(
            total_turns=total_turns,
            n_subtasks=n_subtasks,
            rebalance_strategy=rebalance_strategy,
            tokens_remaining=tokens_remaining,
        )
        return cls(
            planner=planner,
            in_container_agent=in_container_agent,
            decomposer=decomposer,
            max_subtasks=max_subtasks,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _name_for(index: int, statement: str) -> str:
        head = statement.strip().splitlines()[0] if statement.strip() else ""
        # Slug-ish first 24 chars for diagnostics, never used as a key
        # downstream of the planner.
        head = re.sub(r"[^A-Za-z0-9]+", "_", head)[:24].strip("_") or "subtask"
        return f"subtask_{index + 1:02d}_{head.lower()}"
