"""Followup-loop abstraction.

Phase 3.2: introduces a ``FollowupLoop`` abstract base class so the
four orchestrator follow-up loops (near-miss, structural-recovery,
coverage-gap, selection) can share budget tracking and transition
plumbing instead of repeating the scaffolding inline in
``ApexOrchestrator.solve()``.

The concrete subclasses delegate the heavy lifting back to the
orchestrator's existing ``_launch_*_followup`` /
``_select_*_anchor`` methods so this is a pure refactoring layer:
behavior is identical to the inline implementation, but tests can
target the loop abstraction directly via mocks.
"""

from __future__ import annotations

import abc
from typing import Any, Optional

from ..planning.manager import IssuePlan, IssuePlanner
from ..preprocessing.repo_analyzer import RepoContext
from ..rollout.engine import RolloutEngine, RolloutResult
from ..task_state_graph import TaskStateGraph


class FollowupLoop(abc.ABC):
    """Abstract base for the four orchestrator follow-up loops.

    Each subclass owns its own activation predicate (``should_run``),
    anchor selection, request building, and result application. Shared
    budget tracking lives here.
    """

    name: str = "followup"

    def __init__(self, orchestrator: Any) -> None:
        self._orchestrator = orchestrator

    # ---- shared budget plumbing --------------------------------------

    def is_exhausted(
        self,
        *,
        rollout_results: list[RolloutResult],
        followup_round: int,
    ) -> bool:
        """Phase 2 10.T: stop the followup loop when either cap fires."""
        return self._orchestrator._followup_budget_exhausted(
            rollout_results=rollout_results,
            followup_round=followup_round,
        )

    # ---- subclass surface --------------------------------------------

    @abc.abstractmethod
    def should_run(
        self,
        *,
        rollout_results: list[RolloutResult],
        successful: list[RolloutResult],
        best_result: Optional[RolloutResult] = None,
    ) -> bool:
        """Return True when this followup loop should fire on the current state."""

    @abc.abstractmethod
    def select_anchor(
        self,
        rollout_results: list[RolloutResult],
        *,
        best_result: Optional[RolloutResult] = None,
    ) -> Optional[RolloutResult]:
        """Pick the anchor rollout this followup will branch from."""


class NearMissFollowup(FollowupLoop):
    """One follow-up rollout seeded from the highest near-miss patch."""

    name = "near_miss"

    def should_run(
        self,
        *,
        rollout_results: list[RolloutResult],
        successful: list[RolloutResult],
        best_result: Optional[RolloutResult] = None,
    ) -> bool:
        if not bool(self._orchestrator.config.rollout.enable_residual_followup):
            return False
        return self.select_anchor(rollout_results, best_result=best_result) is not None

    def select_anchor(
        self,
        rollout_results: list[RolloutResult],
        *,
        best_result: Optional[RolloutResult] = None,
    ) -> Optional[RolloutResult]:
        return self._orchestrator._select_near_miss_anchor(rollout_results)

    def launch(
        self,
        *,
        repo_path: str,
        repo_context: RepoContext,
        issue_description: str,
        issue_plan: IssuePlan,
        planner: IssuePlanner,
        test_command: Optional[str],
        engine: RolloutEngine,
        rollout_results: list[RolloutResult],
        task_state_graph: Optional[TaskStateGraph],
        wallclock_deadline: Optional[float] = None,
    ) -> tuple[list[RolloutResult], list[dict[str, Any]]]:
        return self._orchestrator._launch_near_miss_followup(
            repo_path=repo_path,
            repo_context=repo_context,
            issue_description=issue_description,
            issue_plan=issue_plan,
            planner=planner,
            test_command=test_command,
            engine=engine,
            rollout_results=rollout_results,
            task_state_graph=task_state_graph,
            wallclock_deadline=wallclock_deadline,
        )


class StructuralRecoveryFollowup(FollowupLoop):
    """Targeted-recovery followup when multiple rollouts share a structural break."""

    name = "structural_recovery"

    def should_run(
        self,
        *,
        rollout_results: list[RolloutResult],
        successful: list[RolloutResult],
        best_result: Optional[RolloutResult] = None,
    ) -> bool:
        if successful:
            return False
        return bool(self._orchestrator.config.rollout.enable_residual_followup)

    def select_anchor(
        self,
        rollout_results: list[RolloutResult],
        *,
        best_result: Optional[RolloutResult] = None,
    ) -> Optional[RolloutResult]:
        # Structural-recovery deliberately starts from a CLEAN baseline,
        # so it doesn't pick a single anchor — it consumes the recurring
        # blocker pattern across rollouts. We expose a pseudo-anchor
        # (the first affected rollout) for shape parity with the other
        # loops; callers should not rely on the specific id.
        kind, affected = self._orchestrator._detect_recurring_structural_blocker(rollout_results)
        if not kind or not affected:
            return None
        return affected[0]

    def launch(
        self,
        *,
        repo_path: str,
        repo_context: RepoContext,
        issue_description: str,
        issue_plan: IssuePlan,
        planner: IssuePlanner,
        test_command: Optional[str],
        engine: RolloutEngine,
        rollout_results: list[RolloutResult],
        task_state_graph: Optional[TaskStateGraph],
        wallclock_deadline: Optional[float] = None,
    ) -> tuple[list[RolloutResult], list[dict[str, Any]]]:
        return self._orchestrator._launch_structural_recovery_followup(
            repo_path=repo_path,
            repo_context=repo_context,
            issue_description=issue_description,
            issue_plan=issue_plan,
            planner=planner,
            test_command=test_command,
            engine=engine,
            rollout_results=rollout_results,
            task_state_graph=task_state_graph,
            wallclock_deadline=wallclock_deadline,
        )


class CoverageGapFollowup(FollowupLoop):
    """Bounded retry from the strongest local-full-suite-pass rollout."""

    name = "coverage_gap"

    def should_run(
        self,
        *,
        rollout_results: list[RolloutResult],
        successful: list[RolloutResult],
        best_result: Optional[RolloutResult] = None,
    ) -> bool:
        if best_result is not None:
            return False
        if not bool(self._orchestrator.config.rollout.enable_residual_followup):
            return False
        if successful:
            return True
        return self.select_anchor(rollout_results) is not None

    def select_anchor(
        self,
        rollout_results: list[RolloutResult],
        *,
        best_result: Optional[RolloutResult] = None,
    ) -> Optional[RolloutResult]:
        return self._orchestrator._select_coverage_gap_anchor(rollout_results)

    def launch(
        self,
        *,
        repo_path: str,
        repo_context: RepoContext,
        issue_description: str,
        issue_plan: IssuePlan,
        planner: IssuePlanner,
        test_command: Optional[str],
        engine: RolloutEngine,
        rollout_results: list[RolloutResult],
        task_state_graph: Optional[TaskStateGraph],
        followup_round: int = 1,
        wallclock_deadline: Optional[float] = None,
    ) -> tuple[list[RolloutResult], list[dict[str, Any]], Optional[dict[str, Any]]]:
        return self._orchestrator._launch_coverage_gap_followup(
            repo_path=repo_path,
            repo_context=repo_context,
            issue_description=issue_description,
            issue_plan=issue_plan,
            planner=planner,
            test_command=test_command,
            engine=engine,
            rollout_results=rollout_results,
            task_state_graph=task_state_graph,
            followup_round=followup_round,
            wallclock_deadline=wallclock_deadline,
        )


class SelectionFollowup(FollowupLoop):
    """Residual-selection followup: extra rollouts when best candidate not accepted."""

    name = "selection"

    def should_run(
        self,
        *,
        rollout_results: list[RolloutResult],
        successful: list[RolloutResult],
        best_result: Optional[RolloutResult] = None,
    ) -> bool:
        if best_result is None:
            return False
        if not self._orchestrator._selected_result_needs_followup(best_result):
            return False
        return bool(self._orchestrator.config.rollout.enable_residual_followup)

    def select_anchor(
        self,
        rollout_results: list[RolloutResult],
        *,
        best_result: Optional[RolloutResult] = None,
    ) -> Optional[RolloutResult]:
        return best_result


__all__ = [
    "FollowupLoop",
    "NearMissFollowup",
    "StructuralRecoveryFollowup",
    "CoverageGapFollowup",
    "SelectionFollowup",
]
