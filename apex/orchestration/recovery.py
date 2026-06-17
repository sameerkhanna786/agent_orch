"""Recovery / followup orchestration glue.

Phase 3.2: previously inline inside ``ApexOrchestrator.solve()``.
The four followup loops (near-miss, structural-recovery, coverage-gap,
selection) used to live as a flat sequence of conditional blocks
inside ``solve()``. They are now factored into discrete functions so
``solve()`` reads as a thin coordinator and each follow-up step is
testable in isolation.

Each entry point takes the orchestrator instance and the run-state
locals it needs, mutates the local lists in-place where appropriate,
and returns updated successful + best-so-far state.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..planning.manager import IssuePlan, IssuePlanner
from ..preprocessing.repo_analyzer import RepoContext
from ..rollout.engine import RolloutEngine, RolloutResult
from ..selection.verifier import BaselineResult, PatchVerifier
from ..task_state_graph import TaskStateGraph
from .followups import (
    CoverageGapFollowup,
    NearMissFollowup,
    SelectionFollowup,
    StructuralRecoveryFollowup,
)

logger = logging.getLogger("apex.orchestrator")


def run_near_miss_recovery(
    orchestrator: Any,
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
    successful: list[RolloutResult],
    orchestration_transitions: list[dict[str, Any]],
    wallclock_deadline: Optional[float] = None,
) -> list[RolloutResult]:
    """Run the near-miss followup loop. Mutates lists in place; returns
    refreshed ``successful`` slice."""
    loop = NearMissFollowup(orchestrator)
    if not loop.should_run(
        rollout_results=rollout_results,
        successful=successful,
    ):
        return successful
    near_miss_results, near_miss_transitions = loop.launch(
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
    if near_miss_transitions:
        orchestration_transitions.extend(near_miss_transitions)
    if near_miss_results:
        rollout_results.extend(near_miss_results)
        successful = [r for r in rollout_results if r.success and r.patch]
    return successful


def run_structural_recovery(
    orchestrator: Any,
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
    successful: list[RolloutResult],
    orchestration_transitions: list[dict[str, Any]],
    wallclock_deadline: Optional[float] = None,
) -> list[RolloutResult]:
    loop = StructuralRecoveryFollowup(orchestrator)
    if not loop.should_run(
        rollout_results=rollout_results,
        successful=successful,
    ):
        return successful
    recovery_results, recovery_transitions = loop.launch(
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
    if recovery_transitions:
        orchestration_transitions.extend(recovery_transitions)
    if recovery_results:
        rollout_results.extend(recovery_results)
        successful = [r for r in rollout_results if r.success and r.patch]
    return successful


def run_coverage_gap_recovery(
    orchestrator: Any,
    *,
    repo_path: str,
    repo_context: RepoContext,
    issue_description: str,
    issue_plan: IssuePlan,
    planner: IssuePlanner,
    test_command: Optional[str],
    verification_test_command: Optional[str],
    verifier: PatchVerifier,
    baseline_result: Optional[BaselineResult],
    engine: RolloutEngine,
    rollout_results: list[RolloutResult],
    task_state_graph: Optional[TaskStateGraph],
    successful: list[RolloutResult],
    best_result: Optional[RolloutResult],
    orchestration_transitions: list[dict[str, Any]],
    search_summary: Optional[dict[str, Any]],
    wallclock_deadline: Optional[float] = None,
) -> tuple[Optional[RolloutResult], list[RolloutResult], Optional[dict[str, Any]]]:
    """Loop the coverage-gap followup until best is found or rounds exhausted."""
    loop = CoverageGapFollowup(orchestrator)
    coverage_gap_followup_round = 0
    max_rounds = int(orchestrator.config.orchestration.max_coverage_gap_followup_rounds)
    while (
        loop.should_run(
            rollout_results=rollout_results,
            successful=successful,
            best_result=best_result,
        )
        and coverage_gap_followup_round < max_rounds
    ):
        coverage_gap_followup_round += 1
        (
            coverage_gap_results,
            coverage_gap_transitions,
            coverage_gap_search_summary,
        ) = loop.launch(
            repo_path=repo_path,
            repo_context=repo_context,
            issue_description=issue_description,
            issue_plan=issue_plan,
            planner=planner,
            test_command=test_command,
            engine=engine,
            rollout_results=rollout_results,
            task_state_graph=task_state_graph,
            followup_round=coverage_gap_followup_round,
            wallclock_deadline=wallclock_deadline,
        )
        if coverage_gap_transitions:
            orchestration_transitions.extend(coverage_gap_transitions)
        search_summary = orchestrator._merge_search_summary(
            search_summary,
            coverage_gap_search_summary,
        )
        if not coverage_gap_results:
            break
        rollout_results.extend(coverage_gap_results)
        successful = [r for r in rollout_results if r.success and r.patch]
        best_result = orchestrator._select_best_patch(
            repo_path=repo_path,
            rollout_results=rollout_results,
            issue_description=issue_description,
            test_command=verification_test_command,
            verifier=verifier,
            baseline_result=baseline_result,
            issue_plan=issue_plan,
        )
    return best_result, successful, search_summary


def run_selection_followups(
    orchestrator: Any,
    *,
    repo_path: str,
    repo_context: RepoContext,
    issue_description: str,
    issue_plan: IssuePlan,
    planner: IssuePlanner,
    test_command: Optional[str],
    verification_test_command: Optional[str],
    verifier: PatchVerifier,
    baseline_result: Optional[BaselineResult],
    engine: RolloutEngine,
    rollout_results: list[RolloutResult],
    task_state_graph: Optional[TaskStateGraph],
    best_result: Optional[RolloutResult],
    orchestration_transitions: list[dict[str, Any]],
    search_summary: Optional[dict[str, Any]],
    wallclock_deadline: Optional[float] = None,
) -> tuple[Optional[RolloutResult], IssuePlan, Optional[dict[str, Any]]]:
    """Drive the residual selection-followup loop with adaptive budget."""
    loop = SelectionFollowup(orchestrator)
    followup_round = 0
    followup_anchor = best_result
    if followup_anchor is None:
        followup_anchor = orchestrator._select_invalid_selection_followup_anchor(rollout_results)
    while loop.should_run(
        rollout_results=rollout_results,
        successful=[r for r in rollout_results if r.success and r.patch],
        best_result=followup_anchor,
    ) and followup_round < orchestrator._effective_max_selection_followup_rounds(followup_anchor):
        if loop.is_exhausted(
            rollout_results=rollout_results,
            followup_round=followup_round,
        ):
            break

        additional_rollouts = planner.recommend_followup_rollouts(
            issue_plan,
            rollout_results,
            best_candidate=followup_anchor,
            current_total_rollouts=len(rollout_results),
        )
        additional_rollouts = orchestrator._cap_followup_rollouts_for_token_budget(
            rollout_results,
            requested_rollouts=additional_rollouts,
        )
        if additional_rollouts <= 0:
            break

        residual_summary = orchestrator._build_selection_residual_summary(
            issue_plan=issue_plan,
            rollout_results=rollout_results,
            best_result=followup_anchor,
        )
        residual_focus_files = orchestrator._selection_residual_focus_files(
            issue_plan=issue_plan,
            rollout_results=rollout_results,
            best_result=followup_anchor,
        )
        followup_plan = planner.build_followup_plan(
            issue_plan,
            repo_context,
            rollout_results,
            additional_rollouts=additional_rollouts,
            residual_summary=residual_summary,
            residual_focus_files=residual_focus_files,
            task_state_context=issue_plan.task_state_context,
        )
        if not followup_plan.rollout_briefs:
            break

        orchestrator._relaunch_with_alternative_backend_or_sandbox(
            followup_plan,
            rollout_results,
        )

        followup_round += 1
        transition = orchestrator._build_selection_followup_transition(
            current_plan=issue_plan,
            followup_plan=followup_plan,
            best_result=followup_anchor,
            followup_round=followup_round,
        )
        logger.info(
            "Launching residual follow-up round %s with %s additional rollouts: %s",
            followup_round,
            additional_rollouts,
            residual_summary,
        )
        orchestration_transitions.append(transition)
        issue_plan = followup_plan
        orchestrator._save_issue_plan(issue_plan)
        followup_seed = orchestrator._select_rollout_seed(
            planner,
            rollout_results,
            preferred_result=followup_anchor,
        )

        (
            followup_results,
            issue_plan,
            _next_rollout_id,
            followup_transitions,
            followup_search_summary,
        ) = orchestrator._execute_progressive_rollout_plan(
            repo_path=repo_path,
            repo_context=repo_context,
            issue_description=issue_description,
            issue_plan=issue_plan,
            planner=planner,
            test_command=test_command,
            engine=engine,
            rollout_id_offset=orchestrator._next_rollout_id_after(rollout_results),
            task_state_graph=task_state_graph,
            initial_workspace_seed=followup_seed,
            wallclock_deadline=wallclock_deadline,
        )
        orchestration_transitions.extend(followup_transitions)
        search_summary = orchestrator._merge_search_summary(
            search_summary,
            followup_search_summary,
        )
        rollout_results.extend(followup_results)
        best_result = orchestrator._select_best_patch(
            repo_path=repo_path,
            rollout_results=rollout_results,
            issue_description=issue_description,
            test_command=verification_test_command,
            verifier=verifier,
            baseline_result=baseline_result,
            issue_plan=issue_plan,
        )
        issue_plan = orchestrator._refresh_task_state_from_selected_result(
            issue_plan,
            task_state_graph,
            best_result,
        )
        followup_anchor = best_result
        if followup_anchor is None:
            followup_anchor = orchestrator._select_invalid_selection_followup_anchor(
                rollout_results
            )

    return best_result, issue_plan, search_summary


__all__ = [
    "run_near_miss_recovery",
    "run_structural_recovery",
    "run_coverage_gap_recovery",
    "run_selection_followups",
]
