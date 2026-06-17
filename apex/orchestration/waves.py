"""Wave-execution helpers.

Phase 3.2: factored out of the monolithic orchestrator. The actual
loop bodies (``_execute_with_dynamic_transitions`` and
``_execute_progressive_rollout_plan``) remain on the orchestrator
instance because they pull on a wide slice of state (frontier search,
task-state graph, planner, engine, etc.); this module exposes thin
function wrappers so callers and tests can import them by name from
the new package.
"""

from __future__ import annotations

from typing import Any, Optional

from ..planning.manager import IssuePlan, IssuePlanner
from ..preprocessing.repo_analyzer import RepoContext
from ..rollout.engine import RolloutEngine, RolloutResult, WorkspaceSeed
from ..selection.verifier import BaselineResult
from ..task_state_graph import TaskStateGraph


def execute_with_dynamic_transitions(
    orchestrator: Any,
    *,
    repo_path: str,
    repo_context: RepoContext,
    issue_description: str,
    issue_plan: IssuePlan,
    planner: IssuePlanner,
    initial_strategy: Any,
    test_command: Optional[str],
    verification_test_command: Optional[str] = None,
    engine: Optional[RolloutEngine] = None,
    transitions: Optional[list[dict[str, Any]]] = None,
    baseline_result: Optional[BaselineResult] = None,
    task_state_graph: Optional[TaskStateGraph] = None,
    benchmark_metadata: Optional[dict[str, Any]] = None,
) -> tuple[
    list[RolloutResult],
    IssuePlan,
    list[dict[str, Any]],
    Optional[dict[str, Any]],
]:
    """Run the dynamic-transition wave loop until acceptance or escalation
    exhaustion. Delegates to the orchestrator instance which holds the
    cross-cutting state (config, planner, etc.).
    """
    return orchestrator._execute_with_dynamic_transitions(
        repo_path=repo_path,
        repo_context=repo_context,
        issue_description=issue_description,
        issue_plan=issue_plan,
        planner=planner,
        initial_strategy=initial_strategy,
        test_command=test_command,
        verification_test_command=verification_test_command,
        engine=engine,
        transitions=transitions,
        baseline_result=baseline_result,
        task_state_graph=task_state_graph,
        benchmark_metadata=benchmark_metadata,
    )


def execute_progressive_rollout_plan(
    orchestrator: Any,
    *,
    repo_path: str,
    repo_context: RepoContext,
    issue_description: str,
    issue_plan: IssuePlan,
    planner: IssuePlanner,
    test_command: Optional[str],
    engine: Optional[RolloutEngine] = None,
    rollout_id_offset: int = 0,
    task_state_graph: Optional[TaskStateGraph] = None,
    initial_workspace_seed: Optional[WorkspaceSeed] = None,
) -> tuple[
    list[RolloutResult],
    IssuePlan,
    int,
    list[dict[str, Any]],
    Optional[dict[str, Any]],
]:
    """Run the progressive (multi-wave) rollout plan loop."""
    return orchestrator._execute_progressive_rollout_plan(
        repo_path=repo_path,
        repo_context=repo_context,
        issue_description=issue_description,
        issue_plan=issue_plan,
        planner=planner,
        test_command=test_command,
        engine=engine,
        rollout_id_offset=rollout_id_offset,
        task_state_graph=task_state_graph,
        initial_workspace_seed=initial_workspace_seed,
    )


__all__ = [
    "execute_with_dynamic_transitions",
    "execute_progressive_rollout_plan",
]
