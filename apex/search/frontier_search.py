"""
Explicit frontier search over branchable code states.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..acceptance import (
    quick_verification_signal_score,
    rollout_has_authoritative_scoring_stop_signal,
    rollout_has_preemptive_authoritative_scoring_request,
    rollout_has_preemptive_completion,
    rollout_has_repairable_near_miss,
)
from ..controller_policy import ShadowPolicyOption, build_shadow_policy_trace
from ..controller_trace import append_controller_decision
from ..core.config import ApexConfig, SearchMode
from ..planning.manager import IssuePlan, IssuePlanner, RolloutBrief, _rollout_brief_allocation_key
from ..preprocessing.repo_analyzer import RepoContext
from ..rollout.engine import (
    RolloutEngine,
    RolloutRequest,
    RolloutResult,
    WorkspaceSeed,
    _runtime_outer_task_parallelism,
    build_workspace_seed_from_rollout_result,
)
from ..task_state_graph import TaskStateGraph

logger = logging.getLogger("apex.search")


def _near_miss_minimum_signal_score(config: Optional[ApexConfig]) -> float:
    try:
        near_miss_threshold = float(
            getattr(
                getattr(config, "orchestration", None),
                "adaptive_followup_near_miss_pass_rate",
                0.95,
            )
            or 0.95
        )
    except (TypeError, ValueError):
        near_miss_threshold = 0.95
    return max(0.999, near_miss_threshold)


def _frontier_rollout_stop_signal(
    result: RolloutResult,
    config: Optional[ApexConfig],
) -> Optional[str]:
    if rollout_has_preemptive_completion(result):
        return "preemptive_completion"
    if rollout_has_repairable_near_miss(
        result,
        minimum_signal_score=_near_miss_minimum_signal_score(config),
        residual_fraction_cap=0.001,
        max_residual_count=50,
    ):
        return "repairable_near_miss"
    if rollout_has_preemptive_authoritative_scoring_request(result):
        return "preemptive_authoritative_scoring"
    if rollout_has_authoritative_scoring_stop_signal(result):
        return "requires_authoritative_scoring"
    return None


def _next_rollout_id_after(
    rollout_results: list[RolloutResult],
    *,
    fallback: int = 0,
) -> int:
    try:
        next_id = max(0, int(fallback))
    except (TypeError, ValueError):
        next_id = 0
    for result in list(rollout_results or []):
        rollout_id = getattr(result, "rollout_id", None)
        if isinstance(rollout_id, bool):
            continue
        try:
            parsed = int(rollout_id)
        except (TypeError, ValueError):
            continue
        next_id = max(next_id, parsed + 1)
    return next_id


def _build_preemptive_completion_stop_on_result(
    config: Optional[ApexConfig] = None,
):
    def stop_on_result(result: RolloutResult) -> bool:
        setattr(stop_on_result, "preempt_active_rollouts", False)
        setattr(stop_on_result, "continue_dispatch_after_nonpreemptive_stop", False)
        setattr(stop_on_result, "drain_active_rollouts_after_nonpreemptive_stop", False)
        stop_signal = _frontier_rollout_stop_signal(result, config)
        if stop_signal:
            if stop_signal in {
                "preemptive_completion",
                "repairable_near_miss",
                "preemptive_authoritative_scoring",
            }:
                setattr(stop_on_result, "preempt_active_rollouts", True)
            elif stop_signal == "requires_authoritative_scoring":
                setattr(stop_on_result, "drain_active_rollouts_after_nonpreemptive_stop", True)
            return True
        return False

    setattr(stop_on_result, "preempt_active_rollouts", False)
    setattr(stop_on_result, "continue_dispatch_after_nonpreemptive_stop", False)
    setattr(stop_on_result, "drain_active_rollouts_after_nonpreemptive_stop", False)
    return stop_on_result


def _clamp(value: float, *, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _search_fallback_rollout_brief(
    issue_plan: IssuePlan,
    rollout_id: int,
) -> RolloutBrief:
    summary = (issue_plan.summary or "").strip() or "Resolve the issue."
    return RolloutBrief(
        title=f"Search Rollout {rollout_id + 1}",
        goal=summary,
        focus_files=list(issue_plan.relevant_files[:8]),
        success_criteria=list(issue_plan.success_criteria[:6]),
        prompt_hint="Search expansion produced no rollout briefs; use the issue summary directly.",
    )


def _search_brief_allocator_key(brief: RolloutBrief) -> str:
    key = _rollout_brief_allocation_key(brief)
    if key:
        return key
    title = str(getattr(brief, "title", "") or "").strip()
    goal = str(getattr(brief, "goal", "") or "").strip()
    return f"title={title}|goal={goal}"


def _brief_requests_root_execution_anchor(brief: RolloutBrief) -> tuple[bool, str]:
    policy = brief.search_policy if isinstance(brief.search_policy, dict) else {}
    delegation_policy = brief.delegation_policy if isinstance(brief.delegation_policy, dict) else {}
    mode = str(policy.get("mode") or "").strip().lower()
    if bool(policy.get("preserve_agent_mode")):
        return True, "preserve_agent_mode"
    if mode == "agentless_pipeline":
        return True, "agentless_pipeline"
    if "enabled" in delegation_policy and not brief.delegation_enabled("patcher"):
        return True, "explicit_single_worker"
    return False, ""


@dataclass
class SearchState:
    """One search state rooted at a branchable checkpoint and local task-state graph."""

    state_id: str
    parent_state_id: Optional[str]
    depth: int
    graph_snapshot: dict[str, Any]
    task_state_context: dict[str, Any]
    frontier_targets: list[dict[str, Any]]
    checkpoint_seed: Optional[WorkspaceSeed] = None
    visit_count: int = 0
    value_sum: float = 0.0
    value_count: int = 0
    best_value: float = 0.0
    terminal: bool = False
    incoming_rollout_id: Optional[int] = None
    incoming_target_id: Optional[str] = None
    cumulative_reward: float = 0.0

    @property
    def average_value(self) -> float:
        if self.value_count <= 0:
            return 0.0
        return self.value_sum / self.value_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "parent_state_id": self.parent_state_id,
            "depth": self.depth,
            "task_state_context": dict(self.task_state_context),
            "frontier_targets": [dict(item) for item in self.frontier_targets],
            "checkpoint_seed": (
                self.checkpoint_seed.to_dict() if self.checkpoint_seed is not None else None
            ),
            "visit_count": self.visit_count,
            "value_sum": round(self.value_sum, 4),
            "value_count": self.value_count,
            "average_value": round(self.average_value, 4),
            "best_value": round(self.best_value, 4),
            "terminal": self.terminal,
            "incoming_rollout_id": self.incoming_rollout_id,
            "incoming_target_id": self.incoming_target_id,
            "cumulative_reward": round(self.cumulative_reward, 4),
        }


@dataclass
class SearchEdgeStats:
    """Visit/value statistics for one state-target edge."""

    state_id: str
    target_id: str
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    best_value: float = 0.0
    direct_visit_count: int = 0
    direct_value_sum: float = 0.0
    virtual_loss: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    mean_cost: float = 0.0
    last_rollout_id: Optional[int] = None
    child_state_ids: list[str] = field(default_factory=list)

    @property
    def average_value(self) -> float:
        if self.visit_count <= 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def direct_average_value(self) -> float:
        if self.direct_visit_count <= 0:
            return 0.0
        return self.direct_value_sum / self.direct_visit_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "target_id": self.target_id,
            "prior": round(self.prior, 4),
            "visit_count": self.visit_count,
            "value_sum": round(self.value_sum, 4),
            "direct_visit_count": self.direct_visit_count,
            "direct_value_sum": round(self.direct_value_sum, 4),
            "direct_average_value": round(self.direct_average_value, 4),
            "best_value": round(self.best_value, 4),
            "virtual_loss": round(self.virtual_loss, 4),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "mean_cost": round(self.mean_cost, 4),
            "last_rollout_id": self.last_rollout_id,
            "child_state_ids": list(self.child_state_ids),
        }


@dataclass
class GlobalTargetStats:
    """Cross-state statistics for one frontier target."""

    target_id: str
    visit_count: int = 0
    value_sum: float = 0.0
    best_value: float = 0.0
    failure_count: int = 0

    @property
    def average_value(self) -> float:
        if self.visit_count <= 0:
            return 0.0
        return self.value_sum / self.visit_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "visit_count": self.visit_count,
            "value_sum": round(self.value_sum, 4),
            "best_value": round(self.best_value, 4),
            "failure_count": self.failure_count,
        }


@dataclass
class SearchExpansionCandidate:
    """One candidate action to expand next."""

    state_id: str
    target: dict[str, Any]
    score: float
    prior: float
    q_value: float
    exploration_bonus: float
    optimistic_bound: float
    reason: str

    @property
    def target_id(self) -> str:
        return str(self.target.get("target_id") or "")


@dataclass
class FrontierSearchRunResult:
    """Output of one frontier-search attempt."""

    rollout_results: list[RolloutResult] = field(default_factory=list)
    issue_plan: Optional[IssuePlan] = None
    next_rollout_id: int = 0
    transitions: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


class FrontierSearchController:
    """Best-first or PUCT-style search over frontier targets."""

    def __init__(
        self,
        config: ApexConfig,
        planner: IssuePlanner,
        repo_context: RepoContext,
        *,
        issue_description: str,
        test_command: Optional[str],
        output_dir: str | Path,
    ) -> None:
        self.config = config
        self.planner = planner
        self.repo_context = repo_context
        self.issue_description = issue_description
        self.test_command = test_command
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._state_index = 0

    def _search_shadow_enabled(self) -> bool:
        policy = getattr(self.config.search, "shadow_policy", None)
        return bool(getattr(policy, "enabled", True))

    def _search_shadow_limit(self) -> int:
        policy = getattr(self.config.search, "shadow_policy", None)
        limit = int(getattr(policy, "max_logged_options", 3) or 3)
        return max(1, limit)

    def _root_batch_request_budget(
        self,
        issue_plan: IssuePlan,
        *,
        remaining_budget: int,
        max_parallel: int,
        observation_cap: Optional[int] = None,
    ) -> int:
        planner_metadata = (
            issue_plan.planner_metadata if isinstance(issue_plan.planner_metadata, dict) else {}
        )
        desired = max_parallel
        for key in ("portfolio_rollout_floor", "portfolio_seed_profile_count"):
            try:
                value = int(planner_metadata.get(key) or 0)
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                desired = max(desired, value)
        if observation_cap is not None:
            desired = min(desired, max(1, int(observation_cap)))
        return max(1, min(remaining_budget, desired))

    def _root_batch_observation_cap(self, *, max_parallel: int) -> Optional[int]:
        """Return the live worker cap that can be observed before the next search turn."""

        try:
            global_worker_budget = int(
                getattr(self.config.rollout, "global_parallel_worker_budget", 0) or 0
            )
        except (TypeError, ValueError):
            global_worker_budget = 0
        if global_worker_budget <= 0:
            return None
        try:
            configured_outer_task_parallelism = int(
                getattr(self.config.benchmark, "task_parallelism", 1) or 1
            )
        except (TypeError, ValueError):
            configured_outer_task_parallelism = 1
        outer_task_parallelism, _active_count, _waiting_count, _applied = (
            _runtime_outer_task_parallelism(configured_outer_task_parallelism)
        )
        if outer_task_parallelism <= 1:
            return None
        resource_cap = max(1, global_worker_budget // max(1, outer_task_parallelism))
        return max(1, min(max(1, int(max_parallel)), resource_cap))

    def _build_batch_rollout_requests(
        self,
        *,
        prepared_expansions: list[
            tuple[
                SearchExpansionCandidate,
                IssuePlan,
                SearchState,
                TaskStateGraph,
                dict[str, Any],
            ]
        ],
        batch_request_budget: int,
        remaining_budget: int,
        next_rollout_id: int,
        edge_stats: dict[tuple[str, str], SearchEdgeStats],
        root_issue_plan: Optional[IssuePlan] = None,
    ) -> tuple[
        list[RolloutRequest],
        dict[
            int,
            tuple[
                SearchExpansionCandidate,
                IssuePlan,
                SearchState,
                TaskStateGraph,
                dict[str, Any],
            ],
        ],
        int,
    ]:
        request_limit = min(batch_request_budget, remaining_budget)
        requests: list[RolloutRequest] = []
        request_context: dict[
            int,
            tuple[
                SearchExpansionCandidate,
                IssuePlan,
                SearchState,
                TaskStateGraph,
                dict[str, Any],
            ],
        ] = {}
        candidate_payloads: list[
            tuple[
                SearchExpansionCandidate,
                IssuePlan,
                SearchState,
                TaskStateGraph,
                dict[str, Any],
                list[RolloutBrief],
            ]
        ] = []
        for candidate, expansion_plan, state, local_graph, base_metadata in prepared_expansions:
            available_briefs = list(expansion_plan.rollout_briefs or [])
            if not available_briefs:
                available_briefs = [_search_fallback_rollout_brief(expansion_plan, next_rollout_id)]
            candidate_payloads.append(
                (
                    candidate,
                    expansion_plan,
                    state,
                    local_graph,
                    base_metadata,
                    available_briefs,
                )
            )

        brief_positions = [0] * len(candidate_payloads)
        used_candidate_indices: set[int] = set()
        seen_allocator_arms: set[str] = set()

        def target_affinity(brief: RolloutBrief, target: dict[str, Any]) -> float:
            target_files = {
                str(path) for path in list(target.get("file_paths") or []) if str(path).strip()
            }
            target_tests = {
                str(test_id)
                for test_id in list(target.get("test_ids") or [])
                if str(test_id).strip()
            }
            brief_files = {str(path) for path in list(brief.focus_files or []) if str(path).strip()}
            policy = brief.search_policy if isinstance(brief.search_policy, dict) else {}
            mode = str(policy.get("mode") or "").strip().lower()
            score = float(len(target_files.intersection(brief_files)))
            if target_tests and mode in {"test_rooted", "api_contract", "invariant_guard"}:
                score += 0.5
            return score

        def clone_plan_for_anchor(
            source_plan: IssuePlan,
            brief: RolloutBrief,
        ) -> IssuePlan:
            cloned_plan = IssuePlan.from_dict(source_plan.to_dict())
            cloned_plan.rollout_briefs = [RolloutBrief.from_dict(brief.to_dict())]
            return cloned_plan

        def allocate_root_execution_anchors() -> None:
            nonlocal next_rollout_id
            if root_issue_plan is None or not candidate_payloads:
                return
            anchor_items: list[tuple[RolloutBrief, str]] = []
            for root_brief in list(root_issue_plan.rollout_briefs or []):
                should_anchor, anchor_reason = _brief_requests_root_execution_anchor(root_brief)
                if should_anchor:
                    anchor_items.append((root_brief, anchor_reason))
            primary_singles = [
                item for item in anchor_items[:1] if item[1] == "explicit_single_worker"
            ]
            preserve_items = [
                item
                for item in anchor_items
                if item[1] in {"preserve_agent_mode", "agentless_pipeline"}
            ]
            primary_ids = {id(item[0]) for item in primary_singles}
            preserve_ids = {id(item[0]) for item in preserve_items}
            remaining_items = [
                item
                for item in anchor_items
                if id(item[0]) not in primary_ids and id(item[0]) not in preserve_ids
            ]
            for root_brief, anchor_reason in [
                *primary_singles,
                *preserve_items,
                *remaining_items,
            ]:
                if len(requests) >= request_limit:
                    break
                allocation_key = _search_brief_allocator_key(root_brief)
                if allocation_key and allocation_key in seen_allocator_arms:
                    continue
                (
                    candidate_index,
                    (
                        candidate,
                        expansion_plan,
                        state,
                        local_graph,
                        base_metadata,
                        _available_briefs,
                    ),
                ) = max(
                    enumerate(candidate_payloads),
                    key=lambda item: target_affinity(root_brief, item[1][0].target),
                )
                state.visit_count += 1
                edge = self._edge_stats(edge_stats, candidate.state_id, candidate.target)
                edge.virtual_loss += max(0.0, float(self.config.search.virtual_loss))
                request_metadata = dict(base_metadata)
                request_metadata.update(
                    {
                        "search_brief_rank": -1,
                        "search_brief_title": root_brief.title,
                        "search_root_execution_anchor": True,
                        "search_root_execution_anchor_reason": anchor_reason,
                    }
                )
                if allocation_key:
                    request_metadata["search_allocator_arm"] = allocation_key
                    seen_allocator_arms.add(allocation_key)
                rollout_id = next_rollout_id
                next_rollout_id += 1
                anchor_plan = clone_plan_for_anchor(expansion_plan, root_brief)
                requests.append(
                    RolloutRequest(
                        rollout_id=rollout_id,
                        issue_plan=anchor_plan,
                        rollout_brief=anchor_plan.rollout_briefs[0],
                        workspace_seed=state.checkpoint_seed,
                        metadata=request_metadata,
                    )
                )
                request_context[rollout_id] = (
                    candidate,
                    anchor_plan,
                    state,
                    local_graph,
                    dict(request_metadata),
                )
                used_candidate_indices.add(candidate_index)

        allocate_root_execution_anchors()

        def allocate_for_candidate(index: int, *, allow_duplicate_arm: bool) -> bool:
            nonlocal next_rollout_id
            if len(requests) >= request_limit:
                return False
            (
                candidate,
                expansion_plan,
                state,
                local_graph,
                base_metadata,
                available_briefs,
            ) = candidate_payloads[index]
            start_rank = brief_positions[index]
            chosen_rank: Optional[int] = None
            chosen_brief: Optional[RolloutBrief] = None
            chosen_key = ""
            for rank in range(start_rank, len(available_briefs)):
                brief = available_briefs[rank]
                allocation_key = _search_brief_allocator_key(brief)
                if not allow_duplicate_arm and allocation_key in seen_allocator_arms:
                    continue
                chosen_rank = rank
                chosen_brief = brief
                chosen_key = allocation_key
                break
            if chosen_brief is None or chosen_rank is None:
                return False

            brief_positions[index] = chosen_rank + 1
            used_candidate_indices.add(index)
            seen_allocator_arms.add(chosen_key)
            state.visit_count += 1
            edge = self._edge_stats(edge_stats, candidate.state_id, candidate.target)
            edge.virtual_loss += max(0.0, float(self.config.search.virtual_loss))
            request_metadata = dict(base_metadata)
            request_metadata.update(
                {
                    "search_brief_rank": chosen_rank,
                    "search_brief_title": chosen_brief.title,
                }
            )
            policy = (
                chosen_brief.search_policy if isinstance(chosen_brief.search_policy, dict) else {}
            )
            allocation_key = str(policy.get("allocator_arm") or "").strip()
            if allocation_key:
                request_metadata["search_allocator_arm"] = allocation_key
            rollout_id = next_rollout_id
            next_rollout_id += 1
            requests.append(
                RolloutRequest(
                    rollout_id=rollout_id,
                    issue_plan=expansion_plan,
                    rollout_brief=chosen_brief,
                    workspace_seed=state.checkpoint_seed,
                    metadata=request_metadata,
                )
            )
            request_context[rollout_id] = (
                candidate,
                expansion_plan,
                state,
                local_graph,
                dict(request_metadata),
            )
            return True

        # First give each candidate one chance to claim a distinct allocator arm.
        for index in range(len(candidate_payloads)):
            if len(requests) >= request_limit:
                break
            allocate_for_candidate(index, allow_duplicate_arm=False)

        # Then make sure every selected candidate can still launch even if arm diversity is exhausted.
        for index in range(len(candidate_payloads)):
            if len(requests) >= request_limit:
                break
            if index in used_candidate_indices:
                continue
            allocate_for_candidate(index, allow_duplicate_arm=True)

        # If the search batch wants more requests than candidates, keep filling greedily,
        # preferring unseen allocator arms before allowing duplicates.
        while len(requests) < request_limit:
            progress = False
            for allow_duplicate_arm in (False, True):
                for index in range(len(candidate_payloads)):
                    if len(requests) >= request_limit:
                        break
                    if allocate_for_candidate(index, allow_duplicate_arm=allow_duplicate_arm):
                        progress = True
                if len(requests) >= request_limit:
                    break
            if not progress:
                break

        return requests, request_context, next_rollout_id

    def run(
        self,
        *,
        issue_plan: IssuePlan,
        task_state_graph: TaskStateGraph,
        engine: RolloutEngine,
        rollout_budget: int,
        rollout_id_offset: int = 0,
        root_workspace_seed: Optional[WorkspaceSeed] = None,
        wallclock_deadline: Optional[float] = None,
    ) -> FrontierSearchRunResult:
        mode = self.config.search.mode
        if mode == SearchMode.OFF:
            raise ValueError("FrontierSearchController requires search mode to be enabled.")

        budget = max(1, int(rollout_budget))
        if self.config.search.max_expansions > 0:
            budget = min(budget, int(self.config.search.max_expansions))
        max_parallel = max(1, min(self.config.rollout.parallel_workers, budget))

        root_graph = task_state_graph.clone()
        global_graph = task_state_graph
        root_state = self._make_state(
            parent_state_id=None,
            depth=0,
            graph=root_graph,
            checkpoint_seed=root_workspace_seed,
            incoming_rollout_id=None,
            incoming_target_id=None,
            cumulative_reward=0.0,
        )
        states: dict[str, SearchState] = {root_state.state_id: root_state}
        edge_stats: dict[tuple[str, str], SearchEdgeStats] = {}
        global_target_stats: dict[str, GlobalTargetStats] = {}
        transitions: list[dict[str, Any]] = []
        shadow_policy_log: list[dict[str, Any]] = []
        all_results: list[RolloutResult] = []
        best_reward = float("-inf")
        best_reward_rollout_id: Optional[int] = None
        best_total_value = float("-inf")
        best_rollout_id: Optional[int] = None
        remaining_budget = budget
        next_rollout_id = rollout_id_offset
        stopped_early = False
        stop_reason: Optional[str] = None

        while remaining_budget > 0:
            candidates = self._rank_candidates(
                states=states,
                edge_stats=edge_stats,
                global_target_stats=global_target_stats,
            )
            if not candidates:
                break
            should_stop, stop_reason = self._should_stop_search(
                candidates=candidates,
                best_total_value=best_total_value,
                states=states,
            )
            if should_stop:
                stopped_early = True
                break

            root_batch_observation_cap: Optional[int] = None
            root_batch = bool(
                not all_results
                and candidates
                and all(candidate.state_id == root_state.state_id for candidate in candidates)
            )
            selection_parallel = max_parallel
            if root_batch:
                root_batch_observation_cap = self._root_batch_observation_cap(
                    max_parallel=max_parallel
                )
                if root_batch_observation_cap is not None:
                    selection_parallel = min(selection_parallel, root_batch_observation_cap)
            selected = candidates[: min(len(candidates), selection_parallel, remaining_budget)]
            if self._search_shadow_enabled():
                option_limit = max(self._search_shadow_limit(), len(selected))
                ranked_options: list[ShadowPolicyOption] = []
                selected_target_ids = {candidate.target_id for candidate in selected}
                for candidate in candidates[: max(option_limit, len(selected))]:
                    ranked_options.append(
                        ShadowPolicyOption(
                            option_id=candidate.target_id,
                            score=float(candidate.score),
                            rationale=candidate.reason,
                            selected=candidate.target_id in selected_target_ids,
                            category="search_target",
                            metadata={
                                "state_id": candidate.state_id,
                                "kind": str(candidate.target.get("kind") or ""),
                                "description": str(candidate.target.get("description") or ""),
                                "frontier_score": float(
                                    candidate.target.get("frontier_score") or 0.0
                                ),
                            },
                        )
                    )
                if ranked_options:
                    trace = build_shadow_policy_trace(
                        decision=f"search_candidate_rank_{len(shadow_policy_log) + 1}",
                        options=ranked_options,
                        max_logged_options=self._search_shadow_limit(),
                    )
                    trace["selected_batch_target_ids"] = list(selected_target_ids)
                    shadow_policy_log.append(trace)
                    append_controller_decision(
                        self.config,
                        output_dir=self.output_dir,
                        stage="search",
                        decision_type="candidate_rank",
                        chosen_option=",".join(sorted(selected_target_ids)),
                        feature_view={
                            "remaining_budget": float(remaining_budget),
                            "candidate_count": float(len(candidates)),
                            "selected_count": float(len(selected)),
                            "best_total_value": float(best_total_value)
                            if best_total_value > float("-inf")
                            else 0.0,
                        },
                        options=ranked_options,
                        metadata={
                            "selected_batch_target_ids": list(selected_target_ids),
                            "root_state_id": root_state.state_id,
                        },
                    )
            batch_request_budget = len(selected)
            if (
                not all_results
                and selected
                and all(candidate.state_id == root_state.state_id for candidate in selected)
            ):
                batch_request_budget = self._root_batch_request_budget(
                    issue_plan,
                    remaining_budget=remaining_budget,
                    max_parallel=max_parallel,
                    observation_cap=root_batch_observation_cap,
                )

            prepared_expansions: list[
                tuple[
                    SearchExpansionCandidate,
                    IssuePlan,
                    SearchState,
                    TaskStateGraph,
                    dict[str, Any],
                ]
            ] = []
            for candidate in selected:
                state = states[candidate.state_id]
                local_graph = TaskStateGraph.from_dict(state.graph_snapshot)
                task_state_context = dict(state.task_state_context)
                source_plan = IssuePlan.from_dict(issue_plan.to_dict())
                expansion_plan = self.planner.build_search_expansion_plan(
                    source_plan,
                    self.repo_context,
                    frontier_target=candidate.target,
                    task_state_context=task_state_context,
                    search_depth=state.depth + 1,
                    brief_limit=batch_request_budget,
                )
                prepared_expansions.append(
                    (
                        candidate,
                        expansion_plan,
                        state,
                        local_graph,
                        {
                            "search_mode": mode.value,
                            "search_state_id": state.state_id,
                            "search_parent_state_id": state.parent_state_id,
                            "search_target_id": candidate.target_id,
                            "search_target_kind": str(candidate.target.get("kind") or ""),
                            "search_target_score": float(
                                candidate.target.get("frontier_score") or 0.0
                            ),
                        },
                    )
                )

            requests, request_context, next_rollout_id = self._build_batch_rollout_requests(
                prepared_expansions=prepared_expansions,
                batch_request_budget=batch_request_budget,
                remaining_budget=remaining_budget,
                next_rollout_id=next_rollout_id,
                edge_stats=edge_stats,
                root_issue_plan=issue_plan
                if (
                    not all_results
                    and selected
                    and all(candidate.state_id == root_state.state_id for candidate in selected)
                )
                else None,
            )
            if not requests:
                break

            execute_kwargs: dict[str, Any] = {
                "issue_description": self.issue_description,
                "rollout_requests": requests,
                "test_command": self.test_command,
                "stop_on_result": _build_preemptive_completion_stop_on_result(self.config),
            }
            if wallclock_deadline is not None:
                execute_kwargs["wallclock_deadline"] = wallclock_deadline
            batch_results = engine.execute_rollout_requests(**execute_kwargs)
            next_rollout_id = _next_rollout_id_after(
                batch_results,
                fallback=next_rollout_id,
            )
            remaining_budget -= sum(
                1
                for result in batch_results
                if not bool(result.search_metadata.get("standalone_agent_anchor"))
            )
            batch_found_authoritative_completion = False

            for result in batch_results:
                if result.rollout_id not in request_context:
                    result.search_metadata.update(
                        {
                            "search_mode": mode.value,
                            "search_reason": "standalone_anchor_guard",
                            "search_budget_charged": False,
                        }
                    )
                    all_results.append(result)
                    stop_signal = _frontier_rollout_stop_signal(result, self.config)
                    if stop_signal:
                        batch_found_authoritative_completion = True
                        stopped_early = True
                        stop_reason = (
                            f"{stop_signal} standalone_anchor_rollout_id={result.rollout_id}"
                        )
                    continue
                context = request_context[result.rollout_id]
                candidate, expansion_plan, parent_state, before_graph, request_metadata = context
                result.search_metadata.update(request_metadata)
                edge = self._edge_stats(edge_stats, parent_state.state_id, candidate.target)
                edge.virtual_loss = max(
                    0.0, edge.virtual_loss - float(self.config.search.virtual_loss)
                )

                global_graph.ingest_rollout_result(expansion_plan, result)
                after_graph = before_graph.clone()
                after_graph.ingest_rollout_result(expansion_plan, result)
                reward = self._compute_transition_reward(
                    before_graph=before_graph,
                    after_graph=after_graph,
                    rollout_result=result,
                    target=candidate.target,
                )
                child_state_id: Optional[str] = None
                seed = build_workspace_seed_from_rollout_result(result)
                can_branch = (
                    seed is not None
                    and parent_state.depth + 1 <= max(1, self.config.search.max_depth)
                    and max(reward, float(result.progress_score or 0.0))
                    >= self.config.search.min_branch_reward
                )
                if can_branch:
                    child_graph = after_graph
                    child_state = self._make_state(
                        parent_state_id=parent_state.state_id,
                        depth=parent_state.depth + 1,
                        graph=child_graph,
                        checkpoint_seed=seed,
                        incoming_rollout_id=result.rollout_id,
                        incoming_target_id=candidate.target_id,
                        cumulative_reward=parent_state.cumulative_reward + reward,
                    )
                    states[child_state.state_id] = child_state
                    child_state_id = child_state.state_id
                total_value = parent_state.cumulative_reward + reward

                self._update_stats(
                    states=states,
                    edge_stats=edge_stats,
                    edge=edge,
                    global_target_stats=global_target_stats,
                    target=candidate.target,
                    parent_state=parent_state,
                    child_state_id=child_state_id,
                    reward=reward,
                    rollout_result=result,
                )

                result.search_metadata.update(
                    {
                        "search_reward": round(reward, 4),
                        "search_total_value": round(total_value, 4),
                        "search_q_value": round(candidate.q_value, 4),
                        "search_exploration_bonus": round(candidate.exploration_bonus, 4),
                        "search_optimistic_bound": round(candidate.optimistic_bound, 4),
                        "search_reason": candidate.reason,
                        "search_child_state_id": child_state_id,
                    }
                )
                transitions.append(
                    {
                        "trigger": "frontier_search_expand",
                        "search_mode": mode.value,
                        "state_id": parent_state.state_id,
                        "child_state_id": child_state_id,
                        "target_id": candidate.target_id,
                        "target_kind": str(candidate.target.get("kind") or ""),
                        "target_description": str(candidate.target.get("description") or ""),
                        "score": round(candidate.score, 4),
                        "q_value": round(candidate.q_value, 4),
                        "exploration_bonus": round(candidate.exploration_bonus, 4),
                        "optimistic_bound": round(candidate.optimistic_bound, 4),
                        "reward": round(reward, 4),
                        "total_value": round(total_value, 4),
                        "rollout_id": result.rollout_id,
                        "branchable": bool(child_state_id),
                    }
                )
                if total_value > best_total_value:
                    best_total_value = total_value
                    best_rollout_id = result.rollout_id
                if reward > best_reward:
                    best_reward = reward
                    best_reward_rollout_id = result.rollout_id
                all_results.append(result)
                stop_signal = _frontier_rollout_stop_signal(result, self.config)
                if stop_signal:
                    batch_found_authoritative_completion = True
                    stopped_early = True
                    stop_reason = f"{stop_signal} rollout_id={result.rollout_id}"

            self._persist_trace(
                states=states,
                edge_stats=edge_stats,
                global_target_stats=global_target_stats,
                transitions=transitions,
                shadow_policy_log=shadow_policy_log,
            )
            if batch_found_authoritative_completion:
                break

        final_task_state_context = global_graph.build_issue_plan_context(
            max_items=max(
                self.config.planning.max_task_state_context_items,
                self.config.planning.max_frontier_targets,
            )
        )
        final_issue_plan = IssuePlan.from_dict(issue_plan.to_dict())
        final_issue_plan.task_state_context = dict(final_task_state_context)
        final_issue_plan = self.planner.apply_task_state_frontier(
            final_issue_plan,
            self.repo_context,
            task_state_context=final_task_state_context,
            stage_label="search_final",
        )
        final_issue_plan.planner_source = f"{issue_plan.planner_source}+{mode.value}"
        final_issue_plan.planner_metadata = dict(final_issue_plan.planner_metadata)
        final_issue_plan.planner_metadata.update(
            {
                "search_mode": mode.value,
                "search_total_expansions": len(all_results),
                "search_state_count": len(states),
                "search_best_rollout_id": best_rollout_id,
                "search_best_total_value": round(best_total_value, 4)
                if best_total_value > float("-inf")
                else None,
                "search_shadow_policy_log": list(shadow_policy_log[-12:]),
            }
        )

        summary = {
            "mode": mode.value,
            "total_expansions": len(all_results),
            "state_count": len(states),
            "edge_count": len(edge_stats),
            "branch_state_count": sum(
                1 for state in states.values() if state.parent_state_id is not None
            ),
            "max_depth_reached": max((state.depth for state in states.values()), default=0),
            "best_rollout_id": best_rollout_id,
            "best_reward": round(best_reward, 4) if best_reward > float("-inf") else None,
            "best_reward_rollout_id": best_reward_rollout_id,
            "best_total_value": round(best_total_value, 4)
            if best_total_value > float("-inf")
            else None,
            "transition_count": len(transitions),
            "checkpoint_reuse_count": sum(
                1
                for result in all_results
                if result.search_metadata.get("search_parent_state_id") is not None
            ),
            "frontier_target_visits": {
                target_id: stats.visit_count
                for target_id, stats in sorted(global_target_stats.items())
            },
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "shadow_policy_log": list(shadow_policy_log),
        }
        self._persist_summary(summary)
        return FrontierSearchRunResult(
            rollout_results=all_results,
            issue_plan=final_issue_plan,
            next_rollout_id=next_rollout_id,
            transitions=transitions,
            summary=summary,
        )

    def _make_state(
        self,
        *,
        parent_state_id: Optional[str],
        depth: int,
        graph: TaskStateGraph,
        checkpoint_seed: Optional[WorkspaceSeed],
        incoming_rollout_id: Optional[int],
        incoming_target_id: Optional[str],
        cumulative_reward: float,
    ) -> SearchState:
        state_id = f"s{self._state_index}"
        self._state_index += 1
        task_state_context = graph.build_issue_plan_context(
            max_items=max(
                self.config.planning.max_task_state_context_items,
                self.config.planning.max_frontier_targets,
            )
        )
        frontier_targets = list(task_state_context.get("frontier_targets") or [])[
            : max(1, self.config.search.max_frontier_branching)
        ]
        terminal = depth >= max(1, self.config.search.max_depth) or not frontier_targets
        if checkpoint_seed is not None:
            checkpoint_seed.state_id = state_id
        return SearchState(
            state_id=state_id,
            parent_state_id=parent_state_id,
            depth=depth,
            graph_snapshot=graph.to_dict(),
            task_state_context=task_state_context,
            frontier_targets=frontier_targets,
            checkpoint_seed=checkpoint_seed,
            terminal=terminal,
            incoming_rollout_id=incoming_rollout_id,
            incoming_target_id=incoming_target_id,
            cumulative_reward=cumulative_reward,
        )

    def _rank_candidates(
        self,
        *,
        states: dict[str, SearchState],
        edge_stats: dict[tuple[str, str], SearchEdgeStats],
        global_target_stats: dict[str, GlobalTargetStats],
    ) -> list[SearchExpansionCandidate]:
        candidates: list[SearchExpansionCandidate] = []
        for state in states.values():
            if state.terminal:
                continue
            normalized_priors = self._normalize_sibling_priors(state)
            for target in state.frontier_targets[
                : max(1, self.config.search.max_frontier_branching)
            ]:
                target_id = str(target.get("target_id") or "").strip()
                if not target_id:
                    continue
                edge = self._edge_stats(edge_stats, state.state_id, target)
                global_stats = global_target_stats.get(target_id)
                raw_prior = edge.prior
                uncertainty = float(target.get("uncertainty_score") or 0.0)
                if self.config.search.mode == SearchMode.PUCT:
                    prior = normalized_priors.get(target_id, raw_prior)
                    # Standard PUCT uses Σ_b N(s, b) — the sum of sibling
                    # edge visit counts — as the parent visit term, NOT
                    # the value-update count of the state. Using
                    # ``state.value_count`` (incremented in
                    # ``_update_state_value_stats``) clamped to 1 collapsed
                    # the exploration bonus to a constant for many siblings.
                    sibling_visits = sum(
                        self._edge_stats(edge_stats, state.state_id, sibling_target).visit_count
                        for sibling_target in state.frontier_targets[
                            : max(1, self.config.search.max_frontier_branching)
                        ]
                    )
                    parent_visits = max(1, sibling_visits)
                    fallback_value = global_stats.average_value if global_stats is not None else 0.0
                    q_value = edge.average_value if edge.visit_count else fallback_value
                    exploration_bonus = (
                        self.config.search.c_puct
                        * prior
                        * math.sqrt(parent_visits)
                        / (1 + edge.visit_count + edge.virtual_loss)
                    )
                    optimistic_q = max(
                        q_value,
                        edge.best_value,
                        global_stats.best_value if global_stats is not None else 0.0,
                    )
                    optimistic_bound = state.cumulative_reward + optimistic_q + exploration_bonus
                    # Drop ``state.cumulative_reward`` from the per-action
                    # score: it is a constant across siblings of the same
                    # state, but biases cross-state ranking toward deeper
                    # paths whose accumulated path-reward is mechanically
                    # larger than shallower frontier states.
                    score = (
                        q_value
                        + exploration_bonus
                        - (0.03 * state.depth)
                        - (0.04 * edge.failure_count)
                    )
                    reason = (
                        f"PUCT q={q_value:.2f} "
                        f"u={exploration_bonus:.2f} prior={prior:.2f} "
                        f"N_parent={parent_visits} N_edge={edge.visit_count}"
                    )
                else:
                    prior = raw_prior
                    observed_visits = (
                        edge.direct_visit_count if edge.direct_visit_count > 0 else edge.visit_count
                    )
                    edge_value = (
                        edge.direct_average_value
                        if edge.direct_visit_count
                        else (edge.average_value if edge.visit_count else 0.0)
                    )
                    global_value = global_stats.average_value if global_stats is not None else 0.0
                    novelty = 1.0 / (1 + observed_visits)
                    failure_penalty = min(edge.failure_count / 3.0, 1.0)
                    q_value = edge_value
                    exploration_bonus = novelty
                    score = (
                        (0.42 * prior)
                        + (0.18 * uncertainty)
                        + (0.16 * edge_value)
                        + (0.10 * global_value)
                        + (0.10 * novelty)
                        - (0.08 * failure_penalty)
                        - (0.03 * state.depth)
                        - edge.virtual_loss
                    )
                    optimistic_bound = state.cumulative_reward + score
                    reason = (
                        f"best_first prior={prior:.2f} uncertainty={uncertainty:.2f} "
                        f"edge={edge_value:.2f} novelty={novelty:.2f}"
                    )
                candidates.append(
                    SearchExpansionCandidate(
                        state_id=state.state_id,
                        target=dict(target),
                        score=float(score),
                        prior=prior,
                        q_value=float(q_value),
                        exploration_bonus=float(exploration_bonus),
                        optimistic_bound=float(optimistic_bound),
                        reason=reason,
                    )
                )
        if self.config.search.mode == SearchMode.PUCT:
            candidates.sort(
                key=lambda item: (item.score, item.optimistic_bound, item.prior, item.target_id),
                reverse=True,
            )
        else:
            candidates.sort(key=lambda item: (item.score, item.prior, item.target_id), reverse=True)
        return candidates

    def _normalize_sibling_priors(self, state: SearchState) -> dict[str, float]:
        priors: dict[str, float] = {}
        for target in state.frontier_targets[: max(1, self.config.search.max_frontier_branching)]:
            target_id = str(target.get("target_id") or "").strip()
            if not target_id:
                continue
            priors[target_id] = max(0.0, float(target.get("frontier_score") or 0.0))
        if not priors:
            return {}
        total = sum(priors.values())
        if total <= 0.0:
            uniform = 1.0 / len(priors)
            return {target_id: uniform for target_id in priors}
        return {target_id: value / total for target_id, value in priors.items()}

    def _edge_stats(
        self,
        edge_stats: dict[tuple[str, str], SearchEdgeStats],
        state_id: str,
        target: dict[str, Any],
    ) -> SearchEdgeStats:
        target_id = str(target.get("target_id") or "").strip()
        key = (state_id, target_id)
        if key not in edge_stats:
            edge_stats[key] = SearchEdgeStats(
                state_id=state_id,
                target_id=target_id,
                prior=float(target.get("frontier_score") or 0.0),
            )
        return edge_stats[key]

    def _edge_stats_for_target_id(
        self,
        edge_stats: dict[tuple[str, str], SearchEdgeStats],
        *,
        state: SearchState,
        target_id: str,
    ) -> SearchEdgeStats:
        key = (state.state_id, target_id)
        stats = edge_stats.get(key)
        if stats is not None:
            return stats
        for target in state.frontier_targets:
            if str(target.get("target_id") or "").strip() == target_id:
                return self._edge_stats(edge_stats, state.state_id, target)
        edge_stats[key] = SearchEdgeStats(state_id=state.state_id, target_id=target_id, prior=0.0)
        return edge_stats[key]

    def _update_stats(
        self,
        *,
        states: dict[str, SearchState],
        edge_stats: dict[tuple[str, str], SearchEdgeStats],
        edge: SearchEdgeStats,
        global_target_stats: dict[str, GlobalTargetStats],
        target: dict[str, Any],
        parent_state: SearchState,
        child_state_id: Optional[str],
        reward: float,
        rollout_result: RolloutResult,
    ) -> None:
        self._update_aggregate_edge_stats(edge, reward)
        if child_state_id:
            self._register_child_state(edge, child_state_id)

        edge.direct_visit_count += 1
        edge.direct_value_sum += reward
        edge.last_rollout_id = rollout_result.rollout_id
        edge.mean_cost = (
            (edge.mean_cost * (edge.direct_visit_count - 1))
            + float(rollout_result.duration_seconds or 0.0)
        ) / edge.direct_visit_count
        if rollout_result.success and rollout_result.patch:
            edge.success_count += 1
        else:
            edge.failure_count += 1

        target_id = str(target.get("target_id") or "").strip()
        stats = global_target_stats.get(target_id)
        if stats is None:
            stats = GlobalTargetStats(target_id=target_id)
            global_target_stats[target_id] = stats
        stats.visit_count += 1
        stats.value_sum += reward
        stats.best_value = max(stats.best_value, reward)
        if not (rollout_result.success and rollout_result.patch):
            stats.failure_count += 1

        leaf_total_value = parent_state.cumulative_reward + reward
        self._update_state_value_stats(parent_state, reward)

        cursor = parent_state
        while cursor.parent_state_id is not None:
            ancestor_parent = states.get(cursor.parent_state_id)
            if ancestor_parent is None:
                break
            incoming_target_id = str(cursor.incoming_target_id or "").strip()
            if incoming_target_id:
                ancestor_edge = self._edge_stats_for_target_id(
                    edge_stats,
                    state=ancestor_parent,
                    target_id=incoming_target_id,
                )
                self._update_aggregate_edge_stats(
                    ancestor_edge,
                    leaf_total_value - ancestor_parent.cumulative_reward,
                )
            self._update_state_value_stats(
                ancestor_parent,
                leaf_total_value - ancestor_parent.cumulative_reward,
            )
            cursor = ancestor_parent

    def _update_aggregate_edge_stats(
        self,
        edge: SearchEdgeStats,
        value: float,
    ) -> None:
        edge.visit_count += 1
        edge.value_sum += value
        edge.best_value = max(edge.best_value, value)

    def _update_state_value_stats(
        self,
        state: SearchState,
        value: float,
    ) -> None:
        state.value_count += 1
        state.value_sum += value
        state.best_value = max(state.best_value, value)

    def _register_child_state(
        self,
        edge: SearchEdgeStats,
        child_state_id: str,
    ) -> None:
        if child_state_id not in edge.child_state_ids:
            edge.child_state_ids.append(child_state_id)

    def _should_stop_search(
        self,
        *,
        candidates: list[SearchExpansionCandidate],
        best_total_value: float,
        states: Optional[dict[str, SearchState]] = None,
    ) -> tuple[bool, Optional[str]]:
        if (
            self.config.search.mode != SearchMode.PUCT
            or not candidates
            or best_total_value <= float("-inf")
        ):
            return False, None
        if states and any(
            not state.terminal
            and state.parent_state_id is not None
            and state.value_count <= 0
            and bool(state.frontier_targets)
            for state in states.values()
        ):
            return False, None
        frontier_bound = max(candidate.optimistic_bound for candidate in candidates)
        margin = max(0.0, float(self.config.search.stop_margin))
        if frontier_bound + margin >= best_total_value:
            return False, None
        reason = (
            f"puct_stop best_total_value={best_total_value:.3f} "
            f"frontier_bound={frontier_bound:.3f} margin={margin:.3f}"
        )
        return True, reason

    def _compute_transition_reward(
        self,
        *,
        before_graph: TaskStateGraph,
        after_graph: TaskStateGraph,
        rollout_result: RolloutResult,
        target: dict[str, Any],
    ) -> float:
        reward_policy = self.config.search.transition_reward
        obligation_id = str(target.get("obligation_id") or "").strip()
        hypothesis_id = str(target.get("hypothesis_id") or "").strip()
        before_obligation = before_graph.obligations.get(obligation_id)
        after_obligation = after_graph.obligations.get(obligation_id)
        before_hypothesis = before_graph.hypotheses.get(hypothesis_id)
        after_hypothesis = after_graph.hypotheses.get(hypothesis_id)

        obligation_delta = 0.0
        if before_obligation is not None and after_obligation is not None:
            baseline = max(before_obligation.outstanding_score, 0.2)
            obligation_delta = (
                before_obligation.outstanding_score - after_obligation.outstanding_score
            ) / baseline
        hypothesis_delta = 0.0
        if before_hypothesis is not None and after_hypothesis is not None:
            hypothesis_delta = after_hypothesis.belief_score - before_hypothesis.belief_score
        before_uncertainty = self._target_uncertainty(before_graph, target)
        after_uncertainty = self._target_uncertainty(after_graph, target)
        uncertainty_reduction = max(0.0, before_uncertainty - after_uncertainty)
        alignment = self._target_alignment(rollout_result, target)
        progress = float(rollout_result.progress_score or 0.0)
        quick_verification = (
            dict(rollout_result.quick_verification)
            if isinstance(rollout_result.quick_verification, dict)
            else {}
        )
        quick_pass_rate = quick_verification_signal_score(quick_verification)
        if quick_pass_rate is not None:
            quick_feedback = quick_pass_rate
            if quick_verification.get("scope") in {"failing_tests", "focus_test_files"}:
                quick_feedback = min(1.0, 0.1 + (0.9 * quick_feedback))
        else:
            quick_feedback = 0.0
        patch_bonus = (
            float(getattr(reward_policy, "patch_bonus", 0.20) or 0.20)
            if rollout_result.success and rollout_result.patch
            else 0.0
        )
        cost_penalty = min(float(rollout_result.duration_seconds or 0.0) / 300.0, 1.0) * float(
            getattr(reward_policy, "cost_penalty_per_300s", 0.05) or 0.05
        )
        failure_penalty = (
            float(getattr(reward_policy, "failure_penalty", 0.12) or 0.12)
            if not rollout_result.success
            else 0.0
        )
        reward = (
            (float(getattr(reward_policy, "obligation_delta", 0.28) or 0.28) * obligation_delta)
            + (float(getattr(reward_policy, "hypothesis_delta", 0.12) or 0.12) * hypothesis_delta)
            + (
                float(getattr(reward_policy, "uncertainty_reduction", 0.18) or 0.18)
                * uncertainty_reduction
            )
            + (float(getattr(reward_policy, "progress", 0.17) or 0.17) * progress)
            + (float(getattr(reward_policy, "quick_feedback", 0.13) or 0.13) * quick_feedback)
            + (float(getattr(reward_policy, "alignment", 0.10) or 0.10) * alignment)
            + patch_bonus
            - cost_penalty
            - failure_penalty
        )
        return round(_clamp(reward, lower=-1.0, upper=1.0), 4)

    def _target_uncertainty(
        self,
        graph: TaskStateGraph,
        target: dict[str, Any],
    ) -> float:
        scores: list[float] = []
        obligation_id = str(target.get("obligation_id") or "").strip()
        hypothesis_id = str(target.get("hypothesis_id") or "").strip()
        if obligation_id and obligation_id in graph.obligations:
            scores.append(graph.obligations[obligation_id].uncertainty_score)
        if hypothesis_id and hypothesis_id in graph.hypotheses:
            scores.append(graph.hypotheses[hypothesis_id].uncertainty_score)
        if not scores:
            scores.append(float(target.get("uncertainty_score") or 0.0))
        return sum(scores) / max(len(scores), 1)

    def _target_alignment(
        self,
        rollout_result: RolloutResult,
        target: dict[str, Any],
    ) -> float:
        target_files = {str(path) for path in list(target.get("file_paths") or []) if path}
        target_tests = {str(test_id) for test_id in list(target.get("test_ids") or []) if test_id}
        changed_files = set(rollout_result.changed_files or [])
        tests_run = {
            str(test_id) for test_id in list(rollout_result.test_descriptions or []) if test_id
        }
        patch_artifact = (
            rollout_result.patch_artifact if isinstance(rollout_result.patch_artifact, dict) else {}
        )
        tests_run.update(str(item) for item in list(patch_artifact.get("tests_run") or []) if item)

        file_alignment = 0.0
        if target_files and changed_files:
            file_alignment = len(target_files.intersection(changed_files)) / max(
                len(target_files), 1
            )
        test_alignment = 0.0
        if target_tests and tests_run:
            test_alignment = len(target_tests.intersection(tests_run)) / max(len(target_tests), 1)
        return _clamp((0.65 * file_alignment) + (0.35 * test_alignment))

    def _persist_trace(
        self,
        *,
        states: dict[str, SearchState],
        edge_stats: dict[tuple[str, str], SearchEdgeStats],
        global_target_stats: dict[str, GlobalTargetStats],
        transitions: list[dict[str, Any]],
        shadow_policy_log: list[dict[str, Any]],
    ) -> None:
        if not self.config.search.persist_trace:
            return
        payload = {
            "states": {state_id: state.to_dict() for state_id, state in states.items()},
            "edge_stats": {
                f"{state_id}:{target_id}": stats.to_dict()
                for (state_id, target_id), stats in edge_stats.items()
            },
            "global_target_stats": {
                target_id: stats.to_dict() for target_id, stats in global_target_stats.items()
            },
            "transitions": list(transitions),
            "shadow_policy_log": list(shadow_policy_log),
        }
        (self.output_dir / "frontier_search_trace.json").write_text(json.dumps(payload, indent=2))

    def _persist_summary(self, summary: dict[str, Any]) -> None:
        if not self.config.search.persist_trace:
            return
        (self.output_dir / "frontier_search_summary.json").write_text(json.dumps(summary, indent=2))
