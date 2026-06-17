"""
Local benchmark runner for APEX.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..acceptance import (
    quick_verification_expected_coverage_ratio,
    quick_verification_signal_score,
)
from ..controller_trace import append_controller_decision
from ..core.config import AgentMode, ApexConfig
from ..core.filesystem import copy_tree
from ..core.subprocess_utils import run_shell_command
from ..orchestrator import ApexOrchestrator
from .checkpointing import (
    RUN_STATE_FILENAME,
    atomic_write_json,
    atomic_write_text,
    build_run_state,
    ensure_clean_directory_for_task,
    load_json_if_exists,
    task_result_path,
    write_task_checkpoint,
)
from .run_artifacts import (
    build_benchmark_policy,
    build_prompt_template_fingerprints,
    build_run_manifest,
    capture_environment_snapshot,
    cluster_failures,
    ensure_run_manifest,
    load_run_manifest,
    manifest_summary,
    summarize_rollout_profiles,
    update_run_manifest,
    write_task_live_state,
)
from .target_runtime import (
    apply_target_tool_env_to_apex_config,
    host_env_runtime,
    target_tool_env_overrides,
)

LOCAL_BENCHMARK_HARNESS_NAME = "local_benchmark_harness"
LOCAL_BENCHMARK_HARNESS_VERSION = "2026-04-14.1"
LOCAL_BENCHMARK_REPORT_KIND = "apex_local_benchmark"


def build_local_benchmark_policy() -> dict[str, Any]:
    return build_benchmark_policy(
        benchmark_name="local",
        benchmark_family="local",
        agent_input_contract={
            "repo_snapshot_visible": True,
            "issue_statement_visible": True,
            "test_command_visible": True,
            "benchmark_specific_metadata_visible_in_prompt": False,
        },
        orchestrator_input_contract={
            "benchmark_metadata_passed_to_orchestrator": False,
        },
        evaluation_protocol={
            "baseline_evaluation_backend": "repo_test_command",
            "final_evaluation_backend": "repo_test_command",
            "rollout_selection_policy": "orchestrator_selected_rollout",
            "primary_metric": "task_success_and_final_tests_passed",
            "sampling_protocol": "single_run_per_task",
        },
        environment_policy={
            "agent_execution_isolation": "temporary_repo_copy",
            "evaluator_execution_isolation": "temporary_repo_copy",
            "agent_network_access": "inherited_host",
            "evaluator_network_access": "inherited_host",
            "persistent_outputs_outside_repo": True,
        },
        benchmark_specifics={
            "fixture_issue_path_visible": True,
            "git_repo_initialized_for_scoring": True,
        },
    )


def build_apex_ablation_config(config: ApexConfig) -> dict[str, Any]:
    scaffold_policy = (
        "adaptive_cop" if config.rollout.agent_mode == AgentMode.ADAPTIVE else "fixed_agent_mode"
    )
    return {
        "allocator": {
            "policy": "bandit" if config.rollout.enable_adaptive_allocation else "fixed",
            "requested_rollouts": (
                config.rollout.max_rollouts
                if config.rollout.enable_adaptive_allocation
                else config.rollout.num_rollouts
            ),
            "min_rollouts": config.rollout.min_rollouts,
            "max_rollouts": config.rollout.max_rollouts,
            "rollout_buckets": list(config.rollout.rollout_buckets),
            "overlap_diversity_cap_enabled": config.rollout.enable_overlap_diversity_cap,
            "min_overlap_diversity_parallel_workers": (
                config.rollout.min_overlap_diversity_parallel_workers
            ),
        },
        "scaffold": {
            "policy": scaffold_policy,
            "configured_agent_mode": config.rollout.agent_mode.value,
            "scaffold_stage_llm_indices": dict(config.rollout.scaffold_stage_llm_indices),
            "orchestrated_multi_agent_enabled": config.rollout.enable_orchestrated_multi_agent,
            "planning_enabled": config.planning.enable_manager_planner,
            "coarse_to_fine_planning_enabled": config.planning.enable_coarse_to_fine_planning,
            "plan_portfolio_enabled": config.planning.enable_plan_portfolio,
            "always_include_single_agent_family": (
                config.planning.always_include_single_agent_family
            ),
            "always_include_agentless_pipeline_family": (
                config.planning.always_include_agentless_pipeline_family
            ),
            "reflective_memory_enabled": config.planning.enable_reflective_memory,
            "task_state_graph_enabled": config.planning.enable_task_state_graph,
            "frontier_targeting_enabled": config.planning.enable_frontier_targeting,
            "task_state_context_items": config.planning.max_task_state_context_items,
            "max_frontier_targets": config.planning.max_frontier_targets,
            "planner_brief_family_cap": config.planning.max_rollout_brief_families,
            "planner_timeout_seconds": config.planning.planner_timeout_seconds,
            "preplanner_timeout_seconds": config.planning.preplanner_timeout_seconds,
            "refinement_timeout_seconds": config.planning.refinement_timeout_seconds,
            "max_reflection_memory_items": config.planning.max_reflection_memory_items,
            "delegation_boundary_pressure_threshold": (
                config.planning.delegation_boundary_pressure_threshold
            ),
            "execution_tree_enabled": config.execution_tree.enabled,
            "dynamic_transitions_enabled": config.rollout.enable_dynamic_cop_transitions,
            "progressive_rollout_allocation_enabled": (
                config.rollout.enable_progressive_rollout_allocation
            ),
            "max_progressive_rollout_waves": config.rollout.max_progressive_rollout_waves,
        },
        "feedback": {
            "quick_verification_enabled": config.rollout.enable_quick_verification,
            "quick_verification_max_tests": config.rollout.quick_verification_max_tests,
            "quick_verification_timeout_seconds": config.rollout.quick_verification_timeout_seconds,
        },
        "agentic_search": {
            "access_mode": config.agentic_search.access_mode.value,
            "local_doc_guidance_enabled": config.agentic_search.enable_local_doc_guidance,
            "local_doc_max_files": config.agentic_search.local_doc_max_files,
            "guided_stage_names": list(config.agentic_search.guided_stage_names),
            "proactive_evidence_enabled": config.agentic_search.enable_proactive_evidence,
            "proactive_evidence_max_items": config.agentic_search.proactive_evidence_max_items,
            "proactive_evidence_stage_names": list(
                config.agentic_search.proactive_evidence_stage_names
            ),
            "external_search_budget": config.agentic_search.external_search_budget,
            "external_search_max_results": config.agentic_search.external_search_max_results,
            "external_search_timeout_seconds": (
                config.agentic_search.external_search_timeout_seconds
            ),
            "semiformal_review_enabled": config.agentic_search.enable_semiformal_review,
            "followup_search_memory_enabled": (config.agentic_search.enable_followup_search_memory),
            "followup_gathered_information_enabled": (
                config.agentic_search.enable_followup_gathered_information
            ),
            "followup_search_memory_max_items": (
                config.agentic_search.followup_search_memory_max_items
            ),
        },
        "search": {
            "mode": config.search.mode.value,
            "max_expansions": config.search.max_expansions,
            "max_depth": config.search.max_depth,
            "max_frontier_branching": config.search.max_frontier_branching,
            "c_puct": config.search.c_puct,
            "virtual_loss": config.search.virtual_loss,
            "stop_margin": config.search.stop_margin,
            "min_branch_reward": config.search.min_branch_reward,
        },
        "selection": {
            "strategy": config.selection.strategy.value,
            "regression_pruning_enabled": config.selection.enable_regression_pruning,
            "cross_validation_enabled": config.selection.cross_validation_enabled,
            "critic_reranking_enabled": config.selection.enable_critic_reranking,
            "critic_weight": config.selection.critic_weight,
            "selector_max_voters": config.selection.selector_max_voters,
            "selector_max_iterations": config.selection.selector_max_iterations,
        },
        "memory": {
            "repo_memory_enabled": config.repo_memory.enabled,
            "repo_memory_decay_factor": config.repo_memory.decay_factor,
            "repo_memory_min_confidence_to_persist": (config.repo_memory.min_confidence_to_persist),
            "repo_memory_max_persisted_insights": (config.repo_memory.max_persisted_insights),
        },
        "benchmark": {
            "commit0_primary_evaluation_backend": (
                config.benchmark.commit0_primary_evaluation_backend.value
            ),
            "commit0_official_audit_selected": config.benchmark.commit0_official_audit_selected,
            "commit0_official_audit_only_if_primary_passes": (
                config.benchmark.commit0_official_audit_only_if_primary_passes
            ),
            "commit0_repo_clone_timeout_seconds": (
                config.benchmark.commit0_repo_clone_timeout_seconds
            ),
            "commit0_runtime_setup_timeout_seconds": (
                config.benchmark.commit0_runtime_setup_timeout_seconds
            ),
            "commit0_dependency_install_timeout_seconds": (
                config.benchmark.commit0_dependency_install_timeout_seconds
            ),
            "commit0_evaluation_timeout_seconds": (
                config.benchmark.commit0_evaluation_timeout_seconds
            ),
            "task_parallelism": config.benchmark.task_parallelism,
        },
    }


def extract_apex_execution_metadata(result: Any) -> dict[str, Any]:
    issue_plan = getattr(result, "issue_plan", None)
    if not isinstance(issue_plan, dict):
        issue_plan = {}

    rollout_briefs = issue_plan.get("rollout_briefs") or []
    selected_agent_modes = sorted(
        {
            brief.get("agent_mode")
            for brief in rollout_briefs
            if isinstance(brief, dict) and brief.get("agent_mode")
        }
    )
    rollout_search_modes = sorted(
        {
            (brief.get("search_policy") or {}).get("mode")
            for brief in rollout_briefs
            if isinstance(brief, dict)
            and isinstance(brief.get("search_policy"), dict)
            and (brief.get("search_policy") or {}).get("mode")
        }
    )

    orchestration_primitives = getattr(result, "orchestration_primitives", None)
    if orchestration_primitives is None:
        orchestration_primitives = issue_plan.get("orchestration_primitives") or []

    orchestration_transitions = getattr(result, "orchestration_transitions", None)
    if orchestration_transitions is None:
        orchestration_transitions = issue_plan.get("orchestration_transitions") or []

    allocator_features = getattr(result, "allocator_features", None)
    if not isinstance(allocator_features, dict):
        allocator_features = issue_plan.get("allocator_features") or {}

    recommended_rollouts = getattr(result, "recommended_rollouts", None)
    if recommended_rollouts is None:
        recommended_rollouts = issue_plan.get("recommended_rollouts")

    difficulty_estimate = getattr(result, "difficulty_estimate", None)
    if difficulty_estimate is None:
        difficulty_estimate = issue_plan.get("difficulty_estimate")

    unsolvable_reason = getattr(result, "unsolvable_reason", None)
    if unsolvable_reason is None:
        unsolvable_reason = issue_plan.get("unsolvable_reason")

    planner_metadata = issue_plan.get("planner_metadata") or {}
    task_state_context = getattr(result, "task_state_context", None)
    if not isinstance(task_state_context, dict):
        task_state_context = issue_plan.get("task_state_context") or {}
    multi_agent_summary = getattr(result, "multi_agent_summary", None)
    if not isinstance(multi_agent_summary, dict):
        multi_agent_summary = {}
    rollout_summaries = getattr(result, "rollout_summaries", None) or []
    selected_rollout_id = getattr(result, "selected_rollout_id", None)
    search_summary = getattr(result, "search_summary", None)
    if not isinstance(search_summary, dict):
        search_summary = {}
    selected_rollout_summary = next(
        (
            dict(summary)
            for summary in rollout_summaries
            if isinstance(summary, dict) and summary.get("rollout_id") == selected_rollout_id
        ),
        {},
    )
    ranking_payloads = [
        dict((summary.get("selection_diagnostics") or {}).get("ranking") or {})
        for summary in rollout_summaries
        if isinstance(summary, dict)
        and isinstance(summary.get("selection_diagnostics"), dict)
        and isinstance((summary.get("selection_diagnostics") or {}).get("ranking"), dict)
    ]
    selection_evidence_mode_counts: Counter[str] = Counter(
        str(payload.get("evidence_mode") or "").strip() or "unknown" for payload in ranking_payloads
    )
    selection_public_signal_scores = [
        float(payload.get("public_signal_score"))
        for payload in ranking_payloads
        if isinstance(payload.get("public_signal_score"), (int, float))
    ]
    backend_anomaly_payloads = []
    for summary in rollout_summaries:
        if not isinstance(summary, dict):
            continue
        selection_diagnostics = summary.get("selection_diagnostics")
        if isinstance(selection_diagnostics, dict) and isinstance(
            selection_diagnostics.get("backend_anomaly"),
            dict,
        ):
            backend_anomaly_payloads.append(dict(selection_diagnostics["backend_anomaly"]))
            continue
        search_metadata = summary.get("search_metadata")
        if isinstance(search_metadata, dict) and isinstance(
            search_metadata.get("backend_anomaly"),
            dict,
        ):
            backend_anomaly_payloads.append(dict(search_metadata["backend_anomaly"]))
    backend_anomaly_kind_counts: Counter[str] = Counter(
        str(payload.get("kind") or "").strip() or "unknown" for payload in backend_anomaly_payloads
    )
    selected_ranking = (
        dict((selected_rollout_summary.get("selection_diagnostics") or {}).get("ranking") or {})
        if isinstance(selected_rollout_summary.get("selection_diagnostics"), dict)
        else {}
    )
    selected_backend_anomaly = {}
    if isinstance(selected_rollout_summary.get("selection_diagnostics"), dict):
        selected_backend_anomaly = dict(
            (selected_rollout_summary.get("selection_diagnostics") or {}).get("backend_anomaly")
            or {}
        )
    if not selected_backend_anomaly and isinstance(
        selected_rollout_summary.get("search_metadata"), dict
    ):
        selected_backend_anomaly = dict(
            (selected_rollout_summary.get("search_metadata") or {}).get("backend_anomaly") or {}
        )
    progress_scores = [
        float(summary.get("progress_score"))
        for summary in rollout_summaries
        if isinstance(summary, dict) and isinstance(summary.get("progress_score"), (int, float))
    ]
    progressive_wave_count = sum(
        1
        for transition in orchestration_transitions
        if isinstance(transition, dict) and transition.get("trigger") == "progressive_wave"
    )
    frontier_search_transition_count = sum(
        1
        for transition in orchestration_transitions
        if isinstance(transition, dict) and transition.get("trigger") == "frontier_search_expand"
    )
    rollout_search_metadata = [
        dict(summary.get("search_metadata") or {})
        for summary in rollout_summaries
        if isinstance(summary, dict) and isinstance(summary.get("search_metadata"), dict)
    ]
    quick_verification_payloads = [
        dict(summary.get("quick_verification") or {})
        for summary in rollout_summaries
        if isinstance(summary, dict) and isinstance(summary.get("quick_verification"), dict)
    ]
    quick_verification_observed_pass_rates = [
        float(payload.get("pass_rate"))
        for payload in quick_verification_payloads
        if isinstance(payload.get("pass_rate"), (int, float))
    ]
    quick_verification_signal_scores = [
        score
        for payload in quick_verification_payloads
        if isinstance((score := quick_verification_signal_score(payload)), (int, float))
    ]
    quick_verification_expected_coverage_ratios = [
        ratio
        for payload in quick_verification_payloads
        if isinstance(
            (ratio := quick_verification_expected_coverage_ratio(payload)),
            (int, float),
        )
    ]
    rollout_multi_agent_summaries = [
        dict(summary.get("multi_agent_summary") or {})
        for summary in rollout_summaries
        if isinstance(summary, dict) and isinstance(summary.get("multi_agent_summary"), dict)
    ]
    rollout_stage_entries = [
        dict(entry)
        for summary in rollout_summaries
        if isinstance(summary, dict)
        for entry in list(summary.get("trajectory") or [])
        if isinstance(entry, dict)
    ]
    followup_routing_events = [
        dict(event)
        for stage_entry in rollout_stage_entries
        for event in list(stage_entry.get("followup_evidence_routing_events") or [])
        if isinstance(event, dict)
    ]
    if not followup_routing_events:
        followup_routing_events = [
            dict(iteration.get("followup_evidence_routing") or {})
            for stage_entry in rollout_stage_entries
            for iteration in list(stage_entry.get("iterations") or [])
            if isinstance(iteration, dict)
            and isinstance(iteration.get("followup_evidence_routing"), dict)
        ]
    followup_pass_rate_deltas = [
        float((event.get("outcome") or {}).get("pass_rate_delta"))
        for event in followup_routing_events
        if isinstance(event, dict)
        and isinstance(event.get("outcome"), dict)
        and isinstance((event.get("outcome") or {}).get("pass_rate_delta"), (int, float))
    ]
    followup_duration_deltas = [
        float((event.get("outcome") or {}).get("duration_delta_seconds"))
        for event in followup_routing_events
        if isinstance(event, dict)
        and isinstance(event.get("outcome"), dict)
        and isinstance((event.get("outcome") or {}).get("duration_delta_seconds"), (int, float))
    ]
    followup_token_deltas = [
        int((event.get("outcome") or {}).get("token_delta"))
        for event in followup_routing_events
        if isinstance(event, dict)
        and isinstance(event.get("outcome"), dict)
        and isinstance((event.get("outcome") or {}).get("token_delta"), (int, float))
    ]
    rollout_profile_table = summarize_rollout_profiles(
        [dict(summary) for summary in rollout_summaries if isinstance(summary, dict)]
    )
    timeout_audit_trail = []
    timeout_terminal_states: Counter[str] = Counter()
    for summary in rollout_summaries:
        if not isinstance(summary, dict):
            continue
        rollout_id = summary.get("rollout_id")
        for entry in list(summary.get("trajectory") or []):
            if not isinstance(entry, dict):
                continue
            timeout_audit = dict(entry.get("timeout_audit") or {})
            if not timeout_audit:
                continue
            terminal_state = str(timeout_audit.get("terminal_state") or "").strip() or "unknown"
            timeout_terminal_states[terminal_state] += 1
            timeout_audit_trail.append(
                {
                    "rollout_id": rollout_id,
                    "stage": entry.get("stage"),
                    "model": entry.get("model"),
                    "terminal_state": terminal_state,
                    "last_progress_source": timeout_audit.get("last_progress_source"),
                    "last_progress_at": timeout_audit.get("last_progress_at"),
                    "started_at": timeout_audit.get("started_at"),
                    "ended_at": timeout_audit.get("ended_at"),
                    "progress_timeout_seconds": timeout_audit.get("progress_timeout_seconds"),
                    "hard_timeout_seconds": timeout_audit.get("hard_timeout_seconds"),
                    "evidence_counts": dict(timeout_audit.get("evidence_counts") or {}),
                    "recovered": bool(timeout_audit.get("recovered")),
                }
            )
    scheduler_configured_parallel_workers_values = sorted(
        {
            int(metadata.get("scheduler_configured_parallel_workers"))
            for metadata in rollout_search_metadata
            if isinstance(metadata.get("scheduler_configured_parallel_workers"), int)
        }
    )
    scheduler_requested_parallel_workers_values = sorted(
        {
            int(metadata.get("scheduler_requested_parallel_workers"))
            for metadata in rollout_search_metadata
            if isinstance(metadata.get("scheduler_requested_parallel_workers"), int)
        }
    )
    scheduler_effective_parallel_workers_values = sorted(
        {
            int(metadata.get("scheduler_effective_parallel_workers"))
            for metadata in rollout_search_metadata
            if isinstance(metadata.get("scheduler_effective_parallel_workers"), int)
        }
    )
    scheduler_diversity_capacity_estimate_values = sorted(
        {
            int(metadata.get("scheduler_diversity_capacity_estimate"))
            for metadata in rollout_search_metadata
            if isinstance(metadata.get("scheduler_diversity_capacity_estimate"), int)
        }
    )
    scheduler_diversity_floor_parallel_workers_values = sorted(
        {
            int(metadata.get("scheduler_diversity_floor_parallel_workers"))
            for metadata in rollout_search_metadata
            if isinstance(metadata.get("scheduler_diversity_floor_parallel_workers"), int)
        }
    )
    scheduler_diversity_limited_rollout_count = sum(
        1
        for metadata in rollout_search_metadata
        if bool(metadata.get("scheduler_diversity_limited"))
    )
    scheduler_diversity_floor_applied_rollout_count = sum(
        1
        for metadata in rollout_search_metadata
        if bool(metadata.get("scheduler_diversity_floor_applied"))
    )
    scheduler_overlap_sensitive_rollout_count_values = sorted(
        {
            int(metadata.get("scheduler_overlap_sensitive_rollout_count"))
            for metadata in rollout_search_metadata
            if isinstance(metadata.get("scheduler_overlap_sensitive_rollout_count"), int)
        }
    )
    search_mode = (
        search_summary.get("mode")
        or planner_metadata.get("search_mode")
        or next(
            (
                metadata.get("search_mode")
                for metadata in rollout_search_metadata
                if metadata.get("search_mode")
            ),
            None,
        )
    )
    checkpoint_reuse_count = int(
        search_summary.get("checkpoint_reuse_count")
        or sum(
            1
            for metadata in rollout_search_metadata
            if metadata.get("search_parent_state_id") is not None
        )
    )
    search_total_expansions = int(
        search_summary.get("total_expansions")
        or (len(rollout_search_metadata) if search_mode else 0)
    )
    search_state_count = search_summary.get("state_count")
    search_run_count = int(
        search_summary.get("run_count") or (1 if search_mode and search_total_expansions > 0 else 0)
    )
    boundary_pressure_count = int(
        multi_agent_summary.get("boundary_pressure_count")
        or sum(
            int(payload.get("boundary_pressure_count") or 0)
            for payload in rollout_multi_agent_summaries
        )
    )
    boundary_requested_files = list(
        dict.fromkeys(
            list(multi_agent_summary.get("boundary_requested_files") or [])
            + [
                str(path).strip()
                for payload in rollout_multi_agent_summaries
                for path in list(payload.get("boundary_requested_files") or [])
                if str(path).strip()
            ]
        )
    )[:8]
    boundary_interface_symbols = list(
        dict.fromkeys(
            list(multi_agent_summary.get("boundary_interface_symbols") or [])
            + [
                str(symbol).strip()
                for payload in rollout_multi_agent_summaries
                for symbol in list(payload.get("boundary_interface_symbols") or [])
                if str(symbol).strip()
            ]
        )
    )[:8]
    boundary_followups = list(
        dict.fromkeys(
            list(multi_agent_summary.get("boundary_followups") or [])
            + [
                str(item).strip()
                for payload in rollout_multi_agent_summaries
                for item in list(payload.get("boundary_followups") or [])
                if str(item).strip()
            ]
        )
    )[:8]
    delegate_parallelism_values = sorted(
        {
            int(payload.get("delegate_parallelism"))
            for payload in rollout_multi_agent_summaries
            if isinstance(payload.get("delegate_parallelism"), int)
            and int(payload.get("delegate_parallelism")) > 0
        }
    )
    delegate_parallelism_requested_values = sorted(
        {
            int(payload.get("delegate_parallelism_requested"))
            for payload in rollout_multi_agent_summaries
            if isinstance(payload.get("delegate_parallelism_requested"), int)
            and int(payload.get("delegate_parallelism_requested")) > 0
        }
    )
    delegate_task_split_count_values = sorted(
        {
            int(payload.get("delegate_task_split_count"))
            for payload in rollout_multi_agent_summaries
            if isinstance(payload.get("delegate_task_split_count"), int)
            and int(payload.get("delegate_task_split_count")) > 0
        }
    )
    delegation_policy_enabled_rollout_count = sum(
        1
        for payload in rollout_multi_agent_summaries
        if bool(payload.get("delegation_policy_enabled"))
    )
    delegate_subtasks_enabled_rollout_count = sum(
        1
        for payload in rollout_multi_agent_summaries
        if bool(payload.get("delegate_subtasks_enabled"))
    )
    planned_delegate_subtask_total = sum(
        int(payload.get("delegate_task_split_count") or 0)
        for payload in rollout_multi_agent_summaries
    )
    delegation_group_count = int(
        multi_agent_summary.get("delegation_group_count")
        or sum(
            int(payload.get("delegation_group_count") or 0)
            for payload in rollout_multi_agent_summaries
        )
    )
    delegated_subtasks = int(
        multi_agent_summary.get("delegated_subtasks")
        or sum(
            int(payload.get("delegated_subtasks") or 0) for payload in rollout_multi_agent_summaries
        )
    )
    successful_delegated_subtasks = int(
        multi_agent_summary.get("successful_delegated_subtasks")
        or sum(
            int(payload.get("successful_delegated_subtasks") or 0)
            for payload in rollout_multi_agent_summaries
        )
    )

    return {
        "planner_source": issue_plan.get("planner_source"),
        "planner_metadata": dict(planner_metadata),
        "planner_pipeline": planner_metadata.get("planner_pipeline"),
        "planning_mode": planner_metadata.get("planning_mode"),
        "preplanner_skip_reason": planner_metadata.get("preplanner_skip_reason"),
        "time_to_first_plan_seconds": planner_metadata.get("time_to_first_plan_seconds"),
        "planner_total_duration_seconds": planner_metadata.get("planner_total_duration_seconds"),
        "coarse_planner_model": planner_metadata.get("coarse_planner_model"),
        "coarse_planner_backend": planner_metadata.get("coarse_planner_backend"),
        "coarse_planner_tokens": planner_metadata.get("coarse_planner_tokens"),
        "refinement_model": planner_metadata.get("refinement_model"),
        "refinement_backend": planner_metadata.get("refinement_backend"),
        "refinement_duration_seconds": planner_metadata.get("refinement_duration_seconds"),
        "difficulty_estimate": difficulty_estimate,
        "recommended_rollouts": recommended_rollouts,
        "orchestration_primitives": list(orchestration_primitives),
        "orchestration_transitions": list(orchestration_transitions),
        "selected_agent_modes": selected_agent_modes,
        "rollout_search_modes": rollout_search_modes,
        "allocator_features": dict(allocator_features),
        "progressive_wave_count": progressive_wave_count,
        "frontier_search_transition_count": frontier_search_transition_count,
        "max_rollout_progress_score": max(progress_scores) if progress_scores else None,
        "task_state_open_obligation_count": len(task_state_context.get("open_obligations") or []),
        "task_state_supported_hypothesis_count": len(
            task_state_context.get("supported_hypotheses") or []
        ),
        "task_state_frontier_target_count": len(task_state_context.get("frontier_targets") or []),
        "task_state_focus_files": list(task_state_context.get("focus_files") or []),
        "task_state_reflection_memory_count": len(
            task_state_context.get("reflection_memory") or []
        ),
        "task_state_progress_ledger_action": (
            (task_state_context.get("progress_ledger") or {}).get("next_action")
            if isinstance(task_state_context.get("progress_ledger"), dict)
            else None
        ),
        "task_state_progress_ledger_summary": (
            (task_state_context.get("progress_ledger") or {}).get("decision_summary")
            if isinstance(task_state_context.get("progress_ledger"), dict)
            else None
        ),
        "search_mode": search_mode,
        "search_total_expansions": search_total_expansions,
        "search_state_count": search_state_count,
        "search_run_count": search_run_count,
        "search_checkpoint_reuse_count": checkpoint_reuse_count,
        "search_summary": copy.deepcopy(search_summary),
        "selected_rollout_evidence_mode": selected_ranking.get("evidence_mode"),
        "selected_rollout_verification_authority": selected_ranking.get("verification_authority"),
        "selected_rollout_ranking_reason": selected_ranking.get("ranking_reason"),
        "selected_rollout_combined_score": selected_ranking.get("combined_score"),
        "selection_evidence_mode_counts": dict(selection_evidence_mode_counts),
        "selection_non_authoritative_rollout_count": sum(
            count
            for mode, count in selection_evidence_mode_counts.items()
            if mode != "authoritative"
        ),
        "selection_public_signal_rollout_count": sum(
            1 for score in selection_public_signal_scores if float(score) > 0.0
        ),
        "max_selection_public_signal_score": (
            max(selection_public_signal_scores) if selection_public_signal_scores else None
        ),
        "avg_selection_public_signal_score": (
            sum(selection_public_signal_scores) / len(selection_public_signal_scores)
            if selection_public_signal_scores
            else None
        ),
        "backend_anomaly_rollout_count": len(backend_anomaly_payloads),
        "backend_anomaly_kind_counts": dict(backend_anomaly_kind_counts),
        "backend_anomaly_recovered_candidate_count": sum(
            1
            for summary in rollout_summaries
            if isinstance(summary, dict)
            and isinstance(summary.get("search_metadata"), dict)
            and bool(
                (summary.get("search_metadata") or {}).get("backend_anomaly_recovered_candidate")
            )
        ),
        "selected_rollout_backend_anomaly_kind": selected_backend_anomaly.get("kind"),
        "rollout_profile_table": rollout_profile_table,
        "timeout_audit_trail": timeout_audit_trail,
        "timeout_terminal_state_counts": dict(timeout_terminal_states),
        "rollout_timeout_event_count": sum(timeout_terminal_states.values()),
        "rollout_stall_timeout_count": int(timeout_terminal_states.get("stall_timeout", 0)),
        "rollout_hard_timeout_count": int(timeout_terminal_states.get("hard_timeout", 0)),
        "rollout_timeout_recovery_count": int(
            timeout_terminal_states.get("recovered_after_timeout", 0)
        ),
        "scheduler_configured_parallel_workers_values": scheduler_configured_parallel_workers_values,
        "scheduler_requested_parallel_workers_values": scheduler_requested_parallel_workers_values,
        "scheduler_effective_parallel_workers_values": scheduler_effective_parallel_workers_values,
        "scheduler_diversity_capacity_estimate_values": scheduler_diversity_capacity_estimate_values,
        "scheduler_diversity_floor_parallel_workers_values": (
            scheduler_diversity_floor_parallel_workers_values
        ),
        "scheduler_diversity_limited": scheduler_diversity_limited_rollout_count > 0,
        "scheduler_diversity_limited_rollout_count": scheduler_diversity_limited_rollout_count,
        "scheduler_diversity_floor_applied": scheduler_diversity_floor_applied_rollout_count > 0,
        "scheduler_diversity_floor_applied_rollout_count": (
            scheduler_diversity_floor_applied_rollout_count
        ),
        "scheduler_overlap_sensitive_rollout_count_values": (
            scheduler_overlap_sensitive_rollout_count_values
        ),
        "rollout_cli_stage_count": len(rollout_stage_entries),
        "rollout_quick_verification_count": len(quick_verification_payloads),
        "max_rollout_quick_test_pass_rate": (
            max(quick_verification_signal_scores) if quick_verification_signal_scores else None
        ),
        "avg_rollout_quick_test_pass_rate": (
            sum(quick_verification_signal_scores) / len(quick_verification_signal_scores)
            if quick_verification_signal_scores
            else None
        ),
        "max_rollout_quick_signal_score": (
            max(quick_verification_signal_scores) if quick_verification_signal_scores else None
        ),
        "avg_rollout_quick_signal_score": (
            sum(quick_verification_signal_scores) / len(quick_verification_signal_scores)
            if quick_verification_signal_scores
            else None
        ),
        "max_rollout_quick_observed_pass_rate": (
            max(quick_verification_observed_pass_rates)
            if quick_verification_observed_pass_rates
            else None
        ),
        "avg_rollout_quick_observed_pass_rate": (
            sum(quick_verification_observed_pass_rates)
            / len(quick_verification_observed_pass_rates)
            if quick_verification_observed_pass_rates
            else None
        ),
        "max_rollout_quick_expected_coverage_ratio": (
            max(quick_verification_expected_coverage_ratios)
            if quick_verification_expected_coverage_ratios
            else None
        ),
        "avg_rollout_quick_expected_coverage_ratio": (
            sum(quick_verification_expected_coverage_ratios)
            / len(quick_verification_expected_coverage_ratios)
            if quick_verification_expected_coverage_ratios
            else None
        ),
        "agentic_search_followup_event_count": len(followup_routing_events),
        "agentic_search_followup_memory_fired_count": sum(
            1 for event in followup_routing_events if bool(event.get("followup_memory_fired"))
        ),
        "agentic_search_gathered_information_fired_count": sum(
            1 for event in followup_routing_events if bool(event.get("gathered_information_fired"))
        ),
        "agentic_search_stall_detected_count": sum(
            1 for event in followup_routing_events if bool(event.get("stall_detected"))
        ),
        "agentic_search_external_contract_uncertainty_count": sum(
            1
            for event in followup_routing_events
            if bool(event.get("external_contract_uncertainty"))
        ),
        "agentic_search_online_evidence_used_count": sum(
            1 for event in followup_routing_events if bool(event.get("online_evidence_used"))
        ),
        "agentic_search_local_evidence_used_count": sum(
            1 for event in followup_routing_events if bool(event.get("local_doc_evidence_used"))
        ),
        "agentic_search_followup_improved_count": sum(
            1
            for event in followup_routing_events
            if bool((event.get("outcome") or {}).get("materially_improved"))
        ),
        "agentic_search_avg_followup_pass_rate_delta": (
            sum(followup_pass_rate_deltas) / len(followup_pass_rate_deltas)
            if followup_pass_rate_deltas
            else None
        ),
        "agentic_search_avg_followup_duration_delta_seconds": (
            sum(followup_duration_deltas) / len(followup_duration_deltas)
            if followup_duration_deltas
            else None
        ),
        "agentic_search_avg_followup_token_delta": (
            sum(followup_token_deltas) / len(followup_token_deltas)
            if followup_token_deltas
            else None
        ),
        "agentic_search_routing_events": copy.deepcopy(followup_routing_events),
        "delegation_policy_enabled_rollout_count": delegation_policy_enabled_rollout_count,
        "delegate_subtasks_enabled_rollout_count": delegate_subtasks_enabled_rollout_count,
        "delegate_parallelism_requested_values": delegate_parallelism_requested_values,
        "delegate_parallelism_values": delegate_parallelism_values,
        "delegate_task_split_count_values": delegate_task_split_count_values,
        "planned_delegate_subtask_total": planned_delegate_subtask_total,
        "delegation_group_count": delegation_group_count,
        "delegated_subtasks": delegated_subtasks,
        "successful_delegated_subtasks": successful_delegated_subtasks,
        "boundary_pressure_count": boundary_pressure_count,
        "boundary_requested_files": boundary_requested_files,
        "boundary_interface_symbols": boundary_interface_symbols,
        "boundary_followups": boundary_followups,
        "unsolvable_reason": unsolvable_reason,
    }


def _benchmark_task_outcome_label(
    *,
    task_success: bool,
    orchestrator_reached: bool,
    orchestrator_success: bool,
    skipped: bool,
) -> str:
    if task_success:
        return "task_success"
    if not orchestrator_reached:
        return "skipped_before_orchestrator" if skipped else "failed_before_orchestrator"
    if orchestrator_success:
        return "benchmark_rejected_after_orchestrator"
    return "orchestrator_failed"


def append_benchmark_task_outcome_trace(
    config: ApexConfig,
    *,
    output_dir: str | Path,
    benchmark_name: str,
    task_id: str,
    task_success: bool,
    orchestrator_reached: bool,
    orchestrator_success: bool,
    baseline_failed: bool,
    baseline_pass_rate: Optional[float] = None,
    final_pass_rate: Optional[float] = None,
    candidate_found: Optional[bool] = None,
    selected_rollout_id: Optional[int] = None,
    skipped: bool = False,
    skip_category: Optional[str] = None,
    duration_seconds: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    return append_controller_decision(
        config,
        output_dir=output_dir,
        stage="benchmark",
        decision_type="task_outcome",
        chosen_option=_benchmark_task_outcome_label(
            task_success=bool(task_success),
            orchestrator_reached=bool(orchestrator_reached),
            orchestrator_success=bool(orchestrator_success),
            skipped=bool(skipped),
        ),
        feature_view={
            "task_success": 1.0 if bool(task_success) else 0.0,
            "orchestrator_reached": 1.0 if bool(orchestrator_reached) else 0.0,
            "orchestrator_success": 1.0 if bool(orchestrator_success) else 0.0,
            "baseline_failed": 1.0 if bool(baseline_failed) else 0.0,
            "candidate_found": 1.0 if bool(candidate_found) else 0.0,
            "selected_rollout_present": 1.0 if selected_rollout_id is not None else 0.0,
            "skipped": 1.0 if bool(skipped) else 0.0,
            "baseline_pass_rate": float(baseline_pass_rate or 0.0),
            "final_pass_rate": float(final_pass_rate or 0.0),
            "duration_seconds": float(duration_seconds or 0.0),
        },
        metadata={
            "benchmark_name": str(benchmark_name or ""),
            "task_id": str(task_id or ""),
            "skip_category": str(skip_category or ""),
        },
        outcome={
            "task_success": bool(task_success),
            "orchestrator_reached": bool(orchestrator_reached),
            "orchestrator_success": bool(orchestrator_success),
            "baseline_failed": bool(baseline_failed),
            "candidate_found": bool(candidate_found),
            "selected_rollout_id": selected_rollout_id,
            "skipped": bool(skipped),
            "skip_category": str(skip_category or ""),
            "baseline_pass_rate": float(baseline_pass_rate or 0.0),
            "final_pass_rate": float(final_pass_rate or 0.0),
            "duration_seconds": float(duration_seconds or 0.0),
        },
    )


@dataclass
class BenchmarkTask:
    """One benchmark fixture."""

    name: str
    fixture_path: str
    repo_path: str
    issue_path: str
    test_command: str = "python3 -m pytest -q"
    notes: str = ""

    def load_issue(self) -> str:
        return Path(self.issue_path).read_text()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fixture_path": self.fixture_path,
            "repo_path": self.repo_path,
            "issue_path": self.issue_path,
            "test_command": self.test_command,
            "notes": self.notes,
        }


@dataclass
class BenchmarkTaskResult:
    """Execution result for one benchmark task."""

    task_name: str
    success: bool
    baseline_failed: bool
    final_tests_passed: bool
    orchestrator_success: bool = False
    selected_rollout_id: Optional[int] = None
    selected_worktree_path: Optional[str] = None
    total_tokens: int = 0
    duration_seconds: float = 0.0
    result_path: Optional[str] = None
    failure_reason: Optional[str] = None
    execution_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "success": self.success,
            "baseline_failed": self.baseline_failed,
            "final_tests_passed": self.final_tests_passed,
            "orchestrator_success": self.orchestrator_success,
            "selected_rollout_id": self.selected_rollout_id,
            "selected_worktree_path": self.selected_worktree_path,
            "total_tokens": self.total_tokens,
            "duration_seconds": self.duration_seconds,
            "result_path": self.result_path,
            "failure_reason": self.failure_reason,
            "execution_metadata": copy.deepcopy(self.execution_metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkTaskResult":
        final_tests_passed = bool(payload.get("final_tests_passed", False))
        uses_explicit_benchmark_success = "orchestrator_success" in payload
        return cls(
            task_name=str(payload["task_name"]),
            success=(
                bool(payload.get("success", False))
                if uses_explicit_benchmark_success
                else final_tests_passed
            ),
            baseline_failed=bool(payload.get("baseline_failed", False)),
            final_tests_passed=final_tests_passed,
            orchestrator_success=bool(
                payload.get("orchestrator_success", payload.get("success", False))
            ),
            selected_rollout_id=payload.get("selected_rollout_id"),
            selected_worktree_path=payload.get("selected_worktree_path"),
            total_tokens=int(payload.get("total_tokens", 0) or 0),
            duration_seconds=float(payload.get("duration_seconds", 0.0) or 0.0),
            result_path=payload.get("result_path"),
            failure_reason=payload.get("failure_reason"),
            execution_metadata=dict(payload.get("execution_metadata") or {}),
        )


@dataclass
class BenchmarkReport:
    """Aggregate benchmark report."""

    tasks: list[BenchmarkTaskResult] = field(default_factory=list)
    requested_task_ids: list[str] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    updated_at: float = 0.0
    report_kind: str = LOCAL_BENCHMARK_REPORT_KIND
    harness_name: str = LOCAL_BENCHMARK_HARNESS_NAME
    harness_version: str = LOCAL_BENCHMARK_HARNESS_VERSION
    config_source: Optional[str] = None
    model_config: list[dict[str, Any]] = field(default_factory=list)
    ablation_config: dict[str, Any] = field(default_factory=dict)
    run_manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_tasks(self) -> int:
        return sum(1 for task in self.tasks if task.success and task.final_tests_passed)

    @property
    def total_tasks(self) -> int:
        if self.requested_task_ids:
            return len(self.requested_task_ids)
        return len(self.tasks)

    @property
    def completed(self) -> bool:
        return self.finished_at > 0.0

    @property
    def duration_seconds(self) -> float:
        if self.started_at <= 0.0:
            return 0.0
        end_time = self.finished_at or self.updated_at or self.started_at
        return max(0.0, end_time - self.started_at)

    @property
    def failure_clusters(self) -> list[dict[str, Any]]:
        return cluster_failures(
            [task.to_dict() for task in self.tasks],
            benchmark_family="local",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_kind": self.report_kind,
            "harness_name": self.harness_name,
            "harness_version": self.harness_version,
            "config_source": self.config_source,
            "requested_task_ids": list(self.requested_task_ids),
            "model_config": copy.deepcopy(self.model_config),
            "started_at": self.started_at,
            "updated_at": self.updated_at or self.finished_at or self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "completed": self.completed,
            "resolved_tasks": self.resolved_tasks,
            "total_tasks": self.total_tasks,
            "ablation_config": copy.deepcopy(self.ablation_config),
            "run_manifest": manifest_summary(self.run_manifest),
            "failure_clusters": self.failure_clusters,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    def to_markdown(self) -> str:
        allocator_policy = (self.ablation_config.get("allocator") or {}).get("policy", "unknown")
        rollout_buckets = (self.ablation_config.get("allocator") or {}).get("rollout_buckets") or []
        overlap_diversity_cap_enabled = (self.ablation_config.get("allocator") or {}).get(
            "overlap_diversity_cap_enabled",
            False,
        )
        min_overlap_diversity_parallel_workers = (self.ablation_config.get("allocator") or {}).get(
            "min_overlap_diversity_parallel_workers",
            "n/a",
        )
        scaffold_policy = (self.ablation_config.get("scaffold") or {}).get("policy", "unknown")
        orchestrated_multi_agent_enabled = (self.ablation_config.get("scaffold") or {}).get(
            "orchestrated_multi_agent_enabled",
            False,
        )
        planner_brief_family_cap = (self.ablation_config.get("scaffold") or {}).get(
            "planner_brief_family_cap",
            "n/a",
        )
        task_state_graph_enabled = (self.ablation_config.get("scaffold") or {}).get(
            "task_state_graph_enabled",
            False,
        )
        frontier_targeting_enabled = (self.ablation_config.get("scaffold") or {}).get(
            "frontier_targeting_enabled",
            False,
        )
        dynamic_transitions = (self.ablation_config.get("scaffold") or {}).get(
            "dynamic_transitions_enabled",
            False,
        )
        feedback_config = self.ablation_config.get("feedback") or {}
        search_mode = (self.ablation_config.get("search") or {}).get("mode", "off")
        selection_config = self.ablation_config.get("selection") or {}
        memory_config = self.ablation_config.get("memory") or {}
        lines = [
            "# APEX Benchmark Report",
            "",
            f"- Harness: {self.harness_name} v{self.harness_version}",
            f"- Report kind: {self.report_kind}",
            f"- Status: {'completed' if self.completed else 'in_progress'}",
            f"- Config source: {self.config_source or 'default'}",
            f"- Resolved tasks: {self.resolved_tasks}/{self.total_tasks}",
            f"- Rollout allocator: {allocator_policy}",
            f"- Rollout buckets: {', '.join(str(bucket) for bucket in rollout_buckets) or 'n/a'}",
            (
                "- Overlap diversity cap: "
                f"{'enabled' if overlap_diversity_cap_enabled else 'disabled'} "
                f"(outer_floor={min_overlap_diversity_parallel_workers})"
            ),
            f"- Scaffold mode: {scaffold_policy}",
            (
                "- Orchestrated multi-agent delegation: "
                f"{'enabled' if orchestrated_multi_agent_enabled else 'disabled'}"
            ),
            f"- Planner brief family cap: {planner_brief_family_cap}",
            f"- Task-state graph: {'enabled' if task_state_graph_enabled else 'disabled'}",
            f"- Frontier targeting: {'enabled' if frontier_targeting_enabled else 'disabled'}",
            f"- Explicit search: {search_mode}",
            f"- COP transitions: {'enabled' if dynamic_transitions else 'disabled'}",
            (
                "- Rollout quick verification: "
                f"{'enabled' if feedback_config.get('quick_verification_enabled') else 'disabled'} "
                f"(max_tests={feedback_config.get('quick_verification_max_tests', 'n/a')}, "
                f"timeout={feedback_config.get('quick_verification_timeout_seconds', 'n/a')}s)"
            ),
            (
                "- Selection critic: "
                f"{'enabled' if selection_config.get('critic_reranking_enabled') else 'disabled'} "
                f"(weight={selection_config.get('critic_weight', 0)})"
            ),
            (
                "- Repo memory: "
                + (
                    "enabled (non-i.i.d.; disclose when comparing to fresh-run baselines)"
                    if memory_config.get("repo_memory_enabled")
                    else "disabled"
                )
            ),
            f"- Duration: {self.duration_seconds:.1f}s",
            "",
            "| Task | Success | Baseline Failed | Final Tests Passed | Selected Rollout | Tokens | Duration (s) |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for task in self.tasks:
            lines.append(
                "| {name} | {success} | {baseline} | {final} | {rollout} | {tokens} | {duration:.1f} |".format(
                    name=task.task_name,
                    success="yes" if task.success else "no",
                    baseline="yes" if task.baseline_failed else "no",
                    final="yes" if task.final_tests_passed else "no",
                    rollout=task.selected_rollout_id
                    if task.selected_rollout_id is not None
                    else "-",
                    tokens=task.total_tokens,
                    duration=task.duration_seconds,
                )
            )
        failure_clusters = self.failure_clusters
        if failure_clusters:
            lines.extend(
                [
                    "",
                    "## Failure Clusters",
                    "",
                    "| Root Cause | Count | Example Tasks |",
                    "| --- | --- | --- |",
                ]
            )
            for cluster in failure_clusters:
                lines.append(
                    "| {bucket} | {count} | {tasks} |".format(
                        bucket=cluster.get("bucket"),
                        count=cluster.get("count"),
                        tasks=", ".join(cluster.get("tasks") or []) or "-",
                    )
                )
        return "\n".join(lines)


class BenchmarkRunner:
    """Run APEX against local benchmark fixtures."""

    def __init__(
        self,
        config: ApexConfig,
        fixtures_dir: str,
        output_dir: str,
    ):
        self.config = config
        self.fixtures_dir = Path(fixtures_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.config_source: Optional[str] = None

    def discover_tasks(self) -> list[BenchmarkTask]:
        tasks = []
        for fixture_dir in sorted(self.fixtures_dir.iterdir()):
            if not fixture_dir.is_dir():
                continue
            repo_path = fixture_dir / "repo"
            issue_path = fixture_dir / "issue.md"
            if not repo_path.exists() or not issue_path.exists():
                continue

            task_json = fixture_dir / "task.json"
            task_data: dict[str, Any] = {}
            if task_json.exists():
                task_data = json.loads(task_json.read_text())

            tasks.append(
                BenchmarkTask(
                    name=task_data.get("name", fixture_dir.name),
                    fixture_path=str(fixture_dir),
                    repo_path=str(repo_path),
                    issue_path=str(issue_path),
                    test_command=task_data.get("test_command", "python3 -m pytest -q"),
                    notes=task_data.get("notes", ""),
                )
            )
        return tasks

    def run(self, task_names: Optional[list[str]] = None) -> BenchmarkReport:
        requested_task_ids: list[str]
        execution = {
            "entrypoint": "benchmark",
            "args": {
                "fixtures_dir": str(self.fixtures_dir),
                "task_names": list(task_names or []),
            },
        }
        existing_state = load_json_if_exists(self.output_dir / RUN_STATE_FILENAME) or {}
        report = BenchmarkReport(
            requested_task_ids=[],
            started_at=float(existing_state.get("started_at") or time.time()),
            updated_at=time.time(),
            config_source=self.config_source,
            model_config=copy.deepcopy(self.config.to_dict().get("llm_configs", [])),
            ablation_config=build_apex_ablation_config(self.config),
        )
        tasks = self.discover_tasks()
        if task_names:
            task_set = set(task_names)
            tasks = [task for task in tasks if task.name in task_set]
        requested_task_ids = [task.name for task in tasks]
        report.requested_task_ids = list(requested_task_ids)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        report.run_manifest = ensure_run_manifest(
            self.output_dir,
            build_run_manifest(
                config=self.config,
                report_kind=report.report_kind,
                harness_name=report.harness_name,
                harness_version=report.harness_version,
                benchmark_family="local",
                output_dir=self.output_dir,
                config_source=report.config_source,
                requested_task_ids=requested_task_ids,
                execution=execution,
                benchmark_policy=build_local_benchmark_policy(),
            ),
        )

        ordered_task_ids = [task.name for task in tasks]
        completed_results: dict[str, BenchmarkTaskResult] = {}
        pending_tasks: list[BenchmarkTask] = []
        for task in tasks:
            checkpointed = self._load_checkpointed_task_result(task)
            if checkpointed is not None:
                completed_results[task.name] = checkpointed
            else:
                pending_tasks.append(task)

        def refresh_report_tasks() -> None:
            report.tasks = [
                completed_results[task_id]
                for task_id in ordered_task_ids
                if task_id in completed_results
            ]

        refresh_report_tasks()
        self._write_report_checkpoint(
            report,
            requested_task_ids,
            completed=False,
            execution=execution,
        )
        for task in pending_tasks:
            completed_results[task.name] = self._run_task_with_checkpoint(task)
            refresh_report_tasks()
            self._write_report_checkpoint(
                report,
                requested_task_ids,
                completed=False,
                execution=execution,
            )

        self._write_report_checkpoint(
            report,
            requested_task_ids,
            completed=True,
            execution=execution,
        )
        return report

    def _task_output_dir(self, task: BenchmarkTask) -> Path:
        return self.output_dir / task.name

    def _task_workspace_dir(self, task: BenchmarkTask) -> Path:
        return self.output_dir / "workspaces" / task.name

    def _run_task_with_checkpoint(self, task: BenchmarkTask) -> BenchmarkTaskResult:
        ensure_clean_directory_for_task(self._task_output_dir(task), completed=False)
        ensure_clean_directory_for_task(self._task_workspace_dir(task), completed=False)
        result = self._run_task(task)
        write_task_checkpoint(self._task_output_dir(task), result.to_dict())
        return result

    def _load_checkpointed_task_result(self, task: BenchmarkTask) -> Optional[BenchmarkTaskResult]:
        payload = load_json_if_exists(task_result_path(self._task_output_dir(task)))
        if payload is None:
            return None
        try:
            return BenchmarkTaskResult.from_dict(payload)
        except Exception:
            return None

    def _write_report_checkpoint(
        self,
        report: BenchmarkReport,
        requested_task_ids: list[str],
        *,
        completed: bool,
        execution: dict[str, Any],
    ) -> None:
        report.updated_at = time.time()
        report.finished_at = report.updated_at if completed else 0.0
        atomic_write_json(
            self.output_dir / RUN_STATE_FILENAME,
            build_run_state(
                report_kind=report.report_kind,
                harness_name=report.harness_name,
                harness_version=report.harness_version,
                started_at=report.started_at,
                requested_task_ids=requested_task_ids,
                completed_task_ids=[task.task_name for task in report.tasks],
                successful_tasks=sum(1 for task in report.tasks if task.success),
                failed_tasks=sum(1 for task in report.tasks if not task.success),
                completed=completed,
                metadata={
                    "config_source": report.config_source,
                    "model_config": copy.deepcopy(report.model_config),
                    "ablation_config": copy.deepcopy(report.ablation_config),
                },
            ),
        )
        update_run_manifest(
            self.output_dir,
            requested_task_ids=requested_task_ids,
            completed_task_ids=[task.task_name for task in report.tasks],
            completed=completed,
            extra_updates={
                "config_payload": self.config.to_dict(),
                "environment_snapshot": capture_environment_snapshot(self.config),
                "prompt_template_fingerprints": build_prompt_template_fingerprints(),
                "execution": dict(execution),
            },
        )
        report.run_manifest = load_run_manifest(self.output_dir) or report.run_manifest
        atomic_write_json(self.output_dir / "benchmark_report.json", report.to_dict())
        atomic_write_text(self.output_dir / "benchmark_report.md", report.to_markdown())

    def _run_task(self, task: BenchmarkTask) -> BenchmarkTaskResult:
        started = time.time()
        sandbox = Path(tempfile.mkdtemp(prefix=f"apex-benchmark-{task.name}-"))
        try:
            repo_copy = sandbox / "repo"
            copy_tree(task.repo_path, repo_copy)
            self._init_git_repo(repo_copy)

            baseline = self._run_command(repo_copy, task.test_command)
            baseline_failed = baseline.returncode != 0

            task_output_dir = self._task_output_dir(task)
            task_workspace_dir = self._task_workspace_dir(task)
            task_output_dir.mkdir(parents=True, exist_ok=True)
            task_workspace_dir.mkdir(parents=True, exist_ok=True)
            write_task_live_state(
                task_output_dir,
                {
                    "task_id": task.name,
                    "phase": "baseline",
                    "status": "active",
                    "last_progress_at": time.time(),
                    "process_pid": os.getpid(),
                },
            )

            config = copy.deepcopy(self.config)
            config.output_dir = str(task_output_dir)
            config.workspace_dir = str(task_workspace_dir)
            keep_worktrees = config.rollout.keep_worktrees
            config.rollout.keep_worktrees = True
            target_tool_env, target_tool_diagnostics = target_tool_env_overrides(
                workdir=repo_copy,
                output_dir=task_output_dir / "target_runtime_tools",
                timeout_seconds=120,
                runtime=host_env_runtime(os.environ, description="local_benchmark_host_runtime"),
                label=f"local_{task.name}",
            )
            apply_target_tool_env_to_apex_config(config, target_tool_env)
            atomic_write_json(
                task_output_dir / "target_runtime_tools.json",
                target_tool_diagnostics,
            )

            orchestrator = ApexOrchestrator(config)
            write_task_live_state(
                task_output_dir,
                {
                    "task_id": task.name,
                    "phase": "solving",
                    "status": "active",
                    "last_progress_at": time.time(),
                    "process_pid": os.getpid(),
                },
            )
            result = orchestrator.solve(
                repo_path=str(repo_copy),
                issue_description=task.load_issue(),
                test_command=task.test_command,
            )

            final_tests_passed = False
            failure_reason = None
            if result.selected_worktree_path:
                write_task_live_state(
                    task_output_dir,
                    {
                        "task_id": task.name,
                        "phase": "evaluation",
                        "status": "active",
                        "last_progress_at": time.time(),
                        "process_pid": os.getpid(),
                    },
                )
                final_run = self._run_command(
                    Path(result.selected_worktree_path), task.test_command
                )
                final_tests_passed = final_run.returncode == 0
                if final_run.returncode != 0:
                    failure_reason = final_run.output
            else:
                failure_reason = result.explanation or "No worktree selected."

            if not keep_worktrees:
                shutil.rmtree(task_workspace_dir, ignore_errors=True)

            task_result = BenchmarkTaskResult(
                task_name=task.name,
                success=final_tests_passed,
                baseline_failed=baseline_failed,
                final_tests_passed=final_tests_passed,
                orchestrator_success=bool(result.success),
                selected_rollout_id=result.selected_rollout_id,
                selected_worktree_path=result.selected_worktree_path,
                total_tokens=result.total_tokens,
                duration_seconds=time.time() - started,
                result_path=str(task_output_dir / "apex_result.json"),
                failure_reason=failure_reason,
                execution_metadata=extract_apex_execution_metadata(result),
            )
            write_task_live_state(
                task_output_dir,
                {
                    "task_id": task.name,
                    "phase": "completed",
                    "status": "completed",
                    "last_progress_at": time.time(),
                    "process_pid": os.getpid(),
                    "success": task_result.success,
                    "final_tests_passed": task_result.final_tests_passed,
                },
            )
            return task_result
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)

    def _init_git_repo(self, repo_path: Path) -> None:
        self._run_command(repo_path, "git init")
        self._run_command(repo_path, "git config user.email apex@example.com")
        self._run_command(repo_path, "git config user.name APEX")
        self._run_command(repo_path, "git add -A")
        self._run_command(repo_path, "git commit -m baseline")

    def _run_command(self, cwd: Path, command: str, timeout: int = 120) -> "_CommandResult":
        result = run_shell_command(
            command,
            cwd,
            timeout=timeout,
        )
        return _CommandResult(
            returncode=result.returncode,
            output=(result.stdout + result.stderr).strip(),
        )


@dataclass
class _CommandResult:
    returncode: int
    output: str
