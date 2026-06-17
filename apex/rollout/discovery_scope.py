"""
Helpers for turning issue plans and stage artifacts into discovery-query scopes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agents.artifacts import (
    coerce_localization_artifact,
    coerce_patch_artifact,
    coerce_reproduction_artifact,
)
from ..planning.manager import IssuePlan, RolloutBrief
from .localizer_scope import is_repo_relative_editable_path, is_test_path


@dataclass
class DiscoveryScope:
    """Prioritized retrieval scope for cross-rollout discoveries."""

    stage_name: str = ""
    file_paths: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "file_paths": list(self.file_paths),
            "symbols": list(self.symbols),
            "test_ids": list(self.test_ids),
        }


def build_discovery_scope(
    issue_plan: IssuePlan,
    rollout_brief: RolloutBrief,
    *,
    stage_name: str,
    reproduction_artifact: Any = None,
    localization_artifact: Any = None,
    patch_artifact: Any = None,
) -> DiscoveryScope:
    """Build a stage-aware discovery retrieval scope from plan/artifact state."""
    reproduction = coerce_reproduction_artifact(reproduction_artifact)
    localization = coerce_localization_artifact(localization_artifact)
    patch = coerce_patch_artifact(patch_artifact)
    test_context = issue_plan.test_context
    evidence_mode = _issue_plan_evidence_mode(issue_plan)
    search_policy = dict(rollout_brief.search_policy or {})
    search_mode = str(search_policy.get("mode") or "").strip().lower()
    verification_focus = str(search_policy.get("verification_focus") or "").strip().lower()
    graph_target_files = [
        str(path) for path in list(search_policy.get("graph_target_file_paths") or []) if path
    ]
    graph_target_symbols = [
        str(symbol) for symbol in list(search_policy.get("graph_target_symbols") or []) if symbol
    ]
    graph_target_test_ids = [
        str(test_id)
        for test_id in list(search_policy.get("graph_target_test_ids") or [])
        if test_id
    ]

    file_paths: list[str] = []
    symbols: list[str] = []
    test_ids: list[str] = []

    if stage_name == "reproducer":
        file_paths.extend(graph_target_files)
        file_paths.extend(test_context.focus_test_files)
        file_paths.extend(rollout_brief.focus_files)
        file_paths.extend(issue_plan.risk_files)
        test_ids.extend(graph_target_test_ids)
        test_ids.extend(test_context.failing_test_ids)
        test_ids.extend(test_context.passing_test_ids[:2])
    elif stage_name == "localizer":
        file_paths.extend(graph_target_files)
        file_paths.extend(rollout_brief.focus_files)
        file_paths.extend(issue_plan.risk_files)
        file_paths.extend(issue_plan.relevant_files)
        file_paths.extend(test_context.focus_test_files)
        symbols.extend(graph_target_symbols)
        test_ids.extend(graph_target_test_ids)
        if reproduction and reproduction.script_path:
            file_paths.append(reproduction.script_path)
        test_ids.extend(test_context.failing_test_ids)
        if reproduction and reproduction.command:
            test_ids.append(reproduction.command)
    elif stage_name in {"patcher", "full_solver"}:
        file_paths.extend(graph_target_files)
        symbols.extend(graph_target_symbols)
        test_ids.extend(graph_target_test_ids)
        if localization:
            file_paths.extend(localization.files)
            symbols.extend(localization.symbols)
        if patch:
            file_paths.extend(patch.changed_files)
            test_ids.extend(patch.tests_run)
        file_paths.extend(rollout_brief.focus_files)
        file_paths.extend(issue_plan.risk_files)
        file_paths.extend(issue_plan.relevant_files)
        test_ids.extend(test_context.failing_test_ids)
        if reproduction and reproduction.command:
            test_ids.append(reproduction.command)
    elif stage_name == "test_writer":
        file_paths.extend(graph_target_files)
        symbols.extend(graph_target_symbols)
        test_ids.extend(graph_target_test_ids)
        file_paths.extend(test_context.focus_test_files)
        if localization:
            file_paths.extend(localization.files)
            symbols.extend(localization.symbols)
        if patch:
            file_paths.extend(patch.changed_files)
            test_ids.extend(patch.tests_run)
        file_paths.extend(rollout_brief.focus_files)
        test_ids.extend(test_context.failing_test_ids)
        if reproduction and reproduction.command:
            test_ids.append(reproduction.command)
    else:
        file_paths.extend(graph_target_files)
        symbols.extend(graph_target_symbols)
        test_ids.extend(graph_target_test_ids)
        file_paths.extend(rollout_brief.focus_files)
        file_paths.extend(issue_plan.risk_files)
        file_paths.extend(issue_plan.relevant_files)
        test_ids.extend(test_context.failing_test_ids)

    if verification_focus == "failing_tests":
        test_ids = list(test_context.failing_test_ids) + test_ids
    elif verification_focus == "focus_test_files":
        file_paths = list(test_context.focus_test_files) + file_paths

    if search_mode == "dependency_trace":
        file_paths.extend(issue_plan.risk_files)
    elif search_mode == "api_contract":
        file_paths.extend(test_context.focus_test_files)
        test_ids.extend(test_context.expectations)
    elif search_mode == "source_cluster":
        file_paths.extend(rollout_brief.focus_files[:4])

    # TIER 2 (T2.3): for a decomposition-scale module-group rollout, narrow the
    # discovery scope to the group's owned + bridge files so retrieval stays
    # inside the group. Pure size/structure-triggered (the keys are only set on
    # module-group briefs); non-decomposition rollouts are unaffected.
    if bool(search_policy.get("decomposition_module_group")):
        owned = [
            str(path)
            for path in list(search_policy.get("module_group_owned_files") or [])
            if path
        ]
        bridge = [
            str(path)
            for path in list(search_policy.get("module_group_bridge_files") or [])
            if path
        ]
        group_test_ids = [
            str(test_id)
            for test_id in list(search_policy.get("module_group_expected_test_ids") or [])
            if test_id
        ]
        if owned:
            # Group-owned files lead; bridge files follow as read-context.
            file_paths = owned + bridge + file_paths
        if group_test_ids:
            test_ids = group_test_ids + test_ids

    return DiscoveryScope(
        stage_name=stage_name,
        file_paths=_normalize_file_paths(
            file_paths,
            evidence_mode=evidence_mode,
            incomplete_test_files=test_context.incomplete_test_files,
        )[:8],
        symbols=_normalize_tokens(symbols)[:6],
        test_ids=_normalize_tokens(test_ids)[:8],
    )


def _issue_plan_evidence_mode(issue_plan: IssuePlan) -> str:
    try:
        policy = issue_plan.evaluation_constraints.resolved_evidence_policy()
        mode = str(getattr(policy, "mode", "") or "").strip()
        if mode:
            return mode
    except Exception:
        pass
    test_context = getattr(issue_plan, "test_context", None)
    return str(getattr(test_context, "evidence_mode", "") or "").strip()


def _normalize_file_paths(
    values: list[str],
    *,
    evidence_mode: str = "",
    incomplete_test_files: list[str] | None = None,
) -> list[str]:
    incomplete = {
        _normalize_path_hint(path)
        for path in list(incomplete_test_files or [])
        if _normalize_path_hint(path)
    }
    normalized: list[str] = []
    for value in values:
        text = (value or "").strip()
        if not text:
            continue
        path = _normalize_path_hint(text)
        if (
            str(evidence_mode or "").strip() == "gold_suite_visible"
            and is_test_path(path)
            and path not in incomplete
        ):
            continue
        if not is_repo_relative_editable_path(
            path,
            evidence_mode=evidence_mode,
            incomplete_test_files=incomplete,
        ):
            continue
        normalized.append(path)
    return list(dict.fromkeys(item for item in normalized if item))


def _normalize_path_hint(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    if "::" in text:
        text = text.split("::", 1)[0]
    if " " in text and text.endswith(".py"):
        text = text.rsplit(" ", 1)[-1]
    return Path(text).as_posix()


def _normalize_tokens(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        text = (value or "").strip()
        if text:
            normalized.append(text)
    return list(dict.fromkeys(normalized))
