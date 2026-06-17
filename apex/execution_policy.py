"""
Backend-agnostic execution policy helpers used by the rollout engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .controller_models import PolicyModelEvaluation, evaluate_policy_model, option_feature_view
from .controller_schema import coerce_controller_action


def _issue_plan_task_regime_probability(issue_plan: Any, state: str) -> float:
    task_regime = getattr(issue_plan, "task_regime", None)
    if isinstance(task_regime, dict):
        probabilities = dict(task_regime.get("state_probabilities") or {})
        return float(probabilities.get(state) or 0.0)
    probability = getattr(task_regime, "probability", None)
    if callable(probability):
        return float(probability(state) or 0.0)
    probabilities = getattr(task_regime, "state_probabilities", {})
    return float(dict(probabilities or {}).get(state) or 0.0)


@dataclass
class CompletionExecutionFeatures:
    action_mode: str
    action_origin: str
    regime_state: str
    failing_test_count: int
    passing_test_count: int
    incomplete_source_count: int
    incomplete_test_count: int
    focus_file_count: int
    relevant_file_count: int
    symbol_count: int
    edit_span_count: int
    difficulty_estimate: float
    requires_broad_validation: bool
    contract_gap_probability: float
    interface_probability: float
    importability_probability: float
    is_completion_task: bool
    delegated_patcher: bool
    delegated_subtask: bool

    def to_feature_dict(self) -> dict[str, float]:
        return {
            "action_mode_is_test_rooted": 1.0 if self.action_mode == "test_rooted" else 0.0,
            "action_mode_is_dependency_trace": (
                1.0 if self.action_mode == "dependency_trace" else 0.0
            ),
            "action_mode_is_api_contract": 1.0 if self.action_mode == "api_contract" else 0.0,
            "action_mode_is_invariant_guard": (
                1.0 if self.action_mode == "invariant_guard" else 0.0
            ),
            "action_origin_is_regime_candidate": (
                1.0 if self.action_origin == "regime_candidate" else 0.0
            ),
            "regime_state_is_importability_blocker": (
                1.0 if self.regime_state == "importability_blocker" else 0.0
            ),
            "failing_test_count": float(self.failing_test_count),
            "passing_test_count": float(self.passing_test_count),
            "incomplete_source_count": float(self.incomplete_source_count),
            "incomplete_test_count": float(self.incomplete_test_count),
            "focus_file_count": float(self.focus_file_count),
            "relevant_file_count": float(self.relevant_file_count),
            "symbol_count": float(self.symbol_count),
            "edit_span_count": float(self.edit_span_count),
            "difficulty_estimate": float(self.difficulty_estimate),
            "requires_broad_validation": 1.0 if self.requires_broad_validation else 0.0,
            "contract_gap_probability": float(self.contract_gap_probability),
            "interface_probability": float(self.interface_probability),
            "importability_probability": float(self.importability_probability),
            "is_completion_task": 1.0 if self.is_completion_task else 0.0,
            "delegated_patcher": 1.0 if self.delegated_patcher else 0.0,
            "delegated_subtask": 1.0 if self.delegated_subtask else 0.0,
        }


def build_completion_execution_features(
    issue_plan: Any,
    brief: Any,
    *,
    requires_broad_validation: bool,
) -> CompletionExecutionFeatures:
    test_context = getattr(issue_plan, "test_context", None)
    search_policy = getattr(brief, "search_policy", None)
    action = coerce_controller_action(
        getattr(brief, "controller_action", None),
        fallback_policy=search_policy if isinstance(search_policy, dict) else {},
        default_files=list(getattr(brief, "focus_files", []) or []),
    )
    failing_test_ids = list(getattr(test_context, "failing_test_ids", []) or [])
    failing_test_count = max(
        int(getattr(test_context, "failing_test_count", 0) or 0),
        len(failing_test_ids),
    )
    return CompletionExecutionFeatures(
        action_mode=str(action.mode or "surgical"),
        action_origin=str(action.origin or "heuristic"),
        regime_state=str(action.regime_state or ""),
        failing_test_count=failing_test_count,
        passing_test_count=max(
            int(getattr(test_context, "passing_test_count", 0) or 0),
            len(list(getattr(test_context, "passing_test_ids", []) or [])),
        ),
        incomplete_source_count=len(
            list(getattr(test_context, "incomplete_source_files", []) or [])
        ),
        incomplete_test_count=len(list(getattr(test_context, "incomplete_test_files", []) or [])),
        focus_file_count=len(list(getattr(brief, "focus_files", []) or [])),
        relevant_file_count=len(list(getattr(issue_plan, "relevant_files", []) or [])),
        symbol_count=len(list(action.symbols or [])),
        edit_span_count=len(list(action.edit_spans or [])),
        difficulty_estimate=float(getattr(issue_plan, "difficulty_estimate", 0.0) or 0.0),
        requires_broad_validation=bool(requires_broad_validation),
        contract_gap_probability=_issue_plan_task_regime_probability(issue_plan, "contract_gap"),
        interface_probability=_issue_plan_task_regime_probability(
            issue_plan, "high_interface_risk"
        ),
        importability_probability=_issue_plan_task_regime_probability(
            issue_plan, "importability_blocker"
        ),
        is_completion_task=bool(
            dict(getattr(issue_plan, "allocator_features", {}) or {}).get("is_completion_task")
        ),
        delegated_patcher=bool(
            callable(getattr(brief, "delegation_enabled", None))
            and brief.delegation_enabled("patcher")
        ),
        # Prefer the typed `ControllerAction.delegated_subtask` field as the
        # single source of truth. Fall back to the planner_metadata dict for
        # back-compat with planners that haven't been migrated to populate the
        # typed field yet.
        delegated_subtask=bool(
            getattr(action, "delegated_subtask", False)
            or dict(getattr(issue_plan, "planner_metadata", {}) or {}).get("delegated_subtask")
        ),
    )


def should_preserve_primary_completion_model(
    config: Any,
    *,
    features: CompletionExecutionFeatures,
) -> tuple[bool, PolicyModelEvaluation, list[str]]:
    completion_policy = getattr(getattr(config, "rollout", None), "completion_policy", None)
    reasons: list[str] = []
    preserve = False
    if features.delegated_patcher:
        preserve = True
        reasons.append("delegated patcher remains on primary model")
    if features.failing_test_count >= int(
        getattr(completion_policy, "preserve_primary_min_failing_tests", 6) or 6
    ):
        preserve = True
        reasons.append("failing surface already broad")
    if features.incomplete_source_count >= int(
        getattr(completion_policy, "preserve_primary_min_incomplete_sources", 3) or 3
    ):
        preserve = True
        reasons.append("incomplete source scaffolds suggest broad completion work")
    if features.incomplete_test_count + features.failing_test_count >= int(
        getattr(completion_policy, "preserve_primary_min_focus_test_failures", 3) or 3
    ) and features.focus_file_count >= int(
        getattr(completion_policy, "preserve_primary_min_focus_files", 4) or 4
    ):
        preserve = True
        reasons.append("focus width is broad enough to keep the primary model")
    if features.relevant_file_count >= int(
        getattr(completion_policy, "preserve_primary_min_relevant_files", 6) or 6
    ):
        preserve = True
        reasons.append("relevant-file surface is broad")
    if features.symbol_count >= 3 or features.edit_span_count >= 3:
        preserve = True
        reasons.append("symbol/edit-span breadth indicates interface-sensitive work")
    if features.difficulty_estimate >= float(
        getattr(completion_policy, "preserve_primary_difficulty_threshold", 0.55) or 0.55
    ):
        preserve = True
        reasons.append("difficulty estimate is already high")
    heuristic_value = 1.0 if preserve else 0.0
    evaluation = evaluate_policy_model(
        getattr(config, "controller_models", None),
        model_name="rollout.preserve_primary_completion",
        features={
            **features.to_feature_dict(),
            "heuristic_score": heuristic_value,
        },
        baseline_value=heuristic_value,
        lower=0.0,
        upper=1.0,
    )
    return bool(evaluation.value >= 0.5), evaluation, reasons


def effective_cli_timeout_seconds(
    config: Any,
    *,
    configured_seconds: Optional[int],
    features: CompletionExecutionFeatures,
) -> tuple[Optional[int], PolicyModelEvaluation, list[str]]:
    if not isinstance(configured_seconds, int) or configured_seconds <= 0:
        return None, PolicyModelEvaluation("", False, 0.0, 0.0, 0.0), []
    completion_policy = getattr(getattr(config, "rollout", None), "completion_policy", None)
    reasons: list[str] = []
    extend = False
    if features.requires_broad_validation:
        extend = True
        reasons.append("broad validation required")
    if features.failing_test_count >= int(
        getattr(completion_policy, "timeout_broad_validation_min_failing_tests", 6) or 6
    ):
        extend = True
        reasons.append("failing test surface warrants extra timeout")
    if (features.incomplete_source_count + features.incomplete_test_count) >= int(
        getattr(completion_policy, "timeout_broad_validation_min_incomplete_files", 2) or 2
    ):
        extend = True
        reasons.append("scaffold breadth warrants extra timeout")
    if features.relevant_file_count >= int(
        getattr(completion_policy, "timeout_broad_validation_min_relevant_files", 8) or 8
    ):
        extend = True
        reasons.append("relevant-file breadth warrants extra timeout")
    heuristic_extension = 0.0
    if extend:
        heuristic_extension = float(
            getattr(completion_policy, "timeout_extension_seconds", 600) or 600
        )
        if (
            features.is_completion_task
            or features.failing_test_count
            >= int(getattr(completion_policy, "timeout_extra_min_failing_tests", 12) or 12)
            or (features.incomplete_source_count + features.incomplete_test_count)
            >= int(getattr(completion_policy, "timeout_extra_min_incomplete_files", 4) or 4)
            or features.relevant_file_count
            >= int(getattr(completion_policy, "timeout_extra_min_relevant_files", 12) or 12)
        ):
            heuristic_extension += float(
                getattr(completion_policy, "timeout_extra_extension_seconds", 300) or 300
            )
            reasons.append("extra-long completion profile detected")
    evaluation = evaluate_policy_model(
        getattr(config, "controller_models", None),
        model_name="rollout.timeout_extension_seconds",
        features={
            **features.to_feature_dict(),
            "heuristic_score": heuristic_extension,
        },
        baseline_value=heuristic_extension,
        lower=0.0,
    )
    extension = max(0, int(round(float(evaluation.value or 0.0))))
    timeout_seconds = configured_seconds + extension
    if not features.delegated_subtask:
        return timeout_seconds, evaluation, reasons

    delegated_multiplier_heuristic = float(
        getattr(completion_policy, "delegated_timeout_multiplier", 0.60) or 0.60
    )
    delegated_evaluation = evaluate_policy_model(
        getattr(config, "controller_models", None),
        model_name="rollout.delegated_timeout_multiplier",
        features={
            **features.to_feature_dict(),
            "heuristic_score": delegated_multiplier_heuristic,
        },
        baseline_value=delegated_multiplier_heuristic,
        lower=0.10,
        upper=1.0,
    )
    delegated_multiplier = min(
        1.0,
        max(0.10, float(delegated_evaluation.value or delegated_multiplier_heuristic)),
    )
    delegated_min_seconds = max(
        1,
        int(getattr(completion_policy, "delegated_timeout_min_seconds", 900) or 900),
    )
    delegated_max_seconds = max(
        delegated_min_seconds,
        int(getattr(completion_policy, "delegated_timeout_max_seconds", 1800) or 1800),
    )
    delegated_timeout_seconds = int(round(timeout_seconds * delegated_multiplier))
    delegated_timeout_seconds = max(
        delegated_min_seconds,
        min(delegated_timeout_seconds, delegated_max_seconds),
    )
    reasons.append("delegated subtasks use scaled timeout policy")
    return delegated_timeout_seconds, delegated_evaluation, reasons


def score_bootstrap_option(
    config: Any,
    *,
    option_id: str,
    heuristic_score: float,
    features: CompletionExecutionFeatures,
) -> PolicyModelEvaluation:
    return evaluate_policy_model(
        getattr(config, "controller_models", None),
        model_name="rollout.bootstrap_score",
        features=option_feature_view(
            features.to_feature_dict(),
            option_id=option_id,
            heuristic_score=heuristic_score,
        ),
        baseline_value=float(heuristic_score or 0.0),
        lower=0.0,
        upper=1.0,
    )
