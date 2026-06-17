"""Opt-in component ablation assignment, enforcement helpers, and telemetry."""

from __future__ import annotations

import hashlib
from typing import Any


ABLATION_COMPONENTS = (
    "localization",
    "reflection_memory",
    "multi_rollout",
    "selector_vote",
    "full_suite_gate",
)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _bool_from_config_mapping(mapping: dict[str, Any], key: str) -> bool:
    value = mapping.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _int_from_config_mapping(mapping: dict[str, Any], key: str, default: int) -> int:
    value = mapping.get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def component_ablation_proof_run(config: Any) -> bool:
    benchmark = getattr(config, "benchmark", None)
    runtime_policy = _as_dict(getattr(benchmark, "runtime_policy", None))
    reporting = _as_dict(getattr(benchmark, "reporting", None))
    return _bool_from_config_mapping(runtime_policy, "proof_run") or _bool_from_config_mapping(
        reporting,
        "proof_run",
    )


def component_ablation_seed(config: Any, default: int = 0) -> int:
    benchmark = getattr(config, "benchmark", None)
    runtime_policy = _as_dict(getattr(benchmark, "runtime_policy", None))
    reporting = _as_dict(getattr(benchmark, "reporting", None))
    if "component_ablation_seed" in runtime_policy:
        return _int_from_config_mapping(runtime_policy, "component_ablation_seed", default)
    return _int_from_config_mapping(reporting, "component_ablation_seed", default)


def _float_from_config_mapping(mapping: dict[str, Any], key: str, default: float) -> float:
    value = mapping.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _runtime_policy(config: Any) -> dict[str, Any]:
    return _as_dict(getattr(getattr(config, "benchmark", None), "runtime_policy", None))


def clarification_abstain_enabled(config: Any) -> bool:
    """WS3I: gate the clarification-abstain behavioral arm. DEFAULT OFF."""
    return _bool_from_config_mapping(_runtime_policy(config), "clarification_abstain_enabled")


def anti_repetition_downrank_enabled(config: Any) -> bool:
    """WS3I: gate the anti-repetition downrank behavioral arm. DEFAULT OFF."""
    return _bool_from_config_mapping(_runtime_policy(config), "anti_repetition_downrank_enabled")


def anti_repetition_downrank_penalty(config: Any, default: float = 0.25) -> float:
    """WS3I: calibration knob — score penalty applied to a repeat-shape candidate."""
    return _float_from_config_mapping(
        _runtime_policy(config), "anti_repetition_downrank_penalty", default
    )


def anti_repetition_min_overlap(config: Any, default: float = 0.75) -> float:
    """WS3I: calibration knob — file-overlap threshold to call a candidate a repeat."""
    return _float_from_config_mapping(
        _runtime_policy(config), "anti_repetition_min_overlap", default
    )


def behavioral_arms_summary(config: Any) -> dict[str, Any]:
    """WS3I: telemetry snapshot of the behavioral arm gates (for diagnostics)."""
    return {
        "clarification_abstain_enabled": clarification_abstain_enabled(config),
        "anti_repetition_downrank_enabled": anti_repetition_downrank_enabled(config),
        "anti_repetition_downrank_penalty": anti_repetition_downrank_penalty(config),
        "anti_repetition_min_overlap": anti_repetition_min_overlap(config),
    }


def component_ablation_task_id(
    *,
    issue_plan: Any = None,
    issue_description: str = "",
    repo_label: str = "",
) -> str:
    summary = str(getattr(issue_plan, "summary", "") or issue_description or "").strip()
    repo = str(repo_label or "").strip()
    return f"{repo}:{summary}" if repo else summary


def component_ablation_assignment_for_task(
    *,
    config: Any,
    issue_plan: Any = None,
    issue_description: str = "",
    repo_label: str = "",
    seed: int | None = None,
    proof_run: bool | None = None,
) -> dict[str, Any]:
    metadata = _as_dict(getattr(issue_plan, "planner_metadata", None))
    existing = _as_dict(metadata.get("component_ablation"))
    if existing:
        return dict(existing)
    return component_ablation_assignment(
        config=config,
        task_id=component_ablation_task_id(
            issue_plan=issue_plan,
            issue_description=issue_description,
            repo_label=repo_label,
        ),
        seed=component_ablation_seed(config) if seed is None else seed,
        proof_run=component_ablation_proof_run(config) if proof_run is None else proof_run,
    )


def component_ablation_assignment(
    *,
    config: Any,
    task_id: str,
    seed: int = 0,
    proof_run: bool = False,
) -> dict[str, Any]:
    benchmark = getattr(config, "benchmark", None)
    runtime_policy = _as_dict(getattr(benchmark, "runtime_policy", None))
    reporting = _as_dict(getattr(benchmark, "reporting", None))
    enabled = bool(
        runtime_policy.get("component_ablation_enabled")
        or reporting.get("component_ablation_enabled")
    )
    if proof_run or not enabled:
        return {
            "enabled": False,
            "disabled_component": None,
            "reason": "disabled_for_proof_run" if proof_run else "component_ablation_not_enabled",
        }
    components = list(runtime_policy.get("component_ablation_components") or ABLATION_COMPONENTS)
    components = [str(component) for component in components if str(component) in ABLATION_COMPONENTS]
    if not components:
        components = list(ABLATION_COMPONENTS)
    digest = hashlib.sha256(f"{task_id}:{seed}".encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % (len(components) + 1)
    disabled = None if index == len(components) else components[index]
    return {
        "enabled": True,
        "disabled_component": disabled,
        "arm": "control" if disabled is None else f"disable_{disabled}",
        "components": components,
    }


def component_enabled(assignment: dict[str, Any], component: str) -> bool:
    if not bool(assignment.get("enabled")):
        return True
    return str(assignment.get("disabled_component") or "") != str(component)


def component_disabled(assignment: dict[str, Any], component: str) -> bool:
    return not component_enabled(assignment, component)


# WS2C: OPT-IN components (opposite polarity to ABLATION_COMPONENTS, which are
# default-ON/ablation-disables). An optional component is OFF unless explicitly
# listed in the assignment's ``enabled_optional_components``.
OPTIONAL_COMPONENTS = ("eg_critic",)


def component_optional_enabled(assignment: dict[str, Any], component: str) -> bool:
    """WS2C: True ONLY when the assignment is enabled AND ``component`` is listed
    in ``enabled_optional_components``. Default False (missing/empty assignment)."""
    assignment = _as_dict(assignment)
    if not assignment.get("enabled"):
        return False
    enabled_optional = assignment.get("enabled_optional_components")
    if not isinstance(enabled_optional, (list, tuple, set)):
        return False
    return component in {str(c) for c in enabled_optional}
