"""
Unified controller decision trace logging.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .controller_models import normalize_numeric_features
from .controller_schema import coerce_controller_action

_TRACE_LOCK = threading.Lock()


def controller_trace_enabled(config: Any) -> bool:
    trace_config = getattr(config, "controller_trace", None)
    return bool(getattr(trace_config, "enabled", True))


def controller_trace_path(
    config: Any,
    *,
    output_dir: Optional[str | Path] = None,
) -> Path:
    root = Path(output_dir or getattr(config, "output_dir", "/tmp/apex_output"))
    trace_config = getattr(config, "controller_trace", None)
    filename = str(
        getattr(trace_config, "filename", "controller_decisions.jsonl")
        or "controller_decisions.jsonl"
    )
    return root / filename


def _max_logged_options(config: Any) -> int:
    trace_config = getattr(config, "controller_trace", None)
    return max(1, int(getattr(trace_config, "max_options", 6) or 6))


def _serialize_option(option: Any) -> dict[str, Any]:
    if hasattr(option, "to_dict"):
        payload = dict(option.to_dict())
    elif isinstance(option, dict):
        payload = dict(option)
    else:
        payload = {"option_id": str(option or "")}
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        metadata = dict(metadata)
        if metadata.get("controller_action") is not None:
            metadata["controller_action"] = coerce_controller_action(
                metadata.get("controller_action")
            ).to_dict()
        serialized_policy_evaluation = _serialize_policy_evaluation(
            metadata.get("policy_evaluation")
        )
        if serialized_policy_evaluation is not None:
            metadata["policy_evaluation"] = serialized_policy_evaluation
        payload["metadata"] = metadata
    return payload


def _serialize_trace_metadata(metadata: Optional[dict[str, Any]]) -> dict[str, Any]:
    payload = dict(metadata or {})
    serialized_chosen_eval = _serialize_policy_evaluation(payload.get("chosen_policy_evaluation"))
    if serialized_chosen_eval is not None:
        payload["chosen_policy_evaluation"] = serialized_chosen_eval
    serialized_policy_eval = _serialize_policy_evaluation(payload.get("policy_evaluation"))
    if serialized_policy_eval is not None:
        payload["policy_evaluation"] = serialized_policy_eval
    return payload


def _serialize_policy_evaluation(payload: Any) -> Optional[dict[str, Any]]:
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()
    if not isinstance(payload, dict):
        return None
    return dict(payload)


def _append_unique_policy_evaluation(
    target: list[dict[str, Any]],
    seen: set[str],
    payload: Any,
) -> None:
    serialized = _serialize_policy_evaluation(payload)
    if not serialized:
        return
    cache_key = json.dumps(serialized, sort_keys=True, default=str)
    if cache_key in seen:
        return
    seen.add(cache_key)
    target.append(serialized)


def _collect_policy_evaluations(
    *,
    chosen_option: str,
    options: list[dict[str, Any]],
    metadata: dict[str, Any],
    explicit_policy_evaluations: Optional[Iterable[Any]],
) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in list(explicit_policy_evaluations or []):
        _append_unique_policy_evaluation(evaluations, seen, item)

    _append_unique_policy_evaluation(
        evaluations,
        seen,
        metadata.get("chosen_policy_evaluation"),
    )
    _append_unique_policy_evaluation(
        evaluations,
        seen,
        metadata.get("policy_evaluation"),
    )

    selected_option: Optional[dict[str, Any]] = None
    for option in options:
        if bool(option.get("selected")):
            selected_option = option
            break
    if selected_option is None:
        for option in options:
            if str(option.get("option_id") or "") == chosen_option:
                selected_option = option
                break
    if isinstance(selected_option, dict):
        option_metadata = selected_option.get("metadata")
        if isinstance(option_metadata, dict):
            _append_unique_policy_evaluation(
                evaluations,
                seen,
                option_metadata.get("policy_evaluation"),
            )
    return evaluations


@dataclass
class ControllerDecisionEvent:
    stage: str
    decision_type: str
    chosen_option: str
    feature_view: dict[str, float] = field(default_factory=dict)
    options: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    outcome: dict[str, Any] = field(default_factory=dict)
    policy_evaluations: list[dict[str, Any]] = field(default_factory=list)
    policy_version: str = ""
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: f"ctrl-{uuid.uuid4().hex}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": float(self.timestamp),
            "stage": self.stage,
            "decision_type": self.decision_type,
            "chosen_option": self.chosen_option,
            "feature_view": {
                str(name): round(float(value), 6)
                for name, value in dict(self.feature_view or {}).items()
            },
            "options": list(self.options),
            "metadata": dict(self.metadata or {}),
            "outcome": dict(self.outcome or {}),
            "policy_evaluations": list(self.policy_evaluations),
            "policy_version": self.policy_version,
        }


def append_controller_decision(
    config: Any,
    *,
    stage: str,
    decision_type: str,
    chosen_option: str,
    feature_view: Optional[dict[str, Any]] = None,
    options: Optional[Iterable[Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
    outcome: Optional[dict[str, Any]] = None,
    policy_evaluations: Optional[Iterable[Any]] = None,
    output_dir: Optional[str | Path] = None,
) -> Optional[dict[str, Any]]:
    if not controller_trace_enabled(config):
        return None
    path = controller_trace_path(config, output_dir=output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    max_options = _max_logged_options(config)
    serialized_options = [_serialize_option(option) for option in list(options or [])]
    serialized_metadata = _serialize_trace_metadata(metadata)
    evaluations = _collect_policy_evaluations(
        chosen_option=str(chosen_option or "").strip(),
        options=serialized_options,
        metadata=serialized_metadata,
        explicit_policy_evaluations=policy_evaluations,
    )
    event = ControllerDecisionEvent(
        stage=str(stage or "").strip(),
        decision_type=str(decision_type or "").strip(),
        chosen_option=str(chosen_option or "").strip(),
        feature_view=normalize_numeric_features(feature_view),
        options=serialized_options[:max_options],
        metadata=serialized_metadata,
        outcome=dict(outcome or {}),
        policy_evaluations=evaluations,
        policy_version=str(
            getattr(getattr(config, "controller_models", None), "policy_version", "") or ""
        ),
    )
    payload = event.to_dict()
    with _TRACE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")
    return payload
