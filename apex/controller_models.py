"""
Config-backed calibrated controller models shared across planning, search, and rollout.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

_CALIBRATED_WEIGHTS_LOCK = threading.Lock()
_CALIBRATED_LIBRARY_CACHE: Optional["ControllerModelLibraryConfig"] = None
_CALIBRATED_LIBRARY_CACHE_DIR: Optional[Path] = None


def calibrated_weights_dir() -> Path:
    """Return the on-disk directory hosting calibrated controller weight bundles."""

    return Path(__file__).resolve().parent / "configs" / "controller_models"


def reset_calibrated_library_cache() -> None:
    """Drop the cached calibrated model library; useful for tests."""

    global _CALIBRATED_LIBRARY_CACHE, _CALIBRATED_LIBRARY_CACHE_DIR
    with _CALIBRATED_WEIGHTS_LOCK:
        _CALIBRATED_LIBRARY_CACHE = None
        _CALIBRATED_LIBRARY_CACHE_DIR = None


def _clamp(value: float, *, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def feature_slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def normalize_numeric_features(features: Optional[Mapping[str, Any]]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in dict(features or {}).items():
        name = str(key or "").strip()
        if not name:
            continue
        if isinstance(value, bool):
            normalized[name] = 1.0 if value else 0.0
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            normalized[name] = float(value)
    return normalized


@dataclass
class LinearPolicyModelConfig:
    """One lightweight offline-fit policy model."""

    enabled: bool = True
    intercept: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)
    transform: str = "identity"
    blend: float = 1.0
    lower: Optional[float] = None
    upper: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "intercept": float(self.intercept or 0.0),
            "weights": {
                str(name): float(weight)
                for name, weight in dict(self.weights or {}).items()
                if str(name)
            },
            "transform": str(self.transform or "identity"),
            "blend": float(self.blend or 0.0),
            "lower": self.lower,
            "upper": self.upper,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LinearPolicyModelConfig":
        return cls(
            enabled=bool(data.get("enabled")),
            intercept=float(data.get("intercept") or 0.0),
            weights={
                str(name): float(weight)
                for name, weight in dict(data.get("weights") or {}).items()
                if str(name)
            },
            transform=str(data.get("transform") or "identity"),
            blend=float(data.get("blend") or 0.0),
            lower=(float(data["lower"]) if isinstance(data.get("lower"), (int, float)) else None),
            upper=(float(data["upper"]) if isinstance(data.get("upper"), (int, float)) else None),
        )


@dataclass
class ControllerModelLibraryConfig:
    """Named calibrated models plus a policy version string for traceability."""

    policy_version: str = "calibrated-v1-synthetic"
    models: dict[str, LinearPolicyModelConfig] = field(default_factory=dict)
    # Phase A.3 (Decisive-Edge): top-level kill switch for the whole
    # calibrated-controller library. ``--benchmark-mode publication``
    # turns this on; ``--benchmark-mode headline`` turns it off so the
    # leaderboard run uses the legacy uncalibrated controller path.
    # When ``False`` the library acts as if no models are registered:
    # ``model(...)`` still works (existing keys remain) but the
    # orchestration layer is expected to consult :attr:`library_enabled`
    # before applying any calibrated decision.
    library_enabled: bool = True

    def model(self, name: str) -> Optional[LinearPolicyModelConfig]:
        return dict(self.models or {}).get(str(name or "").strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": str(self.policy_version or "calibrated-v1-synthetic"),
            "models": {
                str(name): model.to_dict()
                for name, model in dict(self.models or {}).items()
                if str(name) and isinstance(model, LinearPolicyModelConfig)
            },
            "library_enabled": bool(self.library_enabled),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ControllerModelLibraryConfig":
        raw_enabled = data.get("library_enabled", True)
        return cls(
            policy_version=str(data.get("policy_version") or "calibrated-v1-synthetic"),
            models={
                str(name): (
                    model
                    if isinstance(model, LinearPolicyModelConfig)
                    else LinearPolicyModelConfig.from_dict(dict(model or {}))
                )
                for name, model in dict(data.get("models") or {}).items()
                if str(name)
            },
            library_enabled=bool(raw_enabled) if raw_enabled is not None else True,
        )


@dataclass
class PolicyModelEvaluation:
    """Serialized description of one calibrated decision score."""

    model_name: str
    applied: bool
    value: float
    baseline_value: float
    raw_output: float
    features: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "applied": bool(self.applied),
            "value": round(float(self.value), 6),
            "baseline_value": round(float(self.baseline_value), 6),
            "raw_output": round(float(self.raw_output), 6),
            "features": {
                str(name): round(float(value), 6)
                for name, value in dict(self.features or {}).items()
            },
            "metadata": dict(self.metadata or {}),
        }


def option_feature_view(
    features: Optional[Mapping[str, Any]],
    *,
    option_id: str,
    heuristic_score: float,
) -> dict[str, float]:
    result = normalize_numeric_features(features)
    slug = feature_slug(option_id)
    result["heuristic_score"] = float(heuristic_score or 0.0)
    result["option_present"] = 1.0
    if slug:
        result[f"option__{slug}"] = 1.0
    return result


def _apply_transform(raw_output: float, transform: str) -> float:
    mode = str(transform or "identity").strip().lower()
    if mode == "identity":
        return float(raw_output)
    if mode == "logistic":
        bounded = max(min(float(raw_output), 30.0), -30.0)
        return 1.0 / (1.0 + math.exp(-bounded))
    if mode == "clamp01":
        return _clamp(float(raw_output))
    if mode == "round_int":
        return float(int(round(float(raw_output))))
    return float(raw_output)


def _load_calibrated_model_from_disk(
    model_name: str,
    *,
    weights_dir: Optional[Path] = None,
) -> Optional[tuple[LinearPolicyModelConfig, str]]:
    """Best-effort loader for a single calibrated model bundle.

    Returns the resolved ``(model, policy_version)`` pair or ``None`` if no
    bundle is available. The lookup is cached and silently tolerates missing
    or malformed files so the heuristic baseline always remains the fallback.
    """

    target_dir = (weights_dir or calibrated_weights_dir()).resolve()
    global _CALIBRATED_LIBRARY_CACHE, _CALIBRATED_LIBRARY_CACHE_DIR
    with _CALIBRATED_WEIGHTS_LOCK:
        cache_invalid = (
            _CALIBRATED_LIBRARY_CACHE is None or _CALIBRATED_LIBRARY_CACHE_DIR != target_dir
        )
        if cache_invalid:
            library = ControllerModelLibraryConfig(policy_version="heuristic-bootstrap-v1")
            highest_policy_version: Optional[str] = None
            if target_dir.exists():
                for path in sorted(target_dir.glob("*.json")):
                    try:
                        payload = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    if not isinstance(payload, dict):
                        continue
                    regime = str(payload.get("regime") or path.stem).strip()
                    if not regime:
                        continue
                    weights_payload = payload.get("weights")
                    if isinstance(weights_payload, list):
                        feature_names = list(payload.get("feature_names") or [])
                        weights_map = {
                            str(name): float(weight)
                            for name, weight in zip(feature_names, weights_payload)
                            if str(name)
                        }
                    elif isinstance(weights_payload, dict):
                        weights_map = {
                            str(name): float(weight)
                            for name, weight in weights_payload.items()
                            if str(name)
                        }
                    else:
                        weights_map = {}
                    intercept = float(payload.get("intercept") or 0.0)
                    transform = str(payload.get("transform") or "logistic")
                    blend = float(payload.get("blend") or 1.0)
                    lower = payload.get("lower")
                    upper = payload.get("upper")
                    model = LinearPolicyModelConfig(
                        enabled=True,
                        intercept=intercept,
                        weights=weights_map,
                        transform=transform,
                        blend=blend,
                        lower=float(lower) if isinstance(lower, (int, float)) else None,
                        upper=float(upper) if isinstance(upper, (int, float)) else None,
                    )
                    library.models[f"regime.{regime}"] = model
                    candidate_version = str(payload.get("policy_version") or "").strip()
                    if candidate_version and (
                        highest_policy_version is None or candidate_version > highest_policy_version
                    ):
                        highest_policy_version = candidate_version
            if highest_policy_version:
                library.policy_version = highest_policy_version
            _CALIBRATED_LIBRARY_CACHE = library
            _CALIBRATED_LIBRARY_CACHE_DIR = target_dir
        cached_library = _CALIBRATED_LIBRARY_CACHE

    model = cached_library.model(model_name)
    if model is None:
        return None
    return model, cached_library.policy_version


def evaluate_heuristic_baseline(
    *,
    model_name: str,
    features: Optional[Mapping[str, Any]],
    baseline_value: float,
    policy_version: str = "heuristic-bootstrap-v1",
) -> PolicyModelEvaluation:
    """Heuristic-only fallback evaluation, exposed for the A/B harness."""

    baseline = float(baseline_value or 0.0)
    normalized_features = normalize_numeric_features(features)
    return PolicyModelEvaluation(
        model_name=str(model_name or ""),
        applied=False,
        value=baseline,
        baseline_value=baseline,
        raw_output=baseline,
        features=normalized_features,
        metadata={"policy_version": str(policy_version or "heuristic-bootstrap-v1")},
    )


def evaluate_policy_model(
    model_library: Any,
    *,
    model_name: str,
    features: Optional[Mapping[str, Any]],
    baseline_value: float,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
) -> PolicyModelEvaluation:
    normalized_features = normalize_numeric_features(features)
    baseline = float(baseline_value or 0.0)
    library = (
        model_library
        if isinstance(model_library, ControllerModelLibraryConfig)
        else (
            ControllerModelLibraryConfig.from_dict(dict(model_library or {}))
            if isinstance(model_library, dict)
            else None
        )
    )
    library_policy_version = (
        library.policy_version if library is not None else "heuristic-bootstrap-v1"
    )
    model = library.model(model_name) if library is not None else None
    disk_policy_version: Optional[str] = None
    # Only fall through to the on-disk calibrated weights if the library has no
    # entry at all for this model.  An explicit disabled override stays
    # heuristic-only (the A/B harness depends on this contract).
    if model is None and str(model_name or "").strip():
        disk_hit = _load_calibrated_model_from_disk(model_name)
        if disk_hit is not None:
            model, disk_policy_version = disk_hit
    if model is None or not bool(model.enabled):
        metadata: dict[str, Any] = {}
        if library is not None:
            metadata["policy_version"] = library_policy_version
        return PolicyModelEvaluation(
            model_name=str(model_name or ""),
            applied=False,
            value=baseline,
            baseline_value=baseline,
            raw_output=baseline,
            features=normalized_features,
            metadata=metadata,
        )

    raw_output = float(model.intercept or 0.0)
    for feature_name, weight in dict(model.weights or {}).items():
        raw_output += float(weight or 0.0) * float(normalized_features.get(str(feature_name), 0.0))

    transformed = _apply_transform(raw_output, model.transform)
    lower_bound = lower if lower is not None else model.lower
    upper_bound = upper if upper is not None else model.upper
    if isinstance(lower_bound, (int, float)):
        transformed = max(float(lower_bound), transformed)
    if isinstance(upper_bound, (int, float)):
        transformed = min(float(upper_bound), transformed)
    blend = _clamp(float(model.blend or 0.0))
    value = ((1.0 - blend) * baseline) + (blend * transformed)
    if isinstance(lower_bound, (int, float)):
        value = max(float(lower_bound), value)
    if isinstance(upper_bound, (int, float)):
        value = min(float(upper_bound), value)
    resolved_policy_version = disk_policy_version if disk_policy_version else library_policy_version
    return PolicyModelEvaluation(
        model_name=str(model_name or ""),
        applied=True,
        value=float(value),
        baseline_value=baseline,
        raw_output=float(raw_output),
        features=normalized_features,
        metadata={
            "policy_version": resolved_policy_version,
            "transform": str(model.transform or "identity"),
            "blend": blend,
            "lower": lower_bound,
            "upper": upper_bound,
        },
    )
