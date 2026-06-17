"""
Structured artifacts exchanged between staged rollout agents.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, TypeVar

_ArtifactT = TypeVar("_ArtifactT", bound="ArtifactBase")


@dataclass
class ArtifactBase:
    """Base helpers shared by staged-agent artifacts."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls: type[_ArtifactT], payload: dict[str, Any] | None) -> _ArtifactT:
        if not payload:
            return cls()
        allowed = {item.name for item in fields(cls)}
        normalized = {
            key: value for key, value in payload.items() if key in allowed and value is not None
        }
        return cls(**normalized)

    @classmethod
    def from_json(cls: type[_ArtifactT], text: str) -> _ArtifactT:
        return cls.from_dict(json.loads(text))


@dataclass
class ReproductionArtifact(ArtifactBase):
    """Structured output from the reproduction stage."""

    summary: str = ""
    command: str = ""
    script_path: str = ""
    script_content: str = ""
    observed_output: str = ""


@dataclass
class LocalizationArtifact(ArtifactBase):
    """Structured output from the localization stage."""

    summary: str = ""
    files: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)


@dataclass
class PatchArtifact(ArtifactBase):
    """Structured output from the patch stage."""

    summary: str = ""
    tests_run: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    confidence: float | None = None
    followups: list[str] = field(default_factory=list)


@dataclass
class TestSuiteArtifact(ArtifactBase):
    """Structured output from the test-writer stage."""

    __test__ = False
    summary: str = ""
    framework: str = ""
    language: str = ""
    test_code: str = ""
    test_descriptions: list[str] = field(default_factory=list)
    test_artifacts: list[dict[str, Any]] = field(default_factory=list)
    portfolio_summary: str = ""
    promotion_summary: str = ""
    validation_summary: dict[str, Any] = field(default_factory=dict)
    contract_hypotheses: list[str] = field(default_factory=list)
    reference_targets: list[str] = field(default_factory=list)
    task_contract: dict[str, Any] = field(default_factory=dict)
    milestones: list[dict[str, Any]] = field(default_factory=list)
    test_objectives: list[dict[str, Any]] = field(default_factory=list)
    regression_suite_summary: dict[str, Any] = field(default_factory=dict)
    minimization_summary: dict[str, Any] = field(default_factory=dict)
    promoted_artifact_ids: list[str] = field(default_factory=list)
    candidate_artifact_ids: list[str] = field(default_factory=list)
    exploratory_artifact_ids: list[str] = field(default_factory=list)
    public_signal: dict[str, Any] = field(default_factory=dict)


def render_artifact_json(payload: Any) -> str:
    """Render a structured artifact or fallback payload as JSON."""
    if payload is None:
        return ""
    if isinstance(payload, ArtifactBase):
        return payload.to_json()
    if isinstance(payload, dict):
        return json.dumps(payload, indent=2, sort_keys=True)
    if isinstance(payload, str):
        return json.dumps({"summary": payload}, indent=2, sort_keys=True)
    return json.dumps({"value": str(payload)}, indent=2, sort_keys=True)


def _coerce_artifact(
    payload: Any,
    artifact_cls: type[_ArtifactT],
) -> _ArtifactT | None:
    if payload is None:
        return None
    if isinstance(payload, artifact_cls):
        return payload
    if isinstance(payload, dict):
        return artifact_cls.from_dict(payload)
    if isinstance(payload, str):
        if "summary" in {item.name for item in fields(artifact_cls)}:
            return artifact_cls.from_dict({"summary": payload})
        return artifact_cls()
    if isinstance(payload, ArtifactBase):
        return artifact_cls.from_dict(payload.to_dict())
    return artifact_cls.from_dict({"summary": str(payload)})


def coerce_reproduction_artifact(payload: Any) -> ReproductionArtifact | None:
    return _coerce_artifact(payload, ReproductionArtifact)


def coerce_localization_artifact(payload: Any) -> LocalizationArtifact | None:
    return _coerce_artifact(payload, LocalizationArtifact)


def coerce_patch_artifact(payload: Any) -> PatchArtifact | None:
    return _coerce_artifact(payload, PatchArtifact)


def coerce_test_suite_artifact(payload: Any) -> TestSuiteArtifact | None:
    return _coerce_artifact(payload, TestSuiteArtifact)
