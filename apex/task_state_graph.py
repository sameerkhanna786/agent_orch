"""
Shared task-state graph used to coordinate planning, search, and selection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

logger = logging.getLogger(__name__)

from .acceptance import quick_verification_signal_score, rollout_has_authoritative_acceptance
from .controller_policy import TaskRegimeProfile

if TYPE_CHECKING:
    from .planning.manager import IssuePlan


def _clamp(value: float, *, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).lower()


def _dedupe_preserve(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _normalize_file_path_token(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.split("::", 1)[0]
    text = re.sub(r":\d+(?::\d+)?$", "", text)
    text = text.strip().strip("`'\"")
    text = text.strip(".,;:()[]{}<>")
    text = text.replace("\\", "/").lstrip("./")
    if not text:
        return ""
    return Path(text).as_posix()


def _normalize_file_paths(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        text = _normalize_file_path_token(value)
        if not text:
            continue
        normalized.append(text)
    return list(dict.fromkeys(item for item in normalized if item))


def _extract_test_identifiers(values: list[str]) -> list[str]:
    def normalize_identifier(raw: str) -> str:
        identifier = str(raw or "").strip().strip("`'\"")
        identifier = identifier.rstrip(".,;:")
        if identifier[:1] in "[({<" and ".py::" in identifier[1:]:
            opener = identifier[0]
            closer = {"[": "]", "(": ")", "{": "}", "<": ">"}.get(opener)
            if closer and identifier.endswith(closer):
                identifier = identifier[1:-1].strip()
            else:
                identifier = identifier[1:].strip()
        while identifier.endswith(")") and identifier.count("(") < identifier.count(")"):
            identifier = identifier[:-1].rstrip()
        while identifier.endswith("]") and identifier.count("[") < identifier.count("]"):
            identifier = identifier[:-1].rstrip()
        while identifier.endswith("}") and identifier.count("{") < identifier.count("}"):
            identifier = identifier[:-1].rstrip()
        return identifier.rstrip(".,;:")

    identifiers: list[str] = []
    for value in values:
        text = (value or "").strip()
        if not text:
            continue
        if "::" in text:
            identifiers.extend(
                identifier
                for match in re.finditer(r"[^\s'\"]+::[^\s'\"]+", text)
                for identifier in [normalize_identifier(match.group(0))]
                if identifier
            )
            continue
        if text.endswith(".py"):
            identifiers.append(Path(text).as_posix())
        else:
            identifiers.extend(
                Path(match.group(0)).as_posix() for match in re.finditer(r"[^\s'\"]+\.py", text)
            )
    return list(dict.fromkeys(item for item in identifiers if item))


def _normalize_tokens(values: list[str]) -> list[str]:
    tokens = [(value or "").strip() for value in values]
    return list(dict.fromkeys(token for token in tokens if token))


def _model_source_family(model: str) -> str:
    token = (model or "").strip().lower()
    if not token:
        return ""
    if token.startswith("openai/") or "gpt" in token or "codex" in token:
        return "openai"
    if token.startswith("anthropic/") or "claude" in token or token == "opus":
        return "anthropic"
    if token.startswith("gemini") or token.startswith("google/"):
        return "google"
    if token.startswith("meta/") or "llama" in token or "avocado" in token:
        return "meta"
    head = token.split("/", 1)[0].split("-", 1)[0]
    return head


def _artifact_values(payload: Any, key: str) -> list[str]:
    if isinstance(payload, dict):
        values = payload.get(key)
    else:
        values = getattr(payload, key, None)
    if not isinstance(values, (list, tuple, set)):
        return []
    return [str(value) for value in values if value]


def _artifact_text(payload: Any, key: str) -> str:
    if isinstance(payload, dict):
        value = payload.get(key)
    else:
        value = getattr(payload, key, None)
    return (value or "").strip() if isinstance(value, str) else ""


def _looks_like_test_path(path: str) -> bool:
    lowered_path = path.lower()
    name = Path(path).name.lower()
    parts = {part.lower() for part in Path(path).parts}
    return (
        "test" in parts
        or "tests" in parts
        or "__tests__" in parts
        or name.startswith("test_")
        or name.endswith("_test.py")
        or "/tests/" in lowered_path
    )


def _candidate_repo_paths(issue_plan: "IssuePlan", rollout_result: Any) -> list[str]:
    paths: list[str] = []
    paths.extend(list(getattr(issue_plan, "relevant_files", []) or []))
    paths.extend(list(getattr(issue_plan, "risk_files", []) or []))

    test_context = getattr(issue_plan, "test_context", None)
    if test_context is not None:
        for field_name in (
            "focus_test_files",
            "incomplete_test_files",
            "source_focus_files",
            "incomplete_source_files",
            "terminal_source_files",
            "failing_test_ids",
            "passing_test_ids",
            "expectations",
        ):
            values = getattr(test_context, field_name, None)
            if isinstance(values, (list, tuple, set)):
                paths.extend(str(value) for value in values if value)

    paths.extend(list(getattr(rollout_result, "changed_files", []) or []))
    paths.extend(_artifact_values(getattr(rollout_result, "patch_artifact", None), "changed_files"))
    paths.extend(_artifact_values(getattr(rollout_result, "localization_artifact", None), "files"))
    return _normalize_file_paths(paths)


def _normalize_reported_python_path(
    candidate: str,
    known_paths: set[str],
) -> Optional[str]:
    if not candidate:
        return None
    normalized = _normalize_file_path_token(candidate)
    if not normalized:
        return None
    if normalized in known_paths:
        if "/" not in normalized:
            suffix_matches = [
                known_path for known_path in known_paths if known_path.endswith("/" + normalized)
            ]
            if len(suffix_matches) == 1:
                return suffix_matches[0]
        return normalized
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
        for known_path in known_paths:
            if normalized.endswith("/" + known_path) or normalized.endswith(known_path):
                return known_path
        return None
    suffix_matches = [
        known_path for known_path in known_paths if known_path.endswith("/" + normalized)
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return Path(normalized).as_posix()


def _extract_traceback_frame(
    line: str,
    known_paths: set[str],
) -> Optional[tuple[str, int]]:
    match = re.search(r"((?:[A-Za-z]:)?/?[A-Za-z0-9_./\\\\-]+\.py):(\d+)", line)
    if not match:
        return None
    normalized = _normalize_reported_python_path(match.group(1), known_paths)
    if not normalized:
        return None
    try:
        line_number = int(match.group(2))
    except ValueError:
        return None
    return normalized, line_number


def _extract_exception_summary(line: str) -> Optional[str]:
    text = line.strip()
    if not text:
        return None
    summary_patterns = (
        r"^ERROR\s+.+?-\s+([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)):\s*(.+)$",
        r"^(?:E\s+)?([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)):\s*(.+)$",
    )
    for pattern in summary_patterns:
        match = re.match(pattern, text)
        if match:
            return f"{match.group(1)}: {match.group(2).strip()}".strip()
    return None


def _extract_test_ids_from_output(output: str) -> list[str]:
    test_ids: list[str] = []
    for token in re.findall(r"[^\s'\"`]+\.py(?:::[^\s'\"`]+)?", output or ""):
        base, _, suffix = token.partition("::")
        normalized_base = _normalize_file_path_token(base)
        if not _looks_like_test_path(normalized_base):
            continue
        test_ids.append(f"{normalized_base}::{suffix}" if suffix else normalized_base)
    return _dedupe_preserve(test_ids)


def _extract_python_file_paths(
    output: str,
    known_paths: set[str],
) -> list[str]:
    paths: list[str] = []
    for token in re.findall(r"[^\s'\"`]+\.py(?:::[^\s'\"`]+)?", output or ""):
        base = token.split("::", 1)[0]
        normalized = _normalize_known_repo_python_path(base, known_paths)
        if normalized:
            paths.append(normalized)
    return _normalize_file_paths(paths)


def _normalize_known_repo_python_path(
    candidate: str,
    known_paths: set[str],
) -> Optional[str]:
    if not candidate:
        return None
    normalized = candidate.strip().lstrip("./").replace("\\", "/")
    normalized = normalized.split("::", 1)[0]
    if not normalized or not normalized.endswith(".py"):
        return None
    if normalized in known_paths:
        if "/" not in normalized:
            suffix_matches = [
                known_path for known_path in known_paths if known_path.endswith("/" + normalized)
            ]
            if len(suffix_matches) == 1:
                return suffix_matches[0]
        return normalized
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
        for known_path in known_paths:
            if normalized.endswith("/" + known_path) or normalized.endswith(known_path):
                return known_path
        return None
    suffix_matches = [
        known_path for known_path in known_paths if known_path.endswith("/" + normalized)
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if "/" in normalized:
        return Path(normalized).as_posix()
    return None


@dataclass
class _TracebackSignal:
    source_files: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)
    terminal_source_files: list[str] = field(default_factory=list)
    exception_summaries: list[str] = field(default_factory=list)


def _extract_traceback_signal(
    output: str,
    known_paths: set[str],
) -> _TracebackSignal:
    if not output:
        return _TracebackSignal()

    source_frames: list[str] = []
    terminal_source_files: list[str] = []
    exception_summaries: list[str] = []
    current_frames: list[tuple[str, int]] = []

    for line in output.splitlines():
        frame = _extract_traceback_frame(line, known_paths)
        if frame is not None:
            current_frames.append(frame)
            if not _looks_like_test_path(frame[0]):
                source_frames.append(frame[0])
            continue

        exception_summary = _extract_exception_summary(line)
        if not exception_summary:
            continue

        terminal_frame = next(
            (item for item in reversed(current_frames) if not _looks_like_test_path(item[0])),
            None,
        )
        if terminal_frame is not None:
            terminal_source_files.append(terminal_frame[0])
            exception_summaries.append(
                f"{exception_summary} @ {terminal_frame[0]}:{terminal_frame[1]}"
            )
        else:
            exception_summaries.append(exception_summary)
        current_frames = []

    ordered_source_files = _dedupe_preserve(list(reversed(source_frames)))
    if not terminal_source_files and ordered_source_files:
        terminal_source_files = ordered_source_files[:2]

    return _TracebackSignal(
        source_files=ordered_source_files,
        test_ids=_extract_test_ids_from_output(output),
        terminal_source_files=_dedupe_preserve(terminal_source_files),
        exception_summaries=_dedupe_preserve(exception_summaries)[:4],
    )


@dataclass
class TaskEvidence:
    """One piece of structured evidence used by the global controller."""

    evidence_id: str
    kind: str
    source: str
    rollout_id: Optional[int]
    summary: str
    confidence: float
    source_model: str = ""
    source_family: str = ""
    accepted: bool = False
    independently_verified: bool = False
    negative: bool = False
    file_paths: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    related_obligation_ids: list[str] = field(default_factory=list)
    related_hypothesis_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "source": self.source,
            "rollout_id": self.rollout_id,
            "summary": self.summary,
            "confidence": self.confidence,
            "source_model": self.source_model,
            "source_family": self.source_family,
            "accepted": self.accepted,
            "independently_verified": self.independently_verified,
            "negative": self.negative,
            "file_paths": list(self.file_paths),
            "test_ids": list(self.test_ids),
            "symbols": list(self.symbols),
            "related_obligation_ids": list(self.related_obligation_ids),
            "related_hypothesis_ids": list(self.related_hypothesis_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskEvidence":
        return cls(
            evidence_id=str(data.get("evidence_id") or ""),
            kind=str(data.get("kind") or ""),
            source=str(data.get("source") or ""),
            rollout_id=(
                int(data["rollout_id"]) if isinstance(data.get("rollout_id"), int) else None
            ),
            summary=str(data.get("summary") or ""),
            confidence=float(data.get("confidence") or 0.0),
            source_model=str(data.get("source_model") or ""),
            source_family=str(data.get("source_family") or ""),
            accepted=bool(data.get("accepted")),
            independently_verified=bool(data.get("independently_verified")),
            negative=bool(data.get("negative")),
            file_paths=_normalize_file_paths(list(data.get("file_paths") or [])),
            test_ids=_extract_test_identifiers(list(data.get("test_ids") or [])),
            symbols=_normalize_tokens(list(data.get("symbols") or [])),
            related_obligation_ids=_normalize_tokens(
                list(data.get("related_obligation_ids") or [])
            ),
            related_hypothesis_ids=_normalize_tokens(
                list(data.get("related_hypothesis_ids") or [])
            ),
        )


@dataclass
class TaskObligation:
    """Behavioral requirement the controller still needs to satisfy."""

    obligation_id: str
    description: str
    source: str
    priority: float
    file_paths: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)
    resolution_confidence: float = 0.0
    coverage_score: float = 0.0
    stagnation_score: float = 0.0
    attempt_count: int = 0
    accepted_count: int = 0
    evidence_ids: list[str] = field(default_factory=list)

    @property
    def outstanding_score(self) -> float:
        unresolved = max(0.0, 1.0 - self.resolution_confidence)
        return round(
            (self.priority * ((0.7 * unresolved) + (0.3 * max(0.0, 1.0 - self.coverage_score))))
            + (0.25 * self.stagnation_score),
            4,
        )

    @property
    def uncertainty_score(self) -> float:
        resolution_gap = max(0.0, 1.0 - self.resolution_confidence)
        coverage_gap = max(0.0, 1.0 - self.coverage_score)
        retry_pressure = max(
            self.stagnation_score,
            min(max(self.attempt_count - self.accepted_count, 0) / 4.0, 1.0),
        )
        return round(
            _clamp((0.45 * resolution_gap) + (0.35 * coverage_gap) + (0.20 * retry_pressure)),
            4,
        )

    @property
    def frontier_score(self) -> float:
        pressure = _clamp(self.outstanding_score / 1.25)
        return round(
            _clamp((0.60 * pressure) + (0.25 * self.uncertainty_score) + (0.15 * self.priority)),
            4,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "description": self.description,
            "source": self.source,
            "priority": self.priority,
            "file_paths": list(self.file_paths),
            "test_ids": list(self.test_ids),
            "resolution_confidence": self.resolution_confidence,
            "coverage_score": self.coverage_score,
            "stagnation_score": self.stagnation_score,
            "attempt_count": self.attempt_count,
            "accepted_count": self.accepted_count,
            "outstanding_score": self.outstanding_score,
            "uncertainty_score": self.uncertainty_score,
            "frontier_score": self.frontier_score,
            "evidence_ids": list(self.evidence_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskObligation":
        return cls(
            obligation_id=str(data.get("obligation_id") or ""),
            description=str(data.get("description") or ""),
            source=str(data.get("source") or ""),
            priority=float(data.get("priority") or 0.0),
            file_paths=_normalize_file_paths(list(data.get("file_paths") or [])),
            test_ids=_extract_test_identifiers(list(data.get("test_ids") or [])),
            resolution_confidence=float(data.get("resolution_confidence") or 0.0),
            coverage_score=float(data.get("coverage_score") or 0.0),
            stagnation_score=float(data.get("stagnation_score") or 0.0),
            attempt_count=int(data.get("attempt_count") or 0),
            accepted_count=int(data.get("accepted_count") or 0),
            evidence_ids=_normalize_tokens(list(data.get("evidence_ids") or [])),
        )


@dataclass
class TaskHypothesis:
    """Candidate explanation or patch direction shared across rollouts."""

    hypothesis_id: str
    description: str
    family: str
    file_paths: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    support_score: float = 0.0
    contradiction_score: float = 0.0
    support_mass: float = 0.0
    contradiction_mass: float = 0.0
    evidence_ids: list[str] = field(default_factory=list)
    source_rollout_ids: list[int] = field(default_factory=list)
    source_families: list[str] = field(default_factory=list)

    @property
    def belief_score(self) -> float:
        return round(_clamp(self.support_score - (0.5 * self.contradiction_score)), 4)

    @property
    def posterior_mean(self) -> float:
        alpha = 1.0 + max(self.support_mass, 0.0)
        beta = 1.0 + max(self.contradiction_mass, 0.0)
        return round(alpha / (alpha + beta), 4)

    @property
    def uncertainty_score(self) -> float:
        alpha = 1.0 + max(self.support_mass, 0.0)
        beta = 1.0 + max(self.contradiction_mass, 0.0)
        total = alpha + beta
        variance = (alpha * beta) / ((total * total) * (total + 1.0))
        return round(_clamp(variance * 12.0), 4)

    @property
    def conflict_score(self) -> float:
        total = max(self.support_mass + self.contradiction_mass, 0.0)
        if total <= 0.0:
            return 0.0
        return round(
            _clamp((2.0 * min(self.support_mass, self.contradiction_mass)) / total),
            4,
        )

    @property
    def independent_support_score(self) -> float:
        rollout_diversity = min(len(self.source_rollout_ids), 4) / 4.0
        family_diversity = min(len(self.source_families), 4) / 4.0
        return round(_clamp((0.55 * rollout_diversity) + (0.45 * family_diversity)), 4)

    @property
    def frontier_score(self) -> float:
        return round(
            _clamp(
                (0.46 * self.belief_score)
                + (0.24 * self.uncertainty_score)
                + (0.18 * self.conflict_score)
                + (0.12 * self.independent_support_score)
            ),
            4,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "description": self.description,
            "family": self.family,
            "file_paths": list(self.file_paths),
            "symbols": list(self.symbols),
            "support_score": self.support_score,
            "contradiction_score": self.contradiction_score,
            "support_mass": round(self.support_mass, 4),
            "contradiction_mass": round(self.contradiction_mass, 4),
            "belief_score": self.belief_score,
            "posterior_mean": self.posterior_mean,
            "uncertainty_score": self.uncertainty_score,
            "conflict_score": self.conflict_score,
            "independent_support_score": self.independent_support_score,
            "frontier_score": self.frontier_score,
            "evidence_ids": list(self.evidence_ids),
            "source_rollout_ids": list(self.source_rollout_ids),
            "source_families": list(self.source_families),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskHypothesis":
        source_rollout_ids = data.get("source_rollout_ids") or []
        source_families = data.get("source_families") or []
        return cls(
            hypothesis_id=str(data.get("hypothesis_id") or ""),
            description=str(data.get("description") or ""),
            family=str(data.get("family") or ""),
            file_paths=_normalize_file_paths(list(data.get("file_paths") or [])),
            symbols=_normalize_tokens(list(data.get("symbols") or [])),
            support_score=float(data.get("support_score") or 0.0),
            contradiction_score=float(data.get("contradiction_score") or 0.0),
            support_mass=float(data.get("support_mass") or 0.0),
            contradiction_mass=float(data.get("contradiction_mass") or 0.0),
            evidence_ids=_normalize_tokens(list(data.get("evidence_ids") or [])),
            source_rollout_ids=[
                int(value) for value in source_rollout_ids if isinstance(value, int)
            ],
            source_families=[str(value).strip() for value in source_families if str(value).strip()],
        )


@dataclass
class TaskFrontierTarget:
    """Graph-ranked target used to steer the next rollout wave."""

    target_id: str
    kind: str
    description: str
    frontier_score: float
    uncertainty_score: float
    rationale: str
    obligation_id: Optional[str] = None
    hypothesis_id: Optional[str] = None
    obligation_description: str = ""
    hypothesis_description: str = ""
    family: str = ""
    file_paths: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "kind": self.kind,
            "description": self.description,
            "frontier_score": round(self.frontier_score, 4),
            "uncertainty_score": round(self.uncertainty_score, 4),
            "rationale": self.rationale,
            "obligation_id": self.obligation_id,
            "hypothesis_id": self.hypothesis_id,
            "obligation_description": self.obligation_description,
            "hypothesis_description": self.hypothesis_description,
            "family": self.family,
            "file_paths": list(self.file_paths),
            "test_ids": list(self.test_ids),
            "symbols": list(self.symbols),
        }


_MAX_EVIDENCE_RECORDS = 256
"""Soft cap on graph evidence size.

The graph never used to bound evidence count. On long search runs (e.g.
50-repo benchmark with multi-wave allocation) this grew until
``contradiction_pressure`` calls became O(n) per access. We cap with FIFO
eviction of the lowest-confidence non-accepted records, and any pruned
ID is silently tolerated by downstream code (``hypothesis.evidence_ids``
already filters missing IDs at access time).
"""


@dataclass
class TaskStateGraph:
    """Global state shared across planning, search, and selection."""

    obligations: dict[str, TaskObligation] = field(default_factory=dict)
    hypotheses: dict[str, TaskHypothesis] = field(default_factory=dict)
    evidence: dict[str, TaskEvidence] = field(default_factory=dict)
    file_attention: dict[str, float] = field(default_factory=dict)
    rollout_history: list[int] = field(default_factory=list)
    task_regime: TaskRegimeProfile = field(default_factory=TaskRegimeProfile)
    version: str = "v5"

    @classmethod
    def from_issue_plan(cls, issue_plan: "IssuePlan") -> "TaskStateGraph":
        graph = cls(
            task_regime=TaskRegimeProfile.from_dict(
                getattr(issue_plan.task_regime, "to_dict", lambda: {})()
                if getattr(issue_plan, "task_regime", None) is not None
                else {}
            )
        )
        test_context = issue_plan.test_context

        for criterion in issue_plan.success_criteria:
            graph._upsert_obligation(
                description=criterion,
                source="success_criteria",
                priority=0.75,
                file_paths=list(issue_plan.risk_files) + list(issue_plan.relevant_files[:2]),
            )

        for test_id in test_context.failing_test_ids:
            graph._upsert_obligation(
                description=f"Resolve visible failing test: {test_id}",
                source="failing_test",
                priority=1.0,
                file_paths=list(test_context.terminal_source_files)
                + list(test_context.incomplete_source_files)
                + list(test_context.source_focus_files)
                + list(test_context.focus_test_files)
                + [test_id],
                test_ids=[test_id],
            )

        for expectation in test_context.expectations:
            graph._upsert_obligation(
                description=f"Preserve visible-test expectation: {expectation}",
                source="visible_expectation",
                priority=0.85,
                file_paths=list(test_context.terminal_source_files)
                + list(test_context.incomplete_source_files)
                + list(test_context.source_focus_files)
                + list(test_context.focus_test_files)
                + [expectation],
                test_ids=[expectation],
            )

        if bool(
            getattr(issue_plan.evaluation_constraints, "preserve_collected_test_coverage", False)
        ):
            graph._upsert_obligation(
                description="Do not shrink collected test coverage under the main test command.",
                source="evaluation_constraint",
                priority=0.88,
                file_paths=list(issue_plan.risk_files) + list(test_context.terminal_source_files),
            )

        graph._ingest_issue_plan_task_regime()

        for brief in issue_plan.rollout_briefs:
            hypotheses = list(brief.hypotheses)
            if brief.prompt_hint:
                hypotheses.append(brief.prompt_hint)
            for hypothesis in hypotheses[:4]:
                graph._upsert_hypothesis(
                    description=hypothesis,
                    family=brief.title or brief.goal,
                    file_paths=brief.focus_files,
                    support_delta=0.15,
                )

        for path in list(issue_plan.risk_files) + list(issue_plan.relevant_files):
            graph.file_attention[path] = graph.file_attention.get(path, 0.0) + 0.15
        for path in test_context.terminal_source_files:
            graph.file_attention[path] = graph.file_attention.get(path, 0.0) + 0.35
        for path in test_context.incomplete_source_files:
            graph.file_attention[path] = graph.file_attention.get(path, 0.0) + 0.3
        for path in test_context.source_focus_files:
            graph.file_attention[path] = graph.file_attention.get(path, 0.0) + 0.25

        return graph

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskStateGraph":
        obligations = {
            obligation.obligation_id: obligation
            for obligation in (
                TaskObligation.from_dict(item)
                for item in list(data.get("obligations") or [])
                if isinstance(item, dict)
            )
            if obligation.obligation_id
        }
        hypotheses = {
            hypothesis.hypothesis_id: hypothesis
            for hypothesis in (
                TaskHypothesis.from_dict(item)
                for item in list(data.get("hypotheses") or [])
                if isinstance(item, dict)
            )
            if hypothesis.hypothesis_id
        }
        evidence = {
            item.evidence_id: item
            for item in (
                TaskEvidence.from_dict(payload)
                for payload in list(data.get("evidence") or [])
                if isinstance(payload, dict)
            )
            if item.evidence_id
        }
        file_attention = {
            str(key): float(value)
            for key, value in dict(data.get("file_attention") or {}).items()
            if key
        }
        rollout_history = [
            int(value)
            for value in list(data.get("rollout_history") or [])
            if isinstance(value, int)
        ]
        version = str(data.get("version") or "v5")
        return cls(
            obligations=obligations,
            hypotheses=hypotheses,
            evidence=evidence,
            file_attention=file_attention,
            rollout_history=rollout_history,
            task_regime=TaskRegimeProfile.from_dict(data.get("task_regime") or {}),
            version=version,
        )

    def clone(self) -> "TaskStateGraph":
        return TaskStateGraph.from_dict(self.to_dict())

    def ingest_rollout_results(self, issue_plan: "IssuePlan", rollout_results: list[Any]) -> None:
        for rollout_result in rollout_results:
            self.ingest_rollout_result(issue_plan, rollout_result)

    def ingest_rollout_result(self, issue_plan: "IssuePlan", rollout_result: Any) -> None:
        rollout_id = getattr(rollout_result, "rollout_id", None)
        if isinstance(rollout_id, int) and rollout_id not in self.rollout_history:
            self.rollout_history.append(rollout_id)

        changed_files = _normalize_file_paths(
            list(getattr(rollout_result, "changed_files", []) or [])
            + _artifact_values(getattr(rollout_result, "patch_artifact", None), "changed_files")
            + _artifact_values(getattr(rollout_result, "localization_artifact", None), "files")
        )
        verification = getattr(rollout_result, "verification", None)
        if not isinstance(verification, dict):
            verification = {}
        quick_verification = getattr(rollout_result, "quick_verification", None)
        if not isinstance(quick_verification, dict):
            quick_verification = {}
        candidate_test_ids = _extract_test_identifiers(
            _artifact_values(getattr(rollout_result, "patch_artifact", None), "tests_run")
            + list(getattr(rollout_result, "test_descriptions", []) or [])
            + list(quick_verification.get("failed_tests") or [])
        )
        candidate_symbols = _normalize_tokens(
            _artifact_values(getattr(rollout_result, "localization_artifact", None), "symbols")
        )
        accepted = rollout_has_authoritative_acceptance(rollout_result)
        # An explicit verifier rejection (verification.accepted is False) must not
        # be recorded as an ``accepted_patch`` in the task-state graph even when
        # the local full-suite quick verification passed. A full-suite pass is a
        # search/preemption signal, not final acceptance, once the verifier has
        # explicitly rejected the candidate (syntax/lint/prune/coverage/score
        # gates). The progress signal below still credits the full-suite run, so
        # the candidate remains a strong ``patch_attempt`` for planning.
        if isinstance(verification, dict) and verification.get("accepted") is False:
            accepted = False
        source_model = str(getattr(rollout_result, "llm_model", "") or "").strip()
        source_family = _model_source_family(source_model)
        verification_score = (
            _clamp(float(verification.get("overall_score")))
            if isinstance(verification.get("overall_score"), (int, float))
            else (
                _clamp(float(quick_verification_score))
                if isinstance(
                    (
                        quick_verification_score := quick_verification_signal_score(
                            quick_verification
                        )
                    ),
                    (int, float),
                )
                else (1.0 if accepted else 0.0)
            )
        )
        verification_trace_signal = self._verification_trace_signal(
            issue_plan,
            rollout_result,
            include_quick_verification=True,
        )
        candidate_test_ids = _extract_test_identifiers(
            list(candidate_test_ids) + list(verification_trace_signal.test_ids)
        )
        progress_score = (
            _clamp(float(getattr(rollout_result, "progress_score", 0.0)))
            if isinstance(getattr(rollout_result, "progress_score", None), (int, float))
            else 0.0
        )
        signal = max(progress_score, verification_score, 1.0 if accepted else 0.0)
        independently_verified = bool(accepted or verification_score >= 0.85)

        evidence_ids: list[str] = []
        localization_hypotheses = _artifact_values(
            getattr(rollout_result, "localization_artifact", None),
            "hypotheses",
        )
        localization_summary = _artifact_text(
            getattr(rollout_result, "localization_artifact", None),
            "summary",
        )
        if localization_summary or changed_files or candidate_symbols:
            evidence_ids.append(
                self._record_evidence(
                    kind="localization",
                    source="rollout_localization",
                    rollout_id=rollout_id,
                    summary=localization_summary or "Localization narrowed likely files.",
                    confidence=max(0.25, signal),
                    source_model=source_model,
                    source_family=source_family,
                    accepted=accepted,
                    independently_verified=independently_verified,
                    file_paths=changed_files,
                    test_ids=candidate_test_ids,
                    symbols=candidate_symbols,
                )
            )

        patch_summary = (
            _artifact_text(getattr(rollout_result, "patch_artifact", None), "summary")
            or (getattr(rollout_result, "explanation", "") or "").strip()
        )
        if patch_summary or changed_files:
            evidence_ids.append(
                self._record_evidence(
                    kind="accepted_patch" if accepted else "patch_attempt",
                    source="rollout_patch",
                    rollout_id=rollout_id,
                    summary=patch_summary or "Patch attempt changed implementation files.",
                    confidence=max(0.2, signal),
                    source_model=source_model,
                    source_family=source_family,
                    accepted=accepted,
                    independently_verified=independently_verified,
                    negative=not accepted,
                    file_paths=changed_files,
                    test_ids=candidate_test_ids,
                    symbols=candidate_symbols,
                )
            )

        if verification:
            evidence_ids.append(
                self._record_evidence(
                    kind="verification",
                    source="rollout_verification",
                    rollout_id=rollout_id,
                    summary=self._verification_summary(verification, accepted),
                    confidence=max(0.15, verification_score),
                    source_model=source_model,
                    source_family=source_family,
                    accepted=accepted,
                    independently_verified=independently_verified,
                    negative=not accepted,
                    file_paths=changed_files,
                    test_ids=candidate_test_ids,
                    symbols=candidate_symbols,
                )
            )

        verification_evidence_ids, verification_hypothesis_ids = self._ingest_verification_feedback(
            issue_plan,
            rollout_result,
            accepted=accepted,
            verification_score=verification_score,
            verification_trace_signal=verification_trace_signal,
        )
        evidence_ids.extend(verification_evidence_ids)

        for path in changed_files:
            self.file_attention[path] = self.file_attention.get(path, 0.0) + (
                0.7 if accepted else max(0.15, 0.45 * signal)
            )

        supported_hypothesis_ids: list[str] = []
        hypothesis_texts = list(localization_hypotheses)
        if patch_summary:
            hypothesis_texts.append(patch_summary)
        for description in hypothesis_texts[:6]:
            hypothesis = self._upsert_hypothesis(
                description=description,
                family=(getattr(rollout_result, "plan_title", "") or "rollout").strip()
                or "rollout",
                file_paths=changed_files,
                symbols=candidate_symbols,
                source_rollout_id=rollout_id,
                source_family=source_family,
                support_delta=(0.2 + (0.6 * signal)) if accepted else max(0.1, 0.35 * signal),
                contradiction_delta=0.0 if accepted or signal >= 0.3 else 0.15,
            )
            hypothesis.evidence_ids = list(dict.fromkeys(hypothesis.evidence_ids + evidence_ids))
            supported_hypothesis_ids.append(hypothesis.hypothesis_id)
        supported_hypothesis_ids.extend(verification_hypothesis_ids)
        boundary_evidence_ids, boundary_hypothesis_ids = self._ingest_boundary_pressure_feedback(
            issue_plan,
            rollout_result,
            changed_files=changed_files,
            verification_trace_signal=verification_trace_signal,
            signal=signal,
            accepted=accepted,
            source_model=source_model,
            source_family=source_family,
        )
        evidence_ids.extend(boundary_evidence_ids)
        supported_hypothesis_ids.extend(boundary_hypothesis_ids)

        for obligation in self.obligations.values():
            file_overlap = set(obligation.file_paths).intersection(changed_files)
            test_overlap = set(obligation.test_ids).intersection(candidate_test_ids)
            overlap_score = 0.0
            if test_overlap:
                overlap_score += 0.6
            if file_overlap:
                overlap_score += 0.35
            if not obligation.file_paths and not obligation.test_ids and changed_files:
                overlap_score += 0.15
            if overlap_score <= 0.0:
                continue

            obligation.attempt_count += 1
            # Use a non-additive monotone update so that one rollout with a
            # large overlap_score cannot saturate coverage to 1.0 (and mask
            # the obligation as fully covered) when only part of it was
            # actually exercised. The previous additive form would reach
            # ``coverage_score=1.0`` after a single ingestion whenever
            # ``overlap_score · signal ≥ 1 − coverage_score``.
            attempt_coverage = overlap_score * max(signal, 0.15)
            if accepted:
                obligation.coverage_score = _clamp(max(obligation.coverage_score, attempt_coverage))
            else:
                # Failed attempt with overlap: the obligation was *probed*
                # but the patch did not pass. Apply a soft decay to the
                # prior coverage so the search controller can revisit
                # this obligation instead of being permanently locked
                # out by an early "looks-like-it-will-fix-it" rollout.
                # Without this decay an early high-overlap-but-wrong
                # rollout pins coverage near 1.0 and outstanding_score
                # near 0, deprioritising the actual locus of the bug.
                stagnation_factor = 1.0 - min(
                    0.10,
                    0.04 * max(0, obligation.attempt_count - obligation.accepted_count),
                )
                decayed_prior = obligation.coverage_score * stagnation_factor
                obligation.coverage_score = _clamp(max(decayed_prior, attempt_coverage))
            if accepted:
                obligation.accepted_count += 1
                resolution = 0.5 + (0.4 * signal) + (0.1 if test_overlap else 0.0)
                obligation.resolution_confidence = _clamp(
                    max(obligation.resolution_confidence, resolution)
                )
            elif verification and test_overlap:
                obligation.stagnation_score = _clamp(
                    obligation.stagnation_score + (0.2 + (0.2 * (1.0 - signal)))
                )
            elif signal >= 0.2:
                obligation.resolution_confidence = _clamp(
                    max(obligation.resolution_confidence, 0.12 * signal)
                )
            obligation.evidence_ids = list(dict.fromkeys(obligation.evidence_ids + evidence_ids))[
                -32:
            ]

        self._link_evidence_relations(
            evidence_ids,
            supported_hypothesis_ids,
        )

    def _ingest_boundary_pressure_feedback(
        self,
        issue_plan: "IssuePlan",
        rollout_result: Any,
        *,
        changed_files: list[str],
        verification_trace_signal: _TracebackSignal,
        signal: float,
        accepted: bool,
        source_model: str,
        source_family: str,
    ) -> tuple[list[str], list[str]]:
        payload = (
            getattr(rollout_result, "multi_agent_summary", None)
            if isinstance(getattr(rollout_result, "multi_agent_summary", None), dict)
            else {}
        )
        boundary_pressure_count = int(payload.get("boundary_pressure_count") or 0)
        boundary_requested_files = _normalize_file_paths(
            [
                str(path).strip()
                for path in list(payload.get("boundary_requested_files") or [])
                if str(path).strip()
            ]
        )
        boundary_interface_symbols = _normalize_tokens(
            [
                str(symbol).strip()
                for symbol in list(payload.get("boundary_interface_symbols") or [])
                if str(symbol).strip()
            ]
        )
        boundary_followups = _dedupe_preserve(
            [
                str(item).strip()
                for item in list(payload.get("boundary_followups") or [])
                if str(item).strip()
            ]
        )
        if boundary_pressure_count <= 0 and not boundary_requested_files and not boundary_followups:
            return [], []

        known_paths = set(_candidate_repo_paths(issue_plan, rollout_result))
        followup_paths = _extract_python_file_paths("\n".join(boundary_followups), known_paths)
        normalized_requested_files = [
            path
            for path in (
                _normalize_known_repo_python_path(
                    candidate,
                    known_paths.union(followup_paths),
                )
                for candidate in boundary_requested_files
            )
            if path
        ]
        boundary_paths = _normalize_file_paths(normalized_requested_files + followup_paths)
        boundary_test_ids = _extract_test_identifiers(boundary_followups)
        if not boundary_paths and not boundary_test_ids and not boundary_interface_symbols:
            return [], []

        summary = (
            boundary_followups[0]
            if boundary_followups
            else "Delegated workers surfaced adjacent bridge files that the parent integrator should own."
        ).strip()
        if len(summary) > 280:
            summary = summary[:277].rstrip() + "..."
        confidence = _clamp(
            max(
                0.25,
                min(0.82, 0.16 * max(boundary_pressure_count, 1) + (0.45 * signal)),
            )
        )
        rollout_id = getattr(rollout_result, "rollout_id", None)
        evidence_id = self._record_evidence(
            kind="boundary_pressure",
            source="rollout_boundary_pressure",
            rollout_id=rollout_id if isinstance(rollout_id, int) else None,
            summary=summary,
            confidence=confidence,
            source_model=source_model,
            source_family=source_family,
            accepted=accepted,
            independently_verified=accepted,
            negative=not accepted,
            file_paths=boundary_paths,
            test_ids=boundary_test_ids,
            symbols=boundary_interface_symbols,
        )
        hypothesis = self._upsert_hypothesis(
            description=(
                "Delegated boundary pressure indicates the next pass must integrate adjacent bridge files."
            ),
            family="boundary_pressure",
            file_paths=boundary_paths,
            symbols=boundary_interface_symbols,
            source_rollout_id=rollout_id if isinstance(rollout_id, int) else None,
            source_family=source_family,
            support_delta=max(0.14, 0.24 * confidence),
        )
        hypothesis.evidence_ids = list(dict.fromkeys(hypothesis.evidence_ids + [evidence_id]))

        attention_boost = 0.18 + (0.22 * confidence) + (0.03 * min(boundary_pressure_count, 4))
        for path in boundary_paths:
            self.file_attention[path] = self.file_attention.get(path, 0.0) + attention_boost

        boundary_anchor_paths = set(changed_files)
        boundary_anchor_paths.update(verification_trace_signal.source_files)
        boundary_anchor_paths.update(verification_trace_signal.terminal_source_files)
        boundary_anchor_paths.update(
            _artifact_values(getattr(rollout_result, "localization_artifact", None), "files")
        )
        if not boundary_anchor_paths:
            boundary_anchor_paths.update(_candidate_repo_paths(issue_plan, rollout_result))

        matched_obligation = False
        for obligation in self.obligations.values():
            test_overlap = set(obligation.test_ids).intersection(boundary_test_ids)
            file_overlap = set(obligation.file_paths).intersection(boundary_anchor_paths)
            if not test_overlap and not file_overlap:
                continue
            matched_obligation = True
            if boundary_paths:
                obligation.file_paths = list(dict.fromkeys(boundary_paths + obligation.file_paths))
            if boundary_test_ids:
                obligation.test_ids = list(dict.fromkeys(boundary_test_ids + obligation.test_ids))
            obligation.evidence_ids = list(dict.fromkeys(obligation.evidence_ids + [evidence_id]))[
                -32:
            ]

        if not matched_obligation:
            obligation = self._upsert_obligation(
                description="Integrate adjacent bridge files surfaced by delegated workers.",
                source="boundary_pressure",
                priority=min(0.92, 0.72 + (0.04 * min(boundary_pressure_count, 4))),
                file_paths=boundary_paths,
                test_ids=boundary_test_ids[:4],
            )
            obligation.evidence_ids = list(dict.fromkeys(obligation.evidence_ids + [evidence_id]))[
                -32:
            ]

        return [evidence_id], [hypothesis.hypothesis_id]

    def ingest_verification_feedback(self, issue_plan: "IssuePlan", rollout_result: Any) -> None:
        verification = getattr(rollout_result, "verification", None)
        if not isinstance(verification, dict):
            verification = {}
        quick_verification = getattr(rollout_result, "quick_verification", None)
        if not isinstance(quick_verification, dict):
            quick_verification = {}
        accepted = rollout_has_authoritative_acceptance(rollout_result)
        verification_score = (
            _clamp(float(verification.get("overall_score")))
            if isinstance(verification.get("overall_score"), (int, float))
            else (
                _clamp(float(quick_verification_score))
                if isinstance(
                    (
                        quick_verification_score := quick_verification_signal_score(
                            quick_verification
                        )
                    ),
                    (int, float),
                )
                else (1.0 if accepted else 0.0)
            )
        )
        evidence_ids, hypothesis_ids = self._ingest_verification_feedback(
            issue_plan,
            rollout_result,
            accepted=accepted,
            verification_score=verification_score,
            verification_trace_signal=self._verification_trace_signal(
                issue_plan,
                rollout_result,
                include_quick_verification=False,
            ),
        )
        self._link_evidence_relations(evidence_ids, hypothesis_ids)

    def rank_open_obligations(self, *, max_items: int = 4) -> list[TaskObligation]:
        max_items = max(1, int(max_items))
        return [
            obligation
            for obligation in sorted(
                self.obligations.values(),
                key=lambda item: (
                    item.frontier_score,
                    item.outstanding_score,
                    item.priority,
                    -item.resolution_confidence,
                ),
                reverse=True,
            )
            if obligation.outstanding_score > 0.12
        ][:max_items]

    def rank_supported_hypotheses(self, *, max_items: int = 4) -> list[TaskHypothesis]:
        max_items = max(1, int(max_items))
        return [
            hypothesis
            for hypothesis in sorted(
                self.hypotheses.values(),
                key=lambda item: (
                    item.frontier_score,
                    item.independent_support_score,
                    item.belief_score,
                    item.support_score,
                    -item.contradiction_score,
                ),
                reverse=True,
            )
            if hypothesis.belief_score > 0.12 or hypothesis.uncertainty_score > 0.45
        ][:max_items]

    def build_frontier_targets(self, *, max_targets: int = 4) -> list[TaskFrontierTarget]:
        max_targets = max(1, int(max_targets))
        open_obligations = self.rank_open_obligations(max_items=max_targets * 2)
        supported_hypotheses = self.rank_supported_hypotheses(max_items=max_targets * 2)
        targets: list[TaskFrontierTarget] = []
        used_hypothesis_ids: set[str] = set()

        for obligation in open_obligations:
            target = self._build_joint_target(
                obligation,
                hypotheses=supported_hypotheses,
                used_hypothesis_ids=used_hypothesis_ids,
            )
            if target is None:
                target = self._frontier_target_from_obligation(obligation)
            elif target.hypothesis_id:
                used_hypothesis_ids.add(target.hypothesis_id)
            targets.append(target)
            if len(targets) >= max_targets:
                return targets[:max_targets]

        for hypothesis in supported_hypotheses:
            if hypothesis.hypothesis_id in used_hypothesis_ids:
                continue
            targets.append(self._frontier_target_from_hypothesis(hypothesis))
            if len(targets) >= max_targets:
                break
        return self._rank_frontier_targets_with_regime(targets)[:max_targets]

    def build_issue_plan_context(
        self,
        *,
        max_items: int = 4,
    ) -> dict[str, Any]:
        max_items = max(1, int(max_items))
        open_obligations = self.rank_open_obligations(max_items=max_items)
        supported_hypotheses = self.rank_supported_hypotheses(max_items=max_items)
        frontier_targets = self.build_frontier_targets(max_targets=max_items)
        focus_files = self.rank_focus_files(max_files=max_items * 2)
        contradiction_pressure = self.contradiction_pressure()
        contested_files = self.contested_files(max_files=max_items * 2)
        unresolved_test_ids = list(
            dict.fromkeys(
                test_id for obligation in open_obligations for test_id in obligation.test_ids
            )
        )[:max_items]

        summary_parts: list[str] = []
        if frontier_targets:
            top_target = frontier_targets[0]
            if top_target.hypothesis_description:
                summary_parts.append(
                    "Current highest-value frontier target: "
                    f"{top_target.description} via {top_target.hypothesis_description}."
                )
            else:
                summary_parts.append(
                    f"Current highest-value frontier target: {top_target.description}."
                )
        if open_obligations:
            summary_parts.append(
                "Highest-pressure obligations: "
                + "; ".join(obligation.description for obligation in open_obligations[:2])
                + "."
            )
        if supported_hypotheses:
            summary_parts.append(
                "Most supported current hypotheses: "
                + "; ".join(hypothesis.description for hypothesis in supported_hypotheses[:2])
                + "."
            )
        if self.task_regime.summary:
            summary_parts.append(self.task_regime.summary)
        if contradiction_pressure >= 0.25 and contested_files:
            summary_parts.append(
                "Current rollout evidence is still contested around "
                + ", ".join(contested_files[:3])
                + "; prefer independently verified progress over simple agreement."
            )
        if focus_files:
            summary_parts.append(
                "Focus implementation search around " + ", ".join(focus_files[:4]) + "."
            )

        return {
            "summary": " ".join(summary_parts).strip(),
            "open_obligations": [
                {
                    "obligation_id": obligation.obligation_id,
                    "description": obligation.description,
                    "source": obligation.source,
                    "priority": round(obligation.priority, 4),
                    "outstanding_score": obligation.outstanding_score,
                    "uncertainty_score": obligation.uncertainty_score,
                    "frontier_score": obligation.frontier_score,
                    "resolution_confidence": round(obligation.resolution_confidence, 4),
                    "coverage_score": round(obligation.coverage_score, 4),
                    "attempt_count": obligation.attempt_count,
                    "accepted_count": obligation.accepted_count,
                    "file_paths": list(obligation.file_paths[:4]),
                    "test_ids": list(obligation.test_ids[:3]),
                }
                for obligation in open_obligations
            ],
            "supported_hypotheses": [
                {
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "description": hypothesis.description,
                    "family": hypothesis.family,
                    "belief_score": hypothesis.belief_score,
                    "support_score": round(hypothesis.support_score, 4),
                    "contradiction_score": round(hypothesis.contradiction_score, 4),
                    "posterior_mean": hypothesis.posterior_mean,
                    "uncertainty_score": hypothesis.uncertainty_score,
                    "conflict_score": hypothesis.conflict_score,
                    "independent_support_score": hypothesis.independent_support_score,
                    "frontier_score": hypothesis.frontier_score,
                    "file_paths": list(hypothesis.file_paths[:4]),
                    "symbols": list(hypothesis.symbols[:3]),
                    "source_rollout_ids": list(hypothesis.source_rollout_ids[:4]),
                    "source_families": list(hypothesis.source_families[:4]),
                }
                for hypothesis in supported_hypotheses
            ],
            "frontier_targets": [target.to_dict() for target in frontier_targets],
            "focus_files": list(focus_files),
            "contested_files": list(contested_files),
            "unresolved_test_ids": unresolved_test_ids,
            "evidence_count": len(self.evidence),
            "obligation_count": len(self.obligations),
            "hypothesis_count": len(self.hypotheses),
            "frontier_target_count": len(frontier_targets),
            "contradiction_pressure": contradiction_pressure,
            "task_regime_summary": str(self.task_regime.summary or "").strip(),
            "task_regime_probabilities": {
                state: round(float(probability), 4)
                for state, probability in sorted(self.task_regime.state_probabilities.items())
            },
            "task_regime_active_states": list(
                self.task_regime.active_states(minimum_probability=0.25)
            ),
            "task_regime_evidence": [
                item.to_dict()
                for item in list(self.task_regime.evidence or [])[: max(2, max_items)]
            ],
            "conflicted_hypothesis_count": sum(
                1 for hypothesis in self.hypotheses.values() if hypothesis.conflict_score >= 0.25
            ),
        }

    def _ingest_issue_plan_task_regime(self) -> None:
        profile = (
            self.task_regime
            if isinstance(self.task_regime, TaskRegimeProfile)
            else TaskRegimeProfile()
        )
        if not profile.state_probabilities:
            return

        regime_messages = {
            "importability_blocker": (
                "Clear the currently observed import or collection blocker before broadening the patch.",
                "The highest-leverage next step is likely in the traceback or import chain already surfaced by visible failures.",
                0.95,
            ),
            "contract_gap": (
                "Recover the missing behavior or contract without weakening visible tests.",
                "Visible tests and nearby scaffolds likely describe missing functionality rather than a narrow regression.",
                0.86,
            ),
            "broad_regression": (
                "Preserve nearby behavior and collected visible coverage while repairing the bug.",
                "The failure surface is broad enough that regression containment must shape the next rollout.",
                0.82,
            ),
            "high_interface_risk": (
                "Preserve shared interface behavior across touched modules unless visible evidence disproves it.",
                "The likely fix crosses shared symbols or module boundaries and needs interface-aware coordination.",
                0.80,
            ),
        }

        for state, probability in sorted(
            profile.state_probabilities.items(),
            key=lambda item: (float(item[1]), item[0]),
            reverse=True,
        ):
            probability = float(probability or 0.0)
            if probability < 0.25:
                continue
            obligation_text, hypothesis_text, priority = regime_messages.get(
                state,
                (
                    "Preserve the currently inferred controller invariant while advancing the patch.",
                    "The inferred controller regime should guide the next rollout family.",
                    0.75,
                ),
            )
            evidence_items = profile.evidence_for_state(state)
            file_paths = _normalize_file_paths(
                [path for item in evidence_items for path in list(item.file_paths or [])]
            )
            test_ids = _extract_test_identifiers(
                [test_id for item in evidence_items for test_id in list(item.test_ids or [])]
            )
            symbols = _normalize_tokens(
                [symbol for item in evidence_items for symbol in list(item.symbols or [])]
            )
            evidence_summary = (
                profile.summary
                if len(evidence_items) == 0
                else "; ".join(
                    item.rationale
                    for item in evidence_items[:2]
                    if str(item.rationale or "").strip()
                )
            ).strip()
            evidence_id = self._record_evidence(
                kind="task_regime",
                source=f"planner_regime:{state}",
                rollout_id=None,
                summary=evidence_summary or hypothesis_text,
                confidence=probability,
                file_paths=file_paths,
                test_ids=test_ids,
                symbols=symbols,
            )
            obligation = self._upsert_obligation(
                description=obligation_text,
                source=f"regime_{state}",
                priority=max(priority, 0.55 + (0.25 * probability)),
                file_paths=file_paths,
                test_ids=test_ids[:2],
            )
            obligation.evidence_ids = list(dict.fromkeys(obligation.evidence_ids + [evidence_id]))
            hypothesis = self._upsert_hypothesis(
                description=hypothesis_text,
                family=f"regime_{state}",
                file_paths=file_paths,
                symbols=symbols,
                support_delta=max(0.12, 0.35 * probability),
            )
            hypothesis.evidence_ids = list(dict.fromkeys(hypothesis.evidence_ids + [evidence_id]))
            for path in file_paths[:6]:
                self.file_attention[path] = self.file_attention.get(path, 0.0) + (
                    0.10 + (0.25 * probability)
                )

    def _task_regime_bonus(
        self,
        *,
        description: str,
        family: str = "",
        source: str = "",
        file_paths: Optional[list[str]] = None,
        test_ids: Optional[list[str]] = None,
        symbols: Optional[list[str]] = None,
    ) -> tuple[float, list[str]]:
        profile = (
            self.task_regime
            if isinstance(self.task_regime, TaskRegimeProfile)
            else TaskRegimeProfile()
        )
        if not profile.state_probabilities:
            return 0.0, []

        normalized_paths = set(_normalize_file_paths(list(file_paths or [])))
        normalized_tests = set(_extract_test_identifiers(list(test_ids or [])))
        normalized_symbols = set(_normalize_tokens(list(symbols or [])))
        description_text = f"{description} {family} {source}".lower()
        bonus = 0.0
        reasons: list[str] = []

        for state in profile.active_states(minimum_probability=0.25):
            state_probability = profile.probability(state)
            evidence_items = profile.evidence_for_state(state)
            evidence_paths = {
                path
                for item in evidence_items
                for path in _normalize_file_paths(list(item.file_paths or []))
            }
            evidence_tests = {
                test_id
                for item in evidence_items
                for test_id in _extract_test_identifiers(list(item.test_ids or []))
            }
            evidence_symbols = {
                symbol
                for item in evidence_items
                for symbol in _normalize_tokens(list(item.symbols or []))
            }

            state_bonus = 0.0
            if state in description_text:
                state_bonus += 0.08 * state_probability
            if (
                normalized_paths
                and evidence_paths
                and normalized_paths.intersection(evidence_paths)
            ):
                state_bonus += 0.08 * state_probability
            if (
                normalized_tests
                and evidence_tests
                and normalized_tests.intersection(evidence_tests)
            ):
                state_bonus += 0.06 * state_probability
            if (
                normalized_symbols
                and evidence_symbols
                and normalized_symbols.intersection(evidence_symbols)
            ):
                state_bonus += 0.05 * state_probability
            if state_bonus > 0.0:
                bonus += state_bonus
                reasons.append(f"{state}={state_probability:.2f}")

        return min(bonus, 0.22), reasons[:3]

    def _rank_frontier_targets_with_regime(
        self,
        targets: list[TaskFrontierTarget],
    ) -> list[TaskFrontierTarget]:
        ranked: list[TaskFrontierTarget] = []
        for target in list(targets or []):
            bonus, reasons = self._task_regime_bonus(
                description=target.description,
                family=target.family,
                source=(
                    str(self.obligations.get(target.obligation_id).source or "")
                    if target.obligation_id and target.obligation_id in self.obligations
                    else (
                        str(self.hypotheses.get(target.hypothesis_id).family or "")
                        if target.hypothesis_id and target.hypothesis_id in self.hypotheses
                        else ""
                    )
                ),
                file_paths=target.file_paths,
                test_ids=target.test_ids,
                symbols=target.symbols,
            )
            if bonus > 0.0:
                target.frontier_score = round(_clamp(target.frontier_score + bonus), 4)
                target.rationale = (
                    f"{target.rationale} Regime support: {', '.join(reasons)}."
                ).strip()
            ranked.append(target)
        ranked.sort(
            key=lambda item: (item.frontier_score, item.uncertainty_score, item.target_id),
            reverse=True,
        )
        return ranked

    def _verification_trace_signal(
        self,
        issue_plan: "IssuePlan",
        rollout_result: Any,
        *,
        include_quick_verification: bool,
    ) -> _TracebackSignal:
        known_paths = set(_candidate_repo_paths(issue_plan, rollout_result))
        outputs: list[str] = []

        verification = getattr(rollout_result, "verification", None)
        if isinstance(verification, dict):
            test_result = verification.get("test_result")
            if isinstance(test_result, dict):
                for field_name in ("regression_output", "reproduction_output"):
                    output = test_result.get(field_name)
                    if isinstance(output, str) and output.strip():
                        outputs.append(output)

        quick_verification = getattr(rollout_result, "quick_verification", None)
        if include_quick_verification and isinstance(quick_verification, dict):
            output_excerpt = quick_verification.get("output_excerpt")
            if isinstance(output_excerpt, str) and output_excerpt.strip():
                outputs.append(output_excerpt)

        source_files: list[str] = []
        test_ids: list[str] = []
        terminal_source_files: list[str] = []
        exception_summaries: list[str] = []
        for output in outputs:
            signal = _extract_traceback_signal(output, known_paths)
            source_files.extend(signal.source_files)
            test_ids.extend(signal.test_ids)
            terminal_source_files.extend(signal.terminal_source_files)
            exception_summaries.extend(signal.exception_summaries)

        if include_quick_verification and isinstance(quick_verification, dict):
            test_ids.extend(
                str(item) for item in list(quick_verification.get("failed_tests") or []) if item
            )

        return _TracebackSignal(
            source_files=_dedupe_preserve(source_files),
            test_ids=_extract_test_identifiers(test_ids),
            terminal_source_files=_dedupe_preserve(terminal_source_files),
            exception_summaries=_dedupe_preserve(exception_summaries)[:4],
        )

    def _ingest_verification_feedback(
        self,
        issue_plan: "IssuePlan",
        rollout_result: Any,
        *,
        accepted: bool,
        verification_score: float,
        verification_trace_signal: _TracebackSignal,
    ) -> tuple[list[str], list[str]]:
        rollout_id = getattr(rollout_result, "rollout_id", None)
        file_paths = _dedupe_preserve(
            list(verification_trace_signal.terminal_source_files)
            + list(verification_trace_signal.source_files)
        )
        test_ids = _extract_test_identifiers(list(verification_trace_signal.test_ids))
        if not file_paths and not test_ids and not verification_trace_signal.exception_summaries:
            return [], []

        signal_strength = 1.0 - _clamp(verification_score)
        confidence = max(0.2, 0.25 + (0.45 * signal_strength))
        summary_parts: list[str] = []
        if verification_trace_signal.exception_summaries:
            summary_parts.append(
                f"Residual verification exception: {verification_trace_signal.exception_summaries[0]}."
            )
        if verification_trace_signal.terminal_source_files:
            summary_parts.append(
                f"Residual failure now terminates in {verification_trace_signal.terminal_source_files[0]}."
            )
        elif file_paths:
            summary_parts.append(
                "Residual verification still points through " + ", ".join(file_paths[:3]) + "."
            )
        summary = (
            " ".join(summary_parts).strip()
            or "Residual verification produced new traceback signal."
        )

        prior_evidence_count = len(self.evidence)
        evidence_id = self._record_evidence(
            kind="verification_traceback",
            source="rollout_verification_trace",
            rollout_id=rollout_id,
            summary=summary,
            confidence=confidence,
            source_model=str(getattr(rollout_result, "llm_model", "") or "").strip(),
            source_family=_model_source_family(str(getattr(rollout_result, "llm_model", "") or "")),
            accepted=accepted,
            independently_verified=False,
            negative=True,
            file_paths=file_paths,
            test_ids=test_ids,
        )
        if len(self.evidence) == prior_evidence_count:
            return [], []

        terminal_paths = set(verification_trace_signal.terminal_source_files)
        for path in file_paths:
            if path in terminal_paths:
                self.file_attention[path] = self.file_attention.get(path, 0.0) + (
                    0.45 + (0.2 * signal_strength)
                )
            else:
                self.file_attention[path] = self.file_attention.get(path, 0.0) + (
                    0.2 + (0.15 * signal_strength)
                )

        evidence_ids = [evidence_id]
        for test_id in test_ids[:4]:
            obligation = self._upsert_obligation(
                description=f"Resolve visible failing test: {test_id}",
                source="failing_test",
                priority=1.0,
                file_paths=file_paths + [test_id],
                test_ids=[test_id],
            )
            obligation.evidence_ids = list(dict.fromkeys(obligation.evidence_ids + evidence_ids))

        if verification_trace_signal.terminal_source_files:
            obligation = self._upsert_obligation(
                description=(
                    "Resolve residual verification failure in "
                    f"{verification_trace_signal.terminal_source_files[0]}"
                ),
                source="verification_traceback",
                priority=0.9 if not accepted else 0.7,
                file_paths=file_paths,
                test_ids=test_ids[:2],
            )
            obligation.evidence_ids = list(dict.fromkeys(obligation.evidence_ids + evidence_ids))

        family = (getattr(rollout_result, "plan_title", "") or "rollout").strip() or "rollout"
        support_delta = max(0.18, 0.2 + (0.25 * signal_strength))
        hypothesis_ids: list[str] = []
        hypothesis_texts: list[str] = []
        if verification_trace_signal.exception_summaries:
            hypothesis_texts.append(
                f"Residual verification exception: {verification_trace_signal.exception_summaries[0]}."
            )
        if verification_trace_signal.terminal_source_files:
            hypothesis_texts.append(
                f"Residual failure now terminates in {verification_trace_signal.terminal_source_files[0]}."
            )
        if len(file_paths) >= 2:
            hypothesis_texts.append(
                "Residual verification traceback converges through "
                + ", ".join(file_paths[:3])
                + "."
            )

        for description in hypothesis_texts[:3]:
            hypothesis = self._upsert_hypothesis(
                description=description,
                family=f"{family} residual",
                file_paths=file_paths,
                source_rollout_id=rollout_id if isinstance(rollout_id, int) else None,
                source_family=_model_source_family(
                    str(getattr(rollout_result, "llm_model", "") or "")
                ),
                support_delta=support_delta,
            )
            hypothesis.evidence_ids = list(dict.fromkeys(hypothesis.evidence_ids + evidence_ids))
            hypothesis_ids.append(hypothesis.hypothesis_id)

        self._mark_conflicting_hypotheses(
            file_paths=file_paths,
            exclude_hypothesis_ids=hypothesis_ids,
            contradiction_delta=max(0.08, 0.12 + (0.18 * signal_strength)),
        )
        return evidence_ids, hypothesis_ids

    def _link_evidence_relations(
        self,
        evidence_ids: list[str],
        hypothesis_ids: list[str],
    ) -> None:
        for evidence_id in list(dict.fromkeys(evidence_ids)):
            if evidence_id not in self.evidence:
                continue
            evidence = self.evidence[evidence_id]
            evidence.related_obligation_ids = [
                obligation.obligation_id
                for obligation in self.obligations.values()
                if set(obligation.evidence_ids).intersection({evidence_id})
            ]
            evidence.related_hypothesis_ids = list(dict.fromkeys(hypothesis_ids))

    def rank_focus_files(self, *, max_files: int = 8) -> list[str]:
        scores = dict(self.file_attention)
        for obligation in self.obligations.values():
            for path in obligation.file_paths:
                scores[path] = scores.get(path, 0.0) + (1.4 * obligation.outstanding_score)
        for hypothesis in self.hypotheses.values():
            for path in hypothesis.file_paths:
                scores[path] = scores.get(path, 0.0) + (0.9 * hypothesis.belief_score)
                scores[path] = scores.get(path, 0.0) + (0.35 * hypothesis.conflict_score)
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [path for path, _ in ordered[:max_files] if path]

    def contradiction_pressure(self) -> float:
        if not self.hypotheses:
            return 0.0
        conflict_scores = sorted(
            (
                max(hypothesis.conflict_score, 0.8 * hypothesis.contradiction_score)
                for hypothesis in self.hypotheses.values()
            ),
            reverse=True,
        )
        top = conflict_scores[0]
        average_top = sum(conflict_scores[:3]) / min(len(conflict_scores), 3)
        return round(_clamp((0.65 * top) + (0.35 * average_top)), 4)

    def contested_files(self, *, max_files: int = 6) -> list[str]:
        scores: Counter[str] = Counter()
        for hypothesis in self.hypotheses.values():
            if hypothesis.conflict_score < 0.22 and hypothesis.contradiction_score < 0.22:
                continue
            pressure = (
                (1.15 * hypothesis.conflict_score)
                + (0.75 * hypothesis.contradiction_score)
                + (0.35 * max(0.0, 1.0 - hypothesis.independent_support_score))
            )
            for path in hypothesis.file_paths:
                scores[path] += pressure
        return [path for path, _ in scores.most_common(max_files) if path]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "rollout_history": list(self.rollout_history),
            "task_regime": self.task_regime.to_dict(),
            "file_attention": {
                key: round(value, 4) for key, value in sorted(self.file_attention.items())
            },
            "obligations": [item.to_dict() for item in self.obligations.values()],
            "hypotheses": [item.to_dict() for item in self.hypotheses.values()],
            "evidence": [item.to_dict() for item in self.evidence.values()],
            "frontier_targets": [item.to_dict() for item in self.build_frontier_targets()],
            "issue_plan_context": self.build_issue_plan_context(),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Optional["TaskStateGraph"]:
        """WS3D: warm-start load from disk. STRICT NO-OP CONTRACT — returns None
        (never raises) when the path is missing, unreadable, corrupt, or has an
        unsupported schema, so a cold first run / a scrubbed file simply gets a
        fresh graph rather than crashing the solve."""
        try:
            file_path = Path(path)
            if not file_path.exists():
                return None
            data = json.loads(file_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return None
            return cls.from_dict(data)
        except (OSError, ValueError, TypeError, KeyError):
            logger.debug("TaskStateGraph.load failed for %s", path, exc_info=True)
            return None

    def merge_warm_start(self, other: "TaskStateGraph") -> int:
        """WS3D: seed this (freshly issue-plan-derived) graph with durable signal
        from a prior-run graph, additively. Current obligations (bound to THIS
        run's issue plan) are preserved; prior evidence/hypotheses are added only
        when their id is not already present, and file_attention is merged by max.
        Returns the number of records grafted (for diagnostics/tests)."""
        grafted = 0
        for ev_id, ev in (getattr(other, "evidence", {}) or {}).items():
            if ev_id and ev_id not in self.evidence:
                self.evidence[ev_id] = ev
                grafted += 1
        for hyp_id, hyp in (getattr(other, "hypotheses", {}) or {}).items():
            if hyp_id and hyp_id not in self.hypotheses:
                self.hypotheses[hyp_id] = hyp
                grafted += 1
        for path_key, score in (getattr(other, "file_attention", {}) or {}).items():
            try:
                prior_score = float(score)
            except (TypeError, ValueError):
                continue
            self.file_attention[path_key] = max(
                float(self.file_attention.get(path_key, 0.0)), prior_score
            )
        return grafted

    def _upsert_obligation(
        self,
        *,
        description: str,
        source: str,
        priority: float,
        file_paths: Optional[list[str]] = None,
        test_ids: Optional[list[str]] = None,
    ) -> TaskObligation:
        normalized = _normalize_text(description)
        obligation_id = _stable_id("obl", f"{source}:{normalized}")
        obligation = self.obligations.get(obligation_id)
        normalized_paths = _normalize_file_paths(list(file_paths or []) + list(test_ids or []))
        normalized_test_ids = _extract_test_identifiers(list(test_ids or []))
        if obligation is None:
            obligation = TaskObligation(
                obligation_id=obligation_id,
                description=description.strip(),
                source=source,
                priority=_clamp(priority),
                file_paths=normalized_paths,
                test_ids=normalized_test_ids,
            )
            self.obligations[obligation_id] = obligation
            return obligation

        obligation.priority = max(obligation.priority, _clamp(priority))
        obligation.file_paths = list(dict.fromkeys(obligation.file_paths + normalized_paths))
        obligation.test_ids = list(dict.fromkeys(obligation.test_ids + normalized_test_ids))
        return obligation

    def _upsert_hypothesis(
        self,
        *,
        description: str,
        family: str,
        file_paths: Optional[list[str]] = None,
        symbols: Optional[list[str]] = None,
        source_rollout_id: Optional[int] = None,
        source_family: str = "",
        support_delta: float = 0.0,
        contradiction_delta: float = 0.0,
    ) -> TaskHypothesis:
        normalized = _normalize_text(description)
        hypothesis_id = _stable_id("hyp", normalized)
        hypothesis = self.hypotheses.get(hypothesis_id)
        normalized_paths = _normalize_file_paths(file_paths or [])
        normalized_symbols = _normalize_tokens(symbols or [])
        if hypothesis is None:
            hypothesis = TaskHypothesis(
                hypothesis_id=hypothesis_id,
                description=description.strip(),
                family=(family or "rollout").strip() or "rollout",
                file_paths=normalized_paths,
                symbols=normalized_symbols,
            )
            self.hypotheses[hypothesis_id] = hypothesis

        hypothesis.family = hypothesis.family or (family.strip() or "rollout")
        hypothesis.file_paths = list(dict.fromkeys(hypothesis.file_paths + normalized_paths))
        hypothesis.symbols = list(dict.fromkeys(hypothesis.symbols + normalized_symbols))
        hypothesis.support_mass += max(0.0, support_delta)
        hypothesis.contradiction_mass += max(0.0, contradiction_delta)
        hypothesis.support_score = _clamp(hypothesis.support_score + max(0.0, support_delta))
        hypothesis.contradiction_score = _clamp(
            hypothesis.contradiction_score + max(0.0, contradiction_delta)
        )
        if (
            isinstance(source_rollout_id, int)
            and source_rollout_id not in hypothesis.source_rollout_ids
        ):
            hypothesis.source_rollout_ids.append(source_rollout_id)
        normalized_source_family = (source_family or "").strip().lower()
        if normalized_source_family and normalized_source_family not in hypothesis.source_families:
            hypothesis.source_families.append(normalized_source_family)
        return hypothesis

    def _build_joint_target(
        self,
        obligation: TaskObligation,
        *,
        hypotheses: list[TaskHypothesis],
        used_hypothesis_ids: set[str],
    ) -> Optional[TaskFrontierTarget]:
        hypothesis = self._best_hypothesis_for_obligation(
            obligation,
            hypotheses=hypotheses,
            used_hypothesis_ids=used_hypothesis_ids,
        )
        if hypothesis is None:
            return None

        overlap = self._obligation_hypothesis_overlap(obligation, hypothesis)
        score = _clamp(
            (0.60 * obligation.frontier_score)
            + (0.30 * hypothesis.frontier_score)
            + (0.10 * overlap)
        )
        if score < 0.28:
            return None
        return TaskFrontierTarget(
            target_id=_stable_id(
                "frontier", f"{obligation.obligation_id}:{hypothesis.hypothesis_id}"
            ),
            kind="joint",
            description=obligation.description,
            frontier_score=score,
            uncertainty_score=round(
                (obligation.uncertainty_score + hypothesis.uncertainty_score) / 2.0,
                4,
            ),
            rationale=(
                "Highest unresolved behavioral obligation aligned with the strongest "
                "compatible current hypothesis."
            ),
            obligation_id=obligation.obligation_id,
            hypothesis_id=hypothesis.hypothesis_id,
            obligation_description=obligation.description,
            hypothesis_description=hypothesis.description,
            family=hypothesis.family,
            file_paths=list(dict.fromkeys(obligation.file_paths + hypothesis.file_paths))[:6],
            test_ids=list(obligation.test_ids[:4]),
            symbols=list(hypothesis.symbols[:4]),
        )

    def _frontier_target_from_obligation(
        self,
        obligation: TaskObligation,
    ) -> TaskFrontierTarget:
        return TaskFrontierTarget(
            target_id=_stable_id("frontier", obligation.obligation_id),
            kind="obligation",
            description=obligation.description,
            frontier_score=obligation.frontier_score,
            uncertainty_score=obligation.uncertainty_score,
            rationale="Highest unresolved behavioral obligation in the current task state.",
            obligation_id=obligation.obligation_id,
            obligation_description=obligation.description,
            file_paths=list(obligation.file_paths[:6]),
            test_ids=list(obligation.test_ids[:4]),
        )

    def _frontier_target_from_hypothesis(
        self,
        hypothesis: TaskHypothesis,
    ) -> TaskFrontierTarget:
        return TaskFrontierTarget(
            target_id=_stable_id("frontier", hypothesis.hypothesis_id),
            kind="hypothesis",
            description=hypothesis.description,
            frontier_score=hypothesis.frontier_score,
            uncertainty_score=hypothesis.uncertainty_score,
            rationale=(
                "Promising but not yet settled patch direction that still deserves direct exploration."
            ),
            hypothesis_id=hypothesis.hypothesis_id,
            hypothesis_description=hypothesis.description,
            family=hypothesis.family,
            file_paths=list(hypothesis.file_paths[:6]),
            symbols=list(hypothesis.symbols[:4]),
        )

    def _best_hypothesis_for_obligation(
        self,
        obligation: TaskObligation,
        *,
        hypotheses: list[TaskHypothesis],
        used_hypothesis_ids: set[str],
    ) -> Optional[TaskHypothesis]:
        ranked: list[tuple[float, TaskHypothesis]] = []
        for hypothesis in hypotheses:
            if hypothesis.hypothesis_id in used_hypothesis_ids:
                continue
            overlap = self._obligation_hypothesis_overlap(obligation, hypothesis)
            score = (0.55 * overlap) + (0.45 * hypothesis.frontier_score)
            if score <= 0.18:
                continue
            ranked.append((score, hypothesis))
        if not ranked:
            return None
        ranked.sort(
            key=lambda item: (item[0], item[1].frontier_score, item[1].belief_score), reverse=True
        )
        return ranked[0][1]

    def _obligation_hypothesis_overlap(
        self,
        obligation: TaskObligation,
        hypothesis: TaskHypothesis,
    ) -> float:
        obligation_files = set(obligation.file_paths)
        hypothesis_files = set(hypothesis.file_paths)
        if obligation_files and hypothesis_files:
            shared_files = len(obligation_files.intersection(hypothesis_files))
            total_files = len(obligation_files.union(hypothesis_files))
            file_overlap = shared_files / max(total_files, 1)
        else:
            file_overlap = 0.0

        obligation_text = _normalize_text(obligation.description)
        family_text = _normalize_text(hypothesis.family)
        hypothesis_text = _normalize_text(hypothesis.description)
        text_overlap = 0.0
        if obligation_text and (family_text or hypothesis_text):
            obligation_tokens = {token for token in obligation_text.split(" ") if len(token) >= 4}
            hypothesis_tokens = {
                token for token in f"{family_text} {hypothesis_text}".split(" ") if len(token) >= 4
            }
            if obligation_tokens and hypothesis_tokens:
                text_overlap = len(obligation_tokens.intersection(hypothesis_tokens)) / max(
                    len(obligation_tokens.union(hypothesis_tokens)),
                    1,
                )
        return round(_clamp((0.75 * file_overlap) + (0.25 * text_overlap)), 4)

    def _record_evidence(
        self,
        *,
        kind: str,
        source: str,
        rollout_id: Optional[int],
        summary: str,
        confidence: float,
        source_model: str = "",
        source_family: str = "",
        accepted: bool = False,
        independently_verified: bool = False,
        negative: bool = False,
        file_paths: Optional[list[str]] = None,
        test_ids: Optional[list[str]] = None,
        symbols: Optional[list[str]] = None,
    ) -> str:
        normalized_summary = summary.strip() or kind.replace("_", " ")
        evidence_id = _stable_id(
            "ev",
            f"{source}:{rollout_id}:{kind}:{normalized_summary}:{','.join(file_paths or [])}:{','.join(test_ids or [])}",
        )
        if evidence_id not in self.evidence:
            self.evidence[evidence_id] = TaskEvidence(
                evidence_id=evidence_id,
                kind=kind,
                source=source,
                rollout_id=rollout_id,
                summary=normalized_summary,
                confidence=_clamp(confidence),
                source_model=(source_model or "").strip(),
                source_family=(source_family or "").strip().lower(),
                accepted=bool(accepted),
                independently_verified=bool(independently_verified),
                negative=bool(negative),
                file_paths=_normalize_file_paths(file_paths or []),
                test_ids=_extract_test_identifiers(test_ids or []),
                symbols=_normalize_tokens(symbols or []),
            )
            if len(self.evidence) > _MAX_EVIDENCE_RECORDS:
                self._prune_evidence_to_cap()
        return evidence_id

    def _prune_evidence_to_cap(self) -> int:
        """Drop low-value evidence so the graph stays bounded.

        Eviction policy:
          1. Compute the IDs still referenced by any obligation or
             hypothesis ``evidence_ids`` list — those are protected.
          2. Drop unreferenced records, lowest-confidence first, until
             the dict is back inside the soft cap.
          3. As a last resort, drop the oldest non-accepted referenced
             records (keeping at least one record per protected ID
             list to maintain causal traceability).
        Returns the number of evicted records.
        """

        target_size = max(1, int(_MAX_EVIDENCE_RECORDS * 0.85))
        if len(self.evidence) <= target_size:
            return 0
        referenced_ids: set[str] = set()
        for obligation in self.obligations.values():
            referenced_ids.update(obligation.evidence_ids)
        for hypothesis in self.hypotheses.values():
            referenced_ids.update(hypothesis.evidence_ids)
        unreferenced = [
            evidence
            for evidence in self.evidence.values()
            if evidence.evidence_id not in referenced_ids
        ]
        unreferenced.sort(
            key=lambda evidence: (float(evidence.confidence), bool(evidence.accepted))
        )
        evicted = 0
        for evidence in unreferenced:
            if len(self.evidence) <= target_size:
                break
            self.evidence.pop(evidence.evidence_id, None)
            evicted += 1
        if len(self.evidence) <= target_size:
            return evicted
        # Still over cap: evict referenced-but-not-accepted records,
        # oldest by lowest confidence first. Their IDs in obligation /
        # hypothesis lists are tolerated as missing by accessor code.
        weak_referenced = [evidence for evidence in self.evidence.values() if not evidence.accepted]
        weak_referenced.sort(key=lambda evidence: float(evidence.confidence))
        for evidence in weak_referenced:
            if len(self.evidence) <= target_size:
                break
            self.evidence.pop(evidence.evidence_id, None)
            evicted += 1
        return evicted

    def _mark_conflicting_hypotheses(
        self,
        *,
        file_paths: list[str],
        exclude_hypothesis_ids: Optional[list[str]] = None,
        contradiction_delta: float,
    ) -> None:
        normalized_paths = set(_normalize_file_paths(file_paths))
        if not normalized_paths or contradiction_delta <= 0.0:
            return
        excluded = set(exclude_hypothesis_ids or [])
        for hypothesis in self.hypotheses.values():
            if hypothesis.hypothesis_id in excluded:
                continue
            overlap = normalized_paths.intersection(hypothesis.file_paths)
            if not overlap:
                continue
            overlap_ratio = len(overlap) / max(
                len(set(hypothesis.file_paths).union(normalized_paths)), 1
            )
            delta = contradiction_delta * max(0.45, overlap_ratio)
            hypothesis.contradiction_mass += delta
            hypothesis.contradiction_score = _clamp(hypothesis.contradiction_score + delta)

    def _verification_summary(self, verification: dict[str, Any], accepted: bool) -> str:
        if accepted:
            return "Verification accepted the candidate patch."
        test_result = verification.get("test_result")
        if isinstance(test_result, dict):
            passed = int(test_result.get("passed") or 0)
            failed = int(test_result.get("failed") or 0)
            errors = int(test_result.get("errors") or 0)
            return (
                "Verification did not accept the candidate patch "
                f"(passed={passed}, failed={failed}, errors={errors})."
            )
        return "Verification did not accept the candidate patch."
