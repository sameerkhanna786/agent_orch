"""
Language and benchmark adapters that emit generic controller observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


def _dedupe_strings(values: Iterable[Any]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        ordered.append(text)
        seen.add(text)
    return ordered


@dataclass
class ControllerObservation:
    """Generic observation emitted by a benchmark/language adapter."""

    state: str
    source: str
    strength: float
    rationale: str
    file_paths: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceAdapterResult:
    observations: list[ControllerObservation] = field(default_factory=list)
    feature_map: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _dedupe_lower_strings(values: Iterable[Any]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip().lower()
        if not text or text in seen:
            continue
        ordered.append(text)
        seen.add(text)
    return ordered


def _pattern_list(config: Any, field_name: str) -> list[str]:
    return [
        str(pattern).strip().lower()
        for pattern in list(getattr(config, field_name, []) or [])
        if str(pattern).strip()
    ]


def _append_observation(
    result: EvidenceAdapterResult,
    *,
    state: str,
    source: str,
    rationale: str,
    strength: float = 1.0,
    file_paths: Optional[Iterable[Any]] = None,
    test_ids: Optional[Iterable[Any]] = None,
    symbols: Optional[Iterable[Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    result.observations.append(
        ControllerObservation(
            state=state,
            source=source,
            strength=float(strength),
            rationale=str(rationale or ""),
            file_paths=_dedupe_strings(file_paths or []),
            test_ids=_dedupe_strings(test_ids or []),
            symbols=_dedupe_strings(symbols or []),
            metadata=dict(metadata or {}),
        )
    )
    result.feature_map[f"obs__{source}"] = float(
        result.feature_map.get(f"obs__{source}", 0.0)
    ) + float(strength)
    result.feature_map[f"state__{state}"] = float(
        result.feature_map.get(f"state__{state}", 0.0)
    ) + float(strength)


class GenericEvidenceAdapter:
    """Translate benchmark-agnostic task evidence into controller observations."""

    def __init__(self, config: Any):
        self.config = config

    def build(
        self,
        *,
        issue_description: str,
        failing_test_ids: Optional[Iterable[Any]] = None,
        passing_test_ids: Optional[Iterable[Any]] = None,
        terminal_source_files: Optional[Iterable[Any]] = None,
        source_focus_files: Optional[Iterable[Any]] = None,
        incomplete_source_files: Optional[Iterable[Any]] = None,
        incomplete_test_files: Optional[Iterable[Any]] = None,
        relevant_files: Optional[Iterable[Any]] = None,
        exception_summaries: Optional[Iterable[Any]] = None,
        interface_symbols: Optional[Iterable[Any]] = None,
        preserve_collected_test_coverage: bool = False,
        relevant_file_languages: Optional[Iterable[Any]] = None,
        repo_languages: Optional[Iterable[Any]] = None,
        test_command: Optional[str] = None,
    ) -> EvidenceAdapterResult:
        completion_patterns = _pattern_list(self.config, "completion_task_patterns")
        public_api_patterns = _pattern_list(self.config, "public_api_patterns")
        lowered_issue = str(issue_description or "").strip().lower()
        failing_ids = _dedupe_strings(failing_test_ids or [])
        passing_ids = _dedupe_strings(passing_test_ids or [])
        terminal_sources = _dedupe_strings(terminal_source_files or [])
        source_focus = _dedupe_strings(source_focus_files or [])
        incomplete_sources = _dedupe_strings(incomplete_source_files or [])
        incomplete_tests = _dedupe_strings(incomplete_test_files or [])
        relevant = _dedupe_strings(relevant_files or [])
        exceptions = _dedupe_strings(exception_summaries or [])
        interface = _dedupe_strings(interface_symbols or [])
        relevant_languages = _dedupe_lower_strings(relevant_file_languages or [])
        repo_language_hints = _dedupe_lower_strings(repo_languages or [])
        lowered_test_command = str(test_command or "").strip().lower()
        result = EvidenceAdapterResult(
            feature_map={
                "failing_test_count": float(len(failing_ids)),
                "passing_test_count": float(len(passing_ids)),
                "terminal_source_count": float(len(terminal_sources)),
                "source_focus_count": float(len(source_focus)),
                "incomplete_source_count": float(len(incomplete_sources)),
                "incomplete_test_count": float(len(incomplete_tests)),
                "relevant_file_count": float(len(relevant)),
                "interface_symbol_count": float(len(interface)),
                "exception_count": float(len(exceptions)),
                "relevant_language_count": float(len(relevant_languages)),
                "repo_language_count": float(len(repo_language_hints)),
                "preserve_collected_test_coverage": (
                    1.0 if preserve_collected_test_coverage else 0.0
                ),
                "has_test_command": 1.0 if lowered_test_command else 0.0,
                "adapter_is_generic": 1.0,
                "adapter_is_python_pytest": 0.0,
            },
            metadata={
                "adapter": "generic_v1",
                "relevant_file_languages": list(relevant_languages),
                "repo_languages": list(repo_language_hints),
            },
        )

        if any(pattern in lowered_issue for pattern in completion_patterns):
            _append_observation(
                result,
                state="contract_gap",
                source="completion_pattern",
                rationale="Issue text reads like missing behavior or repository-completion work.",
                file_paths=relevant[:4],
            )
        if any(pattern in lowered_issue for pattern in public_api_patterns):
            _append_observation(
                result,
                state="high_interface_risk",
                source="public_api_pattern",
                rationale="Issue text explicitly references public API or compatibility constraints.",
                file_paths=relevant[:4],
                symbols=interface[:4],
            )
            _append_observation(
                result,
                state="contract_gap",
                source="public_api_contract",
                rationale="Public API constraints imply a broader contract-completion task.",
                file_paths=relevant[:4],
                symbols=interface[:4],
                strength=0.75,
            )
        if not passing_ids and failing_ids and (terminal_sources or exceptions):
            _append_observation(
                result,
                state="importability_blocker",
                source="zero_passing_with_traceback",
                rationale=(
                    "No visible tests pass and the failure already converges to a narrow "
                    "traceback/import path."
                ),
                file_paths=terminal_sources + source_focus[:4],
                test_ids=failing_ids[:4],
            )
        if terminal_sources:
            _append_observation(
                result,
                state="importability_blocker",
                source="terminal_source_focus",
                rationale="Traceback evidence identifies a terminal source focus.",
                file_paths=terminal_sources[:4],
                test_ids=failing_ids[:3],
            )
        if incomplete_sources:
            _append_observation(
                result,
                state="contract_gap",
                source="incomplete_source_scaffold",
                rationale="Nearby source files still contain clear implementation scaffolds.",
                file_paths=incomplete_sources[:4],
            )
        if incomplete_tests:
            _append_observation(
                result,
                state="contract_gap",
                source="incomplete_test_scaffold",
                rationale="Visible tests contain scaffolds that imply a missing contract.",
                test_ids=incomplete_tests[:4],
            )
        if len(failing_ids) >= 4:
            _append_observation(
                result,
                state="broad_regression",
                source="failing_test_breadth",
                rationale="The visible failing surface is broad enough to indicate regression pressure.",
                test_ids=failing_ids[:6],
                file_paths=source_focus[:4],
            )
        if passing_ids and failing_ids:
            _append_observation(
                result,
                state="broad_regression",
                source="mixed_pass_fail_surface",
                rationale="Mixed passing and failing visible tests raise regression risk around the fix.",
                test_ids=failing_ids[:4] + passing_ids[:2],
            )
        if len(relevant) >= 6:
            _append_observation(
                result,
                state="broad_regression",
                source="relevant_file_breadth",
                rationale="Relevant evidence spans multiple files.",
                file_paths=relevant[:6],
            )
        if interface:
            _append_observation(
                result,
                state="high_interface_risk",
                source="interface_symbol_signal",
                rationale=(
                    "The current focus exposes shared interface symbols across candidate "
                    "edit regions."
                ),
                file_paths=source_focus[:4] or relevant[:4],
                symbols=interface[:6],
            )
        if len(source_focus) >= 2 and len(relevant) >= 4:
            _append_observation(
                result,
                state="high_interface_risk",
                source="multi_module_focus",
                rationale=(
                    "The likely fix spans multiple source modules and may cross interface "
                    "boundaries."
                ),
                file_paths=source_focus[:4],
            )
        if preserve_collected_test_coverage:
            _append_observation(
                result,
                state="broad_regression",
                source="coverage_preservation_invariant",
                rationale=(
                    "Evaluation requires preserving collected test coverage under the main "
                    "test command."
                ),
                test_ids=failing_ids[:4],
            )
        return result


class PythonPytestEvidenceAdapter(GenericEvidenceAdapter):
    """Translate Python/pytest-specific evidence into generic controller observations."""

    def build(
        self,
        *,
        issue_description: str,
        failing_test_ids: Optional[Iterable[Any]] = None,
        passing_test_ids: Optional[Iterable[Any]] = None,
        terminal_source_files: Optional[Iterable[Any]] = None,
        source_focus_files: Optional[Iterable[Any]] = None,
        incomplete_source_files: Optional[Iterable[Any]] = None,
        incomplete_test_files: Optional[Iterable[Any]] = None,
        relevant_files: Optional[Iterable[Any]] = None,
        exception_summaries: Optional[Iterable[Any]] = None,
        interface_symbols: Optional[Iterable[Any]] = None,
        preserve_collected_test_coverage: bool = False,
        relevant_file_languages: Optional[Iterable[Any]] = None,
        repo_languages: Optional[Iterable[Any]] = None,
        test_command: Optional[str] = None,
    ) -> EvidenceAdapterResult:
        result = super().build(
            issue_description=issue_description,
            failing_test_ids=failing_test_ids,
            passing_test_ids=passing_test_ids,
            terminal_source_files=terminal_source_files,
            source_focus_files=source_focus_files,
            incomplete_source_files=incomplete_source_files,
            incomplete_test_files=incomplete_test_files,
            relevant_files=relevant_files,
            exception_summaries=exception_summaries,
            interface_symbols=interface_symbols,
            preserve_collected_test_coverage=preserve_collected_test_coverage,
            relevant_file_languages=relevant_file_languages,
            repo_languages=repo_languages,
            test_command=test_command,
        )
        failing_ids = _dedupe_strings(failing_test_ids or [])
        terminal_sources = _dedupe_strings(terminal_source_files or [])
        source_focus = _dedupe_strings(source_focus_files or [])
        failing_collection_files = [
            test_id for test_id in failing_ids if test_id.endswith(".py") and "::" not in test_id
        ]
        result.metadata["adapter"] = "python_pytest_v1"
        result.feature_map["adapter_is_generic"] = 0.0
        result.feature_map["adapter_is_python_pytest"] = 1.0
        result.feature_map["failing_collection_file_count"] = float(len(failing_collection_files))
        result.feature_map["python_pytest_command"] = (
            1.0 if "pytest" in str(test_command or "").strip().lower() else 0.0
        )
        if len(failing_collection_files) >= 3:
            _append_observation(
                result,
                state="importability_blocker",
                source="collection_error_cluster",
                rationale="Visible failures concentrate in collection/import-stage modules.",
                test_ids=failing_collection_files[:4],
                file_paths=terminal_sources + source_focus[:4],
            )
        return result
