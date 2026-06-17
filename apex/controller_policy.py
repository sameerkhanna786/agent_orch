"""
Shared controller policy abstractions for planner, search, rollout, and telemetry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .controller_adapters import (
    EvidenceAdapterResult,
    GenericEvidenceAdapter,
    PythonPytestEvidenceAdapter,
)
from .controller_models import evaluate_policy_model
from .core.pytest_utils import parse_pytest_command, render_pytest_command

TASK_REGIME_STATES = (
    "importability_blocker",
    "contract_gap",
    "broad_regression",
    "high_interface_risk",
)

EVIDENCE_MODE_GOLD_SUITE_VISIBLE = "gold_suite_visible"
EVIDENCE_MODE_PARTIAL_SUITE_VISIBLE = "partial_suite_visible"
EVIDENCE_MODE_NO_SUITE_VISIBLE = "no_suite_visible"
EVIDENCE_MODE_EVAL_ONLY_SUITE = "eval_only_suite"

EVIDENCE_MODES = frozenset(
    {
        EVIDENCE_MODE_GOLD_SUITE_VISIBLE,
        EVIDENCE_MODE_PARTIAL_SUITE_VISIBLE,
        EVIDENCE_MODE_NO_SUITE_VISIBLE,
        EVIDENCE_MODE_EVAL_ONLY_SUITE,
    }
)

_VISIBLE_TEST_SOURCES = frozenset(
    {
        "public",
        "repo_public",
        "repo_visible",
        "visible",
        "visible_tests",
        "agent_visible",
        "agent_visible_test_command",
        "commit0_public_test_inventory",
    }
)

_EVAL_ONLY_TEST_SOURCES = frozenset(
    {
        "benchmark_expected",
        "hidden",
        "private",
        "evaluator_only",
        "evaluation_constraints",
        "official_evaluator",
    }
)


def _clamp(value: float, *, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


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


def normalize_evidence_mode(value: Any) -> str:
    """Return the canonical rollout evidence mode."""

    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "gold": EVIDENCE_MODE_GOLD_SUITE_VISIBLE,
        "gold_visible": EVIDENCE_MODE_GOLD_SUITE_VISIBLE,
        "gold_tests_visible": EVIDENCE_MODE_GOLD_SUITE_VISIBLE,
        "gold_suite": EVIDENCE_MODE_GOLD_SUITE_VISIBLE,
        "gold_suite_visible": EVIDENCE_MODE_GOLD_SUITE_VISIBLE,
        "complete_suite_visible": EVIDENCE_MODE_GOLD_SUITE_VISIBLE,
        "authoritative_visible": EVIDENCE_MODE_GOLD_SUITE_VISIBLE,
        "partial": EVIDENCE_MODE_PARTIAL_SUITE_VISIBLE,
        "partial_visible": EVIDENCE_MODE_PARTIAL_SUITE_VISIBLE,
        "partial_suite": EVIDENCE_MODE_PARTIAL_SUITE_VISIBLE,
        "partial_suite_visible": EVIDENCE_MODE_PARTIAL_SUITE_VISIBLE,
        "weak_public": EVIDENCE_MODE_PARTIAL_SUITE_VISIBLE,
        "visible_but_incomplete": EVIDENCE_MODE_PARTIAL_SUITE_VISIBLE,
        "none": EVIDENCE_MODE_NO_SUITE_VISIBLE,
        "no_suite": EVIDENCE_MODE_NO_SUITE_VISIBLE,
        "no_suite_visible": EVIDENCE_MODE_NO_SUITE_VISIBLE,
        "no_tests": EVIDENCE_MODE_NO_SUITE_VISIBLE,
        "hidden": EVIDENCE_MODE_EVAL_ONLY_SUITE,
        "eval_only": EVIDENCE_MODE_EVAL_ONLY_SUITE,
        "eval_only_suite": EVIDENCE_MODE_EVAL_ONLY_SUITE,
        "evaluator_only": EVIDENCE_MODE_EVAL_ONLY_SUITE,
        "official_evaluator_only": EVIDENCE_MODE_EVAL_ONLY_SUITE,
    }
    return aliases.get(token, token if token in EVIDENCE_MODES else "")


@dataclass
class EvidencePolicy:
    """Visibility and scoring policy for the tests/evidence Apex may use."""

    mode: str = EVIDENCE_MODE_NO_SUITE_VISIBLE
    visible_tests_completeness: str = "none"
    visible_tests_allowed_in_prompts: bool = False
    visible_tests_allowed_for_candidate_scoring: bool = False
    visible_failures_allowed_for_followup: bool = False
    evaluation_tests_visibility: str = "none"
    evaluation_feedback_policy: str = "none"
    provenance_label: str = ""

    def normalized(self) -> "EvidencePolicy":
        mode = normalize_evidence_mode(self.mode) or EVIDENCE_MODE_NO_SUITE_VISIBLE
        completeness = str(self.visible_tests_completeness or "").strip().lower()
        evaluation_visibility = str(self.evaluation_tests_visibility or "").strip().lower()
        feedback_policy = str(self.evaluation_feedback_policy or "").strip().lower()
        provenance = str(self.provenance_label or "").strip().lower()

        if mode == EVIDENCE_MODE_GOLD_SUITE_VISIBLE:
            completeness = completeness or "gold"
            evaluation_visibility = evaluation_visibility or "agent_visible"
            feedback_policy = feedback_policy or "full_visible_feedback"
            return EvidencePolicy(
                mode=mode,
                visible_tests_completeness=completeness,
                visible_tests_allowed_in_prompts=True,
                visible_tests_allowed_for_candidate_scoring=True,
                visible_failures_allowed_for_followup=True,
                evaluation_tests_visibility=evaluation_visibility,
                evaluation_feedback_policy=feedback_policy,
                provenance_label=provenance or mode,
            )

        if mode == EVIDENCE_MODE_PARTIAL_SUITE_VISIBLE:
            completeness = completeness or "partial_or_unknown"
            evaluation_visibility = evaluation_visibility or "evaluator_only"
            feedback_policy = feedback_policy or "aggregate_score_only"
            return EvidencePolicy(
                mode=mode,
                visible_tests_completeness=completeness,
                visible_tests_allowed_in_prompts=True,
                visible_tests_allowed_for_candidate_scoring=True,
                visible_failures_allowed_for_followup=True,
                evaluation_tests_visibility=evaluation_visibility,
                evaluation_feedback_policy=feedback_policy,
                provenance_label=provenance or mode,
            )

        if mode == EVIDENCE_MODE_EVAL_ONLY_SUITE:
            return EvidencePolicy(
                mode=mode,
                visible_tests_completeness="none",
                visible_tests_allowed_in_prompts=False,
                visible_tests_allowed_for_candidate_scoring=False,
                visible_failures_allowed_for_followup=False,
                evaluation_tests_visibility=evaluation_visibility or "evaluator_only",
                evaluation_feedback_policy=feedback_policy or "aggregate_score_only",
                provenance_label=provenance or mode,
            )

        return EvidencePolicy(
            mode=EVIDENCE_MODE_NO_SUITE_VISIBLE,
            visible_tests_completeness="none",
            visible_tests_allowed_in_prompts=False,
            visible_tests_allowed_for_candidate_scoring=False,
            visible_failures_allowed_for_followup=False,
            evaluation_tests_visibility=evaluation_visibility or "none",
            evaluation_feedback_policy=feedback_policy or "none",
            provenance_label=provenance or EVIDENCE_MODE_NO_SUITE_VISIBLE,
        )

    def to_dict(self) -> dict[str, Any]:
        policy = self.normalized()
        return {
            "mode": policy.mode,
            "visible_tests_completeness": policy.visible_tests_completeness,
            "visible_tests_allowed_in_prompts": bool(policy.visible_tests_allowed_in_prompts),
            "visible_tests_allowed_for_candidate_scoring": bool(
                policy.visible_tests_allowed_for_candidate_scoring
            ),
            "visible_failures_allowed_for_followup": bool(
                policy.visible_failures_allowed_for_followup
            ),
            "evaluation_tests_visibility": policy.evaluation_tests_visibility,
            "evaluation_feedback_policy": policy.evaluation_feedback_policy,
            "provenance_label": policy.provenance_label,
        }

    def to_artifact_dict(self) -> dict[str, Any]:
        policy = self.normalized()
        return {
            "mode": policy.mode,
            "visible_tests_completeness": policy.visible_tests_completeness,
            "visible_tests_allowed_in_prompts": bool(policy.visible_tests_allowed_in_prompts),
            "visible_tests_allowed_for_candidate_scoring": bool(
                policy.visible_tests_allowed_for_candidate_scoring
            ),
            "visible_failures_allowed_for_followup": bool(
                policy.visible_failures_allowed_for_followup
            ),
            "evaluation_tests_visibility": policy.evaluation_tests_visibility,
            "evaluation_feedback_policy": policy.evaluation_feedback_policy,
            "provenance_label": policy.provenance_label,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "EvidencePolicy":
        if isinstance(data, EvidencePolicy):
            return data.normalized()
        if not isinstance(data, dict):
            return cls().normalized()
        return cls(
            mode=normalize_evidence_mode(data.get("mode")),
            visible_tests_completeness=str(data.get("visible_tests_completeness") or ""),
            visible_tests_allowed_in_prompts=bool(data.get("visible_tests_allowed_in_prompts")),
            visible_tests_allowed_for_candidate_scoring=bool(
                data.get("visible_tests_allowed_for_candidate_scoring")
            ),
            visible_failures_allowed_for_followup=bool(
                data.get("visible_failures_allowed_for_followup")
            ),
            evaluation_tests_visibility=str(data.get("evaluation_tests_visibility") or ""),
            evaluation_feedback_policy=str(data.get("evaluation_feedback_policy") or ""),
            provenance_label=str(data.get("provenance_label") or ""),
        ).normalized()


EvidencePolicy.__test__ = False


def _normalize_test_inventory_framework(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "pytest": "pytest",
        "py_test": "pytest",
        "python_pytest": "pytest",
        "jest": "jest",
        "vitest": "vitest",
        "go": "go_test",
        "go_test": "go_test",
        "gotest": "go_test",
        "cargo": "cargo_test",
        "cargo_test": "cargo_test",
        "rust_test": "cargo_test",
        "junit": "junit",
        "maven": "junit",
        "gradle": "junit",
        "rspec": "rspec",
        "ruby": "rspec",
        "dotnet": "dotnet_test",
        "dotnet_test": "dotnet_test",
        "phpunit": "phpunit",
        "ctest": "ctest",
        "unittest": "unittest",
    }
    return aliases.get(token, token)


def default_test_inventory_language(framework: str) -> str:
    normalized = _normalize_test_inventory_framework(framework)
    aliases = {
        "pytest": "python",
        "unittest": "python",
        "jest": "javascript",
        "vitest": "typescript",
        "go_test": "go",
        "cargo_test": "rust",
        "junit": "java",
        "rspec": "ruby",
        "dotnet_test": "csharp",
        "phpunit": "php",
        "ctest": "cpp",
    }
    return aliases.get(normalized, "")


def _derive_pytest_collection_command(test_command: Optional[str]) -> str:
    command = str(test_command or "").strip()
    if not command:
        return "pytest --collect-only -q"
    parsed = parse_pytest_command(command)
    if parsed is None:
        lowered = command.lower()
        if "pytest" not in lowered:
            return ""
        return "pytest --collect-only -q"

    option_tokens = [
        token
        for token in parsed.option_tokens
        if token
        not in {
            "-q",
            "--quiet",
            "-v",
            "-vv",
            "--tb=no",
            "--collect-only",
        }
    ]
    option_tokens.extend(["--collect-only", "-q"])
    collect_only = type(parsed)(
        shell_prefix_tokens=parsed.shell_prefix_tokens,
        env_prefix_tokens=parsed.env_prefix_tokens,
        invocation_tokens=parsed.invocation_tokens,
        option_tokens=tuple(option_tokens),
        target_tokens=parsed.target_tokens,
    )
    return render_pytest_command(
        collect_only,
        disable_plugin_autoload=False,
    )


def derive_test_collection_command(
    test_command: Optional[str],
    *,
    framework: str = "",
) -> str:
    command = str(test_command or "").strip()
    normalized_framework = infer_test_inventory_framework(
        test_command=command,
        explicit_framework=framework,
    )
    lowered = command.lower()

    if normalized_framework == "pytest":
        return _derive_pytest_collection_command(command)

    if normalized_framework == "unittest":
        if not command:
            return "python -m unittest discover"
        if "discover" in lowered:
            return command
        return "python -m unittest discover"

    if normalized_framework == "jest":
        if not command:
            return "npx jest --listTests"
        if "--listtests" in lowered:
            return command
        if re.search(r"\bnpm\b.*\btest\b", lowered):
            return f"{command} -- --listTests"
        if re.search(r"\bpnpm\b.*\btest\b", lowered):
            return f"{command} -- --listTests"
        if re.search(r"\byarn\b.*\btest\b", lowered):
            return f"{command} --listTests"
        if "jest" in lowered:
            return f"{command} --listTests"
        return "npx jest --listTests"

    if normalized_framework == "vitest":
        if not command:
            return "npx vitest list"
        if re.search(r"\bvitest\s+list\b", lowered):
            return command
        if "vitest" in lowered:
            return re.sub(
                r"\bvitest(?:\s+(?:run|watch))?\b",
                "vitest list",
                command,
                count=1,
            )
        return "npx vitest list"

    if normalized_framework == "go_test":
        if not command:
            return "go test ./... -list ."
        if re.search(r"(?:^|\s)-list(?:=|\s+)\S+", command):
            return command
        sanitized = re.sub(r"\s+-run(?:=|\s+)\S+", "", command)
        sanitized = re.sub(r"\s+-count(?:=|\s+)\S+", "", sanitized)
        sanitized = re.sub(r"\s+-failfast\b", "", sanitized)
        return f"{sanitized.strip()} -list .".strip()

    if normalized_framework == "cargo_test":
        if not command:
            return "cargo test -- --list"
        if "-- --list" in command:
            return command
        sanitized = re.sub(r"\s+--\s+--list\b.*$", "", command).strip()
        return f"{sanitized} -- --list".strip()

    if normalized_framework == "junit":
        if not command:
            return ""
        if "gradle" in lowered:
            if "--test-dry-run" in lowered:
                return command
            return f"{command} --test-dry-run"
        if "mvn" in lowered:
            if "-dtest=" in lowered:
                return command
            if re.search(r"\btest\b", command):
                return re.sub(
                    r"\btest\b",
                    "-Dtest=* -DfailIfNoTests=false test",
                    command,
                    count=1,
                )
            return f"{command} -Dtest=* -DfailIfNoTests=false test"
        return command

    if normalized_framework == "rspec":
        if not command:
            return "rspec --dry-run"
        if "--dry-run" in lowered:
            return command
        return f"{command} --dry-run"

    if normalized_framework == "dotnet_test":
        if not command:
            return "dotnet test --list-tests"
        if "--list-tests" in lowered:
            return command
        return f"{command} --list-tests"

    if normalized_framework == "phpunit":
        if not command:
            return "phpunit --list-tests"
        if "--list-tests" in lowered:
            return command
        return f"{command} --list-tests"

    if normalized_framework == "ctest":
        if not command:
            return "ctest -N"
        if re.search(r"(?:^|\s)-n(?:\s|$)", lowered):
            return command
        return f"{command} -N"

    return ""


def infer_test_inventory_framework(
    *,
    expected_test_ids: Optional[Iterable[Any]] = None,
    test_command: Optional[str] = None,
    explicit_framework: str = "",
) -> str:
    return _infer_test_inventory_framework(
        expected_test_ids=expected_test_ids,
        test_command=test_command,
        explicit_framework=explicit_framework,
    )


@dataclass
class ShadowPolicyOption:
    """One scored policy option for offline counterfactual evaluation."""

    option_id: str
    score: float
    rationale: str = ""
    selected: bool = False
    category: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "score": round(float(self.score), 4),
            "rationale": self.rationale,
            "selected": bool(self.selected),
            "category": self.category,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShadowPolicyOption":
        return cls(
            option_id=str(data.get("option_id") or ""),
            score=float(data.get("score") or 0.0),
            rationale=str(data.get("rationale") or ""),
            selected=bool(data.get("selected")),
            category=str(data.get("category") or ""),
            metadata=dict(data.get("metadata") or {}),
        )


def build_shadow_policy_trace(
    *,
    decision: str,
    options: list[ShadowPolicyOption],
    max_logged_options: int = 3,
) -> dict[str, Any]:
    ranked = sorted(
        options,
        key=lambda item: (float(item.score), item.option_id),
        reverse=True,
    )
    limit = max(1, int(max_logged_options))
    chosen = next((item for item in ranked if item.selected), ranked[0] if ranked else None)
    return {
        "decision": str(decision or "").strip(),
        "chosen_option": chosen.option_id if chosen is not None else "",
        "chosen_score": round(float(chosen.score), 4) if chosen is not None else 0.0,
        "options": [item.to_dict() for item in ranked[:limit]],
        "counterfactual_options": [
            item.to_dict()
            for item in ranked
            if chosen is None or item.option_id != chosen.option_id
        ][: max(0, limit - 1)],
    }


@dataclass
class TaskRegimeEvidence:
    """Observed evidence supporting one generic controller regime state."""

    state: str
    source: str
    weight: float
    rationale: str
    file_paths: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "source": self.source,
            "weight": round(float(self.weight), 4),
            "rationale": self.rationale,
            "file_paths": list(self.file_paths),
            "test_ids": list(self.test_ids),
            "symbols": list(self.symbols),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskRegimeEvidence":
        return cls(
            state=str(data.get("state") or ""),
            source=str(data.get("source") or ""),
            weight=float(data.get("weight") or 0.0),
            rationale=str(data.get("rationale") or ""),
            file_paths=_dedupe_strings(list(data.get("file_paths") or [])),
            test_ids=_dedupe_strings(list(data.get("test_ids") or [])),
            symbols=_dedupe_strings(list(data.get("symbols") or [])),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class TaskRegimeProfile:
    """Calibrated regime probabilities plus the underlying evidence."""

    state_probabilities: dict[str, float] = field(default_factory=dict)
    evidence: list[TaskRegimeEvidence] = field(default_factory=list)
    planner_invariants: list[str] = field(default_factory=list)
    summary: str = ""
    adapter_name: str = ""

    def probability(self, state: str) -> float:
        return float(self.state_probabilities.get(state) or 0.0)

    def active_states(self, *, minimum_probability: float = 0.35) -> list[str]:
        return [
            state
            for state, probability in sorted(
                self.state_probabilities.items(),
                key=lambda item: (float(item[1]), item[0]),
                reverse=True,
            )
            if float(probability) >= float(minimum_probability)
        ]

    def evidence_for_state(self, state: str) -> list[TaskRegimeEvidence]:
        return [item for item in self.evidence if item.state == state]

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_probabilities": {
                state: round(float(probability), 4)
                for state, probability in self.state_probabilities.items()
            },
            "evidence": [item.to_dict() for item in self.evidence],
            "planner_invariants": list(self.planner_invariants),
            "summary": self.summary,
            "adapter_name": self.adapter_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskRegimeProfile":
        probabilities = {
            str(state): float(probability)
            for state, probability in dict(data.get("state_probabilities") or {}).items()
            if str(state)
        }
        return cls(
            state_probabilities=probabilities,
            evidence=[
                TaskRegimeEvidence.from_dict(item)
                for item in list(data.get("evidence") or [])
                if isinstance(item, dict)
            ],
            planner_invariants=_dedupe_strings(list(data.get("planner_invariants") or [])),
            summary=str(data.get("summary") or ""),
            adapter_name=str(data.get("adapter_name") or ""),
        )


@dataclass
class TestInventory:
    """Canonical description of the known/discoverable test surface."""

    framework: str = ""
    language: str = ""
    source: str = ""
    expected_test_count: int = 0
    expected_test_ids: list[str] = field(default_factory=list)
    collection_command: str = ""
    test_command: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "TestInventory":
        expected_ids = _dedupe_strings(list(self.expected_test_ids or []))
        test_command = str(self.test_command or "").strip()
        framework = infer_test_inventory_framework(
            expected_test_ids=expected_ids,
            test_command=test_command,
            explicit_framework=self.framework,
        )
        expected_count = max(
            max(0, int(self.expected_test_count or 0)),
            len(expected_ids),
        )
        return TestInventory(
            framework=framework,
            language=str(self.language or "").strip().lower()
            or default_test_inventory_language(framework),
            source=str(self.source or "").strip().lower(),
            expected_test_count=expected_count,
            expected_test_ids=expected_ids,
            collection_command=str(self.collection_command or "").strip()
            or derive_test_collection_command(
                test_command,
                framework=framework,
            ),
            test_command=test_command,
            metadata=dict(self.metadata or {}),
        )

    def is_empty(self) -> bool:
        normalized = self.normalized()
        return not (
            normalized.framework
            or normalized.language
            or normalized.expected_test_count
            or normalized.expected_test_ids
            or normalized.collection_command
            or normalized.test_command
            or normalized.metadata
        )

    def merged_with(self, fallback: "TestInventory") -> "TestInventory":
        primary = self.normalized()
        backup = fallback.normalized()
        expected_ids = primary.expected_test_ids or backup.expected_test_ids
        expected_count = max(
            primary.expected_test_count,
            backup.expected_test_count,
            len(expected_ids),
        )
        merged_metadata = dict(backup.metadata)
        merged_metadata.update(primary.metadata)
        return TestInventory(
            framework=primary.framework or backup.framework,
            language=primary.language or backup.language,
            source=primary.source or backup.source,
            expected_test_count=expected_count,
            expected_test_ids=expected_ids,
            collection_command=primary.collection_command or backup.collection_command,
            test_command=primary.test_command or backup.test_command,
            metadata=merged_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        normalized = self.normalized()
        return {
            "framework": normalized.framework,
            "language": normalized.language,
            "source": normalized.source,
            "expected_test_count": int(normalized.expected_test_count or 0),
            "expected_test_ids": list(normalized.expected_test_ids),
            "collection_command": normalized.collection_command,
            "test_command": normalized.test_command,
            "metadata": dict(normalized.metadata),
        }

    def to_artifact_dict(self) -> dict[str, Any]:
        normalized = self.normalized()
        return {
            "framework": normalized.framework,
            "language": normalized.language,
            "source": "",
            "expected_test_count": 0,
            "expected_test_ids": [],
            "collection_command": normalized.collection_command,
            "test_command": normalized.test_command,
            "metadata": {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestInventory":
        return cls(
            framework=str(data.get("framework") or ""),
            language=str(data.get("language") or ""),
            source=str(data.get("source") or ""),
            expected_test_count=max(0, int(data.get("expected_test_count") or 0)),
            expected_test_ids=_dedupe_strings(list(data.get("expected_test_ids") or [])),
            collection_command=str(data.get("collection_command") or ""),
            test_command=str(data.get("test_command") or ""),
            metadata=dict(data.get("metadata") or {}),
        ).normalized()


TestInventory.__test__ = False


def infer_evidence_policy(
    *,
    evidence_mode: Any = "",
    test_inventory: Optional[TestInventory] = None,
    has_visible_test_command: bool = False,
    expected_test_count: int = 0,
    expected_test_ids: Optional[Iterable[Any]] = None,
    default_provenance_label: str = "",
) -> EvidencePolicy:
    """Infer a conservative evidence policy from explicit metadata.

    The caller should pass ``evidence_mode`` whenever it knows the suite
    semantics. When the mode is absent, this helper only infers visibility
    from source/provenance labels; it never upgrades an arbitrary expected
    test list to gold authority.
    """

    normalized_mode = normalize_evidence_mode(evidence_mode)
    inventory = (
        test_inventory.normalized()
        if isinstance(test_inventory, TestInventory)
        else TestInventory()
    )
    source = str(inventory.source or default_provenance_label or "").strip().lower()
    expected_ids = _dedupe_strings(list(expected_test_ids or []))
    expected_count = max(max(0, int(expected_test_count or 0)), len(expected_ids))
    has_test_surface = bool(
        has_visible_test_command
        or inventory.test_command
        or inventory.collection_command
        or inventory.expected_test_count
        or inventory.expected_test_ids
        or expected_count
    )

    if normalized_mode:
        return EvidencePolicy(
            mode=normalized_mode,
            provenance_label=source or default_provenance_label,
        ).normalized()

    if source in _VISIBLE_TEST_SOURCES:
        return EvidencePolicy(
            mode=EVIDENCE_MODE_PARTIAL_SUITE_VISIBLE,
            provenance_label=source,
        ).normalized()

    if source in _EVAL_ONLY_TEST_SOURCES and expected_count > 0:
        return EvidencePolicy(
            mode=EVIDENCE_MODE_EVAL_ONLY_SUITE,
            provenance_label=source,
        ).normalized()

    if has_test_surface:
        return EvidencePolicy(
            mode=EVIDENCE_MODE_PARTIAL_SUITE_VISIBLE,
            provenance_label=source or "visible_test_command",
        ).normalized()

    return EvidencePolicy(
        mode=EVIDENCE_MODE_NO_SUITE_VISIBLE,
        provenance_label=source or default_provenance_label,
    ).normalized()


def _infer_test_inventory_framework(
    *,
    expected_test_ids: Optional[Iterable[Any]] = None,
    test_command: Optional[str] = None,
    explicit_framework: str = "",
) -> str:
    explicit = _normalize_test_inventory_framework(explicit_framework)
    if explicit:
        return explicit

    command = str(test_command or "").strip().lower()
    if "python -m unittest" in command or re.search(r"\bunittest\b", command):
        return "unittest"
    if "pytest" in command:
        return "pytest"
    if "jest" in command:
        return "jest"
    if "vitest" in command:
        return "vitest"
    if "go test" in command:
        return "go_test"
    if "cargo test" in command:
        return "cargo_test"
    if "rspec" in command:
        return "rspec"
    if "dotnet test" in command:
        return "dotnet_test"
    if "phpunit" in command:
        return "phpunit"
    if re.search(r"\bctest\b", command):
        return "ctest"
    if "mvn test" in command or "gradle test" in command:
        return "junit"
    if "mvnw" in command and "test" in command:
        return "junit"
    if "gradlew" in command and "test" in command:
        return "junit"

    ids = [
        str(test_id).strip() for test_id in list(expected_test_ids or []) if str(test_id).strip()
    ]
    if any(".py::" in test_id or test_id.endswith(".py") for test_id in ids):
        return "pytest"
    if any(".go::" in test_id or test_id.endswith(".go") for test_id in ids):
        return "go_test"
    if any(".rs::" in test_id or test_id.endswith(".rs") for test_id in ids):
        return "cargo_test"
    if any(
        test_id.endswith(extension) for test_id in ids for extension in (".java", ".kt", ".kts")
    ):
        return "junit"
    if any(".rb:" in test_id or test_id.endswith(".rb") for test_id in ids):
        return "rspec"
    if any(".cs::" in test_id or test_id.endswith(".cs") for test_id in ids):
        return "dotnet_test"
    if any(".php::" in test_id or test_id.endswith(".php") for test_id in ids):
        return "phpunit"
    return ""


def _coerce_test_inventory_from_payload(
    payload: Any,
    *,
    expected_test_count: Optional[int] = None,
    expected_test_ids: Optional[Iterable[Any]] = None,
    test_command: Optional[str] = None,
    default_source: str = "",
) -> TestInventory:
    if isinstance(payload, TestInventory):
        inventory = payload.normalized()
    elif isinstance(payload, dict):
        inventory = TestInventory.from_dict(payload)
    else:
        inventory = TestInventory()

    fallback_ids = _dedupe_strings(list(expected_test_ids or []))
    fallback_count = max(max(0, int(expected_test_count or 0)), len(fallback_ids))
    fallback_framework = _infer_test_inventory_framework(
        expected_test_ids=fallback_ids,
        test_command=test_command,
        explicit_framework=inventory.framework,
    )
    fallback = TestInventory(
        framework=fallback_framework,
        language=default_test_inventory_language(fallback_framework),
        source=str(default_source or "").strip().lower(),
        expected_test_count=fallback_count,
        expected_test_ids=fallback_ids,
        test_command=str(test_command or "").strip(),
    )
    return inventory.merged_with(fallback)


@dataclass
class EvaluationConstraints:
    """Evaluator-only constraints that should not directly shape planner prompts."""

    expected_test_count: int = 0
    expected_test_ids: list[str] = field(default_factory=list)
    preserve_collected_test_coverage: bool = False
    protect_visible_test_files: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    test_inventory: TestInventory = field(default_factory=TestInventory)
    evidence_policy: EvidencePolicy = field(default_factory=EvidencePolicy)

    def resolved_test_inventory(self) -> TestInventory:
        return _coerce_test_inventory_from_payload(
            self.test_inventory,
            expected_test_count=self.expected_test_count,
            expected_test_ids=self.expected_test_ids,
            test_command=(
                str(self.test_inventory.test_command or "").strip()
                if isinstance(self.test_inventory, TestInventory)
                else ""
            ),
            default_source=(
                str(self.test_inventory.source or "").strip().lower()
                if isinstance(self.test_inventory, TestInventory)
                else ""
            )
            or "evaluation_constraints",
        )

    def to_dict(self) -> dict[str, Any]:
        inventory = self.resolved_test_inventory()
        return {
            "expected_test_count": int(inventory.expected_test_count or 0),
            "expected_test_ids": list(inventory.expected_test_ids),
            "preserve_collected_test_coverage": bool(self.preserve_collected_test_coverage),
            "protect_visible_test_files": bool(self.protect_visible_test_files),
            "metadata": dict(self.metadata),
            "test_inventory": inventory.to_dict(),
            "evidence_policy": self.resolved_evidence_policy().to_dict(),
        }

    def to_artifact_dict(self) -> dict[str, Any]:
        inventory = self.resolved_test_inventory()
        return {
            "expected_test_count": 0,
            "expected_test_ids": [],
            "preserve_collected_test_coverage": bool(self.preserve_collected_test_coverage),
            "protect_visible_test_files": bool(self.protect_visible_test_files),
            "metadata": {},
            "test_inventory": inventory.to_artifact_dict(),
            "evidence_policy": self.resolved_evidence_policy().to_artifact_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationConstraints":
        inventory = _coerce_test_inventory_from_payload(
            data.get("test_inventory") or {},
            expected_test_count=data.get("expected_test_count"),
            expected_test_ids=list(data.get("expected_test_ids") or []),
            default_source="evaluation_constraints",
        )
        return cls(
            expected_test_count=int(inventory.expected_test_count or 0),
            expected_test_ids=list(inventory.expected_test_ids),
            preserve_collected_test_coverage=bool(data.get("preserve_collected_test_coverage")),
            protect_visible_test_files=bool(data.get("protect_visible_test_files")),
            metadata=dict(data.get("metadata") or {}),
            test_inventory=inventory,
            evidence_policy=EvidencePolicy.from_dict(data.get("evidence_policy") or {}),
        )

    def resolved_evidence_policy(self) -> EvidencePolicy:
        inventory = self.resolved_test_inventory()
        explicit_policy = (
            self.evidence_policy
            if isinstance(self.evidence_policy, EvidencePolicy)
            else EvidencePolicy.from_dict(self.evidence_policy)
        )
        has_test_surface = bool(
            inventory.test_command
            or inventory.collection_command
            or inventory.expected_test_count
            or inventory.expected_test_ids
        )
        if normalize_evidence_mode(explicit_policy.mode) and (
            explicit_policy.mode != EVIDENCE_MODE_NO_SUITE_VISIBLE or not has_test_surface
        ):
            return explicit_policy.normalized()
        return infer_evidence_policy(
            evidence_mode=(self.metadata or {}).get("evidence_mode"),
            test_inventory=inventory,
            has_visible_test_command=bool(inventory.test_command),
            expected_test_count=inventory.expected_test_count,
            expected_test_ids=inventory.expected_test_ids,
            default_provenance_label=inventory.source,
        )

    def planner_invariants(self) -> list[str]:
        invariants: list[str] = []
        policy = self.resolved_evidence_policy()
        if policy.mode == EVIDENCE_MODE_GOLD_SUITE_VISIBLE:
            invariants.append(
                "The visible test suite is declared as the gold evaluation suite; optimize directly against it while preserving collection."
            )
        elif policy.mode == EVIDENCE_MODE_PARTIAL_SUITE_VISIBLE:
            invariants.append(
                "Visible tests are useful evidence but may be incomplete; satisfy the issue contract beyond the visible cases."
            )
        elif policy.mode == EVIDENCE_MODE_EVAL_ONLY_SUITE:
            invariants.append(
                "Evaluator-only tests/results must not be used as rollout prompt evidence."
            )
        if self.preserve_collected_test_coverage:
            invariants.append("Do not shrink collected test coverage under the main test command.")
        if self.protect_visible_test_files:
            invariants.append(
                "Treat visible tests as read-only specification unless exact `incomplete_test_files` placeholder bodies are identified. Do not modify tests to make failures disappear or reduce coverage."
            )
        return invariants


def _list_field(payload: Any, field_name: str) -> list[Any]:
    if isinstance(payload, dict):
        values = payload.get(field_name)
    else:
        values = getattr(payload, field_name, None)
    if not isinstance(values, (list, tuple, set)):
        return []
    return list(values)


def _int_field(payload: Any, field_name: str) -> int:
    if isinstance(payload, dict):
        value = payload.get(field_name)
    else:
        value = getattr(payload, field_name, None)
    return int(value or 0) if isinstance(value, int) else 0


def _string_field(payload: Any, field_name: str) -> str:
    if isinstance(payload, dict):
        value = payload.get(field_name)
    else:
        value = getattr(payload, field_name, None)
    return str(value or "").strip() if value is not None else ""


def _nested_field(payload: Any, field_name: str) -> Any:
    if isinstance(payload, dict):
        return payload.get(field_name)
    return getattr(payload, field_name, None)


def coerce_evaluation_constraints(payload: Any) -> EvaluationConstraints:
    """Load canonical evaluator constraints from an issue plan or legacy test context."""

    if isinstance(payload, EvaluationConstraints):
        return payload
    if payload is None:
        return EvaluationConstraints()

    payload_test_command = _string_field(payload, "test_command") or _string_field(
        _nested_field(payload, "test_context"), "command"
    )

    direct_constraints = _nested_field(payload, "evaluation_constraints")
    if isinstance(direct_constraints, EvaluationConstraints):
        direct_constraints_has_explicit_content = (
            direct_constraints.expected_test_count
            or direct_constraints.expected_test_ids
            or direct_constraints.preserve_collected_test_coverage
            or direct_constraints.protect_visible_test_files
            or direct_constraints.metadata
            or direct_constraints.resolved_evidence_policy().mode != EVIDENCE_MODE_NO_SUITE_VISIBLE
            or not direct_constraints.resolved_test_inventory().is_empty()
        )
        if direct_constraints_has_explicit_content:
            direct_constraints = EvaluationConstraints(
                expected_test_count=direct_constraints.expected_test_count,
                expected_test_ids=list(direct_constraints.expected_test_ids),
                preserve_collected_test_coverage=bool(
                    direct_constraints.preserve_collected_test_coverage
                ),
                protect_visible_test_files=bool(direct_constraints.protect_visible_test_files),
                metadata=dict(direct_constraints.metadata or {}),
                test_inventory=_coerce_test_inventory_from_payload(
                    direct_constraints.test_inventory,
                    expected_test_count=direct_constraints.expected_test_count,
                    expected_test_ids=direct_constraints.expected_test_ids,
                    test_command=payload_test_command,
                    default_source=(
                        str(direct_constraints.test_inventory.source or "").strip().lower()
                        if isinstance(direct_constraints.test_inventory, TestInventory)
                        else ""
                    )
                    or "evaluation_constraints",
                ),
                evidence_policy=direct_constraints.resolved_evidence_policy(),
            )
            return direct_constraints
    if isinstance(direct_constraints, dict):
        direct_constraints = EvaluationConstraints.from_dict(direct_constraints)
        direct_constraints_has_explicit_content = (
            direct_constraints.expected_test_count
            or direct_constraints.expected_test_ids
            or direct_constraints.preserve_collected_test_coverage
            or direct_constraints.protect_visible_test_files
            or direct_constraints.metadata
            or direct_constraints.resolved_evidence_policy().mode != EVIDENCE_MODE_NO_SUITE_VISIBLE
            or not direct_constraints.resolved_test_inventory().is_empty()
        )
        if direct_constraints_has_explicit_content:
            direct_constraints = EvaluationConstraints(
                expected_test_count=direct_constraints.expected_test_count,
                expected_test_ids=list(direct_constraints.expected_test_ids),
                preserve_collected_test_coverage=bool(
                    direct_constraints.preserve_collected_test_coverage
                ),
                protect_visible_test_files=bool(direct_constraints.protect_visible_test_files),
                metadata=dict(direct_constraints.metadata or {}),
                test_inventory=_coerce_test_inventory_from_payload(
                    direct_constraints.test_inventory,
                    expected_test_count=direct_constraints.expected_test_count,
                    expected_test_ids=direct_constraints.expected_test_ids,
                    test_command=payload_test_command,
                    default_source=(
                        str(direct_constraints.test_inventory.source or "").strip().lower()
                        if isinstance(direct_constraints.test_inventory, TestInventory)
                        else ""
                    )
                    or "evaluation_constraints",
                ),
                evidence_policy=direct_constraints.resolved_evidence_policy(),
            )
            return direct_constraints

    payload_expected_test_ids = _dedupe_strings(_list_field(payload, "expected_test_ids"))
    payload_expected_test_count = _int_field(payload, "expected_test_count")
    payload_raw_test_inventory = _nested_field(payload, "test_inventory")
    payload_has_explicit_test_inventory = False
    if isinstance(payload_raw_test_inventory, TestInventory):
        payload_has_explicit_test_inventory = not payload_raw_test_inventory.is_empty()
    elif isinstance(payload_raw_test_inventory, dict):
        payload_has_explicit_test_inventory = bool(payload_raw_test_inventory)
    payload_test_inventory = _coerce_test_inventory_from_payload(
        payload_raw_test_inventory,
        expected_test_count=payload_expected_test_count,
        expected_test_ids=payload_expected_test_ids,
        test_command=payload_test_command,
        default_source="evaluation_constraints",
    )
    payload_preserve_coverage = bool(_nested_field(payload, "preserve_collected_test_coverage"))
    payload_protect_visible_tests = bool(_nested_field(payload, "protect_visible_test_files"))
    if (
        payload_expected_test_ids
        or payload_expected_test_count
        or payload_has_explicit_test_inventory
        or payload_preserve_coverage
        or payload_protect_visible_tests
    ):
        return EvaluationConstraints(
            expected_test_count=int(payload_test_inventory.expected_test_count or 0),
            expected_test_ids=list(payload_test_inventory.expected_test_ids),
            preserve_collected_test_coverage=(
                payload_preserve_coverage or payload_test_inventory.expected_test_count > 0
            ),
            protect_visible_test_files=payload_protect_visible_tests,
            metadata=dict(_nested_field(payload, "metadata") or {}),
            test_inventory=payload_test_inventory,
            evidence_policy=EvidencePolicy.from_dict(
                _nested_field(payload, "evidence_policy") or {}
            ),
        )

    legacy_context = _nested_field(payload, "test_context")
    if legacy_context is None:
        legacy_context = payload
    legacy_expected_test_ids = _dedupe_strings(_list_field(legacy_context, "expected_test_ids"))
    legacy_expected_test_count = _int_field(legacy_context, "expected_test_count")
    legacy_test_inventory = _coerce_test_inventory_from_payload(
        _nested_field(legacy_context, "test_inventory"),
        expected_test_count=legacy_expected_test_count,
        expected_test_ids=legacy_expected_test_ids,
        test_command=_string_field(legacy_context, "command"),
        default_source="legacy_test_context",
    )
    return EvaluationConstraints(
        expected_test_count=int(legacy_test_inventory.expected_test_count or 0),
        expected_test_ids=list(legacy_test_inventory.expected_test_ids),
        preserve_collected_test_coverage=legacy_test_inventory.expected_test_count > 0,
        protect_visible_test_files=bool(
            _nested_field(legacy_context, "protect_visible_test_files")
        ),
        test_inventory=legacy_test_inventory,
        evidence_policy=EvidencePolicy.from_dict(
            _nested_field(legacy_context, "evidence_policy") or {}
        ),
    )


def canonical_test_inventory(payload: Any) -> TestInventory:
    constraints = coerce_evaluation_constraints(payload)
    return constraints.resolved_test_inventory()


def canonical_expected_test_ids(payload: Any) -> list[str]:
    return list(canonical_test_inventory(payload).expected_test_ids)


def canonical_expected_test_count(payload: Any) -> int:
    return int(canonical_test_inventory(payload).expected_test_count or 0)


def visible_test_edit_protection_enabled(payload: Any) -> bool:
    constraints = coerce_evaluation_constraints(payload)
    return bool(constraints.protect_visible_test_files)


def _normalize_inventory_outcome(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "pass": "passed",
        "passed": "passed",
        "success": "passed",
        "ok": "passed",
        "xfail": "xfailed",
        "xfailed": "xfailed",
        "xpass": "xpassed",
        "xpassed": "xpassed",
        "fail": "failed",
        "failed": "failed",
        "failure": "failed",
        "error": "error",
        "errors": "error",
        "skip": "skipped",
        "skipped": "skipped",
    }
    return aliases.get(normalized, normalized)


def summarize_test_inventory_coverage(
    inventory: Any,
    *,
    observed_test_count: Optional[int] = None,
    observed_test_ids: Optional[Iterable[Any]] = None,
    observed_test_outcomes: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    canonical_inventory = _coerce_test_inventory_from_payload(
        inventory,
        expected_test_count=None,
        expected_test_ids=None,
        default_source="coverage_summary",
    )
    expected_count = int(canonical_inventory.expected_test_count or 0)
    if expected_count <= 0:
        return {}

    observed_ids = _dedupe_strings(list(observed_test_ids or []))
    normalized_outcomes = {
        str(test_id).strip(): _normalize_inventory_outcome(outcome)
        for test_id, outcome in dict(observed_test_outcomes or {}).items()
        if str(test_id).strip()
    }
    if not observed_ids and normalized_outcomes:
        observed_ids = _dedupe_strings(list(normalized_outcomes))
    observed_count_value: Optional[int]
    if isinstance(observed_test_count, int) and observed_test_count >= 0:
        observed_count_value = observed_test_count
    elif observed_ids:
        observed_count_value = len(observed_ids)
    elif normalized_outcomes:
        observed_count_value = len(normalized_outcomes)
    else:
        observed_count_value = None

    summary: dict[str, Any] = {
        "expected_test_count": expected_count,
        "test_inventory_framework": canonical_inventory.framework,
        "test_inventory_language": canonical_inventory.language,
        "test_inventory_source": canonical_inventory.source,
    }
    if canonical_inventory.collection_command:
        summary["test_inventory_collection_command"] = canonical_inventory.collection_command
    if canonical_inventory.test_command:
        summary["test_inventory_test_command"] = canonical_inventory.test_command
    if isinstance(observed_count_value, int):
        summary["collected_test_count"] = observed_count_value

    if canonical_inventory.expected_test_ids and normalized_outcomes:
        if canonical_inventory.framework == "pytest":
            from .core.pytest_report_utils import summarize_expected_pytest_coverage

            coverage = summarize_expected_pytest_coverage(
                canonical_inventory.expected_test_ids,
                normalized_outcomes,
            )
            summary.update(
                {
                    "matched_expected_test_count": int(
                        coverage.get("matched_expected_test_count") or 0
                    ),
                    "missing_expected_test_count": int(
                        coverage.get("missing_expected_test_count") or 0
                    ),
                    "missing_test_ids": list(coverage.get("missing_test_ids") or []),
                    "coverage_preserved": int(coverage.get("missing_expected_test_count") or 0)
                    == 0,
                    "passed": int(coverage.get("passed") or 0),
                    "failed": int(coverage.get("failed") or 0),
                    "errors": int(coverage.get("errors") or 0),
                }
            )
            return summary

        observed_id_set = set(observed_ids or normalized_outcomes)
        matched_ids = [
            test_id
            for test_id in canonical_inventory.expected_test_ids
            if test_id in observed_id_set
        ]
        missing_ids = [
            test_id
            for test_id in canonical_inventory.expected_test_ids
            if test_id not in observed_id_set
        ]
        passed = 0
        errors = 0
        for test_id in matched_ids:
            outcome = normalized_outcomes.get(test_id)
            if outcome in {"passed", "xfailed", "xpassed"}:
                passed += 1
            elif outcome == "error":
                errors += 1
        failed = len(canonical_inventory.expected_test_ids) - passed - errors
        summary.update(
            {
                "matched_expected_test_count": len(matched_ids),
                "missing_expected_test_count": len(missing_ids),
                "missing_test_ids": missing_ids,
                "coverage_preserved": len(missing_ids) == 0,
                "passed": passed,
                "failed": max(failed, 0),
                "errors": errors,
            }
        )
        return summary

    if isinstance(observed_count_value, int):
        # No per-id evidence reached this branch (the id-matched path above did
        # not run). The count-derived matched/missing stay (informative — and the
        # only coverage signal a count-only framework has).
        matched = min(observed_count_value, expected_count)
        summary["matched_expected_test_count"] = matched
        summary["missing_expected_test_count"] = max(expected_count - matched, 0)
        if canonical_inventory.expected_test_ids:
            # FIX A — false-accept guard: we KNOW the expected ids but got NO
            # per-id outcomes, so a bare count >= expected does NOT prove the
            # RIGHT ids ran. coverage_preserved is False when the count is short
            # (cannot be a false full-coverage) and UNKNOWN (None) otherwise —
            # NEVER True without per-id proof. This is the latent count-only edge.
            summary["coverage_preserved"] = (
                False if observed_count_value < expected_count else None
            )
        else:
            # Count-only inventory (no expected ids available — e.g. a non-pytest
            # framework that reports only counts): the count IS the coverage
            # contract for such frameworks, so honor the count comparison.
            summary["coverage_preserved"] = observed_count_value >= expected_count
    return summary


class TaskRegimePolicy:
    """Infer generic task regimes from issue, traceback, and visible test evidence."""

    def __init__(self, config: Any, *, model_library: Any = None):
        self.config = config
        self.model_library = model_library
        self.generic_adapter = GenericEvidenceAdapter(config)
        self.python_pytest_adapter = PythonPytestEvidenceAdapter(config)

    def _normalize_language_hints(
        self,
        values: Optional[Iterable[Any]],
    ) -> list[str]:
        return [
            str(value).strip().lower()
            for value in _dedupe_strings(list(values or []))
            if str(value).strip()
        ]

    def _select_adapter(
        self,
        *,
        failing_test_ids: Optional[Iterable[Any]] = None,
        relevant_file_languages: Optional[Iterable[Any]] = None,
        repo_languages: Optional[Iterable[Any]] = None,
        test_command: Optional[str] = None,
    ) -> Any:
        failing_ids = _dedupe_strings(list(failing_test_ids or []))
        relevant_languages = self._normalize_language_hints(relevant_file_languages)
        repo_language_hints = self._normalize_language_hints(repo_languages)
        lowered_test_command = str(test_command or "").strip().lower()
        python_hint = (
            "pytest" in lowered_test_command
            or any(language == "python" for language in relevant_languages)
            or (
                not relevant_languages
                and any(language == "python" for language in repo_language_hints)
            )
            or any(test_id.endswith(".py") or ".py::" in test_id for test_id in failing_ids)
        )
        if python_hint:
            return self.python_pytest_adapter
        return self.generic_adapter

    def threshold(self, state: str, default: float = 0.5) -> float:
        thresholds = getattr(self.config, "probability_thresholds", {})
        # `or default` would silently override a configured 0.0 threshold.
        # Use explicit None-check so legitimately-zero thresholds round-trip.
        value = dict(thresholds or {}).get(state)
        if value is None:
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _state_feature_view(
        self,
        *,
        state: str,
        feature_map: Optional[dict[str, float]],
        baseline_probability: float,
        evidence_count: int,
    ) -> dict[str, float]:
        features = {
            str(name): float(value)
            for name, value in dict(feature_map or {}).items()
            if isinstance(value, (int, float))
        }
        features["heuristic_score"] = float(baseline_probability or 0.0)
        features["evidence_count"] = float(evidence_count or 0)
        features[f"state__{state}"] = float(features.get(f"state__{state}", 0.0) or 0.0)
        features[f"model_target__{state}"] = 1.0
        return features

    def infer_from_observations(
        self,
        *,
        observations: Iterable[Any],
        feature_map: Optional[dict[str, float]] = None,
        preserve_collected_test_coverage: bool = False,
        adapter_name: str = "",
    ) -> TaskRegimeProfile:
        probabilities = {state: 0.0 for state in TASK_REGIME_STATES}
        evidence: list[TaskRegimeEvidence] = []
        weights = dict(getattr(self.config, "evidence_weights", {}) or {})
        scales = dict(getattr(self.config, "state_scales", {}) or {})

        for raw_observation in list(observations or []):
            state = str(getattr(raw_observation, "state", "") or "").strip()
            source = str(getattr(raw_observation, "source", "") or "").strip()
            if state not in TASK_REGIME_STATES or not source:
                continue
            strength = max(0.0, float(getattr(raw_observation, "strength", 0.0) or 0.0))
            weight = float(weights.get(source, 0.0) or 0.0) * max(strength, 0.0)
            if weight <= 0.0:
                continue
            probabilities[state] = probabilities.get(state, 0.0) + weight
            evidence.append(
                TaskRegimeEvidence(
                    state=state,
                    source=source,
                    weight=weight,
                    rationale=str(getattr(raw_observation, "rationale", "") or ""),
                    file_paths=_dedupe_strings(
                        list(getattr(raw_observation, "file_paths", []) or [])
                    ),
                    test_ids=_dedupe_strings(list(getattr(raw_observation, "test_ids", []) or [])),
                    symbols=_dedupe_strings(list(getattr(raw_observation, "symbols", []) or [])),
                    metadata={
                        **dict(getattr(raw_observation, "metadata", {}) or {}),
                        "strength": round(float(strength), 4),
                    },
                )
            )

        normalized_probabilities = {
            state: round(
                _clamp(
                    float(probabilities.get(state) or 0.0)
                    / max(float(scales.get(state, 1.0) or 1.0), 0.01)
                ),
                4,
            )
            for state in TASK_REGIME_STATES
        }
        for state in TASK_REGIME_STATES:
            evaluation = evaluate_policy_model(
                self.model_library,
                model_name=f"regime.{state}",
                features=self._state_feature_view(
                    state=state,
                    feature_map=feature_map,
                    baseline_probability=normalized_probabilities[state],
                    evidence_count=len([item for item in evidence if item.state == state]),
                ),
                baseline_value=normalized_probabilities[state],
                lower=0.0,
                upper=1.0,
            )
            normalized_probabilities[state] = round(_clamp(float(evaluation.value or 0.0)), 4)

        invariants: list[str] = []
        if preserve_collected_test_coverage:
            invariants.append("Do not shrink collected test coverage under the main test command.")
        if normalized_probabilities["importability_blocker"] >= self.threshold(
            "importability_blocker"
        ):
            invariants.append(
                "Clear the currently observed import or collection blocker before broadening the patch."
            )
        if normalized_probabilities["contract_gap"] >= self.threshold("contract_gap"):
            invariants.append("Prefer completing missing behavior over weakening visible tests.")
        if normalized_probabilities["high_interface_risk"] >= self.threshold("high_interface_risk"):
            invariants.append(
                "Preserve shared interface behavior across touched modules unless visible evidence disproves it."
            )

        ordered_states = [
            state
            for state, probability in sorted(
                normalized_probabilities.items(),
                key=lambda item: (float(item[1]), item[0]),
                reverse=True,
            )
            if float(probability) >= 0.25
        ]
        if ordered_states:
            summary = (
                "Controller regimes: "
                + ", ".join(
                    f"{state}={normalized_probabilities[state]:.2f}" for state in ordered_states[:3]
                )
                + "."
            )
        else:
            summary = (
                "Controller regimes remain low-confidence; continue with evidence-driven search."
            )
        return TaskRegimeProfile(
            state_probabilities=normalized_probabilities,
            evidence=evidence,
            planner_invariants=_dedupe_strings(invariants),
            summary=summary,
            adapter_name=str(adapter_name or ""),
        )

    def infer(
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
    ) -> TaskRegimeProfile:
        adapter = self._select_adapter(
            failing_test_ids=failing_test_ids,
            relevant_file_languages=relevant_file_languages,
            repo_languages=repo_languages,
            test_command=test_command,
        )
        adapter_result: EvidenceAdapterResult = adapter.build(
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
        return self.infer_from_observations(
            observations=adapter_result.observations,
            feature_map=adapter_result.feature_map,
            preserve_collected_test_coverage=preserve_collected_test_coverage,
            adapter_name=str(adapter_result.metadata.get("adapter") or ""),
        )
