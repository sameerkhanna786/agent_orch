"""Benchmark-agnostic evaluation contracts.

The contract layer separates the score-bearing universe from diagnostic
checks. Benchmark adapters can then expose extra runner output without letting
it accidentally override the configured metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class EvaluationMode(str, Enum):
    GOLD_SUITE_VISIBLE = "gold_suite_visible"
    PARTIAL_SUITE_VISIBLE = "partial_suite_visible"
    HIDDEN_SUITE_AUTHORITATIVE = "hidden_suite_authoritative"
    GENERATED_ORACLE = "generated_oracle"
    CUSTOM = "custom"


class RawReturncodePolicy(str, Enum):
    SCORE_BEARING = "score_bearing"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    DIAGNOSTIC_ONLY_WHEN_SCORING_FILTERED = "diagnostic_only_when_scoring_filtered"
    HARNESS_HEALTH = "harness_health"


class ExtraResultPolicy(str, Enum):
    SCORE_BEARING = "score_bearing"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    WARNING = "warning"
    IGNORE = "ignore"


class TimeoutPolicy(str, Enum):
    TASK_FATAL = "task_fatal"
    EVALUATOR_FATAL = "evaluator_fatal"
    DIAGNOSTIC = "diagnostic"
    ATTEMPT_ANYWAY = "attempt_anyway"


class EnvironmentFailurePolicy(str, Enum):
    RETRY = "retry"
    FALLBACK_RUNTIME = "fallback_runtime"
    DIAGNOSTIC = "diagnostic"
    FAIL = "fail"


class RunnerHealth(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SIGNAL = "signal"
    PARSER_ERROR = "parser_error"
    COLLECTION_BROKEN = "collection_broken"
    ENVIRONMENT_FAILURE = "environment_failure"
    HARNESS_FAILURE = "harness_failure"


class EvaluationDecisionKind(str, Enum):
    SOLVED = "solved"
    UNSOLVED = "unsolved"
    INDETERMINATE = "indeterminate"
    INVALID_CANDIDATE = "invalid_candidate"
    HARNESS_FAILURE = "harness_failure"


@dataclass(frozen=False)
class CandidateValidity:
    has_patch: bool
    worktree_materialized: bool
    expected_coverage_preserved: Optional[bool]
    missing_expected_test_count: int
    protected_tests_unchanged: bool
    collection_critical_files_unchanged: bool
    quick_verification_passed: bool
    quality_gate_passed: Optional[bool]
    backend_protocol_error: bool
    coverage_collapse_terminal: bool
    provenance_violation: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def eligible_for_external_scoring(self) -> bool:
        return (
            self.has_patch
            and self.worktree_materialized
            and not self.backend_protocol_error
            and not self.coverage_collapse_terminal
            and not self.provenance_violation
        )

    @property
    def eligible_for_submission(self) -> bool:
        if not self.eligible_for_external_scoring:
            return False
        if not self.quick_verification_passed:
            return False
        if self.expected_coverage_preserved is False:
            return False
        if self.missing_expected_test_count > 0:
            return False
        if self.quality_gate_passed is False:
            return False
        if not self.protected_tests_unchanged:
            return False
        if not self.collection_critical_files_unchanged:
            return False
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "has_patch": bool(self.has_patch),
            "worktree_materialized": bool(self.worktree_materialized),
            "expected_coverage_preserved": self.expected_coverage_preserved,
            "missing_expected_test_count": int(self.missing_expected_test_count),
            "protected_tests_unchanged": bool(self.protected_tests_unchanged),
            "collection_critical_files_unchanged": bool(self.collection_critical_files_unchanged),
            "quick_verification_passed": bool(self.quick_verification_passed),
            "quality_gate_passed": self.quality_gate_passed,
            "backend_protocol_error": bool(self.backend_protocol_error),
            "coverage_collapse_terminal": bool(self.coverage_collapse_terminal),
            "provenance_violation": bool(self.provenance_violation),
            "eligible_for_external_scoring": bool(self.eligible_for_external_scoring),
            "eligible_for_submission": bool(self.eligible_for_submission),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class EvaluationContract:
    mode: EvaluationMode = EvaluationMode.CUSTOM
    scoring_universe: str = "runner_summary"
    diagnostic_universes: tuple[str, ...] = ()
    required_coverage: str = "unknown"
    raw_returncode_policy: RawReturncodePolicy = RawReturncodePolicy.SCORE_BEARING
    extra_result_policy: ExtraResultPolicy = ExtraResultPolicy.DIAGNOSTIC_ONLY
    timeout_policy: TimeoutPolicy = TimeoutPolicy.EVALUATOR_FATAL
    environment_failure_policy: EnvironmentFailurePolicy = EnvironmentFailurePolicy.RETRY

    @classmethod
    def commit0_expected_ids(cls) -> "EvaluationContract":
        return cls(
            mode=EvaluationMode.GOLD_SUITE_VISIBLE,
            scoring_universe="commit0_test_ids",
            diagnostic_universes=("pytest_extra_non_scored_tests", "raw_pytest_returncode"),
            required_coverage="complete",
            raw_returncode_policy=(RawReturncodePolicy.DIAGNOSTIC_ONLY_WHEN_SCORING_FILTERED),
            extra_result_policy=ExtraResultPolicy.DIAGNOSTIC_ONLY,
            timeout_policy=TimeoutPolicy.ATTEMPT_ANYWAY,
            environment_failure_policy=EnvironmentFailurePolicy.FALLBACK_RUNTIME,
        )

    @classmethod
    def full_runner_summary(cls) -> "EvaluationContract":
        return cls(
            mode=EvaluationMode.CUSTOM,
            scoring_universe="runner_summary",
            diagnostic_universes=(),
            required_coverage="unknown",
            raw_returncode_policy=RawReturncodePolicy.SCORE_BEARING,
            extra_result_policy=ExtraResultPolicy.SCORE_BEARING,
        )

    @classmethod
    def swebench_authoritative(cls) -> "EvaluationContract":
        return cls(
            mode=EvaluationMode.HIDDEN_SUITE_AUTHORITATIVE,
            scoring_universe="official_harness",
            diagnostic_universes=("public_tests", "generated_tests"),
            required_coverage="unknown",
            raw_returncode_policy=RawReturncodePolicy.SCORE_BEARING,
            extra_result_policy=ExtraResultPolicy.DIAGNOSTIC_ONLY,
            timeout_policy=TimeoutPolicy.EVALUATOR_FATAL,
            environment_failure_policy=EnvironmentFailurePolicy.RETRY,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "scoring_universe": self.scoring_universe,
            "diagnostic_universes": list(self.diagnostic_universes),
            "required_coverage": self.required_coverage,
            "raw_returncode_policy": self.raw_returncode_policy.value,
            "extra_result_policy": self.extra_result_policy.value,
            "timeout_policy": self.timeout_policy.value,
            "environment_failure_policy": self.environment_failure_policy.value,
        }


@dataclass
class EvaluationDecision:
    kind: EvaluationDecisionKind
    is_success: bool
    is_candidate_viable: bool
    requires_followup: bool
    requires_audit: bool = False
    failure_class: str = ""
    confidence: float = 0.0
    reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "is_success": bool(self.is_success),
            "is_candidate_viable": bool(self.is_candidate_viable),
            "requires_followup": bool(self.requires_followup),
            "requires_audit": bool(self.requires_audit),
            "failure_class": self.failure_class,
            "confidence": float(self.confidence),
            "reason": self.reason,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class ScoredCounts:
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    total: int = 0
    missing: int = 0

    @property
    def runnable(self) -> int:
        return max(0, int(self.passed) + int(self.failed) + int(self.errors))

    @property
    def pass_rate(self) -> float:
        runnable = self.runnable
        return 0.0 if runnable <= 0 else float(self.passed) / float(runnable)


def decide_evaluation(
    *,
    contract: EvaluationContract,
    scored: ScoredCounts,
    raw_returncode: int,
    runner_health: RunnerHealth = RunnerHealth.SUCCESS,
    diagnostics: Optional[dict[str, Any]] = None,
) -> EvaluationDecision:
    """Return the benchmark decision implied by ``contract`` and counts."""

    diag = dict(diagnostics or {})
    if runner_health in {
        RunnerHealth.HARNESS_FAILURE,
        RunnerHealth.PARSER_ERROR,
        RunnerHealth.ENVIRONMENT_FAILURE,
    }:
        return EvaluationDecision(
            kind=EvaluationDecisionKind.HARNESS_FAILURE,
            is_success=False,
            is_candidate_viable=False,
            requires_followup=False,
            failure_class=runner_health.value,
            confidence=1.0,
            reason=f"runner health is {runner_health.value}",
            diagnostics=diag,
        )

    score_passed = (
        int(scored.total) > 0
        and int(scored.failed) == 0
        and int(scored.errors) == 0
        and int(scored.missing) == 0
        and scored.pass_rate >= 1.0
    )
    raw_rc_score_bearing = contract.raw_returncode_policy == RawReturncodePolicy.SCORE_BEARING
    if (
        contract.raw_returncode_policy == RawReturncodePolicy.DIAGNOSTIC_ONLY_WHEN_SCORING_FILTERED
        and contract.scoring_universe in {"runner_summary", "full_suite"}
    ):
        raw_rc_score_bearing = True
    if raw_rc_score_bearing and int(raw_returncode) != 0:
        score_passed = False
        diag.setdefault("raw_returncode_score_bearing", int(raw_returncode))

    if score_passed:
        return EvaluationDecision(
            kind=EvaluationDecisionKind.SOLVED,
            is_success=True,
            is_candidate_viable=True,
            requires_followup=False,
            requires_audit=True,
            confidence=1.0,
            reason=f"{contract.scoring_universe} passed",
            diagnostics=diag,
        )

    failure_class = "scoring_failure"
    if scored.missing:
        failure_class = "missing_scored_tests"
    elif scored.errors:
        failure_class = "scored_test_errors"
    elif scored.failed:
        failure_class = "scored_test_failures"
    elif runner_health == RunnerHealth.TIMEOUT:
        failure_class = "timeout"

    return EvaluationDecision(
        kind=EvaluationDecisionKind.UNSOLVED,
        is_success=False,
        is_candidate_viable=(scored.runnable > 0),
        requires_followup=True,
        failure_class=failure_class,
        confidence=0.85,
        reason=f"{contract.scoring_universe} did not pass",
        diagnostics=diag,
    )
