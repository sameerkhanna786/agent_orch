"""First-class verification outcome taxonomy.

The labels describe what kind of repair path should run next. They are generic
on purpose: benchmark adapters may provide more concrete signals, but selector
policy should route on these broad classes instead of parsing arbitrary text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class VerificationFailureKind(str, Enum):
    ACCEPTED = "accepted"
    TRUE_CODE_FAILURE = "true_code_failure"
    COLLECTION_FAILURE = "collection_failure"
    HARNESS_CONFIG_FAILURE = "harness_config_failure"
    TIMEOUT_INCONCLUSIVE = "timeout_inconclusive"
    PROTECTED_TEST_VIOLATION = "protected_test_violation"
    COVERAGE_COLLAPSE = "coverage_collapse"
    SCORER_DISAGREEMENT = "scorer_disagreement"
    QUALITY_GATE_REJECTION = "quality_gate_rejection"
    SYNTAX_FAILURE = "syntax_failure"
    ENVIRONMENT_FAILURE = "environment_failure"
    UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class VerificationTaxonomy:
    kind: VerificationFailureKind
    repair_policy: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "repair_policy": self.repair_policy,
            "reasons": list(self.reasons),
        }


_REPAIR_POLICY_BY_KIND = {
    VerificationFailureKind.ACCEPTED: "accept",
    VerificationFailureKind.TRUE_CODE_FAILURE: "repair_code_against_failing_tests",
    VerificationFailureKind.COLLECTION_FAILURE: "repair_collection_or_import_contract",
    VerificationFailureKind.HARNESS_CONFIG_FAILURE: "escalate_harness_configuration",
    VerificationFailureKind.TIMEOUT_INCONCLUSIVE: "retry_or_shard_verification_before_acceptance",
    VerificationFailureKind.PROTECTED_TEST_VIOLATION: "reject_and_restore_protected_files",
    VerificationFailureKind.COVERAGE_COLLAPSE: "reject_and_restore_test_inventory",
    VerificationFailureKind.SCORER_DISAGREEMENT: "abstain_until_scorers_reconcile",
    VerificationFailureKind.QUALITY_GATE_REJECTION: "repair_validity_quality_gate",
    VerificationFailureKind.SYNTAX_FAILURE: "repair_parse_or_lint_failure",
    VerificationFailureKind.ENVIRONMENT_FAILURE: "retry_or_switch_runtime",
    VerificationFailureKind.UNCLASSIFIED: "collect_more_evidence",
}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text_contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _quick_failure_label(quick: dict[str, Any]) -> str:
    classification = _as_dict(quick.get("failure_classification"))
    return str(classification.get("label") or classification.get("failure_class") or "").lower()


def _make(kind: VerificationFailureKind, reasons: list[str]) -> VerificationTaxonomy:
    return VerificationTaxonomy(
        kind=kind,
        repair_policy=_REPAIR_POLICY_BY_KIND[kind],
        reasons=list(dict.fromkeys(reason for reason in reasons if reason)),
    )


def classify_candidate_verification(
    candidate: Any,
    verification: Any = None,
) -> VerificationTaxonomy:
    reasons: list[str] = []
    if verification is not None and bool(getattr(verification, "accepted", False)):
        return _make(VerificationFailureKind.ACCEPTED, ["verification accepted"])

    validity = getattr(candidate, "validity", None)
    validity_payload = validity.as_dict() if hasattr(validity, "as_dict") else _as_dict(validity)
    validity_reasons = [
        str(item) for item in list(validity_payload.get("reasons") or []) if str(item).strip()
    ]
    reasons.extend(validity_reasons)
    verification_reasons = [
        str(item)
        for item in list(getattr(verification, "validity_reasons", []) or [])
        if str(item).strip()
    ]
    reasons.extend(verification_reasons)
    reason_text = "\n".join(reasons)

    if validity_payload.get("protected_tests_unchanged") is False or _text_contains_any(
        reason_text,
        ("protected test", "protected-test", "visible test edit"),
    ):
        return _make(VerificationFailureKind.PROTECTED_TEST_VIOLATION, reasons)
    if (
        validity_payload.get("coverage_collapse_terminal")
        or validity_payload.get("expected_coverage_preserved") is False
    ):
        return _make(VerificationFailureKind.COVERAGE_COLLAPSE, reasons)
    test_result = getattr(verification, "test_result", None) if verification is not None else None
    if getattr(test_result, "expected_coverage_preserved", None) is False:
        return _make(
            VerificationFailureKind.COVERAGE_COLLAPSE,
            reasons or ["expected coverage was not preserved"],
        )
    if verification is not None and getattr(verification, "quality_gate_passed", None) is False:
        return _make(
            VerificationFailureKind.QUALITY_GATE_REJECTION,
            reasons or ["quality gate rejected candidate"],
        )
    if verification is not None and (
        not bool(getattr(verification, "syntax_valid", True))
        or not bool(getattr(verification, "lint_clean", True))
    ):
        return _make(
            VerificationFailureKind.SYNTAX_FAILURE, reasons or ["syntax or lint validation failed"]
        )

    quick = _as_dict(getattr(candidate, "quick_verification", None))
    quick_failure_clusters = "\n".join(
        str(item) for item in list(quick.get("failure_clusters") or [])
    )
    quick_classification = _as_dict(quick.get("failure_classification"))
    output = "\n".join(
        str(value or "")
        for value in (
            quick.get("output"),
            quick.get("stderr"),
            quick.get("output_excerpt"),
            quick_failure_clusters,
            quick_classification.get("primary_signal"),
            getattr(test_result, "reproduction_output", ""),
            getattr(test_result, "regression_output", ""),
            reason_text,
        )
    )
    quick_label = _quick_failure_label(quick)
    if (
        bool(quick.get("timed_out") or quick.get("full_scope_timed_out"))
        or bool(getattr(test_result, "regression_inconclusive", False))
        or quick_label == "timeout"
    ):
        return _make(
            VerificationFailureKind.TIMEOUT_INCONCLUSIVE,
            reasons or ["verification timed out or was inconclusive"],
        )
    if quick_label == "syntax":
        return _make(
            VerificationFailureKind.SYNTAX_FAILURE,
            reasons or ["quick verification reported syntax failure"],
        )
    if quick_label == "collection":
        return _make(
            VerificationFailureKind.COLLECTION_FAILURE,
            reasons or ["quick verification reported collection failure"],
        )
    if quick_label == "env":
        return _make(
            VerificationFailureKind.ENVIRONMENT_FAILURE,
            reasons or ["quick verification reported environment failure"],
        )
    if _text_contains_any(
        output,
        (
            "scorer disagreement",
            "private scorer",
            "official scorer",
            "audit disagreement",
        ),
    ):
        return _make(
            VerificationFailureKind.SCORER_DISAGREEMENT, reasons or ["scorer disagreement"]
        )
    if _text_contains_any(
        output,
        (
            "requests.exceptions.httperror",
            "requests.exceptions.connectionerror",
            "requests.exceptions.timeout",
            "rate limit exceeded",
            "could not resolve host",
            "temporary failure in name resolution",
            "network is unreachable",
            "connection refused",
            "connection reset",
            "connection aborted",
            "max retries exceeded with url",
            "nameresolutionerror",
            "remotedisconnected",
        ),
    ):
        return _make(
            VerificationFailureKind.ENVIRONMENT_FAILURE,
            reasons or ["external service or network failure"],
        )
    if _text_contains_any(
        output,
        (
            "no such file or directory",
            "filenotfounderror",
            "submodule is not available",
            "please import the submodule",
            "command not found",
            "could not find a version",
            "failed to build",
            "dependency",
            "permission denied",
            "not logged in",
            "authentication",
        ),
    ):
        return _make(
            VerificationFailureKind.HARNESS_CONFIG_FAILURE,
            reasons or ["harness or runtime setup failed"],
        )
    if _text_contains_any(
        output,
        (
            "collected 0 items",
            "collection error",
            "error during collection",
            "importerror while importing",
            "importerror while loading conftest",
            "module not found",
        ),
    ):
        return _make(
            VerificationFailureKind.COLLECTION_FAILURE, reasons or ["test collection failed"]
        )
    failure_class = str(
        getattr(test_result, "failure_class", "")
        or _as_dict(getattr(test_result, "failure_classification", None)).get("failure_class")
        or ""
    ).lower()
    if any(token in failure_class for token in ("env", "network", "resource")):
        return _make(VerificationFailureKind.ENVIRONMENT_FAILURE, reasons or [failure_class])

    failed = _safe_int(getattr(test_result, "failed", 0) or quick.get("failed") or 0)
    errors = _safe_int(getattr(test_result, "errors", 0) or quick.get("errors") or 0)
    if failed > 0 or errors > 0:
        return _make(VerificationFailureKind.TRUE_CODE_FAILURE, reasons or ["tests failed"])

    return _make(
        VerificationFailureKind.UNCLASSIFIED, reasons or ["no decisive verification class"]
    )
