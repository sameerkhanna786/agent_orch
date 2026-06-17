"""Structured evidence accounting for candidate selection.

The ledger is intentionally benchmark-agnostic: it does not know what a
repository, language, or harness is. Callers attach generic evidence tags and
the selector can prefer candidates whose support is structured, fresh, scored,
and produced inside the controlled runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _clamp01(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(float(value), 1.0))
    return default


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _counts_signal(payload: dict[str, Any]) -> bool:
    return any(
        isinstance(payload.get(key), (int, float))
        for key in ("passed", "failed", "errors", "pass_rate", "expected_test_count")
    )


@dataclass(frozen=True)
class EvidenceSignal:
    """One selector-consumable evidence item."""

    name: str
    source: str
    structured: bool = False
    fresh: bool = False
    scored: bool = False
    in_container: bool = False
    benchmark_owned: bool = False
    agent_produced: bool = False
    trusted: bool = False
    noisy: bool = False
    confidence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        score = 0.12
        score += 0.14 if self.structured else 0.0
        score += 0.10 if self.fresh else 0.0
        score += 0.13 if self.scored else 0.0
        score += 0.10 if self.in_container else 0.0
        score += 0.11 if self.benchmark_owned else 0.0
        score += 0.18 if self.trusted else 0.0
        score += 0.12 * _clamp01(self.confidence)
        score -= 0.18 if self.agent_produced and not self.benchmark_owned else 0.0
        score -= 0.26 if self.noisy else 0.0
        return max(0.0, min(score, 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "structured": bool(self.structured),
            "fresh": bool(self.fresh),
            "scored": bool(self.scored),
            "in_container": bool(self.in_container),
            "benchmark_owned": bool(self.benchmark_owned),
            "agent_produced": bool(self.agent_produced),
            "trusted": bool(self.trusted),
            "noisy": bool(self.noisy),
            "confidence": round(_clamp01(self.confidence), 4),
            "score": round(self.score, 4),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class EvidenceLedger:
    """Aggregate evidence view for one candidate."""

    signals: list[EvidenceSignal] = field(default_factory=list)

    @property
    def score(self) -> float:
        if not self.signals:
            return 0.5
        weighted = 0.0
        total = 0.0
        for signal in self.signals:
            weight = 1.35 if signal.trusted else 1.0
            if signal.noisy:
                weight = 0.75
            weighted += signal.score * weight
            total += weight
        return max(0.0, min(weighted / max(total, 1e-9), 1.0))

    @property
    def trusted_signal_count(self) -> int:
        return sum(1 for signal in self.signals if signal.trusted)

    @property
    def noisy_signal_count(self) -> int:
        return sum(1 for signal in self.signals if signal.noisy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "trusted_signal_count": self.trusted_signal_count,
            "noisy_signal_count": self.noisy_signal_count,
            "signals": [signal.to_dict() for signal in self.signals],
        }


def _quick_signal_score(quick: dict[str, Any]) -> float:
    explicit = _clamp01(quick.get("signal_score"), default=-1.0)
    if explicit >= 0.0:
        return explicit
    pass_rate = _clamp01(quick.get("pass_rate"), default=-1.0)
    if pass_rate >= 0.0:
        return pass_rate
    passed = quick.get("passed")
    failed = quick.get("failed")
    errors = quick.get("errors")
    if all(isinstance(value, (int, float)) for value in (passed, failed, errors)):
        total = float(passed or 0) + float(failed or 0) + float(errors or 0)
        if total > 0:
            return max(0.0, min(float(passed or 0) / total, 1.0))
    return 0.0


def _quick_evidence_is_in_container(candidate: Any, quick: dict[str, Any]) -> bool:
    if bool(quick.get("in_container")) or bool(quick.get("target_runtime")):
        return True
    metadata = _as_dict(getattr(candidate, "search_metadata", None))
    if bool(metadata.get("target_runtime")) or bool(metadata.get("containerized_runtime")):
        return True
    runtime = str(quick.get("runtime") or metadata.get("runtime") or "").lower()
    return any(token in runtime for token in ("container", "docker", "target_runtime"))


def build_candidate_evidence_ledger(
    candidate: Any,
    *,
    verification: Any = None,
) -> EvidenceLedger:
    """Build a normalized evidence ledger from a rollout and verification."""

    signals: list[EvidenceSignal] = []
    quick = _as_dict(getattr(candidate, "quick_verification", None))
    if quick:
        has_counts = _counts_signal(quick)
        score = _quick_signal_score(quick)
        failed = int(quick.get("failed") or 0) if isinstance(quick.get("failed"), int) else 0
        errors = int(quick.get("errors") or 0) if isinstance(quick.get("errors"), int) else 0
        timed_out = bool(quick.get("timed_out") or quick.get("full_scope_timed_out"))
        structured_report = bool(
            quick.get("json_report")
            or quick.get("json_report_path")
            or quick.get("pytest_json_report")
            or quick.get("pytest_report")
        )
        signals.append(
            EvidenceSignal(
                name="rollout_quick_verification",
                source="rollout_harness",
                structured=has_counts or structured_report,
                fresh=True,
                scored=has_counts or isinstance(quick.get("signal_score"), (int, float)),
                in_container=_quick_evidence_is_in_container(candidate, quick),
                benchmark_owned=bool(
                    quick.get("expected_test_count")
                    or quick.get("expected_coverage_preserved") is not None
                    or quick.get("scope") == "full_test_command"
                ),
                agent_produced=False,
                trusted=bool(
                    (has_counts or structured_report)
                    and not timed_out
                    and failed == 0
                    and errors == 0
                    and _clamp01(score) >= 0.98
                ),
                noisy=bool(timed_out or (not has_counts and not structured_report)),
                confidence=score,
                details={
                    "scope": quick.get("scope"),
                    "returncode": quick.get("returncode"),
                    "passed": quick.get("passed"),
                    "failed": quick.get("failed"),
                    "errors": quick.get("errors"),
                    "structured_report": structured_report,
                },
            )
        )

    if verification is not None:
        test_result = getattr(verification, "test_result", None)
        accepted = bool(getattr(verification, "accepted", False))
        overall_score = _clamp01(getattr(verification, "overall_score", 0.0))
        if test_result is not None:
            pass_rate = _clamp01(getattr(test_result, "pass_rate", 0.0))
            coverage_preserved = getattr(test_result, "expected_coverage_preserved", None)
            failed = int(getattr(test_result, "failed", 0) or 0)
            errors = int(getattr(test_result, "errors", 0) or 0)
            expected_count = int(getattr(test_result, "expected_test_count", 0) or 0)
            signals.append(
                EvidenceSignal(
                    name="selector_verification",
                    source="selector_verifier",
                    structured=True,
                    fresh=True,
                    scored=True,
                    in_container=False,
                    benchmark_owned=bool(expected_count or coverage_preserved is not None),
                    agent_produced=False,
                    trusted=bool(accepted and coverage_preserved is not False and failed == 0 and errors == 0),
                    noisy=bool(getattr(test_result, "regression_inconclusive", False)),
                    confidence=max(pass_rate, overall_score),
                    details={
                        "accepted": accepted,
                        "passed": getattr(test_result, "passed", None),
                        "failed": failed,
                        "errors": errors,
                        "expected_coverage_preserved": coverage_preserved,
                        "expected_test_count": expected_count,
                    },
                )
            )
        else:
            signals.append(
                EvidenceSignal(
                    name="selector_static_verification",
                    source="selector_verifier",
                    structured=True,
                    fresh=True,
                    scored=True,
                    agent_produced=False,
                    trusted=accepted,
                    noisy=not accepted,
                    confidence=overall_score,
                    details={"accepted": accepted},
                )
            )

    validity = getattr(candidate, "validity", None)
    if validity is not None:
        as_dict = validity.as_dict() if hasattr(validity, "as_dict") else _as_dict(validity)
        eligible = bool(as_dict.get("eligible_for_submission"))
        signals.append(
            EvidenceSignal(
                name="candidate_validity_contract",
                source="validity_gate",
                structured=True,
                fresh=True,
                scored=True,
                benchmark_owned=True,
                trusted=eligible,
                noisy=not eligible,
                confidence=1.0 if eligible else 0.0,
                details=as_dict,
            )
        )

    patch_artifact = _as_dict(getattr(candidate, "patch_artifact", None))
    confidence = _clamp01(patch_artifact.get("confidence"), default=0.0)
    if patch_artifact or getattr(candidate, "patch", None):
        signals.append(
            EvidenceSignal(
                name="agent_patch_claim",
                source="agent_submission",
                structured=bool(patch_artifact),
                fresh=True,
                scored=isinstance(patch_artifact.get("confidence"), (int, float)),
                agent_produced=True,
                trusted=False,
                noisy=confidence <= 0.0,
                confidence=confidence,
                details={
                    "changed_file_count": len(list(getattr(candidate, "changed_files", []) or [])),
                    "has_patch": bool(getattr(candidate, "patch", None)),
                },
            )
        )

    return EvidenceLedger(signals=signals)
