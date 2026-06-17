"""Acceptance and progressive-signal helpers extracted from the
monolithic orchestrator.

Phase 3.2: pure functions whose behavior is determined by the
``RolloutResult`` payload alone (no orchestrator state). These were
previously methods on ``ApexOrchestrator`` even though they did not
read ``self``; pulling them out makes them re-usable from the V5
in-container agent path and from tests without standing up an
orchestrator instance.
"""

from __future__ import annotations

from typing import Optional

from ..acceptance import (
    quick_verification_expected_coverage_ratio,
    quick_verification_has_local_full_scope_pass,
    quick_verification_has_strong_signal,
    rollout_has_authoritative_acceptance,
    rollout_has_submission_blocking_validity,
    verification_has_explicit_validity_rejection,
)
from ..rollout.engine import RolloutResult


def selected_result_is_accepted(result: Optional[RolloutResult]) -> bool:
    """Phase 2C 2.2: STRICT acceptance gate.

    Acceptance requires either:
      * ``verification.accepted == True`` (authoritative verifier signal), OR
      * ``quick_verification`` has a strong full-scope signal
        (``require_full_scope=True`` floor).

    The legacy ``overall_score >= 0.9`` short-circuit is REMOVED — a
    soft heuristic score is no longer sufficient to mark a candidate
    as solved.
    """
    if result is None or not result.patch:
        return False
    verification = result.verification
    if verification_has_explicit_validity_rejection(verification):
        return False
    if rollout_has_submission_blocking_validity(result):
        return False
    if isinstance(verification, dict) and "accepted" in verification:
        return bool(verification["accepted"])
    quick_verification = (
        result.quick_verification if isinstance(result.quick_verification, dict) else {}
    )
    if quick_verification_has_strong_signal(
        quick_verification,
        require_full_scope=True,
    ):
        return True
    return False


def rollout_has_authoritative_completion_signal(
    result: Optional[RolloutResult],
) -> bool:
    return rollout_has_authoritative_acceptance(result)


def rollout_has_strong_progressive_signal(
    result: Optional[RolloutResult],
) -> bool:
    if result is None or not result.success or not result.patch:
        return False

    quick_verification = (
        result.quick_verification if isinstance(result.quick_verification, dict) else {}
    )
    if quick_verification_has_local_full_scope_pass(quick_verification):
        return True
    if quick_verification_has_strong_signal(quick_verification):
        return True

    verification = result.verification if isinstance(result.verification, dict) else {}
    if verification.get("accepted") is True:
        return True

    return False


def rollout_has_local_full_suite_completion_signal(
    result: Optional[RolloutResult],
) -> bool:
    if result is None or not result.patch:
        return False
    quick_verification = (
        result.quick_verification if isinstance(result.quick_verification, dict) else {}
    )
    return quick_verification_has_local_full_scope_pass(quick_verification)


def rollout_has_expected_coverage_gap(
    result: Optional[RolloutResult],
) -> bool:
    if result is None or not result.patch:
        return False
    quick_verification = (
        result.quick_verification if isinstance(result.quick_verification, dict) else {}
    )
    missing_expected_test_count = quick_verification.get("missing_expected_test_count")
    if isinstance(missing_expected_test_count, int) and missing_expected_test_count > 0:
        return True
    if quick_verification.get("coverage_preserved") is False:
        return True
    expected_coverage_ratio = quick_verification_expected_coverage_ratio(quick_verification)
    return (
        isinstance(expected_coverage_ratio, (int, float)) and float(expected_coverage_ratio) < 0.999
    )


__all__ = [
    "selected_result_is_accepted",
    "rollout_has_authoritative_completion_signal",
    "rollout_has_strong_progressive_signal",
    "rollout_has_local_full_suite_completion_signal",
    "rollout_has_expected_coverage_gap",
]
