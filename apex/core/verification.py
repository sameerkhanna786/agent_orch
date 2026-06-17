"""Shared verification outcome taxonomy."""

from __future__ import annotations

from enum import Enum
from typing import Any


class VerificationOutcome(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"
    INDETERMINATE = "indeterminate"
    INVALID = "invalid"


def classify_quick_verification(
    quick_verification: dict[str, Any] | None,
    *,
    has_expected_coverage_gap: bool = False,
    harness_indeterminate: bool = False,
    invalid_candidate: bool = False,
) -> VerificationOutcome:
    if invalid_candidate:
        return VerificationOutcome.INVALID
    if not isinstance(quick_verification, dict) or not quick_verification:
        return VerificationOutcome.INDETERMINATE
    # A harness/launch failure (E2BIG, exec abort) means the test process never
    # ran — INDETERMINATE, not a genuine FAILED. Honored both via the explicit
    # arg and via the flag the engine stamps on the payload, so every caller that
    # forwards the payload is covered without threading the arg through each one.
    if harness_indeterminate or bool(quick_verification.get("harness_indeterminate")):
        return VerificationOutcome.INDETERMINATE
    if bool(quick_verification.get("timed_out")) or bool(
        quick_verification.get("full_scope_timed_out")
    ):
        return VerificationOutcome.INDETERMINATE
    failed = quick_verification.get("failed")
    errors = quick_verification.get("errors")
    if isinstance(failed, int) and failed > 0:
        return VerificationOutcome.FAILED
    if isinstance(errors, int) and errors > 0:
        return VerificationOutcome.FAILED
    returncode = quick_verification.get("returncode")
    if isinstance(returncode, int) and returncode != 0:
        return VerificationOutcome.FAILED
    pass_rate = quick_verification.get("pass_rate")
    if isinstance(pass_rate, (int, float)) and float(pass_rate) < 0.999:
        return VerificationOutcome.FAILED
    if has_expected_coverage_gap:
        return VerificationOutcome.INCOMPLETE
    if isinstance(returncode, int):
        return VerificationOutcome.PASSED
    if isinstance(pass_rate, (int, float)):
        return VerificationOutcome.PASSED
    return VerificationOutcome.INDETERMINATE
