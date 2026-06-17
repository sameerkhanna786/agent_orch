"""
Shared acceptance and verification-signal helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _unit_interval(value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)):
        return None
    return max(0.0, min(float(value), 1.0))


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def _quick_verification_observed_outcome_count(
    quick_verification: dict[str, Any],
) -> int:
    executed = _positive_int(quick_verification.get("executed_test_count"))
    if executed is not None:
        return executed

    total = 0
    observed_any = False
    for key in ("passed", "failed", "errors", "skipped"):
        value = quick_verification.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            continue
        observed_any = True
        total += int(value)

    case_total = 0
    for key in ("passed_tests", "failed_tests", "error_tests", "skipped_tests"):
        value = quick_verification.get(key)
        if isinstance(value, list):
            case_total += len(value)
    if observed_any:
        return total if total > 0 or case_total <= 0 else case_total
    return case_total


_REDUCED_EXPECTED_COVERAGE_SCOPES = {
    "candidate_test_paths",
    "failing_tests",
    "focus_test_files",
    "module_group_subset",
    "sampled_expected_suite",
    "structural_precheck",
}

# Search-control only: a clean stratified sample this large should stop spending
# new rollout budget and let authoritative scoring decide; it is never accepted.
_MATERIAL_SCORING_MIN_OBSERVED_TEST_COUNT = 512


def _quick_verification_is_reduced_scope(quick_verification: dict[str, Any]) -> bool:
    scope = str(quick_verification.get("scope") or "").strip()
    if not scope:
        return False
    return scope != "full_test_command" or scope in _REDUCED_EXPECTED_COVERAGE_SCOPES


def _result_metadata(rollout_result: Any) -> dict[str, Any]:
    return _as_mapping(getattr(rollout_result, "search_metadata", None))


def _gold_suite_visible_result(rollout_result: Any) -> bool:
    metadata = _result_metadata(rollout_result)
    return str(metadata.get("evidence_mode") or "").strip() == "gold_suite_visible"


def _is_visible_test_path(path: Any) -> bool:
    normalized = str(path or "").replace("\\", "/").strip()
    if not normalized:
        return False
    parts = [part for part in normalized.split("/") if part]
    name = normalized.rsplit("/", 1)[-1]
    if any(part in {"tests", "test", "__tests__"} for part in parts[:-1]):
        return name.endswith(".py")
    return len(parts) == 1 and name.startswith("test_") and name.endswith(".py")


def _changed_files(rollout_result: Any) -> list[str]:
    values = getattr(rollout_result, "changed_files", None)
    if isinstance(values, (list, tuple, set)):
        return [str(value) for value in values if str(value or "").strip()]
    patch_artifact = _as_mapping(getattr(rollout_result, "patch_artifact", None))
    values = patch_artifact.get("changed_files")
    if isinstance(values, (list, tuple, set)):
        return [str(value) for value in values if str(value or "").strip()]
    return []


def _patch_touches_visible_test(patch: Any) -> bool:
    if not isinstance(patch, str) or not patch:
        return False
    for line in patch.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        for raw in parts[2:4]:
            path = raw[2:] if raw.startswith(("a/", "b/")) else raw
            if _is_visible_test_path(path):
                return True
    return False


def rollout_touches_gold_visible_tests(rollout_result: Any) -> bool:
    """Return whether a gold-suite-visible candidate edits visible tests.

    The final benchmark harness may apply more detailed policy, but preemptive
    orchestration needs a conservative validity signal before stopping sibling
    agents. A patch that touches visible tests is not a safe early-stop proof.
    """

    if not _gold_suite_visible_result(rollout_result):
        return False
    if any(_is_visible_test_path(path) for path in _changed_files(rollout_result)):
        return True
    return _patch_touches_visible_test(getattr(rollout_result, "patch", None))


def quick_verification_expected_coverage_ratio(
    quick_verification: Optional[dict[str, Any]],
) -> Optional[float]:
    if not isinstance(quick_verification, dict) or not quick_verification:
        return None

    expected_test_count = quick_verification.get("expected_test_count")
    if not isinstance(expected_test_count, int) or expected_test_count <= 0:
        return None

    matched_expected_test_count = quick_verification.get("matched_expected_test_count")
    if isinstance(matched_expected_test_count, int) and matched_expected_test_count >= 0:
        return max(
            0.0,
            min(float(matched_expected_test_count) / float(expected_test_count), 1.0),
        )

    collected_test_count = quick_verification.get("collected_test_count")
    if isinstance(collected_test_count, int) and collected_test_count >= 0:
        return max(
            0.0,
            min(float(collected_test_count) / float(expected_test_count), 1.0),
        )

    coverage_preserved = quick_verification.get("coverage_preserved")
    if isinstance(coverage_preserved, bool):
        return 1.0 if coverage_preserved else 0.0

    return None


def quick_verification_has_literal_full_coverage_pass(
    quick_verification: Optional[dict[str, Any]],
) -> bool:
    """Return whether the quick verification is a LITERAL full-coverage zero-failure win.

    This is a SCOPE-AGNOSTIC completion signal. It returns True only when the
    rollout literally covered the entire expected-test universe with zero
    failures/errors and a clean return code, regardless of the quick
    verification ``scope`` label (``full_test_command`` vs ``failing_tests``).

    It is strictly STRONGER than the scope-label strong-signal gate it augments:
    full expected coverage + zero failures/errors + clean return code implies
    the suite passed, so it can never recognize anything weaker than the
    existing label-based gate already accepts. Any field-read/parse error
    fails open (returns False), preserving today's no-preempt behavior.
    """

    try:
        if not isinstance(quick_verification, dict) or not quick_verification:
            return False
        if bool(quick_verification.get("timed_out")):
            return False
        if bool(quick_verification.get("full_scope_timed_out")):
            return False

        expected = quick_verification.get("expected_test_count")
        if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
            return False

        matched = quick_verification.get("matched_expected_test_count")
        if not isinstance(matched, int) or isinstance(matched, bool) or matched != expected:
            return False

        missing = quick_verification.get("missing_expected_test_count")
        if not isinstance(missing, int) or isinstance(missing, bool) or missing != 0:
            return False

        failed = quick_verification.get("failed")
        if not isinstance(failed, int) or isinstance(failed, bool) or failed != 0:
            return False

        errors = quick_verification.get("errors")
        if not isinstance(errors, int) or isinstance(errors, bool) or errors != 0:
            return False

        returncode = quick_verification.get("returncode")
        if returncode not in (0, None):
            return False

        coverage_ratio = quick_verification_expected_coverage_ratio(quick_verification)
        if not isinstance(coverage_ratio, (int, float)):
            return False
        if float(coverage_ratio) < 0.999:
            return False

        if _quick_verification_observed_outcome_count(quick_verification) < expected:
            return False

        return True
    except Exception:
        return False


# A1: collection-progress value floor. The ceiling is intentionally strictly
# below any genuine passing signal AND below the existing <=0.35 failing-candidate
# selection clamp (selector._cluster_public_signal_score). It only changes which
# FAILING partial gets invested in (rollout reward + cluster public signal seed),
# never which candidate is accepted/selected: a 0-pass candidate can never reach
# this floor and a genuinely-passing candidate always exceeds it.
COLLECTION_PROGRESS_CEIL = 0.40


def _collection_progress_floor(
    quick_verification: dict[str, Any],
) -> float:
    """Return a small value floor in ``[0, COLLECTION_PROGRESS_CEIL]``.

    The floor rewards a rollout that recovered test-collection surface (so the
    deepest partial becomes the highest-reward repair seed) without ever
    elevating it to or above a real pass. It is derived from, in order of
    preference, collected/expected, matched/expected, then coverage_preserved.
    A rollout that collected/matched nothing, or whose coverage collapsed,
    yields ``0.0`` so a 0-progress candidate can never be lifted.
    """

    expected = quick_verification.get("expected_test_count")
    ratio: Optional[float] = None
    if isinstance(expected, int) and expected > 0:
        collected = quick_verification.get("collected_test_count")
        matched = quick_verification.get("matched_expected_test_count")
        if isinstance(collected, int) and collected > 0:
            ratio = float(collected) / float(expected)
        elif isinstance(matched, int) and matched > 0:
            ratio = float(matched) / float(expected)
    if ratio is None:
        coverage_preserved = quick_verification.get("coverage_preserved")
        if coverage_preserved is True:
            ratio = 1.0
        else:
            return 0.0
    floor = COLLECTION_PROGRESS_CEIL * max(0.0, min(ratio, 1.0))
    return max(0.0, min(floor, COLLECTION_PROGRESS_CEIL))


def quick_verification_signal_score(
    quick_verification: Optional[dict[str, Any]],
) -> Optional[float]:
    if not isinstance(quick_verification, dict) or not quick_verification:
        return None

    pass_rate = _unit_interval(quick_verification.get("pass_rate"))
    passed = quick_verification.get("passed")
    failed = quick_verification.get("failed")
    errors = quick_verification.get("errors")
    if all(isinstance(value, int) and value >= 0 for value in (passed, failed, errors)):
        total = int(passed) + int(failed) + int(errors)
        if total > 0:
            count_pass_rate = max(0.0, min(float(passed) / float(total), 1.0))
            pass_rate = count_pass_rate if pass_rate is None else min(pass_rate, count_pass_rate)
    if pass_rate is None and quick_verification.get("returncode") == 0:
        pass_rate = 1.0
    if pass_rate is None:
        return None

    coverage_ratio = quick_verification_expected_coverage_ratio(quick_verification)
    if coverage_ratio is not None:
        pass_rate = min(pass_rate, coverage_ratio)

    # A1: after the coverage min, raise a deep-but-failing partial to a small
    # collection-progress floor so it outranks a do-nothing rollout as a repair
    # seed. The ceil (0.40) keeps it strictly below any genuine pass and below
    # the failing-candidate selection clamp, so acceptance/selection is never
    # affected — only which failing partial APEX invests in.
    floor = _collection_progress_floor(quick_verification)
    if floor > pass_rate:
        pass_rate = floor

    return pass_rate


def quick_verification_has_scored_expected_suite_pass(
    quick_verification: Optional[dict[str, Any]],
) -> bool:
    """Return whether the benchmark-scored expected-test universe passed.

    Some benchmark adapters define a scoring universe that is narrower than the
    raw local test command. In that case a nonzero raw command return code can
    coexist with a perfect scored expected-suite result when only non-scored
    extra tests failed. This helper intentionally does not claim a local full
    suite pass; it only identifies a benchmark-scored expected-suite pass.
    """

    if not isinstance(quick_verification, dict) or not quick_verification:
        return False
    if bool(quick_verification.get("timed_out")):
        return False
    if bool(quick_verification.get("full_scope_timed_out")):
        return False

    scope = str(quick_verification.get("scope") or "").strip()
    if scope != "full_test_command":
        return False

    expected = quick_verification.get("expected_test_count")
    matched = quick_verification.get("matched_expected_test_count")
    missing = quick_verification.get("missing_expected_test_count")
    if not isinstance(expected, int) or expected <= 0:
        return False
    if not isinstance(matched, int) or matched < expected:
        return False
    if not isinstance(missing, int) or missing != 0:
        return False
    if quick_verification.get("coverage_preserved") is False:
        return False
    if bool(quick_verification.get("empty_expected_suite_execution")):
        return False
    if _quick_verification_observed_outcome_count(quick_verification) < expected:
        return False

    failed = quick_verification.get("failed")
    errors = quick_verification.get("errors")
    if not isinstance(failed, int) or failed != 0:
        return False
    if not isinstance(errors, int) or errors != 0:
        return False

    passed = quick_verification.get("passed")
    if not isinstance(passed, int) or passed <= 0:
        return False

    pass_rate = quick_verification_signal_score(quick_verification)
    return pass_rate is not None and pass_rate >= 0.999


def quick_verification_has_strong_signal(
    quick_verification: Optional[dict[str, Any]],
    *,
    require_full_scope: bool = False,
) -> bool:
    if not isinstance(quick_verification, dict) or not quick_verification:
        return False
    if bool(quick_verification.get("timed_out")):
        return False
    if bool(quick_verification.get("full_scope_timed_out")):
        return False
    if bool(quick_verification.get("empty_expected_suite_execution")):
        return False

    failed = quick_verification.get("failed")
    errors = quick_verification.get("errors")
    pass_rate = quick_verification_signal_score(quick_verification)
    if pass_rate is None:
        return False
    if pass_rate < 0.999:
        return False
    if not isinstance(failed, int) or failed != 0:
        return False
    if not isinstance(errors, int) or errors != 0:
        return False
    observed = _quick_verification_observed_outcome_count(quick_verification)
    if observed <= 0:
        return False

    expected = quick_verification.get("expected_test_count")
    if isinstance(expected, int) and not isinstance(expected, bool) and expected > 0:
        if observed < expected:
            return False

    scope = str(quick_verification.get("scope") or "").strip()
    if require_full_scope:
        return scope == "full_test_command"

    covered_targets = bool(
        quick_verification.get("selected_tests")
        or quick_verification.get("passed_tests")
        or quick_verification.get("failed_tests")
    )
    return scope == "full_test_command" or covered_targets


def quick_verification_requires_authoritative_scoring(
    quick_verification: Optional[dict[str, Any]],
    *,
    metadata: Optional[dict[str, Any]] = None,
) -> bool:
    """Return whether clean reduced-scope evidence should be escalated to scoring.

    This is candidate-preservation evidence, not an acceptance or terminal
    search-control signal. It identifies a candidate whose bounded verification
    passed but explicitly cannot prove the full acceptance universe.
    """

    if not isinstance(quick_verification, dict) or not quick_verification:
        return False
    metadata = _as_mapping(metadata)
    explicit_scoring_request = bool(quick_verification.get("requires_full_scoring")) or bool(
        metadata.get("requires_full_scoring")
    )
    expected = quick_verification.get("expected_test_count")
    reduced_scope_scoring_request = bool(
        _quick_verification_is_reduced_scope(quick_verification)
        and isinstance(expected, int)
        and not isinstance(expected, bool)
        and expected > 0
        and not quick_verification_has_literal_full_coverage_pass(quick_verification)
    )
    if not (explicit_scoring_request or reduced_scope_scoring_request):
        return False
    if bool(quick_verification.get("timed_out")):
        return False
    if bool(quick_verification.get("full_scope_timed_out")):
        return False

    outcome = str(quick_verification.get("verification_outcome") or "").strip().lower()
    if outcome and outcome not in {"incomplete", "passed"}:
        return False

    failed = quick_verification.get("failed")
    errors = quick_verification.get("errors")
    if not isinstance(failed, int) or isinstance(failed, bool) or failed != 0:
        return False
    if not isinstance(errors, int) or isinstance(errors, bool) or errors != 0:
        return False

    passed = quick_verification.get("passed")
    if not isinstance(passed, int) or isinstance(passed, bool) or passed <= 0:
        return False

    returncode = quick_verification.get("returncode")
    if returncode not in (0, None):
        return False

    pass_rate = _unit_interval(quick_verification.get("pass_rate"))
    if pass_rate is not None and pass_rate < 0.999:
        return False

    missing = quick_verification.get("missing_expected_test_count")
    observed = _quick_verification_observed_outcome_count(quick_verification)
    has_coverage_gap = bool(quick_verification.get("coverage_preserved") is False)
    if isinstance(expected, int) and not isinstance(expected, bool) and expected > 0:
        has_coverage_gap = has_coverage_gap or observed < expected
    if isinstance(missing, int) and not isinstance(missing, bool):
        has_coverage_gap = has_coverage_gap or missing > 0
    if not has_coverage_gap and not reduced_scope_scoring_request:
        return False

    return not quick_verification_has_literal_full_coverage_pass(quick_verification)


def quick_verification_has_local_full_scope_pass(
    quick_verification: Optional[dict[str, Any]],
) -> bool:
    """Return whether the rollout itself observed a clean local full-suite pass.

    This signal intentionally ignores benchmark-level expected-ID coverage.
    It is useful for search control when Apex should stop spending extra
    rollout budget after a corroborated local full-suite win, even if
    hidden benchmark expectations keep the patch from being marked
    ``accepted`` internally. Do NOT use this as a substitute for final
    acceptance.
    """

    if not isinstance(quick_verification, dict) or not quick_verification:
        return False
    if bool(quick_verification.get("timed_out")):
        return False
    if bool(quick_verification.get("full_scope_timed_out")):
        return False

    scope = str(quick_verification.get("scope") or "").strip()
    if scope != "full_test_command":
        return False

    returncode = quick_verification.get("returncode")
    if isinstance(returncode, int) and returncode != 0:
        return False

    failed = quick_verification.get("failed")
    errors = quick_verification.get("errors")
    if not isinstance(failed, int) or failed != 0:
        return False
    if not isinstance(errors, int) or errors != 0:
        return False

    pass_rate = _unit_interval(quick_verification.get("pass_rate"))
    passed = quick_verification.get("passed")
    if pass_rate is None and isinstance(passed, int):
        total = passed + failed + errors
        if total > 0:
            pass_rate = max(0.0, min(float(passed) / float(total), 1.0))
    if pass_rate is None and isinstance(returncode, int) and returncode == 0:
        pass_rate = 1.0
    if pass_rate is None or pass_rate < 0.999:
        return False

    expected = quick_verification.get("expected_test_count")
    if isinstance(expected, int) and not isinstance(expected, bool) and expected > 0:
        if _quick_verification_observed_outcome_count(quick_verification) < expected:
            return False

    collected_test_count = quick_verification.get("collected_test_count")
    if isinstance(collected_test_count, int):
        return collected_test_count > 0

    if isinstance(passed, int) and passed > 0:
        return True
    return bool(quick_verification.get("passed_tests") or quick_verification.get("failed_tests"))


def quick_verification_failure_label(
    quick_verification: Optional[dict[str, Any]],
) -> str:
    if not isinstance(quick_verification, dict):
        return ""
    classification = quick_verification.get("failure_classification")
    if not isinstance(classification, dict):
        return ""
    return str(classification.get("label") or classification.get("failure_class") or "").lower()


def verification_is_accepted(verification: Any) -> bool:
    if isinstance(verification, dict):
        return bool(verification.get("accepted"))
    return bool(getattr(verification, "accepted", False))


def verification_has_explicit_validity_rejection(verification: Any) -> bool:
    """Return whether verifier-side validity gates already found a hard defect."""

    if not isinstance(verification, dict):
        return False
    if verification.get("quality_gate_passed") is False:
        return True
    if verification.get("syntax_valid") is False:
        return True
    if verification.get("lint_clean") is False:
        return True
    prune_result = verification.get("prune_result")
    if isinstance(prune_result, dict) and prune_result.get("is_valid") is False:
        return True
    test_result = verification.get("test_result")
    if isinstance(test_result, dict) and test_result.get("expected_coverage_preserved") is False:
        return True
    validity = verification.get("validity")
    if isinstance(validity, dict):
        reasons = {
            str(reason).strip() for reason in validity.get("reasons", []) if str(reason).strip()
        }
        if validity.get("quality_gate_passed") is False:
            return True
        if validity.get("backend_protocol_error") is True:
            return True
        if validity.get("provenance_violation") is True:
            return True
        if validity.get("protected_tests_unchanged") is False:
            return True
        if validity.get("collection_critical_files_unchanged") is False:
            return True
        if validity.get("coverage_collapse_terminal") is True:
            return True
        if validity.get("quick_verification_passed") is False:
            if "requires_authoritative_scoring" in reasons:
                return False
            return True
    return False


def _candidate_validity_payload(rollout_result: Any) -> dict[str, Any]:
    validity = getattr(rollout_result, "validity", None)
    if validity is None:
        return {}
    if isinstance(validity, dict):
        return validity
    as_dict = getattr(validity, "as_dict", None)
    if callable(as_dict):
        try:
            payload = as_dict()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            return payload
    return {
        "has_patch": getattr(validity, "has_patch", None),
        "worktree_materialized": getattr(validity, "worktree_materialized", None),
        "eligible_for_external_scoring": getattr(
            validity,
            "eligible_for_external_scoring",
            None,
        ),
        "eligible_for_submission": getattr(validity, "eligible_for_submission", None),
        "quick_verification_passed": getattr(validity, "quick_verification_passed", None),
        "protected_tests_unchanged": getattr(validity, "protected_tests_unchanged", None),
        "collection_critical_files_unchanged": getattr(
            validity,
            "collection_critical_files_unchanged",
            None,
        ),
        "expected_coverage_preserved": getattr(validity, "expected_coverage_preserved", None),
        "missing_expected_test_count": getattr(validity, "missing_expected_test_count", None),
        "quality_gate_passed": getattr(validity, "quality_gate_passed", None),
        "backend_protocol_error": getattr(validity, "backend_protocol_error", False),
        "coverage_collapse_terminal": getattr(validity, "coverage_collapse_terminal", False),
        "provenance_violation": getattr(validity, "provenance_violation", False),
        "reasons": list(getattr(validity, "reasons", []) or []),
    }


def _validity_contract_allows_visible_test_touch(validity_payload: dict[str, Any]) -> bool:
    if not validity_payload:
        return False
    if validity_payload.get("protected_tests_unchanged") is not True:
        return False
    if validity_payload.get("eligible_for_submission") is False:
        return False
    if validity_payload.get("collection_critical_files_unchanged") is False:
        return False
    if validity_payload.get("expected_coverage_preserved") is False:
        return False
    return True


def rollout_has_submission_blocking_validity(rollout_result: Any) -> bool:
    if rollout_result is None:
        return False
    validity_payload = _candidate_validity_payload(rollout_result)
    if rollout_touches_gold_visible_tests(
        rollout_result
    ) and not _validity_contract_allows_visible_test_touch(validity_payload):
        return True
    verification = getattr(rollout_result, "verification", None)
    if verification_has_explicit_validity_rejection(verification):
        return True
    if validity_payload:
        if validity_payload.get("eligible_for_submission") is False:
            return True
        if validity_payload.get("expected_coverage_preserved") is False:
            return True
        missing_expected = validity_payload.get("missing_expected_test_count")
        if (
            isinstance(missing_expected, int)
            and not isinstance(missing_expected, bool)
            and missing_expected > 0
        ):
            return True
        if verification_has_explicit_validity_rejection({"validity": validity_payload}):
            return True
    failure_reason = str(getattr(rollout_result, "failure_reason", "") or "").lower()
    if "protected test" in failure_reason or "protected visible test" in failure_reason:
        return True
    return False


def rollout_has_authoritative_acceptance(rollout_result: Any) -> bool:
    if rollout_result is None:
        return False
    if not getattr(rollout_result, "success", False):
        return False
    if not getattr(rollout_result, "patch", None):
        return False
    if rollout_has_submission_blocking_validity(rollout_result):
        return False

    verification = getattr(rollout_result, "verification", None)
    if isinstance(verification, dict) and verification.get("accepted") is True:
        return True
    if verification_has_explicit_validity_rejection(verification):
        return False

    quick_verification = getattr(rollout_result, "quick_verification", None)
    if quick_verification_has_strong_signal(
        quick_verification if isinstance(quick_verification, dict) else {},
        require_full_scope=True,
    ):
        return True

    # Scope-agnostic completion path: a LITERAL full-coverage zero-failure win
    # (matched==expected, missing==0, failed==0, errors==0, rc in {0,None},
    # coverage_ratio==1.0) is authoritative even when the quick verification
    # ``scope`` label is ``failing_tests`` rather than ``full_test_command``.
    # This sits below the success / patch / submission-blocking-validity /
    # explicit-rejection guards above, so those already gate it. It is strictly
    # stronger than the scope-label strong-signal gate it augments and never
    # accepts anything weaker than today. Fails open inside the helper.
    if quick_verification_has_literal_full_coverage_pass(
        quick_verification if isinstance(quick_verification, dict) else {},
    ):
        return True

    if isinstance(verification, dict) and "accepted" in verification:
        return bool(verification.get("accepted"))

    return verification_is_accepted(verification)


def rollout_requires_authoritative_scoring(rollout_result: Any) -> bool:
    """Return whether generation should pause so verifier/scorer can decide.

    A clean reduced-scope pass is valuable evidence, but it is not an acceptance.
    This helper only asks orchestration to stop spending more rollout budget and
    escalate the candidate to the normal authoritative verifier/scorer path.
    """

    if rollout_result is None:
        return False
    if not getattr(rollout_result, "success", False):
        return False
    if not getattr(rollout_result, "patch", None):
        return False

    verification = getattr(rollout_result, "verification", None)
    if isinstance(verification, dict):
        if verification.get("syntax_valid") is False:
            return False
        if verification.get("lint_clean") is False:
            return False
        if verification.get("quality_gate_passed") is False:
            return False
        prune_result = verification.get("prune_result")
        if isinstance(prune_result, dict) and prune_result.get("is_valid") is False:
            return False

    validity = getattr(rollout_result, "validity", None)
    if getattr(validity, "backend_protocol_error", False):
        return False
    if getattr(validity, "coverage_collapse_terminal", False):
        return False
    validity_payload = _candidate_validity_payload(rollout_result)
    if validity_payload:
        if validity_payload.get("protected_tests_unchanged") is False:
            return False
        if validity_payload.get("collection_critical_files_unchanged") is False:
            return False
        if validity_payload.get("quality_gate_passed") is False:
            return False
        if validity_payload.get("backend_protocol_error") is True:
            return False
        if validity_payload.get("coverage_collapse_terminal") is True:
            return False
        if validity_payload.get("provenance_violation") is True:
            return False

    quick_verification = getattr(rollout_result, "quick_verification", None)
    return quick_verification_requires_authoritative_scoring(
        quick_verification if isinstance(quick_verification, dict) else {},
        metadata=_result_metadata(rollout_result),
    )


def quick_verification_has_material_scoring_coverage(
    quick_verification: Optional[dict[str, Any]],
    *,
    min_expected_coverage_ratio: float = 0.5,
    min_observed_test_count: int = _MATERIAL_SCORING_MIN_OBSERVED_TEST_COUNT,
) -> bool:
    """Return whether scoring-request evidence is broad enough to prioritize.

    ``quick_verification_requires_authoritative_scoring`` is a candidate
    preservation signal: the scorer may still prove a clean sampled candidate.
    This helper is stricter and only describes materiality. Reduced-scope
    materiality must not by itself terminate search because full scoring can
    still expose residual failures that need additional rollouts.
    """

    if not quick_verification_requires_authoritative_scoring(quick_verification):
        return False
    if not isinstance(quick_verification, dict) or not quick_verification:
        return False
    if quick_verification_has_literal_full_coverage_pass(quick_verification):
        return True
    if not _quick_verification_is_reduced_scope(quick_verification):
        return True

    expected = quick_verification.get("expected_test_count")
    if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
        return True

    coverage_ratio = quick_verification_expected_coverage_ratio(quick_verification)
    if not isinstance(coverage_ratio, (int, float)):
        observed = _quick_verification_observed_outcome_count(quick_verification)
        if observed > 0:
            coverage_ratio = float(observed) / float(expected)
    else:
        observed = _quick_verification_observed_outcome_count(quick_verification)
    try:
        observed_floor = max(1, int(min_observed_test_count))
    except (TypeError, ValueError):
        observed_floor = _MATERIAL_SCORING_MIN_OBSERVED_TEST_COUNT
    if observed > 0 and bool(quick_verification.get("selection_budget_exhausted")):
        return True
    if observed >= observed_floor:
        return True
    if not isinstance(coverage_ratio, (int, float)):
        return True
    try:
        threshold = max(0.0, min(float(min_expected_coverage_ratio), 1.0))
    except (TypeError, ValueError):
        threshold = 0.5
    return float(coverage_ratio) >= threshold


def rollout_has_authoritative_scoring_stop_signal(rollout_result: Any) -> bool:
    """Return whether a scoring request is strong enough to stop generation."""

    if not rollout_requires_authoritative_scoring(rollout_result):
        return False
    quick_verification = getattr(rollout_result, "quick_verification", None)
    quick_verification = quick_verification if isinstance(quick_verification, dict) else {}
    if _quick_verification_is_reduced_scope(quick_verification):
        return False
    return quick_verification_has_material_scoring_coverage(quick_verification)


def rollout_has_preemptive_authoritative_scoring_request(rollout_result: Any) -> bool:
    """Return whether a score-worthy primary anchor should interrupt siblings.

    This is a search-control signal only. A reduced-scope pass is still not an
    acceptance. It can pause dispatch for scoring, but it should not kill active
    sibling rollouts unless the scorer request came from non-reduced evidence.
    """

    if not rollout_has_authoritative_scoring_stop_signal(rollout_result):
        return False
    quick_verification = getattr(rollout_result, "quick_verification", None)
    if isinstance(quick_verification, dict) and _quick_verification_is_reduced_scope(
        quick_verification,
    ):
        return False
    metadata = _result_metadata(rollout_result)
    return bool(metadata.get("standalone_agent_anchor")) or (
        str(metadata.get("search_reason") or "").strip() == "standalone_anchor_guard"
    )


def rollout_has_local_full_scope_completion(rollout_result: Any) -> bool:
    if rollout_result is None:
        return False
    if not getattr(rollout_result, "success", False):
        return False
    quick_verification = getattr(rollout_result, "quick_verification", None)
    return quick_verification_has_local_full_scope_pass(
        quick_verification if isinstance(quick_verification, dict) else {},
    )


def rollout_has_materialized_repair_seed(rollout_result: Any) -> bool:
    """Return whether a follow-up rollout can branch from this result."""

    if rollout_result is None:
        return False
    if bool(getattr(rollout_result, "patch", None)):
        return True
    changed_files = _changed_files(rollout_result)
    worktree_path = str(getattr(rollout_result, "worktree_path", "") or "").strip()
    if not changed_files or not worktree_path:
        return False
    return Path(worktree_path).is_dir()


def rollout_has_repairable_near_miss(
    rollout_result: Any,
    *,
    minimum_signal_score: float = 0.999,
    residual_fraction_cap: float = 0.001,
    max_residual_count: int = 50,
) -> bool:
    """Return whether a failing full-suite result is close enough to seed repair.

    This is a search-control signal, not final acceptance. It lets every
    orchestration path pivot from a preserved near-complete workspace instead
    of spending more sibling rollout budget from scratch.
    """

    if rollout_result is None or not rollout_has_materialized_repair_seed(rollout_result):
        return False
    quick_verification = _as_mapping(getattr(rollout_result, "quick_verification", None))
    if not quick_verification:
        return False
    if bool(quick_verification.get("timed_out")) or bool(
        quick_verification.get("full_scope_timed_out")
    ):
        return False
    if str(quick_verification.get("scope") or "").strip() != "full_test_command":
        return False
    if quick_verification_failure_label(quick_verification) in {"env", "environment_failure"}:
        return False

    passed = quick_verification.get("passed")
    failed = quick_verification.get("failed")
    errors = quick_verification.get("errors")
    if not all(isinstance(value, int) and value >= 0 for value in (passed, failed, errors)):
        return False
    residual = int(failed) + int(errors)
    if residual <= 0:
        return False
    total = int(passed) + residual
    if total <= 0:
        return False

    scaled_cap = int(total * max(0.0, float(residual_fraction_cap)))
    residual_cap = max(1, min(max(1, int(max_residual_count)), scaled_cap))
    if residual > residual_cap:
        return False

    threshold = max(0.0, min(1.0, float(minimum_signal_score)))
    signal_score = quick_verification_signal_score(quick_verification)
    if not isinstance(signal_score, (int, float)) or float(signal_score) < threshold:
        return False
    expected_coverage_ratio = quick_verification_expected_coverage_ratio(quick_verification)
    if (
        isinstance(expected_coverage_ratio, (int, float))
        and float(expected_coverage_ratio) < threshold
    ):
        return False
    return True


def rollout_made_collection_progress_vs_seed(
    quick_verification: Optional[dict[str, Any]],
    seed_quick_verification: Optional[dict[str, Any]],
) -> bool:
    """A2: milestone deepening predicate, decoupled from the 0.95 near-miss gate.

    Returns whether a rollout made *collection-surface* progress relative to its
    seed: it reduced ``missing_expected_test_count``, flipped
    ``coverage_preserved`` False -> True, or strictly increased the number of
    collected/matched/passed tests. This grants deep-but-not-near-miss
    candidates the extended deepening budget that the 99%-pass candidate already
    gets, without ever promoting a non-passing candidate to acceptance (it only
    decides whether to keep investing follow-up budget in this partial).

    The predicate is inert (returns ``False``) when no seed baseline is
    available and, by construction, zero-delta when collection already started
    at 100% (nothing left to reduce/flip/increase), so it never changes behavior
    on benchmarks that collect the full expected suite from the seed.
    """

    qv = _as_mapping(quick_verification)
    seed = _as_mapping(seed_quick_verification)
    if not qv or not seed:
        return False

    seed_missing = seed.get("missing_expected_test_count")
    cand_missing = qv.get("missing_expected_test_count")
    if (
        isinstance(seed_missing, int)
        and isinstance(cand_missing, int)
        and seed_missing > 0
        and 0 <= cand_missing < seed_missing
    ):
        return True

    if seed.get("coverage_preserved") is False and qv.get("coverage_preserved") is True:
        return True

    for key in ("collected_test_count", "matched_expected_test_count", "passed"):
        seed_value = seed.get(key)
        cand_value = qv.get(key)
        if (
            isinstance(seed_value, int)
            and isinstance(cand_value, int)
            and seed_value >= 0
            and cand_value > seed_value
        ):
            return True

    return False


def rollout_has_preemptive_completion(rollout_result: Any) -> bool:
    if rollout_has_submission_blocking_validity(rollout_result):
        return False
    return rollout_has_authoritative_acceptance(
        rollout_result
    ) or rollout_has_local_full_scope_completion(rollout_result)
