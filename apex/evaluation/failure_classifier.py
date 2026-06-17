"""Failure taxonomy for test-generation validation and benchmark runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from .test_style import TestStyleProfile


class FailureClass(str, Enum):
    APEX_SYNTAX = "apex_syntax"
    APEX_SPLICE_COLLISION = "apex_splice_collision"
    APEX_MISSING_SYMBOL = "apex_missing_symbol"
    APEX_MISSING_IMPORT = "apex_missing_import"
    APEX_WRONG_ASSERTION = "apex_wrong_assertion"
    APEX_WRONG_EXCEPTION = "apex_wrong_exception"
    APEX_TEST_ISOLATION = "apex_test_isolation"
    APEX_ADDED_DEP = "apex_added_dep"
    APEX_INFINITE_LOOP = "apex_infinite_loop"
    APEX_EMPTY_FILTER = "apex_empty_filter"
    ENV_RUNNER_MISSING = "env_runner_missing"
    ENV_DEP_MISSING = "env_dep_missing"
    ENV_DB_MISSING = "env_db_missing"
    ENV_NETWORK_BLOCKED = "env_network_blocked"
    ENV_DISK_FULL = "env_disk_full"
    ENV_PERMISSION_DENIED = "env_permission_denied"
    ENV_TIMEOUT = "env_timeout"
    ENV_UNKNOWN = "env_unknown"
    HARNESS_BUG = "harness_bug"
    NO_TESTS_COLLECTED = "apex_no_tests_collected"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FailureClassification:
    failure_class: FailureClass
    reason: str = ""
    charged_to_apex: bool = True
    repair_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failure_class"] = self.failure_class.value
        payload["repair_action"] = self.repair_action or failure_routing_action(self)
        return payload


def classify_testgen_failure(
    run_payload: dict[str, Any],
    *,
    style: TestStyleProfile | None = None,
) -> FailureClassification:
    text = "\n".join(
        str(run_payload.get(key) or "")
        for key in (
            "stdout_tail",
            "stderr_tail",
            "error",
            "status",
            "diagnostic",
            "message",
        )
    )
    lowered = text.lower()
    per_test = run_payload.get("per_test_status")
    duration = float(run_payload.get("duration_seconds") or 0.0)
    if _payload_indicates_test_isolation(run_payload):
        return _classified(
            FailureClass.APEX_TEST_ISOLATION,
            "generated tests pass in isolation but fail in the combined suite",
        )
    if (
        isinstance(per_test, dict)
        and not per_test
        and int(run_payload.get("returncode") or 0) == 0
        and str(run_payload.get("status") or "ok").lower() == "ok"
        and (duration <= 1.0 or "teststime: 0.0" in lowered)
    ):
        return _classified(
            FailureClass.APEX_EMPTY_FILTER,
            "runner completed with an empty generated-test subset",
        )
    if "no space left on device" in lowered:
        return _classified(FailureClass.ENV_DISK_FULL, "disk full", False)
    if "permission denied" in lowered or "operation not permitted" in lowered:
        return _classified(FailureClass.ENV_PERMISSION_DENIED, "permission denied", False)
    if "network is unreachable" in lowered or "temporary failure in name resolution" in lowered:
        return _classified(FailureClass.ENV_NETWORK_BLOCKED, "network unavailable", False)
    if "database" in lowered and ("does not exist" in lowered or "connection refused" in lowered):
        return _classified(FailureClass.ENV_DB_MISSING, "database unavailable", False)
    if (
        "already registered" in lowered
        or "reloading models is not advised" in lowered
        or (
            "destroying test database" in lowered
            and any(token in lowered for token in ("fail", "error", "traceback"))
        )
        or ("database" in lowered and "errors=" in lowered and "generated" in lowered)
    ):
        return _classified(
            FailureClass.APEX_TEST_ISOLATION,
            "generated test leaks process or database state across tests",
        )
    if (
        "conflicting" in lowered
        and "model" in lowered
        or "duplicate top-level" in lowered
        or "already defined" in lowered
        or "splice collision" in lowered
    ):
        return _classified(
            FailureClass.APEX_SPLICE_COLLISION,
            "generated test collides with post-splice symbols",
        )
    if "syntaxerror" in lowered or "indentationerror" in lowered or "unterminated" in lowered:
        return _classified(FailureClass.APEX_SYNTAX, "generated test is not syntactically valid")
    if "forbidden import" in lowered or (
        "pytest_django" in lowered and style and style.runner != "pytest"
    ):
        return _classified(
            FailureClass.APEX_ADDED_DEP, "generated test introduced a forbidden test dependency"
        )
    if "modulenotfounderror" in lowered or "no module named" in lowered:
        if (
            style
            and style.runner in {"django-runtests", "unittest", "sympy-bin-test"}
            and "pytest" in lowered
        ):
            return _classified(
                FailureClass.APEX_ADDED_DEP, "generated test imported pytest for a non-pytest style"
            )
        if style and style.runner == "pytest" and "pytest" in lowered:
            return _classified(
                FailureClass.ENV_RUNNER_MISSING, "configured test runner is missing", False
            )
        return _classified(FailureClass.ENV_DEP_MISSING, "dependency missing", False)
    if "cannot import name" in lowered:
        return _classified(
            FailureClass.APEX_MISSING_IMPORT, "generated test imports a name that is unavailable"
        )
    if "attributeerror" in lowered and (
        "has no attribute" in lowered or "object has no attribute" in lowered
    ):
        return _classified(FailureClass.APEX_MISSING_SYMBOL, "generated test used a missing symbol")
    if "nameerror" in lowered and "is not defined" in lowered:
        if any(name in lowered for name in ("pytest", "mock", "unittest", "np", "pd")):
            return _classified(
                FailureClass.APEX_MISSING_IMPORT,
                "generated test used a helper without importing it",
            )
        return _classified(
            FailureClass.APEX_MISSING_SYMBOL, "generated test used an undefined name"
        )
    if "no tests collected" in lowered or "no tests ran" in lowered:
        return _classified(FailureClass.NO_TESTS_COLLECTED, "no generated tests collected")
    if bool(run_payload.get("timed_out")) or "timeout" in lowered:
        if _text_mentions_generated_test_stack(lowered):
            return _classified(FailureClass.APEX_INFINITE_LOOP, "timeout inside generated test")
        return _classified(FailureClass.ENV_TIMEOUT, "environment or runner timeout", False)
    if (
        "traceback" in lowered
        and "run_evaluation.py" in lowered
        and "evaluate_instance" not in lowered
    ):
        return _classified(FailureClass.HARNESS_BUG, "harness raised before user code", False)
    if "did not raise" in lowered or "wrong exception" in lowered:
        return _classified(
            FailureClass.APEX_WRONG_EXCEPTION,
            "generated test expected the wrong exception behavior",
        )
    if isinstance(per_test, dict) and any(
        str(status).lower() in {"fail", "error"} for status in per_test.values()
    ):
        return _classified(FailureClass.APEX_WRONG_ASSERTION, "generated tests failed")
    if any(token in lowered for token in ("docker", "container", "image pull", "mount")):
        return _classified(FailureClass.ENV_UNKNOWN, "unclassified infrastructure failure", False)
    return _classified(FailureClass.UNKNOWN, "unclassified failure")


def failure_routing_action(classification: FailureClassification) -> str:
    cls = classification.failure_class
    if cls == FailureClass.APEX_ADDED_DEP:
        return "regenerate_artifact_imports_only"
    if cls == FailureClass.APEX_WRONG_ASSERTION:
        return "rewrite_assertion_with_actual_value"
    if cls in {FailureClass.APEX_EMPTY_FILTER, FailureClass.NO_TESTS_COLLECTED}:
        return "regenerate_full"
    if cls == FailureClass.APEX_TEST_ISOLATION:
        return "drop_test"
    if cls.value.startswith("apex_"):
        return "regenerate_full"
    if cls in {FailureClass.ENV_RUNNER_MISSING, FailureClass.ENV_DEP_MISSING}:
        return "install_dep"
    if cls in {
        FailureClass.ENV_DISK_FULL,
        FailureClass.ENV_PERMISSION_DENIED,
        FailureClass.HARNESS_BUG,
    }:
        return "abort_run"
    if cls == FailureClass.ENV_TIMEOUT:
        return "infra_retry"
    if cls.value.startswith("env_"):
        return "infra_retry"
    return "record_and_continue"


def _text_mentions_generated_test_stack(lowered: str) -> bool:
    return "test_" in lowered or "_apex_generated" in lowered or "generated" in lowered


def _payload_indicates_test_isolation(run_payload: dict[str, Any]) -> bool:
    isolated = _status_map(
        run_payload.get("isolated_per_test_status")
        or run_payload.get("per_test_status_isolated")
        or run_payload.get("isolated_status")
    )
    combined = _status_map(
        run_payload.get("combined_per_test_status")
        or run_payload.get("suite_per_test_status")
        or run_payload.get("per_test_status_combined")
    )
    if not isolated or not combined:
        offenders = run_payload.get("isolation_offenders")
        return bool(offenders)
    for nodeid, isolated_status in isolated.items():
        combined_status = combined.get(nodeid)
        if not combined_status:
            continue
        if isolated_status == "pass" and combined_status in {"fail", "error"}:
            return True
    return False


def _status_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(status or "").lower() for key, status in value.items() if str(key)}


def _classified(
    failure_class: FailureClass,
    reason: str,
    charged_to_apex: bool = True,
) -> FailureClassification:
    classification = FailureClassification(
        failure_class=failure_class,
        reason=reason,
        charged_to_apex=charged_to_apex,
    )
    return FailureClassification(
        failure_class=failure_class,
        reason=reason,
        charged_to_apex=charged_to_apex,
        repair_action=failure_routing_action(classification),
    )
