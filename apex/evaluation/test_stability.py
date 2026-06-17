"""Multi-run stability checks for generated tests."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .f2p_oracle import _resolve_test_runner_adapter, _run_result_to_dict, _run_tests_on_paths


@dataclass
class TestStabilityReport:
    status: str
    language: str = "python"
    test_paths: list[str] = field(default_factory=list)
    run_count: int = 0
    passed_run_count: int = 0
    failed_run_count: int = 0
    flaky_nodeids: list[str] = field(default_factory=list)
    run_results: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    duration_seconds: float = 0.0

    @property
    def stable(self) -> bool:
        return self.status == "ok" and self.failed_run_count == 0 and not self.flaky_nodeids

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "language": self.language,
            "test_paths": list(self.test_paths),
            "run_count": self.run_count,
            "passed_run_count": self.passed_run_count,
            "failed_run_count": self.failed_run_count,
            "flaky_nodeids": list(self.flaky_nodeids),
            "stable": self.stable,
            "run_results": [dict(result) for result in self.run_results],
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 3),
        }


def evaluate_test_stability(
    *,
    worktree_path: str | Path,
    test_paths: list[str],
    language: str = "python",
    runs: int = 3,
    timeout_seconds: float = 60.0,
    python_executable: Optional[str] = None,
) -> TestStabilityReport:
    """Rerun generated tests with varied process env and flag instability."""
    started = time.time()
    normalized_language = (language or "").lower() or "python"
    paths = [str(path).strip() for path in test_paths or [] if str(path).strip()]
    run_count = max(1, int(runs or 1))
    report = TestStabilityReport(
        status="ok",
        language=normalized_language,
        test_paths=paths,
        run_count=run_count,
    )
    if not paths:
        report.status = "no_test_paths"
        report.duration_seconds = time.time() - started
        return report

    worktree = Path(worktree_path)
    adapter = _resolve_test_runner_adapter(
        fixed_dir=worktree,
        language=normalized_language,
    )
    original_env = {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "TZ": os.environ.get("TZ"),
    }
    node_statuses: dict[str, set[str]] = {}
    try:
        for index in range(run_count):
            os.environ["PYTHONHASHSEED"] = str(1009 + index)
            os.environ.setdefault("TZ", "UTC")
            result = _run_tests_on_paths(
                adapter=adapter,
                sandbox_dir=worktree,
                test_paths=paths,
                timeout_seconds=timeout_seconds,
                python_executable=python_executable,
            )
            result_dict = _run_result_to_dict(result)
            result_dict["stability_run_index"] = index
            result_dict["python_hash_seed"] = os.environ.get("PYTHONHASHSEED")
            report.run_results.append(result_dict)
            statuses = {
                str(status).strip().lower()
                for status in dict(getattr(result, "per_test_status", {}) or {}).values()
            }
            returncode = getattr(result, "returncode", 0)
            run_passed = (
                result.status == "ok"
                and not (isinstance(returncode, int) and returncode != 0)
                and not (statuses & {"fail", "error"})
            )
            if run_passed:
                report.passed_run_count += 1
            else:
                report.failed_run_count += 1
            for nodeid, status in dict(getattr(result, "per_test_status", {}) or {}).items():
                node_statuses.setdefault(str(nodeid), set()).add(str(status))
        report.flaky_nodeids = sorted(
            nodeid for nodeid, statuses in node_statuses.items() if len(statuses) > 1
        )
        report.duration_seconds = time.time() - started
        return report
    except Exception as exc:  # pragma: no cover - defensive
        report.status = "exception"
        report.error = f"{type(exc).__name__}: {exc}"
        report.duration_seconds = time.time() - started
        return report
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
