"""SWE-Bench (classic / Verified / Multilingual) test-runner adapter.

Parses the public ``swebench`` package's ``report.json`` schema into the
APEX :class:`RunResult` shape so the orchestrator's selector / residual
followup hooks can reason about SWE-Bench harness results uniformly.

Schema (per-instance ``report.json`` written by
``swebench.harness.run_evaluation``)::

    {
      "<instance_id>": {
        "patch_is_None": false,
        "patch_exists": true,
        "patch_successfully_applied": true,
        "resolved": true,
        "tests_status": {
          "FAIL_TO_PASS": {"success": [...], "failure": [...]},
          "PASS_TO_PASS": {"success": [...], "failure": [...]}
        }
      }
    }

This adapter is harness-driven (the harness's docker images own the
actual test execution); it only exposes the TestRunnerAdapter protocol
so APEX's selector code can read SWE-Bench results without branching on
benchmark name.

Mirrors :class:`SWEBenchProAdapter` for the per-language stub patterns
and infrastructure paths, since SWE-Bench Multilingual has the same
9-language coverage as Pro.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Optional

from . import RunResult, register_adapter
from .swebench_pro_adapter import (
    _INFRASTRUCTURE_FILENAMES_BY_LANG,
    _STUB_PATTERNS_BY_LANG,
    _normalize_language,
)


def _unwrap_instance(payload: dict, instance_id: Optional[str] = None) -> dict:
    """Pull the inner per-instance dict out of a swebench report.json.

    The harness writes ``{instance_id: {...}}``; older versions wrote
    just the inner dict. Tolerate both shapes so the adapter can be
    used against any swebench-package version.
    """

    if not isinstance(payload, dict):
        return {}
    if instance_id and instance_id in payload and isinstance(payload[instance_id], dict):
        return payload[instance_id]
    if len(payload) == 1:
        only_value = next(iter(payload.values()))
        if isinstance(only_value, dict):
            return only_value
    return payload


class SWEBenchAdapter:
    """TestRunnerAdapter for the public swebench harness output."""

    name = "swebench"
    language = "polyglot"

    def __init__(
        self,
        repo_language: Optional[str] = None,
        instance_id: Optional[str] = None,
    ) -> None:
        self.repo_language = _normalize_language(repo_language)
        self.instance_id = instance_id

    def list_tests(self, workspace: Path, env: dict[str, str]) -> set[str]:
        # Test discovery for SWE-Bench happens entirely inside the
        # harness's per-instance docker image; we don't re-derive it
        # here. Returning empty is the honest "not handled by this
        # adapter" signal.
        return set()

    def build_run_command(
        self,
        workspace: Path,
        test_ids: list[str],
        report_path: Path,
        *,
        executable: Optional[str] = None,
    ) -> str:
        # Same caveat as list_tests: the harness docker image owns the
        # actual test command. This placeholder satisfies the protocol
        # contract for code paths that call build_run_command without
        # checking adapter capabilities first.
        ids_arg = " ".join(shlex.quote(t) for t in test_ids)
        return f"# swebench: harness-driven; ids={ids_arg}; report={report_path}"

    def parse_report(self, report_path: Path) -> RunResult:
        if not report_path.exists():
            return RunResult(returncode=1, error=f"report.json not found at {report_path}")
        try:
            payload = json.loads(report_path.read_text())
        except (OSError, ValueError) as exc:
            return RunResult(
                returncode=1,
                error=f"failed to parse report.json: {type(exc).__name__}: {exc}",
            )
        inner = _unwrap_instance(payload, self.instance_id)
        tests_status = inner.get("tests_status") or {}
        f2p = tests_status.get("FAIL_TO_PASS") or {}
        p2p = tests_status.get("PASS_TO_PASS") or {}
        f2p_success = list(f2p.get("success") or [])
        f2p_failure = list(f2p.get("failure") or [])
        p2p_success = list(p2p.get("success") or [])
        p2p_failure = list(p2p.get("failure") or [])
        outcomes: dict[str, str] = {}
        for nid in f2p_success:
            outcomes[str(nid)] = "passed"
        for nid in p2p_success:
            outcomes[str(nid)] = "passed"
        for nid in f2p_failure:
            outcomes[str(nid)] = "failed"
        for nid in p2p_failure:
            outcomes[str(nid)] = "failed"
        passed = len(f2p_success) + len(p2p_success)
        failed = len(f2p_failure) + len(p2p_failure)
        collected = passed + failed
        # ``resolved`` is the harness's verdict for the entire instance:
        # all FAIL_TO_PASS now pass AND all PASS_TO_PASS still pass. Use
        # it to set returncode so downstream code that only inspects
        # ``returncode`` matches the harness's own opinion.
        resolved = bool(inner.get("resolved", False))
        returncode = 0 if resolved else 1
        return RunResult(
            returncode=returncode,
            passed=passed,
            failed=failed,
            errors=0,
            skipped=0,
            collected=collected,
            outcomes=outcomes,
            report_path=str(report_path),
        )

    def extract_failure_excerpt(self, test_id: str, report_path: Path) -> str:
        if not report_path.exists():
            return ""
        try:
            payload = json.loads(report_path.read_text())
        except (OSError, ValueError):
            return ""
        inner = _unwrap_instance(payload, self.instance_id)
        # The public swebench report.json deliberately stores only
        # pass/fail lists — no per-test error excerpt. Operators who
        # need the failure text read the corresponding test_output.txt
        # in the same log directory; when present, surface it.
        log_dir = report_path.parent
        for candidate in ("test_output.txt", "run_instance.log"):
            log_path = log_dir / candidate
            if not log_path.exists():
                continue
            try:
                text = log_path.read_text(errors="ignore")
            except OSError:
                continue
            if test_id in text:
                lines = text.splitlines()
                for idx, line in enumerate(lines):
                    if test_id in line:
                        start = max(0, idx - 2)
                        end = min(len(lines), idx + 25)
                        return "\n".join(lines[start:end]).strip()
        # Fall back to an opaque "appeared in failure list" hint so the
        # selector can record that a test failed even if the verbose
        # log isn't reachable.
        tests_status = inner.get("tests_status") or {}
        for category in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            failures = list(((tests_status.get(category) or {}).get("failure") or []))
            if test_id in failures:
                return f"Test {test_id} appeared in {category}.failure list."
        return ""

    def infrastructure_paths(self, workspace: Path) -> set[str]:
        result: set[str] = set()
        candidates = _INFRASTRUCTURE_FILENAMES_BY_LANG.get(self.repo_language or "python", set())
        for fname in candidates:
            if (workspace / fname).exists():
                result.add(fname)
        for path in workspace.rglob("conftest.py"):
            try:
                result.add(str(path.relative_to(workspace)))
            except ValueError:
                continue
        return result

    def stub_patterns(self) -> list[str]:
        return list(_STUB_PATTERNS_BY_LANG.get(self.repo_language or "python", []))


# Default registration — defaults to Python, the dominant language for
# classic/Verified. Multilingual callers should construct an instance
# with the per-task ``repo_language`` to get the right stub patterns.
register_adapter(SWEBenchAdapter())
