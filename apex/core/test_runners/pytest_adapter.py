"""Pytest adapter — wraps the existing wrapper-script + plugin pathway.

Heavy lifting (the per-rollout wrapper that bypasses pytest auto-discovery
and the plugin that filters collection to expected IDs) is already
implemented in ``apex/core/_apex_run_expected_ids.py`` and
``apex/core/_apex_expected_ids_filter.py``. This adapter exposes those
through the language-neutral protocol.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from . import RunResult, register_adapter

_PYTEST_INFRASTRUCTURE_FILENAMES = {
    "conftest.py",
    "pytest.ini",
}
_PYTEST_DUAL_USE_INFRASTRUCTURE_SECTIONS = {
    "pyproject.toml": ("[tool.pytest.ini_options]",),
    "setup.cfg": ("[tool:pytest]", "[pytest]"),
    "tox.ini": ("[pytest]",),
}
_PYTEST_INFRASTRUCTURE_PATH_MARKERS = (
    "/conftest.py",
    "/__init__.py",  # tests-tree __init__.py defines collection
)
_PYTEST_STUB_PATTERNS = [
    r"^\s*pass\s*$",
    r"^\s*\.\.\.\s*$",
    r"^\s*return\s+None\s*$",
    r"^\s*raise\s+NotImplementedError",
]


class PytestAdapter:
    name = "pytest"
    language = "python"

    def list_tests(self, workspace: Path, env: dict[str, str]) -> set[str]:
        """Collect test nodeids without running them.

        Uses ``pytest --collect-only -q --co-only`` (the ``-q`` keeps
        output to one nodeid per line; ``--co-only`` is a no-op flag we
        keep so the command shape stays grep-friendly in logs). Falls
        back to an empty set on any error so callers treat absence as
        "unknown" rather than "zero".
        """
        executable = env.get("VIRTUAL_ENV")
        if executable:
            python = str(Path(executable) / "bin" / "python")
        else:
            python = shutil.which("python") or "python"
        try:
            result = subprocess.run(
                [
                    python,
                    "-m",
                    "pytest",
                    "--collect-only",
                    "-q",
                    "--no-header",
                    "--continue-on-collection-errors",
                ],
                cwd=str(workspace),
                env={**env},
                capture_output=True,
                text=True,
                timeout=180,
            )
        except (subprocess.TimeoutExpired, OSError):
            return set()
        nodeids: set[str] = set()
        for line in (result.stdout or "").splitlines():
            line = line.strip()
            if not line or line.startswith("=") or line.startswith("-"):
                continue
            if "::" in line and "/" in line.split("::", 1)[0]:
                nodeids.add(line)
        return nodeids

    def build_run_command(
        self,
        workspace: Path,
        test_ids: list[str],
        report_path: Path,
        *,
        executable: Optional[str] = None,
    ) -> str:
        """Build the shell command. The caller is expected to have already
        staged the wrapper + ids file via ``_stage_expected_ids_filter``.

        We don't re-stage here to avoid two writes — the existing
        ``commit0_benchmark._build_test_command`` is the integration
        point; this method is for *new* call sites that want a pytest
        invocation without going through that path.
        """
        python = executable or "python"
        ids_arg = " ".join(shlex.quote(t) for t in test_ids)
        return (
            f"{shlex.quote(python)} -m pytest {ids_arg} "
            f"--json-report --json-report-file={shlex.quote(str(report_path))} "
            f"--continue-on-collection-errors --cache-clear -q"
        )

    def parse_report(self, report_path: Path) -> RunResult:
        if not report_path.exists():
            return RunResult(returncode=1)
        try:
            payload = json.loads(report_path.read_text())
        except (OSError, ValueError):
            return RunResult(returncode=1)
        # Delegate per-test outcome extraction to the canonical helper —
        # it correctly inspects ``call.outcome`` to surface xfailed /
        # xpassed cases that the top-level ``test.outcome`` collapses
        # into ``passed`` or ``failed``.
        from ..pytest_report_utils import extract_pytest_report_outcomes

        outcomes = extract_pytest_report_outcomes(payload.get("tests") or [])
        summary = payload.get("summary") or {}
        return RunResult(
            returncode=int(payload.get("exitcode", 0) or 0),
            passed=int(summary.get("passed") or 0)
            + int(summary.get("xpassed") or 0)
            + int(summary.get("xfailed") or 0),
            failed=int(summary.get("failed") or 0),
            errors=int(summary.get("error") or 0) + int(summary.get("errors") or 0),
            skipped=int(summary.get("skipped") or 0),
            collected=int(summary.get("collected") or 0),
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
        for test in payload.get("tests") or []:
            if test.get("nodeid") != test_id:
                continue
            for phase_key in ("call", "setup", "teardown"):
                phase = test.get(phase_key) or {}
                if not isinstance(phase, dict):
                    continue
                longrepr = phase.get("longrepr")
                if isinstance(longrepr, str) and longrepr.strip():
                    return longrepr.strip()
                crash = phase.get("crash") or {}
                if isinstance(crash, dict):
                    msg = crash.get("message")
                    if isinstance(msg, str) and msg.strip():
                        return msg.strip()
        return ""

    def infrastructure_paths(self, workspace: Path) -> set[str]:
        result: set[str] = set()
        for path in workspace.rglob("conftest.py"):
            try:
                result.add(str(path.relative_to(workspace)))
            except ValueError:
                continue
        for fname in _PYTEST_INFRASTRUCTURE_FILENAMES:
            candidate = workspace / fname
            if candidate.exists():
                result.add(fname)
        for fname, pytest_sections in _PYTEST_DUAL_USE_INFRASTRUCTURE_SECTIONS.items():
            candidate = workspace / fname
            if not candidate.exists():
                continue
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            # Python packaging files are dual-use; only pytest-specific sections
            # control the test universe and belong in the infrastructure gate.
            if any(section in text for section in pytest_sections):
                result.add(fname)
        return result

    def stub_patterns(self) -> list[str]:
        return list(_PYTEST_STUB_PATTERNS)


register_adapter(PytestAdapter())
