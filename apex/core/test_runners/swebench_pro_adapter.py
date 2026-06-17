"""SWE-Bench Pro adapter — wraps the per-task Docker parser output.

SWE-Bench Pro tasks ship a per-task ``parser.py`` inside the Docker
image; the harness runs the task's test command in the container, the
parser converts test output → ``output.json`` with this schema::

    {
      "passed_tests":  [...],
      "failed_tests":  [...],
      "skipped_tests": [...],
      "error_tests":   [...],
      "test_outputs":  { "<test_id>": "verbatim failure text", ... }   # optional
    }

This adapter doesn't drive container execution (the existing
``SWEBenchProHarness`` already does that). It only exposes the
TestRunnerAdapter protocol so the solver self-check, residual followup,
and selector hooks can read SWE-Bench Pro results uniformly. ``language``
defaults to ``polyglot`` because a single Pro task can span any of
Python / JS / TS / Go / Rust / Java / C# — per-task language is read
from the dataset's ``repo_language`` field, not from the adapter.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Optional

from . import RunResult, register_adapter

_INFRASTRUCTURE_FILENAMES_BY_LANG: dict[str, set[str]] = {
    "python": {"conftest.py", "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml"},
    "javascript": {
        "jest.config.js",
        "jest.config.ts",
        "vitest.config.js",
        "vitest.config.ts",
        "package.json",
        "tsconfig.json",
        "babel.config.js",
        ".babelrc",
    },
    "typescript": {
        "jest.config.js",
        "jest.config.ts",
        "vitest.config.js",
        "vitest.config.ts",
        "package.json",
        "tsconfig.json",
        "babel.config.js",
        ".babelrc",
    },
    "go": {"go.mod", "go.sum"},
    "rust": {"Cargo.toml", "Cargo.lock"},
    "java": {"pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle"},
    "ruby": {"Gemfile", "Gemfile.lock", ".rspec"},
}

_STUB_PATTERNS_BY_LANG: dict[str, list[str]] = {
    "python": [
        r"^\s*pass\s*$",
        r"^\s*\.\.\.\s*$",
        r"^\s*return\s+None\s*$",
        r"^\s*raise\s+NotImplementedError",
    ],
    "javascript": [
        r"throw new Error\((['\"]).*not.{0,5}implemented.*\1\)",
        r"throw new Error\((['\"]).*TODO.*\1\)",
        r"return undefined;",
        r"//\s*TODO[: ]",
    ],
    "typescript": [
        r"throw new Error\((['\"]).*not.{0,5}implemented.*\1\)",
        r"throw new Error\((['\"]).*TODO.*\1\)",
        r"return undefined;",
        r"//\s*TODO[: ]",
    ],
    "go": [
        r'panic\("not implemented"\)',
        r'panic\("TODO"\)',
        r"//\s*TODO[: ]",
    ],
    "rust": [
        r"unimplemented!\(",
        r"todo!\(",
        r'panic!\((["\']).*not.{0,5}implemented.*\1\)',
        r"//\s*TODO[: ]",
    ],
    "java": [
        r"throw new UnsupportedOperationException\(",
        r"throw new RuntimeException\((['\"]).*not.{0,5}implemented.*\1\)",
        r"throw new NotImplementedError\(",
        r"//\s*TODO[: ]",
    ],
    "ruby": [
        r"raise NotImplementedError",
        r"#\s*TODO[: ]",
    ],
}


_LANGUAGE_ALIASES = {
    "py": "python",
    "python3": "python",
    "js": "javascript",
    "jsx": "javascript",
    "node": "javascript",
    "nodejs": "javascript",
    "mjs": "javascript",
    "cjs": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "c#": "java",  # JUnit-style structure; close enough for infra files
    "csharp": "java",
}


def _normalize_language(language: Optional[str]) -> str:
    if not language:
        return ""
    norm = language.strip().lower()
    return _LANGUAGE_ALIASES.get(norm, norm)


class SWEBenchProAdapter:
    name = "swebench-pro"
    language = "polyglot"

    def __init__(self, repo_language: Optional[str] = None) -> None:
        self.repo_language = _normalize_language(repo_language)

    def list_tests(self, workspace: Path, env: dict[str, str]) -> set[str]:
        # Test discovery for SWE-Bench Pro happens via the dataset's
        # selected_test_files_to_run + per-language test command shape.
        # We don't re-derive it here; the harness calls language-aware
        # helpers in SWEBenchProHarness directly. Returning empty is the
        # honest signal for "not handled by this adapter."
        return set()

    def build_run_command(
        self,
        workspace: Path,
        test_ids: list[str],
        report_path: Path,
        *,
        executable: Optional[str] = None,
    ) -> str:
        # Same caveat as list_tests: the actual run command is built by
        # SWEBenchProHarness (Docker entryscript). Surface a no-op
        # placeholder so the protocol contract is satisfied.
        ids_arg = " ".join(shlex.quote(t) for t in test_ids)
        return f"# swebench-pro: harness-driven; ids={ids_arg}; report={report_path}"

    def parse_report(self, report_path: Path) -> RunResult:
        if not report_path.exists():
            return RunResult(returncode=1)
        try:
            payload = json.loads(report_path.read_text())
        except (OSError, ValueError):
            return RunResult(returncode=1)
        outcomes: dict[str, str] = {}
        for nid in payload.get("passed_tests") or []:
            outcomes[str(nid)] = "passed"
        for nid in payload.get("failed_tests") or []:
            outcomes[str(nid)] = "failed"
        for nid in payload.get("error_tests") or []:
            outcomes[str(nid)] = "error"
        for nid in payload.get("skipped_tests") or []:
            outcomes[str(nid)] = "skipped"
        passed = len(payload.get("passed_tests") or [])
        failed = len(payload.get("failed_tests") or [])
        errors = len(payload.get("error_tests") or [])
        skipped = len(payload.get("skipped_tests") or [])
        collected = passed + failed + errors + skipped
        return RunResult(
            returncode=0 if (failed == 0 and errors == 0) else 1,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
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
        # SWE-Bench Pro task parsers may surface per-test verbatim
        # outputs under either ``test_outputs`` (string per id) or
        # ``test_results`` (richer structure with ``message``).
        outputs = payload.get("test_outputs") or {}
        if isinstance(outputs, dict):
            value = outputs.get(test_id)
            if isinstance(value, str) and value.strip():
                return value.strip()
        results = payload.get("test_results") or []
        if isinstance(results, list):
            for entry in results:
                if not isinstance(entry, dict):
                    continue
                if entry.get("test_id") != test_id and entry.get("name") != test_id:
                    continue
                msg = entry.get("message") or entry.get("longrepr") or entry.get("stdout")
                if isinstance(msg, str) and msg.strip():
                    return msg.strip()
        # Fall back to the raw stdout/stderr blocks if the parser
        # bundled them instead of per-test text.
        for key in ("stdout", "stderr", "output"):
            value = payload.get(key)
            if isinstance(value, str) and test_id in value:
                # Slice ~25 lines around the test_id mention.
                lines = value.splitlines()
                for idx, line in enumerate(lines):
                    if test_id in line:
                        start = max(0, idx - 2)
                        end = min(len(lines), idx + 25)
                        return "\n".join(lines[start:end]).strip()
        return ""

    def infrastructure_paths(self, workspace: Path) -> set[str]:
        result: set[str] = set()
        candidates = _INFRASTRUCTURE_FILENAMES_BY_LANG.get(self.repo_language or "python", set())
        for fname in candidates:
            if (workspace / fname).exists():
                result.add(fname)
        # Common test-fixture / setup directories.
        for path in workspace.rglob("conftest.py"):
            try:
                result.add(str(path.relative_to(workspace)))
            except ValueError:
                continue
        for marker_dir in ("__mocks__", "testdata"):
            for path in workspace.rglob(marker_dir):
                if path.is_dir():
                    try:
                        result.add(str(path.relative_to(workspace)))
                    except ValueError:
                        continue
        return result

    def stub_patterns(self) -> list[str]:
        return list(_STUB_PATTERNS_BY_LANG.get(self.repo_language or "python", []))


# Default registration uses Python as the language fallback. Callers
# that know the per-task ``repo_language`` should construct their own
# instance to override (and the orchestrator's adapter resolution does
# this when a SWEBenchProTask is in scope).
register_adapter(SWEBenchProAdapter())
