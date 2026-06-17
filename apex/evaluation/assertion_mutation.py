"""Dynamic assertion-effect checks for generated tests.

Static quality checks catch obvious weak tests. This module adds a cheap
execution check: invert Python ``assert`` statements in generated test files,
rerun the suite, and flag suites that still pass. If a suite passes after its
assertions are inverted, those assertions are not affecting the outcome.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .f2p_oracle import _run_result_to_dict, _run_tests_on_paths

_ASSERT_LINE_RE = re.compile(r"^(\s*)assert\b(.+)$")


@dataclass
class AssertionMutationReport:
    status: str
    language: str = "python"
    test_paths: list[str] = field(default_factory=list)
    mutated_assertion_count: int = 0
    survived: bool = False
    original_run: dict[str, Any] = field(default_factory=dict)
    mutated_run: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_seconds: float = 0.0

    @property
    def assertion_effective(self) -> bool:
        return self.status == "ok" and self.mutated_assertion_count > 0 and not self.survived

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "language": self.language,
            "test_paths": list(self.test_paths),
            "mutated_assertion_count": self.mutated_assertion_count,
            "survived": self.survived,
            "assertion_effective": self.assertion_effective,
            "original_run": dict(self.original_run),
            "mutated_run": dict(self.mutated_run),
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 3),
        }


def evaluate_assertion_effect_in_loop(
    *,
    worktree_path: str | Path,
    test_paths: list[str],
    language: str = "python",
    timeout_seconds: float = 60.0,
    python_executable: Optional[str] = None,
) -> AssertionMutationReport:
    """Invert Python assertions and verify tests fail.

    Returns ``unsupported_language`` for non-Python callers; the check is
    intentionally conservative until language-specific AST/transforms are
    added for JS/TS/etc.
    """
    started = time.time()
    normalized_language = (language or "").lower()
    paths = [str(path).strip() for path in test_paths or [] if str(path).strip()]
    report = AssertionMutationReport(
        status="ok",
        language=normalized_language or "python",
        test_paths=paths,
    )
    if normalized_language not in {"", "python", "py", "python3"}:
        report.status = f"unsupported_language:{normalized_language or 'unknown'}"
        report.duration_seconds = time.time() - started
        return report
    if not paths:
        report.status = "no_test_paths"
        report.duration_seconds = time.time() - started
        return report

    worktree = Path(worktree_path)
    backups: dict[Path, str] = {}
    try:
        original = _run_tests_on_paths(
            adapter=None,
            sandbox_dir=worktree,
            test_paths=paths,
            timeout_seconds=timeout_seconds,
            python_executable=python_executable,
        )
        report.original_run = _run_result_to_dict(original)
        if not _run_fully_passed(original):
            report.status = f"baseline_{original.status}"
            report.duration_seconds = time.time() - started
            return report

        mutated_count = 0
        for rel_path in paths:
            if "::" in rel_path:
                rel_path = rel_path.partition("::")[0]
            if not rel_path.endswith(".py"):
                continue
            path = worktree / rel_path
            if not path.is_file():
                continue
            original_text = path.read_text(encoding="utf-8", errors="ignore")
            mutated_text, count = _invert_python_assert_lines(original_text)
            if count <= 0 or mutated_text == original_text:
                continue
            backups[path] = original_text
            path.write_text(mutated_text, encoding="utf-8")
            mutated_count += count

        report.mutated_assertion_count = mutated_count
        if mutated_count <= 0:
            report.status = "no_assertions_mutated"
            report.duration_seconds = time.time() - started
            return report

        mutated = _run_tests_on_paths(
            adapter=None,
            sandbox_dir=worktree,
            test_paths=paths,
            timeout_seconds=timeout_seconds,
            python_executable=python_executable,
        )
        report.mutated_run = _run_result_to_dict(mutated)
        report.survived = _run_fully_passed(mutated)
        report.duration_seconds = time.time() - started
        return report
    except Exception as exc:  # pragma: no cover - defensive
        report.status = "exception"
        report.error = f"{type(exc).__name__}: {exc}"
        report.duration_seconds = time.time() - started
        return report
    finally:
        for path, original_text in backups.items():
            try:
                path.write_text(original_text, encoding="utf-8")
            except OSError:
                pass


def _invert_python_assert_lines(source: str) -> tuple[str, int]:
    """Replace simple one-line ``assert expr`` statements with ``assert not (expr)``."""
    lines = source.splitlines(keepends=True)
    changed = 0
    out: list[str] = []
    for line in lines:
        newline = ""
        body = line
        if line.endswith("\r\n"):
            body = line[:-2]
            newline = "\r\n"
        elif line.endswith("\n"):
            body = line[:-1]
            newline = "\n"
        match = _ASSERT_LINE_RE.match(body)
        if match and "\\" not in body:
            indent = match.group(1)
            expression = match.group(2).strip()
            out.append(f"{indent}assert not ({expression})  # apex assertion mutation{newline}")
            changed += 1
            continue
        out.append(line)
    return "".join(out), changed


def _run_fully_passed(result: Any) -> bool:
    if getattr(result, "status", "") != "ok":
        return False
    returncode = getattr(result, "returncode", 0)
    if isinstance(returncode, int) and returncode != 0:
        return False
    statuses = {
        str(status).strip().lower()
        for status in dict(getattr(result, "per_test_status", {}) or {}).values()
    }
    return not (statuses & {"fail", "error"})
