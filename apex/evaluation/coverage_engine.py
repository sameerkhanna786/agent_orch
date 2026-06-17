"""Branch-coverage engine for in-loop testgen feedback.

Phase I.3 — Coverage-driven targeting. The closed-loop testgen
agent benefits from knowing WHICH lines of the focus file(s) the
current candidate test suite exercises and which it does not. The
unexercised lines are the natural next targets: a bug on an
unexercised branch is invisible to any test in the portfolio, no
matter how high the F2P or mutation-sensitivity scores climb.

Implementation: shells out to the standard ``coverage`` package
(``python -m coverage run --branch ... -m pytest ...``; then
``python -m coverage json``) inside the worktree. Parses the JSON
report, converts the per-file ``missing_lines`` integer list into
contiguous (start, end) ranges, and returns a ``CoverageReport``.

Defensive about missing-tool / no-tests / timeout. When
``coverage`` is not importable in the worktree's interpreter, the
report's ``status`` is ``no_coverage_tool`` and the caller can
silently degrade to "no feedback" rather than crashing.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


_DEFAULT_TIMEOUT_SECONDS = 120.0


@dataclass
class CoverageReport:
    """Per-file uncovered line ranges from a single in-loop coverage pass.

    ``per_file_uncovered_ranges`` maps repo-relative source paths to a
    list of inclusive (start, end) line ranges that the tests did NOT
    execute. Single-line gaps are encoded as (n, n).

    ``status`` is "ok" when the report is usable. Other values:
        - "no_target_source_paths": caller passed an empty file list.
        - "no_test_paths": caller passed no candidate tests.
        - "no_coverage_tool": ``coverage`` package is not importable
          in the worktree's interpreter (silent degrade — caller
          should skip the prompt block).
        - "timeout": pytest run exceeded the wallclock budget.
        - "exception": something else went wrong (see ``error``).
    """

    target_source_paths: list[str]
    per_file_uncovered_ranges: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    per_file_total_lines: dict[str, int] = field(default_factory=dict)
    per_file_covered_lines: dict[str, int] = field(default_factory=dict)
    per_file_total_branches: dict[str, int] = field(default_factory=dict)
    per_file_covered_branches: dict[str, int] = field(default_factory=dict)
    per_file_missing_branches: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    missing_target_source_paths: list[str] = field(default_factory=list)
    overall_coverage_ratio: float = 0.0
    overall_branch_coverage_ratio: float = 0.0
    language: str = "python"
    coverage_backend: str = "coverage.py"
    status: str = "ok"
    error: Optional[str] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_source_paths": list(self.target_source_paths),
            "per_file_uncovered_ranges": {
                path: [list(r) for r in ranges]
                for path, ranges in self.per_file_uncovered_ranges.items()
            },
            "per_file_total_lines": dict(self.per_file_total_lines),
            "per_file_covered_lines": dict(self.per_file_covered_lines),
            "per_file_total_branches": dict(self.per_file_total_branches),
            "per_file_covered_branches": dict(self.per_file_covered_branches),
            "per_file_missing_branches": {
                path: [list(branch) for branch in branches]
                for path, branches in self.per_file_missing_branches.items()
            },
            "missing_target_source_paths": list(self.missing_target_source_paths),
            "overall_coverage_ratio": round(self.overall_coverage_ratio, 4),
            "overall_branch_coverage_ratio": round(
                self.overall_branch_coverage_ratio,
                4,
            ),
            "language": self.language,
            "coverage_backend": self.coverage_backend,
            "status": self.status,
            "metric_status": _coverage_metric_status(self.status),
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 3),
        }


def _python_executable() -> str:
    return sys.executable or "python3"


def _coverage_metric_status(status: str) -> str:
    text = str(status or "").lower()
    if text == "ok":
        return "available"
    if text.startswith("unsupported_language"):
        return "unsupported"
    if text in {"no_coverage_tool", "no_js_coverage_runner"}:
        return "unavailable"
    if text == "timeout":
        return "timeout"
    if text in {"no_coverage_data", "exception"}:
        return "infra_error"
    if text in {"no_target_source_paths", "no_test_paths"}:
        return "unavailable"
    return "unknown"


def _coverage_tool_available(executable: str) -> bool:
    """Cheap import probe — runs ``python -c 'import coverage'`` once.

    The probe runs the same interpreter the actual coverage pass will
    use, so a venv-without-coverage is correctly diagnosed even when
    the test runner's own venv has coverage installed.
    """
    try:
        completed = subprocess.run(
            [executable, "-c", "import coverage"],
            capture_output=True,
            text=True,
            timeout=15.0,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
    return completed.returncode == 0


def _missing_lines_to_ranges(missing: list[int]) -> list[tuple[int, int]]:
    """Compress a sorted list of integers into (start, end) ranges.

    [1, 2, 3, 7, 9, 10] -> [(1, 3), (7, 7), (9, 10)]
    """
    if not missing:
        return []
    ordered = sorted(set(int(n) for n in missing if int(n) > 0))
    ranges: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for n in ordered[1:]:
        if n == prev + 1:
            prev = n
            continue
        ranges.append((start, prev))
        start = prev = n
    ranges.append((start, prev))
    return ranges


def _coverage_path_key(worktree: Path, raw_path: str) -> str:
    """Normalize coverage.py file keys to worktree-relative POSIX paths."""
    path_text = str(raw_path or "").strip()
    if not path_text:
        return ""
    path = Path(path_text)
    try:
        if path.is_absolute():
            return path.resolve().relative_to(worktree.resolve()).as_posix()
    except (OSError, ValueError):
        pass
    return path.as_posix()


def _source_line_count(worktree: Path, repo_relative_path: str) -> int:
    path = Path(repo_relative_path)
    source_path = path if path.is_absolute() else worktree / path
    try:
        if not source_path.is_file():
            return 0
        return len(source_path.read_text(encoding="utf-8", errors="ignore").splitlines())
    except OSError:
        return 0


def evaluate_coverage_in_loop(
    *,
    worktree_path: str | Path,
    test_paths: list[str],
    target_source_paths: list[str],
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    python_executable: Optional[str] = None,
) -> CoverageReport:
    """Run pytest with coverage on the worktree and return uncovered ranges.

    Mirrors the in-loop mutation engine's contract: takes the worktree
    directly (no clone), the agent's candidate test paths, and the
    focus source files. Returns a structured report whose
    ``per_file_uncovered_ranges`` field feeds the iteration prompt.
    """
    target_paths = [str(p).strip() for p in target_source_paths if str(p).strip()]
    test_path_strs = [str(p).strip() for p in test_paths if str(p).strip()]
    report = CoverageReport(target_source_paths=target_paths)
    if not target_paths:
        report.status = "no_target_source_paths"
        return report
    if not test_path_strs:
        report.status = "no_test_paths"
        return report

    worktree = Path(worktree_path)
    interpreter = python_executable or _python_executable()
    started = time.time()
    target_keys = {key for path in target_paths if (key := _coverage_path_key(worktree, path))}

    if not _coverage_tool_available(interpreter):
        report.status = "no_coverage_tool"
        report.duration_seconds = time.time() - started
        return report

    cov_data_file = worktree / ".apex_coverage_in_loop"
    cov_json_file = worktree / ".apex_coverage_in_loop.json"
    # Pre-clean any leftover artifacts from a previous iteration so we
    # don't accidentally read stale results if the new run aborts.
    for stale in (cov_data_file, cov_json_file):
        try:
            if stale.exists():
                stale.unlink()
        except OSError:
            pass

    include_arg = ",".join(target_paths)

    from apex.core.subprocess_utils import build_command_env

    env = build_command_env({"COVERAGE_FILE": str(cov_data_file)})
    # Audit H9: prepend the worktree to PYTHONPATH (don't ``setdefault``
    # — that would silently keep a stale parent value pointing at a
    # different worktree, producing wrong coverage rows).
    parent_pythonpath = env.get("PYTHONPATH", "").strip()
    worktree_str = str(worktree)
    if parent_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join([worktree_str, parent_pythonpath])
    else:
        env["PYTHONPATH"] = worktree_str

    run_cmd = [
        interpreter,
        "-m",
        "coverage",
        "run",
        "--branch",
        f"--include={include_arg}",
        "--data-file",
        str(cov_data_file),
        "-m",
        "pytest",
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
        "--rootdir",
        str(worktree),
        *test_path_strs,
    ]
    try:
        run_completed = subprocess.run(
            run_cmd,
            cwd=str(worktree),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        report.status = "timeout"
        report.error = f"coverage run exceeded {int(timeout_seconds)}s"
        report.duration_seconds = time.time() - started
        return report
    except Exception as exc:  # pragma: no cover — defensive
        report.status = "exception"
        report.error = f"{type(exc).__name__}: {exc}"
        report.duration_seconds = time.time() - started
        return report

    # pytest exit 5 (no tests collected) or fatal collection error
    # leaves no .coverage data. Surface as exception so callers know
    # the report is empty by reason of test-side failure, not absence
    # of coverage.
    if not cov_data_file.exists():
        report.status = "no_coverage_data"
        report.error = (
            f"pytest exit={run_completed.returncode}; "
            f"stderr_tail={(run_completed.stderr or '')[-400:]}"
        )
        report.duration_seconds = time.time() - started
        return report

    json_cmd = [
        interpreter,
        "-m",
        "coverage",
        "json",
        "--data-file",
        str(cov_data_file),
        "-o",
        str(cov_json_file),
    ]
    try:
        json_completed = subprocess.run(
            json_cmd,
            cwd=str(worktree),
            capture_output=True,
            text=True,
            timeout=60.0,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        report.status = "timeout"
        report.error = "coverage json export exceeded 60s"
        report.duration_seconds = time.time() - started
        return report
    except Exception as exc:  # pragma: no cover — defensive
        report.status = "exception"
        report.error = f"{type(exc).__name__}: {exc}"
        report.duration_seconds = time.time() - started
        return report

    no_reported_coverage = not cov_json_file.exists() and "No data to report" in (
        (json_completed.stdout or "") + (json_completed.stderr or "")
    )
    if no_reported_coverage:
        payload = {"files": {}}
    elif json_completed.returncode != 0 or not cov_json_file.exists():
        report.status = "exception"
        report.error = (
            f"coverage json failed exit={json_completed.returncode}; "
            f"stderr_tail={(json_completed.stderr or '')[-400:]}"
        )
        report.duration_seconds = time.time() - started
        return report
    else:
        try:
            payload = json.loads(cov_json_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            report.status = "exception"
            report.error = f"failed to read coverage json: {type(exc).__name__}: {exc}"
            report.duration_seconds = time.time() - started
            return report
        finally:
            # Cleanup: don't pollute the worktree with .coverage state
            for cleanup in (cov_data_file, cov_json_file):
                try:
                    if cleanup.exists():
                        cleanup.unlink()
                except OSError:
                    pass
    if no_reported_coverage:
        for cleanup in (cov_data_file, cov_json_file):
            try:
                if cleanup.exists():
                    cleanup.unlink()
            except OSError:
                pass

    files_section = payload.get("files") or {}
    total_covered = 0
    total_lines = 0
    total_covered_branches = 0
    total_branches = 0
    for raw_path, file_info in files_section.items():
        if not isinstance(file_info, dict):
            continue
        # Normalize against worktree-relative form. coverage.py emits
        # paths exactly as the source was reached at runtime, which
        # is normally relative to PYTHONPATH/CWD — both are the
        # worktree, so paths should match `target_paths` directly.
        norm_path = _coverage_path_key(worktree, str(raw_path or ""))
        if not norm_path:
            continue
        if target_keys and norm_path not in target_keys:
            continue
        summary = file_info.get("summary") or {}
        covered_lines = int(summary.get("covered_lines", 0) or 0)
        num_statements = int(summary.get("num_statements", 0) or 0)
        covered_branches = int(summary.get("covered_branches", 0) or 0)
        num_branches = int(summary.get("num_branches", 0) or 0)
        missing_lines_raw = list(file_info.get("missing_lines") or [])
        missing_branches_raw = list(file_info.get("missing_branches") or [])
        ranges = _missing_lines_to_ranges(missing_lines_raw)
        if ranges:
            report.per_file_uncovered_ranges[norm_path] = ranges
        missing_branches: list[tuple[int, int]] = []
        for branch in missing_branches_raw:
            if not isinstance(branch, (list, tuple)) or len(branch) != 2:
                continue
            try:
                missing_branches.append((int(branch[0]), int(branch[1])))
            except (TypeError, ValueError):
                continue
        if missing_branches:
            report.per_file_missing_branches[norm_path] = missing_branches
        report.per_file_covered_lines[norm_path] = covered_lines
        report.per_file_total_lines[norm_path] = num_statements
        report.per_file_covered_branches[norm_path] = covered_branches
        report.per_file_total_branches[norm_path] = num_branches
        total_covered += covered_lines
        total_lines += num_statements
        total_covered_branches += covered_branches
        total_branches += num_branches

    reported_targets = set(report.per_file_total_lines)
    missing_targets = sorted(target_keys - reported_targets)
    for target in missing_targets:
        line_count = _source_line_count(worktree, target)
        if line_count <= 0:
            continue
        report.missing_target_source_paths.append(target)
        report.per_file_uncovered_ranges[target] = [(1, line_count)]
        report.per_file_covered_lines[target] = 0
        report.per_file_total_lines[target] = line_count
        report.per_file_covered_branches[target] = 0
        report.per_file_total_branches[target] = 0
        total_lines += line_count

    report.overall_coverage_ratio = (total_covered / total_lines) if total_lines > 0 else 0.0
    report.overall_branch_coverage_ratio = (
        (total_covered_branches / total_branches)
        if total_branches > 0
        else (0.0 if report.missing_target_source_paths else 1.0)
    )
    report.duration_seconds = time.time() - started
    return report


def evaluate_coverage_for_language_in_loop(
    *,
    worktree_path: str | Path,
    test_paths: list[str],
    target_source_paths: list[str],
    language: str = "python",
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    python_executable: Optional[str] = None,
) -> CoverageReport:
    """Language-aware coverage dispatcher for in-loop testgen feedback."""
    normalized_language = (language or "").lower()
    if normalized_language in {"", "python", "py", "python3"}:
        report = evaluate_coverage_in_loop(
            worktree_path=worktree_path,
            test_paths=test_paths,
            target_source_paths=target_source_paths,
            timeout_seconds=timeout_seconds,
            python_executable=python_executable,
        )
        report.language = "python"
        report.coverage_backend = "coverage.py"
        return report
    if normalized_language in {"javascript", "js", "jsx", "typescript", "ts", "tsx"}:
        return evaluate_js_ts_coverage_in_loop(
            worktree_path=worktree_path,
            test_paths=test_paths,
            target_source_paths=target_source_paths,
            language=normalized_language,
            timeout_seconds=timeout_seconds,
        )
    report = CoverageReport(
        target_source_paths=[
            str(path).strip() for path in target_source_paths if str(path).strip()
        ],
        language=normalized_language or "unknown",
        coverage_backend="unsupported",
        status=f"unsupported_language:{normalized_language or 'unknown'}",
    )
    return report


def evaluate_js_ts_coverage_in_loop(
    *,
    worktree_path: str | Path,
    test_paths: list[str],
    target_source_paths: list[str],
    language: str = "javascript",
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> CoverageReport:
    """Run Jest/Vitest coverage and map Istanbul JSON to CoverageReport."""
    target_paths = [str(p).strip() for p in target_source_paths if str(p).strip()]
    test_path_strs = [str(p).strip() for p in test_paths if str(p).strip()]
    report = CoverageReport(
        target_source_paths=target_paths,
        language=(language or "javascript").lower(),
        coverage_backend="istanbul",
    )
    if not target_paths:
        report.status = "no_target_source_paths"
        return report
    if not test_path_strs:
        report.status = "no_test_paths"
        return report

    worktree = Path(worktree_path)
    started = time.time()
    runner = _detect_js_ts_coverage_runner(worktree)
    if not runner:
        report.status = "no_js_coverage_runner"
        report.duration_seconds = time.time() - started
        return report

    coverage_dir = worktree / ".apex_js_coverage"
    coverage_final = coverage_dir / "coverage-final.json"
    if coverage_dir.exists():
        for child in coverage_dir.rglob("*"):
            try:
                if child.is_file():
                    child.unlink()
            except OSError:
                pass
    coverage_dir.mkdir(parents=True, exist_ok=True)

    if runner == "vitest":
        command = [
            "npx",
            "vitest",
            "run",
            f"--coverage.reportsDirectory={coverage_dir}",
            "--coverage.reporter=json",
            "--coverage.enabled=true",
            *test_path_strs,
        ]
        report.coverage_backend = "vitest/istanbul"
    else:
        command = [
            "npx",
            "jest",
            "--coverage",
            f"--coverageDirectory={coverage_dir}",
            "--coverageReporters=json",
            "--runInBand",
            *test_path_strs,
        ]
        report.coverage_backend = "jest/istanbul"

    from apex.core.subprocess_utils import build_command_env

    env = build_command_env({"CI": "1"})
    try:
        completed = subprocess.run(
            command,
            cwd=str(worktree),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        report.status = "timeout"
        report.error = f"{runner} coverage exceeded {int(timeout_seconds)}s"
        report.duration_seconds = time.time() - started
        return report
    except FileNotFoundError:
        report.status = "no_js_coverage_tool"
        report.error = "npx not found"
        report.duration_seconds = time.time() - started
        return report
    except Exception as exc:  # pragma: no cover - defensive
        report.status = "exception"
        report.error = f"{type(exc).__name__}: {exc}"
        report.duration_seconds = time.time() - started
        return report

    if completed.returncode != 0 and not coverage_final.exists():
        report.status = "no_coverage_data"
        report.error = (
            f"{runner} coverage exit={completed.returncode}; "
            f"stderr_tail={(completed.stderr or '')[-400:]}"
        )
        report.duration_seconds = time.time() - started
        return report
    if not coverage_final.exists():
        report.status = "no_coverage_data"
        report.error = "coverage-final.json was not produced"
        report.duration_seconds = time.time() - started
        return report
    try:
        payload = json.loads(coverage_final.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.status = "exception"
        report.error = f"failed to read Istanbul coverage json: {type(exc).__name__}: {exc}"
        report.duration_seconds = time.time() - started
        return report

    _populate_report_from_istanbul_payload(report, worktree=worktree, payload=payload)
    report.duration_seconds = time.time() - started
    return report


def _detect_js_ts_coverage_runner(worktree: Path) -> str:
    package_json = worktree / "package.json"
    if not package_json.exists():
        return ""
    try:
        data = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    deps: dict[str, Any] = {}
    for key in ("dependencies", "devDependencies"):
        section = data.get(key) or {}
        if isinstance(section, dict):
            deps.update(section)
    scripts = data.get("scripts") or {}
    script_text = (
        " ".join(str(value) for value in scripts.values()) if isinstance(scripts, dict) else ""
    )
    if "vitest" in deps or "vitest" in script_text:
        return "vitest"
    if "jest" in deps or "jest" in script_text:
        return "jest"
    return ""


def _populate_report_from_istanbul_payload(
    report: CoverageReport,
    *,
    worktree: Path,
    payload: dict[str, Any],
) -> None:
    target_keys = {
        key for path in report.target_source_paths if (key := _coverage_path_key(worktree, path))
    }
    total_covered = 0
    total_lines = 0
    total_covered_branches = 0
    total_branches = 0
    reported_targets: set[str] = set()
    for raw_path, file_info in dict(payload or {}).items():
        if not isinstance(file_info, dict):
            continue
        norm_path = _coverage_path_key(
            worktree,
            str(file_info.get("path") or raw_path or ""),
        )
        if not norm_path or (target_keys and norm_path not in target_keys):
            continue
        reported_targets.add(norm_path)
        statement_map = dict(file_info.get("statementMap") or {})
        statement_counts = dict(file_info.get("s") or {})
        line_counts: dict[int, int] = {}
        for statement_id, raw_location in statement_map.items():
            if not isinstance(raw_location, dict):
                continue
            start = dict(raw_location.get("start") or {})
            end = dict(raw_location.get("end") or {})
            start_line = int(start.get("line") or 0)
            end_line = int(end.get("line") or start_line or 0)
            count = int(statement_counts.get(str(statement_id), 0) or 0)
            for line in range(start_line, end_line + 1):
                if line > 0:
                    line_counts[line] = max(line_counts.get(line, 0), count)
        covered_lines = sum(1 for count in line_counts.values() if count > 0)
        total_file_lines = len(line_counts)
        missing_lines = [line for line, count in line_counts.items() if count <= 0]
        ranges = _missing_lines_to_ranges(missing_lines)
        if ranges:
            report.per_file_uncovered_ranges[norm_path] = ranges
        report.per_file_covered_lines[norm_path] = covered_lines
        report.per_file_total_lines[norm_path] = total_file_lines

        branch_map = dict(file_info.get("branchMap") or {})
        branch_counts = dict(file_info.get("b") or {})
        file_branch_total = 0
        file_branch_covered = 0
        missing_branches: list[tuple[int, int]] = []
        for branch_id, raw_branch in branch_map.items():
            if not isinstance(raw_branch, dict):
                continue
            branch_line = int(dict(raw_branch.get("loc") or {}).get("line") or 0)
            locations = list(raw_branch.get("locations") or [])
            counts = list(branch_counts.get(str(branch_id), []) or [])
            file_branch_total += len(locations)
            for index, location in enumerate(locations):
                count = int(counts[index] or 0) if index < len(counts) else 0
                target_line = int(dict(location or {}).get("start", {}).get("line") or branch_line)
                if count > 0:
                    file_branch_covered += 1
                elif branch_line > 0 and target_line > 0:
                    missing_branches.append((branch_line, target_line))
        if missing_branches:
            report.per_file_missing_branches[norm_path] = missing_branches
        report.per_file_total_branches[norm_path] = file_branch_total
        report.per_file_covered_branches[norm_path] = file_branch_covered

        total_covered += covered_lines
        total_lines += total_file_lines
        total_covered_branches += file_branch_covered
        total_branches += file_branch_total

    for target in sorted(target_keys - reported_targets):
        line_count = _source_line_count(worktree, target)
        if line_count <= 0:
            continue
        report.missing_target_source_paths.append(target)
        report.per_file_uncovered_ranges[target] = [(1, line_count)]
        report.per_file_covered_lines[target] = 0
        report.per_file_total_lines[target] = line_count
        report.per_file_covered_branches[target] = 0
        report.per_file_total_branches[target] = 0
        total_lines += line_count

    report.overall_coverage_ratio = (total_covered / total_lines) if total_lines > 0 else 0.0
    report.overall_branch_coverage_ratio = (
        (total_covered_branches / total_branches)
        if total_branches > 0
        else (0.0 if report.missing_target_source_paths else 1.0)
    )
