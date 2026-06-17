"""Fast local dual-state benchmark for testcase generation.

This module gives Apex a cheap F2P-first validation target before running
large benchmark suites. Each task supplies a broken repo, a fixed repo, native
test command, issue/contract metadata, and candidate test artifacts. The runner
materializes the artifacts into both repos, executes the same command under a
strict task deadline, and classifies the candidate suite as F2P/P2P/F2F/P2F.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Optional

F2P = "F2P"
P2P = "P2P"
F2F = "F2F"
P2F = "P2F"
NO_TESTS = "NO_TESTS"
NO_TESTS_COLLECTED = "NO_TESTS_COLLECTED"
TIMEOUT = "TIMEOUT"
ERROR = "ERROR"


_COPY_IGNORE_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
    "target",
    "dist",
    "build",
}


@dataclass(frozen=True)
class CandidateTestArtifact:
    """A generated test file to materialize into a benchmark repo."""

    path: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "content": self.content}


@dataclass(frozen=True)
class LocalDualStateTask:
    """Benchmark-agnostic local testcase-generation task."""

    task_id: str
    language: str
    issue_text: str
    broken_repo: Path
    fixed_repo: Path
    test_command: str
    candidate_tests: tuple[CandidateTestArtifact, ...] = ()
    expected_fixed_behavior: str = ""
    expected_broken_failure_mode: str = ""
    authoritative_source: str = ""
    public_surface: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def contract_dict(self) -> dict[str, str]:
        return {
            "expected_fixed_behavior": self.expected_fixed_behavior,
            "expected_broken_failure_mode": self.expected_broken_failure_mode,
            "authoritative_source": self.authoritative_source,
            "public_surface": self.public_surface,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "language": self.language,
            "issue_text": self.issue_text,
            "broken_repo": str(self.broken_repo),
            "fixed_repo": str(self.fixed_repo),
            "test_command": self.test_command,
            "candidate_tests": [artifact.to_dict() for artifact in self.candidate_tests],
            "contract": self.contract_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass
class CommandRunResult:
    """One command execution on either the broken or fixed repo."""

    side: str
    command: str
    returncode: Optional[int] = None
    timed_out: bool = False
    duration_seconds: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: str = ""

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.error

    @property
    def failed(self) -> bool:
        return self.returncode not in (None, 0) and not self.timed_out

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "command": self.command,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "duration_seconds": round(self.duration_seconds, 3),
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "error": self.error,
        }


@dataclass
class LocalDualStateTaskResult:
    """Classification and diagnostics for one local dual-state task."""

    task_id: str
    language: str
    classification: str
    status: str
    broken: CommandRunResult
    fixed: CommandRunResult
    candidate_test_paths: list[str] = field(default_factory=list)
    failure_excerpt: str = ""
    contract: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    @property
    def fixed_passed(self) -> bool:
        return self.fixed.passed

    @property
    def reliable_f2p(self) -> bool:
        return self.classification == F2P and self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "language": self.language,
            "classification": self.classification,
            "status": self.status,
            "candidate_test_paths": list(self.candidate_test_paths),
            "failure_excerpt": self.failure_excerpt,
            "contract": dict(self.contract),
            "metadata": dict(self.metadata),
            "broken": self.broken.to_dict(),
            "fixed": self.fixed.to_dict(),
            "duration_seconds": round(self.duration_seconds, 3),
        }


@dataclass
class LocalDualStateBenchmarkReport:
    """Aggregate local dual-state benchmark report."""

    results: list[LocalDualStateTaskResult]
    total_duration_seconds: float = 0.0
    skipped_languages: dict[str, str] = field(default_factory=dict)
    candidate_source: str = "unknown"

    @property
    def task_count(self) -> int:
        return len(self.results)

    @property
    def class_counts(self) -> dict[str, int]:
        return dict(Counter(result.classification for result in self.results))

    @property
    def fixed_pass_count(self) -> int:
        return sum(1 for result in self.results if result.fixed_passed)

    @property
    def reliable_f2p_count(self) -> int:
        return sum(1 for result in self.results if result.reliable_f2p)

    @property
    def timeout_count(self) -> int:
        return sum(1 for result in self.results if result.classification == TIMEOUT)

    @property
    def runnable_rate(self) -> float:
        if not self.results:
            return 0.0
        runnable = sum(1 for result in self.results if result.fixed.returncode is not None)
        return runnable / len(self.results)

    @property
    def fixed_pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.fixed_pass_count / len(self.results)

    @property
    def reliable_f2p_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.reliable_f2p_count / len(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_kind": "apex_local_dual_state_testgen",
            "task_count": self.task_count,
            "class_counts": self.class_counts,
            "fixed_pass_count": self.fixed_pass_count,
            "fixed_pass_rate": round(self.fixed_pass_rate, 4),
            "reliable_f2p_count": self.reliable_f2p_count,
            "reliable_f2p_rate": round(self.reliable_f2p_rate, 4),
            "runnable_rate": round(self.runnable_rate, 4),
            "timeout_count": self.timeout_count,
            "skipped_languages": dict(self.skipped_languages),
            "candidate_source": self.candidate_source,
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "results": [result.to_dict() for result in self.results],
        }


def run_local_dual_state_benchmark(
    tasks: Iterable[LocalDualStateTask],
    *,
    output_dir: str | Path,
    task_timeout_seconds: float = 30.0,
    jobs: int = 1,
    skipped_languages: Optional[dict[str, str]] = None,
    candidate_source: str = "unknown",
) -> LocalDualStateBenchmarkReport:
    """Run local dual-state tasks with optional task-level parallelism."""

    started_at = time.monotonic()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    task_list = list(tasks)
    max_workers = max(1, int(jobs or 1))
    results: list[LocalDualStateTaskResult] = []

    if max_workers == 1:
        for task in task_list:
            results.append(
                run_local_dual_state_task(
                    task,
                    output_dir=output_root / "tasks" / task.task_id,
                    task_timeout_seconds=task_timeout_seconds,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_local_dual_state_task,
                    task,
                    output_dir=output_root / "tasks" / task.task_id,
                    task_timeout_seconds=task_timeout_seconds,
                ): task.task_id
                for task in task_list
            }
            for future in as_completed(futures):
                results.append(future.result())
        task_order = {task.task_id: index for index, task in enumerate(task_list)}
        results.sort(key=lambda result: task_order.get(result.task_id, 0))

    report = LocalDualStateBenchmarkReport(
        results=results,
        total_duration_seconds=time.monotonic() - started_at,
        skipped_languages=dict(skipped_languages or {}),
        candidate_source=candidate_source,
    )
    report_path = output_root / "benchmark_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return report


def generate_apex_default_candidate_tasks(
    tasks: Iterable[LocalDualStateTask],
    *,
    output_dir: str | Path,
) -> list[LocalDualStateTask]:
    """Use Apex's default test generator to create candidates for tasks.

    This is intentionally optional: the built-in oracle candidates keep the
    evaluator deterministic, while ``apex-default`` makes the same fixtures a
    real fast testcase-generation benchmark for Apex itself.
    """

    from apex._default_generators import default_test_generator

    output_root = Path(output_dir)
    generation_root = output_root / "generation"
    generation_root.mkdir(parents=True, exist_ok=True)
    generated_tasks: list[LocalDualStateTask] = []

    for task in tasks:
        task_generation_dir = generation_root / task.task_id
        worktree = task_generation_dir / "broken_repo"
        if task_generation_dir.exists():
            shutil.rmtree(task_generation_dir, ignore_errors=True)
        task_generation_dir.mkdir(parents=True, exist_ok=True)
        _copy_repo(task.broken_repo, worktree)
        raw_artifacts = default_test_generator(
            worktree,
            _build_generation_prompt(task),
        )
        artifacts = tuple(
            CandidateTestArtifact(
                path=str(raw.get("path") or "").strip(),
                content=str(raw.get("content") or ""),
            )
            for raw in raw_artifacts
            if isinstance(raw, dict)
            and str(raw.get("path") or "").strip()
            and str(raw.get("content") or "")
        )
        generated_payload = {
            "task": task.to_dict(),
            "raw_artifacts": raw_artifacts,
            "candidate_artifact_count": len(artifacts),
        }
        generated_path = task_generation_dir / "generated_artifacts.json"
        generated_path.write_text(
            json.dumps(generated_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        metadata = dict(task.metadata)
        metadata.update(
            {
                "candidate_source": "apex-default",
                "generated_artifact_count": len(artifacts),
                "generated_artifacts_path": str(generated_path),
            }
        )
        generated_tasks.append(replace(task, candidate_tests=artifacts, metadata=metadata))

    return generated_tasks


def run_local_dual_state_task(
    task: LocalDualStateTask,
    *,
    output_dir: str | Path,
    task_timeout_seconds: float = 30.0,
) -> LocalDualStateTaskResult:
    """Materialize and evaluate one candidate suite against broken/fixed repos."""

    started_at = time.monotonic()
    output_root = Path(output_dir)
    if output_root.exists():
        shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)

    candidate_paths = [artifact.path for artifact in task.candidate_tests]
    empty_run = CommandRunResult(side="not_run", command=task.test_command)
    if not task.candidate_tests:
        result = LocalDualStateTaskResult(
            task_id=task.task_id,
            language=task.language,
            classification=NO_TESTS,
            status="no_candidate_tests",
            broken=replace(empty_run, side="broken"),
            fixed=replace(empty_run, side="fixed"),
            candidate_test_paths=candidate_paths,
            contract=task.contract_dict(),
            metadata=dict(task.metadata),
            duration_seconds=time.monotonic() - started_at,
        )
        _write_task_result(output_root, result)
        return result

    work_root = output_root / "workspaces"
    broken_worktree = work_root / "broken"
    fixed_worktree = work_root / "fixed"
    try:
        _copy_repo(task.broken_repo, broken_worktree)
        _copy_repo(task.fixed_repo, fixed_worktree)
        _materialize_candidate_tests(broken_worktree, task.candidate_tests)
        _materialize_candidate_tests(fixed_worktree, task.candidate_tests)
    except Exception as exc:  # pragma: no cover - defensive filesystem guard
        result = LocalDualStateTaskResult(
            task_id=task.task_id,
            language=task.language,
            classification=ERROR,
            status="setup_error",
            broken=replace(empty_run, side="broken", error=str(exc)),
            fixed=replace(empty_run, side="fixed", error=str(exc)),
            candidate_test_paths=candidate_paths,
            failure_excerpt=str(exc),
            contract=task.contract_dict(),
            metadata=dict(task.metadata),
            duration_seconds=time.monotonic() - started_at,
        )
        _write_task_result(output_root, result)
        return result

    deadline = time.monotonic() + max(0.1, float(task_timeout_seconds))
    broken_run = _run_command_with_deadline(
        command=task.test_command,
        cwd=broken_worktree,
        side="broken",
        deadline=deadline,
    )
    fixed_run = _run_command_with_deadline(
        command=task.test_command,
        cwd=fixed_worktree,
        side="fixed",
        deadline=deadline,
    )
    classification, status = _classify_task_runs(broken_run, fixed_run)
    result = LocalDualStateTaskResult(
        task_id=task.task_id,
        language=task.language,
        classification=classification,
        status=status,
        broken=broken_run,
        fixed=fixed_run,
        candidate_test_paths=candidate_paths,
        failure_excerpt=_select_failure_excerpt(classification, broken_run, fixed_run),
        contract=task.contract_dict(),
        metadata=dict(task.metadata),
        duration_seconds=time.monotonic() - started_at,
    )
    _write_task_result(output_root, result)
    if classification == TIMEOUT:
        (output_root / "timeout_checkpoint.json").write_text(
            json.dumps(result.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def build_builtin_local_dual_state_tasks(
    root_dir: str | Path,
    *,
    languages: Iterable[str] = ("python", "javascript", "go"),
) -> tuple[list[LocalDualStateTask], dict[str, str]]:
    """Create small built-in dual-state fixtures for fast local validation."""

    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    normalized = [_normalize_language(language) for language in languages]
    tasks: list[LocalDualStateTask] = []
    skipped: dict[str, str] = {}

    for language in normalized:
        if language == "python":
            tasks.append(_build_python_clamp_task(root))
        elif language == "javascript":
            if shutil.which("node") is None:
                skipped[language] = "node_not_available"
            else:
                tasks.append(_build_javascript_clamp_task(root))
        elif language == "go":
            if shutil.which("go") is None:
                skipped[language] = "go_not_available"
            else:
                tasks.append(_build_go_clamp_task(root))
        else:
            skipped[language] = "unsupported_builtin_fixture"

    return tasks, skipped


def _build_generation_prompt(task: LocalDualStateTask) -> str:
    contract = task.contract_dict()
    return (
        f"{task.issue_text}\n\n"
        "Generate regression tests for this bug. The tests should fail on the "
        "current repo state and pass on the fixed behavior.\n\n"
        "# Contract\n"
        f"- Language: {task.language}\n"
        f"- Public surface: {contract['public_surface']}\n"
        f"- Expected fixed behavior: {contract['expected_fixed_behavior']}\n"
        f"- Expected broken failure mode: {contract['expected_broken_failure_mode']}\n"
        f"- Authoritative source: {contract['authoritative_source']}\n\n"
        "# Native test command\n"
        f"{task.test_command}\n"
    )


def _write_task_result(output_root: Path, result: LocalDualStateTaskResult) -> None:
    (output_root / "task_result.json").write_text(
        json.dumps(result.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )


def _copy_repo(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"repo does not exist: {src}")
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, ignore=_copy_ignore)


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in _COPY_IGNORE_NAMES}


def _materialize_candidate_tests(
    repo: Path,
    artifacts: Iterable[CandidateTestArtifact],
) -> None:
    repo = repo.resolve()
    for artifact in artifacts:
        relative = Path(artifact.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe candidate test path: {artifact.path}")
        target = (repo / relative).resolve()
        try:
            target.relative_to(repo)
        except ValueError:
            raise ValueError(f"candidate test path escapes repo: {artifact.path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(artifact.content, encoding="utf-8")


def _run_command_with_deadline(
    *,
    command: str,
    cwd: Path,
    side: str,
    deadline: float,
) -> CommandRunResult:
    started_at = time.monotonic()
    remaining = max(0.0, deadline - started_at)
    if remaining <= 0.0:
        return CommandRunResult(
            side=side,
            command=command,
            returncode=124,
            timed_out=True,
            duration_seconds=0.0,
            error="deadline exceeded before command started",
        )

    from apex.core.subprocess_utils import build_command_env, terminate_process_tree

    env = build_command_env()
    process: Optional[subprocess.Popen[str]] = None
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        stdout, stderr = process.communicate(timeout=remaining)
        return CommandRunResult(
            side=side,
            command=command,
            returncode=process.returncode,
            duration_seconds=time.monotonic() - started_at,
            stdout_tail=_tail(stdout),
            stderr_tail=_tail(stderr),
        )
    except subprocess.TimeoutExpired:
        if process is not None:
            stdout, stderr = terminate_process_tree(process)
        else:  # pragma: no cover - process is always set before communicate
            stdout = stderr = ""
        return CommandRunResult(
            side=side,
            command=command,
            returncode=124,
            timed_out=True,
            duration_seconds=time.monotonic() - started_at,
            stdout_tail=_tail(stdout),
            stderr_tail=_tail(stderr),
            error="task deadline exceeded",
        )
    except OSError as exc:
        return CommandRunResult(
            side=side,
            command=command,
            returncode=127,
            duration_seconds=time.monotonic() - started_at,
            error=str(exc),
        )


def _classify_task_runs(
    broken: CommandRunResult,
    fixed: CommandRunResult,
) -> tuple[str, str]:
    if broken.timed_out or fixed.timed_out:
        return TIMEOUT, "timeout"
    if broken.error or fixed.error:
        return ERROR, "execution_error"
    if _looks_like_no_tests_collected(broken) and _looks_like_no_tests_collected(fixed):
        return NO_TESTS_COLLECTED, "no_tests_collected"
    if broken.returncode is None or fixed.returncode is None:
        return ERROR, "missing_returncode"

    broken_passed = broken.returncode == 0
    fixed_passed = fixed.returncode == 0
    if not broken_passed and fixed_passed:
        return F2P, "ok"
    if broken_passed and fixed_passed:
        return P2P, "oracle_too_weak"
    if not broken_passed and not fixed_passed:
        return F2F, "test_or_setup_invalid"
    return P2F, "observed_broken_assertion"


def _looks_like_no_tests_collected(result: CommandRunResult) -> bool:
    text = f"{result.stdout_tail}\n{result.stderr_tail}".lower()
    if result.returncode == 5 and "pytest" in result.command:
        return True
    return any(
        marker in text
        for marker in (
            "no tests ran",
            "no tests collected",
            "0 tests",
            "found 0 tests",
            "no test files found",
        )
    )


def _select_failure_excerpt(
    classification: str,
    broken: CommandRunResult,
    fixed: CommandRunResult,
) -> str:
    if classification in {F2P, P2P}:
        return _tail_text_parts(broken.stderr_tail, broken.stdout_tail)
    return _tail_text_parts(fixed.stderr_tail, fixed.stdout_tail, broken.stderr_tail)


def _tail_text_parts(*parts: str, limit: int = 3000) -> str:
    text = "\n".join(part for part in parts if part).strip()
    return text[-limit:]


def _tail(text: str, limit: int = 4000) -> str:
    return (text or "")[-limit:]


def _normalize_language(language: str) -> str:
    value = (language or "").strip().lower()
    aliases = {
        "py": "python",
        "python3": "python",
        "js": "javascript",
        "node": "javascript",
        "ts": "javascript",
        "typescript": "javascript",
        "golang": "go",
    }
    return aliases.get(value, value)


def _build_python_clamp_task(root: Path) -> LocalDualStateTask:
    task_root = root / "python_clamp"
    broken = task_root / "broken"
    fixed = task_root / "fixed"
    _write_python_clamp_repo(broken, upper_return="low")
    _write_python_clamp_repo(fixed, upper_return="high")
    artifact = CandidateTestArtifact(
        path="tests/test_clamp_generated.py",
        content=(
            "from lib.math_utils import clamp\n\n\n"
            "def test_clamp_caps_upper_bound():\n"
            "    assert clamp(10, 0, 5) == 5\n"
        ),
    )
    return LocalDualStateTask(
        task_id="python_clamp_upper_bound",
        language="python",
        issue_text="clamp(value, low, high) returns low instead of high when value exceeds high.",
        broken_repo=broken,
        fixed_repo=fixed,
        test_command=f"{sys.executable} -m pytest -q tests/test_clamp_generated.py",
        candidate_tests=(artifact,),
        expected_fixed_behavior="clamp(10, 0, 5) returns 5.",
        expected_broken_failure_mode="The generated test fails because the broken implementation returns 0.",
        authoritative_source="local fixture contract",
        public_surface="lib.math_utils.clamp(value, low, high)",
        metadata={"fixture_family": "clamp", "runner": "pytest"},
    )


def _write_python_clamp_repo(repo: Path, *, upper_return: str) -> None:
    if repo.exists():
        shutil.rmtree(repo, ignore_errors=True)
    (repo / "lib").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "lib" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "lib" / "math_utils.py").write_text(
        (
            "def clamp(value, low, high):\n"
            "    if value < low:\n"
            "        return low\n"
            "    if value > high:\n"
            f"        return {upper_return}\n"
            "    return value\n"
        ),
        encoding="utf-8",
    )


def _build_javascript_clamp_task(root: Path) -> LocalDualStateTask:
    task_root = root / "javascript_clamp"
    broken = task_root / "broken"
    fixed = task_root / "fixed"
    _write_javascript_clamp_repo(broken, upper_return="low")
    _write_javascript_clamp_repo(fixed, upper_return="high")
    artifact = CandidateTestArtifact(
        path="tests/clamp.test.js",
        content=(
            "const test = require('node:test');\n"
            "const assert = require('node:assert/strict');\n"
            "const { clamp } = require('../math_utils');\n\n"
            "test('clamp caps upper bound', () => {\n"
            "  assert.equal(clamp(10, 0, 5), 5);\n"
            "});\n"
        ),
    )
    return LocalDualStateTask(
        task_id="javascript_clamp_upper_bound",
        language="javascript",
        issue_text="clamp(value, low, high) returns low instead of high when value exceeds high.",
        broken_repo=broken,
        fixed_repo=fixed,
        test_command="node --test tests/clamp.test.js",
        candidate_tests=(artifact,),
        expected_fixed_behavior="clamp(10, 0, 5) returns 5.",
        expected_broken_failure_mode="The node:test assertion fails because the broken implementation returns 0.",
        authoritative_source="local fixture contract",
        public_surface="math_utils.clamp(value, low, high)",
        metadata={"fixture_family": "clamp", "runner": "node:test"},
    )


def _write_javascript_clamp_repo(repo: Path, *, upper_return: str) -> None:
    if repo.exists():
        shutil.rmtree(repo, ignore_errors=True)
    repo.mkdir(parents=True)
    (repo / "math_utils.js").write_text(
        (
            "function clamp(value, low, high) {\n"
            "  if (value < low) return low;\n"
            f"  if (value > high) return {upper_return};\n"
            "  return value;\n"
            "}\n\n"
            "module.exports = { clamp };\n"
        ),
        encoding="utf-8",
    )


def _build_go_clamp_task(root: Path) -> LocalDualStateTask:
    task_root = root / "go_clamp"
    broken = task_root / "broken"
    fixed = task_root / "fixed"
    _write_go_clamp_repo(broken, upper_return="low")
    _write_go_clamp_repo(fixed, upper_return="high")
    artifact = CandidateTestArtifact(
        path="clamp_test.go",
        content=(
            "package clamp\n\n"
            'import "testing"\n\n'
            "func TestClampCapsUpperBound(t *testing.T) {\n"
            "    if got := Clamp(10, 0, 5); got != 5 {\n"
            '        t.Fatalf("Clamp(10, 0, 5) = %d, want 5", got)\n'
            "    }\n"
            "}\n"
        ),
    )
    return LocalDualStateTask(
        task_id="go_clamp_upper_bound",
        language="go",
        issue_text="Clamp(value, low, high) returns low instead of high when value exceeds high.",
        broken_repo=broken,
        fixed_repo=fixed,
        test_command="go test ./...",
        candidate_tests=(artifact,),
        expected_fixed_behavior="Clamp(10, 0, 5) returns 5.",
        expected_broken_failure_mode="The Go test fails because the broken implementation returns 0.",
        authoritative_source="local fixture contract",
        public_surface="clamp.Clamp(value, low, high)",
        metadata={"fixture_family": "clamp", "runner": "go test"},
    )


def _write_go_clamp_repo(repo: Path, *, upper_return: str) -> None:
    if repo.exists():
        shutil.rmtree(repo, ignore_errors=True)
    repo.mkdir(parents=True)
    (repo / "go.mod").write_text(
        "module example.com/clamp\n\ngo 1.20\n",
        encoding="utf-8",
    )
    (repo / "clamp.go").write_text(
        (
            "package clamp\n\n"
            "func Clamp(value, low, high int) int {\n"
            "    if value < low {\n"
            "        return low\n"
            "    }\n"
            "    if value > high {\n"
            f"        return {upper_return}\n"
            "    }\n"
            "    return value\n"
            "}\n"
        ),
        encoding="utf-8",
    )


def _parse_languages(value: str) -> list[str]:
    return [language.strip() for language in (value or "").split(",") if language.strip()]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Apex's fast local dual-state testcase-generation benchmark.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where benchmark_report.json and per-task artifacts are written.",
    )
    parser.add_argument(
        "--languages",
        default="python,javascript,go",
        help="Comma-separated built-in fixture languages to run.",
    )
    parser.add_argument(
        "--task-timeout-seconds",
        type=float,
        default=30.0,
        help="Strict absolute wall-clock budget per task, covering broken and fixed runs.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel task workers.",
    )
    parser.add_argument(
        "--candidate-source",
        choices=("builtin", "apex-default"),
        default="builtin",
        help=(
            "builtin uses oracle fixture tests; apex-default asks Apex's default "
            "test generator to create the candidate tests before scoring."
        ),
    )
    args = parser.parse_args(argv)

    output_root = Path(args.output)
    fixtures_root = output_root / "fixtures"
    tasks, skipped = build_builtin_local_dual_state_tasks(
        fixtures_root,
        languages=_parse_languages(args.languages),
    )
    if args.candidate_source == "apex-default":
        tasks = generate_apex_default_candidate_tasks(
            tasks,
            output_dir=output_root,
        )
    report = run_local_dual_state_benchmark(
        tasks,
        output_dir=output_root,
        task_timeout_seconds=args.task_timeout_seconds,
        jobs=args.jobs,
        skipped_languages=skipped,
        candidate_source=args.candidate_source,
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.task_count and report.reliable_f2p_count == report.task_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
