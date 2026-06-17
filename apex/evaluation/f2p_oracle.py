"""Fail-to-Pass execution oracle for SWE-Bench Pro testgen.

For each generated test artifact, materialize candidate test files into two
sandboxes derived from ``task.base_commit`` — a *broken* sandbox (no patch
applied) and a *fixed* sandbox (gold ``task.patch`` applied) — and execute
the candidate tests in each. A test suite is considered to "catch the bug"
iff at least one test transitions from FAIL on broken to PASS on fixed.

This module is the inner-loop oracle for stages 4-5 of
``test_generation_design.md`` (Mutation-Driven Discrimination + Dual-Version
Verification) which previously existed only as schema fields the generating
LLM was asked to self-report.

Public API:
    evaluate_f2p(task, repo_dir, test_artifacts, *, output_dir, ...)
        Returns a dict with per-test transitions and aggregate F2P signals.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from apex.core.generated_tests import normalize_generated_test_path

logger = logging.getLogger(__name__)


def _python_executable() -> str:
    """The python interpreter to spawn pytest under.

    Prefer the interpreter currently running APEX (so the apex venv's
    `pytest` is on the path); fall back to system `python3` only if
    `sys.executable` is unset or empty.
    """
    return sys.executable or "python3"


# Languages whose F2P pipeline is exercised in CI. The oracle still works on
# any language with a registered TestRunnerAdapter (see apex.core.test_runners),
# but only these are guaranteed end-to-end by tests/test_f2p_oracle.py.
# Adding a language: register an adapter under apex/core/test_runners/ and
# add a basename pattern to _LANGUAGE_TEST_PATH_PATTERNS below.
_SUPPORTED_LANGUAGES = frozenset(
    {
        # Python
        "python",
        "py",
        "python3",
        # JavaScript / TypeScript — the SWE-Bench Pro dataset uses the
        # short forms "js"/"ts" as repo_language values, so the gate
        # MUST accept those. Without the short aliases the F2P oracle
        # rejects every JS/TS instance with skip_unsupported_language
        # even though jest/vitest adapters are wired and ready.
        "javascript",
        "js",
        "jsx",
        "typescript",
        "ts",
        "tsx",
        # Go (dataset value: "go")
        "go",
        "golang",
        # Rust (no current SWE-Bench Pro instances, but the cargo
        # adapter and provisioning plan are ready).
        "rust",
        "rs",
        # JVM
        "java",
        "kotlin",
        "kt",
        # .NET
        "csharp",
        "cs",
        "c#",
        "dotnet",
        # PHP / C++ / Swift
        "php",
        "cpp",
        "c++",
        "cc",
        "swift",
    }
)

# Per-language predicates for "is this artifact a test file we should
# materialize into the sandbox?". The classification is intentionally
# basename-only — directory-based heuristics misclassify conftest.py
# (under tests/) as a test file even though it's a fixture helper.
_LANGUAGE_TEST_PATH_PATTERNS: dict[str, tuple[re.Pattern, ...]] = {
    "python": (
        re.compile(r"(^|/)test_[^/]+\.py$", re.IGNORECASE),
        re.compile(r"(^|/)[^/]+_test\.py$", re.IGNORECASE),
        re.compile(r"(^|/)tests_[^/]+\.py$", re.IGNORECASE),
        re.compile(r"(^|/)[^/]+_tests\.py$", re.IGNORECASE),
    ),
    "javascript": (
        re.compile(r"\.(test|spec)\.[mc]?jsx?$", re.IGNORECASE),
        re.compile(r"(^|/)[^/]+(Test|Spec)\.[mc]?jsx?$"),
        re.compile(r"(^|/)__tests__/", re.IGNORECASE),
    ),
    "typescript": (
        re.compile(r"\.(test|spec)\.[mc]?tsx?$", re.IGNORECASE),
        re.compile(r"(^|/)[^/]+(Test|Spec)\.[mc]?tsx?$"),
        re.compile(r"(^|/)__tests__/", re.IGNORECASE),
    ),
    "go": (re.compile(r"(^|/)[^/]+_test\.go$", re.IGNORECASE),),
    "rust": (
        re.compile(r"(^|/)tests/[^/]+\.rs$", re.IGNORECASE),
        re.compile(r"(^|/)[^/]+_test\.rs$", re.IGNORECASE),
    ),
    "java": (re.compile(r"(^|/)[^/]+(Test|Tests|IT)\.java$"),),
    "kotlin": (re.compile(r"(^|/)[^/]+(Test|Tests|Spec)\.kt$"),),
    "csharp": (re.compile(r"(^|/)[^/]+(Test|Tests)\.cs$"),),
    "php": (re.compile(r"(^|/)[^/]+Test\.php$"),),
    "cpp": (
        re.compile(
            r"(^|/)(test_[^/]+|[^/]+_test|[^/]+Test)\.(cpp|cc|cxx)$",
            re.IGNORECASE,
        ),
    ),
    "swift": (re.compile(r"(^|/)[^/]+Tests\.swift$"),),
}

# Aliases: every entry in _SUPPORTED_LANGUAGES must resolve to a pattern set
# OR to None (None = unsupported short name we just want the gate to accept).
# Keep this in sync with _SUPPORTED_LANGUAGES and the alias map in
# _resolve_test_runner_adapter — any language name accepted by the gate
# MUST be routable by both the artifact selector and the adapter resolver.
_LANGUAGE_TEST_PATH_PATTERNS["py"] = _LANGUAGE_TEST_PATH_PATTERNS["python"]
_LANGUAGE_TEST_PATH_PATTERNS["python3"] = _LANGUAGE_TEST_PATH_PATTERNS["python"]
_LANGUAGE_TEST_PATH_PATTERNS["js"] = _LANGUAGE_TEST_PATH_PATTERNS["javascript"]
_LANGUAGE_TEST_PATH_PATTERNS["jsx"] = _LANGUAGE_TEST_PATH_PATTERNS["javascript"]
_LANGUAGE_TEST_PATH_PATTERNS["ts"] = _LANGUAGE_TEST_PATH_PATTERNS["typescript"]
_LANGUAGE_TEST_PATH_PATTERNS["tsx"] = _LANGUAGE_TEST_PATH_PATTERNS["typescript"]
_LANGUAGE_TEST_PATH_PATTERNS["golang"] = _LANGUAGE_TEST_PATH_PATTERNS["go"]
_LANGUAGE_TEST_PATH_PATTERNS["rs"] = _LANGUAGE_TEST_PATH_PATTERNS["rust"]
_LANGUAGE_TEST_PATH_PATTERNS["kt"] = _LANGUAGE_TEST_PATH_PATTERNS["kotlin"]
_LANGUAGE_TEST_PATH_PATTERNS["cs"] = _LANGUAGE_TEST_PATH_PATTERNS["csharp"]
_LANGUAGE_TEST_PATH_PATTERNS["c#"] = _LANGUAGE_TEST_PATH_PATTERNS["csharp"]
_LANGUAGE_TEST_PATH_PATTERNS["dotnet"] = _LANGUAGE_TEST_PATH_PATTERNS["csharp"]
_LANGUAGE_TEST_PATH_PATTERNS["c++"] = _LANGUAGE_TEST_PATH_PATTERNS["cpp"]
_LANGUAGE_TEST_PATH_PATTERNS["cc"] = _LANGUAGE_TEST_PATH_PATTERNS["cpp"]


def _is_test_path_for_language(language: str, path: str) -> bool:
    """Whether `path` looks like a test file for `language`.

    Returns False for unknown languages — caller should treat that as
    "this artifact is not a test the oracle should materialize", which
    is the conservative choice (worst case: a test suite produces zero
    F2P signal because we filtered out its files).
    """
    patterns = _LANGUAGE_TEST_PATH_PATTERNS.get((language or "").lower())
    if not patterns:
        return False
    return any(p.search(path or "") for p in patterns)


_JS_TS_TEST_IDIOM_RE = re.compile(
    r"\b(?:describe|it|test)\s*\(|\bo\.spec\s*\(|\bo\s*\(\s*['\"]",
    re.MULTILINE,
)
_PYTHON_TEST_IDIOM_RE = re.compile(
    r"^\s*(?:async\s+def\s+test_|def\s+test_|class\s+Test[A-Za-z0-9_]*\b)",
    re.MULTILINE,
)


def _content_looks_like_test_for_language(language: str, content: str) -> bool:
    normalized_language = (language or "").lower()
    text = str(content or "")
    if normalized_language in {"python", "py", "python3"}:
        return bool(_PYTHON_TEST_IDIOM_RE.search(text))
    if normalized_language in {"javascript", "js", "jsx", "typescript", "ts", "tsx"}:
        return bool(_JS_TS_TEST_IDIOM_RE.search(text))
    return False


def _is_test_directory_artifact_for_language(
    language: str,
    path: str,
    content: str,
) -> bool:
    """Conservative path+content fallback for native test directories.

    Some mature repositories keep large runner-specific suites in files like
    ``test/controllers.js`` rather than ``*.test.js``. Basename-only filters
    drop those artifacts and make F2P skip otherwise runnable JS/TS tests.
    Require both a conventional test directory and test-framework idioms so
    helper files under the same directory are not promoted into execution.
    """
    normalized_language = (language or "").lower()
    normalized_path = str(path or "").strip().replace("\\", "/")
    if normalized_language in {"python", "py", "python3"}:
        if Path(normalized_path).suffix.lower() != ".py":
            return False
        if Path(normalized_path).name.lower() == "conftest.py":
            return False
    elif normalized_language in {"javascript", "js", "jsx"}:
        if Path(normalized_path).suffix.lower() not in {".js", ".jsx", ".mjs", ".cjs"}:
            return False
    elif normalized_language in {"typescript", "ts", "tsx"}:
        if Path(normalized_path).suffix.lower() not in {".ts", ".tsx", ".mts", ".cts"}:
            return False
    else:
        return False
    parts = [part.lower() for part in normalized_path.split("/") if part]
    if len(parts) < 2 or not any(
        part in {"__tests__", "spec", "specs", "test", "testing", "tests"} for part in parts[:-1]
    ):
        return False
    return _content_looks_like_test_for_language(normalized_language, content)


# Default per-side wall-clock budget; total budget is roughly 2x this.
_DEFAULT_BROKEN_TIMEOUT_SECONDS = 300.0
_DEFAULT_FIXED_TIMEOUT_SECONDS = 300.0
_DEFAULT_INSTALL_TIMEOUT_SECONDS = 300.0
_DEFAULT_VENV_CREATE_TIMEOUT_SECONDS = 120.0
_DEFAULT_PATCH_APPLY_TIMEOUT_SECONDS = 60.0
_DEFAULT_CLONE_TIMEOUT_SECONDS = 600.0
_PYTEST_INSTALL_TIMEOUT_SECONDS = 120.0


# Regex for parsing pytest -v output. Lines look like:
#   tests/test_foo.py::TestClass::test_bar PASSED                            [ 12%]
#   tests/test_foo.py::test_baz FAILED                                       [ 25%]
#   tests/test_foo.py::test_quux ERROR                                       [ 33%]
#   tests/test_foo.py::test_skip SKIPPED (reason)                            [ 50%]
_PYTEST_OUTCOME_RE = re.compile(
    r"^(?P<nodeid>[^\s]+::[^\s]+)\s+(?P<status>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b"
)


# Materialization helper lives in the rollout engine; import lazily to avoid a
# heavy import cycle at module load time.
def _import_materializer():
    from apex.rollout.engine import _materialize_test_generation_artifacts

    return _materialize_test_generation_artifacts


@dataclass
class F2PRunResult:
    """Outcome of running candidate tests against one sandbox."""

    status: str  # "ok" | "no_tests_collected" | "timeout" | "exception" | "skip_*"
    returncode: Optional[int] = None
    duration_seconds: float = 0.0
    per_test_status: dict[str, str] = field(default_factory=dict)
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: Optional[str] = None


@dataclass
class DualStateTask:
    """Benchmark-agnostic input for dual-version testcase evaluation.

    ``broken_repo`` is the current/pre-fix checkout. Callers may provide either
    ``fixed_repo`` directly or a ``patch`` to apply to a copied broken checkout.
    SWE-Bench, TestGenEval-style slices, local PRs, and CI tasks should all be
    adapted into this shape before invoking the oracle.
    """

    broken_repo: str | Path
    fixed_repo: str | Path | None = None
    patch: str = ""
    language_hint: str = "python"
    issue_text: str = ""
    test_command: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def evaluate_f2p(
    *,
    task: Any,
    repo_dir: str,
    test_artifacts: list[dict[str, Any]],
    output_dir: str,
    language: str = "python",
    broken_timeout_seconds: float = _DEFAULT_BROKEN_TIMEOUT_SECONDS,
    fixed_timeout_seconds: float = _DEFAULT_FIXED_TIMEOUT_SECONDS,
    install_timeout_seconds: float = _DEFAULT_INSTALL_TIMEOUT_SECONDS,
    install_repo: bool = False,
    keep_sandboxes: bool = False,
) -> dict[str, Any]:
    """Benchmark adapter: clone broken/fixed sandboxes from a benchmark
    task and run the F2P oracle on them.

    This is the SWE-Bench-Pro-shaped entry point. It clones two
    sandboxes from ``repo_dir`` at ``task.base_commit``, applies
    ``task.patch`` to the fixed side, and delegates to
    :func:`evaluate_f2p_on_sandboxes` for the actual measurement.

    For real-world TDD / CI integration where you already have two
    prepared directories (current state vs. fix candidate), call
    :func:`evaluate_f2p_on_sandboxes` (or :func:`evaluate_tdd_iteration`)
    directly — no benchmark task object required.

    Returns a dict with shape::

        {
          "status": "ok" | "skip_*" | "error",
          "language": "python" | ...,
          "broken": F2PRunResult-as-dict,
          "fixed":  F2PRunResult-as-dict,
          "transitions": {
            "<nodeid>": {"broken": "fail", "fixed": "pass", "f2p": True},
            ...
          },
          "summary": {
            "candidate_test_paths": [...],
            "tests_observed": int,
            "f2p_count": int,
            "f2p_rate": float,             # f2p_count / tests_observed (0.0 if 0)
            "any_f2p": bool,               # any test transitioned fail->pass
            "p2f_count": int,              # tests that REGRESSED with the fix
            "f2f_count": int,              # tests that always failed
            "p2p_count": int,              # tests that always passed
            "skipped_count": int,
          },
          "broken_path": str,
          "fixed_path": str,
          "wall_clock_seconds": float,
        }
    """
    started_at = time.time()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    sandboxes_root = output_root / "f2p_sandboxes"
    broken_dir = sandboxes_root / "broken"
    fixed_dir = sandboxes_root / "fixed"

    base_result = {
        "status": "ok",
        "language": (language or "").lower(),
        "broken": _empty_run_result_dict(),
        "fixed": _empty_run_result_dict(),
        "transitions": {},
        "summary": _empty_summary_dict(),
        "broken_path": str(broken_dir),
        "fixed_path": str(fixed_dir),
        "wall_clock_seconds": 0.0,
    }

    # Early validation: gates that don't depend on the cloned sandboxes
    # short-circuit BEFORE the (potentially expensive) clone. Without
    # this, an unsupported language or empty artifact list would clone
    # two full repos before discovering the run is a no-op.
    normalized_language = (language or "").lower()
    if normalized_language not in _SUPPORTED_LANGUAGES:
        base_result["status"] = "skip_unsupported_language"
        base_result["wall_clock_seconds"] = time.time() - started_at
        return base_result
    if not _select_test_artifacts_for_language(test_artifacts, language=normalized_language):
        base_result["status"] = "skip_no_python_test_artifacts"
        base_result["wall_clock_seconds"] = time.time() - started_at
        return base_result

    base_commit = str(getattr(task, "base_commit", "") or "").strip()
    if not base_commit:
        base_result["status"] = "skip_missing_base_commit"
        base_result["wall_clock_seconds"] = time.time() - started_at
        return base_result

    repo_path = Path(repo_dir)
    if not repo_path.exists():
        base_result["status"] = "skip_repo_dir_missing"
        base_result["wall_clock_seconds"] = time.time() - started_at
        return base_result

    # ---------- Benchmark stages 1-2: clone + apply gold patch ----------
    # These are the only stages that depend on the SWE-Bench Pro task
    # shape. Once they finish we have two prepared sandboxes and the
    # rest is identical to a real-world TDD invocation.
    try:
        if sandboxes_root.exists():
            shutil.rmtree(sandboxes_root, ignore_errors=True)
        sandboxes_root.mkdir(parents=True, exist_ok=True)
        _clone_at_commit(repo_path, broken_dir, base_commit)
        _clone_at_commit(repo_path, fixed_dir, base_commit)
    except _F2POracleError as exc:
        base_result["status"] = f"error_clone:{exc.kind}"
        base_result["broken"]["error"] = str(exc)
        base_result["wall_clock_seconds"] = time.time() - started_at
        if not keep_sandboxes:
            shutil.rmtree(sandboxes_root, ignore_errors=True)
        return base_result

    gold_patch = str(getattr(task, "patch", "") or "")
    if gold_patch.strip():
        try:
            _apply_patch(fixed_dir, gold_patch)
        except _F2POracleError as exc:
            base_result["status"] = f"error_apply_gold_patch:{exc.kind}"
            base_result["fixed"]["error"] = str(exc)
            base_result["wall_clock_seconds"] = time.time() - started_at
            if not keep_sandboxes:
                shutil.rmtree(sandboxes_root, ignore_errors=True)
            return base_result

    # ---------- Delegate to the decoupled core ----------
    sandbox_result = evaluate_f2p_on_sandboxes(
        broken_dir=broken_dir,
        fixed_dir=fixed_dir,
        test_artifacts=test_artifacts,
        output_dir=output_dir,
        language=language,
        broken_timeout_seconds=broken_timeout_seconds,
        fixed_timeout_seconds=fixed_timeout_seconds,
        install_timeout_seconds=install_timeout_seconds,
        install_repo=install_repo,
        # Always keep sandboxes through the inner call so this wrapper
        # owns cleanup. Without this the inner call would tear down
        # the dirs the wrapper already created.
        keep_sandboxes=True,
    )
    # Preserve broken/fixed paths from the sandbox call (they're the
    # canonical location), and overlay everything else.
    base_result.update(sandbox_result)
    base_result["broken_path"] = str(broken_dir)
    base_result["fixed_path"] = str(fixed_dir)
    base_result["wall_clock_seconds"] = time.time() - started_at

    if not keep_sandboxes:
        shutil.rmtree(sandboxes_root, ignore_errors=True)

    return base_result


def evaluate_dual_state_task(
    task: DualStateTask,
    *,
    test_artifacts: list[dict[str, Any]],
    output_dir: str | Path,
    broken_timeout_seconds: float = _DEFAULT_BROKEN_TIMEOUT_SECONDS,
    fixed_timeout_seconds: float = _DEFAULT_FIXED_TIMEOUT_SECONDS,
    install_timeout_seconds: float = _DEFAULT_INSTALL_TIMEOUT_SECONDS,
    install_repo: bool = False,
    keep_sandboxes: bool = True,
) -> dict[str, Any]:
    """Run the F2P oracle from a benchmark-agnostic dual-state task."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    broken_dir = Path(task.broken_repo)
    owned_fixed_dir = False
    if task.fixed_repo is not None:
        fixed_dir = Path(task.fixed_repo)
    else:
        fixed_dir = output_root / "_dual_state_fixed"
        owned_fixed_dir = True
        if fixed_dir.exists():
            shutil.rmtree(fixed_dir, ignore_errors=True)
        if broken_dir.exists():
            shutil.copytree(broken_dir, fixed_dir)
        if task.patch.strip() and fixed_dir.exists():
            _apply_patch(fixed_dir, task.patch)

    result = evaluate_f2p_on_sandboxes(
        broken_dir=broken_dir,
        fixed_dir=fixed_dir,
        test_artifacts=test_artifacts,
        output_dir=output_root,
        language=task.language_hint,
        broken_timeout_seconds=broken_timeout_seconds,
        fixed_timeout_seconds=fixed_timeout_seconds,
        install_timeout_seconds=install_timeout_seconds,
        install_repo=install_repo,
        # If we created the fixed side from a patch, do not let the generic
        # cleanup remove the caller-owned broken repo.
        keep_sandboxes=True if owned_fixed_dir else keep_sandboxes,
    )
    result["dual_state_task"] = {
        "language_hint": task.language_hint,
        "issue_text": task.issue_text,
        "test_command": task.test_command,
        "metadata": dict(task.metadata or {}),
        "fixed_materialized_from_patch": owned_fixed_dir and bool(task.patch.strip()),
    }
    if owned_fixed_dir and not keep_sandboxes:
        shutil.rmtree(fixed_dir, ignore_errors=True)
    return result


def evaluate_f2p_on_sandboxes(
    *,
    broken_dir: str | Path,
    fixed_dir: str | Path,
    test_artifacts: list[dict[str, Any]],
    output_dir: str | Path,
    language: str = "python",
    broken_timeout_seconds: float = _DEFAULT_BROKEN_TIMEOUT_SECONDS,
    fixed_timeout_seconds: float = _DEFAULT_FIXED_TIMEOUT_SECONDS,
    install_timeout_seconds: float = _DEFAULT_INSTALL_TIMEOUT_SECONDS,
    install_repo: bool = False,
    keep_sandboxes: bool = True,
) -> dict[str, Any]:
    """Decoupled F2P oracle that operates on two pre-prepared sandboxes.

    This is the real-world / TDD / CI entry point. The caller is
    responsible for creating ``broken_dir`` (current state, pre-fix)
    and ``fixed_dir`` (state after the fix candidate is applied) and
    for cleaning them up afterward (controlled by ``keep_sandboxes``).

    No assumptions about benchmark shape — works for:

      * TDD: ``broken = current source``,
        ``fixed = current source after the agent's patch``
      * Bug regression: ``broken = pre-bug-fix``,
        ``fixed = post-bug-fix``
      * PR review: ``broken = parent commit``,
        ``fixed = PR HEAD``

    The agent / orchestrator that calls this can run it iteratively
    after each fix attempt to get a transition signal without
    re-cloning.

    See :func:`evaluate_f2p` for the SWE-Bench-Pro-shaped wrapper that
    builds the two sandboxes from a benchmark task.
    """
    started_at = time.time()
    broken_dir = Path(broken_dir)
    fixed_dir = Path(fixed_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    base_result = {
        "status": "ok",
        "language": (language or "").lower(),
        "broken": _empty_run_result_dict(),
        "fixed": _empty_run_result_dict(),
        "transitions": {},
        "summary": _empty_summary_dict(),
        "broken_path": str(broken_dir),
        "fixed_path": str(fixed_dir),
        "wall_clock_seconds": 0.0,
    }

    normalized_language = (language or "").lower()
    if normalized_language not in _SUPPORTED_LANGUAGES:
        base_result["status"] = "skip_unsupported_language"
        base_result["wall_clock_seconds"] = time.time() - started_at
        return base_result

    relevant_artifacts = _select_test_artifacts_for_language(
        test_artifacts, language=normalized_language
    )
    if not relevant_artifacts:
        # Status name preserved for backward compatibility — historical
        # callers grep for "skip_no_python_test_artifacts". Future
        # non-python paths can be reported as the same skip code since
        # the meaning is identical: "no artifacts looked like tests".
        base_result["status"] = "skip_no_python_test_artifacts"
        base_result["wall_clock_seconds"] = time.time() - started_at
        return base_result

    candidate_paths = sorted(
        {
            normalized
            for a in relevant_artifacts
            if (normalized := normalize_generated_test_path(a.get("path")))
        }
    )
    base_result["summary"]["candidate_test_paths"] = candidate_paths

    if not broken_dir.exists():
        base_result["status"] = "skip_broken_dir_missing"
        base_result["wall_clock_seconds"] = time.time() - started_at
        return base_result
    if not fixed_dir.exists():
        base_result["status"] = "skip_fixed_dir_missing"
        base_result["wall_clock_seconds"] = time.time() - started_at
        return base_result

    # ---------- Stage 3: materialize candidate test files in both sandboxes ----------
    materialize = _import_materializer()
    all_artifacts = [a for a in list(test_artifacts or []) if isinstance(a, dict)]
    materialized_broken = materialize(worktree_path=str(broken_dir), artifacts=all_artifacts)
    materialized_fixed = materialize(worktree_path=str(fixed_dir), artifacts=all_artifacts)
    executable_paths = sorted(
        set(candidate_paths).intersection(materialized_broken, materialized_fixed)
    )
    base_result["summary"]["materialized_artifact_paths"] = sorted(
        set(materialized_broken).intersection(materialized_fixed)
    )
    base_result["summary"]["materialized_support_paths"] = sorted(
        set(base_result["summary"]["materialized_artifact_paths"]) - set(executable_paths)
    )
    if not executable_paths:
        base_result["status"] = "skip_no_materialized_paths"
        base_result["wall_clock_seconds"] = time.time() - started_at
        _cleanup_sandbox_dirs(broken_dir, fixed_dir, keep=keep_sandboxes)
        return base_result

    # ---------- Stage 4: per-sandbox provisioning ----------
    # Resolve the TestRunnerAdapter once and reuse for both provisioning
    # (Stage 4) and test execution (Stage 5). Detection runs against the
    # fixed sandbox (post-patch state); the broken sandbox shares its
    # project layout so either side would match.
    adapter = _resolve_test_runner_adapter(fixed_dir=fixed_dir, language=normalized_language)
    base_result["test_runner_adapter"] = adapter.name if adapter is not None else None

    # Each side gets its own throwaway environment so installs cannot
    # pollute the apex venv or race between concurrent tasks. The
    # environment is reused across test invocations within the same task.
    broken_python = _python_executable()
    fixed_python = _python_executable()
    install_metadata: dict[str, Any] = {"broken": "skipped", "fixed": "skipped"}
    install_failures: dict[str, str] = {}
    if install_repo:
        for label, sandbox in (("broken", broken_dir), ("fixed", fixed_dir)):
            executable, install_status = _provision_sandbox_environment(
                adapter=adapter,
                sandbox_dir=sandbox,
                venv_timeout_seconds=_DEFAULT_VENV_CREATE_TIMEOUT_SECONDS,
                install_timeout_seconds=install_timeout_seconds,
            )
            install_metadata[label] = install_status
            if executable is not None:
                if label == "broken":
                    broken_python = str(executable)
                else:
                    fixed_python = str(executable)
            if install_status.startswith("install_failed") or install_status == "install_timeout":
                install_failures[label] = install_status

    base_result["install_metadata"] = install_metadata

    # ---------- Stage 5: run tests on both sandboxes ----------
    broken_run = _run_tests_on_paths(
        adapter=adapter,
        sandbox_dir=broken_dir,
        test_paths=executable_paths,
        timeout_seconds=broken_timeout_seconds,
        python_executable=broken_python,
    )
    fixed_run = _run_tests_on_paths(
        adapter=adapter,
        sandbox_dir=fixed_dir,
        test_paths=executable_paths,
        timeout_seconds=fixed_timeout_seconds,
        python_executable=fixed_python,
    )
    base_result["broken"] = _run_result_to_dict(broken_run)
    base_result["fixed"] = _run_result_to_dict(fixed_run)

    # ---------- Stage 6: classify per-test transitions ----------
    raw_transitions, raw_summary = _classify_transitions(
        broken_status=broken_run.per_test_status,
        fixed_status=fixed_run.per_test_status,
    )
    transitions = raw_transitions
    summary = raw_summary
    generated_names_by_path = _build_generated_test_name_filter(
        relevant_artifacts,
        language=normalized_language,
    )
    generated_broken_status = _filter_status_to_generated_nodeids(
        broken_run.per_test_status,
        generated_names_by_path=generated_names_by_path,
    )
    generated_fixed_status = _filter_status_to_generated_nodeids(
        fixed_run.per_test_status,
        generated_names_by_path=generated_names_by_path,
    )
    generated_nodeids = sorted(set(generated_broken_status) | set(generated_fixed_status))
    if generated_nodeids:
        transitions, summary = _classify_transitions(
            broken_status=generated_broken_status,
            fixed_status=generated_fixed_status,
        )
        summary["transition_scope"] = "generated_nodeids"
        summary["raw_transition_summary"] = raw_summary
        summary["generated_test_nodeids"] = generated_nodeids
        summary["generated_test_names_by_path"] = {
            path: sorted(names) for path, names in generated_names_by_path.items()
        }
    else:
        summary["transition_scope"] = "executed_paths"
    unreliable_statuses = {"timeout", "exception"}
    if broken_run.status in unreliable_statuses or fixed_run.status in unreliable_statuses:
        summary["unreliable_execution"] = True
        summary["unreliable_execution_reason"] = (
            f"broken_status={broken_run.status}; fixed_status={fixed_run.status}"
        )
        summary["unreliable_raw_f2p_count"] = int(summary.get("f2p_count") or 0)
        summary["unreliable_raw_any_f2p"] = bool(summary.get("any_f2p"))
        summary["f2p_count"] = 0
        summary["f2p_rate"] = 0.0
        summary["any_f2p"] = False
        for info in transitions.values():
            if info.get("kind") == "f2p":
                info["reliable_f2p"] = False
                info["unreliable_reason"] = summary["unreliable_execution_reason"]
    base_result["transitions"] = transitions
    base_result["summary"].update(summary)
    base_result["summary"]["candidate_test_paths"] = candidate_paths
    if install_failures:
        base_result["summary"]["install_warnings"] = dict(install_failures)

    if summary.get("unreliable_execution"):
        base_result["status"] = "skip_unreliable_test_execution"

    # If both runs collected zero tests, surface a more informative status.
    if broken_run.status == "no_tests_collected" and fixed_run.status == "no_tests_collected":
        base_result["status"] = "skip_no_tests_collected"
    if (
        install_failures
        and int(base_result["summary"].get("tests_observed") or 0) == 0
        and base_result["status"] in {"ok", "skip_no_tests_collected"}
    ):
        first_failure = next(iter(install_failures.values()))
        base_result["status"] = f"skip_install_failed:{first_failure}"

    repair_feedback = _build_f2p_repair_feedback(
        status=str(base_result.get("status") or ""),
        broken_run=broken_run,
        fixed_run=fixed_run,
        summary=dict(base_result.get("summary") or {}),
    )
    base_result["repair_feedback"] = dict(repair_feedback)
    base_result["summary"].update(repair_feedback)
    base_result["wall_clock_seconds"] = time.time() - started_at
    _cleanup_sandbox_dirs(broken_dir, fixed_dir, keep=keep_sandboxes)
    return base_result


def _cleanup_sandbox_dirs(broken_dir: Path, fixed_dir: Path, *, keep: bool) -> None:
    """Remove sandbox dirs unless the caller asked to keep them.

    Used by ``evaluate_f2p_on_sandboxes``. The benchmark wrapper
    (``evaluate_f2p``) always passes ``keep=True`` to the inner call
    so this helper no-ops there — the wrapper owns the shared
    sandboxes_root cleanup separately.
    """
    if keep:
        return
    for sandbox in (broken_dir, fixed_dir):
        if sandbox and sandbox.exists():
            shutil.rmtree(sandbox, ignore_errors=True)


def evaluate_tdd_iteration(
    *,
    broken_dir: str | Path,
    fixed_dir: str | Path,
    test_artifacts: list[dict[str, Any]],
    output_dir: str | Path,
    language: str = "python",
    timeout_seconds: float = _DEFAULT_BROKEN_TIMEOUT_SECONDS,
    install_repo: bool = False,
) -> dict[str, Any]:
    """Friendly wrapper for the real-world TDD use case.

    Use this from IDE plugins, CI gates, or agentic-coding orchestrators
    where you already have two prepared directories: the *current*
    state of the code (no fix yet) and the *fix candidate* state. The
    return shape matches :func:`evaluate_f2p` and
    :func:`evaluate_f2p_on_sandboxes` so downstream tooling (mutation
    engine, minimizer, judge) plugs in unchanged.

    Example::

        # The agent just wrote tests; check whether they actually
        # constrain the contract change. broken_dir = current source,
        # fixed_dir = current source + the agent's proposed patch.
        report = evaluate_tdd_iteration(
            broken_dir="/sandbox/before",
            fixed_dir="/sandbox/after",
            test_artifacts=[{"path": "tests/test_x.py", "content": "..."}],
            output_dir="/tmp/tdd_report",
            language="python",
            install_repo=True,
        )
        if report["summary"]["any_f2p"]:
            print("Tests catch the contract change ✅")
        for nodeid, info in report["transitions"].items():
            if info["kind"] == "p2p":
                print(f"  {nodeid} is useless (passes both versions)")
    """
    return evaluate_f2p_on_sandboxes(
        broken_dir=broken_dir,
        fixed_dir=fixed_dir,
        test_artifacts=test_artifacts,
        output_dir=output_dir,
        language=language,
        broken_timeout_seconds=timeout_seconds,
        fixed_timeout_seconds=timeout_seconds,
        install_timeout_seconds=timeout_seconds,
        install_repo=install_repo,
        # Real-world callers usually own the dir lifecycle; respect that.
        keep_sandboxes=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _F2POracleError(RuntimeError):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass
class _LanguageProvisioningPlan:
    """Per-language sandbox-environment provisioning recipe.

    ``setup_marker_files`` lists the manifest files that must exist for
    this plan to apply. ``commands`` is the sequence of subprocess
    invocations (cwd=sandbox) needed to install dependencies. The
    failing command's stderr tail is propagated up as the install
    status, the same way pytest's `pip install -e .` failure is.
    """

    adapter_name: str
    setup_marker_files: tuple[str, ...]
    commands: tuple[tuple[str, ...], ...]


# Per-adapter provisioning plans. Pytest does NOT live here — it has a
# specialised implementation (``_provision_sandbox_venv``) that builds a
# per-side throwaway venv so concurrent tasks cannot race on
# site-packages. The other languages use the system tool directly: they
# install deps into the sandbox-local manifest cache and don't need
# venv-style isolation.
_LANGUAGE_PROVISIONING_PLANS: dict[str, _LanguageProvisioningPlan] = {
    "jest": _LanguageProvisioningPlan(
        adapter_name="jest",
        setup_marker_files=("package.json",),
        commands=(("npm", "ci", "--no-audit", "--no-fund", "--ignore-scripts"),),
    ),
    "vitest": _LanguageProvisioningPlan(
        adapter_name="vitest",
        setup_marker_files=("package.json",),
        commands=(("npm", "ci", "--no-audit", "--no-fund", "--ignore-scripts"),),
    ),
    "mocha": _LanguageProvisioningPlan(
        adapter_name="mocha",
        setup_marker_files=("package.json",),
        commands=(("npm", "ci", "--no-audit", "--no-fund", "--ignore-scripts"),),
    ),
    "ospec": _LanguageProvisioningPlan(
        adapter_name="ospec",
        setup_marker_files=("package.json",),
        commands=(("npm", "ci", "--no-audit", "--no-fund", "--ignore-scripts"),),
    ),
    "go-test": _LanguageProvisioningPlan(
        adapter_name="go-test",
        setup_marker_files=("go.mod",),
        commands=(("go", "mod", "download"),),
    ),
    "cargo-test": _LanguageProvisioningPlan(
        adapter_name="cargo-test",
        setup_marker_files=("Cargo.toml",),
        commands=(("cargo", "fetch", "--quiet"),),
    ),
    "junit": _LanguageProvisioningPlan(
        adapter_name="junit",
        setup_marker_files=("pom.xml", "build.gradle", "build.gradle.kts"),
        commands=(("mvn", "-q", "dependency:resolve"),),
    ),
    "dotnet-test": _LanguageProvisioningPlan(
        adapter_name="dotnet-test",
        # dotnet picks up *.csproj / *.sln implicitly; the marker check
        # below uses "any file with these suffixes" rather than explicit
        # filenames, so we pass an empty tuple to suppress the marker
        # check and rely on dotnet's own discovery.
        setup_marker_files=(),
        commands=(("dotnet", "restore", "--nologo"),),
    ),
    "phpunit": _LanguageProvisioningPlan(
        adapter_name="phpunit",
        setup_marker_files=("composer.json",),
        commands=(("composer", "install", "--no-interaction", "--quiet"),),
    ),
    "swift-xctest": _LanguageProvisioningPlan(
        adapter_name="swift-xctest",
        setup_marker_files=("Package.swift",),
        commands=(("swift", "package", "resolve"),),
    ),
    "ctest": _LanguageProvisioningPlan(
        adapter_name="ctest",
        setup_marker_files=("CMakeLists.txt",),
        # CMake projects need configuration before tests can run.
        # Out-of-source build under sandbox/build/.
        commands=(("cmake", "-S", ".", "-B", "build"),),
    ),
}


def _provision_sandbox_environment(
    *,
    adapter: Optional[Any],
    sandbox_dir: Path,
    venv_timeout_seconds: float,
    install_timeout_seconds: float,
) -> tuple[Optional[Path], str]:
    """Adapter-aware sandbox provisioner.

    Dispatches to ``_provision_sandbox_venv`` for the pytest adapter
    (preserves the existing per-side venv isolation that all current
    Python F2P runs depend on), and to a per-language plan from
    ``_LANGUAGE_PROVISIONING_PLANS`` otherwise.

    Returns ``(executable, status)`` matching the
    ``_provision_sandbox_venv`` shape:
      * ``executable`` — language-specific runner path (python
        interpreter for pytest, ``None`` for adapters whose tools
        are invoked through their system entrypoint).
      * ``status`` — one of ``installed`` | ``no_setup`` |
        ``install_failed:*`` | ``install_timeout`` |
        ``no_provisioning_plan`` | adapter-specific failure codes.

    Adapters with no registered plan return
    ``(None, "no_provisioning_plan")`` so the F2P run still proceeds
    (with whatever the system already provides), rather than skipping
    the whole task. This matches the ``no_setup`` behavior on Python
    repos with no setup.py / pyproject.toml.
    """
    if adapter is None or getattr(adapter, "name", "") in {"pytest", "django", ""}:
        return _provision_sandbox_venv(
            sandbox_dir,
            venv_timeout_seconds=venv_timeout_seconds,
            install_timeout_seconds=install_timeout_seconds,
        )

    plan = _LANGUAGE_PROVISIONING_PLANS.get(adapter.name)
    if plan is None:
        return None, "no_provisioning_plan"

    # Marker check — the empty-tuple convention means "always proceed"
    # (used for dotnet whose discovery handles missing manifests).
    if plan.setup_marker_files and not any(
        (sandbox_dir / marker).exists() for marker in plan.setup_marker_files
    ):
        return None, "no_setup"

    for command in plan.commands:
        try:
            completed = subprocess.run(
                list(command),
                cwd=str(sandbox_dir),
                capture_output=True,
                text=True,
                timeout=install_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None, "install_timeout"
        except FileNotFoundError:
            # The system tool (npm / cargo / go / etc.) isn't installed.
            # Treat as a soft failure — a missing toolchain is an env
            # issue, not an APEX failure.
            return None, f"install_failed:tool_not_found:{command[0]}"
        if completed.returncode != 0:
            stderr_tail = (completed.stderr or "").strip()[-300:]
            suffix = (":" + stderr_tail.replace("\n", " | ")) if stderr_tail else ""
            return None, f"install_failed{suffix}"
    return None, "installed"


def _resolve_test_runner_adapter(
    *,
    fixed_dir: Path,
    language: str,
) -> Optional[Any]:
    """Pick a TestRunnerAdapter for the F2P run.

    Tries manifest-aware JavaScript/TypeScript selection first because one
    language token can mean Jest, Vitest, or Mocha. Other languages use an
    explicit registry lookup (`get_adapter(language)`), then fall back to
    filesystem detection (`detect_adapter(fixed_dir)`). Returns None when
    no adapter matches — caller treats that as "use the fall-through pytest
    path", preserving historical behavior on python repos even before the
    adapter registry was wired up.
    """
    try:
        from apex.core.test_runners import detect_adapter, get_adapter
    except Exception:  # pragma: no cover — defensive
        return None
    normalized_language = (language or "").lower()

    if normalized_language in {"python", "py", "python3"}:
        if (fixed_dir / "tests" / "runtests.py").exists() and (fixed_dir / "django").is_dir():
            explicit = get_adapter("django")
            if explicit is not None:
                return explicit

    # JavaScript/TypeScript repos need manifest-aware selection because Jest,
    # Vitest, and Mocha all share the same language token. Do this before the
    # generic alias map; otherwise every JS/TS task would be forced through
    # Jest even when package.json declares a different runner.
    if normalized_language in {"javascript", "js", "jsx", "typescript", "ts", "tsx"}:
        package_json = fixed_dir / "package.json"
        if package_json.exists():
            try:
                data = json.loads(package_json.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            deps: dict[str, Any] = {}
            for key in ("dependencies", "devDependencies"):
                section = data.get(key) or {}
                if isinstance(section, dict):
                    deps.update(section)
            for runner in ("vitest", "ospec", "jest", "mocha"):
                if runner in deps:
                    explicit = get_adapter(runner)
                    if explicit is not None:
                        return explicit

    # Keep in sync with _SUPPORTED_LANGUAGES and _LANGUAGE_TEST_PATH_PATTERNS.
    name_aliases = {
        # Python
        "python": "pytest",
        "py": "pytest",
        "python3": "pytest",
        # JavaScript / TypeScript
        "javascript": "mocha",
        "js": "mocha",
        "jsx": "mocha",
        "typescript": "mocha",
        "ts": "mocha",
        "tsx": "mocha",
        # Go
        "go": "go-test",
        "golang": "go-test",
        # Rust
        "rust": "cargo-test",
        "rs": "cargo-test",
        # JVM
        "java": "junit",
        "kotlin": "junit",
        "kt": "junit",
        # .NET
        "csharp": "dotnet-test",
        "cs": "dotnet-test",
        "c#": "dotnet-test",
        "dotnet": "dotnet-test",
        # Other
        "php": "phpunit",
        "cpp": "ctest",
        "c++": "ctest",
        "cc": "ctest",
        "swift": "swift-xctest",
    }
    adapter_name = name_aliases.get(normalized_language)
    if adapter_name:
        explicit = get_adapter(adapter_name)
        if explicit is not None:
            return explicit
    return detect_adapter(fixed_dir)


def _run_tests_on_paths(
    *,
    adapter: Optional[Any],
    sandbox_dir: Path,
    test_paths: list[str],
    timeout_seconds: float,
    python_executable: Optional[str] = None,
) -> "F2PRunResult":
    """Dispatch to a TestRunnerAdapter, falling back to in-process pytest.

    The pytest path stays in ``_run_pytest_on_paths`` because it carries
    nodeid normalization, partial-output preservation on TimeoutExpired,
    and a distinctive exit-code-5 ('no tests collected') translation
    that the generic adapter Protocol does not yet model. For any other
    adapter the helper builds the run command via the Protocol, executes
    it, and translates ``RunResult`` outcomes back into the ``per_test_status``
    shape the F2P transition classifier expects.
    """
    if adapter is None or getattr(adapter, "name", "") in {"pytest", ""}:
        return _run_pytest_on_paths(
            sandbox_dir=sandbox_dir,
            test_paths=test_paths,
            timeout_seconds=timeout_seconds,
            python_executable=python_executable,
        )
    if not test_paths:
        return F2PRunResult(status="no_paths_supplied")

    started = time.time()
    report_path = sandbox_dir / "_apex_f2p_adapter_report.json"
    # ``python_executable`` is meaningful for Python-native non-pytest adapters
    # such as Django, but must not leak into JS/TS/Go/etc. runners.
    adapter_executable = python_executable if getattr(adapter, "name", "") == "django" else None
    from apex.core.subprocess_utils import build_command_env

    env = build_command_env({"PYTHONUNBUFFERED": "1"})
    run_paths = getattr(adapter, "run_paths", None)
    try:
        if run_paths is None:
            raise TypeError("legacy adapter has no run_paths")
        run_result = run_paths(
            sandbox_dir,
            list(test_paths),
            report_path,
            timeout_seconds=timeout_seconds,
            env=env,
            executable=adapter_executable,
        )
    except TypeError:
        # Out-of-tree adapters may not have been registered through
        # apex.core.test_runners.register_adapter. Fall back to the historical
        # build/parse surface so those integrations keep working.
        try:
            command = adapter.build_run_command(
                sandbox_dir,
                list(test_paths),
                report_path,
                executable=adapter_executable,
            )
        except TypeError:
            command = adapter.build_run_command(sandbox_dir, list(test_paths), report_path)
        try:
            completed = subprocess.run(
                command,
                cwd=str(sandbox_dir),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
                shell=True,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            partial_stdout = (
                (exc.stdout or "").decode("utf-8", errors="ignore")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            return F2PRunResult(
                status="timeout",
                duration_seconds=time.time() - started,
                stdout_tail=_tail(partial_stdout),
                error=f"adapter '{getattr(adapter, 'name', '?')}' exceeded {int(timeout_seconds)}s",
            )
        try:
            run_result = adapter.parse_report(report_path)
        except Exception as exc:  # pragma: no cover — defensive
            return F2PRunResult(
                status="exception",
                returncode=completed.returncode,
                duration_seconds=time.time() - started,
                stdout_tail=_tail(completed.stdout or ""),
                stderr_tail=_tail(completed.stderr or ""),
                error=f"adapter parse_report failed: {type(exc).__name__}: {exc}",
            )
        run_result.raw_output = "\n".join(
            part
            for part in (
                str(getattr(run_result, "raw_output", "") or ""),
                completed.stdout or "",
                completed.stderr or "",
            )
            if part
        )
        run_result.returncode = completed.returncode
    except Exception as exc:  # pragma: no cover — defensive
        return F2PRunResult(
            status="exception",
            duration_seconds=time.time() - started,
            error=f"{type(exc).__name__}: {exc}",
        )

    duration = time.time() - started
    raw_output = str(getattr(run_result, "raw_output", "") or "")
    if bool(getattr(run_result, "timed_out", False)):
        return F2PRunResult(
            status="timeout",
            returncode=int(getattr(run_result, "returncode", 124) or 124),
            duration_seconds=duration,
            stdout_tail=_tail(raw_output),
            error=str(
                getattr(run_result, "error", "")
                or f"adapter '{getattr(adapter, 'name', '?')}' exceeded {int(timeout_seconds)}s"
            ),
        )

    # Normalize adapter outcome vocabulary to the {pass|fail|skip} keys
    # the transition classifier expects. The Protocol contract says
    # ``RunResult.outcomes`` may use any vendor-specific status string
    # (pytest's "passed"/"failed", jest's "passed"/"failed"/"pending",
    # cargo's "ok"/"FAILED", etc.).
    per_test_status: dict[str, str] = {}
    for nodeid, raw_status in (run_result.outcomes or {}).items():
        status_lower = str(raw_status or "").strip().lower()
        if status_lower in {"pass", "passed", "ok", "success"}:
            per_test_status[nodeid] = "pass"
        elif status_lower in {"fail", "failed", "error", "failure"}:
            per_test_status[nodeid] = "fail"
        else:
            per_test_status[nodeid] = "skip"
    per_test_status = _normalize_nodeid_keys(
        per_test_status,
        sandbox_dir=sandbox_dir,
    )

    if not per_test_status and run_result.collected == 0:
        return F2PRunResult(
            status="no_tests_collected",
            returncode=int(getattr(run_result, "returncode", 1) or 1),
            duration_seconds=duration,
            stdout_tail=_tail(raw_output),
        )

    return F2PRunResult(
        status="ok",
        returncode=int(getattr(run_result, "returncode", 1) or 1),
        duration_seconds=duration,
        per_test_status=per_test_status,
        stdout_tail=_tail(raw_output),
    )


def _select_test_artifacts_for_language(
    artifacts: list[dict[str, Any]],
    *,
    language: str,
) -> list[dict[str, Any]]:
    """Filter test_writer artifacts to the ones that look like tests
    for ``language``.

    Replaces the prior python-only ``_select_python_test_artifacts``;
    the predicate is now keyed by language so the same oracle can run
    on JavaScript, Go, Rust, Java, etc. once an adapter is registered
    in apex/core/test_runners/ and a basename pattern is added to
    ``_LANGUAGE_TEST_PATH_PATTERNS`` above.
    """
    selected: list[dict[str, Any]] = []
    for artifact in artifacts or []:
        if not isinstance(artifact, dict):
            continue
        path = str(artifact.get("path") or "").strip()
        content = str(artifact.get("content") or "")
        if not path or not content.strip():
            continue
        if not (
            _is_test_path_for_language(language, path)
            or _is_test_directory_artifact_for_language(language, path, content)
        ):
            continue
        selected.append(artifact)
    return selected


# Backward-compatible alias. Existing tests and the historical call site in
# evaluate_f2p both still work with this name; new code should call
# _select_test_artifacts_for_language directly.
def _select_python_test_artifacts(
    artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _select_test_artifacts_for_language(artifacts, language="python")


def _empty_run_result_dict() -> dict[str, Any]:
    return _run_result_to_dict(F2PRunResult(status="not_run"))


def _empty_summary_dict() -> dict[str, Any]:
    return {
        "candidate_test_paths": [],
        "tests_observed": 0,
        "f2p_count": 0,
        "f2p_rate": 0.0,
        "any_f2p": False,
        "p2f_count": 0,
        "f2f_count": 0,
        "p2p_count": 0,
        "skipped_count": 0,
    }


_MODULE_NOT_FOUND_PATTERNS = (
    re.compile(r"ModuleNotFoundError:\s+No module named ['\"](?P<module>[^'\"]+)['\"]"),
    re.compile(r"ImportError:\s+No module named ['\"](?P<module>[^'\"]+)['\"]"),
    re.compile(r"Cannot find module ['\"](?P<module>[^'\"]+)['\"]", re.IGNORECASE),
    re.compile(r"ERR_MODULE_NOT_FOUND.*?(?P<module>[A-Za-z0-9_@./-]+)", re.IGNORECASE),
    re.compile(r"Can't resolve ['\"](?P<module>[^'\"]+)['\"]", re.IGNORECASE),
    re.compile(r"Could not resolve ['\"](?P<module>[^'\"]+)['\"]", re.IGNORECASE),
    re.compile(r"Cannot resolve module ['\"](?P<module>[^'\"]+)['\"]", re.IGNORECASE),
    re.compile(r"no required module provides package (?P<module>\S+)", re.IGNORECASE),
    re.compile(r"cannot find package ['\"]?(?P<module>[^'\"\s]+)", re.IGNORECASE),
)


def _run_output_text(result: F2PRunResult) -> str:
    return "\n".join(
        part
        for part in (
            result.error or "",
            result.stdout_tail or "",
            result.stderr_tail or "",
        )
        if part
    )


def _detect_missing_modules(*results: F2PRunResult) -> list[str]:
    modules: set[str] = set()
    combined = "\n".join(_run_output_text(result) for result in results)
    for pattern in _MODULE_NOT_FOUND_PATTERNS:
        for match in pattern.finditer(combined):
            module = str(match.groupdict().get("module") or "").strip().strip(".,;:")
            if module:
                modules.add(module)
    return sorted(modules)


def _failure_excerpt(result: F2PRunResult, *, max_chars: int = 1200) -> str:
    text = _run_output_text(result).strip()
    if not text:
        return ""
    return _tail(text, max_chars=max_chars)


def _fixed_side_passed(result: F2PRunResult) -> bool:
    statuses = [str(status or "").lower() for status in result.per_test_status.values()]
    if result.status != "ok" or not statuses:
        return False
    return any(status == "pass" for status in statuses) and not any(
        status == "fail" for status in statuses
    )


def _build_f2p_repair_feedback(
    *,
    status: str,
    broken_run: F2PRunResult,
    fixed_run: F2PRunResult,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Convert measured F2P outcomes into repair-policy prompt material."""
    failure_classes: list[str] = []
    repair_hints: list[dict[str, str]] = []

    def add_hint(code: str, policy: str, action: str) -> None:
        if code not in failure_classes:
            failure_classes.append(code)
        if not any(hint.get("class") == code for hint in repair_hints):
            repair_hints.append(
                {
                    "class": code,
                    "policy": policy,
                    "action": action,
                }
            )

    if status == "skip_no_tests_collected" or (
        broken_run.status == "no_tests_collected" and fixed_run.status == "no_tests_collected"
    ):
        add_hint(
            "no_tests_collected",
            "Fix path, runner, framework placement, or test registration before changing assertions.",
            "Mine the nearest existing tests and repo-native test command, then place the artifact where that runner discovers it.",
        )

    missing_modules = _detect_missing_modules(broken_run, fixed_run)
    if missing_modules:
        add_hint(
            "module_not_found",
            "Fix bootstrap/import/setup before changing assertions.",
            "Mine existing test bootstrap files, fixture registration, module aliases, and test-local support imports; add only the required setup artifact.",
        )

    if int(summary.get("f2f_count") or 0) > 0:
        add_hint(
            "f2f",
            "Fix malformed tests or setup so they pass on the fixed state.",
            "Use the fixed-side failure excerpt first; do not strengthen the oracle until the fixed side is green.",
        )
    if int(summary.get("p2p_count") or 0) > 0:
        add_hint(
            "p2p",
            "Strengthen the oracle so it asserts the changed behavior.",
            "Replace smoke/presence checks with concrete public-surface outcomes from the issue, patch, docs, or executable templates.",
        )
    if int(summary.get("p2f_count") or 0) > 0:
        add_hint(
            "p2f",
            "Reject assertions that encode observed broken behavior.",
            "Regenerate expected values from the issue or patch contract rather than copying current output from the broken checkout.",
        )

    broken_excerpt = _failure_excerpt(broken_run)
    fixed_excerpt = _failure_excerpt(fixed_run)
    fixed_passed = _fixed_side_passed(fixed_run)
    tests_observed = int(summary.get("tests_observed") or 0)
    runnable = (
        tests_observed > 0
        and broken_run.status == "ok"
        and fixed_run.status == "ok"
        and not bool(summary.get("unreliable_execution"))
    )
    return {
        "failure_classes": failure_classes,
        "repair_hints": repair_hints,
        "failure_excerpts": {
            key: value
            for key, value in {
                "broken": broken_excerpt,
                "fixed": fixed_excerpt,
            }.items()
            if value
        },
        "missing_modules": missing_modules,
        "fixed_side_passed": fixed_passed,
        "runnable": runnable,
    }


def _run_result_to_dict(result: F2PRunResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "returncode": result.returncode,
        "duration_seconds": round(result.duration_seconds, 3),
        "per_test_status": dict(result.per_test_status),
        "stdout_tail": result.stdout_tail,
        "stderr_tail": result.stderr_tail,
        "error": result.error,
    }


_PYTHON_TEST_DEF_RE = re.compile(
    r"^[ \t]*(?:async[ \t]+def|def)[ \t]+(?P<name>test_[A-Za-z_][A-Za-z0-9_]*)[ \t]*\(",
    re.MULTILINE,
)


def _extract_generated_test_names(
    *,
    artifact: dict[str, Any],
    language: str,
) -> set[str]:
    """Best-effort extraction of generated test node names from content.

    F2P must execute whole files for many runners, but when an artifact
    appends tests into an existing file the raw runner output includes old
    P2P tests too. Extracting generated names lets the summary focus on the
    tests Apex actually wrote while retaining raw whole-file telemetry.
    """
    content = str((artifact or {}).get("content") or "")
    if not content.strip():
        return set()
    normalized_language = (language or "").lower()
    if normalized_language in {"python", "py", "python3"}:
        return {
            match.group("name")
            for match in _PYTHON_TEST_DEF_RE.finditer(content)
            if match.group("name")
        }
    return set()


def _build_generated_test_name_filter(
    artifacts: list[dict[str, Any]],
    *,
    language: str,
) -> dict[str, set[str]]:
    names_by_path: dict[str, set[str]] = {}
    for artifact in artifacts or []:
        if not isinstance(artifact, dict):
            continue
        rel_path = normalize_generated_test_path(artifact.get("path"))
        if not rel_path:
            continue
        names = _extract_generated_test_names(
            artifact=artifact,
            language=language,
        )
        if names:
            names_by_path.setdefault(rel_path, set()).update(names)
    return names_by_path


def _filter_status_to_generated_nodeids(
    statuses: dict[str, str],
    *,
    generated_names_by_path: dict[str, set[str]],
) -> dict[str, str]:
    if not statuses or not generated_names_by_path:
        return {}
    sole_path = next(iter(generated_names_by_path)) if len(generated_names_by_path) == 1 else ""
    filtered: dict[str, str] = {}
    for nodeid, status in statuses.items():
        path, _, suffix = str(nodeid).partition("::")
        canonical_nodeid = str(nodeid)
        if not path and sole_path and suffix:
            path = sole_path
            canonical_nodeid = f"{path}::{suffix}"
        names = generated_names_by_path.get(path)
        if not names or not suffix:
            continue
        node_parts = [part.split("[", 1)[0] for part in suffix.split("::")]
        if any(part in names for part in node_parts):
            filtered[canonical_nodeid] = status
    return filtered


def _clone_at_commit(source_repo: Path, dest: Path, commit: str) -> None:
    """git clone --shared the source repo into dest, then checkout commit."""
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)

    clone_cmd = [
        "git",
        "clone",
        "--shared",
        "--no-hardlinks",
        "--quiet",
        str(source_repo),
        str(dest),
    ]
    completed = subprocess.run(
        clone_cmd,
        capture_output=True,
        text=True,
        timeout=_DEFAULT_CLONE_TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode != 0:
        raise _F2POracleError(
            "clone_failed",
            f"git clone failed (rc={completed.returncode}): {completed.stderr.strip()[-400:]}",
        )

    checkout = subprocess.run(
        ["git", "-C", str(dest), "checkout", "--quiet", commit],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if checkout.returncode != 0:
        raise _F2POracleError(
            "checkout_failed",
            f"git checkout {commit[:12]} failed (rc={checkout.returncode}): "
            f"{checkout.stderr.strip()[-400:]}",
        )


def _apply_patch(repo_dir: Path, patch_text: str) -> None:
    if not patch_text.strip():
        return
    # Try `git apply` first (handles unified diffs cleanly), then fall back to
    # `patch -p1` if git refuses (some SWE-Bench Pro patches use mailbox-style
    # headers that confuse git).
    apply_cmd = subprocess.run(
        ["git", "-C", str(repo_dir), "apply", "--whitespace=nowarn", "-"],
        input=patch_text,
        capture_output=True,
        text=True,
        timeout=_DEFAULT_PATCH_APPLY_TIMEOUT_SECONDS,
        check=False,
    )
    if apply_cmd.returncode == 0:
        return
    # 3-way merge fallback (patches that depend on already-staged content)
    apply_cmd_3way = subprocess.run(
        ["git", "-C", str(repo_dir), "apply", "--3way", "--whitespace=nowarn", "-"],
        input=patch_text,
        capture_output=True,
        text=True,
        timeout=_DEFAULT_PATCH_APPLY_TIMEOUT_SECONDS,
        check=False,
    )
    if apply_cmd_3way.returncode == 0:
        return
    raise _F2POracleError(
        "patch_apply_failed",
        f"git apply failed: rc={apply_cmd.returncode}; 3way rc={apply_cmd_3way.returncode}; "
        f"stderr={apply_cmd.stderr.strip()[-400:]}",
    )


def _provision_sandbox_venv(
    sandbox_dir: Path,
    *,
    venv_timeout_seconds: float,
    install_timeout_seconds: float,
) -> tuple[Optional[Path], str]:
    """Build a per-sandbox venv + install pytest + the sandbox repo.

    Returns ``(venv_python, status)`` where ``venv_python`` is the path to the
    venv's python interpreter (or ``None`` if no venv was created), and
    ``status`` is one of: ``installed`` | ``no_setup`` | ``install_failed:*``
    | ``install_timeout`` | ``venv_failed`` | ``pytest_install_failed``.

    Each side gets its own venv so concurrent tasks do not race on the apex
    venv's site-packages and so tests cannot accidentally see packages from
    the orchestrator process.
    """
    has_setup_py = (sandbox_dir / "setup.py").exists()
    has_pyproject = (sandbox_dir / "pyproject.toml").exists()
    if not (has_setup_py or has_pyproject):
        return None, "no_setup"

    venv_dir = sandbox_dir / ".f2p_venv"
    try:
        venv_create = subprocess.run(
            [_python_executable(), "-m", "venv", "--clear", str(venv_dir)],
            capture_output=True,
            text=True,
            timeout=venv_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "venv_timeout"
    if venv_create.returncode != 0:
        return None, "venv_failed"

    venv_python = venv_dir / "bin" / "python"
    if not venv_python.exists():
        # Some platforms (rare) put it at /Scripts/python.exe — fall back.
        alt = venv_dir / "Scripts" / "python.exe"
        if alt.exists():
            venv_python = alt
        else:
            return None, "venv_no_interpreter"

    # Install pytest into the venv first; if this fails the sandbox is unusable.
    pytest_install = subprocess.run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            "--no-input",
            "pytest",
        ],
        capture_output=True,
        text=True,
        timeout=_PYTEST_INSTALL_TIMEOUT_SECONDS,
        check=False,
    )
    if pytest_install.returncode != 0:
        return venv_python, "pytest_install_failed"

    # Install the repo (editable so the sandbox source is what runs).
    try:
        repo_install = subprocess.run(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                "--no-input",
                "-e",
                ".",
            ],
            cwd=str(sandbox_dir),
            capture_output=True,
            text=True,
            timeout=install_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return venv_python, "install_timeout"
    if repo_install.returncode != 0:
        # Surface a small slice of stderr so the artifact has a hint about
        # which dependency or build step refused to install.
        stderr_tail = (repo_install.stderr or "").strip()[-300:]
        suffix = (":" + stderr_tail.replace("\n", " | ")) if stderr_tail else ""
        return venv_python, f"install_failed{suffix}"
    return venv_python, "installed"


def _run_pytest_on_paths(
    *,
    sandbox_dir: Path,
    test_paths: list[str],
    timeout_seconds: float,
    python_executable: Optional[str] = None,
) -> F2PRunResult:
    if not test_paths:
        return F2PRunResult(status="no_paths_supplied")

    started = time.time()
    interpreter = python_executable or _python_executable()
    cmd = [
        interpreter,
        "-m",
        "pytest",
        "-v",
        "--tb=line",
        "--no-header",
        "-p",
        "no:cacheprovider",
        "--rootdir",
        str(sandbox_dir),
        *test_paths,
    ]
    from apex.core.subprocess_utils import build_command_env

    env = build_command_env()
    is_django_repo = (sandbox_dir / "tests" / "runtests.py").exists() and (
        sandbox_dir / "django"
    ).is_dir()
    if not is_django_repo:
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    else:
        env.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONPATH", str(sandbox_dir))
    if is_django_repo:
        env.setdefault("DJANGO_SETTINGS_MODULE", "test_sqlite")
        pythonpath_parts = [str(sandbox_dir / "tests"), str(sandbox_dir)]
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(sandbox_dir),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        partial_stdout = (
            (exc.stdout or "").decode("utf-8", errors="ignore")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        partial_stderr = (
            (exc.stderr or "").decode("utf-8", errors="ignore")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        return F2PRunResult(
            status="timeout",
            returncode=None,
            duration_seconds=time.time() - started,
            per_test_status=_parse_pytest_outcomes(partial_stdout),
            stdout_tail=_tail(partial_stdout),
            stderr_tail=_tail(partial_stderr),
            error=f"pytest exceeded {int(timeout_seconds)}s",
        )
    except Exception as exc:  # pragma: no cover — defensive
        return F2PRunResult(
            status="exception",
            duration_seconds=time.time() - started,
            error=f"{type(exc).__name__}: {exc}",
        )

    duration = time.time() - started
    per_test_status = _normalize_nodeid_keys(
        _parse_pytest_outcomes(completed.stdout or ""),
        sandbox_dir=sandbox_dir,
    )
    if not per_test_status and completed.returncode == 5:
        # pytest exit code 5 = no tests collected
        return F2PRunResult(
            status="no_tests_collected",
            returncode=completed.returncode,
            duration_seconds=duration,
            stdout_tail=_tail(completed.stdout or ""),
            stderr_tail=_tail(completed.stderr or ""),
        )

    return F2PRunResult(
        status="ok",
        returncode=completed.returncode,
        duration_seconds=duration,
        per_test_status=per_test_status,
        stdout_tail=_tail(completed.stdout or ""),
        stderr_tail=_tail(completed.stderr or ""),
    )


def _normalize_nodeid_keys(
    outcomes: dict[str, str],
    *,
    sandbox_dir: Path,
) -> dict[str, str]:
    """Strip the sandbox path prefix from pytest nodeids.

    pytest emits nodeids relative to the working directory it actually
    used as rootdir, which can include extra "../" hops when
    `--rootdir` was set explicitly. We normalize every nodeid to a
    sandbox-relative form so per-test outcomes match between the
    broken and fixed runs.
    """
    if not outcomes:
        return {}
    sandbox_resolved = sandbox_dir.resolve()
    sandbox_str = str(sandbox_resolved)
    normalized: dict[str, str] = {}
    for nodeid, status in outcomes.items():
        path_part, _, test_part = nodeid.partition("::")
        if not path_part:
            normalized[nodeid] = status
            continue
        try:
            resolved = (sandbox_dir / path_part).resolve()
        except OSError:
            resolved = None
        if resolved is not None:
            resolved_str = str(resolved)
            if resolved_str.startswith(sandbox_str + os.sep):
                rel_path = resolved_str[len(sandbox_str) + 1 :]
            elif resolved_str == sandbox_str:
                rel_path = ""
            else:
                rel_path = path_part
        else:
            rel_path = path_part
        canonical = rel_path if not test_part else f"{rel_path}::{test_part}"
        normalized[canonical] = status
    return normalized


def _parse_pytest_outcomes(stdout: str) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    for line in (stdout or "").splitlines():
        match = _PYTEST_OUTCOME_RE.match(line.strip())
        if not match:
            continue
        nodeid = match.group("nodeid").strip()
        status = match.group("status").strip().lower()
        # PASSED -> pass; FAILED/ERROR -> fail; SKIPPED/XFAIL/XPASS -> skip
        if status in {"passed"}:
            outcomes[nodeid] = "pass"
        elif status in {"failed", "error"}:
            outcomes[nodeid] = "fail"
        else:
            outcomes[nodeid] = "skip"
    return outcomes


def _classify_transitions(
    *,
    broken_status: dict[str, str],
    fixed_status: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    nodeids = sorted(set(broken_status) | set(fixed_status))
    transitions: dict[str, dict[str, Any]] = {}
    f2p_count = p2f_count = f2f_count = p2p_count = skipped_count = 0
    for nodeid in nodeids:
        broken = broken_status.get(nodeid, "missing")
        fixed = fixed_status.get(nodeid, "missing")
        is_f2p = broken == "fail" and fixed == "pass"
        is_p2f = broken == "pass" and fixed == "fail"
        if is_f2p:
            f2p_count += 1
            kind = "f2p"
        elif is_p2f:
            p2f_count += 1
            kind = "p2f"
        elif broken == "fail" and fixed == "fail":
            f2f_count += 1
            kind = "f2f"
        elif broken == "pass" and fixed == "pass":
            p2p_count += 1
            kind = "p2p"
        else:
            skipped_count += 1
            kind = "skip_or_missing"
        transitions[nodeid] = {
            "broken": broken,
            "fixed": fixed,
            "f2p": is_f2p,
            "kind": kind,
        }
    tests_observed = len(transitions)
    f2p_rate = (f2p_count / tests_observed) if tests_observed else 0.0
    summary = {
        "tests_observed": tests_observed,
        "f2p_count": f2p_count,
        "f2p_rate": round(f2p_rate, 4),
        "any_f2p": f2p_count > 0,
        "p2f_count": p2f_count,
        "f2f_count": f2f_count,
        "p2p_count": p2p_count,
        "skipped_count": skipped_count,
    }
    return transitions, summary


def _tail(text: str, *, max_chars: int = 4000) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return "...[truncated]\n" + text[-max_chars:]


def write_f2p_artifact(*, output_dir: str, payload: dict[str, Any]) -> str:
    """Persist the F2P evaluation result alongside other per-task artifacts."""
    target = Path(output_dir) / "f2p_evaluation.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(target)
