"""
Patch verification, regression pruning, and cross-validation.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

from ..controller_policy import (
    TestInventory,
    default_test_inventory_language,
    derive_test_collection_command,
    infer_test_inventory_framework,
    summarize_test_inventory_coverage,
)
from ..core.failure_classifier import (
    ClassificationResult as CoreClassificationResult,
)
from ..core.failure_classifier import (
    FailureClass as CoreFailureClass,
)
from ..core.filesystem import copy_tree
from ..core.git_utils import (
    clone_git_repo_with_overlay,
    is_git_repo,
)
from ..core.git_utils import (
    list_changed_files as list_git_changed_files,
)
from ..core.pytest_report_utils import (
    count_pytest_report_outcomes,
    extract_pytest_report_outcomes,
    extract_pytest_report_summary_counts,
    extract_pytest_report_tests,
    load_pytest_json_report,
    looks_like_test_path,
    parse_pytest_terminal_summary_counts,
)
from ..core.pytest_utils import (
    build_ephemeral_pytest_command,
    build_pytest_recovery_commands,
    build_runtime_python_command,
    build_targeted_pytest_command,
    is_pytest_command,
    normalize_pytest_command,
    output_indicates_missing_pytest,
    parse_pytest_command,
    render_pytest_command,
    should_disable_pytest_plugin_autoload,
)
from ..core.cli_backend import _resolve_active_rollout_cli_context
from ..core.subprocess_utils import run_shell_command
from ..core.terminal_output import normalize_terminal_output
from ..test_portfolio import (
    _artifact_design_metadata_gaps,
    _pass_then_invert_complete,
    normalize_test_suite_artifact_payload,
    select_cross_validation_test_artifacts,
)

logger = logging.getLogger("apex.selection.verifier")

_SYNTHETIC_SUITE_TEST = "<full-suite>"
_SYNTHETIC_MUTATION_MAX_FILES = 2
_SYNTHETIC_MUTATION_MAX_MUTANTS_PER_FILE = 2
_SYNTHETIC_MUTATION_MAX_MUTANTS_PER_ARTIFACT = 4


def _synthetic_validation_output_excerpt(
    output: Any,
    *,
    max_lines: int = 24,
    max_chars: int = 2000,
) -> str:
    text = normalize_terminal_output(str(output or "")).strip()
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    excerpt = "\n".join(lines).strip()
    if len(excerpt) > max_chars:
        excerpt = excerpt[-max_chars:].lstrip()
    return excerpt


def _summarize_synthetic_validation_feedback(validation: dict[str, Any]) -> str:
    if not bool(validation.get("artifact_discovered")):
        return "Artifact was not discovered by the repository test runner."
    if not bool(validation.get("collection_succeeded")):
        return "Artifact failed collection; inspect collection_output_excerpt."
    if not bool(validation.get("execution_succeeded")) and bool(
        validation.get("execution_command")
    ):
        return "Artifact collected but execution failed; inspect execution_output_excerpt."
    if bool(validation.get("baseline_preservation_measured")) and not bool(
        validation.get("baseline_preserved")
    ):
        return (
            "Artifact executed but regressed previously passing baseline tests; "
            "inspect baseline_regression_output_excerpt."
        )
    plausible_mutant_survived_count = int(validation.get("plausible_mutant_survived_count") or 0)
    if plausible_mutant_survived_count > 0:
        noun = "mutant" if plausible_mutant_survived_count == 1 else "mutants"
        return (
            f"{plausible_mutant_survived_count} plausible {noun} survived; "
            "strengthen observable assertions."
        )
    if bool(validation.get("execution_succeeded")) and (
        not bool(validation.get("baseline_preservation_measured"))
        or bool(validation.get("baseline_preserved"))
    ):
        return "Artifact executed cleanly and preserved baseline tests."
    return "Artifact validation completed; inspect stored output excerpts for details."


@dataclass
class TestResult:
    """Result of executing reproduction and regression checks."""

    __test__ = False
    passed: int = 0
    failed: int = 0
    errors: int = 0
    reproduction_passes: bool = False
    regression_passes: bool = False
    # ``regression_inconclusive`` distinguishes a TIMEOUT from a genuine
    # test failure. Before this flag, a timed-out regression command set
    # ``regression_passes=False`` indistinguishably from "tests actually
    # failed", causing the orchestrator to reject patches whose visible
    # tests passed cleanly but whose full regression suite simply ran
    # too long (commit0/fastapi, commit0/pexpect). Inconclusive ≠
    # failure: it should not award the +0.35 regression bonus, but it
    # should also not block acceptance when other strong signals exist.
    regression_inconclusive: bool = False
    reproduction_output: str = ""
    regression_output: str = ""
    expected_test_count: int = 0
    matched_expected_test_count: int = 0
    missing_expected_test_count: int = 0
    missing_expected_test_ids: list[str] = field(default_factory=list)
    expected_coverage_preserved: Optional[bool] = None
    collected_test_count: int = 0
    test_inventory_framework: str = ""
    test_inventory_language: str = ""
    test_inventory_source: str = ""
    test_inventory_collection_command: str = ""
    # Phase 0.1: coarse, orchestrator-wide failure taxonomy. Populated by
    # the test runner when a failure is observed; remains ``None`` for
    # successful runs. ``failure_class`` is a denormalised mirror of
    # ``failure_classification.failure_class`` for cheap filtering.
    failure_class: Optional["CoreFailureClass"] = None
    failure_classification: Optional["CoreClassificationResult"] = None

    @property
    def pass_rate(self) -> float:
        total = self.passed + self.failed + self.errors
        if total == 0:
            return 0.0
        return self.passed / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "pass_rate": self.pass_rate,
            "reproduction_passes": self.reproduction_passes,
            "regression_passes": self.regression_passes,
            "regression_inconclusive": self.regression_inconclusive,
            "reproduction_output": self.reproduction_output,
            "regression_output": self.regression_output,
            "expected_test_count": self.expected_test_count,
            "matched_expected_test_count": self.matched_expected_test_count,
            "missing_expected_test_count": self.missing_expected_test_count,
            "missing_expected_test_ids": list(self.missing_expected_test_ids),
            "expected_coverage_preserved": self.expected_coverage_preserved,
            "collected_test_count": self.collected_test_count,
            "test_inventory_framework": self.test_inventory_framework,
            "test_inventory_language": self.test_inventory_language,
            "test_inventory_source": self.test_inventory_source,
            "test_inventory_collection_command": self.test_inventory_collection_command,
            "failure_class": (self.failure_class.value if self.failure_class is not None else None),
            "failure_classification": (
                self.failure_classification.to_dict()
                if self.failure_classification is not None
                else None
            ),
        }


@dataclass
class BaselineResult:
    """Baseline status of repository tests before patch application."""

    passing_tests: set[str] = field(default_factory=set)
    failing_tests: set[str] = field(default_factory=set)
    collected_tests: set[str] = field(default_factory=set)
    collected_test_count: int = 0
    total_duration: float = 0.0
    output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passing_tests": sorted(self.passing_tests),
            "failing_tests": sorted(self.failing_tests),
            "collected_tests": sorted(self.collected_tests),
            "collected_test_count": self.collected_test_count,
            "total_duration": self.total_duration,
            "output": self.output,
        }


@dataclass
class PruneResult:
    """Binary prune decision for one rollout candidate."""

    is_valid: bool
    regressed_tests: list[str] = field(default_factory=list)
    still_passing: list[str] = field(default_factory=list)
    output: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "regressed_tests": list(self.regressed_tests),
            "still_passing": list(self.still_passing),
            "output": self.output,
            "reason": self.reason,
        }


@dataclass
class VerificationResult:
    """Verification summary for one candidate patch."""

    rollout_id: int
    syntax_valid: bool = True
    lint_clean: bool = True
    # ``lint_applied`` is False when flake8 could not be executed (missing
    # binary, timeout, etc). The previous behaviour silently treated those
    # cases as ``lint_clean=True`` and added the +0.1 lint bonus to the
    # overall score, inflating ranks in flake8-less environments. Scoring
    # now requires ``lint_applied=True`` to award the bonus.
    lint_applied: bool = True
    accepted: bool = False
    changed_files: list[str] = field(default_factory=list)
    lint_output: str = ""
    test_result: Optional[TestResult] = None
    cross_validation_scores: list[float] = field(default_factory=list)
    prune_result: Optional[dict[str, Any]] = None
    overall_score: float = 0.0
    quality_gate_passed: Optional[bool] = None
    validity_reasons: list[str] = field(default_factory=list)
    verification_taxonomy: dict[str, Any] = field(default_factory=dict)
    repair_policy: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_id": self.rollout_id,
            "syntax_valid": self.syntax_valid,
            "lint_clean": self.lint_clean,
            "lint_applied": self.lint_applied,
            "accepted": self.accepted,
            "changed_files": list(self.changed_files),
            "lint_output": self.lint_output,
            "test_result": self.test_result.to_dict() if self.test_result else None,
            "cross_validation_scores": list(self.cross_validation_scores),
            "prune_result": self.prune_result,
            "overall_score": self.overall_score,
            "quality_gate_passed": self.quality_gate_passed,
            "validity_reasons": list(self.validity_reasons),
            "verification_taxonomy": dict(self.verification_taxonomy),
            "repair_policy": self.repair_policy,
        }


@dataclass
class _CommandResult:
    returncode: int
    output: str


class _SandboxValidationError(ValueError):
    """Raised when test_code violates the sandbox path policy."""


def _command_with_absolute_json_report_path(
    command: str,
    report_path: Optional[Path],
) -> str:
    if report_path is None:
        return command
    return re.sub(
        r"--json-report-file=\S+",
        f"--json-report-file={str(report_path)}",
        command,
        count=1,
    )


class PatchVerifier:
    """Run static and dynamic checks against a rollout workspace."""

    def __init__(
        self,
        repo_path: str,
        timeout: int = 120,
        full_test_timeout: Optional[int] = None,
        custom_test_timeout: Optional[int] = None,
        runtime_env_overrides: Optional[dict[str, str]] = None,
        verification_helper_files: Optional[list[str]] = None,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.timeout = timeout
        self.full_test_timeout = max(full_test_timeout or 900, timeout)
        self.custom_test_timeout = max(custom_test_timeout or 120, 1)
        self.runtime_env_overrides = {
            str(key): str(value)
            for key, value in dict(runtime_env_overrides or {}).items()
            if str(key).strip() and str(value).strip()
        }
        self.verification_helper_files = self._normalize_verification_helper_files(
            verification_helper_files or []
        )
        self._baseline_cache: dict[tuple[str, str], BaselineResult] = {}
        self._python_definition_index_cache: dict[str, dict[str, list[str]]] = {}
        self._quarantined_paths: dict[str, str] = {}
        # Set by ``PatchSelector.__init__`` from ``config.selection``.
        # When True, cross-validation skips the sandbox copy (legacy
        # behavior reserved for ablations).
        self.cross_validation_sandbox_disabled: bool = False

    @property
    def quarantined_paths(self) -> dict[str, str]:
        return dict(self._quarantined_paths)

    def release_quarantine(self, rel_path: str) -> None:
        normalized = str(rel_path or "").strip().replace("\\", "/")
        if normalized:
            self._quarantined_paths.pop(normalized, None)

    def _quarantine_path(self, rel_path: str, reason: str) -> None:
        normalized = str(rel_path or "").strip().replace("\\", "/")
        if not normalized:
            return
        self._quarantined_paths[normalized] = str(reason or "syntax_invalid")[:240]

    def verify_patch(
        self,
        rollout_id: int,
        worktree_path: str,
        reproduction_command: Optional[str] = None,
        reproduction_script: Optional[str] = None,
        test_command: Optional[str] = None,
        baseline_ref: Optional[str] = None,
        reproduction_script_path: Optional[str] = None,
        expected_test_count: Optional[int] = None,
        expected_test_ids: Optional[list[str]] = None,
        test_inventory: Optional[Any] = None,
        baseline_result: Optional[BaselineResult] = None,
    ) -> VerificationResult:
        worktree = Path(worktree_path)
        changed_files = self._resolve_changed_files(worktree, baseline_ref=baseline_ref)
        verification = VerificationResult(rollout_id=rollout_id, changed_files=changed_files)

        verification.syntax_valid = self._check_syntax(worktree, changed_files)
        if not verification.syntax_valid:
            verification.overall_score = 0.0
            return verification

        verification.lint_clean, verification.lint_output, verification.lint_applied = (
            self._check_lint(worktree, changed_files)
        )
        self._sync_verification_helper_files(worktree)

        test_result = TestResult()
        resolved_test_inventory = self._resolve_test_inventory(
            test_inventory=test_inventory,
            expected_test_count=expected_test_count,
            expected_test_ids=expected_test_ids,
            test_command=test_command,
            baseline_result=baseline_result,
        )
        if reproduction_command or reproduction_script:
            command_result = self._run_reproduction(
                worktree,
                reproduction_command,
                reproduction_script,
                script_path=reproduction_script_path,
                test_command=test_command,
            )
            test_result.reproduction_passes = command_result.returncode == 0
            test_result.reproduction_output = command_result.output
            # Reproduction outcome is tracked separately on
            # ``reproduction_passes`` and contributes to the score via
            # ``_compute_score``. We deliberately do NOT add a synthetic
            # +1 to passed/failed here, because that would poison
            # ``pass_rate`` (which is used as an acceptance gate) with a
            # quantity that is not a regression test outcome.

        if test_command:
            regression_command = test_command
            report_path: Optional[Path] = None
            report_started_at: Optional[float] = None
            if self._is_pytest_command(test_command):
                report_started_at = time.time()
                regression_command, report_path = (
                    self._prepare_pytest_json_report_command_for_worktree(
                        test_command,
                        label=f"verify-{rollout_id}",
                        worktree=worktree,
                    )
                )
            try:
                regression = self._run_command_with_completion_report(
                    worktree,
                    regression_command,
                    timeout=self.full_test_timeout,
                    completion_report_path=report_path,
                    completion_report_started_at=report_started_at,
                )
                if (
                    report_path is not None
                    and regression.returncode != 0
                    and self._pytest_json_report_command_needs_plain_retry(regression.output)
                ):
                    self._cleanup_report_path(report_path)
                    report_path = None
                    regression = self._run_command(
                        worktree,
                        test_command,
                        timeout=self.full_test_timeout,
                    )
                test_result.regression_output = regression.output
                parsed = self._parse_test_output(regression.output, regression.returncode)
                payload = self._load_pytest_json_report_payload(
                    report_path,
                    minimum_mtime=report_started_at,
                )
                report_tests = extract_pytest_report_tests(payload) if payload is not None else []
                report_outcomes = extract_pytest_report_outcomes(report_tests)
                # pytest-json-report avoids false counts from traceback/log text in terminal output.
                structured_counts = (
                    count_pytest_report_outcomes(report_outcomes)
                    if report_outcomes
                    else extract_pytest_report_summary_counts(payload)
                )
                if any(structured_counts.values()):
                    parsed = structured_counts
                test_result.passed += parsed["passed"]
                test_result.failed += parsed["failed"]
                test_result.errors += parsed["errors"]
                observed_test_count = self._parse_observed_test_count(
                    regression.output,
                    parsed,
                )
                self._apply_test_inventory_coverage(
                    test_result,
                    report_path=report_path,
                    minimum_mtime=report_started_at,
                    test_inventory=resolved_test_inventory,
                    observed_test_count=observed_test_count if observed_test_count > 0 else None,
                )
                # 124 = subprocess timeout. A timeout is INCONCLUSIVE — we
                # don't know if the regression suite would have passed; we
                # only know it didn't finish in time. Treat as separate
                # axis from genuine failure so downstream scoring and
                # acceptance can decide intelligently (see
                # ``_compute_score`` and ``_verification_meets_acceptance_bar``
                # in selector.py for how this flag is consumed).
                if regression.returncode == 124 or "Command timed out after" in (
                    regression.output or ""
                ):
                    test_result.regression_passes = False
                    test_result.regression_inconclusive = True
                else:
                    test_result.regression_passes = regression.returncode == 0
                    test_result.regression_inconclusive = False
            finally:
                self._cleanup_report_path(report_path)

        verification.test_result = test_result
        verification.overall_score = self._compute_score(verification)
        return verification

    @staticmethod
    def _normalize_verification_helper_files(paths: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_path in paths:
            value = str(raw_path or "").strip().replace("\\", "/")
            if not value or value.startswith("/") or value.startswith("../"):
                continue
            parts = [part for part in value.split("/") if part and part != "."]
            if not parts or any(part == ".." for part in parts):
                continue
            rel_path = "/".join(parts)
            if rel_path in seen:
                continue
            seen.add(rel_path)
            normalized.append(rel_path)
        return normalized

    def _sync_verification_helper_files(self, worktree: Path) -> None:
        if not self.verification_helper_files:
            return
        for rel_path in self.verification_helper_files:
            source = self.repo_path / rel_path
            target = worktree / rel_path
            try:
                if source.exists() or source.is_symlink():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists() or target.is_symlink():
                        if target.is_dir() and not target.is_symlink():
                            shutil.rmtree(target, ignore_errors=True)
                        else:
                            target.unlink()
                    shutil.copy2(source, target, follow_symlinks=False)
            except OSError as exc:
                logger.debug(
                    "Failed to sync verification helper %s into %s: %s",
                    rel_path,
                    worktree,
                    exc,
                )

    def capture_baseline(self, repo_path: str, test_command: str) -> BaselineResult:
        """Run the unmodified repository test suite once and cache the result."""
        cache_key = (str(Path(repo_path).resolve()), test_command)
        if cache_key in self._baseline_cache:
            return self._baseline_cache[cache_key]

        start = time.time()
        worktree = Path(repo_path)
        normalized_command = self._normalize_pytest_command(
            test_command,
            force_verbose=True,
            worktree=worktree,
        )
        command = normalized_command
        report_path: Optional[Path] = None
        report_started_at: Optional[float] = None
        if self._is_pytest_command(test_command):
            report_started_at = start
            command, report_path = self._prepare_pytest_json_report_command_for_worktree(
                normalized_command,
                label="baseline",
                worktree=worktree,
            )

        try:
            result = self._run_command_with_completion_report(
                worktree,
                command,
                timeout=self.full_test_timeout,
                completion_report_path=report_path,
                completion_report_started_at=report_started_at,
            )
            if (
                report_path is not None
                and result.returncode != 0
                and self._pytest_json_report_command_needs_plain_retry(result.output)
            ):
                self._cleanup_report_path(report_path)
                report_path = None
                result = self._run_command(
                    worktree,
                    normalized_command,
                    timeout=self.full_test_timeout,
                )

            stdout_outcomes = self._parse_case_outcomes(result.output)
            report_outcomes: dict[str, str] = {}
            collected_tests: set[str] = set()
            collected_test_count = 0
            payload = self._load_pytest_json_report_payload(
                report_path,
                minimum_mtime=report_started_at,
            )
            if payload is not None:
                report_tests = extract_pytest_report_tests(payload)
                report_outcomes = extract_pytest_report_outcomes(report_tests)
                collected_tests = set(report_outcomes)
                summary = payload.get("summary") if isinstance(payload, dict) else {}
                if isinstance(summary, dict):
                    collected_total = summary.get("collected")
                    if isinstance(collected_total, int) and collected_total >= 0:
                        collected_test_count = collected_total
                if collected_test_count <= 0:
                    collected_test_count = len(report_outcomes) or len(report_tests)
            elif re.search(
                r"\b(\d+)\s+(passed|pass|failed|failures|failure|errors|error)\b",
                result.output,
                flags=re.IGNORECASE,
            ):
                parsed = self._parse_test_output(result.output, result.returncode)
                parsed_total = self._parse_observed_test_count(result.output, parsed)
                if parsed_total > 0:
                    collected_test_count = parsed_total

            if stdout_outcomes:
                baseline = BaselineResult(
                    passing_tests={
                        name for name, status in stdout_outcomes.items() if status == "PASSED"
                    },
                    failing_tests={
                        name
                        for name, status in stdout_outcomes.items()
                        if status in {"FAILED", "ERROR"}
                    },
                    collected_tests=collected_tests,
                    collected_test_count=collected_test_count,
                    total_duration=time.time() - start,
                    output=result.output,
                )
            elif report_outcomes:
                baseline = BaselineResult(
                    passing_tests={
                        name
                        for name, outcome in report_outcomes.items()
                        if outcome in {"passed", "xfailed", "xpassed"}
                    },
                    failing_tests={
                        name
                        for name, outcome in report_outcomes.items()
                        if outcome in {"failed", "error"}
                    },
                    collected_tests=collected_tests,
                    collected_test_count=collected_test_count,
                    total_duration=time.time() - start,
                    output=result.output,
                )
            else:
                baseline = BaselineResult(
                    passing_tests={_SYNTHETIC_SUITE_TEST} if result.returncode == 0 else set(),
                    failing_tests=set() if result.returncode == 0 else {_SYNTHETIC_SUITE_TEST},
                    collected_tests=collected_tests,
                    collected_test_count=collected_test_count,
                    total_duration=time.time() - start,
                    output=result.output,
                )
        finally:
            self._cleanup_report_path(report_path)

        probe_output = self._capture_collection_trace_probe(
            Path(repo_path),
            test_command,
            baseline,
        )
        if self._is_informative_collection_probe_output(probe_output):
            baseline.output = (
                baseline.output.rstrip()
                + "\n\n[APEX baseline trace probe]\n"
                + probe_output.strip()
            ).strip()

        self._baseline_cache[cache_key] = baseline
        return baseline

    def prune_by_regression(
        self,
        patch_worktree: str,
        baseline: BaselineResult,
        test_command: str,
        baseline_ref: Optional[str] = None,
    ) -> PruneResult:
        """Discard syntactically invalid or regressing candidates."""
        worktree = Path(patch_worktree)
        changed_files = self._resolve_changed_files(worktree, baseline_ref=baseline_ref)
        if not self._check_syntax(worktree, changed_files):
            return PruneResult(
                is_valid=False,
                regressed_tests=[],
                still_passing=[],
                reason="syntax_invalid",
            )

        if not baseline.passing_tests:
            return PruneResult(is_valid=True, still_passing=[])

        regression_output = ""
        regression_returncode = 0
        outcomes: dict[str, str] = {}

        if _SYNTHETIC_SUITE_TEST not in baseline.passing_tests:
            targeted_result = self._run_baseline_passing_tests(
                worktree,
                test_command,
                sorted(baseline.passing_tests),
            )
            if targeted_result is not None:
                regression_returncode = targeted_result.returncode
                regression_output = targeted_result.output
                outcomes = self._parse_case_outcomes(regression_output)

        if not outcomes and _SYNTHETIC_SUITE_TEST not in baseline.passing_tests:
            fallback_regression = self._run_command(
                worktree,
                self._normalize_pytest_command(
                    test_command,
                    force_verbose=True,
                    worktree=worktree,
                ),
                timeout=self.full_test_timeout,
            )
            regression_returncode = fallback_regression.returncode
            regression_output = fallback_regression.output
            outcomes = self._parse_case_outcomes(regression_output)

        regression = _CommandResult(returncode=regression_returncode, output=regression_output)
        if not regression.output:
            regression = self._run_command(
                worktree,
                self._normalize_pytest_command(
                    test_command,
                    force_verbose=True,
                    worktree=worktree,
                ),
                timeout=self.full_test_timeout,
            )
            outcomes = self._parse_case_outcomes(regression.output)

        if _SYNTHETIC_SUITE_TEST in baseline.passing_tests:
            if regression.returncode == 0:
                return PruneResult(
                    is_valid=True,
                    regressed_tests=[],
                    still_passing=[_SYNTHETIC_SUITE_TEST],
                    output=regression.output,
                )
            return PruneResult(
                is_valid=False,
                regressed_tests=[_SYNTHETIC_SUITE_TEST],
                still_passing=[],
                output=regression.output,
                reason="suite_regressed",
            )

        regressed = sorted(
            test_id
            for test_id in baseline.passing_tests
            if outcomes.get(test_id) in {"FAILED", "ERROR"}
        )
        # Collection errors flagged by ``_parse_case_outcomes`` use the
        # offending file path as the key (e.g. ``tests/test_foo.py``)
        # while baseline ``passing_tests`` are full nodeids
        # (``tests/test_foo.py::test_x``). Without expansion, a collection
        # error that breaks every test in a file would not appear in
        # ``regressed`` and the coarse ``returncode != 0`` fallback below
        # would mark *all* baseline tests as regressed instead of only the
        # affected file's tests. Expand collection-error keys to all
        # baseline nodeids that live in that file.
        collection_failed_files = sorted(
            file_id
            for file_id, status in outcomes.items()
            if status in {"FAILED", "ERROR"} and "::" not in file_id
        )
        if collection_failed_files:
            for collection_file in collection_failed_files:
                file_prefix = f"{collection_file}::"
                for test_id in baseline.passing_tests:
                    if test_id.startswith(file_prefix) and test_id not in regressed:
                        regressed.append(test_id)
            regressed = sorted(regressed)
        if regression.returncode != 0 and not regressed and baseline.passing_tests:
            regressed = sorted(baseline.passing_tests)

        still_passing = sorted(
            test_id for test_id in baseline.passing_tests if test_id not in regressed
        )
        return PruneResult(
            is_valid=not regressed,
            regressed_tests=regressed,
            still_passing=still_passing,
            output=regression.output,
            reason="" if not regressed else "regression_detected",
        )

    def validate_synthetic_test_portfolio(
        self,
        *,
        worktree_path: str,
        test_portfolio: Any,
        test_command: Optional[str] = None,
        test_inventory: Optional[Any] = None,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> dict[str, Any]:
        def _emit_progress(phase: str, **extra_fields: Any) -> None:
            if progress_callback is None:
                return
            payload = {"phase": phase}
            if extra_fields:
                payload.update(extra_fields)
            try:
                progress_callback(payload)
            except Exception:
                logger.debug(
                    "synthetic test portfolio progress callback failed",
                    exc_info=True,
                )

        normalized = normalize_test_suite_artifact_payload(test_portfolio)
        worktree = Path(worktree_path).resolve()
        effective_test_command = str(test_command or normalized.get("test_command") or "").strip()
        raw_artifacts = list(normalized.get("test_artifacts") or [])
        _emit_progress(
            "portfolio_validation_start",
            artifact_count=len(raw_artifacts),
            has_test_command=bool(effective_test_command),
        )

        baseline_result: Optional[BaselineResult] = None
        if effective_test_command:
            _emit_progress(
                "baseline_capture_start",
                test_command=effective_test_command,
            )
            try:
                baseline_result = self.capture_baseline(str(worktree), effective_test_command)
                _emit_progress(
                    "baseline_capture_complete",
                    collected_test_count=int(baseline_result.collected_test_count or 0),
                    failing_test_count=len(baseline_result.failing_tests),
                    passing_test_count=len(baseline_result.passing_tests),
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning(
                    "synthetic test portfolio baseline capture failed: %s",
                    exc,
                )
                _emit_progress(
                    "baseline_capture_failed",
                    error=str(exc),
                )

        resolved_inventory = self._resolve_test_inventory(
            test_inventory=test_inventory,
            expected_test_count=None,
            expected_test_ids=None,
            test_command=effective_test_command,
            baseline_result=baseline_result,
        )
        baseline_compare_worktree = (
            self.repo_path
            if self.repo_path.exists() and self.repo_path.resolve() != worktree
            else None
        )

        validated_entries: list[dict[str, Any]] = []
        targeted_supported_count = 0
        execution_success_count = 0
        baseline_preserving_count = 0
        differentiating_count = 0
        dual_version_verified_count = 0
        mutation_measured_count = 0
        mutation_ready_count = 0
        plausible_mutant_measured_artifact_count = 0
        plausible_mutant_total_count = 0
        plausible_mutant_survivor_artifact_count = 0

        for artifact_index, raw_entry in enumerate(raw_artifacts, start=1):
            entry = dict(raw_entry)
            artifact_path = str(entry.get("path") or "").strip()
            _emit_progress(
                "artifact_validation_start",
                artifact_index=artifact_index,
                artifact_count=len(raw_artifacts),
                artifact_path=artifact_path,
            )
            validation = self._validate_synthetic_test_artifact(
                worktree,
                entry,
                test_command=effective_test_command,
                test_inventory=resolved_inventory,
                baseline_result=baseline_result,
                check_baseline_preservation=True,
                rerun=True,
                measure_mutation=True,
            )
            primary_mutation_signal = float(validation.get("mutation_signal") or 0.0)
            primary_mutation_measured = bool(validation.get("mutation_signal_measured"))
            primary_mutation_mode = str(
                validation.get("mutation_discrimination_mode") or ""
            ).strip()
            primary_mutation_limitations = str(
                validation.get("mutation_discrimination_limitations") or ""
            ).strip()
            if baseline_compare_worktree is not None and validation.get(
                "execution_targeted_supported"
            ):
                baseline_variant = self._validate_synthetic_test_artifact(
                    baseline_compare_worktree,
                    entry,
                    test_command=effective_test_command,
                    test_inventory=resolved_inventory,
                    baseline_result=None,
                    check_baseline_preservation=False,
                    rerun=False,
                    measure_mutation=False,
                )
                baseline_variant_execution = bool(baseline_variant.get("execution_succeeded"))
                baseline_variant_supported = bool(
                    baseline_variant.get("execution_targeted_supported")
                )
                baseline_variant_collection_succeeded = bool(
                    baseline_variant.get("collection_succeeded")
                )
                baseline_variant_behavioral_failure_measured = bool(
                    baseline_variant.get("execution_behavioral_failure_measured")
                )
                baseline_variant_behavioral_failure = bool(
                    baseline_variant.get("execution_behavioral_failure")
                )
                baseline_variant_behaviorally_differentiating = bool(
                    baseline_variant_supported
                    and baseline_variant_collection_succeeded
                    and not baseline_variant_execution
                    and (
                        baseline_variant_behavioral_failure
                        if baseline_variant_behavioral_failure_measured
                        else True
                    )
                )
                baseline_variant_signal = (
                    1.0
                    if validation.get("execution_succeeded")
                    and baseline_variant_behaviorally_differentiating
                    else 0.0
                )
                validation["baseline_variant_execution_succeeded"] = baseline_variant_execution
                validation["baseline_variant_execution_supported"] = baseline_variant_supported
                validation["baseline_variant_collection_succeeded"] = (
                    baseline_variant_collection_succeeded
                )
                validation["baseline_variant_behavioral_failure_measured"] = (
                    baseline_variant_behavioral_failure_measured
                )
                validation["baseline_variant_behavioral_failure"] = (
                    baseline_variant_behavioral_failure
                )
                validation["dual_version_verified"] = bool(
                    validation.get("execution_succeeded")
                    and baseline_variant_behaviorally_differentiating
                )
                validation["dual_version_measured"] = bool(
                    baseline_variant_supported and baseline_variant_collection_succeeded
                )
                validation["baseline_variant_signal"] = round(baseline_variant_signal, 4)
                if primary_mutation_measured:
                    validation["mutation_signal"] = round(primary_mutation_signal, 4)
                    validation["mutation_signal_measured"] = True
                    validation["mutation_discrimination_mode"] = (
                        "plausible_mutant_bank+baseline_variant"
                        if baseline_variant_supported
                        else "plausible_mutant_bank"
                    )
                    limitation_bits = [
                        primary_mutation_limitations,
                        (
                            "Dual-version verification also ran against the original repository snapshot."
                            if baseline_variant_behaviorally_differentiating
                            else (
                                "Original-repository differential checking reached the synthetic file but it failed before a behavioral test-call failure was established."
                                if baseline_variant_supported
                                and not baseline_variant_collection_succeeded
                                else (
                                    "Original-repository differential checking observed a setup/teardown failure rather than a behavioral test-call failure."
                                    if baseline_variant_supported
                                    and baseline_variant_behavioral_failure_measured
                                    and not baseline_variant_behavioral_failure
                                    else (
                                        "Original-repository differential checking ran, but no passing-current / failing-original behavioral split was established."
                                        if baseline_variant_supported
                                        else "Original-repository differential checking was unavailable or unsupported for this artifact."
                                    )
                                )
                            )
                        ),
                    ]
                    validation["mutation_discrimination_limitations"] = " ".join(
                        bit.strip() for bit in limitation_bits if bit and bit.strip()
                    ).strip()
                else:
                    validation["mutation_signal"] = round(baseline_variant_signal, 4)
                    validation["mutation_signal_measured"] = bool(
                        baseline_variant_supported and baseline_variant_collection_succeeded
                    )
                    validation["mutation_discrimination_mode"] = (
                        "baseline_variant_only"
                        if baseline_variant_supported and baseline_variant_collection_succeeded
                        else "baseline_variant_unavailable"
                    )
                    validation["mutation_discrimination_limitations"] = (
                        "Differentially checked against the original repository snapshot only; "
                        "no plausible mutant bank was executed."
                        if baseline_variant_behaviorally_differentiating
                        else (
                            "An original-code comparison workspace existed, but the synthetic artifact did not collect cleanly there."
                            if baseline_variant_supported
                            and not baseline_variant_collection_succeeded
                            else (
                                "An original-code comparison workspace existed, but the synthetic artifact failed in setup/teardown rather than in the behavioral test call."
                                if baseline_variant_supported
                                and baseline_variant_behavioral_failure_measured
                                and not baseline_variant_behavioral_failure
                                else "An original-code comparison workspace existed, but targeted execution was not supported there."
                            )
                        )
                    )
            else:
                validation["dual_version_verified"] = False
                validation["dual_version_measured"] = False
                if primary_mutation_measured:
                    validation["mutation_signal"] = round(primary_mutation_signal, 4)
                    validation["mutation_signal_measured"] = True
                    validation["mutation_discrimination_mode"] = (
                        primary_mutation_mode or "plausible_mutant_bank"
                    )
                    validation["mutation_discrimination_limitations"] = (
                        primary_mutation_limitations
                        or "Executed a bounded plausible-mutant bank only; no original-code comparison workspace was available."
                    )
                else:
                    validation["mutation_signal"] = 0.0
                    validation["mutation_signal_measured"] = False
                    validation["mutation_discrimination_mode"] = "unmeasured"
                    validation["mutation_discrimination_limitations"] = (
                        primary_mutation_limitations
                        or "No separate original-code comparison workspace was available."
                    )
            validation["mutation_discrimination_passed"] = (
                self._validation_mutation_discrimination_passed(validation)
            )
            if bool(validation.get("quarantined")):
                _emit_progress(
                    "artifact_validation_quarantined",
                    artifact_index=artifact_index,
                    artifact_count=len(raw_artifacts),
                    artifact_path=artifact_path,
                    quarantine_reason=str(validation.get("quarantine_reason") or ""),
                )
                continue

            if validation.get("execution_targeted_supported"):
                targeted_supported_count += 1
            if validation.get("execution_succeeded"):
                execution_success_count += 1
            if validation.get("baseline_preserved"):
                baseline_preserving_count += 1
            if float(validation.get("mutation_signal") or 0.0) > 0.0:
                differentiating_count += 1
            if bool(validation.get("dual_version_verified")):
                dual_version_verified_count += 1
            if bool(validation.get("mutation_signal_measured")):
                mutation_measured_count += 1
            if bool(validation.get("mutation_discrimination_passed")):
                mutation_ready_count += 1
            if int(validation.get("plausible_mutant_count") or 0) > 0:
                plausible_mutant_measured_artifact_count += 1
                plausible_mutant_total_count += int(validation.get("plausible_mutant_count") or 0)
            if int(validation.get("plausible_mutant_survived_count") or 0) > 0:
                plausible_mutant_survivor_artifact_count += 1

            entry["validation"] = validation
            entry["dual_version_verified"] = bool(validation.get("dual_version_verified"))
            validated_entries.append(entry)
            _emit_progress(
                "artifact_validation_complete",
                artifact_index=artifact_index,
                artifact_count=len(raw_artifacts),
                artifact_path=artifact_path,
                execution_succeeded=bool(validation.get("execution_succeeded")),
                dual_version_verified=bool(validation.get("dual_version_verified")),
                mutation_signal=float(validation.get("mutation_signal") or 0.0),
                mutation_discrimination_passed=bool(
                    validation.get("mutation_discrimination_passed")
                ),
            )

        normalized["test_artifacts"] = validated_entries
        normalized["framework"] = (
            str(normalized.get("framework") or "").strip() or resolved_inventory.framework
        )
        normalized["language"] = (
            str(normalized.get("language") or "").strip().lower() or resolved_inventory.language
        )

        existing_summary = dict(normalized.get("validation_summary") or {})
        objective_validation: dict[str, dict[str, Any]] = {}
        for raw_objective in list(normalized.get("test_objectives") or []):
            objective_id = str(dict(raw_objective).get("objective_id") or "").strip()
            if not objective_id:
                continue
            objective_validation.setdefault(
                objective_id,
                {
                    "artifact_count": 0,
                    "execution_success_count": 0,
                    "baseline_preserved_count": 0,
                    "dual_version_verified_count": 0,
                    "mutation_discrimination_passed_count": 0,
                    "pass_then_invert_complete_count": 0,
                    "design_metadata_complete_count": 0,
                },
            )
        for entry in validated_entries:
            objective_id = str(entry.get("objective_id") or "").strip()
            if not objective_id:
                continue
            objective_summary = objective_validation.setdefault(
                objective_id,
                {
                    "artifact_count": 0,
                    "execution_success_count": 0,
                    "baseline_preserved_count": 0,
                    "dual_version_verified_count": 0,
                    "mutation_discrimination_passed_count": 0,
                    "pass_then_invert_complete_count": 0,
                    "design_metadata_complete_count": 0,
                },
            )
            objective_summary["artifact_count"] += 1
            validation = dict(entry.get("validation") or {})
            if validation.get("execution_succeeded"):
                objective_summary["execution_success_count"] += 1
            if validation.get("baseline_preserved"):
                objective_summary["baseline_preserved_count"] += 1
            if validation.get("dual_version_verified"):
                objective_summary["dual_version_verified_count"] += 1
            if validation.get("mutation_discrimination_passed"):
                objective_summary["mutation_discrimination_passed_count"] += 1
            if _pass_then_invert_complete(entry.get("pass_then_invert")):
                objective_summary["pass_then_invert_complete_count"] += 1
            if not _artifact_design_metadata_gaps(entry):
                objective_summary["design_metadata_complete_count"] += 1

        for objective_summary in objective_validation.values():
            artifact_count = int(objective_summary.get("artifact_count") or 0)
            execution_success_count = int(objective_summary.get("execution_success_count") or 0)
            baseline_preserved_count = int(objective_summary.get("baseline_preserved_count") or 0)
            dual_version_verified_count = int(
                objective_summary.get("dual_version_verified_count") or 0
            )
            mutation_discrimination_passed_count = int(
                objective_summary.get("mutation_discrimination_passed_count") or 0
            )
            pass_then_invert_complete_count = int(
                objective_summary.get("pass_then_invert_complete_count") or 0
            )
            design_metadata_complete_count = int(
                objective_summary.get("design_metadata_complete_count") or 0
            )
            core_ready = artifact_count > 0 and execution_success_count >= artifact_count
            iso_ready = bool(
                core_ready
                and baseline_preserved_count >= artifact_count
                and pass_then_invert_complete_count >= artifact_count
            )
            strict_ready = bool(
                iso_ready
                and dual_version_verified_count >= artifact_count
                and mutation_discrimination_passed_count >= artifact_count
                and design_metadata_complete_count >= artifact_count
            )
            objective_summary["core_ready"] = core_ready
            objective_summary["iso_ready"] = iso_ready
            objective_summary["strict_ready"] = strict_ready
            objective_summary["validation_level_status"] = (
                "strict"
                if strict_ready
                else "iso"
                if iso_ready
                else "core"
                if core_ready
                else "draft"
            )
            if strict_ready:
                objective_summary["status"] = "verified"
            elif execution_success_count > 0:
                objective_summary["status"] = "candidate"
            else:
                objective_summary["status"] = "draft"

        milestone_validation: dict[str, dict[str, Any]] = {}
        milestone_objective_ids: dict[str, set[str]] = {}
        objective_to_milestone: dict[str, str] = {}

        for raw_milestone in list(normalized.get("milestones") or []):
            milestone = dict(raw_milestone or {})
            milestone_id = str(milestone.get("milestone_id") or "").strip()
            if not milestone_id:
                continue
            milestone_validation.setdefault(
                milestone_id,
                {
                    "artifact_count": 0,
                    "execution_success_count": 0,
                    "baseline_preserved_count": 0,
                    "dual_version_verified_count": 0,
                    "mutation_discrimination_passed_count": 0,
                    "pass_then_invert_complete_count": 0,
                    "design_metadata_complete_count": 0,
                    "validation_level": str(milestone.get("validation_level") or "strict")
                    .strip()
                    .lower(),
                },
            )
            milestone_objective_ids.setdefault(milestone_id, set()).update(
                str(value).strip()
                for value in list(milestone.get("objective_ids") or [])
                if str(value).strip()
            )

        for raw_objective in list(normalized.get("test_objectives") or []):
            objective = dict(raw_objective or {})
            objective_id = str(objective.get("objective_id") or "").strip()
            milestone_id = str(objective.get("milestone_id") or "").strip()
            if not objective_id or not milestone_id:
                continue
            objective_to_milestone[objective_id] = milestone_id
            milestone_objective_ids.setdefault(milestone_id, set()).add(objective_id)
            milestone_validation.setdefault(
                milestone_id,
                {
                    "artifact_count": 0,
                    "execution_success_count": 0,
                    "baseline_preserved_count": 0,
                    "dual_version_verified_count": 0,
                    "mutation_discrimination_passed_count": 0,
                    "pass_then_invert_complete_count": 0,
                    "design_metadata_complete_count": 0,
                    "validation_level": "strict",
                },
            )

        for entry in validated_entries:
            objective_id = str(entry.get("objective_id") or "").strip()
            milestone_id = str(entry.get("milestone_id") or "").strip()
            if not milestone_id:
                milestone_id = objective_to_milestone.get(objective_id, "")
            if not milestone_id:
                continue
            milestone_summary = milestone_validation.setdefault(
                milestone_id,
                {
                    "artifact_count": 0,
                    "execution_success_count": 0,
                    "baseline_preserved_count": 0,
                    "dual_version_verified_count": 0,
                    "mutation_discrimination_passed_count": 0,
                    "pass_then_invert_complete_count": 0,
                    "design_metadata_complete_count": 0,
                    "validation_level": "strict",
                },
            )
            if objective_id:
                milestone_objective_ids.setdefault(milestone_id, set()).add(objective_id)
            milestone_summary["artifact_count"] += 1
            validation = dict(entry.get("validation") or {})
            if validation.get("execution_succeeded"):
                milestone_summary["execution_success_count"] += 1
            if validation.get("baseline_preserved"):
                milestone_summary["baseline_preserved_count"] += 1
            if validation.get("dual_version_verified"):
                milestone_summary["dual_version_verified_count"] += 1
            if validation.get("mutation_discrimination_passed"):
                milestone_summary["mutation_discrimination_passed_count"] += 1
            if _pass_then_invert_complete(entry.get("pass_then_invert")):
                milestone_summary["pass_then_invert_complete_count"] += 1
            if not _artifact_design_metadata_gaps(entry):
                milestone_summary["design_metadata_complete_count"] += 1

        core_ready_milestone_count = 0
        iso_ready_milestone_count = 0
        strict_ready_milestone_count = 0
        for milestone_id, milestone_summary in milestone_validation.items():
            related_objective_ids = sorted(milestone_objective_ids.get(milestone_id) or set())
            related_objective_summaries = [
                dict(objective_validation.get(objective_id) or {})
                for objective_id in related_objective_ids
                if objective_id in objective_validation
            ]
            if related_objective_summaries:
                core_ready = all(
                    bool(summary.get("core_ready")) for summary in related_objective_summaries
                )
                iso_ready = all(
                    bool(summary.get("iso_ready")) for summary in related_objective_summaries
                )
                strict_ready = all(
                    bool(summary.get("strict_ready")) for summary in related_objective_summaries
                )
            else:
                artifact_count = int(milestone_summary.get("artifact_count") or 0)
                core_ready = bool(
                    artifact_count > 0
                    and int(milestone_summary.get("execution_success_count") or 0) >= artifact_count
                )
                iso_ready = bool(
                    core_ready
                    and int(milestone_summary.get("baseline_preserved_count") or 0)
                    >= artifact_count
                    and int(milestone_summary.get("pass_then_invert_complete_count") or 0)
                    >= artifact_count
                )
                strict_ready = bool(
                    iso_ready
                    and int(milestone_summary.get("dual_version_verified_count") or 0)
                    >= artifact_count
                    and int(milestone_summary.get("mutation_discrimination_passed_count") or 0)
                    >= artifact_count
                    and int(milestone_summary.get("design_metadata_complete_count") or 0)
                    >= artifact_count
                )
            if core_ready:
                core_ready_milestone_count += 1
            if iso_ready:
                iso_ready_milestone_count += 1
            if strict_ready:
                strict_ready_milestone_count += 1
            required_level = (
                str(milestone_summary.get("validation_level") or "strict").strip().lower()
            )
            readiness_by_level = {
                "strict": strict_ready,
                "iso": iso_ready,
                "core": core_ready,
            }
            milestone_summary["objective_ids"] = related_objective_ids
            milestone_summary["objective_count"] = len(related_objective_ids)
            milestone_summary["core_ready"] = core_ready
            milestone_summary["iso_ready"] = iso_ready
            milestone_summary["strict_ready"] = strict_ready
            milestone_summary["validation_level_status"] = (
                "strict"
                if strict_ready
                else "iso"
                if iso_ready
                else "core"
                if core_ready
                else "draft"
            )
            milestone_summary["blocking_objectives"] = [
                objective_id
                for objective_id in related_objective_ids
                if not bool(
                    dict(objective_validation.get(objective_id) or {}).get(
                        f"{required_level}_ready"
                    )
                )
            ]
            milestone_summary["ready_for_required_level"] = bool(
                readiness_by_level.get(required_level, strict_ready)
            )

        regression_suite_summary = dict(normalized.get("regression_suite_summary") or {})
        regression_suite_summary.update(
            {
                "milestone_validation": milestone_validation,
                "milestone_count": len(milestone_validation),
                "core_ready_milestone_count": core_ready_milestone_count,
                "iso_ready_milestone_count": iso_ready_milestone_count,
                "strict_ready_milestone_count": strict_ready_milestone_count,
                "strict_ready": bool(milestone_validation)
                and strict_ready_milestone_count >= len(milestone_validation),
            }
        )
        normalized["regression_suite_summary"] = regression_suite_summary
        mutation_modes = {
            str(
                dict(entry.get("validation") or {}).get("mutation_discrimination_mode") or ""
            ).strip()
            for entry in validated_entries
            if str(
                dict(entry.get("validation") or {}).get("mutation_discrimination_mode") or ""
            ).strip()
        }
        if any(mode == "plausible_mutant_bank+baseline_variant" for mode in mutation_modes):
            portfolio_mutation_mode = "plausible_mutant_bank+baseline_variant"
        elif any("plausible_mutant_bank" in mode for mode in mutation_modes):
            portfolio_mutation_mode = "plausible_mutant_bank"
        elif baseline_compare_worktree is not None:
            portfolio_mutation_mode = "baseline_variant_only"
        else:
            portfolio_mutation_mode = "unmeasured"
        existing_summary.update(
            {
                "artifact_count": len(validated_entries),
                "quarantined_test_paths": self.quarantined_paths,
                "quarantined_test_path_count": len(self._quarantined_paths),
                "validated_artifact_count": sum(
                    1
                    for entry in validated_entries
                    if bool(dict(entry.get("validation") or {}).get("artifact_discovered"))
                ),
                "targeted_execution_supported_count": targeted_supported_count,
                "execution_success_count": execution_success_count,
                "baseline_preserving_artifact_count": baseline_preserving_count,
                "differentiating_artifact_count": differentiating_count,
                "dual_version_verified_artifact_count": dual_version_verified_count,
                "mutation_measured_artifact_count": mutation_measured_count,
                "mutation_discrimination_passed_artifact_count": mutation_ready_count,
                "baseline_comparison_available": baseline_compare_worktree is not None,
                "test_inventory_framework": resolved_inventory.framework,
                "test_inventory_language": resolved_inventory.language,
                "test_inventory_collection_command": resolved_inventory.collection_command,
                "mutation_discrimination_mode": portfolio_mutation_mode,
                "mutation_discrimination_limitations": (
                    "Executed a bounded plausible-mutant bank on selected source files for the measured artifacts; "
                    "coverage remains partial and may fall back to original-snapshot differential checks for some artifacts."
                    if plausible_mutant_measured_artifact_count > 0
                    else "Current validation checks generated tests against the original repository "
                    "snapshot only when available; no plausible mutant bank was measured."
                    if baseline_compare_worktree is not None
                    else "No plausible mutant bank or original-code differential check was available."
                ),
                "plausible_mutant_measured_artifact_count": plausible_mutant_measured_artifact_count,
                "plausible_mutant_total_count": plausible_mutant_total_count,
                "plausible_mutant_survivor_artifact_count": plausible_mutant_survivor_artifact_count,
                "objective_validation": objective_validation,
                "milestone_validation": milestone_validation,
                "core_ready_milestone_count": core_ready_milestone_count,
                "iso_ready_milestone_count": iso_ready_milestone_count,
                "strict_ready_milestone_count": strict_ready_milestone_count,
                "strict_ready": bool(milestone_validation)
                and strict_ready_milestone_count >= len(milestone_validation),
            }
        )
        normalized["validation_summary"] = existing_summary
        _emit_progress(
            "portfolio_validation_complete",
            artifact_count=len(validated_entries),
            execution_success_count=execution_success_count,
            dual_version_verified_count=dual_version_verified_count,
            mutation_ready_count=mutation_ready_count,
        )
        return normalized

    def _validate_synthetic_test_artifact(
        self,
        worktree: Path,
        artifact: dict[str, Any],
        *,
        test_command: Optional[str],
        test_inventory: TestInventory,
        baseline_result: Optional[BaselineResult],
        check_baseline_preservation: bool,
        rerun: bool,
        measure_mutation: bool = True,
    ) -> dict[str, Any]:
        normalized_inventory = (
            test_inventory.normalized()
            if isinstance(test_inventory, TestInventory)
            else TestInventory()
        )
        effective_test_command = str(
            test_command or artifact.get("test_command") or normalized_inventory.test_command or ""
        ).strip()
        framework = infer_test_inventory_framework(
            explicit_framework=str(
                artifact.get("framework") or normalized_inventory.framework or ""
            ),
            test_command=effective_test_command,
        )
        language = str(
            artifact.get("language") or normalized_inventory.language or ""
        ).strip().lower() or default_test_inventory_language(framework)
        relative_path = str(artifact.get("path") or "").strip().replace("\\", "/")
        content = str(artifact.get("content") or "").replace("\r\n", "\n")
        validation: dict[str, Any] = {
            "artifact_discovered": False,
            "collection_succeeded": False,
            "execution_targeted_supported": False,
            "execution_succeeded": False,
            "execution_failed_outcome_count": 0,
            "execution_error_outcome_count": 0,
            "execution_behavioral_failure_measured": False,
            "execution_behavioral_failure": False,
            "rerun_consistent": False,
            "baseline_preservation_measured": False,
            "baseline_preserved": False,
            "coverage_signal": 0.0,
            "coverage_signal_measured": False,
            "mutation_signal": 0.0,
            "mutation_signal_measured": False,
            "mutation_discrimination_mode": "unmeasured",
            "mutation_discrimination_limitations": "",
            "plausible_mutant_count": 0,
            "plausible_mutant_killed_count": 0,
            "plausible_mutant_survived_count": 0,
            "plausible_mutant_kill_rate": 0.0,
            "plausible_mutant_source_files": [],
            "plausible_mutant_kills": [],
            "plausible_mutant_survivors": [],
            "flake_signal": 0.0,
            "collection_output_excerpt": "",
            "execution_output_excerpt": "",
            "baseline_regression_output_excerpt": "",
            "execution_feedback_summary": "",
            "framework": framework,
            "language": language,
            "materialization_mode": "replace",
        }
        if relative_path in self._quarantined_paths:
            validation["quarantined"] = True
            validation["quarantine_reason"] = self._quarantined_paths[relative_path]
            validation["execution_feedback_summary"] = (
                "Artifact path is quarantined after a prior syntax failure; "
                "drop it or rewrite it under a fresh valid artifact path."
            )
            return validation
        if not worktree.exists() or not relative_path or not content.strip():
            validation["execution_feedback_summary"] = (
                "Artifact path or content was missing, so validation could not run."
            )
            return validation

        sandbox_root, sanitized_env = self._prepare_sandboxed_workspace_copy(worktree)
        if sandbox_root is None:
            validation["execution_feedback_summary"] = (
                "Sandbox preparation failed before artifact validation could run."
            )
            return validation

        try:
            validation["materialization_mode"] = self._materialize_synthetic_portfolio_artifact(
                sandbox_root=sandbox_root,
                artifact_path=relative_path,
                content=content,
                requested_mode=str(artifact.get("materialization_mode") or ""),
            )

            collection_command, targeted_collection = (
                self._build_synthetic_portfolio_collection_command(
                    artifact_path=relative_path,
                    framework=framework,
                    test_command=effective_test_command,
                    worktree=sandbox_root,
                )
            )
            execution_command, execution_targeted = (
                self._build_synthetic_portfolio_execution_command(
                    artifact_path=relative_path,
                    framework=framework,
                    test_command=effective_test_command,
                    worktree=sandbox_root,
                )
            )
            validation["execution_targeted_supported"] = bool(
                execution_command and execution_targeted
            )
            if collection_command:
                validation["collection_command"] = collection_command
                collection = self._run_synthetic_portfolio_command(
                    sandbox_root,
                    collection_command,
                    env=sanitized_env,
                )
                validation["collection_returncode"] = int(collection.returncode)
                collection_succeeded = (
                    collection.returncode == 0
                    and not self._output_indicates_no_tests(collection.output)
                )
                validation["collection_succeeded"] = collection_succeeded
                if not collection_succeeded:
                    validation["collection_output_excerpt"] = _synthetic_validation_output_excerpt(
                        collection.output
                    )
                    if "syntaxerror" in str(collection.output or "").lower():
                        reason = validation["collection_output_excerpt"] or "SyntaxError"
                        self._quarantine_path(relative_path, reason)
                        validation["quarantined"] = True
                        validation["quarantine_reason"] = self._quarantined_paths[relative_path]
                validation["artifact_discovered"] = self._portfolio_artifact_discovered(
                    artifact_path=relative_path,
                    output=collection.output,
                    collection_succeeded=collection_succeeded,
                    targeted_collection=targeted_collection,
                )

            if execution_command:
                validation["execution_command"] = execution_command
                execution_report_path: Optional[Path] = None
                execution_report_minimum_mtime: Optional[float] = None
                execution_command_to_run = execution_command
                if framework == "pytest":
                    execution_command_to_run, execution_report_path = (
                        self._prepare_pytest_json_report_command_for_worktree(
                            execution_command,
                            label=f"synthetic-{Path(relative_path).stem}",
                            worktree=sandbox_root,
                        )
                    )
                    execution_report_minimum_mtime = time.time()
                try:
                    execution = self._run_synthetic_portfolio_command(
                        sandbox_root,
                        execution_command_to_run,
                        env=sanitized_env,
                    )
                    if (
                        execution_report_path is not None
                        and self._pytest_json_report_command_needs_plain_retry(execution.output)
                    ):
                        self._cleanup_report_path(execution_report_path)
                        execution_report_path = None
                        execution_report_minimum_mtime = None
                        execution = self._run_synthetic_portfolio_command(
                            sandbox_root,
                            execution_command,
                            env=sanitized_env,
                        )
                    if framework == "pytest":
                        payload = self._load_pytest_json_report_payload(
                            execution_report_path,
                            minimum_mtime=execution_report_minimum_mtime,
                        )
                        report_tests = (
                            extract_pytest_report_tests(payload) if payload is not None else []
                        )
                        report_outcomes = extract_pytest_report_outcomes(report_tests)
                        failed_outcome_count = sum(
                            1 for outcome in report_outcomes.values() if outcome == "failed"
                        )
                        error_outcome_count = sum(
                            1 for outcome in report_outcomes.values() if outcome == "error"
                        )
                        validation["execution_failed_outcome_count"] = int(failed_outcome_count)
                        validation["execution_error_outcome_count"] = int(error_outcome_count)
                        if failed_outcome_count > 0 or error_outcome_count > 0:
                            validation["execution_behavioral_failure_measured"] = True
                            validation["execution_behavioral_failure"] = bool(
                                failed_outcome_count > 0 and error_outcome_count == 0
                            )
                finally:
                    self._cleanup_report_path(execution_report_path)
                validation["execution_returncode"] = int(execution.returncode)
                validation["execution_succeeded"] = execution.returncode == 0
                if not validation["execution_succeeded"]:
                    validation["execution_output_excerpt"] = _synthetic_validation_output_excerpt(
                        execution.output
                    )
                    if "syntaxerror" in str(execution.output or "").lower():
                        reason = validation["execution_output_excerpt"] or "SyntaxError"
                        self._quarantine_path(relative_path, reason)
                        validation["quarantined"] = True
                        validation["quarantine_reason"] = self._quarantined_paths[relative_path]
                if validation["execution_succeeded"]:
                    validation["artifact_discovered"] = True
                    validation["collection_succeeded"] = True
                    if rerun:
                        rerun_result = self._run_synthetic_portfolio_command(
                            sandbox_root,
                            execution_command,
                            env=sanitized_env,
                        )
                        validation["rerun_consistent"] = rerun_result.returncode == 0
                    else:
                        validation["rerun_consistent"] = True

            if (
                check_baseline_preservation
                and baseline_result is not None
                and effective_test_command
            ):
                prune = self.prune_by_regression(
                    str(sandbox_root),
                    baseline_result,
                    effective_test_command,
                )
                validation["baseline_preservation_measured"] = True
                validation["baseline_preserved"] = bool(prune.is_valid)
                if not prune.is_valid and validation["execution_succeeded"]:
                    validation["baseline_regression_output_excerpt"] = (
                        _synthetic_validation_output_excerpt(prune.output)
                    )

            if (
                measure_mutation
                and validation["execution_succeeded"]
                and validation["execution_targeted_supported"]
                and execution_command
            ):
                validation.update(
                    self._measure_plausible_mutant_discrimination(
                        sandbox_root=sandbox_root,
                        artifact=artifact,
                        language=language,
                        execution_command=execution_command,
                        env=sanitized_env,
                    )
                )

            coverage_signal = 0.0
            if validation["execution_succeeded"]:
                coverage_signal = 1.0 if validation["rerun_consistent"] else 0.85
            elif validation["artifact_discovered"] and validation["collection_succeeded"]:
                coverage_signal = (
                    0.65 if not validation.get("execution_targeted_supported") else 0.35
                )
            elif validation["artifact_discovered"]:
                coverage_signal = 0.2
            validation["coverage_signal"] = round(coverage_signal, 4)
            validation["coverage_signal_measured"] = True
            validation["flake_signal"] = 1.0 if validation["rerun_consistent"] else 0.0
            validation["execution_feedback_summary"] = _summarize_synthetic_validation_feedback(
                validation
            )
            return validation
        finally:
            shutil.rmtree(sandbox_root.parent, ignore_errors=True)

    def _measure_plausible_mutant_discrimination(
        self,
        *,
        sandbox_root: Path,
        artifact: dict[str, Any],
        language: str,
        execution_command: str,
        env: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        default = {
            "mutation_signal": 0.0,
            "mutation_signal_measured": False,
            "mutation_discrimination_mode": "unmeasured",
            "mutation_discrimination_limitations": "",
            "plausible_mutant_count": 0,
            "plausible_mutant_killed_count": 0,
            "plausible_mutant_survived_count": 0,
            "plausible_mutant_kill_rate": 0.0,
            "plausible_mutant_source_files": [],
            "plausible_mutant_kills": [],
            "plausible_mutant_survivors": [],
        }
        normalized_language = str(language or "").strip().lower()
        if normalized_language != "python":
            default["mutation_discrimination_limitations"] = (
                "Plausible mutant generation is currently implemented for Python source artifacts only."
            )
            return default

        candidate_files = self._mutation_candidate_source_files(
            artifact,
            worktree=sandbox_root,
            language=normalized_language,
        )
        default["plausible_mutant_source_files"] = list(candidate_files)
        if not candidate_files:
            default["mutation_discrimination_limitations"] = (
                "No non-test Python source focus files were attached to this artifact, "
                "so no plausible mutant bank could be generated."
            )
            return default

        symbol_hints = self._python_mutation_symbol_hints(artifact)
        mutants: list[dict[str, Any]] = []
        for rel_path in candidate_files[:_SYNTHETIC_MUTATION_MAX_FILES]:
            file_path = sandbox_root / rel_path
            try:
                source_text = file_path.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            mutants.extend(
                self._build_python_plausible_mutants(
                    artifact_path=rel_path,
                    source_text=source_text,
                    symbol_hints=symbol_hints,
                )
            )
            if len(mutants) >= _SYNTHETIC_MUTATION_MAX_MUTANTS_PER_ARTIFACT:
                break
        mutants = mutants[:_SYNTHETIC_MUTATION_MAX_MUTANTS_PER_ARTIFACT]
        if not mutants:
            default["mutation_discrimination_limitations"] = (
                "No bounded Python AST mutation opportunities were found in the artifact's focused source files."
            )
            return default

        original_sources: dict[str, str] = {}
        killed: list[dict[str, Any]] = []
        survived: list[dict[str, Any]] = []
        for mutant in mutants:
            rel_path = str(mutant.get("path") or "").strip()
            if not rel_path:
                continue
            target_path = sandbox_root / rel_path
            try:
                original_text = original_sources.setdefault(rel_path, target_path.read_text())
            except (OSError, UnicodeDecodeError):
                continue
            try:
                target_path.write_text(str(mutant.get("content") or ""))
                result = self._run_synthetic_portfolio_command(
                    sandbox_root,
                    execution_command,
                    env=env,
                )
            finally:
                try:
                    target_path.write_text(original_text)
                except OSError:
                    logger.warning("failed to restore mutant target %s", rel_path)
            mutant_result = {
                "mutant_id": str(mutant.get("mutant_id") or "").strip(),
                "path": rel_path,
                "kind": str(mutant.get("kind") or "").strip(),
                "line": int(mutant.get("line") or 0),
                "label": str(mutant.get("label") or "").strip(),
                "returncode": int(result.returncode),
            }
            if result.returncode == 0:
                survived.append(mutant_result)
            else:
                killed.append(mutant_result)

        measured_count = len(killed) + len(survived)
        if measured_count <= 0:
            default["mutation_discrimination_limitations"] = (
                "Mutant generation succeeded, but no mutants could be executed successfully."
            )
            return default

        kill_rate = len(killed) / measured_count
        return {
            "mutation_signal": round(kill_rate, 4),
            "mutation_signal_measured": True,
            "mutation_discrimination_passed": len(survived) == 0,
            "mutation_discrimination_mode": "plausible_mutant_bank",
            "mutation_discrimination_limitations": (
                f"Executed a bounded plausible-mutant bank ({measured_count} mutants) over "
                f"{len(candidate_files[:_SYNTHETIC_MUTATION_MAX_FILES])} focused Python source file(s)."
            ),
            "plausible_mutant_count": measured_count,
            "plausible_mutant_killed_count": len(killed),
            "plausible_mutant_survived_count": len(survived),
            "plausible_mutant_kill_rate": round(kill_rate, 4),
            "plausible_mutant_source_files": list(candidate_files),
            "plausible_mutant_kills": killed[:6],
            "plausible_mutant_survivors": survived[:6],
        }

    def _validation_mutation_discrimination_passed(
        self,
        validation: dict[str, Any],
    ) -> bool:
        plausible_mutant_count = int(validation.get("plausible_mutant_count") or 0)
        if plausible_mutant_count > 0:
            return (
                bool(validation.get("mutation_signal_measured"))
                and int(validation.get("plausible_mutant_survived_count") or 0) == 0
            )
        normalized_language = str(validation.get("language") or "").strip().lower()
        if (
            normalized_language == "python"
            and bool(validation.get("execution_succeeded"))
            and bool(validation.get("execution_targeted_supported"))
        ):
            return False
        if bool(validation.get("mutation_signal_measured")):
            return float(validation.get("mutation_signal") or 0.0) > 0.0
        return bool(validation.get("dual_version_verified"))

    def _mutation_candidate_source_files(
        self,
        artifact: dict[str, Any],
        *,
        worktree: Path,
        language: str,
    ) -> list[str]:
        candidates: list[str] = []
        for key in ("focus_files", "interface_specification", "contract_targets"):
            for value in list(artifact.get(key) or []):
                rel_path = self._artifact_mutation_repo_path(value, worktree=worktree)
                if not rel_path:
                    continue
                if not self._mutation_source_file_allowed(rel_path, language=language):
                    continue
                candidates.append(rel_path)
        deduped = list(dict.fromkeys(candidates))
        if deduped or language != "python":
            return deduped[:_SYNTHETIC_MUTATION_MAX_FILES]

        hints = self._python_mutation_symbol_hints(artifact)
        if not hints:
            return []
        definition_index = self._python_definition_index(worktree)
        scored: dict[str, int] = {}
        for hint in hints:
            for rel_path in list(definition_index.get(hint, []) or []):
                if not self._mutation_source_file_allowed(rel_path, language=language):
                    continue
                scored[rel_path] = int(scored.get(rel_path) or 0) + 1
        ranked = sorted(
            scored.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return [path for path, _score in ranked[:_SYNTHETIC_MUTATION_MAX_FILES]]

    def _artifact_mutation_repo_path(
        self,
        value: Any,
        *,
        worktree: Path,
    ) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        candidate = Path(str(raw).split("::", 1)[0].strip())
        if candidate.is_absolute():
            try:
                rel_path = candidate.resolve().relative_to(worktree.resolve()).as_posix()
            except ValueError:
                return ""
        else:
            rel_path = candidate.as_posix()
        rel_path = rel_path.replace("\\", "/").lstrip("./")
        if not rel_path:
            return ""
        if not (worktree / rel_path).is_file():
            return ""
        return rel_path

    def _mutation_source_file_allowed(self, rel_path: str, *, language: str) -> bool:
        normalized = str(rel_path or "").strip().replace("\\", "/")
        if not normalized or looks_like_test_path(normalized):
            return False
        if language == "python":
            return Path(normalized).suffix == ".py"
        return True

    def _python_mutation_symbol_hints(
        self,
        artifact: dict[str, Any],
    ) -> list[str]:
        hints: list[str] = []
        raw_values: list[Any] = (
            list(artifact.get("contract_targets") or [])
            + list(artifact.get("interface_specification") or [])
            + [artifact.get("objective") or ""]
        )
        for raw_value in raw_values:
            text = str(raw_value or "").strip()
            if not text:
                continue
            if self._artifact_mutation_text_looks_like_path(text):
                continue
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text.replace("::", ".")):
                lowered = token.lower()
                if lowered in {
                    "test",
                    "tests",
                    "path",
                    "paths",
                    "function",
                    "method",
                    "class",
                    "module",
                    "interface",
                    "specification",
                }:
                    continue
                hints.append(lowered)
        return list(dict.fromkeys(hints))

    def _artifact_mutation_text_looks_like_path(self, value: str) -> bool:
        text = str(value or "").strip().replace("\\", "/")
        if not text:
            return False
        candidate = text.split("::", 1)[0].strip()
        suffix = Path(candidate).suffix.lower()
        return bool(
            "/" in candidate
            or suffix in {".py", ".js", ".ts", ".tsx", ".java", ".rb", ".go", ".php"}
        )

    def _python_definition_index(self, worktree: Path) -> dict[str, list[str]]:
        cache_key = str(worktree.resolve())
        cached = self._python_definition_index_cache.get(cache_key)
        if cached is not None:
            return cached

        ignored_dirs = {
            ".git",
            ".hg",
            ".jj",
            ".sl",
            ".svn",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".venv",
            "venv",
            "node_modules",
            ".apex_agent_teams",
        }
        index: dict[str, list[str]] = {}
        for root, dirs, files in os.walk(worktree):
            dirs[:] = [
                directory
                for directory in dirs
                if directory not in ignored_dirs
                and not directory.startswith(".apex_")
                and not directory.startswith(".swebench_")
            ]
            root_path = Path(root)
            for filename in files:
                if not filename.endswith(".py"):
                    continue
                file_path = root_path / filename
                rel_path = file_path.relative_to(worktree).as_posix()
                if looks_like_test_path(rel_path):
                    continue
                try:
                    tree = ast.parse(file_path.read_text())
                except (OSError, UnicodeDecodeError, SyntaxError):
                    continue
                names = {
                    str(node.name).strip().lower()
                    for node in ast.walk(tree)
                    if isinstance(
                        node,
                        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                    )
                    and str(node.name).strip()
                }
                for name in names:
                    index.setdefault(name, []).append(rel_path)
        normalized_index = {name: list(dict.fromkeys(paths)) for name, paths in index.items()}
        self._python_definition_index_cache[cache_key] = normalized_index
        return normalized_index

    def _build_python_plausible_mutants(
        self,
        *,
        artifact_path: str,
        source_text: str,
        symbol_hints: list[str],
    ) -> list[dict[str, Any]]:
        try:
            parsed = ast.parse(source_text)
        except SyntaxError:
            return []

        symbol_hint_set = {
            str(hint).strip().lower() for hint in list(symbol_hints or []) if str(hint).strip()
        }
        compare_operator_rewrites = {
            "Eq": ast.NotEq,
            "NotEq": ast.Eq,
            "In": ast.NotIn,
            "NotIn": ast.In,
            "Is": ast.IsNot,
            "IsNot": ast.Is,
            "Lt": ast.GtE,
            "LtE": ast.Gt,
            "Gt": ast.LtE,
            "GtE": ast.Lt,
        }
        binary_operator_rewrites = {
            "Add": ast.Sub,
            "Sub": ast.Add,
        }
        kind_priority = {
            "compare_op": 5,
            "bool_constant": 4,
            "remove_not": 4,
            "binop_flip": 3,
            "numeric_constant": 2,
        }
        seen_candidates: set[tuple[Any, ...]] = set()
        candidates: list[dict[str, Any]] = []

        class Collector(ast.NodeVisitor):
            def __init__(self) -> None:
                self.stack: list[str] = []

            def _priority(self, kind: str) -> int:
                base = kind_priority.get(kind, 0)
                if not symbol_hint_set:
                    return base
                scoped = {name.lower() for name in self.stack if str(name).strip()}
                if scoped.intersection(symbol_hint_set):
                    return base + 10
                return base

            def _record(self, kind: str, node: ast.AST, **extra: Any) -> None:
                lineno = int(getattr(node, "lineno", 0) or 0)
                col_offset = int(getattr(node, "col_offset", 0) or 0)
                if lineno <= 0:
                    return
                candidate_key = (
                    kind,
                    lineno,
                    col_offset,
                    tuple(sorted(extra.items())),
                )
                if candidate_key in seen_candidates:
                    return
                seen_candidates.add(candidate_key)
                candidates.append(
                    {
                        "kind": kind,
                        "lineno": lineno,
                        "col_offset": col_offset,
                        "priority": self._priority(kind),
                        "context": tuple(self.stack[-2:]),
                        **extra,
                    }
                )

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.stack.append(str(node.name))
                self.generic_visit(node)
                self.stack.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self.stack.append(str(node.name))
                self.generic_visit(node)
                self.stack.pop()

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                self.stack.append(str(node.name))
                self.generic_visit(node)
                self.stack.pop()

            def visit_Compare(self, node: ast.Compare) -> None:
                if node.ops:
                    rewrite = type(node.ops[0]).__name__
                    if rewrite in compare_operator_rewrites:
                        self._record("compare_op", node, op_index=0, rewrite=rewrite)
                self.generic_visit(node)

            def visit_Constant(self, node: ast.Constant) -> None:
                if isinstance(node.value, bool):
                    self._record("bool_constant", node, new_value=not node.value)
                elif type(node.value) in {int, float}:
                    current_value = node.value
                    if abs(float(current_value)) <= 10:
                        if current_value == 0:
                            new_value = 1
                        elif current_value == 1:
                            new_value = 0
                        elif current_value == -1:
                            new_value = 0
                        else:
                            new_value = current_value + (1 if current_value > 0 else -1)
                        if new_value != current_value:
                            self._record("numeric_constant", node, new_value=new_value)
                self.generic_visit(node)

            def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
                if isinstance(node.op, ast.Not):
                    self._record("remove_not", node)
                self.generic_visit(node)

            def visit_BinOp(self, node: ast.BinOp) -> None:
                rewrite = type(node.op).__name__
                if rewrite in binary_operator_rewrites:
                    self._record("binop_flip", node, rewrite=rewrite)
                self.generic_visit(node)

        Collector().visit(parsed)
        candidates.sort(
            key=lambda item: (
                -int(item.get("priority") or 0),
                int(item.get("lineno") or 0),
                int(item.get("col_offset") or 0),
                str(item.get("kind") or ""),
            )
        )

        def mutate_candidate(candidate: dict[str, Any]) -> str:
            try:
                tree = ast.parse(source_text)
            except SyntaxError:
                return ""
            target_lineno = int(candidate.get("lineno") or 0)
            target_col = int(candidate.get("col_offset") or 0)
            target_kind = str(candidate.get("kind") or "").strip()

            class Mutator(ast.NodeTransformer):
                def __init__(self) -> None:
                    self.applied = False

                def _matches(self, node: ast.AST) -> bool:
                    return (
                        not self.applied
                        and int(getattr(node, "lineno", 0) or 0) == target_lineno
                        and int(getattr(node, "col_offset", 0) or 0) == target_col
                    )

                def visit_Compare(self, node: ast.Compare) -> ast.AST:
                    self.generic_visit(node)
                    if target_kind == "compare_op" and self._matches(node) and node.ops:
                        op_index = int(candidate.get("op_index") or 0)
                        rewrite_cls = compare_operator_rewrites.get(
                            str(candidate.get("rewrite") or "")
                        )
                        if rewrite_cls is not None and 0 <= op_index < len(node.ops):
                            node.ops[op_index] = rewrite_cls()
                            self.applied = True
                    return node

                def visit_Constant(self, node: ast.Constant) -> ast.AST:
                    if (
                        target_kind == "bool_constant"
                        and self._matches(node)
                        and isinstance(node.value, bool)
                    ):
                        self.applied = True
                        return ast.copy_location(
                            ast.Constant(value=bool(candidate.get("new_value"))),
                            node,
                        )
                    if (
                        target_kind == "numeric_constant"
                        and self._matches(node)
                        and type(node.value) in {int, float}
                    ):
                        self.applied = True
                        return ast.copy_location(
                            ast.Constant(value=candidate.get("new_value")),
                            node,
                        )
                    return node

                def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
                    self.generic_visit(node)
                    if (
                        target_kind == "remove_not"
                        and self._matches(node)
                        and isinstance(node.op, ast.Not)
                    ):
                        self.applied = True
                        return ast.copy_location(node.operand, node)
                    return node

                def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
                    self.generic_visit(node)
                    if target_kind == "binop_flip" and self._matches(node):
                        rewrite_cls = binary_operator_rewrites.get(
                            str(candidate.get("rewrite") or "")
                        )
                        if rewrite_cls is not None:
                            node.op = rewrite_cls()
                            self.applied = True
                    return node

            mutator = Mutator()
            mutated_tree = mutator.visit(tree)
            if not mutator.applied:
                return ""
            try:
                mutated_source = ast.unparse(ast.fix_missing_locations(mutated_tree))
            except Exception:
                return ""
            normalized_original = source_text.replace("\r\n", "\n").strip()
            normalized_mutated = mutated_source.replace("\r\n", "\n").strip()
            if not normalized_mutated or normalized_mutated == normalized_original:
                return ""
            return mutated_source if mutated_source.endswith("\n") else f"{mutated_source}\n"

        mutants: list[dict[str, Any]] = []
        for candidate in candidates:
            mutated_source = mutate_candidate(candidate)
            if not mutated_source:
                continue
            context = [
                str(value).strip()
                for value in list(candidate.get("context") or [])
                if str(value).strip()
            ]
            label_prefix = ".".join(context) if context else Path(artifact_path).stem
            mutant_id = (
                f"{artifact_path}:{int(candidate.get('lineno') or 0)}:"
                f"{str(candidate.get('kind') or '').strip()}"
            )
            mutants.append(
                {
                    "mutant_id": mutant_id,
                    "path": artifact_path,
                    "kind": str(candidate.get("kind") or "").strip(),
                    "line": int(candidate.get("lineno") or 0),
                    "label": f"{label_prefix}:{str(candidate.get('kind') or '').strip()}",
                    "content": mutated_source,
                }
            )
            if len(mutants) >= _SYNTHETIC_MUTATION_MAX_MUTANTS_PER_FILE:
                break
        return mutants

    def _materialize_synthetic_portfolio_artifact(
        self,
        *,
        sandbox_root: Path,
        artifact_path: str,
        content: str,
        requested_mode: str,
    ) -> str:
        materialized_path = sandbox_root / artifact_path
        materialized_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_mode = self._resolve_synthetic_portfolio_materialization_mode(
            materialized_path=materialized_path,
            content=content,
            requested_mode=requested_mode,
        )
        normalized_content = content if content.endswith("\n") else f"{content}\n"
        if resolved_mode == "append" and materialized_path.exists():
            existing_content = materialized_path.read_text()
            merged = existing_content.rstrip("\n")
            addition = normalized_content.strip("\n")
            if merged and addition:
                normalized_content = f"{merged}\n\n{addition}\n"
            elif merged:
                normalized_content = f"{merged}\n"
        materialized_path.write_text(normalized_content)
        return resolved_mode

    def _resolve_synthetic_portfolio_materialization_mode(
        self,
        *,
        materialized_path: Path,
        content: str,
        requested_mode: str,
    ) -> str:
        mode = str(requested_mode or "").strip().lower().replace("-", "_").replace(" ", "_")
        if mode in {"append", "replace"}:
            if mode == "append" and not materialized_path.exists():
                return "replace"
            return mode
        if not materialized_path.exists():
            return "replace"
        existing_content = materialized_path.read_text()
        if self._synthetic_artifact_looks_like_full_file(existing_content, content):
            return "replace"
        return "append"

    def _synthetic_artifact_looks_like_full_file(
        self,
        existing_content: str,
        generated_content: str,
    ) -> bool:
        existing = existing_content.replace("\r\n", "\n").strip()
        generated = generated_content.replace("\r\n", "\n").strip()
        if not existing or not generated:
            return False
        if existing == generated:
            return True
        existing_lines = [line.strip() for line in existing.splitlines() if line.strip()]
        if not existing_lines:
            return False
        shared_prefix_lines = sum(1 for line in existing_lines[:4] if line in generated)
        if shared_prefix_lines >= 2:
            return True
        return len(generated) >= max(400, int(len(existing) * 0.8)) and shared_prefix_lines >= 1

    def _build_synthetic_portfolio_collection_command(
        self,
        *,
        artifact_path: str,
        framework: str,
        test_command: str,
        worktree: Path,
    ) -> tuple[str, bool]:
        normalized_framework = infer_test_inventory_framework(
            explicit_framework=framework,
            test_command=test_command,
        )
        quoted_path = shlex.quote(artifact_path)
        if normalized_framework == "pytest":
            base_command = (
                derive_test_collection_command(
                    test_command or "python3 -m pytest -q",
                    framework="pytest",
                )
                or "python3 -m pytest --collect-only -q"
            )
            targeted = self._build_targeted_pytest_command(
                base_command,
                [artifact_path],
                worktree=worktree,
            )
            return targeted or base_command, True
        if normalized_framework == "jest":
            base_command = (
                derive_test_collection_command(
                    test_command or "npx jest",
                    framework="jest",
                )
                or "npx jest --listTests"
            )
            return f"{base_command} {quoted_path}".strip(), True
        if normalized_framework == "vitest":
            base_command = (
                derive_test_collection_command(
                    test_command or "npx vitest run",
                    framework="vitest",
                )
                or "npx vitest list"
            )
            return f"{base_command} {quoted_path}".strip(), True
        if normalized_framework == "rspec":
            base_command = (
                derive_test_collection_command(
                    test_command or "rspec",
                    framework="rspec",
                )
                or "rspec --dry-run"
            )
            return f"{base_command} {quoted_path}".strip(), True
        if normalized_framework == "phpunit":
            base_command = (
                derive_test_collection_command(
                    test_command or "phpunit",
                    framework="phpunit",
                )
                or "phpunit --list-tests"
            )
            return f"{base_command} {quoted_path}".strip(), True
        if normalized_framework == "cargo_test":
            selector = self._cargo_test_selector(artifact_path)
            if selector:
                return (f"{test_command or 'cargo test'} --test {selector} -- --list".strip(), True)
        if normalized_framework == "junit":
            selector = self._junit_test_selector(artifact_path)
            targeted = self._build_targeted_junit_command(
                test_command,
                selector,
                dry_run=True,
            )
            if targeted:
                return targeted, True
        if normalized_framework == "dotnet_test":
            selector = self._dotnet_test_selector(artifact_path)
            targeted = self._build_targeted_dotnet_test_command(
                test_command,
                selector,
                list_only=True,
            )
            if targeted:
                return targeted, True
        if normalized_framework == "unittest":
            return "", False
        if normalized_framework == "go_test":
            package_arg = self._go_test_package_arg(artifact_path)
            if package_arg:
                return f"go test {package_arg} -list .", True
        return "", False

    def _build_synthetic_portfolio_execution_command(
        self,
        *,
        artifact_path: str,
        framework: str,
        test_command: str,
        worktree: Path,
    ) -> tuple[str, bool]:
        normalized_framework = infer_test_inventory_framework(
            explicit_framework=framework,
            test_command=test_command,
        )
        quoted_path = shlex.quote(artifact_path)
        if normalized_framework == "pytest":
            if test_command:
                targeted = self._build_targeted_pytest_command(
                    test_command,
                    [artifact_path],
                    worktree=worktree,
                )
                if targeted:
                    return targeted, True
            disable_plugin_autoload = self._should_disable_pytest_plugin_autoload(
                test_command or "python3 -m pytest -q",
                worktree=worktree,
            )
            prefix = "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 " if disable_plugin_autoload else ""
            return f"{prefix}python3 -m pytest {quoted_path} -x --tb=no -q", True
        if normalized_framework == "jest":
            return (f"{test_command or 'npx jest'} {quoted_path}".strip(), True)
        if normalized_framework == "vitest":
            return (f"{test_command or 'npx vitest run'} {quoted_path}".strip(), True)
        if normalized_framework == "unittest":
            return (f"python -m unittest {quoted_path}".strip(), True)
        if normalized_framework == "rspec":
            return (f"{test_command or 'rspec'} {quoted_path}".strip(), True)
        if normalized_framework == "phpunit":
            return (f"{test_command or 'phpunit'} {quoted_path}".strip(), True)
        if normalized_framework == "cargo_test":
            selector = self._cargo_test_selector(artifact_path)
            if selector:
                return (f"{test_command or 'cargo test'} --test {selector}".strip(), True)
        if normalized_framework == "junit":
            selector = self._junit_test_selector(artifact_path)
            targeted = self._build_targeted_junit_command(
                test_command,
                selector,
                dry_run=False,
            )
            if targeted:
                return targeted, True
        if normalized_framework == "dotnet_test":
            selector = self._dotnet_test_selector(artifact_path)
            targeted = self._build_targeted_dotnet_test_command(
                test_command,
                selector,
                list_only=False,
            )
            if targeted:
                return targeted, True
        if normalized_framework == "go_test":
            package_arg = self._go_test_package_arg(artifact_path)
            if package_arg:
                return (f"go test {package_arg}", True)
        return str(test_command or "").strip(), False

    def _run_synthetic_portfolio_command(
        self,
        worktree: Path,
        command: str,
        *,
        env: Optional[dict[str, str]] = None,
    ) -> _CommandResult:
        result = self._invoke_run_command(
            worktree,
            command,
            min(self.custom_test_timeout, self.full_test_timeout),
            env=env,
        )
        if result.returncode != 0 and output_indicates_missing_pytest(result.output):
            for recovery_command in build_pytest_recovery_commands(
                command,
                repo_root=worktree,
            ):
                if recovery_command.strip() == command.strip():
                    continue
                result = self._invoke_run_command(
                    worktree,
                    recovery_command,
                    min(self.custom_test_timeout, self.full_test_timeout),
                    env=env,
                )
                if result.returncode == 0 or not output_indicates_missing_pytest(result.output):
                    break
        return result

    def _portfolio_artifact_discovered(
        self,
        *,
        artifact_path: str,
        output: str,
        collection_succeeded: bool,
        targeted_collection: bool,
    ) -> bool:
        if not collection_succeeded:
            return False
        if targeted_collection:
            return True
        text = normalize_terminal_output(output)
        name = Path(artifact_path).name
        return artifact_path in text or name in text or not self._output_indicates_no_tests(text)

    def _output_indicates_no_tests(self, output: str) -> bool:
        lowered = normalize_terminal_output(output).strip().lower()
        if not lowered:
            return False
        return any(
            token in lowered
            for token in (
                "collected 0 items",
                "collected 0 item",
                "no tests collected",
                "no tests found",
                "no tests to run",
                "no tests were found",
                "no matching tests found",
                "0 tests collected",
                "list of tests matched no",
                "did not match any test files",
            )
        )

    def _cargo_test_selector(self, artifact_path: str) -> str:
        path = Path(artifact_path)
        if path.suffix != ".rs":
            return ""
        return path.stem

    def _junit_test_selector(self, artifact_path: str) -> str:
        path = Path(artifact_path)
        if path.suffix not in {".java", ".kt", ".groovy"}:
            return path.stem
        if not path.name:
            return ""
        stem_parts = list(path.with_suffix("").parts)
        for index, part in enumerate(stem_parts):
            lowered = part.lower()
            if (
                lowered in {"java", "kotlin", "groovy"}
                and index > 0
                and stem_parts[index - 1].lower() == "test"
            ):
                package_parts = stem_parts[index + 1 :]
                if package_parts:
                    return ".".join(package_parts)
        return path.stem

    def _build_targeted_junit_command(
        self,
        test_command: str,
        selector: str,
        *,
        dry_run: bool,
    ) -> str:
        if not selector:
            return ""
        command = str(test_command or "mvn test").strip()
        lowered = command.lower()
        if "gradle" in lowered:
            targeted = command
            if "--tests" not in lowered:
                targeted = f"{targeted} --tests {shlex.quote(selector)}"
            if dry_run and "--test-dry-run" not in lowered:
                targeted = f"{targeted} --test-dry-run"
            return targeted.strip()
        if "mvn" in lowered:
            if dry_run:
                return ""
            sanitized = re.sub(r"\s+-Dtest=\S+", "", command, flags=re.IGNORECASE)
            sanitized = re.sub(r"\s+-DfailIfNoTests=\S+", "", sanitized, flags=re.IGNORECASE)
            insertion = f"-Dtest={selector} -DfailIfNoTests=false test"
            if re.search(r"\btest\b", sanitized):
                return re.sub(r"\btest\b", insertion, sanitized, count=1).strip()
            return f"{sanitized} {insertion}".strip()
        return ""

    def _dotnet_test_selector(self, artifact_path: str) -> str:
        path = Path(artifact_path)
        return path.stem

    def _build_targeted_dotnet_test_command(
        self,
        test_command: str,
        selector: str,
        *,
        list_only: bool,
    ) -> str:
        if not selector:
            return ""
        command = str(test_command or "dotnet test").strip()
        sanitized = re.sub(r"\s+--list-tests\b", "", command, flags=re.IGNORECASE)
        sanitized = re.sub(r"\s+--filter\s+\S+", "", sanitized, flags=re.IGNORECASE)
        filter_expr = shlex.quote(f"FullyQualifiedName~{selector}")
        if list_only:
            return f"{sanitized} --list-tests --filter {filter_expr}".strip()
        return f"{sanitized} --filter {filter_expr}".strip()

    def _go_test_package_arg(self, artifact_path: str) -> str:
        directory = Path(artifact_path).parent.as_posix()
        if not directory or directory == ".":
            return "./..."
        return f"./{directory}"

    def build_cross_validation_matrix(
        self,
        results: list[Any],
        max_workers: Optional[int] = None,
        test_command: Optional[str] = None,
    ) -> Any:
        """Build an N x M matrix of patch-vs-test-suite pass indicators."""
        size = len(results)
        if np is not None:
            matrix = np.zeros((size, size))
        else:
            matrix = [[0.0 for _ in range(size)] for _ in range(size)]

        effective_workers = max(1, min(max_workers or 1, size))
        if effective_workers == 1:
            for patch_index, patch_result in enumerate(results):
                row = self._build_cross_validation_row(
                    patch_result,
                    results,
                    test_command=test_command,
                )
                for test_index, value in enumerate(row):
                    matrix[patch_index][test_index] = value
            return matrix

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(
                    self._build_cross_validation_row,
                    patch_result,
                    results,
                    test_command=test_command,
                ): patch_index
                for patch_index, patch_result in enumerate(results)
            }
            for future in as_completed(futures):
                patch_index = futures[future]
                row = future.result()
                for test_index, value in enumerate(row):
                    matrix[patch_index][test_index] = value
        return matrix

    def _build_cross_validation_row(
        self,
        patch_result: Any,
        results: list[Any],
        test_command: Optional[str] = None,
    ) -> list[float]:
        row = [0.0 for _ in range(len(results))]
        for test_index, test_result in enumerate(results):
            portfolio_payload = normalize_test_suite_artifact_payload(
                getattr(test_result, "test_suite_artifact", None)
            )
            portfolio_artifacts = select_cross_validation_test_artifacts(portfolio_payload)
            if portfolio_artifacts:
                artifact_scores: list[float] = []
                portfolio_inventory = self._resolve_test_inventory(
                    test_inventory=portfolio_payload,
                    expected_test_count=None,
                    expected_test_ids=None,
                    test_command=test_command,
                    baseline_result=None,
                )
                for artifact in portfolio_artifacts:
                    validation = self._validate_synthetic_test_artifact(
                        Path(str(patch_result.worktree_path)).resolve(),
                        dict(artifact),
                        test_command=test_command,
                        test_inventory=portfolio_inventory,
                        baseline_result=None,
                        check_baseline_preservation=False,
                        rerun=False,
                        measure_mutation=False,
                    )
                    artifact_scores.append(1.0 if validation.get("execution_succeeded") else 0.0)
                if artifact_scores:
                    row[test_index] = sum(artifact_scores) / len(artifact_scores)
                continue
            if list(portfolio_payload.get("test_artifacts") or []):
                continue
            if not getattr(test_result, "test_suite", None):
                continue
            try:
                passed = self._run_test_suite(
                    patch_result.worktree_path,
                    test_result.test_suite,
                    test_command=test_command,
                )
            except TypeError as exc:
                if "test_command" not in str(exc):
                    raise
                passed = self._run_test_suite(
                    patch_result.worktree_path,
                    test_result.test_suite,
                )
            row[test_index] = 1.0 if passed else 0.0
        return row

    def cross_validate(
        self,
        worktree_path: str,
        reproduction_artifacts: list[dict[str, Any]],
        test_command: Optional[str] = None,
    ) -> list[float]:
        scores = []
        worktree = Path(worktree_path)
        for artifact in reproduction_artifacts:
            command = artifact.get("command")
            script = artifact.get("script_content")
            if not command and not script:
                continue
            result = self._run_reproduction(
                worktree,
                command,
                script,
                script_path=artifact.get("script_path"),
                test_command=test_command,
            )
            scores.append(1.0 if result.returncode == 0 else 0.0)
        return scores

    def _list_changed_files(
        self,
        worktree: Path,
        baseline_ref: Optional[str] = None,
    ) -> list[str]:
        return list_git_changed_files(worktree, baseline_ref=baseline_ref)

    def _resolve_changed_files(
        self,
        worktree: Path,
        baseline_ref: Optional[str] = None,
    ) -> list[str]:
        if baseline_ref is None:
            return self._list_changed_files(worktree)
        try:
            return self._list_changed_files(worktree, baseline_ref=baseline_ref)
        except TypeError as exc:
            if "baseline_ref" not in str(exc):
                raise
            return self._list_changed_files(worktree)

    def _check_syntax(self, worktree: Path, changed_files: list[str]) -> bool:
        for rel_path in changed_files:
            if not rel_path.endswith(".py"):
                continue
            if rel_path in self._quarantined_paths:
                logger.warning(
                    "Syntax check skipped for quarantined %s: %s",
                    rel_path,
                    self._quarantined_paths[rel_path],
                )
                return False
            file_path = worktree / rel_path
            if not file_path.exists() or not file_path.is_file():
                continue
            try:
                ast.parse(file_path.read_text(errors="replace"))
            except SyntaxError as exc:
                self._quarantine_path(
                    rel_path,
                    f"SyntaxError@line{getattr(exc, 'lineno', 0) or 0}: {exc.msg}",
                )
                logger.warning("Syntax error in %s: %s", rel_path, exc)
                return False
        return True

    def _check_lint(
        self,
        worktree: Path,
        changed_files: list[str],
    ) -> tuple[bool, str, bool]:
        """Run flake8 against changed Python files.

        Returns ``(lint_clean, lint_output, lint_applied)``. ``lint_applied``
        is False when flake8 could not be executed (missing binary, timeout,
        FileNotFoundError on the python3 binary). Callers must avoid
        awarding the lint bonus when ``lint_applied`` is False — the
        previous behaviour silently returned ``True`` and inflated the
        overall score in environments without flake8.
        """

        python_files = [
            path for path in changed_files if path.endswith(".py") and (worktree / path).is_file()
        ]
        if not python_files:
            # No python changes to lint — vacuously clean and applied.
            return True, "", True
        try:
            result = subprocess.run(
                ["python3", "-m", "flake8", "--select=E9,F63,F7,F82", "--", *python_files],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(worktree),
            )
            output = (result.stdout + result.stderr).strip()
            if "No module named flake8" in output:
                return True, "", False  # not applied — abstain
            return result.returncode == 0, output, True
        except FileNotFoundError:
            return True, "", False
        except subprocess.TimeoutExpired:
            return True, "lint timed out", False

    def _run_reproduction(
        self,
        worktree: Path,
        command: Optional[str],
        script_content: Optional[str],
        *,
        script_path: Optional[str] = None,
        test_command: Optional[str] = None,
    ) -> _CommandResult:
        materialized_path: Optional[Path] = None
        cleanup_path: Optional[Path] = None
        try:
            if script_content:
                try:
                    materialized_path, cleanup_path = self._materialize_reproduction_script(
                        worktree,
                        script_content,
                        script_path=script_path,
                    )
                except OSError as exc:
                    return _CommandResult(
                        returncode=1,
                        output=f"Failed to materialize reproduction script: {exc}",
                    )
            if command:
                command = self._rewrite_reproduction_command_script_path(
                    command,
                    worktree,
                    materialized_path,
                    script_path=script_path,
                )
                return self._run_command(worktree, command, timeout=self.timeout)
            if materialized_path is not None:
                runtime_command = self._build_runtime_python_command(
                    test_command,
                    self._command_path(worktree, materialized_path),
                )
                if runtime_command is None:
                    runtime_command = (
                        f"python3 {shlex.quote(self._command_path(worktree, materialized_path))}"
                    )
                return self._run_command(worktree, runtime_command, timeout=self.timeout)
        finally:
            if cleanup_path is not None and cleanup_path.exists():
                cleanup_path.unlink()
        return _CommandResult(returncode=1, output="No reproduction command or script supplied.")

    def _run_command(
        self,
        worktree: Path,
        command: str,
        timeout: Optional[int],
        *,
        env: Optional[dict[str, str]] = None,
        sanitized: bool = False,
        completion_report_path: Optional[Path] = None,
        completion_report_started_at: Optional[float] = None,
    ) -> _CommandResult:
        task_id: str | None = None
        active_rollout_context = _resolve_active_rollout_cli_context()
        if active_rollout_context is not None:
            registry, rollout_id = active_rollout_context
            task_id = registry.process_task_id(rollout_id)
        try:
            if sanitized:
                # Bypass the run_shell_command env merger so secrets in
                # ``os.environ`` aren't reintroduced. Run directly via
                # bash with the explicit sanitized env. ``--noprofile
                # --norc`` is critical: ``bash -lc`` would re-source
                # ``~/.bash_profile`` / ``~/.bashrc`` and re-export
                # whatever the user keeps there (including
                # ``OPENAI_API_KEY`` and similar host secrets), defeating
                # the sanitization.
                completed = subprocess.run(
                    ["bash", "--noprofile", "--norc", "-c", command],
                    cwd=str(worktree),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env or {},
                )
                output = normalize_terminal_output(
                    (completed.stdout or "") + (completed.stderr or "")
                ).strip()
                return _CommandResult(returncode=completed.returncode, output=output)
            merged_runtime_env = self._merge_runtime_env(
                {
                    "APEX_TARGET_TOOL_WORKDIR": str(worktree),
                    **dict(env or {}),
                }
            )
            completion_probe = self._structured_report_completion_probe(
                completion_report_path,
                minimum_mtime=completion_report_started_at,
                runtime_env=merged_runtime_env,
            )
            shell_kwargs: dict[str, Any] = {
                "timeout": timeout,
                "task_id": task_id,
                "env": merged_runtime_env,
            }
            if completion_probe is not None:
                shell_kwargs["completion_probe"] = completion_probe
            result = run_shell_command(command, worktree, **shell_kwargs)
            return _CommandResult(
                returncode=result.returncode,
                output=normalize_terminal_output(result.stdout + result.stderr).strip(),
            )
        except subprocess.TimeoutExpired:
            timeout_label = timeout if timeout is not None else "unbounded"
            return _CommandResult(
                returncode=124, output=f"Command timed out after {timeout_label} seconds."
            )

    def _run_command_with_completion_report(
        self,
        worktree: Path,
        command: str,
        timeout: Optional[int],
        *,
        completion_report_path: Optional[Path],
        completion_report_started_at: Optional[float],
    ) -> _CommandResult:
        try:
            return self._run_command(
                worktree,
                command,
                timeout=timeout,
                completion_report_path=completion_report_path,
                completion_report_started_at=completion_report_started_at,
            )
        except TypeError as exc:
            message = str(exc)
            new_kwarg_signature = (
                "got an unexpected keyword argument 'completion_report_path'" in message
                or "got an unexpected keyword argument 'completion_report_started_at'" in message
            )
            if not new_kwarg_signature:
                raise
            legacy_command = _command_with_absolute_json_report_path(
                command,
                completion_report_path,
            )
            return self._run_command(worktree, legacy_command, timeout=timeout)

    def _structured_report_completion_probe(
        self,
        report_path: Optional[Path],
        *,
        minimum_mtime: Optional[float],
        runtime_env: Optional[dict[str, str]] = None,
    ) -> Optional[Callable[[], Optional[int]]]:
        if report_path is None:
            return None

        def _load_payload() -> Optional[dict[str, Any]]:
            payload = self._load_pytest_json_report_payload(
                report_path,
                minimum_mtime=minimum_mtime,
            )
            if isinstance(payload, dict):
                return payload
            if not runtime_env:
                return None
            try:
                from ..evaluation.target_runtime import read_target_runtime_file_text

                text = read_target_runtime_file_text(runtime_env, report_path)
            except Exception:
                return None
            if not text:
                return None
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError, json.JSONDecodeError):
                return None
            return parsed if isinstance(parsed, dict) else None

        def _probe() -> Optional[int]:
            payload = _load_payload()
            if not isinstance(payload, dict):
                return None
            summary = payload.get("summary")
            exitcode = payload.get("exitcode")
            if not isinstance(summary, dict) or isinstance(exitcode, bool):
                return None
            try:
                return int(exitcode)
            except (TypeError, ValueError):
                return None

        return _probe

    def _prepare_pytest_json_report_command_for_worktree(
        self,
        command: str,
        *,
        label: str,
        worktree: Path,
    ) -> tuple[str, Optional[Path]]:
        try:
            return self._prepare_pytest_json_report_command(
                command,
                label=label,
                worktree=worktree,
            )
        except TypeError as exc:
            if "got an unexpected keyword argument 'worktree'" not in str(exc):
                raise
            return self._prepare_pytest_json_report_command(command, label=label)

    def _parse_test_output(self, output: str, returncode: int) -> dict[str, int]:
        output = normalize_terminal_output(output)
        summary_counts = parse_pytest_terminal_summary_counts(output)
        if any(summary_counts.values()):
            return {
                "passed": (
                    summary_counts["passed"] + summary_counts["xfailed"] + summary_counts["xpassed"]
                ),
                "failed": summary_counts["failed"],
                "errors": summary_counts["errors"],
            }
        result = {"passed": 0, "failed": 0, "errors": 0}
        for key, aliases in {
            "passed": ["passed", "pass"],
            "failed": ["failed", "failures", "failure"],
            "errors": ["errors", "error"],
        }.items():
            for alias in aliases:
                match = re.search(rf"(\d+)\s+{alias}\b", output, flags=re.IGNORECASE)
                if match:
                    result[key] = max(result[key], int(match.group(1)))
        if result == {"passed": 0, "failed": 0, "errors": 0}:
            outcomes = self._parse_case_outcomes(output)
            if outcomes:
                result["passed"] = sum(1 for status in outcomes.values() if status == "PASSED")
                result["failed"] = sum(1 for status in outcomes.values() if status == "FAILED")
                result["errors"] = sum(1 for status in outcomes.values() if status == "ERROR")
        if result == {"passed": 0, "failed": 0, "errors": 0}:
            # No parsable pytest summary AND no per-case outcomes were
            # found. Synthesizing ``passed=1`` on returncode==0 used to
            # mark "tests passed" for any command that exited 0 with
            # silent output (e.g., a no-op shell command). That path
            # propagated into pass_rate-based acceptance gates and was
            # a primary source of false-positive accepts. Treat unknown
            # output as a single error signal so downstream gates stay
            # conservative; ``regression_passes`` (which checks the
            # raw exit code) is still surfaced separately for callers
            # that want that signal explicitly.
            if returncode == 0:
                result["errors"] = 1
            else:
                result["failed"] = 1
        return result

    def _parse_observed_test_count(self, output: str, parsed: dict[str, int]) -> int:
        summary_counts = parse_pytest_terminal_summary_counts(normalize_terminal_output(output))
        summary_total = sum(int(value or 0) for value in summary_counts.values())
        if summary_total > 0:
            return summary_total
        return (
            int(parsed.get("passed") or 0)
            + int(parsed.get("failed") or 0)
            + int(parsed.get("errors") or 0)
        )

    def _parse_case_outcomes(self, output: str) -> dict[str, str]:
        output = normalize_terminal_output(output)
        outcomes: dict[str, str] = {}
        patterns = [
            re.compile(
                r"^_+\s+(?P<status>ERROR|FAILED)\s+collecting\s+(?P<test>\S+\.py)\s+_+$",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"^(?P<status>ERROR|FAILED)\s+collecting\s+(?P<test>\S+\.py)\b",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"^(?P<status>ERROR|FAILED)\s+(?P<test>\S+\.py(?:::\S+)*)\s+-\s+",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"^\[TEST\]\s+(?P<status>PASSED|FAILED|ERROR|SKIPPED|XPASS|XFAIL)\s+(?P<test>.+?)\s*$",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"^\[TEST\]\s+(?P<test>.+?)\s+(?P<status>PASSED|FAILED|ERROR|SKIPPED|XPASS|XFAIL)\s*$",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"^(?P<test>\S+::\S+(?:::\S+)*)\s+(?P<status>PASSED|FAILED|ERROR|SKIPPED|XPASS|XFAIL)\b",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"^(?P<status>PASSED|FAILED|ERROR|SKIPPED|XPASS|XFAIL)\s+(?P<test>\S+::\S+(?:::\S+)*)\b",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"^(?P<test>\S+\.py::\S+(?:::\S+)*)\s+(?P<status>PASSED|FAILED|ERROR|SKIPPED|XPASS|XFAIL)\b",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"^(?P<status>PASSED|FAILED|ERROR|SKIPPED|XPASS|XFAIL)\s+(?P<test>\S+\.py::\S+(?:::\S+)*)\b",
                flags=re.IGNORECASE,
            ),
        ]
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            for pattern in patterns:
                match = pattern.search(line)
                if not match:
                    continue
                test_id = match.group("test")
                status = match.group("status").upper()
                if status in {"SKIPPED", "XPASS", "XFAIL"}:
                    break
                outcomes[test_id] = (
                    "ERROR" if status == "ERROR" else ("FAILED" if status == "FAILED" else "PASSED")
                )
                break
        return outcomes

    def _run_test_suite(
        self,
        worktree_path: str,
        test_code: str,
        test_command: Optional[str] = None,
    ) -> bool:
        worktree = Path(worktree_path)
        if not worktree.exists():
            return False

        # Legacy ablation path: behavior used to write the sibling rollout's
        # raw ``test_code`` directly into the candidate's worktree and run
        # it in-place. That path is preserved only for ablations and the
        # sandbox knob defaults to OFF.
        if self.cross_validation_sandbox_disabled:
            return self._run_test_suite_in_worktree(worktree, test_code, test_command)

        return self._run_test_suite_sandboxed(worktree, test_code, test_command)

    def _run_test_suite_sandboxed(
        self,
        worktree: Path,
        test_code: str,
        test_command: Optional[str],
    ) -> bool:
        try:
            sandbox_root, sanitized_env = self._prepare_sandboxed_test_environment(
                worktree,
                test_code,
                allowed_paths=set(),
            )
        except _SandboxValidationError as exc:
            logger.warning("sandboxed cross-validation: rejecting test_code (%s)", exc)
            return False
        if sandbox_root is None:
            return False

        try:
            test_path = sandbox_root / "_apex_cross_test.py"
            test_path.write_text(test_code)
            if "def test_" in test_code or "import pytest" in test_code:
                disable_plugin_autoload = self._should_disable_pytest_plugin_autoload(
                    test_command or "python3 -m pytest -q",
                    worktree=sandbox_root,
                )
                command = self._build_ephemeral_pytest_command(
                    test_command,
                    test_path.name,
                    worktree=sandbox_root,
                ) or (
                    f"{'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ' if disable_plugin_autoload else ''}"
                    f"python3 -m pytest {test_path.name} -x --tb=no -q"
                )
            else:
                command = (
                    self._build_runtime_python_command(test_command, test_path.name)
                    or f"python3 {shlex.quote(test_path.name)}"
                )
            # Run through the standard ``_run_command`` plumbing so callers
            # that monkeypatch ``_run_command`` keep their hook. The
            # ``sanitized=True`` flag bypasses ``run_shell_command``'s
            # ``os.environ`` merger so secrets cannot be reintroduced.
            result = self._invoke_run_command(
                sandbox_root,
                command,
                self.custom_test_timeout,
                env=sanitized_env,
            )
            if result.returncode != 0 and output_indicates_missing_pytest(result.output):
                for recovery_command in build_pytest_recovery_commands(
                    command,
                    repo_root=sandbox_root,
                ):
                    if recovery_command.strip() == command.strip():
                        continue
                    result = self._invoke_run_command(
                        sandbox_root,
                        recovery_command,
                        self.custom_test_timeout,
                        env=sanitized_env,
                    )
                    if result.returncode == 0 or not output_indicates_missing_pytest(result.output):
                        break
            return result.returncode == 0
        finally:
            shutil.rmtree(sandbox_root.parent, ignore_errors=True)

    def _run_test_suite_in_worktree(
        self,
        worktree: Path,
        test_code: str,
        test_command: Optional[str],
    ) -> bool:
        # Legacy unsafe path: kept ONLY behind ``cross_validation_sandbox_disabled``
        # for ablation runs that need to reproduce historical numbers.
        temporary = tempfile.NamedTemporaryFile(
            mode="w",
            suffix="_apex_cross_test.py",
            dir=str(worktree),
            delete=False,
        )
        try:
            temporary.write(test_code)
            temporary.flush()
            temporary.close()
            test_path = Path(temporary.name)
            if "def test_" in test_code or "import pytest" in test_code:
                disable_plugin_autoload = self._should_disable_pytest_plugin_autoload(
                    test_command or "python3 -m pytest -q",
                    worktree=worktree,
                )
                command = self._build_ephemeral_pytest_command(
                    test_command,
                    test_path.name,
                    worktree=worktree,
                ) or (
                    f"{'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ' if disable_plugin_autoload else ''}"
                    f"python3 -m pytest {test_path.name} -x --tb=no -q"
                )
            else:
                command = (
                    self._build_runtime_python_command(test_command, test_path.name)
                    or f"python3 {shlex.quote(test_path.name)}"
                )
            result = self._run_command(worktree, command, timeout=self.custom_test_timeout)
            if result.returncode != 0 and output_indicates_missing_pytest(result.output):
                for recovery_command in build_pytest_recovery_commands(
                    command,
                    repo_root=worktree,
                ):
                    if recovery_command.strip() == command.strip():
                        continue
                    result = self._run_command(
                        worktree,
                        recovery_command,
                        timeout=self.custom_test_timeout,
                    )
                    if result.returncode == 0 or not output_indicates_missing_pytest(result.output):
                        break
            return result.returncode == 0
        finally:
            try:
                Path(temporary.name).unlink()
            except FileNotFoundError:
                pass

    def _invoke_run_command(
        self,
        worktree: Path,
        command: str,
        timeout: Optional[int],
        *,
        env: Optional[dict[str, str]] = None,
    ) -> _CommandResult:
        """Call ``_run_command`` while tolerating thin test stand-ins.

        Tests in this codebase frequently monkeypatch ``_run_command``
        with a function that only accepts ``(worktree, command, timeout)``
        positionally. When that happens we fall back to a minimal call
        signature and skip the env injection — the test owns the
        environment in those cases anyway.
        """
        try:
            return self._run_command(worktree, command, timeout=timeout, env=env, sanitized=True)
        except TypeError as exc:
            # Match strictly against the new-kwarg unexpected-keyword
            # error so a real TypeError elsewhere in ``_run_command``
            # doesn't silently fall back to the legacy unsandboxed path.
            message = str(exc)
            new_kwarg_signature = (
                "got an unexpected keyword argument 'sanitized'" in message
                or "got an unexpected keyword argument 'env'" in message
            )
            if not new_kwarg_signature:
                raise
            return self._run_command(worktree, command, timeout=timeout)

    def _prepare_sandboxed_workspace_copy(
        self,
        worktree: Path,
    ) -> tuple[Optional[Path], dict[str, str]]:
        if not worktree.exists():
            return None, {}
        ephemeral_root = Path(tempfile.gettempdir()) / f"apex-xval-{uuid.uuid4().hex}"
        sandbox_root = ephemeral_root / "workspace"
        sandbox_home = ephemeral_root / "home"
        snapshot_ignore = shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            ".mypy_cache",
            ".git",
            ".hg",
            ".jj",
            ".sl",
            ".svn",
        )
        try:
            cloned = False
            if is_git_repo(worktree):
                cloned = clone_git_repo_with_overlay(
                    worktree,
                    sandbox_root,
                    ignore=snapshot_ignore,
                    restrict_symlinks_to_root=True,
                )
            if not cloned:
                copy_tree(
                    worktree,
                    sandbox_root,
                    ignore=snapshot_ignore,
                    restrict_symlinks_to_root=True,
                )
        except (OSError, shutil.Error) as exc:
            shutil.rmtree(ephemeral_root, ignore_errors=True)
            logger.warning("sandboxed cross-validation: copy failed: %s", exc)
            return None, {}
        sandbox_home.mkdir(parents=True, exist_ok=True)
        return sandbox_root, self._build_sanitized_env(sandbox_home)

    def _prepare_sandboxed_test_environment(
        self,
        worktree: Path,
        test_code: str,
        *,
        allowed_paths: set[Path],
    ) -> tuple[Optional[Path], dict[str, str]]:
        """Create an ephemeral copy of ``worktree`` and a sanitized env.

        Returns ``(sandbox_root, env)``. Caller must rm-rf
        ``sandbox_root.parent`` in a ``finally`` block.

        Raises ``_SandboxValidationError`` if ``test_code`` references
        absolute paths outside the ephemeral root.
        """
        sandbox_root, sanitized_env = self._prepare_sandboxed_workspace_copy(worktree)
        if sandbox_root is None:
            return None, {}
        ephemeral_root = sandbox_root.parent

        validate_paths = {sandbox_root.resolve(), ephemeral_root.resolve()} | {
            allowed.resolve() for allowed in allowed_paths
        }
        try:
            self._validate_test_code_paths(test_code, allowed_roots=validate_paths)
        except _SandboxValidationError:
            shutil.rmtree(ephemeral_root, ignore_errors=True)
            raise
        return sandbox_root, sanitized_env

    def _build_sanitized_env(self, sandbox_home: Path) -> dict[str, str]:
        # Allowlist of variable *names* and prefixes that are safe to
        # propagate to the sandbox. Everything else (and especially
        # *_API_KEY/*_TOKEN/AWS_*/GCP_*/SECRET_* etc.) is dropped.
        allowed_exact = {
            "PATH",
            "LANG",
            "TMPDIR",
            "PYTHONPATH",
            "PYTHONDONTWRITEBYTECODE",
            "SHELL",
            "USER",
            "LOGNAME",
            "VIRTUAL_ENV",
        }
        allowed_prefixes = ("LC_", "PYTEST_")
        sanitized: dict[str, str] = {}
        for key, value in os.environ.items():
            if key in allowed_exact or key.startswith(allowed_prefixes):
                sanitized[key] = value
        for key, value in self.runtime_env_overrides.items():
            if key in allowed_exact or key.startswith(allowed_prefixes):
                sanitized[key] = value
        current_python_dir = str(Path(sys.executable).resolve().parent)
        existing_path = str(sanitized.get("PATH") or "")
        path_entries = [entry for entry in existing_path.split(os.pathsep) if entry]
        if current_python_dir not in path_entries:
            path_entries.insert(0, current_python_dir)
        sanitized["PATH"] = os.pathsep.join(path_entries)
        sanitized["HOME"] = str(sandbox_home)
        sanitized["TMPDIR"] = sanitized.get("TMPDIR") or str(sandbox_home)
        sanitized["PYTHONDONTWRITEBYTECODE"] = "1"
        return sanitized

    def _merge_runtime_env(
        self,
        env: Optional[dict[str, str]] = None,
    ) -> dict[str, str]:
        merged = dict(self.runtime_env_overrides)
        if env:
            merged.update(env)
        return merged

    def _validate_test_code_paths(
        self,
        test_code: str,
        *,
        allowed_roots: set[Path],
    ) -> None:
        """Best-effort lint for *obvious* path-escape literals in test code.

        IMPORTANT: this is a HEURISTIC LINT, not a containment boundary.
        It catches:
          - regex-visible string literals beginning with ``/`` (e.g.
            ``"/etc/passwd"``);
          - ``ast.Constant`` string nodes beginning with ``/``;
          - parse failures (rejected to keep opaque inputs out).
        It does NOT defend against:
          - ``../../etc/passwd`` (relative escape; may be path-walked
            from the sandbox root which IS the actual containment);
          - dynamically constructed paths (``open("/" + suffix)``);
          - ``eval`` / ``exec`` / dynamic ``__import__``;
          - bytes literals (``b"/etc/passwd"``);
          - shell calls that read sensitive paths via ``os.system`` etc.
        The real containment boundary is the copytree-into-ephemeral-root
        + sanitized-env (with secrets stripped + ``HOME`` redirected) +
        ``rm -rf`` of the ephemeral tree in the caller's ``finally``.
        """
        for match in re.finditer(r"['\"](/[A-Za-z0-9_./\\-]+)['\"]", test_code):
            literal = match.group(1)
            if not self._path_within_allowed_roots(Path(literal), allowed_roots):
                raise _SandboxValidationError(
                    f"absolute path literal {literal!r} not under sandbox root"
                )
        try:
            tree = ast.parse(test_code)
        except SyntaxError as exc:
            raise _SandboxValidationError(f"test_code is not parseable Python: {exc}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if value.startswith("/") and not self._path_within_allowed_roots(
                    Path(value), allowed_roots
                ):
                    raise _SandboxValidationError(
                        f"AST string {value!r} resolves outside sandbox root"
                    )

    def _path_within_allowed_roots(
        self,
        path: Path,
        allowed_roots: set[Path],
    ) -> bool:
        try:
            resolved = path.resolve()
        except OSError:
            return False
        for root in allowed_roots:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def _run_baseline_passing_tests(
        self,
        worktree: Path,
        test_command: str,
        passing_tests: list[str],
    ) -> Optional[_CommandResult]:
        if not passing_tests:
            return None

        normalized = self._normalize_pytest_command(
            test_command,
            force_verbose=True,
            worktree=worktree,
        )
        if normalized == test_command and not self._is_pytest_command(test_command):
            return None

        outputs: list[str] = []
        overall_returncode = 0
        for chunk_start in range(0, len(passing_tests), 50):
            chunk = passing_tests[chunk_start : chunk_start + 50]
            command = self._build_targeted_pytest_command(
                test_command,
                chunk,
                worktree=worktree,
            )
            if command is None:
                return None
            result = self._run_command(worktree, command, timeout=self.full_test_timeout)
            outputs.append(result.output)
            if result.returncode != 0:
                overall_returncode = result.returncode
        return _CommandResult(
            returncode=overall_returncode,
            output="\n".join(part for part in outputs if part).strip(),
        )

    def _capture_collection_trace_probe(
        self,
        worktree: Path,
        test_command: str,
        baseline: BaselineResult,
    ) -> str:
        failing_files = [
            test_id
            for test_id in sorted(baseline.failing_tests)
            if test_id.endswith(".py") and "::" not in test_id
        ]
        if not failing_files and not baseline.passing_tests:
            failing_files = self._extract_collection_error_test_files_from_output(
                baseline.output,
            )
        if baseline.passing_tests or not failing_files:
            return ""

        command = self._build_targeted_pytest_command(
            test_command,
            failing_files[:1],
            worktree=worktree,
        )
        if command is None:
            return ""
        if "--tb=no" in command:
            command = command.replace("--tb=no", "--tb=short", 1)
        elif "--tb=" not in command:
            command = f"{command} --tb=short"

        probe = self._run_command(
            worktree,
            command,
            timeout=min(self.custom_test_timeout, self.full_test_timeout),
        )
        return probe.output or ""

    def _is_informative_collection_probe_output(self, output: str) -> bool:
        cleaned = normalize_terminal_output(output).strip()
        if not cleaned:
            return False
        lowered = cleaned.lower()
        if lowered.startswith("error: usage:"):
            return False
        if "unrecognized arguments:" in lowered or "expected one argument" in lowered:
            return False
        if ".py:" in cleaned or 'File "' in cleaned or "E   " in cleaned:
            return True
        return any(
            token in lowered
            for token in (
                "traceback",
                "importerror while loading conftest",
                "importerror while importing test module",
                "error collecting",
                "syntaxerror:",
                "nameerror:",
                "typeerror:",
                "attributeerror:",
                "assertionerror:",
                "valueerror:",
                "runtimeerror:",
                "exception:",
            )
        )

    def _extract_collection_error_test_files_from_output(
        self,
        output: str,
    ) -> list[str]:
        output = normalize_terminal_output(output)
        if not output:
            return []
        candidates: list[str] = []
        patterns = [
            re.compile(
                r"ImportError while loading conftest ['\"](?:.+/)?(?P<path>[^'\"]+\.py)['\"]",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"^(?:ERROR|FAILED)\s+collecting\s+(?P<path>\S+\.py)\b",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"^(?P<path>(?:tests?|testing)/\S+\.py):\d+",
                flags=re.IGNORECASE,
            ),
        ]
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            for pattern in patterns:
                match = pattern.search(line)
                if match:
                    candidates.append(match.group("path"))
                    break
        return list(dict.fromkeys(candidate for candidate in candidates if candidate))

    def _build_targeted_pytest_command(
        self,
        test_command: str,
        selected_tests: list[str],
        *,
        worktree: Optional[Path] = None,
    ) -> Optional[str]:
        return build_targeted_pytest_command(
            test_command,
            selected_tests,
            disable_plugin_autoload=self._should_disable_pytest_plugin_autoload(
                test_command,
                worktree=worktree,
            ),
        )

    def _normalize_pytest_command(
        self,
        test_command: str,
        *,
        force_verbose: bool = False,
        worktree: Optional[Path] = None,
    ) -> str:
        normalized = normalize_pytest_command(
            test_command,
            force_verbose=force_verbose,
            disable_plugin_autoload=self._should_disable_pytest_plugin_autoload(
                test_command,
                worktree=worktree,
            ),
        )
        if force_verbose and "--tb=no" in normalized:
            return normalized.replace("--tb=no", "--tb=short", 1)
        return normalized

    def _is_pytest_command(self, test_command: str) -> bool:
        return is_pytest_command(test_command)

    def _split_pytest_command(self, test_command: str) -> Optional[tuple[list[str], list[str]]]:
        parsed = parse_pytest_command(test_command)
        if parsed is None:
            return None
        prefix_tokens = list(parsed.env_prefix_tokens + parsed.invocation_tokens)
        option_tokens = list(parsed.option_tokens)
        return prefix_tokens, option_tokens

    def _build_ephemeral_pytest_command(
        self,
        test_command: Optional[str],
        selected_test_path: str,
        *,
        worktree: Optional[Path] = None,
    ) -> Optional[str]:
        if not test_command:
            return None
        return build_ephemeral_pytest_command(
            test_command,
            selected_test_path,
            disable_plugin_autoload=self._should_disable_pytest_plugin_autoload(
                test_command,
                worktree=worktree,
            ),
        )

    def _prepare_pytest_json_report_command(
        self,
        command: str,
        *,
        label: str,
        worktree: Optional[Path] = None,
    ) -> tuple[str, Optional[Path]]:
        parsed = parse_pytest_command(command)
        if parsed is None:
            return command, None

        option_tokens: list[str] = []
        index = 0
        while index < len(parsed.option_tokens):
            token = parsed.option_tokens[index]
            normalized = token.split("=", 1)[0]
            if normalized == "--json-report":
                index += 1
                continue
            if normalized == "--json-report-file":
                index += 1
                if "=" not in token and index < len(parsed.option_tokens):
                    index += 1
                continue
            option_tokens.append(token)
            index += 1

        safe_label = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-") or "verify"
        report_arg: str
        if worktree is not None:
            report_dir = worktree / ".apex_verification_reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_name = f"apex-{safe_label}-{uuid.uuid4().hex}.json"
            report_path = report_dir / report_name
            try:
                from ..evaluation.target_runtime import target_runtime_path_for_file

                report_arg = target_runtime_path_for_file(
                    self.runtime_env_overrides,
                    report_path,
                )
            except Exception:
                report_arg = str(report_path)
        else:
            with tempfile.NamedTemporaryFile(
                prefix=f"apex-{safe_label}-",
                suffix=".json",
                delete=False,
            ) as handle:
                report_path = Path(handle.name)
            self._cleanup_report_path(report_path)
            report_arg = str(report_path)

        autoload_disabled = any(
            token.startswith("PYTEST_DISABLE_PLUGIN_AUTOLOAD=")
            and token.split("=", 1)[1].strip() not in {"", "0", "false", "False"}
            for token in parsed.env_prefix_tokens
        )
        if autoload_disabled:
            option_tokens.extend(
                [
                    "-p",
                    "pytest_jsonreport.plugin",
                    "--json-report",
                    f"--json-report-file={report_arg}",
                ]
            )
        else:
            option_tokens.extend(
                [
                    "--json-report",
                    f"--json-report-file={report_arg}",
                ]
            )
        rewritten = type(parsed)(
            shell_prefix_tokens=parsed.shell_prefix_tokens,
            env_prefix_tokens=parsed.env_prefix_tokens,
            invocation_tokens=parsed.invocation_tokens,
            option_tokens=tuple(option_tokens),
            target_tokens=parsed.target_tokens,
        )
        return (
            render_pytest_command(
                rewritten,
                disable_plugin_autoload=False,
            ),
            report_path,
        )

    def _pytest_json_report_command_needs_plain_retry(self, output: Any) -> bool:
        text = normalize_terminal_output(str(output or "")).strip().lower()
        if not text:
            return False
        return any(
            token in text
            for token in (
                "no module named 'pytest_jsonreport'",
                'no module named "pytest_jsonreport"',
                'error importing plugin "pytest_jsonreport.plugin"',
                "error importing plugin 'pytest_jsonreport.plugin'",
                "unrecognized arguments: --json-report",
                "no such option: --json-report",
                "no such option: --json-report-file",
            )
        )

    def _load_pytest_json_report_payload(
        self,
        report_path: Optional[Path],
        *,
        minimum_mtime: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        if report_path is None or not report_path.exists():
            return None
        try:
            if minimum_mtime is not None and report_path.stat().st_mtime < (minimum_mtime - 1e-6):
                return None
        except OSError:
            return None
        payload = load_pytest_json_report(report_path)
        return payload if isinstance(payload, dict) else None

    def _resolve_test_inventory(
        self,
        *,
        test_inventory: Optional[Any],
        expected_test_count: Optional[int],
        expected_test_ids: Optional[list[str]],
        test_command: Optional[str],
        baseline_result: Optional[BaselineResult],
    ) -> TestInventory:
        if isinstance(test_inventory, TestInventory):
            inventory = test_inventory.normalized()
        elif isinstance(test_inventory, dict):
            inventory = TestInventory.from_dict(test_inventory)
        else:
            inventory = TestInventory()

        fallback_expected_ids = [
            str(test_id).strip()
            for test_id in list(expected_test_ids or [])
            if str(test_id).strip()
        ]
        fallback_framework = inventory.framework or infer_test_inventory_framework(
            expected_test_ids=fallback_expected_ids,
            test_command=test_command,
        )
        inventory = inventory.merged_with(
            TestInventory(
                framework=fallback_framework,
                source=inventory.source or "verification_expected",
                expected_test_count=max(int(expected_test_count or 0), len(fallback_expected_ids)),
                expected_test_ids=fallback_expected_ids,
                collection_command=derive_test_collection_command(
                    test_command,
                    framework=fallback_framework,
                ),
                test_command=str(test_command or "").strip(),
            )
        )
        if inventory.expected_test_count > 0 or baseline_result is None:
            return inventory

        baseline_framework = inventory.framework or infer_test_inventory_framework(
            expected_test_ids=sorted(baseline_result.collected_tests),
            test_command=test_command,
        )
        return inventory.merged_with(
            TestInventory(
                framework=baseline_framework,
                source=inventory.source or "baseline_collected_surface",
                expected_test_count=int(baseline_result.collected_test_count or 0),
                expected_test_ids=sorted(baseline_result.collected_tests),
                collection_command=derive_test_collection_command(
                    test_command,
                    framework=baseline_framework,
                ),
                test_command=str(test_command or "").strip(),
            )
        )

    def _apply_test_inventory_coverage(
        self,
        test_result: TestResult,
        *,
        report_path: Optional[Path],
        minimum_mtime: Optional[float],
        test_inventory: TestInventory,
        observed_test_count: Optional[int],
    ) -> None:
        inventory = (
            test_inventory.normalized()
            if isinstance(test_inventory, TestInventory)
            else TestInventory()
        )
        if inventory.expected_test_count <= 0:
            return

        payload = self._load_pytest_json_report_payload(
            report_path,
            minimum_mtime=minimum_mtime,
        )
        report_tests = extract_pytest_report_tests(payload) if payload is not None else []
        outcomes = extract_pytest_report_outcomes(report_tests)
        summary = payload.get("summary") if isinstance(payload, dict) else {}
        if isinstance(summary, dict):
            collected_total = summary.get("collected")
            if isinstance(collected_total, int) and collected_total >= 0:
                test_result.collected_test_count = collected_total
        if not test_result.collected_test_count and outcomes:
            test_result.collected_test_count = len(outcomes)
        if (
            not test_result.collected_test_count
            and isinstance(observed_test_count, int)
            and observed_test_count >= 0
        ):
            test_result.collected_test_count = observed_test_count

        coverage_summary = summarize_test_inventory_coverage(
            inventory,
            observed_test_count=(
                test_result.collected_test_count
                if test_result.collected_test_count > 0
                else observed_test_count
            ),
            observed_test_ids=list(outcomes),
            observed_test_outcomes=outcomes,
        )
        if not coverage_summary:
            return

        test_result.expected_test_count = int(
            coverage_summary.get("expected_test_count") or inventory.expected_test_count
        )
        test_result.test_inventory_framework = str(
            coverage_summary.get("test_inventory_framework") or inventory.framework
        )
        test_result.test_inventory_language = str(
            coverage_summary.get("test_inventory_language") or inventory.language
        )
        test_result.test_inventory_source = str(
            coverage_summary.get("test_inventory_source") or inventory.source
        )
        test_result.test_inventory_collection_command = str(
            coverage_summary.get("test_inventory_collection_command")
            or inventory.collection_command
        )
        if not test_result.collected_test_count and isinstance(
            coverage_summary.get("collected_test_count"), int
        ):
            test_result.collected_test_count = int(
                coverage_summary.get("collected_test_count") or 0
            )
        if isinstance(coverage_summary.get("matched_expected_test_count"), int):
            test_result.matched_expected_test_count = int(
                coverage_summary.get("matched_expected_test_count") or 0
            )
        if isinstance(coverage_summary.get("missing_expected_test_count"), int):
            test_result.missing_expected_test_count = int(
                coverage_summary.get("missing_expected_test_count") or 0
            )
        if "missing_test_ids" in coverage_summary:
            test_result.missing_expected_test_ids = list(
                coverage_summary.get("missing_test_ids") or []
            )[:32]
        if isinstance(coverage_summary.get("coverage_preserved"), bool):
            test_result.expected_coverage_preserved = bool(
                coverage_summary.get("coverage_preserved")
            )
        if isinstance(coverage_summary.get("passed"), int):
            test_result.passed = int(coverage_summary.get("passed") or 0)
        if isinstance(coverage_summary.get("failed"), int):
            test_result.failed = int(coverage_summary.get("failed") or 0)
        if isinstance(coverage_summary.get("errors"), int):
            test_result.errors = int(coverage_summary.get("errors") or 0)

    def _cleanup_report_path(self, report_path: Optional[Path]) -> None:
        if report_path is None:
            return
        try:
            report_path.unlink()
        except FileNotFoundError:
            return
        except OSError:
            return
        parent = report_path.parent
        if parent.name == ".apex_verification_reports":
            try:
                parent.rmdir()
            except OSError:
                pass

    def _should_disable_pytest_plugin_autoload(
        self,
        test_command: str,
        *,
        worktree: Optional[Path] = None,
    ) -> bool:
        return should_disable_pytest_plugin_autoload(
            test_command,
            repo_root=worktree or self.repo_path,
        )

    def _build_runtime_python_command(
        self,
        test_command: Optional[str],
        script_path: str,
    ) -> Optional[str]:
        return build_runtime_python_command(test_command, script_path)

    def _materialize_reproduction_script(
        self,
        worktree: Path,
        script_content: str,
        *,
        script_path: Optional[str] = None,
    ) -> tuple[Path, Optional[Path]]:
        if script_path:
            target = self._resolve_reproduction_script_target(worktree, script_path)
            if target.exists():
                return target, None
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(script_content)
            return target, target

        target = worktree / "_apex_reproduction.py"
        if target.exists():
            return target, None
        target.write_text(script_content)
        return target, target

    def _resolve_reproduction_script_target(self, worktree: Path, script_path: str) -> Path:
        raw_path = str(script_path or "").strip()
        if not raw_path:
            return worktree / "_apex_reproduction.py"

        target = Path(raw_path)
        if target.is_absolute():
            mapped = self._target_runtime_host_path(raw_path)
            target = Path(mapped) if mapped else target
        else:
            target = worktree / target

        if self._path_is_within_worktree(target, worktree):
            return target
        return worktree / ".apex_reproduction_scripts" / self._safe_script_path_name(raw_path)

    def _rewrite_reproduction_command_script_path(
        self,
        command: str,
        worktree: Path,
        materialized_path: Optional[Path],
        *,
        script_path: Optional[str],
    ) -> str:
        raw_path = str(script_path or "").strip()
        if not command or not raw_path or materialized_path is None:
            return command
        if raw_path not in command:
            return command
        replacement = self._command_path(worktree, materialized_path)
        return command.replace(raw_path, shlex.quote(replacement))

    def _target_runtime_host_path(self, raw_path: str) -> str:
        try:
            from ..evaluation.target_runtime import target_runtime_host_path_for_file

            return target_runtime_host_path_for_file(self.runtime_env_overrides, raw_path)
        except Exception:
            return raw_path

    @staticmethod
    def _path_is_within_worktree(path: Path, worktree: Path) -> bool:
        try:
            path.resolve(strict=False).relative_to(worktree.resolve(strict=False))
            return True
        except Exception:
            return False

    @staticmethod
    def _safe_script_path_name(raw_path: str) -> str:
        normalized = str(raw_path or "").strip().replace("\\", "/")
        parts = [part for part in normalized.split("/") if part and part not in {".", ".."}]
        basename = "-".join(parts[-3:]) if parts else "reproduction.py"
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", basename).strip(".-")
        return safe or "reproduction.py"

    def _command_path(self, worktree: Path, target: Path) -> str:
        try:
            return str(target.relative_to(worktree))
        except ValueError:
            return str(target)

    def _compute_score(self, verification: VerificationResult) -> float:
        if not verification.syntax_valid:
            return 0.0

        score = 0.0
        # Only award the lint bonus when flake8 actually ran. Abstaining
        # silently on missing flake8 / timeouts (previous behaviour) gave
        # every candidate a free +0.1 in environments without flake8 and
        # made cross-environment scores incomparable.
        if verification.lint_clean and verification.lint_applied:
            score += 0.1
        if verification.test_result:
            test_result = verification.test_result
            if test_result.reproduction_passes:
                score += 0.35
            # Regression bonus is awarded ONLY on a clean pass. A timeout
            # is INCONCLUSIVE (we do not know if regression would have
            # passed) — do not award the bonus, but also do not penalise
            # below the inconclusive baseline. The acceptance gate
            # (selector._verification_meets_acceptance_bar) reads
            # ``regression_inconclusive`` separately so a candidate with
            # strong quick-verification signal can still be accepted
            # despite a regression-suite timeout.
            if test_result.regression_passes:
                score += 0.35
            elif test_result.regression_inconclusive:
                # Partial credit reflecting "we tried, ran out of clock
                # but produced no failure evidence." Larger than zero so
                # an inconclusive candidate ranks above one whose
                # regression suite definitively failed; smaller than the
                # full bonus so a confirmed pass still wins.
                score += 0.15
            score += 0.1 * test_result.pass_rate
        if verification.cross_validation_scores:
            score += 0.1 * (
                sum(verification.cross_validation_scores)
                / len(verification.cross_validation_scores)
            )
        return min(score, 1.0)
