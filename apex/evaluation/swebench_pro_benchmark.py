"""
SWE-Bench Pro public benchmark runner and shared evaluation harness.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import logging
import os
import platform as py_platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..core.cli_backend import ensure_cli_process_cleanup_hooks
from ..core.config import ApexConfig
from ..core.docker_pinning import resolve_image as _resolve_docker_image
from ..core.git_utils import (
    expand_changed_paths,
    ignored_change_pathspecs,
    is_ignored_change_path,
    parse_porcelain_path,
)
from ..core.subprocess_utils import run_process_command, run_shell_command
from ..orchestrator import ApexOrchestrator
from .benchmark import (
    append_benchmark_task_outcome_trace,
    build_apex_ablation_config,
    extract_apex_execution_metadata,
)
from .checkpointing import (
    RUN_STATE_FILENAME,
    atomic_write_json,
    atomic_write_text,
    build_run_state,
    ensure_clean_directory_for_task,
    load_json_if_exists,
    task_result_path,
    write_task_checkpoint,
)
from .commit0_benchmark import serialize_llm_configs
from .run_artifacts import (
    build_benchmark_policy,
    build_prompt_template_fingerprints,
    build_run_manifest,
    capture_environment_snapshot,
    cluster_failures,
    ensure_run_manifest,
    load_run_manifest,
    manifest_summary,
    update_run_manifest,
    write_task_live_state,
)
from .runners._active_manifest import get_active_manifest
from .target_runtime import (
    apply_target_tool_env_to_apex_config,
    docker_image_runtime,
    target_tool_env_overrides,
)

logger = logging.getLogger("apex.evaluation.swebench_pro")


SWEBENCH_PRO_DATASET_NAME = "ScaleAI/SWE-bench_Pro"
SWEBENCH_PRO_DATASET_SPLIT = "test"
SWEBENCH_PRO_DOCKERHUB_USERNAME = "jefzda"
SWEBENCH_PRO_SCRIPTS_BASE_URL = (
    "https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main/run_scripts"
)
SWEBENCH_PRO_SCRIPTS_TREE_URL = (
    "https://api.github.com/repos/scaleapi/SWE-bench_Pro-os/git/trees/main?recursive=1"
)
SWEBENCH_PRO_HARNESS_NAME = "swebench_pro_shared_harness"
SWEBENCH_PRO_HARNESS_VERSION = "2026-04-22.1"
SWEBENCH_PRO_REPORT_KIND_APEX = "apex_swebench_pro"
SWEBENCH_PRO_REPORT_KIND_RAW = "raw_swebench_pro"
SWEBENCH_AGENT_VISIBILITY_PUBLISHED_PARITY = "published_parity"
SWEBENCH_AGENT_VISIBILITY_ONLINE_FAIR = "online_fair"
SWEBENCH_AGENT_VISIBILITY_BENCHMARK_AWARE = "benchmark_aware"
SWEBENCH_ROLLOUT_SELECTION_ORCHESTRATOR = "orchestrator"
SWEBENCH_ROLLOUT_SELECTION_OFFICIAL_EVALUATOR = "official_evaluator"
SWEBENCH_AGENT_TEST_COMMAND_REPO_PUBLIC = "repo_public_heuristic"
SWEBENCH_AGENT_TEST_COMMAND_OFFICIAL = "official_evaluator_wrapper"
SWEBENCH_PREPARE_REPO_FROM_DOCKER_IMAGE = "docker_image"
SWEBENCH_PREPARE_REPO_FROM_HOST_GIT = "host_git"


def _normalize_agent_visibility_mode(mode: str) -> str:
    if mode == SWEBENCH_AGENT_VISIBILITY_ONLINE_FAIR:
        return SWEBENCH_AGENT_VISIBILITY_PUBLISHED_PARITY
    return mode


def _build_swebench_pro_benchmark_policy(
    *,
    block_network: bool,
    docker_platform: Optional[str],
    agent_visibility_mode: str,
    agent_test_command_source: str,
    rollout_selection_policy: str,
) -> dict[str, Any]:
    benchmark_aware = agent_visibility_mode == SWEBENCH_AGENT_VISIBILITY_BENCHMARK_AWARE
    host_rollout_verification = (
        "disabled_non_authoritative"
        if agent_test_command_source == SWEBENCH_AGENT_TEST_COMMAND_REPO_PUBLIC
        else "agent_visible_test_command"
    )
    official_scope = (
        "baseline_rollout_selection_and_final"
        if rollout_selection_policy == SWEBENCH_ROLLOUT_SELECTION_OFFICIAL_EVALUATOR
        else "baseline_and_final_only"
    )
    return build_benchmark_policy(
        benchmark_name="swebench_pro",
        benchmark_family="swebench_pro",
        agent_input_contract={
            "repo_snapshot_visible": True,
            "problem_statement_visible": True,
            "requirements_visible": True,
            "interface_visible": True,
            "test_command_visible": True,
            "agent_visibility_mode": agent_visibility_mode,
            "benchmark_guardrails_visible_in_prompt": benchmark_aware,
            "benchmark_metadata_visible_in_prompt": benchmark_aware,
            "selected_test_targets_visible_in_prompt": benchmark_aware,
            "required_test_identifiers_visible_in_prompt": benchmark_aware,
        },
        orchestrator_input_contract={
            "benchmark_metadata_passed_to_orchestrator": False,
            "agent_test_command_source": agent_test_command_source,
            "host_rollout_verification": host_rollout_verification,
        },
        evaluation_protocol={
            "baseline_evaluation_backend": "official_swebench_pro_docker",
            "final_evaluation_backend": "official_swebench_pro_docker",
            "rollout_selection_policy": rollout_selection_policy,
            "official_evaluator_scope": official_scope,
            "host_rollout_verification": host_rollout_verification,
            "primary_metric": "required_test_accuracy",
            "sampling_protocol": "single_run_per_task",
        },
        environment_policy={
            "agent_execution_isolation": "per_task_temp_sandbox",
            "evaluator_execution_isolation": "official_docker_image",
            "agent_network_access": "inherited_host",
            "evaluator_network_access": "blocked" if block_network else "docker_default",
            "docker_platform": docker_platform or "auto",
            "persistent_outputs_outside_repo": True,
        },
        benchmark_specifics={
            "benchmark_controlled_files_protected": True,
            "evidence_mode": "partial_suite_visible",
            "published_parity_clean": (
                agent_visibility_mode == SWEBENCH_AGENT_VISIBILITY_PUBLISHED_PARITY
                and rollout_selection_policy == SWEBENCH_ROLLOUT_SELECTION_ORCHESTRATOR
            ),
            "agent_test_command_source": agent_test_command_source,
        },
    )


_SWEBENCH_HIDDEN_LIST_FIELD_COUNTS = {
    "fail_to_pass": "fail_to_pass_count",
    "pass_to_pass": "pass_to_pass_count",
    "issue_specificity": "issue_specificity_count",
    "issue_categories": "issue_categories_count",
    "selected_test_files_to_run": "selected_test_file_count",
    "benchmark_test_files": "benchmark_test_file_count",
    "prepared_baseline_files": "prepared_baseline_file_count",
    "passed_tests": "passed_test_count",
    "failed_tests": "failed_test_count",
    "skipped_tests": "skipped_test_count",
    "error_tests": "error_test_count",
    "required_tests": "required_test_count",
    "missing_required_tests": "missing_required_test_count",
    "selected_test_targets": "selected_test_target_count",
}
_SWEBENCH_HIDDEN_TEXT_FIELDS = {
    "base_commit",
    "before_repo_set_cmd",
    "docker_output_path",
    "dockerhub_tag",
    "patch",
    "output_json_path",
    "stderr_log_path",
    "stdout_log_path",
    "test_patch",
}
_SWEBENCH_PUBLISHED_PARITY_EVAL_TEXT_ARTIFACTS = (
    "docker_run.log",
    "entryscript.sh",
    "stderr.log",
    "stdout.log",
)
_SWEBENCH_REDACTED_ARTIFACT_TEXT = (
    "Benchmark-private evaluator artifact redacted in published_parity mode.\n"
)


def _redact_swebench_hidden_test_text(value: str) -> str:
    text = str(value or "")
    if not text:
        return text
    return re.sub(
        r"(?m)^Missing required tests:.*$",
        "Missing required tests: [redacted]",
        text,
    )


def _scrub_swebench_published_parity_artifact_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        scrubbed: dict[str, Any] = {}
        for key, value in payload.items():
            if key in _SWEBENCH_HIDDEN_LIST_FIELD_COUNTS:
                count_key = _SWEBENCH_HIDDEN_LIST_FIELD_COUNTS[key]
                if isinstance(value, (list, tuple, set)):
                    scrubbed[count_key] = len(list(value))
                scrubbed[key] = []
                continue
            if key in _SWEBENCH_HIDDEN_TEXT_FIELDS:
                scrubbed[key] = ""
                continue
            scrubbed[key] = _scrub_swebench_published_parity_artifact_payload(value)
        return scrubbed
    if isinstance(payload, list):
        return [_scrub_swebench_published_parity_artifact_payload(item) for item in payload]
    if isinstance(payload, str):
        return _redact_swebench_hidden_test_text(payload)
    return copy.deepcopy(payload)


def _artifact_safe_swebench_payload(
    payload: Any,
    *,
    include_benchmark_metadata: bool,
) -> Any:
    if include_benchmark_metadata:
        return copy.deepcopy(payload)
    return _scrub_swebench_published_parity_artifact_payload(payload)


def _artifact_safe_swebench_output_json_payload(
    payload: Any,
    *,
    include_benchmark_metadata: bool,
) -> Any:
    if include_benchmark_metadata:
        return copy.deepcopy(payload)
    tests = list((payload or {}).get("tests") or []) if isinstance(payload, dict) else []
    status_counts: dict[str, int] = {}
    for test in tests:
        if not isinstance(test, dict):
            continue
        status = str(test.get("status", "") or "").strip().upper() or "UNKNOWN"
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "benchmark_metadata_redacted": True,
        "test_count": len(tests),
        "status_counts": status_counts,
    }


def _scrub_swebench_evaluation_artifacts(
    artifact_dir: str | Path,
    *,
    include_benchmark_metadata: bool,
) -> None:
    if include_benchmark_metadata:
        return
    artifact_path = Path(artifact_dir)
    output_json_path = artifact_path / "output.json"
    output_payload = load_json_if_exists(output_json_path)
    if output_payload is not None:
        atomic_write_json(
            output_json_path,
            _artifact_safe_swebench_output_json_payload(
                output_payload,
                include_benchmark_metadata=False,
            ),
        )
    for filename in _SWEBENCH_PUBLISHED_PARITY_EVAL_TEXT_ARTIFACTS:
        path = artifact_path / filename
        if path.exists():
            atomic_write_text(path, _SWEBENCH_REDACTED_ARTIFACT_TEXT)


@dataclass
class SWEBenchProTask:
    """One SWE-Bench Pro public benchmark task."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    requirements: str = ""
    interface: str = ""
    repo_language: str = ""
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    issue_specificity: list[str] = field(default_factory=list)
    issue_categories: list[str] = field(default_factory=list)
    before_repo_set_cmd: str = ""
    selected_test_files_to_run: list[str] = field(default_factory=list)
    dockerhub_tag: str = ""
    patch: str = ""
    test_patch: str = ""
    benchmark_test_files: list[str] = field(default_factory=list)
    prepared_baseline_files: list[str] = field(default_factory=list)

    @property
    def required_tests(self) -> list[str]:
        return sorted(set(self.fail_to_pass) | set(self.pass_to_pass))

    @property
    def repo_name(self) -> str:
        return self.repo.split("/")[-1]

    @property
    def benchmark_controlled_files(self) -> list[str]:
        return sorted(set(self.benchmark_test_files) | set(self.prepared_baseline_files))

    def build_issue_description(
        self,
        test_command: Optional[str] = None,
        *,
        include_benchmark_guardrails: bool = False,
        include_benchmark_metadata: bool = False,
        include_selected_test_targets: bool = False,
        include_required_tests: bool = False,
    ) -> str:
        lines = ["Resolve the repository issue by changing application code."]
        if include_benchmark_guardrails:
            lines.append("Do not modify benchmark-controlled tests or benchmark harness files.")
        lines.extend(["", f"Repository: {self.repo}"])

        if self.repo_language:
            lines.append(f"Repository language: {self.repo_language}")
        if include_benchmark_metadata:
            lines.extend(
                [
                    f"Benchmark instance: {self.instance_id}",
                    f"Base commit: {self.base_commit}",
                ]
            )
        if test_command:
            lines.append(f"Repository test command: {test_command}")
        lines.extend(["", "Problem statement:", self.problem_statement.strip()])

        if self.requirements.strip():
            lines.extend(["", "Additional requirements:", self.requirements.strip()])
        if self.interface.strip():
            lines.extend(["", "Interface notes:", self.interface.strip()])
        if include_selected_test_targets and self.selected_test_files_to_run:
            lines.extend(
                [
                    "",
                    "Selected benchmark test targets:",
                    _format_list_preview(self.selected_test_files_to_run, max_items=40),
                ]
            )
        if include_required_tests and self.required_tests:
            lines.extend(
                [
                    "",
                    "Required tests that must pass after the fix:",
                    _format_list_preview(self.required_tests, max_items=40),
                ]
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "requirements": self.requirements,
            "interface": self.interface,
            "repo_language": self.repo_language,
            "fail_to_pass": list(self.fail_to_pass),
            "pass_to_pass": list(self.pass_to_pass),
            "issue_specificity": list(self.issue_specificity),
            "issue_categories": list(self.issue_categories),
            "before_repo_set_cmd": self.before_repo_set_cmd,
            "selected_test_files_to_run": list(self.selected_test_files_to_run),
            "dockerhub_tag": self.dockerhub_tag,
            "patch": self.patch,
            "test_patch": self.test_patch,
            "benchmark_test_files": list(self.benchmark_test_files),
            "prepared_baseline_files": list(self.prepared_baseline_files),
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "SWEBenchProTask":
        test_patch = row.get("test_patch", "") or ""
        before_repo_set_cmd = row.get("before_repo_set_cmd", "") or ""
        benchmark_test_files = _parse_patch_paths(test_patch)
        prepared_baseline_files = _parse_before_repo_set_paths(before_repo_set_cmd)
        return cls(
            instance_id=str(row["instance_id"]),
            repo=str(row["repo"]),
            base_commit=str(row["base_commit"]),
            problem_statement=str(row.get("problem_statement", "") or ""),
            requirements=str(row.get("requirements", "") or ""),
            interface=str(row.get("interface", "") or ""),
            repo_language=str(row.get("repo_language", "") or ""),
            fail_to_pass=_parse_literal_list(row.get("fail_to_pass")),
            pass_to_pass=_parse_literal_list(row.get("pass_to_pass")),
            issue_specificity=_parse_literal_list(row.get("issue_specificity")),
            issue_categories=_parse_literal_list(row.get("issue_categories")),
            before_repo_set_cmd=before_repo_set_cmd,
            selected_test_files_to_run=_parse_literal_list(row.get("selected_test_files_to_run")),
            dockerhub_tag=str(row.get("dockerhub_tag", "") or ""),
            patch=str(row.get("patch", "") or ""),
            test_patch=test_patch,
            benchmark_test_files=benchmark_test_files,
            prepared_baseline_files=prepared_baseline_files,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "SWEBenchProTask":
        payload = json.loads(Path(path).read_text())
        return cls(
            instance_id=payload["instance_id"],
            repo=payload["repo"],
            base_commit=payload["base_commit"],
            problem_statement=payload.get("problem_statement", ""),
            requirements=payload.get("requirements", ""),
            interface=payload.get("interface", ""),
            repo_language=payload.get("repo_language", ""),
            fail_to_pass=list(payload.get("fail_to_pass", [])),
            pass_to_pass=list(payload.get("pass_to_pass", [])),
            issue_specificity=list(payload.get("issue_specificity", [])),
            issue_categories=list(payload.get("issue_categories", [])),
            before_repo_set_cmd=payload.get("before_repo_set_cmd", ""),
            selected_test_files_to_run=list(payload.get("selected_test_files_to_run", [])),
            dockerhub_tag=payload.get("dockerhub_tag", ""),
            patch=payload.get("patch", ""),
            test_patch=payload.get("test_patch", ""),
            benchmark_test_files=list(payload.get("benchmark_test_files", [])),
            prepared_baseline_files=list(payload.get("prepared_baseline_files", [])),
        )


def _artifact_safe_swebench_task_payload(
    task: "SWEBenchProTask",
    *,
    include_benchmark_metadata: bool,
) -> dict[str, Any]:
    payload = task.to_dict()
    if include_benchmark_metadata:
        return payload
    scrubbed = _scrub_swebench_published_parity_artifact_payload(payload)
    if isinstance(scrubbed, dict):
        scrubbed["benchmark_metadata_redacted"] = True
    return scrubbed


@dataclass
class SWEBenchProEvaluation:
    """One SWE-Bench Pro test execution summary."""

    returncode: int
    output: str = ""
    docker_returncode: int = 0
    passed_tests: set[str] = field(default_factory=set)
    failed_tests: set[str] = field(default_factory=set)
    skipped_tests: set[str] = field(default_factory=set)
    error_tests: set[str] = field(default_factory=set)
    required_tests: set[str] = field(default_factory=set)
    selected_test_targets: list[str] = field(default_factory=list)
    ignored_changes: list[str] = field(default_factory=list)
    stdout_log_path: Optional[str] = None
    stderr_log_path: Optional[str] = None
    output_json_path: Optional[str] = None
    docker_output_path: Optional[str] = None
    scoring_source: str = "swebench_required_tests"
    duration_seconds: float = 0.0

    @property
    def missing_required_tests(self) -> list[str]:
        return sorted(self.required_tests - self.passed_tests)

    @property
    def total_required_tests(self) -> int:
        return len(self.required_tests)

    @property
    def required_pass_rate(self) -> float:
        if not self.required_tests:
            return 1.0 if self.returncode == 0 else 0.0
        return (len(self.required_tests) - len(self.missing_required_tests)) / len(
            self.required_tests
        )

    @property
    def all_required_tests_passed(self) -> bool:
        return not self.missing_required_tests

    def to_dict(self) -> dict[str, Any]:
        return {
            "returncode": self.returncode,
            "docker_returncode": self.docker_returncode,
            "output": self.output,
            "passed_tests": sorted(self.passed_tests),
            "failed_tests": sorted(self.failed_tests),
            "skipped_tests": sorted(self.skipped_tests),
            "error_tests": sorted(self.error_tests),
            "required_tests": sorted(self.required_tests),
            "missing_required_tests": self.missing_required_tests,
            "required_pass_rate": self.required_pass_rate,
            "all_required_tests_passed": self.all_required_tests_passed,
            "selected_test_targets": list(self.selected_test_targets),
            "ignored_changes": list(self.ignored_changes),
            "stdout_log_path": self.stdout_log_path,
            "stderr_log_path": self.stderr_log_path,
            "output_json_path": self.output_json_path,
            "docker_output_path": self.docker_output_path,
            "scoring_source": self.scoring_source,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SWEBenchProEvaluation":
        return cls(
            returncode=int(payload.get("returncode", 1)),
            output=str(payload.get("output", "") or ""),
            docker_returncode=int(payload.get("docker_returncode", 0) or 0),
            passed_tests=set(payload.get("passed_tests") or []),
            failed_tests=set(payload.get("failed_tests") or []),
            skipped_tests=set(payload.get("skipped_tests") or []),
            error_tests=set(payload.get("error_tests") or []),
            required_tests=set(payload.get("required_tests") or []),
            selected_test_targets=list(payload.get("selected_test_targets") or []),
            ignored_changes=list(payload.get("ignored_changes") or []),
            stdout_log_path=payload.get("stdout_log_path"),
            stderr_log_path=payload.get("stderr_log_path"),
            output_json_path=payload.get("output_json_path"),
            docker_output_path=payload.get("docker_output_path"),
            scoring_source=str(
                payload.get("scoring_source", "swebench_required_tests")
                or "swebench_required_tests"
            ),
            duration_seconds=float(payload.get("duration_seconds", 0.0) or 0.0),
        )


@dataclass
class SWEBenchProTaskResult:
    """Execution result for one SWE-Bench Pro task."""

    task_name: str
    instance_id: str
    repo: str
    success: bool
    baseline_failed: bool
    final_tests_passed: bool
    baseline: SWEBenchProEvaluation
    final: SWEBenchProEvaluation
    orchestrator_success: bool = False
    candidate_found: bool = False
    orchestrator_selected_rollout_id: Optional[int] = None
    orchestrator_selected_worktree_path: Optional[str] = None
    selected_rollout_id: Optional[int] = None
    selected_worktree_path: Optional[str] = None
    total_tokens: int = 0
    duration_seconds: float = 0.0
    result_path: Optional[str] = None
    failure_reason: Optional[str] = None
    skipped: bool = False
    skip_category: Optional[str] = None
    execution_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pass_rate_delta(self) -> float:
        return self.final.required_pass_rate - self.baseline.required_pass_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "instance_id": self.instance_id,
            "repo": self.repo,
            "success": self.success,
            "baseline_failed": self.baseline_failed,
            "final_tests_passed": self.final_tests_passed,
            "baseline": self.baseline.to_dict(),
            "final": self.final.to_dict(),
            "orchestrator_success": self.orchestrator_success,
            "candidate_found": self.candidate_found,
            "orchestrator_selected_rollout_id": self.orchestrator_selected_rollout_id,
            "orchestrator_selected_worktree_path": self.orchestrator_selected_worktree_path,
            "pass_rate_delta": self.pass_rate_delta,
            "selected_rollout_id": self.selected_rollout_id,
            "selected_worktree_path": self.selected_worktree_path,
            "total_tokens": self.total_tokens,
            "duration_seconds": self.duration_seconds,
            "result_path": self.result_path,
            "failure_reason": self.failure_reason,
            "skipped": self.skipped,
            "skip_category": self.skip_category,
            "execution_metadata": copy.deepcopy(self.execution_metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SWEBenchProTaskResult":
        uses_explicit_benchmark_success = any(
            key in payload
            for key in (
                "orchestrator_success",
                "candidate_found",
                "orchestrator_selected_rollout_id",
                "orchestrator_selected_worktree_path",
            )
        )
        final_tests_passed = bool(payload.get("final_tests_passed", False))
        return cls(
            task_name=str(payload["task_name"]),
            instance_id=str(payload["instance_id"]),
            repo=str(payload["repo"]),
            success=(
                bool(payload.get("success", False))
                if uses_explicit_benchmark_success
                else final_tests_passed
            ),
            baseline_failed=bool(payload.get("baseline_failed", False)),
            final_tests_passed=final_tests_passed,
            baseline=SWEBenchProEvaluation.from_dict(dict(payload.get("baseline") or {})),
            final=SWEBenchProEvaluation.from_dict(dict(payload.get("final") or {})),
            orchestrator_success=bool(
                payload.get("orchestrator_success", payload.get("success", False))
            ),
            candidate_found=bool(payload.get("candidate_found", False)),
            orchestrator_selected_rollout_id=payload.get("orchestrator_selected_rollout_id"),
            orchestrator_selected_worktree_path=payload.get("orchestrator_selected_worktree_path"),
            selected_rollout_id=payload.get("selected_rollout_id"),
            selected_worktree_path=payload.get("selected_worktree_path"),
            total_tokens=int(payload.get("total_tokens", 0) or 0),
            duration_seconds=float(payload.get("duration_seconds", 0.0) or 0.0),
            result_path=payload.get("result_path"),
            failure_reason=payload.get("failure_reason"),
            skipped=bool(payload.get("skipped", False)),
            skip_category=payload.get("skip_category"),
            execution_metadata=dict(payload.get("execution_metadata") or {}),
        )


@dataclass
class SWEBenchProBenchmarkReport:
    """Aggregate SWE-Bench Pro benchmark report."""

    tasks: list[SWEBenchProTaskResult] = field(default_factory=list)
    requested_task_ids: list[str] = field(default_factory=list)
    requested_repo_names: list[str] = field(default_factory=list)
    started_at: float = 0.0
    updated_at: float = 0.0
    finished_at: float = 0.0
    dataset_name: str = SWEBENCH_PRO_DATASET_NAME
    dataset_split: str = SWEBENCH_PRO_DATASET_SPLIT
    report_kind: str = SWEBENCH_PRO_REPORT_KIND_APEX
    harness_name: str = SWEBENCH_PRO_HARNESS_NAME
    harness_version: str = SWEBENCH_PRO_HARNESS_VERSION
    config_source: Optional[str] = None
    model_config: list[dict[str, Any]] = field(default_factory=list)
    ablation_config: dict[str, Any] = field(default_factory=dict)
    agent_visibility_mode: str = SWEBENCH_AGENT_VISIBILITY_PUBLISHED_PARITY
    agent_test_command_source: str = SWEBENCH_AGENT_TEST_COMMAND_REPO_PUBLIC
    official_evaluator_scope: str = "baseline_and_final_only"
    benchmark_clean: bool = True
    run_manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def repo_names(self) -> list[str]:
        if self.requested_repo_names:
            return list(self.requested_repo_names)
        return sorted({task.repo for task in self.tasks})

    @property
    def total_tasks(self) -> int:
        if self.requested_task_ids:
            return len(self.requested_task_ids)
        return len(self.tasks)

    @property
    def completed_tasks(self) -> int:
        return len(self.tasks)

    @property
    def completed(self) -> bool:
        return self.finished_at > 0.0

    @property
    def duration_seconds(self) -> float:
        if self.started_at <= 0.0:
            return 0.0
        end_time = self.finished_at or self.updated_at or self.started_at
        return max(0.0, end_time - self.started_at)

    @property
    def solved_tasks(self) -> int:
        return sum(1 for task in self.tasks if task.final_tests_passed)

    @property
    def skipped_tasks(self) -> int:
        return sum(1 for task in self.tasks if task.skipped)

    @property
    def runnable_tasks(self) -> int:
        return sum(1 for task in self.tasks if not task.skipped)

    @property
    def solved_runnable_tasks(self) -> int:
        return sum(1 for task in self.tasks if not task.skipped and task.final_tests_passed)

    @property
    def score(self) -> float:
        if not self.tasks:
            return 0.0
        return self.solved_tasks / len(self.tasks)

    @property
    def baseline_score(self) -> float:
        if not self.tasks:
            return 0.0
        return sum(1 for task in self.tasks if task.baseline.all_required_tests_passed) / len(
            self.tasks
        )

    @property
    def score_improvement(self) -> float:
        return self.score - self.baseline_score

    @property
    def runnable_score(self) -> float:
        runnable = [task for task in self.tasks if not task.skipped]
        if not runnable:
            return 0.0
        return sum(1 for task in runnable if task.final_tests_passed) / len(runnable)

    @property
    def runnable_baseline_score(self) -> float:
        runnable = [task for task in self.tasks if not task.skipped]
        if not runnable:
            return 0.0
        return sum(1 for task in runnable if task.baseline.all_required_tests_passed) / len(
            runnable
        )

    @property
    def runnable_score_improvement(self) -> float:
        return self.runnable_score - self.runnable_baseline_score

    @property
    def score_percent(self) -> float:
        return 100.0 * self.score

    @property
    def baseline_score_percent(self) -> float:
        return 100.0 * self.baseline_score

    @property
    def score_improvement_percent(self) -> float:
        return 100.0 * self.score_improvement

    @property
    def runnable_score_percent(self) -> float:
        return 100.0 * self.runnable_score

    @property
    def runnable_baseline_score_percent(self) -> float:
        return 100.0 * self.runnable_baseline_score

    @property
    def runnable_score_improvement_percent(self) -> float:
        return 100.0 * self.runnable_score_improvement

    @property
    def scoring_method(self) -> str:
        return "required_test_accuracy"

    @property
    def scoring_source(self) -> str:
        return self.scoring_method

    @property
    def failure_clusters(self) -> list[dict[str, Any]]:
        return cluster_failures(
            [task.to_dict() for task in self.tasks],
            benchmark_family="swebench_pro",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_kind": self.report_kind,
            "harness_name": self.harness_name,
            "harness_version": self.harness_version,
            "config_source": self.config_source,
            "requested_task_ids": list(self.requested_task_ids),
            "repo_names": self.repo_names,
            "model_config": copy.deepcopy(self.model_config),
            "ablation_config": copy.deepcopy(self.ablation_config),
            "evaluation_fairness": {
                "benchmark_clean": self.benchmark_clean,
                "agent_visibility_mode": self.agent_visibility_mode,
                "agent_test_command_source": self.agent_test_command_source,
                "official_evaluator_scope": self.official_evaluator_scope,
            },
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "completed": self.completed,
            "dataset_name": self.dataset_name,
            "dataset_split": self.dataset_split,
            "completed_tasks": self.completed_tasks,
            "solved_tasks": self.solved_tasks,
            "total_tasks": self.total_tasks,
            "skipped_tasks": self.skipped_tasks,
            "runnable_tasks": self.runnable_tasks,
            "solved_runnable_tasks": self.solved_runnable_tasks,
            "score": self.score,
            "score_percent": self.score_percent,
            "baseline_score": self.baseline_score,
            "baseline_score_percent": self.baseline_score_percent,
            "score_improvement": self.score_improvement,
            "score_improvement_percent": self.score_improvement_percent,
            "runnable_score": self.runnable_score,
            "runnable_score_percent": self.runnable_score_percent,
            "runnable_baseline_score": self.runnable_baseline_score,
            "runnable_baseline_score_percent": self.runnable_baseline_score_percent,
            "runnable_score_improvement": self.runnable_score_improvement,
            "runnable_score_improvement_percent": self.runnable_score_improvement_percent,
            "scoring_source": self.scoring_source,
            "scoring_method": self.scoring_method,
            "run_manifest": manifest_summary(self.run_manifest),
            "failure_clusters": self.failure_clusters,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    def to_markdown(self) -> str:
        rollout_buckets = (self.ablation_config.get("allocator") or {}).get("rollout_buckets") or []
        planner_brief_family_cap = (self.ablation_config.get("scaffold") or {}).get(
            "planner_brief_family_cap",
            "n/a",
        )
        task_state_graph_enabled = (self.ablation_config.get("scaffold") or {}).get(
            "task_state_graph_enabled",
            False,
        )
        frontier_targeting_enabled = (self.ablation_config.get("scaffold") or {}).get(
            "frontier_targeting_enabled",
            False,
        )
        dynamic_transitions = (self.ablation_config.get("scaffold") or {}).get(
            "dynamic_transitions_enabled",
            False,
        )
        feedback_config = self.ablation_config.get("feedback") or {}
        search_mode = (self.ablation_config.get("search") or {}).get("mode", "off")
        selection_config = self.ablation_config.get("selection") or {}
        memory_config = self.ablation_config.get("memory") or {}
        status = "completed" if self.completed else "in_progress"
        lines = [
            "# APEX SWE-Bench Pro Benchmark Report",
            "",
            f"- Harness: {self.harness_name} v{self.harness_version}",
            f"- Report kind: {self.report_kind}",
            f"- Status: {status}",
            f"- Config source: {self.config_source or 'default'}",
            f"- Model config: {_format_model_config_summary(self.model_config)}",
            f"- Rollout allocator: {(self.ablation_config.get('allocator') or {}).get('policy', 'unknown')}",
            f"- Rollout buckets: {', '.join(str(bucket) for bucket in rollout_buckets) or 'n/a'}",
            f"- Scaffold mode: {(self.ablation_config.get('scaffold') or {}).get('policy', 'unknown')}",
            f"- Planner brief family cap: {planner_brief_family_cap}",
            f"- Task-state graph: {'enabled' if task_state_graph_enabled else 'disabled'}",
            f"- Frontier targeting: {'enabled' if frontier_targeting_enabled else 'disabled'}",
            f"- Explicit search: {search_mode}",
            f"- COP transitions: {'enabled' if dynamic_transitions else 'disabled'}",
            (
                "- Rollout quick verification: "
                f"{'enabled' if feedback_config.get('quick_verification_enabled') else 'disabled'} "
                f"(max_tests={feedback_config.get('quick_verification_max_tests', 'n/a')}, "
                f"timeout={feedback_config.get('quick_verification_timeout_seconds', 'n/a')}s)"
            ),
            (
                "- Selection critic: "
                f"{'enabled' if selection_config.get('critic_reranking_enabled') else 'disabled'} "
                f"(weight={selection_config.get('critic_weight', 0)})"
            ),
            (
                "- Repo memory: "
                + (
                    "enabled (non-i.i.d.; disclose when comparing to fresh-run baselines)"
                    if memory_config.get("repo_memory_enabled")
                    else "disabled"
                )
            ),
            f"- Repos: {', '.join(self.repo_names) or 'none'}",
            f"- Completed repos: {self.completed_tasks}/{self.total_tasks}",
            f"- Benchmark cleanliness: {'published-parity' if self.benchmark_clean else 'benchmark-aware'}",
            f"- Agent visibility mode: {self.agent_visibility_mode}",
            f"- Agent test command source: {self.agent_test_command_source}",
            f"- Official evaluator usage: {self.official_evaluator_scope}",
            f"- Accuracy: {self.score_percent:.1f}%",
            f"- Baseline accuracy: {self.baseline_score_percent:.1f}%",
            f"- Accuracy delta: {self.score_improvement_percent:+.1f}%",
            f"- Solved tasks: {self.solved_tasks}/{self.total_tasks}",
            f"- Duration: {self.duration_seconds:.1f}s",
            "",
        ]
        if self.skipped_tasks:
            lines.extend(
                [
                    f"- Runnable-only accuracy: {self.runnable_score_percent:.1f}%",
                    f"- Runnable-only baseline: {self.runnable_baseline_score_percent:.1f}%",
                    f"- Runnable-only delta: {self.runnable_score_improvement_percent:+.1f}%",
                    f"- Runnable tasks solved: {self.solved_runnable_tasks}/{self.runnable_tasks}",
                    f"- Skipped tasks: {self.skipped_tasks}",
                    "",
                ]
            )
        lines.extend(
            [
                "| Instance | Repo | Baseline | Final | Delta | Solved | Status | Tokens | Duration (s) |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for task in self.tasks:
            status = "skipped" if task.skipped else "scored"
            if task.skip_category:
                status = f"{status} ({task.skip_category})"
            lines.append(
                "| {name} | {repo} | {baseline:.1f}% | {final:.1f}% | {delta:+.1f}% | {solved} | {status} | {tokens} | {duration:.1f} |".format(
                    name=task.task_name,
                    repo=task.repo,
                    baseline=100.0 * task.baseline.required_pass_rate,
                    final=100.0 * task.final.required_pass_rate,
                    delta=100.0 * task.pass_rate_delta,
                    solved="yes" if task.final_tests_passed else "no",
                    status=status,
                    tokens=task.total_tokens,
                    duration=task.duration_seconds,
                )
            )
        failure_clusters = self.failure_clusters
        if failure_clusters:
            lines.extend(
                [
                    "",
                    "## Failure Clusters",
                    "",
                    "| Root Cause | Count | Example Tasks |",
                    "| --- | --- | --- |",
                ]
            )
            for cluster in failure_clusters:
                lines.append(
                    "| {bucket} | {count} | {tasks} |".format(
                        bucket=cluster.get("bucket"),
                        count=cluster.get("count"),
                        tasks=", ".join(cluster.get("tasks") or []) or "-",
                    )
                )
        return "\n".join(lines)


@dataclass
class _CandidateEvaluation:
    rollout_id: int
    worktree_path: Path
    evaluation: SWEBenchProEvaluation


class SWEBenchProHarness:
    """Shared dataset, repo materialization, and evaluation helpers."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        dataset_name: str = SWEBENCH_PRO_DATASET_NAME,
        dataset_split: str = SWEBENCH_PRO_DATASET_SPLIT,
        dockerhub_username: str = SWEBENCH_PRO_DOCKERHUB_USERNAME,
        scripts_cache_dir: str | Path | None = None,
        docker_platform: Optional[str] = None,
        block_network: bool = False,
        agent_visibility_mode: str = SWEBENCH_AGENT_VISIBILITY_PUBLISHED_PARITY,
        prepare_repo_mode: str = SWEBENCH_PREPARE_REPO_FROM_DOCKER_IMAGE,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.dockerhub_username = dockerhub_username
        self.scripts_cache_dir = (
            Path(scripts_cache_dir).resolve()
            if scripts_cache_dir is not None
            else (self.output_dir / "_scripts_cache").resolve()
        )
        self.scripts_cache_dir.mkdir(parents=True, exist_ok=True)
        self.repo_mirror_cache_dir = (self.scripts_cache_dir / "_repo_mirrors").resolve()
        self.repo_mirror_cache_dir.mkdir(parents=True, exist_ok=True)
        self.docker_platform = docker_platform or self._detect_default_docker_platform()
        self.block_network = block_network
        self.agent_visibility_mode = _normalize_agent_visibility_mode(agent_visibility_mode)
        self.prepare_repo_mode = (
            str(prepare_repo_mode or SWEBENCH_PREPARE_REPO_FROM_DOCKER_IMAGE).strip().lower()
        )
        self.project_root = Path(__file__).resolve().parents[2]
        self._scripts_manifest_cache: Optional[set[str]] = None
        self._synced_host_repo_mirrors: set[str] = set()
        self._host_repo_mirror_lock = threading.Lock()
        self._host_repo_mirror_locks: dict[str, threading.Lock] = {}
        self._validate_agent_visibility_mode()
        self._validate_prepare_repo_mode()

    def discover_tasks(
        self,
        *,
        instances: Optional[list[str]] = None,
        repos: Optional[list[str]] = None,
        languages: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> list[SWEBenchProTask]:
        from datasets import load_dataset

        dataset = load_dataset(self.dataset_name, split=self.dataset_split)
        allowed_instances = set(instances or [])
        allowed_repos = {item for item in repos or []}
        allowed_repo_suffixes = {item.split("/")[-1] for item in repos or []}
        allowed_languages = {item.lower() for item in languages or []}

        tasks: list[SWEBenchProTask] = []
        for row in dataset:
            if allowed_instances and row["instance_id"] not in allowed_instances:
                continue
            if (
                allowed_repos
                and row["repo"] not in allowed_repos
                and row["repo"].split("/")[-1] not in allowed_repo_suffixes
            ):
                continue
            if (
                allowed_languages
                and str(row.get("repo_language", "")).lower() not in allowed_languages
            ):
                continue

            tasks.append(SWEBenchProTask.from_row(dict(row)))
            if limit is not None and len(tasks) >= limit:
                break
        return tasks

    def load_task(
        self,
        *,
        task_file: str | Path | None = None,
        instance_id: Optional[str] = None,
    ) -> SWEBenchProTask:
        if task_file is not None:
            return SWEBenchProTask.from_file(task_file)
        if not instance_id:
            raise ValueError("Either task_file or instance_id is required.")
        tasks = self.discover_tasks(instances=[instance_id], limit=1)
        if not tasks:
            raise RuntimeError(f"SWE-Bench Pro instance not found: {instance_id}")
        return tasks[0]

    def resolve_image_uri(self, task: SWEBenchProTask) -> str:
        if not task.dockerhub_tag:
            raise RuntimeError(f"Task {task.instance_id} is missing dockerhub_tag.")
        tag = f"{self.dockerhub_username}/sweap-images:{task.dockerhub_tag}"
        return _resolve_docker_image(
            tag,
            record_to_manifest=get_active_manifest(),
        ).image_ref

    def write_task_metadata(
        self,
        task: SWEBenchProTask,
        path: str | Path,
        *,
        include_benchmark_metadata: bool = True,
    ) -> Path:
        metadata_path = Path(path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(
                _artifact_safe_swebench_task_payload(
                    task,
                    include_benchmark_metadata=include_benchmark_metadata,
                ),
                indent=2,
            )
        )
        return metadata_path

    def build_official_evaluator_command(self, task_file: str | Path) -> str:
        command = [
            "python3",
            "-m",
            "apex.evaluation.swebench_pro_benchmark",
            "eval-repo",
            "--repo-root",
            ".",
            "--task-file",
            str(Path(task_file).resolve()),
            "--scripts-cache-dir",
            str(self.scripts_cache_dir),
            "--dockerhub-username",
            self.dockerhub_username,
        ]
        if self.docker_platform:
            command.extend(["--docker-platform", self.docker_platform])
        if self.block_network:
            command.append("--block-network")
        return f"PYTHONPATH={shlex.quote(str(self.project_root))}:$PYTHONPATH " + " ".join(
            shlex.quote(token) for token in command
        )

    def build_agent_test_command(
        self,
        task: SWEBenchProTask,
        repo_dir: str | Path,
    ) -> Optional[str]:
        return self.build_public_test_command(task, Path(repo_dir))

    def build_public_test_command(
        self,
        task: SWEBenchProTask,
        repo_dir: Path,
    ) -> Optional[str]:
        builders = self._ordered_public_test_command_builders(task.repo_language)
        for builder in builders:
            command = builder(repo_dir)
            if command:
                return command
        return None

    def _restore_published_parity_benchmark_controlled_files(
        self,
        task: SWEBenchProTask,
        repo_dir: Path,
    ) -> None:
        if self.agent_visibility_mode != SWEBENCH_AGENT_VISIBILITY_PUBLISHED_PARITY:
            return

        benchmark_paths = [
            str(path or "").strip().replace("\\", "/") for path in task.benchmark_controlled_files
        ]
        benchmark_paths = [path for path in benchmark_paths if path]
        if not benchmark_paths:
            return

        checkout_paths: list[str] = []
        remove_paths: list[str] = []
        for rel_path in benchmark_paths:
            if self._path_exists_in_commit(repo_dir, task.base_commit, rel_path):
                checkout_paths.append(rel_path)
            else:
                remove_paths.append(rel_path)

        if checkout_paths:
            self._run_process(
                ["git", "checkout", task.base_commit, "--", *checkout_paths],
                cwd=repo_dir,
                timeout=300,
            )

        for rel_path in remove_paths:
            target = repo_dir / rel_path
            if target.is_symlink() or target.is_file():
                target.unlink(missing_ok=True)
            elif target.is_dir():
                shutil.rmtree(target, ignore_errors=True)

    def _path_exists_in_commit(
        self,
        repo_dir: Path,
        commit: str,
        rel_path: str,
    ) -> bool:
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}:{rel_path}"],
            capture_output=True,
            text=True,
            cwd=str(repo_dir),
        )
        return result.returncode == 0

    def _host_repo_clone_timeout_seconds(self) -> int:
        return 3600

    def _host_repo_remote_url(self, task: SWEBenchProTask) -> str:
        return f"https://github.com/{task.repo}.git"

    def _host_repo_mirror_dir(self, task: SWEBenchProTask) -> Path:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "__", str(task.repo or "").strip())
        slug = slug.strip("._-") or "repo"
        return self.repo_mirror_cache_dir / f"{slug}.git"

    def _ensure_host_repo_mirror(self, task: SWEBenchProTask) -> Path:
        mirror_dir = self._host_repo_mirror_dir(task)
        remote_url = self._host_repo_remote_url(task)
        with self._host_repo_mirror_lock:
            repo_lock = self._host_repo_mirror_locks.setdefault(task.repo, threading.Lock())
        with repo_lock:
            if not mirror_dir.exists() or not (mirror_dir / "HEAD").exists():
                if mirror_dir.exists():
                    shutil.rmtree(mirror_dir, ignore_errors=True)
                self._run_process(
                    ["git", "clone", "--mirror", remote_url, str(mirror_dir)],
                    timeout=self._host_repo_clone_timeout_seconds(),
                )
                self._synced_host_repo_mirrors.add(task.repo)
                return mirror_dir

            if task.repo not in self._synced_host_repo_mirrors:
                self._run_process(
                    ["git", "remote", "set-url", "origin", remote_url],
                    cwd=mirror_dir,
                    timeout=60,
                )
                self._run_process(
                    ["git", "fetch", "--prune", "origin"],
                    cwd=mirror_dir,
                    timeout=self._host_repo_clone_timeout_seconds(),
                )
                self._synced_host_repo_mirrors.add(task.repo)
        return mirror_dir

    def _prepare_repo_from_docker_image(self, task: SWEBenchProTask, repo_dir: Path) -> None:
        repo_dir.mkdir(parents=True, exist_ok=True)
        image_uri = self.resolve_image_uri(task)
        self._ensure_image_available(image_uri)
        container_id = self._create_container(image_uri)
        try:
            self._run_process(
                ["docker", "cp", f"{container_id}:/app/.", str(repo_dir)],
                timeout=3600,
            )
        finally:
            subprocess.run(["docker", "rm", "-f", "-v", container_id], capture_output=True)

    def _prepare_repo_from_host_git(self, task: SWEBenchProTask, repo_dir: Path) -> None:
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        if repo_dir.exists():
            shutil.rmtree(repo_dir, ignore_errors=True)
        mirror_dir = self._ensure_host_repo_mirror(task)
        self._run_process(
            ["git", "clone", str(mirror_dir), str(repo_dir)],
            timeout=self._host_repo_clone_timeout_seconds(),
        )
        self._run_process(
            ["git", "remote", "set-url", "origin", self._host_repo_remote_url(task)],
            cwd=repo_dir,
            timeout=60,
        )
        self._run_process(
            ["git", "checkout", "-B", "apex-base", task.base_commit],
            cwd=repo_dir,
            timeout=300,
        )

    def prepare_repo(self, task: SWEBenchProTask, repo_dir: Path) -> None:
        if self.prepare_repo_mode == SWEBENCH_PREPARE_REPO_FROM_HOST_GIT:
            self._prepare_repo_from_host_git(task, repo_dir)
        else:
            self._prepare_repo_from_docker_image(task, repo_dir)

        if not (repo_dir / ".git").exists():
            source = (
                "host git clone"
                if self.prepare_repo_mode == SWEBENCH_PREPARE_REPO_FROM_HOST_GIT
                else "Docker image"
            )
            raise RuntimeError(f"{source} for {task.instance_id} did not expose a git repository.")

        self._run_process(
            ["git", "config", "user.email", "apex@example.com"], cwd=repo_dir, timeout=60
        )
        self._run_process(["git", "config", "user.name", "APEX"], cwd=repo_dir, timeout=60)

        if task.before_repo_set_cmd.strip():
            self._run_command(repo_dir, task.before_repo_set_cmd, timeout=600, check=True)
        self._restore_published_parity_benchmark_controlled_files(task, repo_dir)
        self._commit_prepared_baseline(repo_dir)

    def resolve_test_runner_adapter(
        self,
        task: SWEBenchProTask,
        repo_dir: Optional[Path] = None,
    ) -> Optional[Any]:
        """Per-task TestRunnerAdapter instance.

        Returns a SWEBenchProAdapter primed with this task's repo_language
        so the protocol-level helpers (stub_patterns, infrastructure_paths,
        extract_failure_excerpt, parse_report) target the right language.
        Falls back to whatever ``detect_adapter`` finds in the workspace
        when the SWE-Bench Pro adapter isn't registered (which can happen
        in tests that monkeypatch the registry).
        """
        try:
            from ..core.test_runners import detect_adapter, get_adapter
            from ..core.test_runners.swebench_pro_adapter import SWEBenchProAdapter
        except ImportError:
            return None
        language = (getattr(task, "repo_language", "") or "").strip()
        if get_adapter("swebench-pro") is not None:
            return SWEBenchProAdapter(repo_language=language or None)
        if repo_dir is not None and repo_dir.exists():
            return detect_adapter(repo_dir)
        return None

    def evaluate_repo(
        self,
        task: SWEBenchProTask,
        repo_dir: Path,
        *,
        artifacts_dir: str | Path | None = None,
    ) -> SWEBenchProEvaluation:
        ignored_changes = self._list_ignored_changes(repo_dir, task.benchmark_controlled_files)
        patch = self._build_filtered_patch(repo_dir, task.benchmark_controlled_files)
        evaluation = self.evaluate_patch(task, patch, artifacts_dir=artifacts_dir)
        evaluation.ignored_changes = ignored_changes
        if ignored_changes and "Ignored benchmark-controlled changes" not in evaluation.output:
            ignored_summary = ", ".join(ignored_changes[:10])
            prefix = f"Ignored benchmark-controlled changes: {ignored_summary}" + (
                " ..." if len(ignored_changes) > 10 else ""
            )
            evaluation.output = (prefix + "\n" + evaluation.output).strip()
        return evaluation

    def evaluate_patch(
        self,
        task: SWEBenchProTask,
        patch: str,
        *,
        artifacts_dir: str | Path | None = None,
    ) -> SWEBenchProEvaluation:
        created_temp = False
        if artifacts_dir is None:
            artifacts_dir = tempfile.mkdtemp(prefix=f"apex-swebench-eval-{task.repo_name}-")
            created_temp = True

        artifact_path = Path(artifacts_dir).resolve()
        artifact_path.mkdir(parents=True, exist_ok=True)
        start = time.time()
        image_uri = self.resolve_image_uri(task)
        self._ensure_image_available(image_uri)

        run_script_path, parser_path = self._ensure_task_scripts(task)
        cleaned_patch = _strip_binary_hunks(patch)
        selected_targets = ",".join(task.selected_test_files_to_run)

        entryscript = self._build_entryscript(task, selected_targets)
        workspace_files = {
            "patch.diff": cleaned_patch,
            "run_script.sh": run_script_path.read_text(),
            "parser.py": parser_path.read_text(),
            "entryscript.sh": entryscript,
        }
        for rel_name, content in workspace_files.items():
            (artifact_path / rel_name).write_text(content)
        subprocess.run(["chmod", "+x", str(artifact_path / "run_script.sh")], capture_output=True)
        subprocess.run(["chmod", "+x", str(artifact_path / "entryscript.sh")], capture_output=True)

        docker_run_output = self._run_container(image_uri, artifact_path)
        (artifact_path / "docker_run.log").write_text(docker_run_output.output)

        output_payload = self._load_output_json(artifact_path / "output.json")
        evaluation = self._build_evaluation(
            task=task,
            docker_result=docker_run_output,
            output_payload=output_payload,
            artifact_path=artifact_path,
            duration_seconds=time.time() - start,
        )

        if created_temp:
            shutil.rmtree(artifact_path, ignore_errors=True)
        return evaluation

    def format_for_test_command(self, evaluation: SWEBenchProEvaluation) -> str:
        lines: list[str] = []
        for status, tests in (
            ("PASSED", sorted(evaluation.passed_tests)),
            ("FAILED", sorted(evaluation.failed_tests)),
            ("ERROR", sorted(evaluation.error_tests)),
            ("SKIPPED", sorted(evaluation.skipped_tests)),
        ):
            for test_name in tests:
                lines.append(f"[TEST] {status} {test_name}")

        lines.append(
            "Summary: {passed} passed, {failed} failed, {errors} errors, {skipped} skipped".format(
                passed=len(evaluation.passed_tests),
                failed=len(evaluation.failed_tests),
                errors=len(evaluation.error_tests),
                skipped=len(evaluation.skipped_tests),
            )
        )
        lines.append(
            "Required tests: {passed}/{total} passed".format(
                passed=evaluation.total_required_tests - len(evaluation.missing_required_tests),
                total=evaluation.total_required_tests,
            )
        )
        if evaluation.ignored_changes:
            lines.append(
                "Ignored benchmark-controlled changes: "
                + ", ".join(evaluation.ignored_changes[:10])
                + (" ..." if len(evaluation.ignored_changes) > 10 else "")
            )
        if evaluation.missing_required_tests:
            lines.append(
                "Missing required tests: "
                + ", ".join(evaluation.missing_required_tests[:10])
                + (" ..." if len(evaluation.missing_required_tests) > 10 else "")
            )
        if evaluation.output:
            lines.append(evaluation.output)
        return "\n".join(lines).strip()

    def _detect_default_docker_platform(self) -> Optional[str]:
        machine = py_platform.machine().lower()
        if machine in {"arm64", "aarch64"}:
            return "linux/amd64"
        return None

    @property
    def agent_test_command_source(self) -> str:
        if self.agent_visibility_mode == SWEBENCH_AGENT_VISIBILITY_BENCHMARK_AWARE:
            return SWEBENCH_AGENT_TEST_COMMAND_OFFICIAL
        return SWEBENCH_AGENT_TEST_COMMAND_REPO_PUBLIC

    def _validate_agent_visibility_mode(self) -> None:
        if self.agent_visibility_mode not in {
            SWEBENCH_AGENT_VISIBILITY_PUBLISHED_PARITY,
            SWEBENCH_AGENT_VISIBILITY_BENCHMARK_AWARE,
        }:
            raise ValueError(
                f"Unsupported SWE-Bench Pro agent visibility mode: {self.agent_visibility_mode}"
            )

    def _validate_prepare_repo_mode(self) -> None:
        if self.prepare_repo_mode not in {
            SWEBENCH_PREPARE_REPO_FROM_DOCKER_IMAGE,
            SWEBENCH_PREPARE_REPO_FROM_HOST_GIT,
        }:
            raise ValueError(
                f"Unsupported SWE-Bench Pro repo preparation mode: {self.prepare_repo_mode}"
            )

    def _ordered_public_test_command_builders(
        self,
        repo_language: str,
    ) -> list[Any]:
        language = repo_language.lower().strip()
        language_aliases = {
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
        }
        language = language_aliases.get(language, language)
        common = [
            self._detect_python_test_command,
            self._detect_node_test_command,
            self._detect_maven_test_command,
            self._detect_gradle_test_command,
            self._detect_go_test_command,
            self._detect_rust_test_command,
            self._detect_dotnet_test_command,
        ]
        preferred: dict[str, list[Any]] = {
            "python": [
                self._detect_python_test_command,
                self._detect_node_test_command,
            ],
            "javascript": [
                self._detect_node_test_command,
                self._detect_python_test_command,
            ],
            "typescript": [
                self._detect_node_test_command,
                self._detect_python_test_command,
            ],
            "java": [
                self._detect_maven_test_command,
                self._detect_gradle_test_command,
            ],
            "go": [
                self._detect_go_test_command,
            ],
            "rust": [
                self._detect_rust_test_command,
            ],
            "c#": [
                self._detect_dotnet_test_command,
            ],
            "csharp": [
                self._detect_dotnet_test_command,
            ],
        }
        ordered = preferred.get(language, []) + common
        unique: list[Any] = []
        seen: set[str] = set()
        for builder in ordered:
            key = getattr(getattr(builder, "__func__", builder), "__name__", repr(builder))
            if key in seen:
                continue
            seen.add(key)
            unique.append(builder)
        return unique

    def _repo_contains_matching_file(
        self,
        repo_dir: Path,
        *,
        suffixes: tuple[str, ...],
        preferred_roots: tuple[str, ...] = (),
        excluded_dirs: tuple[str, ...] = (
            ".git",
            ".hg",
            ".jj",
            ".sl",
            ".svn",
            "node_modules",
            ".venv",
            "venv",
            "env",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".tox",
            "dist",
            "build",
            "coverage",
        ),
    ) -> bool:
        normalized_suffixes = tuple(suffix.lower() for suffix in suffixes if str(suffix).strip())
        if not normalized_suffixes:
            return False

        search_roots: list[Path] = []
        for rel_root in preferred_roots:
            candidate = repo_dir / rel_root
            if candidate.exists() and candidate.is_dir():
                search_roots.append(candidate)
        search_roots.append(repo_dir)

        seen_roots: set[Path] = set()
        for search_root in search_roots:
            resolved_root = search_root.resolve()
            if resolved_root in seen_roots:
                continue
            seen_roots.add(resolved_root)
            for root, dirs, files in os.walk(resolved_root):
                dirs[:] = [entry for entry in dirs if entry not in excluded_dirs]
                for name in files:
                    if name.lower().endswith(normalized_suffixes):
                        return True
        return False

    def _repo_has_python_files(self, repo_dir: Path) -> bool:
        return self._repo_contains_matching_file(
            repo_dir,
            suffixes=(".py",),
            preferred_roots=("tests", "test", "src"),
        )

    def _detect_python_test_command(self, repo_dir: Path) -> Optional[str]:
        if any(
            (repo_dir / marker).exists()
            for marker in (
                "pytest.ini",
                "tox.ini",
                "conftest.py",
                "manage.py",
                "setup.py",
            )
        ):
            return "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q --tb=no"
        if self._repo_has_python_files(repo_dir) and (
            (repo_dir / "tests").exists() or (repo_dir / "test").exists()
        ):
            return "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q --tb=no"
        pyproject = repo_dir / "pyproject.toml"
        if pyproject.exists():
            contents = pyproject.read_text(errors="replace")
            if self._repo_has_python_files(repo_dir) and (
                "pytest" in contents
                or "setuptools" in contents
                or "hatchling" in contents
                or "[project]" in contents
                or "[tool.poetry]" in contents
            ):
                return "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q --tb=no"
        return None

    def _detect_node_test_command(self, repo_dir: Path) -> Optional[str]:
        package_json = repo_dir / "package.json"
        if not package_json.exists():
            return None
        try:
            payload = json.loads(package_json.read_text())
        except json.JSONDecodeError:
            return None
        scripts = payload.get("scripts") or {}
        if not isinstance(scripts, dict):
            scripts = {}

        package_manager = self._detect_js_package_manager(repo_dir)
        for script_name in ("test", "test:unit", "test-unit", "test:ci", "ci:test"):
            script_value = scripts.get(script_name)
            if not isinstance(script_value, str):
                continue
            lowered = script_value.lower()
            if "no test specified" in lowered:
                continue
            if script_name == "test":
                return f"CI=1 {package_manager} test"
            return f"CI=1 {package_manager} run {shlex.quote(script_name)}"

        if scripts:
            return f"CI=1 {package_manager} test"
        return None

    def _detect_js_package_manager(self, repo_dir: Path) -> str:
        if (repo_dir / "pnpm-lock.yaml").exists() or (repo_dir / "pnpm-workspace.yaml").exists():
            return "pnpm"
        if (repo_dir / "yarn.lock").exists():
            return "yarn"
        return "npm"

    def _detect_go_test_command(self, repo_dir: Path) -> Optional[str]:
        if (repo_dir / "go.mod").exists():
            return "go test ./..."
        return None

    def _detect_rust_test_command(self, repo_dir: Path) -> Optional[str]:
        if (repo_dir / "Cargo.toml").exists():
            return "cargo test"
        return None

    def _detect_gradle_test_command(self, repo_dir: Path) -> Optional[str]:
        if (repo_dir / "gradlew").exists():
            return "./gradlew test"
        if (repo_dir / "build.gradle").exists() or (repo_dir / "build.gradle.kts").exists():
            return "gradle test"
        return None

    def _detect_maven_test_command(self, repo_dir: Path) -> Optional[str]:
        if (repo_dir / "mvnw").exists():
            return "./mvnw -q test"
        if (repo_dir / "pom.xml").exists():
            return "mvn -q test"
        return None

    def _detect_dotnet_test_command(self, repo_dir: Path) -> Optional[str]:
        if any(repo_dir.glob("*.sln")):
            return "dotnet test"
        try:
            next(repo_dir.rglob("*.csproj"))
        except StopIteration:
            return None
        return "dotnet test"

    def _ensure_image_available(self, image_uri: str) -> None:
        inspect = subprocess.run(
            ["docker", "image", "inspect", image_uri],
            capture_output=True,
            text=True,
        )
        if inspect.returncode == 0:
            return

        command = ["docker", "pull"]
        if self.docker_platform:
            command.extend(["--platform", self.docker_platform])
        command.append(image_uri)
        self._run_process(command, timeout=3600)

    def _create_container(self, image_uri: str) -> str:
        command = ["docker", "create"]
        if self.docker_platform:
            command.extend(["--platform", self.docker_platform])
        command.append(image_uri)
        result = run_process_command(command, timeout=300)
        if result.returncode != 0:
            raise RuntimeError((result.stdout + result.stderr).strip() or "docker create failed")
        container_id = result.stdout.strip()
        if not container_id:
            raise RuntimeError("docker create did not return a container id")
        return container_id

    def _commit_prepared_baseline(self, repo_dir: Path) -> None:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=str(repo_dir),
        )
        if not status.stdout.strip():
            return
        self._run_process(["git", "add", "-A"], cwd=repo_dir, timeout=300)
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(repo_dir),
        )
        if diff.returncode != 0:
            self._run_process(
                ["git", "commit", "-m", "apex benchmark prepared baseline"],
                cwd=repo_dir,
                timeout=300,
            )

    def _build_entryscript(self, task: SWEBenchProTask, selected_targets: str) -> str:
        quoted_targets = shlex.quote(selected_targets) if selected_targets else ""
        run_script_command = "bash /workspace/run_script.sh"
        if quoted_targets:
            run_script_command += f" {quoted_targets}"

        benchmark_prep_commands = _extract_benchmark_prep_commands(task.before_repo_set_cmd)
        lines = [
            "#!/bin/bash",
            "set -e",
            "cd /app",
            f"git reset --hard {shlex.quote(task.base_commit)}",
            "git clean -fd",
            f"git checkout {shlex.quote(task.base_commit)}",
        ]
        lines.extend(benchmark_prep_commands)
        lines.extend(
            [
                "if [ -s /workspace/patch.diff ]; then git apply -v /workspace/patch.diff; fi",
                f"{run_script_command} > /workspace/stdout.log 2> /workspace/stderr.log",
                "python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json",
            ]
        )
        return "\n".join(lines) + "\n"

    def _ensure_task_scripts(self, task: SWEBenchProTask) -> tuple[Path, Path]:
        last_error: Optional[RuntimeError] = None
        for script_dir_name in self._candidate_script_dirs(task.instance_id):
            task_dir = self.scripts_cache_dir / script_dir_name
            task_dir.mkdir(parents=True, exist_ok=True)
            run_script_path = task_dir / "run_script.sh"
            parser_path = task_dir / "parser.py"
            created_paths: list[Path] = []
            try:
                if not run_script_path.exists():
                    run_script_path.write_text(
                        self._fetch_remote_text(
                            f"{SWEBENCH_PRO_SCRIPTS_BASE_URL}/{script_dir_name}/run_script.sh"
                        )
                    )
                    created_paths.append(run_script_path)
                if not parser_path.exists():
                    parser_path.write_text(
                        self._fetch_remote_text(
                            f"{SWEBENCH_PRO_SCRIPTS_BASE_URL}/{script_dir_name}/parser.py"
                        )
                    )
                    created_paths.append(parser_path)
                return run_script_path, parser_path
            except RuntimeError as exc:
                last_error = exc
                for path in created_paths:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                try:
                    next(task_dir.iterdir())
                except StopIteration:
                    task_dir.rmdir()

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Unable to resolve benchmark scripts for {task.instance_id}")

    def _candidate_script_dirs(self, instance_id: str) -> list[str]:
        manifest = self._load_scripts_manifest()
        stripped_instance_id = _strip_instance_version_suffix(instance_id)
        candidates: list[str] = []

        if manifest and instance_id in manifest:
            return [instance_id]
        if manifest and stripped_instance_id != instance_id and stripped_instance_id in manifest:
            candidates.append(stripped_instance_id)

        candidates.append(instance_id)
        if stripped_instance_id != instance_id:
            candidates.append(stripped_instance_id)

        unique_candidates: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            unique_candidates.append(candidate)
        return unique_candidates

    def _load_scripts_manifest(self) -> Optional[set[str]]:
        if self._scripts_manifest_cache is not None:
            return self._scripts_manifest_cache

        manifest_path = self.scripts_cache_dir / "_scripts_manifest.json"
        if manifest_path.exists():
            try:
                manifest_payload = json.loads(manifest_path.read_text())
                if isinstance(manifest_payload, list):
                    self._scripts_manifest_cache = {str(item) for item in manifest_payload}
                    return self._scripts_manifest_cache
            except json.JSONDecodeError:
                pass

        request = Request(
            SWEBENCH_PRO_SCRIPTS_TREE_URL,
            headers={
                "User-Agent": "apex-swebench-pro-benchmark/1.0",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.load(response)
        except (HTTPError, URLError, json.JSONDecodeError):
            return None

        entries = sorted(
            {
                str(item["path"]).split("/", 2)[1]
                for item in payload.get("tree", [])
                if item.get("type") == "blob"
                and str(item.get("path", "")).startswith("run_scripts/")
                and str(item["path"]).count("/") >= 2
            }
        )
        if not entries:
            return None

        manifest_path.write_text(json.dumps(entries, indent=2))
        self._scripts_manifest_cache = set(entries)
        return self._scripts_manifest_cache

    def _fetch_remote_text(self, url: str) -> str:
        request = Request(
            url,
            headers={
                "User-Agent": "apex-swebench-pro-benchmark/1.0",
                "Accept": "text/plain",
            },
        )
        try:
            with urlopen(request, timeout=60) as response:
                return response.read().decode("utf-8")
        except HTTPError as exc:
            raise RuntimeError(
                f"Failed to download benchmark asset {url}: HTTP {exc.code}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"Failed to download benchmark asset {url}: {exc.reason}") from exc

    def _run_container(self, image_uri: str, workspace_dir: Path) -> "_CommandResult":
        command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{workspace_dir}:/workspace",
            "--entrypoint",
            "/bin/bash",
        ]
        if self.block_network:
            command.extend(["--network", "none"])
        if self.docker_platform:
            command.extend(["--platform", self.docker_platform])
        command.extend(
            [
                image_uri,
                "-c",
                "bash /workspace/entryscript.sh",
            ]
        )
        result = run_process_command(command, timeout=3600)
        output = (result.stdout + result.stderr).strip()
        return _CommandResult(returncode=result.returncode, output=output)

    def _load_output_json(self, path: Path) -> Optional[dict[str, Any]]:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None

    def _cleanup_task_docker_artifacts(self, task: SWEBenchProTask) -> None:
        if shutil.which("docker") is None:
            return

        try:
            image_uri = self.resolve_image_uri(task)
        except Exception as exc:
            logger.debug(
                "Skipping Docker cleanup for SWE-Bench Pro task %s: %s",
                task.instance_id,
                exc,
            )
            return

        container_ids = self._list_task_docker_container_ids(image_uri)
        for container_id in container_ids:
            self._run_best_effort_docker_cleanup(
                ["docker", "rm", "-f", "-v", container_id],
                task=task,
                artifact_kind="container",
                artifact_id=container_id,
                missing_markers=("No such container",),
            )

        self._run_best_effort_docker_cleanup(
            ["docker", "image", "rm", "-f", image_uri],
            task=task,
            artifact_kind="image",
            artifact_id=image_uri,
            missing_markers=("No such image",),
        )

    def _list_task_docker_container_ids(self, image_uri: str) -> list[str]:
        result = run_process_command(
            ["docker", "ps", "-aq", "--filter", f"ancestor={image_uri}"],
            timeout=120,
        )
        if result.returncode != 0:
            output = (result.stdout + result.stderr).strip()
            logger.warning(
                "Failed to enumerate Docker containers for SWE-Bench Pro image %s: %s",
                image_uri,
                output or f"exit {result.returncode}",
            )
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def _run_best_effort_docker_cleanup(
        self,
        command: list[str],
        *,
        task: SWEBenchProTask,
        artifact_kind: str,
        artifact_id: str,
        missing_markers: tuple[str, ...] = (),
    ) -> bool:
        result = run_process_command(command, timeout=300)
        if result.returncode == 0:
            return True

        output = (result.stdout + result.stderr).strip()
        lowered_output = output.lower()
        if any(marker.lower() in lowered_output for marker in missing_markers):
            return False

        logger.warning(
            "Best-effort Docker cleanup failed for SWE-Bench Pro task %s %s %s: %s",
            task.instance_id,
            artifact_kind,
            artifact_id,
            output or f"exit {result.returncode}",
        )
        return False

    def _build_evaluation(
        self,
        *,
        task: SWEBenchProTask,
        docker_result: "_CommandResult",
        output_payload: Optional[dict[str, Any]],
        artifact_path: Path,
        duration_seconds: float,
    ) -> SWEBenchProEvaluation:
        passed_tests: set[str] = set()
        failed_tests: set[str] = set()
        skipped_tests: set[str] = set()
        error_tests: set[str] = set()

        tests = list((output_payload or {}).get("tests") or [])
        for test in tests:
            name = str(test.get("name", "")).strip()
            status = str(test.get("status", "")).strip().upper()
            if not name:
                continue
            if status == "PASSED":
                passed_tests.add(name)
            elif status == "FAILED":
                failed_tests.add(name)
            elif status == "ERROR":
                error_tests.add(name)
            else:
                skipped_tests.add(name)

        required_tests = set(task.required_tests)
        summary_lines = [
            "Docker evaluation return code: {}".format(docker_result.returncode),
            "Required tests passed: {}/{}".format(
                len(required_tests & passed_tests),
                len(required_tests),
            ),
        ]
        missing_required = sorted(required_tests - passed_tests)
        if missing_required:
            summary_lines.append(
                "Missing required tests: "
                + ", ".join(missing_required[:10])
                + (" ..." if len(missing_required) > 10 else "")
            )
        if docker_result.output:
            summary_lines.append(docker_result.output.strip())

        evaluation = SWEBenchProEvaluation(
            returncode=0 if not missing_required else (docker_result.returncode or 1),
            output="\n".join(part for part in summary_lines if part).strip(),
            docker_returncode=docker_result.returncode,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            error_tests=error_tests,
            required_tests=required_tests,
            selected_test_targets=list(task.selected_test_files_to_run),
            stdout_log_path=str(artifact_path / "stdout.log"),
            stderr_log_path=str(artifact_path / "stderr.log"),
            output_json_path=str(artifact_path / "output.json"),
            docker_output_path=str(artifact_path / "docker_run.log"),
            duration_seconds=duration_seconds,
        )
        return evaluation

    def _build_filtered_patch(self, repo_dir: Path, disallowed_paths: list[str]) -> str:
        subprocess.run(["git", "add", "-N", "."], capture_output=True, cwd=str(repo_dir))
        command = ["git", "diff", "--binary", "--relative", "HEAD", "--", "."]
        for disallowed_path in sorted(set(disallowed_paths)):
            command.append(f":(exclude){disallowed_path}")
        for pattern in ignored_change_pathspecs():
            command.append(pattern)
        result = run_process_command(command, cwd=repo_dir, timeout=300)
        if result.returncode != 0:
            raise RuntimeError((result.stdout + result.stderr).strip() or "git diff failed")
        return result.stdout

    def _list_ignored_changes(self, repo_dir: Path, disallowed_paths: list[str]) -> list[str]:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=str(repo_dir),
        )
        raw_paths = [parse_porcelain_path(line) for line in status.stdout.splitlines()]
        changed_paths = expand_changed_paths(
            repo_dir,
            raw_paths,
            ignored_predicate=_is_ignored_repo_artifact,
        )
        return sorted(
            {
                rel_path
                for rel_path in changed_paths
                if _path_is_excluded(rel_path, disallowed_paths)
            }
        )

    def _run_process(
        self,
        command: list[str],
        *,
        cwd: Optional[Path] = None,
        timeout: int = 300,
    ) -> None:
        result = run_process_command(command, cwd=cwd, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                (result.stdout + result.stderr).strip() or f"Command failed: {command}"
            )

    def _run_command(
        self,
        cwd: Path,
        command: str,
        *,
        timeout: int = 300,
        check: bool = False,
    ) -> "_CommandResult":
        result = run_shell_command(command, cwd, timeout=timeout)
        output = (result.stdout + result.stderr).strip()
        if check and result.returncode != 0:
            raise RuntimeError(output or f"Command failed: {command}")
        return _CommandResult(returncode=result.returncode, output=output)


class SWEBenchProBenchmarkRunner(SWEBenchProHarness):
    """Run APEX against the SWE-Bench Pro public benchmark."""

    def __init__(
        self,
        config: ApexConfig,
        output_dir: str,
        *,
        dataset_name: str = SWEBENCH_PRO_DATASET_NAME,
        dataset_split: str = SWEBENCH_PRO_DATASET_SPLIT,
        dockerhub_username: str = SWEBENCH_PRO_DOCKERHUB_USERNAME,
        scripts_cache_dir: str | Path | None = None,
        docker_platform: Optional[str] = None,
        block_network: bool = False,
        agent_visibility_mode: str = SWEBENCH_AGENT_VISIBILITY_PUBLISHED_PARITY,
        rollout_selection_policy: str = SWEBENCH_ROLLOUT_SELECTION_ORCHESTRATOR,
        prepare_repo_mode: str = SWEBENCH_PREPARE_REPO_FROM_DOCKER_IMAGE,
    ) -> None:
        super().__init__(
            output_dir=output_dir,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            dockerhub_username=dockerhub_username,
            scripts_cache_dir=scripts_cache_dir,
            docker_platform=docker_platform,
            block_network=block_network,
            agent_visibility_mode=agent_visibility_mode,
            prepare_repo_mode=prepare_repo_mode,
        )
        self.config = copy.deepcopy(config)
        self.config_source: Optional[str] = None
        self.rollout_selection_policy = rollout_selection_policy
        if self.rollout_selection_policy not in {
            SWEBENCH_ROLLOUT_SELECTION_ORCHESTRATOR,
            SWEBENCH_ROLLOUT_SELECTION_OFFICIAL_EVALUATOR,
        }:
            raise ValueError(
                "Unsupported SWE-Bench Pro rollout selection policy: "
                f"{self.rollout_selection_policy}"
            )
        if (
            self.agent_test_command_source == SWEBENCH_AGENT_TEST_COMMAND_REPO_PUBLIC
            and self.config.rollout.enable_quick_verification
        ):
            logger.info(
                "Disabling rollout quick verification for SWE-Bench Pro published-parity "
                "runs because the host repo-public pytest command is not a reliable proxy "
                "for the official Docker evaluation environment."
            )
            self.config.rollout.enable_quick_verification = False

    def run(
        self,
        *,
        instances: Optional[list[str]] = None,
        repos: Optional[list[str]] = None,
        languages: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> SWEBenchProBenchmarkReport:
        ensure_cli_process_cleanup_hooks()
        execution = {
            "entrypoint": "swebench-pro-benchmark",
            "args": {
                "instances": list(instances or []),
                "repos": list(repos or []),
                "languages": list(languages or []),
                "limit": limit,
                "dataset_name": self.dataset_name,
                "dataset_split": self.dataset_split,
                "dockerhub_username": self.dockerhub_username,
                "scripts_cache_dir": str(self.scripts_cache_dir),
                "docker_platform": self.docker_platform,
                "block_network": self.block_network,
                "agent_visibility_mode": self.agent_visibility_mode,
                "rollout_selection_policy": self.rollout_selection_policy,
                "prepare_repo_mode": self.prepare_repo_mode,
            },
        }
        tasks = self.discover_tasks(
            instances=instances,
            repos=repos,
            languages=languages,
            limit=limit,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        requested_task_ids = [task.instance_id for task in tasks]
        existing_state = load_json_if_exists(self.output_dir / RUN_STATE_FILENAME) or {}
        report = SWEBenchProBenchmarkReport(
            requested_task_ids=requested_task_ids,
            requested_repo_names=sorted({task.repo for task in tasks}),
            started_at=float(existing_state.get("started_at") or time.time()),
            dataset_name=self.dataset_name,
            dataset_split=self.dataset_split,
            config_source=self.config_source,
            model_config=serialize_llm_configs(self.config),
            ablation_config=build_apex_ablation_config(self.config),
            agent_visibility_mode=self.agent_visibility_mode,
            agent_test_command_source=self.agent_test_command_source,
            official_evaluator_scope=(
                "baseline_rollout_selection_and_final"
                if self.rollout_selection_policy == SWEBENCH_ROLLOUT_SELECTION_OFFICIAL_EVALUATOR
                else "baseline_and_final_only"
            ),
            benchmark_clean=(
                self.agent_visibility_mode == SWEBENCH_AGENT_VISIBILITY_PUBLISHED_PARITY
                and self.rollout_selection_policy == SWEBENCH_ROLLOUT_SELECTION_ORCHESTRATOR
            ),
        )
        report.run_manifest = ensure_run_manifest(
            self.output_dir,
            build_run_manifest(
                config=self.config,
                report_kind=report.report_kind,
                harness_name=report.harness_name,
                harness_version=report.harness_version,
                benchmark_family="swebench_pro",
                output_dir=self.output_dir,
                config_source=self.config_source,
                requested_task_ids=requested_task_ids,
                execution=execution,
                extra_settings={
                    "dataset_name": self.dataset_name,
                    "dataset_split": self.dataset_split,
                    "agent_visibility_mode": self.agent_visibility_mode,
                    "official_evaluator_scope": report.official_evaluator_scope,
                },
                benchmark_policy=_build_swebench_pro_benchmark_policy(
                    block_network=self.block_network,
                    docker_platform=self.docker_platform,
                    agent_visibility_mode=self.agent_visibility_mode,
                    agent_test_command_source=self.agent_test_command_source,
                    rollout_selection_policy=self.rollout_selection_policy,
                ),
            ),
        )

        for task in tasks:
            checkpointed = self._load_checkpointed_task_result(task)
            if checkpointed is not None:
                report.tasks.append(checkpointed)
                continue

        self._write_report_checkpoint(report, requested_task_ids, completed=False)

        for task in tasks:
            checkpointed = self._load_checkpointed_task_result(task)
            if checkpointed is not None:
                continue
            ensure_clean_directory_for_task(self._task_output_dir(task), completed=False)
            ensure_clean_directory_for_task(self._task_workspace_dir(task), completed=False)

            result = self._run_task(task)
            task_payload = result.to_dict()
            if self.agent_visibility_mode == SWEBENCH_AGENT_VISIBILITY_PUBLISHED_PARITY:
                task_payload = _artifact_safe_swebench_payload(
                    task_payload,
                    include_benchmark_metadata=False,
                )
            write_task_checkpoint(self._task_output_dir(task), task_payload)
            report.tasks.append(result)
            self._write_report_checkpoint(report, requested_task_ids, completed=False)

        self._write_report_checkpoint(report, requested_task_ids, completed=True)
        return report

    def _task_output_dir(self, task: SWEBenchProTask) -> Path:
        return self.output_dir / task.instance_id

    def _task_workspace_dir(self, task: SWEBenchProTask) -> Path:
        return self.output_dir / "workspaces" / task.instance_id

    def _load_checkpointed_task_result(
        self,
        task: SWEBenchProTask,
    ) -> Optional[SWEBenchProTaskResult]:
        checkpoint_path = task_result_path(self._task_output_dir(task))
        payload = load_json_if_exists(checkpoint_path)
        if payload is None:
            return None
        try:
            return SWEBenchProTaskResult.from_dict(payload)
        except Exception as exc:
            logger.warning(
                "Ignoring corrupt SWE-Bench Pro checkpoint for %s at %s: %s",
                task.instance_id,
                checkpoint_path,
                exc,
            )
            return None

    def _write_report_checkpoint(
        self,
        report: SWEBenchProBenchmarkReport,
        requested_task_ids: list[str],
        *,
        completed: bool,
    ) -> None:
        report.updated_at = time.time()
        report.finished_at = report.updated_at if completed else 0.0
        update_run_manifest(
            self.output_dir,
            requested_task_ids=requested_task_ids,
            completed_task_ids=[task.instance_id for task in report.tasks],
            completed=completed,
            extra_updates={
                "config_payload": self.config.to_dict(),
                "environment_snapshot": capture_environment_snapshot(self.config),
                "prompt_template_fingerprints": build_prompt_template_fingerprints(),
                "dataset_name": report.dataset_name,
                "dataset_split": report.dataset_split,
                "agent_visibility_mode": report.agent_visibility_mode,
                "official_evaluator_scope": report.official_evaluator_scope,
                "execution": {
                    "entrypoint": "swebench-pro-benchmark",
                    "args": {
                        "instances": list(report.requested_task_ids),
                        "repos": list(report.requested_repo_names),
                        "languages": [],
                        "limit": None,
                        "dataset_name": report.dataset_name,
                        "dataset_split": report.dataset_split,
                        "dockerhub_username": self.dockerhub_username,
                        "scripts_cache_dir": str(self.scripts_cache_dir),
                        "docker_platform": self.docker_platform,
                        "block_network": self.block_network,
                        "agent_visibility_mode": report.agent_visibility_mode,
                        "rollout_selection_policy": self.rollout_selection_policy,
                    },
                },
            },
        )
        report.run_manifest = load_run_manifest(self.output_dir) or report.run_manifest
        report_payload = report.to_dict()
        if report.agent_visibility_mode == SWEBENCH_AGENT_VISIBILITY_PUBLISHED_PARITY:
            report_payload = _artifact_safe_swebench_payload(
                report_payload,
                include_benchmark_metadata=False,
            )
        atomic_write_json(self.output_dir / "benchmark_report.json", report_payload)
        atomic_write_text(self.output_dir / "benchmark_report.md", report.to_markdown())
        atomic_write_json(
            self.output_dir / RUN_STATE_FILENAME,
            build_run_state(
                report_kind=report.report_kind,
                harness_name=report.harness_name,
                harness_version=report.harness_version,
                started_at=report.started_at,
                requested_task_ids=requested_task_ids,
                completed_task_ids=[task.instance_id for task in report.tasks],
                successful_tasks=sum(1 for task in report.tasks if task.success),
                failed_tasks=sum(1 for task in report.tasks if not task.success),
                completed=completed,
                metadata={
                    "config_source": report.config_source,
                    "dataset_name": report.dataset_name,
                    "dataset_split": report.dataset_split,
                    "agent_visibility_mode": report.agent_visibility_mode,
                    "agent_test_command_source": report.agent_test_command_source,
                    "official_evaluator_scope": report.official_evaluator_scope,
                    "benchmark_clean": report.benchmark_clean,
                    "model_config": copy.deepcopy(report.model_config),
                    "ablation_config": copy.deepcopy(report.ablation_config),
                },
            ),
        )

    def _build_task_config(
        self,
        task_output_dir: Path,
        task_workspace_dir: Path,
    ) -> ApexConfig:
        config = copy.deepcopy(self.config)
        config.output_dir = str(task_output_dir)
        config.workspace_dir = str(task_workspace_dir)

        if config.rollout.num_rollouts <= 1 and config.planning.enable_manager_planner:
            logger.info(
                "Disabling LLM manager planner for single-rollout SWE-Bench Pro task; "
                "using heuristic planning instead."
            )
            config.planning.enable_manager_planner = False

        return config

    def _run_task(self, task: SWEBenchProTask) -> SWEBenchProTaskResult:
        started = time.time()
        sandbox = Path(tempfile.mkdtemp(prefix=f"apex-swebench-{task.repo_name}-"))
        repo_dir = sandbox / "repo"
        task_output_dir = self.output_dir / task.instance_id
        task_workspace_dir = self.output_dir / "workspaces" / task.instance_id
        task_output_dir.mkdir(parents=True, exist_ok=True)
        task_workspace_dir.mkdir(parents=True, exist_ok=True)
        task_file: Optional[Path] = None
        baseline: Optional[SWEBenchProEvaluation] = None
        orchestrator_reached = False
        include_benchmark_metadata = (
            self.agent_visibility_mode == SWEBENCH_AGENT_VISIBILITY_BENCHMARK_AWARE
        )

        try:
            write_task_live_state(
                task_output_dir,
                {
                    "task_id": task.instance_id,
                    "phase": "prepare_repo",
                    "status": "active",
                    "process_pid": os.getpid(),
                    "last_progress_at": time.time(),
                },
            )
            self.prepare_repo(task, repo_dir)
            write_task_live_state(
                task_output_dir,
                {
                    "task_id": task.instance_id,
                    "phase": "baseline_eval",
                    "status": "active",
                    "process_pid": os.getpid(),
                    "last_progress_at": time.time(),
                },
            )
            baseline = self.evaluate_repo(
                task,
                repo_dir,
                artifacts_dir=task_output_dir / "baseline_eval",
            )
            _scrub_swebench_evaluation_artifacts(
                task_output_dir / "baseline_eval",
                include_benchmark_metadata=include_benchmark_metadata,
            )

            config = self._build_task_config(task_output_dir, task_workspace_dir)
            keep_worktrees = config.rollout.keep_worktrees
            config.rollout.keep_worktrees = True
            target_tool_env, target_tool_diagnostics = target_tool_env_overrides(
                workdir=repo_dir,
                output_dir=task_output_dir / "target_runtime_tools",
                timeout_seconds=max(
                    1, int(self.config.selection.verification_timeout_seconds or 600)
                ),
                runtime=docker_image_runtime(
                    image=self.resolve_image_uri(task),
                    docker_workdir="/app",
                    docker_platform=self.docker_platform or "",
                    description="swebench_pro_official_docker_image",
                ),
                label=f"swebench_pro_{task.instance_id}",
            )
            apply_target_tool_env_to_apex_config(config, target_tool_env)
            atomic_write_json(
                task_output_dir / "target_runtime_tools.json",
                target_tool_diagnostics,
            )

            if include_benchmark_metadata:
                task_file = self.write_task_metadata(
                    task,
                    task_output_dir / "swebench_pro_task.json",
                    include_benchmark_metadata=True,
                )
                test_command = self.build_official_evaluator_command(task_file)
            else:
                test_command = self.build_agent_test_command(task, repo_dir)
            verification_test_command = (
                None
                if self.agent_test_command_source == SWEBENCH_AGENT_TEST_COMMAND_REPO_PUBLIC
                else test_command
            )
            orchestrator = ApexOrchestrator(config)
            write_task_live_state(
                task_output_dir,
                {
                    "task_id": task.instance_id,
                    "phase": "solving",
                    "status": "active",
                    "process_pid": os.getpid(),
                    "last_progress_at": time.time(),
                },
            )
            orchestrator_reached = True
            # Phase A.1 (Decisive-Edge): when V5 is the configured agent
            # surface, the orchestrator routes through
            # ``solve_in_container_agent`` and the per-task SWE-Bench Pro
            # docker image (``<dockerhub_username>/sweap-images:<tag>``)
            # is the ContainerSupervisor target.
            v5_benchmark_metadata: dict[str, Any] = {
                "benchmark_name": "swebench_pro",
                "instance_id": task.instance_id,
                "evidence_mode": "partial_suite_visible",
                "test_suite_evidence_mode": "partial_suite_visible",
            }
            try:
                v5_benchmark_metadata["docker_image"] = self.resolve_image_uri(task)
            except Exception:
                pass
            result = orchestrator.solve(
                repo_path=str(repo_dir),
                issue_description=task.build_issue_description(
                    test_command,
                    include_benchmark_guardrails=include_benchmark_metadata,
                    include_benchmark_metadata=include_benchmark_metadata,
                    include_selected_test_targets=include_benchmark_metadata,
                    include_required_tests=include_benchmark_metadata,
                ),
                test_command=test_command,
                verification_test_command=verification_test_command,
                benchmark_metadata=v5_benchmark_metadata,
            )

            selected_rollout_id = result.selected_rollout_id
            selected_worktree_path = result.selected_worktree_path
            failure_reason = None
            final = SWEBenchProEvaluation(
                returncode=1,
                output=result.explanation or "No rollout selected.",
                required_tests=set(task.required_tests),
                selected_test_targets=list(task.selected_test_files_to_run),
            )

            candidate = None
            if self.rollout_selection_policy == SWEBENCH_ROLLOUT_SELECTION_OFFICIAL_EVALUATOR:
                candidate = self._select_best_rollout_candidate(
                    task,
                    task_workspace_dir,
                    task_output_dir,
                )
            if candidate is not None:
                write_task_live_state(
                    task_output_dir,
                    {
                        "task_id": task.instance_id,
                        "phase": "final_eval",
                        "status": "active",
                        "process_pid": os.getpid(),
                        "last_progress_at": time.time(),
                    },
                )
                final = candidate.evaluation
                selected_rollout_id = candidate.rollout_id
                selected_worktree_path = str(candidate.worktree_path)
                if not final.all_required_tests_passed:
                    failure_reason = final.output or result.explanation
            elif result.selected_worktree_path:
                write_task_live_state(
                    task_output_dir,
                    {
                        "task_id": task.instance_id,
                        "phase": "final_eval",
                        "status": "active",
                        "process_pid": os.getpid(),
                        "last_progress_at": time.time(),
                    },
                )
                final = self.evaluate_repo(
                    task,
                    Path(result.selected_worktree_path),
                    artifacts_dir=task_output_dir / "selected_rollout_eval",
                )
                _scrub_swebench_evaluation_artifacts(
                    task_output_dir / "selected_rollout_eval",
                    include_benchmark_metadata=include_benchmark_metadata,
                )
                if not final.all_required_tests_passed:
                    failure_reason = final.output or result.explanation
            else:
                failure_reason = result.explanation or "No worktree selected."

            (task_output_dir / "baseline_metrics.json").write_text(
                json.dumps(
                    _artifact_safe_swebench_payload(
                        baseline.to_dict(),
                        include_benchmark_metadata=include_benchmark_metadata,
                    ),
                    indent=2,
                )
            )
            if task_file is None:
                task_file = self.write_task_metadata(
                    task,
                    task_output_dir / "swebench_pro_task.json",
                    include_benchmark_metadata=include_benchmark_metadata,
                )
            (task_output_dir / "final_metrics.json").write_text(
                json.dumps(
                    _artifact_safe_swebench_payload(
                        final.to_dict(),
                        include_benchmark_metadata=include_benchmark_metadata,
                    ),
                    indent=2,
                )
            )

            reported_orchestrator_selected_worktree_path = result.selected_worktree_path
            reported_selected_worktree_path = selected_worktree_path
            if not keep_worktrees:
                shutil.rmtree(task_workspace_dir, ignore_errors=True)
                reported_orchestrator_selected_worktree_path = None
                reported_selected_worktree_path = None
            else:
                if (
                    reported_orchestrator_selected_worktree_path
                    and not Path(reported_orchestrator_selected_worktree_path).exists()
                ):
                    reported_orchestrator_selected_worktree_path = None
                if (
                    reported_selected_worktree_path
                    and not Path(reported_selected_worktree_path).exists()
                ):
                    reported_selected_worktree_path = None

            task_result = SWEBenchProTaskResult(
                task_name=task.instance_id,
                instance_id=task.instance_id,
                repo=task.repo,
                success=final.all_required_tests_passed,
                baseline_failed=not baseline.all_required_tests_passed,
                final_tests_passed=final.all_required_tests_passed,
                baseline=baseline,
                final=final,
                orchestrator_success=bool(result.success),
                candidate_found=candidate is not None,
                orchestrator_selected_rollout_id=result.selected_rollout_id,
                orchestrator_selected_worktree_path=reported_orchestrator_selected_worktree_path,
                selected_rollout_id=selected_rollout_id,
                selected_worktree_path=reported_selected_worktree_path,
                total_tokens=result.total_tokens,
                duration_seconds=time.time() - started,
                result_path=str(task_result_path(task_output_dir)),
                failure_reason=failure_reason,
                execution_metadata=extract_apex_execution_metadata(result),
            )
            write_task_live_state(
                task_output_dir,
                {
                    "task_id": task.instance_id,
                    "phase": "completed",
                    "status": "completed",
                    "process_pid": os.getpid(),
                    "last_progress_at": time.time(),
                    "success": task_result.success,
                    "final_tests_passed": task_result.final_tests_passed,
                },
            )
            append_benchmark_task_outcome_trace(
                self.config,
                output_dir=task_output_dir,
                benchmark_name="swebench_pro",
                task_id=task.instance_id,
                task_success=task_result.success,
                orchestrator_reached=orchestrator_reached,
                orchestrator_success=task_result.orchestrator_success,
                baseline_failed=task_result.baseline_failed,
                baseline_pass_rate=task_result.baseline.required_pass_rate,
                final_pass_rate=task_result.final.required_pass_rate,
                candidate_found=task_result.candidate_found,
                selected_rollout_id=task_result.selected_rollout_id,
                skipped=task_result.skipped,
                skip_category=task_result.skip_category,
                duration_seconds=task_result.duration_seconds,
            )
            return task_result
        except Exception as exc:
            output = str(exc)
            include_benchmark_metadata = (
                self.agent_visibility_mode == SWEBENCH_AGENT_VISIBILITY_BENCHMARK_AWARE
            )
            if baseline is not None:
                (task_output_dir / "baseline_metrics.json").write_text(
                    json.dumps(
                        _artifact_safe_swebench_payload(
                            baseline.to_dict(),
                            include_benchmark_metadata=include_benchmark_metadata,
                        ),
                        indent=2,
                    )
                )
            if task_file is None:
                self.write_task_metadata(
                    task,
                    task_output_dir / "swebench_pro_task.json",
                    include_benchmark_metadata=include_benchmark_metadata,
                )
            failure = SWEBenchProEvaluation(
                returncode=1,
                output=output,
                required_tests=set(task.required_tests),
                selected_test_targets=list(task.selected_test_files_to_run),
            )
            skipped, skip_category = self._classify_prepare_error(exc)
            (task_output_dir / "prepare_error.txt").write_text(output)
            task_result = SWEBenchProTaskResult(
                task_name=task.instance_id,
                instance_id=task.instance_id,
                repo=task.repo,
                success=False,
                baseline_failed=False,
                final_tests_passed=False,
                baseline=failure,
                final=failure,
                duration_seconds=time.time() - started,
                result_path=str(task_result_path(task_output_dir)),
                failure_reason=output,
                skipped=skipped,
                skip_category=skip_category,
            )
            write_task_live_state(
                task_output_dir,
                {
                    "task_id": task.instance_id,
                    "phase": "completed",
                    "status": "error",
                    "process_pid": os.getpid(),
                    "last_progress_at": time.time(),
                    "success": False,
                    "failure_reason": output,
                    "skipped": skipped,
                    "skip_category": skip_category,
                },
            )
            append_benchmark_task_outcome_trace(
                self.config,
                output_dir=task_output_dir,
                benchmark_name="swebench_pro",
                task_id=task.instance_id,
                task_success=task_result.success,
                orchestrator_reached=orchestrator_reached,
                orchestrator_success=False,
                baseline_failed=task_result.baseline_failed,
                baseline_pass_rate=task_result.baseline.required_pass_rate,
                final_pass_rate=task_result.final.required_pass_rate,
                candidate_found=task_result.candidate_found,
                selected_rollout_id=task_result.selected_rollout_id,
                skipped=task_result.skipped,
                skip_category=task_result.skip_category,
                duration_seconds=task_result.duration_seconds,
            )
            return task_result
        finally:
            self._cleanup_task_docker_artifacts(task)
            shutil.rmtree(sandbox, ignore_errors=True)

    def _classify_prepare_error(self, exc: Exception) -> tuple[bool, Optional[str]]:
        message = str(exc).lower()
        if "docker" in message and ("not found" in message or "cannot connect" in message):
            return True, "unsupported_host"
        return False, None

    def _select_best_rollout_candidate(
        self,
        task: SWEBenchProTask,
        workspace_dir: Path,
        task_output_dir: Path,
    ) -> Optional[_CandidateEvaluation]:
        candidates: list[_CandidateEvaluation] = []
        for worktree_path in sorted(workspace_dir.glob("rollout_*")):
            if not worktree_path.is_dir():
                continue
            rollout_id = _rollout_id_from_name(worktree_path.name)
            if rollout_id is None:
                continue
            artifacts_dir = task_output_dir / "rollout_scores" / worktree_path.name
            evaluation = self.evaluate_repo(
                task,
                worktree_path,
                artifacts_dir=artifacts_dir,
            )
            _scrub_swebench_evaluation_artifacts(
                artifacts_dir,
                include_benchmark_metadata=(
                    self.agent_visibility_mode == SWEBENCH_AGENT_VISIBILITY_BENCHMARK_AWARE
                ),
            )
            candidates.append(
                _CandidateEvaluation(
                    rollout_id=rollout_id,
                    worktree_path=worktree_path,
                    evaluation=evaluation,
                )
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                item.evaluation.all_required_tests_passed,
                item.evaluation.required_pass_rate,
                -len(item.evaluation.ignored_changes),
                -item.rollout_id,
            ),
            reverse=True,
        )
        return candidates[0]


def default_swebench_pro_output_dir(
    config: ApexConfig,
    run_kind: str,
    base_dir: str | Path | None = None,
) -> Path:
    if run_kind not in {"apex", "raw"}:
        raise ValueError(f"Unsupported SWE-Bench Pro run kind: {run_kind}")

    llm_configs = serialize_llm_configs(config)
    primary = llm_configs[0] if llm_configs else {}
    backend = _slugify_output_component(str(primary.get("backend", "default")))
    model = _slugify_output_component(str(primary.get("model", "default")))
    output_root = Path(base_dir) if base_dir is not None else Path.cwd()
    return output_root / f".{run_kind}_swebench_pro_{backend}_{model}"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m apex.evaluation.swebench_pro_benchmark",
        description="Shared SWE-Bench Pro helper commands.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    eval_parser = subparsers.add_parser(
        "eval-repo",
        help="Evaluate the current repository state against a SWE-Bench Pro task.",
    )
    eval_parser.add_argument("--repo-root", required=True, help="Path to the repository root")
    task_group = eval_parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task-file", help="Path to a serialized SWE-Bench Pro task JSON file")
    task_group.add_argument("--instance-id", help="Task instance id")
    eval_parser.add_argument(
        "--dataset-name",
        default=SWEBENCH_PRO_DATASET_NAME,
        help="HuggingFace dataset name",
    )
    eval_parser.add_argument(
        "--dataset-split",
        default=SWEBENCH_PRO_DATASET_SPLIT,
        help="HuggingFace dataset split",
    )
    eval_parser.add_argument(
        "--dockerhub-username",
        default=SWEBENCH_PRO_DOCKERHUB_USERNAME,
        help="Docker Hub username hosting the sweap-images repository",
    )
    eval_parser.add_argument(
        "--scripts-cache-dir",
        default=None,
        help="Optional directory used to cache official run scripts and parsers",
    )
    eval_parser.add_argument(
        "--docker-platform",
        default=None,
        help="Docker platform override, e.g. linux/amd64",
    )
    eval_parser.add_argument(
        "--block-network",
        action="store_true",
        help="Disable network access inside evaluation containers",
    )
    eval_parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Optional directory to persist evaluation logs for this invocation",
    )

    args = parser.parse_args()

    if args.command != "eval-repo":
        parser.print_help()
        sys.exit(1)

    harness = SWEBenchProHarness(
        output_dir=tempfile.mkdtemp(prefix="apex-swebench-eval-cli-"),
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        dockerhub_username=args.dockerhub_username,
        scripts_cache_dir=args.scripts_cache_dir,
        docker_platform=args.docker_platform,
        block_network=args.block_network,
    )
    task = harness.load_task(task_file=args.task_file, instance_id=args.instance_id)
    evaluation = harness.evaluate_repo(
        task,
        Path(args.repo_root).resolve(),
        artifacts_dir=args.artifacts_dir,
    )
    print(harness.format_for_test_command(evaluation))
    sys.exit(0 if evaluation.all_required_tests_passed else 1)


def _parse_literal_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text]
    if isinstance(parsed, (list, tuple, set)):
        return [str(item) for item in parsed]
    return [str(parsed)]


def _parse_patch_paths(patch_text: str) -> list[str]:
    paths: list[str] = []
    for match in re.finditer(
        r"^diff --git a/(?P<old>.+?) b/(?P<new>.+?)$",
        patch_text,
        flags=re.MULTILINE,
    ):
        old_path = match.group("old")
        new_path = match.group("new")
        if new_path != "/dev/null":
            paths.append(new_path)
        elif old_path != "/dev/null":
            paths.append(old_path)
    return sorted(set(paths))


def _parse_before_repo_set_paths(command: str) -> list[str]:
    paths: list[str] = []
    for line in _extract_benchmark_prep_commands(command):
        payload = line.split(" -- ", 1)[1]
        try:
            tokens = shlex.split(payload)
        except ValueError:
            tokens = payload.split()
        paths.extend(token for token in tokens if token and not token.startswith("-"))
    return sorted(set(paths))


def _format_list_preview(items: list[str], *, max_items: int) -> str:
    preview = items[:max_items]
    lines = [f"- {item}" for item in preview]
    if len(items) > max_items:
        lines.append(f"- ... ({len(items) - max_items} more)")
    return "\n".join(lines)


def _strip_binary_hunks(patch: str) -> str:
    if not patch:
        return patch
    sections = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
    kept: list[str] = []
    for section in sections:
        if not section.strip():
            continue
        if re.search(r"^Binary files .* differ$", section, re.MULTILINE):
            continue
        if re.search(r"^GIT binary patch$", section, re.MULTILINE):
            continue
        kept.append(section)
    return "".join(kept)


def _path_is_excluded(rel_path: str, disallowed_paths: list[str]) -> bool:
    normalized = rel_path.strip().lstrip("./")
    return any(
        normalized == candidate or normalized.startswith(candidate.rstrip("/") + "/")
        for candidate in disallowed_paths
    )


def _is_ignored_repo_artifact(path: str) -> bool:
    return is_ignored_change_path(path)


def _rollout_id_from_name(name: str) -> Optional[int]:
    if not name.startswith("rollout_"):
        return None
    suffix = name[len("rollout_") :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _format_model_config_summary(model_config: list[dict[str, Any]]) -> str:
    if not model_config:
        return "none"
    summaries = []
    for entry in model_config:
        backend = entry.get("backend", "unknown")
        model = entry.get("model", "default")
        timeout = entry.get("cli_timeout")
        if timeout is None:
            timeout = entry.get("timeout")
        if timeout is None:
            summaries.append(f"{backend}/{model}")
            continue
        summaries.append(f"{backend}/{model} (timeout={timeout}s)")
    return ", ".join(summaries)


def _slugify_output_component(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return slug.strip("_") or "default"


def _extract_benchmark_prep_commands(command: str) -> list[str]:
    commands: list[str] = []
    for raw_line in command.splitlines():
        line = raw_line.strip()
        if line.startswith("git checkout ") and " -- " in line:
            commands.append(line)
    return commands


def _strip_instance_version_suffix(instance_id: str) -> str:
    return re.sub(r"-v[^-/]+$", "", instance_id)


@dataclass
class _CommandResult:
    returncode: int
    output: str


if __name__ == "__main__":
    main()
