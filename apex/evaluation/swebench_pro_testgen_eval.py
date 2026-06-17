"""
SWE-Bench Pro test-generation-only evaluator.

This runner prepares each benchmark repository from either the official Docker
image or a host-side git checkout, builds the same public-parity issue
description Apex would see during a normal run, forces the
reproducer/localizer/test-writer path, then compares the generated synthetic
test portfolio against the gold benchmark tests offline.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import multiprocessing
import os
import re
import shutil
import signal
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any, Iterator, Optional

from ..agents.artifacts import (
    coerce_localization_artifact,
    coerce_reproduction_artifact,
)
from ..core.cli_backend import ensure_cli_process_cleanup_hooks
from ..core.config import ApexConfig, LLMConfig
from ..orchestrator import ApexOrchestrator
from ..planning.manager import IssuePlanner
from ..rollout.engine import (
    EpisodicMemoryBus,
    GitWorktreeManager,
    _execute_rollout_test_generation,
    _resolve_stage_llm_config,
    _run_scaffold_localizer_stage,
    _run_scaffold_reproducer_stage,
    _select_rollout_brief,
    _test_writer_candidate_is_better,
    _test_writer_issue_surface_repair_signal,
)
from ..test_portfolio import extract_issue_contract_targets, normalize_test_suite_artifact_payload
from .benchmark import build_apex_ablation_config
from .benchmark_adapters import SWEBENCH_PRO_TESTGEN_ADAPTER
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
    build_run_manifest,
    capture_environment_snapshot,
    ensure_run_manifest,
    load_run_manifest,
    manifest_summary,
    update_run_manifest,
    write_task_live_state,
)
from .swebench_pro_benchmark import (
    SWEBENCH_AGENT_VISIBILITY_PUBLISHED_PARITY,
    SWEBENCH_PREPARE_REPO_FROM_DOCKER_IMAGE,
    SWEBENCH_PREPARE_REPO_FROM_HOST_GIT,
    SWEBENCH_PRO_DATASET_NAME,
    SWEBENCH_PRO_DATASET_SPLIT,
    SWEBENCH_PRO_DOCKERHUB_USERNAME,
    SWEBENCH_PRO_HARNESS_NAME,
    SWEBENCH_PRO_HARNESS_VERSION,
    SWEBenchProHarness,
    SWEBenchProTask,
)
from .synthetic_test_analysis import (
    attach_semantic_review_to_comparison,
    build_swebench_gold_comparison_packet_for_task,
    compare_generated_and_gold_portfolios,
    review_generated_vs_gold_test_semantics,
)
from .target_runtime import (
    apply_target_tool_env_to_apex_config,
    docker_image_runtime,
    target_tool_env_overrides,
)
from .test_style import TestStyleProfile, infer_test_style
from .validation_gate import validate_testgen_portfolio_static

logger = logging.getLogger("apex.evaluation.swebench_pro_testgen")

SWEBENCH_PRO_TESTGEN_REPORT_KIND = "apex_swebench_pro_testgen"
SWEBENCH_PRO_TESTGEN_HARNESS_NAME = f"{SWEBENCH_PRO_HARNESS_NAME}_testgen"
SWEBENCH_PRO_TESTGEN_HARNESS_VERSION = f"{SWEBENCH_PRO_HARNESS_VERSION}-testgen.1"
SWEBENCH_PRO_TESTGEN_TASK_TIMEOUT_SKIP_CATEGORY = "task_timeout"
SYSTEMIC_TESTGEN_SKIP_CATEGORIES = frozenset(
    {
        "unsupported_host",
        "host_storage_exhausted",
        "container_runtime_failure",
        "artifact_sync_failure",
    }
)


def _stable_mutation_seed(instance_id: str, source_path: str) -> int:
    digest = hashlib.sha256(
        f"{instance_id}\0{source_path}".encode("utf-8", errors="ignore")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _mutation_payload_measured(payload: dict[str, Any]) -> bool:
    if not payload:
        return False
    if payload.get("skip_reason") or payload.get("error"):
        return False
    classified_total = int(payload.get("killed") or 0) + int(payload.get("survived") or 0)
    return classified_total > 0


def _mutation_payload_skip_reason(payload: dict[str, Any]) -> str:
    if not payload:
        return ""
    explicit = str(payload.get("skip_reason") or "").strip()
    if explicit:
        return explicit
    if payload.get("error"):
        return "error"
    total = int(payload.get("total_mutants") or 0)
    classified_total = int(payload.get("killed") or 0) + int(payload.get("survived") or 0)
    if total > 0 and classified_total == 0:
        return "no_classified_mutants"
    return ""


def _iteration_assertion_mutation_enabled() -> bool:
    return os.environ.get("APEX_DISABLE_ITERATION_ASSERTION_MUTATION", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
        "enabled",
    }


def _f2p_summary_mutation_attempted(summary: dict[str, Any]) -> bool:
    if "mutation_attempted" in summary:
        return bool(summary.get("mutation_attempted"))
    return (
        "mutation_score" in summary
        or "mutation_total" in summary
        or "mutation_skip_reason" in summary
        or "mutation_error" in summary
    )


def _f2p_summary_mutation_measured(summary: dict[str, Any]) -> bool:
    if "mutation_measured" in summary:
        return bool(summary.get("mutation_measured"))
    if not _f2p_summary_mutation_attempted(summary):
        return False
    if summary.get("mutation_skip_reason") or summary.get("mutation_error"):
        return False
    if "mutation_total" in summary:
        return int(summary.get("mutation_total") or 0) > 0
    # Backward compatibility for older reports that only persisted a score.
    return "mutation_score" in summary


def _rollout_scoped_artifact_path(path: str, rollout_id: int) -> str:
    posix = PurePosixPath(str(path or "").replace("\\", "/"))
    suffix = "".join(posix.suffixes)
    name = posix.name
    stem = name[: -len(suffix)] if suffix else posix.stem
    scoped_name = f"{stem}_rollout_{rollout_id}{suffix or posix.suffix}"
    parent = str(posix.parent)
    if not parent or parent == ".":
        return scoped_name
    return f"{parent}/{scoped_name}"


def _artifact_path_can_be_rollout_scoped(path: str) -> bool:
    normalized = str(path or "").replace("\\", "/").lower()
    name = PurePosixPath(normalized).name
    return "__tests__/" in normalized or "test" in name or "spec" in name


def _build_f2p_positive_rollout_union_portfolio(
    candidates: list["_PerRolloutTestGenerationCandidate"],
    *,
    f2p_payloads: dict[int, dict[str, Any]],
    selected_rollout_id: Optional[int],
) -> Optional[dict[str, Any]]:
    """Union test artifacts from rollouts with measured F2P signal.

    Best-of-N selection is good at picking one strong rollout, but it can
    discard complementary tests when different rollouts catch different bug
    surfaces. This helper keeps the selected rollout first, appends other
    F2P-positive portfolios, and lets the downstream F2P + minimizer decide
    which artifacts survive.
    """
    if not candidates or not f2p_payloads:
        return None
    by_id = {candidate.rollout_id: candidate for candidate in candidates}
    positive_ids = [
        candidate.rollout_id
        for candidate in candidates
        if bool(
            dict((f2p_payloads.get(candidate.rollout_id) or {}).get("summary") or {}).get("any_f2p")
        )
    ]
    if len(positive_ids) <= 1:
        return None

    ordered_ids: list[int] = []
    if selected_rollout_id in positive_ids:
        ordered_ids.append(int(selected_rollout_id))
    ordered_ids.extend(rollout_id for rollout_id in positive_ids if rollout_id not in ordered_ids)

    selected_candidate = by_id.get(
        int(selected_rollout_id) if selected_rollout_id is not None else ordered_ids[0]
    )
    base_portfolio = dict(
        (selected_candidate.generated_portfolio if selected_candidate else {}) or {}
    )
    selected_artifacts = list(base_portfolio.get("test_artifacts") or [])
    artifacts: list[dict[str, Any]] = []
    seen_by_path: dict[str, str] = {}
    source_rollout_ids: list[int] = []

    for rollout_id in ordered_ids:
        candidate = by_id.get(rollout_id)
        if candidate is None:
            continue
        source_rollout_ids.append(rollout_id)
        for raw_artifact in list((candidate.generated_portfolio or {}).get("test_artifacts") or []):
            if not isinstance(raw_artifact, dict):
                continue
            path = str(raw_artifact.get("path") or "").strip().replace("\\", "/")
            content = str(raw_artifact.get("content") or "")
            if not path or not content.strip():
                continue
            artifact = dict(raw_artifact)
            existing_content = seen_by_path.get(path)
            if existing_content is not None:
                if existing_content == content:
                    continue
                if not _artifact_path_can_be_rollout_scoped(path):
                    # Support-file conflicts are unsafe to rename blindly
                    # (e.g. conftest.py fixtures). Preserve the selected
                    # rollout's support file and let minimization/F2P judge.
                    continue
                scoped_path = _rollout_scoped_artifact_path(path, rollout_id)
                if scoped_path in seen_by_path:
                    continue
                path = scoped_path
            artifact["path"] = path
            artifact.setdefault("source_rollout_id", rollout_id)
            artifacts.append(artifact)
            seen_by_path[path] = content

    if len(artifacts) <= len(selected_artifacts):
        return None
    union_portfolio = dict(base_portfolio)
    union_portfolio["test_artifacts"] = artifacts
    union_portfolio["portfolio_strategy"] = "f2p_positive_rollout_union"
    union_portfolio["source_rollout_ids"] = source_rollout_ids
    union_portfolio["selected_rollout_id"] = selected_rollout_id
    return union_portfolio


def _testgen_candidate_f2p_score_tuple(
    candidate: "_PerRolloutTestGenerationCandidate",
    f2p_payloads: Optional[dict[int, dict[str, Any]]],
) -> tuple[int, int, int, float, int, float, int, int, int, int]:
    payload = (f2p_payloads or {}).get(candidate.rollout_id) or {}
    summary = dict(payload.get("summary") or {})
    mutation = dict(payload.get("mutation") or {})
    status = str(payload.get("status") or "")
    tests_observed = int(summary.get("tests_observed") or 0)
    return (
        1 if bool(summary.get("any_f2p")) else 0,
        1 if bool(summary.get("fixed_side_passed")) else 0,
        1 if bool(summary.get("runnable")) or tests_observed > 0 else 0,
        float(mutation.get("mutation_score") or 0.0),
        int(summary.get("f2p_count") or 0),
        float(summary.get("f2p_rate") or 0.0),
        -int(summary.get("p2f_count") or 0),
        -int(summary.get("f2f_count") or 0),
        -int(summary.get("p2p_count") or 0),
        1 if status == "ok" else 0,
    )


def _artifact_contract_metadata_gaps(artifact: dict[str, Any]) -> list[str]:
    """Return required contract-mining fields missing from an assertion artifact."""
    content = str((artifact or {}).get("content") or "")
    if not re.search(
        r"\b(assert|expect|should|verify|throws|equals|deepEquals|Fatalf?|Errorf?)\b",
        content,
    ):
        return []

    def concrete_text(value: Any, *, min_chars: int = 12) -> bool:
        text = str(value or "").strip()
        if len(text) < min_chars:
            return False
        return text.lower() not in {"n/a", "na", "none", "unknown", "todo", "tbd"}

    gaps: list[str] = []
    pass_then_invert = dict(artifact.get("pass_then_invert") or {})
    expected_fixed_behavior = (
        artifact.get("expected_fixed_behavior")
        or artifact.get("objective")
        or artifact.get("summary")
    )
    if not concrete_text(expected_fixed_behavior):
        gaps.append("expected_fixed_behavior")
    expected_broken_failure_mode = (
        artifact.get("expected_broken_failure_mode")
        or pass_then_invert.get("inversion_summary")
        or artifact.get("justification")
    )
    if not concrete_text(expected_broken_failure_mode):
        gaps.append("expected_broken_failure_mode")
    authoritative_source = artifact.get("authoritative_source") or " ".join(
        str(item) for item in list(artifact.get("contract_sources") or [])
    )
    if not concrete_text(authoritative_source):
        gaps.append("authoritative_source")
    public_surface = artifact.get("public_surface") or " ".join(
        str(item) for item in list(artifact.get("contract_targets") or [])
    )
    if not concrete_text(public_surface, min_chars=3):
        gaps.append("public_surface")
    return gaps


def _select_static_quality_test_artifacts(
    artifacts: list[dict[str, Any]],
    *,
    language: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return executable generated tests for static quality analysis.

    F2P materializes support artifacts such as fixtures and config helpers,
    but it only executes paths that look like tests for the task language.
    The weak-artifact metric should use that same executable surface; support
    files should not be reported as ``no_test_functions`` candidate tests.
    """

    candidate_artifacts = [
        artifact for artifact in list(artifacts or []) if isinstance(artifact, dict)
    ]
    if not candidate_artifacts:
        return [], []
    try:
        from apex.evaluation.f2p_oracle import _select_test_artifacts_for_language

        selected = _select_test_artifacts_for_language(
            candidate_artifacts,
            language=language,
        )
    except Exception:  # pragma: no cover - defensive fallback
        selected = candidate_artifacts
    selected_paths = {
        str(artifact.get("path") or "").strip()
        for artifact in selected
        if str(artifact.get("path") or "").strip()
    }
    skipped_paths = [
        path
        for artifact in candidate_artifacts
        if (path := str(artifact.get("path") or "").strip()) and path not in selected_paths
    ]
    return list(selected), skipped_paths


def _attach_static_validation_to_testgen_portfolio(
    *,
    task: SWEBenchProTask,
    generated_portfolio: dict[str, Any],
    test_command: str = "",
) -> dict[str, Any]:
    """Attach shared test-generation validation telemetry to a portfolio."""

    style = _infer_generated_portfolio_style(
        task=task,
        generated_portfolio=generated_portfolio,
        test_command=test_command,
    )
    validation = validate_testgen_portfolio_static(
        generated_portfolio,
        style=style,
        splice_simulator=SWEBENCH_PRO_TESTGEN_ADAPTER.splice_simulator(),
    )
    validation.update(
        {
            "benchmark_adapter": "swebench_pro_testgen",
            "validation_scope": "static_pre_comparison",
        }
    )
    generated_portfolio["apex_validation"] = dict(validation)
    return validation


def _infer_generated_portfolio_style(
    *,
    task: SWEBenchProTask,
    generated_portfolio: dict[str, Any],
    test_command: str = "",
) -> TestStyleProfile:
    artifacts = [
        artifact
        for artifact in list((generated_portfolio or {}).get("test_artifacts") or [])
        if isinstance(artifact, dict)
    ]
    first_artifact = artifacts[0] if artifacts else {}
    language = str(task.repo_language or "").strip().lower() or "python"
    probe_path = str(first_artifact.get("path") or "").strip()
    if not probe_path:
        probe_path = _default_test_style_path_for_language(language)
    style = infer_test_style(
        existing_test_source=str(first_artifact.get("content") or ""),
        existing_test_path=probe_path,
        focal_path=probe_path,
    )
    return _style_with_runner_from_command(style, test_command)


def _default_test_style_path_for_language(language: str) -> str:
    normalized = (language or "").lower()
    if normalized in {"javascript", "js"}:
        return "tests/generated.test.js"
    if normalized in {"typescript", "ts"}:
        return "tests/generated.test.ts"
    if normalized == "go":
        return "generated_test.go"
    if normalized == "java":
        return "GeneratedTest.java"
    return "tests/test_generated.py"


def _style_with_runner_from_command(
    style: TestStyleProfile,
    test_command: str,
) -> TestStyleProfile:
    command = str(test_command or "").lower()
    if (style.language or "").lower() in {"python", "py", "python3"}:
        if "runtests.py" in command or "django" in command:
            return replace(
                style,
                runner="django-runtests",
                runner_source="command",
                function_naming="Django TestCase/SimpleTestCase subclass with test_* methods",
                fixture_style="unittest setUp/tearDown and Django test helpers",
                assertion_style="self.assert* assertions used by Django's unittest runner",
            )
        if "unittest" in command and "pytest" not in command:
            return replace(
                style,
                runner="unittest",
                runner_source="command",
                function_naming="unittest.TestCase subclass with test_* methods",
                fixture_style="setUp/tearDown methods when setup is needed",
                assertion_style="self.assert* assertions",
            )
        if "pytest" in command:
            return replace(style, runner="pytest", runner_source="command")
    if (style.language or "").lower() in {"javascript", "typescript"}:
        if "vitest" in command:
            return replace(style, runner="vitest", runner_source="command")
        if "jest" in command:
            return replace(style, runner="jest", runner_source="command")
    return style


def _testgen_candidate_selection_evidence(
    candidate: "_PerRolloutTestGenerationCandidate",
    *,
    f2p_payloads: Optional[dict[int, dict[str, Any]]],
    selected_rollout_id: Optional[int],
) -> dict[str, Any]:
    payload = (f2p_payloads or {}).get(candidate.rollout_id) or {}
    summary = dict(payload.get("summary") or {})
    mutation = dict(payload.get("mutation") or {})
    score = _testgen_candidate_f2p_score_tuple(candidate, f2p_payloads)
    return {
        "rollout_id": candidate.rollout_id,
        "selected": selected_rollout_id == candidate.rollout_id,
        "selection_basis": (
            "f2p_mutation_score_tuple" if f2p_payloads else "heuristic_issue_surface_signal"
        ),
        "score_tuple": {
            "any_f2p": score[0],
            "fixed_side_passed": score[1],
            "runnable": score[2],
            "mutation_score": score[3],
            "f2p_count": score[4],
            "f2p_rate": score[5],
            "p2f_penalty": score[6],
            "f2f_penalty": score[7],
            "p2p_penalty": score[8],
            "status_ok": score[9],
        },
        "f2p_status": payload.get("status"),
        "f2p_summary": {
            key: summary.get(key)
            for key in (
                "candidate_test_paths",
                "tests_observed",
                "f2p_count",
                "f2p_rate",
                "any_f2p",
                "p2f_count",
                "f2f_count",
                "p2p_count",
                "skipped_count",
                "unreliable_execution",
                "unreliable_raw_f2p_count",
                "fixed_side_passed",
                "runnable",
                "failure_classes",
                "repair_hints",
            )
            if key in summary
        },
        "mutation_summary": {
            key: mutation.get(key)
            for key in (
                "mutation_score",
                "total_mutants",
                "killed",
                "survived",
                "errored",
                "timed_out",
                "skip_reason",
                "language",
                "source_paths",
            )
            if key in mutation
        },
    }


@dataclass
class _PerRolloutTestGenerationCandidate:
    rollout_id: int
    brief: Any
    worktree_path: Path
    baseline_commit: str
    reproduction_artifact: Any
    localization_artifact: Any
    generated_portfolio: dict[str, Any]
    tokens_used: int
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    loop_summary: dict[str, Any] = field(default_factory=dict)
    issue_surface_signal: dict[str, Any] = field(default_factory=dict)


@contextmanager
def _interruptible_thread_pool(max_workers: int) -> Iterator[ThreadPoolExecutor]:
    executor = ThreadPoolExecutor(max_workers=max_workers)
    interrupted = False
    try:
        yield executor
    except BaseException:
        interrupted = True
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if not interrupted:
            executor.shutdown(wait=True)


@contextmanager
def _temporary_environ(overrides: dict[str, str]) -> Iterator[None]:
    previous: dict[str, Optional[str]] = {key: os.environ.get(key) for key in overrides}
    os.environ.update({key: str(value) for key, value in overrides.items()})
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _slugify_output_component(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return slug.strip("_") or "default"


def default_swebench_pro_testgen_output_dir(
    config: ApexConfig,
    base_dir: str | Path | None = None,
) -> Path:
    llm_configs = serialize_llm_configs(config)
    primary = llm_configs[0] if llm_configs else {}
    backend = _slugify_output_component(str(primary.get("backend", "default")))
    model = _slugify_output_component(str(primary.get("model", "default")))
    output_root = Path(base_dir) if base_dir is not None else Path.cwd()
    return output_root / f".apex_swebench_pro_testgen_{backend}_{model}"


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _median(values: list[float]) -> float:
    return round(float(median(values)), 4) if values else 0.0


def _counter_preview(counter: Counter[str], *, max_items: int = 10) -> list[dict[str, Any]]:
    return [{"name": name, "count": count} for name, count in counter.most_common(max_items)]


def _task_summary_float(payload: dict[str, Any], key: str) -> Optional[float]:
    value = payload.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _task_comparison_status_ok(task: "SWEBenchProTestGenTaskResult") -> bool:
    metadata = dict(task.execution_metadata or {})
    status = str(metadata.get("comparison_status") or "").strip().lower()
    if not status:
        return bool(task.success)
    return status == "ok"


def _build_testgen_benchmark_policy(
    *,
    docker_platform: Optional[str],
    block_network: bool,
    prepare_repo_mode: str,
) -> dict[str, Any]:
    repo_source = (
        "host_git_clone_with_local_mirror_cache"
        if prepare_repo_mode == SWEBENCH_PREPARE_REPO_FROM_HOST_GIT
        else "official_swebench_pro_docker_image"
    )
    evaluator_network_access = (
        "blocked"
        if block_network
        else (
            "host_inherited"
            if prepare_repo_mode == SWEBENCH_PREPARE_REPO_FROM_HOST_GIT
            else "docker_default"
        )
    )
    return build_benchmark_policy(
        benchmark_name="swebench_pro_testgen",
        benchmark_family="swebench_pro",
        agent_input_contract={
            "repo_snapshot_visible": True,
            "problem_statement_visible": True,
            "requirements_visible": True,
            "interface_visible": True,
            "test_command_visible": True,
            "agent_visibility_mode": SWEBENCH_AGENT_VISIBILITY_PUBLISHED_PARITY,
            "benchmark_private_gold_tests_visible_during_generation": False,
        },
        orchestrator_input_contract={
            "forced_test_generation": True,
            "executed_stages": ["reproducer", "localizer", "test_writer"],
            "patch_generation_executed": False,
        },
        evaluation_protocol={
            "primary_metric": "synthetic_vs_gold_test_coverage",
            "comparison_scope": "offline_post_generation",
            "canonical_rollout_count": 1,
            "gold_tests_hidden_during_generation": True,
        },
        environment_policy={
            "agent_execution_isolation": "per_task_temp_sandbox",
            "repo_source": repo_source,
            "prepare_repo_mode": prepare_repo_mode,
            "agent_network_access": "inherited_host",
            "docker_platform": docker_platform or "auto",
            "evaluator_network_access": evaluator_network_access,
        },
        benchmark_specifics={
            "published_parity_clean": True,
            "gold_tests_only_materialized_after_generation": True,
        },
    )


@dataclass
class SWEBenchProTestGenTaskResult:
    instance_id: str
    repo: str
    success: bool
    generated_summary: dict[str, Any] = field(default_factory=dict)
    gold_summary: dict[str, Any] = field(default_factory=dict)
    coverage_summary: dict[str, Any] = field(default_factory=dict)
    target_comparison: list[dict[str, Any]] = field(default_factory=list)
    semantic_review: dict[str, Any] = field(default_factory=dict)
    issue_contract_targets: list[str] = field(default_factory=list)
    total_tokens: int = 0
    duration_seconds: float = 0.0
    result_path: Optional[str] = None
    generated_portfolio_path: Optional[str] = None
    semantic_review_error: Optional[str] = None
    failure_reason: Optional[str] = None
    skipped: bool = False
    skip_category: Optional[str] = None
    execution_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "success": self.success,
            "generated_summary": copy.deepcopy(self.generated_summary),
            "gold_summary": copy.deepcopy(self.gold_summary),
            "coverage_summary": copy.deepcopy(self.coverage_summary),
            "target_comparison": copy.deepcopy(self.target_comparison),
            "semantic_review": copy.deepcopy(self.semantic_review),
            "issue_contract_targets": list(self.issue_contract_targets),
            "total_tokens": self.total_tokens,
            "duration_seconds": self.duration_seconds,
            "result_path": self.result_path,
            "generated_portfolio_path": self.generated_portfolio_path,
            "semantic_review_error": self.semantic_review_error,
            "failure_reason": self.failure_reason,
            "skipped": self.skipped,
            "skip_category": self.skip_category,
            "execution_metadata": copy.deepcopy(self.execution_metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SWEBenchProTestGenTaskResult":
        return cls(
            instance_id=str(payload.get("instance_id") or ""),
            repo=str(payload.get("repo") or ""),
            success=bool(payload.get("success", False)),
            generated_summary=dict(payload.get("generated_summary") or {}),
            gold_summary=dict(payload.get("gold_summary") or {}),
            coverage_summary=dict(payload.get("coverage_summary") or {}),
            target_comparison=list(payload.get("target_comparison") or []),
            semantic_review=dict(payload.get("semantic_review") or {}),
            issue_contract_targets=[
                str(item)
                for item in list(payload.get("issue_contract_targets") or [])
                if str(item).strip()
            ],
            total_tokens=int(payload.get("total_tokens", 0) or 0),
            duration_seconds=float(payload.get("duration_seconds", 0.0) or 0.0),
            result_path=payload.get("result_path"),
            generated_portfolio_path=payload.get("generated_portfolio_path"),
            semantic_review_error=payload.get("semantic_review_error"),
            failure_reason=payload.get("failure_reason"),
            skipped=bool(payload.get("skipped", False)),
            skip_category=payload.get("skip_category"),
            execution_metadata=dict(payload.get("execution_metadata") or {}),
        )


def _run_testgen_task_process_entry(
    evaluator: Any,
    task: SWEBenchProTask,
    result_queue: Any,
) -> None:
    """Run one benchmark task in a child process and checkpoint its result.

    The parent treats the checkpoint as authoritative. The queue is only a
    low-latency success path; if the child exits before posting, the parent
    reloads `task_result.json` or synthesizes a failed result.
    """

    started = time.time()
    try:
        os.setsid()
    except Exception as exc:
        # Audit H12: ``setsid`` failure is non-fatal but worth logging —
        # without our own session, the child can't be reliably killed
        # by the orphan sweeper.
        logger = __import__("logging").getLogger(__name__)
        logger.warning(
            "swebench_pro_testgen_eval: os.setsid failed (%s: %s); "
            "child process group not isolated",
            type(exc).__name__,
            exc,
        )
    try:
        result = evaluator._run_task_with_checkpoint(task)
    except BaseException as exc:  # pragma: no cover - child-process fallback
        task_output_dir = evaluator._task_output_dir(task)
        payload = SWEBenchProTestGenTaskResult(
            instance_id=task.instance_id,
            repo=task.repo,
            success=False,
            duration_seconds=time.time() - started,
            failure_reason=f"subprocess_error:{type(exc).__name__}: {exc}",
            skipped=False,
            execution_metadata={
                "task_process_pid": os.getpid(),
                "subprocess_error_type": type(exc).__name__,
            },
        ).to_dict()
        try:
            write_task_live_state(
                task_output_dir,
                {
                    "task_id": task.instance_id,
                    "phase": "failed",
                    "status": "failed",
                    "process_pid": os.getpid(),
                    "last_progress_at": time.time(),
                    "error": payload["failure_reason"],
                },
            )
            write_task_checkpoint(task_output_dir, payload)
        finally:
            try:
                result_queue.put(payload)
            except Exception as exc:
                # Audit H12: queue put failure leaves the parent waiting
                # forever on the result. Surface the cause so the parent
                # can join with a useful diagnostic.
                logger = __import__("logging").getLogger(__name__)
                logger.warning(
                    "swebench_pro_testgen_eval: result_queue.put failed "
                    "for %s (%s: %s); parent will time out waiting",
                    task.instance_id,
                    type(exc).__name__,
                    exc,
                )
        return
    try:
        result_queue.put(result.to_dict())
    except Exception as exc:
        logger = __import__("logging").getLogger(__name__)
        logger.warning(
            "swebench_pro_testgen_eval: result_queue.put (success path) failed for %s (%s: %s)",
            task.instance_id,
            type(exc).__name__,
            exc,
        )


@dataclass
class SWEBenchProTestGenReport:
    tasks: list[SWEBenchProTestGenTaskResult] = field(default_factory=list)
    requested_task_ids: list[str] = field(default_factory=list)
    requested_repo_names: list[str] = field(default_factory=list)
    started_at: float = 0.0
    updated_at: float = 0.0
    finished_at: float = 0.0
    dataset_name: str = SWEBENCH_PRO_DATASET_NAME
    dataset_split: str = SWEBENCH_PRO_DATASET_SPLIT
    report_kind: str = SWEBENCH_PRO_TESTGEN_REPORT_KIND
    harness_name: str = SWEBENCH_PRO_TESTGEN_HARNESS_NAME
    harness_version: str = SWEBENCH_PRO_TESTGEN_HARNESS_VERSION
    config_source: Optional[str] = None
    model_config: list[dict[str, Any]] = field(default_factory=list)
    ablation_config: dict[str, Any] = field(default_factory=dict)
    run_manifest: dict[str, Any] = field(default_factory=dict)
    evaluation_mode: str = "published_parity_forced_testgen"
    enable_testgen_memory: bool = False
    allow_gold_oracle_selection: bool = False
    testgen_memory_directory: Optional[str] = None
    testgen_memory_per_task_summary: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def total_tasks(self) -> int:
        return len(self.requested_task_ids) if self.requested_task_ids else len(self.tasks)

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
    def successful_tasks(self) -> int:
        return sum(1 for task in self.tasks if task.success)

    @property
    def failed_tasks(self) -> int:
        return sum(1 for task in self.tasks if not task.skipped and not task.success)

    @property
    def skipped_tasks(self) -> int:
        return sum(1 for task in self.tasks if task.skipped)

    @property
    def runnable_tasks(self) -> int:
        return sum(1 for task in self.tasks if not task.skipped)

    @property
    def repo_names(self) -> list[str]:
        if self.requested_repo_names:
            return list(self.requested_repo_names)
        return sorted({task.repo for task in self.tasks})

    def _successful_coverage_summaries(self) -> list[dict[str, Any]]:
        return [
            dict(task.coverage_summary)
            for task in self.tasks
            if _task_comparison_status_ok(task) and dict(task.coverage_summary)
        ]

    def _successful_target_comparisons(self) -> list[list[dict[str, Any]]]:
        return [
            list(task.target_comparison)
            for task in self.tasks
            if _task_comparison_status_ok(task) and list(task.target_comparison)
        ]

    @property
    def compared_tasks(self) -> int:
        return len(self._successful_coverage_summaries())

    @property
    def aggregate_metrics(self) -> dict[str, Any]:
        summaries = self._successful_coverage_summaries()
        comparisons = self._successful_target_comparisons()
        contract_axis_recalls = [
            float(value)
            for summary in summaries
            for value in [_task_summary_float(summary, "overall_contract_axis_recall")]
            if value is not None
        ]
        required_axis_coverage_scores = [
            float(value)
            for summary in summaries
            for value in [_task_summary_float(summary, "required_axis_coverage_score")]
            if value is not None
        ]
        missing_required_axis_counter: Counter[str] = Counter()
        for summary in summaries:
            for axis in list(summary.get("missing_required_axes") or []):
                axis_text = str(axis).strip()
                if axis_text:
                    missing_required_axis_counter[axis_text] += 1
        gold_target_recalls = [
            float(value)
            for summary in summaries
            for value in [_task_summary_float(summary, "gold_target_recall")]
            if value is not None
        ]
        generated_contract_target_coverage = [
            float(value)
            for summary in summaries
            for value in [_task_summary_float(summary, "generated_contract_target_coverage_ratio")]
            if value is not None
        ]
        gold_field_path_recalls = [
            float(value)
            for summary in summaries
            if int(summary.get("gold_field_path_count", 0) or 0) > 0
            for value in [_task_summary_float(summary, "gold_field_path_recall")]
            if value is not None
        ]
        # `gold_field_path_recall` is gated on `gold_field_path_count > 0`,
        # which excludes tasks where gold tests don't assert on JSON / payload
        # field paths but the *issue* documents some. The authoritative recall
        # metric is gated on the union (issue OR gold) so downstream dashboards
        # see a non-vacuous number for those tasks too.
        authoritative_field_path_recalls = [
            float(value)
            for summary in summaries
            if int(summary.get("authoritative_field_path_count", 0) or 0) > 0
            for value in [_task_summary_float(summary, "authoritative_field_path_recall")]
            if value is not None
        ]
        gold_field_path_shape_recalls = [
            float(value)
            for summary in summaries
            if int(summary.get("gold_field_path_shape_count", 0) or 0) > 0
            for value in [_task_summary_float(summary, "gold_field_path_shape_recall")]
            if value is not None
        ]
        generated_negative_shape_coverage = [
            float(value)
            for summary in summaries
            if int(summary.get("gold_field_path_shape_count", 0) or 0) > 0
            for value in [
                _task_summary_float(summary, "generated_field_path_negative_shape_coverage_ratio")
            ]
            if value is not None
        ]
        semantic_review_summaries = [
            summary
            for summary in summaries
            if str(summary.get("semantic_review_verdict") or "").strip()
        ]
        semantic_strict_behavioral_recalls = [
            float(value)
            for summary in semantic_review_summaries
            for value in [_task_summary_float(summary, "semantic_review_strict_behavioral_recall")]
            if value is not None
        ]
        semantic_lenient_behavioral_recalls = [
            float(value)
            for summary in semantic_review_summaries
            for value in [_task_summary_float(summary, "semantic_review_lenient_behavioral_recall")]
            if value is not None
        ]
        semantic_review_confidences = [
            float(value)
            for summary in semantic_review_summaries
            for value in [_task_summary_float(summary, "semantic_review_confidence")]
            if value is not None
        ]
        semantic_review_verdict_counter: Counter[str] = Counter()
        for summary in semantic_review_summaries:
            verdict = str(summary.get("semantic_review_verdict") or "").strip()
            if verdict:
                semantic_review_verdict_counter[verdict] += 1

        missing_axis_counter: Counter[str] = Counter()
        for comparison_entries in comparisons:
            for target_entry in comparison_entries:
                for axis in list(dict(target_entry or {}).get("missing_axes") or []):
                    axis_text = str(axis).strip()
                    if axis_text:
                        missing_axis_counter[axis_text] += 1

        missing_shape_counter: Counter[str] = Counter()
        missing_field_path_counter: Counter[str] = Counter()
        for summary in summaries:
            for shape in list(summary.get("missing_gold_field_path_shapes") or []):
                shape_text = str(shape).strip()
                if shape_text:
                    missing_shape_counter[shape_text] += 1
            for field_path in list(summary.get("missing_gold_field_paths") or []):
                field_path_text = str(field_path).strip()
                if field_path_text:
                    missing_field_path_counter[field_path_text] += 1

        hardest_tasks = sorted(
            [
                {
                    "instance_id": task.instance_id,
                    "repo": task.repo,
                    "overall_contract_axis_recall": float(
                        task.coverage_summary.get("overall_contract_axis_recall") or 0.0
                    ),
                    "gold_target_recall": float(
                        task.coverage_summary.get("gold_target_recall") or 0.0
                    ),
                    "gold_field_path_recall": float(
                        task.coverage_summary.get("gold_field_path_recall") or 0.0
                    ),
                    "gold_field_path_shape_recall": float(
                        task.coverage_summary.get("gold_field_path_shape_recall") or 0.0
                    ),
                    "generated_artifact_count": int(
                        task.coverage_summary.get("generated_artifact_count") or 0
                    ),
                    "total_tokens": int(task.total_tokens or 0),
                    "duration_seconds": round(float(task.duration_seconds or 0.0), 1),
                }
                for task in self.tasks
                if task.success and dict(task.coverage_summary)
            ],
            key=lambda item: (
                item["overall_contract_axis_recall"],
                item["gold_target_recall"],
                item["gold_field_path_recall"],
                item["gold_field_path_shape_recall"],
                item["generated_artifact_count"],
            ),
        )[:15]

        return {
            "compared_tasks": self.compared_tasks,
            "quality_successful_tasks": self.successful_tasks,
            "comparable_task_denominator": self.compared_tasks,
            "mean_overall_contract_axis_recall": _mean(contract_axis_recalls),
            "median_overall_contract_axis_recall": _median(contract_axis_recalls),
            "full_overall_contract_axis_recall_tasks": sum(
                1 for value in contract_axis_recalls if value >= 0.9999
            ),
            "mean_required_axis_coverage_score": _mean(required_axis_coverage_scores),
            "full_required_axis_coverage_tasks": sum(
                1 for value in required_axis_coverage_scores if value >= 0.9999
            ),
            "top_missing_required_axes": _counter_preview(missing_required_axis_counter),
            "mean_gold_target_recall": _mean(gold_target_recalls),
            "full_gold_target_recall_tasks": sum(
                1 for value in gold_target_recalls if value >= 0.9999
            ),
            "mean_generated_contract_target_coverage_ratio": _mean(
                generated_contract_target_coverage
            ),
            "tasks_with_gold_field_paths": len(gold_field_path_recalls),
            "mean_gold_field_path_recall": _mean(gold_field_path_recalls),
            "full_gold_field_path_recall_tasks": sum(
                1 for value in gold_field_path_recalls if value >= 0.9999
            ),
            "tasks_with_authoritative_field_paths": len(authoritative_field_path_recalls),
            "mean_authoritative_field_path_recall": _mean(authoritative_field_path_recalls),
            "full_authoritative_field_path_recall_tasks": sum(
                1 for value in authoritative_field_path_recalls if value >= 0.9999
            ),
            "tasks_with_gold_field_path_shapes": len(gold_field_path_shape_recalls),
            "mean_gold_field_path_shape_recall": _mean(gold_field_path_shape_recalls),
            "full_gold_field_path_shape_recall_tasks": sum(
                1 for value in gold_field_path_shape_recalls if value >= 0.9999
            ),
            "mean_generated_field_path_negative_shape_coverage_ratio": _mean(
                generated_negative_shape_coverage
            ),
            "semantic_review_tasks": len(semantic_review_summaries),
            "semantic_review_error_tasks": sum(
                1
                for task in self.tasks
                if task.success and str(task.semantic_review_error or "").strip()
            ),
            "mean_semantic_review_confidence": _mean(semantic_review_confidences),
            "mean_semantic_review_strict_behavioral_recall": _mean(
                semantic_strict_behavioral_recalls
            ),
            "mean_semantic_review_lenient_behavioral_recall": _mean(
                semantic_lenient_behavioral_recalls
            ),
            "full_semantic_review_strict_behavioral_recall_tasks": sum(
                1 for value in semantic_strict_behavioral_recalls if value >= 0.9999
            ),
            "full_semantic_review_lenient_behavioral_recall_tasks": sum(
                1 for value in semantic_lenient_behavioral_recalls if value >= 0.9999
            ),
            "semantic_review_tasks_with_no_material_gaps": sum(
                1
                for summary in semantic_review_summaries
                if bool(summary.get("semantic_review_no_material_gaps"))
            ),
            "semantic_review_tasks_with_no_weaker_assertions": sum(
                1
                for summary in semantic_review_summaries
                if bool(summary.get("semantic_review_no_weaker_assertions"))
            ),
            "semantic_review_verdicts": _counter_preview(semantic_review_verdict_counter),
            "top_missing_axes": _counter_preview(missing_axis_counter),
            "top_missing_gold_field_path_shapes": _counter_preview(missing_shape_counter),
            "top_missing_gold_field_paths": _counter_preview(
                missing_field_path_counter,
                max_items=15,
            ),
            "lowest_recall_tasks": hardest_tasks,
            **self._test_quality_aggregate_metrics(),
            **self._f2p_aggregate_metrics(),
        }

    def _test_quality_aggregate_metrics(self) -> dict[str, Any]:
        """Aggregate static oracle-quality checks for generated tests."""
        quality_summaries: list[dict[str, Any]] = []
        issue_counter: Counter[str] = Counter()
        assertion_effect_scores: list[float] = []
        for task in self.tasks:
            metadata = dict(task.execution_metadata or {})
            summary = dict(metadata.get("test_quality_summary") or {})
            if not summary:
                continue
            quality_summaries.append(summary)
            for code, count in dict(summary.get("issue_counts") or {}).items():
                code_text = str(code).strip()
                if code_text:
                    issue_counter[code_text] += int(count or 0)
            artifact_scores = [
                float(dict(artifact).get("assertion_effect_score") or 0.0)
                for artifact in list(summary.get("artifacts") or [])
                if isinstance(artifact, dict)
            ]
            if artifact_scores:
                assertion_effect_scores.extend(artifact_scores)
                continue
            mean_score = _task_summary_float(
                summary,
                "mean_assertion_effect_score",
            )
            if mean_score is not None:
                assertion_effect_scores.append(float(mean_score))

        if not quality_summaries:
            return {
                "test_quality_tasks": 0,
                "test_quality_artifact_count": 0,
                "test_quality_weak_artifact_count": 0,
                "test_quality_weak_artifact_rate": 0.0,
                "test_quality_issue_count": 0,
                "mean_test_quality_assertion_effect_score": 0.0,
                "test_quality_analysis_error_tasks": 0,
                "test_quality_issue_counts": {},
                "top_test_quality_issues": [],
            }

        artifact_count = sum(
            int(summary.get("artifact_count") or 0) for summary in quality_summaries
        )
        weak_artifact_count = sum(
            int(summary.get("weak_artifact_count") or 0) for summary in quality_summaries
        )
        issue_count = sum(int(summary.get("issue_count") or 0) for summary in quality_summaries)
        analysis_error_tasks = sum(
            1 for summary in quality_summaries if str(summary.get("error") or "").strip()
        )
        return {
            "test_quality_tasks": len(quality_summaries),
            "test_quality_artifact_count": artifact_count,
            "test_quality_weak_artifact_count": weak_artifact_count,
            "test_quality_weak_artifact_rate": round(
                (weak_artifact_count / artifact_count) if artifact_count else 0.0,
                4,
            ),
            "test_quality_issue_count": issue_count,
            "mean_test_quality_assertion_effect_score": _mean(assertion_effect_scores),
            "test_quality_analysis_error_tasks": analysis_error_tasks,
            "test_quality_issue_counts": dict(issue_counter),
            "top_test_quality_issues": _counter_preview(issue_counter),
        }

    def _f2p_aggregate_metrics(self) -> dict[str, Any]:
        """Aggregate F2P (Fail-to-Pass) execution-oracle results.

        F2P is opt-in (`--enable-f2p`); when disabled the resulting metrics
        block is small-and-zero rather than absent so downstream consumers can
        always pull the same keys.
        """
        f2p_summaries: list[dict[str, Any]] = []
        f2p_status_counter: Counter[str] = Counter()
        for task in self.tasks:
            metadata = dict(task.execution_metadata or {})
            if not metadata.get("f2p_enabled"):
                continue
            summary = dict(metadata.get("f2p_summary") or {})
            if not summary:
                continue
            status = str(summary.get("status") or "unknown")
            f2p_status_counter[status] += 1
            if status == "ok":
                f2p_summaries.append(summary)

        if not f2p_status_counter:
            return {
                "f2p_evaluated_tasks": 0,
                "f2p_status_counts": {},
                "f2p_tasks_with_any_f2p": 0,
                "mean_f2p_rate": 0.0,
                "mean_f2p_count_per_task": 0.0,
                "tasks_with_p2f_regressions": 0,
                "assertion_mutation_attempted_tasks": 0,
                "assertion_mutation_measured_tasks": 0,
                "assertion_mutation_survived_tasks": 0,
            }

        any_f2p_count = sum(1 for summary in f2p_summaries if bool(summary.get("any_f2p")))
        f2p_rates = [float(summary.get("f2p_rate") or 0.0) for summary in f2p_summaries]
        f2p_counts = [int(summary.get("f2p_count") or 0) for summary in f2p_summaries]
        p2f_count = sum(1 for summary in f2p_summaries if int(summary.get("p2f_count") or 0) > 0)
        assertion_mutation_attempt_summaries = [
            s for s in f2p_summaries if bool(s.get("assertion_mutation_attempted"))
        ]
        assertion_mutation_measured_summaries = [
            s
            for s in assertion_mutation_attempt_summaries
            if bool(s.get("assertion_mutation_measured"))
        ]
        # Mutation discrimination is opt-in (`--enable-mutation`) AND only
        # runs on tasks where F2P confirmed the bug was caught. Skipped
        # attempts (for example no language-supported patch targets) are
        # useful telemetry but must not enter score aggregates.
        mutation_attempt_summaries = [
            s for s in f2p_summaries if _f2p_summary_mutation_attempted(s)
        ]
        mutation_summaries = [
            s for s in mutation_attempt_summaries if _f2p_summary_mutation_measured(s)
        ]
        mutation_skip_counts: Counter[str] = Counter(
            str(s.get("mutation_skip_reason") or "unknown")
            for s in mutation_attempt_summaries
            if not _f2p_summary_mutation_measured(s)
        )
        if mutation_summaries:
            mutation_scores = [float(s.get("mutation_score") or 0.0) for s in mutation_summaries]
            mutation_killed_total = sum(
                int(s.get("mutation_killed") or 0) for s in mutation_summaries
            )
            mutation_total = sum(int(s.get("mutation_total") or 0) for s in mutation_summaries)
            effective_mutation_evaluable_total = sum(
                int(
                    s.get("effective_mutation_evaluable")
                    if s.get("effective_mutation_evaluable") is not None
                    else s.get("mutation_total") or 0
                )
                for s in mutation_summaries
            )
            baseline_subset_tasks = sum(
                1
                for s in mutation_summaries
                if str(s.get("mutation_score_denominator") or "") == "baseline_passing_subset"
            )
            mutation_block: dict[str, Any] = {
                "mutation_attempted_tasks": len(mutation_attempt_summaries),
                "mutation_measured_tasks": len(mutation_summaries),
                "mutation_evaluated_tasks": len(mutation_summaries),
                "mean_mutation_score": _mean(mutation_scores),
                "median_mutation_score": _median(mutation_scores),
                "mean_measured_mutation_score": _mean(mutation_scores),
                "median_measured_mutation_score": _median(mutation_scores),
                "total_mutants_killed": mutation_killed_total,
                "total_mutants_evaluated": mutation_total,
                "effective_mutation_evaluable_total": effective_mutation_evaluable_total,
                "effective_mutation_evaluable_tasks": sum(
                    1
                    for s in mutation_summaries
                    if int(s.get("effective_mutation_evaluable") or 0) > 0
                ),
                "baseline_passing_subset_mutation_tasks": baseline_subset_tasks,
                "mutation_skipped_tasks": sum(mutation_skip_counts.values()),
                "mutation_skip_counts": dict(mutation_skip_counts),
                "tasks_with_perfect_mutation_score": sum(1 for s in mutation_scores if s >= 0.999),
            }
        else:
            mutation_block = {
                "mutation_attempted_tasks": len(mutation_attempt_summaries),
                "mutation_measured_tasks": 0,
                "mutation_evaluated_tasks": 0,
                "mean_mutation_score": 0.0,
                "median_mutation_score": 0.0,
                "mean_measured_mutation_score": 0.0,
                "median_measured_mutation_score": 0.0,
                "total_mutants_killed": 0,
                "total_mutants_evaluated": 0,
                "effective_mutation_evaluable_total": 0,
                "effective_mutation_evaluable_tasks": 0,
                "baseline_passing_subset_mutation_tasks": 0,
                "mutation_skipped_tasks": sum(mutation_skip_counts.values()),
                "mutation_skip_counts": dict(mutation_skip_counts),
                "tasks_with_perfect_mutation_score": 0,
            }
        return {
            "f2p_evaluated_tasks": sum(f2p_status_counter.values()),
            "f2p_runnable_tasks": len(f2p_summaries),
            "f2p_status_counts": dict(f2p_status_counter),
            "f2p_tasks_with_any_f2p": any_f2p_count,
            "mean_f2p_rate": _mean(f2p_rates),
            "mean_f2p_count_per_task": _mean([float(value) for value in f2p_counts]),
            "tasks_with_p2f_regressions": p2f_count,
            "assertion_mutation_attempted_tasks": len(assertion_mutation_attempt_summaries),
            "assertion_mutation_measured_tasks": len(assertion_mutation_measured_summaries),
            "assertion_mutation_survived_tasks": sum(
                1
                for s in assertion_mutation_measured_summaries
                if bool(s.get("assertion_mutation_survived"))
            ),
            **mutation_block,
        }

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
            "evaluation_mode": self.evaluation_mode,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "completed": self.completed,
            "dataset_name": self.dataset_name,
            "dataset_split": self.dataset_split,
            "completed_tasks": self.completed_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "skipped_tasks": self.skipped_tasks,
            "runnable_tasks": self.runnable_tasks,
            "total_tasks": self.total_tasks,
            "aggregate_metrics": self.aggregate_metrics,
            "methodology": self.methodology_block(),
            "run_manifest": manifest_summary(self.run_manifest),
            "enable_testgen_memory": self.enable_testgen_memory,
            "allow_gold_oracle_selection": self.allow_gold_oracle_selection,
            "testgen_memory_directory": self.testgen_memory_directory,
            "testgen_memory_per_task_summary": copy.deepcopy(self.testgen_memory_per_task_summary),
            "tasks": [task.to_dict() for task in self.tasks],
        }

    def methodology_block(self) -> dict[str, Any]:
        """Audit-friendly methodology metadata derived from current state.

        Reviewers consistently challenge testgen / coding-agent papers on
        five surfaces: (a) subset selection bias, (b) model-class confound
        (single model vs. portfolio), (c) rollout-count framing, (d)
        repo-memory contamination, (e) external-evidence access. This
        block surfaces all five at the top of the report so a reviewer
        does not have to reverse-engineer the run from per-task metadata.

        Derived from existing report fields only — no runner support
        needed, so older reports that pre-date this method still produce
        a well-formed (if more sparse) block.
        """
        unique_models = sorted(
            {
                str((entry or {}).get("model") or "").strip()
                for entry in self.model_config or []
                if (entry or {}).get("model")
            }
        )
        model_class = "single" if len(unique_models) <= 1 else "portfolio"

        f2p_enabled_tasks = 0
        mutation_attempted_tasks = 0
        mutation_evaluated_tasks = 0
        effective_mutation_evaluable_total = 0
        effective_mutation_evaluable_tasks = 0
        assertion_mutation_attempted_tasks = 0
        assertion_mutation_measured_tasks = 0
        assertion_mutation_survived_tasks = 0
        minimization_applied_tasks = 0
        repo_memory_enabled_tasks = 0
        repo_memory_disabled_via_env_tasks = 0
        # rollout_count is a per-run choice; the benchmark canonical is 1
        # but APEX may run more for best-of-N selection. Surfacing both
        # the per-task observed count (from execution_metadata) and the
        # canonical lets a reviewer cross-check.
        rollout_counts: list[int] = []
        for task in self.tasks:
            metadata = dict(task.execution_metadata or {})
            if isinstance(metadata.get("rollout_count"), int):
                rollout_counts.append(int(metadata["rollout_count"]))
            if metadata.get("f2p_enabled"):
                f2p_enabled_tasks += 1
            f2p_summary = dict(metadata.get("f2p_summary") or {})
            if _f2p_summary_mutation_attempted(f2p_summary):
                mutation_attempted_tasks += 1
            if _f2p_summary_mutation_measured(f2p_summary):
                mutation_evaluated_tasks += 1
                effective_evaluable = int(
                    f2p_summary.get("effective_mutation_evaluable")
                    if f2p_summary.get("effective_mutation_evaluable") is not None
                    else f2p_summary.get("mutation_total") or 0
                )
                effective_mutation_evaluable_total += effective_evaluable
                if effective_evaluable > 0:
                    effective_mutation_evaluable_tasks += 1
            if bool(f2p_summary.get("assertion_mutation_attempted")):
                assertion_mutation_attempted_tasks += 1
            if bool(f2p_summary.get("assertion_mutation_measured")):
                assertion_mutation_measured_tasks += 1
                if bool(f2p_summary.get("assertion_mutation_survived")):
                    assertion_mutation_survived_tasks += 1
            if "minimized_count" in f2p_summary:
                minimization_applied_tasks += 1
            repo_memory_summary = dict(metadata.get("repo_memory_summary") or {})
            if repo_memory_summary.get("enabled"):
                repo_memory_enabled_tasks += 1
            if repo_memory_summary.get("disabled_via_env"):
                repo_memory_disabled_via_env_tasks += 1

        rollout_count_min = min(rollout_counts) if rollout_counts else None
        rollout_count_max = max(rollout_counts) if rollout_counts else None

        # Phase G feature surface — these are wired into the test_writer
        # path and run unconditionally where they have signal. Surfacing
        # them in the methodology block lets a reviewer cross-check
        # which loop-level signals were available for the run.
        memory_per_task_summary = {
            str(task_id): dict(summary)
            for task_id, summary in dict(self.testgen_memory_per_task_summary or {}).items()
            if isinstance(summary, dict)
        }
        memory_enabled = bool(self.enable_testgen_memory)
        phase_g_features_wired = {
            "iteration_axis_coverage_feedback": True,  # G.0
            "iteration_f2p_feedback": True,  # F.2 (predecessor of G)
            "in_loop_mutation_sensitivity": (
                os.environ.get("APEX_DISABLE_ITERATION_MUTATION", "").strip().lower()
                not in {"1", "true", "yes", "on", "enabled"}
            ),  # G.1 / I.1 (always-on, opt-OUT)
            "in_loop_coverage_gaps": (
                os.environ.get("APEX_DISABLE_ITERATION_COVERAGE", "").strip().lower()
                not in {"1", "true", "yes", "on", "enabled"}
            ),  # I.3 (always-on, opt-OUT)
            "in_loop_assertion_mutation": _iteration_assertion_mutation_enabled(),
            "cross_rollout_mutation_kill_sharing": True,  # I.2
            "cross_task_testgen_memory": {
                # Phase I.7: opt-IN for benchmarks (contamination risk).
                # Per-run isolated directory by default so cross-
                # benchmark-run contamination is impossible.
                "enabled": memory_enabled,
                "directory": self.testgen_memory_directory if memory_enabled else None,
                "per_task_summary": memory_per_task_summary,
                "tasks_with_loaded_insights": sum(
                    1
                    for v in memory_per_task_summary.values()
                    if isinstance(v, dict) and int(v.get("insights_loaded") or 0) > 0
                ),
                "tasks_with_persisted_insights": sum(
                    1
                    for v in memory_per_task_summary.values()
                    if isinstance(v, dict) and int(v.get("insights_persisted") or 0) > 0
                ),
            },
            "repo_test_exemplars": True,  # G.2
            "predicted_edges_in_submission_schema": True,  # G.3
            "testgen_rollout_morphs": True,  # G.5
            "property_metamorphic_prompt_guidance": True,  # G.6
            "cross_rollout_testgen_learning": True,  # G.9
        }

        return {
            "tasks_evaluated_count": int(self.total_tasks),
            "tasks_requested_count": len(self.requested_task_ids),
            "repo_count": len(self.repo_names),
            "evaluation_mode": self.evaluation_mode,
            "model_class": model_class,
            "model_count": len(unique_models),
            "model_ids": unique_models,
            "rollout_count_min": rollout_count_min,
            "rollout_count_max": rollout_count_max,
            "rollout_count_canonical": 1,  # SWE-Bench Pro testgen canonical
            "f2p_enabled_tasks": f2p_enabled_tasks,
            "allow_gold_oracle_selection": self.allow_gold_oracle_selection,
            "mutation_attempted_tasks": mutation_attempted_tasks,
            "mutation_measured_tasks": mutation_evaluated_tasks,
            "mutation_evaluated_tasks": mutation_evaluated_tasks,
            "effective_mutation_evaluable_total": effective_mutation_evaluable_total,
            "effective_mutation_evaluable_tasks": effective_mutation_evaluable_tasks,
            "assertion_mutation_attempted_tasks": assertion_mutation_attempted_tasks,
            "assertion_mutation_measured_tasks": assertion_mutation_measured_tasks,
            "assertion_mutation_survived_tasks": assertion_mutation_survived_tasks,
            "minimization_applied_tasks": minimization_applied_tasks,
            "repo_memory_enabled_tasks": repo_memory_enabled_tasks,
            "repo_memory_disabled_via_env_tasks": repo_memory_disabled_via_env_tasks,
            "phase_g_features_wired": phase_g_features_wired,
        }

    def to_markdown(self) -> str:
        metrics = self.aggregate_metrics
        status = "completed" if self.completed else "in_progress"
        skipped_tasks = [task for task in self.tasks if task.skipped]
        methodology = self.methodology_block()
        memory_state = dict(
            (methodology.get("phase_g_features_wired") or {}).get("cross_task_testgen_memory") or {}
        )
        lines = [
            "# APEX SWE-Bench Pro Testgen Evaluation",
            "",
            "## Methodology",
            "",
            (
                f"- Subset: {methodology['tasks_evaluated_count']} task(s) evaluated "
                f"of {methodology['tasks_requested_count']} requested "
                f"across {methodology['repo_count']} repo(s)"
            ),
            f"- Evaluation mode: {methodology['evaluation_mode']}",
            (
                f"- Model class: {methodology['model_class']} "
                f"({methodology['model_count']} unique model(s): "
                f"{', '.join(methodology['model_ids']) or 'unknown'})"
            ),
            (
                "- Rollout count: "
                + (
                    f"{methodology['rollout_count_min']}"
                    if (
                        methodology.get("rollout_count_min") is not None
                        and methodology.get("rollout_count_min")
                        == methodology.get("rollout_count_max")
                    )
                    else (
                        f"{methodology['rollout_count_min']}-{methodology['rollout_count_max']}"
                        if methodology.get("rollout_count_min") is not None
                        else "unknown"
                    )
                )
                + f" per task (canonical: {methodology['rollout_count_canonical']})"
            ),
            (
                f"- F2P oracle: enabled on {methodology['f2p_enabled_tasks']} "
                f"of {methodology['tasks_evaluated_count']} task(s)"
            ),
            (
                f"- Mutation discrimination: measured on "
                f"{methodology['mutation_measured_tasks']} task(s)"
                + (
                    f", attempted on {methodology['mutation_attempted_tasks']} task(s)"
                    if methodology.get("mutation_attempted_tasks")
                    != methodology.get("mutation_measured_tasks")
                    else ""
                )
                + (
                    f"; effective evaluable mutants/tests: "
                    f"{methodology['effective_mutation_evaluable_total']} "
                    f"across {methodology['effective_mutation_evaluable_tasks']} task(s)"
                    if methodology.get("effective_mutation_evaluable_total")
                    else ""
                )
            ),
            (
                f"- Minimization (Stage 5): applied on "
                f"{methodology['minimization_applied_tasks']} task(s)"
            ),
            (
                f"- Assertion mutation: measured on "
                f"{methodology['assertion_mutation_measured_tasks']} task(s), "
                f"survived on {methodology['assertion_mutation_survived_tasks']} task(s)"
            ),
            (
                f"- Repo memory: "
                f"{'enabled' if memory_state.get('enabled') else 'disabled'}; "
                f"loaded on {memory_state.get('tasks_with_loaded_insights', 0)} task(s), "
                f"persisted on {memory_state.get('tasks_with_persisted_insights', 0)} task(s)"
                + (
                    f" ({methodology['repo_memory_disabled_via_env_tasks']} forced off via "
                    "APEX_DISABLE_REPO_MEMORY)"
                    if methodology["repo_memory_disabled_via_env_tasks"]
                    else ""
                )
            ),
            (
                "- Phase G features wired: "
                + ", ".join(
                    f"{k}={'✓' if v else '✗'}"
                    for k, v in (methodology.get("phase_g_features_wired") or {}).items()
                )
                if methodology.get("phase_g_features_wired")
                else ""
            ),
            "",
            "## Run",
            "",
            f"- Harness: {self.harness_name} v{self.harness_version}",
            f"- Status: {status}",
            f"- Config source: {self.config_source or 'default'}",
            f"- Model config: {json.dumps(self.model_config)}",
            f"- Evaluation mode: {self.evaluation_mode}",
            f"- Repos: {', '.join(self.repo_names) or 'none'}",
            f"- Completed tasks: {self.completed_tasks}/{self.total_tasks}",
            f"- Runnable tasks: {self.runnable_tasks}",
            f"- Successful comparisons: {self.successful_tasks}",
            f"- Failed tasks: {self.failed_tasks}",
            f"- Skipped tasks: {self.skipped_tasks}",
            f"- Mean contract-axis recall: {100.0 * metrics.get('mean_overall_contract_axis_recall', 0.0):.1f}%",
            f"- Median contract-axis recall: {100.0 * metrics.get('median_overall_contract_axis_recall', 0.0):.1f}%",
            f"- Mean gold-target recall: {100.0 * metrics.get('mean_gold_target_recall', 0.0):.1f}%",
            f"- Mean generated contract-target coverage: {100.0 * metrics.get('mean_generated_contract_target_coverage_ratio', 0.0):.1f}%",
            (
                "- Mean gold field-path recall: "
                f"{100.0 * metrics.get('mean_gold_field_path_recall', 0.0):.1f}% "
                f"across {metrics.get('tasks_with_gold_field_paths', 0)} tasks"
            ),
            (
                "- Mean gold field-path shape recall: "
                f"{100.0 * metrics.get('mean_gold_field_path_shape_recall', 0.0):.1f}% "
                f"across {metrics.get('tasks_with_gold_field_path_shapes', 0)} tasks"
            ),
            f"- Mean generated negative-shape coverage ratio: {100.0 * metrics.get('mean_generated_field_path_negative_shape_coverage_ratio', 0.0):.1f}%",
        ]
        if int(metrics.get("test_quality_tasks", 0) or 0) > 0:
            lines.extend(
                [
                    (
                        "- Test-quality checked artifacts: "
                        f"{metrics.get('test_quality_artifact_count', 0)} "
                        f"across {metrics.get('test_quality_tasks', 0)} task(s)"
                    ),
                    (
                        "- Weak generated test artifacts: "
                        f"{metrics.get('test_quality_weak_artifact_count', 0)} "
                        f"({100.0 * metrics.get('test_quality_weak_artifact_rate', 0.0):.1f}%)"
                    ),
                    (
                        "- Mean assertion-effect score: "
                        f"{100.0 * metrics.get('mean_test_quality_assertion_effect_score', 0.0):.1f}%"
                    ),
                ]
            )
        if int(metrics.get("semantic_review_tasks", 0) or 0) > 0:
            lines.extend(
                [
                    f"- Semantic review tasks: {metrics.get('semantic_review_tasks', 0)}",
                    f"- Mean semantic-review confidence: {100.0 * metrics.get('mean_semantic_review_confidence', 0.0):.1f}%",
                    (
                        "- Mean semantic strict behavioral recall: "
                        f"{100.0 * metrics.get('mean_semantic_review_strict_behavioral_recall', 0.0):.1f}%"
                    ),
                    (
                        "- Mean semantic lenient behavioral recall: "
                        f"{100.0 * metrics.get('mean_semantic_review_lenient_behavioral_recall', 0.0):.1f}%"
                    ),
                    (
                        "- Semantic tasks with no material gaps: "
                        f"{metrics.get('semantic_review_tasks_with_no_material_gaps', 0)}"
                    ),
                    (
                        "- Semantic tasks with no weaker assertions: "
                        f"{metrics.get('semantic_review_tasks_with_no_weaker_assertions', 0)}"
                    ),
                ]
            )
        lines.extend(
            [
                f"- Duration: {self.duration_seconds:.1f}s",
                "",
                "## Common Gaps",
                "",
                "| Gap Type | Count |",
                "| --- | --- |",
            ]
        )
        has_common_gaps = False
        for entry in list(metrics.get("top_missing_axes") or []):
            lines.append(f"| missing axis: {entry.get('name')} | {entry.get('count')} |")
            has_common_gaps = True
        for entry in list(metrics.get("top_missing_gold_field_path_shapes") or []):
            lines.append(
                f"| missing field-path shape: {entry.get('name')} | {entry.get('count')} |"
            )
            has_common_gaps = True
        for entry in list(metrics.get("top_test_quality_issues") or []):
            lines.append(f"| weak test issue: {entry.get('name')} | {entry.get('count')} |")
            has_common_gaps = True
        if not has_common_gaps:
            lines.append("| none | 0 |")
        lines.extend(
            [
                "",
                "## Lowest Recall Tasks",
                "",
                "| Instance | Repo | Axis Recall | Target Recall | Field-Path Recall | Shape Recall | Artifacts | Tokens | Duration (s) |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for entry in list(metrics.get("lowest_recall_tasks") or []):
            lines.append(
                "| {instance_id} | {repo} | {axis:.1f}% | {target:.1f}% | {field:.1f}% | {shape:.1f}% | {artifacts} | {tokens} | {duration:.1f} |".format(
                    instance_id=entry.get("instance_id"),
                    repo=entry.get("repo"),
                    axis=100.0 * float(entry.get("overall_contract_axis_recall") or 0.0),
                    target=100.0 * float(entry.get("gold_target_recall") or 0.0),
                    field=100.0 * float(entry.get("gold_field_path_recall") or 0.0),
                    shape=100.0 * float(entry.get("gold_field_path_shape_recall") or 0.0),
                    artifacts=entry.get("generated_artifact_count"),
                    tokens=entry.get("total_tokens"),
                    duration=float(entry.get("duration_seconds") or 0.0),
                )
            )
        if skipped_tasks:
            lines.extend(
                [
                    "",
                    "## Skipped Tasks",
                    "",
                    "| Instance | Repo | Category | Reason |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for task in skipped_tasks:
                lines.append(
                    "| {instance_id} | {repo} | {category} | {reason} |".format(
                        instance_id=task.instance_id,
                        repo=task.repo,
                        category=task.skip_category or "unclassified",
                        reason=(task.failure_reason or "").replace("\n", " ").strip() or "n/a",
                    )
                )
        return "\n".join(lines)


class SWEBenchProTestGenEvaluator(SWEBenchProHarness):
    def __init__(
        self,
        config: ApexConfig,
        output_dir: str | Path,
        *,
        dataset_name: str = SWEBENCH_PRO_DATASET_NAME,
        dataset_split: str = SWEBENCH_PRO_DATASET_SPLIT,
        dockerhub_username: str = SWEBENCH_PRO_DOCKERHUB_USERNAME,
        scripts_cache_dir: str | Path | None = None,
        docker_platform: Optional[str] = None,
        block_network: bool = False,
        rollout_count: int = 1,
        prepare_repo_mode: str = SWEBENCH_PREPARE_REPO_FROM_DOCKER_IMAGE,
        semantic_review: bool = False,
        semantic_review_model: Optional[str] = None,
        enable_f2p: bool = False,
        f2p_install_repo: bool = True,
        f2p_per_side_timeout_seconds: float = 300.0,
        enable_mutation: bool = False,
        mutation_max_per_file: int = 16,
        mutation_max_files: int = 3,
        mutation_per_mutant_timeout_seconds: float = 60.0,
        enable_minimization: bool = False,
        enable_testgen_judge: bool = False,
        enable_testgen_memory: bool = False,
        testgen_memory_directory: Optional[str] = None,
        enable_testgen_delegation: bool = False,
        allow_gold_oracle_selection: bool = False,
    ) -> None:
        super().__init__(
            output_dir=output_dir,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            dockerhub_username=dockerhub_username,
            scripts_cache_dir=scripts_cache_dir,
            docker_platform=docker_platform,
            block_network=block_network,
            agent_visibility_mode=SWEBENCH_AGENT_VISIBILITY_PUBLISHED_PARITY,
            prepare_repo_mode=prepare_repo_mode,
        )
        self.config = copy.deepcopy(config)
        self.enable_testgen_delegation = bool(enable_testgen_delegation)
        if not self.enable_testgen_delegation:
            self.config.rollout.enable_orchestrated_multi_agent = False
        self.config_source: Optional[str] = None
        self.rollout_count = max(1, int(rollout_count or 1))
        self.semantic_review = bool(semantic_review)
        self.semantic_review_model = str(semantic_review_model or "").strip() or None
        self.enable_f2p = bool(enable_f2p)
        self.allow_gold_oracle_selection = bool(allow_gold_oracle_selection)
        self.f2p_install_repo = bool(f2p_install_repo)
        self.f2p_per_side_timeout_seconds = max(30.0, float(f2p_per_side_timeout_seconds or 300.0))
        # Mutation discrimination is meaningful only on candidates whose tests
        # already catch the bug (any_f2p=True), so it implicitly requires
        # enable_f2p. We still permit setting the flag independently — the
        # per-task code short-circuits when F2P is disabled.
        self.enable_mutation = bool(enable_mutation)
        self.mutation_max_per_file = max(1, int(mutation_max_per_file or 16))
        self.mutation_max_files = max(1, int(mutation_max_files or 3))
        self.mutation_per_mutant_timeout_seconds = max(
            5.0, float(mutation_per_mutant_timeout_seconds or 60.0)
        )
        # Minimization is opt-in and only meaningful with at least F2P
        # signal — without coverage data the greedy short-circuits to
        # "keep all" so leaving enable_minimization on without F2P costs
        # nothing but adds noise to the comparison JSON.
        self.enable_minimization = bool(enable_minimization)
        # Phase E.3: opt-in LLM judge for breaking F2P-tuple ties among
        # multi-rollout candidates. Implies --enable-f2p (the judge runs
        # only on the F2P top tier). Without enable_testgen_judge=True
        # the selector falls through to the heuristic comparator on ties.
        self.enable_testgen_judge = bool(enable_testgen_judge)
        # Phase I.7: cross-task persistent testgen memory (opt-IN for
        # benchmarks because of contamination risk — insights from
        # prior tasks would leak benchmark-derived signal across the
        # eval set if always-on). Default off; when on, we ALWAYS
        # use a per-run isolated directory so cross-benchmark-run
        # contamination is impossible. The user can override the
        # directory if they explicitly want shared memory across
        # runs (e.g., real-world TDD use).
        self.enable_testgen_memory = bool(enable_testgen_memory)
        self.testgen_memory_directory: Optional[str] = (
            str(testgen_memory_directory).strip()
            if testgen_memory_directory and str(testgen_memory_directory).strip()
            else None
        )
        # Track per-task memory operations for the methodology block
        # so reviewers can see which tasks loaded prior insights and
        # which persisted new ones.
        self._testgen_memory_per_task_summary: dict[str, dict[str, int]] = {}
        self.config.rollout.num_rollouts = self.rollout_count
        self.config.rollout.min_rollouts = min(
            self.config.rollout.min_rollouts,
            self.rollout_count,
        )
        self.config.rollout.max_rollouts = max(
            self.config.rollout.min_rollouts,
            self.rollout_count,
        )
        self.config.rollout.parallel_workers = 1
        self.config.selection.cross_validation_enabled = True

    def _semantic_review_timeout_seconds(self) -> int:
        primary = self.config.llm_configs[0]
        upper_bounds: list[int] = []
        for value in (primary.cli_timeout, primary.timeout):
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                upper_bounds.append(parsed)
        if not upper_bounds:
            return 300
        return max(120, min(max(upper_bounds), 600))

    def _build_semantic_review_config(self) -> LLMConfig:
        primary = self.config.llm_configs[0]
        selected_model = (
            self.semantic_review_model or self.config.selection.judge_model or primary.model
        )
        review_timeout_seconds = self._semantic_review_timeout_seconds()
        bounded_timeout = max(60, min(int(primary.timeout), review_timeout_seconds))
        bounded_cli_timeout = max(60, min(int(primary.cli_timeout), review_timeout_seconds))
        hard_timeout_cap = bounded_cli_timeout + 60
        existing_hard_timeout = (
            int(primary.cli_hard_timeout_seconds)
            if primary.cli_hard_timeout_seconds is not None
            else hard_timeout_cap
        )
        bounded_hard_timeout = max(
            bounded_cli_timeout,
            min(existing_hard_timeout, hard_timeout_cap),
        )
        return LLMConfig(
            model=selected_model,
            backend=primary.backend,
            api_key_env=primary.api_key_env,
            base_url=primary.base_url,
            temperature=self.config.selection.judge_temperature,
            max_tokens=primary.max_tokens,
            timeout=bounded_timeout,
            cli_command=primary.cli_command,
            cli_args=list(primary.cli_args),
            cli_model_id=primary.cli_model_id if selected_model == primary.model else None,
            cli_timeout=bounded_cli_timeout,
            cli_hard_timeout_seconds=bounded_hard_timeout,
            cli_disable_osx_sandbox=primary.cli_disable_osx_sandbox,
            cli_permission_mode=primary.cli_permission_mode,
            cli_env_overrides=dict(primary.cli_env_overrides),
            cli_env_redaction_disabled=primary.cli_env_redaction_disabled,
        )

    def _build_semantic_review_packet(
        self,
        task: SWEBenchProTask,
        *,
        comparison: dict[str, Any],
        issue_description: Optional[str] = None,
        required_contract_targets: Optional[list[str]] = None,
        test_command: Optional[str] = None,
    ) -> dict[str, Any]:
        effective_test_command = str(test_command or "").strip() or None
        effective_issue_description = issue_description
        if effective_issue_description is None:
            effective_issue_description = task.build_issue_description(
                effective_test_command,
                include_benchmark_guardrails=False,
                include_benchmark_metadata=False,
                include_selected_test_targets=False,
                include_required_tests=False,
            )
        targets = [
            str(item).strip() for item in list(required_contract_targets or []) if str(item).strip()
        ]
        if not targets:
            targets = extract_issue_contract_targets(effective_issue_description)
        if not targets:
            targets = [
                str(item).strip()
                for item in list(
                    dict(comparison.get("generated_summary") or {}).get("targets") or []
                )
                if str(item).strip()
            ]
        if not targets:
            targets = [
                str(item).strip()
                for item in list(dict(comparison.get("gold_summary") or {}).get("targets") or [])
                if str(item).strip()
            ]
        return {
            "instance_id": task.instance_id,
            "repo": task.repo,
            "issue_description": effective_issue_description,
            "required_contract_targets": targets,
        }

    def _apply_semantic_review_to_comparison(
        self,
        *,
        task: SWEBenchProTask,
        comparison: dict[str, Any],
        working_dir: str,
        issue_description: Optional[str] = None,
        required_contract_targets: Optional[list[str]] = None,
        test_command: Optional[str] = None,
    ) -> tuple[dict[str, Any], Optional[str], int]:
        updated_comparison = dict(comparison or {})
        if not self.semantic_review:
            return updated_comparison, None, 0

        generated_portfolio = dict(updated_comparison.get("generated_portfolio") or {})
        gold_portfolio = dict(updated_comparison.get("gold_portfolio") or {})
        if not generated_portfolio or not gold_portfolio:
            error = (
                "Comparison payload is missing generated or gold portfolios for semantic review."
            )
            updated_comparison["semantic_review_error"] = error
            return updated_comparison, error, 0

        try:
            semantic_review = review_generated_vs_gold_test_semantics(
                packet=self._build_semantic_review_packet(
                    task,
                    comparison=updated_comparison,
                    issue_description=issue_description,
                    required_contract_targets=required_contract_targets,
                    test_command=test_command,
                ),
                generated_portfolio=generated_portfolio,
                gold_portfolio=gold_portfolio,
                judge_config=self._build_semantic_review_config(),
                working_dir=working_dir,
            )
            updated_comparison = attach_semantic_review_to_comparison(
                updated_comparison,
                semantic_review,
            )
            updated_comparison.pop("semantic_review_error", None)
            semantic_review_tokens = int(
                dict(updated_comparison.get("semantic_review") or {}).get("judge_total_tokens") or 0
            )
            return updated_comparison, None, semantic_review_tokens
        except Exception as exc:
            error = str(exc)
            updated_comparison["semantic_review_error"] = error
            logger.warning(
                "Semantic review failed for %s: %s",
                task.instance_id,
                exc,
            )
            return updated_comparison, error, 0

    def _backfill_checkpointed_semantic_review(
        self,
        task: SWEBenchProTask,
        result: SWEBenchProTestGenTaskResult,
    ) -> SWEBenchProTestGenTaskResult:
        if not self.semantic_review or not result.success or dict(result.semantic_review):
            return result

        comparison_path = Path(
            str(
                result.result_path
                or (self._task_output_dir(task) / "synthetic_vs_gold_comparison.json")
            )
        )
        comparison = load_json_if_exists(comparison_path)
        updated_result = SWEBenchProTestGenTaskResult.from_dict(result.to_dict())
        execution_metadata = dict(updated_result.execution_metadata)
        execution_metadata["semantic_review"] = True
        execution_metadata["semantic_review_model"] = (
            self.semantic_review_model or self.config.selection.judge_model or ""
        )
        updated_result.execution_metadata = execution_metadata

        if comparison is None:
            updated_result.semantic_review_error = (
                f"Missing comparison payload for semantic review: {comparison_path}"
            )
            write_task_checkpoint(self._task_output_dir(task), updated_result.to_dict())
            return updated_result

        updated_comparison, semantic_review_error, semantic_review_tokens = (
            self._apply_semantic_review_to_comparison(
                task=task,
                comparison=comparison,
                working_dir=str(self.project_root),
                required_contract_targets=list(updated_result.issue_contract_targets),
                test_command=str(execution_metadata.get("test_command") or ""),
            )
        )
        atomic_write_json(comparison_path, updated_comparison)
        updated_result.generated_summary = dict(
            updated_comparison.get("generated_summary") or updated_result.generated_summary
        )
        updated_result.gold_summary = dict(
            updated_comparison.get("gold_summary") or updated_result.gold_summary
        )
        updated_result.coverage_summary = dict(
            updated_comparison.get("coverage_summary") or updated_result.coverage_summary
        )
        updated_result.target_comparison = list(
            updated_comparison.get("target_comparison") or updated_result.target_comparison
        )
        updated_result.semantic_review = dict(updated_comparison.get("semantic_review") or {})
        updated_result.semantic_review_error = semantic_review_error
        updated_result.result_path = str(comparison_path)
        updated_result.total_tokens = int(updated_result.total_tokens or 0) + int(
            semantic_review_tokens or 0
        )
        write_task_checkpoint(self._task_output_dir(task), updated_result.to_dict())
        return updated_result

    def run(
        self,
        *,
        instances: Optional[list[str]] = None,
        repos: Optional[list[str]] = None,
        languages: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> SWEBenchProTestGenReport:
        ensure_cli_process_cleanup_hooks()
        tasks = self.discover_tasks(
            instances=instances,
            repos=repos,
            languages=languages,
            limit=limit,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        requested_task_ids = [task.instance_id for task in tasks]
        existing_state = load_json_if_exists(self.output_dir / RUN_STATE_FILENAME) or {}
        report = SWEBenchProTestGenReport(
            requested_task_ids=requested_task_ids,
            requested_repo_names=sorted({task.repo for task in tasks}),
            started_at=float(existing_state.get("started_at") or time.time()),
            dataset_name=self.dataset_name,
            dataset_split=self.dataset_split,
            config_source=self.config_source,
            model_config=serialize_llm_configs(self.config),
            ablation_config=build_apex_ablation_config(self.config),
            enable_testgen_memory=self.enable_testgen_memory,
            allow_gold_oracle_selection=self.allow_gold_oracle_selection,
            testgen_memory_directory=self._testgen_memory_directory_for_repo(""),
        )
        report.run_manifest = ensure_run_manifest(
            self.output_dir,
            build_run_manifest(
                config=self.config,
                report_kind=report.report_kind,
                harness_name=report.harness_name,
                harness_version=report.harness_version,
                benchmark_family="swebench_pro_testgen",
                output_dir=self.output_dir,
                config_source=self.config_source,
                requested_task_ids=requested_task_ids,
                execution={
                    "entrypoint": "swebench-pro-testgen-eval",
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
                        "rollout_count": self.rollout_count,
                        "task_parallelism": self.config.benchmark.task_parallelism,
                        "prepare_repo_mode": self.prepare_repo_mode,
                        "semantic_review": self.semantic_review,
                        "semantic_review_model": self.semantic_review_model,
                        "enable_f2p": self.enable_f2p,
                        "f2p_install_repo": self.f2p_install_repo,
                        "f2p_per_side_timeout_seconds": self.f2p_per_side_timeout_seconds,
                        "enable_mutation": self.enable_mutation,
                        "mutation_max_per_file": self.mutation_max_per_file,
                        "mutation_max_files": self.mutation_max_files,
                        "mutation_per_mutant_timeout_seconds": (
                            self.mutation_per_mutant_timeout_seconds
                        ),
                        "enable_minimization": self.enable_minimization,
                        "enable_testgen_judge": self.enable_testgen_judge,
                        "enable_testgen_memory": self.enable_testgen_memory,
                        "testgen_memory_directory": self.testgen_memory_directory,
                        "enable_testgen_delegation": self.enable_testgen_delegation,
                        "testgen_task_timeout_seconds": (
                            self.config.benchmark.testgen_task_timeout_seconds
                        ),
                    },
                },
                extra_settings={
                    "dataset_name": self.dataset_name,
                    "dataset_split": self.dataset_split,
                    "rollout_count": self.rollout_count,
                    "evaluation_mode": report.evaluation_mode,
                    "prepare_repo_mode": self.prepare_repo_mode,
                    "semantic_review": self.semantic_review,
                    "semantic_review_model": self.semantic_review_model,
                    "enable_f2p": self.enable_f2p,
                    "f2p_install_repo": self.f2p_install_repo,
                    "f2p_per_side_timeout_seconds": self.f2p_per_side_timeout_seconds,
                    "enable_mutation": self.enable_mutation,
                    "mutation_max_per_file": self.mutation_max_per_file,
                    "mutation_max_files": self.mutation_max_files,
                    "mutation_per_mutant_timeout_seconds": (
                        self.mutation_per_mutant_timeout_seconds
                    ),
                    "enable_minimization": self.enable_minimization,
                    "enable_testgen_judge": self.enable_testgen_judge,
                    "enable_testgen_memory": self.enable_testgen_memory,
                    "testgen_memory_directory": self.testgen_memory_directory,
                    "enable_testgen_delegation": self.enable_testgen_delegation,
                    "testgen_task_timeout_seconds": (
                        self.config.benchmark.testgen_task_timeout_seconds
                    ),
                },
                benchmark_policy=_build_testgen_benchmark_policy(
                    docker_platform=self.docker_platform,
                    block_network=self.block_network,
                    prepare_repo_mode=self.prepare_repo_mode,
                ),
            ),
        )

        completed_results: dict[str, SWEBenchProTestGenTaskResult] = {}
        pending_tasks: list[SWEBenchProTask] = []
        ordered_instance_ids = [task.instance_id for task in tasks]
        for task in tasks:
            checkpointed = self._load_checkpointed_task_result(task)
            if checkpointed is not None:
                checkpointed = self._backfill_checkpointed_semantic_review(task, checkpointed)
                completed_results[task.instance_id] = checkpointed
            else:
                pending_tasks.append(task)

        def refresh_report_tasks() -> None:
            report.tasks = [
                completed_results[instance_id]
                for instance_id in ordered_instance_ids
                if instance_id in completed_results
            ]

        refresh_report_tasks()
        self._write_report_checkpoint(report, requested_task_ids, completed=False)
        prior_systemic_skip = next(
            (
                task_result
                for task_result in report.tasks
                if task_result.skipped
                and task_result.skip_category in SYSTEMIC_TESTGEN_SKIP_CATEGORIES
            ),
            None,
        )
        if prior_systemic_skip is not None:
            logger.error(
                "Found prior systemic SWE-Bench Pro testgen environment failure on %s (%s); not resuming remaining tasks.",
                prior_systemic_skip.instance_id,
                prior_systemic_skip.skip_category,
            )
            self._write_report_checkpoint(report, requested_task_ids, completed=True)
            return report

        max_workers = self._task_worker_limit(len(pending_tasks))
        systemic_skip_detected = False
        if max_workers == 1:
            for task in pending_tasks:
                result = self._run_task_benchmark_safe(task)
                completed_results[task.instance_id] = result
                refresh_report_tasks()
                self._write_report_checkpoint(report, requested_task_ids, completed=False)
                if result.skipped and result.skip_category in SYSTEMIC_TESTGEN_SKIP_CATEGORIES:
                    logger.error(
                        "Aborting remaining SWE-Bench Pro testgen tasks after systemic environment failure on %s (%s): %s",
                        task.instance_id,
                        result.skip_category,
                        result.failure_reason or "no failure reason recorded",
                    )
                    systemic_skip_detected = True
                    break
        else:
            pending_iter = iter(pending_tasks)

            def submit_next(
                executor: ThreadPoolExecutor,
                future_map: dict[Any, SWEBenchProTask],
            ) -> bool:
                try:
                    task = next(pending_iter)
                except StopIteration:
                    return False
                future = executor.submit(self._run_task_benchmark_safe, task)
                future_map[future] = task
                return True

            with _interruptible_thread_pool(max_workers) as executor:
                future_map: dict[Any, SWEBenchProTask] = {}
                for _ in range(min(max_workers, len(pending_tasks))):
                    submit_next(executor, future_map)
                while future_map:
                    future = next(as_completed(list(future_map)))
                    task = future_map.pop(future)
                    result = future.result()
                    completed_results[task.instance_id] = result
                    refresh_report_tasks()
                    self._write_report_checkpoint(report, requested_task_ids, completed=False)
                    if result.skipped and result.skip_category in SYSTEMIC_TESTGEN_SKIP_CATEGORIES:
                        logger.error(
                            "Aborting remaining SWE-Bench Pro testgen tasks after systemic environment failure on %s (%s): %s",
                            task.instance_id,
                            result.skip_category,
                            result.failure_reason or "no failure reason recorded",
                        )
                        systemic_skip_detected = True
                    if not systemic_skip_detected:
                        submit_next(executor, future_map)

        self._write_report_checkpoint(report, requested_task_ids, completed=True)
        return report

    def _run_task_with_checkpoint(
        self,
        task: SWEBenchProTask,
    ) -> SWEBenchProTestGenTaskResult:
        ensure_clean_directory_for_task(self._task_output_dir(task), completed=False)
        ensure_clean_directory_for_task(self._task_workspace_dir(task), completed=False)
        result = self._run_task(task)
        write_task_checkpoint(self._task_output_dir(task), result.to_dict())
        return result

    def _task_timeout_seconds(self) -> float:
        try:
            return float(self.config.benchmark.testgen_task_timeout_seconds or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _run_task_benchmark_safe(
        self,
        task: SWEBenchProTask,
    ) -> SWEBenchProTestGenTaskResult:
        timeout_seconds = self._task_timeout_seconds()
        if timeout_seconds <= 0:
            return self._run_task_with_checkpoint(task)
        return self._run_task_with_checkpoint_in_subprocess(
            task,
            timeout_seconds=timeout_seconds,
        )

    def _terminate_task_process(
        self,
        process: multiprocessing.Process,
        *,
        grace_seconds: float = 5.0,
    ) -> None:
        if not process.is_alive():
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass
        process.join(timeout=max(0.1, grace_seconds))
        if not process.is_alive():
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
        process.join(timeout=1.0)

    def _timeout_task_result(
        self,
        task: SWEBenchProTask,
        *,
        timeout_seconds: float,
        started_at: float,
        child_pid: Optional[int] = None,
    ) -> SWEBenchProTestGenTaskResult:
        task_output_dir = self._task_output_dir(task)
        result = SWEBenchProTestGenTaskResult(
            instance_id=task.instance_id,
            repo=task.repo,
            success=False,
            duration_seconds=time.time() - started_at,
            failure_reason=(f"Task exceeded strict benchmark timeout of {timeout_seconds:.1f}s"),
            skipped=True,
            skip_category=SWEBENCH_PRO_TESTGEN_TASK_TIMEOUT_SKIP_CATEGORY,
            execution_metadata={
                "task_timeout_seconds": timeout_seconds,
                "task_process_pid": child_pid,
                "timeout_kind": "strict_task_wallclock",
            },
        )
        write_task_live_state(
            task_output_dir,
            {
                "task_id": task.instance_id,
                "phase": "timeout",
                "status": "failed",
                "process_pid": os.getpid(),
                "child_process_pid": child_pid,
                "last_progress_at": time.time(),
                "skipped": True,
                "skip_category": SWEBENCH_PRO_TESTGEN_TASK_TIMEOUT_SKIP_CATEGORY,
                "timeout_seconds": timeout_seconds,
                "error": result.failure_reason,
            },
        )
        payload = result.to_dict()
        atomic_write_json(task_output_dir / "task_timeout_checkpoint.json", payload)
        write_task_checkpoint(task_output_dir, payload)
        return result

    def _run_task_with_checkpoint_in_subprocess(
        self,
        task: SWEBenchProTask,
        *,
        timeout_seconds: float,
    ) -> SWEBenchProTestGenTaskResult:
        ensure_clean_directory_for_task(self._task_output_dir(task), completed=False)
        ensure_clean_directory_for_task(self._task_workspace_dir(task), completed=False)
        started = time.time()
        try:
            ctx = multiprocessing.get_context(
                "fork"
                if "fork" in multiprocessing.get_all_start_methods()
                else multiprocessing.get_start_method()
            )
        except Exception:  # pragma: no cover - platform fallback
            ctx = multiprocessing.get_context()
        result_queue: Any = ctx.Queue(maxsize=1)
        process = ctx.Process(
            target=_run_testgen_task_process_entry,
            args=(self, task, result_queue),
            name=f"apex-testgen-task-{task.instance_id[:48]}",
        )
        process.start()
        process.join(timeout=max(0.1, timeout_seconds))
        if process.is_alive():
            self._terminate_task_process(process)
            return self._timeout_task_result(
                task,
                timeout_seconds=timeout_seconds,
                started_at=started,
                child_pid=process.pid,
            )
        payload: Optional[dict[str, Any]] = None
        try:
            payload = result_queue.get_nowait()
        except Exception:
            checkpointed = self._load_checkpointed_task_result(task)
            if checkpointed is not None:
                return checkpointed
        if isinstance(payload, dict):
            try:
                result = SWEBenchProTestGenTaskResult.from_dict(payload)
                write_task_checkpoint(self._task_output_dir(task), result.to_dict())
                return result
            except Exception as exc:
                logger.warning(
                    "Ignoring invalid subprocess payload for %s: %s",
                    task.instance_id,
                    exc,
                )
        result = SWEBenchProTestGenTaskResult(
            instance_id=task.instance_id,
            repo=task.repo,
            success=False,
            duration_seconds=time.time() - started,
            failure_reason=(
                f"Task subprocess exited with code {process.exitcode} without a valid checkpoint."
            ),
            execution_metadata={
                "task_process_pid": process.pid,
                "task_process_exitcode": process.exitcode,
            },
        )
        write_task_checkpoint(self._task_output_dir(task), result.to_dict())
        return result

    def _task_worker_limit(self, task_count: int) -> int:
        if task_count <= 0:
            return 1
        configured = max(int(self.config.benchmark.task_parallelism or 1), 1)
        return max(1, min(task_count, configured))

    def _task_output_dir(self, task: SWEBenchProTask) -> Path:
        return self.output_dir / task.instance_id

    def _task_workspace_dir(self, task: SWEBenchProTask) -> Path:
        return self.output_dir / "workspaces" / task.instance_id

    def _load_checkpointed_task_result(
        self,
        task: SWEBenchProTask,
    ) -> Optional[SWEBenchProTestGenTaskResult]:
        payload = load_json_if_exists(task_result_path(self._task_output_dir(task)))
        if payload is None:
            return None
        try:
            return SWEBenchProTestGenTaskResult.from_dict(payload)
        except Exception as exc:
            logger.warning(
                "Ignoring corrupt SWE-Bench Pro testgen checkpoint for %s: %s",
                task.instance_id,
                exc,
            )
            return None

    def _write_report_checkpoint(
        self,
        report: SWEBenchProTestGenReport,
        requested_task_ids: list[str],
        *,
        completed: bool,
    ) -> None:
        report.updated_at = time.time()
        report.finished_at = report.updated_at if completed else 0.0
        report.enable_testgen_memory = self.enable_testgen_memory
        report.allow_gold_oracle_selection = self.allow_gold_oracle_selection
        report.testgen_memory_directory = self._testgen_memory_directory_for_repo("")
        report.testgen_memory_per_task_summary = copy.deepcopy(
            self._testgen_memory_per_task_summary
        )
        update_run_manifest(
            self.output_dir,
            requested_task_ids=requested_task_ids,
            completed_task_ids=[task.instance_id for task in report.tasks],
            completed=completed,
            extra_updates={
                "config_payload": self.config.to_dict(),
                "environment_snapshot": capture_environment_snapshot(self.config),
                "dataset_name": report.dataset_name,
                "dataset_split": report.dataset_split,
                "evaluation_mode": report.evaluation_mode,
                "rollout_count": self.rollout_count,
                "aggregate_metrics": report.aggregate_metrics,
            },
        )
        report.run_manifest = load_run_manifest(self.output_dir) or report.run_manifest
        atomic_write_json(self.output_dir / "benchmark_report.json", report.to_dict())
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
                successful_tasks=report.successful_tasks,
                failed_tasks=report.failed_tasks,
                completed=completed,
                metadata={
                    "config_source": report.config_source,
                    "dataset_name": report.dataset_name,
                    "dataset_split": report.dataset_split,
                    "evaluation_mode": report.evaluation_mode,
                    "rollout_count": self.rollout_count,
                    "model_config": copy.deepcopy(report.model_config),
                    "ablation_config": copy.deepcopy(report.ablation_config),
                    "aggregate_metrics": report.aggregate_metrics,
                },
            ),
        )

    def _run_testgen_rollout_candidate(
        self,
        *,
        task: SWEBenchProTask,
        config: ApexConfig,
        repo_dir: Path,
        task_output_dir: Path,
        worktree_manager: GitWorktreeManager,
        repo_context: Any,
        issue_description: str,
        issue_plan: Any,
        test_command: str,
        rollout_id: int,
        memory_bus: Optional[EpisodicMemoryBus] = None,
    ) -> _PerRolloutTestGenerationCandidate:
        brief = _select_rollout_brief(issue_plan, rollout_id)
        worktree_path = worktree_manager.create_worktree(rollout_id)
        baseline_commit = worktree_manager.get_baseline_commit(worktree_path)
        temperature = config.get_temperature_for_rollout(rollout_id)
        rollout_output_dir = task_output_dir / f"rollout_{rollout_id}"
        rollout_output_dir.mkdir(parents=True, exist_ok=True)

        write_task_live_state(
            task_output_dir,
            {
                "task_id": task.instance_id,
                "phase": "reproducer",
                "status": "active",
                "process_pid": os.getpid(),
                "last_progress_at": time.time(),
                "rollout_id": rollout_id,
            },
        )
        reproduction_submission, reproduction_tokens, _ = _run_scaffold_reproducer_stage(
            llm_config=_resolve_stage_llm_config(config, rollout_id, "reproducer", brief=brief),
            config=config,
            repo_context=repo_context,
            issue_description=issue_description,
            issue_plan=issue_plan,
            brief=brief,
            worktree_path=str(worktree_path),
            test_command=test_command,
            memory_bus=memory_bus,
            rollout_id=rollout_id,
            resolved_baseline=baseline_commit,
            execution_tree=None,
            temperature=temperature,
            status_output_dir=str(rollout_output_dir),
        )
        reproduction_artifact = coerce_reproduction_artifact(reproduction_submission)

        write_task_live_state(
            task_output_dir,
            {
                "task_id": task.instance_id,
                "phase": "localizer",
                "status": "active",
                "process_pid": os.getpid(),
                "last_progress_at": time.time(),
                "rollout_id": rollout_id,
            },
        )
        localization_submission, localization_tokens, _ = _run_scaffold_localizer_stage(
            llm_config=_resolve_stage_llm_config(config, rollout_id, "localizer", brief=brief),
            config=config,
            repo_context=repo_context,
            issue_description=issue_description,
            issue_plan=issue_plan,
            brief=brief,
            worktree_path=str(worktree_path),
            reproduction_artifact=reproduction_artifact,
            memory_bus=memory_bus,
            rollout_id=rollout_id,
            resolved_baseline=baseline_commit,
            execution_tree=None,
            temperature=temperature,
            status_output_dir=str(rollout_output_dir),
        )
        localization_artifact = coerce_localization_artifact(localization_submission)

        write_task_live_state(
            task_output_dir,
            {
                "task_id": task.instance_id,
                "phase": "test_writer",
                "status": "active",
                "process_pid": os.getpid(),
                "last_progress_at": time.time(),
                "rollout_id": rollout_id,
            },
        )
        task_budget = float(getattr(config.benchmark, "testgen_task_timeout_seconds", 0.0) or 0.0)
        rollout_budget = 480.0
        if task_budget > 0:
            rollout_budget = max(180.0, min(480.0, task_budget * 0.45))
        with _temporary_environ(
            {
                # Benchmark slices should submit the first materialized usable
                # suite to F2P/mutation instead of spending the task budget on
                # precursor/inversion/repair prompt rounds.
                "APEX_TESTGEN_SINGLE_PASS": "1",
                "APEX_TESTGEN_REPAIR_MAX_ROUNDS": "0",
                "APEX_TEST_GENERATION_ROLLOUT_WALLCLOCK_SECONDS": str(rollout_budget),
            }
        ):
            test_generation_result = _execute_rollout_test_generation(
                config=config,
                repo_path=str(repo_dir),
                repo_context=repo_context,
                issue_description=issue_description,
                issue_plan=issue_plan,
                brief=brief,
                worktree_path=str(worktree_path),
                reproduction_artifact=reproduction_artifact,
                localization_artifact=localization_artifact,
                memory_bus=memory_bus,
                rollout_id=rollout_id,
                resolved_baseline=baseline_commit,
                execution_tree=None,
                temperature=temperature,
                test_command=test_command,
                changed_files=[],
                status_output_dir=str(rollout_output_dir),
                stop_after_usable_submission=True,
            )
        test_submission = dict(test_generation_result.submission or {})
        generated_portfolio = (
            test_generation_result.artifact.to_dict()
            if test_generation_result.artifact is not None
            else normalize_test_suite_artifact_payload(test_submission)
        )
        issue_surface_signal = _test_writer_issue_surface_repair_signal(
            submission=generated_portfolio,
            worktree_path=str(worktree_path),
            issue_description=issue_description,
            issue_plan=issue_plan,
            reproduction_artifact=reproduction_artifact,
            localization_artifact=localization_artifact,
            repo_context=repo_context,
        )
        atomic_write_json(
            rollout_output_dir / "generated_test_portfolio.json",
            generated_portfolio,
        )
        atomic_write_json(
            rollout_output_dir / "issue_surface_signal.json",
            issue_surface_signal,
        )
        atomic_write_json(
            rollout_output_dir / "rollout_summary.json",
            {
                "rollout_id": rollout_id,
                "tokens_used": (
                    int(reproduction_tokens or 0)
                    + int(localization_tokens or 0)
                    + int(test_generation_result.tokens_used or 0)
                ),
                "issue_surface_signal": issue_surface_signal,
                "loop_summary": dict(test_generation_result.loop_summary or {}),
            },
        )
        return _PerRolloutTestGenerationCandidate(
            rollout_id=rollout_id,
            brief=brief,
            worktree_path=worktree_path,
            baseline_commit=baseline_commit,
            reproduction_artifact=reproduction_artifact,
            localization_artifact=localization_artifact,
            generated_portfolio=generated_portfolio,
            tokens_used=(
                int(reproduction_tokens or 0)
                + int(localization_tokens or 0)
                + int(test_generation_result.tokens_used or 0)
            ),
            trajectory=list(test_generation_result.trajectory or []),
            loop_summary=dict(test_generation_result.loop_summary or {}),
            issue_surface_signal=issue_surface_signal,
        )

    def _build_testgen_judge_caller(self) -> Optional[Any]:
        """Construct an LLMCaller bound to the eval's primary backend.

        For CLI backends (codex_cli / claude_cli / gemini_cli) we use
        ``CLIModelClient.run_structured_prompt`` so the judge response is
        schema-constrained at the decoder level. For OpenAI-compatible
        backends we fall through to ``LLMClient.chat`` and parse JSON
        from the assistant message. Either way the call is one-shot
        per task — no multi-turn agent.

        Returns None when no usable backend is available so the selector
        gracefully falls back to the heuristic comparator.
        """
        if not self.config.llm_configs:
            return None
        primary = self.config.llm_configs[0]
        if primary.is_cli_backend:
            try:
                from apex.core.cli_backend import CLIModelClient
            except Exception:  # pragma: no cover — defensive
                return None
            cli_client = CLIModelClient(primary)

            import tempfile

            def _cli_caller(prompt: str, schema: dict[str, Any]) -> Optional[dict[str, Any]]:
                with tempfile.TemporaryDirectory(prefix="apex-testgen-judge-") as workdir:
                    result = cli_client.run_structured_prompt(
                        prompt=prompt,
                        working_dir=workdir,
                        schema=schema,
                        allow_edits=False,
                        internet_enabled=False,
                    )
                if not result.success:
                    return None
                if isinstance(result.parsed_json, dict):
                    return dict(result.parsed_json)
                try:
                    return json.loads(result.text or "{}")
                except (TypeError, ValueError, json.JSONDecodeError):
                    return None

            return _cli_caller

        try:
            from apex.core.llm import LLMClient, Message
        except Exception:  # pragma: no cover
            return None
        api_client = LLMClient(primary, temperature_override=0.0)

        def _api_caller(prompt: str, schema: dict[str, Any]) -> Optional[dict[str, Any]]:
            response = api_client.chat(
                messages=[
                    Message(
                        role="system",
                        content=(
                            "Respond ONLY with valid JSON matching this schema: "
                            + json.dumps(schema)
                        ),
                    ),
                    Message(role="user", content=prompt),
                ]
            )
            try:
                return json.loads(response.content or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                return None

        return _api_caller

    def _select_best_testgen_rollout_candidate(
        self,
        candidates: list[_PerRolloutTestGenerationCandidate],
        *,
        f2p_payloads: Optional[dict[int, dict[str, Any]]] = None,
        judge_caller: Optional[Any] = None,
        judge_context: Optional[dict[str, Any]] = None,
    ) -> _PerRolloutTestGenerationCandidate:
        if not candidates:
            raise RuntimeError("No test-generation rollout candidates were produced.")
        # When F2P payloads are available, prefer the candidate whose tests
        # actually catch the bug (any_f2p=True), tie-broken by f2p_count and
        # then f2p_rate, with the heuristic surface-repair signal used only
        # as a final tie-break. This makes selection optimize for the metric
        # the public benchmarks (SWT-Bench / TDD-Bench) actually measure on,
        # rather than the indirect surface-repair signal alone.
        if f2p_payloads:
            sorted_by_f2p = sorted(
                candidates,
                key=lambda c: _testgen_candidate_f2p_score_tuple(c, f2p_payloads),
                reverse=True,
            )
            top_score = _testgen_candidate_f2p_score_tuple(
                sorted_by_f2p[0],
                f2p_payloads,
            )
            top_tier = [
                c
                for c in sorted_by_f2p
                if _testgen_candidate_f2p_score_tuple(c, f2p_payloads) == top_score
            ]
            if len(top_tier) == 1:
                return top_tier[0]
            # Phase E.3: when the F2P-tuple ties multiple candidates, an
            # optional LLM judge breaks the tie by reading each
            # candidate's measured outcomes (F2P/mutation counts, sample
            # test paths) and picking the suite most likely to catch
            # production bugs. Only runs on the top tier so it cannot
            # downgrade a bug-catching suite to a non-bug-catching one.
            if judge_caller is not None and len(top_tier) > 1:
                try:
                    from apex.evaluation.testgen_judge import (
                        judge_testgen_candidates,
                        summarize_candidate_for_judge,
                    )

                    summaries = []
                    for c in top_tier:
                        artifacts = list((c.generated_portfolio or {}).get("test_artifacts") or [])
                        test_paths = [
                            str(a.get("path") or "").strip() for a in artifacts if a.get("path")
                        ]
                        summaries.append(
                            summarize_candidate_for_judge(
                                rollout_id=c.rollout_id,
                                f2p_payload=f2p_payloads.get(c.rollout_id) or {},
                                test_paths=test_paths,
                            )
                        )
                    ctx = dict(judge_context or {})
                    outcome = judge_testgen_candidates(
                        candidates_summary=summaries,
                        issue_description=str(ctx.get("issue_description") or ""),
                        repo_name=str(ctx.get("repo_name") or ""),
                        llm_caller=judge_caller,
                    )
                    if outcome.judge_used and outcome.selected_rollout_id is not None:
                        for c in top_tier:
                            if c.rollout_id == outcome.selected_rollout_id:
                                return c
                        # Judge picked a rollout_id not in our top tier;
                        # log and fall through to heuristic.
                        logger.warning(
                            "Testgen judge picked rollout %s which is not in the F2P top tier; "
                            "falling back to heuristic comparator.",
                            outcome.selected_rollout_id,
                        )
                except Exception as exc:  # pragma: no cover — defensive
                    logger.warning("Testgen judge invocation failed: %s", exc)
            # Multiple candidates tied on F2P — fall through to heuristic
            # comparison among the tied set so we still pick the strongest
            # surface-repair signal among equally-bug-catching candidates.
            candidates = top_tier

        best = candidates[0]
        best_signal = dict(best.issue_surface_signal or {})
        best_payload = dict(best.generated_portfolio or {})
        for candidate in candidates[1:]:
            candidate_signal = dict(candidate.issue_surface_signal or {})
            candidate_payload = dict(candidate.generated_portfolio or {})
            if _test_writer_candidate_is_better(
                candidate_signal=candidate_signal,
                candidate_payload=candidate_payload,
                baseline_signal=best_signal,
                baseline_payload=best_payload,
            ):
                best = candidate
                best_signal = candidate_signal
                best_payload = candidate_payload
        return best

    def _run_mutation_for_payload(
        self,
        *,
        task: SWEBenchProTask,
        f2p_payload: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Run mutation discrimination against a F2P-kept fixed sandbox.

        Returns the MutationReport.to_dict() payload, or None when the
        oracle was disabled, the F2P run did not catch the bug, or the
        gold patch did not modify any language-supported source file we
        can mutate.

        Caller is responsible for cleaning up the F2P sandbox afterward.
        """
        if not self.enable_mutation:
            return None
        summary = dict(f2p_payload.get("summary") or {})
        if not summary.get("any_f2p"):
            # Mutation against a sandbox whose tests don't catch the gold
            # bug is uninformative — every mutation will "survive" because
            # the tests can't even kill the original mismatched behavior.
            return None
        fixed_path = str(f2p_payload.get("fixed_path") or "").strip()
        if not fixed_path or not Path(fixed_path).exists():
            return None
        gold_patch = str(getattr(task, "patch", "") or "")
        try:
            from apex.evaluation.mutation_engine import (
                evaluate_mutation_score,
                generate_mutants,
                source_paths_from_patch,
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("Mutation engine unavailable: %s", exc)
            return None

        repo_language = str(getattr(task, "repo_language", "") or "python").lower()
        target_paths = source_paths_from_patch(
            gold_patch,
            language=repo_language,
        )[: self.mutation_max_files]
        if not target_paths:
            return {
                "mutation_score": 0.0,
                "total_mutants": 0,
                "killed": 0,
                "survived": 0,
                "errored": 0,
                "timed_out": 0,
                "skip_reason": "no_mutation_targets_in_gold_patch",
                "language": repo_language,
                "source_paths": [],
            }

        candidate_test_paths = list(summary.get("candidate_test_paths") or [])
        if not candidate_test_paths:
            return {
                "mutation_score": 0.0,
                "total_mutants": 0,
                "skip_reason": "no_candidate_test_paths",
            }

        mutants: list[Any] = []
        for rel_path in target_paths:
            absolute = Path(fixed_path) / rel_path
            if not absolute.exists():
                continue
            file_mutants = generate_mutants(
                source_path=absolute,
                language=repo_language,
                max_mutants=self.mutation_max_per_file,
                seed=_stable_mutation_seed(task.instance_id, rel_path),
            )
            # Re-key each mutant's source_path to the sandbox-relative path
            # the engine uses to write back into the sandbox.
            for m in file_mutants:
                m.source_path = rel_path
            mutants.extend(file_mutants)
        if not mutants:
            return {
                "mutation_score": 0.0,
                "total_mutants": 0,
                "skip_reason": "no_mutants_generated",
                "source_paths": target_paths,
            }
        try:
            report = evaluate_mutation_score(
                fixed_dir=fixed_path,
                mutants=mutants,
                test_paths=candidate_test_paths,
                language=repo_language,
                per_mutant_timeout_seconds=self.mutation_per_mutant_timeout_seconds,
                baseline_timeout_seconds=self.f2p_per_side_timeout_seconds,
            )
        except Exception as mut_exc:  # pragma: no cover — defensive
            logger.warning("Mutation evaluation failed for %s: %s", task.instance_id, mut_exc)
            return {
                "mutation_score": 0.0,
                "total_mutants": len(mutants),
                "error": f"{type(mut_exc).__name__}: {mut_exc}",
            }
        return report.to_dict()

    def _run_assertion_mutation_for_payload(
        self,
        *,
        task: SWEBenchProTask,
        f2p_payload: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        summary = dict(f2p_payload.get("summary") or {})
        if not summary.get("any_f2p"):
            return None
        fixed_path = str(f2p_payload.get("fixed_path") or "").strip()
        if not fixed_path or not Path(fixed_path).exists():
            return None
        candidate_test_paths = list(summary.get("candidate_test_paths") or [])
        if not candidate_test_paths:
            return {
                "status": "no_candidate_test_paths",
                "test_paths": [],
                "mutated_assertion_count": 0,
                "survived": False,
                "assertion_effective": False,
            }
        try:
            from apex.evaluation.assertion_mutation import (
                evaluate_assertion_effect_in_loop,
            )

            report = evaluate_assertion_effect_in_loop(
                worktree_path=fixed_path,
                test_paths=candidate_test_paths,
                language=str(getattr(task, "repo_language", "") or "python"),
                timeout_seconds=self.f2p_per_side_timeout_seconds,
            )
            return report.to_dict()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Assertion mutation evaluation failed for %s: %s",
                task.instance_id,
                exc,
            )
            return {
                "status": "exception",
                "error": f"{type(exc).__name__}: {exc}",
                "test_paths": candidate_test_paths,
                "mutated_assertion_count": 0,
                "survived": False,
                "assertion_effective": False,
            }

    def _build_task_config(
        self,
        task_output_dir: Path,
        task_workspace_dir: Path,
    ) -> ApexConfig:
        config = copy.deepcopy(self.config)
        config.output_dir = str(task_output_dir)
        config.workspace_dir = str(task_workspace_dir)
        return config

    # ---------- Phase I.7: cross-task testgen memory ----------

    def _testgen_memory_directory_for_repo(self, repo_path: str) -> Optional[str]:
        """Resolve the per-repo memory directory used by RepoMemoryStore.

        When the user passes ``--testgen-memory-directory`` we use it
        verbatim (their explicit choice — could be shared across runs).
        Otherwise we default to ``<output_dir>/_testgen_memory``, which
        gives EACH BENCHMARK RUN an isolated memory store. That removes
        the cross-benchmark-run contamination risk: re-running the
        eval doesn't bleed prior-run insights into the new run.
        """
        if not self.enable_testgen_memory:
            return None
        if self.testgen_memory_directory:
            return self.testgen_memory_directory
        return str(Path(self.output_dir) / "_testgen_memory")

    def _focus_files_for_task(self, issue_plan: Any) -> list[str]:
        """Pull the union of focus file paths across the issue plan and
        rollout briefs. Defensive about missing/non-iterable fields."""
        focus: set[str] = set()
        for item in list(getattr(issue_plan, "focus_files", []) or []):
            s = str(item or "").strip()
            if s:
                focus.add(s)
        for brief in list(getattr(issue_plan, "rollout_briefs", []) or []):
            for item in list(getattr(brief, "focus_files", []) or []):
                s = str(item or "").strip()
                if s:
                    focus.add(s)
        return sorted(focus)

    def _inject_prior_testgen_memory(self, *, task: Any, issue_plan: Any) -> None:
        """Stash prior cross-task testgen insights on the issue plan's
        planner_metadata so the test_writer prompt builder can render
        them. No-op when memory is disabled."""
        if not self.enable_testgen_memory:
            return
        memory_dir = self._testgen_memory_directory_for_repo(getattr(task, "repo", ""))
        if not memory_dir:
            return
        focus_files = self._focus_files_for_task(issue_plan)
        if not focus_files:
            return
        try:
            from ..persistence import (
                query_prior_testgen_insights_for_focus_files,
            )

            insights = query_prior_testgen_insights_for_focus_files(
                repo_path=str(task.repo or ""),
                focus_files=focus_files,
                directory=memory_dir,
                max_insights=12,
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "Phase I.7: prior testgen memory query failed for %s: %s",
                task.instance_id,
                exc,
            )
            return
        if not insights:
            self._testgen_memory_per_task_summary[task.instance_id] = {
                "insights_loaded": 0,
                "insights_persisted": 0,
            }
            return
        existing = (
            dict(issue_plan.planner_metadata)
            if isinstance(getattr(issue_plan, "planner_metadata", None), dict)
            else {}
        )
        existing["prior_testgen_memory_insights"] = [i.to_dict() for i in insights]
        issue_plan.planner_metadata = existing
        self._testgen_memory_per_task_summary[task.instance_id] = {
            "insights_loaded": len(insights),
            "insights_persisted": 0,
        }

    def _persist_testgen_memory_for_task(
        self,
        *,
        task: Any,
        issue_plan: Any,
        f2p_summary: Optional[dict[str, Any]],
        comparison: Optional[dict[str, Any]],
    ) -> None:
        """Extract cross-task insights from this task's outcomes and
        merge them into the per-repo memory store. No-op when memory
        is disabled."""
        if not self.enable_testgen_memory:
            return
        memory_dir = self._testgen_memory_directory_for_repo(getattr(task, "repo", ""))
        if not memory_dir:
            return
        focus_files = self._focus_files_for_task(issue_plan)
        mutation_summary = (
            dict(comparison.get("mutation_summary") or {}) if isinstance(comparison, dict) else {}
        )
        coverage_gap_summary = (
            dict(comparison.get("coverage_summary") or {}) if isinstance(comparison, dict) else {}
        )
        target_comparison = (
            list(comparison.get("target_comparison") or []) if isinstance(comparison, dict) else []
        )
        try:
            from ..persistence import (
                extract_testgen_insights_from_run_summary,
                persist_testgen_insights_for_repo,
            )

            insights = extract_testgen_insights_from_run_summary(
                focus_files=focus_files,
                f2p_summary=f2p_summary or {},
                mutation_summary=mutation_summary,
                coverage_gap_summary=coverage_gap_summary,
                axis_coverage_summary=coverage_gap_summary,
                target_comparison=target_comparison,
            )
            if not insights:
                return
            summary = persist_testgen_insights_for_repo(
                repo_path=str(task.repo or ""),
                insights=insights,
                directory=memory_dir,
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "Phase I.7: persist testgen memory failed for %s: %s",
                task.instance_id,
                exc,
            )
            return
        per_task = self._testgen_memory_per_task_summary.setdefault(
            task.instance_id,
            {"insights_loaded": 0, "insights_persisted": 0},
        )
        per_task["insights_persisted"] = int(summary.get("persisted_insight_count") or 0)

    def _run_task(self, task: SWEBenchProTask) -> SWEBenchProTestGenTaskResult:
        started = time.time()
        sandbox = Path(tempfile.mkdtemp(prefix=f"apex-swebench-testgen-{task.repo_name}-"))
        repo_dir = sandbox / "repo"
        task_output_dir = self._task_output_dir(task)
        task_workspace_dir = self._task_workspace_dir(task)
        task_output_dir.mkdir(parents=True, exist_ok=True)
        task_workspace_dir.mkdir(parents=True, exist_ok=True)
        worktree_manager: Optional[GitWorktreeManager] = None
        issue_contract_targets: list[str] = []

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
            test_command = self.build_agent_test_command(task, repo_dir)
            issue_description = task.build_issue_description(
                test_command,
                include_benchmark_guardrails=False,
                include_benchmark_metadata=False,
                include_selected_test_targets=False,
                include_required_tests=False,
            )
            issue_contract_targets = extract_issue_contract_targets(issue_description)

            config = self._build_task_config(task_output_dir, task_workspace_dir)
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
                    description="swebench_pro_testgen_official_docker_image",
                ),
                label=f"swebench_pro_testgen_{task.instance_id}",
            )
            apply_target_tool_env_to_apex_config(config, target_tool_env)
            atomic_write_json(
                task_output_dir / "target_runtime_tools.json",
                target_tool_diagnostics,
            )
            orchestrator = ApexOrchestrator(config)

            write_task_live_state(
                task_output_dir,
                {
                    "task_id": task.instance_id,
                    "phase": "preprocess_repo",
                    "status": "active",
                    "process_pid": os.getpid(),
                    "last_progress_at": time.time(),
                },
            )
            repo_context = orchestrator._preprocess_repo(str(repo_dir))
            repo_context.save(task_output_dir / "repo_context.json")

            write_task_live_state(
                task_output_dir,
                {
                    "task_id": task.instance_id,
                    "phase": "planning",
                    "status": "active",
                    "process_pid": os.getpid(),
                    "last_progress_at": time.time(),
                },
            )
            planner = IssuePlanner(config)
            strategy = planner.build_execution_strategy(
                issue_description,
                repo_context,
                rollout_count=self.rollout_count,
                baseline_result=None,
            )
            issue_plan = orchestrator._plan_issue(
                issue_description,
                repo_context,
                planner=planner,
                rollout_count=strategy.rollout_count,
                difficulty=strategy.difficulty_estimate,
                baseline_result=None,
            )
            issue_plan = planner.enrich_issue_plan(
                issue_plan,
                issue_description=issue_description,
                repo_context=repo_context,
                test_command=test_command,
                baseline_result=None,
            )
            issue_plan = planner.apply_execution_strategy(issue_plan, strategy)
            # Phase I.7: query cross-task testgen memory BEFORE saving
            # the issue plan so the prior insights ride along with it
            # to every rollout's test_writer prompt. Guarded by
            # enable_testgen_memory; when off this is a no-op so smoke
            # A vs smoke B comparison is clean.
            self._inject_prior_testgen_memory(task=task, issue_plan=issue_plan)
            issue_plan.save(task_output_dir / "issue_plan.json")

            write_task_live_state(
                task_output_dir,
                {
                    "task_id": task.instance_id,
                    "phase": "create_worktrees",
                    "status": "active",
                    "process_pid": os.getpid(),
                    "last_progress_at": time.time(),
                },
            )
            worktree_manager = GitWorktreeManager(
                str(repo_dir),
                str(task_workspace_dir),
                use_git_worktrees=config.rollout.use_git_worktrees,
            )
            testgen_memory_bus = EpisodicMemoryBus()
            planned_rollout_count = max(
                1,
                len(getattr(issue_plan, "rollout_briefs", []) or []) or self.rollout_count,
            )
            rollout_candidates: list[_PerRolloutTestGenerationCandidate] = []
            rollout_failures: list[dict[str, Any]] = []
            for rollout_id in range(planned_rollout_count):
                try:
                    rollout_candidates.append(
                        self._run_testgen_rollout_candidate(
                            task=task,
                            config=config,
                            repo_dir=repo_dir,
                            task_output_dir=task_output_dir,
                            worktree_manager=worktree_manager,
                            repo_context=repo_context,
                            issue_description=issue_description,
                            issue_plan=issue_plan,
                            test_command=test_command,
                            rollout_id=rollout_id,
                            memory_bus=testgen_memory_bus,
                        )
                    )
                except FileNotFoundError as exc:
                    # The rollout's worktree disappeared mid-stage (e.g. an
                    # agentic CLI tool nuked its own cwd). Record the failure
                    # but keep going: the gold comparator only needs
                    # ``repo_dir``, so we can still produce a well-formed
                    # synthetic_vs_gold_comparison.json with an empty
                    # generated portfolio rather than crashing the task.
                    logger.warning(
                        "Rollout %s for %s lost its worktree (%s); continuing with empty generated portfolio.",
                        rollout_id,
                        task.instance_id,
                        exc,
                    )
                    rollout_failures.append(
                        {
                            "rollout_id": rollout_id,
                            "error_type": "worktree_unavailable",
                            "error": str(exc),
                        }
                    )
                except Exception as exc:
                    logger.warning(
                        "Rollout %s for %s failed (%s); continuing so the gold comparator can still run.",
                        rollout_id,
                        task.instance_id,
                        exc,
                    )
                    rollout_failures.append(
                        {
                            "rollout_id": rollout_id,
                            "error_type": exc.__class__.__name__,
                            "error": str(exc),
                        }
                    )

            write_task_live_state(
                task_output_dir,
                {
                    "task_id": task.instance_id,
                    "phase": "select_rollout",
                    "status": "active",
                    "process_pid": os.getpid(),
                    "last_progress_at": time.time(),
                    "candidate_rollout_count": len(rollout_candidates),
                    "failed_rollout_count": len(rollout_failures),
                },
            )
            # Per-candidate F2P before selection uses the gold patch as an
            # oracle. That is useful for diagnostics and ablations, but it is
            # not benchmark-fair SOTA methodology unless explicitly enabled.
            # Default runs evaluate F2P only after selection.
            per_candidate_f2p: dict[int, dict[str, Any]] = {}
            per_candidate_f2p_skipped_for_methodology = bool(
                self.enable_f2p and not self.allow_gold_oracle_selection and rollout_candidates
            )
            if self.enable_f2p and self.allow_gold_oracle_selection and rollout_candidates:
                assertion_mutation_enabled = _iteration_assertion_mutation_enabled()
                try:
                    from apex.evaluation.f2p_oracle import evaluate_f2p
                except Exception:  # pragma: no cover — defensive
                    evaluate_f2p = None  # type: ignore[assignment]
                if evaluate_f2p is not None:
                    f2p_root = task_output_dir / "_f2p_per_candidate"
                    f2p_root.mkdir(parents=True, exist_ok=True)
                    for candidate in rollout_candidates:
                        candidate_artifacts = list(
                            (candidate.generated_portfolio or {}).get("test_artifacts") or []
                        )
                        candidate_dir = f2p_root / f"rollout_{candidate.rollout_id}"
                        try:
                            payload = evaluate_f2p(
                                task=task,
                                repo_dir=str(repo_dir),
                                test_artifacts=candidate_artifacts,
                                output_dir=str(candidate_dir),
                                language=str(task.repo_language or "python"),
                                broken_timeout_seconds=self.f2p_per_side_timeout_seconds,
                                fixed_timeout_seconds=self.f2p_per_side_timeout_seconds,
                                install_repo=self.f2p_install_repo,
                                # Keep the fixed sandbox alive so mutation
                                # and assertion discrimination can reuse it
                                # without paying another clone+install round-trip.
                                keep_sandboxes=(self.enable_mutation or assertion_mutation_enabled),
                            )
                        except Exception as f2p_exc:  # pragma: no cover
                            logger.warning(
                                "Per-candidate F2P failed for rollout %s of %s: %s",
                                candidate.rollout_id,
                                task.instance_id,
                                f2p_exc,
                            )
                            continue
                        if self.enable_mutation:
                            mutation_report = self._run_mutation_for_payload(
                                task=task,
                                f2p_payload=payload,
                            )
                            if mutation_report is not None:
                                payload["mutation"] = mutation_report
                        if assertion_mutation_enabled:
                            assertion_report = self._run_assertion_mutation_for_payload(
                                task=task,
                                f2p_payload=payload,
                            )
                            if assertion_report is not None:
                                payload["assertion_mutation"] = assertion_report
                        if self.enable_mutation or assertion_mutation_enabled:
                            # Cleanup sandboxes now that discrimination is done.
                            sandboxes_root = Path(payload.get("broken_path") or "").parent
                            if sandboxes_root.exists():
                                shutil.rmtree(sandboxes_root, ignore_errors=True)
                        try:
                            from apex.evaluation.iteration_feedback import (
                                classify_dual_state_f2p_feedback,
                            )

                            iteration_feedback = classify_dual_state_f2p_feedback(
                                f2p_payload=payload,
                                iteration_index=0,
                            )
                            if iteration_feedback.is_actionable():
                                feedback_dict = iteration_feedback.to_dict()
                                payload["iteration_feedback"] = dict(feedback_dict)
                                candidate.generated_portfolio["prior_iteration_f2p_feedback"] = (
                                    dict(feedback_dict)
                                )
                                candidate.loop_summary["prior_iteration_f2p_feedback"] = dict(
                                    feedback_dict
                                )
                        except Exception as feedback_exc:  # pragma: no cover
                            logger.warning(
                                "Failed to derive dual-state F2P prompt feedback for rollout %s of %s: %s",
                                candidate.rollout_id,
                                task.instance_id,
                                feedback_exc,
                            )
                        per_candidate_f2p[candidate.rollout_id] = payload

            if rollout_candidates:
                judge_caller = (
                    self._build_testgen_judge_caller()
                    if self.enable_testgen_judge and per_candidate_f2p
                    else None
                )
                selected_candidate = self._select_best_testgen_rollout_candidate(
                    rollout_candidates,
                    f2p_payloads=per_candidate_f2p or None,
                    judge_caller=judge_caller,
                    judge_context={
                        "issue_description": issue_description,
                        "repo_name": task.repo,
                    },
                )
                generated_portfolio = dict(selected_candidate.generated_portfolio or {})
                selected_rollout_id: Optional[int] = selected_candidate.rollout_id
                portfolio_selection_mode = "single_rollout"
                union_portfolio = _build_f2p_positive_rollout_union_portfolio(
                    rollout_candidates,
                    f2p_payloads=per_candidate_f2p,
                    selected_rollout_id=selected_rollout_id,
                )
                if union_portfolio is not None:
                    generated_portfolio = union_portfolio
                    portfolio_selection_mode = "f2p_positive_rollout_union"
            else:
                # All rollouts failed — degrade to an empty portfolio so the
                # gold comparator can still run and we still emit
                # synthetic_vs_gold_comparison.json. Without this the task
                # would crash here and we'd lose all evaluation artifacts.
                selected_candidate = None
                generated_portfolio = {}
                selected_rollout_id = None
                portfolio_selection_mode = "no_rollout"
            test_tokens = sum(int(candidate.tokens_used or 0) for candidate in rollout_candidates)
            test_quality_summary: dict[str, Any] = {}
            portfolio_validation: dict[str, Any] = {}
            if generated_portfolio.get("test_artifacts"):
                try:
                    from apex.evaluation.test_quality import (
                        analyze_test_artifacts_quality,
                    )

                    quality_artifacts, skipped_quality_paths = (
                        _select_static_quality_test_artifacts(
                            [
                                artifact
                                for artifact in list(
                                    generated_portfolio.get("test_artifacts") or []
                                )
                                if isinstance(artifact, dict)
                            ],
                            language=str(task.repo_language or "python"),
                        )
                    )
                    test_quality_summary = analyze_test_artifacts_quality(
                        quality_artifacts,
                        language=str(task.repo_language or "python"),
                    ).to_dict()
                    if skipped_quality_paths:
                        test_quality_summary["skipped_static_quality_artifact_count"] = len(
                            skipped_quality_paths
                        )
                        test_quality_summary["skipped_static_quality_artifact_paths"] = (
                            skipped_quality_paths[:12]
                        )
                    generated_portfolio["test_quality_summary"] = dict(test_quality_summary)
                except Exception as quality_exc:  # pragma: no cover - defensive
                    test_quality_summary = {"error": f"{type(quality_exc).__name__}: {quality_exc}"}
            try:
                portfolio_validation = _attach_static_validation_to_testgen_portfolio(
                    task=task,
                    generated_portfolio=generated_portfolio,
                    test_command=test_command,
                )
            except Exception as validation_exc:  # pragma: no cover - defensive
                portfolio_validation = {
                    "status": "error",
                    "error": f"{type(validation_exc).__name__}: {validation_exc}",
                    "benchmark_adapter": "swebench_pro_testgen",
                    "validation_scope": "static_pre_comparison",
                }
                generated_portfolio["apex_validation"] = dict(portfolio_validation)
            atomic_write_json(
                task_output_dir / "rollout_selection.json",
                {
                    "executed_rollout_count": len(rollout_candidates),
                    "selected_rollout_id": selected_rollout_id,
                    "portfolio_selection_mode": portfolio_selection_mode,
                    "union_source_rollout_ids": list(
                        generated_portfolio.get("source_rollout_ids") or []
                    ),
                    "candidate_rollouts": [
                        {
                            "rollout_id": candidate.rollout_id,
                            "tokens_used": int(candidate.tokens_used or 0),
                            "issue_surface_signal": dict(candidate.issue_surface_signal or {}),
                            "loop_summary": dict(candidate.loop_summary or {}),
                            "selection_evidence": _testgen_candidate_selection_evidence(
                                candidate,
                                f2p_payloads=per_candidate_f2p or None,
                                selected_rollout_id=selected_rollout_id,
                            ),
                        }
                        for candidate in rollout_candidates
                    ],
                    "rollout_failures": rollout_failures,
                    "shared_testgen_memory_bus_enabled": True,
                    "gold_oracle_selection": {
                        "allowed": self.allow_gold_oracle_selection,
                        "per_candidate_f2p_used": bool(per_candidate_f2p),
                        "per_candidate_f2p_skipped_for_methodology": (
                            per_candidate_f2p_skipped_for_methodology
                        ),
                    },
                },
            )
            generated_portfolio_path = task_output_dir / "generated_test_portfolio.json"
            atomic_write_json(generated_portfolio_path, generated_portfolio)

            write_task_live_state(
                task_output_dir,
                {
                    "task_id": task.instance_id,
                    "phase": "compare_gold",
                    "status": "active",
                    "process_pid": os.getpid(),
                    "last_progress_at": time.time(),
                },
            )
            # Build the comparator packet and run the offline gold comparison.
            # ``base_repo_root=repo_dir`` (not the rollout worktree) is what
            # ``materialize_gold_test_files`` reads from, but defend against
            # FileNotFoundError anyway: if any filesystem dependency under
            # the comparator vanishes (worktree zapped by a rogue agentic
            # tool, sandbox already torn down by an outer signal handler,
            # etc.) we still want a comparison artifact on disk so the run
            # report can distinguish "missing inputs" from "real APEX miss".
            comparison: dict[str, Any] = {}
            comparison_status = "ok" if rollout_candidates else "no_generated_portfolio"
            comparison_error: Optional[str] = None
            try:
                comparison_packet = build_swebench_gold_comparison_packet_for_task(
                    task=task,
                    base_repo_root=repo_dir,
                    generated_portfolio=generated_portfolio,
                    test_command=test_command,
                    issue_description=issue_description,
                )
                comparison = compare_generated_and_gold_portfolios(comparison_packet)
            except FileNotFoundError as exc:
                comparison_status = "worktree_unavailable"
                comparison_error = str(exc)
                logger.warning(
                    "Gold comparator could not access required files for %s: %s",
                    task.instance_id,
                    exc,
                )
            except Exception as exc:
                comparison_status = "comparator_error"
                comparison_error = str(exc)
                logger.warning(
                    "Gold comparator failed for %s: %s",
                    task.instance_id,
                    exc,
                )

            semantic_review_error: Optional[str] = None
            semantic_review_tokens = 0
            if comparison_status == "ok":
                try:
                    comparison, semantic_review_error, semantic_review_tokens = (
                        self._apply_semantic_review_to_comparison(
                            task=task,
                            comparison=comparison,
                            working_dir=str(repo_dir),
                            issue_description=issue_description,
                            required_contract_targets=issue_contract_targets,
                            test_command=test_command,
                        )
                    )
                except FileNotFoundError as exc:
                    semantic_review_error = str(exc)
                    logger.warning(
                        "Semantic review could not access working dir for %s: %s",
                        task.instance_id,
                        exc,
                    )

            comparison["comparison_status"] = comparison_status
            comparison["rollout_failures"] = list(rollout_failures)
            if test_quality_summary:
                comparison["test_quality_summary"] = dict(test_quality_summary)
            if portfolio_validation:
                comparison["apex_validation"] = dict(portfolio_validation)
            if comparison_error:
                comparison["comparison_error"] = comparison_error

            # Optional Fail-to-Pass execution oracle. The structural comparator
            # only checks recall against gold; F2P actually runs the candidate
            # tests against ``base_commit`` (broken) and ``base_commit + patch``
            # (fixed) and counts how many transition fail->pass. This is the
            # oracle for stages 4-5 of test_generation_design.md and the
            # metric public benchmarks (SWT-Bench, TDD-Bench) actually score on.
            f2p_summary: dict[str, Any] = {}
            if self.enable_f2p and rollout_candidates:
                try:
                    from apex.evaluation.f2p_oracle import (
                        evaluate_f2p,
                        write_f2p_artifact,
                    )

                    selected_id = (
                        selected_candidate.rollout_id if selected_candidate is not None else None
                    )
                    cached_payload = (
                        per_candidate_f2p.get(selected_id)
                        if (
                            selected_id is not None and portfolio_selection_mode == "single_rollout"
                        )
                        else None
                    )
                    if cached_payload is not None:
                        # Reuse the per-candidate payload computed before
                        # selection — saves a redundant 2-sandbox install
                        # and (when mutation was enabled) the kept sandbox
                        # is already cleaned up.
                        f2p_payload = cached_payload
                    else:
                        f2p_artifacts = list(
                            (generated_portfolio or {}).get("test_artifacts") or []
                        )
                        assertion_mutation_enabled = _iteration_assertion_mutation_enabled()
                        f2p_payload = evaluate_f2p(
                            task=task,
                            repo_dir=str(repo_dir),
                            test_artifacts=f2p_artifacts,
                            output_dir=str(task_output_dir),
                            language=str(task.repo_language or "python"),
                            broken_timeout_seconds=self.f2p_per_side_timeout_seconds,
                            fixed_timeout_seconds=self.f2p_per_side_timeout_seconds,
                            install_repo=self.f2p_install_repo,
                            keep_sandboxes=(self.enable_mutation or assertion_mutation_enabled),
                        )
                        if self.enable_mutation:
                            mutation_report = self._run_mutation_for_payload(
                                task=task,
                                f2p_payload=f2p_payload,
                            )
                            if mutation_report is not None:
                                f2p_payload["mutation"] = mutation_report
                        if assertion_mutation_enabled:
                            assertion_report = self._run_assertion_mutation_for_payload(
                                task=task,
                                f2p_payload=f2p_payload,
                            )
                            if assertion_report is not None:
                                f2p_payload["assertion_mutation"] = assertion_report
                        if self.enable_mutation or assertion_mutation_enabled:
                            sandboxes_root = Path(f2p_payload.get("broken_path") or "").parent
                            if sandboxes_root.exists():
                                shutil.rmtree(sandboxes_root, ignore_errors=True)
                    write_f2p_artifact(output_dir=str(task_output_dir), payload=f2p_payload)
                    f2p_summary = dict(f2p_payload.get("summary") or {})
                    f2p_summary["status"] = f2p_payload.get("status")
                    f2p_summary["selected_via"] = (
                        "f2p_positive_rollout_union"
                        if portfolio_selection_mode == "f2p_positive_rollout_union"
                        else "per_candidate_f2p"
                        if cached_payload is not None
                        else "post_selection_f2p"
                    )
                    mutation_payload = dict(f2p_payload.get("mutation") or {})
                    if mutation_payload:
                        f2p_summary["mutation_attempted"] = True
                        f2p_summary["mutation_measured"] = _mutation_payload_measured(
                            mutation_payload
                        )
                        f2p_summary["mutation_score"] = float(
                            mutation_payload.get("mutation_score") or 0.0
                        )
                        f2p_summary["mutation_killed"] = int(mutation_payload.get("killed") or 0)
                        f2p_summary["mutation_total"] = int(
                            mutation_payload.get("total_mutants") or 0
                        )
                        f2p_summary["effective_mutation_evaluable"] = int(
                            mutation_payload.get("effective_mutation_evaluable") or 0
                        )
                        f2p_summary["mutation_score_denominator"] = str(
                            mutation_payload.get("mutation_score_denominator") or ""
                        )
                        mutation_skip_reason = _mutation_payload_skip_reason(mutation_payload)
                        if mutation_skip_reason:
                            f2p_summary["mutation_skip_reason"] = mutation_skip_reason
                        if mutation_payload.get("error"):
                            f2p_summary["mutation_error"] = str(mutation_payload.get("error"))
                    comparison["f2p_summary"] = dict(f2p_summary)
                    if mutation_payload:
                        comparison["mutation_summary"] = mutation_payload
                    assertion_payload = dict(f2p_payload.get("assertion_mutation") or {})
                    if assertion_payload:
                        f2p_summary["assertion_mutation_attempted"] = True
                        f2p_summary["assertion_mutation_measured"] = (
                            str(assertion_payload.get("status") or "") == "ok"
                        )
                        f2p_summary["assertion_mutation_survived"] = bool(
                            assertion_payload.get("survived")
                        )
                        f2p_summary["assertion_mutation_mutated_assertion_count"] = int(
                            assertion_payload.get("mutated_assertion_count") or 0
                        )
                        comparison["assertion_mutation_summary"] = assertion_payload
                        comparison["f2p_summary"] = dict(f2p_summary)

                    # Stage 5: minimize the suite using the F2P + mutation
                    # coverage maps we just computed. We persist BOTH the
                    # original portfolio (so we can audit how aggressive the
                    # minimization was) and the minimized one (which becomes
                    # the canonical artifact for downstream consumers).
                    if (
                        self.enable_minimization
                        and selected_candidate is not None
                        and bool(f2p_summary.get("any_f2p"))
                    ):
                        try:
                            from apex.evaluation.test_minimizer import (
                                minimize_suite,
                            )

                            original_artifacts = list(
                                (generated_portfolio or {}).get("test_artifacts") or []
                            )
                            minimized_artifacts, min_report = minimize_suite(
                                test_artifacts=original_artifacts,
                                f2p_payload=f2p_payload,
                                mutation_report=mutation_payload or None,
                            )
                            comparison["minimized_test_artifacts"] = list(minimized_artifacts)
                            comparison["minimization_summary"] = min_report.to_dict()
                            f2p_summary["minimized_count"] = min_report.minimized_count
                            f2p_summary["original_count"] = min_report.original_count
                            comparison["f2p_summary"] = dict(f2p_summary)
                        except Exception as min_exc:  # pragma: no cover
                            logger.warning(
                                "Minimization failed for %s: %s",
                                task.instance_id,
                                min_exc,
                            )
                            comparison["minimization_summary"] = {
                                "skipped": True,
                                "skip_reason": f"error:{type(min_exc).__name__}",
                            }
                except Exception as f2p_exc:  # pragma: no cover - defensive
                    logger.warning("F2P oracle failed for %s: %s", task.instance_id, f2p_exc)
                    f2p_summary = {"status": f"error:{type(f2p_exc).__name__}"}
                    comparison["f2p_summary"] = dict(f2p_summary)

            comparison_path = task_output_dir / "synthetic_vs_gold_comparison.json"
            hard_quality_gate_failures: list[str] = []
            generated_artifacts = [
                item
                for item in list((generated_portfolio or {}).get("test_artifacts") or [])
                if isinstance(item, dict)
            ]
            contract_metadata_gaps = [
                {
                    "path": str(item.get("path") or ""),
                    "artifact_id": str(item.get("artifact_id") or ""),
                    "missing_or_vague_fields": gaps,
                }
                for item in generated_artifacts
                if (gaps := _artifact_contract_metadata_gaps(item))
            ]
            if contract_metadata_gaps:
                comparison["artifact_contract_metadata_gaps"] = contract_metadata_gaps
            if not generated_artifacts:
                hard_quality_gate_failures.append("no_generated_test_artifacts")
            elif portfolio_validation and portfolio_validation.get("status") != "pass":
                hard_quality_gate_failures.append("static_validation_failed")
            elif contract_metadata_gaps:
                hard_quality_gate_failures.append("artifact_contract_metadata_failed")
            if self.enable_f2p:
                if not bool(f2p_summary.get("any_f2p")):
                    hard_quality_gate_failures.append("no_fail_to_pass_tests")
                if bool(f2p_summary.get("unreliable_execution")):
                    hard_quality_gate_failures.append("unreliable_f2p_execution")
            if self.enable_mutation and bool(f2p_summary.get("mutation_measured")):
                if float(f2p_summary.get("mutation_score") or 0.0) <= 0.0:
                    hard_quality_gate_failures.append("zero_mutation_score")
            if bool(f2p_summary.get("assertion_mutation_measured")) and bool(
                f2p_summary.get("assertion_mutation_survived")
            ):
                hard_quality_gate_failures.append("assertion_mutation_survived")
            if test_quality_summary:
                artifact_count = int(test_quality_summary.get("artifact_count") or 0)
                weak_count = int(test_quality_summary.get("weak_artifact_count") or 0)
                mean_effect = float(test_quality_summary.get("mean_assertion_effect_score") or 0.0)
                if artifact_count > 0 and weak_count >= artifact_count and mean_effect <= 0.0:
                    hard_quality_gate_failures.append("static_oracle_quality_failed")
            coverage_summary = dict(comparison.get("coverage_summary") or {})
            if generated_artifacts and coverage_summary:
                target_recall = float(coverage_summary.get("gold_target_recall") or 0.0)
                axis_recall = float(coverage_summary.get("overall_contract_axis_recall") or 0.0)
                if target_recall <= 0.0 and axis_recall <= 0.0:
                    hard_quality_gate_failures.append("no_target_or_axis_coverage")
            if hard_quality_gate_failures:
                comparison["hard_quality_gate_failures"] = list(hard_quality_gate_failures)
            atomic_write_json(comparison_path, comparison)

            task_success = comparison_status == "ok" and not hard_quality_gate_failures
            write_task_live_state(
                task_output_dir,
                {
                    "task_id": task.instance_id,
                    "phase": "completed" if task_success else "degraded",
                    "status": "completed" if task_success else "failed",
                    "process_pid": os.getpid(),
                    "last_progress_at": time.time(),
                    "comparison_status": comparison_status,
                    "failed_rollout_count": len(rollout_failures),
                },
            )
            failure_reason = (
                None
                if task_success
                else (
                    "; ".join(hard_quality_gate_failures)
                    if hard_quality_gate_failures
                    else comparison_error
                    or (
                        f"All {planned_rollout_count} rollout(s) failed: "
                        + "; ".join(
                            f"rollout_{item.get('rollout_id')}={item.get('error')}"
                            for item in rollout_failures
                        )
                        if rollout_failures
                        else "Gold comparator did not produce a usable result."
                    )
                )
            )
            # Phase I.7: persist cross-task testgen insights AFTER the
            # task summaries are finalized, so the next task on the
            # same repo can pick them up. Guarded by enable_testgen_memory.
            self._persist_testgen_memory_for_task(
                task=task,
                issue_plan=issue_plan,
                f2p_summary=f2p_summary,
                comparison=comparison,
            )
            return SWEBenchProTestGenTaskResult(
                instance_id=task.instance_id,
                repo=task.repo,
                success=task_success,
                generated_summary=dict(comparison.get("generated_summary") or {}),
                gold_summary=dict(comparison.get("gold_summary") or {}),
                coverage_summary=dict(comparison.get("coverage_summary") or {}),
                target_comparison=list(comparison.get("target_comparison") or []),
                semantic_review=dict(comparison.get("semantic_review") or {}),
                issue_contract_targets=issue_contract_targets,
                total_tokens=(
                    int(issue_plan.planner_tokens or 0)
                    + int(test_tokens or 0)
                    + int(semantic_review_tokens or 0)
                ),
                duration_seconds=time.time() - started,
                result_path=str(comparison_path),
                generated_portfolio_path=str(generated_portfolio_path),
                semantic_review_error=semantic_review_error,
                failure_reason=failure_reason,
                execution_metadata={
                    "planner_source": str(issue_plan.planner_source or ""),
                    "orchestration_primitives": list(issue_plan.orchestration_primitives or []),
                    "test_command": str(test_command or ""),
                    "rollout_count": self.rollout_count,
                    "executed_rollout_count": len(rollout_candidates),
                    "failed_rollout_count": len(rollout_failures),
                    "selected_rollout_id": selected_rollout_id,
                    "portfolio_selection_mode": portfolio_selection_mode,
                    "union_source_rollout_ids": list(
                        generated_portfolio.get("source_rollout_ids") or []
                    ),
                    "shared_testgen_memory_bus_enabled": True,
                    "repo_language": str(task.repo_language or ""),
                    "semantic_review": self.semantic_review,
                    "semantic_review_model": (
                        self.semantic_review_model or self.config.selection.judge_model or ""
                    ),
                    "comparison_status": comparison_status,
                    "test_quality_summary": dict(test_quality_summary or {}),
                    "apex_validation": dict(portfolio_validation or {}),
                    "f2p_enabled": self.enable_f2p,
                    "allow_gold_oracle_selection": self.allow_gold_oracle_selection,
                    "per_candidate_f2p_used_for_selection": bool(per_candidate_f2p),
                    "per_candidate_f2p_skipped_for_methodology": (
                        per_candidate_f2p_skipped_for_methodology
                    ),
                    "f2p_summary": dict(f2p_summary or {}),
                },
            )
        except Exception as exc:
            skipped, skip_category = self._classify_environment_failure(exc)
            write_task_live_state(
                task_output_dir,
                {
                    "task_id": task.instance_id,
                    "phase": "failed",
                    "status": "failed",
                    "process_pid": os.getpid(),
                    "last_progress_at": time.time(),
                    "error": str(exc),
                    "skipped": skipped,
                    "skip_category": skip_category,
                },
            )
            return SWEBenchProTestGenTaskResult(
                instance_id=task.instance_id,
                repo=task.repo,
                success=False,
                duration_seconds=time.time() - started,
                failure_reason=str(exc),
                issue_contract_targets=issue_contract_targets,
                skipped=skipped,
                skip_category=skip_category,
                execution_metadata={
                    "rollout_count": self.rollout_count,
                    "repo_language": str(task.repo_language or ""),
                },
            )
        finally:
            if worktree_manager is not None:
                try:
                    worktree_manager.cleanup_all()
                except Exception as cleanup_exc:  # pragma: no cover - defensive
                    logger.warning(
                        "Worktree cleanup failed for %s: %s", task.instance_id, cleanup_exc
                    )
            self._cleanup_task_docker_artifacts(task)
            shutil.rmtree(sandbox, ignore_errors=True)

    def _classify_environment_failure(
        self,
        exc_or_message: Exception | str,
    ) -> tuple[bool, Optional[str]]:
        message = str(exc_or_message).lower()
        if "docker" in message and ("not found" in message or "cannot connect" in message):
            return True, "unsupported_host"
        if "no space left on device" in message:
            return True, "host_storage_exhausted"
        if "failed to retrieve image list" in message or "failed to perform sync" in message:
            return True, "artifact_sync_failure"

        container_runtime_markers = (
            "containerd",
            "/var/lib/containerd",
            "meta.db",
            "content digest",
            "content store",
            "content ingest",
            "blob",
            "layer",
            "image pull",
            "docker pull",
        )
        if any(marker in message for marker in container_runtime_markers) and (
            "input/output error" in message or "i/o error" in message
        ):
            return True, "container_runtime_failure"
        if (
            ("docker" in message or "container" in message)
            and any(
                marker in message
                for marker in ("image", "pull", "blob", "content", "storage", "layer")
            )
            and any(marker in message for marker in ("input/output error", "i/o error", "write "))
        ):
            return True, "container_runtime_failure"
        return False, None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m apex.evaluation.swebench_pro_testgen_eval",
        description="Run SWE-Bench Pro synthetic test generation evaluation.",
    )
    parser.add_argument("--config", required=True, help="Path to the Apex JSON config")
    parser.add_argument("--output", help="Optional output directory")
    parser.add_argument("--dataset-name", default=SWEBENCH_PRO_DATASET_NAME)
    parser.add_argument("--dataset-split", default=SWEBENCH_PRO_DATASET_SPLIT)
    parser.add_argument("--dockerhub-username", default=SWEBENCH_PRO_DOCKERHUB_USERNAME)
    parser.add_argument("--scripts-cache-dir", default=None)
    parser.add_argument("--docker-platform", default=None)
    parser.add_argument("--block-network", action="store_true")
    parser.add_argument(
        "--rollout-count",
        type=int,
        default=4,
        help=(
            "Number of independent test_writer rollouts per task. The "
            "F2P/mutation/judge selector needs multiple candidates to "
            "discriminate; default 4 gives best-of-4 selection without "
            "burning excessive compute. Pass --rollout-count 1 to match "
            "the benchmark's canonical_rollout_count for parity reporting."
        ),
    )
    parser.add_argument("--task-parallelism", type=int, default=None)
    parser.add_argument(
        "--task-timeout-seconds",
        type=float,
        default=1800.0,
        help=(
            "Strict wall-clock timeout for each benchmark task. Each task runs "
            "in a child process; on timeout Apex writes task_result.json with "
            "skip_category=task_timeout and continues to the next task. Pass "
            "0 to preserve legacy in-process execution."
        ),
    )
    parser.add_argument(
        "--enable-testgen-delegation",
        action="store_true",
        help=(
            "Allow orchestrated multi-agent delegation during SWE-Bench Pro "
            "test generation. Default is disabled for benchmark slices until "
            "delegated children have strict inherited deadlines."
        ),
    )
    parser.add_argument(
        "--semantic-review",
        action="store_true",
        help="Use an LLM judge to semantically compare generated tests against gold tests.",
    )
    parser.add_argument(
        "--semantic-review-model",
        default=None,
        help="Optional semantic-review model override on the primary evaluation backend.",
    )
    parser.add_argument(
        "--prepare-repo-mode",
        choices=[
            SWEBENCH_PREPARE_REPO_FROM_DOCKER_IMAGE,
            SWEBENCH_PREPARE_REPO_FROM_HOST_GIT,
        ],
        default=SWEBENCH_PREPARE_REPO_FROM_DOCKER_IMAGE,
        help="How to materialize benchmark repos before test generation.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--repos", nargs="*")
    parser.add_argument("--languages", nargs="*")
    parser.add_argument(
        "--enable-f2p",
        action="store_true",
        help=(
            "After gold comparison, run candidate tests against base_commit "
            "(broken) and base_commit+gold_patch (fixed) and report Fail-to-Pass "
            "transitions. Adds ~1-3 minutes per task."
        ),
    )
    parser.add_argument(
        "--f2p-install-repo",
        dest="f2p_install_repo",
        action="store_true",
        default=True,
        help=(
            "Attempt `pip install -e .` in each F2P sandbox before running "
            "tests (default). Most non-trivial repos (ansible, openlibrary, "
            "etc.) need this to import their own modules — without it pytest "
            "fails collection on `ModuleNotFoundError`."
        ),
    )
    parser.add_argument(
        "--no-f2p-install-repo",
        dest="f2p_install_repo",
        action="store_false",
        help=(
            "Skip `pip install -e .` in F2P sandboxes. Useful for repos with "
            "heavy install side effects, or for fast smoke runs where the "
            "test files have no repo-internal imports."
        ),
    )
    parser.add_argument(
        "--f2p-per-side-timeout-seconds",
        type=float,
        default=300.0,
        help="Wall-clock budget for the broken and fixed pytest invocations (default 300s each).",
    )
    parser.add_argument(
        "--enable-mutation",
        action="store_true",
        help=(
            "Stage 4 mutation discrimination. After F2P confirms a candidate "
            "catches the bug, generate AST-based mutants of the gold-patched "
            "source files and run the candidate's tests against each. Reports "
            "a real mutation_score in addition to F2P. Implies --enable-f2p."
        ),
    )
    parser.add_argument(
        "--mutation-max-per-file",
        type=int,
        default=16,
        help="Cap mutants per source file (default 16). Higher = better signal, longer runtime.",
    )
    parser.add_argument(
        "--mutation-max-files",
        type=int,
        default=3,
        help=(
            "Cap source files mutated per task (default 3). Files are taken "
            "in the order they appear in the gold patch."
        ),
    )
    parser.add_argument(
        "--mutation-per-mutant-timeout-seconds",
        type=float,
        default=60.0,
        help="Per-mutant pytest wall-clock budget (default 60s). Above this the mutant is recorded as timeout.",
    )
    parser.add_argument(
        "--enable-minimization",
        action="store_true",
        help=(
            "Stage 5 minimization. Greedy set-cover over F2P + mutation "
            "kills, drops files whose contribution is fully subsumed by "
            "kept files. Implies --enable-f2p; recommended with --enable-mutation."
        ),
    )
    parser.add_argument(
        "--enable-testgen-judge",
        action="store_true",
        help=(
            "Phase E.3 LLM judge for selection ties. When the F2P-tuple "
            "ranking ties multiple candidates, an LLM reads each candidate's "
            "F2P/mutation outcomes + sample test paths and picks the "
            "suite most likely to catch production bugs. One LLM call "
            "per task (when invoked). Implies --enable-f2p; only useful "
            "with --rollout-count > 1."
        ),
    )
    parser.add_argument(
        "--allow-gold-oracle-selection",
        action="store_true",
        help=(
            "Allow pre-selection F2P, mutation, and judge ranking to use the "
            "gold patch as an oracle. This is an ablation/debug mode; default "
            "benchmark runs select candidates without gold-patch feedback."
        ),
    )
    parser.add_argument(
        "--enable-testgen-memory",
        action="store_true",
        help=(
            "Phase I.7 cross-task persistent testgen memory. After each "
            "task, extract insights (focus-file hotspots, F2P bug patterns, "
            "mutation classes killed/survived, coverage gaps) and merge "
            "into a per-repo store. The next task on the same repo gets "
            "those insights as priors in its test_writer prompt. OPT-IN "
            "for benchmarks because of contamination risk: prior-task "
            "insights leak benchmark-derived signal across the eval set. "
            "Default off; methodology block reports per-task load/persist "
            "counts when on."
        ),
    )
    parser.add_argument(
        "--testgen-memory-directory",
        default=None,
        help=(
            "Override the cross-task memory directory. Defaults to "
            "<output>/_testgen_memory which gives EACH BENCHMARK RUN an "
            "isolated store (no cross-benchmark-run contamination). Set "
            "this to a shared path only if you explicitly want memory "
            "to persist across benchmark invocations (e.g., real-world "
            "TDD use)."
        ),
    )
    args = parser.parse_args(argv)

    config = ApexConfig.from_file(args.config)
    if args.task_parallelism is not None:
        config.benchmark.task_parallelism = max(1, int(args.task_parallelism))
    if args.task_timeout_seconds is not None:
        config.benchmark.testgen_task_timeout_seconds = max(
            0.0,
            float(args.task_timeout_seconds or 0.0),
        )
    output_dir = (
        Path(args.output).resolve()
        if args.output
        else default_swebench_pro_testgen_output_dir(config)
    )
    semantic_review_enabled = bool(args.semantic_review or args.semantic_review_model)
    evaluator = SWEBenchProTestGenEvaluator(
        config,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        dockerhub_username=args.dockerhub_username,
        scripts_cache_dir=args.scripts_cache_dir,
        docker_platform=args.docker_platform,
        block_network=args.block_network,
        rollout_count=args.rollout_count,
        prepare_repo_mode=args.prepare_repo_mode,
        semantic_review=semantic_review_enabled,
        semantic_review_model=args.semantic_review_model,
        enable_f2p=(
            args.enable_f2p
            or args.enable_mutation
            or args.enable_minimization
            or args.enable_testgen_judge
        ),
        f2p_install_repo=args.f2p_install_repo,
        f2p_per_side_timeout_seconds=args.f2p_per_side_timeout_seconds,
        enable_mutation=args.enable_mutation,
        mutation_max_per_file=args.mutation_max_per_file,
        mutation_max_files=args.mutation_max_files,
        mutation_per_mutant_timeout_seconds=args.mutation_per_mutant_timeout_seconds,
        enable_minimization=args.enable_minimization,
        enable_testgen_judge=args.enable_testgen_judge,
        enable_testgen_memory=args.enable_testgen_memory,
        testgen_memory_directory=args.testgen_memory_directory,
        enable_testgen_delegation=args.enable_testgen_delegation,
        allow_gold_oracle_selection=args.allow_gold_oracle_selection,
    )
    evaluator.config_source = str(Path(args.config).resolve())
    report = evaluator.run(
        instances=list(args.instances or []),
        repos=list(args.repos or []),
        languages=list(args.languages or []),
        limit=args.limit,
    )
    print(json.dumps(report.aggregate_metrics, indent=2))
    return (
        0
        if report.failed_tasks == 0
        and report.skipped_tasks == 0
        and report.completed_tasks == report.total_tasks
        else 1
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
