"""APEX orchestrator class (Phase 3.2 decomposed package).

Houses ``ApexOrchestrator`` and ``ApexResult``. The class previously
lived in ``apex/orchestrator.py``; that path is now a thin shim that
re-exports from this module for back-compat.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, cast

from ..acceptance import (
    quick_verification_expected_coverage_ratio,
    quick_verification_has_local_full_scope_pass,
    quick_verification_has_strong_signal,
    quick_verification_signal_score,
    rollout_has_authoritative_acceptance,
    rollout_has_authoritative_scoring_stop_signal,
    rollout_has_materialized_repair_seed,
    rollout_has_preemptive_authoritative_scoring_request,
    rollout_has_preemptive_completion,
    rollout_has_repairable_near_miss,
    rollout_has_submission_blocking_validity,
    rollout_made_collection_progress_vs_seed,
    rollout_requires_authoritative_scoring,
    verification_has_explicit_validity_rejection,
)
from ..agents.artifacts import (
    LocalizationArtifact,
    ReproductionArtifact,
    coerce_localization_artifact,
    coerce_reproduction_artifact,
)
from ..agents.solver import (
    reset_active_repo_episodes,
    set_active_repo_episodes,
)
from ..controller_policy import (
    derive_test_collection_command,
    infer_test_inventory_framework,
)
from ..controller_trace import append_controller_decision
from ..core.config import (
    AGENT_MODE_CHOICES,
    GLOBAL_DEFAULT_AGENT_MODE,
    ApexConfig,
    SearchMode,
)
from ..core.failure_classifier import (
    ClassificationResult as CoreClassificationResult,
)
from ..core.failure_classifier import (
    FailureClass as CoreFailureClass,
)
from ..core.failure_classifier import (
    classify_failure,
)
from ..core.llm_routing import llm_backend_is_available
from ..core.status import Status
from ..planning.manager import IssuePlan, IssuePlanner
from ..preprocessing.repo_analyzer import RepoAnalyzer, RepoContext
from ..rollout.engine import (
    RolloutEngine,
    RolloutResult,
    WorkspaceSeed,
    _issue_plan_expected_test_count,
    _quick_verification_blocker_summary,
    _quick_verification_followup_guidance,
    _quick_verification_structural_blocker_kind,
    _rollout_budget_size_factor,
    build_workspace_seed_from_rollout_result,
)
from ..rollout.localizer_scope import is_apex_harness_path, is_test_path
from ..rollout.patch_sanitizer import (
    PatchPathCategory,
    classify_patch_path,
    filter_solution_paths,
)
from ..search.frontier_search import FrontierSearchController
from ..selection.selector import PatchSelector
from ..selection.verifier import BaselineResult, PatchVerifier
from ..task_state_graph import TaskStateGraph
from . import recovery as _recovery
from .abstention import ConfidenceBreakdown, ConfidenceScorer

_INHERIT_VERIFICATION_TEST_COMMAND = object()


def _resolve_patched(name: str, default: Any) -> Any:
    """Phase 3.2 shim: callers (existing tests) monkeypatch
    ``apex.orchestrator.X`` directly. The orchestrator class lives in
    this submodule, so direct ``X(...)`` lookups bypass that
    monkeypatch. Resolving via the legacy ``apex.orchestrator`` module
    when present preserves the old behavior; falls back to the import
    bound at module load time when the shim isn't imported yet
    (e.g. very early bootstrap during __init__)."""
    import sys

    legacy = sys.modules.get("apex.orchestrator")
    if legacy is None:
        return default
    return getattr(legacy, name, default)


def _jaccard_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        # Two empty footprints are conservatively treated as identical
        # so we don't flood the seed slate with empty-changed-files
        # rollouts (those produce nothing to branch from).
        return 1.0
    if not a or not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / float(union)


_CLEAN_BASELINE_FILE_SCOPE_KEYS = frozenset(
    {
        "action_file_paths",
        "allowed_files",
        "boundary_requested_files",
        "bridge_files",
        "file_paths",
        "focus_files",
        "forbidden_files",
        "graph_target_file_paths",
        "owned_files",
        "peer_files",
        "relevant_files",
        "risk_files",
    }
)


def _clean_baseline_file_hint_text(value: Any) -> str:
    text = str(value or "").split("::", 1)[0].strip().strip("`'\"")
    if not text or "\n" in text or "\r" in text:
        return ""
    text = text.replace("\\", "/")
    # Candidate artifacts may report copied worktree paths instead of repo
    # paths. Residual planning must reason over the candidate repo namespace.
    text = re.sub(r"^(?:.*?/)?workspaces/_pool/[^/]+/workspace/", "", text)
    text = re.sub(r"^(?:.*?/)?workspaces/[^/]+/workspace/", "", text)
    text = re.sub(r"^workspace/workspaces/_pool/[^/]+/", "", text)
    text = re.sub(r"^workspace/workspaces/[^/]+/", "", text)
    text = re.sub(r":\d+(?::\d+)?$", "", text)
    return text.strip()


def _safe_repo_relative_path(value: str) -> str:
    text = str(value or "").replace("\\", "/").lstrip("./")
    if not text:
        return ""
    path = Path(text)
    if path.is_absolute():
        return ""
    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return ""
    if parts[0] == ".git":
        return ""
    return path.as_posix()


def _repo_relative_file_hint(repo_path: str, text: str) -> str:
    if not text:
        return ""
    candidate_path = Path(text)
    if not candidate_path.is_absolute():
        return _safe_repo_relative_path(text)
    try:
        repo_root = Path(repo_path).resolve()
        relative = candidate_path.resolve().relative_to(repo_root)
    except (OSError, ValueError):
        return ""
    return _safe_repo_relative_path(relative.as_posix())


def _baseline_file_hint_exists(
    repo_path: str,
    repo_context: RepoContext,
    candidate: str,
) -> bool:
    if repo_context.get_file_info(candidate) is not None:
        return True
    try:
        return (Path(repo_path) / candidate).is_file()
    except OSError:
        return False


def _clean_baseline_repo_file_hint(
    repo_path: str,
    repo_context: RepoContext,
    value: Any,
) -> Optional[str]:
    text = _clean_baseline_file_hint_text(value)
    if not text:
        return None
    candidates = []
    direct = _repo_relative_file_hint(repo_path, text)
    if direct:
        candidates.append(direct)
    normalized = repo_context.normalize_repo_path_candidate(text)
    if normalized:
        normalized = _safe_repo_relative_path(normalized)
        if normalized:
            candidates.append(normalized)
    for candidate in dict.fromkeys(candidates):
        if _baseline_file_hint_exists(repo_path, repo_context, candidate):
            return candidate
    return None


def _clean_baseline_existing_file_hints(
    repo_path: str,
    repo_context: RepoContext,
    values: list[Any] | tuple[Any, ...] | set[Any],
) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    dropped: list[str] = []
    for value in values:
        text = _clean_baseline_file_hint_text(value)
        if not text:
            continue
        normalized = _clean_baseline_repo_file_hint(repo_path, repo_context, text)
        if normalized:
            if normalized not in kept:
                kept.append(normalized)
        elif text not in dropped:
            dropped.append(text)
    return kept, dropped


def _scrub_clean_baseline_edit_spans(
    repo_path: str,
    repo_context: RepoContext,
    values: Any,
) -> tuple[list[Any], list[str]]:
    if not isinstance(values, (list, tuple, set)):
        return [], []
    kept: list[Any] = []
    dropped: list[str] = []
    seen: set[tuple[str, str, int, int]] = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        raw_path = value.get("file_path")
        normalized = _clean_baseline_repo_file_hint(repo_path, repo_context, raw_path)
        if not normalized:
            text = _clean_baseline_file_hint_text(raw_path)
            if text and text not in dropped:
                dropped.append(text)
            continue
        cloned = dict(value)
        cloned["file_path"] = normalized
        key = (
            normalized,
            str(cloned.get("symbol") or ""),
            int(cloned.get("start_line") or 0),
            int(cloned.get("end_line") or 0),
        )
        if key in seen:
            continue
        seen.add(key)
        kept.append(cloned)
    return kept, dropped


def _scrub_clean_baseline_file_scope_mapping(
    repo_path: str,
    repo_context: RepoContext,
    mapping: Any,
) -> tuple[Any, list[str]]:
    if not isinstance(mapping, dict):
        return mapping, []
    scrubbed: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in mapping.items():
        if key in _CLEAN_BASELINE_FILE_SCOPE_KEYS and isinstance(
            value,
            (list, tuple, set),
        ):
            kept, missing = _clean_baseline_existing_file_hints(
                repo_path,
                repo_context,
                value,
            )
            scrubbed[key] = kept
            dropped.extend(missing)
            continue
        if key == "edit_spans":
            kept_spans, missing = _scrub_clean_baseline_edit_spans(
                repo_path,
                repo_context,
                value,
            )
            scrubbed[key] = kept_spans
            dropped.extend(missing)
            continue
        if isinstance(value, dict):
            nested, missing = _scrub_clean_baseline_file_scope_mapping(
                repo_path,
                repo_context,
                value,
            )
            scrubbed[key] = nested
            dropped.extend(missing)
            continue
        if isinstance(value, list):
            nested_values: list[Any] = []
            for item in value:
                if isinstance(item, dict):
                    nested, missing = _scrub_clean_baseline_file_scope_mapping(
                        repo_path,
                        repo_context,
                        item,
                    )
                    nested_values.append(nested)
                    dropped.extend(missing)
                else:
                    nested_values.append(item)
            scrubbed[key] = nested_values
            continue
        scrubbed[key] = value
    return scrubbed, dropped


def _scrub_clean_baseline_controller_action(
    repo_path: str,
    repo_context: RepoContext,
    action: Any,
) -> list[str]:
    dropped: list[str] = []
    file_paths, missing = _clean_baseline_existing_file_hints(
        repo_path,
        repo_context,
        list(getattr(action, "file_paths", []) or []),
    )
    action.file_paths = file_paths
    dropped.extend(missing)
    kept_spans = []
    seen: set[tuple[str, str, int, int]] = set()
    for span in list(getattr(action, "edit_spans", []) or []):
        raw_path = getattr(span, "file_path", "")
        normalized = _clean_baseline_repo_file_hint(repo_path, repo_context, raw_path)
        if not normalized:
            text = _clean_baseline_file_hint_text(raw_path)
            if text and text not in dropped:
                dropped.append(text)
            continue
        span.file_path = normalized
        key = (
            normalized,
            str(getattr(span, "symbol", "") or ""),
            int(getattr(span, "start_line", 0) or 0),
            int(getattr(span, "end_line", 0) or 0),
        )
        if key in seen:
            continue
        seen.add(key)
        kept_spans.append(span)
    action.edit_spans = kept_spans
    return dropped


def _scrub_clean_baseline_issue_plan_file_hints(
    issue_plan: IssuePlan,
    repo_path: str,
    repo_context: RepoContext,
) -> list[str]:
    dropped: list[str] = []
    issue_plan.relevant_files, missing = _clean_baseline_existing_file_hints(
        repo_path,
        repo_context,
        list(issue_plan.relevant_files or []),
    )
    dropped.extend(missing)
    issue_plan.risk_files, missing = _clean_baseline_existing_file_hints(
        repo_path,
        repo_context,
        list(issue_plan.risk_files or []),
    )
    dropped.extend(missing)
    for brief in list(issue_plan.rollout_briefs or []):
        brief.focus_files, missing = _clean_baseline_existing_file_hints(
            repo_path,
            repo_context,
            list(brief.focus_files or []),
        )
        dropped.extend(missing)
        search_policy, missing = _scrub_clean_baseline_file_scope_mapping(
            repo_path,
            repo_context,
            brief.search_policy,
        )
        brief.search_policy = search_policy
        dropped.extend(missing)
        action = brief.resolved_controller_action()
        dropped.extend(_scrub_clean_baseline_controller_action(repo_path, repo_context, action))
        brief.set_controller_action(action, merge_policy=brief.search_policy)
        delegation_policy, missing = _scrub_clean_baseline_file_scope_mapping(
            repo_path,
            repo_context,
            brief.delegation_policy,
        )
        brief.delegation_policy = delegation_policy
        dropped.extend(missing)
    return list(dict.fromkeys(dropped))


def _resolve_verification_test_command(
    test_command: Optional[str],
    verification_test_command: Any,
) -> Optional[str]:
    if verification_test_command is _INHERIT_VERIFICATION_TEST_COMMAND:
        return test_command
    if not verification_test_command:
        return None
    return str(verification_test_command)


def _humanize_test_inventory_framework(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "go_test": "go test",
        "cargo_test": "cargo test",
        "dotnet_test": "dotnet test",
    }
    return aliases.get(token, token.replace("_", " "))


def _quick_verification_inventory_context(
    payload: Optional[dict[str, Any]],
) -> tuple[str, str, str]:
    details = payload if isinstance(payload, dict) else {}
    test_command = str(details.get("test_inventory_test_command") or "").strip()
    framework = infer_test_inventory_framework(
        expected_test_ids=list(details.get("missing_expected_test_ids") or []),
        test_command=test_command,
        explicit_framework=str(details.get("test_inventory_framework") or "").strip(),
    )
    collection_command = str(
        details.get("test_inventory_collection_command") or ""
    ).strip() or derive_test_collection_command(
        test_command,
        framework=framework,
    )
    return framework, collection_command, test_command


logger = logging.getLogger("apex.orchestrator")
# Phase 2C 2.10: a child logger that the orchestrator owns. Sub-component
# loggers attach below this namespace (e.g. ``apex.orchestrator.solve``,
# ``apex.orchestrator.dynamic_transitions``) so user code can target the
# orchestrator without touching the root logger.
_ORCHESTRATOR_LOGGER_NAMESPACE = "apex.orchestrator"
# Mark APEX-installed handlers so re-init replaces only OUR handlers
# (idempotent), never user-installed ones.
_APEX_LOGGING_HANDLER_MARKER = "_apex_orchestrator_owned_handler"


@dataclass
class ApexResult:
    """Final orchestrator result."""

    success: bool
    patch: Optional[str] = None
    explanation: Optional[str] = None
    selected_rollout_id: Optional[int] = None
    selected_worktree_path: Optional[str] = None
    selected_changed_files: list[str] = field(default_factory=list)
    verification_summary: Optional[dict[str, Any]] = None
    selection_diagnostics: Optional[dict[str, Any]] = None
    selected_for_submission: bool = False
    internally_accepted: bool = False
    officially_accepted: Optional[bool] = None
    salvaged_for_external_scoring: bool = False
    # Phase 2C 2.2: structured outcome status. Derived from ``success`` +
    # the verifier signal at construction time when not explicitly set.
    status: Status = Status.FAILED
    salvaged: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)
    issue_plan: Optional[dict[str, Any]] = None
    total_rollouts: int = 0
    successful_rollouts: int = 0
    total_tokens: int = 0
    planner_tokens: int = 0
    total_duration_seconds: float = 0.0
    rollout_summaries: list[dict[str, Any]] = field(default_factory=list)
    external_scoring_candidates: list[dict[str, Any]] = field(default_factory=list)
    repo_stats: dict[str, Any] = field(default_factory=dict)
    difficulty_estimate: Optional[float] = None
    recommended_rollouts: Optional[int] = None
    baseline_summary: Optional[dict[str, Any]] = None
    orchestration_primitives: list[str] = field(default_factory=list)
    orchestration_transitions: list[dict[str, Any]] = field(default_factory=list)
    allocator_features: dict[str, Any] = field(default_factory=dict)
    unsolvable_reason: Optional[str] = None
    task_state_context: Optional[dict[str, Any]] = None
    task_state_graph: Optional[dict[str, Any]] = None
    search_summary: Optional[dict[str, Any]] = None
    multi_agent_summary: dict[str, Any] = field(default_factory=dict)
    repo_memory_summary: dict[str, Any] = field(default_factory=dict)
    # Phase 0.1: coarse, orchestrator-wide failure taxonomy. Populated
    # when the overall solve fails for an env / harness / agent reason;
    # remains ``None`` for successful solves. ``failure_class`` is a
    # denormalised mirror of ``failure_classification.failure_class`` for
    # cheap filtering at the report layer.
    failure_class: Optional["CoreFailureClass"] = None
    failure_classification: Optional["CoreClassificationResult"] = None
    # Phase 6.3: calibrated abstention. Optional so consumers built
    # before 6.3 keep working — populated in ``_build_final_result`` when
    # the orchestrator can compute a real score, otherwise None.
    confidence: Optional["ConfidenceBreakdown"] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "patch": self.patch,
            "explanation": self.explanation,
            "selected_rollout_id": self.selected_rollout_id,
            "selected_worktree_path": self.selected_worktree_path,
            "selected_changed_files": list(self.selected_changed_files),
            "verification_summary": self.verification_summary,
            "selection_diagnostics": self.selection_diagnostics,
            "selected_for_submission": self.selected_for_submission,
            "internally_accepted": self.internally_accepted,
            "officially_accepted": self.officially_accepted,
            "salvaged_for_external_scoring": self.salvaged_for_external_scoring,
            "status": self.status.value if isinstance(self.status, Status) else self.status,
            "salvaged": self.salvaged,
            "diagnostics": dict(self.diagnostics),
            "issue_plan": self.issue_plan,
            "total_rollouts": self.total_rollouts,
            "successful_rollouts": self.successful_rollouts,
            "total_tokens": self.total_tokens,
            "planner_tokens": self.planner_tokens,
            "total_duration_seconds": self.total_duration_seconds,
            "rollout_summaries": self.rollout_summaries,
            "external_scoring_candidates": self.external_scoring_candidates,
            "repo_stats": self.repo_stats,
            "difficulty_estimate": self.difficulty_estimate,
            "recommended_rollouts": self.recommended_rollouts,
            "baseline_summary": self.baseline_summary,
            "orchestration_primitives": list(self.orchestration_primitives),
            "orchestration_transitions": list(self.orchestration_transitions),
            "allocator_features": dict(self.allocator_features),
            "unsolvable_reason": self.unsolvable_reason,
            "task_state_context": self.task_state_context,
            "task_state_graph": self.task_state_graph,
            "search_summary": self.search_summary,
            "multi_agent_summary": dict(self.multi_agent_summary),
            "repo_memory_summary": dict(self.repo_memory_summary),
            "failure_class": (self.failure_class.value if self.failure_class is not None else None),
            "failure_classification": (
                self.failure_classification.to_dict()
                if self.failure_classification is not None
                else None
            ),
            "confidence": (self.confidence.to_dict() if self.confidence is not None else None),
        }

    def save(self, path: str | Path) -> None:
        from ..evaluation.checkpointing import atomic_write_json

        atomic_write_json(Path(path), self.to_dict())


class ApexOrchestrator:
    """Coordinate preprocessing, planning, rollout execution, and selection."""

    def __init__(self, config: ApexConfig):
        self.config = config
        self._artifact_safe_issue_plan = False
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Phase 2C 2.10: install a dedicated handler on the apex.orchestrator
        logger instead of mutating the root logger via ``basicConfig``.

        The legacy implementation called ``logging.basicConfig(...)`` per
        construction. ``basicConfig`` is a no-op when the root logger
        already has any handler — so the SECOND orchestrator constructed
        in a process silently kept the first orchestrator's log level
        and never honoured the new ``log_level``. This change:

        - never touches the root logger
        - replaces previously APEX-installed handlers (idempotent)
        - leaves user-installed handlers in place and emits a one-time
          notice so the operator knows their handler is winning
        """
        try:
            level = getattr(logging, str(self.config.log_level).upper())
        except AttributeError:
            level = logging.INFO

        target_logger = logging.getLogger(_ORCHESTRATOR_LOGGER_NAMESPACE)
        target_logger.setLevel(level)
        # Do NOT propagate to root if we have our own handler — otherwise
        # the operator sees every line twice.
        existing_apex_handlers = [
            handler
            for handler in list(target_logger.handlers)
            if getattr(handler, _APEX_LOGGING_HANDLER_MARKER, False)
        ]
        non_apex_handlers = [
            handler
            for handler in list(target_logger.handlers)
            if not getattr(handler, _APEX_LOGGING_HANDLER_MARKER, False)
        ]
        for handler in existing_apex_handlers:
            target_logger.removeHandler(handler)
        if non_apex_handlers:
            target_logger.info(
                "apex.orchestrator already has %s user-installed handler(s); "
                "leaving them in place and not adding the default APEX handler.",
                len(non_apex_handlers),
            )
            return
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        # Tag our handler so re-init can replace it without disturbing
        # user handlers.
        setattr(handler, _APEX_LOGGING_HANDLER_MARKER, True)
        target_logger.addHandler(handler)
        target_logger.propagate = False

    def solve(
        self,
        repo_path: str,
        issue_description: str,
        test_command: Optional[str] = None,
        benchmark_metadata: Optional[dict[str, Any]] = None,
        verification_test_command: Any = _INHERIT_VERIFICATION_TEST_COMMAND,
    ) -> ApexResult:
        """Phase 3.2: thin coordinator. Heavy lifting lives in helper
        methods plus :mod:`apex.orchestration.recovery` and
        :mod:`apex.orchestration.waves`.
        """
        start_time = time.time()
        # Phase A.1 (Decisive-Edge): when the benchmark layer has opted
        # into the V5 in-container agent surface
        # (``BenchmarkConfig.default_agent_mode == "in_container_v5"``)
        # AND we are in a benchmark context (``benchmark_metadata``
        # supplied), short-circuit the legacy MASAI/scaffolded pipeline
        # and route the solve through ``solve_in_container_agent``. The
        # V5 result is bridged into ``ApexResult`` so the rest of the
        # benchmark report code (rollout summaries, selection
        # diagnostics, repo stats) keeps working with by-construction
        # empty fields where they are not meaningful for V5.
        v5_routed = self._maybe_solve_via_in_container_v5(
            start_time=start_time,
            repo_path=repo_path,
            issue_description=issue_description,
            benchmark_metadata=benchmark_metadata,
            test_command=test_command,
        )
        if v5_routed is not None:
            return v5_routed
        prior_artifact_safe_issue_plan = self._artifact_safe_issue_plan
        # Decisive-Edge C.3: pre-load per-repo episodic patterns from
        # ~/.apex/repo_episodic/<repo_signature>/episodes.jsonl. The
        # contextvar is read by ``apex.agents.solver.BaseAgent.__init__``
        # so each agent's system prompt can include a "Repo conventions"
        # section. The token is reset in ``finally`` so a crashing solve
        # doesn't leak episodes into a subsequent in-process solve. A
        # disabled or empty store is treated as "no priors" and is fully
        # silent — never blocks the solve.
        repo_episodes_token, repo_episodes_loaded = self._load_repo_episodes_for_solve(repo_path)
        try:
            (
                repo_context,
                verifier,
                planner,
                strategy,
                issue_plan,
                task_state_graph,
                baseline_result,
                resolved_verification_test_command,
                orchestration_transitions,
            ) = self._prepare_run(
                repo_path=repo_path,
                issue_description=issue_description,
                test_command=test_command,
                benchmark_metadata=benchmark_metadata,
                verification_test_command=verification_test_command,
            )
            if strategy.unsolvable_reason:
                return self._build_unsolvable_result(
                    start_time=start_time,
                    strategy=strategy,
                    issue_plan=issue_plan,
                    repo_context=repo_context,
                    baseline_result=baseline_result,
                    task_state_graph=task_state_graph,
                    orchestration_transitions=orchestration_transitions,
                )
            engine = RolloutEngine(self.config, repo_path, repo_context)
            # WS3E: seed cross-solve (per-task) episodic priors into the engine
            # memory bus BEFORE rollouts run. Default OFF (returns [] unless the
            # flag is set); never blocks the solve. Complements the C.3 per-repo
            # priors (which seed from RepoMemoryStore lazily inside the engine).
            task_episode_priors = self._load_task_episodes_for_solve(
                repo_path=repo_path,
                issue_description=issue_description,
                benchmark_metadata=benchmark_metadata,
            )
            if task_episode_priors:
                try:
                    engine.memory_bus.seed_priors(task_episode_priors)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("WS3E: seeding task episode priors failed: %s", exc)
            try:
                result = self._run_pipeline(
                    start_time=start_time,
                    repo_path=repo_path,
                    repo_context=repo_context,
                    issue_description=issue_description,
                    issue_plan=issue_plan,
                    planner=planner,
                    strategy=strategy,
                    test_command=test_command,
                    resolved_verification_test_command=resolved_verification_test_command,
                    benchmark_metadata=benchmark_metadata,
                    verifier=verifier,
                    baseline_result=baseline_result,
                    task_state_graph=task_state_graph,
                    engine=engine,
                    orchestration_transitions=orchestration_transitions,
                )
                # C.3: post-run pattern capture. Always best-effort —
                # store failures must never fail the solve.
                self._persist_repo_episodes_after_solve(
                    repo_path=repo_path,
                    loaded_count=repo_episodes_loaded,
                )
                # WS3E: record this solve's terminal outcome under the task
                # signature for the next attempt. Best-effort, gated, silent.
                self._persist_task_episode_after_solve(
                    repo_path=repo_path,
                    issue_description=issue_description,
                    benchmark_metadata=benchmark_metadata,
                    result=result,
                )
                return result
            finally:
                self._safe_cleanup_engine(engine)
        finally:
            if repo_episodes_token is not None:
                reset_active_repo_episodes(repo_episodes_token)
            self._artifact_safe_issue_plan = prior_artifact_safe_issue_plan

    def _maybe_solve_via_in_container_v5(
        self,
        *,
        start_time: float,
        repo_path: str,
        issue_description: str,
        benchmark_metadata: Optional[dict[str, Any]],
        test_command: Optional[str] = None,
    ) -> Optional[ApexResult]:
        """Phase A.1 (Decisive-Edge): V5 in-container agent dispatch.

        Returns an :class:`ApexResult` when the benchmark configuration
        opts into the V5 surface (``BenchmarkConfig.default_agent_mode
        == "in_container_v5"``) AND the caller is in a benchmark context
        (``benchmark_metadata`` supplied). Returns ``None`` when the
        legacy scaffolded / cli_agent pipeline should be used.

        ``benchmark_metadata`` may carry an optional ``"docker_image"``
        key (the per-task image tag, e.g. the SWT-Bench
        ``aorwall/sweb.eval.x86_64.<instance>`` ref). When present the
        V5 agent is wrapped in a :class:`ContainerSupervisor` for true
        container isolation; otherwise the V5 agent runs in the V1 host
        bash shim against ``repo_path``.

        Phase A-α revalidate fix: when V5 emits a non-empty patch, this
        method now also materializes a worktree and applies the patch
        so that downstream benchmark evaluators (e.g. Commit0) can run
        their final pytest grading against a *patched* tree instead of
        the unmodified baseline. A synthetic ``RolloutResult`` is added
        to ``rollout_summaries`` and ``selected_rollout_id`` /
        ``selected_worktree_path`` are populated. Without this step the
        evaluator would silently grade on the baseline checkout and
        report the placeholder explanation as the "test output" — see
        the smoke run that motivated the fix.

        Result mapping:
          * V5 returns a patch that applies cleanly  → ``Status.SOLVED``
            (or ``Status.FAILED`` if downstream pytest fails — the
            evaluator decides), synthetic rollout with ``success=True``
            and a real worktree path.
          * V5 returns a patch that does NOT apply  → ``Status.FAILED``,
            synthetic rollout with ``success=False``, no worktree path
            so the evaluator falls back to baseline scoring.
          * V5 returns ``None`` because it gave up  → ``Status.ABSTAINED``,
            no synthetic rollout.
          * V5 hit ``max_turns`` / parse failure / llm failure with no
            patch → ``Status.FAILED``, no synthetic rollout.
        """
        try:
            requested = str(getattr(self.config.benchmark, "default_agent_mode", "") or "").strip()
        except AttributeError:
            return None
        if requested != "in_container_v5":
            return None
        if not benchmark_metadata:
            return None

        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.workspace_dir, exist_ok=True)
        self._artifact_safe_issue_plan = True

        from ..orchestrator_in_container_agent import (
            DEFAULT_MAX_TURNS,
            DEFAULT_TURN_TIMEOUT_SECONDS,
            solve_in_container_agent,
        )

        docker_image = ""
        if isinstance(benchmark_metadata, dict):
            docker_image = str(benchmark_metadata.get("docker_image") or "").strip()
        target_runtime_required = self._target_runtime_isolation_required(benchmark_metadata)

        llm_config = self.config.llm_configs[0] if self.config.llm_configs else None
        rollout_cfg = self.config.rollout
        base_max_turns = int(
            getattr(rollout_cfg, "max_iterations_per_rollout", 0) or DEFAULT_MAX_TURNS
        )
        if base_max_turns <= 0:
            base_max_turns = DEFAULT_MAX_TURNS
        # 1C: difficulty-scaled turn budget — raise for hard tasks, never below
        # the resolved base (no-cost-reduction). Best-effort; never crashes.
        max_turns = self._v5_scaled_max_turns(
            base=base_max_turns,
            issue_description=issue_description,
            benchmark_metadata=benchmark_metadata,
        )

        md = benchmark_metadata if isinstance(benchmark_metadata, dict) else {}
        expected_ids = [str(x) for x in (md.get("expected_test_ids") or []) if str(x).strip()]
        verify_command = self._v5_verify_command(test_command, md, expected_ids)
        # 1D: read the V5 activation knobs.
        reject_cap = int(getattr(rollout_cfg, "v5_patch_verifier_reject_cap", 3) or 3)
        recent_verbatim = int(getattr(rollout_cfg, "v5_recent_verbatim_turns", 3) or 3)
        stall_repeat = int(getattr(rollout_cfg, "v5_stall_repeat_threshold", 3) or 3)
        stall_cap = int(getattr(rollout_cfg, "v5_stall_terminate_cap", 5) or 5)

        workspace_dir = repo_path
        patch: Optional[str] = None
        v5_error: Optional[str] = None

        def _run_v5(supervisor: Any) -> Optional[str]:
            # 1A: build the in-loop verifier in THIS runtime so tests run with the
            # agent's deps/env (container when present). None -> legacy accept.
            verifier = None
            try:
                from ..modes import _build_v5_patch_verifier

                verifier = _build_v5_patch_verifier(
                    workspace_dir=Path(workspace_dir),
                    test_command=verify_command,
                    container_supervisor=supervisor,
                )
            except Exception:  # noqa: BLE001 - verifier build must not break dispatch
                logger.debug("V5 patch_verifier build failed; running without it", exc_info=True)
                verifier = None
            return solve_in_container_agent(
                llm_config=llm_config,
                workspace_dir=workspace_dir,
                problem_statement=issue_description,
                max_turns=max_turns,
                per_tool_timeout_seconds=DEFAULT_TURN_TIMEOUT_SECONDS,
                container_supervisor=supervisor,
                patch_verifier=verifier,
                patch_verifier_reject_cap=reject_cap,
                recent_verbatim_turns=recent_verbatim,
                stall_repeat_threshold=stall_repeat,
                stall_terminate_cap=stall_cap,
            )

        supervisor_cm = None
        if target_runtime_required and not docker_image:
            v5_error = "Target runtime isolation is required, but no docker image was supplied."
        elif docker_image:
            try:
                from ..core.container_supervisor import ContainerSupervisor

                supervisor_cm = ContainerSupervisor(
                    image=docker_image,
                    workspace_dir=Path(repo_path),
                )
            except Exception as exc:
                if target_runtime_required:
                    v5_error = (
                        "Target runtime isolation is required, but ContainerSupervisor "
                        f"init failed for image={docker_image}: {exc}"
                    )
                else:
                    logger.warning(
                        "ContainerSupervisor init failed for image=%s: %s; falling back to host shim.",
                        docker_image,
                        exc,
                    )
                supervisor_cm = None

        if v5_error is None:
            try:
                if supervisor_cm is not None:
                    with supervisor_cm as supervisor:
                        patch = _run_v5(supervisor)
                else:
                    patch = _run_v5(None)
            except Exception as exc:  # pragma: no cover — defensive
                logger.exception("V5 in-container agent dispatch failed: %s", exc)
                v5_error = f"{type(exc).__name__}: {exc}"

        # Phase A-α revalidate: when V5 produced a patch, we MUST
        # materialize it on a worktree so the benchmark evaluator can
        # run pytest against a patched tree. Without this, the smoke run
        # observed score_strict=0 with placeholder "V5 ... solved the
        # task." text being treated as test output.
        has_v5_patch = bool(patch and patch.strip()) and v5_error is None
        synthetic_rollout, patch_apply_diagnostics = (
            self._materialize_v5_synthetic_rollout(
                repo_path=repo_path,
                patch_text=patch if has_v5_patch else None,
            )
            if has_v5_patch
            else (None, None)
        )

        total_duration = time.time() - start_time
        diagnostics: dict[str, Any] = {
            "agent_mode": "in_container_v5",
            "docker_image": docker_image or None,
            "v5_routed_via_benchmark_default": True,
            "target_runtime_required": target_runtime_required,
        }
        if v5_error:
            diagnostics["v5_error"] = v5_error
        if patch_apply_diagnostics is not None:
            diagnostics["v5_patch_apply"] = patch_apply_diagnostics

        # Status taxonomy:
        #   * patch + clean apply  → SOLVED (evaluator decides final pass/fail
        #     from pytest, but APEX-side bookkeeping says we produced a
        #     candidate that survived the local apply gate)
        #   * patch + dirty apply  → FAILED (V5 hallucinated; evaluator will
        #     fall back to baseline scoring)
        #   * no patch + give_up   → ABSTAINED (deliberate non-attempt)
        #   * no patch + other     → FAILED
        # Without rich AgentRunSummary visibility here the give_up vs
        # max_turns distinction is best-effort: an empty patch with no
        # exception is treated as ABSTAINED on the grounds that any
        # mode that returned cleanly without a diff is most-charitably
        # interpreted as "I did not attempt to submit a patch".
        patch_applied_ok = synthetic_rollout is not None and synthetic_rollout.success
        if has_v5_patch and patch_applied_ok:
            status_value = Status.SOLVED
            success_for_apex = True
        elif has_v5_patch and not patch_applied_ok:
            status_value = Status.FAILED
            success_for_apex = False
        elif v5_error is not None:
            status_value = Status.ENV_SKIPPED if target_runtime_required else Status.FAILED
            success_for_apex = False
        else:
            # No patch, no exception: most-likely give_up / max_turns.
            status_value = Status.ABSTAINED
            success_for_apex = False

        rollout_summaries: list[dict[str, Any]] = []
        selected_rollout_id: Optional[int] = None
        selected_worktree_path: Optional[str] = None
        selected_changed_files: list[str] = []
        if synthetic_rollout is not None:
            payload = synthetic_rollout.to_dict()
            payload["trajectory"] = []
            rollout_summaries.append(payload)
            if synthetic_rollout.success:
                selected_rollout_id = synthetic_rollout.rollout_id
                selected_worktree_path = synthetic_rollout.worktree_path
                selected_changed_files = list(synthetic_rollout.changed_files)

        # Preserve the V5-emitted patch on ApexResult.patch even when
        # it failed to apply — the operator (and the
        # ``apex_result.json`` debug artifact) needs visibility into
        # what V5 tried. The downstream evaluator gates on
        # ``selected_worktree_path`` (not ``patch``), so a non-None
        # patch with a None worktree path correctly leads to a
        # baseline-scored task rather than a phantom-grade.
        result = ApexResult(
            success=success_for_apex,
            patch=patch if has_v5_patch else None,
            explanation=(
                "V5 in-container agent solved the task."
                if success_for_apex
                else (
                    v5_error
                    or (
                        "V5 in-container agent produced a patch that did not apply cleanly."
                        if has_v5_patch and not patch_applied_ok
                        else "V5 in-container agent produced no patch."
                    )
                )
            ),
            status=status_value,
            diagnostics=diagnostics,
            total_rollouts=1,
            successful_rollouts=1 if success_for_apex else 0,
            total_duration_seconds=total_duration,
            internally_accepted=success_for_apex,
            selected_for_submission=success_for_apex,
            rollout_summaries=rollout_summaries,
            external_scoring_candidates=self._external_scoring_candidates(
                [synthetic_rollout] if synthetic_rollout is not None else []
            ),
            selected_rollout_id=selected_rollout_id,
            selected_worktree_path=selected_worktree_path,
            selected_changed_files=selected_changed_files,
        )
        try:
            result.save(Path(self.config.output_dir) / "apex_result.json")
        except Exception:  # pragma: no cover — best-effort artifact emission
            pass
        return result

    def _target_runtime_isolation_required(
        self,
        benchmark_metadata: Optional[dict[str, Any]],
    ) -> bool:
        policy = getattr(self.config.benchmark, "runtime_policy", None)
        if isinstance(policy, dict) and bool(policy.get("target_evaluation_runtime_required")):
            return True
        if isinstance(benchmark_metadata, dict):
            metadata_policy = benchmark_metadata.get("runtime_policy")
            if isinstance(metadata_policy, dict) and bool(
                metadata_policy.get("target_evaluation_runtime_required")
            ):
                return True
        return False

    def _v5_scaled_max_turns(
        self,
        *,
        base: int,
        issue_description: str,
        benchmark_metadata: Optional[dict[str, Any]],
    ) -> int:
        """1C: scale the V5 turn budget by estimated difficulty.

        Hard tasks get more turns (memory now makes extra turns productive);
        easy tasks stay at ``base``. Per the no-cost-reduction rule the result is
        NEVER below ``base`` — difficulty only RAISES it, clamped to the config
        ceiling. Best-effort: any failure falls back to ``base`` (never crashes
        dispatch). The stall detector remains the safety valve against thrash.
        """
        floor = int(getattr(self.config.rollout, "v5_max_turns_floor", 8) or 8)
        ceiling = int(getattr(self.config.rollout, "v5_max_turns_ceiling", 60) or 60)
        result = max(int(base), floor)
        try:
            from ..planning.manager import IssuePlanner

            md = benchmark_metadata if isinstance(benchmark_metadata, dict) else {}
            expected_ids = list(md.get("expected_test_ids") or [])
            features = {
                "issue_length": len(issue_description or ""),
                "failing_test_count": len(expected_ids),
                "estimated_files_to_edit": int(md.get("estimated_files_to_edit") or 1),
                "repo_size_files": int(md.get("repo_size_files") or 0),
                "has_stack_trace": bool(md.get("has_stack_trace")),
                "has_tests": bool(expected_ids) or bool(md.get("has_tests")),
            }
            difficulty = float(IssuePlanner(self.config).estimate_difficulty(features))
            if difficulty >= 0.8:
                factor = 2.0
            elif difficulty >= 0.55:
                factor = 1.5
            else:
                factor = 1.0
            scaled = int(round(int(base) * factor))
            result = min(ceiling, max(int(base), floor, scaled))
        except Exception:  # noqa: BLE001 - difficulty scaling must never crash dispatch
            logger.debug("V5 difficulty scaling failed; using base max_turns", exc_info=True)
            result = min(ceiling, max(int(base), floor))
        return result

    def _v5_verify_command(
        self,
        test_command: Optional[str],
        benchmark_metadata: dict[str, Any],
        expected_ids: list[str],
    ) -> Optional[str]:
        """1A: resolve the in-loop verify command (fast subset preferred).

        Priority: an explicit threaded ``test_command`` / ``benchmark_metadata``
        command, else a fast expected-FILES pytest subset derived from the
        expected test IDs. Returns ``None`` (verifier becomes a no-op) when no
        runnable command can be derived, so non-Commit0 / no-test paths keep the
        legacy immediate-accept behavior.
        """
        explicit = str(
            test_command
            or benchmark_metadata.get("v5_verify_command")
            or benchmark_metadata.get("test_command")
            or ""
        ).strip()
        if explicit:
            return explicit
        files = sorted({eid.split("::", 1)[0] for eid in expected_ids if eid.split("::", 1)[0]})
        if not files:
            return None
        import shlex

        quoted = " ".join(shlex.quote(f) for f in files[:200])
        return f"python -m pytest -q -p no:cacheprovider {quoted}"

    def _task_state_graph_warm_path(self) -> Optional[Path]:
        """WS3D: signature-keyed path for the persisted TaskStateGraph, or None
        when no stable signature is available (warm-start becomes a no-op)."""
        sig = str(getattr(self, "_active_repo_signature", "") or "").strip()
        if not sig:
            return None
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", sig).strip("._-") or "task"
        return Path(self.config.output_dir) / "task_state_graphs" / f"{safe}.json"

    def _warm_start_task_state_graph(self, graph: "TaskStateGraph") -> None:
        if not getattr(self.config.planning, "warm_start_task_state_graph", True):
            return
        path = self._task_state_graph_warm_path()
        if path is None:
            return
        try:
            prior = TaskStateGraph.load(path)
            if prior is not None:
                grafted = graph.merge_warm_start(prior)
                if grafted:
                    logger.info(
                        "Warm-started task-state graph from %s (%d records grafted)",
                        path,
                        grafted,
                    )
        except Exception:  # noqa: BLE001 - warm-start must never break a solve
            logger.debug("Task-state-graph warm-start failed", exc_info=True)

    def _persist_task_state_graph_warm(self, graph: "TaskStateGraph") -> None:
        if not getattr(self.config.planning, "warm_start_task_state_graph", True):
            return
        path = self._task_state_graph_warm_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            graph.save(path)
        except Exception:  # noqa: BLE001 - persistence is best-effort
            logger.debug("Task-state-graph persist failed", exc_info=True)

    def _materialize_v5_synthetic_rollout(
        self,
        *,
        repo_path: str,
        patch_text: Optional[str],
    ) -> tuple[Optional[RolloutResult], Optional[dict[str, Any]]]:
        """Build a synthetic :class:`RolloutResult` from a V5 patch.

        Phase A-α revalidate fix. The Commit0 / SWE-EVO benchmark
        evaluators consume ``ApexResult.selected_worktree_path`` and run
        pytest against the worktree directly. The V5 dispatch path
        previously returned an ``ApexResult`` with that field set to
        ``None``, causing every grading run to score against the
        unmodified baseline checkout and report the V5 explanation
        ("V5 in-container agent solved the task.") as the test output.

        This helper:
          1. Allocates a per-task ``v5_rollout_0`` worktree under the
             configured ``workspace_dir`` using :class:`GitWorktreeManager`.
          2. Applies the V5 patch via ``apex.modes._apply_patch`` (with
             ``git apply --3way`` fallback).
          3. Computes the changed file list with ``git diff --name-only``.
          4. Returns a synthetic ``RolloutResult`` carrying the worktree
             path, the patch verbatim, ``is_synthetic=True``, and a
             diagnostics blob describing the apply path. The caller
             (:meth:`_maybe_solve_via_in_container_v5`) wires the
             rollout into ``ApexResult`` and sets ``selected_*`` fields.

        On any failure (worktree creation, dirty apply, etc.) returns a
        synthetic rollout with ``success=False`` and ``worktree_path=None``
        so the evaluator falls back to baseline scoring instead of
        crashing.
        """
        if not patch_text or not patch_text.strip():
            return None, None

        from ..modes import _apply_patch as _modes_apply_patch
        from ..rollout.engine import GitWorktreeManager

        diagnostics: dict[str, Any] = {"rollout_id": "v5_rollout_0"}
        manager = GitWorktreeManager(
            repo_path=repo_path,
            workspace_dir=self.config.workspace_dir,
            use_git_worktrees=True,
        )
        worktree_path: Optional[Path] = None
        try:
            worktree_path = manager.create_worktree(
                rollout_id=cast(Any, "v5_rollout_0"),
            )
        except Exception as exc:
            logger.warning("V5 synthetic rollout: worktree creation failed: %s", exc)
            diagnostics["worktree_error"] = f"{type(exc).__name__}: {exc}"
            return (
                RolloutResult(
                    rollout_id=0,
                    success=False,
                    patch=patch_text,
                    explanation="V5 synthetic rollout: worktree creation failed.",
                    is_synthetic=True,
                    agent_mode="in_container_v5",
                    selection_diagnostics=diagnostics,
                    failure_reason=str(exc),
                ),
                diagnostics,
            )

        try:
            apply_diag = _modes_apply_patch(worktree_path, patch_text)
        except Exception as exc:
            # Both direct and 3-way apply failed.
            logger.warning(
                "V5 synthetic rollout: patch did not apply at %s: %s",
                worktree_path,
                exc,
            )
            diagnostics["patch_apply_path"] = "failed"
            diagnostics["patch_apply_error"] = str(exc)
            # Tear the worktree down — keeping a baseline-only worktree
            # would mislead the evaluator into "scoring" the patch.
            try:
                manager.remove_worktree(worktree_path)
            except Exception:  # pragma: no cover — best-effort cleanup
                pass
            return (
                RolloutResult(
                    rollout_id=0,
                    success=False,
                    patch=patch_text,
                    explanation="V5 synthetic rollout: git apply failed in BOTH direct and --3way modes.",
                    is_synthetic=True,
                    agent_mode="in_container_v5",
                    selection_diagnostics=diagnostics,
                    failure_reason=str(exc),
                ),
                diagnostics,
            )

        # apply_diag is None for an empty patch (impossible here — we
        # gate on ``patch_text.strip()`` above) or a structured dict on
        # success. Surface the path / stderrs back to the caller for
        # benchmark observability.
        if isinstance(apply_diag, dict):
            diagnostics.update(apply_diag)

        # Compute changed files relative to HEAD so the evaluator and
        # report layer can show what V5 touched.
        changed_files: list[str] = []
        try:
            import subprocess as _subprocess

            diff_proc = _subprocess.run(
                ["git", "-C", str(worktree_path), "diff", "--name-only", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            if diff_proc.returncode == 0:
                changed_files = [
                    line.strip() for line in (diff_proc.stdout or "").splitlines() if line.strip()
                ]
        except Exception:  # pragma: no cover — diagnostics only
            changed_files = []

        synthetic = RolloutResult(
            rollout_id=0,
            success=True,
            patch=patch_text,
            explanation="V5 in-container agent patch materialized on synthetic rollout.",
            changed_files=changed_files,
            worktree_path=str(worktree_path),
            is_synthetic=True,
            agent_mode="in_container_v5",
            selection_diagnostics=diagnostics,
            internally_accepted=True,
            selected_for_submission=True,
        )
        return synthetic, diagnostics

    def _prepare_run(
        self,
        *,
        repo_path: str,
        issue_description: str,
        test_command: Optional[str],
        benchmark_metadata: Optional[dict[str, Any]],
        verification_test_command: Any,
    ) -> tuple[
        RepoContext,
        PatchVerifier,
        IssuePlanner,
        Any,
        IssuePlan,
        Optional[TaskStateGraph],
        Optional[BaselineResult],
        Optional[str],
        list[dict[str, Any]],
    ]:
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.workspace_dir, exist_ok=True)
        self._artifact_safe_issue_plan = bool(benchmark_metadata)

        logger.info("=" * 60)
        logger.info("APEX orchestrator starting")
        # SINGLE, AUTHORITATIVE declaration of the orchestration surface. Every
        # run — CLI or direct-API — passes through here, so this is the one place
        # guaranteed to record which path the orchestration took. An unresolved
        # value (a caller that bypassed agent-mode resolution) is guarded back to
        # the global default with a LOUD warning rather than silently defaulting,
        # so a run can never quietly take an unintended avenue.
        _resolved_agent_mode = str(
            getattr(self.config.benchmark, "default_agent_mode", "") or ""
        ).strip()
        if _resolved_agent_mode not in AGENT_MODE_CHOICES:
            logger.warning(
                "Orchestration agent surface was UNRESOLVED (%r) — a caller bypassed "
                "agent-mode resolution; falling back to %r.",
                _resolved_agent_mode,
                GLOBAL_DEFAULT_AGENT_MODE,
            )
            _resolved_agent_mode = GLOBAL_DEFAULT_AGENT_MODE
            try:
                self.config.benchmark.default_agent_mode = _resolved_agent_mode
            except AttributeError:
                pass
        logger.info("Orchestration agent surface: %s", _resolved_agent_mode)
        logger.info("=" * 60)

        repo_context = self._preprocess_repo(repo_path)
        repo_context.save(Path(self.config.output_dir) / "repo_context.json")

        verifier = self._build_verifier(repo_path)
        resolved_verification_test_command = _resolve_verification_test_command(
            test_command,
            verification_test_command,
        )
        if test_command and resolved_verification_test_command is None:
            logger.info(
                "Verifier-side test command disabled; preserving the agent-visible "
                "test command for planning and rollout context only."
            )
        baseline_result: Optional[BaselineResult] = None
        if resolved_verification_test_command:
            logger.info("Capturing baseline test results before planning and rollouts.")
            baseline_result = verifier.capture_baseline(
                repo_path,
                resolved_verification_test_command,
            )
            from ..evaluation.checkpointing import atomic_write_json

            atomic_write_json(
                Path(self.config.output_dir, "baseline_result.json"),
                baseline_result.to_dict(),
            )

        planner_cls = _resolve_patched("IssuePlanner", IssuePlanner)
        planner = planner_cls(self.config)
        strategy = planner.build_execution_strategy(
            issue_description,
            repo_context,
            baseline_result=baseline_result,
        )
        logger.info(
            "Execution strategy: rollouts=%s difficulty=%.2f primitives=%s unsolvable=%s",
            strategy.rollout_count,
            strategy.difficulty_estimate,
            ", ".join(primitive.value for primitive in strategy.primitives) or "none",
            bool(strategy.unsolvable_reason),
        )

        issue_plan = self._plan_issue(
            issue_description,
            repo_context,
            planner=planner,
            rollout_count=strategy.rollout_count,
            difficulty=strategy.difficulty_estimate,
            baseline_result=baseline_result,
        )
        issue_plan = planner.enrich_issue_plan(
            issue_plan,
            issue_description=issue_description,
            repo_context=repo_context,
            test_command=test_command,
            baseline_result=baseline_result,
            benchmark_metadata=benchmark_metadata,
        )
        issue_plan = planner.apply_execution_strategy(issue_plan, strategy)
        task_state_graph: Optional[TaskStateGraph] = None
        if self.config.planning.enable_task_state_graph:
            task_state_graph = TaskStateGraph.from_issue_plan(issue_plan)
            # WS3D: warm-start the fresh graph with durable signal from a prior
            # solve of the same task (additive; current obligations preserved).
            self._warm_start_task_state_graph(task_state_graph)
            issue_plan = self._refresh_task_state_context(issue_plan, task_state_graph)
            issue_plan = planner.apply_task_state_frontier(
                issue_plan,
                repo_context,
                stage_label="initial",
            )
            # Persist so the NEXT solve of this task can warm-start from it.
            self._persist_task_state_graph_warm(task_state_graph)
        self._save_issue_plan(issue_plan)
        return (
            repo_context,
            verifier,
            planner,
            strategy,
            issue_plan,
            task_state_graph,
            baseline_result,
            resolved_verification_test_command,
            [],
        )

    def _build_unsolvable_result(
        self,
        *,
        start_time: float,
        strategy: Any,
        issue_plan: IssuePlan,
        repo_context: RepoContext,
        baseline_result: Optional[BaselineResult],
        task_state_graph: Optional[TaskStateGraph],
        orchestration_transitions: list[dict[str, Any]],
    ) -> ApexResult:
        total_duration = time.time() - start_time
        result = ApexResult(
            success=False,
            explanation=strategy.unsolvable_reason,
            issue_plan=issue_plan.to_dict(),
            total_rollouts=0,
            successful_rollouts=0,
            total_tokens=issue_plan.planner_tokens,
            planner_tokens=issue_plan.planner_tokens,
            total_duration_seconds=total_duration,
            rollout_summaries=[],
            repo_stats=self._repo_stats(repo_context),
            difficulty_estimate=issue_plan.difficulty_estimate,
            recommended_rollouts=issue_plan.recommended_rollouts,
            baseline_summary=baseline_result.to_dict() if baseline_result else None,
            orchestration_primitives=list(issue_plan.orchestration_primitives),
            orchestration_transitions=list(orchestration_transitions),
            allocator_features=dict(issue_plan.allocator_features),
            unsolvable_reason=strategy.unsolvable_reason,
            task_state_context=dict(issue_plan.task_state_context),
            task_state_graph=task_state_graph.to_dict() if task_state_graph else None,
            multi_agent_summary={},
        )
        result.save(Path(self.config.output_dir) / "apex_result.json")
        return result

    def _run_pipeline(
        self,
        *,
        start_time: float,
        repo_path: str,
        repo_context: RepoContext,
        issue_description: str,
        issue_plan: IssuePlan,
        planner: IssuePlanner,
        strategy: Any,
        test_command: Optional[str],
        resolved_verification_test_command: Optional[str],
        benchmark_metadata: Optional[dict[str, Any]],
        verifier: PatchVerifier,
        baseline_result: Optional[BaselineResult],
        task_state_graph: Optional[TaskStateGraph],
        engine: RolloutEngine,
        orchestration_transitions: list[dict[str, Any]],
    ) -> ApexResult:
        wallclock_deadline = self._task_wallclock_deadline(start_time)
        (
            rollout_results,
            issue_plan,
            orchestration_transitions,
            search_summary,
        ) = self._execute_with_dynamic_transitions(
            repo_path=repo_path,
            repo_context=repo_context,
            issue_description=issue_description,
            issue_plan=issue_plan,
            planner=planner,
            initial_strategy=strategy,
            test_command=test_command,
            verification_test_command=resolved_verification_test_command,
            engine=engine,
            transitions=orchestration_transitions,
            baseline_result=baseline_result,
            task_state_graph=task_state_graph,
            benchmark_metadata=benchmark_metadata,
            wallclock_deadline=wallclock_deadline,
        )
        successful = [
            r
            for r in rollout_results
            if r.patch and (r.success or self._result_has_score_bearing_success(r))
        ]

        # Followup loops 1-2: near-miss + structural-recovery (both pre-best).
        successful = _recovery.run_near_miss_recovery(
            self,
            repo_path=repo_path,
            repo_context=repo_context,
            issue_description=issue_description,
            issue_plan=issue_plan,
            planner=planner,
            test_command=test_command,
            engine=engine,
            rollout_results=rollout_results,
            task_state_graph=task_state_graph,
            successful=successful,
            orchestration_transitions=orchestration_transitions,
            wallclock_deadline=wallclock_deadline,
        )
        successful = _recovery.run_structural_recovery(
            self,
            repo_path=repo_path,
            repo_context=repo_context,
            issue_description=issue_description,
            issue_plan=issue_plan,
            planner=planner,
            test_command=test_command,
            engine=engine,
            rollout_results=rollout_results,
            task_state_graph=task_state_graph,
            successful=successful,
            orchestration_transitions=orchestration_transitions,
            wallclock_deadline=wallclock_deadline,
        )

        if not successful and not self._external_scoring_candidates(rollout_results):
            return self._build_no_patch_result(
                start_time=start_time,
                rollout_results=rollout_results,
                issue_plan=issue_plan,
                repo_context=repo_context,
                baseline_result=baseline_result,
                task_state_graph=task_state_graph,
                search_summary=search_summary,
                orchestration_transitions=orchestration_transitions,
                engine=engine,
            )

        best_result = self._select_best_patch(
            repo_path=repo_path,
            rollout_results=rollout_results,
            issue_description=issue_description,
            test_command=resolved_verification_test_command,
            verifier=verifier,
            baseline_result=baseline_result,
            issue_plan=issue_plan,
        )
        # Followup loop 3: coverage-gap.
        best_result, successful, search_summary = _recovery.run_coverage_gap_recovery(
            self,
            repo_path=repo_path,
            repo_context=repo_context,
            issue_description=issue_description,
            issue_plan=issue_plan,
            planner=planner,
            test_command=test_command,
            verification_test_command=resolved_verification_test_command,
            verifier=verifier,
            baseline_result=baseline_result,
            engine=engine,
            rollout_results=rollout_results,
            task_state_graph=task_state_graph,
            successful=successful,
            best_result=best_result,
            orchestration_transitions=orchestration_transitions,
            search_summary=search_summary,
            wallclock_deadline=wallclock_deadline,
        )
        issue_plan = self._refresh_task_state_from_selected_result(
            issue_plan, task_state_graph, best_result
        )
        # Followup loop 4: residual selection.
        best_result, issue_plan, search_summary = _recovery.run_selection_followups(
            self,
            repo_path=repo_path,
            repo_context=repo_context,
            issue_description=issue_description,
            issue_plan=issue_plan,
            planner=planner,
            test_command=test_command,
            verification_test_command=resolved_verification_test_command,
            verifier=verifier,
            baseline_result=baseline_result,
            engine=engine,
            rollout_results=rollout_results,
            task_state_graph=task_state_graph,
            best_result=best_result,
            orchestration_transitions=orchestration_transitions,
            search_summary=search_summary,
            wallclock_deadline=wallclock_deadline,
        )
        return self._build_final_result(
            start_time=start_time,
            best_result=best_result,
            rollout_results=rollout_results,
            issue_plan=issue_plan,
            repo_context=repo_context,
            baseline_result=baseline_result,
            task_state_graph=task_state_graph,
            search_summary=search_summary,
            orchestration_transitions=orchestration_transitions,
            engine=engine,
            benchmark_metadata=benchmark_metadata,
        )

    def _build_no_patch_result(
        self,
        *,
        start_time: float,
        rollout_results: list[RolloutResult],
        issue_plan: IssuePlan,
        repo_context: RepoContext,
        baseline_result: Optional[BaselineResult],
        task_state_graph: Optional[TaskStateGraph],
        search_summary: Optional[dict[str, Any]],
        orchestration_transitions: list[dict[str, Any]],
        engine: RolloutEngine,
    ) -> ApexResult:
        total_duration = time.time() - start_time
        # Persist negative insights even on full failure.
        repo_memory_summary = engine.persist_repo_memory(accepted_rollout_ids=None)
        no_patch_status = (
            Status.ENV_SKIPPED if self._all_rollouts_env_failed(rollout_results) else Status.FAILED
        )
        result = ApexResult(
            success=False,
            status=no_patch_status,
            explanation="No rollout produced a valid patch.",
            issue_plan=issue_plan.to_dict(),
            total_rollouts=len(rollout_results),
            successful_rollouts=0,
            total_tokens=sum(item.total_tokens for item in rollout_results)
            + issue_plan.planner_tokens,
            planner_tokens=issue_plan.planner_tokens,
            total_duration_seconds=total_duration,
            rollout_summaries=self._serialize_rollout_summaries(rollout_results),
            external_scoring_candidates=self._external_scoring_candidates(rollout_results),
            repo_stats=self._repo_stats(repo_context),
            difficulty_estimate=issue_plan.difficulty_estimate,
            recommended_rollouts=issue_plan.recommended_rollouts,
            baseline_summary=baseline_result.to_dict() if baseline_result else None,
            orchestration_primitives=list(issue_plan.orchestration_primitives),
            orchestration_transitions=list(orchestration_transitions),
            allocator_features=dict(issue_plan.allocator_features),
            unsolvable_reason=issue_plan.unsolvable_reason,
            task_state_context=dict(issue_plan.task_state_context),
            task_state_graph=task_state_graph.to_dict() if task_state_graph else None,
            search_summary=search_summary,
            multi_agent_summary=self._summarize_multi_agent_usage(rollout_results),
            repo_memory_summary=repo_memory_summary,
        )
        result.save(Path(self.config.output_dir) / "apex_result.json")
        return result

    def _build_final_result(
        self,
        *,
        start_time: float,
        best_result: Optional[RolloutResult],
        rollout_results: list[RolloutResult],
        issue_plan: IssuePlan,
        repo_context: RepoContext,
        baseline_result: Optional[BaselineResult],
        task_state_graph: Optional[TaskStateGraph],
        search_summary: Optional[dict[str, Any]],
        orchestration_transitions: list[dict[str, Any]],
        engine: RolloutEngine,
        benchmark_metadata: Optional[dict[str, Any]] = None,
    ) -> ApexResult:
        total_duration = time.time() - start_time
        successful = [
            r
            for r in rollout_results
            if r.patch and (r.success or self._result_has_score_bearing_success(r))
        ]
        selected_patch_accepted = self._selected_result_is_accepted(best_result)
        accepted_rollout_ids: Optional[set[int]] = None
        if selected_patch_accepted and best_result is not None:
            accepted_rollout_ids = {int(best_result.rollout_id)}
        repo_memory_summary = engine.persist_repo_memory(
            accepted_rollout_ids=accepted_rollout_ids,
        )

        # Phase 2C 2.2: classify the outcome status BEFORE building the
        # result so the salvage gate can null out the patch consistently.
        # See SOLVED / ABSTAINED / ENV_SKIPPED / FAILED docs in Status.
        allow_salvage = bool(getattr(self.config.rollout, "allow_salvage", False))
        best_is_salvage_only = bool(
            best_result is not None
            and not selected_patch_accepted
            and getattr(best_result, "salvaged_for_external_scoring", False)
            and best_result.patch
        )
        if selected_patch_accepted:
            status = Status.SOLVED
            effective_patch = best_result.patch if best_result else None
            effective_success = True
            effective_salvaged_flag = False
        elif best_is_salvage_only and allow_salvage:
            status = Status.SOLVED
            assert best_result is not None
            effective_patch = best_result.patch
            effective_success = True
            effective_salvaged_flag = True
        elif best_is_salvage_only:
            status = Status.ABSTAINED
            effective_patch = None
            effective_success = False
            effective_salvaged_flag = False
        elif self._all_rollouts_env_failed(rollout_results):
            status = Status.ENV_SKIPPED
            effective_patch = None
            effective_success = False
            effective_salvaged_flag = False
        else:
            status = Status.FAILED
            effective_patch = None
            effective_success = False
            effective_salvaged_flag = False

        # Phase 6.3: build a draft result so the confidence scorer can
        # introspect verifier_summary / cluster_consensus / mutation_kill
        # signals without double-walking the rollouts. We THEN apply the
        # abstention override (if any) and rebuild the final ApexResult.
        # Decisive-Edge C.2: pass the benchmark id so the scorer can pick
        # the per-benchmark calibrated threshold (when available) ahead
        # of the global default.
        run_benchmark_id: Optional[str] = None
        if isinstance(benchmark_metadata, dict):
            for key in ("benchmark_name", "benchmark_id"):
                value = benchmark_metadata.get(key)
                if isinstance(value, str) and value.strip():
                    run_benchmark_id = value.strip()
                    break
        confidence = self._compute_confidence(
            best_result=best_result,
            rollout_results=rollout_results,
            status=status,
            effective_salvaged_flag=effective_salvaged_flag,
            benchmark_id=run_benchmark_id,
        )
        if (
            confidence is not None
            and confidence.recommended_action == "abstain"
            and status == Status.SOLVED
            and not allow_salvage
            and self._calibrated_signals_have_evidence_against(confidence)
        ):
            # Calibrated confidence overrides the acceptance gate: the
            # gate said SOLVED but the calibrated score is below the
            # configured threshold. Phase 2C's salvage-as-success is
            # already gone; Phase 6.3 extends the same principle to
            # honestly-verified-but-low-confidence patches.
            #
            # The override fires only when we have actual evidence
            # against the patch (a non-zero secondary signal that scored
            # low). When secondary signals are simply absent (mutation
            # / f2p / controller_policy all = 0 because the data wasn't
            # collected for this run), we DEFER to the strict acceptance
            # gate rather than abstain on missing-data alone.
            logger.info(
                "Phase 6.3 abstention override: status=SOLVED but "
                "confidence=%.4f < threshold=%.4f; abstaining.",
                confidence.overall,
                confidence.threshold_used,
            )
            status = Status.ABSTAINED
            effective_patch = None
            effective_success = False
            effective_salvaged_flag = False

        expose_selected_worktree = bool(
            best_result
            and effective_success
            and effective_patch
            and (selected_patch_accepted or effective_salvaged_flag)
            and (self.config.rollout.keep_worktrees or best_result.is_synthetic)
        )
        result = ApexResult(
            success=effective_success,
            patch=effective_patch,
            explanation=best_result.explanation if best_result else None,
            selected_rollout_id=best_result.rollout_id if best_result else None,
            selected_worktree_path=best_result.worktree_path if expose_selected_worktree else None,
            selected_changed_files=best_result.changed_files if best_result else [],
            verification_summary=best_result.verification if best_result else None,
            selection_diagnostics=best_result.selection_diagnostics if best_result else None,
            selected_for_submission=(
                bool(getattr(best_result, "selected_for_submission", False))
                if best_result is not None
                else False
            ),
            internally_accepted=selected_patch_accepted,
            officially_accepted=(
                getattr(best_result, "officially_accepted", None)
                if best_result is not None
                else None
            ),
            salvaged_for_external_scoring=(
                bool(getattr(best_result, "salvaged_for_external_scoring", False))
                if best_result is not None
                else False
            ),
            status=status,
            salvaged=effective_salvaged_flag,
            issue_plan=issue_plan.to_dict(),
            total_rollouts=len(rollout_results),
            successful_rollouts=len(successful),
            total_tokens=sum(item.total_tokens for item in rollout_results)
            + issue_plan.planner_tokens,
            planner_tokens=issue_plan.planner_tokens,
            total_duration_seconds=total_duration,
            rollout_summaries=self._serialize_rollout_summaries(rollout_results),
            external_scoring_candidates=self._external_scoring_candidates(rollout_results),
            repo_stats=self._repo_stats(repo_context),
            difficulty_estimate=issue_plan.difficulty_estimate,
            recommended_rollouts=issue_plan.recommended_rollouts,
            baseline_summary=baseline_result.to_dict() if baseline_result else None,
            orchestration_primitives=list(issue_plan.orchestration_primitives),
            orchestration_transitions=list(orchestration_transitions),
            allocator_features=dict(issue_plan.allocator_features),
            unsolvable_reason=issue_plan.unsolvable_reason,
            task_state_context=dict(issue_plan.task_state_context),
            task_state_graph=task_state_graph.to_dict() if task_state_graph else None,
            search_summary=search_summary,
            multi_agent_summary=self._summarize_multi_agent_usage(rollout_results),
            repo_memory_summary=repo_memory_summary,
            confidence=confidence,
        )
        # Decisive-Edge C.2: stamp the benchmark id onto diagnostics so the
        # per-benchmark Pareto-frontier sweep
        # (:func:`apex.orchestration.abstention.compute_pareto_frontier_per_benchmark`)
        # can group runs without each consumer having to re-derive it from
        # the benchmark_metadata dict.
        if run_benchmark_id:
            result.diagnostics["benchmark_id"] = run_benchmark_id
        result.save(Path(self.config.output_dir) / "apex_result.json")
        if self.config.save_trajectories:
            self._save_trajectories(rollout_results)
        logger.info(
            "APEX completed in %.1fs (success=%s, selected_rollout=%s)",
            total_duration,
            result.success,
            result.selected_rollout_id,
        )
        return result

    def _safe_cleanup_engine(self, engine: RolloutEngine) -> None:
        if self.config.rollout.keep_worktrees:
            return
        try:
            engine.cleanup()
        except Exception as cleanup_exc:  # noqa: BLE001
            # Phase 2C 5.5: classify before swallowing. Cleanup
            # exceptions are usually filesystem races (env), so
            # log + continue. Non-env failures are real bugs and bubble.
            classification = classify_failure(
                stderr=str(cleanup_exc),
                stdout="",
                returncode=1,
                context={"phase": "scoring"},
            )
            if classification.failure_class.is_environment:
                logger.warning(
                    "Engine cleanup raised env-class failure (%s, %s); "
                    "worktrees may need manual removal.",
                    classification.failure_class.value,
                    cleanup_exc,
                )
            else:
                logger.error(
                    "Engine cleanup raised non-env failure (%s, %s); "
                    "re-raising so the bug isn't silently swallowed.",
                    classification.failure_class.value,
                    cleanup_exc,
                )
                raise

    # ------------------------------------------------------------------
    # Decisive-Edge C.3 — per-repo episodic memory hooks
    # ------------------------------------------------------------------

    def _load_repo_episodes_for_solve(self, repo_path: str) -> tuple[Any, int]:
        """Load per-repo episodic patterns and install them into the agent contextvar.

        Returns ``(token, loaded_count)`` where ``token`` is the
        contextvars.Token to ``reset()`` once the solve completes (None
        when load failed and no token was installed). ``loaded_count``
        is the number of episodes the agents will see in their prompts.

        All failures (missing module, missing repo signature, IO errors)
        degrade silently to "no episodes" — the C.3 path must NEVER
        block the solve.
        """
        try:
            from ..persistence import (
                RepoEpisodicStore,
                repo_signature_for_path,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("C.3 repo episodic memory unavailable (import failed: %s)", exc)
            return None, 0
        try:
            sig = repo_signature_for_path(repo_path)
        except Exception as exc:
            logger.debug("C.3 repo signature unavailable for %s: %s", repo_path, exc)
            return None, 0
        try:
            store = RepoEpisodicStore()
            episodes = list(store.get_episodes(sig))
        except Exception as exc:
            logger.debug("C.3 repo episodic load failed for %s: %s", repo_path, exc)
            return None, 0
        # Stash the resolved signature so the post-run capture step
        # doesn't need to re-derive it (and so it's the same string we
        # loaded against).
        self._active_repo_signature = sig
        token = set_active_repo_episodes(episodes)
        if episodes:
            logger.info(
                "C.3: loaded %d repo episode(s) for repo signature %s",
                len(episodes),
                sig,
            )
        return token, len(episodes)

    def _persist_repo_episodes_after_solve(
        self,
        *,
        repo_path: str,
        loaded_count: int,
    ) -> None:
        """Mine the just-completed run for repo-level patterns and persist them.

        Best-effort: every failure path (missing apex_result.json,
        unreadable run_dir, store IO error) is swallowed with a debug
        log. The solve has already returned successfully by the time we
        get here.
        """
        try:
            from ..persistence import RepoEpisodicStore
        except Exception:  # pragma: no cover - defensive
            return
        sig = getattr(self, "_active_repo_signature", "") or ""
        if not sig:
            try:
                from ..persistence import repo_signature_for_path

                sig = repo_signature_for_path(repo_path)
            except Exception:
                return
        run_dir = Path(self.config.output_dir)
        try:
            store = RepoEpisodicStore()
            extracted = store.extract_patterns_from_run(
                run_dir,
                repo_signature=sig,
                repo_root=Path(repo_path) if repo_path else None,
            )
        except Exception as exc:
            logger.debug("C.3 pattern extraction failed for run_dir=%s: %s", run_dir, exc)
            return
        if not extracted:
            return
        persisted = 0
        for episode in extracted:
            try:
                store.add_episode(sig, episode)
                persisted += 1
            except Exception as exc:
                logger.debug(
                    "C.3 add_episode failed for pattern_type=%s: %s",
                    episode.pattern_type,
                    exc,
                )
        if persisted:
            logger.info(
                "C.3: persisted %d new repo episode(s) for signature %s (loaded %d at solve start)",
                persisted,
                sig,
                loaded_count,
            )

    # ------------------------------------------------------------------
    # WS3E — cross-solve (per-task) episodic memory
    # ------------------------------------------------------------------

    def _cross_solve_task_id(
        self,
        *,
        issue_description: str,
        benchmark_metadata: Optional[dict[str, Any]],
    ) -> str:
        """Stable task identity for the cross-solve episodic signature.

        Prefers an explicit benchmark id (``instance_id`` / ``task_id``); else
        falls back to a short hash of the issue description so re-attempts of the
        same issue match. Never raises."""
        if isinstance(benchmark_metadata, dict):
            for key in ("instance_id", "task_id"):
                value = benchmark_metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        digest = hashlib.sha256(str(issue_description or "").encode("utf-8")).hexdigest()
        return f"issue:{digest[:16]}"

    def _load_task_episodes_for_solve(
        self,
        *,
        repo_path: str,
        issue_description: str,
        benchmark_metadata: Optional[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """WS3E: load prior cross-solve episodes for this exact task and convert
        them into memory-bus prior-insight dicts. Gated by
        ``RolloutConfig.enable_cross_solve_episodic_memory`` (default OFF).

        Returns ``[]`` on any failure or when disabled — never blocks the solve.
        """
        if not getattr(self.config.rollout, "enable_cross_solve_episodic_memory", False):
            return []
        try:
            from ..capabilities.episodic_memory import learn_from_prior_run
            from ..persistence import repo_signature_for_path
            from ..persistence.episodic_store import EpisodicStore
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("WS3E cross-solve episodic memory unavailable: %s", exc)
            return []
        try:
            sig = getattr(self, "_active_repo_signature", "") or repo_signature_for_path(repo_path)
            task_id = self._cross_solve_task_id(
                issue_description=issue_description,
                benchmark_metadata=benchmark_metadata,
            )
            store = EpisodicStore()
            hypotheses = learn_from_prior_run(
                store,
                repo_signature=sig,
                task_id=task_id,
            )
        except Exception as exc:
            logger.debug("WS3E cross-solve episodic load failed for %s: %s", repo_path, exc)
            return []
        priors: list[dict[str, Any]] = []
        for hyp in hypotheses:
            description = str(getattr(hyp, "description", "") or "").strip()
            insight_type = str(getattr(hyp, "episode_type", "") or "").strip()
            if not description or not insight_type:
                continue
            priors.append(
                {
                    "insight_type": insight_type,
                    "description": description,
                    "confidence": float(getattr(hyp, "confidence", 0.0) or 0.0),
                    "file_paths": list(getattr(hyp, "file_paths", []) or []),
                    "symbols": list(getattr(hyp, "symbols", []) or []),
                    "test_ids": list(getattr(hyp, "test_ids", []) or []),
                    "negative": bool(getattr(hyp, "negative", False)),
                    "support_count": int(getattr(hyp, "support_count", 1) or 1),
                    "provenance": "cross_solve_episodic",
                }
            )
        if priors:
            logger.info(
                "WS3E: loaded %d cross-solve episode prior(s) for task %s",
                len(priors),
                self._cross_solve_task_id(
                    issue_description=issue_description,
                    benchmark_metadata=benchmark_metadata,
                ),
            )
        return priors

    def _persist_task_episode_after_solve(
        self,
        *,
        repo_path: str,
        issue_description: str,
        benchmark_metadata: Optional[dict[str, Any]],
        result: Any,
    ) -> None:
        """WS3E: broadcast this solve's terminal outcome under the task signature
        so the next attempt on the SAME task can learn from it. Best-effort;
        gated by the cross-solve flag and silent on every failure."""
        if not getattr(self.config.rollout, "enable_cross_solve_episodic_memory", False):
            return
        try:
            from ..capabilities.episodic_memory import record_outcome
            from ..persistence import repo_signature_for_path
            from ..persistence.episodic_store import EpisodicStore
        except Exception:  # pragma: no cover - defensive
            return
        try:
            sig = getattr(self, "_active_repo_signature", "") or repo_signature_for_path(repo_path)
            task_id = self._cross_solve_task_id(
                issue_description=issue_description,
                benchmark_metadata=benchmark_metadata,
            )
            status_obj = getattr(result, "status", None)
            status = str(getattr(status_obj, "value", status_obj) or "unknown")
            outcome: dict[str, Any] = {
                "status": status,
                "success": bool(getattr(result, "success", False)),
            }
            confidence = getattr(result, "confidence", None)
            score = getattr(confidence, "score", None)
            if isinstance(score, (int, float)):
                outcome["confidence"] = float(score)
            store = EpisodicStore()
            record_outcome(
                store,
                repo_signature=sig,
                task_id=task_id,
                rollout_id="solve",
                outcome=outcome,
            )
        except Exception as exc:
            logger.debug("WS3E cross-solve outcome persist failed for %s: %s", repo_path, exc)

    def _preprocess_repo(self, repo_path: str) -> RepoContext:
        analyzer = RepoAnalyzer(repo_path)
        return analyzer.analyze()

    def _plan_issue(
        self,
        issue_description: str,
        repo_context: RepoContext,
        planner: Optional[IssuePlanner] = None,
        rollout_count: Optional[int] = None,
        difficulty: Optional[float] = None,
        baseline_result: Optional[BaselineResult] = None,
    ) -> IssuePlan:
        planner_cls = _resolve_patched("IssuePlanner", IssuePlanner)
        planner = planner or planner_cls(self.config)
        return planner.plan_issue(
            issue_description,
            repo_context,
            rollout_count=rollout_count,
            difficulty=difficulty,
            baseline_result=baseline_result,
        )

    def _refresh_task_state_context(
        self,
        issue_plan: IssuePlan,
        task_state_graph: Optional[TaskStateGraph],
    ) -> IssuePlan:
        if task_state_graph is None or not self.config.planning.enable_task_state_graph:
            return issue_plan
        prior_context = (
            dict(issue_plan.task_state_context)
            if isinstance(issue_plan.task_state_context, dict)
            else {}
        )
        graph_context = task_state_graph.build_issue_plan_context(
            max_items=self.config.planning.max_task_state_context_items,
        )
        merged_context = self._merge_task_state_context_payloads(
            graph_context,
            prior_context,
        )
        issue_plan.task_state_context = merged_context
        self._persist_task_state_graph(task_state_graph)
        return issue_plan

    def _merge_task_state_context_payloads(
        self,
        base_context: Optional[dict[str, Any]],
        prior_context: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        merged_context = dict(base_context) if isinstance(base_context, dict) else {}
        preserved_context = dict(prior_context) if isinstance(prior_context, dict) else {}
        for key, value in preserved_context.items():
            if key == "summary":
                continue
            if key not in merged_context or not merged_context.get(key):
                merged_context[key] = copy.deepcopy(value)

        prior_summary = str(preserved_context.get("summary") or "").strip()
        base_summary = str(merged_context.get("summary") or "").strip()
        if prior_summary and base_summary and prior_summary not in base_summary:
            merged_context["summary"] = f"{base_summary} {prior_summary}".strip()
        elif prior_summary and not base_summary:
            merged_context["summary"] = prior_summary
        return merged_context

    def _refresh_task_state_from_selected_result(
        self,
        issue_plan: IssuePlan,
        task_state_graph: Optional[TaskStateGraph],
        selected_result: Optional[RolloutResult],
    ) -> IssuePlan:
        if (
            task_state_graph is None
            or selected_result is None
            or not self.config.planning.enable_task_state_graph
        ):
            return issue_plan
        task_state_graph.ingest_verification_feedback(issue_plan, selected_result)
        return self._refresh_task_state_context(issue_plan, task_state_graph)

    def _save_issue_plan(self, issue_plan: IssuePlan) -> None:
        issue_plan.save(
            Path(self.config.output_dir) / "issue_plan.json",
            artifact_safe=self._artifact_safe_issue_plan,
        )

    def _persist_task_state_graph(
        self,
        task_state_graph: Optional[TaskStateGraph],
    ) -> None:
        if task_state_graph is None or not self.config.planning.enable_task_state_graph:
            return
        output_path = Path(self.config.output_dir) / "task_state_graph.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        task_state_graph.save(output_path)

    def _task_wallclock_deadline(self, start_time: float) -> Optional[float]:
        try:
            budget_seconds = float(
                getattr(self.config.rollout, "task_wallclock_budget_seconds", 0) or 0
            )
        except (TypeError, ValueError):
            return None
        if budget_seconds <= 0:
            return None
        return float(start_time) + budget_seconds

    def _execute_rollouts(
        self,
        repo_path: str,
        repo_context: RepoContext,
        issue_description: str,
        issue_plan: IssuePlan,
        test_command: Optional[str],
        engine: Optional[RolloutEngine] = None,
        rollout_id_offset: int = 0,
        stop_on_result: Optional[Callable[[RolloutResult], bool]] = None,
        workspace_seed: Optional[WorkspaceSeed] = None,
        workspace_seeds: Optional[list[Optional[WorkspaceSeed]]] = None,
        wallclock_deadline: Optional[float] = None,
        advisory_seed_reproduction_artifact: Optional[ReproductionArtifact] = None,
        advisory_seed_localization_artifact: Optional[LocalizationArtifact] = None,
        advisory_seed_source_rollout_id: Optional[int] = None,
    ) -> list[RolloutResult]:
        engine = engine or RolloutEngine(self.config, repo_path, repo_context)
        return engine.execute_rollouts(
            issue_description=issue_description,
            issue_plan=issue_plan,
            test_command=test_command,
            rollout_id_offset=rollout_id_offset,
            stop_on_result=stop_on_result,
            workspace_seed=workspace_seed,
            workspace_seeds=workspace_seeds,
            wallclock_deadline=wallclock_deadline,
            advisory_seed_reproduction_artifact=advisory_seed_reproduction_artifact,
            advisory_seed_localization_artifact=advisory_seed_localization_artifact,
            advisory_seed_source_rollout_id=advisory_seed_source_rollout_id,
        )

    @staticmethod
    def _next_rollout_id_after(
        rollout_results: list[RolloutResult],
        *,
        fallback: int = 0,
    ) -> int:
        """Return the next unused rollout id after a possibly sparse result set."""

        try:
            next_id = max(0, int(fallback))
        except (TypeError, ValueError):
            next_id = 0
        for result in list(rollout_results or []):
            rollout_id = getattr(result, "rollout_id", None)
            if isinstance(rollout_id, bool):
                continue
            try:
                parsed = int(rollout_id)
            except (TypeError, ValueError):
                continue
            next_id = max(next_id, parsed + 1)
        return next_id

    def _planned_rollout_budget(self, issue_plan: IssuePlan) -> int:
        """Return the rollout allocation selected for this plan.

        Decomposition can intentionally collapse many rollout variants into a
        smaller set of seed target briefs. The execution budget must follow the
        allocator's chosen rollout count, with the current brief count as a
        floor, so search can sample multiple approaches per target.
        """

        candidates: list[int] = []

        def add_candidate(value: Any) -> None:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return
            if parsed > 0:
                candidates.append(parsed)

        add_candidate(getattr(issue_plan, "recommended_rollouts", None))
        planner_metadata = (
            issue_plan.planner_metadata if isinstance(issue_plan.planner_metadata, dict) else {}
        )
        for key in (
            "execution_strategy_rollout_count",
            "requested_rollouts",
            "portfolio_rollout_floor",
        ):
            add_candidate(planner_metadata.get(key))
        add_candidate(len(issue_plan.rollout_briefs or []))
        if not candidates:
            add_candidate(getattr(self.config.rollout, "num_rollouts", 0))
        return max(1, max(candidates or [1]))

    def _clone_issue_plan_with_rollout_briefs(
        self,
        issue_plan: IssuePlan,
        rollout_briefs: list[Any],
    ) -> IssuePlan:
        cloned = IssuePlan.from_dict(issue_plan.to_dict())
        cloned.rollout_briefs = [
            type(brief).from_dict(brief.to_dict()) if hasattr(type(brief), "from_dict") else brief
            for brief in rollout_briefs
        ]
        return cloned

    def _speculative_first_attempt_enabled(self, issue_plan: IssuePlan) -> bool:
        """WS3B: True when the planner flagged this task for a single-seed
        speculative first attempt (computed from estimated difficulty and gated
        by ``RolloutConfig.enable_speculative_first_attempt``)."""
        if not getattr(self.config.rollout, "enable_speculative_first_attempt", False):
            return False
        metadata = getattr(issue_plan, "planner_metadata", None)
        if not isinstance(metadata, dict):
            return False
        return bool(metadata.get("speculative_first_attempt"))

    def _run_speculative_first_attempt(
        self,
        *,
        repo_path: str,
        repo_context: RepoContext,
        issue_description: str,
        issue_plan: IssuePlan,
        test_command: Optional[str],
        engine: Optional[RolloutEngine],
        rollout_id_offset: int,
        current_workspace_seed: Optional[WorkspaceSeed],
        wallclock_deadline: Optional[float],
    ) -> tuple[Optional[list[RolloutResult]], int, list[RolloutResult]]:
        """WS3B: dispatch the single highest-priority seed rollout first.

        Returns ``(results, next_rollout_id, seed_results)`` where ``results``
        is the combined slate. When the speculative seed yields the same
        terminal search-control signal used by later strategy waves, we stop
        early and return ONLY the seed result (the parallel slate was never
        dispatched). When it does not, we return
        ``None`` for ``results`` so the caller falls through to the normal full
        dispatch — this never reduces coverage. ``seed_results`` is ALWAYS the
        raw list the seed rollout produced (even on the fall-through), so the
        caller can harvest its already-computed reproduction/localization
        discovery as an advisory warm-start seed for the full slate instead of
        discarding it (SPEED LEVER: cross-rollout discovery REUSE).
        """
        briefs = list(issue_plan.rollout_briefs or [])
        seed_plan = self._clone_issue_plan_with_rollout_briefs(issue_plan, briefs[:1])
        next_rollout_id = rollout_id_offset
        seed_results = self._execute_rollouts(
            repo_path=repo_path,
            repo_context=repo_context,
            issue_description=issue_description,
            issue_plan=seed_plan,
            test_command=test_command,
            engine=engine,
            rollout_id_offset=next_rollout_id,
            stop_on_result=self._build_authoritative_completion_stop_on_result(),
            workspace_seed=current_workspace_seed,
            workspace_seeds=None,
            wallclock_deadline=wallclock_deadline,
        )
        next_rollout_id += len(seed_results)
        if any(
            self._rollout_has_authoritative_completion_signal(result)
            or self._rollout_has_local_full_suite_completion_signal(result)
            or self._rollout_has_authoritative_scoring_stop_signal(result)
            for result in seed_results
        ):
            logger.info(
                "WS3B speculative first attempt accepted after %s seed rollout(s); "
                "skipped %s remaining parallel rollout(s)",
                len(seed_results),
                max(0, len(briefs) - len(seed_results)),
            )
            return seed_results, next_rollout_id, seed_results
        return None, rollout_id_offset, seed_results

    @staticmethod
    def _is_harvestable_discovery(result: RolloutResult) -> bool:
        """SPEED LEVER (cross-rollout discovery REUSE) high-confidence gate.

        True only for a result whose reproduction ACTUALLY RAN and observed
        something (a non-empty ``observed_output`` AND a ``command`` or
        ``script_path``) AND whose localization names at least one concrete file
        — never an empty/aspirational/stub artifact. Discovery is valid even
        when the solve itself failed, so we do NOT require ``result.success``;
        but we DO skip results whose failure was classified as an env/infra
        failure, since their artifacts are untrustworthy. Strict by design: a
        confidently-wrong reproduction would otherwise be inherited by every
        sibling, so we keep the bar high and rely on the advisory framing for
        the rest. Fully fail-open: any exception => not harvestable.
        """

        try:
            failure_class = getattr(result, "failure_class", None)
            if failure_class is not None and getattr(failure_class, "is_environment", False):
                return False
            repro = coerce_reproduction_artifact(getattr(result, "reproduction_artifact", None))
            loc = coerce_localization_artifact(getattr(result, "localization_artifact", None))
            if repro is None or loc is None:
                return False
            observed = (repro.observed_output or "").strip()
            ran = bool((repro.command or "").strip()) or bool((repro.script_path or "").strip())
            if not (observed and ran):
                return False
            return bool(loc.files)
        except Exception:  # noqa: BLE001 - fail-open
            return False

    def _harvest_advisory_discovery(
        self,
        results: Optional[list[RolloutResult]],
    ) -> tuple[Optional[ReproductionArtifact], Optional[LocalizationArtifact], Optional[int]]:
        """Scan ``results`` for the FIRST high-confidence discovery and return
        its reproduction + localization artifacts (coerced) plus the source
        rollout id. Returns ``(None, None, None)`` when nothing qualifies or on
        any error (fail-open). O(N) over already-materialized results — adds no
        agentic call and never blocks dispatch.
        """

        try:
            for result in list(results or []):
                if not self._is_harvestable_discovery(result):
                    continue
                repro = coerce_reproduction_artifact(getattr(result, "reproduction_artifact", None))
                loc = coerce_localization_artifact(getattr(result, "localization_artifact", None))
                if repro is None or loc is None:
                    continue
                source_id = getattr(result, "rollout_id", None)
                return (
                    repro,
                    loc,
                    int(source_id) if isinstance(source_id, int) else None,
                )
        except Exception as harvest_exc:  # noqa: BLE001 - fail-open
            logger.debug("Advisory discovery harvest failed: %s", harvest_exc)
        return None, None, None

    def _advisory_discovery_reuse_allowed(self, issue_plan: IssuePlan) -> bool:
        """Giants size-gate for the discovery-reuse lever (Layer-A, measured).

        Only reuse discovery when the suite is NOT a giant: giants
        (size_factor at the configured max) keep today's fully-independent
        per-rollout localization, which is where independent localization is
        most load-bearing. Mirrors the giants-keep-full-behavior guard in
        ``_size_aware_followup_round_cap``. Fail-open: any error => allowed
        (the engine-side gate + high-confidence predicate still apply).
        """

        rollout_cfg = getattr(self.config, "rollout", None)
        if not bool(getattr(rollout_cfg, "enable_cross_rollout_discovery_reuse", True)):
            return False
        try:
            max_size_factor = int(getattr(rollout_cfg, "rollout_budget_max_size_factor", 6) or 6)
            tests_per_unit = int(
                getattr(rollout_cfg, "rollout_budget_tests_per_unit", 2000) or 2000
            )
            size_factor = _rollout_budget_size_factor(
                _issue_plan_expected_test_count(issue_plan),
                tests_per_unit=tests_per_unit,
                max_size_factor=max_size_factor,
            )
            return size_factor < max(2, max_size_factor)
        except Exception as gate_exc:  # noqa: BLE001 - fail-open
            logger.debug("Advisory discovery reuse size-gate failed: %s", gate_exc)
            return True

    def _frontier_search_enabled(
        self,
        issue_plan: IssuePlan,
        task_state_graph: Optional[TaskStateGraph],
    ) -> bool:
        if self.config.search.mode == SearchMode.OFF:
            return False
        for brief in list(issue_plan.rollout_briefs or []):
            policy = brief.search_policy if isinstance(brief.search_policy, dict) else {}
            if (
                policy.get("verifier_validity_repair") is True
                or str(policy.get("origin") or "").strip() == "verifier_validity_repair"
                or policy.get("skip_frontier_search") is True
                or policy.get("direct_workspace_seed_repair") is True
            ):
                return False
        if task_state_graph is None:
            logger.warning(
                "Explicit search is enabled, but no task-state graph is available; "
                "falling back to rollout execution without frontier search."
            )
            return False
        return bool(issue_plan.rollout_briefs)

    def _merge_search_summary(
        self,
        current: Optional[dict[str, Any]],
        new_summary: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if not new_summary:
            return current

        merged = dict(current or {})
        prior_runs = [dict(run) for run in list(merged.get("runs") or []) if isinstance(run, dict)]
        new_runs = [
            dict(run) for run in list(new_summary.get("runs") or []) if isinstance(run, dict)
        ]
        if not new_runs:
            new_runs = [dict(new_summary)]
        runs = prior_runs + new_runs

        best_reward = None
        best_rollout_id = None
        for run in runs:
            reward = run.get("best_reward")
            if not isinstance(reward, (int, float)):
                continue
            if best_reward is None or float(reward) > best_reward:
                best_reward = float(reward)
                best_rollout_id = run.get("best_rollout_id")

        merged.update(
            {
                "enabled": True,
                "mode": str(
                    new_summary.get("mode") or merged.get("mode") or self.config.search.mode.value
                ),
                "runs": runs,
                "run_count": len(runs),
                "total_expansions": sum(int(run.get("total_expansions") or 0) for run in runs),
                "state_count": sum(int(run.get("state_count") or 0) for run in runs),
                "branch_state_count": sum(int(run.get("branch_state_count") or 0) for run in runs),
                "transition_count": sum(int(run.get("transition_count") or 0) for run in runs),
                "checkpoint_reuse_count": sum(
                    int(run.get("checkpoint_reuse_count") or 0) for run in runs
                ),
                "max_depth_reached": max(
                    (int(run.get("max_depth_reached") or 0) for run in runs),
                    default=0,
                ),
                "best_reward": round(best_reward, 4) if best_reward is not None else None,
                "best_rollout_id": best_rollout_id,
            }
        )
        return merged

    def _summarize_multi_agent_usage(
        self,
        rollout_results: list[RolloutResult],
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "subagent_enabled_rollouts": 0,
            "delegate_enabled_rollouts": 0,
            "investigation_calls": 0,
            "debugger_calls": 0,
            "delegated_subtasks": 0,
            "successful_delegated_subtasks": 0,
            "delegated_tokens": 0,
            "boundary_pressure_count": 0,
            "boundary_requested_files": [],
            "boundary_interface_symbols": [],
            "boundary_followups": [],
        }
        for rollout in rollout_results:
            payload = (
                rollout.multi_agent_summary if isinstance(rollout.multi_agent_summary, dict) else {}
            )
            if bool(payload.get("subagent_tools_enabled")):
                summary["subagent_enabled_rollouts"] += 1
            if bool(payload.get("delegate_subtasks_enabled")):
                summary["delegate_enabled_rollouts"] += 1
            for key in (
                "investigation_calls",
                "debugger_calls",
                "delegated_subtasks",
                "successful_delegated_subtasks",
                "delegated_tokens",
                "boundary_pressure_count",
            ):
                summary[key] += int(payload.get(key) or 0)
            for key in (
                "boundary_requested_files",
                "boundary_interface_symbols",
                "boundary_followups",
            ):
                summary[key] = list(
                    dict.fromkeys(
                        list(summary.get(key) or [])
                        + [
                            str(item).strip()
                            for item in list(payload.get(key) or [])
                            if str(item).strip()
                        ]
                    )
                )[:8]
        return summary

    # A4: deepening waves relax the seed-diversity overlap gate so that
    # complementary near-identical partials (e.g. two rollouts that each filled
    # a different stub in the same files) can BOTH be retained as warm-start
    # seeds for the next deepening wave, rather than one being dropped for
    # overlap. Kept strictly below 1.0 so byte-identical seeds are still pruned.
    _DEEPENING_SEED_OVERLAP_THRESHOLD = 0.92

    def _select_rollout_seeds(
        self,
        planner: IssuePlanner,
        rollout_results: list[RolloutResult],
        *,
        k: int = 3,
        preferred_result: Optional[RolloutResult] = None,
        deepening: bool = False,
    ) -> list[WorkspaceSeed]:
        """Pick up to `k` diverse workspace seeds from prior rollouts.

        Replaces the prior single-seed picker which caused all wave/follow-up
        rollouts to branch from the same checkpoint, producing identical
        patches. We rank by `planner.score_rollout_progress`, then drop any
        candidate whose `changed_files` Jaccard overlap with any already-picked
        seed exceeds ``OrchestrationConfig.seed_diversity_overlap_threshold``.

        A4: when ``deepening`` is True (later progressive waves), the overlap
        gate is relaxed to ``_DEEPENING_SEED_OVERLAP_THRESHOLD`` so complementary
        deep partials are retained for warm-start instead of pruned.
        """
        target_k = max(1, int(k))
        overlap_threshold = float(self.config.orchestration.seed_diversity_overlap_threshold)
        if deepening:
            overlap_threshold = max(
                overlap_threshold,
                float(self._DEEPENING_SEED_OVERLAP_THRESHOLD),
            )
        ordered_candidates: list[RolloutResult] = []
        seen_rollout_ids: set[int] = set()
        if preferred_result is not None:
            ordered_candidates.append(preferred_result)
            seen_rollout_ids.add(preferred_result.rollout_id)
        ranked = sorted(
            rollout_results,
            key=planner.score_rollout_progress,
            reverse=True,
        )
        for result in ranked:
            if result.rollout_id in seen_rollout_ids:
                continue
            ordered_candidates.append(result)
            seen_rollout_ids.add(result.rollout_id)

        picked_seeds: list[WorkspaceSeed] = []
        picked_change_sets: list[frozenset[str]] = []
        considered: list[dict[str, Any]] = []
        for candidate in ordered_candidates:
            seed = build_workspace_seed_from_rollout_result(candidate)
            if seed is None:
                continue
            change_set = frozenset(
                str(path).strip() for path in (candidate.changed_files or []) if str(path).strip()
            )
            is_preferred = (
                preferred_result is not None and candidate.rollout_id == preferred_result.rollout_id
            )
            max_overlap = 0.0
            for prior_change_set in picked_change_sets:
                overlap = _jaccard_similarity(change_set, prior_change_set)
                if overlap > max_overlap:
                    max_overlap = overlap
            # Preferred result is always admitted: callers explicitly endorse
            # it (e.g. residual-followup needs to branch from the current
            # best). Diversity gating only applies to the planner-ranked tail.
            rejected_for_diversity = (
                not is_preferred and picked_change_sets and max_overlap > overlap_threshold
            )
            considered.append(
                {
                    "rollout_id": int(candidate.rollout_id),
                    "progress_score": float(planner.score_rollout_progress(candidate)),
                    "max_overlap": round(float(max_overlap), 4),
                    "is_preferred": bool(is_preferred),
                    "rejected_for_diversity": bool(rejected_for_diversity),
                    "changed_file_count": len(change_set),
                }
            )
            if rejected_for_diversity:
                continue
            picked_seeds.append(seed)
            picked_change_sets.append(change_set)
            if len(picked_seeds) >= target_k:
                break

        # Trace decision so analysts can audit when diversity rejected
        # otherwise high-scoring candidates. Failures here must never block
        # the orchestrator — append_controller_decision wraps its own IO.
        try:
            append_decision = _resolve_patched(
                "append_controller_decision", append_controller_decision
            )
            append_decision(
                self.config,
                stage="orchestration",
                decision_type="rollout_seed_diversity",
                chosen_option=",".join(str(seed.source_rollout_id) for seed in picked_seeds)
                or "none",
                feature_view={
                    "k": float(target_k),
                    "candidate_count": float(len(ordered_candidates)),
                    "picked_count": float(len(picked_seeds)),
                    "diversity_threshold": float(overlap_threshold),
                    "deepening": 1.0 if deepening else 0.0,
                },
                metadata={
                    "preferred_rollout_id": (
                        int(preferred_result.rollout_id) if preferred_result is not None else None
                    ),
                    "considered": considered,
                },
            )
        except Exception as trace_exc:  # noqa: BLE001
            logger.debug("rollout_seed_diversity trace failed: %s", trace_exc)

        return picked_seeds

    def _select_rollout_seed(
        self,
        planner: IssuePlanner,
        rollout_results: list[RolloutResult],
        *,
        preferred_result: Optional[RolloutResult] = None,
    ) -> Optional[WorkspaceSeed]:
        # TODO: callers should migrate to _select_rollout_seeds once the
        # engine surface accepts a per-request workspace_seed batch.
        seeds = self._select_rollout_seeds(
            planner,
            rollout_results,
            k=1,
            preferred_result=preferred_result,
        )
        return seeds[0] if seeds else None

    def _execute_progressive_rollout_plan(
        self,
        *,
        repo_path: str,
        repo_context: RepoContext,
        issue_description: str,
        issue_plan: IssuePlan,
        planner: IssuePlanner,
        test_command: Optional[str],
        engine: Optional[RolloutEngine] = None,
        rollout_id_offset: int = 0,
        task_state_graph: Optional[TaskStateGraph] = None,
        initial_workspace_seed: Optional[WorkspaceSeed] = None,
        wallclock_deadline: Optional[float] = None,
    ) -> tuple[
        list[RolloutResult],
        IssuePlan,
        int,
        list[dict[str, Any]],
        Optional[dict[str, Any]],
    ]:
        engine = engine or RolloutEngine(self.config, repo_path, repo_context)
        transitions: list[dict[str, Any]] = []
        next_rollout_id = rollout_id_offset
        search_summary: Optional[dict[str, Any]] = None
        current_workspace_seed = initial_workspace_seed
        current_workspace_seeds: Optional[list[Optional[WorkspaceSeed]]] = None

        if self._frontier_search_enabled(issue_plan, task_state_graph):
            controller_cls = _resolve_patched("FrontierSearchController", FrontierSearchController)
            controller = controller_cls(
                self.config,
                planner,
                repo_context,
                issue_description=issue_description,
                test_command=test_command,
                output_dir=self.config.output_dir,
            )
            search_kwargs: dict[str, Any] = {
                "issue_plan": issue_plan,
                "task_state_graph": task_state_graph,
                "engine": engine,
                "rollout_budget": self._planned_rollout_budget(issue_plan),
                "rollout_id_offset": next_rollout_id,
                "wallclock_deadline": wallclock_deadline,
            }
            if current_workspace_seed is not None:
                search_kwargs["root_workspace_seed"] = current_workspace_seed
            search_result = controller.run(
                **search_kwargs,
            )
            search_issue_plan = search_result.issue_plan or issue_plan
            self._persist_task_state_graph(task_state_graph)
            self._save_issue_plan(search_issue_plan)
            return (
                search_result.rollout_results,
                search_issue_plan,
                self._next_rollout_id_after(
                    search_result.rollout_results,
                    fallback=search_result.next_rollout_id,
                ),
                search_result.transitions,
                search_result.summary,
            )

        if not planner.should_use_progressive_rollout_allocation(issue_plan):
            # WS3B: speculative first attempt. For tasks the planner flagged as
            # easy, dispatch ONE seed rollout before fanning out the full slate;
            # accept immediately on an authoritative completion signal and skip
            # the remaining (now-redundant) parallel rollouts. Falls through to
            # the normal full dispatch when the seed does not authoritatively
            # pass, so it never reduces coverage on a task that needs it.
            advisory_seed_repro: Optional[ReproductionArtifact] = None
            advisory_seed_loc: Optional[LocalizationArtifact] = None
            advisory_seed_source_id: Optional[int] = None
            if (
                self._speculative_first_attempt_enabled(issue_plan)
                and len(issue_plan.rollout_briefs or []) >= 2
            ):
                (
                    speculative_results,
                    next_rollout_id,
                    speculative_seed_results,
                ) = self._run_speculative_first_attempt(
                    repo_path=repo_path,
                    repo_context=repo_context,
                    issue_description=issue_description,
                    issue_plan=issue_plan,
                    test_command=test_command,
                    engine=engine,
                    rollout_id_offset=next_rollout_id,
                    current_workspace_seed=current_workspace_seed,
                    wallclock_deadline=wallclock_deadline,
                )
                if speculative_results is not None:
                    if task_state_graph is not None:
                        task_state_graph.ingest_rollout_results(issue_plan, speculative_results)
                        issue_plan = self._refresh_task_state_context(issue_plan, task_state_graph)
                    return (
                        speculative_results,
                        issue_plan,
                        next_rollout_id,
                        transitions,
                        search_summary,
                    )
                # SPEED LEVER (cross-rollout discovery REUSE): the speculative
                # seed did NOT authoritatively pass, so the full slate is about
                # to fan out. Harvest the seed rollout's already-computed
                # reproduction + localization discovery and inject it as an
                # ADVISORY warm-start so the siblings spend their turns on
                # differentiated solving instead of re-deriving the identical
                # facts. Strictly advisory + high-confidence gated + giants-
                # size-gated; fully fail-open.
                if self._advisory_discovery_reuse_allowed(issue_plan):
                    (
                        advisory_seed_repro,
                        advisory_seed_loc,
                        advisory_seed_source_id,
                    ) = self._harvest_advisory_discovery(speculative_seed_results)
            results = self._execute_rollouts(
                repo_path=repo_path,
                repo_context=repo_context,
                issue_description=issue_description,
                issue_plan=issue_plan,
                test_command=test_command,
                engine=engine,
                rollout_id_offset=next_rollout_id,
                stop_on_result=self._build_authoritative_completion_stop_on_result(),
                workspace_seed=current_workspace_seed,
                workspace_seeds=current_workspace_seeds,
                wallclock_deadline=wallclock_deadline,
                advisory_seed_reproduction_artifact=advisory_seed_repro,
                advisory_seed_localization_artifact=advisory_seed_loc,
                advisory_seed_source_rollout_id=advisory_seed_source_id,
            )
            if task_state_graph is not None:
                task_state_graph.ingest_rollout_results(issue_plan, results)
                issue_plan = self._refresh_task_state_context(issue_plan, task_state_graph)
            next_rollout_id = self._next_rollout_id_after(
                results,
                fallback=next_rollout_id + len(results),
            )
            return (
                results,
                issue_plan,
                next_rollout_id,
                transitions,
                search_summary,
            )

        total_budget = len(issue_plan.rollout_briefs)
        seed_briefs = planner.select_progressive_seed_briefs(issue_plan)
        if len(seed_briefs) >= total_budget:
            results = self._execute_rollouts(
                repo_path=repo_path,
                repo_context=repo_context,
                issue_description=issue_description,
                issue_plan=issue_plan,
                test_command=test_command,
                engine=engine,
                rollout_id_offset=next_rollout_id,
                stop_on_result=self._build_authoritative_completion_stop_on_result(),
                workspace_seed=current_workspace_seed,
                workspace_seeds=current_workspace_seeds,
                wallclock_deadline=wallclock_deadline,
            )
            if task_state_graph is not None:
                task_state_graph.ingest_rollout_results(issue_plan, results)
                issue_plan = self._refresh_task_state_context(issue_plan, task_state_graph)
            next_rollout_id = self._next_rollout_id_after(
                results,
                fallback=next_rollout_id + len(results),
            )
            return (
                results,
                issue_plan,
                next_rollout_id,
                transitions,
                search_summary,
            )

        source_plan = IssuePlan.from_dict(issue_plan.to_dict())
        active_plan = self._clone_issue_plan_with_rollout_briefs(issue_plan, seed_briefs)
        all_results: list[RolloutResult] = []
        wave_index = 1
        wave_advisory_reuse_allowed = self._advisory_discovery_reuse_allowed(issue_plan)

        while True:
            if task_state_graph is not None:
                active_plan = self._refresh_task_state_context(active_plan, task_state_graph)
            self._save_issue_plan(active_plan)
            # SPEED LEVER (cross-rollout discovery REUSE): once a prior wave has
            # produced a high-confidence reproduction + localization, inject it
            # as an ADVISORY warm-start into the later waves so deepening
            # rollouts skip the redundant discovery and focus on solving. The
            # first wave has no prior results, so this is a no-op there. Strictly
            # advisory + high-confidence gated + giants-size-gated; fail-open.
            wave_advisory_repro: Optional[ReproductionArtifact] = None
            wave_advisory_loc: Optional[LocalizationArtifact] = None
            wave_advisory_source_id: Optional[int] = None
            if wave_advisory_reuse_allowed and all_results:
                (
                    wave_advisory_repro,
                    wave_advisory_loc,
                    wave_advisory_source_id,
                ) = self._harvest_advisory_discovery(all_results)
            wave_results = self._execute_rollouts(
                repo_path=repo_path,
                repo_context=repo_context,
                issue_description=issue_description,
                issue_plan=active_plan,
                test_command=test_command,
                engine=engine,
                rollout_id_offset=next_rollout_id,
                stop_on_result=self._build_progressive_stop_on_result(
                    prefix_results=all_results,
                ),
                workspace_seed=current_workspace_seed,
                workspace_seeds=current_workspace_seeds,
                wallclock_deadline=wallclock_deadline,
                advisory_seed_reproduction_artifact=wave_advisory_repro,
                advisory_seed_localization_artifact=wave_advisory_loc,
                advisory_seed_source_rollout_id=wave_advisory_source_id,
            )
            next_rollout_id = self._next_rollout_id_after(
                wave_results,
                fallback=next_rollout_id + len(wave_results),
            )
            all_results.extend(wave_results)
            # Rotate a diverse slate of seeds across sibling follow-up
            # rollouts so later waves do not all branch from the same
            # checkpoint. A4: widen the warm-start breadth as we deepen
            # (3 -> min(wave, 5)) and relax the overlap gate for deepening
            # waves so complementary partials are both retained.
            seed_breadth = min(max(3, wave_index + 1), 5)
            candidate_seeds = self._select_rollout_seeds(
                planner,
                all_results,
                k=seed_breadth,
                deepening=True,
            )
            if candidate_seeds:
                current_workspace_seeds = list(candidate_seeds)
                current_workspace_seed = candidate_seeds[(wave_index - 1) % len(candidate_seeds)]
            else:
                current_workspace_seeds = None
            if task_state_graph is not None:
                task_state_graph.ingest_rollout_results(source_plan, wave_results)
                active_plan = self._refresh_task_state_context(active_plan, task_state_graph)

            if self._should_stop_progressive_waves_early(wave_results, all_results=all_results):
                return all_results, active_plan, next_rollout_id, transitions, search_summary

            # SPEED LEVER (RANK-3B: verified-primary WAVE-STOP). Stop dispatching
            # FURTHER waves once K = max(2, ceil(size_factor)) rollouts already
            # carry a VERIFIED literal-1.0 primary (rollout_has_authoritative_
            # acceptance, which includes rank-1's literal-full-coverage path).
            # This ONLY short-circuits AFTER genuine verified 1.0 primaries
            # exist (never before 1.0). On giants K stays large (size_factor
            # saturates at max), so it is a practical no-op there. Already-
            # dispatched / in-flight rollouts and the deferred briefs in
            # ``issue_plan.rollout_briefs`` are untouched, so synthesis /
            # best-of-N / residual still draw from the full deque. Fully
            # fail-open: any error falls through to today's continuation logic.
            if self._verified_primary_wave_stop(issue_plan, all_results):
                return all_results, active_plan, next_rollout_id, transitions, search_summary

            remaining_budget = max(0, total_budget - len(all_results))
            if remaining_budget <= 0:
                return all_results, active_plan, next_rollout_id, transitions, search_summary
            if wave_index >= max(1, self.config.rollout.max_progressive_rollout_waves):
                return all_results, active_plan, next_rollout_id, transitions, search_summary
            if not planner.should_continue_progressive_waves(
                all_results,
                remaining_budget=remaining_budget,
                issue_plan=active_plan,
                task_state_context=active_plan.task_state_context,
            ):
                return all_results, active_plan, next_rollout_id, transitions, search_summary

            additional_rollouts = planner.recommend_followup_rollouts(
                source_plan,
                all_results,
                current_total_rollouts=len(all_results),
            )
            additional_rollouts = max(0, min(remaining_budget, additional_rollouts))
            additional_rollouts = self._cap_followup_rollouts_for_token_budget(
                all_results,
                requested_rollouts=additional_rollouts,
            )
            if additional_rollouts <= 0:
                return all_results, active_plan, next_rollout_id, transitions, search_summary

            focus_files = planner.extract_progressive_focus_files(source_plan, all_results)
            progress_summary = planner.summarize_progressive_signals(source_plan, all_results)
            next_plan = planner.build_progressive_wave_plan(
                source_plan,
                repo_context,
                all_results,
                additional_rollouts=additional_rollouts,
                progressive_summary=progress_summary,
                progressive_focus_files=focus_files,
                task_state_context=active_plan.task_state_context,
            )
            if not next_plan.rollout_briefs:
                return all_results, active_plan, next_rollout_id, transitions, search_summary

            # Record transition first so wave_index reported in the trace
            # matches the wave that just executed (+1 to count next plan, not stale).
            next_wave_index = wave_index + 1
            transitions.append(
                self._build_progressive_wave_transition(
                    planner=planner,
                    current_plan=active_plan,
                    next_plan=next_plan,
                    rollout_results=all_results,
                    wave_index=next_wave_index,
                    remaining_budget=remaining_budget,
                )
            )
            wave_index = next_wave_index
            active_plan = next_plan

    def _execute_with_dynamic_transitions(
        self,
        repo_path: str,
        repo_context: RepoContext,
        issue_description: str,
        issue_plan: IssuePlan,
        planner: IssuePlanner,
        initial_strategy: Any,
        test_command: Optional[str],
        verification_test_command: Optional[str] = None,
        engine: Optional[RolloutEngine] = None,
        transitions: Optional[list[dict[str, Any]]] = None,
        baseline_result: Optional[BaselineResult] = None,
        task_state_graph: Optional[TaskStateGraph] = None,
        benchmark_metadata: Optional[dict[str, Any]] = None,
        wallclock_deadline: Optional[float] = None,
    ) -> tuple[
        list[RolloutResult],
        IssuePlan,
        list[dict[str, Any]],
        Optional[dict[str, Any]],
    ]:
        engine = engine or RolloutEngine(self.config, repo_path, repo_context)
        transitions = list(transitions or [])
        current_plan = issue_plan
        current_strategy = initial_strategy
        all_rollout_results: list[RolloutResult] = []
        next_rollout_id = 0
        aggregated_search_summary: Optional[dict[str, Any]] = None
        current_workspace_seed: Optional[WorkspaceSeed] = None

        # Phase 2C 2.9: backstop the ``while True`` loop. The outer
        # planner is supposed to terminate by returning ``None`` from
        # ``escalate_execution_strategy`` — but a misbehaving planner
        # (or one stuck on a single primitive) could spin forever. Track
        # the iteration count and the last-seen strategy identity so we
        # can break with structured diagnostics.
        max_strategy_iterations = int(
            getattr(self.config.orchestration, "max_strategy_iterations", 0) or 0
        )
        strategy_iteration = 0
        previous_strategy_identity: Optional[Any] = None
        loop_guard_logger = logging.getLogger(
            f"{_ORCHESTRATOR_LOGGER_NAMESPACE}.dynamic_transitions"
        )

        while True:
            strategy_iteration += 1
            if max_strategy_iterations > 0 and strategy_iteration > max_strategy_iterations:
                loop_guard_logger.warning(
                    "loop_iteration_limit_reached: dynamic-transition loop hit "
                    "the max_strategy_iterations cap (%s). Returning best-so-far "
                    "to avoid an unbounded loop.",
                    max_strategy_iterations,
                )
                transitions.append(
                    {
                        "kind": "loop_iteration_limit_reached",
                        "max_strategy_iterations": max_strategy_iterations,
                        "iterations_executed": strategy_iteration - 1,
                        "reason": (
                            "dynamic-transition loop reached max_strategy_iterations; "
                            "returning best-so-far to bound runtime"
                        ),
                    }
                )
                return (
                    all_rollout_results,
                    current_plan,
                    transitions,
                    aggregated_search_summary,
                )
            (
                attempt_results,
                current_plan,
                next_rollout_id,
                progressive_transitions,
                search_summary,
            ) = self._execute_progressive_rollout_plan(
                repo_path=repo_path,
                repo_context=repo_context,
                issue_description=issue_description,
                issue_plan=current_plan,
                planner=planner,
                test_command=test_command,
                engine=engine,
                rollout_id_offset=next_rollout_id,
                task_state_graph=task_state_graph,
                initial_workspace_seed=current_workspace_seed,
                wallclock_deadline=wallclock_deadline,
            )
            transitions.extend(progressive_transitions)
            aggregated_search_summary = self._merge_search_summary(
                aggregated_search_summary,
                search_summary,
            )
            all_rollout_results.extend(attempt_results)

            # An authoritative full-scope pass — rc=0, every collected test
            # passing, coverage preserved — is a terminal completion signal even
            # when the agent CLI never serialized patch text (the materialized
            # worktree is the candidate). Stop the strategy loop on it instead of
            # escalating into another wave: that wasted wave is what a later
            # wall-clock/scheduler cancel discards together with this winning
            # candidate (a confident 1.0 rollout was produced early but the task
            # kept running until it was externally killed, leaving no result).
            # The same strategy-loop boundary applies to clean reduced-scope
            # candidates that explicitly require authoritative scoring: once
            # generation can no longer prove acceptance locally, return to the
            # selector/scorer instead of launching another strategy wave.
            if any(
                self._rollout_has_authoritative_completion_signal(result)
                or self._rollout_has_local_full_suite_completion_signal(result)
                or self._rollout_has_authoritative_scoring_stop_signal(result)
                for result in attempt_results
            ):
                return all_rollout_results, current_plan, transitions, aggregated_search_summary

            successful_results = [
                result for result in attempt_results if result.success and result.patch
            ]
            if not attempt_results:
                return all_rollout_results, current_plan, transitions, aggregated_search_summary
            if successful_results:
                if any(
                    self._rollout_has_strong_progressive_signal(result)
                    for result in successful_results
                ):
                    return all_rollout_results, current_plan, transitions, aggregated_search_summary
                best_attempt_result = self._select_best_patch(
                    repo_path=repo_path,
                    rollout_results=attempt_results,
                    issue_description=issue_description,
                    test_command=verification_test_command,
                    baseline_result=baseline_result,
                    issue_plan=current_plan,
                )
                if self._selected_result_is_accepted(best_attempt_result):
                    return all_rollout_results, current_plan, transitions, aggregated_search_summary

            current_workspace_seed = self._select_rollout_seed(
                planner,
                all_rollout_results,
            )
            next_strategy = planner.escalate_execution_strategy(current_strategy)
            if next_strategy is None:
                return all_rollout_results, current_plan, transitions, aggregated_search_summary

            # Phase 2C 2.9: detect a stuck planner. If ``escalate_*``
            # returns the SAME strategy identity twice in a row, the
            # planner is not actually escalating; another rollout would
            # produce the same wrong patch. Break with diagnostics.
            next_identity = self._strategy_identity_for_loop_guard(next_strategy)
            if (
                previous_strategy_identity is not None
                and next_identity is not None
                and next_identity == previous_strategy_identity
            ):
                loop_guard_logger.warning(
                    "strategy_stuck: planner returned identical strategy %r in "
                    "consecutive iterations; aborting dynamic-transition loop "
                    "to avoid burning rollouts on a wedged planner.",
                    next_identity,
                )
                transitions.append(
                    {
                        "kind": "strategy_stuck",
                        "strategy_identity": str(next_identity),
                        "iterations_executed": strategy_iteration,
                        "reason": (
                            "planner returned identical strategy in consecutive "
                            "iterations; returning best-so-far"
                        ),
                    }
                )
                return (
                    all_rollout_results,
                    current_plan,
                    transitions,
                    aggregated_search_summary,
                )
            previous_strategy_identity = next_identity

            transition = self._build_orchestration_transition(
                current_plan=current_plan,
                next_strategy=next_strategy,
                attempt_results=attempt_results,
                transition_index=len(transitions) + 1,
            )
            logger.info(
                "Escalating orchestration after attempt %s: %s -> %s (%s)",
                transition["attempt"],
                ", ".join(transition["from_primitives"]) or "none",
                ", ".join(transition["to_primitives"]) or "none",
                transition["reason"],
            )
            transitions.append(transition)
            current_strategy = next_strategy
            prior_task_state_context = (
                dict(current_plan.task_state_context)
                if isinstance(current_plan.task_state_context, dict)
                else {}
            )
            current_plan = self._plan_issue(
                issue_description,
                repo_context,
                planner=planner,
                rollout_count=next_strategy.rollout_count,
                difficulty=next_strategy.difficulty_estimate,
                baseline_result=baseline_result,
            )
            current_plan = planner.enrich_issue_plan(
                current_plan,
                issue_description=issue_description,
                repo_context=repo_context,
                test_command=test_command,
                baseline_result=baseline_result,
                benchmark_metadata=benchmark_metadata,
            )
            current_plan = planner.apply_execution_strategy(current_plan, next_strategy)
            current_plan.task_state_context = self._merge_task_state_context_payloads(
                current_plan.task_state_context,
                prior_task_state_context,
            )
            if task_state_graph is not None:
                current_plan = self._refresh_task_state_context(current_plan, task_state_graph)
                current_plan = planner.apply_task_state_frontier(
                    current_plan,
                    repo_context,
                    stage_label="escalation",
                )
            self._save_issue_plan(current_plan)

    def _all_rollouts_env_failed(
        self,
        rollout_results: list[RolloutResult],
    ) -> bool:
        """Phase 2C 2.2: True iff every rollout failed for an env reason.

        We use the per-rollout ``failure_class`` populated by Phase 1
        (rollout/engine.py) — when EVERY rollout has an env-class
        failure and none has a usable patch, the run is honestly
        ``ENV_SKIPPED`` rather than a real APEX miss. This is what the
        report layer needs to exclude env failures from the published
        APEX denominator.
        """
        if not rollout_results:
            return False
        any_real_attempt = False
        for result in rollout_results:
            patch = getattr(result, "patch", None)
            if patch:
                # If we got a patch out, even an unaccepted one, this
                # was not a pure env skip.
                return False
            failure_class = getattr(result, "failure_class", None)
            if failure_class is None:
                # If at least one rollout has no env classification, we
                # don't have grounds to call this an env skip.
                return False
            if isinstance(failure_class, CoreFailureClass):
                if not failure_class.is_environment:
                    return False
            else:
                # Unknown classification shape — be conservative and don't
                # categorise as ENV_SKIPPED.
                return False
            any_real_attempt = True
        return any_real_attempt

    def _strategy_identity_for_loop_guard(self, strategy: Any) -> Optional[Any]:
        """Phase 2C 2.9: derive a hashable identity for a strategy object.

        We compare ``next_strategy`` to the previously-seen one via this
        identity so a planner that returns the same strategy in
        consecutive iterations is detected and the loop short-circuits
        with ``strategy_stuck`` diagnostics.

        Strategy objects can vary in shape (PlanningDecision in normal
        operation, mocks in tests). We try common identity-bearing
        attributes; if none are present, fall back to the object's
        ``repr`` (which is hashable but order-stable enough for back-to-
        back comparison).
        """
        if strategy is None:
            return None
        # PlanningDecision-style identity
        primitives = getattr(strategy, "primitives", None)
        if primitives is not None:
            try:
                primitive_keys = tuple(getattr(p, "value", p) for p in primitives)
                rollout_count = getattr(strategy, "rollout_count", None)
                difficulty = getattr(strategy, "difficulty_estimate", None)
                return (primitive_keys, rollout_count, difficulty)
            except TypeError:
                pass
        # Generic fallback — repr is stable across consecutive comparisons
        # in a single process and hashable as a string.
        try:
            return repr(strategy)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _calibrated_signals_have_evidence_against(
        confidence: ConfidenceBreakdown,
    ) -> bool:
        """Phase 6.3: gate the abstention override on actual contradicting
        evidence.

        Returns True iff at least one SECONDARY signal (anything besides
        verifier_strength) is present and weak enough to contradict the
        acceptance gate. Supportive non-zero signals, for example full
        expected-test coverage, must never be treated as evidence against
        a verifier-accepted patch.

        This guard prevents a regression where a verifier-accepted patch
        is downgraded to ABSTAINED purely because mutation testing /
        f2p tracking / controller policy weren't wired in for that run.
        """
        weak_signal_threshold = min(0.35, max(0.0, float(confidence.threshold_used or 0.0)))
        secondary_keys = (
            "cluster_consensus",
            "controller_policy_certainty",
            "mutation_kill_rate",
            "f2p_consensus_rate",
        )
        for key in secondary_keys:
            value = float(confidence.breakdown.get(key, 0.0) or 0.0)
            if 0.0 < value < weak_signal_threshold:
                return True
        if float(confidence.breakdown.get("salvage_penalty", 0.0) or 0.0) > 0.0:
            return True
        return False

    def _compute_confidence(
        self,
        *,
        best_result: Optional[RolloutResult],
        rollout_results: list[RolloutResult],
        status: Status,
        effective_salvaged_flag: bool,
        benchmark_id: Optional[str] = None,
    ) -> Optional[ConfidenceBreakdown]:
        """Phase 6.3: build a calibrated ConfidenceBreakdown for the run.

        We construct a lightweight stand-in that mirrors the fields the
        scorer reads from ApexResult (verification_summary,
        selected_changed_files, salvaged) so we don't have to materialize
        the full ApexResult twice. ``status`` is the post-classification
        status BEFORE the abstention override.

        Decisive-Edge C.2: when ``benchmark_id`` is supplied, the scorer
        consults the per-benchmark calibrated thresholds JSON
        (``apex/configs/abstention_thresholds_per_benchmark.json``) and
        ``BenchmarkConfig.abstention_threshold_override`` ahead of the
        global ``OrchestrationConfig.abstention_threshold``.
        """
        try:
            threshold = float(getattr(self.config.orchestration, "abstention_threshold", 0.50))
        except (TypeError, ValueError):
            threshold = 0.50
        weights = getattr(self.config.orchestration, "abstention_weights", None)
        weights_dict: Optional[dict[str, float]] = None
        if isinstance(weights, dict) and weights:
            weights_dict = {str(k): float(v) for k, v in weights.items()}
        # Per-benchmark threshold override from BenchmarkConfig.
        benchmark_override: Optional[float] = None
        try:
            raw_override = getattr(self.config.benchmark, "abstention_threshold_override", None)
            if raw_override is not None:
                benchmark_override = float(raw_override)
        except (AttributeError, TypeError, ValueError):
            benchmark_override = None
        try:
            scorer = ConfidenceScorer(
                threshold=threshold,
                weights=weights_dict,
                benchmark_threshold_override=benchmark_override,
            )
        except Exception as exc:  # noqa: BLE001 — defensive, never fatal
            logger.warning(
                "Phase 6.3 confidence scorer init failed (%s); skipping calibration this run.",
                exc,
            )
            return None

        # Stand-in shaped like ApexResult — the scorer reads attributes,
        # not concrete classes.
        class _StandIn:
            pass

        stand_in = _StandIn()
        stand_in.verification_summary = (  # type: ignore[attr-defined]
            best_result.verification if best_result is not None else None
        )
        stand_in.selected_changed_files = (  # type: ignore[attr-defined]
            list(best_result.changed_files) if best_result is not None else []
        )
        stand_in.selected_rollout_id = (  # type: ignore[attr-defined]
            best_result.rollout_id if best_result is not None else None
        )
        stand_in.internally_accepted = (  # type: ignore[attr-defined]
            self._selected_result_is_accepted(best_result) if best_result else False
        )
        stand_in.salvaged = bool(effective_salvaged_flag)  # type: ignore[attr-defined]
        stand_in.salvaged_for_external_scoring = (  # type: ignore[attr-defined]
            bool(getattr(best_result, "salvaged_for_external_scoring", False))
            if best_result is not None
            else False
        )
        stand_in.status = status  # type: ignore[attr-defined]
        try:
            return scorer.score(
                stand_in,
                controller_action=None,
                rollout_results=rollout_results,
                benchmark_id=benchmark_id,
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            logger.warning(
                "Phase 6.3 confidence scoring raised (%s); leaving result.confidence=None.",
                exc,
            )
            return None

    def _select_best_patch(
        self,
        repo_path: str,
        rollout_results: list[RolloutResult],
        issue_description: str,
        test_command: Optional[str] = None,
        verifier: Optional[PatchVerifier] = None,
        baseline_result: Optional[BaselineResult] = None,
        issue_plan: Optional[IssuePlan] = None,
    ) -> Optional[RolloutResult]:
        verifier = verifier or self._build_verifier(repo_path)
        selector = PatchSelector(self.config, repo_path, verifier)
        # WS3C: inject a fresh-context LLM final-acceptance reviewer when enabled
        # (DEFAULT OFF). Prefer a distinct-family backend so the reviewer is an
        # independent judge; falls back to the actor family otherwise.
        if self.config.selection.enable_final_acceptance_reviewer:
            try:
                selector.final_acceptance_reviewer = self._build_final_acceptance_reviewer(
                    repo_path
                )
            except Exception:  # noqa: BLE001 - reviewer construction must not block selection
                logger.debug("Final-acceptance reviewer construction failed", exc_info=True)
                selector.final_acceptance_reviewer = None
        # Feature E: inject the perspective-diverse model critic when enabled
        # (DEFAULT ON). It is a LOW-PRIORITY tiebreaker among execution-verified
        # clusters only and fails open, so construction failures are a no-op.
        if self.config.selection.enable_perspective_review:
            try:
                selector.perspective_reviewer = self._build_perspective_reviewer(repo_path)
            except Exception:  # noqa: BLE001 - reviewer construction must not block selection
                logger.debug("Perspective reviewer construction failed", exc_info=True)
                selector.perspective_reviewer = None
        return selector.select_best_patch(
            rollout_results=rollout_results,
            issue_description=issue_description,
            test_command=test_command,
            baseline_result=baseline_result,
            issue_plan=issue_plan,
        )

    def _build_final_acceptance_reviewer(self, repo_path: str):
        """WS3C: construct the final-acceptance reviewer over a distinct-family
        backend when available (else the actor family)."""
        from ..selection.final_acceptance_reviewer import FinalAcceptanceReviewer

        configs = list(self.config.llm_configs or [])
        if not configs:
            return None
        actor = configs[0]
        actor_backend = str(getattr(getattr(actor, "backend", None), "value", "") or "")
        preferred = str(self.config.selection.final_acceptance_reviewer_backend or "").strip()
        chosen = None
        for cfg in configs:
            backend = str(getattr(getattr(cfg, "backend", None), "value", "") or "")
            if preferred and backend == preferred:
                chosen = cfg
                break
            if not preferred and backend and backend != actor_backend and chosen is None:
                chosen = cfg
        if chosen is None:
            if (
                self.config.selection.final_acceptance_reviewer_require_distinct_family
                and not preferred
            ):
                # No distinct-family reviewer available; fall back to the actor.
                chosen = actor
            else:
                chosen = actor
        reviewer_backend = str(getattr(getattr(chosen, "backend", None), "value", "") or "")
        try:
            if bool(getattr(chosen, "is_cli_backend", False)):
                from ..core.cli_backend import CLIModelClient

                reviewer_llm: Any = CLIModelClient(chosen)
            else:
                from ..core.llm import LLMClient

                reviewer_llm = LLMClient(chosen)
        except Exception:  # noqa: BLE001
            logger.debug("Reviewer LLM client construction failed", exc_info=True)
            return None
        return FinalAcceptanceReviewer(
            reviewer_llm,
            reviewer_backend=reviewer_backend,
            actor_backend=actor_backend,
            require_distinct_family=bool(
                self.config.selection.final_acceptance_reviewer_require_distinct_family
            ),
            working_dir=repo_path,
            timeout_seconds=int(self.config.selection.final_acceptance_reviewer_timeout_seconds),
        )

    def _build_perspective_reviewer(self, repo_path: str):
        """Feature E: construct the perspective-diverse model critic over a
        distinct-family backend when available (else the actor family), mirroring
        ``_build_final_acceptance_reviewer``. Returns None when disabled / no
        client so the selector skips the tiebreaker entirely (fail open)."""
        from ..selection.final_acceptance_reviewer import build_perspective_reviewer

        configs = list(self.config.llm_configs or [])
        if not configs:
            return None
        actor = configs[0]
        actor_backend = str(getattr(getattr(actor, "backend", None), "value", "") or "")
        preferred = str(self.config.selection.perspective_review_backend or "").strip()
        chosen = None
        for cfg in configs:
            backend = str(getattr(getattr(cfg, "backend", None), "value", "") or "")
            if preferred and backend == preferred:
                chosen = cfg
                break
            if not preferred and backend and backend != actor_backend and chosen is None:
                chosen = cfg
        if chosen is None:
            chosen = actor
        try:
            if bool(getattr(chosen, "is_cli_backend", False)):
                from ..core.cli_backend import CLIModelClient

                reviewer_llm: Any = CLIModelClient(chosen)
            else:
                from ..core.llm import LLMClient

                reviewer_llm = LLMClient(chosen)
        except Exception:  # noqa: BLE001 - client construction must not block selection
            logger.debug("Perspective reviewer LLM client construction failed", exc_info=True)
            return None
        reviewer = build_perspective_reviewer(self.config, reviewer_llm)
        if reviewer is not None:
            reviewer.working_dir = repo_path
            reviewer.actor_backend = actor_backend
        return reviewer

    def _build_verifier(self, repo_path: str) -> PatchVerifier:
        runtime_env_overrides = dict(self.config.llm_configs[0].cli_env_overrides or {})
        # Phase 3.2: resolve via the legacy ``apex.orchestrator`` shim
        # so existing tests that monkeypatch ``apex.orchestrator.PatchVerifier``
        # still intercept construction.
        verifier_cls = _resolve_patched("PatchVerifier", PatchVerifier)
        try:
            return verifier_cls(
                repo_path,
                timeout=self.config.selection.verification_timeout_seconds,
                full_test_timeout=self.config.selection.full_test_timeout_seconds,
                custom_test_timeout=self.config.selection.custom_test_timeout_seconds,
                runtime_env_overrides=runtime_env_overrides,
                verification_helper_files=list(self.config.selection.verification_helper_files),
            )
        except TypeError as exc:
            if (
                "timeout" not in str(exc)
                and "full_test_timeout" not in str(exc)
                and "custom_test_timeout" not in str(exc)
                and "runtime_env_overrides" not in str(exc)
                and "verification_helper_files" not in str(exc)
            ):
                raise
            if "verification_helper_files" in str(exc):
                try:
                    return verifier_cls(
                        repo_path,
                        timeout=self.config.selection.verification_timeout_seconds,
                        full_test_timeout=self.config.selection.full_test_timeout_seconds,
                        custom_test_timeout=self.config.selection.custom_test_timeout_seconds,
                        runtime_env_overrides=runtime_env_overrides,
                    )
                except TypeError as helper_fallback_exc:
                    if "runtime_env_overrides" not in str(helper_fallback_exc):
                        raise
            if "runtime_env_overrides" in str(exc):
                try:
                    return verifier_cls(
                        repo_path,
                        timeout=self.config.selection.verification_timeout_seconds,
                        full_test_timeout=self.config.selection.full_test_timeout_seconds,
                        custom_test_timeout=self.config.selection.custom_test_timeout_seconds,
                    )
                except TypeError as env_fallback_exc:
                    if (
                        "timeout" not in str(env_fallback_exc)
                        and "full_test_timeout" not in str(env_fallback_exc)
                        and "custom_test_timeout" not in str(env_fallback_exc)
                    ):
                        raise
            return verifier_cls(repo_path)

    def _selected_result_is_accepted(self, result: Optional[RolloutResult]) -> bool:
        """Phase 2C 2.2: STRICT acceptance gate.

        Acceptance requires either:
          * ``verification.accepted == True`` (authoritative verifier signal), OR
          * ``quick_verification`` has a strong full-scope signal
            (``require_full_scope=True`` floor).

        A stale ``accepted=False`` marker must not override a later clean
        full-suite quick-verification pass. That marker is often produced by
        selection fallback metadata before quick verification has a chance to
        corroborate the candidate.

        The legacy ``overall_score >= 0.9`` short-circuit is REMOVED — a
        soft heuristic score is no longer sufficient to mark a candidate
        as solved. A high score with no accepted=True and no strong
        full-scope signal must abstain; the salvage path now uses
        ``allow_salvage`` (RolloutConfig) instead of silently elevating
        salvage candidates to acceptance.
        """
        if result is None or not result.patch:
            return False
        verification = result.verification
        if verification_has_explicit_validity_rejection(verification):
            return False
        if rollout_has_submission_blocking_validity(result):
            return False
        if self._result_has_score_bearing_success(result):
            return True
        quick_verification = (
            result.quick_verification if isinstance(result.quick_verification, dict) else {}
        )
        if isinstance(verification, dict) and verification.get("accepted") is True:
            return True
        mode = ""
        metadata = getattr(result, "search_metadata", None)
        if isinstance(metadata, dict):
            mode = str(metadata.get("evidence_mode") or "").strip()
        if quick_verification_has_strong_signal(
            quick_verification,
            require_full_scope=True,
        ):
            if mode in {"partial_suite_visible", "hidden_suite_authoritative"}:
                return False
            return True
        if isinstance(verification, dict) and "accepted" in verification:
            return bool(verification["accepted"])
        # Default to NOT accepted: require positive evidence rather than
        # silently treating a missing-quick-verification + missing-acceptance
        # state as success. The previous ``overall_score >= 0.9`` branch
        # accepted salvage candidates with no verifier signal — Phase 2C 2.2
        # explicitly removes that, in favour of calibrated abstention.
        return False

    def _score_bearing_decision_payload(
        self,
        result: Optional[RolloutResult],
    ) -> Optional[dict[str, Any]]:
        if result is None:
            return None
        containers: list[dict[str, Any]] = []
        for candidate in (
            getattr(result, "search_metadata", None),
            getattr(result, "verification", None),
            getattr(result, "quick_verification", None),
            getattr(result, "selection_diagnostics", None),
        ):
            if isinstance(candidate, dict):
                containers.append(candidate)
        for container in containers:
            for key in (
                "evaluation_decision",
                "contract_decision",
                "scoring_decision",
                "benchmark_decision",
            ):
                payload = container.get(key)
                if isinstance(payload, dict) and "is_success" in payload:
                    return payload
        return None

    def _result_has_score_bearing_success(
        self,
        result: Optional[RolloutResult],
    ) -> bool:
        decision = self._score_bearing_decision_payload(result)
        if not isinstance(decision, dict):
            return False
        if decision.get("is_success") is not True:
            return False
        kind = str(decision.get("kind") or "").strip().lower()
        if kind and kind not in {"solved", "success", "passed"}:
            return False
        return True

    @staticmethod
    def _rollout_has_materialized_repair_seed(result: Optional[RolloutResult]) -> bool:
        return rollout_has_materialized_repair_seed(result)

    # Phase 3.2: ``_ADAPTIVE_FOLLOWUP_NEAR_MISS_MULTIPLIER`` and
    # ``_ADAPTIVE_FOLLOWUP_NEAR_MISS_PASS_RATE`` were class attributes
    # here. They now live on ``OrchestrationConfig`` so callers can
    # tune them without monkey-patching the class. Defaults preserved.

    def _followup_budget_exhausted(
        self,
        *,
        rollout_results: list[RolloutResult],
        followup_round: int,
    ) -> bool:
        """Phase 2 10.T: stop the followup loop when either cap fires.

        Two caps protect any long-running benchmark from token-runaway:

        * ``rollout.max_followup_iterations`` — total followup rounds across
          this repo (broader than ``max_selection_followup_rounds``, which
          is the per-repo wave cap that the adaptive multiplier already
          stretches for near-miss best results).
        * ``rollout.max_tokens_per_repo_followup`` — sum of ``total_tokens``
          across every rollout (initial waves + followups) for this repo.

        Returns True when either cap is hit; the orchestrator then breaks
        the followup loop and accepts best-so-far. ``<= 0`` disables the
        cap (so test fixtures and ablations can opt out cleanly).
        """

        max_iterations = int(getattr(self.config.rollout, "max_followup_iterations", 0) or 0)
        if max_iterations > 0 and followup_round >= max_iterations:
            logger.warning(
                "Phase 2 10.T followup-iteration cap hit: round=%s >= "
                "max_followup_iterations=%s; accepting best-so-far.",
                followup_round,
                max_iterations,
            )
            return True

        max_tokens = self._followup_token_cap()
        if max_tokens > 0:
            tokens_used = 0
            for rollout in rollout_results:
                try:
                    tokens_used += int(getattr(rollout, "total_tokens", 0) or 0)
                except (TypeError, ValueError):
                    continue
            if tokens_used >= max_tokens:
                logger.warning(
                    "Phase 2 10.T followup-token cap hit: tokens_used=%s "
                    ">= max_tokens_per_repo_followup=%s; accepting "
                    "best-so-far rather than relaunching.",
                    tokens_used,
                    max_tokens,
                )
                return True
        return False

    def _cap_followup_rollouts_for_token_budget(
        self,
        rollout_results: list[RolloutResult],
        *,
        requested_rollouts: int,
    ) -> int:
        requested = max(0, int(requested_rollouts or 0))
        if requested <= 0:
            return 0
        max_tokens = self._followup_token_cap()
        if max_tokens <= 0:
            return requested
        tokens_used = 0
        prior_costs: list[int] = []
        for rollout in rollout_results:
            try:
                cost = int(getattr(rollout, "total_tokens", 0) or 0)
            except (TypeError, ValueError):
                continue
            if cost > 0:
                tokens_used += cost
                prior_costs.append(cost)
        remaining = max_tokens - tokens_used
        if remaining <= 0:
            logger.warning(
                "Follow-up token budget exhausted before launch: tokens_used=%s "
                ">= max_tokens_per_repo_followup=%s.",
                tokens_used,
                max_tokens,
            )
            return 0
        if not prior_costs:
            return requested
        prior_costs.sort()
        p75_index = min(len(prior_costs) - 1, int(0.75 * (len(prior_costs) - 1)))
        estimated_rollout_cost = max(1, prior_costs[p75_index])
        allowed_by_budget = remaining // estimated_rollout_cost
        if allowed_by_budget <= 0:
            logger.warning(
                "Skipping follow-up launch: remaining token budget %s is below "
                "estimated rollout cost %s.",
                remaining,
                estimated_rollout_cost,
            )
            return 0
        capped = min(requested, int(allowed_by_budget))
        if capped < requested:
            logger.warning(
                "Capping follow-up rollouts from %s to %s to fit remaining "
                "token budget %s (estimated rollout cost=%s).",
                requested,
                capped,
                remaining,
                estimated_rollout_cost,
            )
        return capped

    @staticmethod
    def _seed_quick_verification_for_result(
        result: Optional[RolloutResult],
    ) -> dict[str, Any]:
        """Return the seed/baseline quick_verification recorded for a rollout.

        Used by the A2 milestone deepening predicate to compare a rollout
        against the partial it branched from. We read whatever baseline the
        engine recorded on the result's ``search_metadata`` (any of a few
        forward/back-compatible key spellings). When no baseline is available
        the predicate is inert, which keeps A2 zero-delta on already-fully
        collected benchmarks.
        """

        metadata = getattr(result, "search_metadata", None)
        if not isinstance(metadata, dict):
            return {}
        for key in (
            "seed_quick_verification",
            "baseline_quick_verification",
            "previous_quick_verification",
            "seed_collection_quick_verification",
        ):
            payload = metadata.get(key)
            if isinstance(payload, dict) and payload:
                return payload
        return {}

    def _rollout_made_collection_progress_vs_seed(
        self,
        result: Optional[RolloutResult],
    ) -> bool:
        """A2: True when a rollout deepened collection vs its seed (zero-delta if no seed)."""

        if result is None:
            return False
        qv = result.quick_verification if isinstance(result.quick_verification, dict) else {}
        if not qv:
            return False
        seed_qv = self._seed_quick_verification_for_result(result)
        return rollout_made_collection_progress_vs_seed(qv, seed_qv)

    def _effective_max_selection_followup_rounds(
        self,
        best_result: Optional[RolloutResult],
    ) -> int:
        configured = max(0, int(self.config.rollout.max_selection_followup_rounds))
        if configured == 0 or best_result is None:
            return configured
        qv = (
            best_result.quick_verification
            if isinstance(best_result.quick_verification, dict)
            else {}
        )
        near_miss_pass_rate = float(self.config.orchestration.adaptive_followup_near_miss_pass_rate)
        multiplier = int(self.config.orchestration.adaptive_followup_near_miss_multiplier)
        pass_rate = qv.get("pass_rate")
        if isinstance(pass_rate, (int, float)) and float(pass_rate) >= near_miss_pass_rate:
            return configured * multiplier
        # Also check the verification test_result for the same signal —
        # the QV pass_rate isn't always populated when the candidate
        # made it through full verification first.
        verification = best_result.verification or {}
        if isinstance(verification, dict):
            test_result = verification.get("test_result")
            if isinstance(test_result, dict):
                tr_pass = test_result.get("pass_rate")
                if isinstance(tr_pass, (int, float)) and float(tr_pass) >= near_miss_pass_rate:
                    return configured * multiplier
        # A2: milestone deepening — a candidate that recovered collection
        # surface vs its seed (but is not a 0.95 near-miss) earns the same
        # extended budget the near-miss candidate gets, so compounding effort
        # on the deepest partial is not abandoned. Inert when no seed baseline
        # exists (returns False), so this is zero-delta at full collection.
        if self._rollout_made_collection_progress_vs_seed(best_result):
            return configured * multiplier
        return configured

    def _followup_token_cap(self) -> int:
        benchmark_config = getattr(self.config, "benchmark", None)
        power_mode = (
            str(
                getattr(benchmark_config, "evaluation_power_mode", "")
                or getattr(benchmark_config, "power_mode", "")
                or ""
            )
            .strip()
            .lower()
        )
        unbounded = bool(
            getattr(benchmark_config, "unbounded_followup_budget", False)
            or power_mode in {"max", "maximum", "max_quality", "unlimited", "full_max"}
        )
        if unbounded:
            return 0
        return int(getattr(self.config.rollout, "max_tokens_per_repo_followup", 0) or 0)

    def _selected_result_needs_followup(self, result: Optional[RolloutResult]) -> bool:
        if result is None or not self._rollout_has_materialized_repair_seed(result):
            return False
        decision = self._score_bearing_decision_payload(result)
        if isinstance(decision, dict) and decision.get("is_success") is True:
            return bool(decision.get("requires_followup", False))
        if self._selected_result_is_accepted(result):
            return False
        if rollout_requires_authoritative_scoring(result):
            return False
        verification = result.verification if isinstance(result.verification, dict) else {}
        if verification.get("accepted") is False:
            return True
        if self._rollout_has_local_full_suite_completion_signal(result):
            mode = ""
            metadata = getattr(result, "search_metadata", None)
            if isinstance(metadata, dict):
                mode = str(metadata.get("evidence_mode") or "").strip()
            # In gold-suite-visible mode a clean full-suite rollout signal is
            # enough to stop residual selection followups; the benchmark
            # scorer will run the complete public scoring universe next. In
            # partial/hidden modes the same visible-suite pass is not
            # authoritative, so followups should continue unless a verifier
            # accepted the candidate.
            if mode in {"partial_suite_visible", "hidden_suite_authoritative"}:
                return True
            # A2: even with a clean local full-suite signal, if this rollout is
            # still actively deepening collection vs its seed (e.g. recovering
            # missing expected tests the local suite never exercised), keep the
            # followup loop alive so the milestone progress is not abandoned.
            # Inert (zero-delta) when no seed baseline / already-full collection.
            if self._rollout_made_collection_progress_vs_seed(result):
                return True
            return False
        return True

    def _select_invalid_selection_followup_anchor(
        self,
        rollout_results: list[RolloutResult],
    ) -> Optional[RolloutResult]:
        """Pick a repairable invalid candidate when strict selection abstained.

        This is intentionally not a fallback selector. The returned rollout is
        only a seed for residual repair; callers should still return ``None``
        as the final result unless a later selection pass accepts a candidate.
        """

        candidates: list[tuple[float, float, int, int, int, RolloutResult]] = []
        for result in rollout_results:
            if result is None or not result.success or not result.patch:
                continue
            if self._selected_result_is_accepted(result):
                continue
            if not self._result_has_repairable_validity_diagnostics(result):
                continue
            signal_score = self._selection_followup_anchor_signal(result)
            residual_defects = self._selection_followup_anchor_residual_defect_count(result)
            changed_file_count = len(list(result.changed_files or []))
            candidates.append(
                (
                    signal_score,
                    self._selection_followup_anchor_coverage_ratio(result),
                    -residual_defects,
                    -changed_file_count,
                    -int(result.rollout_id),
                    result,
                )
            )
        if not candidates:
            return None
        # Sort by the numeric ranking prefix ONLY, never the trailing
        # RolloutResult: rollout_id repeats across residual follow-up rounds, so
        # the whole-tuple sort could tie on the prefix and fall through to
        # comparing two RolloutResult objects (no __lt__) -> TypeError that
        # crashes the entire repo to 0.0 (observed on scrapy: 16 rollouts +
        # residual rounds). Keying on c[:-1] is stable on ties and never touches
        # the object.
        candidates.sort(key=lambda candidate_entry: candidate_entry[:-1], reverse=True)
        anchor = candidates[0][-1]
        anchor.selected_for_submission = False
        anchor.internally_accepted = False
        anchor.salvaged_for_external_scoring = False
        diagnostics = (
            anchor.selection_diagnostics if isinstance(anchor.selection_diagnostics, dict) else {}
        )
        diagnostics["residual_followup_anchor"] = {
            "reason": "strict_selection_abstained_with_repairable_validity_diagnostics",
            "signal_score": candidates[0][0],
            "expected_coverage_ratio": candidates[0][1],
            "residual_defect_count": -candidates[0][2],
        }
        anchor.selection_diagnostics = diagnostics
        return anchor

    def _result_has_repairable_validity_diagnostics(self, result: RolloutResult) -> bool:
        verification = result.verification if isinstance(result.verification, dict) else {}
        if verification_has_explicit_validity_rejection(verification):
            return True
        if rollout_has_submission_blocking_validity(result):
            return True
        validity_payload = self._candidate_validity_payload(result)
        if verification_has_explicit_validity_rejection({"validity": validity_payload}):
            return True
        diagnostics = (
            result.selection_diagnostics if isinstance(result.selection_diagnostics, dict) else {}
        )
        for key in ("stub_residue", "public_symbol_losses"):
            value = diagnostics.get(key)
            if isinstance(value, list) and value:
                return True
        failure_reason = str(result.failure_reason or "").lower()
        return any(
            phrase in failure_reason
            for phrase in (
                "quality gate",
                "stub residue",
                "protected test",
                "coverage collapse",
                "validity",
            )
        )

    @staticmethod
    def _candidate_validity_payload(result: RolloutResult) -> dict[str, Any]:
        validity = getattr(result, "validity", None)
        if isinstance(validity, dict):
            return validity
        if validity is None:
            return {}
        as_dict = getattr(validity, "as_dict", None)
        if callable(as_dict):
            try:
                payload = as_dict()
            except Exception:  # noqa: BLE001 - defensive normalization
                payload = {}
            if isinstance(payload, dict):
                return payload
        return {
            "eligible_for_submission": getattr(validity, "eligible_for_submission", None),
            "quick_verification_passed": getattr(validity, "quick_verification_passed", None),
            "protected_tests_unchanged": getattr(validity, "protected_tests_unchanged", None),
            "collection_critical_files_unchanged": getattr(
                validity,
                "collection_critical_files_unchanged",
                None,
            ),
            "expected_coverage_preserved": getattr(
                validity,
                "expected_coverage_preserved",
                None,
            ),
            "quality_gate_passed": getattr(validity, "quality_gate_passed", None),
        }

    def _selection_followup_anchor_signal(self, result: RolloutResult) -> float:
        quick_verification = (
            result.quick_verification if isinstance(result.quick_verification, dict) else {}
        )
        signal_score = quick_verification_signal_score(quick_verification)
        if isinstance(signal_score, (int, float)):
            return float(signal_score)
        pass_rate = quick_verification.get("pass_rate")
        if isinstance(pass_rate, (int, float)):
            return float(pass_rate)
        verification = result.verification if isinstance(result.verification, dict) else {}
        test_result = verification.get("test_result")
        if isinstance(test_result, dict):
            test_pass_rate = test_result.get("pass_rate")
            if isinstance(test_pass_rate, (int, float)):
                return float(test_pass_rate)
        return 0.0

    @staticmethod
    def _selection_followup_anchor_coverage_ratio(result: RolloutResult) -> float:
        quick_verification = (
            result.quick_verification if isinstance(result.quick_verification, dict) else {}
        )
        expected_coverage_ratio = quick_verification_expected_coverage_ratio(quick_verification)
        if isinstance(expected_coverage_ratio, (int, float)):
            return float(expected_coverage_ratio)
        if quick_verification.get("coverage_preserved") is True:
            return 1.0
        return 0.0

    def _selection_followup_anchor_residual_defect_count(
        self,
        result: RolloutResult,
    ) -> int:
        defects: set[str] = set()
        verification = result.verification if isinstance(result.verification, dict) else {}
        if verification.get("quality_gate_passed") is False:
            defects.add("quality_gate")
        if verification.get("syntax_valid") is False:
            defects.add("syntax")
        if verification.get("lint_clean") is False:
            defects.add("lint")
        prune_result = verification.get("prune_result")
        if isinstance(prune_result, dict) and prune_result.get("is_valid") is False:
            defects.add("prune")
        test_result = verification.get("test_result")
        if (
            isinstance(test_result, dict)
            and test_result.get("expected_coverage_preserved") is False
        ):
            defects.add("expected_coverage")
        validity_payload = self._candidate_validity_payload(result)
        if validity_payload.get("quality_gate_passed") is False:
            defects.add("quality_gate")
        if validity_payload.get("protected_tests_unchanged") is False:
            defects.add("protected_tests")
        if validity_payload.get("collection_critical_files_unchanged") is False:
            defects.add("collection_critical_files")
        if validity_payload.get("expected_coverage_preserved") is False:
            defects.add("expected_coverage")
        if validity_payload.get("eligible_for_submission") is False:
            defects.add("eligible_for_submission")
        if validity_payload.get("quick_verification_passed") is False:
            defects.add("quick_verification")
        diagnostics = (
            result.selection_diagnostics if isinstance(result.selection_diagnostics, dict) else {}
        )
        stub_residue = diagnostics.get("stub_residue")
        if isinstance(stub_residue, list):
            defects.update(f"stub_residue:{index}" for index, _ in enumerate(stub_residue))
        symbol_losses = diagnostics.get("public_symbol_losses")
        if isinstance(symbol_losses, list):
            defects.update(f"public_symbol_loss:{index}" for index, _ in enumerate(symbol_losses))
        if rollout_has_submission_blocking_validity(result):
            defects.add("submission_blocking_validity")
        return max(1, len(defects))

    def _rollout_has_expected_coverage_gap(
        self,
        result: Optional[RolloutResult],
    ) -> bool:
        if result is None or not self._rollout_has_materialized_repair_seed(result):
            return False
        quick_verification = (
            result.quick_verification if isinstance(result.quick_verification, dict) else {}
        )
        missing_expected_test_count = quick_verification.get("missing_expected_test_count")
        if isinstance(missing_expected_test_count, int) and missing_expected_test_count > 0:
            return True
        if quick_verification.get("coverage_preserved") is False:
            return True
        expected_coverage_ratio = quick_verification_expected_coverage_ratio(quick_verification)
        return (
            isinstance(expected_coverage_ratio, (int, float))
            and float(expected_coverage_ratio) < 0.999
        )

    def _rollout_has_local_full_suite_completion_signal(
        self,
        result: Optional[RolloutResult],
    ) -> bool:
        if result is None or not self._rollout_has_materialized_repair_seed(result):
            return False
        quick_verification = (
            result.quick_verification if isinstance(result.quick_verification, dict) else {}
        )
        return quick_verification_has_local_full_scope_pass(quick_verification)

    # Phase 2C 3.3: ``_REPO_TOKEN_HARD_CAP`` was a hard-coded 100M-token
    # ceiling that aborted progressive waves on any repo with no
    # accepted rollout. Per the project directive ("never optimize for
    # cost"), this magic constant is removed; the cap now lives on
    # ``OrchestrationConfig.repo_token_cap`` and defaults to ``None`` so
    # callers must opt in explicitly. The abort path
    # (``_cumulative_token_cap_exceeded``) is preserved unchanged but is
    # a no-op when the config value is None.
    # Phase 3.2: ``_REPEATED_BLOCKER_STOP_AFTER`` is now
    # ``OrchestrationConfig.repeated_blocker_stop_after`` (default 3).

    def _verified_primary_wave_stop(
        self,
        issue_plan: Optional[IssuePlan],
        all_results: list[RolloutResult],
    ) -> bool:
        """RANK-3B: stop further waves once >= K verified literal-1.0 primaries.

        ``K = max(2, ceil(size_factor))`` where ``size_factor`` is the EXISTING
        ``_rollout_budget_size_factor`` signal. A rollout counts only when
        ``rollout_has_authoritative_acceptance`` is True (success + patch +
        no submission-blocking validity + a verified completion, including
        rank-1's literal-full-coverage path). This can ONLY short-circuit AFTER
        genuine verified 1.0 primaries exist — it NEVER fires before a literal
        1.0, and on giants ``size_factor`` saturates at the configured max so
        ``K`` stays large (practical no-op there). Already-dispatched / in-flight
        rollouts and the deferred briefs are untouched. Fully fail-open: any
        error returns False (today's continuation behavior).
        """

        try:
            if not all_results:
                return False
            rollout_cfg = getattr(self.config, "rollout", None)
            max_size_factor = int(getattr(rollout_cfg, "rollout_budget_max_size_factor", 6) or 6)
            tests_per_unit = int(
                getattr(rollout_cfg, "rollout_budget_tests_per_unit", 2000) or 2000
            )
            size_factor = _rollout_budget_size_factor(
                _issue_plan_expected_test_count(issue_plan),
                tests_per_unit=tests_per_unit,
                max_size_factor=max_size_factor,
            )
            # K = max(2, ceil(size_factor)). size_factor is already an int >= 1,
            # so ceil is the identity; the floor of 2 preserves a best-of-2
            # verified bar on small suites, and K grows with the suite (== the
            # full size_factor on giants, which is >= max => never reached before
            # a large number of verified 1.0 primaries).
            required = max(2, int(size_factor))
            verified = 0
            for result in all_results:
                if rollout_has_authoritative_acceptance(result):
                    verified += 1
                    if verified >= required:
                        return True
            return False
        except Exception as stop_exc:  # noqa: BLE001 - fail-open to continuation
            logger.debug("Verified-primary wave-stop gate failed: %s", stop_exc)
            return False

    def _should_stop_progressive_waves_early(
        self,
        wave_results: list[RolloutResult],
        all_results: Optional[list[RolloutResult]] = None,
    ) -> bool:
        if any(self._progressive_stop_on_result(result) for result in wave_results):
            return True
        if not all_results:
            return False
        if self._repeated_blocker_should_stop(all_results):
            return True
        if self._cumulative_token_cap_exceeded(all_results):
            return True
        return False

    def _repeated_blocker_should_stop(
        self,
        all_results: list[RolloutResult],
    ) -> bool:
        stop_after = int(self.config.orchestration.repeated_blocker_stop_after)
        if stop_after <= 0 or len(all_results) < stop_after:
            return False
        if any(r is not None and r.success and r.patch for r in all_results):
            return False
        counts: Counter[str] = Counter()
        for result in all_results:
            if result is None:
                continue
            kind = self._rollout_structural_blocker_kind(result)
            if not kind:
                continue
            counts[kind] += 1
            if counts[kind] >= stop_after:
                return True
        return False

    def _build_progressive_stop_on_result(
        self,
        *,
        prefix_results: list[RolloutResult],
    ) -> Callable[[RolloutResult], bool]:
        observed_results: list[RolloutResult] = []
        prefix_snapshot = list(prefix_results)

        def stop_on_result(result: RolloutResult) -> bool:
            setattr(stop_on_result, "preempt_active_rollouts", False)
            setattr(stop_on_result, "continue_dispatch_after_nonpreemptive_stop", False)
            setattr(stop_on_result, "drain_active_rollouts_after_nonpreemptive_stop", False)
            observed_results.append(result)
            if self._progressive_stop_on_result(result):
                if (
                    self._rollout_has_preemptive_completion_signal(result)
                    or self._rollout_has_repairable_near_miss_signal(result)
                    or self._rollout_has_preemptive_authoritative_scoring_request(result)
                ):
                    setattr(stop_on_result, "preempt_active_rollouts", True)
                elif self._rollout_has_authoritative_scoring_stop_signal(result):
                    setattr(
                        stop_on_result,
                        "drain_active_rollouts_after_nonpreemptive_stop",
                        True,
                    )
                return True
            repeated_blocker_stop = self._repeated_blocker_should_stop(
                [*prefix_snapshot, *observed_results],
            )
            if repeated_blocker_stop:
                setattr(stop_on_result, "preempt_active_rollouts", True)
                return True
            return False

        setattr(stop_on_result, "preempt_active_rollouts", False)
        setattr(stop_on_result, "continue_dispatch_after_nonpreemptive_stop", False)
        setattr(stop_on_result, "drain_active_rollouts_after_nonpreemptive_stop", False)
        return stop_on_result

    def _build_authoritative_completion_stop_on_result(self) -> Callable[[RolloutResult], bool]:
        def stop_on_result(result: RolloutResult) -> bool:
            setattr(stop_on_result, "preempt_active_rollouts", False)
            setattr(stop_on_result, "continue_dispatch_after_nonpreemptive_stop", False)
            setattr(stop_on_result, "drain_active_rollouts_after_nonpreemptive_stop", False)
            if (
                self._rollout_has_preemptive_completion_signal(result)
                or self._rollout_has_repairable_near_miss_signal(result)
                or self._rollout_has_preemptive_authoritative_scoring_request(result)
            ):
                setattr(stop_on_result, "preempt_active_rollouts", True)
                return True
            if self._rollout_has_authoritative_scoring_stop_signal(result):
                setattr(stop_on_result, "drain_active_rollouts_after_nonpreemptive_stop", True)
                return True
            return False

        setattr(stop_on_result, "preempt_active_rollouts", False)
        setattr(stop_on_result, "continue_dispatch_after_nonpreemptive_stop", False)
        setattr(stop_on_result, "drain_active_rollouts_after_nonpreemptive_stop", False)
        return stop_on_result

    def _cumulative_token_cap_exceeded(
        self,
        all_results: list[RolloutResult],
    ) -> bool:
        # Phase 2C 3.3: cap is config-driven now. ``None`` (default)
        # means no cap — never abort on token spend alone.
        cap = getattr(self.config.orchestration, "repo_token_cap", None)
        if cap is None or int(cap) <= 0:
            return False
        if any(r is not None and r.success and r.patch for r in all_results):
            return False
        total_tokens = 0
        for result in all_results:
            usage = getattr(result, "usage", None)
            if isinstance(usage, dict):
                tokens = usage.get("total_tokens") or usage.get("tokens")
                if isinstance(tokens, (int, float)):
                    total_tokens += int(tokens)
                    continue
            tokens = getattr(result, "total_tokens", None)
            if isinstance(tokens, (int, float)):
                total_tokens += int(tokens)
        return total_tokens > int(cap)

    def _progressive_stop_on_result(
        self,
        result: Optional[RolloutResult],
    ) -> bool:
        if self._rollout_has_repairable_near_miss_signal(result):
            return True
        if self._rollout_has_authoritative_scoring_stop_signal(result):
            return True
        if self.config.rollout.progressive_stop_on_strong_signal:
            return self._rollout_has_strong_progressive_signal(result)
        return self._rollout_has_authoritative_completion_signal(result)

    def _rollout_has_repairable_near_miss_signal(
        self,
        result: Optional[RolloutResult],
    ) -> bool:
        """Return whether a full-suite candidate is better repaired than ignored.

        This is a search-control signal, not acceptance. A candidate that has
        run the full command, preserved coverage, and missed only a tiny
        residual slice should seed follow-up repair immediately instead of
        waiting for every remaining baseline rollout in the current wave.
        """

        configured_threshold = float(
            getattr(self.config.orchestration, "adaptive_followup_near_miss_pass_rate", 0.95)
            or 0.95
        )
        return rollout_has_repairable_near_miss(
            result,
            minimum_signal_score=max(0.999, configured_threshold),
            residual_fraction_cap=0.001,
            max_residual_count=50,
        )

    def _rollout_has_authoritative_completion_signal(
        self,
        result: Optional[RolloutResult],
    ) -> bool:
        return rollout_has_authoritative_acceptance(result)

    def _rollout_has_preemptive_completion_signal(
        self,
        result: Optional[RolloutResult],
    ) -> bool:
        return rollout_has_preemptive_completion(result)

    def _rollout_requires_authoritative_scoring(
        self,
        result: Optional[RolloutResult],
    ) -> bool:
        return rollout_requires_authoritative_scoring(result)

    def _rollout_has_authoritative_scoring_stop_signal(
        self,
        result: Optional[RolloutResult],
    ) -> bool:
        return rollout_has_authoritative_scoring_stop_signal(result)

    def _rollout_has_preemptive_authoritative_scoring_request(
        self,
        result: Optional[RolloutResult],
    ) -> bool:
        return rollout_has_preemptive_authoritative_scoring_request(result)

    def _rollout_has_strong_progressive_signal(
        self,
        result: Optional[RolloutResult],
    ) -> bool:
        if result is None or not result.success or not result.patch:
            return False

        quick_verification = (
            result.quick_verification if isinstance(result.quick_verification, dict) else {}
        )
        if quick_verification_has_local_full_scope_pass(quick_verification):
            return True
        if quick_verification_has_strong_signal(quick_verification):
            return True

        verification = result.verification if isinstance(result.verification, dict) else {}
        if verification.get("accepted") is True:
            return True

        return False

    def _build_selection_residual_summary(
        self,
        *,
        issue_plan: IssuePlan,
        rollout_results: list[RolloutResult],
        best_result: RolloutResult,
    ) -> str:
        verification = best_result.verification or {}
        diagnostics = (
            best_result.selection_diagnostics
            if isinstance(best_result.selection_diagnostics, dict)
            else {}
        )
        critic = diagnostics.get("critic") if isinstance(diagnostics, dict) else {}
        test_result = verification.get("test_result") if isinstance(verification, dict) else None
        parts = [
            "Previous rollouts produced candidate patches, but none met the acceptance bar.",
        ]
        critic_summary = (critic.get("summary") or "").strip() if isinstance(critic, dict) else ""
        if critic_summary:
            parts.append(f"Selection critic: {critic_summary}")
        if isinstance(test_result, dict):
            if test_result.get("regression_passes") is False:
                parts.append("The current best candidate still regresses visible tests.")
            if test_result.get("reproduction_passes") is False:
                parts.append("The current best candidate still fails the reproduction step.")
            passed = test_result.get("passed")
            failed = test_result.get("failed")
            errors = test_result.get("errors")
            if any(isinstance(value, int) and value > 0 for value in (passed, failed, errors)):
                parts.append(
                    "Observed verification counts: "
                    f"passed={passed or 0}, failed={failed or 0}, errors={errors or 0}."
                )
        static_focus_files: list[str] = []
        if isinstance(verification, dict):
            if verification.get("syntax_valid") is False:
                parts.append(
                    "Verifier static validity rejection: candidate syntax validation failed."
                )
            lint_output = str(verification.get("lint_output") or "").strip()
            if verification.get("lint_clean") is False:
                lint_lines = [line.strip() for line in lint_output.splitlines() if line.strip()][:8]
                if isinstance(test_result, dict) and test_result.get("regression_passes") is True:
                    parts.append(
                        "The current best candidate already passes the verifier regression "
                        "suite; make a minimal static-validity repair to the named diagnostics "
                        "and preserve the passing test behavior. Rerun the verifier/static "
                        "validity check after editing; a behavioral test pass alone is not enough."
                    )
                if lint_lines:
                    parts.append(
                        "Verifier lint rejection diagnostics to repair: "
                        + "; ".join(lint_lines)
                        + "."
                    )
                    for line in lint_lines:
                        match = re.match(r"^(?P<path>[^:\n]+):\d+(?::\d+)?:", line)
                        if match:
                            rel_path = match.group("path").strip()
                            if rel_path.startswith("./"):
                                rel_path = rel_path[2:]
                            if rel_path and rel_path not in static_focus_files:
                                static_focus_files.append(rel_path)
                    source_context = self._collect_verifier_diagnostic_source_context(
                        best_result,
                        lint_lines,
                    )
                    if source_context:
                        parts.append(
                            "Verifier diagnostic source context from the best candidate:\n"
                            + source_context
                        )
                else:
                    parts.append(
                        "Verifier lint rejection: candidate has static lint failures. "
                        "Repair the verifier-reported diagnostics before broadening."
                    )
            prune_result = verification.get("prune_result")
            if isinstance(prune_result, dict) and prune_result.get("is_valid") is False:
                reason = str(prune_result.get("reason") or "regression pruning failed").strip()
                parts.append(f"Verifier prune rejection: {reason}.")
            if (
                isinstance(test_result, dict)
                and test_result.get("expected_coverage_preserved") is False
            ):
                missing = test_result.get("missing_expected_test_count")
                if isinstance(missing, int):
                    parts.append(
                        "Verifier coverage rejection: expected test coverage collapsed "
                        f"({missing} expected tests missing)."
                    )
                else:
                    parts.append("Verifier coverage rejection: expected test coverage collapsed.")
                verifier_missing_ids = [
                    str(raw).strip()
                    for raw in list(test_result.get("missing_expected_test_ids") or [])
                    if str(raw or "").strip()
                ]
                if verifier_missing_ids:
                    missing_groups, has_parametrized_missing_ids = (
                        self._summarize_missing_expected_test_groups(verifier_missing_ids)
                    )
                    if missing_groups:
                        parts.append(
                            "Verifier missing expected-test groups: "
                            + "; ".join(missing_groups)
                            + "."
                        )
                    shown = verifier_missing_ids[:12]
                    extra = (
                        f" (and {len(verifier_missing_ids) - len(shown)} more)"
                        if len(verifier_missing_ids) > len(shown)
                        else ""
                    )
                    parts.append(
                        "Sample expected test IDs missing from the final verifier collection: "
                        + ", ".join(shown)
                        + extra
                        + "."
                    )
                    if has_parametrized_missing_ids:
                        parts.append(
                            "The missing IDs include parametrized cases; preserve the "
                            "test collection and parameterization surface while repairing "
                            "the source behavior."
                        )
        # Enumerate the specific failing/missing test IDs from the best
        # rollout's quick verification so the followup agent can target them
        # directly instead of re-deriving the residual from focus files.
        # 25 is enough to surface a cluster pattern (e.g. all failures in
        # one module) without overwhelming the prompt budget.
        anchor_qv = (
            best_result.quick_verification
            if isinstance(best_result.quick_verification, dict)
            else {}
        )
        failing_ids, failing_ids_match_counts = self._residual_failed_test_ids(
            anchor_qv=anchor_qv,
            test_result=test_result if isinstance(test_result, dict) else {},
        )
        if failing_ids:
            shown = failing_ids[:25]
            extra = (
                f" (and {len(failing_ids) - len(shown)} more)"
                if len(failing_ids) > len(shown)
                else ""
            )
            label = (
                "Specific failing tests still observed in the best candidate: "
                if failing_ids_match_counts
                else "Recent quick-verification failing tests to recheck: "
            )
            parts.append(label + ", ".join(shown) + extra + ".")
        missing_ids = list(anchor_qv.get("missing_expected_test_ids") or [])
        if missing_ids:
            shown = missing_ids[:25]
            extra = (
                f" (and {len(missing_ids) - len(shown)} more)"
                if len(missing_ids) > len(shown)
                else ""
            )
            parts.append(
                "Expected tests not collected by the best candidate (likely import / collection breakage): "
                + ", ".join(shown)
                + extra
                + "."
            )
        # Public-symbol losses — when the patch deleted top-level public
        # functions/classes that existed in baseline, name them so the
        # followup explicitly restores them.
        symbol_losses = (
            diagnostics.get("public_symbol_losses") if isinstance(diagnostics, dict) else None
        )
        if symbol_losses:
            shown = list(symbol_losses)[:8]
            entries = [
                f"{item.get('path', '?')}::{item.get('symbol', '?')} ({item.get('kind', '?')})"
                for item in shown
                if isinstance(item, dict)
            ]
            if entries:
                extra_count = len(symbol_losses) - len(entries)
                suffix = f" (and {extra_count} more)" if extra_count > 0 else ""
                parts.append(
                    "Public symbols present in the baseline but missing from the candidate: "
                    + "; ".join(entries)
                    + suffix
                    + ". Restore these (or rename their callers if intentional)."
                )

        # Stub-residue findings (cross-language) — names the unimplemented
        # functions in the patch so the followup agent has a concrete target
        # rather than re-discovering the same gap from test failures.
        stub_residue = diagnostics.get("stub_residue") if isinstance(diagnostics, dict) else None
        stub_residue_drives_residual = self._stub_residue_should_drive_residual(best_result)
        if stub_residue and stub_residue_drives_residual:
            shown = list(stub_residue)[:12]
            entries = [
                f"{item.get('path', '?')}::{item.get('symbol', '?')} ({item.get('reason', '?')})"
                for item in shown
                if isinstance(item, dict)
            ]
            if entries:
                extra_count = len(stub_residue) - len(entries)
                suffix = f" (and {extra_count} more)" if extra_count > 0 else ""
                parts.append(
                    "Unimplemented function bodies still in the candidate patch: "
                    + "; ".join(entries)
                    + suffix
                    + "."
                )
            for path in self._stub_residue_focus_files(best_result):
                if path not in static_focus_files:
                    static_focus_files.append(path)

        # Verbatim failure excerpts for the small residual failing set, sourced
        # via the test-runner adapter so the residual prompt shows the
        # exact "expected vs actual" the test asserts on. Critical for
        # output-text-precision repos like pylint/snapshot suites.
        excerpt_limit = len(failing_ids) if 0 < len(failing_ids) <= 8 else 3
        excerpt_block = self._collect_failure_excerpts(
            best_result,
            max_failures=excerpt_limit,
        )
        if excerpt_block:
            parts.append("Failure excerpts from the best candidate: " + excerpt_block)

        partial_signal_files = [
            path
            for path in list(best_result.changed_files or [])
            if not self._is_non_source_residual_focus_path(path)
        ]
        risk_source_files = [
            path
            for path in list(issue_plan.risk_files or [])
            if not self._is_non_source_residual_focus_path(path)
        ]
        if static_focus_files:
            parts.append(
                "Focus follow-up search on verifier/validity-rejected files: "
                + ", ".join(static_focus_files[:12])
                + "."
            )
        elif partial_signal_files:
            parts.append(
                "Focus follow-up search on files with partial signal: "
                + ", ".join(partial_signal_files[:4])
                + "."
            )
        elif risk_source_files:
            parts.append(
                "Focus follow-up search on the highest-risk files: "
                + ", ".join(risk_source_files[:4])
                + "."
            )

        high_signal_files = self._selection_residual_focus_files(
            issue_plan=issue_plan,
            rollout_results=rollout_results,
            best_result=best_result,
        )
        if high_signal_files:
            parts.append(
                "Cross-rollout residual focus files: " + ", ".join(high_signal_files[:12]) + "."
            )
        return " ".join(parts)

    def _extract_test_source_for_id(
        self,
        worktree: Path,
        test_id: str,
        *,
        max_chars: int = 600,
    ) -> str:
        """Best-effort extraction of the verbatim test function body.

        Works for any language whose nodeid prefix is the source file
        path. The first segment before ``::`` is treated as the file;
        the last segment is the symbol name. The lookup is intentionally
        cheap (read file, regex for the symbol), avoiding any AST cost.
        Returns empty string when the file or symbol can't be found.
        """
        if not test_id or "::" not in test_id:
            return ""
        file_part, _, rest = test_id.partition("::")
        symbol = rest.split("::")[-1].split("[")[0]
        if not file_part or not symbol:
            return ""
        path = worktree / file_part
        if not path.exists() or not path.is_file():
            return ""
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        lines = source.splitlines()
        # Find a line that defines this symbol (works for Python def,
        # JS/TS function/class methods, Go func, Rust fn, Java methods,
        # Ruby def; the heuristic is "the symbol token followed by `(`").
        signature_idx: Optional[int] = None
        target = symbol
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if (
                f" {target}(" in line
                or f"\t{target}(" in line
                or stripped.startswith(f"{target}(")
                or stripped.startswith(f"def {target}(")
                or stripped.startswith(f"async def {target}(")
                or stripped.startswith(f"function {target}(")
                or stripped.startswith(f"fn {target}(")
                or stripped.startswith(f"func {target}(")
                or f" {target} =" in stripped[:80]  # JS arrow funcs
            ):
                signature_idx = idx
                break
        if signature_idx is None:
            return ""
        # Capture up to ~60 lines or until the next sibling at the same
        # indent. This is a heuristic snip — sufficient for a residual
        # prompt; not a full parser.
        first_line = lines[signature_idx]
        indent = len(first_line) - len(first_line.lstrip())
        captured = [first_line]
        for line in lines[signature_idx + 1 : signature_idx + 60]:
            stripped = line.strip()
            current_indent = len(line) - len(line.lstrip())
            if stripped and current_indent <= indent and not line.startswith(" "):
                # Hit a same-or-lesser-indent non-blank — end of body.
                # (Only true for Python; for brace languages the body
                # often extends past this, but the heuristic keeps the
                # excerpt bounded.)
                break
            captured.append(line)
        excerpt = "\n".join(captured)
        if len(excerpt) > max_chars:
            excerpt = excerpt[:max_chars] + "…"
        return excerpt

    def _collect_failure_excerpts(
        self,
        best_result: RolloutResult,
        *,
        max_failures: int = 3,
        max_excerpt_chars: int = 600,
    ) -> str:
        """Use the test-runner adapter to extract verbatim failure text
        for the top failing tests, suitable for embedding in the residual
        followup prompt.
        """
        anchor_qv = (
            best_result.quick_verification
            if isinstance(best_result.quick_verification, dict)
            else {}
        )
        failing_ids = list(anchor_qv.get("failed_tests") or [])[:max_failures]
        if not failing_ids:
            return ""
        worktree_path = getattr(best_result, "worktree_path", None) or ""
        if not worktree_path:
            return ""
        try:
            from ..core.test_runners import detect_adapter
        except ImportError:
            return ""
        workspace = Path(worktree_path)
        if not workspace.exists():
            return ""
        adapter = detect_adapter(workspace)
        if adapter is None:
            return ""
        # Locate the report that produced the failed_tests list. Best
        # rollout's verification path tends to live under the rollout's
        # workspace; we try the canonical relative locations in order.
        candidate_reports = [
            workspace / "rollout_report.json",
            workspace / ".pytest_cache" / "rollout_report.json",
        ]
        report_path = next((p for p in candidate_reports if p.exists()), None)
        if report_path is None:
            return ""
        excerpts: list[str] = []
        for test_id in failing_ids:
            failure = adapter.extract_failure_excerpt(test_id, report_path)
            test_source = self._extract_test_source_for_id(workspace, test_id)
            if not failure and not test_source:
                continue
            if failure and len(failure) > max_excerpt_chars:
                failure = failure[:max_excerpt_chars] + "…"
            block_parts = [f"[{test_id}]"]
            if test_source:
                block_parts.append(f"TEST SOURCE:\n{test_source}")
            if failure:
                block_parts.append(f"FAILURE:\n{failure}")
            excerpts.append("\n".join(block_parts))
        return "\n\n---\n\n".join(excerpts)

    def _collect_verifier_diagnostic_source_context(
        self,
        best_result: RolloutResult,
        lint_lines: list[str],
        *,
        max_locations: int = 6,
        context_lines: int = 4,
        max_chars: int = 5000,
    ) -> str:
        """Return bounded source snippets around verifier diagnostics.

        The verifier already reduced the problem to concrete file/line
        diagnostics. Feeding those exact local lines back into repair
        prompts is a general signal: it helps any language/toolchain whose
        diagnostics use ``path:line`` without encoding repository-specific
        policy in the orchestrator.
        """

        if not lint_lines:
            return ""
        worktree_path = getattr(best_result, "worktree_path", None) or ""
        if not worktree_path:
            return ""
        workspace = Path(worktree_path)
        if not workspace.exists() or not workspace.is_dir():
            return ""

        blocks: list[str] = []
        seen: set[tuple[str, int]] = set()
        total_chars = 0
        for raw_line in lint_lines:
            if len(blocks) >= max_locations:
                break
            match = re.match(
                r"^(?P<path>[^:\n]+):(?P<line>\d+)(?::(?P<column>\d+))?:\s*(?P<message>.*)$",
                raw_line.strip(),
            )
            if not match:
                continue
            rel_path = match.group("path").strip().replace("\\", "/")
            if rel_path.startswith("./"):
                rel_path = rel_path[2:]
            path_obj = Path(rel_path)
            if (
                not rel_path
                or path_obj.is_absolute()
                or any(part == ".." for part in path_obj.parts)
            ):
                continue
            try:
                line_no = int(match.group("line"))
            except ValueError:
                continue
            key = (rel_path, line_no)
            if key in seen:
                continue
            seen.add(key)
            source_path = workspace / rel_path
            if not source_path.exists() or not source_path.is_file():
                continue
            try:
                source_lines = source_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).splitlines()
            except OSError:
                continue
            if line_no < 1 or line_no > len(source_lines):
                continue
            start = max(1, line_no - context_lines)
            end = min(len(source_lines), line_no + context_lines)
            excerpt_lines = []
            for current in range(start, end + 1):
                marker = ">" if current == line_no else " "
                excerpt_lines.append(f"{marker} {current}: {source_lines[current - 1]}")
            message = match.group("message").strip()
            header = f"[{rel_path}:{line_no}]"
            if message:
                header += f" {message}"
            block = header + "\n" + "\n".join(excerpt_lines)
            if total_chars + len(block) > max_chars:
                if not blocks:
                    blocks.append(block[:max_chars] + "...")
                break
            blocks.append(block)
            total_chars += len(block) + 2

        return "\n\n".join(blocks)

    @staticmethod
    def _residual_failed_test_ids(
        *,
        anchor_qv: dict[str, Any],
        test_result: dict[str, Any],
    ) -> tuple[list[str], bool]:
        ids = [
            str(test_id).strip()
            for test_id in (
                test_result.get("failed_tests")
                or test_result.get("failed_test_ids")
                or anchor_qv.get("failed_tests")
                or []
            )
            if str(test_id).strip()
        ]
        if not ids:
            return [], True
        expected_failed = None
        for source in (test_result, anchor_qv):
            try:
                value = int(source.get("failed") or 0)
            except (AttributeError, TypeError, ValueError):
                value = 0
            if value > 0:
                expected_failed = value
                break
        if expected_failed is None:
            return ids, True
        return ids, len(ids) == expected_failed

    def _selection_residual_focus_files(
        self,
        *,
        issue_plan: IssuePlan,
        rollout_results: list[RolloutResult],
        best_result: RolloutResult,
    ) -> list[str]:
        verifier_rejection_files = self._verifier_rejection_focus_files(best_result)
        all_stub_residue_files = self._stub_residue_focus_files(best_result)
        stub_residue_files = (
            all_stub_residue_files if self._stub_residue_should_drive_residual(best_result) else []
        )
        suppressed_stub_residue_files = set(all_stub_residue_files) - set(stub_residue_files)
        hard_validity_source_frontier = [
            path
            for path in list(dict.fromkeys(verifier_rejection_files + stub_residue_files))
            if not self._is_non_source_residual_focus_path(path)
        ]
        counts: Counter[str] = Counter()
        for result in rollout_results:
            if not result.success or not result.patch:
                continue
            for path in result.changed_files:
                counts[path] += 1
        ordered = [path for path, _ in counts.most_common(8)]
        boundary_requested_files = self._rollout_boundary_requested_files(rollout_results)
        diagnostics = (
            best_result.selection_diagnostics
            if isinstance(best_result.selection_diagnostics, dict)
            else {}
        )
        critic = diagnostics.get("critic") if isinstance(diagnostics, dict) else {}
        critic_focus_files = critic.get("focus_files") if isinstance(critic, dict) else []
        failure_evidence_files = self._residual_failure_evidence_source_files(best_result)
        invalid_discovery_files = self._failed_rollout_discovery_files(rollout_results)
        residual = list(
            dict.fromkeys(
                verifier_rejection_files
                + stub_residue_files
                + failure_evidence_files
                + list(critic_focus_files or [])
                + best_result.changed_files
                + boundary_requested_files
                + invalid_discovery_files
                + list(issue_plan.test_context.terminal_source_files or [])
                + list(issue_plan.test_context.source_focus_files or [])
                + ordered
                + issue_plan.risk_files
            )
        )
        if suppressed_stub_residue_files:
            residual = [path for path in residual if path not in suppressed_stub_residue_files]
        if hard_validity_source_frontier and self._result_has_repairable_validity_diagnostics(
            best_result
        ):
            residual = [
                path
                for path in residual
                if path in hard_validity_source_frontier
                or not self._is_non_source_residual_focus_path(path)
            ]
        cleaned: list[str] = []
        for raw_path in residual:
            path = _safe_repo_relative_path(_clean_baseline_file_hint_text(raw_path))
            if path and not self._is_non_source_residual_focus_path(path) and path not in cleaned:
                cleaned.append(path)
        return cleaned[:12]

    @classmethod
    def _residual_failure_evidence_source_files(
        cls,
        result: RolloutResult,
    ) -> list[str]:
        quick_verification = (
            result.quick_verification if isinstance(result.quick_verification, dict) else {}
        )
        verification = result.verification if isinstance(result.verification, dict) else {}
        test_result = verification.get("test_result") if isinstance(verification, dict) else {}
        if not isinstance(test_result, dict):
            test_result = {}
        texts: list[str] = []
        for cluster in list(quick_verification.get("failure_clusters") or []):
            if isinstance(cluster, dict):
                for key in ("path", "file", "source_path", "summary", "message", "label"):
                    value = cluster.get(key)
                    if value:
                        texts.append(str(value))
            elif cluster:
                texts.append(str(cluster))
        for key in (
            "output_excerpt",
            "stdout",
            "stderr",
            "failure_summary",
            "regression_output",
            "reproduction_output",
        ):
            value = quick_verification.get(key)
            if value:
                texts.append(str(value))
            value = test_result.get(key)
            if value:
                texts.append(str(value))
        source_paths: list[str] = []
        path_pattern = re.compile(r"(?P<path>(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.py)(?::\d+)?")
        for text in texts:
            for match in path_pattern.finditer(text):
                path = cls._normalize_residual_failure_evidence_path(match.group("path"))
                if path and not cls._is_non_source_residual_focus_path(path):
                    source_paths.append(path)
        return list(dict.fromkeys(source_paths))

    @staticmethod
    def _normalize_residual_failure_evidence_path(path: Any) -> str:
        text = str(path or "").strip().replace("\\", "/").lstrip("/")
        while text.startswith("./"):
            text = text[2:]
        if not text:
            return ""
        parts = [part for part in text.split("/") if part]
        for index, part in enumerate(parts):
            if part == "workspaces":
                if (
                    index + 3 < len(parts)
                    and parts[index + 1] == "_pool"
                    and parts[index + 2].startswith("pool_")
                ):
                    return "/".join(parts[index + 3 :])
                if index + 2 < len(parts) and parts[index + 1].startswith("rollout_"):
                    return "/".join(parts[index + 2 :])
            if part == "_pool" and index + 2 < len(parts) and parts[index + 1].startswith("pool_"):
                return "/".join(parts[index + 2 :])
        return "/".join(parts)

    @staticmethod
    def _stub_residue_should_drive_residual(result: Optional[RolloutResult]) -> bool:
        if result is None:
            return False
        diagnostics = (
            result.selection_diagnostics if isinstance(result.selection_diagnostics, dict) else {}
        )
        stub_residue = diagnostics.get("stub_residue")
        if not isinstance(stub_residue, list) or not stub_residue:
            return False
        if diagnostics.get("stub_residue_advisory"):
            return False
        quick_verification = (
            result.quick_verification if isinstance(result.quick_verification, dict) else {}
        )
        verification = result.verification if isinstance(result.verification, dict) else {}
        test_result = verification.get("test_result") if isinstance(verification, dict) else {}
        if not isinstance(test_result, dict):
            test_result = {}

        def _as_int(value: Any) -> int:
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0

        def _expected_coverage_is_clean(payload: dict[str, Any]) -> bool:
            missing_expected = payload.get("missing_expected_test_count")
            if missing_expected is not None and _as_int(missing_expected) > 0:
                return False
            expected = payload.get("expected_test_count")
            matched = payload.get("matched_expected_test_count")
            if expected is not None and matched is not None:
                return _as_int(expected) > 0 and _as_int(matched) >= _as_int(expected)
            return payload.get("expected_coverage_preserved") is not False

        def _test_result_has_clean_objective_evidence(payload: dict[str, Any]) -> bool:
            if not payload:
                return False
            if _as_int(payload.get("failed")) > 0 or _as_int(payload.get("errors")) > 0:
                return False
            if payload.get("regression_passes") is False:
                return False
            if payload.get("reproduction_passes") is False:
                return False
            if not _expected_coverage_is_clean(payload):
                return False
            pass_rate = payload.get("pass_rate")
            if isinstance(pass_rate, (int, float)) and float(pass_rate) >= 0.999:
                return True
            if payload.get("returncode") == 0:
                return True
            if payload.get("regression_passes") is True:
                return True
            return _as_int(payload.get("passed")) > 0

        def _test_result_reports_residual_failure(payload: dict[str, Any]) -> bool:
            if not payload:
                return False
            if _as_int(payload.get("failed")) > 0 or _as_int(payload.get("errors")) > 0:
                return True
            if payload.get("regression_passes") is False:
                return True
            if payload.get("reproduction_passes") is False:
                return True
            if payload.get("expected_coverage_preserved") is False:
                return True
            return _as_int(payload.get("missing_expected_test_count")) > 0

        if not _test_result_reports_residual_failure(test_result) and (
            quick_verification_has_local_full_scope_pass(quick_verification)
            or quick_verification_has_strong_signal(quick_verification)
        ):
            if _expected_coverage_is_clean(quick_verification):
                return False
        if verification.get("accepted") is True:
            return False
        if _test_result_has_clean_objective_evidence(test_result):
            return False

        failed_tests = [
            str(item)
            for item in (
                quick_verification.get("failed_tests")
                or test_result.get("failed_tests")
                or test_result.get("failed_test_ids")
                or []
            )
        ]
        failed = _as_int(quick_verification.get("failed") or test_result.get("failed"))
        errors = _as_int(quick_verification.get("errors") or test_result.get("errors"))
        if not failed_tests and failed <= 0 and errors <= 0:
            return True
        direct_failure_parts = [
            *failed_tests,
            *[str(item) for item in quick_verification.get("failure_clusters") or []],
            str(quick_verification.get("failure_summary") or ""),
            str(test_result.get("failure_summary") or ""),
            str(test_result.get("regression_output") or ""),
            str(test_result.get("reproduction_output") or ""),
        ]
        broad_failure_parts = [
            str(quick_verification.get("output_excerpt") or ""),
            str(test_result.get("stdout") or ""),
            str(test_result.get("stderr") or ""),
        ]

        def _text_points_at_stub(failure_text: str) -> bool:
            text = failure_text.lower()
            if any(
                marker in text
                for marker in (
                    "notimplementederror",
                    "not implemented",
                    "unimplemented",
                    "todo",
                )
            ):
                return True
            for item in stub_residue:
                if not isinstance(item, dict):
                    continue
                symbol = str(item.get("symbol") or "").strip().lower()
                path = str(item.get("path") or "").strip().replace("\\", "/").lower()
                basename = path.rsplit("/", 1)[-1]
                if symbol and symbol in text:
                    return True
                if basename and basename in text:
                    return True
            return False

        direct_failure_text = "\n".join(part for part in direct_failure_parts if part)
        if _text_points_at_stub(direct_failure_text):
            return True
        if failed_tests:
            # A concrete residual test list is a stronger repair target than
            # broad pytest transcript noise; only direct failure evidence can
            # promote changed-source stubs back into the residual frontier.
            return False
        return _text_points_at_stub(
            "\n".join(
                part
                for part in (
                    *direct_failure_parts,
                    *broad_failure_parts,
                )
                if part
            )
        )

    @staticmethod
    def _is_non_source_residual_focus_path(path: Any) -> bool:
        text = str(path or "").strip().replace("\\", "/")
        if not text:
            return True
        if "::" in text:
            text = text.split("::", 1)[0]
        while text.startswith("./"):
            text = text[2:]
        if not text:
            return True
        parts = tuple(part for part in text.lstrip("/").split("/") if part)
        if "site-packages" in parts or ".venv" in parts:
            return True
        if parts[:3] == ("usr", "local", "lib") or parts[:2] == ("usr", "lib"):
            return True
        if parts[:2] == ("opt", "apex-commit0"):
            return True
        if is_test_path(text, repo_relative=False):
            return True
        if is_apex_harness_path(text, repo_relative=False):
            return True
        category = classify_patch_path(text, evidence_mode="gold_suite_visible")
        return category in {
            PatchPathCategory.APEX_CONTROL_FILE,
            PatchPathCategory.GENERATED_ARTIFACT,
            PatchPathCategory.TEMPORARY_ARTIFACT,
            PatchPathCategory.DEPENDENCY_ARTIFACT,
            PatchPathCategory.GOLD_PROTECTED_TEST,
        }

    @staticmethod
    def _stub_residue_focus_files(result: Optional[RolloutResult]) -> list[str]:
        if result is None:
            return []
        diagnostics = (
            result.selection_diagnostics if isinstance(result.selection_diagnostics, dict) else {}
        )
        stub_residue = diagnostics.get("stub_residue")
        if not isinstance(stub_residue, list):
            return []
        raw_worktree_path = str(result.worktree_path or "").strip()
        if not raw_worktree_path:
            return []
        worktree = Path(raw_worktree_path).resolve(strict=False)
        focus_files: list[str] = []
        for item in stub_residue:
            if not isinstance(item, dict):
                continue
            raw_path = str(item.get("path") or "").strip()
            if not raw_path:
                continue
            rel_path = raw_path.replace("\\", "/")
            candidate_path = Path(raw_path).expanduser()
            if candidate_path.is_absolute() and str(worktree):
                try:
                    rel_path = candidate_path.resolve(strict=False).relative_to(worktree).as_posix()
                except ValueError:
                    continue
            if rel_path and rel_path not in focus_files:
                focus_files.append(rel_path)
        return focus_files[:12]

    @staticmethod
    def _verifier_rejection_focus_files(result: Optional[RolloutResult]) -> list[str]:
        if result is None or not isinstance(result.verification, dict):
            return []
        verification = result.verification
        focus_files: list[str] = []
        lint_output = str(verification.get("lint_output") or "")
        if verification.get("lint_clean") is False and lint_output.strip():
            for line in lint_output.splitlines():
                match = re.match(r"^(?P<path>[^:\n]+):\d+(?::\d+)?:", line.strip())
                if not match:
                    continue
                rel_path = match.group("path").strip()
                if rel_path.startswith("./"):
                    rel_path = rel_path[2:]
                if rel_path and rel_path not in focus_files:
                    focus_files.append(rel_path)
        prune_result = verification.get("prune_result")
        if isinstance(prune_result, dict):
            for raw in list(prune_result.get("regressed_tests") or []):
                path = str(raw or "").split("::", 1)[0].strip()
                if path and path not in focus_files:
                    focus_files.append(path)
        test_result = verification.get("test_result")
        if (
            isinstance(test_result, dict)
            and test_result.get("expected_coverage_preserved") is False
        ):
            for raw in list(test_result.get("missing_expected_test_ids") or [])[:8]:
                path = str(raw or "").split("::", 1)[0].strip()
                if path and path not in focus_files:
                    focus_files.append(path)
        return focus_files[:8]

    def _rollout_boundary_requested_files(
        self,
        rollout_results: list[RolloutResult],
    ) -> list[str]:
        """Return files requested by delegated workers as boundary expansions.

        Delegated workers are intentionally prevented from editing outside
        their owned slice. When multiple workers report adjacent files as
        necessary, that is a planning signal for the next parent-controlled
        scope, not permission for the child patch to bypass the sanitizer.
        """

        requested: list[str] = []
        for result in rollout_results:
            payload = (
                result.multi_agent_summary if isinstance(result.multi_agent_summary, dict) else {}
            )
            for raw in list(payload.get("boundary_requested_files") or []):
                text = _clean_baseline_file_hint_text(raw)
                if text and text not in requested:
                    requested.append(text)
        return requested

    def _rollout_candidate_file_hints(
        self,
        result: RolloutResult,
        *,
        include_patch_footprint: bool = True,
    ) -> list[str]:
        """Return clean solution-context file hints contributed by a rollout.

        These hints are evidence for the next parent-owned plan only. They do
        not make an invalid rollout selectable and they do not bypass patch
        sanitizer or validity checks.
        """

        raw_paths: list[Any] = list(result.changed_files or []) if include_patch_footprint else []
        for container in (
            result.patch_artifact if isinstance(result.patch_artifact, dict) else {},
            result.search_metadata if isinstance(result.search_metadata, dict) else {},
        ):
            for key in (
                "changed_files",
                "solution_files",
                "partial_discovery_files",
                "out_of_scope_changed_files",
            ):
                values = container.get(key)
                if isinstance(values, (list, tuple, set)):
                    if not include_patch_footprint and key in {
                        "changed_files",
                        "out_of_scope_changed_files",
                        "solution_files",
                    }:
                        continue
                    raw_paths.extend(values)
        hints: list[str] = []
        for raw_path in filter_solution_paths(raw_paths, evidence_mode=""):
            text = _clean_baseline_file_hint_text(raw_path)
            path = _safe_repo_relative_path(text)
            if path and path not in hints:
                hints.append(path)
        return hints

    def _failed_rollout_discovery_files(
        self,
        rollout_results: list[RolloutResult],
        *,
        min_recurrence: Optional[int] = None,
        include_singletons: bool = False,
        suppress_broad_invalid_patch_footprints: bool = False,
        blocker_kind: str = "",
    ) -> list[str]:
        """Merge useful file discoveries from invalid rollouts.

        A failed rollout's patch is still invalid. Repeatedly touched clean
        solution-context files, however, are useful evidence that the parent
        planner can use to widen or redirect the next rollout's search scope.
        """

        threshold = (
            1
            if include_singletons
            else max(
                1,
                int(min_recurrence or self._STRUCTURAL_RECOVERY_MIN_RECURRENCE),
            )
        )
        counts: Counter[str] = Counter()
        first_seen: dict[str, int] = {}
        for result in rollout_results:
            if result is None or result.success:
                continue
            include_patch_footprint = not (
                suppress_broad_invalid_patch_footprints
                and self._rollout_has_broad_invalid_patch_footprint(
                    result,
                    blocker_kind=blocker_kind,
                )
            )
            per_rollout = list(
                dict.fromkeys(
                    self._rollout_candidate_file_hints(
                        result,
                        include_patch_footprint=include_patch_footprint,
                    )
                )
            )
            for path in per_rollout:
                counts[path] += 1
                first_seen.setdefault(path, len(first_seen))
        ordered = sorted(
            (path for path, count in counts.items() if count >= threshold),
            key=lambda path: (-counts[path], first_seen[path], path),
        )
        return ordered[:8]

    # Recovery prompts carry at most a small focus set; larger structurally
    # invalid patch footprints are negative evidence, not implementation scope.
    _STRUCTURAL_RECOVERY_FOCUS_FILE_CAP = 8
    _STRUCTURAL_RECOVERY_BROAD_PATCH_FILE_COUNT = _STRUCTURAL_RECOVERY_FOCUS_FILE_CAP + 1
    _STRUCTURAL_RECOVERY_PATCH_FOOTPRINT_BLOCKER_KINDS = {
        "abstract_class",
        "api_contract",
        "collection_broken",
        "conftest_import",
        "module_import",
        "syntax",
        "test_collection",
        "test_selection",
        "timeout",
        "unimplemented",
        "verification_blocker",
    }

    def _rollout_has_broad_invalid_patch_footprint(
        self,
        result: RolloutResult,
        *,
        blocker_kind: str = "",
    ) -> bool:
        """Return true when an invalid rollout's own patch footprint is too broad.

        This is negative recovery evidence. It must not become the next
        clean-baseline rollout's required implementation scope, because the
        prior patch already proved that broad edit pattern breaks collection.
        """

        if result is None or result.success:
            return False
        kind = str(blocker_kind or self._rollout_structural_blocker_kind(result)).strip()
        if kind not in self._STRUCTURAL_RECOVERY_PATCH_FOOTPRINT_BLOCKER_KINDS:
            return False
        patch_footprint = list(
            dict.fromkeys(filter_solution_paths(list(result.changed_files or []), evidence_mode=""))
        )
        return len(patch_footprint) >= self._STRUCTURAL_RECOVERY_BROAD_PATCH_FILE_COUNT

    def _failed_rollout_broad_invalid_patch_files(
        self,
        rollout_results: list[RolloutResult],
        *,
        blocker_kind: str = "",
    ) -> list[str]:
        counts: Counter[str] = Counter()
        first_seen: dict[str, int] = {}
        for result in rollout_results:
            if not self._rollout_has_broad_invalid_patch_footprint(
                result,
                blocker_kind=blocker_kind,
            ):
                continue
            for path in filter_solution_paths(list(result.changed_files or []), evidence_mode=""):
                text = _clean_baseline_file_hint_text(path)
                clean_path = _safe_repo_relative_path(text)
                if not clean_path:
                    continue
                counts[clean_path] += 1
                first_seen.setdefault(clean_path, len(first_seen))
        ordered = sorted(
            counts,
            key=lambda path: (-counts[path], first_seen[path], path),
        )
        return ordered[:12]

    def _failed_rollouts_with_repeated_discovery_files(
        self,
        rollout_results: list[RolloutResult],
    ) -> list[RolloutResult]:
        repeated = set(self._failed_rollout_discovery_files(rollout_results))
        if not repeated:
            return []
        affected: list[RolloutResult] = []
        for result in rollout_results:
            if result is None or result.success:
                continue
            if repeated.intersection(self._rollout_candidate_file_hints(result)):
                affected.append(result)
        return affected

    def _rollout_structural_blocker_kind(self, result: RolloutResult) -> str:
        qv = result.quick_verification if isinstance(result.quick_verification, dict) else {}
        kind = _quick_verification_structural_blocker_kind(qv)
        if kind:
            return kind
        reason = str(result.failure_reason or "").strip()
        if not reason:
            return ""
        lowered = reason.lower()
        if "importerror while loading conftest" in lowered:
            return "conftest_import"
        if "importerror while importing test module" in lowered or "error collecting " in lowered:
            return "test_collection"
        if "no tests ran" in lowered:
            return "test_selection"
        if "syntaxerror:" in lowered or "indentationerror:" in lowered:
            return "syntax"
        if (" in __init__" in lowered or " in __new__" in lowered) and any(
            token in lowered
            for token in (
                "nameerror:",
                "attributeerror:",
                "typeerror:",
                "importerror:",
                "valueerror:",
            )
        ):
            return "api_contract"
        if (
            "can't instantiate abstract class" in lowered
            or "without an implementation for abstract method" in lowered
        ):
            return "abstract_class"
        if "notimplementederror" in lowered or "need to implement" in lowered:
            return "unimplemented"
        if "timed out" in lowered or "timeout" in lowered:
            return "timeout"
        if "quick verification failed" in lowered or "verification failed" in lowered:
            return "verification_blocker"
        return ""

    def _save_trajectories(self, rollout_results: list[RolloutResult]) -> None:
        trajectories_dir = Path(self.config.output_dir) / "trajectories"
        trajectories_dir.mkdir(parents=True, exist_ok=True)
        for result in rollout_results:
            if not result.trajectory:
                continue
            path = trajectories_dir / f"rollout_{result.rollout_id}.json"
            from ..evaluation.checkpointing import atomic_write_json

            atomic_write_json(
                path,
                {
                    "rollout_id": result.rollout_id,
                    "model": result.llm_model,
                    "rollout_profile_index": result.rollout_profile_index,
                    "rollout_profile_signature": list(result.rollout_profile_signature),
                    "stage_model_routing": dict(result.stage_model_routing),
                    "trajectory": result.trajectory,
                },
            )

    def _repo_stats(self, repo_context: RepoContext) -> dict[str, Any]:
        return {
            "files": len(repo_context.files),
            "symbols": sum(len(file_info.symbols) for file_info in repo_context.files),
        }

    def _serialize_rollout_summaries(
        self,
        rollout_results: list[RolloutResult],
    ) -> list[dict[str, Any]]:
        summaries = []
        for rollout in rollout_results:
            payload = rollout.to_dict()
            payload["trajectory"] = copy.deepcopy(rollout.trajectory)
            if not self.config.rollout.keep_worktrees:
                payload["worktree_path"] = None
            summaries.append(payload)
        return summaries

    def _external_scoring_candidates(
        self,
        rollout_results: list[RolloutResult],
    ) -> list[dict[str, Any]]:
        """APEX-owned candidate manifest for benchmark-side scoring.

        Benchmarks own the hidden evaluator, but they should not infer
        APEX's workspace layout or acceptance internals. This manifest is
        the stable handoff: every patched rollout with a retained worktree is
        listed whether or not APEX internally accepted it.
        """

        if not self.config.rollout.keep_worktrees:
            retained = {
                result.rollout_id
                for result in rollout_results
                if bool(getattr(result, "is_synthetic", False))
            }
        else:
            retained = {result.rollout_id for result in rollout_results}

        candidates: list[dict[str, Any]] = []
        for result in rollout_results:
            patch = result.patch
            if not isinstance(patch, str) or not patch.strip():
                continue
            if result.rollout_id not in retained:
                continue
            worktree_path = str(result.worktree_path or "").strip()
            if not worktree_path:
                continue
            scoring_decision = self._score_bearing_decision_payload(result)
            scoring_success = self._result_has_score_bearing_success(result)
            validity_obj = result.validity
            validity = validity_obj.as_dict() if validity_obj is not None else None
            diagnostic_score_only = self._external_scoring_candidate_is_diagnostic_only(
                result,
                validity,
            )
            candidates.append(
                {
                    "rollout_id": result.rollout_id,
                    "worktree_path": worktree_path,
                    "patch": patch,
                    "changed_files": list(result.changed_files),
                    "success": bool(result.success or scoring_success),
                    "rollout_success": bool(result.success),
                    "contract_success": scoring_success,
                    "diagnostic_score_only": diagnostic_score_only,
                    "evaluation_decision": copy.deepcopy(scoring_decision),
                    "selected_for_submission": bool(result.selected_for_submission),
                    "internally_accepted": bool(result.internally_accepted),
                    "officially_accepted": result.officially_accepted,
                    "salvaged_for_external_scoring": bool(result.salvaged_for_external_scoring),
                    "verification": copy.deepcopy(result.verification),
                    "quick_verification": copy.deepcopy(result.quick_verification),
                    "validity": copy.deepcopy(validity),
                    "failure_reason": result.failure_reason,
                    "llm_model": result.llm_model,
                    "agent_mode": result.agent_mode,
                    "rollout_profile_index": result.rollout_profile_index,
                    "rollout_profile_signature": list(result.rollout_profile_signature),
                    "stage_model_routing": dict(result.stage_model_routing),
                }
            )
        return candidates

    def _external_scoring_candidate_is_diagnostic_only(
        self,
        result: RolloutResult,
        validity: Optional[dict[str, Any]],
    ) -> bool:
        if not isinstance(validity, dict):
            return False
        if validity.get("eligible_for_submission") is not False:
            return False
        if validity.get("eligible_for_external_scoring") is False:
            return True
        return not rollout_requires_authoritative_scoring(result)

    def _build_orchestration_transition(
        self,
        *,
        current_plan: IssuePlan,
        next_strategy: Any,
        attempt_results: list[RolloutResult],
        transition_index: int,
    ) -> dict[str, Any]:
        candidate_rollouts = [
            result for result in attempt_results if result.success and result.patch
        ]
        accepted_candidate_rollouts = [
            result
            for result in candidate_rollouts
            if self._selected_result_is_accepted(result)
            or self._rollout_has_strong_progressive_signal(result)
        ]
        if accepted_candidate_rollouts:
            reason = (
                "Candidate patches were produced with strong local completion signals, "
                "but none crossed the final acceptance bar."
            )
            trigger = "local_completion_unaccepted"
        elif candidate_rollouts:
            reason = (
                "Candidate patches were produced, but none showed an acceptance-strength signal."
            )
            trigger = "no_accepted_rollouts"
        else:
            reason = "No successful rollout produced a candidate patch."
            trigger = "no_successful_rollouts"
        return {
            "attempt": transition_index,
            "reason": reason,
            "trigger": trigger,
            "from_primitives": list(current_plan.orchestration_primitives),
            "to_primitives": [primitive.value for primitive in next_strategy.primitives],
            "from_rollouts": len(current_plan.rollout_briefs),
            "to_rollouts": next_strategy.rollout_count,
            "failed_rollouts": sum(1 for result in attempt_results if not result.success),
            "successful_rollouts": sum(1 for result in attempt_results if result.success),
            "candidate_rollouts": len(candidate_rollouts),
            "accepted_candidate_rollouts": len(accepted_candidate_rollouts),
        }

    def _build_progressive_wave_transition(
        self,
        *,
        planner: IssuePlanner,
        current_plan: IssuePlan,
        next_plan: IssuePlan,
        rollout_results: list[RolloutResult],
        wave_index: int,
        remaining_budget: int,
    ) -> dict[str, Any]:
        ranked = sorted(
            rollout_results,
            key=planner.score_rollout_progress,
            reverse=True,
        )
        top_results = ranked[:3]
        best_score = planner.score_rollout_progress(top_results[0]) if top_results else 0.0
        return {
            "attempt": wave_index,
            "reason": "Promising partial signal was detected; reallocating remaining rollout budget.",
            "trigger": "progressive_wave",
            "wave": wave_index,
            "remaining_budget": remaining_budget,
            "from_rollouts": len(current_plan.rollout_briefs),
            "to_rollouts": len(next_plan.rollout_briefs),
            "best_progress_score": best_score,
            "top_rollout_ids": [result.rollout_id for result in top_results],
            "top_plan_titles": [result.plan_title for result in top_results if result.plan_title],
            "focus_files": list(next_plan.relevant_files[:4]),
        }

    def _build_selection_followup_transition(
        self,
        *,
        current_plan: IssuePlan,
        followup_plan: IssuePlan,
        best_result: RolloutResult,
        followup_round: int,
    ) -> dict[str, Any]:
        verification = (
            best_result.verification if isinstance(best_result.verification, dict) else {}
        )
        return {
            "attempt": followup_round,
            "reason": "Selected candidate was not accepted; launching a residual follow-up search.",
            "trigger": "selection_unaccepted",
            "followup_round": followup_round,
            "from_primitives": list(current_plan.orchestration_primitives),
            "to_primitives": list(followup_plan.orchestration_primitives),
            "from_rollouts": len(current_plan.rollout_briefs),
            "to_rollouts": len(followup_plan.rollout_briefs),
            "best_rollout_id": best_result.rollout_id,
            "best_changed_files": list(best_result.changed_files),
            "best_verification_score": verification.get("overall_score"),
            "best_accepted": verification.get("accepted"),
            "best_selection_critic_score": (
                (best_result.selection_diagnostics.get("critic") or {}).get("score")
                if isinstance(best_result.selection_diagnostics, dict)
                else None
            ),
        }

    # Phase 3.2: ``_MAX_COVERAGE_GAP_FOLLOWUP_ROUNDS`` was a class
    # attribute. It now lives on
    # ``OrchestrationConfig.max_coverage_gap_followup_rounds`` (default 2).
    _STRUCTURAL_RECOVERY_MIN_RECURRENCE = 2
    """Minimum number of failed rollouts that must share the same
    structural blocker kind (conftest_import, syntax, module_import,
    test_collection) before the structural-recovery followup fires.
    Single-rollout structural failures are noisy — the regular
    in-rollout repair loop usually handles them. Recurring breaks
    across rollouts indicate a systematic agent failure mode that
    needs an explicit recovery prompt."""

    def _select_near_miss_anchor(
        self,
        rollout_results: list[RolloutResult],
    ) -> Optional[RolloutResult]:
        """Pick the highest-signal non-accepted rollout that is branchable.

        A near-miss anchor must:
          * carry a patch or materialized changed workspace and still fail
            authoritative acceptance (for example: residual failures, timeout
            fallback, or expected-coverage collapse),
          * have a non-None worktree path that still exists on disk
            (so the followup can clone from it),
          * be convertible to a ``WorkspaceSeed`` either through an execution
            checkpoint or a materialized workspace patch,
          * have a preserved full-suite quick-verification signal whose
            residual failure count is tiny enough to repair surgically.
        """

        candidates: list[tuple[float, RolloutResult]] = []
        if any(
            self._rollout_has_preemptive_completion_signal(result) for result in rollout_results
        ):
            return None
        for result in rollout_results:
            if result is None or not self._rollout_has_materialized_repair_seed(result):
                continue
            if self._selected_result_is_accepted(result):
                continue
            if not self._rollout_has_repairable_near_miss_signal(result):
                continue
            quick_verification = (
                result.quick_verification if isinstance(result.quick_verification, dict) else {}
            )
            signal_score = quick_verification_signal_score(quick_verification)
            if not isinstance(signal_score, (int, float)):
                continue
            worktree_path = result.worktree_path
            if not worktree_path:
                continue
            if not Path(worktree_path).exists():
                continue
            if build_workspace_seed_from_rollout_result(result) is None:
                continue
            candidates.append((float(signal_score), result))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], -item[1].rollout_id), reverse=True)
        return candidates[0][1]

    def _select_coverage_gap_anchor(
        self,
        rollout_results: list[RolloutResult],
    ) -> Optional[RolloutResult]:
        """Pick the strongest branchable rollout with a local full-suite win but
        benchmark-level expected-coverage shrinkage.

        This recovery path is intentionally separate from the normal
        selection followup logic. A local full-suite pass is strong
        progress, so `_selected_result_needs_followup` stays false to
        avoid burning budget forever after a green local run. But when
        selection returns `None` because *every* locally-green rollout
        collapsed expected coverage, we still want one bounded retry
        seeded from the best existing workspace.
        """

        candidates: list[tuple[float, int, float, int, RolloutResult]] = []
        for result in rollout_results:
            if result is None or not self._rollout_has_materialized_repair_seed(result):
                continue
            if self._selected_result_is_accepted(result):
                continue
            if not self._rollout_has_local_full_suite_completion_signal(result):
                continue
            if not self._rollout_has_expected_coverage_gap(result):
                continue
            worktree_path = getattr(result, "worktree_path", None)
            if not worktree_path or not Path(worktree_path).exists():
                continue
            if build_workspace_seed_from_rollout_result(result) is None:
                continue
            quick_verification = (
                result.quick_verification if isinstance(result.quick_verification, dict) else {}
            )
            expected_coverage_ratio = quick_verification_expected_coverage_ratio(quick_verification)
            if not isinstance(expected_coverage_ratio, (int, float)):
                continue
            signal_score = quick_verification_signal_score(quick_verification)
            missing_expected_test_count = quick_verification.get("missing_expected_test_count")
            normalized_missing_expected = (
                int(missing_expected_test_count)
                if isinstance(missing_expected_test_count, int) and missing_expected_test_count >= 0
                else 10**9
            )
            candidates.append(
                (
                    float(expected_coverage_ratio)
                    if isinstance(expected_coverage_ratio, (int, float))
                    else (float(signal_score) if isinstance(signal_score, (int, float)) else 0.0),
                    -normalized_missing_expected,
                    float(signal_score) if isinstance(signal_score, (int, float)) else 0.0,
                    -int(result.rollout_id),
                    result,
                )
            )
        if not candidates:
            return None
        # Key on the numeric prefix only (never the trailing RolloutResult): the
        # -rollout_id tiebreak repeats across residual rounds, so a whole-tuple
        # sort can tie through the prefix and crash comparing RolloutResult
        # objects (no __lt__). Same defect class as the invalid-selection anchor.
        candidates.sort(key=lambda candidate_entry: candidate_entry[:-1], reverse=True)
        return candidates[0][-1]

    def _launch_coverage_gap_followup(
        self,
        *,
        repo_path: str,
        repo_context: RepoContext,
        issue_description: str,
        issue_plan: IssuePlan,
        planner: IssuePlanner,
        test_command: Optional[str],
        engine: RolloutEngine,
        rollout_results: list[RolloutResult],
        task_state_graph: Optional[TaskStateGraph],
        followup_round: int = 1,
        wallclock_deadline: Optional[float] = None,
    ) -> tuple[list[RolloutResult], list[dict[str, Any]], Optional[dict[str, Any]]]:
        """Launch one bounded retry from the strongest local full-suite rollout
        whose expected benchmark coverage still shrank."""

        anchor = self._select_coverage_gap_anchor(rollout_results)
        if anchor is None:
            return [], [], None

        additional_rollouts = 1
        anchor_quick_verification = (
            anchor.quick_verification if isinstance(anchor.quick_verification, dict) else {}
        )
        anchor_signal_score = quick_verification_signal_score(anchor_quick_verification)
        anchor_expected_coverage_ratio = quick_verification_expected_coverage_ratio(
            anchor_quick_verification
        )
        anchor_missing_expected_test_count = anchor_quick_verification.get(
            "missing_expected_test_count"
        )
        editable_test_files = self._editable_test_files_for_followup(issue_plan)
        residual_summary = self._build_near_miss_residual_summary(
            anchor=anchor,
            anchor_quick_verification=anchor_quick_verification,
            editable_test_files=editable_test_files,
        )
        coverage_gap_intro = (
            "The prior rollout passed its local full-suite execution, but the "
            "benchmark-level expected test coverage still shrank."
        )
        if isinstance(anchor_expected_coverage_ratio, (int, float)):
            coverage_gap_intro += (
                f" Expected coverage ratio was {float(anchor_expected_coverage_ratio):.4f}."
            )
        if (
            isinstance(anchor_missing_expected_test_count, int)
            and anchor_missing_expected_test_count > 0
        ):
            coverage_gap_intro += (
                f" {int(anchor_missing_expected_test_count)} expected tests were still missing."
            )
        coverage_gap_intro += (
            " Treat this as a discovery or coverage gap rooted in the current "
            "workspace, and preserve the existing edits while repairing the "
            "missing coverage."
        )
        residual_summary = f"{coverage_gap_intro}\n\n{residual_summary}"
        residual_focus_files = self._selection_residual_focus_files(
            issue_plan=issue_plan,
            rollout_results=rollout_results,
            best_result=anchor,
        )
        if editable_test_files:
            residual_focus_files = list(
                dict.fromkeys(list(editable_test_files) + list(residual_focus_files))
            )
        followup_plan = planner.build_followup_plan(
            issue_plan,
            repo_context,
            rollout_results,
            additional_rollouts=additional_rollouts,
            residual_summary=residual_summary,
            residual_focus_files=residual_focus_files,
            task_state_context=issue_plan.task_state_context,
        )
        if not followup_plan.rollout_briefs:
            return [], [], None

        alt_profile = self._alternative_llm_profile_index(rollout_results)
        if alt_profile is not None:
            self._apply_profile_index_override_to_briefs(followup_plan, alt_profile)
            logger.info(
                "Coverage-gap followup pinned to LLM profile index %s "
                "(prior rollouts used a different model).",
                alt_profile,
            )
        anchor_seed = build_workspace_seed_from_rollout_result(anchor)
        if anchor_seed is None:
            logger.info(
                "Coverage-gap followup skipped: anchor rollout %s has no branchable workspace seed.",
                anchor.rollout_id,
            )
            return [], [], None
        self._save_issue_plan(followup_plan)
        transition_reason = (
            "No rollout was selectable under the acceptance policy, but rollout "
            f"{anchor.rollout_id} passed the local full suite while expected "
            "benchmark coverage still shrank; launching one bounded followup "
            "seeded from that workspace."
        )
        transition = {
            "attempt": followup_round,
            "reason": transition_reason,
            "trigger": "coverage_gap_followup",
            "followup_round": followup_round,
            "from_primitives": list(issue_plan.orchestration_primitives),
            "to_primitives": list(followup_plan.orchestration_primitives),
            "from_rollouts": len(issue_plan.rollout_briefs),
            "to_rollouts": len(followup_plan.rollout_briefs),
            "anchor_rollout_id": anchor.rollout_id,
            "anchor_signal_score": (
                float(anchor_signal_score)
                if isinstance(anchor_signal_score, (int, float))
                else None
            ),
            "anchor_expected_coverage_ratio": (
                float(anchor_expected_coverage_ratio)
                if isinstance(anchor_expected_coverage_ratio, (int, float))
                else None
            ),
            "anchor_missing_expected_test_count": anchor_missing_expected_test_count,
            "anchor_failed": anchor_quick_verification.get("failed"),
            "anchor_errors": anchor_quick_verification.get("errors"),
        }
        logger.info(
            (
                "Launching coverage-gap followup: anchor=rollout_%s "
                "signal=%s expected_coverage=%s missing_expected=%s"
            ),
            anchor.rollout_id,
            (
                f"{float(anchor_signal_score):.4f}"
                if isinstance(anchor_signal_score, (int, float))
                else None
            ),
            (
                f"{float(anchor_expected_coverage_ratio):.4f}"
                if isinstance(anchor_expected_coverage_ratio, (int, float))
                else None
            ),
            anchor_missing_expected_test_count,
        )
        try:
            (
                extra_results,
                _refreshed_plan,
                _next_rollout_id,
                progressive_transitions,
                search_summary,
            ) = self._execute_progressive_rollout_plan(
                repo_path=repo_path,
                repo_context=repo_context,
                issue_description=issue_description,
                issue_plan=followup_plan,
                planner=planner,
                test_command=test_command,
                engine=engine,
                rollout_id_offset=self._next_rollout_id_after(rollout_results),
                task_state_graph=task_state_graph,
                initial_workspace_seed=anchor_seed,
                wallclock_deadline=wallclock_deadline,
            )
        except Exception as exc:  # noqa: BLE001 - best-effort followup
            logger.warning("Coverage-gap followup raised %s; skipping.", exc)
            return [], [transition], None
        transitions = [transition] + list(progressive_transitions)
        return list(extra_results), transitions, search_summary

    def _launch_near_miss_followup(
        self,
        *,
        repo_path: str,
        repo_context: RepoContext,
        issue_description: str,
        issue_plan: IssuePlan,
        planner: IssuePlanner,
        test_command: Optional[str],
        engine: RolloutEngine,
        rollout_results: list[RolloutResult],
        task_state_graph: Optional[TaskStateGraph],
        wallclock_deadline: Optional[float] = None,
    ) -> tuple[list[RolloutResult], list[dict[str, Any]]]:
        """One follow-up rollout seeded from the highest near-miss patch.

        Returns ``(extra_results, transitions)``. Both empty when no
        near-miss anchor exists or the planner refuses to build a
        followup plan.
        """

        anchor = self._select_near_miss_anchor(rollout_results)
        if anchor is None:
            return [], []
        # One extra rollout, ignoring the planner's recommendation
        # (which is often 0 because the best candidate still lacks an
        # acceptance-strength signal due to residual failures, timeout
        # fallback, or expected-coverage collapse).
        additional_rollouts = 1
        anchor_quick_verification = (
            anchor.quick_verification if isinstance(anchor.quick_verification, dict) else {}
        )
        anchor_signal_score = quick_verification_signal_score(anchor_quick_verification)
        anchor_observed_pass_rate = anchor_quick_verification.get("pass_rate")
        anchor_expected_coverage_ratio = quick_verification_expected_coverage_ratio(
            anchor_quick_verification
        )
        display_pass_rate = (
            float(anchor_signal_score)
            if isinstance(anchor_signal_score, (int, float))
            else float(anchor_observed_pass_rate or 0.0)
        )
        editable_test_files = self._editable_test_files_for_followup(issue_plan)
        residual_summary = self._build_near_miss_residual_summary(
            anchor=anchor,
            anchor_quick_verification=anchor_quick_verification,
            editable_test_files=editable_test_files,
        )
        residual_focus_files = self._selection_residual_focus_files(
            issue_plan=issue_plan,
            rollout_results=rollout_results,
            best_result=anchor,
        )
        # Pull editable test files to the FRONT of the focus list so the
        # followup brief surfaces them. Without this, on completion tasks,
        # ``incomplete_test_files`` (which the
        # policy explicitly allows the agent to edit) get pushed out by
        # the brief's per-list 8-file cap and the agent never sees the
        # editable stubs in the focus_files section of its prompt.
        if editable_test_files:
            existing = list(residual_focus_files)
            preserved_editable = [path for path in editable_test_files if path]
            residual_focus_files = list(dict.fromkeys(preserved_editable + existing))
        followup_plan = planner.build_followup_plan(
            issue_plan,
            repo_context,
            rollout_results,
            additional_rollouts=additional_rollouts,
            residual_summary=residual_summary,
            residual_focus_files=residual_focus_files,
            task_state_context=issue_plan.task_state_context,
        )
        if not followup_plan.rollout_briefs:
            return [], []
        self._prepare_near_miss_followup_plan_for_direct_repair(followup_plan)
        # Cross-model diversity: when the configured llm_profiles offer
        # an LLM the prior failed rollouts did NOT use, pin this
        # followup to it. Same model + same prompt usually reproduces
        # the same wrong patch; rotating to a different model is the
        # cheapest meaningful diversity lever.
        alt_profile = self._alternative_llm_profile_index(rollout_results)
        if alt_profile is not None:
            self._apply_profile_index_override_to_briefs(followup_plan, alt_profile)
            logger.info(
                "Near-miss followup pinned to LLM profile index %s "
                "(prior rollouts used a different model).",
                alt_profile,
            )
        anchor_seed = build_workspace_seed_from_rollout_result(anchor)
        if anchor_seed is None:
            logger.info(
                "Near-miss followup skipped: anchor rollout %s has no branchable workspace seed.",
                anchor.rollout_id,
            )
            return [], []
        self._save_issue_plan(followup_plan)
        transition_reason = (
            "No rollout produced an accepted patch, but rollout "
            f"{anchor.rollout_id} reached {display_pass_rate:.4f} quick-verification score"
        )
        if (
            isinstance(anchor_observed_pass_rate, (int, float))
            and abs(float(anchor_observed_pass_rate) - display_pass_rate) >= 1e-6
        ):
            transition_reason += f" (observed pass_rate={float(anchor_observed_pass_rate):.4f}"
            if isinstance(anchor_expected_coverage_ratio, (int, float)):
                transition_reason += (
                    f", expected_coverage={float(anchor_expected_coverage_ratio):.4f}"
                )
            transition_reason += ")"
        transition_reason += "; launching one near-miss followup seeded from it."
        transition = {
            "attempt": 1,
            "reason": transition_reason,
            "trigger": "near_miss_followup",
            "followup_round": 1,
            "from_primitives": list(issue_plan.orchestration_primitives),
            "to_primitives": list(followup_plan.orchestration_primitives),
            "from_rollouts": len(issue_plan.rollout_briefs),
            "to_rollouts": len(followup_plan.rollout_briefs),
            "anchor_rollout_id": anchor.rollout_id,
            "anchor_pass_rate": display_pass_rate,
            "anchor_signal_score": (
                float(anchor_signal_score)
                if isinstance(anchor_signal_score, (int, float))
                else None
            ),
            "anchor_observed_pass_rate": (
                float(anchor_observed_pass_rate)
                if isinstance(anchor_observed_pass_rate, (int, float))
                else None
            ),
            "anchor_expected_coverage_ratio": (
                float(anchor_expected_coverage_ratio)
                if isinstance(anchor_expected_coverage_ratio, (int, float))
                else None
            ),
            "anchor_missing_expected_test_count": anchor_quick_verification.get(
                "missing_expected_test_count"
            ),
            "anchor_failed": anchor_quick_verification.get("failed"),
            "anchor_errors": anchor_quick_verification.get("errors"),
        }
        logger.info(
            (
                "Launching near-miss followup: anchor=rollout_%s signal=%.4f "
                "observed_pass_rate=%s expected_coverage=%s failed=%s errors=%s "
                "missing_expected=%s"
            ),
            anchor.rollout_id,
            display_pass_rate,
            (
                f"{float(anchor_observed_pass_rate):.4f}"
                if isinstance(anchor_observed_pass_rate, (int, float))
                else None
            ),
            (
                f"{float(anchor_expected_coverage_ratio):.4f}"
                if isinstance(anchor_expected_coverage_ratio, (int, float))
                else None
            ),
            anchor_quick_verification.get("failed"),
            anchor_quick_verification.get("errors"),
            anchor_quick_verification.get("missing_expected_test_count"),
        )
        try:
            (
                extra_results,
                refreshed_plan,
                _next_rollout_id,
                progressive_transitions,
                _search_summary,
            ) = self._execute_progressive_rollout_plan(
                repo_path=repo_path,
                repo_context=repo_context,
                issue_description=issue_description,
                issue_plan=followup_plan,
                planner=planner,
                test_command=test_command,
                engine=engine,
                rollout_id_offset=self._next_rollout_id_after(rollout_results),
                task_state_graph=task_state_graph,
                initial_workspace_seed=anchor_seed,
                wallclock_deadline=wallclock_deadline,
            )
        except Exception as exc:  # noqa: BLE001 - best-effort followup
            logger.warning("Near-miss followup raised %s; skipping.", exc)
            return [], [transition]
        followup_plan = refreshed_plan
        transitions = [transition] + list(progressive_transitions)
        return list(extra_results), transitions

    @staticmethod
    def _prepare_near_miss_followup_plan_for_direct_repair(
        followup_plan: IssuePlan,
    ) -> None:
        """Make near-miss followups repair the seeded workspace directly."""

        for brief in list(followup_plan.rollout_briefs or []):
            policy = dict(brief.search_policy or {})
            policy.update(
                {
                    "origin": "near_miss_residual_repair",
                    "verification_focus": str(
                        policy.get("verification_focus") or "gold_expected_suite"
                    ),
                    "direct_workspace_seed_repair": True,
                    "skip_frontier_search": True,
                    "disable_strategy_prefix": True,
                    "cli_agent_use_masai_preround": "off",
                }
            )
            brief.search_policy = policy
            if not str(brief.title or "").lower().startswith("follow-up:"):
                brief.title = f"Follow-up: {brief.title}"

    _NEAR_MISS_TEST_SOURCE_MAX_FILES = 4
    _NEAR_MISS_TEST_SOURCE_MAX_CHARS = 1200
    _NEAR_MISS_MARKER_MAX_HITS = 8
    _MISSING_EXPECTED_IDS_PROMPT_LIMIT = 50

    def _alternative_llm_profile_index(
        self,
        used_rollouts: list[RolloutResult],
    ) -> Optional[int]:
        """Pick a profile index that resolves to a DIFFERENT llm_configs slot
        than any of the prior failed rollouts.

        When multiple LLM backends are configured (e.g. claude + codex +
        gemini in benchmark_commit0_max), rerunning the same model on a
        followup typically reproduces the same wrong patch. Routing the
        followup through an unused llm_configs slot is the cheapest
        cross-rollout diversity lever available — no prompt or scaffold
        change required, just pick another model. When only ONE
        llm_configs slot exists (or no llm_profiles are configured), this
        is a no-op and returns None.
        """

        # Audit H12: log the exception type/message instead of bare-passing
        # so a misconfigured ``llm_configs`` / ``llm_profiles`` doesn't
        # silently disable follow-up LLM diversity selection.
        try:
            llm_configs = list(self.config.llm_configs or [])
        except Exception as exc:
            logger.warning(
                "alternative_llm_profile_index: llm_configs unavailable "
                "(%s: %s); skipping diversity selection",
                type(exc).__name__,
                exc,
            )
            return None
        if len(llm_configs) <= 1:
            return None
        try:
            profiles = list(self.config.rollout.llm_profiles or [])
        except Exception as exc:
            logger.warning(
                "alternative_llm_profile_index: rollout.llm_profiles "
                "unavailable (%s: %s); skipping diversity selection",
                type(exc).__name__,
                exc,
            )
            return None
        if not profiles:
            return None

        used_llm_keys: set[tuple[str, str]] = set()
        for rollout in used_rollouts:
            model = str(getattr(rollout, "llm_model", "") or "").strip().lower()
            backend = ""
            metadata = (
                getattr(rollout, "search_metadata", None)
                if isinstance(getattr(rollout, "search_metadata", None), dict)
                else {}
            )
            if isinstance(metadata, dict):
                backend = str(metadata.get("rollout_llm_backend") or "").strip().lower()
            if model:
                used_llm_keys.add((backend, model))

        # Walk profiles in declared order; first profile that resolves
        # to a different (backend, model) tuple wins. Stable choice so
        # the followup is reproducible across runs.
        for profile_index, _profile in enumerate(profiles):
            try:
                resolved = self.config.get_llm_for_rollout_profile(profile_index)
            except (IndexError, KeyError, ValueError, AttributeError) as exc:
                # Phase 2C 5.5: log explicit exception types instead of
                # silently skipping. A misconfigured profile that raises
                # an *unexpected* exception type should surface so the
                # operator can fix it; the silent ``except Exception``
                # hid those before.
                logger.warning(
                    "alternative_llm_profile_index: skipping profile %s due to %s: %s",
                    profile_index,
                    type(exc).__name__,
                    exc,
                )
                continue
            backend_check = _resolve_patched("llm_backend_is_available", llm_backend_is_available)
            if not backend_check(resolved):
                continue
            backend = ""
            try:
                backend = str(resolved.backend.value).strip().lower()
            except AttributeError:
                pass
            model = str(getattr(resolved, "model", "") or "").strip().lower()
            if not model and not backend:
                continue
            if (backend, model) not in used_llm_keys:
                return profile_index
        return None

    def _apply_profile_index_override_to_briefs(
        self,
        followup_plan: IssuePlan,
        profile_index: int,
    ) -> None:
        """Pin every brief in the followup plan to the given profile index.

        Sets ``brief.search_policy["rollout_profile_index"] = N`` so the
        engine's ``_resolve_rollout_llm_config`` picks the desired LLM
        instead of the rollout-id-derived default. Idempotent and
        no-op when ``profile_index`` is None.
        """

        if profile_index is None:
            return
        for brief in list(followup_plan.rollout_briefs or []):
            policy = brief.search_policy if isinstance(brief.search_policy, dict) else {}
            policy = dict(policy)
            policy["rollout_profile_index"] = int(profile_index)
            brief.search_policy = policy

    def _relaunch_with_alternative_backend_or_sandbox(
        self,
        followup_plan: IssuePlan,
        rollout_results: list[RolloutResult],
    ) -> dict[str, Any]:
        """Apply the strongest backend-anomaly relaunch hint to the followup plan.

        Phase 2 10.K: when prior rollouts were salvaged with a backend
        anomaly, ``rollout.search_metadata["backend_anomaly_relaunch_hint"]``
        carries the recovery axis the engine recommends:

        * ``sandbox_bypass``      → set ``APEX_CODEX_BYPASS_SANDBOX=1`` in the
          followup brief's ``cli_env_overrides`` (subprocess-scoped, never
          mutating the parent process environment).
        * ``double_cli_timeout``  → multiply the followup brief's
          ``cli_hard_timeout_seconds`` by the hint's multiplier.
        * ``alternative_profile`` → reuse ``_alternative_llm_profile_index``
          to route through a different LLM backend.

        Returns the merged hint payload that was applied (or an empty dict
        when no hint was found). Safe to call when no salvaged anomalies
        exist; this is a no-op in that case.
        """

        hints: list[dict[str, Any]] = []
        for rollout in rollout_results:
            metadata = (
                getattr(rollout, "search_metadata", None)
                if isinstance(getattr(rollout, "search_metadata", None), dict)
                else {}
            )
            hint = (
                metadata.get("backend_anomaly_relaunch_hint")
                if isinstance(metadata, dict)
                else None
            )
            if isinstance(hint, dict) and hint.get("axis"):
                hints.append(dict(hint))
        if not hints:
            return {}

        applied: dict[str, Any] = {}
        # Apply env overrides (sandbox bypass) to every brief in the plan.
        env_overrides: dict[str, str] = {}
        for hint in hints:
            extra_env = hint.get("env_overrides")
            if isinstance(extra_env, dict):
                for key, value in extra_env.items():
                    env_overrides[str(key)] = str(value)
        if env_overrides:
            for brief in list(followup_plan.rollout_briefs or []):
                policy = dict(brief.search_policy) if isinstance(brief.search_policy, dict) else {}
                existing_cli_env = policy.get("cli_env_overrides")
                existing_env = dict(existing_cli_env) if isinstance(existing_cli_env, dict) else {}
                existing_env.update(env_overrides)
                policy["cli_env_overrides"] = existing_env
                brief.search_policy = policy
            applied["env_overrides"] = dict(env_overrides)

        # Apply timeout multiplier to every brief in the plan.
        timeout_multipliers = [
            float(hint.get("timeout_multiplier") or 1.0)
            for hint in hints
            if hint.get("axis") == "double_cli_timeout"
        ]
        if timeout_multipliers:
            multiplier = max(timeout_multipliers)
            for brief in list(followup_plan.rollout_briefs or []):
                policy = dict(brief.search_policy) if isinstance(brief.search_policy, dict) else {}
                base_timeout = policy.get("cli_hard_timeout_seconds")
                if not isinstance(base_timeout, (int, float)) or base_timeout <= 0:
                    profile_index = policy.get("rollout_profile_index")
                    llm_config = None
                    try:
                        if isinstance(profile_index, int):
                            llm_config = self.config.get_llm_for_rollout_profile(profile_index)
                    except (IndexError, KeyError, ValueError, AttributeError) as exc:
                        # Phase 2C 5.5: log explicit exception types
                        # instead of silently swallowing. We still fall
                        # back to llm_configs[0] below — but a crash that
                        # isn't a config lookup error indicates a bug.
                        logger.warning(
                            "_relaunch_with_alternative_backend_or_sandbox: "
                            "profile_index=%s lookup failed (%s: %s); "
                            "falling back to llm_configs[0]",
                            profile_index,
                            type(exc).__name__,
                            exc,
                        )
                        llm_config = None
                    if llm_config is None and self.config.llm_configs:
                        llm_config = self.config.llm_configs[0]
                    base_timeout = float(
                        getattr(llm_config, "cli_hard_timeout_seconds", 0)
                        or getattr(llm_config, "cli_timeout", 0)
                        or 0
                    )
                if base_timeout and base_timeout > 0:
                    applied_timeout = int(float(base_timeout) * multiplier)
                    policy["cli_hard_timeout_seconds"] = applied_timeout
                    policy["cli_timeout_multiplier"] = multiplier
                    brief.search_policy = policy
                    applied["cli_hard_timeout_seconds"] = applied_timeout
            applied["timeout_multiplier"] = multiplier

        # Route through an alternative profile when finalization failed.
        wants_alt_profile = any(hint.get("axis") == "alternative_profile" for hint in hints)
        if wants_alt_profile:
            alt_profile = self._alternative_llm_profile_index(rollout_results)
            if alt_profile is not None:
                self._apply_profile_index_override_to_briefs(followup_plan, alt_profile)
                applied["alternative_profile_index"] = int(alt_profile)

        if applied:
            logger.info(
                "Phase 2 10.K reroute applied to %s followup brief(s): %s",
                len(followup_plan.rollout_briefs or []),
                applied,
            )
        return applied

    def _detect_recurring_structural_blocker(
        self,
        rollout_results: list[RolloutResult],
    ) -> tuple[str, list[RolloutResult]]:
        """Find the most-common structural blocker shared across failed rollouts.

        Returns ``(blocker_kind, affected_rollouts)``. If no blocker
        recurs at least ``_STRUCTURAL_RECOVERY_MIN_RECURRENCE`` times,
        returns an empty kind so the structural-recovery followup is a
        no-op. Recurrence is what justifies an additional rollout —
        a single structural failure is usually addressed by the
        in-rollout repair loop.
        """

        kind_to_rollouts: dict[str, list[RolloutResult]] = {}
        for result in rollout_results:
            if result is None or result.success:
                continue
            kind = self._rollout_structural_blocker_kind(result)
            if not kind:
                multi_agent = (
                    result.multi_agent_summary
                    if isinstance(result.multi_agent_summary, dict)
                    else {}
                )
                if int(multi_agent.get("boundary_pressure_count") or 0) > 0:
                    kind = "delegation_boundary"
            if not kind:
                continue
            kind_to_rollouts.setdefault(kind, []).append(result)
        discovery_affected = self._failed_rollouts_with_repeated_discovery_files(
            rollout_results,
        )
        if len(discovery_affected) >= self._STRUCTURAL_RECOVERY_MIN_RECURRENCE:
            kind_to_rollouts.setdefault("partial_discovery", discovery_affected)
        if not kind_to_rollouts:
            return "", []
        best_kind = max(
            kind_to_rollouts,
            key=lambda k: (
                len(kind_to_rollouts[k]),
                0 if k == "partial_discovery" else 1,
                k,
            ),
        )
        affected = kind_to_rollouts[best_kind]
        if len(affected) < self._STRUCTURAL_RECOVERY_MIN_RECURRENCE:
            return "", []
        return best_kind, affected

    def _build_structural_recovery_summary(
        self,
        *,
        blocker_kind: str,
        affected_rollouts: list[RolloutResult],
        repo_path: Optional[str] = None,
        repo_context: Optional[RepoContext] = None,
    ) -> str:
        """Compose a recovery prompt for the recurring-structural-failure path.

        The prompt focuses the agent on diagnosis-before-edit: figure
        out which prior edit broke imports / collection, run targeted
        verification first, and avoid wide source rewrites that
        re-create the same break.
        """

        guidance = _quick_verification_followup_guidance(blocker_kind) or (
            "Multiple prior rollouts failed at the same structural stage. "
            "Investigate and fix the recurring blocker before any other change."
        )
        # Sample blocker text from the highest-changed-files rollout —
        # that's the most informative trace.
        ranked = sorted(
            affected_rollouts,
            key=lambda r: len(getattr(r, "changed_files", []) or []),
            reverse=True,
        )
        sample_blockers: list[str] = []
        for rollout in ranked[:3]:
            qv = rollout.quick_verification if isinstance(rollout.quick_verification, dict) else {}
            blocker_summary = _quick_verification_blocker_summary(qv).strip()
            if not blocker_summary:
                blocker_summary = str(rollout.failure_reason or "").strip()
            if blocker_summary and blocker_summary not in sample_blockers:
                sample_blockers.append(blocker_summary[:300])
        if blocker_kind == "partial_discovery":
            opening = (
                "### Recurring invalid-rollout discovery\n"
                f"{len(affected_rollouts)} prior rollouts were not valid "
                "candidates, but they repeatedly surfaced the same clean "
                "solution-context files. Treat those files as evidence for "
                "parent-controlled planning only; do not reuse or trust the "
                "invalid patches themselves."
            )
        else:
            opening = (
                f"### Recurring structural blocker: {blocker_kind}\n"
                f"{len(affected_rollouts)} prior rollouts all failed before "
                f"any test could meaningfully run, with the same kind of "
                f"structural break ({blocker_kind}). The pattern indicates a "
                f"consistent failure mode in the agent's edits — diagnose the "
                f"recurring break before introducing more source changes."
            )
        parts: list[str] = [opening, f"\n{guidance}"]
        if sample_blockers:
            parts.append("\n### Observed blocker output (across rollouts)")
            for blocker in sample_blockers:
                parts.append(f"  | {blocker}")
        suppress_broad_patch_footprints = (
            blocker_kind in self._STRUCTURAL_RECOVERY_PATCH_FOOTPRINT_BLOCKER_KINDS
        )
        sample_changed_files = self._failed_rollout_discovery_files(
            ranked[:3],
            include_singletons=True,
            suppress_broad_invalid_patch_footprints=suppress_broad_patch_footprints,
            blocker_kind=blocker_kind,
        )
        if repo_path and repo_context:
            sample_changed_files, _dropped = _clean_baseline_existing_file_hints(
                repo_path,
                repo_context,
                sample_changed_files,
            )
        repeated_discovery_files = self._failed_rollout_discovery_files(
            affected_rollouts,
            suppress_broad_invalid_patch_footprints=suppress_broad_patch_footprints,
            blocker_kind=blocker_kind,
        )
        if repo_path and repo_context:
            repeated_discovery_files, _dropped = _clean_baseline_existing_file_hints(
                repo_path,
                repo_context,
                repeated_discovery_files,
            )
        broad_invalid_patch_files = self._failed_rollout_broad_invalid_patch_files(
            affected_rollouts,
            blocker_kind=blocker_kind,
        )
        if repo_path and repo_context:
            broad_invalid_patch_files, _dropped = _clean_baseline_existing_file_hints(
                repo_path,
                repo_context,
                broad_invalid_patch_files,
            )
        if repeated_discovery_files:
            parts.append("\n### Useful source/test discovery from invalid rollouts")
            parts.append(
                "These files recurred across failed candidates. Use them as "
                "search and diagnosis context for the clean followup; the "
                "prior patches remain invalid and must not be copied wholesale:"
            )
            for path in repeated_discovery_files[:8]:
                parts.append(f"  - {path}")
        if sample_changed_files:
            parts.append("\n### Files repeatedly touched by failed rollouts")
            parts.append(
                "These files were edited by the prior failing rollouts. "
                "If your patch must touch them, verify imports and "
                "module-load behavior FIRST:"
            )
            for path in sample_changed_files[:8]:
                parts.append(f"  - {path}")
        if broad_invalid_patch_files:
            parts.append("\n### Broad invalid patch footprint")
            parts.append(
                "These files came from collection-collapsing invalid patches. "
                "They are not a required implementation scope; use them only "
                "to avoid copying the failed broad edit pattern. Start from "
                "the blocker output and baseline-supported focus, then widen "
                "only after collection remains healthy:"
            )
            for path in broad_invalid_patch_files[:12]:
                parts.append(f"  - {path}")
        boundary_requested_files = self._rollout_boundary_requested_files(affected_rollouts)
        if repo_path and repo_context:
            boundary_requested_files, _dropped = _clean_baseline_existing_file_hints(
                repo_path,
                repo_context,
                boundary_requested_files,
            )
        if boundary_requested_files:
            parts.append("\n### Adjacent files requested by worker boundaries")
            parts.append(
                "Prior delegated workers were not allowed to edit these files, "
                "but reported them as necessary adjacent scope. Include them in "
                "the parent recovery plan instead of bypassing child-worker "
                "scope enforcement:"
            )
            for path in boundary_requested_files[:8]:
                parts.append(f"  - {path}")
        inventory_framework = ""
        inventory_collection_command = ""
        for rollout in ranked:
            quick_verification = (
                rollout.quick_verification if isinstance(rollout.quick_verification, dict) else {}
            )
            inventory_framework, inventory_collection_command, _ = (
                _quick_verification_inventory_context(quick_verification)
            )
            if inventory_framework or inventory_collection_command:
                break
        discovery_instruction = (
            "2. Run the repository's test-discovery or collection command "
            "to verify discovery succeeds after EACH source edit. If "
            "discovery or collection breaks, revert that edit and try a "
            "narrower change.\n"
        )
        if inventory_collection_command:
            discovery_instruction = (
                f"2. Run `{inventory_collection_command}` to verify test discovery "
                "or collection succeeds after EACH source edit. If discovery or "
                "collection breaks, revert that edit and try a narrower change.\n"
            )
        parts.append(
            "\n### Recovery strategy — DIAGNOSE BEFORE YOU EDIT\n"
            "1. Run the repository's test command from the baseline FIRST to "
            "confirm it works on the unmodified tree. Only then apply edits.\n"
            + discovery_instruction
            + "3. The recurring blocker usually originates in the test bootstrap "
            "or top-level module-load path (for example a removed export, a bad "
            "initializer side effect, or an attribute that the harness depends on). "
            "Read the broken module's full dependency chain before editing.\n"
            "4. Prefer additive edits to existing modules over replacing "
            "entire files. Removing top-level names breaks imports and "
            "is the most common cause of conftest_import / module_import "
            "failures.\n"
            "5. If a function or class signature must change, also update "
            "every import site in the same patch."
        )
        return "\n".join(parts)

    def _launch_structural_recovery_followup(
        self,
        *,
        repo_path: str,
        repo_context: RepoContext,
        issue_description: str,
        issue_plan: IssuePlan,
        planner: IssuePlanner,
        test_command: Optional[str],
        engine: RolloutEngine,
        rollout_results: list[RolloutResult],
        task_state_graph: Optional[TaskStateGraph],
        wallclock_deadline: Optional[float] = None,
    ) -> tuple[list[RolloutResult], list[dict[str, Any]]]:
        """Targeted-recovery followup when multiple rollouts share a structural break.

        Unlike the near-miss path (which seeds from a high-pass-rate
        anchor), this one starts from a CLEAN baseline. The prior
        rollouts were broken in a consistent way; seeding from any of
        them would replay the broken state. The recovery prompt
        explicitly explains the recurring blocker and instructs the
        agent to diagnose-then-edit narrowly.

        Returns ``(extra_results, transitions)``. Empty when no
        recurring structural blocker is detected.
        """

        blocker_kind, affected_rollouts = self._detect_recurring_structural_blocker(
            rollout_results,
        )
        if not blocker_kind or not affected_rollouts:
            return [], []
        residual_summary = self._build_structural_recovery_summary(
            blocker_kind=blocker_kind,
            affected_rollouts=affected_rollouts,
            repo_path=repo_path,
            repo_context=repo_context,
        )
        # Structural recovery starts from a clean baseline, so file-scoped
        # recovery hints must refer to files present in that baseline.
        source_issue_plan = copy.deepcopy(issue_plan)
        dropped_file_hints = _scrub_clean_baseline_issue_plan_file_hints(
            source_issue_plan,
            repo_path,
            repo_context,
        )
        focus_candidates: list[str] = list(source_issue_plan.relevant_files)
        suppress_broad_patch_footprints = (
            blocker_kind in self._STRUCTURAL_RECOVERY_PATCH_FOOTPRINT_BLOCKER_KINDS
        )
        discovery_focus_candidates = self._failed_rollout_discovery_files(
            affected_rollouts,
            include_singletons=True,
            suppress_broad_invalid_patch_footprints=suppress_broad_patch_footprints,
            blocker_kind=blocker_kind,
        )
        for path in discovery_focus_candidates:
            if path and path not in focus_candidates:
                focus_candidates.append(path)
        boundary_focus_candidates = self._rollout_boundary_requested_files(affected_rollouts)
        for path in boundary_focus_candidates:
            if path and path not in focus_candidates:
                focus_candidates.append(path)
        focus_files, missing_focus_files = _clean_baseline_existing_file_hints(
            repo_path,
            repo_context,
            focus_candidates,
        )
        dropped_file_hints.extend(missing_focus_files)
        boundary_focus_files, missing_boundary_files = _clean_baseline_existing_file_hints(
            repo_path,
            repo_context,
            boundary_focus_candidates,
        )
        dropped_file_hints.extend(missing_boundary_files)
        partial_discovery_files, missing_discovery_files = _clean_baseline_existing_file_hints(
            repo_path,
            repo_context,
            self._failed_rollout_discovery_files(
                affected_rollouts,
                suppress_broad_invalid_patch_footprints=suppress_broad_patch_footprints,
                blocker_kind=blocker_kind,
            ),
        )
        dropped_file_hints.extend(missing_discovery_files)
        followup_plan = planner.build_followup_plan(
            source_issue_plan,
            repo_context,
            rollout_results,
            additional_rollouts=1,
            residual_summary=residual_summary,
            residual_focus_files=focus_files,
            task_state_context=source_issue_plan.task_state_context,
        )
        if not followup_plan.rollout_briefs:
            return [], []
        dropped_file_hints.extend(
            _scrub_clean_baseline_issue_plan_file_hints(
                followup_plan,
                repo_path,
                repo_context,
            )
        )
        dropped_file_hints = list(dict.fromkeys(dropped_file_hints))
        if dropped_file_hints:
            planner_metadata = dict(followup_plan.planner_metadata or {})
            planner_metadata["clean_baseline_recovery_dropped_file_hints"] = dropped_file_hints[:50]
            followup_plan.planner_metadata = planner_metadata
        # Pin to an unused LLM profile when one is available — recurring
        # structural breaks are often caused by the same model
        # consistently making the same import-breaking edit; routing the
        # recovery through a different LLM is the cheapest cross-rollout
        # diversity lever.
        alt_profile = self._alternative_llm_profile_index(rollout_results)
        if alt_profile is not None:
            self._apply_profile_index_override_to_briefs(followup_plan, alt_profile)
            logger.info(
                "Structural-recovery followup pinned to LLM profile index %s "
                "(prior rollouts used a different model).",
                alt_profile,
            )
        self._save_issue_plan(followup_plan)
        transition = {
            "attempt": 1,
            "reason": (
                f"{len(affected_rollouts)} prior rollouts failed with the same "
                f"structural blocker ({blocker_kind}); launching one recovery "
                "rollout from a clean baseline with diagnosis-first guidance."
            ),
            "trigger": "structural_recovery_followup",
            "followup_round": 1,
            "from_primitives": list(issue_plan.orchestration_primitives),
            "to_primitives": list(followup_plan.orchestration_primitives),
            "from_rollouts": len(issue_plan.rollout_briefs),
            "to_rollouts": len(followup_plan.rollout_briefs),
            "blocker_kind": blocker_kind,
            "affected_rollout_ids": [r.rollout_id for r in affected_rollouts],
            "boundary_requested_files": boundary_focus_files,
            "partial_discovery_files": partial_discovery_files,
        }
        logger.info(
            "Launching structural-recovery followup: blocker=%s affected_rollouts=%s",
            blocker_kind,
            ", ".join(str(r.rollout_id) for r in affected_rollouts),
        )
        try:
            (
                extra_results,
                _refreshed_plan,
                _next_rollout_id,
                progressive_transitions,
                _search_summary,
            ) = self._execute_progressive_rollout_plan(
                repo_path=repo_path,
                repo_context=repo_context,
                issue_description=issue_description,
                issue_plan=followup_plan,
                planner=planner,
                test_command=test_command,
                engine=engine,
                rollout_id_offset=self._next_rollout_id_after(rollout_results),
                task_state_graph=task_state_graph,
                # Critical: NO seed. Start fresh — prior rollouts broke
                # consistently from the same baseline; resetting gives
                # the agent a clean slate to apply the recovery prompt.
                initial_workspace_seed=None,
                wallclock_deadline=wallclock_deadline,
            )
        except Exception as exc:  # noqa: BLE001 - best-effort followup
            logger.warning("Structural-recovery followup raised %s; skipping.", exc)
            return [], [transition]
        transitions = [transition] + list(progressive_transitions)
        return list(extra_results), transitions

    def _editable_test_files_for_followup(
        self,
        issue_plan: IssuePlan,
    ) -> list[str]:
        """Return visible test files the agent IS allowed to edit.

        On completion-task repositories (``allocator_features.is_completion_task``
        true) the selector's policy specifically permits edits to
        ``test_context.incomplete_test_files`` — those are stub tests
        the benchmark expects the agent to fill in. The followup
        machinery must surface this list explicitly to the agent;
        otherwise the focus-file list (which only contains source
        files) leaves the agent unable to act on a residual that
        actually lives in a test stub.
        """

        try:
            test_context = getattr(issue_plan, "test_context", None)
            if test_context is None:
                return []
            files = list(getattr(test_context, "incomplete_test_files", None) or [])
            return [str(path) for path in files if str(path).strip()]
        except Exception as exc:  # noqa: BLE001 - never fail the followup on metadata extraction
            # Audit H12: log so a malformed test_context shape doesn't
            # silently strip the editable-test-stub list from the followup
            # prompt. The fallback is still ``[]`` so behavior is
            # backward-compatible.
            logger.warning(
                "_extract_incomplete_test_files: %s: %s; "
                "returning empty list (followup prompt will lose stub list)",
                type(exc).__name__,
                exc,
            )
            return []

    def _build_near_miss_residual_summary(
        self,
        *,
        anchor: RolloutResult,
        anchor_quick_verification: dict[str, Any],
        editable_test_files: Optional[list[str]] = None,
    ) -> str:
        """Compose a high-density residual-failure summary for the followup prompt.

        Earlier near-miss followups failed when the retry prompt was a thin
        paragraph and the agent kept re-producing the same patch with the
        same residual ``NotImplementedError`` failures. This version is deliberately verbose: it includes
        the failure-cluster grouping, the SOURCE CODE of the failing
        tests, the implementation-marker scan of the anchor's changed
        files (NotImplementedError / TODO / `raise NotImplemented` /
        `pass  # TODO`), the list of editable visible test stubs (so the
        agent knows it CAN fix tests that themselves contain
        NotImplementedError), and an explicit "different angle"
        instruction. The agent's prompt budget can absorb ~5KB of
        structured failure evidence cheaply, and this is the
        highest-leverage place to spend it.
        """

        parts: list[str] = []
        signal_score = quick_verification_signal_score(anchor_quick_verification)
        observed_pass_rate = anchor_quick_verification.get("pass_rate")
        coverage_ratio = quick_verification_expected_coverage_ratio(anchor_quick_verification)
        inventory_framework, inventory_collection_command, inventory_test_command = (
            _quick_verification_inventory_context(anchor_quick_verification)
        )
        inventory_framework_label = _humanize_test_inventory_framework(inventory_framework)
        matched_expected = anchor_quick_verification.get("matched_expected_test_count")
        expected_total = anchor_quick_verification.get("expected_test_count")
        failed = anchor_quick_verification.get("failed")
        errors = anchor_quick_verification.get("errors")
        precise_residual_repair = self._near_miss_has_precise_residual_repair_signal(
            anchor_quick_verification
        )

        # 1. Headline status: tell the model exactly how close it is and
        #    that it must NOT discard the existing edits.
        if isinstance(signal_score, (int, float)):
            headline = (
                f"PRIOR ROLLOUT REACHED {float(signal_score):.4f} "
                "COVERAGE-AWARE QUICK-VERIFICATION SCORE"
            )
            if (
                isinstance(observed_pass_rate, (int, float))
                and abs(float(observed_pass_rate) - float(signal_score)) >= 1e-6
            ):
                headline += f" (observed local pass_rate={float(observed_pass_rate):.4f}"
                if isinstance(coverage_ratio, (int, float)):
                    headline += f", expected_coverage={float(coverage_ratio):.4f}"
                if (
                    isinstance(matched_expected, int)
                    and isinstance(expected_total, int)
                    and expected_total > 0
                ):
                    headline += f", matched_expected={matched_expected}/{expected_total}"
                headline += ")"
            headline += (
                f" (failed={failed or 0}, errors={errors or 0}). "
                "You are branching from that rollout's workspace — DO NOT discard or "
                "rewrite its existing edits; resolve the RESIDUAL failures only."
            )
            parts.append(headline)
        elif isinstance(observed_pass_rate, (int, float)):
            parts.append(
                f"PRIOR ROLLOUT REACHED {float(observed_pass_rate):.4f} VISIBLE-TEST PASS RATE "
                f"(failed={failed or 0}, errors={errors or 0}). "
                "You are branching from that rollout's workspace — DO NOT discard or "
                "rewrite its existing edits; resolve the RESIDUAL failures only."
            )
        else:
            parts.append(
                "Prior rollout produced a near-passing patch; you are branching from "
                "its workspace and must resolve the residual failures only."
            )

        # 2. Failure-cluster grouping. Repeated identical errors indicate a
        #    SINGLE root cause; the model must fix the source pattern, not
        #    each test individually.
        clusters: list[dict[str, Any]] = []
        for raw_cluster in anchor_quick_verification.get("failure_clusters") or []:
            if isinstance(raw_cluster, dict):
                clusters.append(raw_cluster)
                continue
            cluster_text = str(raw_cluster or "").strip()
            if cluster_text:
                clusters.append({"summary": cluster_text})
        if clusters:
            parts.append("\n### Failure clusters (same error → same root cause)")
            for cluster in clusters[:5]:
                count = cluster.get("count") or len(
                    cluster.get("test_ids") or cluster.get("tests") or []
                )
                label = (cluster.get("label") or cluster.get("kind") or "").strip()
                tests = list(cluster.get("test_ids") or cluster.get("tests") or [])
                summary = (cluster.get("summary") or cluster.get("message") or "").strip()
                if not label and summary:
                    line = f"- {summary[:220]}"
                else:
                    line = f"- {count}x {label}"
                    if summary:
                        line += f" - {summary[:120]}"
                if tests:
                    line += f"\n    tests: {', '.join(tests[:5])}"
                parts.append(line)
            parts.append(
                "If two or more clusters share the same error message, the fix is "
                "almost always ONE source change, not N individual edits."
            )

        # 3. Failing-test source code. Reading the failing tests is what
        #    most patcher rollouts skip; pasting the source forces the
        #    model to reason about expected behaviour.
        failed_tests = list(anchor_quick_verification.get("failed_tests") or [])
        if failed_tests:
            parts.append("\n### Residual failing test IDs")
            for test_id in failed_tests[:25]:
                parts.append(f"- {test_id}")
            extra_failed = max(0, len(failed_tests) - 25)
            if extra_failed:
                parts.append(f"- ... ({extra_failed} more)")
        test_sources = self._collect_failing_test_source(anchor, failed_tests)
        if test_sources:
            parts.append("\n### Failing test source (read first, then implement)")
            for path, source in test_sources:
                parts.append(f"\n```python\n# {path}\n{source}\n```")

        # 4. Implementation-marker scan. For broad residual recovery,
        #    NotImplementedError / TODO / `pass  # TODO` in the anchor's
        #    changed files are useful "fix me here" tags. For a precise
        #    one-test residual with full expected coverage, those markers are
        #    weaker than the failing test/import path and have repeatedly pulled
        #    agents into broad completion work. Keep them out of the prompt
        #    unless the residual is not precise.
        markers = [] if precise_residual_repair else self._collect_implementation_markers(anchor)
        if markers:
            parts.append("\n### Implementation markers in anchor's changed files")
            parts.append(
                "Each line below is an explicit 'fix me' marker the prior rollout left in place:"
            )
            for path, line_no, line_text in markers:
                parts.append(f"  {path}:{line_no}  {line_text}")

        # 5. Output excerpt — the raw failure tail, in case the cluster
        #    summary missed something the model can grep for.
        excerpt = (anchor_quick_verification.get("output_excerpt") or "").strip()
        if excerpt:
            parts.append("\n### Test runner output excerpt")
            parts.append(excerpt[:1200])

        # 5b. Per-test traceback signal — pull file:line of the
        #     exception terminus from the output excerpt. Helps the
        #     agent jump directly to the failure site instead of
        #     re-reading the entire stack trace.
        traceback_terminals = self._extract_traceback_terminals(excerpt, anchor)
        if traceback_terminals:
            parts.append("\n### Failure terminus file:line signals")
            for path, line_no, exception_label in traceback_terminals:
                parts.append(f"  {path}:{line_no}  {exception_label}")

        # 5c. Git diff of the anchor's edits — useful for broad residual
        #     recovery, but too noisy for a precise one-test repair. In precise
        #     mode the workspace already contains the anchor edits; the next
        #     agent should inspect the failing path rather than reread a broad
        #     patch.
        anchor_diff = "" if precise_residual_repair else self._collect_anchor_git_diff(anchor)
        if anchor_diff:
            parts.append("\n### Anchor git diff (your starting point — preserve these edits)")
            parts.append(f"```diff\n{anchor_diff}\n```")

        # 6. Anchor patch footprint.
        if anchor.changed_files:
            if precise_residual_repair:
                parts.append(
                    "\nAnchor patch is already applied in this workspace. Treat its "
                    "changed files as continuation context, not as required repair "
                    "scope; edit only files required by the residual failing test path."
                )
            else:
                parts.append(
                    "\nAnchor patch touched: "
                    + ", ".join(list(anchor.changed_files)[:8])
                    + " — these files already contain partial work; modify them in place."
                )

        # 6a. Discovery gap — when the official benchmark expects test
        #     IDs that the agent's pytest run did NOT collect (marker
        #     filters, conftest skips, parametrize counted differently,
        #     missing collection root). Listing the missing IDs lets the
        #     agent diff its `pytest --collect-only` against the
        #     authoritative set instead of silently undercounting. This
        #     is the symptom that produced partial expected-ID coverage
        #     even when local pytest reported 100%.
        missing_test_ids = list(anchor_quick_verification.get("missing_expected_test_ids") or [])
        missing_count_total = anchor_quick_verification.get("missing_expected_test_count")
        if missing_test_ids:
            shown = missing_test_ids[: self._MISSING_EXPECTED_IDS_PROMPT_LIMIT]
            extra = max(0, len(missing_test_ids) - len(shown))
            missing_test_groups, has_parametrized_missing_ids = (
                self._summarize_missing_expected_test_groups(missing_test_ids)
            )
            missing_test_sources = self._collect_failing_test_source(
                anchor,
                missing_test_ids,
            )
            parts.append(
                "\n### Discovery gap — known tests the current discovery path did NOT collect"
            )
            count_label = (
                f"{int(missing_count_total)}"
                if isinstance(missing_count_total, int) and missing_count_total > 0
                else f"{len(missing_test_ids)}"
            )
            discovery_subject = "current test-discovery path"
            if inventory_framework_label:
                discovery_subject = f"{inventory_framework_label} discovery path"
            parts.append(
                f"The benchmark scorer expects {count_label} test IDs that the "
                f"rollout's {discovery_subject} did NOT discover — the "
                "orchestrator treats them as failures. Resolve the discovery "
                "gap (a filter, harness skip, or generated-case mismatch is "
                "likely responsible) before attempting more source edits:"
            )
            parts.append(
                "Do not modify any file that, when reverted, causes additional "
                "expected IDs to disappear during collection."
            )
            for test_id in shown:
                parts.append(f"  - {test_id}")
            if extra:
                parts.append(f"  ... ({extra} more)")
            if missing_test_groups:
                parts.append(
                    "Grouped missing IDs by visible test function to help spot "
                    "dynamic parametrization or provider gaps:"
                )
                for group in missing_test_groups:
                    parts.append(f"  - {group}")
            if missing_test_sources:
                parts.append("\n### Representative source for missing-coverage tests")
                parts.append(
                    "Read these visible test definitions and trace any helper, "
                    "registry, or provider code that determines how many cases "
                    "they generate:"
                )
                for path, source in missing_test_sources:
                    parts.append(f"\n```python\n# {path}\n{source}\n```")
            if inventory_collection_command:
                parts.append(
                    f"Action: run `{inventory_collection_command}`, compare against "
                    "the IDs above, and adjust the test command, filters, or "
                    "framework configuration so the missing tests get collected."
                )
            elif inventory_test_command:
                parts.append(
                    f"Action: rerun `{inventory_test_command}` with the framework's "
                    "discovery or listing mode enabled, compare against the IDs "
                    "above, and adjust the test command or filters so the missing "
                    "tests get collected."
                )
            else:
                parts.append(
                    "Action: run the repository's test-discovery or collection "
                    "command, compare against the IDs above, and adjust filters or "
                    "framework configuration so the missing tests get collected."
                )
            if has_parametrized_missing_ids:
                parts.append(
                    "If the missing IDs are parametrized variants of one visible "
                    "test, inspect the helper/provider/registry code that "
                    "controls how many cases that test emits before changing "
                    "unrelated source files."
                )

        # 6b. Editable visible-test stubs. Critical for completion tasks
        #     where the failing test FILE itself is a stub the benchmark
        #     expects the agent to implement. Without this section the agent assumes
        #     ``tests/*.py`` is read-only specification.
        editable_test_files = list(editable_test_files or [])
        if editable_test_files:
            parts.append(
                "\n### Editable visible-test stubs\n"
                "These visible test files are tagged ``incomplete_test_files`` by "
                "the planner — the selector policy ALLOWS edits to them, and the "
                "benchmark expects the agent to fill in their bodies if a residual "
                "failure originates inside one of them (e.g. ``raise NotImplementedError`` "
                "inside the test function itself):"
            )
            for path in editable_test_files[:8]:
                parts.append(f"  - {path}")
            parts.append(
                "If the residual failures above point to test functions that themselves "
                "raise NotImplementedError, IMPLEMENT THE TEST BODY — that is the intended "
                "fix, not a separate source change."
            )

        # 7. Explicit repair strategy. Precise residuals need the opposite of a
        #    broad "different angle" sweep: follow the one failing test path,
        #    make the smallest edit, and avoid unrelated anchor TODOs.
        if precise_residual_repair:
            parts.append(
                "\n### Strategy guidance — PRECISE RESIDUAL REPAIR\n"
                "The verifier preserved expected-test coverage and now names a tiny "
                "residual failing-test set. Treat those failing test IDs and their "
                "source as the authoritative subgoal:\n"
                "  1. Open each residual failing test and trace only the imports, "
                "helpers, callees, and data transformations needed by that test.\n"
                "  2. Ignore unrelated TODOs, NotImplementedError markers, and broad "
                "anchor changed-file clusters unless they are on the failing test's "
                "runtime path.\n"
                "  3. Make the smallest source edit that fixes the residual behavior "
                "while preserving the anchor workspace edits.\n"
                "  4. Rerun the exact failing test first, then the broader expected "
                "suite only after the targeted check passes.\n"
                "  5. If a fix makes a previously-passing test fail, revert that "
                "narrow edit and trace the residual path again."
            )
        else:
            parts.append(
                "\n### Strategy guidance — DIFFERENT ANGLE REQUIRED\n"
                "Multiple earlier rollouts produced patches that hit the same residual "
                "failures listed above. You are the LAST attempt. Take a different angle:\n"
                "  1. If the discovery-gap section above lists missing test IDs, fix "
                "the test-discovery problem FIRST — running tests the current "
                "collection path never sees is the most common cause of a high local "
                "pass rate but a low scored rate.\n"
                "  2. Open each failing test source — do not guess at the contract.\n"
                "  3. Identify each implementation marker and replace it with a real implementation.\n"
                "  4. If a function in the anchor's changed files raises NotImplementedError, "
                "that IS the function you need to implement — not a separate one.\n"
                "  5. If a TEST function itself raises NotImplementedError and the path is in "
                "the editable stubs list above, IMPLEMENT THE TEST BODY (do not skip it).\n"
                "  6. After each fix, rerun only the failing tests or the smallest "
                "targeted command your framework supports before re-running the "
                "broader suite.\n"
                "  7. If a fix makes a previously-passing test fail, revert and try a "
                "narrower change."
            )

        # 8. Feature C': optional model-driven root-cause diagnosis. When
        #    enabled, make ONE LLM call passing the anchor diff + residual
        #    failing-test evidence and append the model's concise root-cause +
        #    suggested-fix-direction as a clearly-labeled extra section. Fails
        #    open: any error/timeout leaves the heuristic dossier unchanged.
        diagnosis_section = self._diagnose_near_miss(
            anchor=anchor,
            anchor_quick_verification=anchor_quick_verification,
            heuristic_summary="\n".join(parts),
        )
        if diagnosis_section:
            parts.append(diagnosis_section)

        return "\n".join(parts)

    # Feature C': bound the evidence we feed into the single diagnosis call so
    # the prompt stays cost-bounded even for very large diffs / output tails.
    _NEAR_MISS_DIAGNOSIS_DIFF_MAX_CHARS = 6000
    _NEAR_MISS_DIAGNOSIS_OUTPUT_MAX_CHARS = 4000
    _NEAR_MISS_DIAGNOSIS_MAX_TEST_IDS = 25
    _NEAR_MISS_DIAGNOSIS_MAX_DIAGNOSIS_CHARS = 4000

    def _diagnose_near_miss(
        self,
        *,
        anchor: RolloutResult,
        anchor_quick_verification: dict[str, Any],
        heuristic_summary: str,
    ) -> str:
        """Feature C': one LLM call producing a near-miss root-cause diagnosis.

        Returns a clearly-labeled diagnosis section (string) to append to the
        residual repair dossier, or an empty string when the feature is
        disabled or anything goes wrong. The call is made with the
        planner/selection LLM client already resolvable from ``self.config``
        and is hard-bounded by ``near_miss_diagnosis_timeout_seconds`` so the
        cost stays predictable.

        FAIL OPEN: any disabled flag, missing client, empty evidence,
        exception, or timeout returns ``""`` — the caller then ships the
        unchanged heuristic dossier.
        """

        orchestration = getattr(self.config, "orchestration", None)
        if not bool(getattr(orchestration, "enable_near_miss_diagnosis", False)):
            return ""

        try:
            timeout_seconds = int(
                getattr(orchestration, "near_miss_diagnosis_timeout_seconds", 0) or 0
            )
        except (TypeError, ValueError):
            timeout_seconds = 0
        if timeout_seconds <= 0:
            timeout_seconds = 120

        # Gather the residual evidence: failing-test IDs, raw output tail, and
        # the candidate diff. If there is nothing actionable to reason about,
        # skip the call entirely.
        try:
            failed_tests = [
                str(item).strip()
                for item in list(anchor_quick_verification.get("failed_tests") or [])
                if str(item).strip()
            ]
            output_excerpt = str(anchor_quick_verification.get("output_excerpt") or "").strip()
            candidate_diff = self._collect_anchor_git_diff(anchor)
        except Exception:  # noqa: BLE001 — diagnosis is best-effort.
            logger.debug("near-miss diagnosis evidence collection failed", exc_info=True)
            return ""

        if not failed_tests and not output_excerpt and not candidate_diff:
            return ""

        client = self._build_near_miss_diagnosis_llm()
        if client is None:
            return ""

        # Prompt assembly + Message construction are pure string/list work, but
        # guard them anyway so the diagnosis can NEVER raise into the residual
        # summary builder (fail open: any error returns the heuristic dossier).
        try:
            from ..core.llm import Message

            prompt = self._build_near_miss_diagnosis_prompt(
                failed_tests=failed_tests,
                output_excerpt=output_excerpt,
                candidate_diff=candidate_diff,
            )
            messages = [
                Message(
                    role="system",
                    content=(
                        "You are a senior software engineer diagnosing why a nearly-"
                        "passing patch still fails a small set of tests. Respond with "
                        "a concise root-cause analysis and a concrete suggested fix "
                        "direction. Be specific and actionable; do not restate the "
                        "diff or the test output verbatim."
                    ),
                ),
                Message(role="user", content=prompt),
            ]
        except Exception:  # noqa: BLE001 — never let diagnosis break the retry.
            logger.debug("near-miss diagnosis prompt assembly failed", exc_info=True)
            return ""

        def _invoke() -> str:
            response = client.chat(messages)
            return str(getattr(response, "content", "") or "").strip()

        diagnosis = ""
        try:
            import concurrent.futures

            # NB: do NOT use the executor as a context manager here. ``__exit__``
            # calls ``shutdown(wait=True)`` which would block on a hung LLM call
            # even after the timeout fires (``future.cancel()`` is a no-op once
            # the worker has started). To honour the wall-clock bound we manage
            # the executor explicitly and shut it down WITHOUT waiting on
            # timeout, letting the worker drain in the background while we fall
            # back to the heuristic dossier.
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(_invoke)
                try:
                    diagnosis = future.result(timeout=timeout_seconds)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    logger.debug(
                        "near-miss diagnosis timed out after %ss; "
                        "falling back to heuristic dossier",
                        timeout_seconds,
                    )
                    executor.shutdown(wait=False)
                    return ""
            finally:
                # Successful/failed (non-timeout) paths: the worker is already
                # done, so a non-blocking shutdown reclaims the thread without
                # blocking the followup.
                executor.shutdown(wait=False)
        except Exception:  # noqa: BLE001 — never let diagnosis break the retry.
            logger.debug("near-miss diagnosis LLM call failed", exc_info=True)
            return ""

        diagnosis = (diagnosis or "").strip()
        if not diagnosis:
            return ""
        if len(diagnosis) > self._NEAR_MISS_DIAGNOSIS_MAX_DIAGNOSIS_CHARS:
            diagnosis = (
                diagnosis[: self._NEAR_MISS_DIAGNOSIS_MAX_DIAGNOSIS_CHARS]
                + "\n... (diagnosis truncated)"
            )

        return (
            "\n### Model-driven root-cause diagnosis (advisory)\n"
            "A diagnostic model reviewed the candidate diff and the residual "
            "failing tests and proposed the following root cause and fix "
            "direction. Treat it as a strong hint, but VERIFY against the "
            "failing test source above before acting:\n"
            f"{diagnosis}"
        )

    def _build_near_miss_diagnosis_prompt(
        self,
        *,
        failed_tests: list[str],
        output_excerpt: str,
        candidate_diff: str,
    ) -> str:
        """Compose the (bounded) user prompt for the diagnosis LLM call."""

        sections: list[str] = []
        sections.append(
            "A patch is one or a few tests short of passing. Diagnose the most "
            "likely ROOT CAUSE of the remaining failures and give a concrete "
            "SUGGESTED FIX DIRECTION. Keep it concise (a few sentences plus a "
            "short bullet list at most)."
        )
        if failed_tests:
            shown = failed_tests[: self._NEAR_MISS_DIAGNOSIS_MAX_TEST_IDS]
            extra = max(0, len(failed_tests) - len(shown))
            sections.append("## Remaining failing test IDs")
            sections.extend(f"- {test_id}" for test_id in shown)
            if extra:
                sections.append(f"- ... ({extra} more)")
        if output_excerpt:
            excerpt = output_excerpt[: self._NEAR_MISS_DIAGNOSIS_OUTPUT_MAX_CHARS]
            sections.append("## Test runner output (tail)")
            sections.append(excerpt)
        if candidate_diff:
            diff = candidate_diff[: self._NEAR_MISS_DIAGNOSIS_DIFF_MAX_CHARS]
            sections.append("## Candidate diff")
            sections.append(f"```diff\n{diff}\n```")
        sections.append(
            "## Output format\n"
            "ROOT CAUSE: <one or two sentences>\n"
            "SUGGESTED FIX DIRECTION: <concrete, actionable steps>"
        )
        return "\n".join(sections)

    def _build_near_miss_diagnosis_llm(self) -> Any:
        """Construct the planner/selection LLM client for the diagnosis call.

        Reuses the configured ``llm_configs`` (slot 0, the planner/selection
        profile). Returns ``None`` on any construction failure so the caller
        fails open. CLI/agentic backends are skipped: the diagnosis is a
        single chat completion, not an agent loop.
        """

        try:
            configs = list(getattr(self.config, "llm_configs", None) or [])
        except Exception:  # noqa: BLE001
            return None
        if not configs:
            return None
        chosen = configs[0]
        try:
            if bool(getattr(chosen, "is_cli_backend", False)):
                return None
            from ..core.llm import LLMClient

            return LLMClient(chosen)
        except Exception:  # noqa: BLE001
            logger.debug("near-miss diagnosis LLM client construction failed", exc_info=True)
            return None

    @staticmethod
    def _near_miss_has_precise_residual_repair_signal(
        anchor_quick_verification: dict[str, Any],
    ) -> bool:
        """Return true when a near-miss should be handled as a surgical residual.

        Full expected coverage plus a tiny explicit failing-test set means the
        residual test path is stronger evidence than anchor-wide changed files
        or generic implementation markers. Missing expected IDs, errors, or a
        broad failure count still need the wider recovery prompt.
        """

        if not isinstance(anchor_quick_verification, dict):
            return False
        failed_tests = [
            str(item).strip()
            for item in list(anchor_quick_verification.get("failed_tests") or [])
            if str(item).strip()
        ]
        if not failed_tests or len(failed_tests) > 3:
            return False

        def _as_int(value: Any) -> int:
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0

        failed_count = _as_int(anchor_quick_verification.get("failed"))
        error_count = _as_int(anchor_quick_verification.get("errors"))
        missing_count = _as_int(anchor_quick_verification.get("missing_expected_test_count"))
        if error_count > 0 or missing_count > 0:
            return False
        if failed_count > max(3, len(failed_tests)):
            return False

        expected_count = _as_int(anchor_quick_verification.get("expected_test_count"))
        matched_count = _as_int(anchor_quick_verification.get("matched_expected_test_count"))
        if expected_count > 0 and matched_count > 0:
            return matched_count >= expected_count

        coverage_preserved = anchor_quick_verification.get("expected_coverage_preserved")
        if coverage_preserved is True:
            return True
        coverage_ratio = quick_verification_expected_coverage_ratio(anchor_quick_verification)
        return isinstance(coverage_ratio, (int, float)) and float(coverage_ratio) >= 0.999

    # Test-source-bearing extensions across the languages APEX supports
    # (Python, JS / TS, Go, Rust, Java / Kotlin, Ruby, C / C++, C#).
    # Used to gate which test files to read for the residual summary.
    _TEST_FILE_EXTENSIONS = (
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".mjs",
        ".cjs",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".kts",
        ".rb",
        ".cs",
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hpp",
    )

    def _collect_failing_test_source(
        self,
        anchor: RolloutResult,
        failed_tests: list[str],
    ) -> list[tuple[str, str]]:
        """Read the source of failing test files from the anchor worktree.

        Returns ``[(rel_path, source), ...]`` truncated to a few files /
        a small total budget so the prompt stays compact. Best-effort:
        any read or parse error simply drops that file from the result.

        The function is multi-language: it accepts test files in any
        of the languages APEX supports (see ``_TEST_FILE_EXTENSIONS``).
        For each test file it tries to extract a focused snippet of
        the failing test (Python ``def``, JS / TS ``test()``/``it()``,
        Go ``func TestX``, Rust ``#[test] fn``, etc.); if no
        recognisable form matches it falls back to the file head so
        the agent at least sees imports and intent.
        """

        worktree_path = getattr(anchor, "worktree_path", None) or ""
        if not worktree_path:
            return []
        try:
            worktree = Path(worktree_path)
            if not worktree.exists() or not worktree.is_dir():
                return []
        except OSError:
            return []
        seen_files: set[str] = set()
        results: list[tuple[str, str]] = []
        budget = self._NEAR_MISS_TEST_SOURCE_MAX_CHARS
        for test_id in failed_tests:
            if len(results) >= self._NEAR_MISS_TEST_SOURCE_MAX_FILES or budget <= 0:
                break
            text = str(test_id or "").strip()
            if not text:
                continue
            file_path, _, test_function = text.partition("::")
            file_path = file_path.strip()
            if not file_path or file_path in seen_files:
                continue
            if not any(file_path.lower().endswith(ext) for ext in self._TEST_FILE_EXTENSIONS):
                continue
            seen_files.add(file_path)
            abs_path = worktree / file_path
            try:
                source = abs_path.read_text(errors="replace")
            except OSError:
                continue
            extracted = self._extract_test_function_source(
                source,
                test_function or "",
                file_path=file_path,
                max_chars=min(budget, 600),
            )
            if extracted is None:
                continue
            results.append((file_path, extracted))
            budget -= len(extracted)
        return results

    @staticmethod
    def _summarize_missing_expected_test_groups(
        missing_test_ids: list[Any],
    ) -> tuple[list[str], bool]:
        """Group missing expected IDs by visible test function."""

        counts: Counter[str] = Counter()
        has_parametrized_ids = False
        for raw in missing_test_ids:
            text = str(raw or "").strip()
            if not text:
                continue
            base = text
            if "[" in base:
                has_parametrized_ids = True
                base = base.split("[", 1)[0]
            counts[base] += 1
        grouped: list[str] = []
        for base, count in counts.most_common(6):
            label = f"{base} ({count} missing case"
            if count != 1:
                label += "s"
            label += ")"
            if has_parametrized_ids and count > 1:
                label += " — likely missing parametrized variants from one visible test"
            grouped.append(label)
        return grouped, has_parametrized_ids

    @staticmethod
    def _extract_test_function_source(
        module_source: str,
        function_name: str,
        *,
        max_chars: int,
        file_path: str = "",
    ) -> Optional[str]:
        """Pull the source of one test function plus the module's imports.

        Multi-language support: tries language-appropriate signatures
        based on file extension (Python ``def``/``async def``,
        Go ``func``, Rust ``fn``, Java / Kotlin ``void`` methods,
        JS / TS ``function``/``test``/``it``, Ruby ``def``). If no
        signature matches, returns the first ``max_chars`` characters
        of the module so the agent at least sees imports and intent.
        Pure substring match — never executes user-controlled code.
        """

        if not module_source:
            return None
        clean_function_name = (function_name or "").strip()
        if "[" in clean_function_name:
            clean_function_name = clean_function_name.split("[", 1)[0]
        position = -1
        if clean_function_name:
            ext = ""
            if file_path:
                lowered = file_path.lower()
                dot = lowered.rfind(".")
                if dot != -1:
                    ext = lowered[dot:]
            # Candidate signature prefixes per language. Order matters —
            # we want the most-specific patterns first.
            candidates: list[str] = []
            if ext in {".py"} or not ext:
                candidates.append(f"def {clean_function_name}(")
                candidates.append(f"async def {clean_function_name}(")
            if ext in {".go"}:
                candidates.append(f"func {clean_function_name}(")
            if ext in {".rs"}:
                candidates.append(f"fn {clean_function_name}(")
                candidates.append(f"async fn {clean_function_name}(")
            if ext in {".java", ".kt", ".kts"}:
                candidates.append(f"void {clean_function_name}(")
                candidates.append(f"fun {clean_function_name}(")
            if ext in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
                # JS / TS test frameworks express tests as quoted strings,
                # not function names. Match either the quoted-name form
                # (jest / vitest / mocha: ``test('name', ...)``,
                # ``it('name', ...)``, ``describe('name', ...)``) or a
                # named function declaration.
                candidates.append(f"function {clean_function_name}(")
                for keyword in ("test", "it", "describe"):
                    candidates.append(f"{keyword}('{clean_function_name}'")
                    candidates.append(f'{keyword}("{clean_function_name}"')
                    candidates.append(f"{keyword}(`{clean_function_name}`")
            if ext in {".rb"}:
                candidates.append(f"def {clean_function_name}")
            if ext in {".cs"}:
                candidates.append(f"void {clean_function_name}(")
            if ext in {".c", ".cc", ".cpp", ".h", ".hpp"}:
                candidates.append(f"void {clean_function_name}(")
                candidates.append(f"int {clean_function_name}(")
                candidates.append(f"TEST({clean_function_name}")  # Google Test
                candidates.append(f"TEST_F({clean_function_name}")
            for marker in candidates:
                hit = module_source.find(marker)
                if hit != -1:
                    position = hit
                    break
        if position != -1:
            line_start = module_source.rfind("\n", 0, position) + 1
            imports_block = ApexOrchestrator._extract_module_imports_block(
                module_source[:line_start]
            )
            budget = max(0, max_chars - len(imports_block))
            snippet = module_source[line_start : line_start + budget]
            combined = (imports_block + snippet).rstrip()
            return combined
        return module_source[:max_chars].rstrip()

    @staticmethod
    def _extract_module_imports_block(prefix: str) -> str:
        """Return the leading import / from / docstring block of a module."""

        if not prefix:
            return ""
        lines = prefix.splitlines(keepends=True)
        kept: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                kept.append(line)
                continue
            if (
                stripped.startswith("import ")
                or stripped.startswith("from ")
                or stripped.startswith("#")
                or stripped.startswith('"""')
                or stripped.startswith("'''")
            ):
                kept.append(line)
                continue
            # First non-import / non-comment line — stop.
            break
        block = "".join(kept).rstrip("\n")
        if block and not block.endswith("\n"):
            block += "\n\n"
        return block

    _NEAR_MISS_DIFF_MAX_CHARS = 4000
    _NEAR_MISS_TRACEBACK_MAX_HITS = 6

    def _collect_anchor_git_diff(self, anchor: RolloutResult) -> str:
        """Return the anchor worktree's diff vs. its baseline commit.

        The diff is *exactly* what the prior rollout produced; passing it
        into the next rollout's prompt means the agent can see what
        edits to KEEP and what files it has already touched. Without
        this, repeated rollouts on stuck completion tasks keep producing
        similar but not identical patches because the agent
        has no anchor for "what work is already done." Bounded to
        ``_NEAR_MISS_DIFF_MAX_CHARS`` so the prompt stays manageable.
        """

        worktree_path = getattr(anchor, "worktree_path", None) or ""
        baseline_commit = getattr(anchor, "baseline_commit", None) or ""
        if not worktree_path or not baseline_commit:
            return ""
        try:
            worktree = Path(worktree_path)
            if not worktree.exists():
                return ""
        except OSError:
            return ""
        try:
            import subprocess

            completed = subprocess.run(
                ["git", "diff", "--no-color", "--unified=3", str(baseline_commit)],
                cwd=str(worktree),
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        if completed.returncode != 0:
            return ""
        diff = (completed.stdout or "").strip()
        if not diff:
            return ""
        if len(diff) > self._NEAR_MISS_DIFF_MAX_CHARS:
            diff = (
                diff[: self._NEAR_MISS_DIFF_MAX_CHARS]
                + f"\n... (truncated; full diff is {len(diff)} chars)"
            )
        return diff

    def _extract_traceback_terminals(
        self,
        excerpt: str,
        anchor: RolloutResult,
    ) -> list[tuple[str, int, str]]:
        """Pull (file, line, exception_label) tuples from a test excerpt.

        Most language test runners surface a terminal frame of the form
        ``relative/path.<ext>:<line>: <ErrorClass>`` (pytest, rspec,
        rust panics, jest snapshot diffs, etc.). Surfacing these gives
        the agent a direct file:line jump-to-fix anchor instead of
        making it re-read the whole stack. We restrict the search to
        files that exist inside the anchor's worktree so we don't pull
        in references to ``site-packages`` / ``node_modules`` paths
        that the agent cannot edit. Languages whose stack traces use a
        different shape (Go's ``+0xNN``, Java's ``at pkg.Foo(File:L)``)
        are simply not matched here — they still flow through other
        signals like ``_collect_implementation_markers``.
        """

        if not excerpt:
            return []
        worktree_path = getattr(anchor, "worktree_path", None) or ""
        try:
            worktree = Path(worktree_path) if worktree_path else None
        except OSError:
            worktree = None
        terminals: list[tuple[str, int, str]] = []
        seen: set[tuple[str, int]] = set()
        # Pattern matches: ``relative/path.<lang-ext>:42: ExceptionName``
        # plus optional message tail. Filters out ``site-packages``,
        # ``node_modules``, ``.runtime``, and absolute system paths
        # automatically: only paths that exist inside the worktree
        # are accepted.
        pattern = re.compile(
            r"([A-Za-z0-9_./\\\\-]+\.(?:py|pyx|pyi|js|jsx|ts|tsx|mjs|cjs|"
            r"go|rs|java|kt|kts|scala|rb|cs|c|cc|cpp|cxx|h|hpp|hxx|swift|"
            r"php|ex|exs)):(\d+):\s+"
            r"([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning|Panic|Failure))"
            r"(?::\s*(.+))?"
        )
        for line in excerpt.splitlines():
            for match in pattern.finditer(line):
                rel_path = match.group(1).strip()
                if (
                    "site-packages" in rel_path
                    or "node_modules" in rel_path
                    or ".runtime" in rel_path
                    or "vendor/" in rel_path
                    or "/target/" in rel_path
                ):
                    continue
                if rel_path.startswith("/") or re.match(r"^[A-Za-z]:", rel_path):
                    continue
                try:
                    line_no = int(match.group(2))
                except ValueError:
                    continue
                exception = match.group(3).strip()
                tail = (match.group(4) or "").strip()
                if worktree is not None and not (worktree / rel_path).exists():
                    continue
                key = (rel_path, line_no)
                if key in seen:
                    continue
                seen.add(key)
                label = f"{exception}: {tail}" if tail else exception
                terminals.append((rel_path, line_no, label[:200]))
                if len(terminals) >= self._NEAR_MISS_TRACEBACK_MAX_HITS:
                    return terminals
        return terminals

    # Stub / fix-me markers across the languages APEX supports.
    # Grouped by surface so the pattern bank stays auditable:
    #   * Python / Ruby   — ``raise NotImplementedError`` and ``#`` comments
    #   * Rust            — ``unimplemented!()``, ``todo!()``, ``panic!("todo")``
    #   * Go              — ``panic("not implemented")``, ``errors.New(... not implemented)``
    #   * JS / TS         — ``throw new Error("not implemented")``, ``// TODO``
    #   * Java / Kotlin / C# — ``throw new UnsupportedOperationException``,
    #                          ``throw new NotImplementedException``, Kotlin ``TODO()``
    #   * C / C++ / generic C-style — ``// TODO``, ``/* TODO */``
    _IMPL_MARKER_PATTERNS = (
        # Python / Ruby raise-style
        re.compile(r"raise\s+NotImplementedError"),
        re.compile(r"raise\s+NotImplemented\b"),
        # Python ``pass # TODO`` placeholder body
        re.compile(r"^\s*pass\s*#\s*TODO", re.IGNORECASE | re.MULTILINE),
        # Hash-comment markers (Python, Ruby, shell, YAML, etc.)
        re.compile(r"#\s*TODO[:\s]"),
        re.compile(r"#\s*FIXME[:\s]"),
        re.compile(r"#\s*XXX[:\s]"),
        re.compile(r"#\s*BUG[:\s]"),
        re.compile(r"#\s*Need to implement", re.IGNORECASE),
        re.compile(r"#\s*Implement this", re.IGNORECASE),
        # C-style line + block comment markers (JS / TS / Go / Rust /
        # Java / Kotlin / Scala / C / C++ / C# / Swift / PHP)
        re.compile(r"//\s*TODO[:\s]"),
        re.compile(r"//\s*FIXME[:\s]"),
        re.compile(r"//\s*XXX[:\s]"),
        re.compile(r"//\s*BUG[:\s]"),
        re.compile(r"//\s*Need to implement", re.IGNORECASE),
        re.compile(r"//\s*Implement this", re.IGNORECASE),
        re.compile(r"/\*\s*TODO[:\s\*]", re.IGNORECASE),
        re.compile(r"/\*\s*FIXME[:\s\*]", re.IGNORECASE),
        # Rust stub macros
        re.compile(r"\bunimplemented!\s*\("),
        re.compile(r"\btodo!\s*\("),
        re.compile(
            r"\bpanic!\s*\(\s*[\"'][^\"']*(?:not\s+implemented|todo|unimplemented)", re.IGNORECASE
        ),
        # Go stub patterns
        re.compile(
            r"\bpanic\s*\(\s*[\"'][^\"']*(?:not\s+implemented|todo|unimplemented)", re.IGNORECASE
        ),
        re.compile(
            r"errors\.New\s*\(\s*[\"'][^\"']*(?:not\s+implemented|todo|unimplemented)",
            re.IGNORECASE,
        ),
        re.compile(
            r"fmt\.Errorf\s*\(\s*[\"'][^\"']*(?:not\s+implemented|todo|unimplemented)",
            re.IGNORECASE,
        ),
        # JS / TS stub patterns
        re.compile(
            r"throw\s+new\s+Error\s*\(\s*[\"'`][^\"'`]*(?:not\s+implemented|todo|unimplemented)",
            re.IGNORECASE,
        ),
        # Java / C# / Kotlin stub patterns
        re.compile(r"throw\s+new\s+UnsupportedOperationException"),
        re.compile(r"throw\s+new\s+NotImplementedException"),
        # Kotlin built-in stub
        re.compile(r"\bTODO\s*\(\s*[\"']"),
    )

    # Source-file extensions for marker scanning. Broader than
    # ``_TEST_FILE_EXTENSIONS`` so it covers implementation-only
    # languages encountered in SWE-Bench Pro and similar benches.
    _SOURCE_FILE_EXTENSIONS = (
        ".py",
        ".pyx",
        ".pyi",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".mjs",
        ".cjs",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".kts",
        ".scala",
        ".rb",
        ".cs",
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hpp",
        ".hxx",
        ".swift",
        ".php",
        ".ex",
        ".exs",
    )

    def _collect_implementation_markers(
        self,
        anchor: RolloutResult,
    ) -> list[tuple[str, int, str]]:
        """Scan changed source files for explicit fix-me markers.

        Returns up to ``_NEAR_MISS_MARKER_MAX_HITS`` entries of
        ``(rel_path, line_number, stripped_line_text)``. The patterns
        cover the common stub / fix-me surfaces across all supported
        languages: ``raise NotImplementedError`` (Python / Ruby),
        ``unimplemented!()`` / ``todo!()`` / ``panic!("todo")`` (Rust),
        ``throw new Error("not implemented")`` (JS / TS),
        ``throw new UnsupportedOperationException`` (Java / Kotlin),
        ``panic("not implemented")`` / ``errors.New("not implemented")``
        (Go), Kotlin's built-in ``TODO("...")``, and both ``#`` and
        ``//`` style ``TODO`` / ``FIXME`` comments.

        SWE-Bench Pro uses repos in JS / TS / Go / Rust / Java / Ruby in
        addition to Python. The marker bank is wide enough that any of
        those repos get a useful signal here.
        """

        worktree_path = getattr(anchor, "worktree_path", None) or ""
        if not worktree_path:
            return []
        try:
            worktree = Path(worktree_path)
            if not worktree.exists():
                return []
        except OSError:
            return []
        hits: list[tuple[str, int, str]] = []
        for rel_path in list(anchor.changed_files or []):
            if len(hits) >= self._NEAR_MISS_MARKER_MAX_HITS:
                break
            if not any(rel_path.lower().endswith(ext) for ext in self._SOURCE_FILE_EXTENSIONS):
                continue
            abs_path = worktree / rel_path
            if not abs_path.is_file():
                continue
            try:
                source_lines = abs_path.read_text(errors="replace").splitlines()
            except OSError:
                continue
            for line_index, line in enumerate(source_lines):
                if len(hits) >= self._NEAR_MISS_MARKER_MAX_HITS:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                if any(pattern.search(line) for pattern in self._IMPL_MARKER_PATTERNS):
                    hits.append((rel_path, line_index + 1, stripped[:160]))
        return hits
