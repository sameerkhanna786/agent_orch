"""
APEX configuration models.

The runtime is intentionally opinionated:
- OpenAI-compatible LLMs are the primary execution backend
- planner / selector stages can fall back to heuristics
- rollout execution can fall back to a local repair engine when LLM access fails
"""

from __future__ import annotations

import copy
import json
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .cli_tool_hooks import (
    get_cli_tool_hook_support,
    require_independent_cli_reviewer,
)
from ..controller_models import ControllerModelLibraryConfig


class PromptStrategy(str, Enum):
    """Prompt diversity strategies for parallel rollouts."""

    MINIMAL = "minimal"
    COMPREHENSIVE = "comprehensive"
    TEST_DRIVEN = "test_driven"


class SelectionStrategy(str, Enum):
    """Patch selection strategies."""

    AST_CLUSTER = "ast_cluster"
    CROSS_VALIDATE = "cross_validate"
    LLM_JUDGE = "llm_judge"
    MULTI_STAGE = "multi_stage"


class AgentMode(str, Enum):
    """How each rollout should execute the issue-solving workflow."""

    FULL_SOLVER = "full_solver"
    SCAFFOLDED = "scaffolded"
    CLI_AGENT = "cli_agent"
    ADAPTIVE = "adaptive"


class LLMBackend(str, Enum):
    """Execution backend used for a model entry."""

    OPENAI_API = "openai_api"
    CLAUDE_CLI = "claude_cli"
    GEMINI_CLI = "gemini_cli"
    CODEX_CLI = "codex_cli"
    OPENCODE_CLI = "opencode_cli"
    METACODE_CLI = "metacode_cli"


class BenchmarkEvaluationBackend(str, Enum):
    """Backend used for Commit0 benchmark scoring."""

    LOCAL_PYTEST = "local_pytest_json_report"
    OFFICIAL_DOCKER = "commit0_official_local_docker"


class SearchMode(str, Enum):
    """Search policy used for graph-guided rollout allocation."""

    OFF = "off"
    BEST_FIRST = "best_first"
    PUCT = "puct"


class KnowledgeAccessMode(str, Enum):
    """External-knowledge access policy for agentic search."""

    AIR_GAPPED = "air_gapped"
    INTERNET_AWARE = "internet_aware"


SCAFFOLD_STAGE_NAMES = frozenset({"reproducer", "localizer", "patcher", "test_writer"})
ROLLOUT_PROFILE_STAGE_ORDER = ("reproducer", "localizer", "patcher", "test_writer")
ROLLOUT_LLM_PROFILE_BASE_KEY = "rollout"
SUPPORTED_EXECUTION_MODELS = frozenset(
    {
        "gpt-5.5",
        "opus",
        "gemini-3.1-pro",
        "meta/avocado-tester",
        "meta/avocado-code-latest",
    }
)
_CLAUDE_OPUS_EXECUTION_MODEL_ID = "claude-opus-4-8[1m]"
_MODEL_NAME_ALIASES = {
    "gpt-5.5": "gpt-5.5",
    "gpt 5.5": "gpt-5.5",
    "gpt5.5": "gpt-5.5",
    "gpt_5_5": "gpt-5.5",
    "opus": "opus",
    "claude-opus-4-8[1m]": "opus",
    "gemini-3.1-pro": "gemini-3.1-pro",
    "gemini 3.1 pro": "gemini-3.1-pro",
    "gemini_3_1_pro": "gemini-3.1-pro",
    "gemini3.1pro": "gemini-3.1-pro",
    "avocado": "meta/avocado-tester",
    "avocado-tester": "meta/avocado-tester",
    "meta/avocado-tester": "meta/avocado-tester",
    "avocado-code-latest": "meta/avocado-code-latest",
    "meta/avocado-code-latest": "meta/avocado-code-latest",
    "metacode": "meta/avocado-code-latest",
}
_SUPPORTED_BACKEND_MODELS = {
    LLMBackend.OPENAI_API: frozenset({"gpt-5.5"}),
    LLMBackend.CLAUDE_CLI: frozenset({"opus"}),
    LLMBackend.GEMINI_CLI: frozenset({"gemini-3.1-pro"}),
    LLMBackend.CODEX_CLI: frozenset({"gpt-5.5"}),
    LLMBackend.OPENCODE_CLI: frozenset({"meta/avocado-tester"}),
    LLMBackend.METACODE_CLI: frozenset({"meta/avocado-code-latest"}),
}
_DEFAULT_MODEL_BY_BACKEND = {
    LLMBackend.OPENAI_API: "gpt-5.5",
    LLMBackend.CLAUDE_CLI: "opus",
    LLMBackend.GEMINI_CLI: "gemini-3.1-pro",
    LLMBackend.CODEX_CLI: "gpt-5.5",
    LLMBackend.OPENCODE_CLI: "meta/avocado-tester",
    LLMBackend.METACODE_CLI: "meta/avocado-code-latest",
}
_LLM_CONFIG_KEEP = object()
_CLI_PERMISSION_MODES_BY_BACKEND = {
    LLMBackend.CODEX_CLI: frozenset({"never", "on-request", "on-failure", "untrusted"}),
    LLMBackend.CLAUDE_CLI: frozenset({"default", "acceptEdits", "bypassPermissions", "plan"}),
    LLMBackend.GEMINI_CLI: frozenset({"default", "yolo", "auto_edit", "suggest"}),
}


def normalize_supported_model_name(model: Any) -> str:
    token = str(model).strip().lower()
    canonical = _MODEL_NAME_ALIASES.get(token)
    if canonical:
        return canonical
    supported = ", ".join(sorted(SUPPORTED_EXECUTION_MODELS))
    raise ValueError(f"Unsupported model '{model}'. APEX only supports: {supported}.")


def _validate_backend_model_pair(backend: LLMBackend, model: str) -> None:
    supported_models = _SUPPORTED_BACKEND_MODELS.get(backend)
    if not supported_models:
        supported_backends = ", ".join(sorted(item.value for item in _SUPPORTED_BACKEND_MODELS))
        raise ValueError(
            f"Unsupported backend '{backend.value}'. APEX only supports: {supported_backends}."
        )
    if model not in supported_models:
        supported = ", ".join(sorted(supported_models))
        raise ValueError(
            f"Model '{model}' is not supported for backend '{backend.value}'. "
            f"Allowed models for this backend: {supported}."
        )


def _validate_cli_permission_mode_pair(
    backend: LLMBackend,
    permission_mode: Optional[str],
) -> None:
    mode = str(permission_mode or "").strip()
    if not mode:
        return
    supported_modes = _CLI_PERMISSION_MODES_BY_BACKEND.get(backend)
    if supported_modes is None:
        return
    if mode not in supported_modes:
        allowed = ", ".join(sorted(supported_modes))
        raise ValueError(
            f"CLI permission mode '{mode}' is not valid for backend "
            f"'{backend.value}'. Allowed modes: {allowed}."
        )


@dataclass
class LLMConfig:
    """Configuration for one LLM backend."""

    backend: LLMBackend = LLMBackend.OPENAI_API
    model: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 8192
    timeout: int = 120
    cli_command: Optional[str] = None
    cli_args: list[str] = field(default_factory=list)
    cli_model_id: Optional[str] = None
    cli_timeout: int = 1200
    cli_hard_timeout_seconds: Optional[int] = None
    cli_health_probe_timeout_seconds: Optional[int] = None
    cli_target_runtime_warmup_timeout_seconds: Optional[int] = None
    # When True, hard_timeout fires on wall-clock alone (no progress-grace).
    # The default behavior allows trickle output (heartbeats / retry
    # messages) to keep pushing back the kill, which can let codex hang for
    # 30+ minutes despite hard_timeout being set. Set this to True for
    # batch-eval workflows where a strict per-call wall-clock cap matters.
    cli_strict_hard_timeout: bool = False
    # Progress-based liveness (K2): the CLI watchdog's uniform stall window and
    # the in-flight-LLM-request (S7) ceiling. These mirror
    # ``RolloutConfig.stall_window_seconds`` /
    # ``RolloutConfig.max_inflight_request_seconds`` and are propagated onto the
    # per-rollout ``LLMConfig`` by the engine so the watchdog (which only sees
    # an ``LLMConfig``) can resolve them. Generous defaults that match the
    # RolloutConfig defaults so a bare LLMConfig still gets liveness semantics.
    cli_stall_window_seconds: int = 1200
    cli_max_inflight_request_seconds: int = 1800
    # Streaming CLI startup liveness. Some agentic CLIs can hang during
    # bootstrap before emitting their first machine-readable event while still
    # burning CPU; CPU-only liveness must not mask a dead pre-agent handshake.
    # ``None`` uses the backend default, ``0`` disables the first-output gate.
    cli_first_output_timeout_seconds: Optional[int] = None
    # No-edit-progress window (token-runaway governor). Re-armed (was 0/disabled):
    # reaps a session that produces NO stdout, NO worktree edit, and NO
    # in-container activity for this window while burning host CPU — i.e. a silent
    # spin (observed: multi-hour claude rollouts with stdout=0 / 0 edits). It
    # ignores host CPU on purpose, and now that Claude streams (stream-json) a
    # genuinely-working agent refreshes the clock via stdout, so this never
    # false-kills real work; it is a no-PROGRESS reaper, NOT a max-wall-clock cap.
    # Mirrors ``RolloutConfig.no_edit_progress_window_seconds``; the engine
    # propagates the rollout value onto this field.
    cli_no_edit_progress_window_seconds: int = 1800
    cli_output_capture_max_chars: int = 16 * 1024 * 1024
    cli_disable_osx_sandbox: bool = True
    cli_permission_mode: Optional[str] = None
    cli_env_overrides: dict[str, str] = field(default_factory=dict)
    # Researcher debug knob: when True, host secret redaction is bypassed for
    # CLI subprocesses (and for the ACI bash tool when wired through). A
    # warning is emitted on every subprocess spawn so the bypass is visible.
    cli_env_redaction_disabled: bool = False
    # Opt-in native CLI hook insertion for agentic tool-call review. The
    # reviewer command must invoke a different agentic CLI family than the
    # actor backend so Codex is not reviewing Codex, Claude is not reviewing
    # Claude, etc.
    cli_tool_review_enabled: bool = False
    cli_tool_review_reviewer_backend: Optional[str] = None
    cli_tool_review_reviewer_command: Optional[str] = None
    cli_tool_review_timeout_seconds: int = 60

    def __post_init__(self) -> None:
        if not isinstance(self.backend, LLMBackend):
            self.backend = LLMBackend(self.backend)
        if not self.model:
            default_model = _DEFAULT_MODEL_BY_BACKEND.get(self.backend)
            if not default_model:
                supported_backends = ", ".join(
                    sorted(item.value for item in _SUPPORTED_BACKEND_MODELS)
                )
                raise ValueError(
                    f"Unsupported backend '{self.backend.value}'. "
                    f"APEX only supports: {supported_backends}."
                )
            self.model = default_model
        self.model = normalize_supported_model_name(self.model)
        _validate_backend_model_pair(self.backend, self.model)
        if self.cli_permission_mode is not None:
            self.cli_permission_mode = str(self.cli_permission_mode).strip() or None
        _validate_cli_permission_mode_pair(self.backend, self.cli_permission_mode)
        if self.cli_model_id is not None:
            self.cli_model_id = str(self.cli_model_id).strip() or None
        if self.cli_health_probe_timeout_seconds is not None:
            self.cli_health_probe_timeout_seconds = int(self.cli_health_probe_timeout_seconds)
        if self.cli_target_runtime_warmup_timeout_seconds is not None:
            self.cli_target_runtime_warmup_timeout_seconds = int(
                self.cli_target_runtime_warmup_timeout_seconds
            )
        self.cli_output_capture_max_chars = max(1024, int(self.cli_output_capture_max_chars))
        self.cli_tool_review_enabled = bool(self.cli_tool_review_enabled)
        if self.cli_tool_review_reviewer_backend is not None:
            self.cli_tool_review_reviewer_backend = (
                str(self.cli_tool_review_reviewer_backend).strip() or None
            )
        if self.cli_tool_review_reviewer_command is not None:
            self.cli_tool_review_reviewer_command = (
                str(self.cli_tool_review_reviewer_command).strip() or None
            )
        self.cli_tool_review_timeout_seconds = max(
            1,
            int(self.cli_tool_review_timeout_seconds or 60),
        )
        if self.cli_tool_review_enabled:
            support = get_cli_tool_hook_support(self.backend)
            if not support.supports_direct_pre_tool_hook:
                raise ValueError(
                    f"CLI tool-call review cannot be injected into '{self.backend.value}': "
                    f"{support.notes or 'no native pre-tool hook contract is registered'}"
                )
            if not self.cli_tool_review_reviewer_backend:
                raise ValueError(
                    "cli_tool_review_reviewer_backend is required when "
                    "cli_tool_review_enabled is true."
                )
            if not self.cli_tool_review_reviewer_command:
                raise ValueError(
                    "cli_tool_review_reviewer_command is required when "
                    "cli_tool_review_enabled is true."
                )
            require_independent_cli_reviewer(
                actor_backend=self.backend,
                reviewer_backend=self.cli_tool_review_reviewer_backend,
            )

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    @property
    def is_cli_backend(self) -> bool:
        return self.backend in {
            LLMBackend.CLAUDE_CLI,
            LLMBackend.GEMINI_CLI,
            LLMBackend.CODEX_CLI,
            LLMBackend.OPENCODE_CLI,
            LLMBackend.METACODE_CLI,
        }

    @property
    def is_agentic_backend(self) -> bool:
        """True when each invocation runs its own multi-turn agent loop
        (reads files, multi-LLM-call internally, decides when to stop)
        rather than performing a single LLM completion.

        All Meta CLI backends today are agentic. Setting this on a config
        flips testgen-pipeline defaults (single candidate, single repair
        attempt, skip post-hoc deterministic helpers) because the agent
        already iterates internally and can validate its own output. Set
        manually only if you point a CLI backend at a non-agentic
        underlying CLI; default ``backend in {CODEX,CLAUDE,GEMINI,OPENCODE}``
        is the right answer for today.
        """
        return self.backend in {
            LLMBackend.CLAUDE_CLI,
            LLMBackend.GEMINI_CLI,
            LLMBackend.CODEX_CLI,
            LLMBackend.OPENCODE_CLI,
            LLMBackend.METACODE_CLI,
        }

    @property
    def resolved_cli_command(self) -> str:
        if self.cli_command:
            return self.cli_command
        defaults = {
            LLMBackend.CLAUDE_CLI: "claude",
            LLMBackend.GEMINI_CLI: "gemini",
            LLMBackend.CODEX_CLI: "codex",
            LLMBackend.OPENCODE_CLI: "opencode",
            LLMBackend.METACODE_CLI: "metacode",
        }
        return defaults.get(self.backend, "")

    @property
    def resolved_cli_model(self) -> Optional[str]:
        if self.cli_model_id:
            return self.cli_model_id
        if self.backend == LLMBackend.CLAUDE_CLI and self.model == "opus":
            return _CLAUDE_OPUS_EXECUTION_MODEL_ID
        return self.model

    @property
    def cli_available(self) -> bool:
        if not self.is_cli_backend:
            return False
        return shutil.which(self.resolved_cli_command) is not None


@dataclass
class ShadowPolicyConfig:
    """Counterfactual logging settings for major controller decisions."""

    enabled: bool = True
    max_logged_options: int = 3


@dataclass
class ControllerTraceConfig:
    """Unified controller decision trace settings."""

    enabled: bool = True
    filename: str = "controller_decisions.jsonl"
    max_options: int = 6


@dataclass
class RegimePolicyConfig:
    """Config-backed evidence weights for generic task-regime inference."""

    completion_task_patterns: list[str] = field(
        default_factory=lambda: [
            "intentionally incomplete",
            "implement missing",
            "missing functionality",
            "repository completion",
            "complete the implementation",
            "complete the library",
            "fill in the missing",
            "missing library functionality",
            "long-horizon repository completion",
        ]
    )
    public_api_patterns: list[str] = field(
        default_factory=lambda: [
            "public api",
            "api contract",
            "backward compatible",
            "without changing the public api",
            "callers",
            "library behavior",
            "contract",
        ]
    )
    probability_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "importability_blocker": 0.55,
            "contract_gap": 0.5,
            "broad_regression": 0.45,
            "high_interface_risk": 0.45,
        }
    )
    state_scales: dict[str, float] = field(
        default_factory=lambda: {
            "importability_blocker": 1.0,
            "contract_gap": 0.95,
            "broad_regression": 1.0,
            "high_interface_risk": 0.9,
        }
    )
    evidence_weights: dict[str, float] = field(
        default_factory=lambda: {
            "completion_pattern": 0.55,
            "public_api_pattern": 0.45,
            "public_api_contract": 0.20,
            "collection_error_cluster": 0.55,
            "zero_passing_with_traceback": 0.35,
            "terminal_source_focus": 0.20,
            "incomplete_source_scaffold": 0.45,
            "incomplete_test_scaffold": 0.25,
            "failing_test_breadth": 0.40,
            "mixed_pass_fail_surface": 0.22,
            "relevant_file_breadth": 0.20,
            "interface_symbol_signal": 0.40,
            "multi_module_focus": 0.20,
            "coverage_preservation_invariant": 0.18,
        }
    )


@dataclass
class DelegationPolicyConfig:
    """Planner/delegation policy thresholds that can later be calibrated offline."""

    split_confidence_threshold: float = 0.60
    boundary_pressure_threshold: int = 2
    bridge_cross_ratio: float = 0.45
    bridge_weight_min: float = 3.0
    low_leverage_cluster_max_files: int = 2
    low_leverage_cluster_max_work: float = 2.6
    low_leverage_cluster_work_ratio: float = 0.45
    low_leverage_outbound_ratio: float = 0.55
    low_leverage_peer_weight_min: float = 1.0
    low_leverage_confidence_penalty: float = 0.18
    thin_cluster_max_work: float = 1.75
    thin_cluster_work_ratio: float = 0.60
    thin_file_max_lines: int = 12
    thin_file_max_symbols: int = 1
    exhaustive_bisection_max_files: int = 8
    symbol_interface_bonus: float = 0.12
    edit_span_bonus: float = 0.08


@dataclass
class CompletionExecutionPolicyConfig:
    """Model-selection and timeout heuristics for broad-validation tasks."""

    preserve_primary_min_failing_tests: int = 6
    preserve_primary_min_incomplete_sources: int = 3
    preserve_primary_min_focus_tests: int = 4
    preserve_primary_min_focus_test_failures: int = 3
    preserve_primary_min_focus_files: int = 4
    preserve_primary_min_relevant_files: int = 6
    preserve_primary_difficulty_threshold: float = 0.55
    timeout_broad_validation_min_failing_tests: int = 6
    timeout_broad_validation_min_incomplete_files: int = 2
    timeout_broad_validation_min_relevant_files: int = 8
    timeout_extension_seconds: int = 600
    timeout_extra_extension_seconds: int = 300
    timeout_extra_min_failing_tests: int = 12
    timeout_extra_min_incomplete_files: int = 4
    timeout_extra_min_relevant_files: int = 12
    delegated_timeout_multiplier: float = 0.60
    delegated_timeout_min_seconds: int = 900
    delegated_timeout_max_seconds: int = 1800


@dataclass
class OverlapPolicyConfig:
    """Similarity thresholds used by planner and scheduler overlap policies."""

    source_overlap_threshold: float = 0.70
    test_overlap_threshold: float = 0.60
    combined_overlap_threshold: float = 0.68


@dataclass
class TransitionRewardConfig:
    """Reward weights for frontier-search transitions."""

    obligation_delta: float = 0.28
    hypothesis_delta: float = 0.12
    uncertainty_reduction: float = 0.18
    progress: float = 0.17
    quick_feedback: float = 0.13
    alignment: float = 0.10
    patch_bonus: float = 0.20
    cost_penalty_per_300s: float = 0.05
    failure_penalty: float = 0.12


@dataclass
class RolloutConfig:
    """Configuration for parallel rollout execution."""

    num_rollouts: int = 5
    enable_adaptive_allocation: bool = False
    # WS3B (FASTPATH-SPECULATIVE): on an easy task (difficulty below the max),
    # dispatch ONE seed first and accept on an authoritative full-scope pass,
    # else fan out the full wave. Only consulted when enable_adaptive_allocation
    # is False; never lowers the rollout cap (no-cost-reduction). Short-horizon
    # latency win that cannot hurt hard tasks.
    enable_speculative_first_attempt: bool = True
    speculative_first_attempt_max_difficulty: float = 0.25
    # SPEED LEVER (cross-rollout discovery REUSE / warm-start). When True, the
    # FIRST best-of-N rollout to produce a high-confidence reproduction +
    # localization (e.g. the WS3B speculative-first seed) has its read-only
    # discovery artifacts harvested and injected as an ADVISORY warm-start seed
    # into the sibling rollouts, so siblings spend their turns on differentiated
    # SOLVING instead of re-deriving the identical failing-test reproduction and
    # file localization. Strictly advisory (hint, not constraint): siblings keep
    # full freedom to re-localize/re-reproduce and diverge, and the distinct-
    # hypothesis ``localizer_top_k`` fan-out is untouched. Scales DOWN for giants
    # (size_factor >= rollout_budget_max_size_factor keeps today's fully-
    # independent behavior, where independent localization is most load-bearing).
    # Fail-open: any harvest/inject error falls back to today's per-rollout
    # derivation. Pre-solve context only — never gates acceptance, so it adds no
    # false-accept surface. Ablation off => byte-identical to today.
    enable_cross_rollout_discovery_reuse: bool = True
    # WS3E: cross-solve (per-task) episodic memory. When True, the orchestrator
    # loads prior ROOT_CAUSE/DEAD_END/RELEVANT_FILE episodes for the SAME task
    # signature (repo + task_id) and seeds them as decayed priors into the
    # rollout memory bus, then records this solve's terminal outcome for the
    # next attempt. Default OFF — opt-in; complements the per-repo C.3 path.
    enable_cross_solve_episodic_memory: bool = False
    min_rollouts: int = 1
    max_rollouts: int = 16
    rollout_buckets: list[int] = field(default_factory=lambda: [1, 4, 8, 16])
    scaffold_stage_llm_indices: dict[str, int] = field(default_factory=dict)
    llm_profiles: list[dict[str, int]] = field(default_factory=list)
    portfolio_seed_profile_count: int = 0
    portfolio_diversity_include_prompt_strategy: bool = False
    portfolio_diversity_include_temperature: bool = False
    # Profile indices that MUST appear in the seed wave even when the
    # profile_budget would otherwise exclude them. Default ``[0]`` keeps
    # the raw-Codex profile in the mix on every commit0 task — the
    # selection layer needs at least one Codex run to satisfy the
    # raw-Codex floor (see ``_select_best_rollout_candidate``). Set to
    # ``[]`` to disable.
    always_include_profiles: list[int] = field(default_factory=lambda: [0])
    enable_dynamic_cop_transitions: bool = True
    enable_progressive_rollout_allocation: bool = True
    max_progressive_rollout_waves: int = 6
    progressive_stop_on_strong_signal: bool = True
    enable_residual_followup: bool = True
    max_selection_followup_rounds: int = 4
    # Optional cost-bounded experiment guard. The max-quality Apex defaults
    # leave this disabled; set a positive value only for budget-constrained
    # ablations where accepting best-so-far is preferable to more search.
    max_tokens_per_repo_followup: int = 0
    max_followup_iterations: int = 24
    max_iterations_per_rollout: int = 30
    # Small-repo (size_factor == 1) follow-up-round floor for the size-aware
    # patcher cap (see ``_size_aware_followup_round_cap`` in rollout/engine.py).
    # Giants are governed by ``max_iterations_per_rollout`` (30) via
    # ``rollout_budget_max_size_factor`` scaling and are byte-identical to today.
    # Default 3 == "a handful of turns like a plain CLI run" for easy repos that
    # have already converged or plateaued; the loop still self-stops earlier on
    # suite-pass / stall / near-miss, so this only bounds the worst case. One
    # knob to tune the easy-repo floor without any per-repo conditional.
    min_completion_followup_rounds: int = 3
    # SPEED LEVER (rank-4: fold the reproducer SESSION into the localizer for
    # SMALL completion/stub tasks). In the completion regime the reproducer only
    # re-derives the "stubs incomplete / collection blocker" signal the planner
    # already supplies (test_context.incomplete_source_files +
    # allocator_features.is_completion_task), so for the smallest suites we SEED
    # the reproduction artifact (skipping the standalone reproducer session) and
    # let the localizer still run as a full session, receiving the deterministic
    # reproduction summary via the existing build_localizer_prompt path. The
    # localizer's ranking is load-bearing for best-of-N diversity, so it is never
    # collapsed. Size-gated: the merge only activates when the size_factor
    # (``_rollout_budget_size_factor``) is <= this threshold, so for any larger /
    # giant suite (size_factor >= max(2, rollout_budget_max_size_factor)) BOTH the
    # reproducer and localizer sessions still run, byte-identical to today.
    # Default 1 == "smallest suites only"; mirrors ``min_completion_followup_rounds``
    # as the single small-suite knob. Set 0 to disable entirely (provable no-op).
    completion_reproducer_merge_max_size_factor: int = 1
    # SPEED LEVER (rank-5/6: size-proportionate broad re-validation + advisory
    # convergence budget). The solver prompt normally tells the agent to rerun
    # the *broader* repository test command every round ("a narrow slice passing
    # is not sufficient evidence"). On easy repos that re-runs the full suite on
    # every turn even after the targeted scope is green — a large chunk of the
    # observed 300-700 turns. When the size_factor (``_rollout_budget_size_factor``)
    # is <= this threshold AND the round is NOT the final eligible round AND the
    # prior-round quick-verify is NOT a near-pass, ``build_solver_prompt`` switches
    # to ``broad_revalidation_mode="convergence"``: validate broadly on convergence
    # / the final eligible round (still a MANDATORY green broad confirmation run
    # before submit) plus an advisory ``# Convergence Budget`` section. For any
    # larger / giant suite (size_factor >= max(2, rollout_budget_max_size_factor)),
    # the final eligible round, and any prior-round near-pass, the prompt stays
    # ``"continuous"`` and is byte-identical to today. Mirrors
    # ``completion_reproducer_merge_max_size_factor`` as the single small-suite
    # knob. Default 1 == "smallest suites only"; set 0 to disable (provable no-op).
    broad_revalidation_convergence_max_size_factor: int = 1
    # Small-repo (size_factor == 1) near-miss residual-failure tolerance for the
    # patcher continue-loop scheduling signal. Only loosened for the smallest
    # suites (sf == 1); medium/giant suites keep the strict 3-failure cap and
    # must still reach literal final_pass_rate == 1.0. Acceptance is gated by
    # selection/verification, not by this scheduling signal, so this cannot
    # induce a false-accept on any repo.
    small_repo_near_miss_residual_cap: int = 5
    enable_orchestrated_multi_agent: bool = True
    # Max-quality / dominance mode: add one extra rollout that is as close as
    # possible to the strongest standalone agentic CLI baseline. The selector
    # can then preserve this candidate unless another candidate has strictly
    # stronger verifier evidence. Disabled by default so low-budget callers do
    # not silently get an extra rollout; max benchmark configs should enable it.
    enable_standalone_anchor: bool = False
    standalone_anchor_profile_index: int = 0
    standalone_anchor_label: str = "codex_cli:gpt-5.5"
    # Ordered candidate specs for the standalone anchor. Each entry may
    # specify ``backend``, ``model``, optional ``cli_model_id``, optional
    # ``label``, and optional ``harness`` (currently ``"cli_agent"``).
    # The rollout engine picks the first configured LLM matching this
    # ordered list and embeds the exact llm_config index into the anchor
    # brief. This keeps the "strongest standalone CLI" choice explicit and
    # future-proof instead of relying on profile 0 forever.
    standalone_anchor_candidates: list[dict[str, Any]] = field(default_factory=list)
    standalone_anchor_run_all_candidates: bool = False
    standalone_anchor_strict_candidate_match: bool = False
    standalone_anchor_allow_llm_fallback: bool = False
    use_git_worktrees: bool = True
    # When true, snapshot workspaces are materialized as a fresh one-commit git
    # repository from the visible file tree instead of cloning upstream git
    # history. This preserves normal git status/diff for agents while making
    # prior commits physically unavailable in sandboxed solve environments.
    historyless_snapshots: bool = False
    keep_worktrees: bool = True
    # Phase 2.1: when False (the default) the RolloutEngine creates a
    # private per-engine scratch directory (``tempfile.mkdtemp``) so two
    # concurrent ``solve()`` calls in the same parent process can never
    # contend on the same ``rollout_<id>`` slot. Set True when the operator
    # explicitly wants a persistent shared workspace (e.g. for cross-solve
    # caches or checkpoint reuse). File-locking still applies in that case.
    shared_workspace: bool = False
    enable_heuristic_fallback: bool = True
    enable_quick_verification: bool = True
    quick_verification_max_tests: int = 8
    quick_verification_timeout_seconds: int = 180
    quick_verification_full_collection_when_expected_set_known: bool = True
    # Conservative per-test wall-clock estimate used to decide whether a broad
    # quick-verification over the full expected suite would blow the QV budget
    # (T1.1). Only consulted for huge suites; small repos never cross the
    # threshold so this is a no-op for them. Overridable for ecosystems with
    # atypically heavy/cheap tests.
    quick_verification_estimated_seconds_per_test: float = 0.5
    # Lower/upper clamp on the deterministic stratified expected-id sample size
    # used when the full suite is over budget (T1.1). The sample is breadth-first
    # across every test file so candidate ranking still sees every module, but it
    # is structurally a *sample* (missing>0) so it can never prove full coverage.
    quick_verification_sampled_suite_min: int = 200
    quick_verification_sampled_suite_max: int = 2000
    # ------------------------------------------------------------------
    # Progress-based liveness (replaces strict wall-clock timeouts).
    # ------------------------------------------------------------------
    # CORE INVARIANT: a rollout / agent CLI / test process is killed IFF
    # (a) it produced no *meaningful progress* for ``stall_window_seconds``
    # (a deadlock / hung / disconnected worker), or (b) its process is
    # provably dead / disconnected. Elapsed wall-clock alone NEVER kills a
    # progressing agent. ``stall_window_seconds`` is the single uniform
    # stall window applied at all three kill sites (scheduler K1, CLI
    # watchdog K2, per-test-run QV K3). Generous default (20 min) so a fast
    # rollout finishes long before the window — a structural no-op on the
    # normal path; the window only fires on a true total silence.
    stall_window_seconds: int = 1200
    # Stall window for the per-test-run quick-verification poll loop (K3).
    # ``0`` (the default sentinel) means "inherit ``stall_window_seconds``".
    qv_stall_window_seconds: int = 0
    # NO-EDIT-PROGRESS window (token-runaway governor). RE-ARMED (was 0/disabled):
    # an EDIT-CAPABLE CLI stage that for this many seconds shows NO new stdout, NO
    # worktree edit, AND NO in-container (target-runtime) activity — only
    # host-process CPU spin — is reaped as no-meaningful-progress. It was disabled
    # to avoid false-killing the host-invisible in-container agent, but with Claude
    # now STREAMING its turns (stream-json) a genuinely-working agent refreshes the
    # clock via stdout every turn, so the false-kill risk is gone, while the
    # observed pathology — multi-hour claude rollouts with stdout=0 and 0 edits
    # (APEX was blind to them) — is now reaped. This is a no-PROGRESS reaper keyed
    # on output/edit/container signals, NOT a max-wall-clock cap. The 4h emergency
    # silence cap remains the absolute backstop. Set 0 to disable.
    no_edit_progress_window_seconds: int = 1800
    # An in-flight LLM request (S7) holds the watchdog's progress clock
    # frozen (treated ALIVE) while ``process.poll() is None`` and the
    # request marker is set — a multi-minute "thinking" turn is liveness,
    # not inactivity. Bounded by this ceiling so a TCP black-hole (no FIN)
    # or a crashed worker that left the marker set still trips stall timing.
    max_inflight_request_seconds: int = 1800
    # §4 EMERGENCY SILENCE CAP (task-level, the single absolute ceiling).
    # Kill iff ``now - max(last_stdout_at, last_worktree_at) >=`` this
    # window — CPU is EXPLICITLY IGNORED so a silent CPU-spinning livelock
    # is bounded while a working agent that streams output / edits files is
    # never touched. ``0`` => unlimited. Tightened from 24 h to 4 h: this is the
    # coarse task-wide backstop (the granular per-rollout no-edit-progress kill
    # above is the primary tool); 4 h still keeps ~8x headroom over stacked
    # silent agentic turns while cutting worst-case waste from a full day.
    emergency_silence_window_seconds: int = 14400
    # Once a candidate is selection-ready (for example: clean reduced-scope
    # evidence that explicitly requires authoritative scoring), sibling rollouts
    # are optional extra evidence. Give them a short grace window, then proceed
    # to selection so a repo slot is not held indefinitely by unrelated siblings.
    selection_ready_drain_grace_seconds: int = 300
    # ``task_wallclock_budget_seconds`` is REPURPOSED as the task-level
    # emergency-cap anchor (see §4); it is no longer a wall-clock kill on a
    # progressing agent. ``0`` => unlimited.
    task_wallclock_budget_seconds: int = 7200
    # ``rollout_wallclock_budget_seconds`` is advisory / a scheduling hint
    # only under progress-based liveness (default ``0`` => unlimited): K1 no
    # longer compares against it, so it contributes nothing to a kill.
    rollout_wallclock_budget_seconds: int = 0
    # T1.4 size-scaling: legacy size-factor knobs retained only as
    # scheduling / emergency-cap sizing hints. Under progress-based
    # liveness wall-clock is not a kill, so these no longer extend any kill
    # threshold. ``size_factor == 1`` for small repos => identical behavior.
    rollout_budget_tests_per_unit: int = 2000
    rollout_budget_max_size_factor: int = 6
    # ------------------------------------------------------------------
    # TIER 2 decomposition (accumulation architecture). SIZE/STRUCTURE
    # triggered ONLY — small/non-decomposition repos never trip the
    # predicate, so these are no-ops off the giants. No repo/language
    # conditional: every threshold here is a general measured-size knob.
    # ------------------------------------------------------------------
    enable_decomposition_scale_partitioning: bool = True
    # A repo is "decomposition-scale" iff (expected_test_count >= this) OR
    # (repo-wide stub-file count >= ``decomposition_min_stub_files``) OR its
    # inferred scope class is ``library_reconstruction``.
    decomposition_min_expected_tests: int = 4000
    decomposition_min_stub_files: int = 120
    # FILES_PER_GROUP: target owned-file count per module group. The number
    # of partitions is clamp(ceil(stub_files / this), 2, max(num_rollouts, cap)).
    decomposition_files_per_group: int = 25
    decomposition_max_partitions_cap: int = 12
    enable_overlap_diversity_cap: bool = True
    min_overlap_diversity_parallel_workers: int = 1
    overlap_diversity_include_prompt_strategy: bool = False
    overlap_diversity_include_temperature: bool = False
    completion_policy: CompletionExecutionPolicyConfig = field(
        default_factory=CompletionExecutionPolicyConfig
    )
    overlap_policy: OverlapPolicyConfig = field(default_factory=OverlapPolicyConfig)
    shadow_policy: ShadowPolicyConfig = field(default_factory=ShadowPolicyConfig)
    agent_mode: AgentMode = AgentMode.ADAPTIVE
    diversity_temperatures: list[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8])
    diversity_prompts: list[PromptStrategy] = field(
        default_factory=lambda: [
            PromptStrategy.MINIMAL,
            PromptStrategy.COMPREHENSIVE,
            PromptStrategy.TEST_DRIVEN,
            PromptStrategy.COMPREHENSIVE,
            PromptStrategy.TEST_DRIVEN,
        ]
    )
    parallel_workers: int = 3
    # Optional global cap for simultaneously running rollout workers across
    # concurrent benchmark tasks. ``0`` preserves the per-task worker setting.
    global_parallel_worker_budget: int = 0
    # Phase 2C 2.2: opt-in salvage. When False (the default), candidates
    # whose only "acceptance" signal is selector salvage
    # (``salvaged_for_external_scoring=True``, ``internally_accepted=False``)
    # are returned with ``ApexResult.success=False`` and a structured
    # abstain status. When True, salvage candidates surface as success
    # with ``salvaged=True`` for downstream external scoring.
    allow_salvage: bool = False
    # Phase 2C 2.7 / Decisive-Edge B.1: localizer constraint enforcement
    # on ``submit_patch``.
    #
    # ``advisory``       — no check (legacy behaviour).
    # ``warning``        — count off-target patches into diagnostics, allow.
    # ``hard_constraint``— legacy high-severity diagnostic for patches whose
    #                      changed files fall outside the localizer scope plus
    #                      the project-wide allowlist.
    #
    # Localization is a search prior, not validity evidence. General
    # orchestration relies on objective verification and protected-file gates
    # instead of dropping useful source progress solely for being off scope.
    localizer_enforcement: str = "advisory"
    # Files always allowed to be touched even under hard_constraint mode.
    # Tests are always allowed (they're how the agent demonstrates the fix).
    # Decisive-Edge B.1: extended with ``conftest.py``, ``Makefile`` and
    # the ``.github/workflows/*`` glob (the executor's ``allowlist_globs``
    # already accepts glob patterns alongside literal file paths).
    localizer_allowlist_files: list[str] = field(
        default_factory=lambda: [
            "setup.py",
            "pyproject.toml",
            "setup.cfg",
            "tox.ini",
            "MANIFEST.in",
            "requirements.txt",
            "requirements-dev.txt",
            "conftest.py",
            "Makefile",
            ".github/workflows/*",
        ]
    )
    # Phase 3.5: resolve the MASAI bypass in CLI mode.
    #
    # Historically the CLI patcher (codex/claude/gemini/opencode) skipped
    # MASAI's Reproducer + Localizer entirely. The scaffolded path was the
    # only one that ran the three-stage decomposition. CLI agents are
    # arguably the production codegen mode, so bypassing MASAI here drops
    # grounded context that the CLI agent would otherwise have to
    # rediscover at significant token / latency cost.
    #
    # ``cli_agent_use_masai_preround`` controls the new behaviour:
    #
    #   * ``"off"``               — legacy behaviour. CLI agent goes
    #                               directly to patching, MASAI bypassed.
    #   * ``"advisory"``          — run Reproducer + Localizer pre-rounds;
    #                               prefix the CLI agent's prompt with a
    #                               free-form natural-language paragraph
    #                               summarising the suspected files /
    #                               reproduction.
    #   * ``"structured_prompt"`` — run Reproducer + Localizer pre-rounds;
    #                               prefix the CLI agent's prompt with a
    #                               structured "Grounded Context" YAML
    #                               block (reproduction:, localized_files:,
    #                               localized_symbols:, hypotheses:) that
    #                               the agent must acknowledge.
    #
    # Default is ``"structured_prompt"`` (Option A in the Phase 3.5 plan).
    # Flip back to ``"off"`` for ablation / monolith-agent experiments.
    cli_agent_use_masai_preround: str = "structured_prompt"
    # Optional override for the LLM used by the MASAI pre-rounds. When
    # ``None`` (the default), the pre-rounds re-use the CLI agent's own
    # llm_config. When set to a model name, the pre-rounds are routed to
    # that model instead — useful for running cheaper / faster models in
    # the pre-round stages while keeping the patcher on a stronger model.
    # Resolved against the configured ``LLMConfig`` profiles by the
    # rollout engine.
    cli_agent_preround_llm: Optional[str] = None
    # Phase A.4 (Decisive-Edge): strategy-axis diversity. K rollouts now
    # round-robin through K *strategically distinct* approaches rather
    # than K samples of the same approach. Empty list = use the canonical
    # 7-axis library at :data:`apex.rollout.diversity_strategies.STRATEGY_AXES`.
    # See ``apex/rollout/diversity_strategies.py`` for the per-axis
    # prompt prefixes. ``diversity_temperatures`` and ``diversity_prompts``
    # are preserved as tertiary diversity for back-compat; callers that
    # want pure strategy-driven diversity should set
    # ``diversity_temperatures=[0.7]`` and let strategies own the variance.
    diversity_strategies: list[str] = field(default_factory=list)
    # Decisive-Edge B.8 — agent prompt module version.
    #
    #   * ``"v1"``  — the historical free-prose prompts in
    #                 ``apex/agents/prompts.py`` (current default; the
    #                 prompts that produced the published 86.3% Commit0-Lite
    #                 score). Stays default until an A/B confirms v2 wins.
    #   * ``"v2"``  — the structured / strategy-aware rewrite in
    #                 ``apex/agents/prompts_v2.py`` (clearer role cards,
    #                 explicit YAML/JSON output envelopes, few-shot
    #                 examples, reduced ambiguity, strategy-axis aware).
    #
    # Honoured by ``apex/agents/solver.py`` — the agents pull their
    # prompt builders + system prompts from the matching module. Run the
    # ``apex/scripts/ab_prompts.py`` harness on a representative slice
    # before flipping the default.
    prompts_version: str = "v1"
    # Decisive-Edge B.2: opt-in to weak agent CLIs (currently
    # ``opencode_cli``) in default ensembles.
    #
    # The four-agent ensemble historically rolled in opencode for
    # diversity, but headline runs show it consistently underperforms
    # the strong three (codex / claude / gemini) and dilutes selection.
    # The default ensemble in
    # ``apex/_default_generators.default_agent_names`` is now the
    # strong three; opencode remains in
    # ``AGENT_NAME_TO_CONFIG`` so callers that ask for it explicitly
    # still work, and operators who want the four-agent ablation can
    # re-enable it in defaults by setting ``allow_weak_models=True``.
    # ``apex/_default_generators.default_agent_names`` consults this
    # flag.
    allow_weak_models: bool = False
    # Decisive-Edge C.1 — Per-rollout context variation via top-K
    # Localizer hypotheses.
    #
    # When ``> 1``, the MASAI Localizer pre-round produces ``K`` ranked
    # hypotheses (each with its own files / symbols / hypotheses) and
    # rollout ``i`` is dispatched the ``i % K``-th hypothesis. Combined
    # with ``diversity_strategies`` this turns K identical samples into
    # K (strategy, hypothesis) pairs, dramatically increasing the
    # qualitative span of a rollout batch.
    #
    # ``localizer_top_k = 1`` (the default) preserves legacy behaviour
    # — every rollout sees a single shared localization. Benchmark
    # configs may flip this to 4 (one localization per K=4 rollouts).
    localizer_top_k: int = 1
    # WS3H: spawn N fresh-context isolated localization sub-agents and merge their
    # findings by cross-worker agreement (context-isolation as the aging paper's
    # decay mitigation). Default 1 == single worker (current behaviour, inert).
    localization_subagent_fanout: int = 1
    # Decisive-Edge C.5 — Worktree pool: pre-warm a small pool of
    # worktrees per (task, base_commit) and reuse them across rollouts
    # rather than tearing down + recreating between rollouts. Saves
    # roughly 4s of `git worktree add` + `pip install` warmup per
    # rollout on commit0-style tasks.
    #
    # ``use_worktree_pool=True``  — opt in. Falls back to per-rollout
    #                               worktree creation on pool failures.
    # ``worktree_pool_size``      — explicit pool size override. ``0``
    #                               (the default) lets the engine pick
    #                               ``num_rollouts`` so progressive and
    #                               residual follow-ups do not exhaust the
    #                               pool when they temporarily exceed
    #                               ``parallel_workers``.
    use_worktree_pool: bool = True
    worktree_pool_size: int = 0

    # V5 in-container agent activation knobs (Workstream 1). Defaults mirror the
    # orchestrator_in_container_agent module DEFAULT_* so existing behavior is
    # preserved; the ``v5_`` prefix avoids colliding with the legacy
    # ``max_iterations_per_rollout``. These are read at the V5 construction sites
    # (solver._maybe_solve_via_in_container_v5 / modes._invoke_in_container_v5_agent).
    v5_recent_verbatim_turns: int = 3
    v5_stall_repeat_threshold: int = 3
    v5_stall_terminate_cap: int = 5
    v5_patch_verifier_reject_cap: int = 3
    # Difficulty-scaled turn budget (1C): never lower than the resolved base
    # (no-cost-reduction); only RAISE for hard tasks, clamped to the ceiling.
    v5_max_turns_floor: int = 8
    v5_max_turns_ceiling: int = 60


@dataclass
class OrchestrationConfig:
    """Top-level orchestration loop knobs.

    These were previously hard-coded module constants in
    ``apex/orchestrator.py`` (Phase 2C 2.9 / 3.3). They are now config-
    driven so callers can opt into stricter loop guarding or cost caps
    without monkey-patching.
    """

    # Phase 2C 2.9: backstop for ``_execute_with_dynamic_transitions``.
    # The dynamic-transition loop is ``while True`` and depends on the
    # planner returning ``None`` from ``escalate_execution_strategy`` to
    # terminate. A misbehaving planner (or one stuck on a single
    # primitive) could spin forever; the iteration cap fires after this
    # many escalations regardless. Default 20 — generous for any
    # realistic search depth, tight enough to bound an infinite loop to
    # bounded time. Set to 0 to disable the cap entirely (legacy).
    max_strategy_iterations: int = 20
    # Phase 2C 3.3: optional repo-level cumulative-token cap for the
    # progressive-wave aborts (``_cumulative_token_cap_exceeded``). The
    # legacy implementation hard-coded 100M; per the project directive
    # ("never optimize for cost") the default is now ``None`` (no cap).
    # Operators who DO want a cap can set this explicitly.
    repo_token_cap: Optional[int] = None
    # Phase 3.2: previously hard-coded class attributes on
    # ``ApexOrchestrator``. Lifted here so callers can opt into stricter
    # or looser loops without monkey-patching the orchestrator class.
    # Defaults preserve prior behavior exactly.
    repeated_blocker_stop_after: int = 3
    adaptive_followup_near_miss_multiplier: int = 3
    adaptive_followup_near_miss_pass_rate: float = 0.95
    max_coverage_gap_followup_rounds: int = 2
    seed_diversity_overlap_threshold: float = 0.70
    # C': model-driven diagnosis pass that enriches the near-miss repair dossier
    # with an explicit root-cause analysis of the remaining failing tests (diff +
    # failure output), so the targeted repair rollout converts more near-misses
    # (e.g. a candidate one syntax error short of passing). Layer-A general; fails
    # open (falls back to the existing heuristic dossier on any error).
    enable_near_miss_diagnosis: bool = True
    # Agentic CLI diagnosis call; floor at 30min so the agent step is not cut
    # short (fails open on timeout, so the cap only bounds the wait).
    near_miss_diagnosis_timeout_seconds: int = 1800

    # Phase 4A item 4.6: optional cap on the number of test artifacts a
    # single ``default_test_generator`` invocation will keep. ``None``
    # (default) means unbounded — the JSON schema's old maxItems=10 cap
    # was silently truncating large axis-coverage portfolios. Operators
    # who want a token-budget cap can set this explicitly; the trim
    # happens post-hoc with a logged warning so the truncation is
    # visible.
    max_test_artifacts_per_invocation: Optional[int] = None
    # Phase 4A item 4.3: ``enforce_final_acceptance`` retries each
    # failing test up to ``final_acceptance_flake_retries`` times before
    # treating it as deterministic. Flaky tests survive; only
    # consistently-failing tests are candidates for drop. Default 3
    # matches the W3 acceptance retry budget elsewhere in the pipeline.
    final_acceptance_flake_retries: int = 3
    # Drop a deterministic-failing test only if its
    # ``mutation_kill_contribution`` (fraction of mutants the test
    # uniquely kills) is below this threshold. High-mutation-kill tests
    # that fail are flagged as ``oracle_disagreement`` and surfaced in
    # diagnostics rather than silently dropped — they're the strongest
    # signal that either the test or the implementation is wrong.
    final_acceptance_min_mutation_contribution: float = 0.05
    # When the post-drop suite has ``mutation_kill_score`` below this
    # threshold the gate emits ``weak_minimized_artifact`` regardless of
    # whether the suite is non-empty. Replaces the previous bias toward
    # easy-to-pass-but-empty-of-signal suites.
    final_acceptance_weak_artifact_threshold: float = 0.20
    # Phase 4A item 4.7: number of surrogate patches synthesized per
    # ``run_testgen_with_fix(surrogate_patcher=...)`` invocation.
    # Default raised from 4 to 8 — consensus-F2P is noisy at low N and
    # the project directive forbids cost-driven downgrades. Per-surrogate
    # backends round-robin through ``surrogate_oracle_models``.
    surrogate_oracle_n: int = 8
    # Diversify surrogate generation across distinct CLI agent backends
    # (codex / claude / gemini). Different training and tool-use
    # patterns surface different bug shapes — same-model sampling hits
    # the same blind spots N times. Empty list = use a single default
    # backend (legacy behavior).
    surrogate_oracle_models: list[str] = field(
        default_factory=lambda: [
            "codex_cli:gpt-5.5",
            "claude_cli:opus",
            "gemini_cli:gemini-3.1-pro",
        ]
    )
    # Phase 4.5: per-mutant wall-clock timeout for in-loop mutation
    # sensitivity (``mutation_engine.evaluate_mutation_sensitivity_in_loop``).
    # Bumped from the legacy 15s to 30s — the legacy budget caused slow
    # imports (django, pandas, ansible test discovery) to time out and
    # get mis-scored as "survived". A higher cap is safe because timed-
    # out mutants now record ``status="timeout"`` and are excluded from
    # the score (with a quality_concern_high_timeout_rate flag if the
    # timeout fraction exceeds 20%).
    mutation_per_mutant_timeout_seconds: float = 30.0
    # Phase 6.3: calibrated abstention threshold. After the strict
    # acceptance gate decides SOLVED, the calibrated ConfidenceScorer
    # weights verifier_strength + cluster_consensus + controller policy
    # certainty + mutation_kill + f2p_consensus and (when the aggregate
    # is below this threshold AND ``rollout.allow_salvage`` is False)
    # downgrades the outcome to ABSTAINED. 0.50 is the literature-informed
    # default — operators can tighten or loosen via OrchestrationConfig.
    abstention_threshold: float = 0.50
    # Optional per-component overrides for the ConfidenceScorer weights.
    # Keys must match :data:`apex.orchestration.abstention.DEFAULT_ABSTENTION_WEIGHTS`.
    # ``None`` (default) uses the literature-informed defaults; a dict
    # overrides the listed components and leaves the rest at default.
    abstention_weights: Optional[dict[str, float]] = None
    # Phase B.6 (Decisive-Edge): cap on
    # :class:`apex.capabilities.active_learning.MutationActiveLearner`
    # iterations. Default 2 mirrors ``DEFAULT_MAX_ITERATIONS`` —
    # iteration 3+ rarely killed additional mutants in Phase 6
    # validation runs. Operators that want a longer loop can bump
    # this; the absolute cap of 10 in active_learning.py still
    # bounds the worst case.
    active_learning_max_iterations: int = 2
    # Max-quality test generation: let agentic CLIs create scratch tests and
    # run them during authoring, then the benchmark runner rematerializes the
    # original task before Apex validates emitted artifacts. Off by default to
    # preserve legacy read-only generation behavior.
    testgen_allow_agentic_edit_loop: bool = False


@dataclass
class ACIConfig:
    """Agent-Computer Interface configuration."""

    file_view_lines: int = 100
    search_max_results: int = 50
    lint_on_edit: bool = True
    edit_feedback_context_lines: int = 10
    bash_timeout: int = 30
    explicit_empty_output: bool = True
    max_output_lines: int = 220
    runtime_env_overrides: dict[str, str] = field(default_factory=dict)
    enable_agent_teams: bool = True
    max_agent_team_depth: int = 2
    max_agent_team_size: int = 3
    max_agent_team_parallelism: int = 2
    max_agent_team_iterations: int = 12
    agent_team_workspace_dirname: str = ".apex_agent_teams"
    agent_team_patch_preview_chars: int = 1600
    keep_agent_team_workspaces: bool = True
    # Phase 5.6 security hardening: default the bash tool to ``bash -c``
    # rather than ``bash -lc``. Sourcing login profiles re-introduces
    # host secrets (``HOME``, conda init, ``~/.bash_profile``) into the
    # agent's environment that we explicitly try to scrub via
    # ``runtime_env_overrides``. Set ``allow_login_shell=True`` only when
    # the benchmark image actually depends on login-profile semantics
    # (typical for benchmark docker images that ship a curated profile).
    allow_login_shell: bool = False
    # Cache TTL applied to ``_project_doc_cache`` and
    # ``_external_search_cache``. Stale doc / search hits reused across
    # edits cause invisible drift; expire entries after this many seconds.
    cache_ttl_seconds: float = 60.0
    # Project-doc cache can also be invalidated whenever the agent edits
    # a file under one of these prefixes (default: any path). Tests can
    # narrow it; production keeps the conservative default.
    cache_invalidate_on_any_edit: bool = True


@dataclass
class AgenticSearchConfig:
    """Mode-aware prompt/tooling hooks for local docs, web retrieval, and review."""

    access_mode: KnowledgeAccessMode = KnowledgeAccessMode.AIR_GAPPED
    enable_local_doc_guidance: bool = False
    local_doc_max_files: int = 6
    guided_stage_names: list[str] = field(default_factory=lambda: ["localizer", "patcher"])
    enable_proactive_evidence: bool = False
    proactive_evidence_max_items: int = 4
    proactive_evidence_stage_names: list[str] = field(default_factory=lambda: ["patcher"])
    external_search_budget: int = 2
    external_search_max_results: int = 5
    external_search_timeout_seconds: int = 12
    enable_semiformal_review: bool = False
    enable_followup_search_memory: bool = False
    enable_followup_gathered_information: bool = False
    followup_search_memory_max_items: int = 3
    # Substrings (case-insensitive) blocked from external evidence on top
    # of the hard-coded benchmark gold denylist. Use this to deny the
    # task's own upstream repo on benchmarks where INTERNET_AWARE is
    # safe in principle but the gold patch lives at a known URL (e.g.
    # `["github.com/<owner>/<repo>", "github.com/<fork>"]`). Substring
    # match — no regex required.
    external_search_denied_domains: list[str] = field(default_factory=list)


@dataclass
class ContextConfig:
    """Configuration for long-horizon context management."""

    max_context_tokens: int = 120000
    target_context_tokens: int = 80000
    protected_head_messages: int = 3
    protected_tail_messages: int = 6
    prune_tool_outputs_first: bool = True
    tool_output_max_tokens: int = 2000
    enable_periodic_summary: bool = True
    summary_interval_iterations: int = 10


@dataclass
class ExecutionTreeConfig:
    """Configuration for rollout checkpointing and backtracking."""

    enabled: bool = True
    max_depth: int = 5
    max_branches: int = 3
    restore_best_state: bool = True


@dataclass
class PlanningConfig:
    """Configuration for issue planning and rollout briefing."""

    enable_manager_planner: bool = True
    enable_task_state_graph: bool = True
    # WS3D: persist the per-task TaskStateGraph and warm-start the next solve of
    # the same task with its durable evidence/hypotheses/file-attention (don't
    # cold-derive the frontier every run). No-op when no prior file exists.
    warm_start_task_state_graph: bool = True
    enable_frontier_targeting: bool = True
    enable_collection_error_planner_bypass: bool = True
    allow_collection_error_fast_path_delegation: bool = False
    allow_heuristic_fallback: bool = True
    enable_coarse_to_fine_planning: bool = True
    allow_preplanner_skip_on_rich_heuristic_seed: bool = True
    enable_plan_portfolio: bool = True
    # D': validate the planner's structured (JSON) output against the expected
    # schema before accepting it; on a validation miss, prefer the existing retry
    # path rather than silently degrading to a coarse heuristic plan. Reuses the
    # in-house validator (apex/core/llm.py); Layer-A general; fails open.
    enable_planner_output_validation: bool = True
    always_include_single_agent_family: bool = True
    # Keep one simple Agentless-style localize->patch->validate family in the
    # rollout portfolio when the family budget is large enough.
    always_include_agentless_pipeline_family: bool = True
    enable_reflective_memory: bool = True
    planner_model: Optional[str] = None
    planner_llm_index: Optional[int] = None
    preplanner_model: Optional[str] = None
    preplanner_llm_index: Optional[int] = None
    # Planner phases are AGENTIC CLI steps (agent loops), not single LLM calls.
    # These hard timeouts apply ONLY to CLI backends (see manager
    # _phase_hard_timeout_seconds / _planner_hard_timeout_seconds), so a sub-30min
    # value here silently starves an agent that is budgeted up to
    # cli_hard_timeout_seconds (7200). Floor every planner phase at 30 minutes so
    # a coarse/refinement/main planning step is never killed mid-thought; the
    # per-task wallclock budget remains the outer cap. (Was 180/600/900 — the 180s
    # coarse cap killed real runs.)
    preplanner_timeout_seconds: Optional[int] = 1800
    refinement_timeout_seconds: Optional[int] = 1800
    planner_timeout_seconds: Optional[int] = 1800
    max_keywords: int = 16
    max_relevant_files: int = 18
    include_dependency_neighbors: int = 6
    max_repo_map_files: int = 24
    max_rollout_brief_families: int = 6
    max_task_state_context_items: int = 4
    max_frontier_targets: int = 6
    max_reflection_memory_items: int = 6
    delegation_boundary_pressure_threshold: int = 2
    regime_policy: RegimePolicyConfig = field(default_factory=RegimePolicyConfig)
    delegation_policy: DelegationPolicyConfig = field(default_factory=DelegationPolicyConfig)
    shadow_policy: ShadowPolicyConfig = field(default_factory=ShadowPolicyConfig)


@dataclass
class SearchConfig:
    """Configuration for explicit search over frontier targets."""

    mode: SearchMode = SearchMode.OFF
    max_expansions: int = 0
    max_depth: int = 6
    max_frontier_branching: int = 3
    c_puct: float = 1.25
    virtual_loss: float = 0.15
    stop_margin: float = 0.1
    min_branch_reward: float = 0.12
    persist_trace: bool = True
    transition_reward: TransitionRewardConfig = field(default_factory=TransitionRewardConfig)
    shadow_policy: ShadowPolicyConfig = field(default_factory=ShadowPolicyConfig)


@dataclass
class SelectionConfig:
    """Patch selection pipeline configuration."""

    strategy: SelectionStrategy = SelectionStrategy.MULTI_STAGE
    ast_similarity_threshold: float = 0.95
    enable_regression_pruning: bool = True
    cross_validation_enabled: bool = True
    judge_model: Optional[str] = None
    judge_temperature: float = 0.0
    min_test_pass_rate: float = 0.5
    selector_max_voters: int = 5
    selector_max_iterations: int = 8
    verification_timeout_seconds: int = 120
    full_test_timeout_seconds: int = 900
    custom_test_timeout_seconds: int = 120
    # Relative, non-submission helper files the benchmark adapter wants copied
    # from the prepared source checkout into candidate worktrees before
    # verifier commands run. Empty by default; benchmark adapters own the
    # contents.
    verification_helper_files: list[str] = field(default_factory=list)
    enable_critic_reranking: bool = True
    critic_weight: float = 0.2
    # Decisive-Edge C.4: explicit "is the LLM critic in the selection
    # path AT ALL" gate. ``enable_critic_reranking`` only suppresses the
    # critic's score contribution to the ranking; ``use_critic`` is the
    # *call-site* gate — when False the SelectionCritic is never
    # constructed, the LLM is never invoked, and the selector falls back
    # to a verifier-only ranking (pass-rate then lowest test-edit count).
    # Default True for back-compat with the published 86.3% headline.
    # The C.4 A/B harness flips this per-arm to measure the critic's
    # marginal contribution.
    use_critic: bool = True
    enable_patch_synthesis: bool = True
    max_synthesis_candidates: int = 6
    max_synthesis_combinations: int = 12
    # ------------------------------------------------------------------
    # TIER 2 N-way greedy synthesis union (T2.5). The legacy pairs/triples
    # path is unchanged; the greedy union is an ADDITIONAL path that only
    # ever fires when there are enough file-disjoint clusters to union, and
    # the verifier remains authoritative (a union is selected iff its
    # verification_score beats every component). General size knobs only.
    enable_greedy_synthesis_union: bool = True
    # Seed pool of cluster representatives the greedy union ranks over. Not
    # truncated to the legacy top-6; capped here so the O(n^2)-once AST
    # conflict check stays bounded.
    max_synthesis_pool: int = 24
    # Upper bound on members in a single synthesized union (ladder ceiling).
    max_synthesis_union_members: int = 12
    # Ablation knob: when True, cross-validation will run sibling test
    # code directly inside the candidate worktree (the legacy unsafe
    # behavior). Off by default; set True only for benchmark ablations
    # that need to reproduce the historical signal verbatim.
    cross_validation_sandbox_disabled: bool = False
    # Phase 4.2: weighted-composite ranking weights for the testgen
    # candidate selector
    # (``apex.evaluation.multi_candidate.select_best_testgen_candidate``).
    # The default is the literature-informed prior in
    # ``DEFAULT_TESTGEN_RANKING_WEIGHTS``. Operators may override this
    # globally here, or per-benchmark via
    # :attr:`BenchmarkConfig.testgen_ranking_weights_override`. See
    # ``apex/scripts/calibrate_testgen_ranking.py`` for the future-data
    # refit pipeline. Empty dict = use the module default.
    testgen_ranking_weights: dict[str, float] = field(default_factory=dict)
    # Decisive-Edge D.2: optional override for the calibrated critic
    # reranker weights JSON. ``None`` (default) falls back to
    # ``apex/configs/critic_weights_calibrated.json`` next to the
    # package; if that file is missing or malformed the literature-prior
    # ``DEFAULT_CRITIC_WEIGHTS`` is used. Operators set this when they
    # want to A/B a candidate calibration without overwriting the
    # shipped JSON.
    critic_weights_calibrated_path: Optional[str] = None
    # When a rollout is marked ``standalone_agent_anchor``, preserve it on
    # verifier ties and only allow an Apex-orchestrated candidate to replace it
    # when the replacement has strictly stronger harness/verification signal.
    # This turns max-config Apex into an expected-value wrapper around the best
    # standalone CLI instead of a selector that can demote it on soft heuristics.
    preserve_standalone_anchor: bool = True
    # WS2C: execution-grounded learned-critic tie-break among EXECUTION-TIED
    # clusters. Two independent default-off gates (this flag AND the eg_critic
    # optional-component arm) + a fitted artifact are all required for live wiring;
    # a non-fitted artifact / missing file is a no-op. Never overrides execution
    # evidence (tie-break only).
    enable_eg_critic_tiebreak: bool = False
    eg_critic_weights_path: Optional[str] = None
    # WS3C: fresh-context LLM final-acceptance reviewer (only downgrades; fails
    # open). All default OFF/empty so the headline is untouched.
    enable_final_acceptance_reviewer: bool = False
    final_acceptance_reviewer_backend: Optional[str] = None
    final_acceptance_reviewer_require_distinct_family: bool = True
    # Agentic CLI reviewer call; floor at 30min (fails open on timeout).
    final_acceptance_reviewer_timeout_seconds: int = 1800
    # E: perspective-diverse model-critic selection layer. Multiple DISTINCT
    # generic lenses score each candidate; they act (a) as a tiebreaker among
    # execution-verified candidates (picks the one least likely to overfit the
    # visible/F2P tests and fail hidden tests), and (b) as a model-judgment
    # acceptance/ranking signal when execution evidence is absent/inconclusive
    # (general, non-code long-horizon tasks). Layer-A general; fails open; NEVER
    # overrides concrete execution evidence (only re-ranks within an accept tier).
    enable_perspective_review: bool = True
    perspective_review_lenses: list[str] = field(
        default_factory=lambda: [
            "minimality",
            "spec_conformance",
            "edge_case_risk",
            "regression_risk",
        ]
    )
    perspective_review_max_workers: int = 4
    perspective_review_backend: Optional[str] = None
    # Per-lens agentic CLI reviewer calls (concurrent); floor at 30min (each lens
    # fails open to neutral 0.5 on timeout, so the cap only bounds the wait).
    perspective_review_timeout_seconds: int = 1800
    perspective_review_min_candidates: int = 2


# ---------------------------------------------------------------------------
# Orchestration agent surface (single source of truth).
#
# There is exactly ONE vocabulary of agent surfaces and ONE global fallback,
# defined here and imported everywhere (CLI resolver + orchestrator startup).
# No other module may define a competing default. The resolved surface is
# always written back to ``BenchmarkConfig.default_agent_mode`` and logged
# loudly, so a developer/researcher/agent can never be unsure which path a run
# took. See ``apex.cli._resolve_agent_mode`` (resolution + provenance) and the
# orchestrator startup banner in ``apex.orchestration.solver`` (loud declare +
# unresolved guard).
# ---------------------------------------------------------------------------
AGENT_MODE_SCAFFOLDED = "scaffolded"
AGENT_MODE_CLI_AGENT = "cli_agent"
AGENT_MODE_IN_CONTAINER_V5 = "in_container_v5"
AGENT_MODE_HIERARCHICAL_V5 = "hierarchical_v5"
AGENT_MODE_CHOICES: tuple[str, ...] = (
    AGENT_MODE_SCAFFOLDED,
    AGENT_MODE_CLI_AGENT,
    AGENT_MODE_IN_CONTAINER_V5,
    AGENT_MODE_HIERARCHICAL_V5,
)
# Ultimate fallback when nothing (explicit CLI / config file / per-subcommand
# default) selects a surface. ``scaffolded`` = the legacy MASAI orchestrator.
GLOBAL_DEFAULT_AGENT_MODE = AGENT_MODE_SCAFFOLDED


@dataclass
class BenchmarkConfig:
    """Benchmark evaluation policy."""

    commit0_primary_evaluation_backend: BenchmarkEvaluationBackend = (
        BenchmarkEvaluationBackend.LOCAL_PYTEST
    )
    # Phase 1.1: the upstream Commit0 docker harness audit is the source of
    # truth for the published headline number. APEX-private local pytest
    # scoring is preserved as a diagnostic but no longer the headline when
    # the audit succeeds.
    commit0_official_audit_selected: bool = True
    commit0_official_audit_only_if_primary_passes: bool = False
    # B5 (non-deterministic-failure firewall): a single Twisted/asyncio teardown
    # ERROR (e.g. DirtyReactorAggregateError) can flip an otherwise-clean
    # official audit (failed==0, coverage preserved) to a published 0. When the
    # audit is green except for scored ERRORS that match a known teardown/finalizer
    # flake signature, re-run the official audit up to this budget until a stable
    # outcome is observed before publishing. ``1`` disables re-auditing.
    commit0_transient_audit_rerun_budget: int = 3
    # Require this many agreeing attempts (or one decisive clean success) before
    # accepting a re-audited outcome. Bounds rerun cost while still giving a
    # non-deterministic teardown failure a chance to resolve to its true state.
    commit0_transient_audit_require_stable: int = 2
    # WS2B (NDFF): when a teardown flake exhausts the re-audit budget, stamp the
    # task NON_DETERMINISTIC and carve it out of the strict-headline denominator
    # (a flaky gold test must never charge an APEX miss). Only ever fires on a
    # genuine budget-exhausted teardown flake, so it cannot inflate a real miss.
    commit0_ndff_exclude_nondeterministic: bool = True
    # B3: memory cgroup limit and /dev/shm size for the shared per-task Commit0
    # runtime container. The default 64MB /dev/shm and unbounded cgroup let large
    # suites (e.g. the 3612-test pytest-on-pytest run) thrash shared memory or get
    # OOM-reaped mid-run; once reaped, every later exec hits "No such container".
    # Docker size strings (e.g. "8g", "512m").
    commit0_docker_memory_limit: str = "8g"
    commit0_docker_shm_size: str = "2g"
    # Commit0 official audits run in an independent benchmark harness lane so
    # long Docker/qemu audit tails do not consume agent solve parallelism.
    commit0_official_audit_parallelism: int = 1
    # Commit0/Python harness accelerator for local pytest evaluation. Empty/0
    # keeps serial pytest; "auto"/"logical"/"max" or a positive integer injects
    # pytest-xdist workers while preserving expected-ID scoring from the JSON
    # report.
    commit0_pytest_xdist_workers: str = ""
    commit0_pytest_xdist_dist: str = ""
    # Commit0 Docker solves can leave large task sandboxes behind after an
    # interrupted run. Max-quality runs may set a floor and prune stale inactive
    # sandboxes before launch so ENOSPC does not masquerade as agent failure.
    commit0_min_free_disk_gb: int = 0
    commit0_prune_stale_task_sandboxes: bool = True
    commit0_stale_task_sandbox_min_age_seconds: int = 1800
    # Phase 1.2: gate on whether the APEX-private pytest-json exit-code
    # rewrite (commit0_benchmark._collect_evaluation) is allowed to override
    # the shell returncode. Default OFF so the published number honours the
    # canonical shell rc; turn ON only for ablations that need to reproduce
    # the historical (pre-Phase-1) signal.
    commit0_use_pytest_json_exitcode: bool = False
    # Diagnostic/oracle-ablation only: use the upstream Docker audit scorer to
    # rerank rollout candidates before choosing the final submission. This must
    # remain OFF for fair published comparisons because it lets the benchmark
    # evaluator choose among multiple candidate patches. ``0`` means evaluate
    # every candidate that survived the local scorer; positive values cap the
    # audit rerank to the top N.
    commit0_audit_candidate_selection: bool = False
    commit0_audit_candidate_selection_top_k: int = 3
    commit0_repo_clone_timeout_seconds: int = 1800
    # Filesystem paths to local mirrors of the Commit0 repos. Tried in order
    # after the GitHub clone retries are exhausted; first hit wins. Each
    # path is treated as a parent dir containing repos named ``<repo_name>``
    # (e.g. ``/srv/commit0_mirror/tinydb``). Useful when running
    # benchmarks behind a flaky network or in air-gapped CI.
    commit0_local_repo_roots: list[str] = field(default_factory=list)
    commit0_runtime_setup_timeout_seconds: int = 1800
    commit0_dependency_install_timeout_seconds: int = 3600
    commit0_evaluation_timeout_seconds: int = 1800
    commit0_baseline_evaluation_timeout_seconds: int = 1800
    # Commit0/Python harness fact: official full-suite scoring needs a long
    # timeout, but exploratory agent-issued target-runtime commands should not
    # inherit that audit budget and monopolize solve slots.
    commit0_agent_target_tool_timeout_seconds: int = 300
    disable_pytest_plugin_autoload: bool = True
    # When the host (macOS) prepare/baseline fails OR the baseline shows a
    # known host-env limitation signature (pytest plugin incompat,
    # collection errors), automatically retry the prepare+baseline inside a
    # Linux Docker container before skipping the repo. The container is
    # torn down by the existing finally clause in `_run_task`, so disk
    # usage stays bounded to one task at a time.
    commit0_docker_fallback_on_failure: bool = True
    # Commit0 CLI agents need the same filesystem boundary as evaluation;
    # "always" prepares every task in the benchmark Docker runtime instead
    # of using host_env first and retrying only on host-specific failures.
    commit0_docker_runtime_mode: str = "fallback"
    # Commit0 max/portfolio runs use backend diversity as part of the
    # experimental contract, so they can opt into failing fast when any
    # configured target-container CLI backend is unavailable.
    commit0_require_all_configured_cli_backends: bool = False
    # Commit0 harness fact: experimental containerized CLI routes can be
    # configured additively while still requiring core backends to pass preflight.
    commit0_optional_configured_cli_backends: list[str] = field(default_factory=list)
    task_parallelism: int = 1
    # General evaluation/remediation contracts. Empty dictionaries preserve
    # legacy configs; benchmark runners can resolve benchmark-specific
    # defaults through ``resolved_evaluation_contract_config``.
    evaluation_power_mode: str = "standard"
    unbounded_followup_budget: bool = False
    evaluation_contract: dict[str, Any] = field(default_factory=dict)
    runtime_policy: dict[str, Any] = field(default_factory=dict)
    patch_hygiene: dict[str, Any] = field(default_factory=dict)
    run_supervisor: dict[str, Any] = field(
        default_factory=lambda: {
            "cleanup_timeout_seconds": 120,
            "signal_escalation_timeout_seconds": 5,
            "docker_cleanup_labels": [
                "apex.run_id",
                "apex.task_id",
                "apex.owner_pid",
                "apex.benchmark",
                "apex.created_at",
            ],
        }
    )
    reporting: dict[str, Any] = field(
        default_factory=lambda: {
            "compact_top_level_state": True,
            "emit_active_tasks": True,
            "emit_evaluation_progress": True,
            "detail_artifact_layout": "task_directories",
        }
    )
    # SWE-Bench Pro test-generation task isolation. A value <= 0 preserves
    # in-process execution for unit tests and legacy callers; the benchmark CLI
    # enables this by default so a hung repo checkpoints as timeout and the
    # slice advances.
    testgen_task_timeout_seconds: float = 0.0

    # ------------------------------------------------------------------
    # Phase 1c: SWT-Bench specific
    # ------------------------------------------------------------------
    # Subprocess retry budget for SWT-Bench harness invocations
    # (apex.core.subprocess_retry). Real APEX failures are NEVER retried;
    # only env_* and harness_bug classifications are. Set to 1 to disable.
    swtbench_subprocess_max_attempts: int = 3
    # Backoff strategy: "exponential" (1s/2s/4s) or "none".
    swtbench_subprocess_backoff: str = "exponential"
    # Fairness audit mode for SWT-Bench. SWT-Bench has only one scorer
    # (the upstream Docker harness); when set to "parallel" the audit
    # framework still emits a per-task entry with by-construction zero
    # delta so reviewers see "comparable" rather than "missing".
    swtbench_fairness_audit_mode: str = "off"
    # Phase 1: side-by-side fairness audit. OFF by default so production
    # runs aren't slowed by a second scoring pass. PARALLEL runs both the
    # APEX-private and upstream-canonical scorers per task and emits
    # ``fairness_audit.json`` + ``FAIRNESS_REPORT.md``. UPSTREAM_ONLY uses
    # the upstream-canonical scorer for the headline and treats the
    # APEX-private number as diagnostic-only. The string default keeps the
    # config dataclass importable without a fairness_audit dep cycle; the
    # benchmark runner coerces it to FairnessAuditMode.
    fairness_audit_mode: str = "off"

    # ------------------------------------------------------------------
    # Phase 1b: TestGenEval specific
    # ------------------------------------------------------------------
    # When True, apply APEX-maintained patches (mutation_timeout flag and
    # the macOS bind-mount fix; see
    # apex/evaluation/upstream_patches/testgeneval/UPSTREAM_PR_PLAN.md)
    # to the upstream kjain14/testgeneval checkout before running the
    # harness. Even with the deprecated memory-swappiness fix removed in
    # Phase 1.3, this remains a divergence vs. the published baseline,
    # so publishable comparison keeps this False by default. Set True only
    # for local operational runs that explicitly accept harness divergence.
    testgeneval_apply_upstream_patches: bool = False
    # When True, apply only the defensive baseline_covs KeyError fix
    # (baseline_covs_keyerror.patch) but NOT the mutation_timeout /
    # bind-mount patch. Used by the upstream-canonical scorer in
    # ``FairnessAuditMode.PARALLEL`` to keep ``generate_report.py`` from
    # crashing on rows that omit baseline_covs while otherwise leaving
    # the harness behaviour untouched.
    testgeneval_apply_baseline_covs_patch_only: bool = False
    # Per-benchmark fairness audit mode override. When non-empty, takes
    # precedence over ``fairness_audit_mode`` for TestGenEval runs only.
    testgeneval_fairness_audit_mode: str = ""
    # Phase 4.2: per-benchmark override of the testgen ranking weights
    # in :attr:`SelectionConfig.testgen_ranking_weights`. ``None`` means
    # "use the global selection.testgen_ranking_weights / module
    # default". A partial dict (e.g. ``{"mutation_score": 0.40}``) merges
    # with the default — missing keys keep the default weight.
    testgen_ranking_weights_override: Optional[dict[str, float]] = None
    # Phase A.1 (Decisive-Edge): per-benchmark default agent surface.
    # The CLI lifts the resolved ``--agent-mode`` flag (or its per-
    # subcommand default) onto this field. ``ApexOrchestrator.solve``
    # consults it when ``benchmark_metadata`` is supplied: the
    # ``in_container_v5`` value routes the solve through the V5
    # in-container agent loop and bridges the result back into the
    # Resolved orchestration agent surface — the SINGLE field every consumer
    # reads (orchestrator startup, solver V5 routing). Default ``""`` means
    # UNRESOLVED: it is filled by ``apex.cli._resolve_agent_mode`` (explicit
    # --agent-mode > config-file value > per-subcommand default >
    # GLOBAL_DEFAULT_AGENT_MODE), and the orchestrator startup banner guards
    # any still-empty value back to GLOBAL_DEFAULT_AGENT_MODE with a loud
    # warning. Never compare against a hardcoded literal elsewhere — use
    # AGENT_MODE_* / AGENT_MODE_CHOICES.
    default_agent_mode: str = ""
    # Phase A.3 (Decisive-Edge): controls which of the two scoring
    # numbers benchmark reports treat as "the headline". Reports ALWAYS
    # emit BOTH ``score_strict`` (treats env-skipped repos as zero) and
    # ``score_runnable`` (denominator excludes skipped repos); this
    # field selects the one rendered as the leaderboard number in the
    # markdown summary. Allowed values: ``"score_strict"`` (default,
    # publication-defensible) and ``"score_runnable"`` (headline-race).
    # The ``--benchmark-mode publication`` preset pins this to strict;
    # ``--benchmark-mode headline`` pins it to runnable.
    report_headline_metric: str = "score_strict"
    # Decisive-Edge B.1: per-benchmark override of the global
    # ``RolloutConfig.localizer_enforcement`` default. When ``None`` the
    # global default applies; when non-empty the value (one of
    # ``"advisory"``, ``"warning"``, ``"hard_constraint"``) is lifted
    # onto ``RolloutConfig.localizer_enforcement`` at run-time by the
    # benchmark runner. Use this to relax enforcement on benchmarks
    # whose localizer is unreliable on very large repos without
    # flipping the global default.
    localizer_enforcement_override: Optional[str] = None
    # Decisive-Edge C.2: per-benchmark override of the global
    # ``OrchestrationConfig.abstention_threshold``. When set (non-None),
    # this value supersedes the global threshold for benchmark runs
    # whose ``benchmark_metadata["benchmark_name"]`` resolves to this
    # config's benchmark family. This override has LOWER priority than
    # the calibrated per-benchmark table on disk
    # (``apex/configs/abstention_thresholds_per_benchmark.json``,
    # produced by ``apex/scripts/calibrate_abstention_threshold.py``):
    #   per-benchmark calibrated  >  per-benchmark override  >  global default
    # ``None`` (default) means "fall through to the global threshold".
    abstention_threshold_override: Optional[float] = None

    def resolved_evaluation_contract_config(
        self,
        benchmark_name: str = "",
    ) -> dict[str, Any]:
        if self.evaluation_contract:
            return copy.deepcopy(self.evaluation_contract)
        name = str(benchmark_name or "").strip().lower().replace("_", "-")
        if "commit0" in name or "commit-0" in name:
            return {
                "mode": "gold_suite_visible",
                "scoring_universe": "expected_test_ids",
                "diagnostic_universes": [
                    "extra_non_scored_tests",
                    "raw_pytest_returncode",
                ],
                "raw_returncode_policy": "diagnostic_only_when_scoring_filtered",
                "extra_result_policy": "diagnostic_only",
                "baseline_timeout_policy": "attempt_anyway",
                "environment_failure_policy": "fallback_runtime",
            }
        if "swebench" in name or "swe-bench" in name:
            return {
                "mode": "hidden_suite_authoritative",
                "scoring_universe": "official_harness",
                "diagnostic_universes": ["public_tests", "generated_tests"],
                "raw_returncode_policy": "score_bearing",
                "extra_result_policy": "diagnostic_only",
                "baseline_timeout_policy": "evaluator_fatal",
                "environment_failure_policy": "retry",
            }
        return {
            "mode": "custom",
            "scoring_universe": "runner_summary",
            "diagnostic_universes": [],
            "raw_returncode_policy": "score_bearing",
            "extra_result_policy": "score_bearing",
        }

    # ------------------------------------------------------------------
    # Calibrated per-benchmark threshold loader (Decisive-Edge C.2)
    # ------------------------------------------------------------------
    @staticmethod
    def load_calibrated_abstention_thresholds(
        path: Optional[str | Path] = None,
    ) -> dict[str, float]:
        """Load the per-benchmark calibrated abstention thresholds JSON.

        Reads ``apex/configs/abstention_thresholds_per_benchmark.json``
        (or ``path`` when supplied) and returns a mapping
        ``{benchmark_id: threshold}``. The ``_metadata`` block emitted by
        the calibrator is stripped. Returns an empty dict when the file
        does not exist (graceful fall-through — calibration may not have
        run yet) or when it cannot be parsed; never raises.
        """
        if path is None:
            # Resolve relative to this module's package root so the loader
            # works whether APEX is installed or run from a checkout.
            try:
                resolved = (
                    Path(__file__).resolve().parent.parent
                    / "configs"
                    / "abstention_thresholds_per_benchmark.json"
                )
            except Exception:
                return {}
        else:
            resolved = Path(path)
        try:
            if not resolved.exists():
                return {}
            with resolved.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        out: dict[str, float] = {}
        for key, value in data.items():
            if str(key).startswith("_"):
                # Skip _metadata and any other underscore-prefixed key.
                continue
            try:
                out[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return out


@dataclass
class RepoMemoryConfig:
    """Persistent per-repository insight memory across solves.

    Disabled by default to keep benchmark runs reproducible. When enabled,
    APEX persists high-confidence rollout discoveries to a JSON store
    keyed by the absolute repo path and re-loads them as priors on the
    next solve of the same repo. The store is intentionally small and
    decay-weighted so that stale beliefs fade.
    """

    enabled: bool = False
    directory: Optional[str] = None
    min_confidence_to_persist: float = 0.7
    decay_factor: float = 0.85
    max_persisted_insights: int = 64
    prefer_high_support_threshold: int = 2


@dataclass
class ApexConfig:
    """Top-level APEX configuration."""

    llm_configs: list[LLMConfig] = field(default_factory=lambda: [LLMConfig()])
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    aci: ACIConfig = field(default_factory=ACIConfig)
    agentic_search: AgenticSearchConfig = field(default_factory=AgenticSearchConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    execution_tree: ExecutionTreeConfig = field(default_factory=ExecutionTreeConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    controller_models: ControllerModelLibraryConfig = field(
        default_factory=ControllerModelLibraryConfig
    )
    controller_trace: ControllerTraceConfig = field(default_factory=ControllerTraceConfig)
    repo_memory: RepoMemoryConfig = field(default_factory=RepoMemoryConfig)
    orchestration: "OrchestrationConfig" = field(default_factory=lambda: OrchestrationConfig())
    use_concise_prompts: bool = True
    enable_planning_tool: bool = True
    workspace_dir: str = "/tmp/apex_workspace"
    output_dir: str = "/tmp/apex_output"
    log_level: str = "INFO"
    save_trajectories: bool = True

    @classmethod
    def from_file(cls, path: str | Path) -> "ApexConfig":
        path = Path(path)
        with path.open() as handle:
            data = json.load(handle)
        return cls._from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApexConfig":
        return cls._from_dict(dict(data or {}))

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "ApexConfig":
        llm_configs = []
        for entry in data.get("llm_configs", [{}]):
            payload = dict(entry)
            if "backend" in payload:
                payload["backend"] = LLMBackend(payload["backend"])
            payload.pop("cli_max_concurrency", None)
            payload.pop("cli_slot_namespace", None)
            llm_configs.append(LLMConfig(**payload))

        rollout_data = dict(data.get("rollout", {}))
        if "agent_mode" in rollout_data:
            rollout_data["agent_mode"] = AgentMode(rollout_data["agent_mode"])
        if "diversity_prompts" in rollout_data:
            rollout_data["diversity_prompts"] = [
                _coerce_enum(value, PromptStrategy) for value in rollout_data["diversity_prompts"]
            ]
        if "scaffold_stage_llm_indices" in rollout_data:
            rollout_data["scaffold_stage_llm_indices"] = _coerce_stage_llm_indices(
                rollout_data["scaffold_stage_llm_indices"],
                llm_config_count=len(llm_configs),
            )
        if "llm_profiles" in rollout_data:
            rollout_data["llm_profiles"] = _coerce_llm_profiles(
                rollout_data["llm_profiles"],
                llm_config_count=len(llm_configs),
            )
        if "standalone_anchor_candidates" in rollout_data:
            rollout_data["standalone_anchor_candidates"] = _coerce_standalone_anchor_candidates(
                rollout_data["standalone_anchor_candidates"]
            )
        if "completion_policy" in rollout_data:
            rollout_data["completion_policy"] = CompletionExecutionPolicyConfig(
                **dict(rollout_data["completion_policy"] or {})
            )
        if "overlap_policy" in rollout_data:
            rollout_data["overlap_policy"] = OverlapPolicyConfig(
                **dict(rollout_data["overlap_policy"] or {})
            )
        if "shadow_policy" in rollout_data:
            rollout_data["shadow_policy"] = ShadowPolicyConfig(
                **dict(rollout_data["shadow_policy"] or {})
            )
        rollout = RolloutConfig(**rollout_data)

        aci = ACIConfig(**data.get("aci", {}))
        agentic_search_data = dict(data.get("agentic_search", {}))
        if "access_mode" in agentic_search_data:
            agentic_search_data["access_mode"] = KnowledgeAccessMode(
                agentic_search_data["access_mode"]
            )
        agentic_search = AgenticSearchConfig(**agentic_search_data)
        context = ContextConfig(**data.get("context", {}))
        execution_tree = ExecutionTreeConfig(**data.get("execution_tree", {}))
        planning_data = dict(data.get("planning", {}))
        planning_data.pop("planner_cli_slot_namespace", None)
        planning_data.pop("preplanner_cli_slot_namespace", None)
        if "regime_policy" in planning_data:
            planning_data["regime_policy"] = RegimePolicyConfig(
                **dict(planning_data["regime_policy"] or {})
            )
        if "delegation_policy" in planning_data:
            planning_data["delegation_policy"] = DelegationPolicyConfig(
                **dict(planning_data["delegation_policy"] or {})
            )
        elif "delegation_boundary_pressure_threshold" in planning_data:
            planning_data["delegation_policy"] = DelegationPolicyConfig(
                boundary_pressure_threshold=int(
                    planning_data.get("delegation_boundary_pressure_threshold") or 0
                )
                or DelegationPolicyConfig().boundary_pressure_threshold
            )
        if "shadow_policy" in planning_data:
            planning_data["shadow_policy"] = ShadowPolicyConfig(
                **dict(planning_data["shadow_policy"] or {})
            )
        planning = PlanningConfig(**planning_data)
        planning.delegation_boundary_pressure_threshold = int(
            planning.delegation_policy.boundary_pressure_threshold
            if isinstance(getattr(planning, "delegation_policy", None), DelegationPolicyConfig)
            else planning.delegation_boundary_pressure_threshold
        )
        search_data = dict(data.get("search", {}))
        if "mode" in search_data:
            search_data["mode"] = SearchMode(search_data["mode"])
        if "transition_reward" in search_data:
            search_data["transition_reward"] = TransitionRewardConfig(
                **dict(search_data["transition_reward"] or {})
            )
        if "shadow_policy" in search_data:
            search_data["shadow_policy"] = ShadowPolicyConfig(
                **dict(search_data["shadow_policy"] or {})
            )
        search = SearchConfig(**search_data)

        selection_data = dict(data.get("selection", {}))
        if "strategy" in selection_data:
            selection_data["strategy"] = SelectionStrategy(selection_data["strategy"])
        selection = SelectionConfig(**selection_data)

        benchmark_data = dict(data.get("benchmark", {}))
        if "commit0_primary_evaluation_backend" in benchmark_data:
            benchmark_data["commit0_primary_evaluation_backend"] = BenchmarkEvaluationBackend(
                benchmark_data["commit0_primary_evaluation_backend"]
            )
        benchmark = BenchmarkConfig(**benchmark_data)
        controller_models = (
            ControllerModelLibraryConfig.from_dict(dict(data.get("controller_models") or {}))
            if isinstance(data.get("controller_models"), dict)
            else ControllerModelLibraryConfig()
        )
        controller_trace = ControllerTraceConfig(**dict(data.get("controller_trace") or {}))
        repo_memory_data = dict(data.get("repo_memory") or {})
        repo_memory = RepoMemoryConfig(**repo_memory_data)

        orchestration_data = dict(data.get("orchestration") or {})
        orchestration = OrchestrationConfig(**orchestration_data)

        config = cls(
            llm_configs=llm_configs,
            rollout=rollout,
            aci=aci,
            agentic_search=agentic_search,
            context=context,
            execution_tree=execution_tree,
            planning=planning,
            search=search,
            selection=selection,
            benchmark=benchmark,
            controller_models=controller_models,
            controller_trace=controller_trace,
            repo_memory=repo_memory,
            orchestration=orchestration,
            use_concise_prompts=data.get("use_concise_prompts", True),
            enable_planning_tool=data.get("enable_planning_tool", True),
            workspace_dir=data.get("workspace_dir", "/tmp/apex_workspace"),
            output_dir=data.get("output_dir", "/tmp/apex_output"),
            log_level=data.get("log_level", "INFO"),
            save_trajectories=data.get("save_trajectories", True),
        )
        return apply_localizer_enforcement_override(config)

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_configs": [
                {
                    "model": config.model,
                    "backend": config.backend.value,
                    "api_key_env": config.api_key_env,
                    "base_url": config.base_url,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "timeout": config.timeout,
                    "cli_command": config.cli_command,
                    "cli_args": list(config.cli_args),
                    "cli_model_id": config.cli_model_id,
                    "cli_timeout": config.cli_timeout,
                    "cli_hard_timeout_seconds": config.cli_hard_timeout_seconds,
                    "cli_health_probe_timeout_seconds": (config.cli_health_probe_timeout_seconds),
                    "cli_target_runtime_warmup_timeout_seconds": (
                        config.cli_target_runtime_warmup_timeout_seconds
                    ),
                    "cli_strict_hard_timeout": config.cli_strict_hard_timeout,
                    "cli_first_output_timeout_seconds": (
                        config.cli_first_output_timeout_seconds
                    ),
                    "cli_output_capture_max_chars": config.cli_output_capture_max_chars,
                    "cli_disable_osx_sandbox": config.cli_disable_osx_sandbox,
                    "cli_permission_mode": config.cli_permission_mode,
                    "cli_env_overrides": dict(config.cli_env_overrides),
                    "cli_env_redaction_disabled": config.cli_env_redaction_disabled,
                    "cli_tool_review_enabled": config.cli_tool_review_enabled,
                    "cli_tool_review_reviewer_backend": (config.cli_tool_review_reviewer_backend),
                    "cli_tool_review_reviewer_command": (config.cli_tool_review_reviewer_command),
                    "cli_tool_review_timeout_seconds": (config.cli_tool_review_timeout_seconds),
                }
                for config in self.llm_configs
            ],
            "rollout": {
                "num_rollouts": self.rollout.num_rollouts,
                "enable_adaptive_allocation": self.rollout.enable_adaptive_allocation,
                "enable_speculative_first_attempt": self.rollout.enable_speculative_first_attempt,
                "speculative_first_attempt_max_difficulty": (
                    self.rollout.speculative_first_attempt_max_difficulty
                ),
                "enable_cross_rollout_discovery_reuse": (
                    self.rollout.enable_cross_rollout_discovery_reuse
                ),
                "enable_cross_solve_episodic_memory": (
                    self.rollout.enable_cross_solve_episodic_memory
                ),
                "min_rollouts": self.rollout.min_rollouts,
                "max_rollouts": self.rollout.max_rollouts,
                "rollout_buckets": list(self.rollout.rollout_buckets),
                "scaffold_stage_llm_indices": dict(self.rollout.scaffold_stage_llm_indices),
                "llm_profiles": [dict(profile) for profile in self.rollout.llm_profiles],
                "portfolio_seed_profile_count": self.rollout.portfolio_seed_profile_count,
                "portfolio_diversity_include_prompt_strategy": (
                    self.rollout.portfolio_diversity_include_prompt_strategy
                ),
                "portfolio_diversity_include_temperature": (
                    self.rollout.portfolio_diversity_include_temperature
                ),
                "enable_dynamic_cop_transitions": self.rollout.enable_dynamic_cop_transitions,
                "enable_progressive_rollout_allocation": (
                    self.rollout.enable_progressive_rollout_allocation
                ),
                "max_progressive_rollout_waves": self.rollout.max_progressive_rollout_waves,
                "progressive_stop_on_strong_signal": (
                    self.rollout.progressive_stop_on_strong_signal
                ),
                "enable_residual_followup": self.rollout.enable_residual_followup,
                "max_selection_followup_rounds": self.rollout.max_selection_followup_rounds,
                "max_tokens_per_repo_followup": self.rollout.max_tokens_per_repo_followup,
                "max_followup_iterations": self.rollout.max_followup_iterations,
                "max_iterations_per_rollout": self.rollout.max_iterations_per_rollout,
                "min_completion_followup_rounds": self.rollout.min_completion_followup_rounds,
                "small_repo_near_miss_residual_cap": (
                    self.rollout.small_repo_near_miss_residual_cap
                ),
                "enable_orchestrated_multi_agent": self.rollout.enable_orchestrated_multi_agent,
                "enable_standalone_anchor": self.rollout.enable_standalone_anchor,
                "standalone_anchor_profile_index": (self.rollout.standalone_anchor_profile_index),
                "standalone_anchor_label": self.rollout.standalone_anchor_label,
                "standalone_anchor_candidates": [
                    dict(candidate) for candidate in self.rollout.standalone_anchor_candidates
                ],
                "standalone_anchor_run_all_candidates": (
                    self.rollout.standalone_anchor_run_all_candidates
                ),
                "standalone_anchor_strict_candidate_match": (
                    self.rollout.standalone_anchor_strict_candidate_match
                ),
                "standalone_anchor_allow_llm_fallback": (
                    self.rollout.standalone_anchor_allow_llm_fallback
                ),
                "use_git_worktrees": self.rollout.use_git_worktrees,
                "historyless_snapshots": self.rollout.historyless_snapshots,
                "keep_worktrees": self.rollout.keep_worktrees,
                "use_worktree_pool": self.rollout.use_worktree_pool,
                "worktree_pool_size": self.rollout.worktree_pool_size,
                "enable_heuristic_fallback": self.rollout.enable_heuristic_fallback,
                "enable_quick_verification": self.rollout.enable_quick_verification,
                "quick_verification_max_tests": self.rollout.quick_verification_max_tests,
                "quick_verification_timeout_seconds": self.rollout.quick_verification_timeout_seconds,
                "quick_verification_full_collection_when_expected_set_known": (
                    self.rollout.quick_verification_full_collection_when_expected_set_known
                ),
                "enable_decomposition_scale_partitioning": (
                    self.rollout.enable_decomposition_scale_partitioning
                ),
                "decomposition_min_expected_tests": (
                    self.rollout.decomposition_min_expected_tests
                ),
                "decomposition_min_stub_files": self.rollout.decomposition_min_stub_files,
                "decomposition_files_per_group": self.rollout.decomposition_files_per_group,
                "decomposition_max_partitions_cap": (
                    self.rollout.decomposition_max_partitions_cap
                ),
                "stall_window_seconds": self.rollout.stall_window_seconds,
                "qv_stall_window_seconds": self.rollout.qv_stall_window_seconds,
                "no_edit_progress_window_seconds": (
                    self.rollout.no_edit_progress_window_seconds
                ),
                "max_inflight_request_seconds": self.rollout.max_inflight_request_seconds,
                "emergency_silence_window_seconds": (
                    self.rollout.emergency_silence_window_seconds
                ),
                "selection_ready_drain_grace_seconds": (
                    self.rollout.selection_ready_drain_grace_seconds
                ),
                "task_wallclock_budget_seconds": (self.rollout.task_wallclock_budget_seconds),
                "rollout_wallclock_budget_seconds": (self.rollout.rollout_wallclock_budget_seconds),
                "enable_overlap_diversity_cap": self.rollout.enable_overlap_diversity_cap,
                "min_overlap_diversity_parallel_workers": (
                    self.rollout.min_overlap_diversity_parallel_workers
                ),
                "overlap_diversity_include_prompt_strategy": (
                    self.rollout.overlap_diversity_include_prompt_strategy
                ),
                "overlap_diversity_include_temperature": (
                    self.rollout.overlap_diversity_include_temperature
                ),
                "completion_policy": {
                    "preserve_primary_min_failing_tests": (
                        self.rollout.completion_policy.preserve_primary_min_failing_tests
                    ),
                    "preserve_primary_min_incomplete_sources": (
                        self.rollout.completion_policy.preserve_primary_min_incomplete_sources
                    ),
                    "preserve_primary_min_focus_tests": (
                        self.rollout.completion_policy.preserve_primary_min_focus_tests
                    ),
                    "preserve_primary_min_focus_test_failures": (
                        self.rollout.completion_policy.preserve_primary_min_focus_test_failures
                    ),
                    "preserve_primary_min_focus_files": (
                        self.rollout.completion_policy.preserve_primary_min_focus_files
                    ),
                    "preserve_primary_min_relevant_files": (
                        self.rollout.completion_policy.preserve_primary_min_relevant_files
                    ),
                    "preserve_primary_difficulty_threshold": (
                        self.rollout.completion_policy.preserve_primary_difficulty_threshold
                    ),
                    "timeout_broad_validation_min_failing_tests": (
                        self.rollout.completion_policy.timeout_broad_validation_min_failing_tests
                    ),
                    "timeout_broad_validation_min_incomplete_files": (
                        self.rollout.completion_policy.timeout_broad_validation_min_incomplete_files
                    ),
                    "timeout_broad_validation_min_relevant_files": (
                        self.rollout.completion_policy.timeout_broad_validation_min_relevant_files
                    ),
                    "timeout_extension_seconds": (
                        self.rollout.completion_policy.timeout_extension_seconds
                    ),
                    "timeout_extra_extension_seconds": (
                        self.rollout.completion_policy.timeout_extra_extension_seconds
                    ),
                    "timeout_extra_min_failing_tests": (
                        self.rollout.completion_policy.timeout_extra_min_failing_tests
                    ),
                    "timeout_extra_min_incomplete_files": (
                        self.rollout.completion_policy.timeout_extra_min_incomplete_files
                    ),
                    "timeout_extra_min_relevant_files": (
                        self.rollout.completion_policy.timeout_extra_min_relevant_files
                    ),
                    "delegated_timeout_multiplier": (
                        self.rollout.completion_policy.delegated_timeout_multiplier
                    ),
                    "delegated_timeout_min_seconds": (
                        self.rollout.completion_policy.delegated_timeout_min_seconds
                    ),
                    "delegated_timeout_max_seconds": (
                        self.rollout.completion_policy.delegated_timeout_max_seconds
                    ),
                },
                "overlap_policy": {
                    "source_overlap_threshold": self.rollout.overlap_policy.source_overlap_threshold,
                    "test_overlap_threshold": self.rollout.overlap_policy.test_overlap_threshold,
                    "combined_overlap_threshold": self.rollout.overlap_policy.combined_overlap_threshold,
                },
                "shadow_policy": {
                    "enabled": self.rollout.shadow_policy.enabled,
                    "max_logged_options": self.rollout.shadow_policy.max_logged_options,
                },
                "agent_mode": self.rollout.agent_mode.value,
                "diversity_temperatures": list(self.rollout.diversity_temperatures),
                "diversity_prompts": [
                    strategy.value for strategy in self.rollout.diversity_prompts
                ],
                # Phase A.4 Decisive-Edge: strategy-axis diversity. Empty
                # list = use the canonical 7-axis library.
                "diversity_strategies": list(self.rollout.diversity_strategies),
                # B.8 (Decisive-Edge): which prompt module to load.
                "prompts_version": self.rollout.prompts_version,
                "parallel_workers": self.rollout.parallel_workers,
                "global_parallel_worker_budget": self.rollout.global_parallel_worker_budget,
                "allow_salvage": self.rollout.allow_salvage,
                "localizer_enforcement": self.rollout.localizer_enforcement,
                "localizer_allowlist_files": list(self.rollout.localizer_allowlist_files),
                "cli_agent_use_masai_preround": (self.rollout.cli_agent_use_masai_preround),
                "cli_agent_preround_llm": self.rollout.cli_agent_preround_llm,
                "v5_recent_verbatim_turns": self.rollout.v5_recent_verbatim_turns,
                "v5_stall_repeat_threshold": self.rollout.v5_stall_repeat_threshold,
                "v5_stall_terminate_cap": self.rollout.v5_stall_terminate_cap,
                "v5_patch_verifier_reject_cap": self.rollout.v5_patch_verifier_reject_cap,
                "v5_max_turns_floor": self.rollout.v5_max_turns_floor,
                "v5_max_turns_ceiling": self.rollout.v5_max_turns_ceiling,
            },
            "aci": {
                "file_view_lines": self.aci.file_view_lines,
                "search_max_results": self.aci.search_max_results,
                "lint_on_edit": self.aci.lint_on_edit,
                "edit_feedback_context_lines": self.aci.edit_feedback_context_lines,
                "bash_timeout": self.aci.bash_timeout,
                "explicit_empty_output": self.aci.explicit_empty_output,
                "max_output_lines": self.aci.max_output_lines,
                "runtime_env_overrides": dict(self.aci.runtime_env_overrides),
                "enable_agent_teams": self.aci.enable_agent_teams,
                "max_agent_team_depth": self.aci.max_agent_team_depth,
                "max_agent_team_size": self.aci.max_agent_team_size,
                "max_agent_team_parallelism": self.aci.max_agent_team_parallelism,
                "max_agent_team_iterations": self.aci.max_agent_team_iterations,
                "agent_team_workspace_dirname": self.aci.agent_team_workspace_dirname,
                "agent_team_patch_preview_chars": self.aci.agent_team_patch_preview_chars,
                "keep_agent_team_workspaces": self.aci.keep_agent_team_workspaces,
            },
            "agentic_search": {
                "access_mode": self.agentic_search.access_mode.value,
                "enable_local_doc_guidance": self.agentic_search.enable_local_doc_guidance,
                "local_doc_max_files": self.agentic_search.local_doc_max_files,
                "guided_stage_names": list(self.agentic_search.guided_stage_names),
                "enable_proactive_evidence": self.agentic_search.enable_proactive_evidence,
                "proactive_evidence_max_items": self.agentic_search.proactive_evidence_max_items,
                "proactive_evidence_stage_names": list(
                    self.agentic_search.proactive_evidence_stage_names
                ),
                "external_search_budget": self.agentic_search.external_search_budget,
                "external_search_max_results": self.agentic_search.external_search_max_results,
                "external_search_timeout_seconds": (
                    self.agentic_search.external_search_timeout_seconds
                ),
                "enable_semiformal_review": self.agentic_search.enable_semiformal_review,
                "enable_followup_search_memory": (
                    self.agentic_search.enable_followup_search_memory
                ),
                "enable_followup_gathered_information": (
                    self.agentic_search.enable_followup_gathered_information
                ),
                "followup_search_memory_max_items": (
                    self.agentic_search.followup_search_memory_max_items
                ),
            },
            "context": {
                "max_context_tokens": self.context.max_context_tokens,
                "target_context_tokens": self.context.target_context_tokens,
                "protected_head_messages": self.context.protected_head_messages,
                "protected_tail_messages": self.context.protected_tail_messages,
                "prune_tool_outputs_first": self.context.prune_tool_outputs_first,
                "tool_output_max_tokens": self.context.tool_output_max_tokens,
                "enable_periodic_summary": self.context.enable_periodic_summary,
                "summary_interval_iterations": self.context.summary_interval_iterations,
            },
            "execution_tree": {
                "enabled": self.execution_tree.enabled,
                "max_depth": self.execution_tree.max_depth,
                "max_branches": self.execution_tree.max_branches,
                "restore_best_state": self.execution_tree.restore_best_state,
            },
            "planning": {
                "enable_manager_planner": self.planning.enable_manager_planner,
                "enable_task_state_graph": self.planning.enable_task_state_graph,
                "warm_start_task_state_graph": self.planning.warm_start_task_state_graph,
                "enable_frontier_targeting": self.planning.enable_frontier_targeting,
                "enable_collection_error_planner_bypass": (
                    self.planning.enable_collection_error_planner_bypass
                ),
                "allow_collection_error_fast_path_delegation": (
                    self.planning.allow_collection_error_fast_path_delegation
                ),
                "allow_heuristic_fallback": self.planning.allow_heuristic_fallback,
                "enable_coarse_to_fine_planning": self.planning.enable_coarse_to_fine_planning,
                "allow_preplanner_skip_on_rich_heuristic_seed": (
                    self.planning.allow_preplanner_skip_on_rich_heuristic_seed
                ),
                "enable_plan_portfolio": self.planning.enable_plan_portfolio,
                "always_include_single_agent_family": self.planning.always_include_single_agent_family,
                "always_include_agentless_pipeline_family": (
                    self.planning.always_include_agentless_pipeline_family
                ),
                "enable_reflective_memory": self.planning.enable_reflective_memory,
                "planner_model": self.planning.planner_model,
                "planner_llm_index": self.planning.planner_llm_index,
                "preplanner_model": self.planning.preplanner_model,
                "preplanner_llm_index": self.planning.preplanner_llm_index,
                "preplanner_timeout_seconds": self.planning.preplanner_timeout_seconds,
                "refinement_timeout_seconds": self.planning.refinement_timeout_seconds,
                "planner_timeout_seconds": self.planning.planner_timeout_seconds,
                "max_keywords": self.planning.max_keywords,
                "max_relevant_files": self.planning.max_relevant_files,
                "include_dependency_neighbors": self.planning.include_dependency_neighbors,
                "max_repo_map_files": self.planning.max_repo_map_files,
                "max_rollout_brief_families": self.planning.max_rollout_brief_families,
                "max_task_state_context_items": self.planning.max_task_state_context_items,
                "max_frontier_targets": self.planning.max_frontier_targets,
                "max_reflection_memory_items": self.planning.max_reflection_memory_items,
                "delegation_boundary_pressure_threshold": (
                    self.planning.delegation_boundary_pressure_threshold
                ),
                "regime_policy": {
                    "completion_task_patterns": list(
                        self.planning.regime_policy.completion_task_patterns
                    ),
                    "public_api_patterns": list(self.planning.regime_policy.public_api_patterns),
                    "probability_thresholds": dict(
                        self.planning.regime_policy.probability_thresholds
                    ),
                    "state_scales": dict(self.planning.regime_policy.state_scales),
                    "evidence_weights": dict(self.planning.regime_policy.evidence_weights),
                },
                "delegation_policy": {
                    "split_confidence_threshold": (
                        self.planning.delegation_policy.split_confidence_threshold
                    ),
                    "boundary_pressure_threshold": (
                        self.planning.delegation_boundary_pressure_threshold
                    ),
                    "bridge_cross_ratio": self.planning.delegation_policy.bridge_cross_ratio,
                    "bridge_weight_min": self.planning.delegation_policy.bridge_weight_min,
                    "low_leverage_cluster_max_files": (
                        self.planning.delegation_policy.low_leverage_cluster_max_files
                    ),
                    "low_leverage_cluster_max_work": (
                        self.planning.delegation_policy.low_leverage_cluster_max_work
                    ),
                    "low_leverage_cluster_work_ratio": (
                        self.planning.delegation_policy.low_leverage_cluster_work_ratio
                    ),
                    "low_leverage_outbound_ratio": (
                        self.planning.delegation_policy.low_leverage_outbound_ratio
                    ),
                    "low_leverage_peer_weight_min": (
                        self.planning.delegation_policy.low_leverage_peer_weight_min
                    ),
                    "low_leverage_confidence_penalty": (
                        self.planning.delegation_policy.low_leverage_confidence_penalty
                    ),
                    "thin_cluster_max_work": (
                        self.planning.delegation_policy.thin_cluster_max_work
                    ),
                    "thin_cluster_work_ratio": (
                        self.planning.delegation_policy.thin_cluster_work_ratio
                    ),
                    "thin_file_max_lines": self.planning.delegation_policy.thin_file_max_lines,
                    "thin_file_max_symbols": self.planning.delegation_policy.thin_file_max_symbols,
                    "exhaustive_bisection_max_files": (
                        self.planning.delegation_policy.exhaustive_bisection_max_files
                    ),
                    "symbol_interface_bonus": (
                        self.planning.delegation_policy.symbol_interface_bonus
                    ),
                    "edit_span_bonus": self.planning.delegation_policy.edit_span_bonus,
                },
                "shadow_policy": {
                    "enabled": self.planning.shadow_policy.enabled,
                    "max_logged_options": self.planning.shadow_policy.max_logged_options,
                },
            },
            "search": {
                "mode": self.search.mode.value,
                "max_expansions": self.search.max_expansions,
                "max_depth": self.search.max_depth,
                "max_frontier_branching": self.search.max_frontier_branching,
                "c_puct": self.search.c_puct,
                "virtual_loss": self.search.virtual_loss,
                "stop_margin": self.search.stop_margin,
                "min_branch_reward": self.search.min_branch_reward,
                "persist_trace": self.search.persist_trace,
                "transition_reward": {
                    "obligation_delta": self.search.transition_reward.obligation_delta,
                    "hypothesis_delta": self.search.transition_reward.hypothesis_delta,
                    "uncertainty_reduction": self.search.transition_reward.uncertainty_reduction,
                    "progress": self.search.transition_reward.progress,
                    "quick_feedback": self.search.transition_reward.quick_feedback,
                    "alignment": self.search.transition_reward.alignment,
                    "patch_bonus": self.search.transition_reward.patch_bonus,
                    "cost_penalty_per_300s": (self.search.transition_reward.cost_penalty_per_300s),
                    "failure_penalty": self.search.transition_reward.failure_penalty,
                },
                "shadow_policy": {
                    "enabled": self.search.shadow_policy.enabled,
                    "max_logged_options": self.search.shadow_policy.max_logged_options,
                },
            },
            "selection": {
                "strategy": self.selection.strategy.value,
                "ast_similarity_threshold": self.selection.ast_similarity_threshold,
                "enable_regression_pruning": self.selection.enable_regression_pruning,
                "cross_validation_enabled": self.selection.cross_validation_enabled,
                "judge_model": self.selection.judge_model,
                "judge_temperature": self.selection.judge_temperature,
                "min_test_pass_rate": self.selection.min_test_pass_rate,
                "selector_max_voters": self.selection.selector_max_voters,
                "selector_max_iterations": self.selection.selector_max_iterations,
                "verification_timeout_seconds": self.selection.verification_timeout_seconds,
                "full_test_timeout_seconds": self.selection.full_test_timeout_seconds,
                "custom_test_timeout_seconds": self.selection.custom_test_timeout_seconds,
                "verification_helper_files": list(self.selection.verification_helper_files),
                "enable_critic_reranking": self.selection.enable_critic_reranking,
                "critic_weight": self.selection.critic_weight,
                "use_critic": self.selection.use_critic,
                "enable_patch_synthesis": self.selection.enable_patch_synthesis,
                "max_synthesis_candidates": self.selection.max_synthesis_candidates,
                "max_synthesis_combinations": self.selection.max_synthesis_combinations,
                "enable_greedy_synthesis_union": self.selection.enable_greedy_synthesis_union,
                "max_synthesis_pool": self.selection.max_synthesis_pool,
                "max_synthesis_union_members": self.selection.max_synthesis_union_members,
                "preserve_standalone_anchor": self.selection.preserve_standalone_anchor,
                "enable_final_acceptance_reviewer": (
                    self.selection.enable_final_acceptance_reviewer
                ),
                "final_acceptance_reviewer_require_distinct_family": (
                    self.selection.final_acceptance_reviewer_require_distinct_family
                ),
            },
            "benchmark": {
                "commit0_primary_evaluation_backend": (
                    self.benchmark.commit0_primary_evaluation_backend.value
                ),
                "commit0_official_audit_selected": self.benchmark.commit0_official_audit_selected,
                "commit0_official_audit_only_if_primary_passes": (
                    self.benchmark.commit0_official_audit_only_if_primary_passes
                ),
                "commit0_transient_audit_rerun_budget": (
                    self.benchmark.commit0_transient_audit_rerun_budget
                ),
                "commit0_transient_audit_require_stable": (
                    self.benchmark.commit0_transient_audit_require_stable
                ),
                "commit0_ndff_exclude_nondeterministic": (
                    self.benchmark.commit0_ndff_exclude_nondeterministic
                ),
                "commit0_docker_memory_limit": self.benchmark.commit0_docker_memory_limit,
                "commit0_docker_shm_size": self.benchmark.commit0_docker_shm_size,
                "commit0_official_audit_parallelism": (
                    self.benchmark.commit0_official_audit_parallelism
                ),
                "commit0_pytest_xdist_workers": (
                    self.benchmark.commit0_pytest_xdist_workers
                ),
                "commit0_pytest_xdist_dist": self.benchmark.commit0_pytest_xdist_dist,
                "commit0_min_free_disk_gb": self.benchmark.commit0_min_free_disk_gb,
                "commit0_prune_stale_task_sandboxes": (
                    self.benchmark.commit0_prune_stale_task_sandboxes
                ),
                "commit0_stale_task_sandbox_min_age_seconds": (
                    self.benchmark.commit0_stale_task_sandbox_min_age_seconds
                ),
                "commit0_use_pytest_json_exitcode": (
                    self.benchmark.commit0_use_pytest_json_exitcode
                ),
                "commit0_audit_candidate_selection": (
                    self.benchmark.commit0_audit_candidate_selection
                ),
                "commit0_audit_candidate_selection_top_k": (
                    self.benchmark.commit0_audit_candidate_selection_top_k
                ),
                "commit0_repo_clone_timeout_seconds": (
                    self.benchmark.commit0_repo_clone_timeout_seconds
                ),
                "commit0_runtime_setup_timeout_seconds": (
                    self.benchmark.commit0_runtime_setup_timeout_seconds
                ),
                "commit0_dependency_install_timeout_seconds": (
                    self.benchmark.commit0_dependency_install_timeout_seconds
                ),
                "commit0_evaluation_timeout_seconds": (
                    self.benchmark.commit0_evaluation_timeout_seconds
                ),
                "commit0_baseline_evaluation_timeout_seconds": (
                    self.benchmark.commit0_baseline_evaluation_timeout_seconds
                ),
                "commit0_agent_target_tool_timeout_seconds": (
                    self.benchmark.commit0_agent_target_tool_timeout_seconds
                ),
                "disable_pytest_plugin_autoload": (self.benchmark.disable_pytest_plugin_autoload),
                "commit0_docker_fallback_on_failure": (
                    self.benchmark.commit0_docker_fallback_on_failure
                ),
                "commit0_docker_runtime_mode": self.benchmark.commit0_docker_runtime_mode,
                "commit0_require_all_configured_cli_backends": (
                    self.benchmark.commit0_require_all_configured_cli_backends
                ),
                "commit0_optional_configured_cli_backends": list(
                    self.benchmark.commit0_optional_configured_cli_backends
                ),
                "task_parallelism": self.benchmark.task_parallelism,
                "evaluation_power_mode": self.benchmark.evaluation_power_mode,
                "unbounded_followup_budget": self.benchmark.unbounded_followup_budget,
                "evaluation_contract": copy.deepcopy(self.benchmark.evaluation_contract),
                "runtime_policy": copy.deepcopy(self.benchmark.runtime_policy),
                "patch_hygiene": copy.deepcopy(self.benchmark.patch_hygiene),
                "run_supervisor": copy.deepcopy(self.benchmark.run_supervisor),
                "reporting": copy.deepcopy(self.benchmark.reporting),
                "testgen_task_timeout_seconds": (self.benchmark.testgen_task_timeout_seconds),
                "fairness_audit_mode": self.benchmark.fairness_audit_mode,
                "swtbench_fairness_audit_mode": self.benchmark.swtbench_fairness_audit_mode,
                "testgeneval_apply_upstream_patches": (
                    self.benchmark.testgeneval_apply_upstream_patches
                ),
                "testgeneval_apply_baseline_covs_patch_only": (
                    self.benchmark.testgeneval_apply_baseline_covs_patch_only
                ),
                "testgeneval_fairness_audit_mode": (self.benchmark.testgeneval_fairness_audit_mode),
                # Phase A.1 / A.3 Decisive-Edge fields. Always serialized
                # so reviewers can audit which agent surface produced a
                # run and which scoring number was published as headline.
                "default_agent_mode": self.benchmark.default_agent_mode,
                "report_headline_metric": self.benchmark.report_headline_metric,
            },
            "controller_models": self.controller_models.to_dict(),
            "controller_trace": {
                "enabled": self.controller_trace.enabled,
                "filename": self.controller_trace.filename,
                "max_options": self.controller_trace.max_options,
            },
            "orchestration": {
                "max_strategy_iterations": self.orchestration.max_strategy_iterations,
                "repo_token_cap": self.orchestration.repo_token_cap,
                "repeated_blocker_stop_after": (self.orchestration.repeated_blocker_stop_after),
                "adaptive_followup_near_miss_multiplier": (
                    self.orchestration.adaptive_followup_near_miss_multiplier
                ),
                "adaptive_followup_near_miss_pass_rate": (
                    self.orchestration.adaptive_followup_near_miss_pass_rate
                ),
                "max_coverage_gap_followup_rounds": (
                    self.orchestration.max_coverage_gap_followup_rounds
                ),
                "seed_diversity_overlap_threshold": (
                    self.orchestration.seed_diversity_overlap_threshold
                ),
                "active_learning_max_iterations": (
                    self.orchestration.active_learning_max_iterations
                ),
                "testgen_allow_agentic_edit_loop": (
                    self.orchestration.testgen_allow_agentic_edit_loop
                ),
            },
            "use_concise_prompts": self.use_concise_prompts,
            "enable_planning_tool": self.enable_planning_tool,
            "workspace_dir": self.workspace_dir,
            "output_dir": self.output_dir,
            "log_level": self.log_level,
            "save_trajectories": self.save_trajectories,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def _get_llm_for_index(self, index: int, *, label: str) -> LLMConfig:
        if index < 0 or index >= len(self.llm_configs):
            raise IndexError(
                f"{label} references llm_configs[{index}], "
                f"but only {len(self.llm_configs)} configs are defined."
            )
        return self.llm_configs[index]

    def _get_llm_profile_for_rollout(self, rollout_idx: int) -> Optional[dict[str, int]]:
        if not self.rollout.llm_profiles:
            return None
        return self.rollout.llm_profiles[rollout_idx % len(self.rollout.llm_profiles)]

    def get_llm_profile(self, profile_index: int) -> Optional[dict[str, int]]:
        if not self.rollout.llm_profiles:
            return None
        normalized_index = int(profile_index)
        if normalized_index < 0 or normalized_index >= len(self.rollout.llm_profiles):
            raise IndexError(
                f"Rollout profile index {normalized_index} is out of range for "
                f"{len(self.rollout.llm_profiles)} configured profiles."
            )
        return self.rollout.llm_profiles[normalized_index]

    def get_llm_for_rollout_profile(self, profile_index: int) -> LLMConfig:
        profile = self.get_llm_profile(profile_index)
        if profile is None:
            return self.get_llm_for_rollout(profile_index)
        override_idx = profile.get(ROLLOUT_LLM_PROFILE_BASE_KEY)
        if override_idx is not None:
            return self._get_llm_for_index(
                override_idx,
                label=f"Rollout profile {profile_index}",
            )
        return self.llm_configs[profile_index % len(self.llm_configs)]

    def get_llm_for_profile_stage(self, profile_index: int, stage_name: str) -> LLMConfig:
        normalized_stage = _normalize_scaffold_stage_name(stage_name)
        profile = self.get_llm_profile(profile_index)
        if profile is None:
            return self.get_llm_for_stage(profile_index, normalized_stage)
        profile_idx = profile.get(normalized_stage)
        if profile_idx is not None:
            return self._get_llm_for_index(
                profile_idx,
                label=f"Rollout profile {profile_index} stage '{normalized_stage}'",
            )
        override_idx = self.rollout.scaffold_stage_llm_indices.get(normalized_stage)
        if override_idx is None:
            rollout_override_idx = profile.get(ROLLOUT_LLM_PROFILE_BASE_KEY)
            if rollout_override_idx is not None:
                return self._get_llm_for_index(
                    rollout_override_idx,
                    label=f"Rollout profile {profile_index}",
                )
            return self.get_llm_for_rollout_profile(profile_index)
        return self._get_llm_for_index(
            override_idx,
            label=f"Scaffold stage '{normalized_stage}'",
        )

    def get_llm_for_rollout(self, rollout_idx: int) -> LLMConfig:
        profile = self._get_llm_profile_for_rollout(rollout_idx)
        if profile is not None:
            return self.get_llm_for_rollout_profile(rollout_idx % len(self.rollout.llm_profiles))
        return self.llm_configs[rollout_idx % len(self.llm_configs)]

    def get_llm_for_stage(self, rollout_idx: int, stage_name: str) -> LLMConfig:
        normalized_stage = _normalize_scaffold_stage_name(stage_name)
        profile = self._get_llm_profile_for_rollout(rollout_idx)
        if profile is not None:
            return self.get_llm_for_profile_stage(
                rollout_idx % len(self.rollout.llm_profiles),
                normalized_stage,
            )
        override_idx = self.rollout.scaffold_stage_llm_indices.get(normalized_stage)
        if override_idx is None:
            return self.get_llm_for_rollout(rollout_idx)
        return self._get_llm_for_index(
            override_idx,
            label=f"Scaffold stage '{normalized_stage}'",
        )

    def get_rollout_stage_model_signature_for_profile(self, profile_index: int) -> tuple[str, ...]:
        return tuple(
            self.get_llm_for_profile_stage(profile_index, stage_name).model or ""
            for stage_name in ROLLOUT_PROFILE_STAGE_ORDER
        )

    def get_rollout_stage_model_signature(self, rollout_idx: int) -> tuple[str, ...]:
        profile = self._get_llm_profile_for_rollout(rollout_idx)
        if profile is not None:
            return self.get_rollout_stage_model_signature_for_profile(
                rollout_idx % len(self.rollout.llm_profiles)
            )
        return tuple(
            self.get_llm_for_stage(rollout_idx, stage_name).model or ""
            for stage_name in ROLLOUT_PROFILE_STAGE_ORDER
        )

    def get_rollout_diversity_signature_for_profile(
        self,
        profile_index: int,
        *,
        prompt_strategy: Optional[PromptStrategy] = None,
        temperature: Optional[float] = None,
        include_prompt_strategy: bool = False,
        include_temperature: bool = False,
    ) -> tuple[str, ...]:
        signature = list(self.get_rollout_stage_model_signature_for_profile(profile_index))
        if include_prompt_strategy:
            strategy = prompt_strategy or self.get_prompt_strategy_for_rollout(profile_index)
            signature.append(strategy.value)
        if include_temperature:
            resolved_temperature = (
                float(temperature)
                if temperature is not None
                else self.get_temperature_for_rollout(profile_index)
            )
            signature.append(f"{resolved_temperature:.2f}")
        return tuple(signature)

    def get_rollout_diversity_signature(
        self,
        rollout_idx: int,
        *,
        include_prompt_strategy: bool = False,
        include_temperature: bool = False,
    ) -> tuple[str, ...]:
        profile = self._get_llm_profile_for_rollout(rollout_idx)
        if profile is not None:
            return self.get_rollout_diversity_signature_for_profile(
                rollout_idx % len(self.rollout.llm_profiles),
                prompt_strategy=self.get_prompt_strategy_for_rollout(rollout_idx),
                temperature=self.get_temperature_for_rollout(rollout_idx),
                include_prompt_strategy=include_prompt_strategy,
                include_temperature=include_temperature,
            )
        signature = list(self.get_rollout_stage_model_signature(rollout_idx))
        if include_prompt_strategy:
            signature.append(self.get_prompt_strategy_for_rollout(rollout_idx).value)
        if include_temperature:
            signature.append(f"{self.get_temperature_for_rollout(rollout_idx):.2f}")
        return tuple(signature)

    def count_distinct_rollout_profiles(
        self,
        rollout_count: int,
        *,
        include_prompt_strategy: bool = False,
        include_temperature: bool = False,
    ) -> int:
        normalized_count = max(0, int(rollout_count))
        if normalized_count <= 0:
            return 0
        return len(
            {
                self.get_rollout_diversity_signature(
                    rollout_idx,
                    include_prompt_strategy=include_prompt_strategy,
                    include_temperature=include_temperature,
                )
                for rollout_idx in range(normalized_count)
            }
        )

    def get_temperature_for_rollout(self, rollout_idx: int) -> float:
        temperatures = self.rollout.diversity_temperatures
        return temperatures[rollout_idx % len(temperatures)]

    def get_prompt_strategy_for_rollout(self, rollout_idx: int) -> PromptStrategy:
        strategies = self.rollout.diversity_prompts
        return strategies[rollout_idx % len(strategies)]

    def _clone_llm_config_with_overrides(
        self,
        primary: LLMConfig,
        *,
        model: Any = _LLM_CONFIG_KEEP,
    ) -> LLMConfig:
        return LLMConfig(
            backend=primary.backend,
            model=primary.model if model is _LLM_CONFIG_KEEP else model,
            api_key_env=primary.api_key_env,
            base_url=primary.base_url,
            temperature=primary.temperature,
            max_tokens=primary.max_tokens,
            timeout=primary.timeout,
            cli_command=primary.cli_command,
            cli_args=list(primary.cli_args),
            cli_model_id=primary.cli_model_id,
            cli_timeout=primary.cli_timeout,
            cli_hard_timeout_seconds=primary.cli_hard_timeout_seconds,
            cli_first_output_timeout_seconds=primary.cli_first_output_timeout_seconds,
            cli_strict_hard_timeout=primary.cli_strict_hard_timeout,
            cli_output_capture_max_chars=primary.cli_output_capture_max_chars,
            cli_disable_osx_sandbox=primary.cli_disable_osx_sandbox,
            cli_permission_mode=primary.cli_permission_mode,
            cli_env_overrides=dict(primary.cli_env_overrides),
            cli_env_redaction_disabled=primary.cli_env_redaction_disabled,
            cli_tool_review_enabled=primary.cli_tool_review_enabled,
            cli_tool_review_reviewer_backend=primary.cli_tool_review_reviewer_backend,
            cli_tool_review_reviewer_command=primary.cli_tool_review_reviewer_command,
            cli_tool_review_timeout_seconds=primary.cli_tool_review_timeout_seconds,
        )

    def get_planner_llm(self) -> LLMConfig:
        planner_index = self.planning.planner_llm_index
        if isinstance(planner_index, int):
            if planner_index < 0 or planner_index >= len(self.llm_configs):
                raise ValueError(
                    f"planning.planner_llm_index={planner_index} is out of range for "
                    f"{len(self.llm_configs)} llm_configs"
                )
            primary = self.llm_configs[planner_index]
        else:
            primary = self.llm_configs[0]
        planner_model = self.planning.planner_model
        if planner_model:
            planner = self._clone_llm_config_with_overrides(
                primary,
                model=planner_model if planner_model else _LLM_CONFIG_KEEP,
            )
            if planner_model:
                planner.temperature = 0.0
            return planner
        return primary

    def get_preplanner_llm(self) -> LLMConfig:
        preplanner_index = self.planning.preplanner_llm_index
        if isinstance(preplanner_index, int):
            if preplanner_index < 0 or preplanner_index >= len(self.llm_configs):
                raise ValueError(
                    f"planning.preplanner_llm_index={preplanner_index} is out of range for "
                    f"{len(self.llm_configs)} llm_configs"
                )
            primary = self.llm_configs[preplanner_index]
        else:
            primary = next(
                (
                    config
                    for config in self.llm_configs
                    if config.model == "gpt-5.5"
                    and config.backend in {LLMBackend.CODEX_CLI, LLMBackend.OPENAI_API}
                ),
                self.get_planner_llm(),
            )
        preplanner_model = self.planning.preplanner_model
        if preplanner_model:
            return self._clone_llm_config_with_overrides(
                primary,
                model=preplanner_model if preplanner_model else _LLM_CONFIG_KEEP,
            )
        return primary

    def agent_team_parallelism_cap(
        self,
        *,
        llm_config: Optional[LLMConfig] = None,
    ) -> int:
        del llm_config
        return max(1, int(self.aci.max_agent_team_parallelism or 1))

    def clamp_agent_team_parallelism(
        self,
        requested_parallelism: Any,
        *,
        llm_config: Optional[LLMConfig] = None,
        max_tasks: Optional[int] = None,
    ) -> int:
        try:
            parallelism = int(requested_parallelism or 1)
        except (TypeError, ValueError):
            parallelism = 1
        parallelism = max(1, parallelism)
        cap = self.agent_team_parallelism_cap(llm_config=llm_config)
        if isinstance(max_tasks, int) and max_tasks > 0:
            cap = min(cap, max_tasks)
        return max(1, min(parallelism, cap))


_VALID_LOCALIZER_ENFORCEMENT_VALUES: frozenset[str] = frozenset(
    {"advisory", "warning", "hard_constraint"}
)


def apply_localizer_enforcement_override(config: "ApexConfig") -> "ApexConfig":
    """Decisive-Edge B.1: lift ``BenchmarkConfig.localizer_enforcement_override``
    onto ``RolloutConfig.localizer_enforcement`` when set.

    The benchmark runner / CLI bootstrap calls this at config-resolution
    time so per-benchmark relaxation (or tightening) of the localizer
    enforcement mode is honoured by every consumer that already reads
    ``config.rollout.localizer_enforcement`` (PatcherAgent, the CLI
    patcher post-validation in ``apex/rollout/engine.py``, and the
    diagnostics dump in :meth:`ApexConfig.to_diagnostics_dict`).

    Mutates ``config`` in place and returns it so callers can chain.
    Unknown / empty override values are ignored (with a debug log) so a
    typo in the JSON config can't silently disable enforcement.
    """
    benchmark = getattr(config, "benchmark", None)
    rollout = getattr(config, "rollout", None)
    if benchmark is None or rollout is None:
        return config
    raw = getattr(benchmark, "localizer_enforcement_override", None)
    if raw is None:
        return config
    override = str(raw or "").strip().lower()
    if not override:
        return config
    if override not in _VALID_LOCALIZER_ENFORCEMENT_VALUES:
        # Bad value: leave the rollout default in place.
        import logging as _logging

        _logging.getLogger("apex.config").warning(
            "BenchmarkConfig.localizer_enforcement_override=%r is not "
            "one of %s; keeping rollout.localizer_enforcement=%r.",
            raw,
            sorted(_VALID_LOCALIZER_ENFORCEMENT_VALUES),
            getattr(rollout, "localizer_enforcement", None),
        )
        return config
    rollout.localizer_enforcement = override
    return config


def _coerce_enum(value: Any, enum_cls: type[Enum]) -> Any:
    if isinstance(value, enum_cls):
        return value
    return enum_cls(value)


def _normalize_scaffold_stage_name(stage_name: Any) -> str:
    normalized = str(stage_name).strip().lower()
    if normalized not in SCAFFOLD_STAGE_NAMES:
        allowed = ", ".join(sorted(SCAFFOLD_STAGE_NAMES))
        raise ValueError(f"Unknown scaffold stage '{stage_name}'. Expected one of: {allowed}.")
    return normalized


def _normalize_llm_profile_key(key: Any) -> str:
    normalized = str(key).strip().lower()
    if normalized == ROLLOUT_LLM_PROFILE_BASE_KEY:
        return normalized
    return _normalize_scaffold_stage_name(normalized)


def _coerce_stage_llm_indices(value: Any, *, llm_config_count: int) -> dict[str, int]:
    if not isinstance(value, dict):
        raise TypeError(
            "rollout.scaffold_stage_llm_indices must be a mapping of stage names to llm config indices."
        )

    normalized: dict[str, int] = {}
    for stage_name, raw_index in value.items():
        normalized_stage = _normalize_scaffold_stage_name(stage_name)
        index = int(raw_index)
        if index < 0 or index >= llm_config_count:
            raise ValueError(
                f"Scaffold stage '{normalized_stage}' references llm_configs[{index}], "
                f"but only {llm_config_count} configs are defined."
            )
        normalized[normalized_stage] = index
    return normalized


def _coerce_llm_profiles(value: Any, *, llm_config_count: int) -> list[dict[str, int]]:
    if not isinstance(value, list):
        raise TypeError("rollout.llm_profiles must be a list of stage/profile mappings.")

    normalized_profiles: list[dict[str, int]] = []
    for profile_index, raw_profile in enumerate(value):
        if not isinstance(raw_profile, dict):
            raise TypeError(
                "rollout.llm_profiles entries must be mappings of stage/profile names to llm config indices."
            )
        normalized_profile: dict[str, int] = {}
        for raw_key, raw_index in raw_profile.items():
            normalized_key = _normalize_llm_profile_key(raw_key)
            index = int(raw_index)
            if index < 0 or index >= llm_config_count:
                raise ValueError(
                    f"rollout.llm_profiles[{profile_index}] key '{normalized_key}' references "
                    f"llm_configs[{index}], but only {llm_config_count} configs are defined."
                )
            normalized_profile[normalized_key] = index
        normalized_profiles.append(normalized_profile)
    return normalized_profiles


def _coerce_standalone_anchor_candidates(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("rollout.standalone_anchor_candidates must be a list.")

    out: list[dict[str, Any]] = []
    for candidate_index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise TypeError("rollout.standalone_anchor_candidates entries must be mappings.")
        payload = dict(entry)
        if "backend" in payload and str(payload.get("backend") or "").strip():
            payload["backend"] = LLMBackend(payload["backend"]).value
        if "model" in payload and str(payload.get("model") or "").strip():
            payload["model"] = normalize_supported_model_name(payload["model"])
        if "llm_config_index" in payload:
            try:
                index = int(payload["llm_config_index"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "rollout.standalone_anchor_candidates"
                    f"[{candidate_index}].llm_config_index must be an integer."
                ) from exc
            if index < 0:
                raise ValueError(
                    "rollout.standalone_anchor_candidates"
                    f"[{candidate_index}].llm_config_index must be >= 0."
                )
            payload["llm_config_index"] = index
        out.append(payload)
    return out
