"""
Shared run artifact helpers for manifests, live status, and comparisons.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from ..core.cli_backend import (
    _BACKEND_AUTH_ALLOWLIST,
    _BACKEND_AUTH_REQUIREMENTS,
    CLIModelClient,
    probe_cli_backend_health,
)
from ..core.config import ApexConfig, LLMBackend, LLMConfig
from ..core.llm_routing import resolve_available_llm_config
from .checkpointing import atomic_write_json, atomic_write_text, load_json_if_exists

RUN_MANIFEST_FILENAME = "run_manifest.json"
TASK_LIVE_STATE_FILENAME = "task_live_state.json"
ROLLOUT_STATUS_DIRNAME = "rollout_status"
CLI_RETRY_DIAGNOSTICS_DIRNAME = "cli_retry_diagnostics"
RUN_ARTIFACT_SCHEMA_VERSION = 1
BENCHMARK_POLICY_SCHEMA_VERSION = 1

_APEX_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ALLOWED_CLI_BACKENDS = (
    LLMBackend.CODEX_CLI,
    LLMBackend.CLAUDE_CLI,
    LLMBackend.GEMINI_CLI,
    LLMBackend.OPENCODE_CLI,
    LLMBackend.METACODE_CLI,
)
_MODEL_PROXY_ENV_NAMES = frozenset(
    {
        "ANTHROPIC_BASE_URL",
        "APEX_AGENT_MODEL_PROXY_URL",
        "APEX_CLAUDE_CLI_MODEL_PROXY_URL",
        "APEX_CLAUDE_MODEL_PROXY_URL",
        "APEX_CODEX_CLI_MODEL_PROXY_URL",
        "APEX_CODEX_MODEL_PROXY_URL",
        "APEX_GEMINI_CLI_MODEL_PROXY_URL",
        "APEX_GEMINI_MODEL_PROXY_URL",
        "APEX_HOST_MODEL_PROXY_URL",
        "APEX_METACODE_CLI_MODEL_PROXY_URL",
        "APEX_METACODE_MODEL_PROXY_URL",
        "APEX_OPENCODE_CLI_MODEL_PROXY_URL",
        "APEX_OPENCODE_MODEL_PROXY_URL",
        "CODEX_BASE_URL",
        "CODE_ASSIST_ENDPOINT",
        "GOOGLE_GEMINI_BASE_URL",
        "GOOGLE_VERTEX_BASE_URL",
        "OPENAI_BASE_URL",
    }
)
_RUN_RESERVED_DIRS = {
    "workspaces",
    ".runtime",
    "_scripts_cache",
    "target_runtime_tools",
    "target_authoring_tool_shims",
    ROLLOUT_STATUS_DIRNAME,
    "__pycache__",
}

_TASK_LIVE_TERMINAL_STATUSES = {
    "completed",
    "failed",
    "error",
    "skipped",
    "cancelled",
    "canceled",
    "interrupted",
    "stopped",
}


def _rollout_live_state_is_terminal(payload: dict[str, Any]) -> bool:
    if bool(payload.get("rollout_terminal")):
        return True
    stage = str(payload.get("stage") or "").strip()
    return stage == "rollout_finished"


def _rollout_live_state_is_scheduler_cancelled(payload: dict[str, Any]) -> bool:
    if bool(payload.get("scheduler_cancelled")):
        return True
    terminal_state = str(payload.get("terminal_state") or "").strip().lower()
    return terminal_state in {"cancelled", "canceled"}


def _task_live_state_is_terminal(payload: dict[str, Any]) -> bool:
    if bool(payload.get("terminal")):
        return True
    status = str(payload.get("status") or "").strip().lower()
    if status in _TASK_LIVE_TERMINAL_STATUSES:
        return True
    phase = str(payload.get("phase") or "").strip().lower()
    return phase == "completed"


_RETRY_PROGRESS_SOURCES = {
    "agent_nonterminal_retry",
    "bootstrap_retry",
    "content_free_retry",
    "startup_retry",
    "transient_infra_retry",
    "transient_infra_retry_after_workspace_activity",
    "transient_infra_degraded_candidate",
}


def _retry_diagnostic_from_progress_payload(payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    source = str(payload.get("last_progress_source") or payload.get("state") or "").strip()
    retry_reason = str(payload.get("retry_reason") or "").strip()
    diagnostic_path = str(
        payload.get("diagnostic_path") or payload.get("retry_diagnostic_path") or ""
    ).strip()
    if not (retry_reason or diagnostic_path or source in _RETRY_PROGRESS_SOURCES):
        return None

    retry_kind = str(payload.get("retry_kind") or source or "retry").strip() or "retry"
    summary: dict[str, Any] = {"retry_kind": retry_kind}
    if retry_reason:
        summary["retry_reason"] = retry_reason
    if diagnostic_path:
        summary["diagnostic_path"] = diagnostic_path
    for source_key, target_key in (
        ("attempt_index", "attempt_index"),
        ("max_attempts", "max_attempts"),
        ("process_pid", "process_pid"),
        ("backend", "backend"),
        ("model", "model"),
        ("stage", "stage"),
        ("last_progress_at", "observed_at"),
    ):
        value = payload.get(source_key)
        if value is not None:
            summary[target_key] = value
    return summary


def _merge_retry_diagnostics(
    existing: Any,
    incoming: Any,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    seen: dict[str, int] = {}

    def _append(item: Any) -> None:
        if not isinstance(item, dict):
            return
        summary = dict(item)
        diagnostic_path = str(summary.get("diagnostic_path") or "").strip()
        identity = (
            f"diagnostic_path:{diagnostic_path}"
            if diagnostic_path
            else json.dumps(summary, sort_keys=True, default=str)
        )
        if identity in seen:
            existing = diagnostics[seen[identity]]
            for key, value in summary.items():
                if key not in existing and value is not None:
                    existing[key] = value
            return
        seen[identity] = len(diagnostics)
        diagnostics.append(summary)

    for value in (existing, incoming):
        if isinstance(value, list):
            for item in value:
                _append(item)
        else:
            _append(value)
    return diagnostics


def _persist_retry_diagnostic_artifacts(
    *,
    task_output_dir: str | Path,
    rollout_id: int,
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    persisted: list[dict[str, Any]] = []
    if not diagnostics:
        return persisted
    task_dir = Path(task_output_dir)
    target_dir = task_dir / CLI_RETRY_DIAGNOSTICS_DIRNAME / f"rollout_{int(rollout_id)}"
    for index, item in enumerate(diagnostics):
        summary = dict(item)
        source_text = str(summary.get("diagnostic_path") or "").strip()
        if not source_text:
            persisted.append(summary)
            continue
        source_path = Path(source_text)
        if not source_path.is_file():
            persisted.append(summary)
            continue
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_path.name).strip("._-")
            if not safe_name:
                safe_name = f"retry_{index}.json"
            target_path = target_dir / safe_name
            if source_path.resolve() != target_path.resolve():
                shutil.copy2(source_path, target_path)
            summary["persisted_diagnostic_path"] = str(target_path)
            try:
                summary["persisted_diagnostic_relative_path"] = str(
                    target_path.relative_to(task_dir)
                )
            except ValueError:
                pass
        except OSError as exc:
            summary["diagnostic_persist_error"] = f"{type(exc).__name__}: {exc}"
        persisted.append(summary)
    return persisted


@dataclass(frozen=True)
class JoinReport:
    records_seen: int
    records_updated: int
    missing_harness_data: int
    harness_rows_without_records: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def hash_config_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def hash_config(config: ApexConfig) -> str:
    return hash_config_payload(config.to_dict())


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _hash_file(path: str | Path) -> Optional[str]:
    candidate = Path(path)
    try:
        return hashlib.sha256(candidate.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def _safe_source_hash(obj: Any) -> Optional[str]:
    try:
        return _hash_text(inspect.getsource(obj))
    except (OSError, TypeError):
        return None


def capture_environment_snapshot(config: Optional[ApexConfig] = None) -> dict[str, Any]:
    path_entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    configured_backends = []
    if config is not None:
        configured_backends = [
            {
                "backend": llm_config.backend.value,
                "model": llm_config.model,
                "command": llm_config.resolved_cli_command if llm_config.is_cli_backend else None,
            }
            for llm_config in config.llm_configs
        ]
    # Reproducibility-relevant runtime knobs that NeurIPS reviewers will
    # want to see captured per run.
    reproducibility = {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
        "DOCKER_HOST": os.environ.get("DOCKER_HOST"),
        "TZ": os.environ.get("TZ"),
        # Capture wall clock and absolute UTC date so re-runs of the same
        # config can be diffed against the underlying model's silent
        # alias-roll on the provider side.
        "wall_clock_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_epoch": int(time.time()),
    }
    return {
        "cwd": str(Path.cwd().resolve()),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "path_entries_head": path_entries[:12],
        "configured_backends": configured_backends,
        "reproducibility": reproducibility,
    }


def build_prompt_template_fingerprints() -> dict[str, Any]:
    from ..agents import prompts as prompt_module
    from ..planning import manager as planning_manager_module

    issue_planner = planning_manager_module.IssuePlanner
    fingerprints = {
        "agents_prompts_py": _hash_file(prompt_module.__file__),
        "planning_manager_py": _hash_file(planning_manager_module.__file__),
        "solver_system_prompt": _hash_text(prompt_module.SOLVER_SYSTEM_PROMPT),
        "reproducer_system_prompt": _hash_text(prompt_module.REPRODUCER_SYSTEM_PROMPT),
        "localizer_system_prompt": _hash_text(prompt_module.LOCALIZER_SYSTEM_PROMPT),
        "test_writer_system_prompt": _hash_text(prompt_module.TEST_WRITER_SYSTEM_PROMPT),
        "build_stage_system_prompt": _safe_source_hash(prompt_module.build_stage_system_prompt),
        "build_solver_prompt": _safe_source_hash(prompt_module.build_solver_prompt),
        "build_reproducer_prompt": _safe_source_hash(prompt_module.build_reproducer_prompt),
        "build_localizer_prompt": _safe_source_hash(prompt_module.build_localizer_prompt),
        "build_test_writer_prompt": _safe_source_hash(prompt_module.build_test_writer_prompt),
        "planner_run_plan_prompt": _safe_source_hash(issue_planner._run_plan_prompt),
        "planner_signal_renderer": _safe_source_hash(
            issue_planner._render_planner_baseline_signal_block
        ),
    }
    fingerprints["combined_hash"] = hash_config_payload(
        {key: value for key, value in fingerprints.items() if value is not None}
    )
    return fingerprints


def _run_capture(
    command: list[str], *, cwd: Optional[Path] = None, env: Optional[dict[str, str]] = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(cwd) if cwd else None,
        env=env,
    )


def detect_git_snapshot(root: Optional[Path] = None) -> dict[str, Any]:
    repo_root = (root or _APEX_PROJECT_ROOT).resolve()
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        return {
            "root": str(repo_root),
            "available": False,
            "sha": None,
            "branch": None,
            "dirty": None,
        }

    sha = _run_capture(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    branch = _run_capture(["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = _run_capture(["git", "-C", str(repo_root), "status", "--porcelain"])
    return {
        "root": str(repo_root),
        "available": sha.returncode == 0,
        "sha": sha.stdout.strip() or None,
        "branch": branch.stdout.strip() or None,
        "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def _auth_snapshot_for_backend(config: LLMConfig) -> dict[str, Any]:
    blocked_env_by_backend = {
        LLMBackend.CODEX_CLI: ("OPENAI_API_KEY", "CODEX_API_KEY"),
        LLMBackend.CLAUDE_CLI: ("ANTHROPIC_API_KEY",),
        LLMBackend.GEMINI_CLI: ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        LLMBackend.OPENCODE_CLI: ("OPENROUTER_API_KEY", "OPENCODE_API_KEY"),
        LLMBackend.METACODE_CLI: ("OPENROUTER_API_KEY", "OPENCODE_API_KEY"),
    }
    blocked = blocked_env_by_backend.get(config.backend, ())
    present_but_ignored = [name for name in blocked if os.environ.get(name)]
    # Record which provider routing/auth env vars are actually present (NAMES
    # only — never values) so the preflight artifact reflects whether the launch
    # shell carried the backend's auth env. A hardcoded empty list previously
    # masked the most common Commit0 failure (claude/gemini CLIs unauthenticated
    # because the launch shell lacked the Vertex/gateway/plugboard env).
    allowlist = _BACKEND_AUTH_ALLOWLIST.get(config.backend, ())
    blocked_set = set(blocked)
    # Provider API keys that APEX deliberately does NOT pass to the agentic CLI
    # (CLI-is-agent) belong in ``ignored_env_vars``, not here — even if they also
    # appear in the backend's auth allowlist (e.g. codex's OPENAI_API_KEY).
    present_env_vars = [
        name
        for name in allowlist
        if name not in blocked_set and str(os.environ.get(name) or "").strip()
    ]
    present_set = set(present_env_vars)
    requirements = _BACKEND_AUTH_REQUIREMENTS.get(config.backend, ())
    auth_env_satisfied = any(
        requirement and all(name in present_set for name in requirement)
        for requirement in requirements
    )
    return {
        "mode": "cli_session",
        "present_env_vars": present_env_vars,
        "ignored_env_vars": present_but_ignored,
        # "session_required" still means APEX must materialize CLI session auth;
        # auth_env_satisfied tells operators whether env-based provider routing
        # was already configured at launch (the fast path for in-container auth).
        "auth_env_satisfied": auth_env_satisfied,
        "status": "session_required",
        "note": "Provider API-key env vars are not passed to agentic CLI subprocesses.",
    }


def _cli_version_snapshot(config: LLMConfig) -> dict[str, Any]:
    env = CLIModelClient(config)._build_subprocess_env()
    resolved = shutil.which(config.resolved_cli_command, path=env.get("PATH")) or shutil.which(
        config.resolved_cli_command
    )
    if str(env.get("APEX_TARGET_TOOL_CONTEXT") or "").strip():
        return {
            "command": config.resolved_cli_command,
            "resolved_path": resolved,
            "version": None,
            "exit_code": None,
            "error": "skipped: target-runtime health probe",
        }
    if resolved is None:
        return {
            "command": config.resolved_cli_command,
            "resolved_path": None,
            "version": None,
            "exit_code": None,
            "error": "not_installed",
        }
    try:
        result = subprocess.run(
            [resolved, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {
            "command": config.resolved_cli_command,
            "resolved_path": resolved,
            "version": None,
            "exit_code": None,
            "error": "timeout",
        }
    except OSError as exc:
        return {
            "command": config.resolved_cli_command,
            "resolved_path": resolved,
            "version": None,
            "exit_code": None,
            "error": str(exc),
        }
    text = (result.stdout or result.stderr or "").strip()
    return {
        "command": config.resolved_cli_command,
        "resolved_path": resolved,
        "version": text.splitlines()[0] if text else None,
        "exit_code": result.returncode,
        "error": None if result.returncode == 0 else (text or "probe_failed"),
    }


def build_backend_snapshot(
    config: LLMConfig,
    *,
    refresh_health: bool = False,
) -> dict[str, Any]:
    probe_started_at = time.time()
    healthy, reason = probe_cli_backend_health(config, refresh=refresh_health)
    probe_ended_at = time.time()
    env = CLIModelClient(config)._build_subprocess_env()
    target_runtime_context = str(env.get("APEX_TARGET_TOOL_CONTEXT") or "").strip()
    reason_text = str(reason or "")
    version = _cli_version_snapshot(config)
    return {
        "backend": config.backend.value,
        "model": config.model,
        "command": config.resolved_cli_command,
        "healthy": healthy,
        "unavailable_reason": reason_text,
        "version": version,
        "auth": _auth_snapshot_for_backend(config),
        "health_probe": {
            "refresh_health": bool(refresh_health),
            "started_at": probe_started_at,
            "ended_at": probe_ended_at,
            "elapsed_seconds": max(0.0, probe_ended_at - probe_started_at),
            "healthy": bool(healthy),
            "target_runtime_context_present": bool(target_runtime_context),
            "target_runtime_context_fingerprint": (
                hashlib.sha256(target_runtime_context.encode("utf-8")).hexdigest()[:12]
                if target_runtime_context
                else ""
            ),
            "target_runtime_auth_mode": str(
                env.get("APEX_TARGET_RUNTIME_CLI_AUTH_MODE") or ""
            ).strip(),
            "model_proxy_env_vars_present": sorted(
                key for key in _MODEL_PROXY_ENV_NAMES if str(env.get(key) or "").strip()
            ),
            "unavailable_reason_excerpt": reason_text[:800],
        },
    }


def _configured_backend_map(config: ApexConfig) -> dict[LLMBackend, LLMConfig]:
    mapping: dict[LLMBackend, LLMConfig] = {}
    for llm_config in config.llm_configs:
        if llm_config.is_cli_backend and llm_config.backend not in mapping:
            mapping[llm_config.backend] = llm_config
    return mapping


def build_allowed_backend_snapshots(
    config: ApexConfig,
    *,
    refresh_health: bool = False,
    include_unconfigured: bool = False,
) -> list[dict[str, Any]]:
    configured = _configured_backend_map(config)
    snapshots: list[dict[str, Any]] = []
    backends = (
        _ALLOWED_CLI_BACKENDS
        if include_unconfigured
        else tuple(backend for backend in _ALLOWED_CLI_BACKENDS if backend in configured)
    )
    for backend in backends:
        llm_config = configured.get(backend, LLMConfig(backend=backend))
        snapshot = build_backend_snapshot(llm_config, refresh_health=refresh_health)
        snapshot["configured"] = backend in configured
        snapshots.append(snapshot)
    return snapshots


_ROLLOUT_PROFILE_STAGE_NAMES = (
    "rollout",
    "reproducer",
    "localizer",
    "patcher",
    "test_writer",
)


def _llm_config_index(config: ApexConfig, llm_config: LLMConfig) -> Optional[int]:
    for index, candidate in enumerate(config.llm_configs):
        if candidate is llm_config:
            return index
    for index, candidate in enumerate(config.llm_configs):
        if (
            candidate.backend == llm_config.backend
            and candidate.model == llm_config.model
            and candidate.cli_model_id == llm_config.cli_model_id
            and candidate.resolved_cli_command == llm_config.resolved_cli_command
        ):
            return index
    return None


def _llm_config_identity(config: ApexConfig, llm_config: LLMConfig) -> dict[str, Any]:
    return {
        "index": _llm_config_index(config, llm_config),
        "backend": llm_config.backend.value,
        "model": llm_config.model,
        "cli_model_id": llm_config.cli_model_id,
        "command": llm_config.resolved_cli_command if llm_config.is_cli_backend else "",
    }


def build_resolved_rollout_profile_snapshot(config: ApexConfig) -> dict[str, Any]:
    profiles: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    profile_count = len(config.rollout.llm_profiles)
    for profile_index in range(profile_count):
        planned_profile = dict(config.rollout.llm_profiles[profile_index])
        stages: dict[str, dict[str, Any]] = {}
        fallback_applied = False
        resolved_signature: list[str] = []

        for stage_name in _ROLLOUT_PROFILE_STAGE_NAMES:
            if stage_name == "rollout":
                requested = config.get_llm_for_rollout_profile(profile_index)
            else:
                requested = config.get_llm_for_profile_stage(profile_index, stage_name)
            resolved, routing = resolve_available_llm_config(
                requested,
                config.llm_configs,
                purpose=f"rollout_profile:{profile_index}:{stage_name}",
            )
            stage_fallback = bool(routing.get("fallback_applied"))
            fallback_applied = fallback_applied or stage_fallback
            resolved_identity = _llm_config_identity(config, resolved)
            resolved_signature.append(
                f"{stage_name}:{resolved_identity['backend']}:{resolved_identity['model']}"
            )
            stages[stage_name] = {
                "planned": _llm_config_identity(config, requested),
                "resolved": resolved_identity,
                "routing": dict(routing),
                "fallback_applied": stage_fallback,
                "requested_unavailable_reason": str(
                    routing.get("requested_unavailable_reason") or ""
                ),
            }
            if stage_fallback:
                warnings.append(
                    {
                        "profile_index": profile_index,
                        "stage": stage_name,
                        "requested": _llm_config_identity(config, requested),
                        "resolved": resolved_identity,
                        "reason": str(routing.get("requested_unavailable_reason") or ""),
                        "fallback_kind": str(routing.get("fallback_kind") or ""),
                    }
                )

        profiles.append(
            {
                "profile_index": profile_index,
                "planned_profile": planned_profile,
                "stages": stages,
                "fallback_applied": fallback_applied,
                "resolved_signature": resolved_signature,
            }
        )

    return {
        "profile_count": profile_count,
        "profiles": profiles,
        "fallback_count": len(warnings),
        "warnings": warnings,
    }


def _standalone_anchor_spec_matches_config(
    spec: dict[str, Any],
    llm_config: LLMConfig,
) -> bool:
    backend = str(spec.get("backend") or "").strip()
    if backend and backend != llm_config.backend.value:
        return False
    model = str(spec.get("model") or "").strip()
    if model and model != str(llm_config.model or "").strip():
        return False
    cli_model_id = str(spec.get("cli_model_id") or "").strip()
    if cli_model_id and cli_model_id != str(llm_config.cli_model_id or "").strip():
        return False
    return True


def build_execution_portfolio_audit(config: ApexConfig) -> dict[str, Any]:
    """Explain which model families the run is configured to exercise."""

    configured = [
        _llm_config_identity(config, llm_config) for llm_config in list(config.llm_configs or [])
    ]
    configured_backends = sorted(
        {str(item.get("backend") or "") for item in configured if str(item.get("backend") or "")}
    )
    standalone_specs = [
        dict(spec)
        for spec in list(getattr(config.rollout, "standalone_anchor_candidates", []) or [])
        if isinstance(spec, dict)
    ]
    standalone_targets: list[dict[str, Any]] = []
    missing_standalone_specs: list[dict[str, Any]] = []
    for spec in standalone_specs:
        matches = [
            _llm_config_identity(config, llm_config)
            for llm_config in list(config.llm_configs or [])
            if _standalone_anchor_spec_matches_config(spec, llm_config)
        ]
        payload = {
            "label": str(spec.get("label") or ""),
            "backend": str(spec.get("backend") or ""),
            "model": str(spec.get("model") or ""),
            "cli_model_id": str(spec.get("cli_model_id") or ""),
            "harness": str(spec.get("harness") or ""),
            "matched_llm_config_indices": [
                item.get("index") for item in matches if item.get("index") is not None
            ],
            "matched": bool(matches),
        }
        standalone_targets.append(payload)
        if not matches:
            missing_standalone_specs.append(payload)

    required_backend_families = sorted(
        {
            str(spec.get("backend") or "").strip()
            for spec in standalone_specs
            if str(spec.get("backend") or "").strip()
        }
    )
    missing_backend_families = sorted(
        set(required_backend_families).difference(configured_backends)
    )
    strict_anchor = bool(getattr(config.rollout, "standalone_anchor_strict_candidate_match", False))
    issues: list[dict[str, Any]] = []
    for backend in missing_backend_families:
        issues.append(
            {
                "severity": "high" if strict_anchor else "medium",
                "kind": "missing_backend_family",
                "backend": backend,
            }
        )
    for spec in missing_standalone_specs:
        issues.append(
            {
                "severity": "high" if strict_anchor else "medium",
                "kind": "missing_standalone_anchor_candidate",
                "candidate": spec,
            }
        )

    return {
        "status": (
            "ok"
            if not issues
            else (
                "error" if any(issue.get("severity") == "high" for issue in issues) else "warning"
            )
        ),
        "configured_backends": configured_backends,
        "configured_llm_count": len(configured),
        "configured_llms": configured,
        "standalone_anchor_enabled": bool(
            getattr(config.rollout, "enable_standalone_anchor", False)
        ),
        "standalone_anchor_run_all_candidates": bool(
            getattr(config.rollout, "standalone_anchor_run_all_candidates", False)
        ),
        "standalone_anchor_strict_candidate_match": strict_anchor,
        "standalone_anchor_targets": standalone_targets,
        "required_backend_families": required_backend_families,
        "missing_backend_families": missing_backend_families,
        "issues": issues,
    }


def _deepcopy_json_mapping(payload: Optional[dict[str, Any]]) -> dict[str, Any]:
    return copy.deepcopy(dict(payload or {}))


def build_benchmark_policy(
    *,
    benchmark_name: str,
    benchmark_family: Optional[str] = None,
    agent_input_contract: Optional[dict[str, Any]] = None,
    orchestrator_input_contract: Optional[dict[str, Any]] = None,
    evaluation_protocol: Optional[dict[str, Any]] = None,
    environment_policy: Optional[dict[str, Any]] = None,
    benchmark_specifics: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "schema_version": BENCHMARK_POLICY_SCHEMA_VERSION,
        "benchmark_name": benchmark_name,
        "benchmark_family": benchmark_family or benchmark_name,
        "agent_input_contract": _deepcopy_json_mapping(agent_input_contract),
        "orchestrator_input_contract": _deepcopy_json_mapping(orchestrator_input_contract),
        "evaluation_protocol": _deepcopy_json_mapping(evaluation_protocol),
        "environment_policy": _deepcopy_json_mapping(environment_policy),
        "benchmark_specifics": _deepcopy_json_mapping(benchmark_specifics),
    }


def build_run_manifest(
    *,
    config: ApexConfig,
    report_kind: str,
    harness_name: str,
    harness_version: str,
    benchmark_family: str,
    output_dir: str | Path,
    config_source: Optional[str] = None,
    requested_task_ids: Optional[list[str]] = None,
    execution: Optional[dict[str, Any]] = None,
    extra_settings: Optional[dict[str, Any]] = None,
    benchmark_policy: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    config_payload = config.to_dict()
    prompt_template_fingerprints = build_prompt_template_fingerprints()
    resolved_rollout_profiles = build_resolved_rollout_profile_snapshot(config)
    portfolio_audit = build_execution_portfolio_audit(config)
    return {
        "version": RUN_ARTIFACT_SCHEMA_VERSION,
        "report_kind": report_kind,
        "benchmark_family": benchmark_family,
        "harness_name": harness_name,
        "harness_version": harness_version,
        "output_dir": str(Path(output_dir).resolve()),
        "config_source": config_source,
        "config_hash": hash_config_payload(config_payload),
        "config_payload": config_payload,
        "python_version": sys.version.split()[0],
        "git": detect_git_snapshot(),
        "environment_snapshot": capture_environment_snapshot(config),
        "benchmark_policy": _deepcopy_json_mapping(benchmark_policy),
        "prompt_template_fingerprints": prompt_template_fingerprints,
        "execution": dict(execution or {}),
        "settings": {
            "num_rollouts": config.rollout.num_rollouts,
            "min_rollouts": config.rollout.min_rollouts,
            "max_rollouts": config.rollout.max_rollouts,
            "parallel_workers": config.rollout.parallel_workers,
            "task_parallelism": config.benchmark.task_parallelism,
            "planner_model": config.get_planner_llm().model,
            "planner_backend": config.get_planner_llm().backend.value,
            "selection_strategy": config.selection.strategy.value,
            "search_mode": config.search.mode.value,
            "rollout_profile_count": len(config.rollout.llm_profiles),
            "rollout_profiles": [dict(profile) for profile in config.rollout.llm_profiles],
            "resolved_rollout_profiles": resolved_rollout_profiles["profiles"],
            "rollout_profile_resolution_warnings": resolved_rollout_profiles["warnings"][:50],
            "execution_portfolio_audit": portfolio_audit,
            **(dict(extra_settings) if extra_settings else {}),
        },
        "rollout_profile_resolution": resolved_rollout_profiles,
        "execution_portfolio_audit": portfolio_audit,
        "model_config": config_payload.get("llm_configs", []),
        "backend_health_snapshot": build_allowed_backend_snapshots(
            config,
            refresh_health=False,
        ),
        "requested_task_ids": list(requested_task_ids or []),
        "completed_task_ids": [],
        "started_at": time.time(),
        "updated_at": time.time(),
        "finished_at": 0.0,
        "completed": False,
    }


def write_run_manifest(output_dir: str | Path, manifest: dict[str, Any]) -> Path:
    return atomic_write_json(Path(output_dir) / RUN_MANIFEST_FILENAME, manifest)


def reconcile_run_manifest(
    existing_manifest: Optional[dict[str, Any]],
    fresh_manifest: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(existing_manifest, dict):
        return copy.deepcopy(fresh_manifest)
    merged = copy.deepcopy(existing_manifest)
    merged.update(copy.deepcopy(fresh_manifest))
    for key in (
        "started_at",
        "updated_at",
        "finished_at",
        "completed",
        "completed_task_ids",
    ):
        if key in existing_manifest:
            merged[key] = copy.deepcopy(existing_manifest[key])
    return merged


def ensure_run_manifest(output_dir: str | Path, fresh_manifest: dict[str, Any]) -> dict[str, Any]:
    existing_manifest = load_run_manifest(output_dir)
    merged = reconcile_run_manifest(existing_manifest, fresh_manifest)
    if merged != existing_manifest:
        write_run_manifest(output_dir, merged)
    return merged


def update_run_manifest(
    output_dir: str | Path,
    *,
    completed_task_ids: Optional[list[str]] = None,
    requested_task_ids: Optional[list[str]] = None,
    completed: Optional[bool] = None,
    extra_updates: Optional[dict[str, Any]] = None,
) -> Optional[Path]:
    manifest_path = Path(output_dir) / RUN_MANIFEST_FILENAME
    payload = load_json_if_exists(manifest_path)
    if payload is None:
        return None
    payload["updated_at"] = time.time()
    if requested_task_ids is not None:
        payload["requested_task_ids"] = list(requested_task_ids)
    if completed_task_ids is not None:
        payload["completed_task_ids"] = list(completed_task_ids)
    if completed is not None:
        payload["completed"] = bool(completed)
        payload["finished_at"] = payload["updated_at"] if completed else 0.0
    if extra_updates:
        payload.update(dict(extra_updates))
    return write_run_manifest(output_dir, payload)


def load_run_manifest(run_dir: str | Path) -> Optional[dict[str, Any]]:
    return load_json_if_exists(Path(run_dir) / RUN_MANIFEST_FILENAME)


def manifest_summary(manifest: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        return {}
    execution = dict(manifest.get("execution") or {})
    backends = [
        {
            "backend": item.get("backend"),
            "model": item.get("model"),
            "healthy": item.get("healthy"),
            "version": ((item.get("version") or {}).get("version")),
        }
        for item in list(manifest.get("backend_health_snapshot") or [])
        if isinstance(item, dict)
    ]
    return {
        "config_hash": manifest.get("config_hash"),
        "git_sha": ((manifest.get("git") or {}).get("sha")),
        "git_branch": ((manifest.get("git") or {}).get("branch")),
        "prompt_template_hash": (
            (manifest.get("prompt_template_fingerprints") or {}).get("combined_hash")
        ),
        "execution_entrypoint": execution.get("entrypoint"),
        "execution_args": dict(execution.get("args") or {}),
        "backends": backends,
        "settings": dict(manifest.get("settings") or {}),
        "benchmark_policy": _deepcopy_json_mapping(manifest.get("benchmark_policy")),
    }


def write_task_live_state(task_output_dir: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(task_output_dir) / TASK_LIVE_STATE_FILENAME
    current = load_json_if_exists(target) or {}
    if not isinstance(current, dict):
        current = {}
    payload_copy = dict(payload)
    clear_keys = payload_copy.pop("_clear_keys", None)
    now = time.time()
    if (
        isinstance(current, dict)
        and current
        and _task_live_state_is_terminal(current)
        and not _task_live_state_is_terminal(payload_copy)
    ):
        preserved = dict(current)
        preserved["updated_at"] = now
        preserved["ignored_nonterminal_update_count"] = (
            int(preserved.get("ignored_nonterminal_update_count") or 0) + 1
        )
        preserved["last_ignored_nonterminal_update_at"] = now
        if payload_copy.get("phase") or payload_copy.get("status"):
            preserved["last_ignored_nonterminal_update"] = {
                "phase": payload_copy.get("phase"),
                "status": payload_copy.get("status"),
                "last_progress_at": payload_copy.get("last_progress_at"),
                "last_progress_source": payload_copy.get("last_progress_source"),
            }
        return atomic_write_json(target, preserved)
    merged = dict(current)
    if isinstance(clear_keys, (list, tuple, set)):
        for key in clear_keys:
            if isinstance(key, str):
                merged.pop(key, None)
    merged.update(payload_copy)
    if _task_live_state_is_terminal(merged):
        merged["terminal"] = True
        merged.setdefault("terminal_at", payload_copy.get("last_progress_at") or now)
    started_at = (
        merged.get("task_started_at")
        or merged.get("started_at")
        or current.get("task_started_at")
        or current.get("started_at")
        or merged.get("last_progress_at")
        or now
    )
    if isinstance(started_at, (int, float)):
        merged["task_started_at"] = float(started_at)
    merged["updated_at"] = now
    return atomic_write_json(target, merged)


def write_task_live_state_terminal(
    task_output_dir: str | Path,
    payload: dict[str, Any],
) -> Path:
    """Write terminal task state while clearing stale active rollout fields."""

    payload_copy = dict(payload)
    status = str(payload_copy.get("status") or "").strip().lower()
    phase = str(payload_copy.get("phase") or "").strip().lower()
    if status in {"completed", "failed", "error"}:
        current_phase = status
    elif phase in {"completed", "failed", "error"}:
        current_phase = phase
    else:
        current_phase = status or phase or "completed"
    existing_clear_keys = payload_copy.pop("_clear_keys", None)
    clear_keys = [
        "last_ignored_nonterminal_update",
        "last_ignored_nonterminal_update_at",
    ]
    if isinstance(existing_clear_keys, (list, tuple, set)):
        clear_keys.extend(key for key in existing_clear_keys if isinstance(key, str))
    payload_copy.update(
        {
            "active_rollout_ids": [],
            "active_rollout_count": 0,
            "current_rollout_id": None,
            "current_stage": None,
            "current_phase": current_phase,
            "terminal": True,
            "_clear_keys": clear_keys,
        }
    )
    return write_task_live_state(task_output_dir, payload_copy)


def write_rollout_live_state(
    task_output_dir: str | Path,
    rollout_id: int,
    payload: dict[str, Any],
) -> Path:
    directory = Path(task_output_dir) / ROLLOUT_STATUS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"rollout_{int(rollout_id)}.json"
    current = load_json_if_exists(target) or {}
    payload_copy = dict(payload)
    clear_keys = payload_copy.pop("_clear_keys", None)
    now = time.time()
    if not isinstance(current, dict):
        current = {}
    if (
        current
        and _rollout_live_state_is_terminal(current)
        and not _rollout_live_state_is_terminal(payload_copy)
    ):
        preserved = dict(current)
        preserved["updated_at"] = now
        preserved["ignored_nonterminal_update_count"] = (
            int(preserved.get("ignored_nonterminal_update_count") or 0) + 1
        )
        preserved["last_ignored_nonterminal_update_at"] = now
        if payload_copy.get("stage") or payload_copy.get("status"):
            preserved["last_ignored_nonterminal_update"] = {
                "stage": payload_copy.get("stage"),
                "status": payload_copy.get("status"),
                "last_progress_at": payload_copy.get("last_progress_at"),
                "last_progress_source": payload_copy.get("last_progress_source"),
            }
        return atomic_write_json(target, preserved)
    progress_retry_diagnostic = _retry_diagnostic_from_progress_payload(payload_copy)
    incoming_retry_diagnostics = _merge_retry_diagnostics(
        payload_copy.get("retry_diagnostics"),
        progress_retry_diagnostic,
    )
    if incoming_retry_diagnostics:
        persisted_retry_diagnostics = _persist_retry_diagnostic_artifacts(
            task_output_dir=task_output_dir,
            rollout_id=rollout_id,
            diagnostics=incoming_retry_diagnostics,
        )
        payload_copy["retry_diagnostics"] = _merge_retry_diagnostics(
            current.get("retry_diagnostics") if isinstance(current, dict) else None,
            persisted_retry_diagnostics,
        )
        payload_copy["retry_count"] = len(payload_copy["retry_diagnostics"])
    if (
        current
        and _rollout_live_state_is_terminal(current)
        and _rollout_live_state_is_scheduler_cancelled(current)
        and not _rollout_live_state_is_scheduler_cancelled(payload_copy)
    ):
        preserved = dict(current)
        preserved["updated_at"] = now
        preserved["ignored_post_cancel_update_count"] = (
            int(preserved.get("ignored_post_cancel_update_count") or 0) + 1
        )
        preserved["last_ignored_post_cancel_update_at"] = now
        if payload_copy.get("stage") or payload_copy.get("status"):
            preserved["last_ignored_post_cancel_update"] = {
                "stage": payload_copy.get("stage"),
                "status": payload_copy.get("status"),
                "terminal_state": payload_copy.get("terminal_state"),
                "last_progress_at": payload_copy.get("last_progress_at"),
                "last_progress_source": payload_copy.get("last_progress_source"),
            }
        return atomic_write_json(target, preserved)
    merged = dict(current)
    if isinstance(clear_keys, (list, tuple, set)):
        for key in clear_keys:
            if isinstance(key, str):
                merged.pop(key, None)
    merged.update(payload_copy)
    merged["rollout_id"] = int(rollout_id)
    merged["updated_at"] = now
    return atomic_write_json(target, merged)


def load_rollout_live_states(task_output_dir: str | Path) -> list[dict[str, Any]]:
    directory = Path(task_output_dir) / ROLLOUT_STATUS_DIRNAME
    if not directory.exists():
        return []
    payloads: list[dict[str, Any]] = []
    for path in sorted(directory.glob("rollout_*.json")):
        payload = load_json_if_exists(path)
        if isinstance(payload, dict):
            payloads.append(payload)
    payloads.sort(key=lambda item: int(item.get("rollout_id", 0)))
    return payloads


def list_task_directories(run_dir: str | Path) -> list[Path]:
    root = Path(run_dir)
    task_dirs: list[Path] = []
    for child in sorted(root.iterdir() if root.exists() else []):
        if not child.is_dir():
            continue
        if child.name in _RUN_RESERVED_DIRS:
            continue
        if child.name.startswith("."):
            continue
        task_dirs.append(child)
    return task_dirs


def _looks_like_collection_error(text: str) -> bool:
    normalized = text.lower()
    return (
        "collected 0 items" in normalized
        or "collection error" in normalized
        or "errors during collection" in normalized
        or "error collecting" in normalized
    )


def _looks_like_import_error(text: str) -> bool:
    normalized = text.lower()
    return (
        "modulenotfounderror" in normalized
        or "no module named" in normalized
        or "importerror" in normalized
        or "cannot import name" in normalized
    )


def _expected_test_coverage(task: dict[str, Any]) -> dict[str, Any]:
    final = dict(task.get("final") or {})
    coverage = dict(final.get("expected_test_coverage") or {})
    return coverage if coverage else {}


def _looks_like_coverage_collapse(task: dict[str, Any]) -> bool:
    coverage = _expected_test_coverage(task)
    missing = coverage.get("missing_expected_test_count")
    try:
        return int(missing or 0) > 0
    except (TypeError, ValueError):
        return False


def _looks_like_verification_failure(task: dict[str, Any]) -> bool:
    final_tests_passed = bool(task.get("final_tests_passed", False))
    if not final_tests_passed and (
        bool(task.get("success"))
        or bool(task.get("orchestrator_success"))
        or bool(task.get("candidate_found"))
    ):
        return True
    metadata = dict(task.get("execution_metadata") or {})
    if metadata.get("rollout_quick_verification_count"):
        if metadata.get("max_rollout_quick_test_pass_rate") == 0:
            return True
    reason = str(task.get("failure_reason") or "").lower()
    return "verification" in reason or "selector" in reason


def _infer_timeout_bucket(task: dict[str, Any]) -> Optional[str]:
    metadata = dict(task.get("execution_metadata") or {})
    trail = list(metadata.get("timeout_audit_trail") or [])
    terminal_states = [
        str(item.get("terminal_state") or "").strip() for item in trail if isinstance(item, dict)
    ]
    if any(state == "hard_timeout" for state in terminal_states):
        return "timeout_hard"
    if any(state == "stall_timeout" for state in terminal_states):
        return "timeout_stall"
    text = (
        str(task.get("failure_reason") or "")
        + "\n"
        + str((task.get("final") or {}).get("output") or "")
    ).lower()
    if "hard timeout" in text:
        return "timeout_hard"
    if "timed out" in text or "stalled after" in text or "timeout" in text:
        return "timeout_stall"
    return None


def classify_failure_root(task: dict[str, Any], benchmark_family: str) -> str:
    if bool(task.get("final_tests_passed", False)):
        return "solved"
    # Skipped tasks are upstream / environment failures (Linux-only
    # repo, baseline-pytest timeout, broken setup.py on declared
    # python version, etc.). The orchestrator never had an
    # opportunity to act, so they belong in their own bucket rather
    # than ``other_failure`` — which would otherwise lump them in
    # with genuine "no rollout produced a valid patch" outcomes and
    # mislead operators about the model's behaviour.
    if bool(task.get("skipped")):
        return f"skipped_{task.get('skip_category') or 'other'}"
    timeout_bucket = _infer_timeout_bucket(task)
    if timeout_bucket:
        return timeout_bucket

    text = (
        str(task.get("failure_reason") or "")
        + "\n"
        + str((task.get("final") or {}).get("output") or "")
    )
    if _looks_like_collection_error(text):
        return "collection_error"
    if _looks_like_import_error(text):
        return "import_error"
    if benchmark_family == "commit0" and _looks_like_coverage_collapse(task):
        return "coverage_collapse"
    if _looks_like_verification_failure(task):
        return "verification_failure"
    normalized = text.lower()
    if "coverage failure" in normalized or "fail-under" in normalized:
        return "evaluation_gate"
    if (
        "permission denied" in normalized
        or "docker" in normalized
        or "command not found" in normalized
    ):
        return "environment"
    return "other_failure"


def cluster_failures(tasks: list[dict[str, Any]], benchmark_family: str) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for task in tasks:
        if not isinstance(task, dict):
            continue
        bucket = classify_failure_root(task, benchmark_family)
        if bucket == "solved":
            continue
        task_id = str(
            task.get("instance_id") or task.get("repo") or task.get("task_name") or "unknown"
        )
        display_name = str(
            task.get("task_name") or task.get("repo") or task.get("instance_id") or task_id
        )
        payload = buckets.setdefault(
            bucket,
            {
                "bucket": bucket,
                "count": 0,
                "tasks": [],
                "sample_failure_reasons": Counter(),
            },
        )
        payload["count"] += 1
        payload["tasks"].append(display_name)
        failure_reason = str(task.get("failure_reason") or "").strip()
        if failure_reason:
            payload["sample_failure_reasons"][failure_reason[:180]] += 1

    clusters = []
    for bucket, payload in buckets.items():
        reasons = payload["sample_failure_reasons"]
        all_tasks = sorted(payload["tasks"])
        # Keep the truncated ``tasks`` list for the summary view (short
        # display in markdown / terminal), but ALSO emit the full list
        # under ``tasks_full`` so triage scripts and post-run analysis
        # can see every task in the bucket. The previous 8-task cap
        # silently hid 16+ tasks per cluster on Commit0 runs.
        clusters.append(
            {
                "bucket": bucket,
                "count": payload["count"],
                "tasks": all_tasks[:8],
                "tasks_full": all_tasks,
                "top_failure_reason": reasons.most_common(1)[0][0] if reasons else None,
            }
        )
    clusters.sort(key=lambda item: (-int(item["count"]), str(item["bucket"])))
    return clusters


def summarize_rollout_profiles(rollout_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for summary in rollout_summaries:
        if not isinstance(summary, dict):
            continue
        table.append(
            {
                "rollout_id": summary.get("rollout_id"),
                "profile_index": summary.get("rollout_profile_index"),
                "agent_mode": summary.get("agent_mode"),
                "prompt_strategy": summary.get("prompt_strategy"),
                "temperature": summary.get("temperature"),
                "llm_model": summary.get("llm_model"),
                "profile_signature": list(summary.get("rollout_profile_signature") or []),
                "stage_model_routing": dict(summary.get("stage_model_routing") or {}),
                "success": bool(summary.get("success", False)),
            }
        )
    table.sort(key=lambda item: int(item.get("rollout_id") or 0))
    return table


def _process_backend_name(command: str) -> Optional[str]:
    normalized = str(command or "").lower()
    if "codex exec" in normalized:
        return "codex_cli"
    if "claude -p" in normalized or normalized.startswith("claude "):
        return "claude_cli"
    if "gemini -p" in normalized or normalized.startswith("gemini "):
        return "gemini_cli"
    if "opencode run" in normalized or normalized.startswith("opencode "):
        return "opencode_cli"
    if "metacode run" in normalized or normalized.startswith("metacode "):
        return "metacode_cli"
    if "pytest" in normalized:
        return "pytest"
    return None


def _collect_process_snapshot() -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss=,%cpu=,command="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {}, []

    entries: dict[int, dict[str, Any]] = {}
    children_by_parent: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) != 5:
            continue
        pid_text, ppid_text, rss_text, cpu_text, command = parts
        try:
            pid = int(pid_text)
            ppid = int(ppid_text)
            rss_kb = int(rss_text)
            cpu_percent = float(cpu_text)
        except ValueError:
            continue
        entries[pid] = {
            "pid": pid,
            "ppid": ppid,
            "rss_kb": rss_kb,
            "cpu_percent": cpu_percent,
            "command": command.strip(),
        }
        children_by_parent.setdefault(ppid, []).append(pid)

    def descendant_count(root_pid: int) -> int:
        count = 0
        seen: set[int] = set()
        stack = list(children_by_parent.get(root_pid, []))
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            seen.add(pid)
            count += 1
            stack.extend(children_by_parent.get(pid, []))
        return count

    for pid, entry in entries.items():
        entry["child_process_count"] = descendant_count(pid)

    backend_totals: dict[str, dict[str, Any]] = {}
    for entry in entries.values():
        backend = _process_backend_name(entry["command"])
        if not backend:
            continue
        bucket = backend_totals.setdefault(
            backend,
            {
                "backend": backend,
                "process_count": 0,
                "total_rss_kb": 0,
                "total_cpu_percent": 0.0,
            },
        )
        bucket["process_count"] += 1
        bucket["total_rss_kb"] += int(entry["rss_kb"])
        bucket["total_cpu_percent"] += float(entry["cpu_percent"])

    totals = []
    for payload in backend_totals.values():
        totals.append(
            {
                **payload,
                "total_rss_mb": round(float(payload["total_rss_kb"]) / 1024.0, 1),
                "total_cpu_percent": round(float(payload["total_cpu_percent"]), 1),
            }
        )
    totals.sort(key=lambda item: (-int(item["process_count"]), str(item["backend"])))
    return entries, totals


def _resource_telemetry_for_pid(
    pid: Optional[int],
    process_snapshot: dict[int, dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if not isinstance(pid, int) or pid <= 0:
        return None
    entry = process_snapshot.get(pid)
    if entry is None:
        return {
            "pid": pid,
            "running": False,
        }
    return {
        "pid": pid,
        "running": True,
        "rss_kb": int(entry["rss_kb"]),
        "rss_mb": round(float(entry["rss_kb"]) / 1024.0, 1),
        "cpu_percent": round(float(entry["cpu_percent"]), 1),
        "child_process_count": int(entry["child_process_count"]),
        "command": entry["command"],
    }


def _task_completed_successfully(result_payload: dict[str, Any]) -> bool:
    if not result_payload:
        return False
    if "final_tests_passed" in result_payload:
        return bool(result_payload.get("final_tests_passed"))
    return bool(result_payload.get("success"))


def join_harness_results_into_records(
    records_dir: str | Path,
    official_full_json: str | Path,
    *,
    benchmark_adapter: Any = None,
    eval_logs_dir: str | Path | None = None,
    eval_log_suffix: str = ".full.eval.log",
) -> JoinReport:
    """Join official TestGenEval per-task output into Apex task records."""

    root = Path(records_dir)
    official_path = Path(official_full_json)
    official_payload = load_json_if_exists(official_path) or {}
    official_by_id = _official_full_rows_by_instance(official_payload)
    log_root = (
        Path(eval_logs_dir) if eval_logs_dir is not None else (root.parent / "official_eval_logs")
    )
    log_completeness_by_id: dict[str, dict[str, Any]] = (
        _scan_eval_logs_for_completeness(log_root, suffix=eval_log_suffix)
        if log_root.exists()
        else {}
    )
    record_paths = [
        path
        for path in sorted(root.glob("*.json"))
        if path.name not in {RUN_MANIFEST_FILENAME, "runner_status.json"}
    ]
    updated = 0
    missing = 0
    seen_ids: set[str] = set()
    for path in record_paths:
        record = load_json_if_exists(path) or {}
        if not isinstance(record, dict):
            continue
        candidate_keys = [
            str(record.get("id") or ""),
            str(record.get("instance_id") or ""),
            str(record.get("task_id") or ""),
            path.stem,
        ]
        task_id = ""
        row = None
        for key in candidate_keys:
            if key and key in official_by_id:
                task_id = key
                row = official_by_id[key]
                break
        if not task_id:
            task_id = next((key for key in candidate_keys if key), path.stem)
        seen_ids.add(task_id)
        if row is None:
            record["success"] = False
            record["pass_at_1"] = 0.0
            record["all_pass_at_1"] = 0.0
            _clear_stale_official_fields(record)
            record.setdefault("diagnostics", {}).setdefault("harness_join", {})[
                "missing_harness_data"
            ] = True
            missing += 1
        else:
            record.update(_record_fields_from_official_row(row))
            log_extras = log_completeness_by_id.get(task_id) or {}
            for key, value in log_extras.items():
                if value is None:
                    continue
                record[key] = value
            record.setdefault("diagnostics", {}).setdefault("harness_join", {})[
                "official_full_json"
            ] = str(official_path)
            if log_extras:
                record["diagnostics"]["harness_join"]["eval_log_metrics"] = dict(log_extras)
            updated += 1
        atomic_write_json(path, record)
    return JoinReport(
        records_seen=len(record_paths),
        records_updated=updated,
        missing_harness_data=missing,
        harness_rows_without_records=len(set(official_by_id) - seen_ids),
    )


def records_from_official_full_json(official_full_json: str | Path) -> list[dict[str, Any]]:
    payload = load_json_if_exists(official_full_json) or {}
    return [
        {"instance_id": task_id, **_record_fields_from_official_row(row)}
        for task_id, row in sorted(_official_full_rows_by_instance(payload).items())
    ]


def _official_full_rows_by_instance(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for task_id, raw in dict(payload or {}).items():
        if isinstance(raw, dict):
            rows[str(task_id)] = dict(raw)
    return rows


_OFFICIAL_JOIN_FIELDS = {
    "coverage",
    "coverage_ratio",
    "mutation_score",
    "mutation_num",
    "mutation_uncertainty",
    "mutation_completeness",
    "test_error",
    "official_filtered_passed",
    "official_unfiltered_passed",
    "official_unfiltered_pass_at_1",
    "official_filtered_pass_at_1",
    "mutation_jobs_total",
    "mutation_jobs_complete",
    "mutation_timeout",
}


def _clear_stale_official_fields(record: dict[str, Any]) -> None:
    for key in _OFFICIAL_JOIN_FIELDS:
        record.pop(key, None)


_EVAL_LOG_TOTAL_JOBS_RE = re.compile(r"total jobs:\s*(\d+)")
_EVAL_LOG_COMPLETE_RE = re.compile(r"complete:\s*(\d+)\s*\(([\d.]+)%\)")
_EVAL_LOG_MUTATION_TIMEOUT_RE = re.compile(r"\bMutationTimeout\b")


def _scan_eval_logs_for_completeness(
    log_root: Path,
    *,
    suffix: str = ".full.eval.log",
) -> dict[str, dict[str, Any]]:
    """Parse mutation completeness markers from official harness eval logs."""

    results: dict[str, dict[str, Any]] = {}
    for log_path in log_root.glob(f"*{suffix}"):
        name = log_path.name
        if not name.endswith(suffix):
            continue
        stem = name[: -len(suffix)]
        # The harness produces "<task_id>.<model_name>". Task ids never contain
        # ".", but model names commonly do, so split on the first dot.
        task_id = stem.split(".", 1)[0] if "." in stem else stem
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        try:
            from apex.evaluation.runners.testgenevallite_aggregate import (
                parse_official_eval_log,
            )

            parsed = parse_official_eval_log(text)
        except Exception:
            parsed = None
        total_match = _EVAL_LOG_TOTAL_JOBS_RE.search(text)
        complete_match = _EVAL_LOG_COMPLETE_RE.search(text)
        if not (total_match or complete_match or parsed):
            continue
        total_jobs = int(total_match.group(1)) if total_match else None
        complete = int(complete_match.group(1)) if complete_match else None
        completeness = None
        if total_jobs and complete is not None and total_jobs > 0:
            completeness = min(1.0, max(0.0, complete / total_jobs))
        timed_out = bool(_EVAL_LOG_MUTATION_TIMEOUT_RE.search(text))
        results[task_id] = {
            "mutation_jobs_total": total_jobs,
            "mutation_jobs_complete": complete,
            "mutation_completeness": completeness,
            "mutation_timeout": timed_out,
        }
        if parsed is not None:
            results[task_id].update(
                {
                    "official_eval_log_status": parsed.status,
                    "official_eval_log_has_mutation": parsed.has_mutation_log,
                    "official_eval_log_has_coverage": parsed.has_coverage_log,
                }
            )
    return results


def _record_fields_from_official_row(row: dict[str, Any]) -> dict[str, Any]:
    full = dict((row or {}).get("full") or row or {})
    unfiltered = bool(_first(full.get("unfiltered_tests_passed"), False))
    filtered = bool(_first(full.get("tests_passed"), False))
    mutation_num = _float_or_none(_first(full.get("mutation_num"), None))
    mutation_complete = _float_or_none(_first(full.get("mutation_jobs_complete"), None))
    completeness = None
    if mutation_num and mutation_complete is not None:
        completeness = min(1.0, max(0.0, mutation_complete / mutation_num))
    return {
        "success": unfiltered,
        "pass_at_1": 1.0 if unfiltered else 0.0,
        "all_pass_at_1": 1.0 if filtered else 0.0,
        "official_unfiltered_pass_at_1": 1.0 if unfiltered else 0.0,
        "official_filtered_pass_at_1": 1.0 if filtered else 0.0,
        "coverage": _float_or_zero(_first(full.get("coverage"), 0.0)),
        "coverage_ratio": _float_or_zero(_first(full.get("coverage"), 0.0)) / 100.0,
        "mutation_score": _float_or_zero(_first(full.get("mutation_score"), 0.0)),
        "mutation_num": int(_float_or_zero(_first(full.get("mutation_num"), 0))),
        "mutation_uncertainty": _float_or_zero(_first(full.get("mutation_uncertainty"), 0.0)),
        "mutation_completeness": completeness,
        "test_error": str(_first(full.get("test_error"), "")),
        "official_filtered_passed": filtered,
        "official_unfiltered_passed": unfiltered,
    }


def _first(value: Any, default: Any = None) -> Any:
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


def _float_or_zero(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _task_timeout_recovery_count(result_payload: dict[str, Any]) -> int:
    metadata = dict(result_payload.get("execution_metadata") or {})
    trail = list(metadata.get("timeout_audit_trail") or [])
    return sum(
        1
        for item in trail
        if isinstance(item, dict)
        and (
            bool(item.get("recovered"))
            or str(item.get("terminal_state") or "").strip() == "recovered_after_timeout"
        )
    )


def _classify_task_health(
    *,
    completed: bool,
    completed_successfully: bool,
    status_bucket: str,
    live_state: dict[str, Any],
    rollout_states: list[dict[str, Any]],
    result_payload: dict[str, Any],
) -> tuple[str, list[str]]:
    if completed and completed_successfully:
        return "healthy", ["completed_success"]
    if completed:
        return "failed", ["completed_failure"]
    if status_bucket == "active":
        return "running", ["active_worker_present"]
    if status_bucket == "stalled":
        return "suspicious", ["stalled_progress"]
    if status_bucket in {"error", "timeout"}:
        return "suspicious", [status_bucket]
    if live_state or rollout_states or result_payload:
        return "suspicious", ["incomplete_artifacts"]
    return "pending", ["not_started"]


def _status_bucket_from_live_state(payload: dict[str, Any], *, now: Optional[float] = None) -> str:
    current_time = now or time.time()
    status = str(payload.get("status") or "unknown").strip() or "unknown"
    if status in {"completed", "error", "timeout"}:
        return status
    last_progress_at = payload.get("last_progress_at") or payload.get("updated_at")
    progress_timeout_seconds = payload.get("progress_timeout_seconds")
    if isinstance(last_progress_at, (int, float)) and isinstance(
        progress_timeout_seconds, (int, float)
    ):
        if current_time - float(last_progress_at) > max(30.0, float(progress_timeout_seconds)):
            return "stalled"
    return "active"


def inspect_run_directory(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    manifest = load_run_manifest(root) or {}
    run_state = load_json_if_exists(root / "benchmark_state.json") or {}
    report = load_json_if_exists(root / "benchmark_report.json") or {}
    benchmark_family = str(manifest.get("benchmark_family") or "") or (
        "commit0"
        if "commit0" in str(report.get("report_kind") or "").lower()
        else "swebench_pro"
        if "swebench" in str(report.get("report_kind") or "").lower()
        else "local"
    )
    requested_ids = list(
        run_state.get("requested_task_ids") or manifest.get("requested_task_ids") or []
    )
    report_tasks = [task for task in list(report.get("tasks") or []) if isinstance(task, dict)]
    task_dirs = list_task_directories(root)
    task_dir_map = {path.name: path for path in task_dirs}

    def _task_key(task: dict[str, Any]) -> str:
        for key in ("task_name", "instance_id", "repo"):
            value = str(task.get(key) or "").strip()
            if value and value in task_dir_map:
                return value
        for key in ("task_name", "instance_id", "repo"):
            value = str(task.get(key) or "").strip()
            if value:
                return value
        return "unknown"

    completed_map = {_task_key(task): task for task in report_tasks}
    tasks: list[dict[str, Any]] = []
    process_snapshot, backend_process_totals = _collect_process_snapshot()
    candidate_ids = list(
        dict.fromkeys(
            list(task_dir_map.keys())
            + list(completed_map.keys())
            + ([] if task_dir_map else requested_ids)
        )
    )
    now = time.time()
    for task_id in candidate_ids:
        task_dir = task_dir_map.get(task_id, root / task_id)
        live_state = load_json_if_exists(task_dir / TASK_LIVE_STATE_FILENAME) or {}
        rollout_states = load_rollout_live_states(task_dir)
        annotated_rollout_states: list[dict[str, Any]] = []
        for rollout_state in rollout_states:
            annotated = dict(rollout_state)
            annotated["resource_telemetry"] = _resource_telemetry_for_pid(
                annotated.get("process_pid"),
                process_snapshot,
            )
            annotated_rollout_states.append(annotated)
        result_payload = (
            load_json_if_exists(task_dir / "task_result.json") or completed_map.get(task_id) or {}
        )
        apex_result = load_json_if_exists(task_dir / "apex_result.json") or {}
        current_rollout = next(
            (
                item
                for item in sorted(
                    annotated_rollout_states,
                    key=lambda payload: float(payload.get("updated_at") or 0.0),
                    reverse=True,
                )
                if _status_bucket_from_live_state(item, now=now) in {"active", "stalled"}
            ),
            None,
        )
        live_state_status_bucket = (
            _status_bucket_from_live_state(live_state, now=now) if live_state else "pending"
        )
        task_completed = bool(result_payload)
        if task_completed and live_state_status_bucket not in {"error", "timeout"}:
            live_state_status_bucket = "completed"
        display_rollout = None if task_completed else current_rollout
        current_stage = (
            (display_rollout or {}).get("stage")
            or live_state.get("current_stage")
            or (
                (((apex_result.get("rollout_summaries") or [{}])[-1]).get("trajectory") or [{}])[
                    -1
                ].get("stage")
                if apex_result.get("rollout_summaries")
                else None
            )
        )
        current_model = (
            (display_rollout or {}).get("model")
            or live_state.get("model")
            or (
                ((apex_result.get("rollout_summaries") or [{}])[-1]).get("llm_model")
                if apex_result.get("rollout_summaries")
                else None
            )
        )
        timeout_budget_remaining_seconds = None
        if isinstance(
            (display_rollout or {}).get("hard_timeout_seconds"), (int, float)
        ) and isinstance(
            (display_rollout or {}).get("stage_started_at"),
            (int, float),
        ):
            timeout_budget_remaining_seconds = max(
                0.0,
                float((display_rollout or {}).get("hard_timeout_seconds"))
                - (now - float((display_rollout or {}).get("stage_started_at"))),
            )
        completed = task_completed
        completed_successfully = _task_completed_successfully(result_payload)
        if task_completed:
            status_bucket = live_state_status_bucket
        else:
            status_bucket = (
                _status_bucket_from_live_state(display_rollout or live_state, now=now)
                if (display_rollout or live_state)
                else "pending"
            )
        task_health, health_reasons = _classify_task_health(
            completed=completed,
            completed_successfully=completed_successfully,
            status_bucket=status_bucket,
            live_state=live_state,
            rollout_states=annotated_rollout_states,
            result_payload=result_payload,
        )
        resource_telemetry = (
            _resource_telemetry_for_pid(
                (display_rollout or live_state).get("process_pid")
                if isinstance((display_rollout or live_state), dict)
                else None,
                process_snapshot,
            )
            if status_bucket in {"active", "stalled"}
            else None
        )
        failure_root = None
        if completed and not completed_successfully:
            failure_root = classify_failure_root(result_payload, benchmark_family)
        timeout_recovery_count = _task_timeout_recovery_count(result_payload)
        tasks.append(
            {
                "task_id": task_id,
                "path": str(task_dir),
                "live_state": live_state,
                "completed": completed,
                "phase": (
                    "completed" if result_payload else str(live_state.get("phase") or "pending")
                ),
                "status": status_bucket,
                "health": task_health,
                "health_reasons": health_reasons,
                "failure_root": failure_root,
                "current_stage": current_stage,
                "model": current_model,
                "last_progress_at": (
                    (live_state.get("last_progress_at") or live_state.get("updated_at"))
                    if task_completed
                    else (
                        (display_rollout or {}).get("last_progress_at")
                        or live_state.get("last_progress_at")
                        or live_state.get("updated_at")
                    )
                ),
                "timeout_budget_remaining_seconds": timeout_budget_remaining_seconds,
                "rollouts": annotated_rollout_states,
                "result": result_payload,
                "resource_telemetry": resource_telemetry,
                "timeout_recovery_count": timeout_recovery_count,
            }
        )

    summary_counter = Counter(str(task.get("health") or "pending") for task in tasks)
    recent_failures = sorted(
        [
            {
                "task_id": task.get("task_id"),
                "failure_root": task.get("failure_root"),
                "failure_reason": str((task.get("result") or {}).get("failure_reason") or "")[:220],
                "last_progress_at": task.get("last_progress_at"),
            }
            for task in tasks
            if task.get("health") == "failed"
        ],
        key=lambda item: float(item.get("last_progress_at") or 0.0),
        reverse=True,
    )[:8]
    timeout_recoveries = sorted(
        [
            {
                "task_id": task.get("task_id"),
                "timeout_recovery_count": task.get("timeout_recovery_count"),
                "last_progress_at": task.get("last_progress_at"),
            }
            for task in tasks
            if int(task.get("timeout_recovery_count") or 0) > 0
        ],
        key=lambda item: (
            int(item.get("timeout_recovery_count") or 0),
            float(item.get("last_progress_at") or 0.0),
        ),
        reverse=True,
    )[:8]
    failure_clusters = list(report.get("failure_clusters") or [])
    if not failure_clusters:
        failure_clusters = cluster_failures(
            [task.get("result") or {} for task in tasks if task.get("completed")],
            benchmark_family=benchmark_family,
        )
    return {
        "run_dir": str(root),
        "manifest": manifest,
        "manifest_summary": manifest_summary(manifest),
        "run_state": run_state,
        "report": report,
        "benchmark_family": benchmark_family,
        "tasks": tasks,
        "summary": {
            "total": len(tasks),
            "healthy": int(summary_counter.get("healthy", 0)),
            "failed": int(summary_counter.get("failed", 0)),
            "running": int(summary_counter.get("running", 0)),
            "suspicious": int(summary_counter.get("suspicious", 0)),
            "pending": int(summary_counter.get("pending", 0)),
            "completed": sum(1 for task in tasks if bool(task.get("completed"))),
        },
        "backend_process_totals": backend_process_totals,
        "recent_failures": recent_failures,
        "timeout_recoveries": timeout_recoveries,
        "failure_clusters": failure_clusters,
    }


def compare_run_directories(left_run: str | Path, right_run: str | Path) -> dict[str, Any]:
    left = inspect_run_directory(left_run)
    right = inspect_run_directory(right_run)
    left_report = left.get("report") or {}
    right_report = right.get("report") or {}
    return {
        "left_run": left["run_dir"],
        "right_run": right["run_dir"],
        "left_manifest": left["manifest_summary"],
        "right_manifest": right["manifest_summary"],
        "benchmark_family": left.get("benchmark_family") or right.get("benchmark_family"),
        "score_delta_percent": (
            float(
                right_report.get("score_percent")
                or right_report.get("average_pass_rate_percent")
                or 0.0
            )
            - float(
                left_report.get("score_percent")
                or left_report.get("average_pass_rate_percent")
                or 0.0
            )
        ),
        "solve_delta_percent": (
            float(right_report.get("solved_rate_percent") or 0.0)
            - float(left_report.get("solved_rate_percent") or 0.0)
        ),
        "config_hash_changed": (
            (left["manifest_summary"].get("config_hash") or None)
            != (right["manifest_summary"].get("config_hash") or None)
        ),
        "git_sha_changed": (
            (left["manifest_summary"].get("git_sha") or None)
            != (right["manifest_summary"].get("git_sha") or None)
        ),
        "prompt_template_hash_changed": (
            (left["manifest_summary"].get("prompt_template_hash") or None)
            != (right["manifest_summary"].get("prompt_template_hash") or None)
        ),
    }


def run_structured_backend_smoke_test(
    config: LLMConfig,
    *,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ping": {"type": "string"},
            "command": {"type": "string"},
        },
        "required": ["ping", "command"],
    }
    temp_dir = Path(tempfile.mkdtemp(prefix="apex-doctor-"))
    try:
        client = CLIModelClient(config)
        result = client.run_structured_prompt(
            prompt=(
                "Return a JSON object with ping='pong' and command set to the backend "
                "name. Do not inspect the repository and do not call tools unless your "
                "runtime requires it to answer."
            ),
            working_dir=str(temp_dir),
            schema=schema,
            allow_edits=False,
            hard_timeout_seconds=timeout_seconds,
        )
        parsed = result.parsed_json if isinstance(result.parsed_json, dict) else {}
        return {
            "success": bool(result.success and parsed.get("ping") == "pong"),
            "duration_seconds": result.duration_seconds,
            "error": result.error,
            "parsed_json": parsed,
        }
    except Exception as exc:
        return {
            "success": False,
            "duration_seconds": 0.0,
            "error": str(exc),
            "parsed_json": None,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_tool_backend_smoke_test(
    config: LLMConfig,
    *,
    timeout_seconds: int = 90,
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "token_file": {"type": "string"},
            "observed_token": {"type": "string"},
            "working_dir_basename": {"type": "string"},
        },
        "required": ["token_file", "observed_token", "working_dir_basename"],
    }
    temp_dir = Path(tempfile.mkdtemp(prefix="apex-doctor-tool-"))
    token_file = f"doctor_token_{int(time.time() * 1_000_000)}.txt"
    token_value = hashlib.sha256(token_file.encode("utf-8")).hexdigest()[:24]
    try:
        (temp_dir / token_file).write_text(token_value)
        (temp_dir / "README.txt").write_text("tool smoke\n")
        client = CLIModelClient(config)
        result = client.run_structured_prompt(
            prompt=(
                "Use available read-only tools to inspect the current working directory. "
                f"Read the exact contents of the file named {token_file} and return a JSON "
                "object matching the schema. Set token_file to that filename, "
                "observed_token to the exact file contents, and working_dir_basename "
                "to the basename of the current working directory. Do not modify files."
            ),
            working_dir=str(temp_dir),
            schema=schema,
            allow_edits=False,
            hard_timeout_seconds=timeout_seconds,
        )
        parsed = result.parsed_json if isinstance(result.parsed_json, dict) else {}
        success = bool(
            result.success
            and parsed.get("token_file") == token_file
            and parsed.get("observed_token") == token_value
            and parsed.get("working_dir_basename") == temp_dir.name
        )
        return {
            "success": success,
            "duration_seconds": result.duration_seconds,
            "error": result.error,
            "parsed_json": parsed,
        }
    except Exception as exc:
        return {
            "success": False,
            "duration_seconds": 0.0,
            "error": str(exc),
            "parsed_json": None,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _build_lookup_only_backend_snapshots(config: ApexConfig) -> list[dict[str, Any]]:
    """PATH-only backend snapshots used when --skip-cli-smoke-tests is set.

    Skips ``probe_cli_backend_health`` (which spawns the CLI binary)
    so that an interactively-wrapped launcher (e.g. ``claude
    --teammate-mode tmux``) cannot hang the doctor. ``healthy`` is set
    to ``True`` purely on the basis of ``shutil.which(command) is not
    None`` — operators who pass the flag have already accepted that
    they're trading startup verification for liveness.
    """
    configured = _configured_backend_map(config)
    snapshots: list[dict[str, Any]] = []
    for backend in _ALLOWED_CLI_BACKENDS:
        llm_config = configured.get(backend, LLMConfig(backend=backend))
        command = llm_config.resolved_cli_command
        on_path = bool(command) and shutil.which(command) is not None
        snapshots.append(
            {
                "backend": backend.value,
                "model": llm_config.model,
                "command": command,
                "healthy": on_path,
                "unavailable_reason": (
                    "" if on_path else f"CLI backend '{command}' is not installed."
                ),
                "version": {
                    "command": command,
                    "resolved_path": shutil.which(command) if command else None,
                    "version": None,
                    "exit_code": None,
                    "error": "skipped: --skip-cli-smoke-tests",
                },
                "auth": _auth_snapshot_for_backend(llm_config),
                "configured": backend in configured,
            }
        )
    return snapshots


def doctor_summary(
    config: ApexConfig,
    *,
    config_source: Optional[str] = None,
    run_smoke_tests: bool = True,
    run_cli_health_probes: bool = True,
) -> dict[str, Any]:
    """Run preflight checks for the doctor command.

    Parameters
    ----------
    run_smoke_tests:
        When True (default), runs the structured-output / tool-call
        smoke probes against each healthy backend. When False the
        smoke results are an empty list.
    run_cli_health_probes:
        When True (default), each CLI backend is started with a
        ``--help`` / ``--version`` probe to verify it can launch. When
        False the doctor only verifies the launcher is on PATH via
        ``shutil.which`` and skips running the binary at all. Useful
        when one of the installed CLIs is wrapped in an interactive
        alias that hangs the probe (e.g. claude wrapped with
        ``--teammate-mode tmux``). With this flag set the doctor will
        also implicitly skip the structured-output smoke tests since
        those would invoke the same hanging wrapper.
    """
    if not run_cli_health_probes:
        backend_snapshots = _build_lookup_only_backend_snapshots(config)
        # The structured-output / tool-call smoke probes invoke the
        # same CLI binary; if the operator asked us to skip CLI startup
        # probes for hang-avoidance reasons, also skip these.
        run_smoke_tests = False
    else:
        backend_snapshots = build_allowed_backend_snapshots(
            config,
            refresh_health=True,
            include_unconfigured=True,
        )
    smoke_results: list[dict[str, Any]] = []
    tool_smoke_results: list[dict[str, Any]] = []
    if run_smoke_tests:
        for snapshot in backend_snapshots:
            llm_config = next(
                (item for item in config.llm_configs if item.backend.value == snapshot["backend"]),
                LLMConfig(backend=LLMBackend(snapshot["backend"])),
            )
            if snapshot.get("healthy"):
                smoke = run_structured_backend_smoke_test(llm_config)
                tool_smoke = run_tool_backend_smoke_test(llm_config)
            else:
                smoke = {
                    "success": False,
                    "duration_seconds": 0.0,
                    "error": snapshot.get("unavailable_reason"),
                    "parsed_json": None,
                }
                tool_smoke = {
                    "success": False,
                    "duration_seconds": 0.0,
                    "error": snapshot.get("unavailable_reason"),
                    "parsed_json": None,
                }
            smoke_results.append(
                {
                    "backend": snapshot["backend"],
                    "model": snapshot["model"],
                    **smoke,
                }
            )
            tool_smoke_results.append(
                {
                    "backend": snapshot["backend"],
                    "model": snapshot["model"],
                    **tool_smoke,
                }
            )

    parity_checks: list[dict[str, Any]] = []
    python_path = Path(sys.executable)
    for name, probe, required, note in (
        (
            "pytest_jsonreport_plugin",
            f'{python_path} -c "import pytest_jsonreport; import pytest_jsonreport.plugin"',
            False,
            "Optional host check only. Commit0 local pytest runs install this inside task environments when needed.",
        ),
        (
            "pkg_resources_legacy_api",
            f"{python_path} -c \"import pkg_resources; dist = pkg_resources.get_distribution('setuptools'); assert dist.version\"",
            False,
            "Optional host check only. Benchmark runners validate legacy setuptools compatibility inside task environments.",
        ),
    ):
        result = subprocess.run(
            probe,
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )
        parity_checks.append(
            {
                "name": name,
                "success": result.returncode == 0,
                "required": required,
                "note": note,
                "error": (result.stdout + result.stderr).strip() if result.returncode != 0 else "",
            }
        )

    command_checks = []
    for command in ("git", "python3", "uv", "docker"):
        command_checks.append(
            {
                "command": command,
                "available": shutil.which(command) is not None,
            }
        )

    # Permissive backend gate: APEX dispatches to whichever CLI agent is
    # available, so the doctor only needs to confirm AT LEAST ONE CLI
    # backend is healthy. Requiring all four would fail every host that
    # only has, say, codex installed — which is the documented common
    # case. The per-backend report still surfaces every CLI's status so
    # operators can see which are working / unavailable / unhealthy.
    healthy_backend_count = sum(1 for item in backend_snapshots if bool(item.get("healthy")))
    backends_overall_ok = healthy_backend_count >= 1 if backend_snapshots else False
    # Smoke tests follow the same "≥1 success" rule when run, restricted
    # to the backends we actually probed (a missing CLI's smoke is
    # synthesised as a failure record above, but its absence shouldn't
    # tank the whole doctor result if a sibling CLI works).
    if run_smoke_tests and smoke_results:
        smoke_overall_ok = any(bool(item.get("success")) for item in smoke_results)
    else:
        smoke_overall_ok = True
    if run_smoke_tests and tool_smoke_results:
        tool_smoke_overall_ok = any(bool(item.get("success")) for item in tool_smoke_results)
    else:
        tool_smoke_overall_ok = True

    overall_success = (
        backends_overall_ok
        and all(bool(item.get("success")) for item in parity_checks if item.get("required", True))
        and all(
            bool(item.get("available")) for item in command_checks if item["command"] != "docker"
        )
        and smoke_overall_ok
        and tool_smoke_overall_ok
    )
    return {
        "success": overall_success,
        "config_source": config_source,
        "config_hash": hash_config(config),
        "git": detect_git_snapshot(),
        "backend_health": backend_snapshots,
        "healthy_backend_count": healthy_backend_count,
        "backend_smoke_tests": smoke_results,
        "backend_tool_smoke_tests": tool_smoke_results,
        "benchmark_env_parity": parity_checks,
        "command_checks": command_checks,
    }


def render_status_table(status: dict[str, Any]) -> str:
    lines = [
        f"Run: {status['run_dir']}",
        f"Benchmark family: {status.get('benchmark_family') or 'unknown'}",
    ]
    manifest = status.get("manifest_summary") or {}
    if manifest.get("git_sha"):
        lines.append(f"Git SHA: {manifest['git_sha']}")
    if manifest.get("config_hash"):
        lines.append(f"Config hash: {manifest['config_hash']}")
    if manifest.get("prompt_template_hash"):
        lines.append(f"Prompt template hash: {manifest['prompt_template_hash']}")
    summary = status.get("summary") or {}
    lines.append(
        "Task health: healthy={healthy} failed={failed} running={running} suspicious={suspicious} pending={pending}".format(
            healthy=summary.get("healthy", 0),
            failed=summary.get("failed", 0),
            running=summary.get("running", 0),
            suspicious=summary.get("suspicious", 0),
            pending=summary.get("pending", 0),
        )
    )
    backend_totals = status.get("backend_process_totals") or []
    if backend_totals:
        lines.append(
            "Backend processes: "
            + "; ".join(
                "{backend}={count} proc {rss:.1f}MB {cpu:.1f}%cpu".format(
                    backend=item.get("backend"),
                    count=int(item.get("process_count") or 0),
                    rss=float(item.get("total_rss_mb") or 0.0),
                    cpu=float(item.get("total_cpu_percent") or 0.0),
                )
                for item in backend_totals
            )
        )
    lines.extend(
        [
            "",
            "| Task | Phase | Health | Status | Stage | Model | RSS (MB) | CPU % | Children | Last Progress | Hard Budget Left (s) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for task in status.get("tasks", []):
        last_progress = task.get("last_progress_at")
        last_progress_text = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(last_progress)))
            if isinstance(last_progress, (int, float))
            else "-"
        )
        budget = task.get("timeout_budget_remaining_seconds")
        telemetry = task.get("resource_telemetry") or {}
        lines.append(
            "| {task_id} | {phase} | {health} | {status_bucket} | {stage} | {model} | {rss} | {cpu} | {children} | {last_progress} | {budget} |".format(
                task_id=task.get("task_id"),
                phase=task.get("phase") or "-",
                health=task.get("health") or "-",
                status_bucket=task.get("status") or "-",
                stage=task.get("current_stage") or "-",
                model=task.get("model") or "-",
                rss=f"{float(telemetry.get('rss_mb') or 0.0):.1f}"
                if telemetry.get("running")
                else "-",
                cpu=f"{float(telemetry.get('cpu_percent') or 0.0):.1f}"
                if telemetry.get("running")
                else "-",
                children=int(telemetry.get("child_process_count") or 0)
                if telemetry.get("running")
                else "-",
                last_progress=last_progress_text,
                budget=f"{float(budget):.0f}" if isinstance(budget, (int, float)) else "-",
            )
        )
    recent_failures = status.get("recent_failures") or []
    if recent_failures:
        lines.extend(["", "Recent failures:"])
        for failure in recent_failures[:5]:
            lines.append(
                "- {task_id} [{root}] {reason}".format(
                    task_id=failure.get("task_id"),
                    root=failure.get("failure_root") or "unknown",
                    reason=failure.get("failure_reason") or "no failure reason recorded",
                )
            )
    timeout_recoveries = status.get("timeout_recoveries") or []
    if timeout_recoveries:
        lines.extend(["", "Timeout recoveries:"])
        for recovery in timeout_recoveries[:5]:
            lines.append(
                "- {task_id}: {count} recovered timeout events".format(
                    task_id=recovery.get("task_id"),
                    count=int(recovery.get("timeout_recovery_count") or 0),
                )
            )
    return "\n".join(lines)


def write_testgen_run_report(
    output_dir: str | Path,
    *,
    summary: dict[str, Any],
    task_records: Optional[list[dict[str, Any]]] = None,
) -> Path:
    """Write a compact RUN_REPORT.md for test-generation benchmark runs."""

    records = [dict(record) for record in list(task_records or []) if isinstance(record, dict)]
    charged_metrics = compute_testgen_charged_metrics(records)
    input_summary = dict(summary or {})
    summary = {**charged_metrics, **input_summary}
    if "full_unfiltered_pass_at_1" in input_summary:
        official_unfiltered = input_summary.get("full_unfiltered_pass_at_1")
        summary["mean_all_pass_at_1"] = official_unfiltered
        summary["pass_at_1_publishable"] = official_unfiltered
        if not records and "pass_at_1_charged" not in input_summary:
            summary["pass_at_1_charged"] = official_unfiltered
    style_totals: Counter[str] = Counter()
    style_passes: Counter[str] = Counter()
    runner_filtered_passes: Counter[str] = Counter()
    runner_coverage_sum: Counter[str] = Counter()
    runner_mutation_sum: Counter[str] = Counter()
    repo_totals: Counter[str] = Counter()
    repo_passes: Counter[str] = Counter()
    failure_totals: Counter[str] = Counter()
    action_totals: Counter[str] = Counter()
    mutation_completeness_bins: Counter[str] = Counter()
    minimizer_tasks = 0
    broaden_tasks = 0
    saved_tasks: list[dict[str, Any]] = []
    failing_tasks: list[dict[str, Any]] = []
    final_gate_samples: list[str] = []
    atomic_samples: list[str] = []
    repair_changed = 0
    repair_attempt_tasks = 0
    style_mismatch = 0
    final_gate_dropped = 0
    atomic_rejected = 0
    for record in records:
        diagnostics = dict(record.get("diagnostics") or {})
        validation = dict(diagnostics.get("apex_validation") or {})
        style = dict(validation.get("style_profile") or {})
        runner = str(style.get("runner") or "unknown")
        task_name = str(record.get("instance_id") or record.get("task_id") or "")
        repo = task_name.split("__", 1)[0] if "__" in task_name else "unknown"
        style_totals[runner] += 1
        repo_totals[repo] += 1
        if bool(record.get("success")) or float(record.get("pass_at_1") or 0.0) > 0:
            style_passes[runner] += 1
            repo_passes[repo] += 1
        if float(record.get("all_pass_at_1") or 0.0) > 0:
            runner_filtered_passes[runner] += 1
        runner_coverage_sum[runner] += float(
            record.get("coverage") or record.get("coverage_ratio") or 0.0
        )
        runner_mutation_sum[runner] += float(record.get("mutation_score") or 0.0)
        failure_class = (
            validation.get("failure_class")
            or (diagnostics.get("failure_classification") or {}).get("failure_class")
            or "none"
        )
        if failure_class and failure_class != "none":
            failure_totals[str(failure_class)] += 1
        repair_action = (
            validation.get("repair_action")
            or (diagnostics.get("failure_classification") or {}).get("repair_action")
            or ""
        )
        if repair_action:
            action_totals[str(repair_action)] += 1
        final_gate = dict(
            validation.get("final_acceptance_gate")
            or diagnostics.get("final_acceptance_gate")
            or {}
        )
        dropped_count = int(
            final_gate.get("dropped_count") or len(final_gate.get("dropped_tests") or [])
        )
        if dropped_count:
            final_gate_dropped += dropped_count
            if len(final_gate_samples) < 10:
                final_gate_samples.append(
                    f"{task_name}: {', '.join(list(final_gate.get('dropped_tests') or [])[:5])}"
                )
        atomic = dict(
            validation.get("atomic_acceptance") or diagnostics.get("atomic_acceptance") or {}
        )
        rejected_count = int(
            atomic.get("rejected_count") or len(atomic.get("rejected_tests") or [])
        )
        if rejected_count:
            atomic_rejected += rejected_count
            if len(atomic_samples) < 10:
                atomic_samples.append(f"{task_name}: {rejected_count} rejected")
        completeness = record.get("mutation_completeness")
        if completeness is None:
            mutation_completeness_bins["unknown"] += 1
        else:
            value = float(completeness or 0.0)
            if value >= 0.95:
                mutation_completeness_bins[">=95%"] += 1
            elif value >= 0.80:
                mutation_completeness_bins["80-95%"] += 1
            else:
                mutation_completeness_bins["<80%"] += 1
        repair_attempts = int(validation.get("repair_attempts") or 0)
        if validation.get("minimizer_dropped") or int(validation.get("minimizer_attempts") or 0):
            minimizer_tasks += 1
        if int(validation.get("broaden_attempts") or 0):
            broaden_tasks += 1
        if repair_attempts:
            repair_attempt_tasks += 1
        pre = dict(diagnostics.get("pre_repair_result") or {})
        if pre and float(record.get("pass_at_1") or 0.0) > float(pre.get("pass_at_1") or 0.0):
            repair_changed += 1
            saved_tasks.append(record)
        elif pre and float(record.get("all_pass_at_1") or 0.0) > float(
            pre.get("all_pass_at_1") or 0.0
        ):
            saved_tasks.append(record)
        if not (bool(record.get("success")) or float(record.get("pass_at_1") or 0.0) > 0):
            failing_tasks.append(record)
        rendered_validation = json.dumps(validation, sort_keys=True).lower()
        if "forbidden import" in rendered_validation:
            style_mismatch += 1

    lines = [
        "# Test Generation Run Report",
        "",
        "## Headline",
        f"- publishable pass@1: {float(summary.get('pass_at_1_publishable') or 0.0):.3f}",
        f"- charged pass@1: {float(summary.get('pass_at_1_charged') or 0.0):.3f}",
        f"- unfiltered all_pass@1: {float(summary.get('mean_all_pass_at_1') or 0.0):.3f}",
        f"- charged denominator: {int(summary.get('charged_denominator') or 0)}",
        f"- publishable denominator: {int(summary.get('publishable_denominator') or 0)}",
        "",
        "## Summary",
    ]
    for key, value in sorted(dict(summary or {}).items()):
        if isinstance(value, (str, int, float, bool)) or value is None:
            lines.append(f"- {key}: {value}")
    if style_totals:
        lines.extend(
            [
                "",
                "## Pass Rate By Style / Pass Rate By Runner",
                "| Runner | Tasks | Unfiltered Passes | Unfiltered Rate | Filtered Passes | Avg Coverage | Avg Mutation |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for style, total in sorted(style_totals.items()):
            passes = style_passes[style]
            rate = passes / total if total else 0.0
            filtered = runner_filtered_passes[style]
            avg_cov = runner_coverage_sum[style] / total if total else 0.0
            avg_mut = runner_mutation_sum[style] / total if total else 0.0
            lines.append(
                f"| {style} | {total} | {passes} | {rate:.1%} | {filtered} | {avg_cov:.3f} | {avg_mut:.3f} |"
            )
    if repo_totals:
        lines.extend(
            [
                "",
                "## Pass Rate By Repo",
                "| Repo | Tasks | Unfiltered Passes | Pass Rate |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for repo, total in sorted(repo_totals.items()):
            passes = repo_passes[repo]
            lines.append(
                f"| {repo} | {total} | {passes} | {(passes / total if total else 0.0):.1%} |"
            )
    if failure_totals:
        lines.extend(
            [
                "",
                "## Loss Attribution",
                "| Failure Class | Count |",
                "| --- | ---: |",
            ]
        )
        for failure_class, count in failure_totals.most_common():
            lines.append(f"| {failure_class} | {count} |")
    if action_totals:
        lines.extend(
            [
                "",
                "## Repair Action Distribution",
                "| Action | Count |",
                "| --- | ---: |",
            ]
        )
        for action, count in action_totals.most_common():
            lines.append(f"| {action} | {count} |")
    publishable = summary.get("pass_at_1_publishable", summary.get("mean_pass_at_1"))
    charged = summary.get("pass_at_1_charged", summary.get("mean_charged_pass_at_1"))
    if publishable is not None and charged is not None:
        try:
            gap = float(charged) - float(publishable)
        except (TypeError, ValueError):
            gap = 0.0
        if gap > 0.05:
            lines.extend(
                [
                    "",
                    "## Infra Quality Alert",
                    f"- Charged pass@1 exceeds publishable pass@1 by {gap:.1%}; inspect env_skipped tasks before comparing externally.",
                ]
            )
    if records:
        lines.extend(
            [
                "",
                "## Repair Yield",
                f"- Tasks with repair attempts: {repair_attempt_tasks}/{len(records)}",
                f"- Tasks improved by repair: {repair_changed}/{len(records)}",
                f"- Tasks where minimizer fired: {minimizer_tasks}/{len(records)}",
                f"- Tasks where broaden fired: {broaden_tasks}/{len(records)}",
                f"- Style-mismatch guard hits: {style_mismatch}/{len(records)}",
                f"- Tests dropped by final acceptance gate: {final_gate_dropped}",
                f"- Tests rejected by atomic acceptance: {atomic_rejected}",
            ]
        )
        if final_gate_samples:
            lines.extend(["", "Final acceptance gate samples:"])
            lines.extend(f"- {sample}" for sample in final_gate_samples)
        if atomic_samples:
            lines.extend(["", "Atomic acceptance samples:"])
            lines.extend(f"- {sample}" for sample in atomic_samples)
        if mutation_completeness_bins:
            lines.extend(
                [
                    "",
                    "## Mutation Completeness",
                    "| Completeness | Tasks |",
                    "| --- | ---: |",
                ]
            )
            for bucket, count in mutation_completeness_bins.most_common():
                lines.append(f"| {bucket} | {count} |")
        lines.extend(
            [
                "",
                "## Longest Tasks",
                "| Task | Duration (s) | Pass@1 | Failure Class |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        longest = sorted(
            records,
            key=lambda item: float(item.get("duration_seconds") or 0.0),
            reverse=True,
        )[:10]
        for record in longest:
            diagnostics = dict(record.get("diagnostics") or {})
            validation = dict(diagnostics.get("apex_validation") or {})
            failure_class = (
                validation.get("failure_class")
                or (diagnostics.get("failure_classification") or {}).get("failure_class")
                or "-"
            )
            lines.append(
                "| {task} | {duration:.1f} | {pass_at_1:.3f} | {failure} |".format(
                    task=record.get("instance_id") or record.get("task_id") or "-",
                    duration=float(record.get("duration_seconds") or 0.0),
                    pass_at_1=float(record.get("pass_at_1") or 0.0),
                    failure=failure_class,
                )
            )
        if saved_tasks:
            lines.extend(
                [
                    "",
                    "## Saved From Failure",
                    "| Task | Pass@1 | all_pass@1 |",
                    "| --- | ---: | ---: |",
                ]
            )
            for record in saved_tasks[:10]:
                lines.append(
                    "| {task} | {pass_at_1:.3f} | {all_pass_at_1:.3f} |".format(
                        task=record.get("instance_id") or record.get("task_id") or "-",
                        pass_at_1=float(record.get("pass_at_1") or 0.0),
                        all_pass_at_1=float(record.get("all_pass_at_1") or 0.0),
                    )
                )
        if failing_tasks:
            lines.extend(
                [
                    "",
                    "## Still Failing",
                    "| Task | Failure Class | Error |",
                    "| --- | --- | --- |",
                ]
            )
            for record in failing_tasks[:10]:
                diagnostics = dict(record.get("diagnostics") or {})
                validation = dict(diagnostics.get("apex_validation") or {})
                failure_class = (
                    validation.get("failure_class")
                    or (diagnostics.get("failure_classification") or {}).get("failure_class")
                    or "-"
                )
                error = str(record.get("error") or "")[:160].replace("|", "\\|")
                lines.append(
                    "| {task} | {failure} | {error} |".format(
                        task=record.get("instance_id") or record.get("task_id") or "-",
                        failure=failure_class,
                        error=error,
                    )
                )
    lines.append("")
    return atomic_write_text(Path(output_dir) / "RUN_REPORT.md", "\n".join(lines))


def compute_testgen_charged_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    publishable_denominator = len(records)
    charged_records = [
        record for record in records if not _record_is_env_or_harness_failure(record)
    ]
    charged_denominator = len(charged_records)
    publishable_passes = sum(
        1
        for record in records
        if bool(record.get("success")) or float(record.get("pass_at_1") or 0.0) > 0
    )
    charged_passes = sum(
        1
        for record in charged_records
        if bool(record.get("success")) or float(record.get("pass_at_1") or 0.0) > 0
    )
    complete_mutation_records = [
        record
        for record in charged_records
        if record.get("mutation_completeness") is not None
        and float(record.get("mutation_completeness") or 0.0) >= 0.8
    ]
    return {
        "publishable_denominator": publishable_denominator,
        "charged_denominator": charged_denominator,
        "env_or_harness_excluded_count": publishable_denominator - charged_denominator,
        "pass_at_1_publishable": (
            publishable_passes / publishable_denominator if publishable_denominator else 0.0
        ),
        "pass_at_1_charged": (charged_passes / charged_denominator if charged_denominator else 0.0),
        "mutation_complete_denominator": len(complete_mutation_records),
        "mean_mutation_score_complete_only": (
            sum(float(record.get("mutation_score") or 0.0) for record in complete_mutation_records)
            / len(complete_mutation_records)
            if complete_mutation_records
            else 0.0
        ),
    }


def _record_is_env_or_harness_failure(record: dict[str, Any]) -> bool:
    diagnostics = dict(record.get("diagnostics") or {})
    validation = dict(diagnostics.get("apex_validation") or record.get("apex_validation") or {})
    classification = dict(diagnostics.get("failure_classification") or {})
    failure_class = str(
        validation.get("failure_class") or classification.get("failure_class") or ""
    )
    if bool(record.get("env_skipped")):
        return True
    return failure_class.startswith("env_") or failure_class.startswith("harness_")
