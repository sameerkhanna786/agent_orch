"""
CLI-backed model execution helpers for Claude Code, Gemini, Codex, OpenCode,
and MetaCode.
"""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import fnmatch
import hashlib
import http.server
import inspect
import json
import logging
import os
import queue
import re
import secrets
import shlex
import shutil
import signal
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import warnings
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional

from .cli_tool_hooks import (
    build_apex_tool_review_hook_command,
    build_cli_tool_review_hook_config,
    build_opencode_tool_review_plugin_source,
    get_cli_tool_hook_support,
)
from .config import LLMBackend, LLMConfig
from .runtime_policy import classify_command_domain

logger = logging.getLogger("apex.cli_backend")
security_logger = logging.getLogger("apex.security")

# Patterns for environment variables considered host secrets that must NOT be
# inherited by CLI subprocesses. Matched with fnmatch (case-insensitive). The
# per-backend allowlist (see _BACKEND_AUTH_ALLOWLIST) re-admits the specific
# auth keys each CLI binary genuinely needs.
_HOST_SECRET_DENYLIST_PATTERNS: tuple[str, ...] = (
    "*_API_KEY",
    "*_TOKEN",
    "*SECRET*",
    "*PASSWORD*",
    "*PASSWD*",
    "*PRIVATE_KEY*",
    "*CREDENTIAL*",
    "AWS_*",
    "GCP_*",
    "GH_*",
    "GITHUB_*",
    "ANTHROPIC_*",
    "OPENAI_*",
    "GOOGLE_*",
    "AZURE_*",
    "CIRCLE_*",
    "CI_JOB_TOKEN",
    "BUILDKITE_*",
    "GITLAB_*",
    "META_*",
    "FB_*",
    "NPM_TOKEN",
    "PYPI_*",
    "DOCKER_PASSWORD",
    "OPENROUTER_API_KEY",
    "OPENCODE_API_KEY",
    # Database/cluster URLs and signing keys often carry embedded
    # credentials. ``*_DSN``/``*_URI``/``KUBECONFIG``/``*_WEBHOOK*`` are
    # the most common naming conventions. ``*_KEY`` catches generic
    # signing/encryption keys (``JWT_SIGNING_KEY``, ``ENCRYPTION_KEY``,
    # ``SIGNING_KEY``) that don't end in ``_API_KEY``.
    "*_KEY",
    "*_KEYFILE",
    "*_DSN",
    "*_URI",
    "*_WEBHOOK*",
    "DATABASE_URL",
    "REDIS_URL",
    "MONGODB_URI",
    "POSTGRES_URL",
    "POSTGRESQL_URL",
    "AMQP_URL",
    "KAFKA_URL",
    "KUBECONFIG",
    "DOCKER_AUTH_CONFIG",
    "VAULT_TOKEN",
    "VAULT_*",
    "CONSUL_HTTP_TOKEN",
    "NETLIFY_*",
    "VERCEL_*",
    "HEROKU_*",
    "TWILIO_*",
    "STRIPE_*",
    "SENDGRID_*",
    "MAILGUN_*",
    "DATADOG_*",
    "DD_*",
    "NEW_RELIC_*",
    "NEWRELIC_*",
    "OKTA_*",
)


@dataclass(frozen=True)
class _CLIAuthStateFile:
    """Minimal host auth file that may be copied into an isolated CLI home."""

    target_home_relative: str
    source_home_relative: str = ""
    source_env_key: str = ""
    target_env_key: str = ""
    required_env_keys: tuple[str, ...] = ()
    marks_auth: bool = True
    mode: int = 0o600


@dataclass(frozen=True)
class _CLIAuthStateDirectory:
    """Host CLI state directory that may contain vendor-managed auth data."""

    source_home_relative: str
    target_home_relative: str
    exclude_names: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()
    auth_filename_patterns: tuple[str, ...] = (
        "*auth*",
        "*credential*",
        "*credentials*",
        "*oauth*",
        "*token*",
    )
    marks_auth: bool = True
    file_mode: int = 0o600
    dir_mode: int = 0o700
    max_depth: int = 2
    max_scan_files: int = 2000


@dataclass(frozen=True)
class _CLIBackendSandboxSpec:
    """Declarative sandbox/auth contract for one agentic coding CLI."""

    backend: LLMBackend
    binary_names: tuple[str, ...]
    auth_env_allowlist: tuple[str, ...]
    auth_requirements: tuple[tuple[str, ...], ...]
    auth_state_env_key: str
    target_runtime_home: dict[str, str] = field(default_factory=dict)
    target_env_defaults: dict[str, str] = field(default_factory=dict)
    target_path_env_keys: tuple[str, ...] = ()
    container_env_keys: tuple[str, ...] = ()
    auth_state_files: tuple[_CLIAuthStateFile, ...] = ()
    auth_state_directories: tuple[_CLIAuthStateDirectory, ...] = ()
    probe_timeout_seconds: int = 10
    probe_requires_node: bool = True


_CLI_BACKEND_SANDBOX_SPECS: dict[LLMBackend, _CLIBackendSandboxSpec] = {
    LLMBackend.CLAUDE_CLI: _CLIBackendSandboxSpec(
        backend=LLMBackend.CLAUDE_CLI,
        binary_names=("claude",),
        auth_env_allowlist=(
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_CUSTOM_HEADERS",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_VERTEX_BASE_URL",
            "ANTHROPIC_VERTEX_PROJECT_ID",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_CODE_SKIP_VERTEX_AUTH",
            "CLAUDE_CODE_USE_VERTEX",
            "CLOUD_ML_REGION",
            "GOOGLE_APPLICATION_CREDENTIALS",
        ),
        auth_requirements=(
            ("ANTHROPIC_API_KEY",),
            ("ANTHROPIC_AUTH_TOKEN",),
            ("CLAUDE_CODE_OAUTH_TOKEN",),
            ("CLAUDE_CODE_USE_VERTEX", "GOOGLE_APPLICATION_CREDENTIALS"),
            ("ANTHROPIC_VERTEX_BASE_URL", "CLAUDE_CODE_SKIP_VERTEX_AUTH"),
        ),
        auth_state_env_key="APEX_CLAUDE_CLI_AUTH_STATE",
        target_runtime_home={"CLAUDE_CONFIG_DIR": ".claude"},
        target_env_defaults={
            # Claude Code in headless Docker should not persist prompt history
            # or start nonessential updater/plugin/telemetry traffic.
            "CLAUDE_CODE_SKIP_PROMPT_HISTORY": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY": "1",
            "CLAUDE_CODE_DISABLE_TRAJECTORY": "1",
            "DISABLE_AUTOUPDATER": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "DISABLE_INSTALLATION_CHECKS": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_UPDATES": "1",
            "META_3PAI_TELEMETRY_DISABLED": "1",
            "META_CLAUDE_DISABLE_PRESET_TELEMETRY": "1",
            "META_CLAUDE_SILENCE_PLUGIN_INSTALL_SPEW": "1",
            "META_DISABLE_SHAMAN": "1",
            "META_SKIP_BUCK2_WARMUP": "1",
            "META_SKIP_DOTSLASH_WARMUP": "1",
            "THREE_PAI_META_CLAUDE_DISABLE_EXTERNAL_WEB_SEARCH": "1",
        },
        target_path_env_keys=("CLAUDE_CONFIG_DIR", "GOOGLE_APPLICATION_CREDENTIALS"),
        container_env_keys=(
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_CUSTOM_HEADERS",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_VERTEX_BASE_URL",
            "ANTHROPIC_VERTEX_PROJECT_ID",
            "CLAUDE_CODE_DISABLE_ADVISOR_TOOL",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS",
            "CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
            "CLAUDE_CODE_DISABLE_TRAJECTORY",
            "CLAUDE_CODE_SKIP_VERTEX_AUTH",
            "CLAUDE_CODE_SKIP_PROMPT_HISTORY",
            "CLAUDE_CODE_USE_VERTEX",
            "CLOUD_ML_REGION",
            "CLAUDE_CONFIG_DIR",
            "DISABLE_AUTOUPDATER",
            "DISABLE_ERROR_REPORTING",
            "DISABLE_INSTALLATION_CHECKS",
            "DISABLE_TELEMETRY",
            "DISABLE_UPDATES",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "META_3PAI_TELEMETRY_DISABLED",
            "META_CLAUDE_DISABLE_PRESET_TELEMETRY",
            "META_CLAUDE_SILENCE_PLUGIN_INSTALL_SPEW",
            "META_DISABLE_SHAMAN",
            "META_SKIP_BUCK2_WARMUP",
            "META_SKIP_DOTSLASH_WARMUP",
            "THREE_PAI_META_CLAUDE_DISABLE_EXTERNAL_WEB_SEARCH",
        ),
        auth_state_files=(
            # Claude Code on Linux stores portable credentials here when
            # CLAUDE_CONFIG_DIR is used; macOS Keychain state is not copied.
            _CLIAuthStateFile(
                source_home_relative=".claude/.credentials.json",
                target_home_relative=".claude/.credentials.json",
            ),
            _CLIAuthStateFile(
                source_env_key="GOOGLE_APPLICATION_CREDENTIALS",
                target_home_relative=".claude/credentials/google_application_credentials.json",
                target_env_key="GOOGLE_APPLICATION_CREDENTIALS",
            ),
        ),
        auth_state_directories=(
            _CLIAuthStateDirectory(
                source_home_relative=".claude",
                target_home_relative=".claude",
                exclude_names=(
                    "backups",
                    "history.jsonl",
                    "logs",
                    "paste-cache",
                    "shell-snapshots",
                    "stats-cache.json",
                    "todos",
                ),
                exclude_patterns=("settings.json.backup-*", "*.tmp.*"),
            ),
        ),
        probe_timeout_seconds=30,
    ),
    LLMBackend.CODEX_CLI: _CLIBackendSandboxSpec(
        backend=LLMBackend.CODEX_CLI,
        binary_names=("codex",),
        auth_env_allowlist=("OPENAI_API_KEY",),
        auth_requirements=(("OPENAI_API_KEY",),),
        auth_state_env_key="APEX_CODEX_CLI_AUTH_STATE",
        target_runtime_home={"CODEX_HOME": ".codex"},
        target_env_defaults={
            "CODEX_DISABLE_TRAJECTORY": "1",
            "META_DISABLE_SHAMAN": "1",
            "META_SKIP_BUCK2_WARMUP": "1",
            "META_SKIP_DOTSLASH_WARMUP": "1",
        },
        target_path_env_keys=("CODEX_HOME",),
        container_env_keys=(
            "CODEX_DISABLE_TRAJECTORY",
            "CODEX_HOME",
            "META_DISABLE_SHAMAN",
            "META_SKIP_BUCK2_WARMUP",
            "META_SKIP_DOTSLASH_WARMUP",
        ),
        auth_state_files=(
            # Codex supports portable auth.json under CODEX_HOME; copy only
            # that file into the rollout-local home, never host ~/.codex.
            _CLIAuthStateFile(
                source_home_relative=".codex/auth.json",
                target_home_relative=".codex/auth.json",
            ),
        ),
        probe_timeout_seconds=10,
    ),
    LLMBackend.GEMINI_CLI: _CLIBackendSandboxSpec(
        backend=LLMBackend.GEMINI_CLI,
        binary_names=("gemini",),
        auth_env_allowlist=(
            "CODE_ASSIST_API_VERSION",
            "CODE_ASSIST_ENDPOINT",
            "GEMINI_CUSTOM_HEADERS",
            "GEMINI_CLI_CUSTOM_HEADERS",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_CLOUD_ACCESS_TOKEN",
            "GOOGLE_CLOUD_LOCATION",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_PROJECT_ID",
            "GOOGLE_CLOUD_QUOTA_PROJECT",
            "GOOGLE_GEMINI_BASE_URL",
            "GOOGLE_GENAI_USE_GCA",
            "GOOGLE_GENAI_USE_VERTEXAI",
            "GOOGLE_VERTEX_BASE_URL",
        ),
        auth_requirements=(
            ("GEMINI_API_KEY",),
            ("GOOGLE_API_KEY",),
            ("GOOGLE_GENAI_USE_GCA", "GOOGLE_CLOUD_ACCESS_TOKEN"),
            ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_API_KEY"),
            (
                "GOOGLE_GENAI_USE_VERTEXAI",
                "GOOGLE_CLOUD_PROJECT",
                "GOOGLE_CLOUD_LOCATION",
                "GOOGLE_APPLICATION_CREDENTIALS",
            ),
            (
                "GOOGLE_GENAI_USE_VERTEXAI",
                "GOOGLE_CLOUD_PROJECT_ID",
                "GOOGLE_CLOUD_LOCATION",
                "GOOGLE_APPLICATION_CREDENTIALS",
            ),
        ),
        auth_state_env_key="APEX_GEMINI_CLI_AUTH_STATE",
        target_runtime_home={"GEMINI_CLI_HOME": "."},
        target_env_defaults={
            # Gemini CLI 0.42 refuses headless untrusted workdirs; APEX rollouts
            # run in isolated benchmark workspaces owned by the target runtime.
            "GEMINI_CLI_TRUST_WORKSPACE": "true",
        },
        target_path_env_keys=(
            "GEMINI_CLI_HOME",
            "GEMINI_CLI_TRUSTED_FOLDERS_PATH",
            "GOOGLE_APPLICATION_CREDENTIALS",
        ),
        container_env_keys=(
            "CODE_ASSIST_API_VERSION",
            "CODE_ASSIST_ENDPOINT",
            "GEMINI_CLI_TRUST_WORKSPACE",
            "GEMINI_CUSTOM_HEADERS",
            "GEMINI_CLI_CUSTOM_HEADERS",
            "GEMINI_CLI_HOME",
            "GEMINI_CLI_TRUSTED_FOLDERS_PATH",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_CLOUD_LOCATION",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_PROJECT_ID",
            "GOOGLE_CLOUD_QUOTA_PROJECT",
            "GOOGLE_GEMINI_BASE_URL",
            "GOOGLE_GENAI_USE_GCA",
            "GOOGLE_GENAI_USE_VERTEXAI",
            "GOOGLE_VERTEX_BASE_URL",
        ),
        auth_state_files=(
            # Gemini CLI stores OAuth credentials under GEMINI_CLI_HOME/.gemini;
            # copy explicit credential files before the filtered state snapshot.
            _CLIAuthStateFile(
                source_home_relative=".gemini/oauth_creds.json",
                target_home_relative=".gemini/oauth_creds.json",
            ),
            _CLIAuthStateFile(
                source_home_relative=".gemini/gemini-credentials.json",
                target_home_relative=".gemini/gemini-credentials.json",
            ),
            _CLIAuthStateFile(
                source_env_key="GOOGLE_APPLICATION_CREDENTIALS",
                target_home_relative=".gemini/credentials/google_application_credentials.json",
                target_env_key="GOOGLE_APPLICATION_CREDENTIALS",
            ),
            _CLIAuthStateFile(
                source_home_relative=".config/gcloud/application_default_credentials.json",
                target_home_relative=".config/gcloud/application_default_credentials.json",
            ),
            _CLIAuthStateFile(
                source_home_relative=".config/gcloud/credentials.db",
                target_home_relative=".config/gcloud/credentials.db",
            ),
            _CLIAuthStateFile(
                source_home_relative=".config/gcloud/access_tokens.db",
                target_home_relative=".config/gcloud/access_tokens.db",
            ),
        ),
        auth_state_directories=(
            _CLIAuthStateDirectory(
                source_home_relative=".gemini",
                target_home_relative=".gemini",
                exclude_names=("tmp",),
            ),
        ),
        probe_timeout_seconds=30,
    ),
    LLMBackend.OPENCODE_CLI: _CLIBackendSandboxSpec(
        backend=LLMBackend.OPENCODE_CLI,
        binary_names=("opencode", "metacode"),
        auth_env_allowlist=(
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "GROQ_API_KEY",
            "OPENCODE_API_KEY",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENROUTER_API_KEY",
            "XAI_API_KEY",
        ),
        auth_requirements=(
            ("ANTHROPIC_API_KEY",),
            ("OPENAI_API_KEY",),
            ("GOOGLE_API_KEY",),
            ("GEMINI_API_KEY",),
            ("OPENROUTER_API_KEY",),
            ("OPENCODE_API_KEY",),
            ("XAI_API_KEY",),
            ("GROQ_API_KEY",),
            ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        ),
        auth_state_env_key="APEX_OPENCODE_CLI_AUTH_STATE",
        target_runtime_home={"OPENCODE_CONFIG_DIR": ".config/opencode"},
        target_env_defaults={
            "OPENCODE_DISABLE_AUTOUPDATE": "1",
            "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
            "METACODE_DISABLE_TRAJECTORY": "1",
        },
        target_path_env_keys=(
            "OPENCODE_CONFIG",
            "OPENCODE_CONFIG_DIR",
            "OPENCODE_MANAGED_SETTINGS_OVERRIDE",
            "OPENCODE_TEST_MANAGED_CONFIG_DIR",
        ),
        container_env_keys=(
            "ANTHROPIC_BASE_URL",
            "OPENAI_BASE_URL",
            "OPENCODE_CONFIG",
            "OPENCODE_CONFIG_CONTENT",
            "OPENCODE_CONFIG_DIR",
            "OPENCODE_DB",
            "OPENCODE_DISABLE_AUTOUPDATE",
            "OPENCODE_DISABLE_MODELS_FETCH",
            "OPENCODE_DISABLE_PROJECT_CONFIG",
            "OPENCODE_MANAGED_SETTINGS_OVERRIDE",
            "OPENCODE_MODELS_PATH",
            "OPENCODE_MODELS_URL",
            "OPENCODE_PURE",
            "OPENCODE_TEST_MANAGED_CONFIG_DIR",
            "METACODE_DISABLE_TRAJECTORY",
        ),
        probe_timeout_seconds=10,
        probe_requires_node=False,
    ),
    LLMBackend.METACODE_CLI: _CLIBackendSandboxSpec(
        backend=LLMBackend.METACODE_CLI,
        binary_names=("metacode",),
        auth_env_allowlist=(
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "GROQ_API_KEY",
            "OPENCODE_API_KEY",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENROUTER_API_KEY",
            "XAI_API_KEY",
        ),
        auth_requirements=(
            ("ANTHROPIC_API_KEY",),
            ("OPENAI_API_KEY",),
            ("GOOGLE_API_KEY",),
            ("GEMINI_API_KEY",),
            ("OPENROUTER_API_KEY",),
            ("OPENCODE_API_KEY",),
            ("XAI_API_KEY",),
            ("GROQ_API_KEY",),
            ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        ),
        auth_state_env_key="APEX_METACODE_CLI_AUTH_STATE",
        target_runtime_home={"OPENCODE_CONFIG_DIR": ".config/opencode"},
        target_env_defaults={
            "OPENCODE_DISABLE_AUTOUPDATE": "1",
            "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
            "METACODE_DISABLE_TRAJECTORY": "1",
        },
        target_path_env_keys=(
            "OPENCODE_CONFIG",
            "OPENCODE_CONFIG_DIR",
            "OPENCODE_MANAGED_SETTINGS_OVERRIDE",
            "OPENCODE_TEST_MANAGED_CONFIG_DIR",
        ),
        container_env_keys=(
            "ANTHROPIC_BASE_URL",
            "METACODE_DISABLE_TRAJECTORY",
            "OPENAI_BASE_URL",
            "OPENCODE_CONFIG",
            "OPENCODE_CONFIG_CONTENT",
            "OPENCODE_CONFIG_DIR",
            "OPENCODE_DB",
            "OPENCODE_DISABLE_AUTOUPDATE",
            "OPENCODE_DISABLE_MODELS_FETCH",
            "OPENCODE_DISABLE_PROJECT_CONFIG",
            "OPENCODE_MANAGED_SETTINGS_OVERRIDE",
            "OPENCODE_MODELS_PATH",
            "OPENCODE_MODELS_URL",
            "OPENCODE_PURE",
            "OPENCODE_TEST_MANAGED_CONFIG_DIR",
        ),
        probe_timeout_seconds=10,
        probe_requires_node=False,
    ),
}


def _cli_backend_sandbox_spec(backend: LLMBackend) -> _CLIBackendSandboxSpec:
    if not isinstance(backend, LLMBackend):
        backend = LLMBackend(backend)
    return _CLI_BACKEND_SANDBOX_SPECS[backend]


_OPENCODE_FAMILY_BACKENDS: frozenset[LLMBackend] = frozenset(
    {LLMBackend.OPENCODE_CLI, LLMBackend.METACODE_CLI}
)


def _is_opencode_family_backend(backend: Any) -> bool:
    try:
        normalized = backend if isinstance(backend, LLMBackend) else LLMBackend(backend)
    except ValueError:
        return False
    return normalized in _OPENCODE_FAMILY_BACKENDS


# CLI backends normally use their logged-in agentic CLI session/config state,
# not provider API-key env vars. Keep target-container pass-through explicit so
# an isolated Docker agent only receives credentials its own backend can use.
_BACKEND_AUTH_ALLOWLIST: dict[LLMBackend, tuple[str, ...]] = {
    backend: spec.auth_env_allowlist for backend, spec in _CLI_BACKEND_SANDBOX_SPECS.items()
}
_BACKEND_AUTH_REQUIREMENTS: dict[LLMBackend, tuple[tuple[str, ...], ...]] = {
    backend: spec.auth_requirements for backend, spec in _CLI_BACKEND_SANDBOX_SPECS.items()
}

_TARGET_RUNTIME_SOURCE_HOME_ENV_RELATIVES: dict[str, str] = {
    "XDG_CONFIG_HOME": ".config",
    "XDG_CACHE_HOME": ".cache",
    "XDG_DATA_HOME": ".local/share",
    "XDG_STATE_HOME": ".local/state",
}


def env_key_matches_secret_denylist(key: str) -> bool:
    """Return True when ``key`` matches any host-secret denylist pattern."""

    if not key:
        return False
    upper = key.upper()
    for pattern in _HOST_SECRET_DENYLIST_PATTERNS:
        if fnmatch.fnmatchcase(upper, pattern):
            return True
    return False


def redact_host_secrets(
    env: dict[str, str],
    *,
    allow_keys: Iterable[str] = (),
) -> tuple[dict[str, str], list[str]]:
    """Strip host secrets from ``env`` while preserving ``allow_keys``.

    Returns the cleaned env and the sorted list of removed key names so callers
    can audit redactions.
    """

    allow_set = {str(name).upper() for name in allow_keys if name}
    removed: list[str] = []
    cleaned: dict[str, str] = {}
    for key, value in env.items():
        upper = key.upper()
        if upper in allow_set:
            cleaned[key] = value
            continue
        if env_key_matches_secret_denylist(key):
            removed.append(key)
            continue
        cleaned[key] = value
    removed.sort()
    return cleaned, removed


_ACTIVE_CLI_PROCESS_LOCK = threading.Lock()
_ACTIVE_CLI_PROCESS_PIDS: set[int] = set()
_ACTIVE_CLI_ATEXIT_REGISTERED = False
_ACTIVE_CLI_SIGNAL_HANDLERS_INSTALLED = False
_ACTIVE_CLI_PREVIOUS_SIGNAL_HANDLERS: dict[signal.Signals, Any] = {}
_ACTIVE_CLI_CANCEL_REQUESTED = threading.Event()
_CLI_HEALTH_CACHE_LOCK = threading.Lock()
_CLI_HEALTH_CACHE: dict[tuple[str, ...], tuple[bool, str]] = {}
_AIR_GAPPED_CLI_PREP_LOCK = threading.Lock()
_AIR_GAPPED_CLI_PREPARED: set[tuple[str, str, str]] = set()
_AIR_GAPPED_CLI_PREPARED_WITHOUT_VERSION: set[tuple[str, str, str]] = set()
_CLI_PROBE_TIMEOUT_DEFAULT = 10
_CLI_PROBE_TIMEOUT_BY_BACKEND: dict[LLMBackend, int] = {
    backend: spec.probe_timeout_seconds for backend, spec in _CLI_BACKEND_SANDBOX_SPECS.items()
}
_CLI_HEALTH_PROBE_LOOKUP = "lookup"
_CLI_HEALTH_PROBE_SUBPROCESS = "subprocess"
_CLIHealthProbe = tuple[str, str | list[str]]
_CLAUDE_INTERNET_MODE_MARKER = "internet-mode-used_DO_NOT_REMOVE_MANUALLY_SECURITY_RISK"
_HARD_TIMEOUT_PROGRESS_GRACE_MIN_SECONDS = 300.0
_HARD_TIMEOUT_PROGRESS_GRACE_RATIO = 0.5
# CLI backends are agent loops: a single agentic step (planner phase, rollout
# scaffold stage, model-critic call) can legitimately run for many minutes. Floor
# every agentic step's HARD wall-clock at 30 minutes so an upstream per-phase
# override can EXTEND but never SHRINK a step below this — the planner phase caps
# (180/600/900s) previously clamped agents far below their configured
# cli_hard_timeout_seconds and killed them mid-step. Strict CLI hard timeouts
# remain the outer per-call termination guarantee when a config opts into them.
# Stall detection (the soft/progress timeout) is independent and stays tight.
_MIN_AGENT_STEP_HARD_TIMEOUT_SECONDS = 1800
# ----------------------------------------------------------------------------
# Infra non-result retry budget (Layer A, backend-agnostic).
# ----------------------------------------------------------------------------
# A CLI invocation that exits during bootstrap/setup (preset installs, version
# checks, auth) WITHOUT ever running the agent loop, or that hits a transient
# infra fault, made ZERO progress — no parsed result, no workspace edits. These
# are pure infrastructure non-results, not agent/candidate failures, and are
# cheap + safe to retry (nothing to redo). Under heavy concurrent load (e.g. 8
# parallel rollouts each bootstrapping at once) such non-results spike, and a
# single retry (the old budget of 2) is empirically too few — a fresh attempt
# very frequently lets the agent run. We therefore give infra non-results a
# larger retry budget with backoff so transient contention can subside; after
# workspace activity, two recovery attempts are allowed before degrading to the
# saved patch candidate. A real agent failure or a success still returns on the
# first attempt, so this never adds cost to the normal path. Dropping such a
# non-result would silently lose that rollout's work-item (e.g. a decomposition
# module group), capping the final pass rate.
_CLI_INFRA_RETRY_MAX_ATTEMPTS = 4
_CLI_WORKSPACE_TRANSIENT_DEGRADE_AFTER_ATTEMPTS = 3
_CLI_INFRA_RETRY_BACKOFF_BASE_SECONDS = 4.0
_CLI_INFRA_RETRY_BACKOFF_MAX_SECONDS = 30.0
_CLI_RETRY_DIAGNOSTIC_EXCERPT_CHARS = 4000
_CLI_RETRY_DIAGNOSTIC_MAX_ITEMS = 40
_CLI_RETRY_DIAGNOSTIC_MAX_DEPTH = 4
_CLI_TRANSIENT_RECOVERY_DEFAULT_SECONDS = 30.0
_CLAUDE_TRANSIENT_RETRY_RESUME_PROMPT = (
    "Continue the interrupted task in this workspace. Inspect the current changes, "
    "finish any remaining work, run relevant checks, and return the requested "
    "final structured response."
)
_CODEX_TRANSIENT_RETRY_RESUME_PROMPT = (
    "Continue the interrupted task in this workspace. Inspect the current changes, "
    "finish any remaining work, run relevant checks, and return the requested "
    "final structured response."
)
_CLI_BACKEND_CONCURRENCY_LOCK = threading.Lock()
_CLI_BACKEND_CONCURRENCY_SEMAPHORES: dict[tuple[str, str, int], threading.BoundedSemaphore] = {}
_CLI_BACKEND_TRANSIENT_RECOVERY_UNTIL: dict[tuple[str, str], float] = {}


def _sleep_infra_retry_backoff(
    attempt_index: int,
    process_pid: int,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> float:
    """Sleep a bounded, per-process-jittered backoff before retrying an infra
    non-result, then return the slept duration.

    Backoff grows linearly with the attempt and is capped; a small pid-derived
    jitter de-synchronises the herd so N rollouts that all bootstrap-failed at
    once do not retry in lockstep (which would just reproduce the contention).
    Isolated as a module function so unit tests can monkeypatch it to a no-op."""
    base = max(0.0, _CLI_INFRA_RETRY_BACKOFF_BASE_SECONDS)
    backoff = min(_CLI_INFRA_RETRY_BACKOFF_MAX_SECONDS, base * max(1, int(attempt_index)))
    jitter = (int(process_pid) % 997) / 997.0 * base
    delay = backoff + jitter
    if delay > 0 and cancel_check is None:
        time.sleep(delay)
    elif delay > 0:
        deadline = time.monotonic() + delay
        while True:
            try:
                if cancel_check():
                    break
            except Exception:  # noqa: BLE001 - cancellation probes must fail open
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.25, remaining))
    return delay


def _parse_positive_int_env(*keys: str) -> Optional[int]:
    for key in keys:
        raw_value = str(os.environ.get(key) or "").strip()
        if not raw_value:
            continue
        try:
            return max(0, int(raw_value))
        except ValueError:
            logger.warning("Ignoring invalid integer %s=%r", key, raw_value)
            return None
    return None


def _parse_nonnegative_float_env(*keys: str) -> Optional[float]:
    for key in keys:
        raw_value = str(os.environ.get(key) or "").strip()
        if not raw_value:
            continue
        try:
            return max(0.0, float(raw_value))
        except ValueError:
            logger.warning("Ignoring invalid float %s=%r", key, raw_value)
            return None
    return None


def _configured_cli_backend_concurrency_limit(
    config: LLMConfig,
    *,
    target_runtime_enforced: bool,
) -> int:
    backend_value = str(getattr(config.backend, "value", config.backend) or "").strip()
    backend_env = re.sub(r"[^A-Za-z0-9]+", "_", backend_value).strip("_").upper()
    scoped_limit = _parse_positive_int_env(
        f"APEX_TARGET_RUNTIME_{backend_env}_MAX_CONCURRENCY",
        f"APEX_{backend_env}_MAX_CONCURRENCY",
        "APEX_TARGET_RUNTIME_CLI_BACKEND_MAX_CONCURRENCY",
        "APEX_CLI_BACKEND_MAX_CONCURRENCY",
    )
    if scoped_limit is not None:
        return scoped_limit
    if target_runtime_enforced and config.backend == LLMBackend.CLAUDE_CLI:
        # Claude Code's target-runtime bootstrap writes session/config state and
        # obtains provider session material before the stream-json result loop.
        # Concurrent *startups* can exit content-free, so serialize the startup
        # window by default while allowing already-running agents to overlap.
        return 1
    return 0


def _configured_cli_backend_active_concurrency_limit(
    config: LLMConfig,
    *,
    target_runtime_enforced: bool,
) -> int:
    backend_value = str(getattr(config.backend, "value", config.backend) or "").strip()
    backend_env = re.sub(r"[^A-Za-z0-9]+", "_", backend_value).strip("_").upper()
    scoped_limit = _parse_positive_int_env(
        f"APEX_TARGET_RUNTIME_{backend_env}_MAX_ACTIVE_CONCURRENCY",
        f"APEX_{backend_env}_MAX_ACTIVE_CONCURRENCY",
        "APEX_TARGET_RUNTIME_CLI_BACKEND_MAX_ACTIVE_CONCURRENCY",
        "APEX_CLI_BACKEND_MAX_ACTIVE_CONCURRENCY",
    )
    if scoped_limit is not None:
        return scoped_limit
    return 0


def _configured_cli_backend_startup_hold_seconds(
    config: LLMConfig,
    *,
    target_runtime_enforced: bool,
) -> float:
    backend_value = str(getattr(config.backend, "value", config.backend) or "").strip()
    backend_env = re.sub(r"[^A-Za-z0-9]+", "_", backend_value).strip("_").upper()
    parsed = _parse_nonnegative_float_env(
        f"APEX_TARGET_RUNTIME_{backend_env}_STARTUP_SERIAL_SECONDS",
        f"APEX_{backend_env}_STARTUP_SERIAL_SECONDS",
        "APEX_TARGET_RUNTIME_CLI_BACKEND_STARTUP_SERIAL_SECONDS",
        "APEX_CLI_BACKEND_STARTUP_SERIAL_SECONDS",
    )
    if parsed is not None:
        return parsed
    if target_runtime_enforced and config.backend == LLMBackend.CLAUDE_CLI:
        return 45.0
    return 0.0


def _configured_cli_backend_transient_recovery_seconds(
    config: LLMConfig,
    *,
    target_runtime_enforced: bool,
) -> float:
    backend_value = str(getattr(config.backend, "value", config.backend) or "").strip()
    backend_env = re.sub(r"[^A-Za-z0-9]+", "_", backend_value).strip("_").upper()
    parsed = _parse_nonnegative_float_env(
        f"APEX_TARGET_RUNTIME_{backend_env}_TRANSIENT_RECOVERY_SECONDS",
        f"APEX_{backend_env}_TRANSIENT_RECOVERY_SECONDS",
        "APEX_TARGET_RUNTIME_CLI_BACKEND_TRANSIENT_RECOVERY_SECONDS",
        "APEX_CLI_BACKEND_TRANSIENT_RECOVERY_SECONDS",
    )
    if parsed is not None:
        return parsed
    if target_runtime_enforced and config.backend in {LLMBackend.CLAUDE_CLI, LLMBackend.CODEX_CLI}:
        return _CLI_TRANSIENT_RECOVERY_DEFAULT_SECONDS
    return 0.0


def _cli_backend_transient_recovery_key(
    config: LLMConfig,
    *,
    target_runtime_enforced: bool,
) -> tuple[str, str]:
    backend_value = str(getattr(config.backend, "value", config.backend) or "").strip()
    scope = "target_runtime" if target_runtime_enforced else "host"
    return (backend_value or "unknown", scope)


def _record_cli_backend_transient_recovery(
    config: LLMConfig,
    *,
    target_runtime_enforced: bool,
) -> Optional[float]:
    cooldown_seconds = _configured_cli_backend_transient_recovery_seconds(
        config,
        target_runtime_enforced=target_runtime_enforced,
    )
    if cooldown_seconds <= 0:
        return None
    key = _cli_backend_transient_recovery_key(
        config,
        target_runtime_enforced=target_runtime_enforced,
    )
    now = time.time()
    next_allowed_at = now + cooldown_seconds
    with _CLI_BACKEND_CONCURRENCY_LOCK:
        _CLI_BACKEND_TRANSIENT_RECOVERY_UNTIL[key] = max(
            float(_CLI_BACKEND_TRANSIENT_RECOVERY_UNTIL.get(key) or 0.0),
            next_allowed_at,
        )
        return _CLI_BACKEND_TRANSIENT_RECOVERY_UNTIL[key]


def _wait_for_cli_backend_transient_recovery(
    config: LLMConfig,
    *,
    target_runtime_enforced: bool,
    working_dir: str,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> float:
    key = _cli_backend_transient_recovery_key(
        config,
        target_runtime_enforced=target_runtime_enforced,
    )
    waited = 0.0
    logged = False
    while True:
        with _CLI_BACKEND_CONCURRENCY_LOCK:
            next_allowed_at = float(_CLI_BACKEND_TRANSIENT_RECOVERY_UNTIL.get(key) or 0.0)
        remaining = next_allowed_at - time.time()
        if remaining <= 0:
            return waited
        if not logged:
            logger.info(
                "Waiting %.1fs for %s CLI %s transient recovery before launch for %s",
                remaining,
                key[0],
                key[1],
                working_dir,
            )
            logged = True
        try:
            if cancel_check is not None and cancel_check():
                return waited
        except Exception:  # noqa: BLE001 - cancellation probes must fail open
            cancel_check = None
        sleep_for = min(0.25, max(0.0, remaining))
        if sleep_for <= 0:
            return waited
        time.sleep(sleep_for)
        waited += sleep_for


@contextlib.contextmanager
def _cli_backend_concurrency_slot_for_limit(
    config: LLMConfig,
    *,
    target_runtime_enforced: bool,
    working_dir: str,
    slot_kind: str,
    limit: int,
) -> Iterator[None]:
    if limit <= 0:
        yield
        return
    backend_value = str(getattr(config.backend, "value", config.backend) or "").strip()
    scope = "target_runtime" if target_runtime_enforced else "host"
    slot_scope = scope if slot_kind == "startup" else f"{scope}_{slot_kind}"
    key = (backend_value, slot_scope, int(limit))
    with _CLI_BACKEND_CONCURRENCY_LOCK:
        semaphore = _CLI_BACKEND_CONCURRENCY_SEMAPHORES.get(key)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(value=int(limit))
            _CLI_BACKEND_CONCURRENCY_SEMAPHORES[key] = semaphore
    started_waiting_at = time.time()
    acquired = semaphore.acquire(timeout=0)
    if not acquired:
        logger.info(
            "Waiting for %s CLI %s concurrency slot (limit=%s) for %s",
            backend_value or "unknown",
            slot_scope,
            limit,
            working_dir,
        )
        semaphore.acquire()
        wait_seconds = time.time() - started_waiting_at
        logger.info(
            "Acquired %s CLI %s concurrency slot after %.1fs for %s",
            backend_value or "unknown",
            slot_scope,
            wait_seconds,
            working_dir,
        )
    try:
        yield
    finally:
        semaphore.release()


@contextlib.contextmanager
def _cli_backend_concurrency_slot(
    config: LLMConfig,
    *,
    target_runtime_enforced: bool,
    working_dir: str,
) -> Iterator[None]:
    limit = _configured_cli_backend_concurrency_limit(
        config,
        target_runtime_enforced=target_runtime_enforced,
    )
    with _cli_backend_concurrency_slot_for_limit(
        config,
        target_runtime_enforced=target_runtime_enforced,
        working_dir=working_dir,
        slot_kind="startup",
        limit=limit,
    ):
        yield


@contextlib.contextmanager
def _cli_backend_active_concurrency_slot(
    config: LLMConfig,
    *,
    target_runtime_enforced: bool,
    working_dir: str,
) -> Iterator[None]:
    limit = _configured_cli_backend_active_concurrency_limit(
        config,
        target_runtime_enforced=target_runtime_enforced,
    )
    with _cli_backend_concurrency_slot_for_limit(
        config,
        target_runtime_enforced=target_runtime_enforced,
        working_dir=working_dir,
        slot_kind="active",
        limit=limit,
    ):
        yield


class _CLIBackendStartupConcurrencyLease:
    """Release a backend startup slot after startup readiness.

    The slot protects CLI bootstrap, not the full agent turn. Backends that have
    reliable stream/worktree readiness can hold until that evidence arrives;
    other backends can still use a timer as the conservative launch window.
    Normal process completion releases it sooner.
    """

    def __init__(
        self,
        slot: contextlib.AbstractContextManager[None],
        *,
        hold_seconds: float,
        release_on_timer: bool = True,
    ) -> None:
        self._slot = slot
        self._hold_seconds = max(0.0, float(hold_seconds or 0.0))
        self._release_on_timer = bool(release_on_timer)
        self._lock = threading.Lock()
        self._released = False
        self._timer: Optional[threading.Timer] = None

    def __enter__(self) -> "_CLIBackendStartupConcurrencyLease":
        self._slot.__enter__()
        return self

    def release_after_startup_window(self) -> None:
        if self._hold_seconds <= 0:
            self.release()
            return
        if not self._release_on_timer:
            return
        timer = threading.Timer(self._hold_seconds, self.release)
        timer.daemon = True
        self._timer = timer
        timer.start()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
            timer = self._timer
            self._timer = None
        if timer is not None:
            timer.cancel()
        self._slot.__exit__(None, None, None)

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        tb: object,
    ) -> None:
        self.release()


_CLI_STARTUP_LEASE_RELEASE_PROGRESS_SOURCES = frozenset(
    {
        "stdout",
        "stderr",
        "worktree",
        "final_output_file",
    }
)


def _progress_payload_releases_cli_startup_slot(progress: Mapping[str, Any]) -> bool:
    """Return true once the CLI has moved past bootstrap-only startup.

    Target-runtime process/CPU samples are liveness evidence, not startup
    readiness: provider bootstrap churn can emit them before the result stream is
    usable. Keep those signals out of early startup-slot release.
    """

    source = str(progress.get("last_progress_source") or "").strip()
    if source in _CLI_STARTUP_LEASE_RELEASE_PROGRESS_SOURCES:
        return True
    evidence_counts = progress.get("evidence_counts")
    if not isinstance(evidence_counts, Mapping):
        return False
    for evidence_source in _CLI_STARTUP_LEASE_RELEASE_PROGRESS_SOURCES:
        try:
            if int(evidence_counts.get(evidence_source) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _cli_backend_startup_release_on_timer(
    config: LLMConfig,
    *,
    target_runtime_enforced: bool,
) -> bool:
    if target_runtime_enforced and config.backend == LLMBackend.CLAUDE_CLI:
        # Claude Code target-runtime startup can be silent long after the process
        # has launched. Keep the bootstrap serialization window bounded so a
        # live-but-quiet agent cannot hold every later Claude launch indefinitely.
        return True
    return True


# ----------------------------------------------------------------------------
# Progress-based liveness (K2 CLI watchdog).
# ----------------------------------------------------------------------------
# Fallbacks used only when a config object does not expose the new knobs (the
# config dataclass defaults are authoritative; these mirror them so a bare /
# legacy config never produces a degenerate window).
_DEFAULT_STALL_WINDOW_SECONDS = 1200.0
_DEFAULT_MAX_INFLIGHT_REQUEST_SECONDS = 1800.0
_DEFAULT_NO_EDIT_PROGRESS_WINDOW_SECONDS = 1800.0
_DEFAULT_STREAMING_FIRST_OUTPUT_TIMEOUT_SECONDS = 180.0
_CODEX_FINAL_OUTPUT_STABLE_SECONDS = 3.0
# Per-COMMAND (Bash tool) timeout for the claude CLI backend. codex enforces a
# native per-command exec timeout (telemetry: codex_core::tools::router
# "Exit code: 124 command timed out after 120147/180196/300065 ms"); claude has
# NO equivalent, so a hung giant test command (twisted reactor / network /
# deadlock) blocks the agent's Bash tool indefinitely — no host CPU, no stdout,
# no in-container work — until the OUTER scheduler reaps the whole rollout. These
# restore for claude the per-command self-protection codex has natively: a hung
# command returns control to the streaming agent loop fast, which then continues
# toward a passing candidate. This is a PER-COMMAND ceiling, never a rollout cap,
# so it can never cap final_pass_rate below 1.0.
#
# DEFAULT is modest (matches codex's native 120s floor) so unspecified commands
# self-bound; MAX is the giants-safe headroom — generous enough that a legitimate
# full giant suite run still completes, scaled UP from the small-repo baseline by
# the existing expected_test_count size signal (the same factor used by
# _rollout_budget_size_factor / active_cli_hard_timeout_size_factor). Monotone
# non-decreasing in suite size; small repos keep the baseline (size_factor == 1).
_DEFAULT_CLAUDE_BASH_DEFAULT_TIMEOUT_MS = 120000  # 2 min — matches codex floor
_BASE_CLAUDE_BASH_MAX_TIMEOUT_MS = 600000  # 10 min baseline (size_factor == 1)
# Hard ceiling on the size-scaled MAX so even the largest suite cannot push the
# per-command timeout into the multi-hour range (45 min is ~22x the codex 120s
# tier and well above any single legitimate giant test command).
_CLAUDE_BASH_MAX_TIMEOUT_CEILING_MS = 2700000  # 45 min
# Codex-native cap on model-visible output for a single tool result. This does
# not block the command; it prevents megabyte-scale shell output from being sent
# back through the provider stream on the next turn.
_DEFAULT_CODEX_TOOL_OUTPUT_TOKEN_LIMIT = 12000
_INTERACTIVE_ACK_PROMPT_PHRASE = "i have reviewed and verified"
# Claude Code at Meta forces a one-line interactive acknowledgement ("Type
# 'I HAVE REVIEWED AND VERIFIED' to proceed") when a working directory (or an
# ancestor) was previously launched with --internet mode but the current launch
# is non-internet. The required response is a fixed phrase; Apex feeds it on the
# child's stdin up front so the host-launched CLI never blocks waiting for a TTY.
_CLAUDE_INTERNET_REVIEW_ACK_RESPONSE = "I HAVE REVIEWED AND VERIFIED"
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_CLI_STREAM_QUEUE_MAX_CHUNKS = 256
_CLI_STREAM_READ_CHARS = 8192
_PERSISTENT_AIR_GAPPED_HOME_LAYOUT_VERSION = "v1"
_PERSISTENT_AIR_GAPPED_HOME_BACKENDS: frozenset[LLMBackend] = frozenset({LLMBackend.CLAUDE_CLI})
_CLI_BOOTSTRAP_HELPERS: tuple[str, ...] = ("dotslash",)
_WORKSPACE_POLICY_MONITORED_COMMANDS: frozenset[str] = frozenset(
    {
        "awk",
        "cat",
        "cp",
        "diff",
        "du",
        "find",
        "fd",
        "fdfind",
        "grep",
        "ag",
        "head",
        "ls",
        "mv",
        "perl",
        "rg",
        "rsync",
        "sed",
        "tail",
        "tar",
        "tee",
        "tree",
    }
)
_TARGET_RUNTIME_DYNAMIC_COMMANDS: frozenset[str] = frozenset(
    {
        "python",
        "python3",
        "python3.10",
        "python3.11",
        "python3.12",
        "python3.13",
        "bash",
        "sh",
        "zsh",
        "env",
        "xargs",
        "git",
        "docker",
        "curl",
        "wget",
        "perl",
        "awk",
        "make",
        "timeout",
        "gtimeout",
        "patch",
        "diff",
        "shasum",
        "sha1sum",
        "sha256sum",
        "md5sum",
        "pytest",
        "py.test",
        "pip",
        "pip3",
        "uv",
        "poetry",
        "hatch",
        "tox",
        "nox",
        "coverage",
        "django-admin",
        "node",
        "npm",
        "npx",
        "pnpm",
        "yarn",
        "bun",
        "deno",
        "go",
        "cargo",
        "mvn",
        "gradle",
        "java",
        "ruby",
        "bundle",
        "rspec",
        "php",
        "phpunit",
        "dotnet",
        "swift",
    }
)
_TARGET_RUNTIME_PROMPT_DYNAMIC_TOOL_NAMES = "|".join(
    re.escape(name) for name in sorted(_TARGET_RUNTIME_DYNAMIC_COMMANDS, key=len, reverse=True)
)
_TARGET_RUNTIME_PROMPT_ABSOLUTE_TOOL_RE = re.compile(
    rf"(?P<prefix>^|[\s`'\"(=])"
    rf"(?P<path>/(?:[^\s`'\"<>|;&$]+/)*(?:bin|sbin|Scripts)/"
    rf"(?P<tool>{_TARGET_RUNTIME_PROMPT_DYNAMIC_TOOL_NAMES}))"
    rf"(?=$|[\s`'\"\),.;:|&])"
)
_WORKSPACE_POLICY_BACKEND_HELPER_MARKERS: tuple[str, ...] = (
    ".apex_agent_runtime/",
    "apex-cli-offline-",
    "cli_airgapped_homes/",
    "fastzip-castree-",
)
# Read-only search commands can briefly run from backend/runtime package
# extraction directories while a CLI starts or finalizes. These are not task
# repository roots; explicit path escapes from the workspace remain fatal.
_WORKSPACE_POLICY_BACKEND_RUNTIME_CWD_PREFIXES: tuple[str, ...] = (
    "/usr/local/",
    "/opt/homebrew/",
    "/System/",
    "/Library/",
)
_WORKSPACE_POLICY_BACKEND_RUNTIME_CWD_MARKERS: tuple[str, ...] = (
    "/_internal/",
    "/site-packages/",
    "/dist-packages/",
    "/node_modules/",
    "/claude_code/",
    "/agent-market/",
    "/fbcode/platform",
    ".framework/Versions/",
)
# Path roots that, when targeted by a monitored read-only command, should be
# treated as backend-helper noise rather than a fatal workspace violation. Keep
# this scoped to APEX-managed helper roots only; arbitrary /tmp, sibling task
# workspaces, and broad system paths remain fatal policy violations.
_WORKSPACE_POLICY_SYSTEM_TARGET_PREFIXES: tuple[str, ...] = (
    ".apex_agent_runtime/",
    "apex-cli-offline-",
    "cli_airgapped_homes/",
    "fastzip-castree-",
)
_WORKSPACE_POLICY_CONTAINER_SHELL_EXECUTABLES: frozenset[str] = frozenset(
    {
        "/bin/bash",
        "/bin/sh",
        "/bin/zsh",
        "/usr/bin/bash",
        "/usr/bin/sh",
        "/usr/bin/zsh",
    }
)
_WORKSPACE_POLICY_CLAUDE_TASK_OUTPUT_SUFFIX = ".output"
_WORKSPACE_POLICY_TRANSIENT_OUTPUT_ROOTS: tuple[str, ...] = (
    "/tmp/",
    "/private/tmp/",
    "/var/tmp/",
)
_WORKSPACE_POLICY_SHELL_OUTPUT_REDIRECT_TOKENS: frozenset[str] = frozenset(
    {
        ">",
        ">|",
        ">>",
        "1>",
        "1>|",
        "1>>",
        "2>",
        "2>|",
        "2>>",
        "&>",
        "&>|",
        "&>>",
    }
)
_WORKSPACE_POLICY_TRANSIENT_OUTPUT_OPTION_MARKERS: tuple[str, ...] = (
    "json-report",
    "junit",
    "coverage",
    "cov-report",
    "report",
    "result",
    "output",
    "outfile",
    "out-file",
    "log-file",
    "logfile",
)
_WORKSPACE_POLICY_EMBEDDED_ABSOLUTE_PATH_ROOTS = (
    "bin",
    "dev",
    "etc",
    "home",
    "Library",
    "opt",
    "private",
    "root",
    "sbin",
    "System",
    "testbed",
    "tmp",
    "Users",
    "usr",
    "var",
    "workspace",
)
_WORKSPACE_POLICY_EMBEDDED_ABSOLUTE_PATH_ROOT_RE = "|".join(
    re.escape(root) for root in _WORKSPACE_POLICY_EMBEDDED_ABSOLUTE_PATH_ROOTS
)
_WORKSPACE_POLICY_EXTERNAL_SOURCE_HELPER_MARKERS: frozenset[str] = frozenset(
    {
        "download",
        "fetch",
        "mirror",
        "vendor",
    }
)
_WORKSPACE_POLICY_TRANSIENT_SCRIPT_SUFFIXES: frozenset[str] = frozenset(
    {".cjs", ".js", ".mjs", ".php", ".pl", ".py", ".rb"}
)
_WORKSPACE_POLICY_LOCAL_NETWORK_HOSTS: frozenset[str] = frozenset(
    {"localhost", "127.0.0.1", "::1", "host.docker.internal"}
)
_WORKSPACE_POLICY_INLINE_URL_PROBE_COMMANDS: frozenset[str] = frozenset(
    {
        "python",
        "python3",
        "python3.10",
        "python3.11",
        "python3.12",
        "python3.13",
        "perl",
        "ruby",
        "node",
        "nodejs",
        "deno",
        "php",
    }
)
_WORKSPACE_POLICY_INLINE_URL_PROBE_MARKERS: tuple[str, ...] = (
    "urlopen",
    "urlretrieve",
    "urllib.request",
    "urllib3",
    "requests.",
    "httpx.",
    "aiohttp.",
    "fetch(",
    "http.client",
    "https",
)
_WORKSPACE_POLICY_FORBIDDEN_GIT_HISTORY_SUBCOMMANDS: frozenset[str] = frozenset(
    {
        "blame",
        "cat-file",
        "log",
        "ls-tree",
        "reflog",
        "rev-list",
        "show",
    }
)
_WORKSPACE_POLICY_GIT_GLOBAL_OPTIONS_WITH_VALUE: frozenset[str] = frozenset(
    {
        "-C",
        "-c",
        "--git-dir",
        "--namespace",
        "--super-prefix",
        "--work-tree",
    }
)
_WORKSPACE_POLICY_GIT_GLOBAL_OPTIONS: frozenset[str] = frozenset(
    {
        "--bare",
        "--no-optional-locks",
        "--no-pager",
        "--paginate",
        "--version",
    }
)
_CLI_CLEANUP_WARNING_SIGNATURES: set[tuple[str, str, str]] = set()
_CLI_CLEANUP_WARNING_LOCK = threading.Lock()


def _process_cwd(pid: int) -> Optional[Path]:
    proc_cwd = Path(f"/proc/{pid}/cwd")
    if proc_cwd.exists():
        try:
            return proc_cwd.resolve()
        except OSError:
            pass
    if not shutil.which("lsof"):
        return None
    result = subprocess.run(
        ["lsof", "-a", "-d", "cwd", "-Fn", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.startswith("n"):
            try:
                return Path(line[1:]).resolve()
            except OSError:
                return None
    return None


def _log_cli_cleanup_warning_once(kind: str, path: str, exc: OSError) -> None:
    signature = (
        str(kind or ""),
        str(path or ""),
        f"{type(exc).__name__}:{getattr(exc, 'errno', None)}:{exc}",
    )
    with _CLI_CLEANUP_WARNING_LOCK:
        first_emission = signature not in _CLI_CLEANUP_WARNING_SIGNATURES
        if first_emission:
            _CLI_CLEANUP_WARNING_SIGNATURES.add(signature)
    if first_emission:
        logger.warning(
            "Failed to remove CLI temp %s %s during cleanup: %s",
            kind,
            path,
            exc,
        )


@dataclass
class CLIModelResult:
    """Normalized result from a CLI-backed execution."""

    success: bool
    text: str = ""
    parsed_json: Optional[dict[str, Any]] = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw_output: str = ""
    duration_seconds: float = 0.0
    error: Optional[str] = None
    timeout_audit: dict[str, Any] = field(default_factory=dict)
    response_status: str = "unknown"
    workspace_status: str = "unknown"
    patch_extraction_status: str = "unknown"
    finalization_status: str = "unknown"
    telemetry_status: str = "unknown"
    backend_diagnostics: dict[str, Any] = field(default_factory=dict)


class CLIProcessTimeout(subprocess.TimeoutExpired):
    """Timeout raised when a CLI subprocess exceeds a specific timeout mode."""

    def __init__(
        self,
        cmd: Any,
        timeout: float,
        *,
        timeout_kind: str,
        output: Optional[str] = None,
        stderr: Optional[str] = None,
        timeout_audit: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(cmd, timeout, output=output, stderr=stderr)
        self.timeout_kind = timeout_kind
        self.timeout_audit = dict(timeout_audit or {})


class CLIProcessPolicyViolation(RuntimeError):
    """Raised when a CLI subprocess violates Apex workspace execution policy."""

    def __init__(
        self,
        reason: str,
        *,
        output: Optional[str] = None,
        stderr: Optional[str] = None,
        policy_audit: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.output = output or ""
        self.stderr = stderr or ""
        self.policy_audit = dict(policy_audit or {})


class CLIProcessInteractionRequired(RuntimeError):
    """Raised when a nested CLI asks for terminal input in batch mode."""

    def __init__(
        self,
        reason: str,
        *,
        output: Optional[str] = None,
        stderr: Optional[str] = None,
        interaction_audit: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.output = output or ""
        self.stderr = stderr or ""
        self.interaction_audit = dict(interaction_audit or {})


class CLIProcessOutputLimitExceeded(RuntimeError):
    """Raised when a CLI subprocess emits more output than Apex will retain."""

    def __init__(
        self,
        reason: str,
        *,
        output: Optional[str] = None,
        stderr: Optional[str] = None,
        output_audit: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.output = output or ""
        self.stderr = stderr or ""
        self.output_audit = dict(output_audit or {})


class CLIProcessProgressAbort(RuntimeError):
    """Raised when a progress observer decides a live CLI run is invalid."""

    def __init__(
        self,
        reason: str,
        *,
        output: Optional[str] = None,
        stderr: Optional[str] = None,
        progress_audit: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.output = output or ""
        self.stderr = stderr or ""
        self.progress_audit = dict(progress_audit or {})


class _BoundedStreamCapture:
    """Capture stream output with a preserved head and tail."""

    def __init__(self, stream_name: str, max_chars: int) -> None:
        self.stream_name = stream_name
        self.max_chars = max(0, int(max_chars))
        self.total_chars = 0
        self._parts: Optional[list[str]] = []
        self._parts_chars = 0
        self._head = ""
        self._tail = ""

    def append(self, chunk: str) -> None:
        if not chunk:
            return
        self.total_chars += len(chunk)
        if self.max_chars <= 0:
            self._parts = None
            self._head = ""
            self._tail = ""
            return
        if self._parts is not None:
            if self._parts_chars + len(chunk) <= self.max_chars:
                self._parts.append(chunk)
                self._parts_chars += len(chunk)
                return
            combined = "".join(self._parts) + chunk
            head_limit = self.max_chars // 2
            tail_limit = self.max_chars - head_limit
            self._head = combined[:head_limit]
            self._tail = combined[-tail_limit:] if tail_limit else ""
            self._parts = None
            self._parts_chars = 0
            return
        tail_limit = self.max_chars - (self.max_chars // 2)
        self._tail = (self._tail + chunk)[-tail_limit:] if tail_limit else ""

    @property
    def omitted_chars(self) -> int:
        if self.max_chars <= 0:
            return self.total_chars
        return max(0, self.total_chars - self.max_chars)

    @property
    def truncated(self) -> bool:
        return self._parts is None and self.omitted_chars > 0

    def text(self) -> str:
        if self._parts is not None:
            return "".join(self._parts)
        if self.max_chars <= 0:
            return f"[apex omitted {self.total_chars} chars from {self.stream_name} capture]"
        marker = (
            f"\n...[apex truncated {self.omitted_chars} chars from {self.stream_name} capture]...\n"
        )
        return f"{self._head}{marker}{self._tail}"

    def audit(self) -> dict[str, Any]:
        return {
            "max_chars": self.max_chars,
            "total_chars": self.total_chars,
            "captured_chars": min(self.total_chars, self.max_chars),
            "omitted_chars": self.omitted_chars,
            "truncated": self.truncated,
        }


def _collect_subprocess_tree_pids(root_pid: int) -> set[int]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return set()

    children_by_parent: dict[int, list[int]] = {}
    visible_pids: set[int] = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        visible_pids.add(pid)
        children_by_parent.setdefault(ppid, []).append(pid)

    if root_pid not in visible_pids:
        return set()

    tracked = {root_pid}
    stack = [root_pid]
    while stack:
        parent = stack.pop()
        for child in children_by_parent.get(parent, []):
            if child in tracked:
                continue
            tracked.add(child)
            stack.append(child)
    return tracked


def _signal_subprocess_tree(
    pids: set[int],
    signum: signal.Signals,
) -> None:
    if not pids:
        return

    if hasattr(os, "killpg"):
        pgids: set[int] = set()
        for pid in pids:
            try:
                pgids.add(os.getpgid(pid))
            except (ProcessLookupError, PermissionError):
                continue
        for pgid in pgids:
            try:
                os.killpg(pgid, signum)
            except (ProcessLookupError, PermissionError):
                continue

    for pid in sorted(pids, reverse=True):
        try:
            os.kill(pid, signum)
        except (ProcessLookupError, PermissionError):
            continue


_TARGET_RUNTIME_CLEANUP_ENV_KEYS: tuple[str, ...] = (
    "APEX_TARGET_TOOL_CONTEXT",
    "APEX_TARGET_TOOL_WORKDIR",
    "APEX_CLI_INVOCATION_ID",
    "APEX_AGENT_CONTAINER",
)
_TARGET_RUNTIME_ACTIVITY_CHECK_INTERVAL_SECONDS = 1.0
_TARGET_RUNTIME_STRUCTURAL_GIT_HISTORY_POLICIES = {
    "allow",
    "allowed",
    "structurally_erased",
    "structurally-erased",
}
_TARGET_RUNTIME_STRUCTURAL_SOURCE_NETWORK_POLICIES = {
    "structurally_denied",
    "structurally-denied",
    "container_denied",
    "container-denied",
    "blocked_by_boundary",
    "blocked-by-boundary",
}
_TARGET_RUNTIME_STRUCTURAL_FILESYSTEM_POLICIES = {
    "structurally_isolated",
    "structurally-isolated",
    "container_isolated",
    "container-isolated",
    "sandbox_isolated",
    "sandbox-isolated",
}


def _target_runtime_policy_value(
    activity: dict[str, Any],
    env: Optional[dict[str, str]],
    key: str,
    default: str,
) -> str:
    value = str(activity.get(key) or "").strip()
    if value:
        return value
    if isinstance(env, dict) and env.get("APEX_TARGET_TOOL_CONTEXT"):
        context = _load_target_runtime_context(env)
        value = str(context.get(key) or "").strip()
        if value:
            return value
    return default


def _target_runtime_git_history_is_structural(policy: str) -> bool:
    return str(policy or "").strip().lower() in _TARGET_RUNTIME_STRUCTURAL_GIT_HISTORY_POLICIES


def _target_runtime_source_network_is_structural(policy: str) -> bool:
    return str(policy or "").strip().lower() in _TARGET_RUNTIME_STRUCTURAL_SOURCE_NETWORK_POLICIES


def _target_runtime_filesystem_boundary_is_structural(policy: str) -> bool:
    return str(policy or "").strip().lower() in _TARGET_RUNTIME_STRUCTURAL_FILESYSTEM_POLICIES


def _target_runtime_args(marker: dict[str, Any]) -> list[str]:
    return [str(arg or "") for arg in list(marker.get("args") or [])]


def _target_runtime_marker_is_package_fetch(tool: str, args: list[str]) -> bool:
    tool_name = Path(str(tool or "")).name.lower()
    arg_words = [str(arg or "").strip().lower() for arg in args if str(arg or "").strip()]
    package_verbs = {
        "add",
        "download",
        "fetch",
        "install",
        "lock",
        "resolve",
        "sync",
        "update",
        "wheel",
    }
    if tool_name in {"pip", "pip3", "pipx"}:
        return any(word in package_verbs for word in arg_words)
    if tool_name in {"uv", "poetry", "pipenv", "conda", "mamba"}:
        return any(word in package_verbs for word in arg_words)
    if tool_name in {"npm", "pnpm", "yarn"}:
        return any(word in package_verbs or word == "ci" for word in arg_words)
    if tool_name.startswith("python"):
        for index, word in enumerate(arg_words[:-1]):
            if word == "-m" and arg_words[index + 1] in {"pip", "piptools", "pip_tool"}:
                return any(candidate in package_verbs for candidate in arg_words[index + 2 :])
    return False


def _target_runtime_marker_is_source_fetch(tool: str, args: list[str]) -> bool:
    tool_name = Path(str(tool or "")).name.lower()
    arg_words = [str(arg or "").strip().lower() for arg in args if str(arg or "").strip()]
    if tool_name in {"curl", "wget"}:
        return any(word.startswith(("http://", "https://")) for word in arg_words)
    if tool_name == "git":
        return any(
            word in {"fetch", "pull", "ls-remote", "submodule", "clone"} for word in arg_words
        )
    return False


def _target_runtime_policy_marker_is_structurally_redundant(
    marker: Any,
    *,
    git_history_structural: bool,
    source_network_structural: bool,
    filesystem_boundary_structural: bool,
) -> bool:
    if not isinstance(marker, dict):
        return False
    reason = str(marker.get("reason") or "").strip().lower()
    policy_kind = str(marker.get("policy_kind") or "").strip().lower()
    path_token = str(marker.get("path_token") or "").strip().lower()
    tool = Path(str(marker.get("tool") or "")).name.lower()
    args = _target_runtime_args(marker)
    if git_history_structural and "git history/object discovery" in reason:
        return True
    if source_network_structural and (
        policy_kind
        in {
            "dependency_fetch",
            "external_source_acquisition",
            "network_egress",
            "package_egress",
            "source_network",
        }
        or path_token.startswith(("http://", "https://"))
        or "external source acquisition" in reason
        or "source/network" in reason
        or "source fetch" in reason
        or "source egress" in reason
        or "package egress" in reason
        or "package fetch" in reason
        or "dependency-fetch" in reason
        or "dependency fetch" in reason
        or "dependency download" in reason
        or "dependency installation" in reason
        or _target_runtime_marker_is_source_fetch(tool, args)
        or _target_runtime_marker_is_package_fetch(tool, args)
    ):
        return True
    if filesystem_boundary_structural and (
        "absolute host dynamic tool paths" in reason
        or "static tool path escapes target workspace" in reason
        or "unsupported mutating find invocation" in reason
    ):
        return True
    return False


def _target_runtime_cleanup_metadata(
    env: Optional[dict[str, str]],
) -> dict[str, Any]:
    if not isinstance(env, dict):
        return {}
    target_env = {
        key: str(env.get(key) or "")
        for key in _TARGET_RUNTIME_CLEANUP_ENV_KEYS
        if str(env.get(key) or "").strip()
    }
    if not target_env.get("APEX_TARGET_TOOL_CONTEXT"):
        return {}
    return {"env": target_env}


def _cleanup_target_runtime_from_metadata(
    metadata: Optional[dict[str, Any]],
    *,
    signum: int,
) -> None:
    env = metadata.get("env") if isinstance(metadata, dict) else None
    if not isinstance(env, dict) or not env.get("APEX_TARGET_TOOL_CONTEXT"):
        return
    try:
        from apex.evaluation.target_runtime import cleanup_target_runtime_processes

        cleanup_target_runtime_processes(env, signum=signum)
    except Exception:
        return


def _cleanup_target_runtime_for_env(
    env: Optional[dict[str, str]],
    *,
    signum: int,
) -> None:
    _cleanup_target_runtime_from_metadata(
        _target_runtime_cleanup_metadata(env),
        signum=signum,
    )


def _cleanup_target_runtime_after_cli_completion(
    env: Optional[dict[str, str]],
) -> None:
    """Drain detached target-runtime processes after a CLI stage exits.

    Agent CLIs can spawn background helpers that keep editing after the main
    ``docker exec`` command returns. Once Apex has a terminal stage result,
    any workdir-scoped child still running is stale and would corrupt later
    diff/verification accounting.
    """

    if not isinstance(env, dict) or not env.get("APEX_TARGET_TOOL_CONTEXT"):
        return
    cleaned = set()
    try:
        from apex.evaluation.target_runtime import cleanup_target_runtime_processes

        cleaned = cleanup_target_runtime_processes(env, signum=signal.SIGTERM)
        if cleaned:
            time.sleep(0.2)
            cleanup_target_runtime_processes(env, signum=signal.SIGKILL)
    except Exception:
        return


def _sample_target_runtime_process_activity(
    env: Optional[dict[str, str]],
) -> dict[str, Any]:
    if not isinstance(env, dict) or not env.get("APEX_TARGET_TOOL_CONTEXT"):
        return {}
    try:
        from apex.evaluation.target_runtime import target_runtime_process_activity

        activity = target_runtime_process_activity(env)
    except Exception:
        return {}
    return dict(activity or {}) if isinstance(activity, dict) else {}


def _target_runtime_completion_policy_audit_base(
    *,
    working_dir: str,
    target_runtime_activity: dict[str, Any],
    policy_violation: dict[str, Any],
) -> dict[str, Any]:
    now = time.time()
    return {
        "started_at": now,
        "ended_at": now,
        "working_dir": working_dir,
        "track_worktree": True,
        "progress_timeout_seconds": None,
        "first_output_timeout_seconds": None,
        "hard_timeout_seconds": None,
        "hard_timeout_progress_grace_seconds": None,
        "last_progress_at": now,
        "last_progress_source": "target_runtime_completion_policy",
        "last_stdout_at": None,
        "last_stderr_at": None,
        "last_worktree_at": None,
        "last_cpu_at": None,
        "last_target_runtime_activity_at": now,
        "target_runtime_activity": dict(target_runtime_activity),
        "evidence_counts": {
            "stdout": 0,
            "stderr": 0,
            "worktree": 0,
            "cpu": 0,
            "target_runtime_cpu": 0,
            "target_runtime_process": 1,
        },
        "terminal_state": "target_runtime_policy_violation",
        "policy_violation": dict(policy_violation),
        "completion_policy_check": True,
    }


def _snapshot_registered_cli_processes(*, clear: bool = False) -> set[int]:
    with _ACTIVE_CLI_PROCESS_LOCK:
        active_pids = set(_ACTIVE_CLI_PROCESS_PIDS)
        if clear:
            _ACTIVE_CLI_PROCESS_PIDS.clear()
    return active_pids


def _signal_registered_cli_processes(
    signum: signal.Signals,
    *,
    clear: bool = False,
) -> set[int]:
    active_pids = _snapshot_registered_cli_processes(clear=clear)
    if not active_pids:
        return set()
    _signal_subprocess_tree(active_pids, signum)
    return active_pids


def _cleanup_registered_cli_processes() -> None:
    active_pids = _snapshot_registered_cli_processes(clear=True)
    if not active_pids:
        return

    tracked_pids: set[int] = set()
    for pid in active_pids:
        tracked_pids.update(_collect_subprocess_tree_pids(pid) or {pid})
    _signal_subprocess_tree(tracked_pids, signal.SIGTERM)

    deadline = time.time() + 1.0
    remaining = set(tracked_pids)
    while remaining and time.time() < deadline:
        time.sleep(0.1)
        remaining = set()
        for pid in active_pids:
            remaining.update(_collect_subprocess_tree_pids(pid))

    if remaining:
        _signal_subprocess_tree(remaining, signal.SIGKILL)


def _handle_cli_cleanup_signal(signum: int, frame: Any) -> None:
    # Keep signal-time cleanup lightweight. The full tree walk happens in the
    # regular atexit path; here we only terminate the registered process groups.
    _ACTIVE_CLI_CANCEL_REQUESTED.set()
    _signal_registered_cli_processes(signal.SIGTERM, clear=False)

    previous = _ACTIVE_CLI_PREVIOUS_SIGNAL_HANDLERS.get(
        signal.Signals(signum),
        signal.SIG_DFL,
    )
    if previous == signal.SIG_IGN:
        return
    if callable(previous):
        try:
            previous(signum, frame)
        except (KeyboardInterrupt, SystemExit):
            logger.info(
                "Suppressed terminal exception from previous CLI cleanup signal handler "
                "after recording cancellation.",
            )
        return


def cli_cleanup_signal_requested() -> bool:
    """Return whether an installed CLI cleanup handler observed cancellation."""

    return _ACTIVE_CLI_CANCEL_REQUESTED.is_set()


def clear_cli_cleanup_signal_requested() -> None:
    """Clear the process-wide CLI cancellation flag.

    Tests and fresh top-level runs use this to avoid inheriting cancellation
    state from an earlier interrupted run in the same Python process.
    """

    _ACTIVE_CLI_CANCEL_REQUESTED.clear()


def _install_cli_process_cleanup_hooks() -> None:
    global _ACTIVE_CLI_ATEXIT_REGISTERED
    global _ACTIVE_CLI_SIGNAL_HANDLERS_INSTALLED

    with _ACTIVE_CLI_PROCESS_LOCK:
        if not _ACTIVE_CLI_ATEXIT_REGISTERED:
            atexit.register(_cleanup_registered_cli_processes)
            _ACTIVE_CLI_ATEXIT_REGISTERED = True

        if threading.current_thread() is not threading.main_thread():
            return
        if _ACTIVE_CLI_SIGNAL_HANDLERS_INSTALLED:
            return

        signals_to_install = [signal.SIGTERM, signal.SIGINT]
        if hasattr(signal, "SIGHUP"):
            signals_to_install.append(signal.SIGHUP)
        for signum in signals_to_install:
            try:
                _ACTIVE_CLI_PREVIOUS_SIGNAL_HANDLERS[signum] = signal.getsignal(signum)
                signal.signal(signum, _handle_cli_cleanup_signal)
            except (OSError, RuntimeError, ValueError):
                continue

        _ACTIVE_CLI_SIGNAL_HANDLERS_INSTALLED = True


def ensure_cli_process_cleanup_hooks() -> None:
    """Install CLI child cleanup hooks for the current process when possible."""

    _install_cli_process_cleanup_hooks()


def terminate_registered_cli_processes() -> set[int]:
    """Terminate ALL currently registered CLI subprocess trees (legacy).

    Deprecated: this kills every registered CLI child regardless of which
    rollout owns it. Two concurrent rollouts that share the global registry
    cross-contaminate on a single deadline expiry. Use
    ``RolloutCLIRegistry.terminate_for_rollout`` instead, which is scoped to
    a specific rollout id.

    Retained for backwards compatibility with callers that legitimately
    want a global drain (atexit cleanup, signal handlers).
    """

    warnings.warn(
        "terminate_registered_cli_processes() terminates CLI children "
        "across all rollouts. Use RolloutCLIRegistry.terminate_for_rollout "
        "for scoped, per-rollout teardown.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _signal_registered_cli_processes(signal.SIGTERM, clear=False)


def _register_active_cli_process(pid: int) -> None:
    if not isinstance(pid, int) or pid <= 0 or pid == os.getpid():
        return
    _install_cli_process_cleanup_hooks()
    with _ACTIVE_CLI_PROCESS_LOCK:
        _ACTIVE_CLI_PROCESS_PIDS.add(pid)


def _unregister_active_cli_process(pid: int) -> None:
    if not isinstance(pid, int) or pid <= 0:
        return
    with _ACTIVE_CLI_PROCESS_LOCK:
        _ACTIVE_CLI_PROCESS_PIDS.discard(pid)


# ---------------------------------------------------------------------------
# Per-rollout CLI process registry (Phase 2.5)
# ---------------------------------------------------------------------------
#
# The legacy global ``_ACTIVE_CLI_PROCESS_PIDS`` set could not distinguish
# between rollouts when an executor expired one of them: ``terminate_registered
# _cli_processes`` killed every CLI child across the parent process. With the
# parallel rollout engine, that meant a slow rollout's deadline could nuke a
# sibling rollout that was about to succeed. ``RolloutCLIRegistry`` scopes
# registration by rollout id so per-deadline teardown only signals the offending
# rollout.
#
# The class is intentionally small and threadsafe: one ``threading.Lock`` guards
# the dict, terminations grab a snapshot under the lock, then signal outside.
# We delegate the actual signal/process-tree walk to the existing
# ``_signal_subprocess_tree`` helper to share OOM/SIGKILL retry behaviour.


# A thread/coroutine-scoped pointer to the currently-active rollout
# context. ``CLIModelClient`` consults this contextvar when no explicit
# ``rollout_registry`` was passed to its constructor, so the engine can
# wire registration in *without* threading the registry through every
# nested helper function. Workers call
# ``with active_rollout_cli_context(registry, rollout_id): ...`` around
# the rollout body.
_ACTIVE_ROLLOUT_CLI_CONTEXT: contextvars.ContextVar[Optional[tuple["RolloutCLIRegistry", Any]]] = (
    contextvars.ContextVar("_ACTIVE_ROLLOUT_CLI_CONTEXT", default=None)
)


@contextlib.contextmanager
def active_rollout_cli_context(
    registry: "RolloutCLIRegistry",
    rollout_id: Any,
) -> Iterator[None]:
    """Bind ``(registry, rollout_id)`` to the current task/thread context.

    Any ``CLIModelClient`` constructed inside this block that doesn't
    receive an explicit registry will pick up these values automatically.
    Safe to nest; the inner block shadows the outer for its lifetime.
    """
    token = _ACTIVE_ROLLOUT_CLI_CONTEXT.set((registry, rollout_id))
    try:
        yield
    finally:
        _ACTIVE_ROLLOUT_CLI_CONTEXT.reset(token)


# T1.4 — per-rollout CLI step hard-timeout EXTENSION multiplier. The engine
# size-scales the per-rollout wall-clock budget for huge suites; this contextvar
# lets it mirror that scaling onto each CLI agent step's HARD timeout so the
# agent session isn't killed mid-step on a giant repo. It is EXTEND-ONLY (factor
# >= 1) and the existing 30-minute floor + outer per-task wallclock budget remain
# the termination guarantees. Defaults to 1.0 => identical behavior for small
# repos (where the engine never sets a factor > 1).
_ACTIVE_CLI_HARD_TIMEOUT_SIZE_FACTOR: contextvars.ContextVar[float] = contextvars.ContextVar(
    "_ACTIVE_CLI_HARD_TIMEOUT_SIZE_FACTOR", default=1.0
)


@contextlib.contextmanager
def active_cli_hard_timeout_size_factor(size_factor: float) -> Iterator[None]:
    """Bind a CLI step hard-timeout extension multiplier to this context.

    Values <= 1 are clamped to 1.0 (extend-only). Safe to nest.
    """
    try:
        factor = max(1.0, float(size_factor))
    except (TypeError, ValueError):
        factor = 1.0
    token = _ACTIVE_CLI_HARD_TIMEOUT_SIZE_FACTOR.set(factor)
    try:
        yield
    finally:
        _ACTIVE_CLI_HARD_TIMEOUT_SIZE_FACTOR.reset(token)


def _resolve_active_cli_hard_timeout_size_factor() -> float:
    try:
        return max(1.0, float(_ACTIVE_CLI_HARD_TIMEOUT_SIZE_FACTOR.get()))
    except (TypeError, ValueError):
        return 1.0


def _claude_bash_timeout_env_overrides() -> dict[str, str]:
    """Size-aware Bash (per-command) timeout env for the claude CLI backend.

    Returns ``BASH_DEFAULT_TIMEOUT_MS`` (modest, matches codex's native floor so
    unspecified commands self-bound) and ``BASH_MAX_TIMEOUT_MS`` (generous,
    size-scaled headroom). The MAX is derived from the SAME ``expected_test_count``
    size signal already bound by the engine via ``active_cli_hard_timeout_size_factor``
    (which mirrors ``_rollout_budget_size_factor``): giants get the largest budget,
    small repos keep the baseline. Monotone non-decreasing in suite size, clamped
    to a hard ceiling so even the biggest suite stays well under any rollout cap.

    Operator override: ``APEX_CLAUDE_BASH_MAX_TIMEOUT_MS`` /
    ``APEX_CLAUDE_BASH_DEFAULT_TIMEOUT_MS`` (positive int milliseconds) replace the
    computed values. This is a PER-COMMAND ceiling only — it returns a single hung
    command to the agent loop and can never cap ``final_pass_rate`` below 1.0.
    """

    def _positive_int_env(name: str) -> Optional[int]:
        raw = str(os.environ.get(name) or "").strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    default_ms = _positive_int_env("APEX_CLAUDE_BASH_DEFAULT_TIMEOUT_MS")
    if default_ms is None:
        default_ms = _DEFAULT_CLAUDE_BASH_DEFAULT_TIMEOUT_MS

    max_ms = _positive_int_env("APEX_CLAUDE_BASH_MAX_TIMEOUT_MS")
    if max_ms is None:
        size_factor = _resolve_active_cli_hard_timeout_size_factor()
        scaled = int(round(_BASE_CLAUDE_BASH_MAX_TIMEOUT_MS * max(1.0, size_factor)))
        max_ms = min(scaled, _CLAUDE_BASH_MAX_TIMEOUT_CEILING_MS)
    # The MAX must never be below the DEFAULT (Claude requires MAX >= DEFAULT;
    # an operator override of only DEFAULT must not invert the relationship).
    max_ms = max(max_ms, default_ms)
    return {
        "BASH_DEFAULT_TIMEOUT_MS": str(int(default_ms)),
        "BASH_MAX_TIMEOUT_MS": str(int(max_ms)),
    }


def _resolve_active_rollout_cli_context() -> Optional[tuple["RolloutCLIRegistry", Any]]:
    """Return the currently-bound (registry, rollout_id) or ``None``."""
    return _ACTIVE_ROLLOUT_CLI_CONTEXT.get()


class RolloutCLIRegistry:
    """Per-rollout CLI subprocess registry.

    Each ``RolloutEngine`` owns one instance. Workers register their
    ``subprocess.Popen`` against a rollout id; on deadline expiry the
    engine calls ``terminate_for_rollout(rollout_id)`` to kill ONLY that
    rollout's CLI children. Sibling rollouts in the same engine remain
    untouched.

    Threadsafety: a single internal ``threading.Lock`` guards the
    rollout->processes mapping. Termination snapshots the PID list under
    the lock then issues signals outside it, so a long-running ``ps``
    or ``killpg`` cannot block sibling registrations.
    """

    __slots__ = (
        "_lock",
        "_processes",
        "_metadata_by_rollout_pid",
        "_install_cleanup_hooks",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # rollout_id (str) -> dict[pid -> Popen]. We key Popen by pid so
        # the same process re-registered twice doesn't cause duplicates,
        # and so ``get_for_rollout`` can return a stable list. We do NOT
        # store the Popen reference longer than necessary because the
        # caller's ``finally`` clause is responsible for unregistering.
        self._processes: dict[str, dict[int, subprocess.Popen[Any]]] = {}
        self._metadata_by_rollout_pid: dict[str, dict[int, dict[str, Any]]] = {}
        # Hook installation is opt-in via register(); zero overhead for
        # tests that never spawn real subprocesses.
        self._install_cleanup_hooks: bool = True

    @staticmethod
    def _normalize_rollout_id(rollout_id: Any) -> str:
        """Coerce a rollout id to a stable string key.

        Rollout ids in apex flow as ``int`` everywhere, but new code (the
        Phase 2 selector / orchestrator scoping) sometimes carries a
        composite ``"{run_id}:{rollout_id}"`` key. Coerce to str so both
        callers route to the same bucket.
        """
        return str(rollout_id) if rollout_id is not None else ""

    def register(
        self,
        rollout_id: Any,
        process: "subprocess.Popen[Any]",
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Register a live CLI subprocess against a rollout id.

        Caller MUST also call the global ``_register_active_cli_process``
        path (already wired in ``CLIModelClient``) so atexit / signal
        handlers can still find the process if the parent crashes before
        the rollout finishes. This registry is additive — it scopes
        teardown, it does not replace the global drain.
        """
        if process is None:
            return
        pid = int(getattr(process, "pid", 0) or 0)
        if pid <= 0 or pid == os.getpid():
            return
        key = self._normalize_rollout_id(rollout_id)
        if not key:
            return
        if self._install_cleanup_hooks:
            _install_cli_process_cleanup_hooks()
        with self._lock:
            self._processes.setdefault(key, {})[pid] = process
            if metadata:
                self._metadata_by_rollout_pid.setdefault(key, {})[pid] = dict(metadata)

    def unregister(
        self,
        rollout_id: Any,
        process: "subprocess.Popen[Any]",
    ) -> None:
        """Drop a process from the registry once it has exited."""
        if process is None:
            return
        pid = int(getattr(process, "pid", 0) or 0)
        if pid <= 0:
            return
        key = self._normalize_rollout_id(rollout_id)
        with self._lock:
            bucket = self._processes.get(key)
            if not bucket:
                return
            bucket.pop(pid, None)
            if not bucket:
                self._processes.pop(key, None)
            metadata_bucket = self._metadata_by_rollout_pid.get(key)
            if metadata_bucket is not None:
                metadata_bucket.pop(pid, None)
                if not metadata_bucket:
                    self._metadata_by_rollout_pid.pop(key, None)

    def get_for_rollout(self, rollout_id: Any) -> list["subprocess.Popen[Any]"]:
        """Return live ``Popen`` objects registered for one rollout."""
        key = self._normalize_rollout_id(rollout_id)
        with self._lock:
            bucket = self._processes.get(key, {})
            return list(bucket.values())

    # ------------------------------------------------------------------
    # S7 in-flight-LLM-request marker (progress-based liveness).
    # ------------------------------------------------------------------
    # The marker lives in the lock-guarded per-(rollout, pid) metadata so the
    # killer reads it under the same lock that guards process registration.
    # ``mark_inflight_request`` records the dispatch time; the watchdog treats
    # a *running* process whose marker is set (and not older than
    # ``max_inflight_request_seconds`` of total silence) as ALIVE so a
    # multi-minute LLM "thinking" turn is not killed as a stall. The marker is
    # cleared in the dispatch ``finally``; a crashed worker that leaves it set
    # is bounded by the watchdog's ceiling.
    def mark_inflight_request(
        self,
        rollout_id: Any,
        pid: int,
        *,
        started_at: Optional[float] = None,
    ) -> None:
        key = self._normalize_rollout_id(rollout_id)
        if not key:
            return
        try:
            pid_int = int(pid)
        except (TypeError, ValueError):
            return
        if pid_int <= 0:
            return
        ts = float(started_at) if started_at is not None else time.time()
        with self._lock:
            self._metadata_by_rollout_pid.setdefault(key, {}).setdefault(pid_int, {})[
                "inflight_request_started_at"
            ] = ts

    def clear_inflight_request(self, rollout_id: Any, pid: int) -> None:
        key = self._normalize_rollout_id(rollout_id)
        if not key:
            return
        try:
            pid_int = int(pid)
        except (TypeError, ValueError):
            return
        with self._lock:
            metadata_bucket = self._metadata_by_rollout_pid.get(key)
            if not metadata_bucket:
                return
            pid_meta = metadata_bucket.get(pid_int)
            if isinstance(pid_meta, dict):
                pid_meta.pop("inflight_request_started_at", None)

    def inflight_request_started_at(self, rollout_id: Any, pid: int) -> Optional[float]:
        """Return the dispatch timestamp of an in-flight request, else None."""
        key = self._normalize_rollout_id(rollout_id)
        if not key:
            return None
        try:
            pid_int = int(pid)
        except (TypeError, ValueError):
            return None
        with self._lock:
            metadata_bucket = self._metadata_by_rollout_pid.get(key)
            if not metadata_bucket:
                return None
            pid_meta = metadata_bucket.get(pid_int)
            if not isinstance(pid_meta, dict):
                return None
            value = pid_meta.get("inflight_request_started_at")
            return float(value) if isinstance(value, (int, float)) else None

    def latest_inflight_request_started_at(self, rollout_id: Any) -> Optional[float]:
        """Most-recent in-flight-request dispatch timestamp across a rollout's pids.

        Returns the MAX ``inflight_request_started_at`` over every live pid bucket
        for ``rollout_id`` (or ``None`` when no marker is set). The MAX is the
        liveness-correct choice: an in-flight request started more recently means
        the freeze should still hold. Used by the scheduler stall check so the
        OUTER reaper honours the SAME S7 freeze the inner CLI watchdog applies,
        instead of out-racing it for a deliberately-frozen (long LLM turn) rollout.
        """
        key = self._normalize_rollout_id(rollout_id)
        if not key:
            return None
        with self._lock:
            metadata_bucket = self._metadata_by_rollout_pid.get(key)
            if not metadata_bucket:
                return None
            timestamps = [
                float(pid_meta["inflight_request_started_at"])
                for pid_meta in metadata_bucket.values()
                if isinstance(pid_meta, dict)
                and isinstance(pid_meta.get("inflight_request_started_at"), (int, float))
            ]
        return max(timestamps) if timestamps else None

    def process_task_id(self, rollout_id: Any) -> str:
        """Return the subprocess-registry key for non-LLM rollout commands."""

        key = self._normalize_rollout_id(rollout_id)
        if not key:
            return ""
        return f"rollout:{id(self)}:{key}"

    def terminate_for_rollout(
        self,
        rollout_id: Any,
        signum: int = signal.SIGTERM,
    ) -> int:
        """Signal every CLI child for ``rollout_id``; return PID count.

        Sibling rollouts in this registry are untouched. Returns 0 when
        the rollout had no registered processes (e.g. it never spawned a
        CLI backend, or the agentic loop was synchronous).
        """
        key = self._normalize_rollout_id(rollout_id)
        with self._lock:
            bucket = self._processes.get(key, {})
            pids: set[int] = set(bucket.keys())
            metadata_by_pid = dict(self._metadata_by_rollout_pid.get(key, {}))
        if not pids:
            return 0
        for metadata in metadata_by_pid.values():
            _cleanup_target_runtime_from_metadata(metadata, signum=int(signum))
        # Walk descendants too; an agent-loop child often spawns its own
        # subprocess tree (pytest, docker exec) that won't share the PID
        # we registered.
        tracked: set[int] = set()
        for pid in pids:
            tracked.update(_collect_subprocess_tree_pids(pid) or {pid})
        try:
            _signal_subprocess_tree(tracked, signal.Signals(int(signum)))
        except ValueError:
            _signal_subprocess_tree(tracked, signal.SIGTERM)
        return len(pids)

    def active_rollout_ids(self) -> list[str]:
        """Snapshot of rollout ids currently holding registered processes."""
        with self._lock:
            return [key for key, bucket in self._processes.items() if bucket]

    def clear(self) -> None:
        """Drop all entries WITHOUT signalling any processes.

        Used by the engine in ``cleanup`` once it has independently torn
        down rollouts via ``terminate_for_rollout``. Calling this without
        a prior terminate leaks the PIDs from this scoped view, but the
        global ``_ACTIVE_CLI_PROCESS_PIDS`` drain still catches them.
        """
        with self._lock:
            self._processes.clear()


def extract_total_tokens(usage: dict[str, Any]) -> int:
    """Best-effort token extraction across backend-specific usage payloads."""

    if not usage:
        return 0

    if "total_tokens" in usage and isinstance(usage["total_tokens"], int):
        return usage["total_tokens"]

    tokens = usage.get("tokens")
    if isinstance(tokens, dict):
        total = tokens.get("total")
        if isinstance(total, int):
            return total

    models = usage.get("models")
    if isinstance(models, dict):
        model_totals = [extract_total_tokens(model_usage) for model_usage in models.values()]
        if any(model_totals):
            return sum(model_totals)

    direct_token_values = [
        value for key, value in usage.items() if isinstance(value, int) and "token" in key.lower()
    ]
    if direct_token_values:
        return sum(direct_token_values)

    total = 0
    for value in usage.values():
        if isinstance(value, dict):
            total += extract_total_tokens(value)
        elif isinstance(value, list):
            total += sum(extract_total_tokens(item) for item in value if isinstance(item, dict))
    return total


def _default_cli_env_overrides(config: LLMConfig) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if config.backend == LLMBackend.CLAUDE_CLI:
        # Use the installed native launcher by default. Forcing
        # CLAUDE_CODE_VERSION_OVERRIDE="latest" makes every launch resolve the
        # newest build via Manifold; that lookup is unreachable in air-gapped /
        # host agent launches and fails CLI startup with "Manifold native
        # version check failed", producing zero-edit rollouts. The installed
        # binary already runs Opus 4.8 (model selection is via --model, not the
        # launcher version), so latest-resolution is opt-in via
        # APEX_CLAUDE_CODE_VERSION_OVERRIDE for online environments where
        # Manifold is reachable. The air-gapped warm-prep below still pops any
        # configured override when "latest" cannot be resolved.
        override = str(os.environ.get("APEX_CLAUDE_CODE_VERSION_OVERRIDE") or "").strip()
        if override:
            overrides["CLAUDE_CODE_VERSION_OVERRIDE"] = override
    return overrides


def _prepare_cli_command_for_target_tool_path(
    command: list[str],
    env: dict[str, str],
) -> list[str]:
    """Launch the CLI itself with host interpreters when PATH has target shims."""

    if not command or not env.get("APEX_TARGET_TOOL_CONTEXT"):
        return command
    executable = command[0]
    resolved: str | None
    if os.path.isabs(executable):
        resolved = executable
    else:
        resolved = shutil.which(executable, path=os.environ.get("PATH", ""))
    if not resolved:
        return command
    prepared = [resolved, *command[1:]]
    try:
        with open(resolved, "rb") as fh:
            first_line = fh.readline(512).decode("utf-8", errors="ignore").strip()
    except OSError:
        return prepared
    if not first_line.startswith("#!"):
        return prepared
    try:
        shebang_parts = shlex.split(first_line[2:].strip())
    except ValueError:
        return prepared
    if not shebang_parts:
        return prepared
    interpreter = shebang_parts[0]
    interpreter_args = shebang_parts[1:]
    if Path(interpreter).name == "env":
        env_args = interpreter_args
        if env_args[:1] == ["-S"]:
            try:
                env_args = shlex.split(" ".join(env_args[1:]))
            except ValueError:
                return prepared
        interpreter_index = next(
            (idx for idx, item in enumerate(env_args) if not item.startswith("-")),
            None,
        )
        if interpreter_index is None:
            return prepared
        interpreter = env_args[interpreter_index]
        interpreter_args = env_args[interpreter_index + 1 :]
    interpreter_path: str | None
    if os.path.isabs(interpreter):
        interpreter_path = interpreter
    else:
        interpreter_path = shutil.which(interpreter, path=os.environ.get("PATH", ""))
    if not interpreter_path:
        return prepared
    return [interpreter_path, *interpreter_args, resolved, *command[1:]]


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_flag_not_disabled(name: str) -> bool:
    return str(os.environ.get(name) or "").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _codex_tool_output_token_limit() -> int:
    raw_value = os.environ.get("APEX_CODEX_TOOL_OUTPUT_TOKEN_LIMIT")
    if raw_value is None or not str(raw_value).strip():
        return _DEFAULT_CODEX_TOOL_OUTPUT_TOKEN_LIMIT
    try:
        return max(0, int(str(raw_value).strip()))
    except ValueError:
        return _DEFAULT_CODEX_TOOL_OUTPUT_TOKEN_LIMIT


def _target_runtime_claude_json_output_enabled(*, force_json: bool = False) -> bool:
    return not _env_flag_enabled("APEX_TARGET_RUNTIME_CLAUDE_STREAM_JSON") and (
        force_json or _env_flag_not_disabled("APEX_TARGET_RUNTIME_CLAUDE_JSON_OUTPUT")
    )


def _coerce_subprocess_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _capture_cli_command_to_file(
    *,
    command: list[str],
    env: dict[str, str] | None,
    working_dir: str | None,
    backend: str,
    attempt_index: int,
) -> None:
    """When ``APEX_CAPTURE_CLI_COMMAND_TO_FILE`` is enabled, write the
    unredacted CLI command (and the env-var KEYS only — never values) to
    a debug file in ``working_dir``. Best-effort: any exception is
    swallowed so capture never blocks a real CLI launch.

    Resolves the gap where prior failed runs only logged
    ``cli_env_redacted`` and the actual command was unrecoverable for
    triage. Used by every backend, every benchmark, every agentic call.
    """

    try:
        target_dir = Path(working_dir).expanduser().resolve() if working_dir else Path.cwd()
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        # Use a per-call timestamped name so multiple captures in the
        # same workdir don't clobber each other.
        out_path = target_dir / f".apex_cli_command_{backend}_{stamp}_{attempt_index}.txt"
        env_keys = sorted((env or os.environ).keys())
        payload_lines = [
            f"# backend: {backend}",
            f"# attempt: {attempt_index}",
            f"# captured_at_utc: {stamp}",
            f"# working_dir: {target_dir}",
            "# env keys (values intentionally redacted):",
            *(f"#   {key}" for key in env_keys),
            "",
            "# command (argv joined with shell-safe quoting):",
            shlex.join(str(arg) for arg in command),
        ]
        out_path.write_text("\n".join(payload_lines) + "\n", encoding="utf-8")
    except Exception:  # pragma: no cover - capture must never block the launch
        return


_CLI_RETRY_TOKEN_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\bsk-[A-Za-z0-9][A-Za-z0-9_-]{7,}\b"), "[REDACTED_TOKEN]"),
    (
        re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{8,}\b"),
        r"\1[REDACTED_TOKEN]",
    ),
    (
        re.compile(
            r"(?i)\b((?:api[_-]?key|token|password|passwd|secret|credential)"
            r"\s*[=:]\s*)[^\s\"']+"
        ),
        r"\1[REDACTED]",
    ),
)


def _truncate_cli_retry_text(text: str, limit: int = _CLI_RETRY_DIAGNOSTIC_EXCERPT_CHARS) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n...[truncated {omitted} chars]"


def _redact_cli_retry_text(
    value: Any,
    *,
    env: Optional[Mapping[str, str]] = None,
    limit: int = _CLI_RETRY_DIAGNOSTIC_EXCERPT_CHARS,
) -> str:
    text = _ANSI_ESCAPE_RE.sub("", _coerce_subprocess_text(value))
    for key, secret_value in (env or {}).items():
        if not env_key_matches_secret_denylist(str(key)):
            continue
        secret_text = str(secret_value or "")
        if len(secret_text) >= 6:
            text = text.replace(secret_text, f"[REDACTED:{key}]")
    for pattern, replacement in _CLI_RETRY_TOKEN_REDACTIONS:
        text = pattern.sub(replacement, text)
    return _truncate_cli_retry_text(text, limit)


def _compact_cli_retry_value(
    value: Any,
    *,
    env: Optional[Mapping[str, str]] = None,
    depth: int = 0,
) -> Any:
    if depth >= _CLI_RETRY_DIAGNOSTIC_MAX_DEPTH:
        return "[TRUNCATED_DEPTH]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _redact_cli_retry_text(value, env=env)
    if isinstance(value, bytes):
        return _redact_cli_retry_text(value, env=env)
    if isinstance(value, Mapping):
        compacted: dict[str, Any] = {}
        items = list(value.items())
        for key, item_value in items[:_CLI_RETRY_DIAGNOSTIC_MAX_ITEMS]:
            compacted[str(key)] = _compact_cli_retry_value(
                item_value,
                env=env,
                depth=depth + 1,
            )
        if len(items) > _CLI_RETRY_DIAGNOSTIC_MAX_ITEMS:
            compacted["__truncated_items__"] = len(items) - _CLI_RETRY_DIAGNOSTIC_MAX_ITEMS
        return compacted
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        compacted_list = [
            _compact_cli_retry_value(item, env=env, depth=depth + 1)
            for item in items[:_CLI_RETRY_DIAGNOSTIC_MAX_ITEMS]
        ]
        if len(items) > _CLI_RETRY_DIAGNOSTIC_MAX_ITEMS:
            compacted_list.append(
                {"__truncated_items__": len(items) - _CLI_RETRY_DIAGNOSTIC_MAX_ITEMS}
            )
        return compacted_list
    return _redact_cli_retry_text(value, env=env)


def _cli_retry_diagnostics_dir(working_dir: str | Path) -> Path:
    return _agent_runtime_state_root_for_workspace(working_dir) / ".cli_retry_diagnostics"


def _write_cli_retry_diagnostic(
    *,
    working_dir: str,
    backend: str,
    model: str,
    retry_kind: str,
    retry_reason: str,
    attempt_index: int,
    max_attempts: int,
    process_pid: int,
    returncode: Any,
    command: Optional[list[str]],
    env: Optional[Mapping[str, str]],
    stdout: str,
    stderr: str,
    raw_output: str,
    result: CLIModelResult,
    timeout_audit: Optional[Mapping[str, Any]],
) -> Optional[str]:
    """Persist bounded retry evidence outside the candidate workspace."""

    try:
        command_args = [str(arg) for arg in (command or [])]
        command_hash = hashlib.sha256(
            json.dumps(command_args, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        env_keys = sorted(str(key) for key in (env or {}).keys())
        env_key_hash = hashlib.sha256(
            json.dumps(env_keys, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        payload = {
            "event": "apex.cli.retry_diagnostic",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "backend": backend,
            "model": model,
            "retry_kind": retry_kind,
            "retry_reason": retry_reason,
            "attempt_index": attempt_index,
            "max_attempts": max_attempts,
            "process_pid": process_pid,
            "returncode": returncode,
            "working_dir": working_dir,
            "command_fingerprint": {
                "sha256": command_hash,
                "argv0": Path(command_args[0]).name if command_args else "",
                "arg_count": len(command_args),
            },
            "env_key_fingerprint": {
                "sha256": env_key_hash,
                "key_count": len(env_keys),
                "secret_key_count": sum(
                    1 for key in env_keys if env_key_matches_secret_denylist(key)
                ),
            },
            "stdout_excerpt": _redact_cli_retry_text(stdout, env=env),
            "stderr_excerpt": _redact_cli_retry_text(stderr, env=env),
            "raw_output_excerpt": _redact_cli_retry_text(raw_output, env=env),
            "result": {
                "success": bool(result.success),
                "parsed_json_present": isinstance(result.parsed_json, dict),
                "error_excerpt": _redact_cli_retry_text(result.error or "", env=env),
                "text_excerpt": _redact_cli_retry_text(result.text or "", env=env),
                "response_status": result.response_status,
                "workspace_status": result.workspace_status,
                "patch_extraction_status": result.patch_extraction_status,
                "finalization_status": result.finalization_status,
                "telemetry_status": result.telemetry_status,
            },
            "timeout_audit": _compact_cli_retry_value(timeout_audit or {}, env=env),
        }
        out_dir = _cli_retry_diagnostics_dir(working_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_kind = re.sub(r"[^A-Za-z0-9_.-]+", "_", retry_kind).strip("._-") or "retry"
        safe_backend = re.sub(r"[^A-Za-z0-9_.-]+", "_", backend).strip("._-") or "backend"
        out_path = out_dir / (
            f"{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_"
            f"{safe_backend}_{safe_kind}_attempt{attempt_index}_"
            f"pid{process_pid}_{secrets.token_hex(4)}.json"
        )
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            out_path.chmod(0o600)
        except OSError:
            pass
        return str(out_path)
    except Exception:  # pragma: no cover - diagnostics must never block retries
        logger.debug("Failed to persist CLI retry diagnostic", exc_info=True)
        return None


def _path_has_claude_internet_mode_marker(path: str | Path) -> bool:
    try:
        current = Path(path).expanduser().resolve(strict=False)
    except OSError:
        current = Path(path).expanduser().absolute()
    for candidate in (current, *current.parents):
        if (candidate / ".claude" / _CLAUDE_INTERNET_MODE_MARKER).exists():
            return True
    return False


def _ambient_cli_state_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    xdg_cache_home = str(os.environ.get("XDG_CACHE_HOME") or "").strip()
    if xdg_cache_home:
        candidates.append(Path(xdg_cache_home).expanduser() / "apex")
    host_home = str(os.environ.get("HOME") or "").strip()
    if host_home:
        candidates.append(Path(host_home).expanduser() / ".cache" / "apex")
    candidates.append(Path(tempfile.gettempdir()) / "apex")
    return candidates


def _default_cli_state_root() -> Path:
    override = str(os.environ.get("APEX_CLI_STATE_HOME") or "").strip()
    if override:
        root = Path(override).expanduser()
    else:
        root = _ambient_cli_state_root_candidates()[0]
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cli_state_root_for_air_gapped_backend(backend: LLMBackend) -> Path:
    root = _default_cli_state_root()
    if backend != LLMBackend.CLAUDE_CLI or not _path_has_claude_internet_mode_marker(root):
        return root
    for candidate in _ambient_cli_state_root_candidates():
        if not _path_has_claude_internet_mode_marker(candidate):
            candidate.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Using detached Claude air-gapped CLI state root %s because configured root %s "
                "is under a Claude internet-mode marker.",
                candidate,
                root,
            )
            return candidate
    return root


def _persistent_air_gapped_cli_home(backend: LLMBackend) -> Path:
    home_path = (
        _cli_state_root_for_air_gapped_backend(backend)
        / "cli_airgapped_homes"
        / _PERSISTENT_AIR_GAPPED_HOME_LAYOUT_VERSION
        / str(backend.value)
    )
    for rel_path in (
        ".config",
        ".cache",
        ".local/share",
        ".local/state",
        "Library/Caches",
    ):
        (home_path / rel_path).mkdir(parents=True, exist_ok=True)
    return home_path


def _temporary_air_gapped_cli_home(
    backend: LLMBackend,
) -> tempfile.TemporaryDirectory[str]:
    """Create an isolated CLI home away from task TMPDIR.

    Benchmark runs may pin TMPDIR inside the repository so Docker and Colima
    can mount task workspaces. Codex refuses to bootstrap helper binaries from
    locations it classifies as temporary, so isolated CLI homes live under
    Apex's state cache instead of inheriting that task TMPDIR.
    """

    root = _default_cli_state_root() / "cli_airgapped_homes" / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(
        prefix=f"apex-cli-offline-{backend.value}-",
        dir=str(root),
    )


def _ensure_cli_bootstrap_helper_dir() -> Optional[Path]:
    """Expose only host helpers required to start provider CLIs."""

    bootstrap_dir = _default_cli_state_root() / "cli_bootstrap_bins" / "v1"
    wrote_any = False
    for helper_name in _CLI_BOOTSTRAP_HELPERS:
        resolved = shutil.which(helper_name, path=os.environ.get("PATH", ""))
        if not resolved:
            continue
        bootstrap_dir.mkdir(parents=True, exist_ok=True)
        wrapper_path = bootstrap_dir / helper_name
        wrapper_body = "#!/bin/sh\nexec " + shlex.quote(resolved) + ' "$@"\n'
        try:
            if (
                not wrapper_path.exists()
                or wrapper_path.read_text(encoding="utf-8") != wrapper_body
            ):
                wrapper_path.write_text(wrapper_body, encoding="utf-8")
                wrapper_path.chmod(0o755)
        except OSError:
            logger.debug(
                "Failed to prepare CLI bootstrap helper wrapper for %s",
                helper_name,
                exc_info=True,
            )
            continue
        wrote_any = True
    return bootstrap_dir if wrote_any else None


def _cli_launch_env_for_target_runtime(
    env: dict[str, str],
    command: list[str],
) -> dict[str, str]:
    """Return process env for starting the provider CLI under target shims."""

    if not command or not env.get("APEX_TARGET_TOOL_CONTEXT"):
        return env
    bootstrap_dir = _ensure_cli_bootstrap_helper_dir()
    if bootstrap_dir is None:
        return env
    bootstrap_text = str(bootstrap_dir)
    path_entries = [entry for entry in str(env.get("PATH") or "").split(os.pathsep) if entry]
    if bootstrap_text in path_entries:
        return env
    insert_at = 1 if path_entries else 0
    launch_env = dict(env)
    path_entries.insert(insert_at, bootstrap_text)
    launch_env["PATH"] = os.pathsep.join(path_entries)
    launch_env["APEX_CLI_BOOTSTRAP_PATH"] = bootstrap_text
    return launch_env


class CLIAgentContainerIsolationError(RuntimeError):
    """Raised when a declared target container cannot contain the CLI launch."""


_CLI_BACKEND_CONTAINER_ENV_KEYS: frozenset[str] = frozenset(
    key
    for spec in _CLI_BACKEND_SANDBOX_SPECS.values()
    for key in (*spec.container_env_keys, *spec.target_path_env_keys)
)
_AGENT_CONTAINER_ENV_ALLOWLIST: frozenset[str] = (
    frozenset(
        {
            "ALL_PROXY",
            "APEX_AGENT_MODEL_PROXY_ACTIVE",
            "APEX_AGENT_MODEL_PROXY_BACKEND",
            "APEX_CLI_INVOCATION_ID",
            "APEX_HOST_DYNAMIC_TOOLS",
            "APEX_TARGET_TOOL_CONTEXT",
            "APEX_TARGET_TOOL_WORKDIR",
            "CLAUDE_CODE_VERSION_OVERRIDE",
            "CODEX_HOME",
            "HOME",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "NO_PROXY",
            "NO_COLOR",
            "PATH",
            "PIP_DISABLE_PIP_VERSION_CHECK",
            "PYTHONDONTWRITEBYTECODE",
            "PYTHONNOUSERSITE",
            "TERM",
            "X2P_AGENT_PROXY_ADDRESS",
            "XDG_CACHE_HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "XDG_STATE_HOME",
            "all_proxy",
            "http_proxy",
            "https_proxy",
            "no_proxy",
        }
    )
    | _CLI_BACKEND_CONTAINER_ENV_KEYS
)
_AGENT_CONTAINER_ENV_PREFIX_ALLOWLIST: tuple[str, ...] = ("LC_",)
_AGENT_CONTAINER_PROXY_ENV_KEYS: frozenset[str] = frozenset(
    {
        "ALL_PROXY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "X2P_AGENT_PROXY_ADDRESS",
        "all_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
)
_CLI_BACKEND_CONTAINER_PATH_ENV_KEYS: frozenset[str] = frozenset(
    key for spec in _CLI_BACKEND_SANDBOX_SPECS.values() for key in spec.target_path_env_keys
)
_AGENT_CONTAINER_PATH_ENV_KEYS: frozenset[str] = (
    frozenset(
        {
            "APEX_TARGET_TOOL_CONTEXT",
            "APEX_TARGET_TOOL_WORKDIR",
            "CODEX_HOME",
            "HOME",
            "XDG_CACHE_HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "XDG_STATE_HOME",
        }
    )
    | _CLI_BACKEND_CONTAINER_PATH_ENV_KEYS
)
_NODE_AGENT_CONTAINER_CLI_BINARIES: frozenset[str] = frozenset(
    binary
    for spec in _CLI_BACKEND_SANDBOX_SPECS.values()
    if spec.probe_requires_node
    for binary in spec.binary_names
)


@dataclass(frozen=True)
class _AgentContainerLaunchContext:
    docker_bin: str
    container_name: str
    host_root: Path
    container_root: str
    working_dir_container: str
    runtime_env: dict[str, str]
    docker_host_env: dict[str, str]
    docker_user: str
    extra_host_path_mappings: tuple[tuple[Path, str], ...] = field(default_factory=tuple)


def _path_relative_to_or_none(path: Path, root: Path) -> Path | None:
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def _agent_container_host_path_mappings(
    context: _AgentContainerLaunchContext,
) -> tuple[tuple[Path, str], ...]:
    mappings: list[tuple[Path, str]] = [
        (context.host_root, context.container_root),
        *context.extra_host_path_mappings,
    ]
    normalized: list[tuple[Path, str]] = []
    for raw_host, raw_container in mappings:
        try:
            host = Path(raw_host).expanduser().resolve(strict=False)
        except OSError:
            host = Path(raw_host).expanduser().absolute()
        container = str(raw_container or "").rstrip("/")
        if not container.startswith("/"):
            continue
        normalized.append((host, container or "/"))
    return tuple(sorted(normalized, key=lambda item: len(str(item[0])), reverse=True))


def _map_host_path_to_agent_container(
    path: str | Path,
    context: _AgentContainerLaunchContext,
) -> str:
    raw_path = Path(path).expanduser()
    try:
        resolved = raw_path.resolve(strict=False)
    except OSError:
        resolved = raw_path.absolute()
    for host_root, container_root in _agent_container_host_path_mappings(context):
        relative = _path_relative_to_or_none(resolved, host_root)
        if relative is None:
            continue
        suffix = relative.as_posix()
        if suffix == ".":
            suffix = ""
        return container_root + (("/" + suffix) if suffix else "")
    raise CLIAgentContainerIsolationError(
        f"{resolved} is outside the declared target container mounts"
    )


def _map_host_root_text_to_agent_container(
    value: str,
    context: _AgentContainerLaunchContext,
) -> str:
    text = str(value)
    for host_root, container_root in _agent_container_host_path_mappings(context):
        text = text.replace(str(host_root), container_root)
    return text


def _agent_visible_file_uri(
    path: str | Path,
    *,
    env: dict[str, str],
    working_dir: str,
) -> str:
    raw_path = Path(path).expanduser()
    try:
        resolved = raw_path.resolve(strict=False)
    except OSError:
        resolved = raw_path.absolute()
    visible_path = str(resolved)
    if env.get("APEX_TARGET_TOOL_CONTEXT"):
        context = _agent_container_launch_context(env, working_dir=working_dir)
        if context is not None:
            visible_path = _map_host_path_to_agent_container(resolved, context)
    return Path(visible_path).as_uri()


def _write_opencode_family_isolated_config(
    config: LLMConfig,
    env: dict[str, str],
    *,
    working_dir: str,
    plugin_paths: Iterable[str | Path] = (),
) -> None:
    """Force OpenCode-family CLIs to load only APEX-authored runtime config."""

    if not _is_opencode_family_backend(config.backend):
        return
    home = Path(env.get("HOME") or "").expanduser()
    if not str(home):
        return
    config_dir = Path(env.get("OPENCODE_CONFIG_DIR") or (home / ".config" / "opencode"))
    config_dir.mkdir(parents=True, exist_ok=True)
    managed_empty_dir = config_dir / "apex-empty-managed-settings"
    managed_empty_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "opencode.json"
    env["OPENCODE_CONFIG_DIR"] = str(config_dir)
    env["OPENCODE_CONFIG"] = str(config_path)
    # MetaCode's package-managed defaults can load Meta MCP/plugins. Use an
    # explicit empty managed-settings directory so benchmark agents only see
    # APEX's isolated config and optional APEX review plugin.
    env["OPENCODE_TEST_MANAGED_CONFIG_DIR"] = str(managed_empty_dir)
    env.setdefault("OPENCODE_DISABLE_AUTOUPDATE", "1")
    env.setdefault("OPENCODE_DISABLE_PROJECT_CONFIG", "1")
    env.setdefault("METACODE_DISABLE_TRAJECTORY", "1")
    plugin_uris = [
        _agent_visible_file_uri(path, env=env, working_dir=working_dir)
        for path in plugin_paths
        if str(path or "").strip()
    ]
    payload: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "agent": {},
        "command": {},
        "experimental": {
            "done_marker": False,
            "reminder": False,
        },
        "instructions": [],
        "mcp": {},
        "mode": {},
        "plugin": plugin_uris,
        "snapshot": False,
    }
    resolved_model = config.resolved_cli_model
    if resolved_model:
        payload["model"] = resolved_model
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _proxy_value_for_agent_container(
    value: str,
    *,
    preserve_loopback: bool = False,
) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parse_value = text if "://" in text else f"http://{text}"
    try:
        parsed = urllib.parse.urlsplit(parse_value)
    except ValueError:
        return text
    hostname = parsed.hostname or ""
    if hostname not in {"localhost", "127.0.0.1", "::1"}:
        return text
    if preserve_loopback:
        return text
    netloc = "host.docker.internal"
    if parsed.port is not None:
        netloc += f":{parsed.port}"
    if parsed.username:
        auth = urllib.parse.quote(parsed.username, safe="")
        if parsed.password:
            auth += ":" + urllib.parse.quote(parsed.password, safe="")
        netloc = f"{auth}@{netloc}"
    rewritten = urllib.parse.urlunsplit(
        (parsed.scheme or "http", netloc, parsed.path, parsed.query, parsed.fragment)
    )
    return rewritten if "://" in text else rewritten.removeprefix("http://")


_MODEL_PROXY_DUMMY_SECRET = "apex-host-model-proxy"
_GLOBAL_MODEL_PROXY_ENV_KEYS: tuple[str, ...] = (
    "APEX_AGENT_MODEL_PROXY_URL",
    "APEX_HOST_MODEL_PROXY_URL",
)
_BACKEND_MODEL_PROXY_ENV_KEYS: dict[LLMBackend, tuple[str, ...]] = {
    LLMBackend.CLAUDE_CLI: (
        "APEX_CLAUDE_CLI_MODEL_PROXY_URL",
        "APEX_CLAUDE_MODEL_PROXY_URL",
    ),
    LLMBackend.CODEX_CLI: (
        "APEX_CODEX_CLI_MODEL_PROXY_URL",
        "APEX_CODEX_MODEL_PROXY_URL",
    ),
    LLMBackend.GEMINI_CLI: (
        "APEX_GEMINI_CLI_MODEL_PROXY_URL",
        "APEX_GEMINI_MODEL_PROXY_URL",
    ),
    LLMBackend.OPENCODE_CLI: (
        "APEX_OPENCODE_CLI_MODEL_PROXY_URL",
        "APEX_OPENCODE_MODEL_PROXY_URL",
    ),
    LLMBackend.METACODE_CLI: (
        "APEX_METACODE_CLI_MODEL_PROXY_URL",
        "APEX_METACODE_MODEL_PROXY_URL",
    ),
}
_MODEL_PROXY_MODE_VALUES = {
    "host_model_proxy",
    "model_proxy",
    "proxy",
    "credentialless_model_proxy",
}
# Host-CLI target-runtime auth mode. Runs the agentic CLI ON THE HOST so its
# native / compiled-in auth works (Meta Plugboard + x2p CAT injection for codex,
# the Vertex gateway for claude/gemini) — the only mode that authenticates a CLI
# whose credentials are NOT portable into a Linux container/sandbox. The agent's
# shell / file / test execution is still routed into the target Docker container
# via the existing target-tool shims, so repo tests run on the real Linux
# toolchain and the agent never executes against the host directly. Opt-in via
# ``APEX_TARGET_RUNTIME_CLI_AUTH_MODE``; default behaviour is unchanged.
_HOST_CLI_AUTH_MODE_VALUES = {
    "host_cli",
    "host_auth_cli",
    "host_cli_container_tools",
    "host_auth_container_tools",
}
# Host transport env that MUST survive host-secret redaction so the host CLI can
# authenticate through the SANCTIONED host proxy/gateway (never public provider
# endpoints). Only preserved in host_cli mode; every other host secret is still
# stripped, so unrelated credentials never reach the agent subprocess (the
# no-data-leakage contract). These are session-transport vars, not third-party
# API keys.
_HOST_CLI_TRANSPORT_ALLOW_KEYS: tuple[str, ...] = (
    "X2P_AGENT_PROXY_ADDRESS",
    "X2P_INJECT_CAT",
    "X2P_SUPPORTS_VPNLESS",
    "CPE_RUST_X2P_SUPPORTS_VPNLESS",
    "DOTSLASH_X2P_EDGETERM",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "NO_PROXY",
    "no_proxy",
    "ALL_PROXY",
    "CODEX_INTERNAL_ORIGINATOR_OVERRIDE",
    "ANTHROPIC_VERTEX_BASE_URL",
    "ANTHROPIC_VERTEX_PROJECT_ID",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_SKIP_VERTEX_AUTH",
    "CLOUD_ML_REGION",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_PROJECT_ID",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_CLOUD_QUOTA_PROJECT",
    "GOOGLE_GENAI_USE_VERTEXAI",
    "GOOGLE_GENAI_USE_GCA",
    "GOOGLE_CLOUD_ACCESS_TOKEN",
    "GOOGLE_GEMINI_BASE_URL",
    "GOOGLE_VERTEX_BASE_URL",
    "CODE_ASSIST_ENDPOINT",
    "CODE_ASSIST_API_VERSION",
    "GEMINI_CUSTOM_HEADERS",
    "GEMINI_CLI_CUSTOM_HEADERS",
    "META_3PAI_AGENT_PLATFORM",
    "META_3PAI_INVOCATION_CONTEXT",
    "META_3PAI_INVOCATION_ID",
    "META_3PAI_ORIGINATING_AGENT_PLATFORM",
    "META_3PAI_SESSION_TRACKING_FILE",
    "META_CLAUDE_AI_GATEWAY_TAILER_SCRIBE_LOGGING",
    "META_CLAUDE_ENABLE_WEB_TOOLS",
    "META_CLAUDE_TOOL_GOVERNANCE",
    "META_CLAUDE_USE_ANTHROPIC_DIRECT",
)
_DOCKER_SANDBOX_AUTH_MODE_VALUES = {
    "docker_sandbox",
    "docker_sandboxes",
    "docker_sandbox_host_auth",
    "host_auth_docker_sandbox",
    "host_docker_sandbox",
}
_DOCKER_SANDBOX_AGENT_BY_BACKEND: dict[LLMBackend, str] = {
    LLMBackend.CLAUDE_CLI: "claude",
    LLMBackend.GEMINI_CLI: "gemini",
    LLMBackend.CODEX_CLI: "codex",
}
_DOCKER_SANDBOX_SYSTEM_PATH = (
    "/usr/local/share/npm-global/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:"
    "/home/agent/.local/bin"
)
_CLAUDE_CLI_EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh", "max"})
_CLAUDE_CLI_DEFAULT_EFFORT = "max"
_CLAUDE_TARGET_RUNTIME_EMPTY_MCP_CONFIG_JSON = '{"mcpServers":{}}'
_CLAUDE_TARGET_RUNTIME_WARMUP_SENTINEL = ".apex_target_runtime_warmup_ok"
_CLAUDE_TARGET_RUNTIME_WARMUP_DEFAULT_TIMEOUT_SECONDS = 15.0
_CLAUDE_TARGET_RUNTIME_WARMUP_FAILURE_COOLDOWN_SECONDS = 300.0
_CLAUDE_TARGET_RUNTIME_WARMUP_COOLDOWN_LOCK = threading.Lock()
_CLAUDE_TARGET_RUNTIME_WARMUP_COOLDOWNS: dict[str, float] = {}
_CLAUDE_TARGET_RUNTIME_RETRY_STATE_PRESERVE_NAMES = frozenset(
    {
        ".credentials.json",
        "credentials",
        _CLAUDE_TARGET_RUNTIME_WARMUP_SENTINEL,
    }
)
_BACKEND_MODEL_PROXY_CONTAINER_CREDENTIAL_ENV_KEYS: dict[LLMBackend, tuple[str, ...]] = {
    LLMBackend.CLAUDE_CLI: (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_CUSTOM_HEADERS",
        "ANTHROPIC_VERTEX_BASE_URL",
        "ANTHROPIC_VERTEX_PROJECT_ID",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_SKIP_VERTEX_AUTH",
        "CLAUDE_CODE_USE_VERTEX",
        "CLOUD_ML_REGION",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ),
    LLMBackend.CODEX_CLI: ("OPENAI_API_KEY",),
    LLMBackend.GEMINI_CLI: (
        "GEMINI_API_KEY",
        "GEMINI_CUSTOM_HEADERS",
        "GEMINI_CLI_CUSTOM_HEADERS",
        "GOOGLE_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_ACCESS_TOKEN",
        "GOOGLE_GEMINI_BASE_URL",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_VERTEX_BASE_URL",
    ),
    LLMBackend.OPENCODE_CLI: (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "XAI_API_KEY",
    ),
    LLMBackend.METACODE_CLI: (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "XAI_API_KEY",
    ),
}


def _configured_cli_model_proxy_url(config: LLMConfig, env: dict[str, str]) -> str:
    keys = (
        *_BACKEND_MODEL_PROXY_ENV_KEYS.get(config.backend, ()),
        *_GLOBAL_MODEL_PROXY_ENV_KEYS,
    )
    for key in keys:
        value = str(env.get(key) or os.environ.get(key) or "").strip()
        if value:
            return value
    return str(getattr(config, "base_url", None) or "").strip()


def _cli_model_proxy_configured(config: LLMConfig, env: dict[str, str]) -> bool:
    return bool(_configured_cli_model_proxy_url(config, env))


def _target_runtime_requires_model_proxy(env: dict[str, str]) -> bool:
    if str(env.get("APEX_TARGET_RUNTIME_REQUIRE_MODEL_PROXY") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True
    mode = str(env.get("APEX_TARGET_RUNTIME_CLI_AUTH_MODE") or "").strip().lower()
    return mode in _MODEL_PROXY_MODE_VALUES


def _target_runtime_cli_auth_mode(env: dict[str, str]) -> str:
    return str(env.get("APEX_TARGET_RUNTIME_CLI_AUTH_MODE") or "").strip().lower()


def _target_runtime_uses_docker_sandbox_cli(
    config: LLMConfig,
    env: dict[str, str],
) -> bool:
    if not env.get("APEX_TARGET_TOOL_CONTEXT"):
        return False
    if _target_runtime_kind(env) not in {"docker_exec", "docker_image"}:
        return False
    if _target_runtime_cli_auth_mode(env) not in _DOCKER_SANDBOX_AUTH_MODE_VALUES:
        return False
    return config.backend in _DOCKER_SANDBOX_AGENT_BY_BACKEND


def _target_runtime_launches_agent_cli_in_docker_image(
    config: LLMConfig,
    env: dict[str, str],
) -> bool:
    if not env.get("APEX_TARGET_TOOL_CONTEXT"):
        return False
    if _target_runtime_uses_docker_sandbox_cli(config, env):
        return False
    if _target_runtime_uses_host_cli(config, env):
        return False
    return _target_runtime_kind(env) == "docker_image"


def _host_cli_auth_mode_requested(env: Optional[dict[str, str]]) -> bool:
    """True when an env mapping selects the host-CLI auth mode."""
    if not isinstance(env, dict):
        return False
    return str(env.get("APEX_TARGET_RUNTIME_CLI_AUTH_MODE") or "").strip().lower() in (
        _HOST_CLI_AUTH_MODE_VALUES
    )


def _target_runtime_uses_host_cli(
    config: LLMConfig,
    env: dict[str, str],
) -> bool:
    """Run the agent CLI ON THE HOST (native/compiled-in auth) while routing
    tool/shell/test execution into the target Docker container via the
    target-tool shims. Opt-in via APEX_TARGET_RUNTIME_CLI_AUTH_MODE; default off.

    Unlike docker_sandbox / agent_cli_in_container this keeps the CLI process on
    the host so credentials that are not portable into a Linux runtime (e.g. the
    Meta codex cryptex binary's Plugboard/x2p auth) keep working.
    """
    if not env.get("APEX_TARGET_TOOL_CONTEXT"):
        return False
    if _target_runtime_kind(env) != "docker_exec":
        return False
    return _target_runtime_cli_auth_mode(env) in _HOST_CLI_AUTH_MODE_VALUES


# --- host_cli read-jail (macOS Seatbelt) -----------------------------------
# A host-running agent CLI can read the whole host filesystem (codex/gemini/claude
# all default to global reads). To honour the no-data-leakage contract we wrap the
# host launch in an Apple Seatbelt jail that BLOCKS reads of sensitive user data
# outside the workspace while leaving system/toolchain reads intact (a strict
# deny-default profile prevents the Meta codex Plugboard launcher from booting, so
# we use allow-default + deny the high-value targets — credentials, other repos,
# personal documents, other users, mounted volumes). Reads matter more than
# writes; we also deny writes to those same areas.
# Minimal, CLI-safe deny set: concrete high-value credential stores + personal
# data. Deliberately excludes paths the Meta launcher / git / keychain need to
# boot (e.g. ~/.gitconfig, ~/Library/Keychains) — denying those makes codex fail
# with 'Operation not permitted' before it can run. Validated to keep codex
# bootable while blocking the real leak vectors.
_HOST_CLI_READ_JAIL_DENY_HOME_SUBPATHS: tuple[str, ...] = (
    ".ssh",
    ".aws",
    ".gnupg",
    ".azure",
    ".config/gcloud",
    ".config/gh",
    ".git-credentials",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".password-store",
    "Documents",
    "Desktop",
    "Downloads",
)
_HOST_CLI_READ_JAIL_DENY_ABS_SUBPATHS: tuple[str, ...] = (
    "/Volumes",
    "/Users/Shared",
)


def _host_cli_read_jail_enabled(env: Optional[dict[str, str]]) -> bool:
    """host_cli read-jail is ON unless explicitly disabled. Only meaningful on
    macOS (sandbox-exec) and in host_cli mode."""
    if sys.platform != "darwin":
        return False
    if not _host_cli_auth_mode_requested(env):
        return False
    value = str((env or {}).get("APEX_HOST_CLI_READ_JAIL") or "").strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    return shutil.which("sandbox-exec") is not None


def _host_cli_read_jail_profile_text(
    *, workspace: str, home: str, extra_repo_root: str = ""
) -> str:
    """Return an allow-default Seatbelt profile that denies reads+writes of the
    sensitive user-data areas outside ``workspace`` (deny-list model — keeps the
    self-Seatbelting Meta CLIs bootable, unlike a strict deny-default profile
    which the Meta codex launcher cannot start under).

    NOTE: we deliberately do NOT deny the workspace's parent directory — POSIX
    path traversal needs read access on ancestor directories, so denying the
    parent would block the CLI from reaching its own workspace. The deny-list
    targets the concrete high-value leak vectors (credentials, personal
    documents, other users, mounted volumes); ``extra_repo_root`` is accepted for
    back-compat but only its *non-ancestor* part is denied.
    """

    def _esc(path: str) -> str:
        return str(path).replace("\\", "\\\\").replace('"', '\\"')

    ws_norm = str(workspace).rstrip("/")
    lines = ["(version 1)", "(allow default)"]
    candidate_targets: list[str] = []
    home_norm = str(home).rstrip("/")
    for rel in _HOST_CLI_READ_JAIL_DENY_HOME_SUBPATHS:
        candidate_targets.append(f"{home_norm}/{rel}")
    candidate_targets.extend(_HOST_CLI_READ_JAIL_DENY_ABS_SUBPATHS)
    if extra_repo_root:
        candidate_targets.append(str(extra_repo_root).rstrip("/"))

    def _is_ancestor_or_self(path: str) -> bool:
        # POSIX path traversal needs read access on every ANCESTOR of the
        # workspace, so a deny on the workspace itself or any ancestor would make
        # the CLI unable to reach its own workspace. Skip those.
        p = str(path).rstrip("/")
        return p == ws_norm or (ws_norm + "/").startswith(p + "/")

    seen: set[str] = set()
    for target in candidate_targets:
        target = str(target).rstrip("/")
        if not target or target in seen or _is_ancestor_or_self(target):
            continue
        seen.add(target)
        lines.append(f'(deny file-read* file-write* (subpath "{_esc(target)}"))')
    # Re-allow the workspace explicitly so a workspace nested under a denied root
    # stays fully readable/writable.
    lines.append(
        f"(allow file-read* file-write* file-write-create file-read-metadata "
        f'file-ioctl (subpath "{_esc(ws_norm)}"))'
    )
    return "\n".join(lines) + "\n"


def _write_host_cli_read_jail_profile(working_dir: str, *, suffix: str = "") -> Optional[Path]:
    """Write the Seatbelt profile into the workspace and return its path."""
    try:
        workspace = Path(working_dir).expanduser().resolve(strict=False)
    except OSError:
        workspace = Path(working_dir).expanduser().absolute()
    home = os.path.expanduser("~")
    # Deny the workspace's repo-tree siblings (other repos under the same parent)
    # without blocking the workspace itself.
    extra_repo_root = ""
    try:
        parent = workspace.parent
        if str(parent) not in ("/", home):
            extra_repo_root = str(parent)
    except Exception:  # pragma: no cover - defensive
        extra_repo_root = ""
    text = _host_cli_read_jail_profile_text(
        workspace=str(workspace), home=home, extra_repo_root=extra_repo_root
    )
    try:
        jail_dir = workspace / ".apex_seatbelt"
        jail_dir.mkdir(parents=True, exist_ok=True)
        profile = jail_dir / f"read_jail{suffix}.sb"
        profile.write_text(text, encoding="utf-8")
        return profile
    except OSError:
        logger.debug("host_cli: failed to write read-jail profile", exc_info=True)
        return None


_HOST_CLI_READ_JAIL_GEMINI_PROFILE_NAME = "apexjail"


def _apply_host_cli_read_jail(
    config: LLMConfig,
    launch_command: list[str],
    launch_env: dict[str, str],
    *,
    working_dir: str,
) -> tuple[list[str], dict[str, str]]:
    """Confine a host-launched agent CLI's filesystem reads to its workspace.

    * codex / claude: wrap the launch in an OUTER ``sandbox-exec`` jail (their
      inner sandboxes are disabled in :meth:`_build_command` so the outer profile
      is the single boundary).
    * gemini: it relaunches its OWN process under ``sandbox-exec``; we just drop
      the profile where it looks for it and point ``SEATBELT_PROFILE`` at it.

    Fails OPEN (returns the command unchanged) if anything goes wrong — auth and
    functionality are never sacrificed to a profile-write error; the gold-mode
    leak-guard test is the backstop that catches an un-jailed launch.
    """
    backend = config.backend
    try:
        if backend == LLMBackend.GEMINI_CLI:
            # Gemini CLI confines reads by self-applying a Seatbelt profile: it
            # relaunches its OWN process under sandbox-exec, so we drop our
            # read-jail profile where it looks for it and point SEATBELT_PROFILE
            # at it. This is FAIL-CLOSED on hosts where Gemini's Seatbelt apply is
            # broken (it errors with "sandbox_apply: Operation not permitted"
            # rather than running unconfined). It cannot be wrapped in an OUTER
            # sandbox-exec (its internal apply would nest and fail). Where
            # Gemini's Seatbelt is broken, drop the gemini backend from the model
            # portfolio rather than run it unconfined (do NOT set
            # GEMINI_SANDBOX=false to force it through — that defeats the
            # read-jail; see configs/benchmark_commit0_max.json which omits gemini
            # on this host for exactly this reason).
            workspace = Path(working_dir).expanduser().resolve(strict=False)
            gemini_dir = workspace / ".gemini"
            gemini_dir.mkdir(parents=True, exist_ok=True)
            name = _HOST_CLI_READ_JAIL_GEMINI_PROFILE_NAME
            profile = gemini_dir / f"sandbox-macos-{name}.sb"
            profile.write_text(
                _host_cli_read_jail_profile_text(
                    workspace=str(workspace),
                    home=os.path.expanduser("~"),
                    extra_repo_root=str(workspace.parent)
                    if str(workspace.parent) not in ("/", os.path.expanduser("~"))
                    else "",
                ),
                encoding="utf-8",
            )
            launch_env = dict(launch_env)
            launch_env["GEMINI_SANDBOX"] = "sandbox-exec"
            launch_env["SEATBELT_PROFILE"] = name
            launch_env.setdefault("GEMINI_CLI_TRUST_WORKSPACE", "true")
            return launch_command, launch_env
        # codex / claude: external sandbox-exec wrapper.
        profile = _write_host_cli_read_jail_profile(working_dir)
        if profile is None:
            return launch_command, launch_env
        sandbox_exec = shutil.which("sandbox-exec") or "/usr/bin/sandbox-exec"
        wrapped = [sandbox_exec, "-f", str(profile), *launch_command]
        return wrapped, launch_env
    except Exception:  # pragma: no cover - fail open
        logger.debug("host_cli: failed to apply read jail; launching unjailed", exc_info=True)
        return launch_command, launch_env


# --- host_cli claude auth pre-mint -----------------------------------------
# Layer-B fact (Meta Claude Code auth mechanics on macOS host_cli): the managed
# apiKeyHelper (`/usr/local/bin/claude_code/api-key-helper`) mints a Plugboard
# CAT via `clicat create-all`. clicat is a host dotslash tool that execs absolute
# host binaries; under the target-tool shim PATH that host_cli mode prepends (so
# the agent's repo tools route into the container) clicat's child invocations
# trip the shim's "absolute host dynamic tool paths are disabled" guard and the
# helper returns empty -> claude auth fails (planner/localizer/patcher stages all
# degrade). The helper honors META_PREFETCHED_API_KEY (its sanctioned "launcher
# pre-fetched the key" escape hatch); we mint the token UNCONFINED here (real
# host PATH, no shim, outside the Seatbelt wrap) and inject it so claude never
# runs clicat under the shim. Fail-open: any error leaves the env untouched and
# claude falls back to its own helper (works when a host cred is already warm).
_HOST_CLI_CLAUDE_API_KEY_HELPER_CANDIDATES: tuple[str, ...] = (
    "/usr/local/bin/claude_code/api-key-helper",
)
_HOST_CLI_CLAUDE_MANAGED_SETTINGS_PATHS: tuple[str, ...] = (
    "/Library/Application Support/ClaudeCode/managed-settings.json",
    "/etc/claude-code/managed-settings.json",
)
# Helper computes token lifetime as TTL_MS * 2 / 1000 seconds; 10800000 ms ->
# a 6h token that outlives any single claude stage. Re-mint in-process well
# before expiry so a long per-repo run never serves a stale token.
_HOST_CLI_CLAUDE_API_KEY_TTL_MS = "10800000"
_HOST_CLI_CLAUDE_API_KEY_REFRESH_SECONDS = 1800.0
_host_cli_claude_api_key_cache: dict[str, tuple[str, float]] = {}
_host_cli_claude_api_key_lock = threading.Lock()


def _resolve_host_cli_claude_api_key_helper() -> Optional[list[str]]:
    """Resolve the apiKeyHelper command claude would run, preferring an explicit
    override, then the Claude Code managed-settings declaration, then the known
    Meta host path."""

    override = str(os.environ.get("APEX_HOST_CLI_CLAUDE_API_KEY_HELPER") or "").strip()
    if override:
        try:
            parts = shlex.split(override)
        except ValueError:
            parts = [override]
        if parts and (
            (os.path.isabs(parts[0]) and os.access(parts[0], os.X_OK)) or shutil.which(parts[0])
        ):
            return parts
    for settings_path in _HOST_CLI_CLAUDE_MANAGED_SETTINGS_PATHS:
        try:
            data = json.loads(Path(settings_path).read_text(encoding="utf-8"))
        except Exception:
            continue
        helper = str((data or {}).get("apiKeyHelper") or "").strip()
        if not helper:
            continue
        try:
            parts = shlex.split(helper)
        except ValueError:
            parts = [helper]
        if parts and os.access(parts[0], os.X_OK):
            return parts
    for candidate in _HOST_CLI_CLAUDE_API_KEY_HELPER_CANDIDATES:
        if os.access(candidate, os.X_OK):
            return [candidate]
    return None


def _host_cli_unconfined_mint_env() -> dict[str, str]:
    """Real host env for minting the claude gateway token: strip any target-tool
    shim context and guarantee the standard system PATH so clicat's host tools
    resolve to real binaries (never the container-routing shims)."""

    env = dict(os.environ)
    for key in (
        "APEX_TARGET_TOOL_CONTEXT",
        "APEX_TARGET_TOOL_WORKDIR",
        "APEX_TARGET_TOOL_BRIDGE_FILE",
        "APEX_CLI_BOOTSTRAP_PATH",
        "META_PREFETCHED_API_KEY",
    ):
        env.pop(key, None)
    system_path = "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    current_path = str(env.get("PATH") or "")
    env["PATH"] = system_path + (os.pathsep + current_path if current_path else "")
    env["CLAUDE_CODE_API_KEY_HELPER_TTL_MS"] = _HOST_CLI_CLAUDE_API_KEY_TTL_MS
    return env


def _prefetch_host_cli_claude_api_key() -> Optional[str]:
    """Mint (or reuse a fresh cached) claude gateway token on the unconfined host.
    Returns None on any failure so the caller fails open."""

    # Mint under the lock with a double-check so a burst of rollout threads at
    # process start does not fan out into one clicat call per thread — the first
    # mints, the rest read the warm cache (~immediate) instead of stampeding.
    with _host_cli_claude_api_key_lock:
        cached = _host_cli_claude_api_key_cache.get("claude")
        if cached and (time.time() - cached[1]) < _HOST_CLI_CLAUDE_API_KEY_REFRESH_SECONDS:
            return cached[0]
        helper = _resolve_host_cli_claude_api_key_helper()
        if not helper:
            return None
        try:
            completed = subprocess.run(
                helper,
                env=_host_cli_unconfined_mint_env(),
                capture_output=True,
                text=True,
                timeout=120,
            )
        except Exception:
            logger.debug("host_cli: claude api-key pre-mint subprocess failed", exc_info=True)
            return None
        token = (completed.stdout or "").strip()
        if completed.returncode != 0 or not token:
            logger.debug(
                "host_cli: claude api-key pre-mint produced no token (rc=%s)",
                completed.returncode,
            )
            return None
        _host_cli_claude_api_key_cache["claude"] = (token, time.time())
        return token


def _host_cli_isolated_claude_config_dir() -> Optional[str]:
    """Return a persistent, operator-free CLAUDE_CONFIG_DIR for host_cli claude.

    Layer-B fact: host_cli keeps the real HOME (so claude can dotslash-materialize
    native binaries from ~/Library/Caches and use the host Meta transport), but
    the default CLAUDE_CONFIG_DIR=~/.claude then drags the OPERATOR's user
    plugins/hooks/settings into every agent rollout — e.g. the agent-market
    meta-trajectory `SessionEnd` hook (`${CLAUDE_PLUGIN_ROOT}/hooks/session-end.sh`,
    which fails -> "CLI finalization failure" -> recovered-but-lost candidates) and
    the skill-evaluation `rg ~/.claude/plugins/cache` that trips the workspace
    guard. Pointing CLAUDE_CONFIG_DIR at an isolated dir loads zero operator
    plugins ("No plugins installed.") and no user-settings hooks, while managed
    settings + the APEX `--settings` tool-review hook still apply and HOME-based
    dotslash/auth keep working. Reused across launches (claude config dirs are
    designed for concurrent sessions) to avoid native re-bootstrap churn."""

    try:
        config_dir = _default_cli_state_root() / "host_cli_isolated_claude_config" / "v1"
        config_dir.mkdir(parents=True, exist_ok=True)
        return str(config_dir)
    except OSError:
        logger.debug("host_cli: failed to prepare isolated claude config dir", exc_info=True)
        return None


def _inject_host_cli_prefetched_claude_api_key(
    config: LLMConfig,
    env: dict[str, str],
    launch_env: dict[str, str],
) -> dict[str, str]:
    """For host_cli claude launches: (1) isolate CLAUDE_CONFIG_DIR so the agent
    never loads the operator's plugins/hooks/settings, and (2) inject a pre-minted
    gateway token via META_PREFETCHED_API_KEY so the managed apiKeyHelper returns
    it instantly instead of running clicat under the confining target-tool shim."""

    if config.backend != LLMBackend.CLAUDE_CLI:
        return launch_env
    if not _target_runtime_uses_host_cli(config, env):
        return launch_env
    launch_env = dict(launch_env)
    # (1) Operator-plugin/hook isolation (opt-out via APEX_HOST_CLI_ISOLATE_CLAUDE_CONFIG=0).
    if str(os.environ.get("APEX_HOST_CLI_ISOLATE_CLAUDE_CONFIG") or "").strip() not in {
        "0",
        "false",
        "no",
        "off",
    }:
        isolated = _host_cli_isolated_claude_config_dir()
        if isolated:
            launch_env["CLAUDE_CONFIG_DIR"] = isolated
    # (2) Pre-minted gateway token.
    if not str(launch_env.get("META_PREFETCHED_API_KEY") or "").strip():
        token = _prefetch_host_cli_claude_api_key()
        if token:
            launch_env["META_PREFETCHED_API_KEY"] = token
    return launch_env


def _container_reachable_model_proxy_url(raw_url: str, env: dict[str, str]) -> str:
    text = str(raw_url or "").strip().rstrip("/")
    if not text:
        return ""
    parse_value = text if "://" in text else f"http://{text}"
    try:
        parsed = urllib.parse.urlsplit(parse_value)
    except ValueError as exc:
        raise CLIAgentContainerIsolationError(
            f"Invalid APEX model proxy URL for target-container CLI: {raw_url!r}"
        ) from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise CLIAgentContainerIsolationError(
            f"APEX model proxy URL must be an http(s) URL with a host: {raw_url!r}"
        )
    if parsed.username or parsed.password:
        raise CLIAgentContainerIsolationError(
            "APEX model proxy URL must not embed credentials; pass proxy auth through "
            "the proxy service, not the containerized agent command line."
        )
    preserve_loopback = str(
        env.get("APEX_AGENT_CONTAINER_PRESERVE_LOOPBACK_PROXY") or ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    return _proxy_value_for_agent_container(text, preserve_loopback=preserve_loopback).rstrip("/")


def _merge_no_proxy_entries(existing: str, additions: Iterable[str]) -> str:
    entries: list[str] = []
    seen: set[str] = set()
    for raw in [*str(existing or "").split(","), *additions]:
        item = str(raw or "").strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        entries.append(item)
    return ",".join(entries)


def _note_model_proxy_no_proxy(env: dict[str, str], proxy_url: str) -> None:
    try:
        parsed = urllib.parse.urlsplit(proxy_url if "://" in proxy_url else f"http://{proxy_url}")
    except ValueError:
        return
    hostname = parsed.hostname or ""
    if hostname not in {"host.docker.internal", "localhost", "127.0.0.1", "::1"}:
        return
    merged = _merge_no_proxy_entries(
        str(env.get("NO_PROXY") or env.get("no_proxy") or ""),
        (hostname, "host.docker.internal", "localhost", "127.0.0.1", "::1"),
    )
    if merged:
        env["NO_PROXY"] = merged


def _apply_cli_model_proxy_for_target_runtime(
    config: LLMConfig,
    env: dict[str, str],
) -> bool:
    """Wire a host-auth model endpoint while keeping the agent CLI in Docker."""

    if not env.get("APEX_TARGET_TOOL_CONTEXT"):
        return False
    raw_url = _configured_cli_model_proxy_url(config, env)
    if not raw_url:
        return False
    proxy_url = _container_reachable_model_proxy_url(raw_url, env)
    if not proxy_url:
        return False
    for key in _BACKEND_MODEL_PROXY_CONTAINER_CREDENTIAL_ENV_KEYS.get(config.backend, ()):
        env.pop(key, None)
    spec = _CLI_BACKEND_SANDBOX_SPECS.get(config.backend)
    if spec is not None:
        env.pop(spec.auth_state_env_key, None)
    backend = config.backend
    if backend == LLMBackend.CODEX_CLI:
        env["CODEX_BASE_URL"] = proxy_url
        env["OPENAI_API_KEY"] = _MODEL_PROXY_DUMMY_SECRET
    elif backend == LLMBackend.CLAUDE_CLI:
        env["ANTHROPIC_BASE_URL"] = proxy_url
        env["ANTHROPIC_API_KEY"] = _MODEL_PROXY_DUMMY_SECRET
    elif backend == LLMBackend.GEMINI_CLI:
        # Gemini CLI exposes endpoint override on its Code Assist path; use a
        # dummy GCA token so auth stays inside the target container.
        env["CODE_ASSIST_ENDPOINT"] = proxy_url
        env.setdefault("CODE_ASSIST_API_VERSION", "v1internal")
        env["GOOGLE_GENAI_USE_GCA"] = "true"
        env["GOOGLE_CLOUD_ACCESS_TOKEN"] = _MODEL_PROXY_DUMMY_SECRET
    elif backend in _OPENCODE_FAMILY_BACKENDS:
        # OpenCode-family CLIs use provider base URLs; set both common provider
        # routes so the configured model/provider can resolve without host auth.
        env["ANTHROPIC_BASE_URL"] = proxy_url
        env["OPENAI_BASE_URL"] = proxy_url
        env["ANTHROPIC_API_KEY"] = _MODEL_PROXY_DUMMY_SECRET
        env["OPENAI_API_KEY"] = _MODEL_PROXY_DUMMY_SECRET
    else:
        return False
    env["APEX_AGENT_MODEL_PROXY_ACTIVE"] = "1"
    env["APEX_AGENT_MODEL_PROXY_BACKEND"] = backend.value
    _note_model_proxy_no_proxy(env, proxy_url)
    return True


def _normalize_gemini_provider_env(config: LLMConfig, env: dict[str, str]) -> None:
    if config.backend != LLMBackend.GEMINI_CLI:
        return
    if not str(env.get("GEMINI_CLI_CUSTOM_HEADERS") or "").strip():
        custom_headers = str(env.get("GEMINI_CUSTOM_HEADERS") or "").strip()
        if custom_headers:
            env["GEMINI_CLI_CUSTOM_HEADERS"] = custom_headers


def _prepare_cli_target_runtime_env(
    config: LLMConfig,
    env: dict[str, str],
    *,
    working_dir: str,
) -> bool:
    _normalize_gemini_provider_env(config, env)
    proxy_first = _cli_model_proxy_configured(config, env)
    if _target_runtime_requires_model_proxy(env) and not proxy_first:
        keys = (
            *_BACKEND_MODEL_PROXY_ENV_KEYS.get(config.backend, ()),
            *_GLOBAL_MODEL_PROXY_ENV_KEYS,
        )
        raise CLIAgentContainerIsolationError(
            "Target-runtime CLI auth mode requires a host model proxy; set one of: "
            + ", ".join(keys)
        )
    proxy_active = _apply_cli_model_proxy_for_target_runtime(config, env) if proxy_first else False
    has_auth_state = _relocate_cli_home_for_target_runtime(
        config,
        env,
        working_dir=working_dir,
        copy_auth_state=not proxy_active,
    )
    if not proxy_first:
        proxy_active = _apply_cli_model_proxy_for_target_runtime(config, env)
    return has_auth_state or proxy_active


def _backend_auth_configured(config: LLMConfig, env: dict[str, str]) -> bool:
    spec = _CLI_BACKEND_SANDBOX_SPECS.get(config.backend)
    if spec is not None:
        auth_state = str(env.get(spec.auth_state_env_key) or "").strip()
        if _auth_state_env_value_is_marker(auth_state) or any(
            path.exists() for path in _explicit_auth_state_source_paths(auth_state)
        ):
            return True
    requirements = _BACKEND_AUTH_REQUIREMENTS.get(config.backend)
    if not requirements:
        return True
    for requirement in requirements:
        if all(str(env.get(key) or "").strip() for key in requirement):
            return True
    return False


def _auth_state_env_value_is_marker(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "configured"}


def _explicit_auth_state_source_paths(value: str) -> list[Path]:
    text = str(value or "").strip()
    if not text or _auth_state_env_value_is_marker(text):
        return []
    raw_parts: list[str]
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = []
        raw_parts = [str(item) for item in parsed if str(item).strip()]
    else:
        raw_parts = [part for part in text.split(os.pathsep) if part.strip()]
    paths: list[Path] = []
    seen: set[str] = set()
    for raw in raw_parts:
        path = Path(raw).expanduser()
        key = os.path.normpath(os.fspath(path))
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return paths


def _auth_env_satisfies_backend_without_file_copy(
    spec: _CLIBackendSandboxSpec,
    env: dict[str, str],
) -> bool:
    file_backed_env_keys = {
        rule.source_env_key
        for rule in spec.auth_state_files
        if rule.source_env_key and rule.target_env_key
    }
    # If a file-backed credential env var (e.g. GOOGLE_APPLICATION_CREDENTIALS)
    # is present, the no-file-copy fast path must NOT short-circuit even when
    # some other, non-file auth requirement is also satisfied. That credential
    # file still has to be relocated into the container mount and its env var
    # rewritten to an in-mount path; otherwise the later host->container path
    # mapping rejects the out-of-mount host path and marks an otherwise-healthy
    # backend unavailable. This is the dominant target-runtime CLI-auth blocker:
    # an ambient host credential (Vertex base URL + skip-vertex-auth) satisfying
    # the API requirement would otherwise suppress relocation of an explicitly
    # configured service-account key file.
    if any(str(env.get(key) or "").strip() for key in file_backed_env_keys):
        return False
    for requirement in spec.auth_requirements:
        if not all(str(env.get(key) or "").strip() for key in requirement):
            continue
        if file_backed_env_keys.intersection(requirement):
            continue
        return True
    return False


def _backend_auth_hint_keys(spec: _CLIBackendSandboxSpec) -> tuple[str, ...]:
    keys: list[str] = []
    seen: set[str] = set()
    for requirement in spec.auth_requirements:
        for key in requirement:
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
    return tuple(keys)


def _copy_cli_auth_state_file(
    source: Path,
    target: Path,
    *,
    mode: int,
) -> bool:
    try:
        source_resolved = source.expanduser().resolve(strict=True)
    except OSError:
        return False
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_resolved, target)
        try:
            target.chmod(mode)
        except OSError:
            pass
        return True
    except OSError:
        logger.debug(
            "Failed to materialize CLI auth state file for isolated target runtime",
            exc_info=True,
        )
        return False


def _auth_state_directory_file_marks_auth(
    path: Path,
    rule: _CLIAuthStateDirectory,
) -> bool:
    name = path.name.lower()
    rel = str(path).lower()
    return any(
        fnmatch.fnmatchcase(name, pattern.lower()) or fnmatch.fnmatchcase(rel, pattern.lower())
        for pattern in rule.auth_filename_patterns
    )


def _copy_cli_auth_state_directory(
    source: Path,
    target: Path,
    *,
    rule: _CLIAuthStateDirectory,
) -> bool:
    try:
        source_resolved = source.expanduser().resolve(strict=True)
    except OSError:
        return False
    if not source_resolved.is_dir():
        return False
    copied_auth_material = False
    exclude_names = {name.lower() for name in rule.exclude_names}
    exclude_patterns = tuple(pattern.lower() for pattern in rule.exclude_patterns)

    def _excluded(name: str) -> bool:
        lowered = name.lower()
        return lowered in exclude_names or any(
            fnmatch.fnmatchcase(lowered, pattern) for pattern in exclude_patterns
        )

    try:
        scanned_files = 0
        for current_root, dir_names, file_names in os.walk(source_resolved):
            root_path = Path(current_root)
            try:
                depth = len(root_path.relative_to(source_resolved).parts)
            except ValueError:
                depth = 0
            if depth >= max(0, int(rule.max_depth)):
                dir_names[:] = []
            dir_names[:] = [
                name for name in dir_names if not _excluded(name) and not name.startswith(".apex")
            ]
            try:
                rel_root = root_path.relative_to(source_resolved)
            except ValueError:
                continue
            for file_name in file_names:
                if _excluded(file_name) or file_name.startswith(".apex"):
                    continue
                scanned_files += 1
                if scanned_files > max(1, int(rule.max_scan_files)):
                    return copied_auth_material
                source_file = root_path / file_name
                if not source_file.is_file() or source_file.is_symlink():
                    continue
                if not _auth_state_directory_file_marks_auth(source_file, rule):
                    continue
                target_root = target / rel_root
                target_file = target_root / file_name
                try:
                    target_root.mkdir(parents=True, exist_ok=True)
                    target_root.chmod(rule.dir_mode)
                    shutil.copyfile(source_file, target_file)
                    target_file.chmod(rule.file_mode)
                except OSError:
                    logger.debug(
                        "Failed to copy CLI auth state file from directory snapshot",
                        exc_info=True,
                    )
                    continue
                copied_auth_material = True
        return copied_auth_material
    except OSError:
        logger.debug(
            "Failed to materialize CLI auth state directory for isolated target runtime",
            exc_info=True,
        )
        return copied_auth_material


def _auth_source_suffix_for_declared_home(
    relative_path: str,
    declared_home_relative: str,
) -> Optional[Path]:
    rel_parts = Path(relative_path).parts
    home_parts = Path(declared_home_relative).parts
    if not home_parts or home_parts == (".",):
        return Path(relative_path)
    if rel_parts[: len(home_parts)] != home_parts:
        return None
    suffix_parts = rel_parts[len(home_parts) :]
    return Path(*suffix_parts) if suffix_parts else Path()


def _cli_auth_source_candidates(
    spec: _CLIBackendSandboxSpec,
    relative_path: str,
    *,
    source_env: Mapping[str, str],
) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = os.path.normpath(os.fspath(path.expanduser()))
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    add(Path.home() / relative_path)
    for env_key, home_relative in (
        *spec.target_runtime_home.items(),
        *_TARGET_RUNTIME_SOURCE_HOME_ENV_RELATIVES.items(),
    ):
        raw_root = str(source_env.get(env_key) or "").strip()
        if not raw_root:
            continue
        suffix = _auth_source_suffix_for_declared_home(relative_path, home_relative)
        if suffix is None:
            continue
        add(Path(raw_root).expanduser() / suffix)

    return candidates


def _primary_cli_auth_target_dir(spec: _CLIBackendSandboxSpec) -> str:
    if spec.target_runtime_home:
        return next(iter(spec.target_runtime_home.values()))
    return "."


def _auth_state_file_matches_rule(source: Path, rule: _CLIAuthStateFile) -> bool:
    source_name = source.name.lower()
    for rel in (rule.source_home_relative, rule.target_home_relative):
        if rel and source_name == Path(rel).name.lower():
            return True
    return False


def _auth_state_directory_has_direct_auth_material(
    source: Path,
    rule: _CLIAuthStateDirectory,
) -> bool:
    try:
        for child in source.iterdir():
            if (
                child.is_file()
                and not child.is_symlink()
                and _auth_state_directory_file_marks_auth(child, rule)
            ):
                return True
    except OSError:
        return False
    return False


def _copy_explicit_cli_auth_state_source(
    source: Path,
    spec: _CLIBackendSandboxSpec,
    *,
    home_path: Path,
) -> bool:
    try:
        resolved = source.expanduser().resolve(strict=True)
    except OSError:
        return False
    if resolved.is_dir():
        copied = False
        for rule in spec.auth_state_files:
            if not rule.source_home_relative:
                continue
            nested_source = resolved / rule.source_home_relative
            if nested_source.is_file() and _copy_cli_auth_state_file(
                nested_source,
                home_path / rule.target_home_relative,
                mode=rule.mode,
            ):
                copied = True

        directory_rules = spec.auth_state_directories
        if not directory_rules:
            directory_rules = (
                _CLIAuthStateDirectory(
                    source_home_relative=os.fspath(resolved),
                    target_home_relative=_primary_cli_auth_target_dir(spec),
                ),
            )
        for rule in directory_rules:
            relative_source = Path(rule.source_home_relative)
            source_dirs: list[Path] = []
            if relative_source.parts and not relative_source.is_absolute():
                nested_source = resolved / relative_source
                if nested_source.is_dir():
                    source_dirs.append(nested_source)
            if resolved.name == Path(
                rule.source_home_relative
            ).name or _auth_state_directory_has_direct_auth_material(resolved, rule):
                source_dirs.append(resolved)
            for source_dir in source_dirs:
                if _copy_cli_auth_state_directory(
                    source_dir,
                    home_path / rule.target_home_relative,
                    rule=rule,
                ):
                    copied = True
                    break
        return copied
    if not resolved.is_file() or resolved.is_symlink():
        return False
    for rule in spec.auth_state_files:
        if _auth_state_file_matches_rule(resolved, rule):
            return _copy_cli_auth_state_file(
                resolved,
                home_path / rule.target_home_relative,
                mode=rule.mode,
            )
    directory_rule = _CLIAuthStateDirectory(
        source_home_relative=str(resolved.parent),
        target_home_relative=_primary_cli_auth_target_dir(spec),
    )
    if not _auth_state_directory_file_marks_auth(resolved, directory_rule):
        return False
    return _copy_cli_auth_state_file(
        resolved,
        home_path / _primary_cli_auth_target_dir(spec) / resolved.name,
        mode=directory_rule.file_mode,
    )


def _chmod_existing_path(path: Path, mode: int) -> None:
    try:
        if path.exists():
            path.chmod(mode)
    except OSError:
        logger.debug("Failed to chmod CLI runtime path %s", path, exc_info=True)


def _ensure_target_runtime_cli_home_container_permissions(
    spec: _CLIBackendSandboxSpec,
    home_path: Path,
    *,
    public_config_files: Iterable[Path] = (),
) -> None:
    """Make Apex-owned CLI runtime state traversable inside Docker containers."""

    directories: set[Path] = set()

    current = home_path
    while True:
        directories.add(current)
        if current.name == _AGENT_RUNTIME_STATE_DIRNAME or current.parent == current:
            break
        current = current.parent

    for rel_path in (
        ".config",
        ".cache",
        ".local",
        ".local/share",
        ".local/state",
        "Library",
        "Library/Caches",
        *spec.target_runtime_home.values(),
    ):
        target_dir = home_path / rel_path
        current = target_dir
        while True:
            directories.add(current)
            if current == home_path or current.parent == current:
                break
            current = current.parent

    for rule in spec.auth_state_files:
        target = home_path / rule.target_home_relative
        directories.add(target.parent)
        _chmod_existing_path(target, rule.mode)
    for rule in spec.auth_state_directories:
        directories.add(home_path / rule.target_home_relative)

    for directory in sorted(directories, key=lambda item: len(item.parts)):
        try:
            if directory.exists() and directory.is_dir():
                directory.chmod(0o755)
        except OSError:
            logger.debug("Failed to chmod CLI runtime directory %s", directory, exc_info=True)

    for path in public_config_files:
        _chmod_existing_path(path, 0o644)


def _materialize_cli_auth_state_for_target_runtime(
    spec: _CLIBackendSandboxSpec,
    env: dict[str, str],
    *,
    home_path: Path,
    source_env: Optional[Mapping[str, str]] = None,
) -> bool:
    explicit_auth_state = str(env.get(spec.auth_state_env_key) or "").strip()
    if explicit_auth_state and not _auth_state_env_value_is_marker(explicit_auth_state):
        copied_explicit = False
        for source in _explicit_auth_state_source_paths(explicit_auth_state):
            copied_explicit = (
                _copy_explicit_cli_auth_state_source(source, spec, home_path=home_path)
                or copied_explicit
            )
        if copied_explicit:
            env[spec.auth_state_env_key] = "1"
            return True
        env.pop(spec.auth_state_env_key, None)
    if _auth_env_satisfies_backend_without_file_copy(spec, env):
        env[spec.auth_state_env_key] = "1"
        return True
    copied_any = False
    auth_source_env = source_env or env
    for rule in spec.auth_state_directories:
        copied = False
        for source in _cli_auth_source_candidates(
            spec,
            rule.source_home_relative,
            source_env=auth_source_env,
        ):
            copied = _copy_cli_auth_state_directory(
                source,
                home_path / rule.target_home_relative,
                rule=rule,
            )
            if copied:
                break
        copied_any = copied_any or (copied and rule.marks_auth)
    for rule in spec.auth_state_files:
        if rule.required_env_keys and not all(
            str(env.get(key) or "").strip() for key in rule.required_env_keys
        ):
            continue
        target = home_path / rule.target_home_relative
        if rule.source_env_key:
            raw_source = str(env.get(rule.source_env_key) or "").strip()
            if not raw_source:
                continue
            copied = _copy_cli_auth_state_file(
                Path(raw_source),
                target,
                mode=rule.mode,
            )
            if copied and rule.target_env_key:
                env[rule.target_env_key] = str(target)
        elif rule.source_home_relative:
            copied = False
            for source in _cli_auth_source_candidates(
                spec,
                rule.source_home_relative,
                source_env=auth_source_env,
            ):
                copied = _copy_cli_auth_state_file(
                    source,
                    target,
                    mode=rule.mode,
                )
                if copied:
                    break
        else:
            copied = False
        copied_any = copied_any or (copied and rule.marks_auth)
    if copied_any:
        env[spec.auth_state_env_key] = "1"
    return copied_any


def _write_gemini_target_runtime_settings(
    env: dict[str, str],
    *,
    home_path: Path,
    has_auth_state: bool,
) -> None:
    gemini_dir = home_path / ".gemini"
    gemini_dir.mkdir(parents=True, exist_ok=True)
    settings_path = gemini_dir / "settings.json"
    selected_type = ""
    if str(env.get("GEMINI_API_KEY") or "").strip():
        selected_type = "gemini-api-key"
    elif str(env.get("GOOGLE_GENAI_USE_VERTEXAI") or "").strip():
        selected_type = "vertex-ai"
    elif str(env.get("GOOGLE_GENAI_USE_GCA") or "").strip() or has_auth_state:
        selected_type = "oauth-personal"
    if not selected_type or settings_path.exists():
        return
    # Gemini CLI stores its selected auth type in user settings. In Docker
    # target runtimes Apex writes only this minimal non-secret selection file.
    settings_path.write_text(
        json.dumps({"security": {"auth": {"selectedType": selected_type}}}, indent=2) + "\n",
        encoding="utf-8",
    )


def _target_runtime_kind(env: dict[str, str]) -> str:
    target_context = _load_target_runtime_context(env)
    runtime = dict(target_context.get("runtime") or {})
    return str(runtime.get("kind") or target_context.get("mode") or "").strip()


def _target_runtime_launches_agent_cli_in_container(
    config: LLMConfig,
    env: dict[str, str],
) -> bool:
    """Return True when the provider CLI itself should run inside Docker."""

    if not env.get("APEX_TARGET_TOOL_CONTEXT"):
        return False
    if _target_runtime_uses_docker_sandbox_cli(config, env):
        return False
    # host_cli mode keeps the agent CLI on the host (native auth) -> never launch
    # it inside the container; only its tool execution is routed in via shims.
    if _target_runtime_uses_host_cli(config, env):
        return False
    return _target_runtime_kind(env) == "docker_exec"


def _agent_container_launch_context(
    env: dict[str, str],
    *,
    working_dir: str,
) -> _AgentContainerLaunchContext | None:
    target_context = _load_target_runtime_context(env)
    runtime = dict(target_context.get("runtime") or {})
    if str(runtime.get("kind") or target_context.get("mode") or "") != "docker_exec":
        return None
    container_name = str(runtime.get("docker_container_name") or "").strip()
    if not container_name:
        raise CLIAgentContainerIsolationError(
            "target runtime requested docker_exec but did not declare a container"
        )
    raw_host_root = str(runtime.get("docker_host_workdir_root") or "").strip()
    if not raw_host_root:
        raise CLIAgentContainerIsolationError(
            "target runtime requested docker_exec but did not declare a host mount root"
        )
    try:
        host_root = Path(raw_host_root).expanduser().resolve(strict=False)
    except OSError:
        host_root = Path(raw_host_root).expanduser().absolute()
    container_root = str(runtime.get("docker_container_workdir_root") or "/workspace").strip()
    if not container_root.startswith("/"):
        raise CLIAgentContainerIsolationError(
            "target runtime requested docker_exec with a non-absolute container mount root"
        )
    raw_docker_bin = str(runtime.get("docker_bin") or "docker").strip() or "docker"
    docker_bin = (
        raw_docker_bin
        if os.path.isabs(raw_docker_bin)
        else (shutil.which(raw_docker_bin, path=os.environ.get("PATH", "")) or raw_docker_bin)
    )
    context = _AgentContainerLaunchContext(
        docker_bin=docker_bin,
        container_name=container_name,
        host_root=host_root,
        container_root=container_root,
        working_dir_container="",
        runtime_env={
            str(key): str(value) for key, value in dict(runtime.get("docker_env") or {}).items()
        },
        docker_host_env={
            str(key): str(value)
            for key, value in dict(runtime.get("docker_host_env") or {}).items()
        },
        docker_user=str(runtime.get("docker_user") or "").strip(),
    )
    working_dir_container = _map_host_path_to_agent_container(working_dir, context)
    context_path = str(env.get("APEX_TARGET_TOOL_CONTEXT") or "").strip()
    if context_path:
        _map_host_path_to_agent_container(context_path, context)
    return _AgentContainerLaunchContext(
        docker_bin=docker_bin,
        container_name=container_name,
        host_root=host_root,
        container_root=container_root,
        working_dir_container=working_dir_container,
        runtime_env=context.runtime_env,
        docker_host_env=context.docker_host_env,
        docker_user=context.docker_user,
    )


def _agent_container_command(
    command: list[str],
    context: _AgentContainerLaunchContext,
    *,
    backend: LLMBackend | None = None,
) -> list[str]:
    container_command: list[str] = []
    for index, item in enumerate(command):
        text = str(item)
        backend_executable = _agent_container_backend_executable_override(
            text,
            context=context,
            backend=backend,
        )
        if backend_executable:
            container_command.append(backend_executable)
            continue
        if index == 0 and os.path.isabs(text):
            try:
                container_command.append(_map_host_path_to_agent_container(text, context))
            except CLIAgentContainerIsolationError:
                if _agent_container_absolute_executable_allowed(text, context):
                    container_command.append(text)
                else:
                    container_command.append(Path(text).name)
            continue
        container_command.append(_map_host_root_text_to_agent_container(text, context))
    return _agent_container_cli_launcher_command(
        container_command,
        context=context,
        backend=backend,
    )


def _agent_container_backend_executable_override(
    executable: str,
    *,
    context: _AgentContainerLaunchContext,
    backend: LLMBackend | None,
) -> str:
    if backend is None:
        return ""
    spec = _CLI_BACKEND_SANDBOX_SPECS.get(backend)
    if spec is None:
        return ""
    text = str(executable or "").strip()
    if not text.startswith("/"):
        return ""
    executable_path = PurePosixPath(text)
    if executable_path.name not in spec.binary_names:
        return ""
    if str(executable_path.parent).startswith("/opt/"):
        return ""
    bundle_bin = _agent_container_node_cli_bundle_bin(context)
    if not bundle_bin:
        return ""
    return str(PurePosixPath(bundle_bin) / executable_path.name)


def _agent_container_cli_launcher_command(
    command: list[str],
    *,
    context: _AgentContainerLaunchContext,
    backend: LLMBackend | None,
) -> list[str]:
    if not command or backend is None:
        return command
    spec = _CLI_BACKEND_SANDBOX_SPECS.get(backend)
    if spec is None or not spec.probe_requires_node:
        return command
    executable = str(command[0] or "").strip()
    executable_path = PurePosixPath(executable)
    if executable_path.name not in spec.binary_names:
        return command
    if executable.startswith("/") and str(executable_path.parent).startswith("/opt/"):
        bundle_bin = str(executable_path.parent)
    else:
        bundle_bin = _agent_container_node_cli_bundle_bin(context)
        if not bundle_bin:
            return command
        executable = str(PurePosixPath(bundle_bin) / executable_path.name)
    node = str(PurePosixPath(bundle_bin) / "node")
    # Node-backed agent CLI wrappers use `/usr/bin/env node`; with target-tool
    # shims first on PATH, that shebang recurses through the command-budgeted
    # tool shim. Launch the agent CLI with its sibling Node runtime directly
    # while preserving the shimmed PATH for tools the agent invokes afterward.
    return [node, executable, *command[1:]]


def _agent_container_node_cli_bundle_bin(
    context: _AgentContainerLaunchContext,
) -> str:
    for raw_path in (
        str(context.runtime_env.get("PATH") or ""),
        str(context.docker_host_env.get("PATH") or ""),
    ):
        for raw_entry in raw_path.split(":"):
            entry = raw_entry.strip().rstrip("/")
            if entry.startswith("/opt/") and entry.endswith("/bin"):
                return entry
    return ""


def _agent_container_absolute_executable_allowed(
    executable: str,
    context: _AgentContainerLaunchContext,
) -> bool:
    text = str(executable or "").strip()
    if not text.startswith("/"):
        return False
    container_root = context.container_root.rstrip("/")
    if text == container_root or text.startswith(container_root + "/"):
        return True
    return text.startswith(
        (
            "/bin/",
            "/sbin/",
            "/usr/bin/",
            "/usr/sbin/",
            "/usr/local/bin/",
            "/usr/local/sbin/",
            "/opt/",
        )
    )


def _merge_path_entries(*path_values: str) -> str:
    entries: list[str] = []
    seen: set[str] = set()
    for path_value in path_values:
        for raw_entry in str(path_value or "").split(os.pathsep):
            entry = raw_entry.strip()
            if not entry or entry in seen:
                continue
            seen.add(entry)
            entries.append(entry)
    return os.pathsep.join(entries)


def _agent_container_path_value(
    path_value: str,
    context: _AgentContainerLaunchContext,
    *,
    allow_container_local: bool = False,
) -> str:
    entries: list[str] = []
    for raw_entry in str(path_value or "").split(os.pathsep):
        entry = _map_host_root_text_to_agent_container(raw_entry.strip(), context)
        if not entry:
            continue
        if _agent_container_path_entry_allowed(
            entry,
            context,
            allow_container_local=allow_container_local,
        ):
            entries.append(entry)
    return os.pathsep.join(entries)


def _agent_container_path_entry_allowed(
    entry: str,
    context: _AgentContainerLaunchContext,
    *,
    allow_container_local: bool = False,
) -> bool:
    container_root = context.container_root.rstrip("/")
    if entry == container_root or entry.startswith(container_root + "/"):
        return True
    if entry in {
        "/usr/local/sbin",
        "/usr/local/bin",
        "/usr/sbin",
        "/usr/bin",
        "/sbin",
        "/bin",
    }:
        return True
    if allow_container_local and (entry.startswith("/opt/") or entry.startswith("/usr/local/lib/")):
        return True
    return False


def _agent_container_env(
    env: dict[str, str],
    context: _AgentContainerLaunchContext,
) -> dict[str, str]:
    container_env: dict[str, str] = dict(context.runtime_env)
    preserve_loopback_proxy = str(
        env.get("APEX_AGENT_CONTAINER_PRESERVE_LOOPBACK_PROXY") or ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    for key, value in env.items():
        if key not in _AGENT_CONTAINER_ENV_ALLOWLIST and not key.startswith(
            _AGENT_CONTAINER_ENV_PREFIX_ALLOWLIST
        ):
            continue
        if key == "PATH":
            continue
        if key in _AGENT_CONTAINER_PROXY_ENV_KEYS:
            container_env[str(key)] = _proxy_value_for_agent_container(
                str(value),
                preserve_loopback=preserve_loopback_proxy,
            )
            continue
        if key in _AGENT_CONTAINER_PATH_ENV_KEYS:
            if str(value).strip():
                container_env[str(key)] = _map_host_path_to_agent_container(str(value), context)
            continue
        container_env[str(key)] = _map_host_root_text_to_agent_container(str(value), context)
    if env.get("PATH") or context.runtime_env.get("PATH"):
        container_env["PATH"] = _merge_path_entries(
            _agent_container_path_value(str(env.get("PATH") or ""), context),
            _agent_container_path_value(
                str(context.runtime_env.get("PATH") or ""),
                context,
                allow_container_local=True,
            ),
        )
    container_env["APEX_AGENT_CONTAINER"] = "1"
    container_env["APEX_TARGET_TOOL_WORKDIR"] = context.working_dir_container
    return container_env


_AGENT_CONTAINER_BASE_ENTRYPOINT_SCRIPT = (
    # Docker bind mounts can present the rollout worktree with a UID that Git
    # distrusts; scope the exception to the mounted cwd before starting the CLI.
    'if [ -n "${APEX_TARGET_TOOL_WORKDIR:-}" ] && command -v git >/dev/null 2>&1; then '
    "git config --global --get-all safe.directory 2>/dev/null "
    '| grep -Fx -- "$APEX_TARGET_TOOL_WORKDIR" >/dev/null 2>&1 '
    '|| git config --global --add safe.directory "$APEX_TARGET_TOOL_WORKDIR" '
    ">/dev/null 2>&1 || true; "
    "fi; "
    # Agent CLIs get their prompt through argv. Keep stdin closed so CLIs that
    # append piped stdin do not block on Docker's non-TTY exec stream.
    'exec "$@" < /dev/null'
)


def _agent_container_root_setup_entrypoint(base_entrypoint: str) -> str:
    quoted_base = shlex.quote(str(base_entrypoint or 'exec "$@" < /dev/null'))
    prelude = rf"""
if [ -z "${{APEX_AGENT_CONTAINER_AFTER_ROOT_SETUP:-}}" ]; then
  if [ -n "${{APEX_AGENT_CONTAINER_ROOT_SETUP_SCRIPT:-}}" ]; then
    if [ "$(id -u)" != "0" ]; then
      echo "agent container root setup requested but container is not running as root" >&2
      exit 126
    fi
    /bin/sh -eu -c "$APEX_AGENT_CONTAINER_ROOT_SETUP_SCRIPT"
  fi
  if [ -n "${{APEX_AGENT_CONTAINER_RUN_AS_USER:-}}" ] && [ "$(id -u)" = "0" ]; then
    apex_run_user="$APEX_AGENT_CONTAINER_RUN_AS_USER"
    apex_uid="${{apex_run_user%%:*}}"
    apex_gid="${{apex_run_user#*:}}"
    if [ "$apex_gid" = "$apex_run_user" ]; then
      apex_gid="$apex_uid"
    fi
    case "$apex_uid" in
      ""|*[!0-9]*)
        apex_resolved_uid="$(id -u "$apex_uid" 2>/dev/null || true)"
        if [ -z "$apex_resolved_uid" ]; then
          echo "agent container cannot resolve docker user uid: $apex_uid" >&2
          exit 126
        fi
        apex_uid="$apex_resolved_uid"
        ;;
    esac
    case "$apex_gid" in
      ""|*[!0-9]*)
        apex_resolved_gid="$(getent group "$apex_gid" 2>/dev/null | awk -F: '{{print $3}}' || true)"
        if [ -z "$apex_resolved_gid" ]; then
          echo "agent container cannot resolve docker user gid: $apex_gid" >&2
          exit 126
        fi
        apex_gid="$apex_resolved_gid"
        ;;
    esac
    if command -v setpriv >/dev/null 2>&1; then
      exec env APEX_AGENT_CONTAINER_AFTER_ROOT_SETUP=1 APEX_AGENT_CONTAINER_ROOT_SETUP_SCRIPT= APEX_AGENT_CONTAINER_RUN_AS_USER= \
        setpriv --reuid "$apex_uid" --regid "$apex_gid" --clear-groups /bin/sh -c {quoted_base} "$0" "$@"
    fi
    echo "agent container cannot drop from root to $APEX_AGENT_CONTAINER_RUN_AS_USER: setpriv not found" >&2
    exit 126
  fi
fi
"""
    return (prelude.strip() + "\n" + str(base_entrypoint or 'exec "$@" < /dev/null')).strip()


_AGENT_CONTAINER_ENTRYPOINT_SCRIPT = _agent_container_root_setup_entrypoint(
    _AGENT_CONTAINER_BASE_ENTRYPOINT_SCRIPT
)


_AGENT_RUNTIME_STATE_DIRNAME = ".apex_agent_runtime"
_AGENT_RUNTIME_CONTAINER_ROOT = "/apex_agent_runtime"


def _agent_runtime_state_root_for_workspace(working_dir: str | Path) -> Path:
    """Return an Apex-owned runtime state root outside the candidate repo."""

    try:
        workspace = Path(working_dir).expanduser().resolve(strict=False)
    except OSError:
        workspace = Path(working_dir).expanduser().absolute()
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", workspace.name).strip("._-") or "workspace"
    digest = hashlib.sha256(str(workspace).encode("utf-8", errors="replace")).hexdigest()[:12]
    return workspace.parent / _AGENT_RUNTIME_STATE_DIRNAME / f"{safe_name}-{digest}"


def _agent_container_context_with_host_mapping(
    context: _AgentContainerLaunchContext,
    *,
    host_root: Path,
    container_root: str,
) -> _AgentContainerLaunchContext:
    return _AgentContainerLaunchContext(
        docker_bin=context.docker_bin,
        container_name=context.container_name,
        host_root=context.host_root,
        container_root=context.container_root,
        working_dir_container=context.working_dir_container,
        runtime_env=context.runtime_env,
        docker_host_env=context.docker_host_env,
        docker_user=context.docker_user,
        extra_host_path_mappings=(
            *context.extra_host_path_mappings,
            (host_root, container_root),
        ),
    )


def _docker_exec_command_for_agent_container(
    command: list[str],
    env: dict[str, str],
    context: _AgentContainerLaunchContext,
    *,
    backend: LLMBackend | None = None,
    auth_env_keys: Iterable[str] = (),
) -> list[str]:
    docker_command = [
        context.docker_bin,
        "exec",
    ]
    if context.docker_user:
        docker_command.extend(["-u", context.docker_user])
    docker_command.extend(
        [
            "-w",
            context.working_dir_container,
        ]
    )
    container_env = _agent_container_env(env, context)
    for key, value in sorted(container_env.items()):
        if value == "":
            continue
        docker_command.extend(["-e", f"{key}={value}"])
    for key in sorted({str(item) for item in auth_env_keys if item}):
        if key not in container_env and str(env.get(key) or "").strip():
            docker_command.extend(["-e", key])
    container_command = _agent_container_command(command, context, backend=backend)
    docker_command.extend(
        [
            context.container_name,
            "/bin/sh",
            "-c",
            _AGENT_CONTAINER_ENTRYPOINT_SCRIPT,
            "apex-agent-cli",
            *container_command,
        ]
    )
    return docker_command


def _agent_image_launch_context(
    env: dict[str, str],
    *,
    working_dir: str,
) -> tuple[_AgentContainerLaunchContext, dict[str, Any]]:
    target_context = _load_target_runtime_context(env)
    runtime = dict(target_context.get("runtime") or {})
    if str(runtime.get("kind") or target_context.get("mode") or "") != "docker_image":
        raise CLIAgentContainerIsolationError(
            "target runtime requested docker image agent launch without a docker_image runtime"
        )
    image = str(runtime.get("docker_image") or "").strip()
    if not image:
        raise CLIAgentContainerIsolationError(
            "target runtime requested docker image agent launch without an image"
        )
    try:
        host_root = Path(working_dir).expanduser().resolve(strict=False)
    except OSError:
        host_root = Path(working_dir).expanduser().absolute()
    container_root = str(runtime.get("docker_workdir") or "/workspace").rstrip("/")
    if not container_root.startswith("/"):
        raise CLIAgentContainerIsolationError(
            "target runtime requested docker image launch with a non-absolute workdir"
        )
    raw_docker_bin = str(runtime.get("docker_bin") or "docker").strip() or "docker"
    docker_bin = (
        raw_docker_bin
        if os.path.isabs(raw_docker_bin)
        else (shutil.which(raw_docker_bin, path=os.environ.get("PATH", "")) or raw_docker_bin)
    )
    return (
        _AgentContainerLaunchContext(
            docker_bin=docker_bin,
            container_name="",
            host_root=host_root,
            container_root=container_root,
            working_dir_container=container_root,
            runtime_env={
                str(key): str(value) for key, value in dict(runtime.get("docker_env") or {}).items()
            },
            docker_host_env={
                str(key): str(value)
                for key, value in dict(runtime.get("docker_host_env") or {}).items()
            },
            docker_user=str(runtime.get("docker_user") or "").strip(),
        ),
        runtime,
    )


def _agent_image_path_env_value(
    value: str,
    context: _AgentContainerLaunchContext,
) -> str:
    text = str(value or "")
    if not text:
        return ""
    try:
        path = Path(text).expanduser().resolve(strict=False)
    except OSError:
        path = Path(text).expanduser().absolute()
    try:
        return _map_host_path_to_agent_container(path, context)
    except CLIAgentContainerIsolationError:
        return ""


def _agent_image_env(
    env: dict[str, str],
    context: _AgentContainerLaunchContext,
    *,
    state_root: Path,
) -> dict[str, str]:
    container_env: dict[str, str] = dict(context.runtime_env)
    preserve_loopback_proxy = str(
        env.get("APEX_AGENT_CONTAINER_PRESERVE_LOOPBACK_PROXY") or ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    for key, value in env.items():
        if key not in _AGENT_CONTAINER_ENV_ALLOWLIST and not key.startswith(
            _AGENT_CONTAINER_ENV_PREFIX_ALLOWLIST
        ):
            continue
        if key == "PATH":
            continue
        if key in _AGENT_CONTAINER_PROXY_ENV_KEYS:
            container_env[str(key)] = _proxy_value_for_agent_container(
                str(value),
                preserve_loopback=preserve_loopback_proxy,
            )
            continue
        if key in _AGENT_CONTAINER_PATH_ENV_KEYS:
            mapped = _agent_image_path_env_value(str(value), context)
            if mapped:
                container_env[str(key)] = mapped
            continue
        container_env[str(key)] = _map_host_root_text_to_agent_container(str(value), context)
    if context.runtime_env.get("PATH"):
        container_env["PATH"] = str(context.runtime_env.get("PATH") or "")
    container_env["APEX_AGENT_CONTAINER"] = "1"
    container_env["APEX_HOST_DYNAMIC_TOOLS"] = "disabled"
    container_env["APEX_TARGET_TOOL_WORKDIR"] = context.working_dir_container
    return container_env


def _docker_run_command_for_agent_image(
    command: list[str],
    env: dict[str, str],
    context: _AgentContainerLaunchContext,
    runtime: Mapping[str, Any],
    *,
    backend: LLMBackend | None = None,
    auth_env_keys: Iterable[str] = (),
) -> list[str]:
    image = str(runtime.get("docker_image") or "").strip()
    network = str(runtime.get("docker_network") or "none").strip() or "none"
    docker_command = [context.docker_bin, "run", "--rm", "--network", network]
    root_setup_script = str(runtime.get("docker_root_setup_script") or "").strip()
    if context.docker_user and not root_setup_script:
        docker_command.extend(["-u", context.docker_user])
    platform = str(runtime.get("docker_platform") or "").strip()
    if platform:
        docker_command.extend(["--platform", platform])
    workdir = context.host_root
    docker_command.extend(
        [
            "--mount",
            f"type=bind,source={workdir},target={context.container_root}",
        ]
    )
    state_root = _agent_runtime_state_root_for_workspace(workdir)
    state_root.mkdir(parents=True, exist_ok=True)
    # Agent-owned prompt/home/output state is host-backed but should not create a
    # host-looking path tree inside the solve container.
    context = _agent_container_context_with_host_mapping(
        context,
        host_root=state_root,
        container_root=_AGENT_RUNTIME_CONTAINER_ROOT,
    )
    docker_command.extend(
        [
            "--mount",
            f"type=bind,source={state_root},target={_AGENT_RUNTIME_CONTAINER_ROOT}",
        ]
    )
    for raw_mount in list(runtime.get("docker_mounts") or []):
        if not isinstance(raw_mount, Mapping):
            continue
        source = str(raw_mount.get("source") or "").strip()
        target = str(raw_mount.get("target") or "").strip()
        if not source or not target:
            continue
        option = f"type=bind,source={source},target={target}"
        if str(raw_mount.get("readonly") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
            "readonly",
            "ro",
        }:
            option += ",readonly"
        docker_command.extend(["--mount", option])
    docker_command.extend(["-w", context.working_dir_container])
    container_env = _agent_image_env(env, context, state_root=state_root)
    if root_setup_script:
        container_env["APEX_AGENT_CONTAINER_ROOT_SETUP_SCRIPT"] = root_setup_script
        if context.docker_user:
            container_env["APEX_AGENT_CONTAINER_RUN_AS_USER"] = context.docker_user
    for key, value in sorted(container_env.items()):
        if value == "":
            continue
        docker_command.extend(["-e", f"{key}={value}"])
    for key in sorted({str(item) for item in auth_env_keys if item}):
        if key not in container_env and str(env.get(key) or "").strip():
            docker_command.extend(["-e", key])
    container_command = _agent_container_command(command, context, backend=backend)
    docker_command.extend(
        [
            "--entrypoint",
            "/bin/sh",
            image,
            "-c",
            _AGENT_CONTAINER_ENTRYPOINT_SCRIPT,
            "apex-agent-cli",
            *container_command,
        ]
    )
    return docker_command


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _TargetToolBridge:
    """Narrow host bridge for Docker Sandbox agents.

    The agent sandbox does not receive the Docker socket. Its PATH shims POST
    tool invocations here; the bridge re-enters the normal target-runtime shim
    on the host, which then executes inside the declared benchmark container.
    """

    def __init__(
        self,
        env: dict[str, str],
        *,
        working_dir: str,
        descriptor_path: Path,
    ) -> None:
        self.env = dict(env)
        self.working_dir = str(Path(working_dir).expanduser().resolve(strict=False))
        self.descriptor_path = descriptor_path
        self.token = secrets.token_urlsafe(32)
        self.server: _ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.url = ""

    def __enter__(self) -> "_TargetToolBridge":
        bridge = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return

            def do_POST(self) -> None:  # noqa: N802
                bridge._handle(self)

        # Docker Sandbox reaches the host via host.docker.internal; bind beyond
        # loopback while retaining the per-launch bearer token below.
        self.server = _ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        port = int(self.server.server_address[1])
        # Docker Desktop sandboxes reach the host through host.docker.internal.
        self.url = f"http://host.docker.internal:{port}/target-tool"
        local_url = f"http://127.0.0.1:{port}/target-tool"
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="apex-target-tool-bridge",
            daemon=True,
        )
        self.thread.start()
        self.descriptor_path.parent.mkdir(parents=True, exist_ok=True)
        self.descriptor_path.write_text(
            json.dumps(
                {"url": self.url, "local_url": local_url, "token": self.token},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            self.descriptor_path.chmod(0o600)
        except OSError:
            logger.debug("Failed to chmod target-tool bridge descriptor", exc_info=True)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            if self.server is not None:
                self.server.shutdown()
                self.server.server_close()
        finally:
            if self.thread is not None:
                self.thread.join(timeout=2)
            try:
                self.descriptor_path.unlink()
            except OSError:
                pass

    def _json_response(
        self,
        handler: http.server.BaseHTTPRequestHandler,
        status: int,
        payload: Mapping[str, Any],
    ) -> None:
        raw = json.dumps(dict(payload)).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(raw)))
        handler.end_headers()
        handler.wfile.write(raw)

    def _handle(self, handler: http.server.BaseHTTPRequestHandler) -> None:
        if handler.path != "/target-tool":
            self._json_response(
                handler,
                404,
                {"returncode": 113, "stdout": "", "stderr": "unknown target-tool bridge path"},
            )
            return
        if handler.headers.get("X-APEX-Target-Tool-Bridge-Token", "") != self.token:
            self._json_response(
                handler,
                403,
                {"returncode": 113, "stdout": "", "stderr": "invalid target-tool bridge token"},
            )
            return
        try:
            length = min(int(handler.headers.get("Content-Length", "0") or "0"), 16_000_000)
        except ValueError:
            length = 0
        try:
            payload = json.loads(handler.rfile.read(length).decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            self._json_response(
                handler,
                400,
                {"returncode": 113, "stdout": "", "stderr": "malformed target-tool bridge JSON"},
            )
            return
        if not isinstance(payload, dict):
            self._json_response(
                handler,
                400,
                {"returncode": 113, "stdout": "", "stderr": "malformed target-tool bridge body"},
            )
            return
        tool = Path(str(payload.get("tool") or "")).name
        try:
            from apex.evaluation import target_runtime

            allowed_tools = {
                *target_runtime.STATIC_READ_ONLY_TOOLS,
                *target_runtime.DYNAMIC_TOOL_NAMES,
            }
        except Exception:
            allowed_tools = set()
        if tool not in allowed_tools:
            self._json_response(
                handler,
                400,
                {
                    "returncode": 113,
                    "stdout": "",
                    "stderr": f"unsupported target-tool bridge tool: {tool}",
                },
            )
            return
        args = payload.get("args")
        if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
            self._json_response(
                handler,
                400,
                {"returncode": 113, "stdout": "", "stderr": "malformed target-tool arguments"},
            )
            return
        context_path = str(
            self.env.get("APEX_TARGET_TOOL_BRIDGE_HOST_CONTEXT")
            or self.env.get("APEX_TARGET_TOOL_CONTEXT")
            or ""
        ).strip()
        try:
            context = json.loads(Path(context_path).read_text(encoding="utf-8"))
            shim_dir = Path(str(context.get("shim_dir") or "")).expanduser().resolve(strict=False)
            shim_path = shim_dir / tool
            shim_path.resolve(strict=False).relative_to(shim_dir)
        except Exception as exc:
            self._json_response(
                handler,
                500,
                {
                    "returncode": 113,
                    "stdout": "",
                    "stderr": f"target-tool bridge context invalid: {exc}",
                },
            )
            return
        if not shim_path.exists():
            self._json_response(
                handler,
                500,
                {
                    "returncode": 113,
                    "stdout": "",
                    "stderr": f"target-tool bridge shim missing: {tool}",
                },
            )
            return
        bridge_env = dict(os.environ)
        bridge_env.update(self.env)
        bridge_env["APEX_TARGET_TOOL_BRIDGE_LOCAL"] = "1"
        bridge_env["APEX_TARGET_TOOL_CONTEXT"] = context_path
        bridge_env["APEX_TARGET_TOOL_WORKDIR"] = self.working_dir
        bridge_env["PWD"] = self.working_dir
        try:
            timeout_seconds = max(1, int(context.get("timeout_seconds") or 60))
        except (TypeError, ValueError):
            timeout_seconds = 60
        try:
            completed = subprocess.run(
                [str(shim_path), *args],
                input=str(payload.get("stdin") or ""),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                cwd=self.working_dir,
                env=bridge_env,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _coerce_subprocess_text(exc.output)
            stderr = _coerce_subprocess_text(exc.stderr)
            self._json_response(
                handler,
                200,
                {
                    "returncode": 124,
                    "stdout": stdout,
                    "stderr": stderr
                    + ("\n" if stderr else "")
                    + f"target-tool bridge command timed out after {timeout_seconds}s",
                },
            )
            return
        except OSError as exc:
            self._json_response(
                handler,
                200,
                {
                    "returncode": 113,
                    "stdout": "",
                    "stderr": f"target-tool bridge failed to start shim: {exc}",
                },
            )
            return
        self._json_response(
            handler,
            200,
            {
                "returncode": int(completed.returncode or 0),
                "stdout": completed.stdout or "",
                "stderr": completed.stderr or "",
            },
        )


def _write_agent_visible_target_tool_shims(
    env: dict[str, str],
    *,
    working_dir: str,
) -> tuple[dict[str, str], Path]:
    """Copy target-tool shims into the Docker Sandbox-mounted workspace."""

    if not env.get("APEX_TARGET_TOOL_CONTEXT"):
        return env, Path()
    try:
        from apex.evaluation import target_runtime
    except Exception as exc:  # pragma: no cover - import should always work here
        raise CLIAgentContainerIsolationError(
            f"Could not prepare Docker Sandbox target shims: {exc}"
        ) from exc
    workspace = Path(working_dir).expanduser().resolve(strict=False)
    shim_dir = workspace / ".apex_docker_sandbox_target_tools"
    shim_dir.mkdir(parents=True, exist_ok=True)
    original_context = _load_target_runtime_context(env)
    original_context_path = str(env.get("APEX_TARGET_TOOL_CONTEXT") or "").strip()
    context_path = shim_dir / "context.json"
    context_payload = dict(original_context)
    context_payload["shim_dir"] = str(shim_dir)
    context_payload["context_path"] = str(context_path)
    context_payload["workdir"] = str(workspace)
    context_path.write_text(json.dumps(context_payload, indent=2) + "\n", encoding="utf-8")
    runner_path = shim_dir / "apex_target_tool.py"
    runner_path.write_text(target_runtime._runner_source(), encoding="utf-8")
    runner_path.chmod(0o755)
    for tool_name in (
        *target_runtime.STATIC_READ_ONLY_TOOLS,
        *target_runtime.DYNAMIC_TOOL_NAMES,
    ):
        target = shim_dir / tool_name
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            target.symlink_to(runner_path.name)
        except OSError:
            shutil.copy2(runner_path, target)
        target.chmod(0o755)
    launch_env = dict(env)
    launch_env["APEX_TARGET_TOOL_CONTEXT"] = str(context_path)
    launch_env["APEX_TARGET_TOOL_BRIDGE_HOST_CONTEXT"] = original_context_path
    launch_env["APEX_TARGET_TOOL_WORKDIR"] = str(workspace)
    launch_env["APEX_HOST_DYNAMIC_TOOLS"] = "disabled"
    launch_env["APEX_TARGET_TOOL_BRIDGE_FILE"] = str(shim_dir / "bridge.json")
    launch_env["PATH"] = str(shim_dir) + os.pathsep + _DOCKER_SANDBOX_SYSTEM_PATH
    return launch_env, shim_dir


def _docker_sandbox_agent_name(config: LLMConfig) -> str:
    agent = _DOCKER_SANDBOX_AGENT_BY_BACKEND.get(config.backend, "")
    if not agent:
        raise CLIAgentContainerIsolationError(
            f"Docker Sandbox target-runtime auth is not supported for {config.backend.value}."
        )
    return agent


def _docker_sandbox_env_options(config: LLMConfig, env: dict[str, str]) -> list[str]:
    env_keys = {
        "APEX_HOST_DYNAMIC_TOOLS",
        "APEX_TARGET_TOOL_BRIDGE_FILE",
        "APEX_TARGET_TOOL_CONTEXT",
        "APEX_TARGET_TOOL_WORKDIR",
        "CLAUDE_CODE_VERSION_OVERRIDE",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "NO_COLOR",
        "PATH",
        "TERM",
    }
    env_keys.update(key for key in env if str(key).startswith("LC_"))
    options: list[str] = []
    for key in sorted(env_keys):
        value = str(env.get(key) or "")
        if value:
            options.extend(["--env", f"{key}={value}"])
    return options


def _docker_sandbox_name(config: LLMConfig, *, working_dir: str) -> str:
    backend = re.sub(r"[^A-Za-z0-9_.+-]+", "-", config.backend.value).strip("-")
    try:
        workspace = str(Path(working_dir).expanduser().resolve(strict=False))
    except OSError:
        workspace = str(Path(working_dir).expanduser().absolute())
    digest = hashlib.sha256(
        f"{backend}:{workspace}:{time.time_ns()}:{secrets.token_hex(8)}".encode("utf-8")
    ).hexdigest()[:16]
    return f"apex-{backend}-{digest}"


def _docker_sandbox_create_timeout_seconds(env: dict[str, str]) -> int:
    raw = str(env.get("APEX_DOCKER_SANDBOX_CREATE_TIMEOUT_SECONDS") or "").strip()
    try:
        value = int(raw) if raw else 900
    except ValueError:
        value = 900
    return max(60, value)


def _create_docker_sandbox_for_agent(
    config: LLMConfig,
    env: dict[str, str],
    *,
    working_dir: str,
    sandbox_name: str,
) -> None:
    docker_bin = shutil.which("docker", path=os.environ.get("PATH", "")) or "docker"
    agent = _docker_sandbox_agent_name(config)
    command = [
        docker_bin,
        "sandbox",
        "create",
        "--quiet",
        "--name",
        sandbox_name,
        agent,
        str(Path(working_dir).expanduser().resolve(strict=False)),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_docker_sandbox_create_timeout_seconds(env),
            cwd=working_dir,
            env=dict(os.environ),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CLIAgentContainerIsolationError(
            f"Docker Sandbox create for {agent} timed out after {int(exc.timeout)}s."
        ) from exc
    except OSError as exc:
        raise CLIAgentContainerIsolationError(
            f"Docker Sandbox create for {agent} failed to start: {exc}"
        ) from exc
    if completed.returncode != 0:
        detail = _compact_cli_probe_detail((completed.stdout or "") + (completed.stderr or ""))
        suffix = f": {detail}" if detail else ""
        raise CLIAgentContainerIsolationError(f"Docker Sandbox create for {agent} failed{suffix}.")


def _remove_docker_sandbox(sandbox_name: str) -> None:
    if not sandbox_name:
        return
    docker_bin = shutil.which("docker", path=os.environ.get("PATH", "")) or "docker"
    try:
        subprocess.run(
            [docker_bin, "sandbox", "rm", sandbox_name],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env=dict(os.environ),
        )
    except Exception:
        logger.debug("Failed to remove Docker Sandbox %s", sandbox_name, exc_info=True)


def _docker_sandbox_exec_command(
    config: LLMConfig,
    command: list[str],
    env: dict[str, str],
    *,
    sandbox_name: str,
    working_dir: str,
) -> list[str]:
    docker_bin = shutil.which("docker", path=os.environ.get("PATH", "")) or "docker"
    agent = _docker_sandbox_agent_name(config)
    agent_args = list(command[1:]) if command else []
    return [
        docker_bin,
        "sandbox",
        "exec",
        "-i",
        "-w",
        str(Path(working_dir).expanduser().resolve(strict=False)),
        *_docker_sandbox_env_options(config, env),
        sandbox_name,
        agent,
        *agent_args,
    ]


def _claude_cli_effort_args(config: LLMConfig) -> list[str]:
    """Return Claude Code effort args unless the operator supplied them."""

    explicit_args = [str(arg) for arg in getattr(config, "cli_args", [])]
    for index, arg in enumerate(explicit_args):
        if arg == "--effort" and index + 1 < len(explicit_args):
            return []
        if arg.startswith("--effort="):
            return []
    raw_effort = os.environ.get("APEX_CLAUDE_CLI_DEFAULT_EFFORT", _CLAUDE_CLI_DEFAULT_EFFORT)
    effort = str(raw_effort or "").strip().lower()
    if effort in {"", "none", "off", "0", "false"}:
        return []
    if effort not in _CLAUDE_CLI_EFFORT_LEVELS:
        allowed = ", ".join(sorted(_CLAUDE_CLI_EFFORT_LEVELS))
        raise ValueError(
            f"Invalid APEX_CLAUDE_CLI_DEFAULT_EFFORT={raw_effort!r}; expected one of: {allowed}."
        )
    return ["--effort", effort]


def _configured_claude_target_runtime_warmup_timeout_seconds(
    config: LLMConfig | None = None,
) -> float:
    raw_value = str(
        os.environ.get("APEX_TARGET_RUNTIME_CLAUDE_WARMUP_TIMEOUT_SECONDS") or ""
    ).strip()
    if not raw_value:
        configured_timeout = (
            getattr(config, "cli_target_runtime_warmup_timeout_seconds", None)
            if config is not None
            else None
        )
        if configured_timeout is not None and float(configured_timeout) > 0:
            return max(1.0, float(configured_timeout))
        return _CLAUDE_TARGET_RUNTIME_WARMUP_DEFAULT_TIMEOUT_SECONDS
    try:
        return max(1.0, float(raw_value))
    except ValueError:
        logger.warning(
            "Ignoring invalid float APEX_TARGET_RUNTIME_CLAUDE_WARMUP_TIMEOUT_SECONDS=%r",
            raw_value,
        )
        return _CLAUDE_TARGET_RUNTIME_WARMUP_DEFAULT_TIMEOUT_SECONDS


def _claude_target_runtime_warmup_sentinel_path(env: Mapping[str, str]) -> Path | None:
    raw_config_dir = str(env.get("CLAUDE_CONFIG_DIR") or "").strip()
    if not raw_config_dir:
        return None
    return Path(raw_config_dir).expanduser() / _CLAUDE_TARGET_RUNTIME_WARMUP_SENTINEL


def _claude_target_runtime_warmup_scope_key(env: Mapping[str, str]) -> str:
    raw_context = str(env.get("APEX_TARGET_TOOL_CONTEXT") or "").strip()
    if raw_context:
        try:
            context = _load_target_runtime_context(dict(env))
        except Exception:  # noqa: BLE001 - best-effort cooldown key only
            context = {}
        runtime = context.get("runtime") if isinstance(context, Mapping) else {}
        if isinstance(runtime, Mapping):
            container_name = str(runtime.get("docker_container_name") or "").strip()
            if container_name:
                return f"docker-container:{container_name}"
            root = str(runtime.get("docker_host_workdir_root") or "").strip()
            if root:
                return f"docker-root:{root}"
        workdir = str(context.get("workdir") or "").strip() if isinstance(context, Mapping) else ""
        if workdir:
            return f"target-workdir:{workdir}"
        return f"context-file:{raw_context}"
    raw_config_dir = str(env.get("CLAUDE_CONFIG_DIR") or "").strip()
    return f"claude-config:{raw_config_dir}" if raw_config_dir else ""


def _claude_target_runtime_warmup_is_suppressed(scope_key: str) -> bool:
    if not scope_key:
        return False
    now = time.time()
    with _CLAUDE_TARGET_RUNTIME_WARMUP_COOLDOWN_LOCK:
        until = _CLAUDE_TARGET_RUNTIME_WARMUP_COOLDOWNS.get(scope_key)
        if until is None:
            return False
        if until > now:
            return True
        _CLAUDE_TARGET_RUNTIME_WARMUP_COOLDOWNS.pop(scope_key, None)
    return False


def _record_claude_target_runtime_warmup_failure(scope_key: str) -> None:
    if not scope_key:
        return
    until = time.time() + _CLAUDE_TARGET_RUNTIME_WARMUP_FAILURE_COOLDOWN_SECONDS
    with _CLAUDE_TARGET_RUNTIME_WARMUP_COOLDOWN_LOCK:
        _CLAUDE_TARGET_RUNTIME_WARMUP_COOLDOWNS[scope_key] = until


def _clear_claude_target_runtime_warmup_failure(scope_key: str) -> None:
    if not scope_key:
        return
    with _CLAUDE_TARGET_RUNTIME_WARMUP_COOLDOWN_LOCK:
        _CLAUDE_TARGET_RUNTIME_WARMUP_COOLDOWNS.pop(scope_key, None)


def _claude_target_runtime_resume_session_available(
    env: Mapping[str, str],
    *,
    session_id: str,
) -> bool:
    """Return True when Claude's isolated home has a local resumable session."""

    session_id = str(session_id or "").strip()
    raw_config_dir = str(env.get("CLAUDE_CONFIG_DIR") or "").strip()
    if not session_id or not raw_config_dir:
        return False
    try:
        config_dir = Path(raw_config_dir).expanduser().resolve(strict=False)
    except OSError:
        return False
    candidate_roots = (config_dir / "sessions", config_dir / "projects")
    scanned = 0
    for root in candidate_roots:
        try:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                scanned += 1
                if scanned > 2000:
                    return False
                if session_id in path.name:
                    return True
        except OSError:
            continue
    return False


def _reset_claude_target_runtime_state_after_startup_failure(
    env: Mapping[str, str],
    *,
    working_dir: str,
) -> None:
    if not str(env.get("APEX_TARGET_TOOL_CONTEXT") or "").strip():
        return
    raw_config_dir = str(env.get("CLAUDE_CONFIG_DIR") or "").strip()
    if not raw_config_dir:
        return
    try:
        workspace = Path(working_dir).expanduser().resolve(strict=False)
        runtime_home = (
            _agent_runtime_state_root_for_workspace(workspace)
            / ".cli_homes"
            / LLMBackend.CLAUDE_CLI.value
        ).resolve(strict=False)
        config_dir = Path(raw_config_dir).expanduser().resolve(strict=False)
        config_dir.relative_to(runtime_home)
    except (OSError, ValueError):
        logger.debug(
            "Skipping Claude target-runtime state reset outside Apex CLI home: %s",
            raw_config_dir,
        )
        return
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
        for child in list(config_dir.iterdir()):
            if child.name in _CLAUDE_TARGET_RUNTIME_RETRY_STATE_PRESERVE_NAMES:
                continue
            try:
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
            except OSError:
                logger.debug(
                    "Failed to remove Claude target-runtime retry state %s",
                    child,
                    exc_info=True,
                )
    except OSError:
        logger.debug(
            "Failed to reset Claude target-runtime state under %s",
            config_dir,
            exc_info=True,
        )


def _build_claude_target_runtime_warmup_command(config: LLMConfig) -> list[str]:
    return [config.resolved_cli_command, "--version"]


def _run_claude_target_runtime_warmup(
    config: LLMConfig,
    env: dict[str, str],
    *,
    working_dir: str,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> None:
    if config.backend != LLMBackend.CLAUDE_CLI:
        return
    if not _env_flag_enabled("APEX_TARGET_RUNTIME_CLAUDE_WARMUP"):
        return
    sentinel = _claude_target_runtime_warmup_sentinel_path(env)
    raw_home = str(env.get("HOME") or "").strip()
    if sentinel is None or not raw_home:
        return
    if sentinel.exists():
        return
    scope_key = _claude_target_runtime_warmup_scope_key(env)
    if _claude_target_runtime_warmup_is_suppressed(scope_key):
        logger.info(
            "Skipping Claude target-runtime warmup for %s after recent warmup failure in %s",
            working_dir,
            scope_key,
        )
        return
    home_path = Path(raw_home).expanduser()
    try:
        home_path.mkdir(parents=True, exist_ok=True)
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        warmup_context = _agent_container_launch_context(env, working_dir=str(home_path))
        if warmup_context is None:
            return
        command = _build_claude_target_runtime_warmup_command(config)
        warmup_command = _docker_exec_command_for_agent_container(
            command,
            env,
            warmup_context,
            backend=config.backend,
            auth_env_keys=_BACKEND_AUTH_ALLOWLIST.get(config.backend, ()),
        )
        warmup_env = dict(env)
        warmup_env.update(warmup_context.docker_host_env)
        warmup_cwd = os.path.abspath(os.sep)
    except CLIAgentContainerIsolationError as exc:
        logger.info(
            "Skipping Claude target-runtime warmup for %s: %s",
            working_dir,
            exc,
        )
        return
    except OSError as exc:
        logger.info(
            "Skipping Claude target-runtime warmup for %s: %s",
            working_dir,
            exc,
        )
        return

    if progress_callback is not None:
        progress_callback(
            {
                "state": "startup_warmup",
                "working_dir": working_dir,
                "last_progress_at": time.time(),
                "last_progress_source": "startup_warmup",
            }
        )
    timeout_seconds = _configured_claude_target_runtime_warmup_timeout_seconds(config)
    try:
        completed = subprocess.run(
            warmup_command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=warmup_cwd,
            env=warmup_env,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        _record_claude_target_runtime_warmup_failure(scope_key)
        logger.warning(
            "Claude target-runtime warmup timed out after %.1fs for %s",
            timeout_seconds,
            working_dir,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "state": "startup_warmup_timeout",
                    "working_dir": working_dir,
                    "last_progress_at": time.time(),
                    "last_progress_source": "startup_warmup_timeout",
                }
            )
        try:
            raw_output = _coerce_subprocess_text(exc.output)
            raw_error = _coerce_subprocess_text(exc.stderr)
            if raw_output or raw_error:
                logger.debug(
                    "Timed-out Claude target-runtime warmup output for %s: stdout=%r stderr=%r",
                    working_dir,
                    raw_output[-2000:],
                    raw_error[-2000:],
                )
        except Exception:  # pragma: no cover - logging only
            pass
        return
    except OSError as exc:
        logger.info(
            "Claude target-runtime warmup launch failed for %s: %s",
            working_dir,
            exc,
        )
        return

    stdout = _coerce_subprocess_text(completed.stdout)
    stderr = _coerce_subprocess_text(completed.stderr)
    if completed.returncode == 0:
        _clear_claude_target_runtime_warmup_failure(scope_key)
        try:
            sentinel.write_text(
                json.dumps(
                    {
                        "warmed_at": time.time(),
                        "backend": config.backend.value,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        except OSError:
            logger.debug(
                "Failed to write Claude target-runtime warmup sentinel %s",
                sentinel,
                exc_info=True,
            )
        if progress_callback is not None:
            progress_callback(
                {
                    "state": "startup_warmup_complete",
                    "working_dir": working_dir,
                    "last_progress_at": time.time(),
                    "last_progress_source": "startup_warmup_complete",
                }
            )
        return
    _record_claude_target_runtime_warmup_failure(scope_key)
    logger.warning(
        "Claude target-runtime warmup did not complete cleanly for %s "
        "(returncode=%s stdout=%r stderr=%r); continuing with normal CLI retry handling",
        working_dir,
        completed.returncode,
        stdout[-2000:],
        stderr[-2000:],
    )


def _codex_output_temp_file(
    *,
    target_runtime_enforced: bool,
    working_dir: str,
    in_workspace: bool = False,
) -> tempfile._TemporaryFileWrapper[str]:
    if not target_runtime_enforced:
        return tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt")
    try:
        workspace = Path(working_dir).expanduser().resolve(strict=False)
    except OSError:
        workspace = Path(working_dir).expanduser().absolute()
    if in_workspace:
        output_dir = workspace / ".apex_cli_outputs"
    else:
        output_dir = _agent_runtime_state_root_for_workspace(workspace) / ".cli_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", dir=str(output_dir))


_TARGET_RUNTIME_WORKSPACE_CLI_HOME_BACKENDS = set(_CLI_BACKEND_SANDBOX_SPECS)


def _relocate_cli_home_for_target_runtime(
    config: LLMConfig,
    env: dict[str, str],
    *,
    working_dir: str,
    copy_auth_state: bool = True,
) -> bool:
    if config.backend not in _TARGET_RUNTIME_WORKSPACE_CLI_HOME_BACKENDS or not env.get(
        "APEX_TARGET_TOOL_CONTEXT"
    ):
        return False
    try:
        workspace = Path(working_dir).expanduser().resolve(strict=False)
    except OSError:
        workspace = Path(working_dir).expanduser().absolute()
    spec = _cli_backend_sandbox_spec(config.backend)
    source_env = dict(env)
    home_path = (
        _agent_runtime_state_root_for_workspace(workspace)
        / ".cli_homes"
        / str(config.backend.value)
    )
    for rel_path in (
        ".config",
        ".cache",
        ".local/share",
        ".local/state",
        "Library/Caches",
    ):
        (home_path / rel_path).mkdir(parents=True, exist_ok=True)
    for rel_path in spec.target_runtime_home.values():
        target_path = home_path / rel_path
        target_path.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "HOME": str(home_path),
            "XDG_CONFIG_HOME": str(home_path / ".config"),
            "XDG_CACHE_HOME": str(home_path / ".cache"),
            "XDG_DATA_HOME": str(home_path / ".local/share"),
            "XDG_STATE_HOME": str(home_path / ".local/state"),
        }
    )
    for key, rel_path in spec.target_runtime_home.items():
        env[key] = str(home_path / rel_path)
    for key, value in spec.target_env_defaults.items():
        env.setdefault(key, value)
    public_config_files: list[Path] = []
    if config.backend == LLMBackend.CODEX_CLI:
        _seed_isolated_codex_home(
            home_path,
            copy_host_config=False,
            copy_host_auth=copy_auth_state,
            config_dir=Path(env["CODEX_HOME"]),
            include_service_tier=False,
        )
        _harden_codex_target_runtime_home(Path(env["CODEX_HOME"]))
        public_config_files.append(Path(env["CODEX_HOME"]) / "config.toml")
    has_auth_state = False
    if copy_auth_state:
        has_auth_state = _materialize_cli_auth_state_for_target_runtime(
            spec,
            env,
            home_path=home_path,
            source_env=source_env,
        )
    else:
        env.pop(spec.auth_state_env_key, None)
    if config.backend == LLMBackend.GEMINI_CLI:
        _write_gemini_target_runtime_settings(
            env,
            home_path=home_path,
            has_auth_state=has_auth_state,
        )
        public_config_files.append(home_path / ".gemini" / "settings.json")
    if _is_opencode_family_backend(config.backend):
        _write_opencode_family_isolated_config(config, env, working_dir=working_dir)
        public_config_files.append(Path(env["OPENCODE_CONFIG"]))
    _ensure_target_runtime_cli_home_container_permissions(
        spec,
        home_path,
        public_config_files=public_config_files,
    )
    return has_auth_state


def _relocate_codex_home_for_target_runtime(
    config: LLMConfig,
    env: dict[str, str],
    *,
    working_dir: str,
) -> None:
    _relocate_cli_home_for_target_runtime(config, env, working_dir=working_dir)


def _cli_sandbox_writable_roots(
    env: dict[str, str],
    *,
    working_dir: str,
    include_cli_home: bool = True,
) -> list[str]:
    """Return explicit non-workspace roots the nested CLI sandbox may need."""

    roots: list[str] = []
    cli_home_keys = (
        "HOME",
        "CODEX_HOME",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "CLAUDE_CONFIG_DIR",
        "GEMINI_CLI_HOME",
        "OPENCODE_CONFIG_DIR",
        "GOOGLE_APPLICATION_CREDENTIALS",
    )
    if include_cli_home:
        for key in cli_home_keys:
            value = str(env.get(key) or "").strip()
            if value:
                roots.append(value)
    context_path = str(env.get("APEX_TARGET_TOOL_CONTEXT") or "").strip()
    if context_path:
        roots.append(str(Path(context_path).expanduser().parent))
    return _normalize_sandbox_roots(roots, working_dir=working_dir)


def _normalize_sandbox_roots(
    roots: Iterable[str],
    *,
    working_dir: str,
) -> list[str]:
    workspace = Path(working_dir).expanduser().resolve(strict=False)
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_root in roots:
        text = str(raw_root or "").strip()
        if not text:
            continue
        try:
            root = Path(text).expanduser().resolve(strict=False)
        except OSError:
            continue
        if root == workspace:
            continue
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


def _prepare_air_gapped_cli_home(
    config: LLMConfig,
    env: dict[str, str],
    *,
    internet_enabled: bool,
) -> None:
    """Warm backend-managed native state before parallel air-gapped prompts."""

    if internet_enabled or config.backend != LLMBackend.CLAUDE_CLI:
        return
    target_context = _load_target_runtime_context(env)
    target_runtime = dict(target_context.get("runtime") or {})
    if str(target_runtime.get("kind") or target_context.get("mode") or "") == "docker_exec":
        return
    home_path = str(env.get("HOME") or "").strip()
    if not home_path:
        return
    version_override = str(env.get("CLAUDE_CODE_VERSION_OVERRIDE") or "").strip()
    cache_key = (config.resolved_cli_command, home_path, version_override)

    def _run_prepare_probe(probe_env: dict[str, str]) -> tuple[bool, str]:
        command = _prepare_cli_command_for_target_tool_path(
            [config.resolved_cli_command, "--version"],
            probe_env,
        )
        launch_env = _cli_launch_env_for_target_runtime(probe_env, command)
        launch_env["PWD"] = str(Path(home_path).expanduser().resolve(strict=False))
        try:
            probe = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_cli_health_probe_timeout_seconds(config),
                env=launch_env,
                cwd=home_path,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, f"timed out for {home_path}"
        except OSError as exc:
            return False, f"failed to start for {home_path}: {exc}"
        except Exception as exc:  # pragma: no cover - best-effort preparation
            return False, f"failed for {home_path}: {exc}"
        if probe.returncode == 0:
            return True, ""
        return (
            False,
            _summarize_cli_probe_failure(
                " ".join(command),
                probe.stdout,
                probe.stderr,
                probe.returncode,
            ),
        )

    with _AIR_GAPPED_CLI_PREP_LOCK:
        if cache_key in _AIR_GAPPED_CLI_PREPARED_WITHOUT_VERSION:
            env.pop("CLAUDE_CODE_VERSION_OVERRIDE", None)
            return
        if cache_key in _AIR_GAPPED_CLI_PREPARED:
            return

        prepared, reason = _run_prepare_probe(env)
        if prepared:
            _AIR_GAPPED_CLI_PREPARED.add(cache_key)
            return

        if version_override.lower() == "latest":
            fallback_env = dict(env)
            fallback_env.pop("CLAUDE_CODE_VERSION_OVERRIDE", None)
            fallback_key = (config.resolved_cli_command, home_path, "")
            if fallback_key in _AIR_GAPPED_CLI_PREPARED:
                env.pop("CLAUDE_CODE_VERSION_OVERRIDE", None)
                _AIR_GAPPED_CLI_PREPARED_WITHOUT_VERSION.add(cache_key)
                return
            fallback_prepared, fallback_reason = _run_prepare_probe(fallback_env)
            if fallback_prepared:
                env.pop("CLAUDE_CODE_VERSION_OVERRIDE", None)
                _AIR_GAPPED_CLI_PREPARED.add(fallback_key)
                _AIR_GAPPED_CLI_PREPARED_WITHOUT_VERSION.add(cache_key)
                logger.info(
                    "Air-gapped Claude home preparation fell back from latest native "
                    "version resolution to the installed launcher default for %s.",
                    home_path,
                )
                return
            logger.warning(
                "Air-gapped Claude home preparation failed for %s with latest override "
                "(%s) and installed launcher default (%s).",
                home_path,
                reason,
                fallback_reason,
            )
            return

        logger.warning("Air-gapped Claude home preparation failed for %s: %s", home_path, reason)


def _sanitize_target_runtime_prompt_tool_paths(prompt: str) -> str:
    """Rewrite absolute dynamic tool paths in prompts to PATH-resolved names."""

    if not prompt:
        return prompt

    def _replace(match: re.Match[str]) -> str:
        return f"{match.group('prefix')}{match.group('tool')}"

    return _TARGET_RUNTIME_PROMPT_ABSOLUTE_TOOL_RE.sub(_replace, prompt)


def _load_target_runtime_context(env: dict[str, str]) -> dict[str, Any]:
    context_path = str(env.get("APEX_TARGET_TOOL_CONTEXT") or "").strip()
    if not context_path:
        return {}
    try:
        loaded = json.loads(Path(context_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _replace_absolute_prompt_path_prefix(prompt: str, raw_path: str) -> str:
    path_text = str(raw_path or "").strip()
    if not path_text:
        return prompt
    pattern = re.compile(
        rf"(?P<prefix>^|[\s`'\"(=\[])"
        rf"{re.escape(path_text)}"
        rf"(?P<suffix>/[^\s`'\"<>\),;:]*)?"
    )

    def _replace(match: re.Match[str]) -> str:
        suffix = str(match.group("suffix") or "")
        replacement = f".{suffix}" if suffix else "$PWD"
        return f"{match.group('prefix')}{replacement}"

    return pattern.sub(_replace, prompt)


def _rebase_absolute_prompt_path_prefix(prompt: str, raw_path: str, replacement_root: str) -> str:
    path_text = str(raw_path or "").strip().rstrip("/")
    replacement = str(replacement_root or "").strip().rstrip("/")
    if not path_text or not replacement:
        return prompt
    pattern = re.compile(
        rf"(?P<prefix>^|[\s`'\"(=\[])"
        rf"{re.escape(path_text)}"
        rf"(?P<suffix>/[^\s`'\"<>\),;:]*)?"
    )

    def _replace(match: re.Match[str]) -> str:
        suffix = str(match.group("suffix") or "")
        return f"{match.group('prefix')}{replacement}{suffix}"

    return pattern.sub(_replace, prompt)


def _sanitize_target_runtime_prompt_workspace_paths(
    prompt: str,
    *,
    env: dict[str, str],
) -> str:
    """Rebase source-worktree paths in target-runtime prompts to the rollout cwd."""

    if not prompt:
        return prompt
    context = _load_target_runtime_context(env)
    raw_workdir = str(context.get("workdir") or "").strip()
    if not raw_workdir:
        return prompt
    try:
        workdir = Path(raw_workdir).expanduser().resolve(strict=False)
    except OSError:
        workdir = Path(raw_workdir).expanduser().absolute()
    sanitized = _replace_absolute_prompt_path_prefix(prompt, str(workdir))
    runtime = dict(context.get("runtime") or {})
    host_root = str(runtime.get("docker_host_workdir_root") or "").strip().rstrip("/")
    container_root = str(runtime.get("docker_container_workdir_root") or "").strip().rstrip("/")
    if host_root and container_root:
        sanitized = _rebase_absolute_prompt_path_prefix(sanitized, host_root, container_root)
        workdir_text = str(workdir)
        if workdir_text == host_root or workdir_text.startswith(host_root + "/"):
            suffix = workdir_text[len(host_root) :].lstrip("/")
            container_workdir = container_root + (("/" + suffix) if suffix else "")
            sanitized = _replace_absolute_prompt_path_prefix(sanitized, container_workdir)
    return sanitized


def _path_is_apex_managed_cli_home(path: str | Path) -> bool:
    try:
        resolved = Path(path).expanduser().resolve(strict=False)
    except OSError:
        resolved = Path(path).expanduser().absolute()
    parts = resolved.parts
    return (
        "cli_airgapped_homes" in parts
        or (".apex_agent_teams" in parts and ".cli_homes" in parts)
        or (_AGENT_RUNTIME_STATE_DIRNAME in parts and ".cli_homes" in parts)
    )


def _prune_target_runtime_cli_shell_snapshots(env: dict[str, str]) -> None:
    """Drop stale backend shell snapshots that can shadow target PATH shims."""

    if not env.get("APEX_TARGET_TOOL_CONTEXT"):
        return
    home_path = str(env.get("HOME") or "").strip()
    if not home_path or not _path_is_apex_managed_cli_home(home_path):
        return
    snapshot_dir = Path(home_path).expanduser() / ".claude" / "shell-snapshots"
    if not snapshot_dir.exists():
        return
    try:
        shutil.rmtree(snapshot_dir)
    except OSError as exc:
        logger.warning(
            "Failed to prune target-runtime CLI shell snapshots under %s: %s",
            snapshot_dir,
            exc,
        )


def _command_invokes_configured_cli(config: LLMConfig, command: list[str]) -> bool:
    if not command:
        return False
    actual = str(command[0] or "")
    expected = str(config.resolved_cli_command or "")
    return actual == expected or Path(actual).name == Path(expected).name


# Minimal config used when the host has no ~/.codex/config.toml. Mirrors the
# settings the user expects from a strongest-default codex invocation:
# xhigh reasoning effort and fast service tier. Trust prompts are silenced
# per-tmp via the --config flag added in `_build_command`.
_CODEX_FALLBACK_CONFIG_TOML = (
    'model_reasoning_effort = "xhigh"\n'
    "metaLauncherPresetPlugins = []\n"
    "[features]\n"
    "plugins = false\n"
    "plugin_hooks = false\n"
    "remote_plugin = false\n"
    "skill_mcp_dependency_install = false\n"
    "\n"
    "[skills.bundled]\n"
    "enabled = false\n"
)
_CODEX_FALLBACK_SERVICE_TIER_TOML = 'service_tier = "fast"\n'


def _codex_fallback_config_toml(*, include_service_tier: bool = True) -> str:
    config = _CODEX_FALLBACK_CONFIG_TOML
    if include_service_tier:
        config += _CODEX_FALLBACK_SERVICE_TIER_TOML
    base_url = _normalized_codex_base_url(os.environ.get("CODEX_BASE_URL") or "")
    if not base_url:
        return config
    return (
        config
        + 'model_provider = "responses"\n'
        + "[model_providers.responses]\n"
        + 'name = "Azure"\n'
        + f'base_url = "{base_url}/codex/passthrough/v1"\n'
        + 'env_key = "OPENAI_API_KEY"\n'
        + 'wire_api = "responses"\n'
    )


def _normalized_codex_base_url(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if normalized == "http://plugboard.x2p.facebook.net":
        # Meta plugboard's Codex responses route is served from plugboardv2.
        normalized = "http://plugboardv2.x2p.facebook.net"
    return normalized


def _codex_provider_config_args(base_url: str) -> list[str]:
    normalized = _normalized_codex_base_url(base_url)
    if not normalized:
        return []
    return [
        "-c",
        'model_provider="responses"',
        "-c",
        'model_providers.responses.name="Azure"',
        "-c",
        f"model_providers.responses.base_url={json.dumps(normalized + '/codex/passthrough/v1')}",
        "-c",
        'model_providers.responses.env_key="OPENAI_API_KEY"',
        "-c",
        'model_providers.responses.wire_api="responses"',
    ]


def _seed_isolated_codex_home(
    home_path: Path,
    *,
    copy_host_config: bool = True,
    copy_host_auth: bool = True,
    config_dir: Optional[Path] = None,
    include_service_tier: bool = True,
) -> None:
    """Best-effort: copy the user's ~/.codex/config.toml + auth.json into the
    isolated CODEX_HOME so reasoning-effort, service-tier, and auth survive.

    Without this, every codex invocation strips xhigh/fast and re-prompts for
    trust on the (always new) sandbox dir. Wrapped in try/except so init
    never fails on a host config oddity.
    """
    try:
        codex_dir = config_dir if config_dir is not None else home_path / ".codex"
        codex_dir.mkdir(parents=True, exist_ok=True)
        host_codex = Path.home() / ".codex"
        host_config = host_codex / "config.toml"
        target_config = codex_dir / "config.toml"
        if copy_host_config and host_config.exists():
            try:
                shutil.copyfile(host_config, target_config)
            except OSError:
                # Fall back to the minimal config rather than leaving codex
                # with empty defaults.
                target_config.write_text(
                    _codex_fallback_config_toml(
                        include_service_tier=include_service_tier,
                    )
                )
        elif not target_config.exists():
            target_config.write_text(
                _codex_fallback_config_toml(include_service_tier=include_service_tier)
            )
        host_auth = host_codex / "auth.json"
        if copy_host_auth and host_auth.exists():
            try:
                shutil.copyfile(host_auth, codex_dir / "auth.json")
            except OSError:
                pass
    except Exception:  # pragma: no cover — best-effort
        logger.debug(
            "seed_isolated_codex_home failed; continuing with empty CODEX_HOME",
            exc_info=True,
        )


def _harden_codex_target_runtime_home(codex_dir: Path) -> None:
    """Prevent launcher preset/plugin materialization in target-runtime homes."""

    try:
        tmp_dir = codex_dir / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            tmp_dir.chmod(0o755)
        except OSError:
            logger.debug("Failed to make Codex tmp dir writable before cleanup", exc_info=True)
        for child in tmp_dir.glob("plugins-clone-*"):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except OSError:
                    logger.debug("Failed to remove Codex plugin temp file %s", child, exc_info=True)
        # Target-runtime agents must not see launcher marketplace/plugin/MCP
        # material. Public Codex still works with this dir non-writable, while
        # plugin clone attempts fail before adding ambient skills or MCP config.
        tmp_dir.chmod(0o555)
    except Exception:  # pragma: no cover - defensive hardening only
        logger.debug("Failed to harden Codex target-runtime home", exc_info=True)


class CLIModelClient:
    """Run CLI-backed models in a given working directory.

    Phase 2.5: callers that own a per-rollout registry (the rollout engine)
    can pass ``rollout_registry`` and ``rollout_id`` so spawned CLI children
    are scoped to a specific rollout. Deadline-driven teardown then signals
    only that rollout's CLI tree, leaving sibling rollouts untouched. Both
    parameters are optional — existing callsites (planning, evaluation,
    surrogates) keep their global-registry behaviour with zero changes.
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        rollout_registry: Optional["RolloutCLIRegistry"] = None,
        rollout_id: Optional[Any] = None,
        turn_observer: Optional[Callable[[Any, Any], Any]] = None,
        turn_observer_context: Optional[Any] = None,
    ):
        if not config.is_cli_backend:
            raise ValueError("CLIModelClient requires a CLI backend config.")
        self.config = config
        self._rollout_registry = rollout_registry
        self._rollout_id = rollout_id
        # Phase B.5: optional mid-stream observer that consumes Turn
        # objects parsed from the CLI's stdout/stderr and may either
        # log a soft course-correction or request abort. ``None`` keeps
        # the legacy (opaque-monolith) behaviour for every existing
        # caller — only the rollout engine wires this in.
        self._turn_observer = turn_observer
        self._turn_observer_context = turn_observer_context

    def _resolve_rollout_registry(
        self,
    ) -> tuple[Optional["RolloutCLIRegistry"], Optional[Any]]:
        """Return the registry/rollout_id pair to use for registration.

        Explicit constructor args take priority. If neither was passed,
        consult the contextvar set by ``active_rollout_cli_context`` so
        the engine can wire scoping in without threading kwargs through
        every helper.
        """
        if self._rollout_registry is not None and self._rollout_id is not None:
            return self._rollout_registry, self._rollout_id
        ctx = _resolve_active_rollout_cli_context()
        if ctx is not None:
            return ctx
        return None, None

    def _liveness_stall_window_seconds(self) -> float:
        """Resolve the uniform stall window (STALL_WINDOW) for K2.

        This is the single, hard/CLI-timeout-decoupled window the watchdog
        uses to detect *no meaningful progress*. Shrinking a planner phase
        budget can no longer shrink the stall window. Fails open: a missing /
        non-positive value falls back to the generous module default.
        """
        # The watchdog only ever sees an ``LLMConfig``; the engine propagates
        # ``RolloutConfig.stall_window_seconds`` onto ``cli_stall_window_seconds``.
        # Accept either name (fail open to the generous module default).
        for attr in ("cli_stall_window_seconds", "stall_window_seconds"):
            configured = getattr(self.config, attr, None)
            if isinstance(configured, (int, float)) and configured > 0:
                return float(configured)
        return _DEFAULT_STALL_WINDOW_SECONDS

    def _liveness_max_inflight_request_seconds(self) -> float:
        """Resolve the in-flight-LLM-request ceiling (S7 bound)."""
        for attr in ("cli_max_inflight_request_seconds", "max_inflight_request_seconds"):
            configured = getattr(self.config, attr, None)
            if isinstance(configured, (int, float)) and configured > 0:
                return float(configured)
        return _DEFAULT_MAX_INFLIGHT_REQUEST_SECONDS

    def _liveness_no_edit_progress_window_seconds(self) -> float:
        """Resolve the no-edit-progress window (token-runaway governor).

        ``0`` is a legitimate value meaning DISABLED, so (unlike the stall
        window) a configured ``0`` is honored rather than treated as "fall back
        to the default". A missing attribute falls open to the module default.
        """
        for attr in (
            "cli_no_edit_progress_window_seconds",
            "no_edit_progress_window_seconds",
        ):
            configured = getattr(self.config, attr, None)
            if isinstance(configured, (int, float)) and configured >= 0:
                return float(configured)
        return _DEFAULT_NO_EDIT_PROGRESS_WINDOW_SECONDS

    def _liveness_first_output_timeout_seconds(self) -> float:
        """Resolve the startup-output timeout for streaming CLI backends.

        This is not a wall-clock cap on an agentic task. It applies only until
        the first stdout/stderr chunk arrives; after that, normal progress-based
        liveness governs. A stream-json backend that never emits its initial
        event has not entered the observable agent loop, so CPU-only activity is
        not meaningful progress.
        """

        configured = getattr(self.config, "cli_first_output_timeout_seconds", None)
        if isinstance(configured, (int, float)):
            return max(0.0, float(configured))
        if self.config.backend == LLMBackend.CLAUDE_CLI:
            return _DEFAULT_STREAMING_FIRST_OUTPUT_TIMEOUT_SECONDS
        return 0.0

    def _effective_hard_timeout_seconds(
        self,
        override: Optional[int],
    ) -> Optional[int]:
        # Progress-based liveness keeps the default path stall-based, but a
        # caller that explicitly opts into ``cli_strict_hard_timeout`` needs a
        # real wall-clock cap. This is only a per-call backend contract; normal
        # orchestration should still advance via evidence and stall signals.
        if not bool(getattr(self.config, "cli_strict_hard_timeout", False)):
            return None

        candidates: list[int] = []
        for value in (override, getattr(self.config, "cli_hard_timeout_seconds", None)):
            if isinstance(value, (int, float)) and value > 0:
                candidates.append(int(value))
        if not candidates:
            return None

        # Use the largest configured bound so narrow planner-stage overrides do
        # not silently shrink a provider profile's explicit hard cap.
        return max(candidates)

    @staticmethod
    def _merge_cli_tool_hook_config(
        existing: Optional[dict[str, Any]],
        addition: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(existing or {})
        hooks = dict(merged.get("hooks") if isinstance(merged.get("hooks"), dict) else {})
        for event_name, groups in (addition.get("hooks") or {}).items():
            event_groups = list(hooks.get(event_name) or [])
            event_groups.extend(list(groups or []))
            hooks[event_name] = event_groups
        merged["hooks"] = hooks
        return merged

    @staticmethod
    def _write_merged_cli_tool_hook_config(
        path: Path,
        addition: dict[str, Any],
    ) -> None:
        existing: Optional[dict[str, Any]] = None
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    existing = loaded
            except (OSError, json.JSONDecodeError):
                existing = None
        merged = CLIModelClient._merge_cli_tool_hook_config(existing, addition)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(merged, handle, indent=2, sort_keys=True)
            handle.write("\n")

    @staticmethod
    def _home_looks_user_owned(home: Path) -> bool:
        try:
            return home.expanduser().resolve() == Path.home().resolve()
        except OSError:
            return False

    def _ensure_session_cli_home(
        self,
        env: dict[str, str],
        *,
        temp_dirs: list[tempfile.TemporaryDirectory[str]],
    ) -> Path:
        home = Path(env.get("HOME") or "").expanduser()
        if not str(home) or self._home_looks_user_owned(home):
            temp_home = tempfile.TemporaryDirectory(prefix="apex_cli_tool_review_home_")
            temp_dirs.append(temp_home)
            home = Path(temp_home.name)
            env.update(
                {
                    "HOME": str(home),
                    "XDG_CONFIG_HOME": str(home / ".config"),
                    "XDG_CACHE_HOME": str(home / ".cache"),
                    "XDG_DATA_HOME": str(home / ".local/share"),
                    "XDG_STATE_HOME": str(home / ".local/state"),
                }
            )
        home.mkdir(parents=True, exist_ok=True)
        return home

    def _prepare_cli_tool_reviewer_env_file(
        self,
        *,
        reviewer_backend: str,
        temp_dirs: list[tempfile.TemporaryDirectory[str]],
    ) -> str:
        try:
            backend = LLMBackend(getattr(reviewer_backend, "value", reviewer_backend))
        except ValueError:
            return ""
        spec = _CLI_BACKEND_SANDBOX_SPECS.get(backend)
        if spec is None:
            return ""

        home_dir = tempfile.TemporaryDirectory(prefix=f"apex_cli_tool_reviewer_{backend.value}_")
        temp_dirs.append(home_dir)
        home = Path(home_dir.name)
        for rel_path in (
            ".config",
            ".cache",
            ".local/share",
            ".local/state",
            "Library/Caches",
        ):
            (home / rel_path).mkdir(parents=True, exist_ok=True)
        for rel_path in spec.target_runtime_home.values():
            (home / rel_path).mkdir(parents=True, exist_ok=True)

        reviewer_env: dict[str, str] = {
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "XDG_DATA_HOME": str(home / ".local/share"),
            "XDG_STATE_HOME": str(home / ".local/state"),
        }
        for key, rel_path in spec.target_runtime_home.items():
            reviewer_env[key] = str(home / rel_path)
        for key, value in spec.target_env_defaults.items():
            reviewer_env.setdefault(key, value)
        for key in spec.auth_env_allowlist:
            raw_value = os.environ.get(key)
            if raw_value:
                reviewer_env[key] = raw_value

        if backend == LLMBackend.CODEX_CLI:
            _seed_isolated_codex_home(
                home,
                config_dir=Path(reviewer_env.get("CODEX_HOME") or home),
            )
        materialize_env = dict(os.environ)
        materialize_env.update(reviewer_env)
        has_auth_state = _materialize_cli_auth_state_for_target_runtime(
            spec,
            materialize_env,
            home_path=home,
        )
        for key in (
            spec.auth_state_env_key,
            *spec.target_path_env_keys,
            *spec.container_env_keys,
        ):
            if str(materialize_env.get(key) or "").strip():
                reviewer_env[key] = str(materialize_env[key])
        if backend == LLMBackend.GEMINI_CLI:
            _write_gemini_target_runtime_settings(
                materialize_env,
                home_path=home,
                has_auth_state=has_auth_state,
            )
            reviewer_env["GEMINI_CLI_HOME"] = str(home)

        env_path = home / ".apex_tool_reviewer_env.json"
        env_path.write_text(json.dumps(reviewer_env, indent=2, sort_keys=True), encoding="utf-8")
        try:
            env_path.chmod(0o600)
        except OSError:
            logger.debug("Failed to chmod reviewer env file %s", env_path, exc_info=True)
        return str(env_path)

    def _prepare_cli_tool_review_hook(
        self,
        env: dict[str, str],
        *,
        working_dir: str,
        temp_dirs: list[tempfile.TemporaryDirectory[str]],
    ) -> list[str]:
        if not getattr(self.config, "cli_tool_review_enabled", False):
            return []
        reviewer_backend = self.config.cli_tool_review_reviewer_backend or ""
        reviewer_command = self.config.cli_tool_review_reviewer_command or ""
        timeout_seconds = self.config.cli_tool_review_timeout_seconds
        if not str(env.get("APEX_TOOL_CALL_REVIEW_METRICS_PATH") or "").strip():
            try:
                metrics_dir = _agent_runtime_state_root_for_workspace(working_dir)
                metrics_dir.mkdir(parents=True, exist_ok=True)
                env["APEX_TOOL_CALL_REVIEW_METRICS_PATH"] = str(
                    metrics_dir / "tool_call_review_metrics.jsonl"
                )
            except OSError:
                pass
        reviewer_env_file = self._prepare_cli_tool_reviewer_env_file(
            reviewer_backend=reviewer_backend,
            temp_dirs=temp_dirs,
        )
        hook_command = build_apex_tool_review_hook_command(
            actor_backend=self.config.backend,
            reviewer_backend=reviewer_backend,
            reviewer_command=reviewer_command,
            timeout_seconds=timeout_seconds,
            reviewer_env_file=reviewer_env_file,
        )
        opencode_hook_command = hook_command
        support = get_cli_tool_hook_support(self.config.backend)
        if support.backend in {"opencode_cli", "metacode_cli"}:
            # OpenCode/MetaCode uses a local TypeScript plugin file, not a JSON
            # hook config, so it must NOT go through build_cli_tool_review_hook_config
            # (which rejects opencode). Install the plugin directly into the
            # isolated config from the raw hook command. Handled before the
            # JSON-hook-config build so this family never trips that rejection.
            home = self._ensure_session_cli_home(env, temp_dirs=temp_dirs)
            config_dir = home / ".config" / "opencode"
            env["OPENCODE_CONFIG_DIR"] = str(config_dir)
            env.setdefault("OPENCODE_DISABLE_AUTOUPDATE", "1")
            env.setdefault("OPENCODE_DISABLE_PROJECT_CONFIG", "1")
            plugin_dir = config_dir / "plugins"
            plugin_dir.mkdir(parents=True, exist_ok=True)
            plugin_path = plugin_dir / "apex-tool-call-reviewer.ts"
            plugin_path.write_text(
                build_opencode_tool_review_plugin_source(hook_command=opencode_hook_command),
                encoding="utf-8",
            )
            _write_opencode_family_isolated_config(
                self.config,
                env,
                working_dir=working_dir,
                plugin_paths=(plugin_path,),
            )
            return []
        hook_configs = [
            build_cli_tool_review_hook_config(
                actor_backend=self.config.backend,
                hook_command=hook_command,
                timeout_seconds=timeout_seconds,
            )
        ]

        if not hook_configs:
            return []
        hook_config: dict[str, Any] = {}
        for addition in hook_configs:
            hook_config = self._merge_cli_tool_hook_config(hook_config, addition)
        if support.backend == "claude_cli":
            hook_dir = tempfile.TemporaryDirectory(prefix="apex_claude_tool_review_")
            temp_dirs.append(hook_dir)
            settings_path = Path(hook_dir.name) / "settings.json"
            self._write_merged_cli_tool_hook_config(settings_path, hook_config)
            return ["--settings", str(settings_path)]
        if support.backend == "gemini_cli":
            home = self._ensure_session_cli_home(env, temp_dirs=temp_dirs)
            env["GEMINI_CLI_HOME"] = str(home)
            settings_path = home / ".gemini" / "settings.json"
            self._write_merged_cli_tool_hook_config(settings_path, hook_config)
            return []
        if support.backend == "codex_cli":
            home = self._ensure_session_cli_home(env, temp_dirs=temp_dirs)
            env["CODEX_HOME"] = str(home)
            _seed_isolated_codex_home(home)
            hook_paths = [home / "hooks.json", home / ".codex" / "hooks.json"]
            for hook_path in {path.resolve() for path in hook_paths}:
                self._write_merged_cli_tool_hook_config(hook_path, hook_config)
            if _env_flag_enabled("APEX_CODEX_REQUIRE_HOOK_TRUST"):
                return []
            return ["--dangerously-bypass-hook-trust"]
        raise ValueError(
            f"Backend '{support.backend}' does not support native tool-call review hooks."
        )

    def run_structured_prompt(
        self,
        prompt: str,
        working_dir: str,
        schema: Optional[dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        allow_edits: bool = False,
        internet_enabled: bool = False,
        hard_timeout_seconds: Optional[int] = None,
        env_overrides: Optional[dict[str, str]] = None,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        cancel_reason: Optional[Callable[[], str]] = None,
    ) -> CLIModelResult:
        start_time = time.time()

        def _cancel_requested() -> bool:
            if cancel_check is None:
                return False
            try:
                return bool(cancel_check())
            except Exception:  # noqa: BLE001 - cancellation probes must fail open
                return False

        def _cancel_reason_text() -> str:
            if cancel_reason is None:
                return "CLI prompt cancelled by scheduler"
            try:
                reason = str(cancel_reason() or "").strip()
            except Exception:  # noqa: BLE001 - cancellation reason is diagnostic only
                reason = ""
            return reason or "CLI prompt cancelled by scheduler"

        def _cancelled_result() -> CLIModelResult:
            now = time.time()
            reason = _cancel_reason_text()
            if progress_callback is not None:
                progress_callback(
                    {
                        "state": "cancelled",
                        "working_dir": working_dir,
                        "last_progress_at": now,
                        "last_progress_source": "scheduler_cancelled",
                        "terminal_state": "scheduler_cancelled",
                        "scheduler_cancelled": True,
                    }
                )
            return CLIModelResult(
                success=False,
                error=reason,
                duration_seconds=now - start_time,
                timeout_audit={
                    "started_at": start_time,
                    "ended_at": now,
                    "working_dir": working_dir,
                    "terminal_state": "scheduler_cancelled",
                    "scheduler_cancelled": True,
                    "cancel_reason": reason,
                },
                response_status="cancelled",
                workspace_status="not_checked" if allow_edits else "not_applicable",
                patch_extraction_status="missing",
                finalization_status="cancelled",
                telemetry_status="unknown",
            )

        if _cancel_requested():
            return _cancelled_result()

        effective_hard_timeout_seconds = self._effective_hard_timeout_seconds(hard_timeout_seconds)
        prompt = self._augment_prompt_for_backend(prompt, schema)
        air_gapped_temp_dirs: list[tempfile.TemporaryDirectory[str]] = []
        env = self._build_subprocess_env(
            env_overrides=env_overrides,
            internet_enabled=internet_enabled,
            temp_dirs=air_gapped_temp_dirs,
        )
        target_runtime_enforced = bool(env.get("APEX_TARGET_TOOL_CONTEXT"))
        agent_cli_in_container = False
        agent_cli_in_docker_image = False
        agent_cli_in_docker_sandbox = False
        docker_sandbox_shim_dir: Optional[Path] = None
        docker_sandbox_name: str = ""
        retry_diagnostics: list[dict[str, Any]] = []

        def _record_retry_diagnostic(
            *,
            retry_kind: str,
            retry_reason: str,
            attempt_index: int,
            max_attempts: int,
            process: subprocess.Popen[str],
            command: list[str],
            launch_env: Mapping[str, str],
            stdout: str,
            stderr: str,
            raw_output: str,
            result: CLIModelResult,
            timeout_audit: Optional[Mapping[str, Any]],
        ) -> dict[str, Any]:
            backend_value = str(getattr(self.config.backend, "value", self.config.backend))
            model_value = str(getattr(self.config, "model", "") or "")
            process_pid = int(getattr(process, "pid", 0) or 0)
            diagnostic_path = _write_cli_retry_diagnostic(
                working_dir=working_dir,
                backend=backend_value,
                model=model_value,
                retry_kind=retry_kind,
                retry_reason=retry_reason,
                attempt_index=attempt_index,
                max_attempts=max_attempts,
                process_pid=process_pid,
                returncode=getattr(process, "returncode", None),
                command=command,
                env=launch_env,
                stdout=stdout,
                stderr=stderr,
                raw_output=raw_output,
                result=result,
                timeout_audit=timeout_audit,
            )
            summary = {
                "retry_kind": retry_kind,
                "retry_reason": retry_reason,
                "attempt_index": attempt_index,
                "max_attempts": max_attempts,
                "process_pid": process_pid,
                "returncode": getattr(process, "returncode", None),
                "diagnostic_path": diagnostic_path,
            }
            backend_recovery_not_before = _record_cli_backend_transient_recovery(
                self.config,
                target_runtime_enforced=target_runtime_enforced,
            )
            if backend_recovery_not_before is not None:
                summary["backend_recovery_not_before"] = backend_recovery_not_before
            retry_diagnostics.append(summary)
            return summary

        def _attach_retry_diagnostics(result: CLIModelResult) -> CLIModelResult:
            if not retry_diagnostics:
                return result
            result.backend_diagnostics = dict(result.backend_diagnostics or {})
            result.backend_diagnostics["retry_diagnostics"] = list(retry_diagnostics)
            result.timeout_audit = dict(result.timeout_audit or {})
            result.timeout_audit.setdefault("retry_diagnostics", list(retry_diagnostics))
            return result

        if target_runtime_enforced:
            _prune_target_runtime_cli_shell_snapshots(env)
            env["APEX_TARGET_TOOL_WORKDIR"] = str(Path(working_dir).expanduser().resolve())
            env.setdefault(
                "APEX_CLI_INVOCATION_ID",
                f"apex-cli-{os.getpid()}-{time.time_ns()}-{secrets.token_hex(8)}",
            )
            agent_cli_in_docker_sandbox = _target_runtime_uses_docker_sandbox_cli(
                self.config,
                env,
            )
            agent_cli_in_container = _target_runtime_launches_agent_cli_in_container(
                self.config,
                env,
            )
            agent_cli_in_docker_image = _target_runtime_launches_agent_cli_in_docker_image(
                self.config,
                env,
            )
            if agent_cli_in_docker_sandbox:
                try:
                    env, docker_sandbox_shim_dir = _write_agent_visible_target_tool_shims(
                        env,
                        working_dir=working_dir,
                    )
                except CLIAgentContainerIsolationError as exc:
                    return CLIModelResult(
                        success=False,
                        error=str(exc),
                        duration_seconds=time.time() - start_time,
                        response_status="failed",
                        workspace_status="not_checked" if allow_edits else "not_applicable",
                        patch_extraction_status="missing",
                        finalization_status="isolation_error",
                        telemetry_status="unknown",
                    )
            elif agent_cli_in_container or agent_cli_in_docker_image:
                try:
                    _prepare_cli_target_runtime_env(
                        self.config,
                        env,
                        working_dir=working_dir,
                    )
                except CLIAgentContainerIsolationError as exc:
                    return CLIModelResult(
                        success=False,
                        error=str(exc),
                        duration_seconds=time.time() - start_time,
                        response_status="failed",
                        workspace_status="not_checked" if allow_edits else "not_applicable",
                        patch_extraction_status="missing",
                        finalization_status="isolation_error",
                        telemetry_status="unknown",
                    )
            _prune_target_runtime_cli_shell_snapshots(env)
            prompt = self._augment_prompt_for_target_runtime(prompt, env=env)
        cli_hook_args = self._prepare_cli_tool_review_hook(
            env,
            working_dir=working_dir,
            temp_dirs=air_gapped_temp_dirs,
        )
        try:
            # Claude/codex agentic backends emit bootstrap/setup chatter (preset
            # installs, version checks, auth) before the agent loop; a non-result
            # there (or a transient infra fault) made zero progress and is cheap
            # to retry, so they get the larger infra-retry budget. Only those two
            # branches (startup-only / transient-infra) actually consume the extra
            # attempts; real failures and successes still return on attempt 1.
            max_attempts = (
                _CLI_INFRA_RETRY_MAX_ATTEMPTS
                if self.config.backend in {LLMBackend.CLAUDE_CLI, LLMBackend.CODEX_CLI}
                else 1
            )
            claude_force_json_output = False
            claude_resume_session_id = ""
            claude_retry_from_workspace_without_session = False
            codex_resume_thread_id = ""
            salvageable_transient_candidate: Optional[CLIModelResult] = None
            salvageable_transient_source_attempt: Optional[int] = None
            first_salvageable_transient_attempt: Optional[int] = None
            salvageable_transient_reason: Optional[str] = None
            for attempt_index in range(1, max_attempts + 1):
                if _cancel_requested():
                    return _cancelled_result()
                _wait_for_cli_backend_transient_recovery(
                    self.config,
                    target_runtime_enforced=target_runtime_enforced,
                    working_dir=working_dir,
                    cancel_check=cancel_check,
                )
                if _cancel_requested():
                    return _cancelled_result()
                temp_files: list[str] = []
                process: Optional[subprocess.Popen[str]] = None
                launch_env_for_cleanup: Optional[dict[str, str]] = None
                target_tool_bridge: Optional[_TargetToolBridge] = None
                startup_concurrency_lease: Optional[_CLIBackendStartupConcurrencyLease] = None
                active_concurrency_slot: Optional[contextlib.AbstractContextManager[None]] = None
                try:
                    target_runtime_command_enforced = target_runtime_enforced and (
                        agent_cli_in_container
                        or agent_cli_in_docker_image
                        or agent_cli_in_docker_sandbox
                    )
                    prompt_for_attempt = prompt
                    if self.config.backend == LLMBackend.CLAUDE_CLI and (
                        claude_resume_session_id or claude_retry_from_workspace_without_session
                    ):
                        prompt_for_attempt = _CLAUDE_TRANSIENT_RETRY_RESUME_PROMPT
                    elif self.config.backend == LLMBackend.CODEX_CLI and codex_resume_thread_id:
                        prompt_for_attempt = _CODEX_TRANSIENT_RETRY_RESUME_PROMPT
                    build_command_kwargs: dict[str, Any] = {
                        "prompt": prompt_for_attempt,
                        "working_dir": working_dir,
                        "schema": schema,
                        "system_prompt": system_prompt,
                        "allow_edits": allow_edits,
                        "internet_enabled": internet_enabled,
                        "target_runtime_enforced": target_runtime_command_enforced,
                        "sandbox_writable_roots": _cli_sandbox_writable_roots(
                            env,
                            working_dir=working_dir,
                            include_cli_home=(
                                not target_runtime_enforced
                                or agent_cli_in_container
                                or agent_cli_in_docker_image
                            ),
                        ),
                        "codex_base_url": str(
                            env.get("CODEX_BASE_URL") or os.environ.get("CODEX_BASE_URL") or ""
                        ),
                        "cli_hook_args": cli_hook_args,
                        "codex_output_in_workspace": agent_cli_in_docker_sandbox,
                        "host_cli_read_jail": _host_cli_read_jail_enabled(env),
                        "claude_force_json_output": (
                            self.config.backend == LLMBackend.CLAUDE_CLI
                            and claude_force_json_output
                        ),
                        "claude_resume_session_id": (
                            claude_resume_session_id
                            if self.config.backend == LLMBackend.CLAUDE_CLI
                            else ""
                        ),
                        "codex_resume_thread_id": (
                            codex_resume_thread_id
                            if self.config.backend == LLMBackend.CODEX_CLI
                            else ""
                        ),
                    }
                    try:
                        signature = inspect.signature(self._build_command)
                    except (TypeError, ValueError):
                        signature = None
                    accepts_var_keyword = signature is not None and any(
                        parameter.kind == inspect.Parameter.VAR_KEYWORD
                        for parameter in signature.parameters.values()
                    )
                    if signature is not None and not accepts_var_keyword:
                        for optional_kwarg in (
                            "internet_enabled",
                            "target_runtime_enforced",
                            "sandbox_writable_roots",
                            "codex_base_url",
                            "cli_hook_args",
                            "codex_output_in_workspace",
                            "host_cli_read_jail",
                            "claude_force_json_output",
                            "claude_resume_session_id",
                            "codex_resume_thread_id",
                        ):
                            if optional_kwarg not in signature.parameters:
                                build_command_kwargs.pop(optional_kwarg, None)
                    command, temp_files = self._build_command(
                        **build_command_kwargs,
                    )
                    if _command_invokes_configured_cli(self.config, command):
                        _prepare_air_gapped_cli_home(
                            self.config,
                            env,
                            internet_enabled=internet_enabled,
                        )
                    agent_container_context = None
                    agent_image_context = None
                    agent_image_runtime = None
                    if agent_cli_in_container:
                        try:
                            agent_container_context = _agent_container_launch_context(
                                env,
                                working_dir=working_dir,
                            )
                        except CLIAgentContainerIsolationError as exc:
                            return CLIModelResult(
                                success=False,
                                error=str(exc),
                                duration_seconds=time.time() - start_time,
                                response_status="failed",
                                workspace_status=(
                                    "not_checked" if allow_edits else "not_applicable"
                                ),
                                patch_extraction_status="missing",
                                finalization_status="isolation_error",
                                telemetry_status="unknown",
                            )
                    elif agent_cli_in_docker_image:
                        try:
                            agent_image_context, agent_image_runtime = _agent_image_launch_context(
                                env,
                                working_dir=working_dir,
                            )
                        except CLIAgentContainerIsolationError as exc:
                            return CLIModelResult(
                                success=False,
                                error=str(exc),
                                duration_seconds=time.time() - start_time,
                                response_status="failed",
                                workspace_status=(
                                    "not_checked" if allow_edits else "not_applicable"
                                ),
                                patch_extraction_status="missing",
                                finalization_status="isolation_error",
                                telemetry_status="unknown",
                            )
                    if agent_container_context is not None:
                        try:
                            launch_command = _docker_exec_command_for_agent_container(
                                command,
                                env,
                                agent_container_context,
                                backend=self.config.backend,
                                auth_env_keys=_BACKEND_AUTH_ALLOWLIST.get(
                                    self.config.backend,
                                    (),
                                ),
                            )
                        except CLIAgentContainerIsolationError as exc:
                            return CLIModelResult(
                                success=False,
                                error=str(exc),
                                duration_seconds=time.time() - start_time,
                                response_status="failed",
                                workspace_status=(
                                    "not_checked" if allow_edits else "not_applicable"
                                ),
                                patch_extraction_status="missing",
                                finalization_status="isolation_error",
                                telemetry_status="unknown",
                            )
                        launch_env = dict(env)
                        launch_env.update(agent_container_context.docker_host_env)
                        launch_cwd = os.path.abspath(os.sep)
                    elif agent_image_context is not None and agent_image_runtime is not None:
                        try:
                            launch_command = _docker_run_command_for_agent_image(
                                command,
                                env,
                                agent_image_context,
                                agent_image_runtime,
                                backend=self.config.backend,
                                auth_env_keys=_BACKEND_AUTH_ALLOWLIST.get(
                                    self.config.backend,
                                    (),
                                ),
                            )
                        except CLIAgentContainerIsolationError as exc:
                            return CLIModelResult(
                                success=False,
                                error=str(exc),
                                duration_seconds=time.time() - start_time,
                                response_status="failed",
                                workspace_status=(
                                    "not_checked" if allow_edits else "not_applicable"
                                ),
                                patch_extraction_status="missing",
                                finalization_status="isolation_error",
                                telemetry_status="unknown",
                            )
                        launch_env = dict(env)
                        launch_env.update(agent_image_context.docker_host_env)
                        launch_cwd = os.path.abspath(os.sep)
                    elif agent_cli_in_docker_sandbox:
                        try:
                            docker_sandbox_name = _docker_sandbox_name(
                                self.config,
                                working_dir=working_dir,
                            )
                            _create_docker_sandbox_for_agent(
                                self.config,
                                env,
                                working_dir=working_dir,
                                sandbox_name=docker_sandbox_name,
                            )
                            target_tool_bridge = _TargetToolBridge(
                                env,
                                working_dir=working_dir,
                                descriptor_path=Path(env["APEX_TARGET_TOOL_BRIDGE_FILE"]),
                            )
                            target_tool_bridge.__enter__()
                            launch_command = _docker_sandbox_exec_command(
                                self.config,
                                command,
                                env,
                                sandbox_name=docker_sandbox_name,
                                working_dir=working_dir,
                            )
                        except CLIAgentContainerIsolationError as exc:
                            return CLIModelResult(
                                success=False,
                                error=str(exc),
                                duration_seconds=time.time() - start_time,
                                response_status="failed",
                                workspace_status=(
                                    "not_checked" if allow_edits else "not_applicable"
                                ),
                                patch_extraction_status="missing",
                                finalization_status="isolation_error",
                                telemetry_status="unknown",
                            )
                        launch_env = dict(os.environ)
                        launch_cwd = working_dir
                    else:
                        launch_command = _prepare_cli_command_for_target_tool_path(
                            command,
                            env,
                        )
                        launch_env = _cli_launch_env_for_target_runtime(env, launch_command)
                        # host_cli read-jail: confine the host CLI's filesystem
                        # reads to its workspace (macOS Seatbelt) so the agent can
                        # never read host data outside the container/workspace.
                        if _host_cli_read_jail_enabled(env):
                            launch_command, launch_env = _apply_host_cli_read_jail(
                                self.config,
                                launch_command,
                                launch_env,
                                working_dir=working_dir,
                            )
                        # host_cli claude auth: pre-mint the gateway token on the
                        # unconfined host so the managed apiKeyHelper never runs
                        # clicat under the target-tool shim PATH (which blocks it).
                        launch_env = _inject_host_cli_prefetched_claude_api_key(
                            self.config,
                            env,
                            launch_env,
                        )
                        launch_cwd = working_dir
                    launch_env_for_cleanup = dict(launch_env)
                    launch_cwd = os.path.abspath(launch_cwd)
                    launch_env["PWD"] = launch_cwd
                    if self.config.backend == LLMBackend.CLAUDE_CLI:
                        # Per-command (Bash tool) timeout — the missing cause-A
                        # ceiling. codex enforces a native per-exec timeout; claude
                        # does not, so a hung giant test command would block the
                        # agent's Bash tool until the outer scheduler reaps the
                        # whole rollout. Size-aware + generous (giants get the
                        # largest budget via the bound expected_test_count factor)
                        # so a legitimate long giant suite still completes; a truly
                        # hung command returns to the agent loop fast. Computed at
                        # launch so the engine-bound size factor is in scope. Does
                        # not override an operator-set value already in launch_env.
                        for _bash_key, _bash_value in _claude_bash_timeout_env_overrides().items():
                            launch_env.setdefault(_bash_key, _bash_value)
                    startup_stdin = ""
                    # When Claude is launched directly on the host (not inside an
                    # agent container or docker sandbox), it can hit the Meta
                    # internet-mode review gate if the workspace tree was ever used
                    # with --internet. Feed the exact acknowledgement phrase up
                    # front; it is consumed only if the gate appears (the prompt
                    # itself arrives via argv, so a no-gate launch ignores stdin).
                    auto_ack_internet_review = (
                        self.config.backend == LLMBackend.CLAUDE_CLI
                        and agent_container_context is None
                        and agent_image_context is None
                        and not agent_cli_in_docker_sandbox
                    )
                    if auto_ack_internet_review:
                        startup_stdin = _CLAUDE_INTERNET_REVIEW_ACK_RESPONSE + "\n"
                    if _env_flag_enabled("APEX_CAPTURE_CLI_COMMAND_TO_FILE"):
                        _capture_cli_command_to_file(
                            command=launch_command,
                            env=launch_env,
                            working_dir=working_dir,
                            backend=str(getattr(self.config, "backend", "unknown")),
                            attempt_index=attempt_index,
                        )
                    active_concurrency_slot = _cli_backend_active_concurrency_slot(
                        self.config,
                        target_runtime_enforced=target_runtime_enforced,
                        working_dir=working_dir,
                    )
                    active_concurrency_slot.__enter__()
                    startup_concurrency_lease = _CLIBackendStartupConcurrencyLease(
                        _cli_backend_concurrency_slot(
                            self.config,
                            target_runtime_enforced=target_runtime_enforced,
                            working_dir=working_dir,
                        ),
                        hold_seconds=_configured_cli_backend_startup_hold_seconds(
                            self.config,
                            target_runtime_enforced=target_runtime_enforced,
                        ),
                        release_on_timer=_cli_backend_startup_release_on_timer(
                            self.config,
                            target_runtime_enforced=target_runtime_enforced,
                        ),
                    )
                    startup_concurrency_lease.__enter__()

                    def _attempt_progress_callback(progress: dict[str, Any]) -> None:
                        if (
                            startup_concurrency_lease is not None
                            and _progress_payload_releases_cli_startup_slot(progress)
                        ):
                            startup_concurrency_lease.release()
                        if progress_callback is not None:
                            progress_callback(progress)

                    if (
                        self.config.backend == LLMBackend.CLAUDE_CLI
                        and target_runtime_command_enforced
                        and agent_container_context is not None
                    ):
                        _run_claude_target_runtime_warmup(
                            self.config,
                            launch_env,
                            working_dir=working_dir,
                            progress_callback=_attempt_progress_callback,
                        )
                    process = subprocess.Popen(
                        launch_command,
                        stdin=subprocess.PIPE if startup_stdin else subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=launch_cwd,
                        env=launch_env,
                        start_new_session=True,
                    )
                    startup_concurrency_lease.release_after_startup_window()
                    stdin_pipe = getattr(process, "stdin", None)
                    if startup_stdin and stdin_pipe is not None:
                        try:
                            stdin_pipe.write(startup_stdin)
                            stdin_pipe.close()
                            process.stdin = None
                        except (OSError, ValueError):
                            logger.debug(
                                "Failed to send CLI startup acknowledgement to %s",
                                self.config.backend.value,
                                exc_info=True,
                            )
                    _register_active_cli_process(process.pid)
                    # Phase 2.5: also register against the per-rollout
                    # registry when one is in scope. The global
                    # registration above remains the safety net for atexit
                    # / signal cleanup; the scoped one only takes effect
                    # for deadline-driven, per-rollout terminations.
                    rollout_registry, rollout_id_for_registry = self._resolve_rollout_registry()
                    inflight_request_active: Optional[Callable[[], Optional[float]]] = None
                    if rollout_registry is not None and rollout_id_for_registry is not None:
                        try:
                            rollout_registry.register(
                                rollout_id_for_registry,
                                process,
                                metadata=_target_runtime_cleanup_metadata(launch_env),
                            )
                        except Exception:  # pragma: no cover - defensive
                            logger.exception(
                                "Failed to register CLI process pid=%s with per-rollout registry",
                                process.pid,
                            )
                        # S7 marker: an agentic CLI subprocess is an in-flight
                        # request dispatch. Mark it before the watchdog enters
                        # so a long "thinking" turn (zero S1–S5, live socket) is
                        # treated as ALIVE up to ``max_inflight_request_seconds``
                        # of silence. The marker is cleared by ``unregister`` in
                        # the dispatch ``finally`` (whole metadata bucket dropped).
                        _registry_for_marker = rollout_registry
                        _rollout_id_for_marker = rollout_id_for_registry
                        _pid_for_marker = int(getattr(process, "pid", 0) or 0)
                        try:
                            _registry_for_marker.mark_inflight_request(
                                _rollout_id_for_marker, _pid_for_marker
                            )
                        except Exception:  # pragma: no cover - defensive
                            logger.exception(
                                "Failed to set S7 in-flight marker for pid=%s",
                                _pid_for_marker,
                            )

                        def _read_inflight_marker() -> Optional[float]:
                            try:
                                return _registry_for_marker.inflight_request_started_at(
                                    _rollout_id_for_marker, _pid_for_marker
                                )
                            except Exception:  # noqa: BLE001 - fail open (no freeze)
                                return None

                        inflight_request_active = _read_inflight_marker
                    _attempt_progress_callback(
                        {
                            "state": "spawned",
                            "process_pid": process.pid,
                            "working_dir": working_dir,
                            "allow_edits": allow_edits,
                            "started_at": start_time,
                            "attempt_index": attempt_index,
                            "max_attempts": max_attempts,
                            "agent_container_isolated": (
                                agent_container_context is not None
                                or agent_image_context is not None
                                or agent_cli_in_docker_sandbox
                            ),
                            "host_sandboxed_target_runtime": agent_cli_in_docker_sandbox,
                            "docker_sandboxed_target_runtime": agent_cli_in_docker_sandbox,
                            "docker_sandbox_name": docker_sandbox_name,
                        }
                    )
                    try:
                        communication_result = self._communicate_with_progress_timeout(
                            process,
                            working_dir=working_dir,
                            track_worktree=allow_edits,
                            hard_timeout_seconds=effective_hard_timeout_seconds,
                            progress_callback=_attempt_progress_callback,
                            target_runtime_enforced=(
                                bool(env.get("APEX_TARGET_TOOL_CONTEXT"))
                                and agent_container_context is None
                            ),
                            target_runtime_env=(
                                launch_env if bool(env.get("APEX_TARGET_TOOL_CONTEXT")) else None
                            ),
                            auto_ack_internet_review=auto_ack_internet_review,
                            inflight_request_active=inflight_request_active,
                            final_output_files=(
                                temp_files if self.config.backend == LLMBackend.CODEX_CLI else None
                            ),
                            cancel_check=cancel_check,
                            cancel_reason=cancel_reason,
                        )
                        if (
                            isinstance(communication_result, tuple)
                            and len(communication_result) == 3
                        ):
                            stdout, stderr, timeout_audit = communication_result
                        else:
                            stdout, stderr = communication_result  # type: ignore[misc]
                            timeout_audit = {}
                        completion_policy_audit = self._target_runtime_completion_policy_audit(
                            launch_env if bool(env.get("APEX_TARGET_TOOL_CONTEXT")) else None,
                            working_dir=working_dir,
                        )
                        if completion_policy_audit is not None:
                            policy_violation = dict(
                                completion_policy_audit.get("policy_violation") or {}
                            )
                            raise CLIProcessPolicyViolation(
                                str(
                                    policy_violation.get("reason")
                                    or "Target-runtime subprocess violated workspace policy."
                                ),
                                output=stdout or "",
                                stderr=stderr or "",
                                policy_audit=completion_policy_audit,
                            )
                    except subprocess.TimeoutExpired as exc:
                        self._kill_process_tree(process, env=launch_env)
                        partial_stdout = _coerce_subprocess_text(exc.output)
                        partial_stderr = _coerce_subprocess_text(exc.stderr)
                        raw_output = partial_stdout + (
                            ("\n" + partial_stderr) if partial_stderr else ""
                        )
                        timeout_audit = dict(getattr(exc, "timeout_audit", {}) or {})
                        recovered_result = self._recover_timed_out_result(
                            returncode=process.returncode,
                            stdout=partial_stdout,
                            stderr=partial_stderr,
                            temp_files=temp_files,
                        )
                        if recovered_result is not None:
                            timeout_kind = getattr(exc, "timeout_kind", "stall")
                            logger.warning(
                                "Recovered structured CLI output after %s timeout for %s",
                                timeout_kind,
                                working_dir,
                            )
                            timeout_audit["terminal_state"] = "recovered_after_timeout"
                            timeout_audit["recovered"] = True
                            recovered_result.raw_output = raw_output.strip()
                            recovered_result.duration_seconds = time.time() - start_time
                            recovered_result.timeout_audit = timeout_audit
                            self._finalize_result_channels(
                                recovered_result,
                                returncode=process.returncode,
                                stdout=partial_stdout,
                                stderr=partial_stderr,
                                allow_edits=allow_edits,
                                finalization_status="timeout_recovered",
                            )
                            if progress_callback is not None:
                                progress_callback(
                                    {
                                        **timeout_audit,
                                        "state": "recovered_after_timeout",
                                        "process_pid": process.pid,
                                    }
                                )
                            return recovered_result
                        timeout_kind = getattr(exc, "timeout_kind", "stall")
                        if timeout_kind == "hard":
                            error = (
                                f"CLI backend timed out after {int(exc.timeout)}s (hard timeout)"
                            )
                        else:
                            error = f"CLI backend stalled after {int(exc.timeout)}s without observable progress"
                        return CLIModelResult(
                            success=False,
                            error=error,
                            raw_output=raw_output.strip(),
                            duration_seconds=time.time() - start_time,
                            timeout_audit=timeout_audit,
                            response_status="failed",
                            workspace_status="not_checked" if allow_edits else "not_applicable",
                            patch_extraction_status="missing",
                            finalization_status="timeout",
                            telemetry_status="unknown",
                        )
                    except CLIProcessPolicyViolation as exc:
                        self._kill_process_tree(process, env=launch_env)
                        partial_stdout = exc.output or ""
                        partial_stderr = exc.stderr or ""
                        raw_output = partial_stdout + (
                            ("\n" + partial_stderr) if partial_stderr else ""
                        )
                        policy_audit = dict(getattr(exc, "policy_audit", {}) or {})
                        return CLIModelResult(
                            success=False,
                            error=exc.reason,
                            raw_output=raw_output.strip(),
                            duration_seconds=time.time() - start_time,
                            timeout_audit=policy_audit,
                            response_status="failed",
                            workspace_status="not_checked" if allow_edits else "not_applicable",
                            patch_extraction_status="missing",
                            finalization_status="policy_violation",
                            telemetry_status="unknown",
                        )
                    except CLIProcessInteractionRequired as exc:
                        self._kill_process_tree(process, env=launch_env)
                        partial_stdout = exc.output or ""
                        partial_stderr = exc.stderr or ""
                        raw_output = partial_stdout + (
                            ("\n" + partial_stderr) if partial_stderr else ""
                        )
                        interaction_audit = dict(getattr(exc, "interaction_audit", {}) or {})
                        return CLIModelResult(
                            success=False,
                            error=exc.reason,
                            raw_output=raw_output.strip(),
                            duration_seconds=time.time() - start_time,
                            timeout_audit=interaction_audit,
                            response_status="failed",
                            workspace_status="not_checked" if allow_edits else "not_applicable",
                            patch_extraction_status="missing",
                            finalization_status="interactive_prompt",
                            telemetry_status="unknown",
                        )
                    except CLIProcessOutputLimitExceeded as exc:
                        self._kill_process_tree(process, env=launch_env)
                        partial_stdout = exc.output or ""
                        partial_stderr = exc.stderr or ""
                        raw_output = partial_stdout + (
                            ("\n" + partial_stderr) if partial_stderr else ""
                        )
                        output_audit = dict(getattr(exc, "output_audit", {}) or {})
                        return CLIModelResult(
                            success=False,
                            error=exc.reason,
                            raw_output=raw_output.strip(),
                            duration_seconds=time.time() - start_time,
                            timeout_audit=output_audit,
                            response_status="failed",
                            workspace_status="not_checked" if allow_edits else "not_applicable",
                            patch_extraction_status="missing",
                            finalization_status="output_limit",
                            telemetry_status="unknown",
                        )
                    except CLIProcessProgressAbort as exc:
                        self._kill_process_tree(process, env=launch_env)
                        partial_stdout = exc.output or ""
                        partial_stderr = exc.stderr or ""
                        raw_output = partial_stdout + (
                            ("\n" + partial_stderr) if partial_stderr else ""
                        )
                        progress_audit = dict(getattr(exc, "progress_audit", {}) or {})
                        return CLIModelResult(
                            success=False,
                            error=exc.reason,
                            raw_output=raw_output.strip(),
                            duration_seconds=time.time() - start_time,
                            timeout_audit=progress_audit,
                            response_status="failed",
                            workspace_status="not_checked" if allow_edits else "not_applicable",
                            patch_extraction_status="missing",
                            finalization_status="progress_abort",
                            telemetry_status="unknown",
                        )

                    raw_output = (stdout or "") + (("\n" + stderr) if stderr else "")
                    model_result = self._parse_result(
                        returncode=process.returncode,
                        stdout=stdout or "",
                        stderr=stderr or "",
                        temp_files=temp_files,
                    )
                    model_result.raw_output = raw_output.strip()
                    model_result.duration_seconds = time.time() - start_time
                    model_result.timeout_audit = timeout_audit
                    self._finalize_result_channels(
                        model_result,
                        returncode=process.returncode,
                        stdout=stdout or "",
                        stderr=stderr or "",
                        allow_edits=allow_edits,
                    )

                    startup_only_failure = False
                    retry_state = "startup_retry"
                    retry_source = "startup_retry"
                    retry_log_label = "CLI backend"
                    startup_error = (
                        "CLI backend exited during startup without producing structured output."
                    )
                    if self.config.backend == LLMBackend.CLAUDE_CLI:
                        startup_only_failure = self._looks_like_claude_bootstrap_only_exit(
                            stdout=stdout or "",
                            stderr=stderr or "",
                            result=model_result,
                        ) or self._looks_like_claude_no_result_exit(
                            # Banner-only / content-free no-result exit (e.g. the
                            # claude cert/agent.id loss-of-access SEV): retry the
                            # transient infra fault instead of erroring the rollout.
                            stdout=stdout or "",
                            stderr=stderr or "",
                            result=model_result,
                        )
                        retry_state = "bootstrap_retry"
                        retry_source = "bootstrap_retry"
                        retry_log_label = "Claude CLI"
                        startup_error = "Claude CLI exited during bootstrap/setup without producing structured output."
                    elif self.config.backend == LLMBackend.CODEX_CLI:
                        startup_only_failure = self._looks_like_codex_startup_only_exit(
                            stdout=stdout or "",
                            stderr=stderr or "",
                            result=model_result,
                        )
                        retry_log_label = "Codex CLI"
                        startup_error = (
                            "Codex CLI exited during startup without producing structured output."
                        )
                    if startup_only_failure and attempt_index < max_attempts:
                        if _cancel_requested():
                            return _cancelled_result()
                        retry_detail = ""
                        if (
                            self.config.backend == LLMBackend.CLAUDE_CLI
                            and target_runtime_command_enforced
                            and not _target_runtime_claude_json_output_enabled(
                                force_json=claude_force_json_output
                            )
                        ):
                            claude_force_json_output = True
                            retry_detail = " with terminal JSON output"
                        if (
                            self.config.backend == LLMBackend.CLAUDE_CLI
                            and target_runtime_command_enforced
                        ):
                            _reset_claude_target_runtime_state_after_startup_failure(
                                launch_env,
                                working_dir=working_dir,
                            )
                        retry_summary = _record_retry_diagnostic(
                            retry_kind=retry_source,
                            retry_reason=startup_error,
                            attempt_index=attempt_index,
                            max_attempts=max_attempts,
                            process=process,
                            command=launch_command,
                            launch_env=launch_env,
                            stdout=stdout or "",
                            stderr=stderr or "",
                            raw_output=raw_output,
                            result=model_result,
                            timeout_audit=timeout_audit,
                        )
                        logger.warning(
                            "Retrying %s prompt after startup-only exit%s for %s "
                            "(reason=%s diagnostic=%s)",
                            retry_log_label,
                            retry_detail,
                            working_dir,
                            startup_error,
                            retry_summary.get("diagnostic_path"),
                        )
                        if progress_callback is not None:
                            progress_callback(
                                {
                                    "state": retry_state,
                                    "process_pid": process.pid,
                                    "working_dir": working_dir,
                                    "attempt_index": attempt_index,
                                    "max_attempts": max_attempts,
                                    "last_progress_at": time.time(),
                                    "last_progress_source": retry_source,
                                    "retry_reason": startup_error,
                                    "retry_diagnostic_path": retry_summary.get("diagnostic_path"),
                                }
                            )
                        _sleep_infra_retry_backoff(
                            attempt_index,
                            getattr(process, "pid", 0) or 0,
                            cancel_check=cancel_check,
                        )
                        if _cancel_requested():
                            return _cancelled_result()
                        continue
                    claude_activity_no_terminal_without_worktree = (
                        self.config.backend == LLMBackend.CLAUDE_CLI
                        and allow_edits
                        and self._looks_like_claude_agent_activity_without_terminal_result(
                            stdout=stdout or "",
                            result=model_result,
                        )
                        and not self._timeout_audit_has_worktree_activity(timeout_audit)
                    )
                    if (
                        claude_activity_no_terminal_without_worktree
                        and attempt_index < max_attempts
                    ):
                        if _cancel_requested():
                            return _cancelled_result()
                        if (
                            target_runtime_command_enforced
                            and not _target_runtime_claude_json_output_enabled(
                                force_json=claude_force_json_output
                            )
                        ):
                            claude_force_json_output = True
                            retry_detail = " with terminal JSON output"
                        else:
                            retry_detail = ""
                        retry_reason = (
                            "Claude CLI exited after agent activity without terminal "
                            "result or workspace changes."
                        )
                        retry_summary = _record_retry_diagnostic(
                            retry_kind="agent_nonterminal_retry",
                            retry_reason=retry_reason,
                            attempt_index=attempt_index,
                            max_attempts=max_attempts,
                            process=process,
                            command=launch_command,
                            launch_env=launch_env,
                            stdout=stdout or "",
                            stderr=stderr or "",
                            raw_output=raw_output,
                            result=model_result,
                            timeout_audit=timeout_audit,
                        )
                        logger.warning(
                            "Retrying Claude CLI prompt after agent activity ended without "
                            "terminal result or workspace changes%s for %s "
                            "(reason=%s diagnostic=%s)",
                            retry_detail,
                            working_dir,
                            retry_reason,
                            retry_summary.get("diagnostic_path"),
                        )
                        if progress_callback is not None:
                            progress_callback(
                                {
                                    "state": "agent_nonterminal_retry",
                                    "process_pid": process.pid,
                                    "working_dir": working_dir,
                                    "attempt_index": attempt_index,
                                    "max_attempts": max_attempts,
                                    "last_progress_at": time.time(),
                                    "last_progress_source": "agent_nonterminal_retry",
                                    "retry_reason": retry_reason,
                                    "retry_diagnostic_path": retry_summary.get("diagnostic_path"),
                                }
                            )
                        _sleep_infra_retry_backoff(
                            attempt_index,
                            getattr(process, "pid", 0) or 0,
                            cancel_check=cancel_check,
                        )
                        if _cancel_requested():
                            return _cancelled_result()
                        continue
                    if claude_activity_no_terminal_without_worktree:
                        model_result.success = False
                        model_result.error = (
                            "Claude CLI exited after agent activity without terminal result "
                            "or workspace changes."
                        )
                        model_result.finalization_status = "missing_terminal_result"
                    transient_infra_reason: Optional[str] = None
                    if not startup_only_failure and not model_result.success:
                        transient_infra_reason = self._transient_infra_failure_reason(
                            stdout=stdout or "",
                            stderr=stderr or "",
                            result=model_result,
                        )
                        # A codex tool-router failure (lost child stdin / unknown
                        # pid / streamable-exec session death) breaks the agent's
                        # tools, so any narration text it emitted is NOT a real
                        # result. Safe: only fires on success=False, and the
                        # worktree persists across attempts so any real edits are
                        # preserved/recovered.
                        if (
                            transient_infra_reason is None
                            and self._looks_like_codex_tool_router_failure(
                                stderr=stderr or "",
                                result=model_result,
                            )
                        ):
                            transient_infra_reason = "codex_tool_router_failure"
                        # Codex can emit partial agent messages and then lose
                        # the provider stream before the terminal turn. That
                        # is transport failure, not a candidate response; retry
                        # in the same worktree instead of charging a degraded
                        # recovered patch to selection.
                        if (
                            transient_infra_reason is None
                            and self._looks_like_codex_transport_disconnect_failure(
                                stdout=stdout or "",
                                stderr=stderr or "",
                                result=model_result,
                            )
                        ):
                            transient_infra_reason = "codex_transport_disconnect"
                    transient_infra_failure = transient_infra_reason is not None
                    content_free_cli_exit = (
                        not startup_only_failure
                        and not transient_infra_failure
                        and self._looks_like_content_free_cli_exit(
                            stdout=stdout or "",
                            stderr=stderr or "",
                            result=model_result,
                            timeout_audit=timeout_audit,
                        )
                    )
                    if transient_infra_failure:
                        if _cancel_requested():
                            return _cancelled_result()
                        claude_resume_session_missing_retry = False
                        if self.config.backend == LLMBackend.CLAUDE_CLI:
                            if transient_infra_reason == "claude_resume_session_missing":
                                # Claude can lose the remote resume handle even
                                # though the workspace has useful edits; retry
                                # once from the workspace instead of failing the
                                # rollout on a stale provider session id.
                                claude_resume_session_id = ""
                                claude_retry_from_workspace_without_session = allow_edits and (
                                    self._timeout_audit_has_worktree_activity(timeout_audit)
                                    or salvageable_transient_candidate is not None
                                )
                                claude_resume_session_missing_retry = (
                                    claude_retry_from_workspace_without_session
                                )
                            else:
                                recovered_session_id = self._extract_claude_session_id(stdout or "")
                                if recovered_session_id:
                                    if (
                                        target_runtime_enforced
                                        and not _claude_target_runtime_resume_session_available(
                                            env,
                                            session_id=recovered_session_id,
                                        )
                                    ):
                                        claude_resume_session_id = ""
                                        claude_retry_from_workspace_without_session = (
                                            self._timeout_audit_has_worktree_activity(timeout_audit)
                                        )
                                    else:
                                        claude_resume_session_id = recovered_session_id
                                        claude_retry_from_workspace_without_session = False
                        elif self.config.backend == LLMBackend.CODEX_CLI:
                            if transient_infra_reason == "codex_transport_disconnect":
                                # Codex transport loss can leave the remote
                                # thread context poisoned or oversized; retry
                                # in the same workspace with a fresh thread
                                # while preserving all filesystem edits.
                                codex_resume_thread_id = ""
                            else:
                                recovered_thread_id = self._extract_codex_thread_id(stdout or "")
                                if recovered_thread_id:
                                    codex_resume_thread_id = recovered_thread_id
                        salvageable_workspace_candidate = (
                            allow_edits and self._timeout_audit_has_worktree_activity(timeout_audit)
                        )
                        if salvageable_workspace_candidate:
                            salvageable_transient_candidate = model_result
                            salvageable_transient_source_attempt = attempt_index
                            salvageable_transient_reason = transient_infra_reason
                            if first_salvageable_transient_attempt is None:
                                first_salvageable_transient_attempt = attempt_index
                        has_salvageable_transient_candidate = (
                            salvageable_transient_candidate is not None
                            and first_salvageable_transient_attempt is not None
                        )
                        retry_kind = "transient_infra_retry"
                        retry_state = "transient_infra_retry"
                        retry_source = "transient_infra_retry"
                        if has_salvageable_transient_candidate:
                            retry_kind = "transient_infra_retry_after_workspace_activity"
                            retry_state = "transient_infra_retry_after_workspace_activity"
                            retry_source = "transient_infra_retry_after_workspace_activity"
                        if has_salvageable_transient_candidate:
                            workspace_recovery_attempts = max(
                                1,
                                int(_CLI_WORKSPACE_TRANSIENT_DEGRADE_AFTER_ATTEMPTS),
                            )
                            effective_retry_ceiling = min(
                                max_attempts,
                                first_salvageable_transient_attempt
                                + workspace_recovery_attempts
                                - 1,
                            )
                        else:
                            effective_retry_ceiling = max_attempts
                        if claude_resume_session_missing_retry and attempt_index < max_attempts:
                            effective_retry_ceiling = min(
                                max_attempts,
                                max(effective_retry_ceiling, attempt_index + 1),
                            )
                        if attempt_index >= effective_retry_ceiling:
                            if has_salvageable_transient_candidate:
                                degraded_result = salvageable_transient_candidate or model_result
                                retry_summary = _record_retry_diagnostic(
                                    retry_kind="transient_infra_degraded_candidate",
                                    retry_reason=(
                                        transient_infra_reason or "transient_infra_failure"
                                    ),
                                    attempt_index=attempt_index,
                                    max_attempts=effective_retry_ceiling,
                                    process=process,
                                    command=launch_command,
                                    launch_env=launch_env,
                                    stdout=stdout or "",
                                    stderr=stderr or "",
                                    raw_output=raw_output,
                                    result=model_result,
                                    timeout_audit=timeout_audit,
                                )
                                degraded_result.backend_diagnostics = dict(
                                    degraded_result.backend_diagnostics or {}
                                )
                                degraded_result.backend_diagnostics[
                                    "transient_infra_degraded_candidate"
                                ] = {
                                    "reason": transient_infra_reason,
                                    "source_reason": salvageable_transient_reason,
                                    "attempt_index": attempt_index,
                                    "source_attempt_index": (salvageable_transient_source_attempt),
                                    "degraded_after_attempt_index": attempt_index,
                                    "max_attempts": effective_retry_ceiling,
                                    "configured_max_attempts": max_attempts,
                                    "diagnostic_path": retry_summary.get("diagnostic_path"),
                                }
                                if not degraded_result.finalization_status:
                                    degraded_result.finalization_status = (
                                        "transient_infra_degraded_candidate"
                                    )
                                if progress_callback is not None:
                                    progress_callback(
                                        {
                                            "state": "transient_infra_degraded_candidate",
                                            "process_pid": process.pid,
                                            "working_dir": working_dir,
                                            "attempt_index": attempt_index,
                                            "max_attempts": effective_retry_ceiling,
                                            "last_progress_at": time.time(),
                                            "last_progress_source": (
                                                "transient_infra_degraded_candidate"
                                            ),
                                            "retry_reason": transient_infra_reason,
                                            "retry_diagnostic_path": retry_summary.get(
                                                "diagnostic_path"
                                            ),
                                        }
                                    )
                                return _attach_retry_diagnostics(degraded_result)
                            return _attach_retry_diagnostics(model_result)
                        retry_summary = _record_retry_diagnostic(
                            retry_kind=retry_kind,
                            retry_reason=transient_infra_reason or "transient_infra_failure",
                            attempt_index=attempt_index,
                            max_attempts=effective_retry_ceiling,
                            process=process,
                            command=launch_command,
                            launch_env=launch_env,
                            stdout=stdout or "",
                            stderr=stderr or "",
                            raw_output=raw_output,
                            result=model_result,
                            timeout_audit=timeout_audit,
                        )
                        logger.warning(
                            "Retrying %s prompt after transient-infra failure for %s "
                            "(reason=%s workspace_activity=%s diagnostic=%s)",
                            retry_log_label,
                            working_dir,
                            transient_infra_reason,
                            salvageable_workspace_candidate,
                            retry_summary.get("diagnostic_path"),
                        )
                        if progress_callback is not None:
                            progress_callback(
                                {
                                    "state": retry_state,
                                    "process_pid": process.pid,
                                    "working_dir": working_dir,
                                    "attempt_index": attempt_index,
                                    "max_attempts": effective_retry_ceiling,
                                    "last_progress_at": time.time(),
                                    "last_progress_source": retry_source,
                                    "retry_reason": transient_infra_reason,
                                    "retry_diagnostic_path": retry_summary.get("diagnostic_path"),
                                }
                            )
                        _sleep_infra_retry_backoff(
                            attempt_index,
                            getattr(process, "pid", 0) or 0,
                            cancel_check=cancel_check,
                        )
                        if _cancel_requested():
                            return _cancelled_result()
                        continue
                    if content_free_cli_exit and attempt_index < max_attempts:
                        if _cancel_requested():
                            return _cancelled_result()
                        retry_reason = (
                            "CLI backend exited without structured output or captured text."
                        )
                        retry_summary = _record_retry_diagnostic(
                            retry_kind="content_free_retry",
                            retry_reason=retry_reason,
                            attempt_index=attempt_index,
                            max_attempts=max_attempts,
                            process=process,
                            command=launch_command,
                            launch_env=launch_env,
                            stdout=stdout or "",
                            stderr=stderr or "",
                            raw_output=raw_output,
                            result=model_result,
                            timeout_audit=timeout_audit,
                        )
                        logger.warning(
                            "Retrying %s prompt after content-free CLI exit for %s "
                            "(reason=%s diagnostic=%s)",
                            retry_log_label,
                            working_dir,
                            retry_reason,
                            retry_summary.get("diagnostic_path"),
                        )
                        if progress_callback is not None:
                            progress_callback(
                                {
                                    "state": "content_free_retry",
                                    "process_pid": process.pid,
                                    "working_dir": working_dir,
                                    "attempt_index": attempt_index,
                                    "max_attempts": max_attempts,
                                    "last_progress_at": time.time(),
                                    "last_progress_source": "content_free_retry",
                                    "retry_reason": retry_reason,
                                    "retry_diagnostic_path": retry_summary.get("diagnostic_path"),
                                }
                            )
                        _sleep_infra_retry_backoff(
                            attempt_index,
                            getattr(process, "pid", 0) or 0,
                            cancel_check=cancel_check,
                        )
                        if _cancel_requested():
                            return _cancelled_result()
                        continue
                    if content_free_cli_exit:
                        model_result.success = False
                        if not model_result.error:
                            model_result.error = (
                                "CLI backend exited without producing structured output "
                                "or captured text."
                            )
                    if startup_only_failure:
                        model_result.success = False
                        if not model_result.error:
                            model_result.error = startup_error
                    return _attach_retry_diagnostics(model_result)
                finally:
                    if startup_concurrency_lease is not None:
                        startup_concurrency_lease.__exit__(None, None, None)
                    if active_concurrency_slot is not None:
                        active_concurrency_slot.__exit__(None, None, None)
                    if process is not None:
                        _unregister_active_cli_process(process.pid)
                        rollout_registry, rollout_id_for_registry = self._resolve_rollout_registry()
                        if rollout_registry is not None and rollout_id_for_registry is not None:
                            # S7 marker is cleared here (and again implicitly by
                            # unregister, which drops the metadata bucket).
                            try:
                                rollout_registry.clear_inflight_request(
                                    rollout_id_for_registry,
                                    int(getattr(process, "pid", 0) or 0),
                                )
                            except Exception:  # pragma: no cover - defensive
                                logger.exception(
                                    "Failed to clear S7 in-flight marker for pid=%s",
                                    process.pid,
                                )
                            try:
                                rollout_registry.unregister(
                                    rollout_id_for_registry,
                                    process,
                                )
                            except Exception:  # pragma: no cover - defensive
                                logger.exception(
                                    "Failed to unregister CLI process pid=%s "
                                    "from per-rollout registry",
                                    process.pid,
                                )
                        if int(getattr(process, "pid", 0) or 0) != os.getpid():
                            _cleanup_target_runtime_after_cli_completion(
                                launch_env_for_cleanup,
                            )
                    if target_tool_bridge is not None:
                        target_tool_bridge.__exit__(None, None, None)
                        target_tool_bridge = None
                    if docker_sandbox_name:
                        _remove_docker_sandbox(docker_sandbox_name)
                        docker_sandbox_name = ""
                    self._cleanup_temp_files(temp_files)
        finally:
            if docker_sandbox_shim_dir:
                shutil.rmtree(docker_sandbox_shim_dir, ignore_errors=True)
            if docker_sandbox_name:
                _remove_docker_sandbox(docker_sandbox_name)
            self._cleanup_temp_dirs(air_gapped_temp_dirs)
        return _attach_retry_diagnostics(
            CLIModelResult(
                success=False,
                error="CLI backend exhausted all prompt attempts without a terminal result.",
                duration_seconds=time.time() - start_time,
                response_status="failed",
                workspace_status="not_checked" if allow_edits else "not_applicable",
                patch_extraction_status="missing",
                finalization_status="exhausted_retries",
                telemetry_status="unknown",
            )
        )

    def _build_subprocess_env(
        self,
        env_overrides: Optional[dict[str, str]] = None,
        *,
        internet_enabled: bool = False,
        temp_dirs: Optional[list[tempfile.TemporaryDirectory[str]]] = None,
        backend_allow_keys: Optional[Iterable[str]] = None,
    ) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        target_runtime_requested = bool(
            (self.config.cli_env_overrides or {}).get("APEX_TARGET_TOOL_CONTEXT")
            or (env_overrides or {}).get("APEX_TARGET_TOOL_CONTEXT")
        )
        # host_cli mode: the CLI runs on the host and must keep its native Meta
        # transport (x2p proxy + CAT, Vertex gateway, originator) so its auth
        # works. Detect it from the same override sources that carry the auth
        # mode, and preserve ONLY the transport allowlist — all other host
        # secrets are still redacted (no data leakage).
        host_cli_mode = _host_cli_auth_mode_requested(
            self.config.cli_env_overrides
        ) or _host_cli_auth_mode_requested(env_overrides)
        allow = (
            tuple(backend_allow_keys)
            if backend_allow_keys is not None
            else (
                _BACKEND_AUTH_ALLOWLIST.get(self.config.backend, ())
                if target_runtime_requested
                else ()
            )
        )
        if host_cli_mode:
            allow = tuple(allow) + _HOST_CLI_TRANSPORT_ALLOW_KEYS
        # Strip host secrets before any nested CLI/tool subprocess sees them.
        # Normal benchmark rollouts use agentic CLI sessions/config files; they
        # must not inherit provider API keys from Apex's host shell.
        if not getattr(self.config, "cli_env_redaction_disabled", False):
            env, removed = redact_host_secrets(env, allow_keys=allow)
            if removed:
                security_logger.info(
                    "cli_env_redacted",
                    extra={
                        "event": "apex.security.cli_env_redacted",
                        "backend": self.config.backend.value,
                        "removed_count": len(removed),
                        "removed_sample": removed[:8],
                    },
                )
        elif getattr(self.config, "cli_env_redaction_disabled", False):
            security_logger.warning(
                "cli_env_redaction_disabled",
                extra={
                    "event": "apex.security.cli_env_redaction_disabled",
                    "backend": self.config.backend.value,
                },
            )
        # Parent coding-agent sessions export sandbox/originator variables that can
        # interfere with nested Codex/Claude CLI subprocesses.
        env.pop("META_CLAUDE_DANGEROUSLY_DISABLE_LINUX_SANDBOX", None)
        env.pop("META_DANGEROUSLY_DISABLE_LINUX_SANDBOX", None)
        env.pop("CLAUDECODE", None)
        if not host_cli_mode:
            # host_cli runs the real host CLI; its originator override is part of
            # the sanctioned Meta transport and must survive.
            env.pop("CODEX_INTERNAL_ORIGINATOR_OVERRIDE", None)
        if (
            temp_dirs is not None
            and not internet_enabled
            and self.config.is_cli_backend
            and not host_cli_mode
        ):
            # host_cli mode runs the real host CLI which dotslash-materializes from
            # the real ~/Library/Caches and authenticates via the host Meta
            # transport; an isolated/air-gapped HOME breaks both, so it is skipped.
            # Air-gapped runs should not inherit user-level MCP/plugin state that can
            # silently re-enable external search or browsing through ambient config.
            #
            # Claude's launcher also bootstraps native binaries/plugins into HOME on
            # first use. Recreating a fresh isolated HOME for every invocation causes
            # repeated bootstrap churn and setup-only exits during benchmark stages,
            # so those runs reuse a dedicated Apex-managed isolated home instead.
            if self.config.backend in _PERSISTENT_AIR_GAPPED_HOME_BACKENDS:
                home_path = _persistent_air_gapped_cli_home(self.config.backend)
            else:
                offline_home = _temporary_air_gapped_cli_home(self.config.backend)
                temp_dirs.append(offline_home)
                home_path = Path(offline_home.name)
            for rel_path in (
                ".config",
                ".cache",
                ".local/share",
                ".local/state",
                "Library/Caches",
            ):
                (home_path / rel_path).mkdir(parents=True, exist_ok=True)
            env.update(
                {
                    "HOME": str(home_path),
                    "XDG_CONFIG_HOME": str(home_path / ".config"),
                    "XDG_CACHE_HOME": str(home_path / ".cache"),
                    "XDG_DATA_HOME": str(home_path / ".local/share"),
                    "XDG_STATE_HOME": str(home_path / ".local/state"),
                }
            )
            if self.config.backend == LLMBackend.CODEX_CLI:
                env["CODEX_HOME"] = str(home_path)
                # 10.C: copy host ~/.codex/config.toml + auth.json (or write a
                # minimal fallback) so xhigh reasoning-effort and service-tier
                # survive the isolated CODEX_HOME. Without this, every codex
                # invocation runs with stripped reasoning effort.
                _seed_isolated_codex_home(home_path)
        merged_overrides = _default_cli_env_overrides(self.config)
        merged_overrides.update(self.config.cli_env_overrides)
        if env_overrides:
            merged_overrides.update(env_overrides)
        if merged_overrides:
            env.update(merged_overrides)
        _normalize_gemini_provider_env(self.config, env)
        # Overrides can come from benchmark target-runtime wiring or config
        # files. Redact again after merging so provider API keys cannot be
        # reintroduced through cli_env_overrides.
        if not getattr(self.config, "cli_env_redaction_disabled", False):
            env, removed = redact_host_secrets(env, allow_keys=allow)
            if removed:
                security_logger.info(
                    "cli_env_redacted_after_overrides",
                    extra={
                        "event": "apex.security.cli_env_redacted_after_overrides",
                        "backend": self.config.backend.value,
                        "removed_count": len(removed),
                        "removed_sample": removed[:8],
                    },
                )
        return env

    def _communicate_with_progress_timeout(
        self,
        process: subprocess.Popen[str],
        *,
        working_dir: str,
        track_worktree: bool,
        hard_timeout_seconds: Optional[int] = None,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
        target_runtime_enforced: bool = False,
        target_runtime_env: Optional[dict[str, str]] = None,
        auto_ack_internet_review: bool = False,
        inflight_request_active: Optional[Callable[[], Optional[float]]] = None,
        final_output_files: Optional[list[str]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        cancel_reason: Optional[Callable[[], str]] = None,
    ) -> tuple[str, str, dict[str, Any]]:
        progress_timeout = self._effective_progress_timeout_seconds(
            track_worktree=track_worktree,
            hard_timeout_seconds=hard_timeout_seconds,
        )
        # S7 in-flight-LLM-request ceiling: while a request marker is set AND
        # the process is running, the stall clock is treated as frozen (the
        # agent is ALIVE) UNTIL total silence exceeds this ceiling — at which
        # point a black-hole socket / crashed worker resumes normal stall
        # timing and the STALL_WINDOW kill applies. ``inflight_request_active``
        # is fail-open: a missing/None marker means the freeze never engages,
        # so the ordinary STALL_WINDOW stall branch governs (it can never make
        # a kill fire *earlier* than the stall branch alone).
        max_inflight_request_seconds = self._liveness_max_inflight_request_seconds()
        first_output_timeout_seconds = self._liveness_first_output_timeout_seconds()
        # No-edit-progress window (token-runaway governor): editable stages only.
        # 0 disables. CPU is excluded from this clock inside the loop below.
        no_edit_progress_window = (
            self._liveness_no_edit_progress_window_seconds() if track_worktree else 0.0
        )
        hard_timeout = (
            float(hard_timeout_seconds)
            if bool(getattr(self.config, "cli_strict_hard_timeout", False))
            and isinstance(hard_timeout_seconds, (int, float))
            and hard_timeout_seconds > 0
            else None
        )
        if (progress_timeout is None or progress_timeout <= 0) and hard_timeout is None:
            stdout, stderr = process.communicate()
            completed_at = time.time()
            timeout_audit = {
                "started_at": completed_at,
                "ended_at": completed_at,
                "working_dir": working_dir,
                "track_worktree": track_worktree,
                "progress_timeout_seconds": progress_timeout,
                "first_output_timeout_seconds": first_output_timeout_seconds,
                "hard_timeout_seconds": hard_timeout,
                "hard_timeout_progress_grace_seconds": self._effective_hard_timeout_progress_grace_seconds(
                    hard_timeout_seconds=hard_timeout_seconds,
                ),
                "last_progress_at": completed_at,
                "last_progress_source": "process_exit",
                "evidence_counts": {
                    "stdout": 0,
                    "stderr": 0,
                    "worktree": 0,
                    "cpu": 0,
                    "target_runtime_cpu": 0,
                    "target_runtime_process": 0,
                },
                "terminal_state": "completed",
            }
            if progress_callback is not None:
                progress_callback(
                    {
                        **timeout_audit,
                        "state": "completed",
                        "process_pid": process.pid,
                    }
                )
            return stdout or "", stderr or "", timeout_audit

        assert process.stdout is not None
        assert process.stderr is not None

        stream_queue: queue.Queue[tuple[str, str]] = queue.Queue(
            maxsize=_CLI_STREAM_QUEUE_MAX_CHUNKS
        )
        capture_max_chars = int(getattr(self.config, "cli_output_capture_max_chars", 0) or 0)
        stdout_capture = _BoundedStreamCapture("stdout", capture_max_chars)
        stderr_capture = _BoundedStreamCapture("stderr", capture_max_chars)

        readers = [
            threading.Thread(
                target=self._read_stream_to_queue,
                args=(process.stdout, "stdout", stream_queue),
                daemon=True,
            ),
            threading.Thread(
                target=self._read_stream_to_queue,
                args=(process.stderr, "stderr", stream_queue),
                daemon=True,
            ),
        ]
        for reader in readers:
            reader.start()

        started_at = time.time()
        last_progress = time.time()
        last_activity_check = 0.0
        activity_check_interval = (
            min(2.0, max(progress_timeout / 4.0, 0.25))
            if progress_timeout is not None and progress_timeout > 0
            else 0.5
        )
        hard_timeout_grace = self._effective_hard_timeout_progress_grace_seconds(
            hard_timeout_seconds=hard_timeout_seconds,
        )
        last_worktree_mtime = (
            self._latest_worktree_mtime(Path(working_dir)) if track_worktree else None
        )
        last_process_cpu_seconds = self._process_tree_cpu_seconds(process.pid)
        last_target_runtime_cpu_ticks = 0.0
        last_target_runtime_pids: set[int] = set()
        last_target_runtime_activity: dict[str, Any] = {}
        last_target_runtime_activity_at: Optional[float] = None
        last_target_runtime_activity_check = 0.0
        last_stdout_at: Optional[float] = None
        last_stderr_at: Optional[float] = None
        last_worktree_at: Optional[float] = None
        last_cpu_at: Optional[float] = None
        last_progress_source = "spawned"
        evidence_counts = {
            "stdout": 0,
            "stderr": 0,
            "worktree": 0,
            "cpu": 0,
            "target_runtime_cpu": 0,
            "target_runtime_process": 0,
        }
        seen_soft_policy_violations: set[tuple[int, str, str]] = set()
        soft_policy_violation_log: list[dict[str, Any]] = []
        final_output_paths = [
            Path(path) for path in list(final_output_files or []) if str(path or "").strip()
        ]
        final_output_state: dict[Path, dict[str, Any]] = {
            path: {"size": -1, "mtime_ns": -1, "changed_at": started_at}
            for path in final_output_paths
        }
        final_output_completion: Optional[dict[str, Any]] = None
        final_output_termination_requested = False

        # Phase B.5: optional mid-stream turn parser. Buffer arriving
        # stdout chunks into lines, run the parser on each completed
        # line, and dispatch any closed Turn to the observer. Observer
        # output (CourseCorrection) is appended to ``course_corrections``
        # so the caller (the engine) can see what fired without having
        # to scrape the trajectory again. ``abort=True`` requests a
        # subprocess termination via the rollout-scoped registry; the
        # engine wrapper then surfaces it as a normal stall-style exit.
        _turn_parser = self._build_turn_parser_for_backend()
        _turn_observer = self._turn_observer
        _turn_observer_ctx = self._turn_observer_context
        _turn_stdout_carry: list[str] = [""]
        course_corrections: list[dict[str, Any]] = []
        observer_aborted: dict[str, Any] = {}

        def _dispatch_turn_observer(closed_turn: Any) -> None:
            """Run the observer on ``closed_turn`` and capture its result."""
            if _turn_observer is None or closed_turn is None:
                return
            try:
                correction = _turn_observer(closed_turn, _turn_observer_ctx)
            except Exception as exc:  # noqa: BLE001 - never crash the agent loop
                logger.warning(
                    "Turn observer raised %s: %s; treating as no-correction.",
                    type(exc).__name__,
                    exc,
                )
                return
            if correction is None:
                return
            entry = {
                "turn_number": int(getattr(closed_turn, "number", 0) or 0),
                "message": str(getattr(correction, "message", "") or ""),
                "abort": bool(getattr(correction, "abort", False)),
                "source": str(getattr(correction, "source", "") or ""),
                "extra": dict(getattr(correction, "extra", {}) or {}),
            }
            course_corrections.append(entry)
            if entry["abort"]:
                observer_aborted.update(entry)

        def _ingest_stdout_chunk_for_turns(chunk: str) -> None:
            """Feed a stdout chunk to the turn parser, line by line."""
            if _turn_parser is None or not chunk:
                return
            buffered = _turn_stdout_carry[0] + chunk
            *complete_lines, tail = buffered.split("\n")
            _turn_stdout_carry[0] = tail
            for line in complete_lines:
                try:
                    closed = _turn_parser.feed_line(line)
                except Exception as exc:  # noqa: BLE001 - parser must not crash agent
                    logger.warning(
                        "Turn parser raised %s on line %r; skipping.",
                        type(exc).__name__,
                        line[:120],
                    )
                    closed = None
                if closed is not None:
                    _dispatch_turn_observer(closed)

        def _emit_progress(state: str, *, source: Optional[str] = None) -> None:
            if progress_callback is None:
                return
            payload = {
                "state": state,
                "process_pid": process.pid,
                "working_dir": working_dir,
                "track_worktree": track_worktree,
                "progress_timeout_seconds": progress_timeout,
                "first_output_timeout_seconds": first_output_timeout_seconds,
                "hard_timeout_seconds": hard_timeout,
                "hard_timeout_progress_grace_seconds": hard_timeout_grace,
                "stage_started_at": started_at,
                "last_progress_at": last_progress,
                "last_progress_source": source or last_progress_source,
                "last_stdout_at": last_stdout_at,
                "last_stderr_at": last_stderr_at,
                "last_worktree_at": last_worktree_at,
                "last_cpu_at": last_cpu_at,
                "last_target_runtime_activity_at": last_target_runtime_activity_at,
                "target_runtime_activity": dict(last_target_runtime_activity),
                "evidence_counts": dict(evidence_counts),
            }
            try:
                progress_callback(payload)
            except CLIProcessProgressAbort as exc:
                now = time.time()
                progress_audit = _base_timeout_audit(now, "progress_abort")
                progress_audit["progress_abort"] = dict(exc.progress_audit or {})
                raise CLIProcessProgressAbort(
                    exc.reason,
                    output=_captured_stdout(),
                    stderr=_captured_stderr(),
                    progress_audit=progress_audit,
                ) from exc

        def _captured_stdout() -> str:
            return stdout_capture.text()

        def _captured_stderr() -> str:
            return stderr_capture.text()

        def _cancel_requested() -> bool:
            if cancel_check is None:
                return False
            try:
                return bool(cancel_check())
            except Exception:  # noqa: BLE001 - cancellation probes must fail open
                return False

        def _cancel_reason_text() -> str:
            if cancel_reason is None:
                return "CLI prompt cancelled by scheduler"
            try:
                reason = str(cancel_reason() or "").strip()
            except Exception:  # noqa: BLE001 - cancellation reason is diagnostic only
                reason = ""
            return reason or "CLI prompt cancelled by scheduler"

        def _output_capture_audit() -> dict[str, Any]:
            return {
                "stdout": stdout_capture.audit(),
                "stderr": stderr_capture.audit(),
            }

        def _base_timeout_audit(now: float, terminal_state: str) -> dict[str, Any]:
            return {
                "started_at": started_at,
                "ended_at": now,
                "working_dir": working_dir,
                "track_worktree": track_worktree,
                "progress_timeout_seconds": progress_timeout,
                "first_output_timeout_seconds": first_output_timeout_seconds,
                "hard_timeout_seconds": hard_timeout,
                "hard_timeout_progress_grace_seconds": hard_timeout_grace,
                "last_progress_at": last_progress,
                "last_progress_source": last_progress_source,
                "last_stdout_at": last_stdout_at,
                "last_stderr_at": last_stderr_at,
                "last_worktree_at": last_worktree_at,
                "last_cpu_at": last_cpu_at,
                "last_target_runtime_activity_at": last_target_runtime_activity_at,
                "target_runtime_activity": dict(last_target_runtime_activity),
                "evidence_counts": dict(evidence_counts),
                "terminal_state": terminal_state,
                "output_capture": _output_capture_audit(),
            }

        def _final_output_ready(now: float) -> Optional[dict[str, Any]]:
            if self.config.backend != LLMBackend.CODEX_CLI:
                return None
            stable_seconds = max(0.0, float(_CODEX_FINAL_OUTPUT_STABLE_SECONDS))
            for output_path in final_output_paths:
                try:
                    stat_result = output_path.stat()
                except OSError:
                    continue
                size = int(getattr(stat_result, "st_size", 0) or 0)
                if size <= 0:
                    continue
                try:
                    text = output_path.read_text(encoding="utf-8").strip()
                except OSError:
                    continue
                if not text:
                    continue
                mtime_ns = int(getattr(stat_result, "st_mtime_ns", 0) or 0)
                state = final_output_state.setdefault(
                    output_path,
                    {"size": -1, "mtime_ns": -1, "changed_at": now},
                )
                if state.get("size") != size or state.get("mtime_ns") != mtime_ns:
                    state["size"] = size
                    state["mtime_ns"] = mtime_ns
                    state["changed_at"] = now
                    continue
                changed_at = float(state.get("changed_at") or now)
                if now - changed_at < stable_seconds:
                    continue
                return {
                    "path": str(output_path),
                    "size": size,
                    "mtime_ns": mtime_ns,
                    "stable_seconds": now - changed_at,
                    "required_stable_seconds": stable_seconds,
                }
            return None

        def _interactive_prompt_reason(stream_name: str, chunk: str) -> Optional[str]:
            normalized = _ANSI_ESCAPE_RE.sub("", chunk or "").lower()
            if _INTERACTIVE_ACK_PROMPT_PHRASE in normalized and (
                "type" in normalized or "incorrect response" in normalized
            ):
                # The only known acknowledgement gate is Claude's internet-mode
                # review prompt. When we have already fed the required phrase on
                # stdin the gate self-resolves, so the prompt text is benign —
                # aborting here would needlessly fail an otherwise-healthy launch.
                if auto_ack_internet_review:
                    return None
                return (
                    "CLI backend requires interactive confirmation, but Apex "
                    f"runs nested CLIs non-interactively ({stream_name})."
                )
            return None

        _emit_progress("active", source="spawned")

        while True:
            try:
                stream_name, chunk = stream_queue.get(timeout=0.25)
                now = time.time()
                if stream_name == "stdout":
                    stdout_capture.append(chunk)
                    last_stdout_at = now
                    # Phase B.5: feed stdout chunk into the turn parser
                    # so observers can run mid-stream. Best-effort —
                    # parser/observer errors are caught internally so a
                    # buggy hook can never crash the rollout loop.
                    _ingest_stdout_chunk_for_turns(chunk)
                else:
                    stderr_capture.append(chunk)
                    last_stderr_at = now
                evidence_counts[stream_name] += 1
                last_progress = now
                last_progress_source = stream_name
                _emit_progress("active", source=stream_name)
                interaction_reason = _interactive_prompt_reason(stream_name, chunk)
                if interaction_reason:
                    interaction_audit = _base_timeout_audit(now, "interactive_prompt")
                    interaction_audit["interactive_prompt"] = {
                        "stream": stream_name,
                        "signature": "acknowledgement_prompt",
                    }
                    _emit_progress("interactive_prompt", source=stream_name)
                    raise CLIProcessInteractionRequired(
                        interaction_reason,
                        output=_captured_stdout(),
                        stderr=_captured_stderr(),
                        interaction_audit=interaction_audit,
                    )
                stream_capture = stdout_capture if stream_name == "stdout" else stderr_capture
                if stream_capture.omitted_chars > 0:
                    output_audit = _base_timeout_audit(now, "output_limit")
                    output_audit["output_limit"] = {
                        "stream": stream_name,
                        "max_chars": stream_capture.max_chars,
                        "total_chars": stream_capture.total_chars,
                        "omitted_chars": stream_capture.omitted_chars,
                    }
                    _emit_progress("output_limit", source=stream_name)
                    raise CLIProcessOutputLimitExceeded(
                        (
                            f"CLI backend exceeded {stream_capture.max_chars} "
                            f"captured {stream_name} chars without completing."
                        ),
                        output=_captured_stdout(),
                        stderr=_captured_stderr(),
                        output_audit=output_audit,
                    )
                # If the observer requested abort, terminate the
                # subprocess via the rollout registry (when scoped) or
                # via _kill_process_tree as the safety fallback. We
                # leave the rest of the loop to drain the queue and
                # return as a normal completion-style exit.
                if observer_aborted:
                    self._abort_for_turn_observer(
                        process,
                        observer_aborted,
                    )
            except queue.Empty:
                pass

            if _cancel_requested():
                now = time.time()
                progress_audit = _base_timeout_audit(now, "scheduler_cancelled")
                progress_audit["scheduler_cancelled"] = True
                progress_audit["cancel_reason"] = _cancel_reason_text()
                _emit_progress("cancelled", source="scheduler_cancelled")
                raise CLIProcessProgressAbort(
                    progress_audit["cancel_reason"],
                    output=_captured_stdout(),
                    stderr=_captured_stderr(),
                    progress_audit=progress_audit,
                )

            if track_worktree:
                current_mtime = self._latest_worktree_mtime(Path(working_dir))
                if current_mtime > (last_worktree_mtime or 0.0):
                    last_worktree_mtime = current_mtime
                    last_progress = time.time()
                    last_worktree_at = last_progress
                    last_progress_source = "worktree"
                    evidence_counts["worktree"] += 1
                    _emit_progress("active", source="worktree")

            now = time.time()
            if now - last_activity_check >= activity_check_interval:
                process_entries = self._collect_process_tree_entries(process.pid)
                violation = self._process_tree_workspace_policy_violation(
                    process_entries,
                    working_dir,
                    target_runtime_enforced=target_runtime_enforced,
                )
                if violation is not None:
                    severity = str(violation.get("severity") or "fatal")
                    if severity in {"backend_helper", "blocked_by_policy"}:
                        # Soft violation — log once per (pid, command, target)
                        # tuple, record in audit, but do not abort the task.
                        # The monitored command set is read-only and the target
                        # is a system-helper path, so the agent's tool call
                        # cannot mutate state outside the workspace.
                        violation_key = (
                            int(violation.get("pid") or 0),
                            str(violation.get("command_name") or ""),
                            str(violation.get("path_token") or violation.get("cwd") or ""),
                        )
                        if violation_key not in seen_soft_policy_violations:
                            seen_soft_policy_violations.add(violation_key)
                            soft_policy_violation_log.append(
                                {**dict(violation), "observed_at": now}
                            )
                            logger.warning(
                                "Workspace policy soft-violation (downgraded): %s",
                                violation.get("reason"),
                            )
                            _emit_progress(
                                "policy_violation_soft",
                                source="workspace_policy",
                            )
                    else:
                        timeout_audit = _base_timeout_audit(now, "policy_violation")
                        timeout_audit["policy_violation"] = dict(violation)
                        timeout_audit["soft_policy_violations"] = list(soft_policy_violation_log)
                        _emit_progress("policy_violation", source="workspace_policy")
                        raise CLIProcessPolicyViolation(
                            str(
                                violation.get("reason")
                                or "CLI subprocess violated workspace policy."
                            ),
                            output=_captured_stdout(),
                            stderr=_captured_stderr(),
                            policy_audit=timeout_audit,
                        )
                current_process_cpu_seconds = sum(
                    float(entry.get("cpu_seconds", 0.0) or 0.0)
                    for entry in process_entries.values()
                )
                # Use a small delta so busy child processes still count as
                # progress on oversubscribed hosts where CPU time accrues slowly.
                #
                # Progress-based liveness (S3, folded UNCONDITIONALLY): agent
                # process-tree CPU advancement now refreshes the liveness clock
                # in *all* sessions, including editable ones. The previous
                # editable-session gate (``if not track_worktree``) suppressed
                # CPU evidence to reap "dead patchers that spin but never edit";
                # under the liveness directive that gate would instead starve a
                # silent CPU-bound patcher. The §4 task-level emergency *silence*
                # cap (output + worktree, CPU ignored) bounds a true
                # edit-test-spin livelock, so dropping the gate here is safe.
                if current_process_cpu_seconds > last_process_cpu_seconds + 0.01:
                    last_cpu_at = now
                    evidence_counts["cpu"] += 1
                    last_progress = now
                    last_progress_source = "cpu"
                    _emit_progress("active", source="cpu")
                last_process_cpu_seconds = current_process_cpu_seconds
                if (
                    isinstance(target_runtime_env, dict)
                    and target_runtime_env.get("APEX_TARGET_TOOL_CONTEXT")
                    and now - last_target_runtime_activity_check
                    >= _TARGET_RUNTIME_ACTIVITY_CHECK_INTERVAL_SECONDS
                ):
                    activity = _sample_target_runtime_process_activity(target_runtime_env)
                    last_target_runtime_activity_check = now
                    if activity:
                        last_target_runtime_activity = dict(activity)
                        last_target_runtime_activity_at = now
                        target_runtime_git_history_policy = _target_runtime_policy_value(
                            activity,
                            target_runtime_env,
                            "git_history_policy",
                            "blocked",
                        )
                        target_runtime_source_network_policy = _target_runtime_policy_value(
                            activity,
                            target_runtime_env,
                            "source_network_policy",
                            "unspecified",
                        )
                        target_runtime_filesystem_boundary_policy = _target_runtime_policy_value(
                            activity,
                            target_runtime_env,
                            "filesystem_boundary_policy",
                            "policy_enforced",
                        )
                        target_runtime_git_history_structural = (
                            _target_runtime_git_history_is_structural(
                                target_runtime_git_history_policy
                            )
                        )
                        target_runtime_source_network_structural = (
                            _target_runtime_source_network_is_structural(
                                target_runtime_source_network_policy
                            )
                        )
                        target_runtime_filesystem_boundary_structural = (
                            _target_runtime_filesystem_boundary_is_structural(
                                target_runtime_filesystem_boundary_policy
                            )
                        )
                        runtime_policy_violations = activity.get("policy_violations")
                        if (
                            isinstance(runtime_policy_violations, list)
                            and runtime_policy_violations
                        ):
                            runtime_policy_violations = [
                                violation
                                for violation in runtime_policy_violations
                                if not _target_runtime_policy_marker_is_structurally_redundant(
                                    violation,
                                    git_history_structural=(target_runtime_git_history_structural),
                                    source_network_structural=(
                                        target_runtime_source_network_structural
                                    ),
                                    filesystem_boundary_structural=(
                                        target_runtime_filesystem_boundary_structural
                                    ),
                                )
                            ]
                        if (
                            isinstance(runtime_policy_violations, list)
                            and runtime_policy_violations
                        ):
                            first_runtime_violation = (
                                runtime_policy_violations[0]
                                if isinstance(runtime_policy_violations[0], dict)
                                else {"reason": str(runtime_policy_violations[0])}
                            )
                            violation_reason = str(
                                first_runtime_violation.get("reason")
                                or "Target-runtime command violated workspace policy."
                            )
                            target_violation = {
                                "severity": "fatal",
                                "reason": violation_reason,
                                "target_runtime_policy_marker": dict(first_runtime_violation),
                            }
                            timeout_audit = _base_timeout_audit(
                                now,
                                "target_runtime_policy_violation",
                            )
                            timeout_audit["policy_violation"] = dict(target_violation)
                            timeout_audit["target_runtime_activity"] = dict(activity)
                            timeout_audit["soft_policy_violations"] = list(
                                soft_policy_violation_log
                            )
                            _emit_progress(
                                "policy_violation",
                                source="target_runtime_workspace_policy",
                            )
                            raise CLIProcessPolicyViolation(
                                violation_reason,
                                output=_captured_stdout(),
                                stderr=_captured_stderr(),
                                policy_audit=timeout_audit,
                            )
                        target_process_entries = activity.get("process_entries")
                        if isinstance(target_process_entries, dict) and target_process_entries:
                            target_workdir = str(
                                activity.get("workdir")
                                or target_runtime_env.get("APEX_TARGET_TOOL_WORKDIR")
                                or working_dir
                            )
                            target_violation = self._process_tree_workspace_policy_violation(
                                target_process_entries,
                                target_workdir,
                                target_runtime_enforced=True,
                                target_runtime_git_history_policy=(
                                    target_runtime_git_history_policy
                                ),
                                target_runtime_source_network_policy=(
                                    target_runtime_source_network_policy
                                ),
                                target_runtime_filesystem_boundary_policy=(
                                    target_runtime_filesystem_boundary_policy
                                ),
                            )
                            if target_violation is not None:
                                severity = str(target_violation.get("severity") or "fatal")
                                if severity in {"backend_helper", "blocked_by_policy"}:
                                    violation_key = (
                                        int(target_violation.get("pid") or 0),
                                        str(target_violation.get("command_name") or ""),
                                        str(
                                            target_violation.get("path_token")
                                            or target_violation.get("cwd")
                                            or ""
                                        ),
                                    )
                                    if violation_key not in seen_soft_policy_violations:
                                        seen_soft_policy_violations.add(violation_key)
                                        soft_policy_violation_log.append(
                                            {**dict(target_violation), "observed_at": now}
                                        )
                                        logger.warning(
                                            "Target-runtime workspace policy soft-violation "
                                            "(downgraded): %s",
                                            target_violation.get("reason"),
                                        )
                                        _emit_progress(
                                            "policy_violation_soft",
                                            source="target_runtime_workspace_policy",
                                        )
                                else:
                                    timeout_audit = _base_timeout_audit(
                                        now,
                                        "target_runtime_policy_violation",
                                    )
                                    timeout_audit["policy_violation"] = dict(target_violation)
                                    timeout_audit["target_runtime_activity"] = dict(activity)
                                    timeout_audit["soft_policy_violations"] = list(
                                        soft_policy_violation_log
                                    )
                                    _emit_progress(
                                        "policy_violation",
                                        source="target_runtime_workspace_policy",
                                    )
                                    raise CLIProcessPolicyViolation(
                                        str(
                                            target_violation.get("reason")
                                            or "Target-runtime subprocess violated workspace policy."
                                        ),
                                        output=_captured_stdout(),
                                        stderr=_captured_stderr(),
                                        policy_audit=timeout_audit,
                                    )
                        try:
                            current_target_cpu_ticks = float(activity.get("cpu_ticks") or 0.0)
                        except (TypeError, ValueError):
                            current_target_cpu_ticks = 0.0
                        current_target_pids = {
                            int(pid)
                            for pid in list(activity.get("pids") or [])
                            if isinstance(pid, int) or str(pid).isdigit()
                        }
                        if current_target_cpu_ticks > last_target_runtime_cpu_ticks + 0.01:
                            last_target_runtime_cpu_ticks = current_target_cpu_ticks
                            last_cpu_at = now
                            evidence_counts["target_runtime_cpu"] += 1
                            last_progress = now
                            last_progress_source = "target_runtime_cpu"
                            _emit_progress("active", source="target_runtime_cpu")
                        elif (
                            current_target_pids and current_target_pids != last_target_runtime_pids
                        ):
                            evidence_counts["target_runtime_process"] += 1
                            last_progress = now
                            last_progress_source = "target_runtime_process"
                            _emit_progress("active", source="target_runtime_process")
                        last_target_runtime_pids = current_target_pids
                last_activity_check = now

            process_exited = process.poll() is not None
            if (
                process_exited
                and stream_queue.empty()
                and not any(reader.is_alive() for reader in readers)
            ):
                break
            if not process_exited and not final_output_termination_requested:
                ready_output = _final_output_ready(now)
                if ready_output is not None:
                    final_output_completion = ready_output
                    final_output_termination_requested = True
                    last_progress = now
                    last_progress_source = "final_output_file"
                    evidence_counts["final_output_file"] = (
                        int(evidence_counts.get("final_output_file", 0) or 0) + 1
                    )
                    _emit_progress("final_output_file", source="final_output_file")
                    logger.info(
                        "CLI backend produced stable final output file before process exit; "
                        "terminating lingering process tree: %s",
                        ready_output.get("path"),
                    )
                    self._kill_process_tree(process, env=target_runtime_env)

            # S7 — in-flight-LLM-request freeze (bounded). While a running
            # process has an in-flight-request marker set, a long "thinking"
            # turn (zero S1-S5 but a live socket) is treated as ALIVE until the
            # marker itself exceeds max_inflight_request_seconds. The bound is
            # marker-age based, not last-progress based: host CPU or other
            # unrelated process evidence must not indefinitely extend a stale
            # provider-request marker.
            stall_silence = now - last_progress
            inflight_freeze = False
            if not process_exited and inflight_request_active is not None:
                try:
                    marker_started_at = inflight_request_active()
                except Exception:  # noqa: BLE001 - never let the reader crash the loop
                    marker_started_at = None
                if (
                    isinstance(marker_started_at, (int, float))
                    and now - float(marker_started_at) < max_inflight_request_seconds
                ):
                    inflight_freeze = True

            if (
                first_output_timeout_seconds
                and first_output_timeout_seconds > 0
                and not process_exited
                and not inflight_freeze
                and last_stdout_at is None
                and last_stderr_at is None
                and last_target_runtime_activity_at is None
                and now - started_at >= first_output_timeout_seconds
            ):
                timeout_audit = _base_timeout_audit(now, "stall_timeout")
                timeout_audit["stall_reason"] = "first_output_timeout"
                timeout_audit["first_output_timeout_seconds"] = first_output_timeout_seconds
                _emit_progress("stall_timeout", source="first_output_timeout")
                raise CLIProcessTimeout(
                    process.args,
                    float(first_output_timeout_seconds),
                    timeout_kind="stall",
                    output=_captured_stdout(),
                    stderr=_captured_stderr(),
                    timeout_audit=timeout_audit,
                )

            if (
                not process_exited
                and not inflight_freeze
                and progress_timeout is not None
                and progress_timeout > 0
                and stall_silence >= progress_timeout
            ):
                timeout_audit = _base_timeout_audit(now, "stall_timeout")
                _emit_progress("stall_timeout", source=last_progress_source)
                raise CLIProcessTimeout(
                    process.args,
                    float(progress_timeout),
                    timeout_kind="stall",
                    output=_captured_stdout(),
                    stderr=_captured_stderr(),
                    timeout_audit=timeout_audit,
                )
            # No-edit-progress kill (token-runaway governor, §13/§15). Reaps an
            # edit-capable stage that shows NO meaningful work for the whole
            # window: no new host stdout, no host-worktree edit, AND no
            # TARGET-RUNTIME (in-container) activity — only HOST-process CPU spin
            # (the 29M-token / 47-min / 0.0 runaway: host cpu high, container
            # idle, stdout=0, worktree=0). Host-process CPU alone is DELIBERATELY
            # excluded (it is what masked that round from the stall window), but
            # CONTAINER activity COUNTS as progress: the dominant in-container
            # agentic backend does its real work (edits + long test/build runs)
            # inside the container, surfacing as target-runtime CPU/process
            # activity rather than host-worktree mtime, so excluding it would
            # FALSE-KILL a legitimately-working agent (e.g. mid-way through
            # statsmodels' multi-minute test suite). Orthogonal to the stall kill,
            # respects the S7 in-flight freeze, and fires only for editable
            # stages. CLIProcessProgressAbort -> CLIModelResult(success=False): a
            # killed stage is FAILED, never accepted.
            if (
                track_worktree
                and not process_exited
                and not inflight_freeze
                and no_edit_progress_window
                and no_edit_progress_window > 0
            ):
                last_meaningful_at = started_at
                if isinstance(last_stdout_at, (int, float)):
                    last_meaningful_at = max(last_meaningful_at, last_stdout_at)
                if isinstance(last_worktree_at, (int, float)):
                    last_meaningful_at = max(last_meaningful_at, last_worktree_at)
                # In-container work surfaces here, NOT on the host worktree.
                if isinstance(last_target_runtime_activity_at, (int, float)):
                    last_meaningful_at = max(last_meaningful_at, last_target_runtime_activity_at)
                if now - last_meaningful_at >= no_edit_progress_window:
                    progress_audit = _base_timeout_audit(now, "no_edit_progress")
                    progress_audit["no_edit_progress_window_seconds"] = no_edit_progress_window
                    progress_audit["no_edit_silence_seconds"] = now - last_meaningful_at
                    _emit_progress("no_edit_progress", source=last_progress_source)
                    raise CLIProcessProgressAbort(
                        "CLI agent produced no new output, no worktree edit, and no "
                        "in-container activity within the no-edit-progress window "
                        f"({no_edit_progress_window:.0f}s) while consuming host CPU — "
                        "reaping as no-meaningful-progress.",
                        output=_captured_stdout(),
                        stderr=_captured_stderr(),
                        progress_audit=progress_audit,
                    )
            # Strict hard timeout: configs that opt in need a real per-call
            # wall-clock cap even when the process is making progress. This is
            # a backend-level backstop, not a substitute for evidence-driven
            # orchestration decisions.
            if (
                not process_exited
                and hard_timeout is not None
                and hard_timeout > 0
                and now - started_at >= hard_timeout
            ):
                timeout_audit = _base_timeout_audit(now, "hard_timeout")
                _emit_progress("hard_timeout", source=last_progress_source)
                raise CLIProcessTimeout(
                    process.args,
                    float(hard_timeout),
                    timeout_kind="hard",
                    output=_captured_stdout(),
                    stderr=_captured_stderr(),
                    timeout_audit=timeout_audit,
                )

        for reader in readers:
            reader.join(timeout=0.1)

        # Phase B.5: flush any remaining buffered stdout (the carry from
        # an unterminated final line) and finalize the turn parser so
        # the terminal turn fires its observer too.
        if _turn_parser is not None:
            try:
                tail = _turn_stdout_carry[0]
                if tail:
                    closed = _turn_parser.feed_line(tail)
                    if closed is not None:
                        _dispatch_turn_observer(closed)
                    _turn_stdout_carry[0] = ""
                terminal_turn = _turn_parser.finalize()
                if terminal_turn is not None:
                    _dispatch_turn_observer(terminal_turn)
            except Exception as exc:  # noqa: BLE001 - never crash the loop
                logger.warning(
                    "Turn parser finalize raised %s: %s; ignoring.",
                    type(exc).__name__,
                    exc,
                )

        completed_at = time.time()
        terminal_state = (
            "completed_after_final_output" if final_output_completion is not None else "completed"
        )
        timeout_audit = _base_timeout_audit(completed_at, terminal_state)
        if final_output_completion is not None:
            timeout_audit["final_output_completion"] = dict(final_output_completion)
        if soft_policy_violation_log:
            timeout_audit["soft_policy_violations"] = list(soft_policy_violation_log)
        if course_corrections:
            timeout_audit["course_corrections"] = list(course_corrections)
        if observer_aborted:
            timeout_audit["observer_aborted"] = dict(observer_aborted)
            timeout_audit["terminal_state"] = "observer_aborted"
        _emit_progress("completed", source=last_progress_source)
        return _captured_stdout(), _captured_stderr(), timeout_audit

    def _effective_progress_timeout_seconds(
        self,
        *,
        track_worktree: bool,
        hard_timeout_seconds: Optional[int] = None,
    ) -> Optional[float]:
        # Progress-based liveness (K2): the watchdog's stall window is the
        # uniform STALL_WINDOW, DECOUPLED from ``cli_timeout`` and the hard
        # timeout. Shrinking a planner phase budget (``hard_timeout_seconds``)
        # can no longer shrink the stall window — it is purely a function of
        # ``rollout.stall_window_seconds``. ``track_worktree`` no longer gates
        # the window size (S3/S4 are folded unconditionally in the loop), so
        # editable and read-only sessions share the same generous window.
        return self._liveness_stall_window_seconds()

    def _effective_hard_timeout_progress_grace_seconds(
        self,
        *,
        hard_timeout_seconds: Optional[int] = None,
    ) -> float:
        hard_timeout = (
            float(hard_timeout_seconds)
            if isinstance(hard_timeout_seconds, (int, float)) and hard_timeout_seconds > 0
            else None
        )
        if hard_timeout is None:
            return _HARD_TIMEOUT_PROGRESS_GRACE_MIN_SECONDS
        return min(
            600.0,
            max(
                _HARD_TIMEOUT_PROGRESS_GRACE_MIN_SECONDS,
                hard_timeout * _HARD_TIMEOUT_PROGRESS_GRACE_RATIO,
            ),
        )

    def _read_stream_to_queue(
        self,
        stream: Any,
        stream_name: str,
        stream_queue: "queue.Queue[tuple[str, str]]",
    ) -> None:
        try:
            while True:
                chunk = stream.readline(_CLI_STREAM_READ_CHARS)
                if not chunk:
                    break
                stream_queue.put((stream_name, chunk))
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _collect_process_tree_entries(self, root_pid: int) -> dict[int, dict[str, Any]]:
        # Audit H2: on macOS hosts with thousands of processes, ``ps -axo``
        # itself can take several seconds; cap to keep the orphan sweeper
        # from blocking the parent.
        try:
            result = subprocess.run(
                ["ps", "-axo", "pid=,ppid=,time=,command="],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return {}
        if result.returncode != 0:
            return {}

        children_by_parent: dict[int, list[int]] = {}
        entries_by_pid: dict[int, dict[str, Any]] = {}
        for line in result.stdout.splitlines():
            parts = line.strip().split(None, 3)
            if len(parts) != 4:
                continue
            try:
                pid = int(parts[0])
                ppid = int(parts[1])
            except ValueError:
                continue
            entries_by_pid[pid] = {
                "ppid": ppid,
                "cpu_seconds": self._parse_ps_time_to_seconds(parts[2]),
                "command": parts[3].strip(),
            }
            children_by_parent.setdefault(ppid, []).append(pid)

        if root_pid not in entries_by_pid:
            return {}

        tree_entries: dict[int, dict[str, Any]] = {}
        stack = [root_pid]
        while stack:
            pid = stack.pop()
            if pid in tree_entries:
                continue
            entry = entries_by_pid.get(pid)
            if entry is None:
                continue
            tree_entries[pid] = dict(entry)
            stack.extend(children_by_parent.get(pid, []))
        return tree_entries

    @staticmethod
    def _looks_like_shell_env_assignment(token: str) -> bool:
        return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", str(token or "")))

    def _tokenize_process_command(self, command: str) -> list[str]:
        raw = str(command or "").strip()
        if not raw:
            return []
        if "\x1f" in raw:
            return [token for token in raw.split("\x1f") if token]
        try:
            return shlex.split(raw, posix=True)
        except ValueError:
            return raw.split()

    def _sampled_cwd_text_looks_like_command(self, cwd: str) -> bool:
        raw = str(cwd or "").strip()
        if not raw:
            return False
        tokens = self._tokenize_process_command(raw)
        if len(tokens) < 2:
            return False
        command_name = Path(tokens[0]).name
        if command_name.startswith("qemu-"):
            return True
        command_names = _WORKSPACE_POLICY_MONITORED_COMMANDS | _TARGET_RUNTIME_DYNAMIC_COMMANDS
        return raw.startswith(("/bin/", "/usr/bin/", "/usr/local/bin/", "/opt/", "/testbed/")) and (
            command_name in command_names
        )

    def _strip_process_wrapper_tokens(self, tokens: list[str]) -> list[str]:
        if not tokens:
            return []
        index = 0
        while index < len(tokens) and self._looks_like_shell_env_assignment(tokens[index]):
            index += 1
        if index < len(tokens) and tokens[index] == "env":
            index += 1
            while index < len(tokens):
                token = tokens[index]
                if token == "--":
                    index += 1
                    break
                if self._looks_like_shell_env_assignment(token):
                    index += 1
                    continue
                if token.startswith("-"):
                    index += 1
                    if token == "-u" and index < len(tokens):
                        index += 1
                    continue
                break
        if index < len(tokens) and Path(tokens[index]).name.startswith("qemu-"):
            index += 1
            while index < len(tokens) and str(tokens[index]).startswith("-"):
                option = str(tokens[index])
                index += 1
                if option in {"-0", "-L", "-E", "-U", "-cpu", "-B", "-R", "-singlestep"}:
                    index += 1
            if index < len(tokens):
                index += 1
        while index < len(tokens):
            command_name = Path(tokens[index]).name
            if command_name == "command":
                index += 1
                continue
            if command_name in {"timeout", "gtimeout"}:
                index += 1
                while index < len(tokens):
                    token = tokens[index]
                    if token == "--":
                        index += 1
                        break
                    if token.startswith("-"):
                        index += 1
                        if token in {"-k", "--kill-after", "-s", "--signal"} and index < len(
                            tokens
                        ):
                            index += 1
                        continue
                    index += 1
                    break
                continue
            break
        return tokens[index:]

    def _command_path_operands(self, command_name: str, tokens: list[str]) -> list[str]:
        args = list(tokens[1:])
        if not args:
            return []
        if command_name == "find":
            operands: list[str] = []
            for token in args:
                if token == "--":
                    continue
                if token.startswith("-") or token in {"(", ")", "!", ",", "o", "-o"}:
                    break
                operands.append(token)
            return operands or ["."]

        positional: list[str] = []
        index = 0
        while index < len(args):
            token = args[index]
            if token == "--":
                positional.extend(args[index + 1 :])
                break
            if token.startswith("-"):
                if command_name in {"rg", "grep", "ag"} and token in {
                    "-A",
                    "-B",
                    "-C",
                    "-E",
                    "-F",
                    "-P",
                    "-e",
                    "-f",
                    "-g",
                    "-m",
                    "-M",
                    "-r",
                    "-t",
                    "-T",
                    "--context",
                    "--glob",
                    "--max-count",
                    "--replace",
                    "--type",
                    "--type-not",
                }:
                    index += 2
                    continue
                if command_name in {"fd", "fdfind"} and token in {
                    "-c",
                    "-d",
                    "-e",
                    "-E",
                    "-g",
                    "-j",
                    "-S",
                    "-t",
                    "--color",
                    "--exclude",
                    "--extension",
                    "--max-depth",
                    "--size",
                    "--threads",
                    "--type",
                }:
                    index += 2
                    continue
                index += 1
                continue
            positional.append(token)
            index += 1

        if command_name in {"rg", "grep", "ag", "fd", "fdfind"}:
            return positional[1:] if len(positional) > 1 else []
        return positional

    def _resolve_monitored_path_token(
        self,
        token: str,
        *,
        working_dir: Path,
        process_cwd: Optional[Path],
    ) -> Optional[Path]:
        raw = str(token or "").strip()
        if not raw or raw in {"|", "||", "&&", ";"}:
            return None
        base_dir = process_cwd or working_dir
        expanded = raw
        if raw.startswith("${PWD}"):
            expanded = str(base_dir) + raw[len("${PWD}") :]
        elif raw.startswith("$PWD"):
            expanded = str(base_dir) + raw[len("$PWD") :]
        elif raw.startswith("${HOME}"):
            expanded = str(working_dir) + raw[len("${HOME}") :]
        elif raw.startswith("$HOME"):
            expanded = str(working_dir) + raw[len("$HOME") :]
        elif raw == "~" or raw.startswith("~/"):
            expanded = str(working_dir) + raw[1:]
        elif raw.startswith("/"):
            expanded = raw
        elif raw in {".", ".."} or raw.startswith("./") or raw.startswith("../"):
            expanded = str(base_dir / raw)
        else:
            return None
        try:
            return Path(expanded).resolve()
        except OSError:
            return None

    def _path_escapes_workspace(
        self,
        candidate: Optional[Path],
        *,
        working_dir: Path,
    ) -> bool:
        if candidate is None:
            return False
        try:
            candidate.resolve().relative_to(working_dir)
        except ValueError:
            return True
        return False

    def _process_cwd(self, pid: int) -> Optional[Path]:
        return _process_cwd(pid)

    def _path_token_is_explicit_workspace_escape(self, token: str) -> bool:
        raw = str(token or "").strip()
        if not raw:
            return False
        if raw.startswith("/"):
            return True
        if raw.startswith("$PWD") or raw.startswith("${PWD}"):
            return True
        if raw.startswith("$HOME") or raw.startswith("${HOME}") or raw.startswith("~"):
            return True
        return raw == ".." or raw.startswith("../")

    def _looks_like_backend_helper_workspace_policy_violation(
        self,
        *,
        process_cwd: Optional[Path],
        path_tokens: list[str],
        working_dir: Path,
    ) -> bool:
        if process_cwd is None:
            return False
        explicit_escape_tokens = [
            token for token in path_tokens if self._path_token_is_explicit_workspace_escape(token)
        ]
        if explicit_escape_tokens:
            for token in explicit_escape_tokens:
                if token.startswith("/") and self._path_text_looks_like_backend_runtime_helper(
                    token
                ):
                    continue
                resolved = self._resolve_monitored_path_token(
                    token,
                    working_dir=working_dir,
                    process_cwd=process_cwd,
                )
                if self._path_escapes_workspace(resolved, working_dir=working_dir):
                    return False
        cwd_text = str(process_cwd)
        if any(marker in cwd_text for marker in _WORKSPACE_POLICY_BACKEND_HELPER_MARKERS):
            return True
        if self._path_text_looks_like_backend_runtime_helper(cwd_text):
            return True
        return False

    @staticmethod
    def _path_text_looks_like_backend_runtime_helper(path_text: str) -> bool:
        text = str(path_text or "")
        return text.startswith(_WORKSPACE_POLICY_BACKEND_RUNTIME_CWD_PREFIXES) and any(
            marker in text for marker in _WORKSPACE_POLICY_BACKEND_RUNTIME_CWD_MARKERS
        )

    def _process_cwd_looks_like_backend_runtime_helper(
        self,
        process_cwd: Optional[Path],
    ) -> bool:
        if process_cwd is None:
            return False
        cwd_text = str(process_cwd)
        if any(marker in cwd_text for marker in _WORKSPACE_POLICY_BACKEND_HELPER_MARKERS):
            return True
        return self._path_text_looks_like_backend_runtime_helper(cwd_text)

    @staticmethod
    def _process_has_ancestor_command_marker(
        process_entries: dict[int, dict[str, Any]],
        pid: int,
        markers: tuple[str, ...],
    ) -> bool:
        current = int(process_entries.get(pid, {}).get("ppid") or 0)
        seen: set[int] = set()
        while current and current not in seen:
            seen.add(current)
            entry = process_entries.get(current)
            if entry is None:
                return False
            command = str(entry.get("command") or "")
            if any(marker in command for marker in markers):
                return True
            current = int(entry.get("ppid") or 0)
        return False

    @staticmethod
    def _path_looks_like_target_runtime_executable(token: str) -> bool:
        path_text = str(token or "")
        return (
            "apex_target_tool.py" in path_text
            or "target_tool_shims" in path_text
            or "/.runtime/.venv/bin/" in path_text
        )

    def _target_runtime_shell_wrapper_is_allowed(self, tokens: list[str]) -> bool:
        """Allow CLI-owned shell wrappers so target PATH shims can run.

        Codex/Claude/Gemini commonly invoke tools as `/bin/zsh -c <cmd>` or
        `/bin/bash -lc <cmd>`. Blocking that parent shell prevents the
        PATH-prefixed target shims from ever intercepting `python`, `pytest`,
        `rg`, `git`, and similar commands. Keep the exception scoped to shell
        `-c` wrappers and still reject obvious absolute host dynamic tools.
        """

        if not tokens or self._workspace_policy_command_name(tokens[0]) not in {
            "bash",
            "sh",
            "zsh",
        }:
            return False
        command_string: Optional[str] = None
        index = 1
        while index < len(tokens):
            token = tokens[index]
            if token == "--":
                index += 1
                continue
            if token.startswith("-"):
                if "c" in token and index + 1 < len(tokens):
                    command_string = tokens[index + 1]
                    break
                index += 1
                continue
            break
        if command_string is None:
            script_index = 1
            while script_index < len(tokens):
                token = tokens[script_index]
                if token == "--":
                    script_index += 1
                    continue
                if token.startswith("-"):
                    script_index += 1
                    continue
                return self._path_looks_like_target_runtime_executable(token)
            return False

        for payload_token in self._tokenize_process_command(command_string):
            if not os.path.isabs(payload_token):
                continue
            if self._path_looks_like_target_runtime_executable(payload_token):
                continue
            if (
                self._workspace_policy_command_name(payload_token)
                in _TARGET_RUNTIME_DYNAMIC_COMMANDS
            ):
                return False
        return True

    def _target_runtime_shell_command_argument(self, tokens: list[str]) -> str:
        if not tokens or self._workspace_policy_command_name(tokens[0]) not in {
            "bash",
            "sh",
            "zsh",
        }:
            return ""
        index = 1
        while index < len(tokens):
            token = tokens[index]
            if token == "--":
                index += 1
                continue
            if token.startswith("-"):
                if "c" in token and index + 1 < len(tokens):
                    return str(tokens[index + 1] or "")
                index += 1
                continue
            break
        return ""

    def _target_runtime_shell_policy_payloads(self, tokens: list[str]) -> list[str]:
        command_string = self._target_runtime_shell_command_argument(tokens)
        if not command_string:
            return []
        command_tokens = self._tokenize_process_command(command_string)
        eval_payloads: list[str] = []
        for index, token in enumerate(command_tokens[:-1]):
            if Path(token).name == "eval":
                payload = str(command_tokens[index + 1] or "").strip()
                if payload:
                    eval_payloads.append(payload)
        return eval_payloads or [command_string]

    @staticmethod
    def _strip_shell_path_candidate(token: str) -> str:
        text = str(token or "").strip()
        if not text:
            return ""
        text = re.sub(r"^\d*(?:>>?|<<?|<>|&>)", "", text).strip()
        while text and text[0] in "({[":
            text = text[1:].strip()
        while text and text[-1] in ";,)]}":
            text = text[:-1].strip()
        return text

    @classmethod
    def _workspace_policy_command_name(cls, token: str) -> str:
        name = Path(cls._strip_shell_path_candidate(token)).name
        if name.endswith(".apex-real"):
            name = name[: -len(".apex-real")]
        return name

    def _workspace_policy_path_candidates_from_token(self, token: str) -> list[str]:
        text = self._strip_shell_path_candidate(token)
        if not text:
            return []
        candidates: list[str] = []
        if self._looks_like_shell_env_assignment(text):
            _, raw_value = text.split("=", 1)
            candidates.extend(part for part in raw_value.split(os.pathsep) if part)
        elif text.startswith("--") and "=" in text:
            _, raw_value = text.split("=", 1)
            candidates.append(raw_value)
        else:
            candidates.append(text)
        candidates.extend(self._workspace_policy_embedded_absolute_paths(text))
        return [
            candidate
            for candidate in (
                self._strip_shell_path_candidate(candidate) for candidate in candidates
            )
            if self._workspace_policy_token_looks_like_path_candidate(candidate)
        ]

    @staticmethod
    def _workspace_policy_embedded_absolute_paths(text: str) -> list[str]:
        candidates: list[str] = []
        embedded_path_re = (
            rf"(^|[\s=([{{,:])"
            rf"(/(?!/)(?:{_WORKSPACE_POLICY_EMBEDDED_ABSOLUTE_PATH_ROOT_RE})"
            r"(?:/|$)[^\s'\"`$;&|<>\)\]\}]*)"
        )
        for match in re.finditer(
            embedded_path_re,
            str(text or ""),
        ):
            candidate = match.group(2).rstrip(".,:;)]}")
            if candidate:
                candidates.append(candidate)
        return candidates

    @staticmethod
    def _workspace_policy_allows_external_shell_path(
        candidate: str,
        *,
        working_dir: Path,
        process_cwd: Optional[Path],
        source_token: str = "",
        previous_token: str = "",
        next_token: str = "",
        self_created_temp_outputs: Optional[set[str]] = None,
    ) -> bool:
        text = str(candidate or "").strip()
        if text == "/dev/null" or text.startswith("/dev/fd/"):
            return True
        if CLIModelClient._workspace_policy_path_literal_is_metadata_filter(
            text,
            previous_token=previous_token,
            next_token=next_token,
        ):
            return True
        if text in (
            self_created_temp_outputs or set()
        ) and CLIModelClient._workspace_policy_path_is_transient_output(text):
            return True
        if CLIModelClient._workspace_policy_path_is_transient_script_argument(
            text,
            previous_token=previous_token,
        ):
            return True
        if CLIModelClient._workspace_policy_path_is_transient_output(
            text
        ) and CLIModelClient._workspace_policy_token_is_output_path_option(
            source_token, previous_token
        ):
            return True
        if text in _WORKSPACE_POLICY_CONTAINER_SHELL_EXECUTABLES:
            return Path(CLIModelClient._strip_shell_path_candidate(previous_token)).name in {
                "exec",
                "command",
            }
        if CLIModelClient._workspace_policy_path_is_current_cli_task_output(
            text,
            working_dir=working_dir,
            process_cwd=process_cwd,
        ):
            return True
        return False

    @staticmethod
    def _workspace_policy_path_literal_is_metadata_filter(
        candidate: str,
        *,
        previous_token: str,
        next_token: str,
    ) -> bool:
        text = str(candidate or "").strip()
        if text not in {"/.git", "/.hg", "/.svn"}:
            return False
        previous_text = CLIModelClient._strip_shell_path_candidate(previous_token).lower()
        next_text = CLIModelClient._strip_shell_path_candidate(next_token).lower()
        return previous_text in {"if", "elif", "while", "and", "or", "not"} and (
            next_text in {"in", "not", "==", "!="}
        )

    @classmethod
    def _workspace_policy_path_literal_is_text_filter_pattern(
        cls,
        tokens: list[str],
        candidate_index: int,
    ) -> bool:
        if candidate_index < 0 or candidate_index >= len(tokens):
            return False
        candidate = cls._strip_shell_path_candidate(tokens[candidate_index])
        if not candidate.startswith("/"):
            return False

        segment_start = candidate_index
        while segment_start > 0 and str(tokens[segment_start - 1]) not in {
            "|",
            "||",
            "&&",
            ";",
        }:
            segment_start -= 1
        command_index = segment_start
        while command_index < candidate_index and cls._looks_like_shell_env_assignment(
            cls._strip_shell_path_candidate(tokens[command_index])
        ):
            command_index += 1
        if command_index >= candidate_index:
            return False

        command_name = cls._workspace_policy_command_name(tokens[command_index])
        if command_name not in {"grep", "egrep", "fgrep", "rg", "ag", "awk", "sed"}:
            return False

        grep_value_options = {
            "-A",
            "-B",
            "-C",
            "-D",
            "-d",
            "-m",
            "--after-context",
            "--before-context",
            "--context",
            "--devices",
            "--directories",
            "--max-count",
        }
        options_with_values = {
            "grep": grep_value_options,
            "egrep": grep_value_options,
            "fgrep": grep_value_options,
            "rg": {
                "-A",
                "-B",
                "-C",
                "-g",
                "-m",
                "-M",
                "-r",
                "-t",
                "-T",
                "--after-context",
                "--before-context",
                "--context",
                "--glob",
                "--max-count",
                "--max-columns",
                "--replace",
                "--type",
                "--type-not",
            },
            "ag": {
                "-A",
                "-B",
                "-C",
                "-G",
                "-g",
                "-m",
                "--after",
                "--before",
                "--context",
                "--file-search-regex",
                "--ignore",
                "--max-count",
            },
            "awk": {"-F", "-v"},
            "sed": {"-e"},
        }
        pattern_options = {
            "grep": {"-e", "--regexp"},
            "egrep": {"-e", "--regexp"},
            "fgrep": {"-e", "--regexp"},
            "rg": {"-e", "--regexp"},
            "ag": {"-e"},
            "sed": {"-e"},
        }
        positional_count = 0
        index = command_index + 1
        while index <= candidate_index:
            current = cls._strip_shell_path_candidate(tokens[index])
            if not current:
                index += 1
                continue
            if current == "--":
                index += 1
                continue
            if current.startswith("-") and current != "-":
                option = current.split("=", 1)[0]
                if option in pattern_options.get(command_name, set()):
                    if index == candidate_index:
                        return False
                    if "=" in current:
                        index += 1
                        continue
                    if index + 1 == candidate_index:
                        return True
                    index += 2
                    continue
                if option in options_with_values.get(command_name, set()):
                    if index == candidate_index:
                        return False
                    index += 1 if "=" in current else 2
                    continue
                index += 1
                continue
            if index == candidate_index:
                return positional_count == 0
            positional_count += 1
            index += 1
        return False

    def _workspace_policy_lone_slash_is_non_operand(
        self,
        tokens: list[str],
        candidate_index: int,
    ) -> bool:
        if candidate_index < 0 or candidate_index >= len(tokens):
            return False
        if self._strip_shell_path_candidate(tokens[candidate_index]) != "/":
            return False

        separators = {"|", "||", "&&", ";"}
        segment_start = candidate_index
        while segment_start > 0 and str(tokens[segment_start - 1]) not in separators:
            segment_start -= 1
        segment_end = candidate_index + 1
        while segment_end < len(tokens) and str(tokens[segment_end]) not in separators:
            segment_end += 1

        command_index = segment_start
        while command_index < segment_end and self._looks_like_shell_env_assignment(
            self._strip_shell_path_candidate(tokens[command_index])
        ):
            command_index += 1
        if command_index >= segment_end:
            return True

        command_name = self._workspace_policy_command_name(tokens[command_index])
        if command_name not in _WORKSPACE_POLICY_MONITORED_COMMANDS:
            return True

        segment_tokens = [
            self._strip_shell_path_candidate(token) for token in tokens[command_index:segment_end]
        ]
        operands = self._command_path_operands(command_name, segment_tokens)
        return "/" not in operands

    @staticmethod
    def _workspace_policy_token_is_output_path_option(*tokens: str) -> bool:
        for raw in tokens:
            token = str(raw or "").strip().lower()
            if not token.startswith("-"):
                continue
            option = token.split("=", 1)[0]
            if any(
                marker in option for marker in _WORKSPACE_POLICY_TRANSIENT_OUTPUT_OPTION_MARKERS
            ):
                return True
        return False

    @staticmethod
    def _workspace_policy_path_is_transient_output(candidate: str) -> bool:
        text = str(candidate or "").strip()
        if not text or not text.startswith("/"):
            return False
        if any(marker in text for marker in ("*", "?", "[")):
            return False
        path = PurePosixPath(text)
        if ".." in path.parts or path.name in {"", ".", ".."}:
            return False
        normalized = str(path)
        return any(
            normalized == root.rstrip("/") or normalized.startswith(root)
            for root in _WORKSPACE_POLICY_TRANSIENT_OUTPUT_ROOTS
        )

    @staticmethod
    def _workspace_policy_shell_output_redirection_candidate(
        tokens: list[str],
        index: int,
    ) -> str:
        token = str(tokens[index] if index < len(tokens) else "").strip()
        if not token:
            return ""
        if token in _WORKSPACE_POLICY_SHELL_OUTPUT_REDIRECT_TOKENS:
            if index + 1 < len(tokens):
                return CLIModelClient._strip_shell_path_candidate(tokens[index + 1])
            return ""
        for prefix in sorted(
            _WORKSPACE_POLICY_SHELL_OUTPUT_REDIRECT_TOKENS,
            key=len,
            reverse=True,
        ):
            if token.startswith(prefix) and len(token) > len(prefix):
                return CLIModelClient._strip_shell_path_candidate(token[len(prefix) :])
        return ""

    @staticmethod
    def _workspace_policy_self_created_temp_outputs(tokens: list[str]) -> set[str]:
        outputs: set[str] = set()
        for index, _token in enumerate(tokens):
            candidate = CLIModelClient._workspace_policy_shell_output_redirection_candidate(
                tokens,
                index,
            )
            if CLIModelClient._workspace_policy_path_is_transient_output(candidate):
                outputs.add(candidate)
        return outputs

    @staticmethod
    def _workspace_policy_encoded_path_segment(path: Path) -> str:
        return "".join(ch if ch.isalnum() else "-" for ch in str(path))

    @staticmethod
    def _workspace_policy_path_is_current_cli_task_output(
        candidate: str,
        *,
        working_dir: Path,
        process_cwd: Optional[Path],
    ) -> bool:
        text = str(candidate or "").strip()
        if not text.startswith("/tmp/claude-"):
            return False
        path = PurePosixPath(text)
        parts = path.parts
        if ".." in parts:
            return False
        if len(parts) != 7:
            return False
        if parts[0] != "/" or parts[1] != "tmp" or not parts[2].startswith("claude-"):
            return False
        if parts[-2] != "tasks" or not parts[-1].endswith(
            _WORKSPACE_POLICY_CLAUDE_TASK_OUTPUT_SUFFIX
        ):
            return False
        current_workspace_segments = {
            CLIModelClient._workspace_policy_encoded_path_segment(working_dir),
        }
        if process_cwd is not None:
            current_workspace_segments.add(
                CLIModelClient._workspace_policy_encoded_path_segment(process_cwd)
            )
        return parts[3] in current_workspace_segments

    @staticmethod
    def _workspace_policy_token_looks_like_path_candidate(candidate: str) -> bool:
        text = str(candidate or "").strip()
        if not text or not text.startswith("/"):
            return True
        if text == "/":
            return True
        first_segment = text[1:].split("/", 1)[0]
        return bool(re.match(r"^[A-Za-z0-9._+-]+$", first_segment))

    def _shell_payload_workspace_path_violation(
        self,
        payload: str,
        *,
        working_dir: Path,
        process_cwd: Optional[Path],
        cross_payload_temp_outputs: Optional[set[str]] = None,
    ) -> Optional[dict[str, Any]]:
        payload_tokens = self._tokenize_process_command(payload)
        self_created_temp_outputs = self._workspace_policy_self_created_temp_outputs(payload_tokens)
        if cross_payload_temp_outputs:
            self_created_temp_outputs.update(cross_payload_temp_outputs)
        payload_cwd = process_cwd or working_dir
        pending_cd_target_index: Optional[int] = None
        pending_cd_cwd: Optional[Path] = None
        for index, token in enumerate(payload_tokens):
            previous_token = payload_tokens[index - 1] if index > 0 else ""
            next_token = payload_tokens[index + 1] if index + 1 < len(payload_tokens) else ""
            stripped_token = self._strip_shell_path_candidate(token)
            if (
                stripped_token == "cd"
                and index + 1 < len(payload_tokens)
                and str(payload_tokens[index + 1]) not in {"|", "||", "&&", ";"}
            ):
                cd_target = self._strip_shell_path_candidate(payload_tokens[index + 1])
                pending_cd_target_index = index + 1
                if cd_target.startswith(("/", "$PWD", "${PWD}", "$HOME", "${HOME}", "~")):
                    pending_cd_cwd = self._resolve_monitored_path_token(
                        cd_target,
                        working_dir=working_dir,
                        process_cwd=payload_cwd,
                    )
                elif cd_target:
                    try:
                        pending_cd_cwd = (payload_cwd / cd_target).resolve()
                    except OSError:
                        pending_cd_cwd = None
            for candidate in self._workspace_policy_path_candidates_from_token(token):
                if self._workspace_policy_lone_slash_is_non_operand(
                    payload_tokens,
                    index,
                ):
                    continue
                if self._workspace_policy_path_literal_is_text_filter_pattern(
                    payload_tokens,
                    index,
                ):
                    continue
                if self._workspace_policy_allows_external_shell_path(
                    candidate,
                    working_dir=working_dir,
                    process_cwd=payload_cwd,
                    source_token=token,
                    previous_token=previous_token,
                    next_token=next_token,
                    self_created_temp_outputs=self_created_temp_outputs,
                ):
                    continue
                resolved_path = self._resolve_monitored_path_token(
                    candidate,
                    working_dir=working_dir,
                    process_cwd=payload_cwd,
                )
                if not self._path_escapes_workspace(
                    resolved_path,
                    working_dir=working_dir,
                ):
                    continue
                if self._path_resolves_to_system_helper_target(resolved_path):
                    continue
                return {
                    "path_token": candidate,
                    "resolved_path": str(resolved_path) if resolved_path is not None else None,
                }
            if (
                pending_cd_target_index == index
                and pending_cd_cwd is not None
                and not self._path_escapes_workspace(
                    pending_cd_cwd,
                    working_dir=working_dir,
                )
            ):
                payload_cwd = pending_cd_cwd
                pending_cd_target_index = None
                pending_cd_cwd = None
        return None

    def _target_runtime_shell_workspace_policy_violation(
        self,
        tokens: list[str],
        *,
        working_dir: Path,
        process_cwd: Optional[Path],
        cross_payload_temp_outputs: Optional[set[str]] = None,
    ) -> Optional[dict[str, Any]]:
        for payload in self._target_runtime_shell_policy_payloads(tokens):
            violation = self._shell_payload_workspace_path_violation(
                payload,
                working_dir=working_dir,
                process_cwd=process_cwd,
                cross_payload_temp_outputs=cross_payload_temp_outputs,
            )
            if violation is not None:
                violation["shell_payload"] = payload
                return violation
        return None

    def _target_runtime_shell_temp_outputs(
        self,
        process_entries: dict[int, dict[str, Any]],
    ) -> set[str]:
        outputs: set[str] = set()
        for entry in process_entries.values():
            raw_argv = entry.get("argv")
            if isinstance(raw_argv, list) and raw_argv:
                argv_tokens = [str(token) for token in raw_argv if str(token)]
            else:
                argv_tokens = self._tokenize_process_command(str(entry.get("command") or ""))
            tokens = self._strip_process_wrapper_tokens(argv_tokens)
            if not tokens:
                continue
            if self._workspace_policy_command_name(tokens[0]) not in {"bash", "sh", "zsh"}:
                continue
            for payload in self._target_runtime_shell_policy_payloads(tokens):
                payload_tokens = self._tokenize_process_command(payload)
                outputs.update(self._workspace_policy_self_created_temp_outputs(payload_tokens))
        return outputs

    def _workspace_policy_resolve_repo_local_path_token(
        self,
        token: str,
        *,
        working_dir: Path,
        process_cwd: Optional[Path],
    ) -> Optional[Path]:
        raw = str(token or "").strip()
        if not raw or raw.startswith("-"):
            return None
        if raw.startswith("/"):
            return self._resolve_monitored_path_token(
                raw,
                working_dir=working_dir,
                process_cwd=process_cwd,
            )
        if raw.startswith(("./", "../")) or "/" in raw:
            base_dir = process_cwd or working_dir
            try:
                return (base_dir / raw).resolve()
            except OSError:
                return None
        return None

    @staticmethod
    def _workspace_policy_path_is_test_file(path_token: str) -> bool:
        path = PurePosixPath(str(path_token or ""))
        parts = [part.lower() for part in path.parts]
        name = path.name.lower()
        return any(part in {"test", "tests", "testing", "__tests__"} for part in parts) or (
            name.startswith("test_")
            or name.endswith("_test.py")
            or name.endswith(".test.py")
            or name.endswith(".spec.py")
        )

    @staticmethod
    def _workspace_policy_path_name_has_external_source_marker(path_token: str) -> bool:
        name = PurePosixPath(str(path_token or "")).name.lower()
        pieces = [piece for piece in re.split(r"[^a-z0-9]+", name) if piece]
        return any(piece in _WORKSPACE_POLICY_EXTERNAL_SOURCE_HELPER_MARKERS for piece in pieces)

    @staticmethod
    def _workspace_policy_path_is_transient_script_argument(
        candidate: str,
        *,
        previous_token: str,
    ) -> bool:
        text = str(candidate or "").strip()
        if not CLIModelClient._workspace_policy_path_is_transient_output(text):
            return False
        path = PurePosixPath(text)
        if path.suffix.lower() not in _WORKSPACE_POLICY_TRANSIENT_SCRIPT_SUFFIXES:
            return False
        if CLIModelClient._workspace_policy_path_name_has_external_source_marker(text):
            return False
        previous_command = CLIModelClient._workspace_policy_command_name(previous_token)
        return previous_command in _WORKSPACE_POLICY_INLINE_URL_PROBE_COMMANDS

    @staticmethod
    def _workspace_policy_external_url_literals(text: str) -> list[str]:
        urls: list[str] = []
        for match in re.finditer(r"(?i)\b(?:https?|ftp)://[^\s'\"<>]+", str(text or "")):
            url = match.group(0).rstrip(").,;]")
            parsed = urllib.parse.urlsplit(url)
            hostname = (parsed.hostname or "").lower()
            if not hostname or hostname in _WORKSPACE_POLICY_LOCAL_NETWORK_HOSTS:
                continue
            urls.append(url)
        return urls

    @staticmethod
    def _target_runtime_inline_code_arguments(
        tokens: list[str],
        command_index: int,
        command_name: str,
    ) -> list[str]:
        inline_args: list[str] = []
        index = command_index + 1
        while index < len(tokens):
            token = CLIModelClient._strip_shell_path_candidate(tokens[index])
            if not token or token in {"&&", "||", "|", ";"}:
                break
            if command_name.startswith("python"):
                if token == "-c" and index + 1 < len(tokens):
                    inline_args.append(str(tokens[index + 1]))
                    index += 2
                    continue
                if token.startswith("-c") and len(token) > 2:
                    inline_args.append(token[2:])
                    index += 1
                    continue
                if token == "-m":
                    break
            elif command_name in {"node", "nodejs", "deno"}:
                if token in {"-e", "--eval"} and index + 1 < len(tokens):
                    inline_args.append(str(tokens[index + 1]))
                    index += 2
                    continue
                if token.startswith("--eval="):
                    inline_args.append(token.split("=", 1)[1])
                    index += 1
                    continue
            elif command_name in {"perl", "ruby"}:
                if token in {"-e", "-E"} and index + 1 < len(tokens):
                    inline_args.append(str(tokens[index + 1]))
                    index += 2
                    continue
                if len(token) > 2 and token[:2] in {"-e", "-E"}:
                    inline_args.append(token[2:])
                    index += 1
                    continue
            elif command_name == "php":
                if token in {"-r", "-B", "-R"} and index + 1 < len(tokens):
                    inline_args.append(str(tokens[index + 1]))
                    index += 2
                    continue
            if token.startswith("-"):
                index += 1
                continue
            break
        return inline_args

    @classmethod
    def _target_runtime_inline_external_url_probe(
        cls,
        tokens: list[str],
        command_index: int,
        command_name: str,
    ) -> str:
        if command_name not in _WORKSPACE_POLICY_INLINE_URL_PROBE_COMMANDS:
            return ""
        for inline_code in cls._target_runtime_inline_code_arguments(
            tokens,
            command_index,
            command_name,
        ):
            lowered = inline_code.lower()
            if not any(marker in lowered for marker in _WORKSPACE_POLICY_INLINE_URL_PROBE_MARKERS):
                continue
            urls = cls._workspace_policy_external_url_literals(inline_code)
            if urls:
                return urls[0]
        return ""

    @staticmethod
    def _target_runtime_python_script_argument(tokens: list[str], python_index: int) -> str:
        index = python_index + 1
        options_with_value = {"-W", "-X", "--check-hash-based-pycs"}
        while index < len(tokens):
            token = CLIModelClient._strip_shell_path_candidate(tokens[index])
            if not token or token in {"&&", "||", "|", ";"}:
                return ""
            if token in {"-c", "-m"}:
                return ""
            if token.startswith("-"):
                if token in options_with_value and index + 1 < len(tokens):
                    index += 2
                else:
                    index += 1
                continue
            return token
        return ""

    def _target_runtime_source_provenance_policy_violation_from_tokens(
        self,
        tokens: list[str],
        *,
        working_dir: Path,
        process_cwd: Optional[Path],
    ) -> Optional[dict[str, Any]]:
        expect_command = True
        index = 0
        while index < len(tokens):
            raw_token = str(tokens[index] or "").strip()
            token = self._strip_shell_path_candidate(raw_token)
            token_ends_command = raw_token in {"&&", "||", "|", ";"} or raw_token.endswith(";")
            if not token:
                index += 1
                continue
            if token in {"&&", "||", "|", ";"}:
                expect_command = True
                index += 1
                continue
            if not expect_command:
                if token_ends_command:
                    expect_command = True
                index += 1
                continue
            if self._looks_like_shell_env_assignment(token):
                index += 1
                continue
            command_name = self._workspace_policy_command_name(token)
            if command_name in {"command", "exec"}:
                index += 1
                continue
            if command_name == "env":
                index += 1
                while index < len(tokens):
                    current = self._strip_shell_path_candidate(tokens[index])
                    if current == "--":
                        index += 1
                        break
                    if self._looks_like_shell_env_assignment(current):
                        index += 1
                        continue
                    if current.startswith("-"):
                        index += 1
                        if current == "-u" and index < len(tokens):
                            index += 1
                        continue
                    break
                continue
            if command_name in {"timeout", "gtimeout"}:
                index += 1
                while index < len(tokens):
                    current = self._strip_shell_path_candidate(tokens[index])
                    if current == "--":
                        index += 1
                        break
                    if current.startswith("-"):
                        index += 1
                        if current in {"-k", "--kill-after", "-s", "--signal"}:
                            index += 1
                        continue
                    if re.match(r"^\d+(?:\.\d+)?[smhd]?$", current):
                        index += 1
                        continue
                    break
                continue
            if command_name in {"curl", "wget"}:
                url_token = next(
                    (
                        self._strip_shell_path_candidate(candidate)
                        for candidate in tokens[index + 1 :]
                        if re.match(
                            r"^(?:https?|ftp)://",
                            self._strip_shell_path_candidate(candidate),
                            flags=re.IGNORECASE,
                        )
                    ),
                    "",
                )
                if url_token:
                    return {
                        "policy_kind": "external_source_acquisition",
                        "path_token": url_token,
                        "resolved_path": None,
                        "reason": (
                            "CLI subprocess attempted external source acquisition "
                            f"with `{command_name}` URL `{url_token}`. Use only the "
                            "current worktree, visible tests, and in-repo documentation/examples."
                        ),
                    }
            inline_url = self._target_runtime_inline_external_url_probe(
                tokens,
                index,
                command_name,
            )
            if inline_url:
                return {
                    "policy_kind": "external_source_acquisition",
                    "path_token": inline_url,
                    "resolved_path": None,
                    "reason": (
                        "CLI subprocess attempted external source acquisition "
                        f"or availability probing through `{command_name}` inline code "
                        f"URL `{inline_url}`. Use only the current worktree, visible "
                        "tests, and in-repo documentation/examples."
                    ),
                }
            if command_name.startswith("python"):
                script_token = self._target_runtime_python_script_argument(tokens, index)
                if (
                    script_token
                    and self._workspace_policy_path_name_has_external_source_marker(script_token)
                    and not self._workspace_policy_path_is_test_file(script_token)
                ):
                    resolved_path = self._workspace_policy_resolve_repo_local_path_token(
                        script_token,
                        working_dir=working_dir,
                        process_cwd=process_cwd,
                    )
                    if resolved_path is not None and not self._path_escapes_workspace(
                        resolved_path,
                        working_dir=working_dir,
                    ):
                        return {
                            "policy_kind": "external_source_acquisition",
                            "path_token": script_token,
                            "resolved_path": str(resolved_path),
                            "reason": (
                                "CLI subprocess attempted external source acquisition "
                                f"through repo-local helper `{script_token}`. Use only the "
                                "current worktree, visible tests, and in-repo "
                                "documentation/examples; do not run downloader/vendor "
                                "helpers as implementation sources."
                            ),
                        }
            expect_command = token_ends_command
            index += 1
        return None

    def _target_runtime_source_provenance_policy_violation(
        self,
        tokens: list[str],
        *,
        working_dir: Path,
        process_cwd: Optional[Path],
    ) -> Optional[dict[str, Any]]:
        if not tokens:
            return None
        command_name = self._workspace_policy_command_name(str(tokens[0] or ""))
        if command_name in {"bash", "sh", "zsh"}:
            for payload in self._target_runtime_shell_policy_payloads(tokens):
                violation = self._target_runtime_source_provenance_policy_violation_from_tokens(
                    self._tokenize_process_command(payload),
                    working_dir=working_dir,
                    process_cwd=process_cwd,
                )
                if violation is not None:
                    violation["shell_payload"] = payload
                    return violation
            return None
        return self._target_runtime_source_provenance_policy_violation_from_tokens(
            tokens,
            working_dir=working_dir,
            process_cwd=process_cwd,
        )

    def _git_history_subcommand(self, tokens: list[str]) -> str:
        for index, token in enumerate(tokens):
            if self._workspace_policy_command_name(token) != "git":
                continue
            cursor = index + 1
            while cursor < len(tokens):
                current = self._strip_shell_path_candidate(tokens[cursor])
                if not current or current in {"--", "&&", "||", "|", ";"}:
                    cursor += 1
                    continue
                if current in _WORKSPACE_POLICY_GIT_GLOBAL_OPTIONS_WITH_VALUE:
                    cursor += 2
                    continue
                if any(
                    current.startswith(f"{option}=")
                    for option in _WORKSPACE_POLICY_GIT_GLOBAL_OPTIONS_WITH_VALUE
                    if option.startswith("--")
                ):
                    cursor += 1
                    continue
                if current in _WORKSPACE_POLICY_GIT_GLOBAL_OPTIONS:
                    cursor += 1
                    continue
                if current.startswith("-"):
                    cursor += 1
                    continue
                subcommand = self._workspace_policy_command_name(current)
                if subcommand in _WORKSPACE_POLICY_FORBIDDEN_GIT_HISTORY_SUBCOMMANDS:
                    return subcommand
                break
        return ""

    def _target_runtime_shell_git_history_policy_violation(
        self,
        tokens: list[str],
    ) -> Optional[dict[str, str]]:
        for payload in self._target_runtime_shell_policy_payloads(tokens):
            subcommand = self._git_history_subcommand(self._tokenize_process_command(payload))
            if subcommand:
                payload_lower = payload.lower()
                bypass_markers = (
                    "git.apex-real",
                    "/usr/bin/git",
                    "/bin/git",
                    "/usr/local/bin/git",
                    "/opt/",
                )
                severity = (
                    "fatal"
                    if any(marker in payload_lower for marker in bypass_markers)
                    else "blocked_by_policy"
                )
                return {
                    "git_subcommand": subcommand,
                    "shell_payload": payload,
                    "severity": severity,
                }
        return None

    def _target_runtime_shell_is_apex_control_helper(self, tokens: list[str]) -> bool:
        if not tokens or self._workspace_policy_command_name(tokens[0]) not in {
            "sh",
            "bash",
            "zsh",
        }:
            return False
        command_string = self._target_runtime_shell_command_argument(tokens)
        if not command_string:
            return False
        token_names = {self._workspace_policy_command_name(str(token or "")) for token in tokens}
        if "apex-target-cleanup" in token_names:
            return (
                "target=$1" in command_string
                and "sig=$2" in command_string
                and "invocation=$3" in command_string
                and "children_file=" in command_string
                and "/proc/[0-9]*" in command_string
                and 'kill -"$sig"' in command_string
            )
        if "apex-target-activity" in token_names:
            return (
                "target=$1" in command_string
                and "invocation=$2" in command_string
                and "sampler_nonce=$3" in command_string
                and "records_file=" in command_string
                and "details_file=" in command_string
                and "/proc/[0-9]*" in command_string
            )
        return False

    def _target_runtime_dynamic_invocation_is_host_bypass(
        self,
        tokens: list[str],
    ) -> bool:
        """True when a dynamic command clearly bypasses the PATH shims."""

        if not tokens:
            return False
        executable = str(tokens[0] or "")
        if not os.path.isabs(executable):
            # Bare commands are exactly what the target-runtime shims are
            # installed to intercept. The previous policy killed legitimate
            # `git diff`, `grep`, and `pytest` children because ps reports the
            # argv name rather than the resolved shim path.
            return False
        if self._path_looks_like_target_runtime_executable(executable):
            return False
        if Path(executable).name in {"bash", "sh", "zsh"}:
            return not self._target_runtime_shell_wrapper_is_allowed(tokens)
        return Path(executable).name in _TARGET_RUNTIME_DYNAMIC_COMMANDS

    @staticmethod
    def _path_resolves_to_system_helper_target(path: Optional[Path]) -> bool:
        """True if `path` lives under an APEX-managed helper root."""
        if path is None:
            return False
        path_text = str(path)
        return CLIModelClient._path_text_looks_like_backend_runtime_helper(path_text) or any(
            marker in path_text
            if not marker.startswith("/")
            else path_text == marker.rstrip("/") or path_text.startswith(marker)
            for marker in _WORKSPACE_POLICY_SYSTEM_TARGET_PREFIXES
        )

    def _process_tree_workspace_policy_violation(
        self,
        process_entries: dict[int, dict[str, Any]],
        working_dir: str,
        *,
        target_runtime_enforced: bool = False,
        target_runtime_git_history_policy: str = "blocked",
        target_runtime_source_network_policy: str = "unspecified",
        target_runtime_filesystem_boundary_policy: str = "policy_enforced",
    ) -> Optional[dict[str, Any]]:
        workspace_root = Path(working_dir).resolve()
        git_history_structural = _target_runtime_git_history_is_structural(
            target_runtime_git_history_policy
        )
        source_network_structural = _target_runtime_source_network_is_structural(
            target_runtime_source_network_policy
        )
        filesystem_boundary_structural = _target_runtime_filesystem_boundary_is_structural(
            target_runtime_filesystem_boundary_policy
        )
        cross_payload_temp_outputs = (
            self._target_runtime_shell_temp_outputs(process_entries)
            if target_runtime_enforced
            else set()
        )
        for pid, entry in process_entries.items():
            command = str(entry.get("command") or "").strip()
            raw_argv = entry.get("argv")
            if isinstance(raw_argv, list) and raw_argv:
                argv_tokens = [str(token) for token in raw_argv if str(token)]
            else:
                argv_tokens = self._tokenize_process_command(command)
            tokens = self._strip_process_wrapper_tokens(argv_tokens)
            if not tokens:
                continue
            command_name = self._workspace_policy_command_name(tokens[0])
            if target_runtime_enforced and self._target_runtime_shell_is_apex_control_helper(
                tokens
            ):
                continue
            raw_cwd_present = "cwd" in entry
            raw_cwd = str(entry.get("cwd") or "").strip()
            if target_runtime_enforced and raw_cwd_present:
                process_cwd = (
                    Path(raw_cwd).resolve()
                    if raw_cwd and not self._sampled_cwd_text_looks_like_command(raw_cwd)
                    else None
                )
            else:
                process_cwd = Path(raw_cwd).resolve() if raw_cwd else self._process_cwd(pid)
            if (
                target_runtime_enforced
                and not filesystem_boundary_structural
                and command_name in _TARGET_RUNTIME_DYNAMIC_COMMANDS
                and self._target_runtime_dynamic_invocation_is_host_bypass(tokens)
            ):
                if self._path_resolves_to_system_helper_target(
                    process_cwd
                ) or self._process_cwd_looks_like_backend_runtime_helper(process_cwd):
                    continue
                domain_decision = classify_command_domain(tokens)
                if domain_decision.allowed_host_bypass:
                    continue
                token_text = " ".join(tokens[:3])
                backend_launcher = any(
                    any(
                        marker in str(token).lower()
                        for marker in ("codex", "claude", "gemini", "opencode", "metacode")
                    )
                    for token in tokens[1:4]
                )
                if (
                    "apex_target_tool.py" not in command
                    and "target_tool_shims" not in command
                    and "docker run" not in command
                    and "docker sandbox run" not in command
                    and "docker exec" not in command
                    and not backend_launcher
                    and not self._path_looks_like_target_runtime_executable(tokens[0])
                    and not self._target_runtime_shell_wrapper_is_allowed(tokens)
                    and not self._process_has_ancestor_command_marker(
                        process_entries,
                        pid,
                        ("apex_target_tool.py", "target_tool_shims"),
                    )
                ):
                    return {
                        "pid": pid,
                        "command_name": command_name,
                        "command": command,
                        "cwd": str(self._process_cwd(pid) or ""),
                        "command_domain": domain_decision.to_dict(),
                        "severity": "fatal",
                        "reason": (
                            "CLI subprocess attempted host dynamic execution while "
                            f"benchmark target runtime enforcement is active: `{token_text}`. "
                            "Use the PATH-provided target tool shim instead."
                        ),
                    }
            if target_runtime_enforced:
                git_subcommand = (
                    "" if git_history_structural else self._git_history_subcommand(tokens)
                )
                if git_subcommand:
                    return {
                        "pid": pid,
                        "command_name": command_name,
                        "command": command,
                        "cwd": str(process_cwd) if process_cwd is not None else None,
                        "git_subcommand": git_subcommand,
                        "likely_backend_helper": False,
                        "severity": "fatal",
                        "reason": (
                            "CLI subprocess attempted git history discovery inside "
                            f"the rollout workspace: `git {git_subcommand}`. Use the "
                            "current worktree, visible tests, and working-tree diff only."
                        ),
                    }
                source_provenance_violation = None
                if not source_network_structural:
                    source_provenance_violation = (
                        self._target_runtime_source_provenance_policy_violation(
                            tokens,
                            working_dir=workspace_root,
                            process_cwd=process_cwd,
                        )
                    )
                if source_provenance_violation is not None:
                    reason = str(source_provenance_violation.get("reason") or "").strip()
                    return {
                        "pid": pid,
                        "command_name": command_name,
                        "command": command,
                        "cwd": str(process_cwd) if process_cwd is not None else None,
                        "path_token": source_provenance_violation.get("path_token"),
                        "resolved_path": source_provenance_violation.get("resolved_path"),
                        "shell_payload": source_provenance_violation.get("shell_payload"),
                        "policy_kind": source_provenance_violation.get("policy_kind"),
                        "likely_backend_helper": False,
                        "severity": "fatal",
                        "reason": reason
                        or (
                            "CLI subprocess attempted external source acquisition. "
                            "Use only the current worktree, visible tests, and "
                            "in-repo documentation/examples."
                        ),
                    }
            if target_runtime_enforced and command_name in {"bash", "sh", "zsh"}:
                shell_violation = None
                if not filesystem_boundary_structural:
                    shell_violation = self._target_runtime_shell_workspace_policy_violation(
                        tokens,
                        working_dir=workspace_root,
                        process_cwd=process_cwd,
                        cross_payload_temp_outputs=cross_payload_temp_outputs,
                    )
                if shell_violation is not None:
                    token_text = str(shell_violation.get("path_token") or "")
                    return {
                        "pid": pid,
                        "command_name": command_name,
                        "command": command,
                        "cwd": str(process_cwd) if process_cwd is not None else None,
                        "path_token": token_text,
                        "resolved_path": shell_violation.get("resolved_path"),
                        "shell_payload": shell_violation.get("shell_payload"),
                        "likely_backend_helper": False,
                        "severity": "fatal",
                        "reason": (
                            "CLI subprocess attempted repository discovery outside the rollout workspace: "
                            f"`{command_name}` shell payload targeted `{token_text}` which resolves to "
                            f"`{shell_violation.get('resolved_path')}`. Keep repository discovery inside "
                            "the current workspace."
                        ),
                    }
                shell_git_violation = (
                    None
                    if git_history_structural
                    else self._target_runtime_shell_git_history_policy_violation(tokens)
                )
                if shell_git_violation is not None:
                    git_subcommand = str(shell_git_violation.get("git_subcommand") or "")
                    severity = str(shell_git_violation.get("severity") or "fatal")
                    if severity == "blocked_by_policy":
                        reason = (
                            "CLI subprocess attempted git history discovery inside "
                            f"the rollout workspace: shell payload ran `git {git_subcommand}`, "
                            "but target-runtime git policy blocks history output before it can "
                            "be used. Continue from the current worktree and visible tests."
                        )
                    else:
                        reason = (
                            "CLI subprocess attempted git history discovery inside "
                            f"the rollout workspace: shell payload ran `git {git_subcommand}`. "
                            "Use the current worktree, visible tests, and working-tree diff only."
                        )
                    return {
                        "pid": pid,
                        "command_name": command_name,
                        "command": command,
                        "cwd": str(process_cwd) if process_cwd is not None else None,
                        "git_subcommand": git_subcommand,
                        "shell_payload": shell_git_violation.get("shell_payload"),
                        "likely_backend_helper": False,
                        "severity": severity,
                        "reason": reason,
                    }
            if command_name not in _WORKSPACE_POLICY_MONITORED_COMMANDS:
                continue
            if target_runtime_enforced and filesystem_boundary_structural:
                continue

            path_tokens = self._command_path_operands(command_name, tokens)
            if self._path_escapes_workspace(process_cwd, working_dir=workspace_root):
                likely_backend_helper = self._looks_like_backend_helper_workspace_policy_violation(
                    process_cwd=process_cwd,
                    path_tokens=path_tokens,
                    working_dir=workspace_root,
                )
                return {
                    "pid": pid,
                    "command_name": command_name,
                    "command": command,
                    "cwd": str(process_cwd) if process_cwd is not None else None,
                    "likely_backend_helper": likely_backend_helper,
                    "severity": "backend_helper" if likely_backend_helper else "fatal",
                    "reason": (
                        (
                            "CLI backend helper executed repository discovery outside the rollout workspace: "
                            if likely_backend_helper
                            else "CLI subprocess attempted repository discovery outside the rollout workspace: "
                        )
                        + f"`{command_name}` is running from `{process_cwd}` instead of under "
                        f"`{workspace_root}`. Keep repository discovery inside the current workspace."
                    ),
                }

            for token in path_tokens:
                if (
                    target_runtime_enforced
                    and str(token or "") in cross_payload_temp_outputs
                    and self._workspace_policy_path_is_transient_output(str(token or ""))
                ):
                    continue
                resolved_path = self._resolve_monitored_path_token(
                    token,
                    working_dir=workspace_root,
                    process_cwd=process_cwd,
                )
                if not self._path_escapes_workspace(
                    resolved_path,
                    working_dir=workspace_root,
                ):
                    continue
                # Only APEX-managed helper paths are downgraded. Sibling
                # task tempdirs, arbitrary /tmp traversal, and broad system
                # paths remain fatal policy violations.
                target_is_system_helper = self._path_resolves_to_system_helper_target(resolved_path)
                likely_backend_helper = target_is_system_helper
                severity = "backend_helper" if likely_backend_helper else "fatal"
                reason_prefix = (
                    "CLI backend helper executed repository discovery outside the rollout workspace: "
                    if likely_backend_helper
                    else "CLI subprocess attempted repository discovery outside the rollout workspace: "
                )
                return {
                    "pid": pid,
                    "command_name": command_name,
                    "command": command,
                    "cwd": str(process_cwd) if process_cwd is not None else None,
                    "path_token": token,
                    "resolved_path": str(resolved_path) if resolved_path is not None else None,
                    "likely_backend_helper": likely_backend_helper,
                    "severity": severity,
                    "reason": (
                        reason_prefix
                        + f"`{command_name}` targeted `{token}` which resolves to `{resolved_path}`. "
                        "Keep repository discovery inside the current workspace."
                    ),
                }
        return None

    def _target_runtime_completion_policy_audit(
        self,
        env: Optional[dict[str, str]],
        *,
        working_dir: str,
    ) -> Optional[dict[str, Any]]:
        activity = _sample_target_runtime_process_activity(env)
        if not activity:
            return None
        target_runtime_git_history_policy = _target_runtime_policy_value(
            activity,
            env,
            "git_history_policy",
            "blocked",
        )
        target_runtime_source_network_policy = _target_runtime_policy_value(
            activity,
            env,
            "source_network_policy",
            "unspecified",
        )
        target_runtime_filesystem_boundary_policy = _target_runtime_policy_value(
            activity,
            env,
            "filesystem_boundary_policy",
            "policy_enforced",
        )
        target_runtime_git_history_structural = _target_runtime_git_history_is_structural(
            target_runtime_git_history_policy
        )
        target_runtime_source_network_structural = _target_runtime_source_network_is_structural(
            target_runtime_source_network_policy
        )
        target_runtime_filesystem_boundary_structural = (
            _target_runtime_filesystem_boundary_is_structural(
                target_runtime_filesystem_boundary_policy
            )
        )
        runtime_policy_violations = activity.get("policy_violations")
        if isinstance(runtime_policy_violations, list) and runtime_policy_violations:
            runtime_policy_violations = [
                violation
                for violation in runtime_policy_violations
                if not _target_runtime_policy_marker_is_structurally_redundant(
                    violation,
                    git_history_structural=target_runtime_git_history_structural,
                    source_network_structural=target_runtime_source_network_structural,
                    filesystem_boundary_structural=(target_runtime_filesystem_boundary_structural),
                )
            ]
        if isinstance(runtime_policy_violations, list) and runtime_policy_violations:
            first_runtime_violation = (
                runtime_policy_violations[0]
                if isinstance(runtime_policy_violations[0], dict)
                else {"reason": str(runtime_policy_violations[0])}
            )
            violation_reason = str(
                first_runtime_violation.get("reason")
                or "Target-runtime command violated workspace policy."
            )
            target_violation = {
                "severity": "fatal",
                "reason": violation_reason,
                "target_runtime_policy_marker": dict(first_runtime_violation),
            }
            return _target_runtime_completion_policy_audit_base(
                working_dir=working_dir,
                target_runtime_activity=activity,
                policy_violation=target_violation,
            )
        target_process_entries = activity.get("process_entries")
        if not isinstance(target_process_entries, dict) or not target_process_entries:
            return None
        target_workdir = str(
            activity.get("workdir") or (env or {}).get("APEX_TARGET_TOOL_WORKDIR") or working_dir
        )
        target_violation = self._process_tree_workspace_policy_violation(
            target_process_entries,
            target_workdir,
            target_runtime_enforced=True,
            target_runtime_git_history_policy=target_runtime_git_history_policy,
            target_runtime_source_network_policy=target_runtime_source_network_policy,
            target_runtime_filesystem_boundary_policy=(target_runtime_filesystem_boundary_policy),
        )
        if target_violation is None:
            return None
        if str(target_violation.get("severity") or "fatal") in {
            "backend_helper",
            "blocked_by_policy",
        }:
            return None
        return _target_runtime_completion_policy_audit_base(
            working_dir=working_dir,
            target_runtime_activity=activity,
            policy_violation=dict(target_violation),
        )

    def _latest_worktree_mtime(self, worktree: Path) -> int:
        latest = 0
        stack = [worktree]
        skip_names = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}

        while stack:
            current = stack.pop()
            try:
                with os.scandir(current) as entries:
                    for entry in entries:
                        if entry.name in skip_names:
                            continue
                        try:
                            stat = entry.stat(follow_symlinks=False)
                        except FileNotFoundError:
                            continue
                        latest = max(latest, int(getattr(stat, "st_mtime_ns", 0) or 0))
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
            except (FileNotFoundError, NotADirectoryError, PermissionError):
                continue

        return latest

    def _process_tree_cpu_seconds(self, root_pid: int) -> float:
        return sum(
            float(entry.get("cpu_seconds", 0.0) or 0.0)
            for entry in self._collect_process_tree_entries(root_pid).values()
        )

    def _parse_ps_time_to_seconds(self, value: str) -> float:
        text = value.strip()
        if not text:
            return 0.0

        days = 0
        if "-" in text:
            day_text, text = text.split("-", 1)
            try:
                days = int(day_text)
            except ValueError:
                days = 0

        fields = text.split(":")
        try:
            if len(fields) == 3:
                hours = int(fields[0])
                minutes = int(fields[1])
                seconds = float(fields[2])
            elif len(fields) == 2:
                hours = 0
                minutes = int(fields[0])
                seconds = float(fields[1])
            else:
                hours = 0
                minutes = 0
                seconds = float(fields[0])
        except ValueError:
            return 0.0

        return (days * 86400) + (hours * 3600) + (minutes * 60) + seconds

    def _build_command(
        self,
        prompt: str,
        working_dir: str,
        schema: Optional[dict[str, Any]],
        system_prompt: Optional[str],
        allow_edits: bool,
        internet_enabled: bool = False,
        target_runtime_enforced: bool = False,
        sandbox_writable_roots: Optional[Iterable[str]] = None,
        codex_base_url: str = "",
        cli_hook_args: Optional[list[str]] = None,
        codex_output_in_workspace: bool = False,
        host_cli_read_jail: bool = False,
        claude_force_json_output: bool = False,
        claude_resume_session_id: str = "",
        codex_resume_thread_id: str = "",
    ) -> tuple[list[str], list[str]]:
        temp_files: list[str] = []
        if _env_flag_enabled("APEX_FORCE_CLI_INTERNET"):
            internet_enabled = True
        command = [self.config.resolved_cli_command]
        command.extend(self._internet_launcher_args(internet_enabled))

        backend = self.config.backend
        if backend == LLMBackend.CLAUDE_CLI:
            host_cli_mode = host_cli_read_jail or _host_cli_auth_mode_requested(
                self.config.cli_env_overrides
            )
            target_runtime_bare = target_runtime_enforced and _env_flag_not_disabled(
                "APEX_TARGET_RUNTIME_CLAUDE_BARE"
            )
            host_cli_bare = host_cli_mode and _env_flag_not_disabled("APEX_HOST_CLI_CLAUDE_BARE")
            if host_cli_bare or target_runtime_bare:
                # --bare runs the agent claude WITHOUT auto-loading the operator's
                # plugins/hooks/LSP/preset-install. host_cli keeps the real
                # ~/.claude (needed for dotslash native binaries), and target
                # runtimes carry benchmark container state; both otherwise pay
                # startup/finalization cost for operator-facing surfaces that are
                # irrelevant to an isolated noninteractive agent. --settings,
                # --agents, and --plugin-dir are still honored when APEX provides
                # them. Opt out with APEX_HOST_CLI_CLAUDE_BARE=0 or
                # APEX_TARGET_RUNTIME_CLAUDE_BARE=0.
                command.append("--bare")
            if host_cli_read_jail or (
                self.config.cli_disable_osx_sandbox and not target_runtime_enforced
            ):
                # host_cli_read_jail wraps the launch in an OUTER sandbox-exec
                # jail, so claude's own Seatbelt must be off (nesting it under our
                # profile would double-apply); the outer jail enforces reads.
                command.append("--dangerously-disable-osx-sandbox")
            target_runtime_json_output = (
                target_runtime_enforced
                and _target_runtime_claude_json_output_enabled(force_json=claude_force_json_output)
            )
            if target_runtime_json_output:
                # Target-runtime Claude stream-json can exit during bootstrap
                # before agent work starts; JSON-first avoids burning rollout
                # attempts while target process/activity liveness still bounds
                # truly silent runs. Host Claude keeps stream-json by default.
                command.extend(["-p", "--output-format", "json"])
            else:
                # Keep turn-by-turn visibility for the progress watchdog and for
                # recovery when Claude's wrapper exits after tool activity but
                # before emitting the terminal result event. --verbose is required
                # by Claude Code for stream-json under -p.
                command.extend(["-p", "--verbose", "--output-format", "stream-json"])
            if claude_resume_session_id:
                command.extend(["--resume", claude_resume_session_id])
            if (
                target_runtime_enforced
                and not claude_resume_session_id
                and _env_flag_enabled("APEX_TARGET_RUNTIME_CLAUDE_NO_SESSION_PERSISTENCE")
            ):
                # Persist target-runtime Claude sessions by default so an
                # interrupted provider stream can resume instead of replaying a
                # long agent turn. Operators can opt back into stateless launches
                # when retry-resume evidence is not needed.
                command.append("--no-session-persistence")
            if target_runtime_enforced and _env_flag_not_disabled(
                "APEX_TARGET_RUNTIME_CLAUDE_DISABLE_SLASH_COMMANDS"
            ):
                # Target-runtime agents receive explicit APEX prompts; loading
                # interactive slash-command surfaces adds startup work with no
                # benchmark authority.
                command.append("--disable-slash-commands")
            if target_runtime_enforced and _env_flag_not_disabled(
                "APEX_TARGET_RUNTIME_CLAUDE_STRICT_MCP_CONFIG"
            ):
                # The target runtime's tool boundary is APEX's shim layer, not
                # operator MCP servers. Empty strict MCP prevents accidental
                # external tools while keeping Claude's built-in tools available.
                command.extend(
                    [
                        "--strict-mcp-config",
                        "--mcp-config",
                        _CLAUDE_TARGET_RUNTIME_EMPTY_MCP_CONFIG_JSON,
                    ]
                )
            feed_prompt_on_stdin = self._should_feed_claude_prompt_on_stdin(
                prompt=prompt,
                working_dir=working_dir,
                target_runtime_enforced=target_runtime_enforced,
            )
            if system_prompt:
                if feed_prompt_on_stdin and _env_flag_not_disabled(
                    "APEX_TARGET_RUNTIME_CLAUDE_SYSTEM_PROMPT_FILE"
                ):
                    system_prompt_file = self._write_claude_prompt_file(
                        system_prompt,
                        working_dir=working_dir,
                        temp_files=temp_files,
                        prefix="claude-system-prompt-",
                    )
                    command.extend(["--system-prompt-file", system_prompt_file])
                else:
                    command.extend(["--system-prompt", system_prompt])
            if allow_edits:
                command.extend(
                    ["--permission-mode", self.config.cli_permission_mode or "bypassPermissions"]
                )
            else:
                # Read-only APEX stages still need noninteractive shell access for
                # repo-local inspection and targeted validation. Claude's plan
                # mode denies Bash in headless runs, so APEX enforces read-only
                # semantics by restoring workspace writes after the stage instead
                # of asking Claude Code to prompt for permission.
                command.extend(
                    ["--permission-mode", self.config.cli_permission_mode or "bypassPermissions"]
                )
            resolved_model = self.config.resolved_cli_model
            if resolved_model:
                command.extend(["--model", resolved_model])
            command.extend(_claude_cli_effort_args(self.config))
            if schema:
                command.extend(["--json-schema", json.dumps(schema)])
            command.extend(cli_hook_args or [])
            command.extend(self.config.cli_args)
            if feed_prompt_on_stdin:
                prompt_file = self._write_claude_prompt_file(
                    prompt,
                    working_dir=working_dir,
                    temp_files=temp_files,
                    prefix="claude-prompt-",
                )
                return (
                    [
                        "/bin/sh",
                        "-c",
                        'prompt_file="$1"; shift; exec "$@" < "$prompt_file"',
                        "apex-claude-stdin",
                        prompt_file,
                        *command,
                    ],
                    temp_files,
                )
            command.append(prompt)
            return command, temp_files

        if backend == LLMBackend.GEMINI_CLI:
            command.extend(["-p", prompt, "--output-format", "json"])
            permission_mode = self.config.cli_permission_mode or ("yolo" if allow_edits else None)
            if permission_mode:
                command.append(f"--approval-mode={permission_mode}")
            resolved_model = self.config.resolved_cli_model
            if resolved_model:
                command.extend(["-m", resolved_model])
            command.extend(cli_hook_args or [])
            command.extend(self.config.cli_args)
            return command, temp_files

        if backend == LLMBackend.CODEX_CLI:
            codex_resume_thread_id = str(codex_resume_thread_id or "").strip()
            if self.config.cli_permission_mode:
                command.extend(["-a", self.config.cli_permission_mode])
            codex_exec_config_args: list[str] = []
            codex_exec_extra_args: list[str] = []

            def add_codex_exec_config(*args: str) -> None:
                if target_runtime_enforced:
                    codex_exec_config_args.extend(args)
                else:
                    command.extend(args)

            add_codex_exec_config(*_codex_provider_config_args(codex_base_url))
            # Root-cause fix for the codex tool-router stall/error class: disable
            # the unified/streamable exec tool. It keeps a PERSISTENT shell
            # session and writes to its child's stdin; under Apex's headless,
            # stdin-closed (non-TTY) launch (`exec "$@" < /dev/null`) that session
            # dies mid-run with "write_stdin failed: stdin is closed for this
            # session; rerun exec_command with tty=true", which either fails the
            # rollout or leaves codex HANGING until the stall watchdog kills it
            # (~50 min wasted). The classic one-shot `shell_tool` (still enabled)
            # runs each command to completion with no persistent stdin and cannot
            # hit this. Equivalent to `--disable unified_exec`. General agent-infra
            # fix (every codex rollout, any task) — not benchmark/repo-specific.
            add_codex_exec_config("-c", "features.unified_exec=false")
            tool_output_limit = _codex_tool_output_token_limit()
            if tool_output_limit > 0:
                add_codex_exec_config(
                    "-c",
                    f"tool_output_token_limit={tool_output_limit}",
                )
            if not internet_enabled and not target_runtime_enforced:
                # Codex at Meta exposes external-search MCP tools by default.
                # Disable them explicitly so air-gapped runs do not inherit
                # ambient web access from the host CLI environment.
                # Target-runtime Codex uses --ignore-user-config plus an isolated
                # config with no MCP servers; adding a partial mcp_servers table
                # there makes public in-container Codex reject the config.
                add_codex_exec_config("-c", "mcp_servers.meta_core.enabled=false")
            if target_runtime_enforced:
                # Target-runtime agents must not inherit user/plugin/rule context:
                # benchmark evidence must come from the mounted task workspace and
                # APEX's declared tool surface, not ambient Codex marketplaces.
                codex_exec_config_args.extend(
                    [
                        "-c",
                        "features.plugins=false",
                        "-c",
                        "features.plugin_hooks=false",
                        "-c",
                        "features.remote_plugin=false",
                        "-c",
                        "features.skill_mcp_dependency_install=false",
                        "-c",
                        "skills.bundled.enabled=false",
                    ]
                )
            # 10.C: silence the per-tmp trust prompt. Each rollout runs in a
            # fresh sandbox dir codex has never seen, so it would otherwise
            # block waiting for trust input. The TOML key uses a quoted
            # string for the path; shlex.quote keeps oddly-named dirs safe.
            add_codex_exec_config(
                "-c",
                f"projects.{shlex.quote(working_dir)}.trust_level=trusted",
            )
            if target_runtime_enforced:
                # Codex honors --ignore-user-config at the exec subcommand layer;
                # keep target-runtime -c overrides there so the Meta launcher does
                # not parse operator config before the isolation flag is active.
                codex_exec_extra_args.extend(cli_hook_args or [])
            else:
                command.extend(cli_hook_args or [])
            output_file = _codex_output_temp_file(
                target_runtime_enforced=target_runtime_enforced,
                working_dir=working_dir,
                in_workspace=codex_output_in_workspace,
            )
            output_file.close()
            temp_files.append(output_file.name)
            if host_cli_read_jail and not target_runtime_enforced:
                # host_cli_read_jail wraps the launch in an OUTER sandbox-exec
                # jail. The Meta codex launcher self-applies Seatbelt (which makes
                # a nested sandbox-exec fail with 'sandbox_apply: Operation not
                # permitted'), so we disable BOTH codex's launcher osx sandbox and
                # its exec sandbox; the outer jail is the single read boundary.
                command.append("--dangerously-disable-osx-sandbox")
                command.append("exec")
                if codex_resume_thread_id:
                    command.append("resume")
                if not codex_resume_thread_id:
                    command.extend(["--cd", working_dir])
                command.append("--skip-git-repo-check")
                command.append("--dangerously-bypass-approvals-and-sandbox")
            elif _env_flag_enabled("APEX_CODEX_BYPASS_SANDBOX") and not target_runtime_enforced:
                command.append("exec")
                if codex_resume_thread_id:
                    command.append("resume")
                if not codex_resume_thread_id:
                    command.extend(["--cd", working_dir])
                command.append("--skip-git-repo-check")
                command.append("--dangerously-bypass-approvals-and-sandbox")
            else:
                sandbox_override = os.environ.get("APEX_CODEX_SANDBOX_MODE")
                # Docker target runtimes are the filesystem boundary for agent
                # rollouts. Keeping Codex's nested Linux sandbox enabled there
                # can require unprivileged user namespaces and fail before the
                # agent can even read /workspace.
                default_sandbox_mode = (
                    "danger-full-access"
                    if target_runtime_enforced
                    else "workspace-write"
                    if allow_edits
                    else "read-only"
                )
                sandbox_mode = (
                    sandbox_override.strip()
                    if sandbox_override and sandbox_override.strip()
                    else default_sandbox_mode
                )
                command.append("exec")
                if codex_resume_thread_id:
                    command.append("resume")
                if target_runtime_enforced:
                    command.extend(["--ignore-user-config", "--ignore-rules"])
                    command.extend(codex_exec_config_args)
                    command.extend(codex_exec_extra_args)
                if sandbox_mode == "workspace-write":
                    writable_roots = _normalize_sandbox_roots(
                        [
                            *(sandbox_writable_roots or ()),
                            str(Path(output_file.name).parent),
                        ],
                        working_dir=working_dir,
                    )
                    if writable_roots:
                        command.extend(
                            [
                                "-c",
                                "sandbox_workspace_write.writable_roots="
                                + json.dumps(writable_roots),
                            ]
                        )
                    if target_runtime_enforced:
                        command.extend(
                            [
                                "-c",
                                "sandbox_workspace_write.exclude_tmpdir_env_var=true",
                                "-c",
                                "sandbox_workspace_write.exclude_slash_tmp=true",
                            ]
                        )
                if codex_resume_thread_id:
                    command.append("--skip-git-repo-check")
                    if target_runtime_enforced:
                        command.append("--dangerously-bypass-approvals-and-sandbox")
                else:
                    command.extend(["--cd", working_dir, "--skip-git-repo-check"])
                    command.extend(["--sandbox", sandbox_mode])
            command.append("--json")
            if not codex_resume_thread_id and (
                not target_runtime_enforced
                or _env_flag_enabled("APEX_TARGET_RUNTIME_CODEX_EPHEMERAL")
            ):
                command.append("--ephemeral")
            resolved_model = self.config.resolved_cli_model
            if resolved_model:
                command.extend(["--model", resolved_model])
            command.extend(["--output-last-message", output_file.name])
            command.extend(self.config.cli_args)
            if system_prompt:
                prompt = f"{system_prompt}\n\n{prompt}"
            if codex_resume_thread_id:
                command.append(codex_resume_thread_id)
            command.append(prompt)
            return command, temp_files

        if backend in _OPENCODE_FAMILY_BACKENDS:
            command.extend(["run", "--format", "json"])
            if allow_edits:
                command.append("--yolo")
            resolved_model = self.config.resolved_cli_model
            if resolved_model:
                command.extend(["--model", resolved_model])
            command.extend(self.config.cli_args)
            command.append(prompt)
            return command, temp_files

        raise ValueError(f"Unsupported CLI backend: {backend}")

    def _internet_launcher_args(self, internet_enabled: bool) -> list[str]:
        if not internet_enabled:
            return []
        if self.config.backend == LLMBackend.CODEX_CLI:
            # Meta's Codex launcher needs both the outer internet-access switch
            # and Codex's native live-search switch for online tool use.
            return ["--dangerously-enable-internet-mode", "--search"]
        if self.config.backend == LLMBackend.CLAUDE_CLI:
            # Claude Code's Meta launcher accepts --internet for online runs.
            return ["--internet"]
        if self.config.backend in {
            LLMBackend.GEMINI_CLI,
            LLMBackend.OPENCODE_CLI,
            LLMBackend.METACODE_CLI,
        }:
            # Use the cross-platform Meta launcher switch so online runs behave
            # the same on macOS and Linux across all wrapped CLIs.
            return ["--dangerously-enable-internet-mode"]
        return []

    def _parse_result(
        self,
        returncode: int,
        stdout: str,
        stderr: str,
        temp_files: list[str],
    ) -> CLIModelResult:
        if self.config.backend == LLMBackend.CLAUDE_CLI:
            return self._parse_claude_result(returncode, stdout, stderr)
        if self.config.backend == LLMBackend.GEMINI_CLI:
            return self._parse_gemini_result(returncode, stdout, stderr)
        if self.config.backend == LLMBackend.CODEX_CLI:
            return self._parse_codex_result(returncode, stdout, stderr, temp_files)
        if self.config.backend in _OPENCODE_FAMILY_BACKENDS:
            return self._parse_opencode_result(returncode, stdout, stderr)
        return CLIModelResult(success=False, error="Unknown CLI backend")

    def _should_feed_claude_prompt_on_stdin(
        self,
        *,
        prompt: str,
        working_dir: str,
        target_runtime_enforced: bool,
    ) -> bool:
        if self.config.backend != LLMBackend.CLAUDE_CLI:
            return False
        if not target_runtime_enforced or not prompt:
            return False
        if not _env_flag_not_disabled("APEX_TARGET_RUNTIME_CLAUDE_STDIN_PROMPT"):
            return False
        try:
            workspace = Path(working_dir).expanduser().resolve(strict=False)
        except OSError:
            workspace = Path(working_dir).expanduser().absolute()
        # APEX maps the prompt file into docker_exec containers by replacing the
        # declared host mount root. Avoid wrapping synthetic root-level paths
        # used by unit tests where no such mount exists.
        return workspace.parent != workspace.parent.parent

    def _write_claude_prompt_file(
        self,
        content: str,
        *,
        working_dir: str,
        temp_files: list[str],
        prefix: str,
    ) -> str:
        prompt_dir = _agent_runtime_state_root_for_workspace(working_dir) / ".cli_prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=prefix,
            suffix=".txt",
            dir=prompt_dir,
            delete=False,
        ) as handle:
            handle.write(content)
            prompt_path = handle.name
        try:
            Path(prompt_path).chmod(0o600)
        except OSError:
            pass
        temp_files.append(prompt_path)
        return prompt_path

    def _finalize_result_channels(
        self,
        result: CLIModelResult,
        *,
        returncode: Optional[int],
        stdout: str,
        stderr: str,
        allow_edits: bool = False,
        finalization_status: Optional[str] = None,
    ) -> CLIModelResult:
        """Populate independent backend health channels.

        A CLI can produce a usable response while its post-response telemetry or
        transport finalization fails.  Those conditions should be visible in
        diagnostics without being collapsed into "no candidate patch exists."
        """

        text = str(result.text or "").strip()
        has_response = bool(text) or result.parsed_json is not None
        haystack = "\n".join(
            [
                str(stdout or ""),
                str(stderr or ""),
                text,
                str(result.error or ""),
            ]
        )
        result.response_status = "ok" if has_response else "failed"
        result.workspace_status = "not_checked" if allow_edits else "not_applicable"
        result.patch_extraction_status = "available" if has_response else "missing"
        result.finalization_status = finalization_status or (
            "ok" if (returncode is None or int(returncode) == 0) else "failed"
        )
        if self._looks_like_cli_telemetry_failure(haystack):
            result.telemetry_status = "failed"
        else:
            result.telemetry_status = "ok" if has_response else "unknown"

        # CLI noise is NEVER, by itself, a failure. A result that carries a usable
        # structured agent response (``parsed_json``) has SUCCEEDED regardless of
        # any stderr chatter, wrapper promo/preset install lines (e.g. Meta host
        # CLI "Installing <plugin> for preset ...", "Help us improve <tool> ..."),
        # telemetry-export errors, or a non-zero WRAPPER returncode — none of
        # those are the agent's verdict. Only the ABSENCE of a usable result is
        # fatal. This generalizes the prior telemetry / stream-disconnect
        # carve-outs (kept for the text-only-response case) to arbitrary noise.
        # Quality is still gated downstream by quick-verification / selection /
        # the official audit, so promoting a noisy-but-present result to success
        # cannot cause a false accept — it only stops noise from discarding real
        # candidate work.
        nonfatal_backend_failure = not result.success and (
            result.parsed_json is not None
            or (
                has_response
                and (
                    self._looks_like_cli_telemetry_failure(haystack)
                    or (
                        self._looks_like_cli_finalization_only_failure(haystack)
                        and not self._looks_like_cli_incomplete_stream_failure(haystack)
                    )
                )
            )
        )
        if nonfatal_backend_failure:
            if result.error:
                result.backend_diagnostics["non_fatal_backend_error"] = result.error
            result.success = True
            result.error = None
        return result

    def _looks_like_cli_telemetry_failure(self, text: str) -> bool:
        lowered = str(text or "").lower()
        signatures = (
            "periodicexportingmetricreader",
            "telemetry export",
            "failed to export telemetry",
            "opentelemetry",
            "metricreader",
        )
        return any(signature in lowered for signature in signatures)

    def _looks_like_cli_finalization_only_failure(self, text: str) -> bool:
        lowered = str(text or "").lower()
        signatures = (
            "stream ended without terminal result after agent activity",
            "connection closed after final response",
            "post-response finalization",
            "finalization failed",
        )
        return any(signature in lowered for signature in signatures)

    def _looks_like_cli_incomplete_stream_failure(self, text: str) -> bool:
        lowered = str(text or "").lower()
        compact = re.sub(r"\s+", "", lowered)
        signatures = (
            "stream disconnected before completion",
            "turn.failed",
            '"type":"turn.failed"',
        )
        return any(signature in lowered or signature in compact for signature in signatures)

    def _parse_claude_result(self, returncode: int, stdout: str, stderr: str) -> CLIModelResult:
        payload = self._load_claude_result_payload(stdout)
        actionable_stderr = self._extract_actionable_claude_stderr(stderr)
        if payload is not None:
            if not self._looks_like_claude_result_payload(payload):
                success = returncode == 0 or not actionable_stderr
                return CLIModelResult(
                    success=success,
                    text=json.dumps(payload),
                    parsed_json=payload,
                    usage={},
                    error=None if success else (actionable_stderr or stderr.strip()),
                )
            structured_output = payload.get("structured_output")
            text = self._coerce_text(payload.get("result", ""))
            parsed_json = None
            if isinstance(structured_output, dict):
                parsed_json = structured_output
                if not text:
                    text = json.dumps(structured_output)
            else:
                parsed_json = self._decode_json_if_possible(text)
            payload_is_error = bool(payload.get("is_error", False))
            payload_type = str(payload.get("type") or "").strip().lower()
            has_response = (
                isinstance(structured_output, dict) or bool(text) or parsed_json is not None
            )
            success = (returncode == 0 and not payload_is_error) or (
                not payload_is_error
                and not actionable_stderr
                and has_response
                and payload_type in ("", "result")
            )
            return CLIModelResult(
                success=success,
                text=text,
                parsed_json=parsed_json,
                usage=payload.get("usage", {}),
                error=None
                if success
                else (actionable_stderr or stderr.strip() or payload.get("result")),
            )
        # No structured payload. STRIP host-CLI wrapper noise (startup banners
        # like "Claude Code at Meta", "--dangerously-disable-osx-sandbox",
        # preset/version chatter) from stdout before using it as the response
        # text. Otherwise a session that exited producing ONLY wrapper noise and
        # NO result event (e.g. SEV S671975: claude certs issued with the wrong
        # agent.id -> loss of downstream access -> rc!=0, banner-only stdout)
        # would surface the harmless banner as the rollout's failure_reason via
        # the engine's `error or text` fallback, AND would not be recognized as a
        # retryable no-result exit. Reuse the stderr noise filter (stream-agnostic).
        stream_activity_summary = self._extract_claude_stream_activity_summary(stdout)
        meaningful_stdout = (
            ""
            if self._claude_output_is_only_nonterminal_stream_or_noise(stdout)
            else (stream_activity_summary or self._extract_actionable_claude_stderr(stdout))
        )
        return CLIModelResult(
            success=returncode == 0,
            text=meaningful_stdout,
            parsed_json=self._decode_json_if_possible(meaningful_stdout),
            error=None if returncode == 0 else (actionable_stderr or None),
        )

    def _looks_like_claude_result_payload(self, payload: dict[str, Any]) -> bool:
        payload_type = str(payload.get("type") or "").strip().lower()
        if payload_type in {"result", "error"}:
            return True
        return any(
            key in payload
            for key in ("structured_output", "result", "usage", "is_error", "subtype")
        )

    def _load_claude_result_payload(self, text: str) -> Optional[dict[str, Any]]:
        """Return Claude's terminal result event, not nested stream fragments."""

        candidates: list[dict[str, Any]] = []
        whole_payload: Optional[dict[str, Any]] = None
        stripped = text.strip()
        if stripped:
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                whole_payload = payload
                candidates.append(payload)

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                candidates.append(payload)

        terminal_events = [
            candidate
            for candidate in candidates
            if str(candidate.get("type") or "").strip().lower() in {"result", "error"}
            or "structured_output" in candidate
            or "result" in candidate
            or "is_error" in candidate
        ]
        if terminal_events:
            return terminal_events[-1]
        if whole_payload is not None:
            return whole_payload
        return None

    def _extract_claude_session_id(self, text: str) -> str:
        """Return a resumable Claude session id from JSON or JSONL output."""

        candidates: list[str] = []
        stripped = str(text or "").strip()
        if stripped:
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                candidates.append(str(payload.get("session_id") or "").strip())
        for raw_line in str(text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                candidates.append(str(payload.get("session_id") or "").strip())
        for session_id in reversed(candidates):
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{7,}", session_id):
                return session_id
        return ""

    def _extract_codex_thread_id(self, text: str) -> str:
        """Return a resumable Codex thread id from JSONL output."""

        candidates: list[str] = []
        for raw_line in str(text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                candidates.append(str(payload.get("thread_id") or "").strip())
                candidates.append(str(payload.get("session_id") or "").strip())
        for thread_id in reversed(candidates):
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{7,}", thread_id):
                return thread_id
        return ""

    def _parse_gemini_result(self, returncode: int, stdout: str, stderr: str) -> CLIModelResult:
        payload = self._load_json_payload(
            stdout,
            preferred_keys=("response", "stats", "session_id"),
        )
        actionable_stderr = self._extract_actionable_gemini_stderr(stderr)
        if payload is not None:
            response = payload.get("response", "")
            text = self._coerce_text(response)
            parsed_json = (
                response if isinstance(response, dict) else self._decode_json_if_possible(text)
            )
            if isinstance(parsed_json, dict) and not text:
                text = json.dumps(parsed_json)
            success = returncode == 0 or (isinstance(parsed_json, dict) and not actionable_stderr)
            return CLIModelResult(
                success=success,
                text=text,
                parsed_json=parsed_json,
                usage=payload.get("stats", {}),
                error=None if success else (actionable_stderr or stderr.strip()),
            )
        return CLIModelResult(
            success=returncode == 0,
            text=stdout.strip(),
            parsed_json=self._decode_json_if_possible(stdout.strip()),
            error=None if returncode == 0 else (actionable_stderr or stderr.strip()),
        )

    def _parse_codex_result(
        self,
        returncode: int,
        stdout: str,
        stderr: str,
        temp_files: list[str],
    ) -> CLIModelResult:
        text = ""
        usage: dict[str, Any] = {}
        errors: list[str] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    text = item.get("text", text)
            if event.get("type") == "error":
                message = self._coerce_text(event.get("message", ""))
                if message:
                    errors.append(message)
            if event.get("type") == "turn.failed":
                message = self._coerce_text(event.get("error", ""))
                if message:
                    errors.append(message)
            if event.get("type") == "turn.completed":
                usage = event.get("usage", usage)

        if temp_files:
            output_file = Path(temp_files[-1])
            if output_file.exists():
                file_text = output_file.read_text().strip()
                if file_text:
                    text = file_text

        parsed_json = self._decode_json_if_possible(text)
        actionable_stderr = self._extract_actionable_codex_stderr(stderr)
        success = not errors and (
            returncode == 0 or (isinstance(parsed_json, dict) and not actionable_stderr)
        )
        return CLIModelResult(
            success=success,
            text=text,
            parsed_json=parsed_json,
            usage=usage,
            error=None if success else (" | ".join(errors) or actionable_stderr or stderr.strip()),
        )

    def _parse_opencode_result(self, returncode: int, stdout: str, stderr: str) -> CLIModelResult:
        text = ""
        usage: dict[str, Any] = {}
        errors: list[str] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_type = event.get("type")
            part = event.get("part", {})
            if event_type == "text" and isinstance(part, dict):
                candidate = self._coerce_text(part.get("text", ""))
                if candidate:
                    text = candidate
            if event_type == "step_finish" and isinstance(part, dict):
                tokens = part.get("tokens")
                if isinstance(tokens, dict):
                    usage = {"tokens": tokens}
            if event_type == "error":
                message = self._coerce_text(event.get("message", ""))
                if not message:
                    error_payload = event.get("error")
                    if isinstance(error_payload, dict):
                        message = self._coerce_text(error_payload.get("message", ""))
                        if not message:
                            data = error_payload.get("data")
                            if isinstance(data, dict):
                                message = self._coerce_text(data.get("message", ""))
                        if not message:
                            message = self._coerce_text(error_payload)
                if message:
                    errors.append(message)

        parsed_json = self._decode_json_if_possible(text)
        success = returncode == 0 and not errors
        return CLIModelResult(
            success=success,
            text=text,
            parsed_json=parsed_json,
            usage=usage,
            error=None if success else (" | ".join(errors) or stderr.strip()),
        )

    def _cleanup_temp_files(self, temp_files: list[str]) -> None:
        for path in temp_files:
            try:
                Path(path).unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                _log_cli_cleanup_warning_once("file", path, exc)

    def _cleanup_temp_dirs(
        self,
        temp_dirs: list[tempfile.TemporaryDirectory[str]],
    ) -> None:
        for temp_dir in temp_dirs:
            temp_path = getattr(temp_dir, "name", "<unknown>")
            try:
                temp_dir.cleanup()
            except FileNotFoundError:
                continue
            except OSError as exc:
                _log_cli_cleanup_warning_once("dir", temp_path, exc)
                if temp_path and temp_path != "<unknown>":
                    shutil.rmtree(temp_path, ignore_errors=True)

    def _recover_timed_out_result(
        self,
        *,
        returncode: Optional[int],
        stdout: str,
        stderr: str,
        temp_files: list[str],
    ) -> Optional[CLIModelResult]:
        parsed = self._parse_result(
            1 if returncode is None else int(returncode),
            stdout,
            stderr,
            temp_files,
        )
        if not isinstance(parsed.parsed_json, dict):
            return None
        if parsed.success:
            return parsed
        if (
            self.config.backend
            in {LLMBackend.CODEX_CLI, LLMBackend.OPENCODE_CLI, LLMBackend.METACODE_CLI}
            and not parsed.error
        ):
            parsed.success = True
            parsed.error = None
            if not parsed.text:
                parsed.text = json.dumps(parsed.parsed_json)
            return parsed
        return None

    def _terminate_process_tree(self, process: subprocess.Popen[str]) -> tuple[str, str]:
        tracked_pids = self._collect_process_tree_pids(process.pid)
        self._signal_process_tree(tracked_pids, signal.SIGTERM)

        try:
            stdout, stderr = process.communicate(timeout=5)
            return stdout or "", stderr or ""
        except subprocess.TimeoutExpired:
            tracked_pids.update(self._collect_process_tree_pids(process.pid))
            self._signal_process_tree(tracked_pids, signal.SIGKILL)
            try:
                stdout, stderr = process.communicate(timeout=2)
                return stdout or "", stderr or ""
            except subprocess.TimeoutExpired:
                for stream in (process.stdout, process.stderr):
                    try:
                        if stream is not None:
                            stream.close()
                    except OSError:
                        pass
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
                return "", "bounded post-kill drain timed out; partial output unavailable"

    def _kill_process_tree(
        self,
        process: subprocess.Popen[str],
        *,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        tracked_pids = self._collect_process_tree_pids(process.pid)
        self._signal_process_tree(tracked_pids, signal.SIGTERM)
        _cleanup_target_runtime_for_env(env, signum=signal.SIGTERM)
        try:
            process.wait(timeout=5)
            _cleanup_target_runtime_for_env(env, signum=signal.SIGKILL)
            return
        except subprocess.TimeoutExpired:
            tracked_pids.update(self._collect_process_tree_pids(process.pid))
            self._signal_process_tree(tracked_pids, signal.SIGKILL)
            _cleanup_target_runtime_for_env(env, signum=signal.SIGKILL)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    # ------------------------------------------------------------------
    # Phase B.5: turn-parser / observer helpers
    # ------------------------------------------------------------------

    def _build_turn_parser_for_backend(self) -> Optional[Any]:
        """Construct a CLITurnParser for the active backend, when wired.

        Returns ``None`` when no observer is configured — saves the
        per-line parsing cost on the existing zero-observer callsites.
        Failures during import are swallowed and demoted to ``None``;
        the parser is best-effort instrumentation, never load-bearing.
        """
        if self._turn_observer is None:
            return None
        try:
            from .cli_turn_parser import CLITurnParser
        except Exception as exc:  # noqa: BLE001 - never crash the agent loop
            logger.warning(
                "Could not import CLITurnParser (%s: %s); turn observers disabled.",
                type(exc).__name__,
                exc,
            )
            return None
        backend_name = ""
        try:
            backend_name = str(getattr(self.config.backend, "value", "") or "")
        except Exception:  # pragma: no cover - defensive
            backend_name = ""
        try:
            return CLITurnParser(backend_name)
        except Exception as exc:  # noqa: BLE001 - defensive
            logger.warning(
                "CLITurnParser(%r) construction raised %s: %s; observers disabled.",
                backend_name,
                type(exc).__name__,
                exc,
            )
            return None

    def _abort_for_turn_observer(
        self,
        process: subprocess.Popen[str],
        correction: dict[str, Any],
    ) -> None:
        """Terminate the agent subprocess on observer-requested abort.

        Uses the rollout-scoped registry when one is in scope (so we
        only kill THIS rollout's children); otherwise falls back to
        the local ``_kill_process_tree`` for the same effect at a
        slightly broader blast radius. Idempotent — multiple aborts
        from successive turns just signal the (already-dead) tree.
        """
        try:
            registry, rollout_id = self._resolve_rollout_registry()
        except Exception:  # pragma: no cover - defensive
            registry, rollout_id = None, None
        terminated_via_registry = False
        if registry is not None and rollout_id is not None:
            try:
                count = registry.terminate_for_rollout(rollout_id)
                terminated_via_registry = bool(count)
            except Exception as exc:  # noqa: BLE001 - fall back to local kill
                logger.warning(
                    "Observer-requested abort: registry.terminate_for_rollout "
                    "raised %s: %s; falling back to local kill.",
                    type(exc).__name__,
                    exc,
                )
        if not terminated_via_registry:
            try:
                self._kill_process_tree(process)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Observer-requested abort: _kill_process_tree raised %s: %s.",
                    type(exc).__name__,
                    exc,
                )
        logger.warning(
            "Turn observer aborted CLI subprocess pid=%s (turn=%s, source=%s): %s",
            getattr(process, "pid", "?"),
            correction.get("turn_number"),
            correction.get("source"),
            (correction.get("message") or "")[:240],
        )

    def _collect_process_tree_pids(self, root_pid: int) -> set[int]:
        return _collect_subprocess_tree_pids(root_pid)

    def _signal_process_tree(self, pids: set[int], signum: int) -> None:
        _signal_subprocess_tree(pids, signal.Signals(signum))

    def _decode_json_if_possible(self, text: str) -> Optional[dict[str, Any]]:
        text = text.strip()
        if not text:
            return None
        candidates = [text]
        if text.startswith("```"):
            stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
            stripped = re.sub(r"\n?```$", "", stripped).strip()
            candidates.append(stripped)
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if 0 <= first_brace < last_brace:
            candidates.append(text[first_brace : last_brace + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

        for value in self._extract_embedded_json_values(text):
            if isinstance(value, dict):
                return value
        return None

    def _augment_prompt_for_backend(
        self,
        prompt: str,
        schema: Optional[dict[str, Any]],
    ) -> str:
        if not schema:
            return prompt
        if self.config.backend == LLMBackend.GEMINI_CLI:
            schema_text = json.dumps(schema, indent=2, sort_keys=True)
            return (
                f"{prompt}\n\n"
                "Return only a JSON object. Do not include prose, markdown, or code fences.\n"
                f"JSON schema:\n{schema_text}"
            )
        if self.config.backend == LLMBackend.CODEX_CLI:
            normalized_schema = self._normalize_schema_for_codex(schema)
            schema_text = json.dumps(normalized_schema, indent=2, sort_keys=True)
            return (
                f"{prompt}\n\n"
                "Return only a JSON object. Do not include prose, markdown, or code fences.\n"
                "Your final response must match this JSON schema:\n"
                f"{schema_text}"
            )
        return prompt

    def _augment_prompt_for_target_runtime(
        self,
        prompt: str,
        *,
        env: dict[str, str],
    ) -> str:
        prompt = _sanitize_target_runtime_prompt_tool_paths(prompt)
        prompt = _sanitize_target_runtime_prompt_workspace_paths(prompt, env=env)
        guidance = "Use the current workspace and PATH-resolved tools. Keep scratch output here."
        if guidance in prompt:
            return prompt
        return f"{prompt}\n\n{guidance}"

    def _normalize_schema_for_codex(self, schema: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in schema.items():
            if isinstance(value, dict):
                normalized[key] = self._normalize_schema_for_codex(value)
            elif isinstance(value, list):
                normalized[key] = [
                    self._normalize_schema_for_codex(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                normalized[key] = value

        if normalized.get("type") == "object":
            normalized.setdefault("additionalProperties", False)
            properties = normalized.get("properties")
            if isinstance(properties, dict):
                normalized["properties"] = {
                    name: self._normalize_schema_for_codex(property_schema)
                    if isinstance(property_schema, dict)
                    else property_schema
                    for name, property_schema in properties.items()
                }
                normalized["required"] = list(normalized["properties"].keys())

        if normalized.get("type") == "array" and isinstance(normalized.get("items"), dict):
            normalized["items"] = self._normalize_schema_for_codex(normalized["items"])

        return normalized

    def _coerce_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return str(value).strip()

    def _extract_actionable_gemini_stderr(self, stderr: str) -> str:
        cleaned = stderr.strip()
        if not cleaned:
            return ""

        cleaned = re.sub(
            r"(?ms)^innerError Error: Cannot find module '\.\./build/Debug/pty\.node'.*?^\}\s*$",
            "",
            cleaned,
        )
        actionable_lines = [
            line
            for raw_line in cleaned.splitlines()
            if (line := raw_line.strip()) and not self._is_benign_gemini_stderr_line(line)
        ]
        return "\n".join(actionable_lines)

    def _extract_actionable_codex_stderr(self, stderr: str) -> str:
        cleaned = stderr.strip()
        if not cleaned:
            return ""
        actionable_lines = [
            line
            for raw_line in cleaned.splitlines()
            if (line := raw_line.strip()) and not self._is_benign_codex_stderr_line(line)
        ]
        return "\n".join(actionable_lines)

    def _extract_actionable_claude_stderr(self, stderr: str) -> str:
        cleaned = stderr.strip()
        if not cleaned:
            return ""
        actionable_lines = [
            line
            for raw_line in cleaned.splitlines()
            if (line := raw_line.strip()) and not self._is_benign_claude_stderr_line(line)
        ]
        return "\n".join(actionable_lines)

    def _looks_like_codex_startup_only_exit(
        self,
        *,
        stdout: str,
        stderr: str,
        result: CLIModelResult,
    ) -> bool:
        if self.config.backend != LLMBackend.CODEX_CLI:
            return False
        if isinstance(result.parsed_json, dict):
            return False
        if str(result.text or "").strip():
            return False
        stderr_lines = [line.strip() for line in stderr.splitlines() if line.strip()]
        if not stderr_lines:
            return False
        saw_startup_line = False
        for line in stderr_lines:
            if self._is_codex_startup_stderr_line(line):
                saw_startup_line = True
                continue
            if self._is_benign_codex_stderr_line(line):
                continue
            return False
        return saw_startup_line

    def _looks_like_transient_infra_failure(
        self,
        *,
        stdout: str,
        stderr: str,
        result: CLIModelResult,
    ) -> bool:
        """Detect codex/claude exits caused by external infra hiccups.

        Examples seen in production: dotslash 503 fetching a CAS artifact,
        exec sandbox refusing to spawn a child (`Failed to create unified exec
        process`), and `503 Service Unavailable` from intermediate proxies.
        These are not model failures; one retry typically clears them.
        """
        return (
            self._transient_infra_failure_reason(
                stdout=stdout,
                stderr=stderr,
                result=result,
            )
            is not None
        )

    def _transient_infra_failure_reason(
        self,
        *,
        stdout: str,
        stderr: str,
        result: CLIModelResult,
    ) -> Optional[str]:
        if isinstance(result.parsed_json, dict):
            return None
        if self._looks_like_codex_tool_router_failure(stderr=stderr, result=result):
            return "codex_tool_router_failure"
        if self._looks_like_codex_transport_disconnect_failure(
            stdout=stdout,
            stderr=stderr,
            result=result,
        ):
            return "codex_transport_disconnect"
        haystack = "\n".join(
            [str(stdout or ""), str(stderr or ""), str(result.error or "")]
        ).lower()
        result_text = str(result.text or "").strip().lower()
        if result_text:
            haystack = "\n".join([haystack, result_text])
        if not haystack.strip():
            return None
        if (
            self.config.backend == LLMBackend.CLAUDE_CLI
            and "no conversation found with session id" in haystack
        ):
            return "claude_resume_session_missing"
        signatures = (
            "api error: unable to connect to api",
            "error from external service",
            "econnreset",
            "dotslash error",
            "503 service unavailable",
            "artifact likely no longer exists",
            "stream disconnected before completion",
            "error sending request for url",
            "failed to create unified exec process",
            "createprocess",
            "connection reset by peer",
            "temporary failure in name resolution",
            "could not resolve host",
            "tls handshake timeout",
            "context deadline exceeded",
            "cas: read access denied",
            "claude_passthrough",
            "socket hang up",
            "socket connection was closed unexpectedly",
            "segmentation fault",
            "core dumped",
            # codex tool-router losing its child's stdin mid-run (it self-suggests
            # "rerun exec_command with tty=true"); a runtime fault, not a model
            # error — a fresh attempt usually avoids the race.
            "stdin is closed for this session",
            "write_stdin failed",
        )
        codex_text_router_signatures = {
            "stdin is closed for this session",
            "write_stdin failed",
        }
        for signature in signatures:
            if signature not in haystack:
                continue
            if (
                result_text
                and self.config.backend == LLMBackend.CODEX_CLI
                and signature in codex_text_router_signatures
            ):
                continue
            return signature
        return None

    def _looks_like_content_free_cli_exit(
        self,
        *,
        stdout: str,
        stderr: str,
        result: CLIModelResult,
        timeout_audit: Optional[dict[str, Any]],
    ) -> bool:
        """Detect a completed CLI invocation that produced no usable signal."""

        if isinstance(result.parsed_json, dict):
            return False
        if str(result.text or "").strip():
            return False
        if str(stdout or "").strip() or str(stderr or "").strip():
            return False
        audit = timeout_audit if isinstance(timeout_audit, dict) else {}
        terminal_state = str(audit.get("terminal_state") or "").strip()
        if terminal_state and terminal_state != "completed":
            return False
        if audit.get("last_worktree_at"):
            return False
        evidence_counts = audit.get("evidence_counts")
        if isinstance(evidence_counts, Mapping):
            try:
                if int(evidence_counts.get("worktree") or 0) > 0:
                    return False
            except (TypeError, ValueError):
                return False
        output_capture = audit.get("output_capture")
        if isinstance(output_capture, Mapping):
            for stream_name in ("stdout", "stderr"):
                stream = output_capture.get(stream_name)
                if not isinstance(stream, Mapping):
                    continue
                try:
                    if int(stream.get("total_chars") or 0) > 0:
                        return False
                except (TypeError, ValueError):
                    return False
        return True

    def _looks_like_codex_tool_router_failure(
        self,
        *,
        stderr: str,
        result: CLIModelResult,
    ) -> bool:
        """Detect a codex streamable-exec / tool-router failure.

        codex's ``exec_command`` (unified/streamable shell) keeps a persistent
        child session and writes commands to its stdin; under our sandboxed,
        stdin-closed launch that session can die mid-run, emitting
        ``codex_core::tools::router: error=write_stdin failed: stdin is closed
        for this session`` / ``Unknown process id`` / ``rerun exec_command with
        tty=true``. When that happens the agent's tools are broken, so ANY
        narration text it produced is not a usable result — unlike the generic
        transient check, this does NOT bail on non-empty text. Only consulted on
        ``success=False`` (a valid structured result returns before the retry
        path), and retries re-run in the same worktree so real edits made before
        the failure are preserved and harvested rather than discarded."""
        if self.config.backend != LLMBackend.CODEX_CLI:
            return False
        if isinstance(result.parsed_json, dict):
            return False
        haystack = "\n".join([str(stderr or ""), str(result.error or "")]).lower()
        if "codex_core::tools::router" not in haystack:
            return False
        signatures = (
            "write_stdin failed",
            "stdin is closed for this session",
            "unknown process id",
            "rerun exec_command with tty=true",
            "failed to create unified exec process",
        )
        return any(sig in haystack for sig in signatures)

    def _looks_like_codex_transport_disconnect_failure(
        self,
        *,
        stdout: str,
        stderr: str,
        result: CLIModelResult,
    ) -> bool:
        """Detect Codex provider-stream loss after partial, nonterminal output."""

        if self.config.backend != LLMBackend.CODEX_CLI:
            return False
        if isinstance(result.parsed_json, dict):
            return False
        haystack = "\n".join(
            [str(stdout or ""), str(stderr or ""), str(result.error or "")]
        ).lower()
        if not haystack.strip():
            return False
        signatures = (
            "stream disconnected before completion",
            "turn.failed",
            '"type":"turn.failed"',
            "error sending request for url",
        )
        return any(signature in haystack for signature in signatures)

    def _looks_like_claude_bootstrap_only_exit(
        self,
        *,
        stdout: str,
        stderr: str,
        result: CLIModelResult,
    ) -> bool:
        if self.config.backend != LLMBackend.CLAUDE_CLI:
            return False
        if isinstance(result.parsed_json, dict):
            return False
        if str(result.text or "").strip():
            return False
        stderr_lines = [line.strip() for line in stderr.splitlines() if line.strip()]
        if not stderr_lines:
            return False
        saw_bootstrap_line = False
        for line in stderr_lines:
            if self._is_claude_bootstrap_stderr_line(line):
                saw_bootstrap_line = True
                continue
            if self._is_benign_claude_stderr_line(line):
                continue
            return False
        return saw_bootstrap_line

    def _looks_like_claude_no_result_exit(
        self,
        *,
        stdout: str,
        stderr: str,
        result: CLIModelResult,
    ) -> bool:
        """A claude session that exited producing NO structured result and whose
        ENTIRE output (stdout + stderr) is only host-CLI wrapper noise / bootstrap
        / benign lines — i.e. it never actually produced agent work.

        This is the transient no-result case that bootstrap-only-exit misses when
        the wrapper banner lands on STDOUT (the streaming path) instead of stderr.
        Live cause: SEV S671975 (claude certs issued with the wrong agent.id ->
        intermittent loss of access to downstream systems -> rc!=0, banner-only
        stdout, no result event). It is a transient INFRA non-result, NOT a real
        agent failure: a genuine failure produces real output or a (failed)
        structured result, both of which fail this check (so they are never
        masked). Routed through the same infra-retry budget as the bootstrap-only
        / codex-startup-only exits so an intermittent backend fault re-runs
        instead of erroring the rollout."""
        if self.config.backend != LLMBackend.CLAUDE_CLI:
            return False
        if isinstance(result.parsed_json, dict):
            return False
        if str(result.text or "").strip():
            return False
        # Any NON-noise (actionable) content on either stream means the agent
        # produced real output -> this is NOT a content-free no-result exit.
        if not self._claude_output_is_only_nonterminal_stream_or_noise(stdout):
            if self._extract_actionable_claude_stderr(stdout).strip():
                return False
        if self._extract_actionable_claude_stderr(stderr).strip():
            return False
        # Got here: no parsed result, no real text, and all output was wrapper
        # noise (or empty) -> a content-free no-result exit worth one retry.
        return True

    def _looks_like_claude_agent_activity_without_terminal_result(
        self,
        *,
        stdout: str,
        result: CLIModelResult,
    ) -> bool:
        """Claude stream-json ended after tool activity but before result emission.

        This is not a usable patcher response. If the workspace also has no
        observable changes, retry it as an infra non-result rather than charging
        the rollout with a phantom success.
        """

        if self.config.backend != LLMBackend.CLAUDE_CLI:
            return False
        if isinstance(result.parsed_json, dict):
            return False
        text = str(result.text or "")
        if "Claude stream ended without terminal result after agent activity" in text:
            return True
        return bool(self._extract_claude_stream_activity_summary(stdout))

    @staticmethod
    def _timeout_audit_has_worktree_activity(timeout_audit: Any) -> bool:
        if not isinstance(timeout_audit, dict):
            return False
        if timeout_audit.get("last_worktree_at"):
            return True
        evidence_counts = timeout_audit.get("evidence_counts")
        if not isinstance(evidence_counts, dict):
            return False
        try:
            return float(evidence_counts.get("worktree") or 0) > 0
        except (TypeError, ValueError):
            return False

    def _claude_output_is_only_nonterminal_stream_or_noise(self, output: str) -> bool:
        """True when Claude emitted stream-json progress but no terminal result."""

        for raw_line in str(output or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if self._is_benign_claude_stderr_line(line):
                continue
            if self._is_claude_nonterminal_stream_event_line(line):
                continue
            return False
        return True

    def _is_claude_nonterminal_stream_event_line(self, line: str) -> bool:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return False
        if not isinstance(payload, dict):
            return False
        event_type = str(payload.get("type") or "").strip().lower()
        if event_type in {"", "result", "error"}:
            return False
        if self._claude_stream_event_has_agent_activity(payload):
            return False
        return True

    def _extract_claude_stream_activity_summary(
        self,
        output: str,
        *,
        max_chars: int = 2000,
    ) -> str:
        fragments: list[str] = []
        for raw_line in str(output or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            fragments.extend(self._claude_stream_event_activity_fragments(payload))
            if len("; ".join(fragments)) >= max_chars:
                break
        if not fragments:
            return ""
        summary = "Claude stream ended without terminal result after agent activity: "
        detail = "; ".join(fragments)
        if len(summary) + len(detail) > max_chars:
            detail = detail[: max(0, max_chars - len(summary) - 3)].rstrip() + "..."
        return summary + detail

    def _claude_stream_event_has_agent_activity(self, payload: dict[str, Any]) -> bool:
        return bool(self._claude_stream_event_activity_fragments(payload))

    def _claude_stream_event_activity_fragments(self, payload: dict[str, Any]) -> list[str]:
        event_type = str(payload.get("type") or "").strip().lower()
        message = payload.get("message")
        if event_type == "assistant" and isinstance(message, dict):
            return self._claude_assistant_activity_fragments(message)
        if event_type == "user" and isinstance(message, dict):
            return self._claude_tool_result_activity_fragments(message)
        return []

    def _claude_assistant_activity_fragments(self, message: dict[str, Any]) -> list[str]:
        fragments: list[str] = []
        content = message.get("content")
        if not isinstance(content, list):
            return fragments
        for item in content:
            if not isinstance(item, dict):
                continue
            content_type = str(item.get("type") or "").strip().lower()
            if content_type == "text":
                text = self._coerce_text(item.get("text", ""))
                if text:
                    fragments.append(self._truncate_claude_activity_fragment(text))
                continue
            if content_type == "tool_use":
                name = self._coerce_text(item.get("name", "")) or "tool"
                detail = self._claude_tool_use_detail(item.get("input"))
                fragment = f"tool_use:{name}"
                if detail:
                    fragment = f"{fragment} ({detail})"
                fragments.append(self._truncate_claude_activity_fragment(fragment))
        return fragments

    def _claude_tool_result_activity_fragments(self, message: dict[str, Any]) -> list[str]:
        fragments: list[str] = []
        content = message.get("content")
        if not isinstance(content, list):
            return fragments
        for item in content:
            if not isinstance(item, dict):
                continue
            content_type = str(item.get("type") or "").strip().lower()
            if content_type != "tool_result":
                continue
            tool_content = self._coerce_text(item.get("content", ""))
            if tool_content:
                fragments.append(
                    self._truncate_claude_activity_fragment(f"tool_result:{tool_content}")
                )
        return fragments

    def _claude_tool_use_detail(self, value: Any) -> str:
        if not isinstance(value, dict):
            return ""
        for key in ("description", "command", "file_path", "path"):
            detail = self._coerce_text(value.get(key, ""))
            if detail:
                return self._truncate_claude_activity_fragment(detail, max_chars=160)
        return ""

    def _truncate_claude_activity_fragment(
        self,
        value: str,
        *,
        max_chars: int = 240,
    ) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= max_chars:
            return text
        return text[: max(0, max_chars - 3)].rstrip() + "..."

    def _is_host_cli_wrapper_noise_line(self, lowered: str) -> bool:
        """Backend-agnostic host-CLI wrapper chatter that is NEVER the agent's
        coding error: promo banners, GK-gated preset/plugin installs (incl their
        failures), and "<tool> at Meta" startup banners. Shared by the claude AND
        codex stderr classifiers so a noise pattern handled for one backend is
        never silently missed on the other (the recurring claude/codex gap that
        kept erroring otherwise-successful rollouts and starving synthesis)."""
        if not lowered:
            return False
        # "<tool> at Meta" startup banners.
        if lowered.startswith(
            (
                "claude code at meta",
                "metacode at meta",
                "avocado at meta",
                "avocado code at meta",
                "codex cli at meta",
            )
        ):
            return True
        # Avocado/Muse/MetaCode promo banners.
        if (
            "help us improve avocado" in lowered
            or "start using avocado" in lowered
            or "avocado/metacode" in lowered
            or "contribute to muse" in lowered
        ):
            return True
        # GK-gated preset/plugin auto-install chatter (any source) + its failures.
        if lowered.startswith(("installing ", "installed ")) and "for preset" in lowered:
            return True
        if lowered.startswith(("installing ", "installed ")) and "@" in lowered:
            return True
        if "failed to run agent-market" in lowered:
            return True
        if lowered.startswith("gk '") and ("' passed" in lowered or "' failed" in lowered):
            return True
        return False

    def _is_codex_startup_stderr_line(self, line: str) -> bool:
        lowered = str(line or "").strip().lower()
        if not lowered:
            return False
        if self._is_host_cli_wrapper_noise_line(lowered):
            return True
        if "--dangerously-disable-osx-sandbox flag is enabled" in lowered:
            return True
        if lowered.startswith("using codex plugboard"):
            return True
        if lowered.startswith("reading additional input from stdin"):
            return True
        return False

    def _is_claude_bootstrap_stderr_line(self, line: str) -> bool:
        lowered = str(line or "").strip().lower()
        if not lowered:
            return False
        # Shared host-CLI wrapper chatter (promos, preset/plugin installs + their
        # failures, GK gating, "<tool> at Meta" banners) — same set both backends.
        if self._is_host_cli_wrapper_noise_line(lowered):
            return True
        # claude-native bootstrap lines (version resolution / gateway).
        if lowered.startswith("'latest' native version check"):
            return True
        if lowered.startswith("latest native version check"):
            return True
        if lowered.startswith("resolved 'latest' native version to:"):
            return True
        if lowered.startswith("using native claude code binary version:"):
            return True
        if lowered.startswith("downloading native claude_code "):
            return True
        if lowered.startswith("using ai gateway"):
            return True
        return False

    def _is_benign_codex_stderr_line(self, line: str) -> bool:
        return self._is_codex_startup_stderr_line(line)

    def _is_benign_claude_stderr_line(self, line: str) -> bool:
        if line.startswith("Claude Code at Meta"):
            return True
        if line.startswith("MetaCode at Meta"):
            return True
        if line.startswith("Avocado at Meta") or line.startswith("Avocado Code at Meta"):
            return True
        if "--dangerously-disable-osx-sandbox" in line:
            return True
        if line.startswith("Using direct GCP connection"):
            return True
        if line.startswith("SessionEnd hook") and "Hook cancelled" in line:
            return True
        if line.startswith("SessionStart hook") and "Hook cancelled" in line:
            return True
        if self._is_claude_bootstrap_stderr_line(line):
            return True
        return False

    def _is_benign_gemini_stderr_line(self, line: str) -> bool:
        if (
            line.startswith("Gemini CLI at Meta")
            or line.startswith("YOLO mode is enabled.")
            or line.startswith("Timeout of ")
            or line.startswith("The 'metricReader' option is deprecated.")
            or "PeriodicExportingMetricReader" in line
            or "telemetry export" in line.lower()
            or line.startswith("Loading extension:")
        ):
            return True
        if "MCP error -32000: Connection closed" in line:
            return True
        if (
            line.startswith("innerError Error: Cannot find module '../build/Debug/pty.node'")
            or line == "Require stack:"
            or line.startswith("code: 'MODULE_NOT_FOUND'")
            or line.startswith("requireStack:")
            or line == "]"
            or line == "}"
            or line.startswith("- /usr/local/bin/gemini_cli/node_modules/node-pty/")
            or line.startswith("at Module._")
            or line.startswith("at defaultResolveImpl")
            or line.startswith("at resolveForCJSWithHooks")
            or line.startswith("at TracingChannel.traceSync")
            or line.startswith("at wrapModuleLoad")
            or line.startswith("at require")
            or line.startswith("at Object.<anonymous>")
            or line.startswith("'/usr/local/bin/gemini_cli/node_modules/node-pty/")
        ):
            return True
        return False

    def _load_json_payload(
        self,
        text: str,
        preferred_keys: tuple[str, ...] = (),
    ) -> Optional[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        stripped = text.strip()
        if stripped:
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                candidates.append(payload)

        for value in self._extract_embedded_json_values(text):
            if isinstance(value, dict):
                candidates.append(value)

        if not candidates:
            return None
        if preferred_keys:
            matching = [
                candidate
                for candidate in candidates
                if any(key in candidate for key in preferred_keys)
            ]
            if matching:
                return matching[-1]
        return candidates[-1]

    def _extract_embedded_json_values(self, text: str) -> list[Any]:
        decoder = json.JSONDecoder()
        values: list[Any] = []
        seen_spans: set[tuple[int, int]] = set()
        for match in re.finditer(r"[{\[]", text):
            start = match.start()
            try:
                value, end = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            span = (start, start + end)
            if span in seen_spans:
                continue
            seen_spans.add(span)
            values.append(value)
        return values


def reset_cli_backend_health_cache() -> None:
    """Clear cached CLI backend startup probe results."""

    with _CLI_HEALTH_CACHE_LOCK:
        _CLI_HEALTH_CACHE.clear()
    with _AIR_GAPPED_CLI_PREP_LOCK:
        _AIR_GAPPED_CLI_PREPARED.clear()
        _AIR_GAPPED_CLI_PREPARED_WITHOUT_VERSION.clear()


def cli_backend_is_healthy(
    config: LLMConfig,
    *,
    refresh: bool = False,
) -> bool:
    """Return whether the configured CLI backend is installed and starts cleanly."""

    healthy, _ = probe_cli_backend_health(config, refresh=refresh)
    return healthy


def cli_backend_unavailable_reason(
    config: LLMConfig,
    *,
    refresh: bool = False,
) -> str:
    """Return a human-readable reason why a CLI backend should be treated as unavailable."""

    healthy, reason = probe_cli_backend_health(config, refresh=refresh)
    if healthy:
        return ""
    return reason or f"Configured CLI backend '{config.resolved_cli_command}' is unavailable."


class NoCLIBackendAvailable(RuntimeError):
    """Raised when the host has no supported CLI agent on PATH.

    APEX dispatches every model call through a CLI agent (each CLI is its
    own internal agent loop, per project memory note "CLI backends are
    agents not LLMs"). When the user invokes ``apex`` without an explicit
    ``--model`` and no supported CLI is installed, we surface this
    actionable error rather than silently falling back to an API path the
    user never authenticated.
    """


# (cli_command, backend identifier suitable for ``--model``) — ordered
# strongest-first so `detect_default_cli_backend` returns the first
# installed CLI in this preference order. Project memory:
#   "Optimize for SOTA results, never for cost."
# Codex is the SOTA on Commit0/SWE-Bench Pro; Claude (opus) is the
# strongest fallback when codex is absent; Gemini (3.1-pro) is the
# next; opencode/metacode is the last-resort weak agent kept for
# completeness so a host that ONLY has opencode still resolves.
_DEFAULT_CLI_PREFERENCE: tuple[tuple[str, str], ...] = (
    ("codex", "codex_cli:gpt-5.5"),
    ("claude", "claude_cli:opus"),
    ("gemini", "gemini_cli:gemini-3.1-pro"),
    ("opencode", "opencode_cli:meta/avocado-tester"),
)


def detect_default_cli_backend() -> str:
    """Return the default ``backend:model`` identifier for the current host.

    Walks :data:`_DEFAULT_CLI_PREFERENCE` (strongest agent first) and
    returns the first identifier whose CLI binary is on PATH. Used by
    the CLI's ``--model``-less default to route through an installed
    CLI agent instead of the legacy API-backed model that requires a
    shell-level API key the operator may not have configured.

    Raises :class:`NoCLIBackendAvailable` when none of the supported
    CLI binaries are installed.
    """

    for command, identifier in _DEFAULT_CLI_PREFERENCE:
        if shutil.which(command) is not None:
            return identifier
    supported = ", ".join(command for command, _ in _DEFAULT_CLI_PREFERENCE)
    raise NoCLIBackendAvailable(
        "No supported LLM CLI was found on PATH. APEX dispatches every "
        "model call through a CLI agent (codex / claude / gemini / "
        "opencode / metacode). Install at least one of the following and ensure it "
        f"is on PATH: {supported}."
    )


def _opencode_redirected_to_metacode() -> Optional[str]:
    """If ``opencode`` on PATH is the Meta deprecation shim that points at
    ``metacode``, return the resolved path to ``metacode``. Otherwise None.

    Meta replaced the ``opencode`` binary with ``metacode``; the legacy
    ``opencode`` launcher only prints a notice and exits non-zero. When
    that's the case we want the doctor / health probe to test the
    actually-callable ``metacode`` binary instead, otherwise OPENCODE_CLI
    is reported unhealthy on every Meta dev host.
    """

    opencode_path = shutil.which("opencode")
    metacode_path = shutil.which("metacode")
    if not opencode_path or not metacode_path:
        return None
    try:
        # The deprecation shim is the canonical Meta-internal version that
        # symlinks to ``opencode_cli/opencode``. Detect it by reading the
        # symlink target — touching the binary directly via subprocess
        # would print the notice banner on every probe.
        target = os.readlink(opencode_path) if os.path.islink(opencode_path) else opencode_path
    except OSError:
        target = opencode_path
    if "opencode_cli" in str(target) or "metacode" in str(target).lower():
        return metacode_path
    return None


def _session_mode_probe(config: LLMConfig) -> Optional[_CLIHealthProbe]:
    if config.backend == LLMBackend.CLAUDE_CLI and config.is_agentic_backend:
        return (_CLI_HEALTH_PROBE_LOOKUP, config.resolved_cli_command)
    return None


def _cli_health_probe_commands(config: LLMConfig) -> list[_CLIHealthProbe]:
    session_probe = _session_mode_probe(config)
    if session_probe is not None:
        return [session_probe]
    command = [config.resolved_cli_command]
    if config.backend == LLMBackend.CLAUDE_CLI:
        # Claude's launcher can block on `--version`, while `--help` returns
        # quickly enough to verify that the wrapped CLI is installed and starts.
        return [
            (_CLI_HEALTH_PROBE_SUBPROCESS, command + ["--help"]),
            (_CLI_HEALTH_PROBE_SUBPROCESS, command + ["--version"]),
        ]
    if config.backend == LLMBackend.OPENCODE_CLI:
        # Meta dev hosts ship ``opencode`` as a deprecation shim that
        # exits non-zero with a "use metacode" banner. Detect that case
        # and probe ``metacode --version`` (which exits 0) instead so
        # the OPENCODE_CLI backend isn't reported unhealthy purely
        # because the launcher was replaced.
        metacode_path = _opencode_redirected_to_metacode()
        if metacode_path is not None:
            return [
                (_CLI_HEALTH_PROBE_SUBPROCESS, [metacode_path, "--version"]),
                (_CLI_HEALTH_PROBE_SUBPROCESS, [metacode_path, "--help"]),
            ]
        # OpenCode's launcher can take too long to answer `--version` on a cold
        # start. `--help` is a cheaper readiness probe.
        return [
            (_CLI_HEALTH_PROBE_SUBPROCESS, command + ["--help"]),
            (_CLI_HEALTH_PROBE_SUBPROCESS, command + ["--version"]),
        ]
    if config.backend == LLMBackend.METACODE_CLI:
        return [
            (_CLI_HEALTH_PROBE_SUBPROCESS, command + ["--version"]),
            (_CLI_HEALTH_PROBE_SUBPROCESS, command + ["--help"]),
        ]
    if config.backend == LLMBackend.CODEX_CLI:
        # Probe plain `--version`/`--help` FIRST: on a healthy host these exit 0
        # and verify the launcher without invoking codex's sandbox machinery (so
        # no extra Avocado/Plugboard round-trip, and a non-codex stand-in command
        # in tests still passes). Only if those fail do we retry with the macOS
        # sandbox disabled: some codex installs ship without their sandbox profile
        # (`profile.sb` absent under os/mac), so a plain probe then exits non-zero
        # applying the sandbox ("Could not locate macOS sandbox profile
        # profile.sb"). The fallback mirrors how rollouts launch codex
        # (--dangerously-disable-osx-sandbox), so a missing profile never
        # false-prunes the entire codex backend. The probe loop returns healthy
        # on the FIRST command that exits 0, so whichever variant works wins.
        probes: list[_CLIHealthProbe] = [
            (_CLI_HEALTH_PROBE_SUBPROCESS, command + ["--version"]),
            (_CLI_HEALTH_PROBE_SUBPROCESS, command + ["--help"]),
        ]
        if sys.platform == "darwin":
            sandbox_disabled = command + ["--dangerously-disable-osx-sandbox"]
            probes.extend(
                [
                    (_CLI_HEALTH_PROBE_SUBPROCESS, sandbox_disabled + ["--version"]),
                    (_CLI_HEALTH_PROBE_SUBPROCESS, sandbox_disabled + ["--help"]),
                ]
            )
        return probes
    return [
        (_CLI_HEALTH_PROBE_SUBPROCESS, command + ["--version"]),
        (_CLI_HEALTH_PROBE_SUBPROCESS, command + ["--help"]),
    ]


def _build_cli_health_probe_env(
    config: LLMConfig,
    *,
    relocate_target_runtime_home: bool = True,
) -> dict[str, str]:
    # Health checks should validate the launcher in the user's ambient shell
    # environment rather than Apex's nested execution sandbox. The latter is
    # tuned for prompt runs and can make lightweight probe commands hang even
    # when the actual backend is healthy.
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    target_runtime_requested = bool(
        (getattr(config, "cli_env_overrides", {}) or {}).get("APEX_TARGET_TOOL_CONTEXT")
    )
    # host_cli mode runs the real host CLI binary, which may dotslash-materialize
    # itself from the real ~/Library/Caches and authenticate through the host
    # Meta transport. Preserve that transport and NEVER relocate its home (a
    # relocated/empty cache breaks dotslash and the launcher init).
    host_cli_mode = _host_cli_auth_mode_requested(getattr(config, "cli_env_overrides", {}) or {})
    allow_keys = _BACKEND_AUTH_ALLOWLIST.get(config.backend, ()) if target_runtime_requested else ()
    if host_cli_mode:
        allow_keys = tuple(allow_keys) + _HOST_CLI_TRANSPORT_ALLOW_KEYS
        relocate_target_runtime_home = False
    if not getattr(config, "cli_env_redaction_disabled", False):
        env, _removed = redact_host_secrets(
            env,
            allow_keys=allow_keys,
        )
    env.update(_default_cli_env_overrides(config))
    if config.cli_env_overrides:
        env.update(config.cli_env_overrides)
    if not getattr(config, "cli_env_redaction_disabled", False):
        env, _removed = redact_host_secrets(
            env,
            allow_keys=allow_keys,
        )
    _normalize_gemini_provider_env(config, env)
    if (
        target_runtime_requested
        and relocate_target_runtime_home
        and not _target_runtime_uses_docker_sandbox_cli(config, env)
    ):
        target_context = _load_target_runtime_context(env)
        workdir = str(target_context.get("workdir") or "").strip()
        if workdir:
            _prepare_cli_target_runtime_env(config, env, working_dir=workdir)
    return env


def _cli_health_probe_timeout_seconds(config: LLMConfig) -> int:
    configured_timeout = getattr(config, "cli_health_probe_timeout_seconds", None)
    if configured_timeout is not None and int(configured_timeout) > 0:
        return int(configured_timeout)
    return _CLI_PROBE_TIMEOUT_BY_BACKEND.get(config.backend, _CLI_PROBE_TIMEOUT_DEFAULT)


def _target_runtime_health_probe_cache_suffix(config: LLMConfig) -> tuple[str, ...]:
    env_overrides = dict(getattr(config, "cli_env_overrides", {}) or {})
    context_path = str(env_overrides.get("APEX_TARGET_TOOL_CONTEXT") or "").strip()
    if not context_path:
        return ()
    target_context = _load_target_runtime_context(env_overrides)
    runtime = dict(target_context.get("runtime") or {})
    mode = str(runtime.get("kind") or target_context.get("mode") or "")
    if mode not in {"docker_exec", "docker_image"}:
        return ()
    target_identity = str(runtime.get("docker_container_name") or "") or str(
        runtime.get("docker_image") or ""
    )
    spec = _CLI_BACKEND_SANDBOX_SPECS.get(config.backend)

    def _digest(value: str) -> str:
        if not value:
            return ""
        return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:12]

    auth_inputs: list[str] = []
    if spec is not None:
        for key in (
            spec.auth_state_env_key,
            "APEX_TARGET_RUNTIME_CLI_AUTH_MODE",
            *spec.auth_env_allowlist,
            *spec.target_path_env_keys,
            *spec.container_env_keys,
            *_BACKEND_MODEL_PROXY_ENV_KEYS.get(config.backend, ()),
            *_GLOBAL_MODEL_PROXY_ENV_KEYS,
        ):
            raw = str(env_overrides.get(key) or os.environ.get(key) or "").strip()
            if raw:
                auth_inputs.append(f"{key}:{_digest(raw)}")
    return (
        mode,
        context_path,
        target_identity,
        str(config.resolved_cli_model or config.model or ""),
        _digest("|".join(sorted(set(auth_inputs)))),
    )


def _target_container_cli_lookup_command(
    config: LLMConfig,
    *,
    env: dict[str, str],
    context: _AgentContainerLaunchContext,
    cli_name: str,
) -> list[str]:
    docker_command = [
        context.docker_bin,
        "exec",
    ]
    if context.docker_user:
        docker_command.extend(["-u", context.docker_user])
    docker_command.extend(["-w", context.working_dir_container])
    spec = _cli_backend_sandbox_spec(config.backend)
    probe_env_keys = {
        *spec.container_env_keys,
        *spec.target_path_env_keys,
        "APEX_TARGET_TOOL_CONTEXT",
        "APEX_TARGET_TOOL_WORKDIR",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "NO_COLOR",
        "PATH",
        "TERM",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
    }
    container_probe_env = {
        key: value
        for key, value in env.items()
        if key in probe_env_keys or str(key).startswith("LC_")
    }
    for key, value in sorted(_agent_container_env(container_probe_env, context).items()):
        if value == "":
            continue
        docker_command.extend(["-e", f"{key}={value}"])
    probe_command = [cli_name]
    if cli_name in _NODE_AGENT_CONTAINER_CLI_BINARIES:
        probe_command.append("node")
    docker_command.extend(
        [
            context.container_name,
            "/bin/sh",
            "-c",
            (
                'command -v "$1" >/dev/null && '
                'if [ "$#" -ge 2 ]; then command -v "$2" >/dev/null; fi'
            ),
            "apex-cli-health",
            *probe_command,
        ]
    )
    return docker_command


def _probe_cli_backend_health_in_agent_container(
    config: LLMConfig,
    *,
    env: dict[str, str],
    timeout_seconds: int,
) -> tuple[bool, str]:
    target_context = _load_target_runtime_context(env)
    workdir = str(target_context.get("workdir") or "").strip()
    if not workdir:
        return (
            False,
            f"CLI backend '{config.resolved_cli_command}' target container context has no workdir.",
        )
    try:
        context = _agent_container_launch_context(env, working_dir=workdir)
    except CLIAgentContainerIsolationError as exc:
        return (False, str(exc))
    if context is None:
        return (
            False,
            f"CLI backend '{config.resolved_cli_command}' has no target container context.",
        )

    cli_name = Path(str(config.resolved_cli_command or "")).name
    if not cli_name:
        return (False, "Configured CLI backend command is empty.")
    auth_keys = tuple(_BACKEND_AUTH_ALLOWLIST.get(config.backend, ()))
    spec = _cli_backend_sandbox_spec(config.backend)
    auth_hint = ", ".join(_backend_auth_hint_keys(spec))
    auth_configured = not auth_keys or _backend_auth_configured(config, env)
    model_proxy_active = str(env.get("APEX_AGENT_MODEL_PROXY_ACTIVE") or "").strip() == "1"
    temp_files: list[str] = []
    lookup_command = _target_container_cli_lookup_command(
        config,
        env=env,
        context=context,
        cli_name=cli_name,
    )
    launch_env = dict(env)
    launch_env.update(context.docker_host_env)
    try:
        lookup_probe = subprocess.run(
            lookup_command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=launch_env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return (
            False,
            (
                f"CLI backend '{cli_name}' did not respond to target-container lookup "
                f"within {timeout_seconds}s."
            ),
        )
    except OSError as exc:
        return (False, f"CLI backend '{cli_name}' target-container lookup failed: {exc}")
    if lookup_probe.returncode != 0:
        detail = _compact_cli_probe_detail(
            (lookup_probe.stdout or "") + (lookup_probe.stderr or "")
        )
        suffix = f": {detail}" if detail else ""
        return (
            False,
            f"CLI backend '{cli_name}' is not installed in target container '{context.container_name}'{suffix}.",
        )

    auth_probe = True
    try:
        auth_command, temp_files = _target_container_auth_smoke_command(
            config,
            env,
            working_dir=workdir,
        )
        docker_command = _docker_exec_command_for_agent_container(
            auth_command,
            env,
            context,
            backend=config.backend,
            auth_env_keys=auth_keys,
        )
    except CLIAgentContainerIsolationError as exc:
        return (False, str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        return (
            False,
            f"CLI backend '{cli_name}' target-container auth probe setup failed: {exc}",
        )
    if not auth_configured and not model_proxy_active:
        logger.debug(
            "Running target-container CLI smoke probe for %s without detected portable auth; "
            "failure will mark the backend unavailable.",
            cli_name,
        )
    effective_timeout_seconds = max(int(timeout_seconds), 60) if auth_probe else timeout_seconds
    probe_completed = False
    try:
        with _cli_backend_concurrency_slot(
            config,
            target_runtime_enforced=True,
            working_dir=workdir,
        ):
            probe = subprocess.run(
                docker_command,
                capture_output=True,
                text=True,
                timeout=effective_timeout_seconds,
                env=launch_env,
                check=False,
            )
        probe_completed = True
    except subprocess.TimeoutExpired:
        if auth_probe and model_proxy_active:
            return (
                False,
                (
                    f"CLI backend '{cli_name}' did not complete target-container "
                    f"model-proxy smoke probe within {effective_timeout_seconds}s; "
                    "verify that the configured APEX model proxy endpoint is reachable "
                    "from the target container and can authenticate model requests."
                ),
            )
        if auth_probe:
            return (
                False,
                (
                    f"CLI backend '{cli_name}' did not complete target-container "
                    f"auth probe within {effective_timeout_seconds}s; set one of: "
                    f"{auth_hint}, set {spec.auth_state_env_key} to a "
                    "portable credential file/directory, or authenticate the CLI "
                    "in a way the target-container binary can use."
                ),
            )
        return (
            False,
            (
                f"CLI backend '{cli_name}' did not respond to target-container lookup "
                f"within {timeout_seconds}s."
            ),
        )
    except OSError as exc:
        return (False, f"CLI backend '{cli_name}' target-container lookup failed: {exc}")
    finally:
        if temp_files and not probe_completed:
            CLIModelClient(config)._cleanup_temp_files(temp_files)
    try:
        if probe.returncode == 0:
            marker_error = _target_container_health_marker_error(
                config,
                returncode=probe.returncode,
                stdout=probe.stdout or "",
                stderr=probe.stderr or "",
                temp_files=temp_files,
            )
            if not marker_error:
                return (True, "")
            suffix = f": {_compact_cli_probe_detail(marker_error)}"
            if auth_probe and model_proxy_active:
                return (
                    False,
                    (
                        f"CLI backend '{cli_name}' target-container model-proxy smoke probe "
                        "failed; verify the configured APEX model proxy endpoint is reachable "
                        f"from the target container and can authenticate model requests{suffix}."
                    ),
                )
            if auth_probe:
                return (
                    False,
                    (
                        f"CLI backend '{cli_name}' target-container auth probe failed; "
                        f"set one of: {auth_hint}, set "
                        f"{spec.auth_state_env_key} to a portable credential file/directory, "
                        "or authenticate the CLI in the terminal so APEX can materialize "
                        f"portable auth state into the target runtime{suffix}."
                    ),
                )
            return (
                False,
                f"CLI backend '{cli_name}' is not installed in target container '{context.container_name}'{suffix}.",
            )
        detail = _compact_cli_probe_detail((probe.stdout or "") + (probe.stderr or ""))
        suffix = f": {detail}" if detail else ""
        if auth_probe and model_proxy_active:
            return (
                False,
                (
                    f"CLI backend '{cli_name}' target-container model-proxy smoke probe "
                    "failed; verify the configured APEX model proxy endpoint is reachable "
                    f"from the target container and can authenticate model requests{suffix}."
                ),
            )
        if auth_probe:
            return (
                False,
                (
                    f"CLI backend '{cli_name}' target-container auth probe failed; "
                    f"set one of: {auth_hint}, set "
                    f"{spec.auth_state_env_key} to a portable credential file/directory, "
                    "or authenticate the CLI in the terminal so APEX can materialize "
                    f"portable auth state into the target runtime{suffix}."
                ),
            )
        return (
            False,
            f"CLI backend '{cli_name}' is not installed in target container '{context.container_name}'{suffix}.",
        )
    finally:
        if temp_files:
            CLIModelClient(config)._cleanup_temp_files(temp_files)


def _probe_cli_backend_health_in_agent_image(
    config: LLMConfig,
    *,
    env: dict[str, str],
    timeout_seconds: int,
) -> tuple[bool, str]:
    target_context = _load_target_runtime_context(env)
    workdir = str(target_context.get("workdir") or "").strip()
    if not workdir:
        return (
            False,
            f"CLI backend '{config.resolved_cli_command}' docker image context has no workdir.",
        )
    cli_name = Path(str(config.resolved_cli_command or "")).name
    auth_keys = tuple(_BACKEND_AUTH_ALLOWLIST.get(config.backend, ()))
    temp_files: list[str] = []
    probe_completed = False
    try:
        auth_command, temp_files = _target_container_auth_smoke_command(
            config,
            env,
            working_dir=workdir,
        )
        context, runtime = _agent_image_launch_context(env, working_dir=workdir)
        docker_command = _docker_run_command_for_agent_image(
            auth_command,
            env,
            context,
            runtime,
            backend=config.backend,
            auth_env_keys=auth_keys,
        )
        launch_env = dict(env)
        launch_env.update(context.docker_host_env)
        with _cli_backend_concurrency_slot(
            config,
            target_runtime_enforced=True,
            working_dir=workdir,
        ):
            probe = subprocess.run(
                docker_command,
                capture_output=True,
                text=True,
                timeout=max(int(timeout_seconds), 90),
                env=launch_env,
                check=False,
            )
        probe_completed = True
    except subprocess.TimeoutExpired:
        return (
            False,
            (
                f"CLI backend '{cli_name}' did not complete docker image auth probe "
                f"within {max(int(timeout_seconds), 90)}s."
            ),
        )
    except CLIAgentContainerIsolationError as exc:
        return (False, str(exc))
    except OSError as exc:
        return (False, f"CLI backend '{cli_name}' docker image launch failed: {exc}")
    except Exception as exc:  # pragma: no cover - defensive
        return (False, f"CLI backend '{cli_name}' docker image probe setup failed: {exc}")
    finally:
        if temp_files and not probe_completed:
            CLIModelClient(config)._cleanup_temp_files(temp_files)
    try:
        if probe.returncode == 0:
            marker_error = _target_container_health_marker_error(
                config,
                returncode=probe.returncode,
                stdout=probe.stdout or "",
                stderr=probe.stderr or "",
                temp_files=temp_files,
            )
            if not marker_error:
                return (True, "")
            suffix = f": {_compact_cli_probe_detail(marker_error)}"
            return (
                False,
                f"CLI backend '{cli_name}' docker image auth probe failed{suffix}.",
            )
        detail = _compact_cli_probe_detail((probe.stdout or "") + (probe.stderr or ""))
        suffix = f": {detail}" if detail else ""
        return (False, f"CLI backend '{cli_name}' docker image auth probe failed{suffix}.")
    finally:
        if temp_files:
            CLIModelClient(config)._cleanup_temp_files(temp_files)


def _probe_cli_backend_health_in_docker_sandbox(
    config: LLMConfig,
    *,
    env: dict[str, str],
    timeout_seconds: int,
) -> tuple[bool, str]:
    target_context = _load_target_runtime_context(env)
    workdir = str(target_context.get("workdir") or "").strip()
    if not workdir:
        return (
            False,
            f"CLI backend '{config.resolved_cli_command}' Docker Sandbox context has no workdir.",
        )
    agent = _DOCKER_SANDBOX_AGENT_BY_BACKEND.get(config.backend, "")
    if not agent:
        return (
            False,
            f"CLI backend '{config.backend.value}' has no Docker Sandbox agent mapping.",
        )
    temp_files: list[str] = []
    shim_dir = Path()
    sandbox_name = ""
    probe_completed = False
    try:
        sandbox_env, shim_dir = _write_agent_visible_target_tool_shims(
            env,
            working_dir=workdir,
        )
        auth_command, temp_files = _target_container_auth_smoke_command(
            config,
            sandbox_env,
            working_dir=workdir,
            codex_output_in_workspace=True,
        )
        sandbox_name = _docker_sandbox_name(config, working_dir=workdir)
        _create_docker_sandbox_for_agent(
            config,
            sandbox_env,
            working_dir=workdir,
            sandbox_name=sandbox_name,
        )
        docker_command = _docker_sandbox_exec_command(
            config,
            auth_command,
            sandbox_env,
            sandbox_name=sandbox_name,
            working_dir=workdir,
        )
        effective_timeout_seconds = max(int(timeout_seconds), 90)
        with _TargetToolBridge(
            sandbox_env,
            working_dir=workdir,
            descriptor_path=Path(sandbox_env["APEX_TARGET_TOOL_BRIDGE_FILE"]),
        ):
            probe = subprocess.run(
                docker_command,
                capture_output=True,
                text=True,
                timeout=effective_timeout_seconds,
                cwd=workdir,
                env=dict(os.environ),
                check=False,
            )
            probe_completed = True
    except subprocess.TimeoutExpired:
        return (
            False,
            (
                f"CLI backend '{agent}' did not complete Docker Sandbox auth probe "
                f"within {max(int(timeout_seconds), 90)}s; verify Docker Sandboxes "
                "and the host-authenticated provider CLI are working."
            ),
        )
    except CLIAgentContainerIsolationError as exc:
        return (False, str(exc))
    except OSError as exc:
        return (False, f"CLI backend '{agent}' Docker Sandbox launch failed: {exc}")
    except Exception as exc:  # pragma: no cover - defensive
        return (False, f"CLI backend '{agent}' Docker Sandbox probe setup failed: {exc}")
    finally:
        if temp_files and not probe_completed:
            CLIModelClient(config)._cleanup_temp_files(temp_files)
        if shim_dir:
            shutil.rmtree(shim_dir, ignore_errors=True)
        if sandbox_name:
            _remove_docker_sandbox(sandbox_name)
    try:
        if probe.returncode == 0:
            marker_error = _target_container_health_marker_error(
                config,
                returncode=probe.returncode,
                stdout=probe.stdout or "",
                stderr=probe.stderr or "",
                temp_files=temp_files,
            )
            if not marker_error:
                return (True, "")
            suffix = f": {_compact_cli_probe_detail(marker_error)}"
            return (
                False,
                (
                    f"CLI backend '{agent}' Docker Sandbox auth probe failed; authenticate the "
                    "provider CLI in the host terminal and verify `docker sandbox run "
                    f"{agent}` is available{suffix}."
                ),
            )
        detail = _compact_cli_probe_detail((probe.stdout or "") + (probe.stderr or ""))
        suffix = f": {detail}" if detail else ""
        return (
            False,
            (
                f"CLI backend '{agent}' Docker Sandbox auth probe failed; authenticate the "
                "provider CLI in the host terminal and verify `docker sandbox run "
                f"{agent}` is available{suffix}."
            ),
        )
    finally:
        if temp_files:
            CLIModelClient(config)._cleanup_temp_files(temp_files)


def _target_container_auth_smoke_command(
    config: LLMConfig,
    env: dict[str, str],
    *,
    working_dir: str,
    codex_output_in_workspace: bool = False,
) -> tuple[list[str], list[str]]:
    """Build a minimal no-edit prompt that proves unmodeled container auth."""

    client = CLIModelClient(config)
    return client._build_command(
        prompt="APEX CLI health check. Reply with exactly: APEX_OK",
        working_dir=working_dir,
        schema=None,
        system_prompt=None,
        allow_edits=False,
        internet_enabled=False,
        target_runtime_enforced=True,
        sandbox_writable_roots=_cli_sandbox_writable_roots(
            env,
            working_dir=working_dir,
            include_cli_home=True,
        ),
        codex_base_url=str(env.get("CODEX_BASE_URL") or os.environ.get("CODEX_BASE_URL") or ""),
        cli_hook_args=[],
        codex_output_in_workspace=codex_output_in_workspace,
    )


def _target_container_health_marker_error(
    config: LLMConfig,
    *,
    returncode: int,
    stdout: str,
    stderr: str,
    temp_files: list[str],
    marker: str = "APEX_OK",
) -> str:
    client = CLIModelClient(config)
    try:
        result = client._parse_result(returncode, stdout, stderr, temp_files)
    except Exception as exc:  # pragma: no cover - defensive parser isolation
        return f"target-container smoke probe output could not be parsed: {exc}"

    fragments: list[str] = []
    text = str(getattr(result, "text", "") or "").strip()
    if text:
        fragments.append(text)
    parsed_json = getattr(result, "parsed_json", None)
    if parsed_json is not None:
        try:
            fragments.append(json.dumps(parsed_json, sort_keys=True))
        except TypeError:
            fragments.append(str(parsed_json))
    error = str(getattr(result, "error", "") or "").strip()
    if error:
        fragments.append(error)
    if not bool(getattr(result, "success", False)):
        detail = _compact_cli_probe_detail("\n".join(fragments) or stdout or stderr)
        return "target-container smoke probe did not complete successfully" + (
            f": {detail}" if detail else ""
        )
    if any(marker in fragment for fragment in fragments):
        return ""
    detail = _compact_cli_probe_detail("\n".join(fragments) or stdout or stderr)
    return f"target-container smoke probe did not produce expected {marker} marker" + (
        f": {detail}" if detail else ""
    )


def _compact_cli_probe_detail(text: str, *, limit: int = 800) -> str:
    detail = _ANSI_ESCAPE_RE.sub("", str(text or "")).strip()
    detail = re.sub(r"\s+", " ", detail)
    if len(detail) <= limit:
        return detail
    return detail[: max(0, limit - 3)].rstrip() + "..."


def _probe_cli_backend_health_on_host(
    config: LLMConfig,
    *,
    env: Optional[dict[str, str]] = None,
) -> tuple[bool, str]:
    resolved = shutil.which(config.resolved_cli_command)
    if resolved is None:
        return (
            False,
            f"CLI backend '{config.resolved_cli_command}' is not installed.",
        )

    probe_env = env if env is not None else _build_cli_health_probe_env(config)
    probe_timeout = _cli_health_probe_timeout_seconds(config)
    failure_reasons: list[str] = []
    result: tuple[bool, str] | None = None
    for probe_kind, probe_target in _cli_health_probe_commands(config):
        if probe_kind == _CLI_HEALTH_PROBE_LOOKUP:
            lookup_command = str(probe_target or "").strip()
            if lookup_command and shutil.which(lookup_command):
                result = (True, "")
                break
            failure_reasons.append(f"CLI backend '{config.resolved_cli_command}' is not installed.")
            continue
        probe_command = (
            list(probe_target) if isinstance(probe_target, list) else [str(probe_target)]
        )
        probe_command = _prepare_cli_command_for_target_tool_path(probe_command, probe_env)
        probe_label = " ".join(probe_command[1:]).strip() or probe_command[0]
        try:
            probe = subprocess.run(
                probe_command,
                capture_output=True,
                text=True,
                timeout=probe_timeout,
                env=_cli_launch_env_for_target_runtime(probe_env, probe_command),
            )
        except subprocess.TimeoutExpired:
            failure_reasons.append(
                (
                    f"CLI backend '{config.resolved_cli_command}' did not respond to "
                    f"'{probe_label}' within {probe_timeout}s."
                )
            )
            continue
        except OSError as exc:
            result = (
                False,
                f"CLI backend '{config.resolved_cli_command}' failed to start: {exc}",
            )
            break

        if probe.returncode == 0:
            result = (True, "")
            break

        failure_reasons.append(
            _summarize_cli_probe_failure(
                " ".join(probe_command),
                probe.stdout,
                probe.stderr,
                probe.returncode,
            )
        )

    if result is not None:
        return result
    if len(failure_reasons) == 1:
        return False, failure_reasons[0]
    if failure_reasons:
        return False, " ; ".join(failure_reasons[:2])
    return (
        False,
        f"CLI backend '{config.resolved_cli_command}' failed its startup probe.",
    )


def probe_cli_backend_health(
    config: LLMConfig,
    *,
    refresh: bool = False,
) -> tuple[bool, str]:
    """Best-effort startup probe for CLI backends.

    A backend counts as healthy only when its executable exists and at least one
    cheap startup probe completes successfully.
    """

    if not config.is_cli_backend:
        return False, "Configured model is not a CLI backend."

    target_cache_suffix = _target_runtime_health_probe_cache_suffix(config)
    cache_key = (config.backend.value, config.resolved_cli_command, *target_cache_suffix)
    with _CLI_HEALTH_CACHE_LOCK:
        if not refresh and cache_key in _CLI_HEALTH_CACHE:
            return _CLI_HEALTH_CACHE[cache_key]

    if target_cache_suffix:
        try:
            env = _build_cli_health_probe_env(config, relocate_target_runtime_home=True)
            if _target_runtime_uses_host_cli(config, env):
                # host_cli mode runs the agent CLI on the host, so its health +
                # auth must be probed on the host (the in-container public CLI has
                # no usable Meta credentials and would be wrongly pruned).
                target_result = _probe_cli_backend_health_on_host(config)
            elif _target_runtime_uses_docker_sandbox_cli(config, env):
                target_result = _probe_cli_backend_health_in_docker_sandbox(
                    config,
                    env=env,
                    timeout_seconds=_cli_health_probe_timeout_seconds(config),
                )
            elif _target_runtime_launches_agent_cli_in_docker_image(config, env):
                target_result = _probe_cli_backend_health_in_agent_image(
                    config,
                    env=env,
                    timeout_seconds=_cli_health_probe_timeout_seconds(config),
                )
            else:
                target_result = _probe_cli_backend_health_in_agent_container(
                    config,
                    env=env,
                    timeout_seconds=_cli_health_probe_timeout_seconds(config),
                )
        except CLIAgentContainerIsolationError as exc:
            target_result = (False, str(exc))
        with _CLI_HEALTH_CACHE_LOCK:
            _CLI_HEALTH_CACHE[cache_key] = target_result
        return target_result

    result = _probe_cli_backend_health_on_host(config)

    with _CLI_HEALTH_CACHE_LOCK:
        _CLI_HEALTH_CACHE[cache_key] = result
    return result


def _summarize_cli_probe_failure(
    command: str,
    stdout: str,
    stderr: str,
    returncode: int,
) -> str:
    lines = [
        line.strip() for line in f"{stdout or ''}\n{stderr or ''}".splitlines() if line.strip()
    ]
    preview = "\n".join(lines[:8]).strip()
    if len(preview) > 600:
        preview = preview[:597].rstrip() + "..."
    if preview:
        return (
            f"CLI backend '{command}' failed its startup probe (exit code {returncode}): {preview}"
        )
    return f"CLI backend '{command}' failed its startup probe (exit code {returncode})."
