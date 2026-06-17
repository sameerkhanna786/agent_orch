"""Normalized vendor-neutral Executor wrapping v1's CLIModelClient (A10).

WRAPS (does not replace) v1 cli_backend/llm_routing/backend_portfolio.  The git
diff over the worktree is THE authoritative artifact (filesystem-as-truth); the
JSON stream is telemetry.  The result is the plan's ExecResult with first-class
``session_id`` + ``finalization_status`` (v1 only surfaces these via
backend_diagnostics/raw_output, so we add them explicitly).

This adapter is what makes a single APEX-Ω solve run on Codex, Claude Code, or a
mixed fleet — the vendor is just a field on the worker spec.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..types import CapabilityProfile, ExecResult, ScopedTask, TokenUsage
from .capability import negotiate as _negotiate

# v1 imports are done lazily inside methods so the engine package can be imported
# (and unit-tested with the FakeExecutor) without the apex venv on the path.


# map v1 CLIModelResult.finalization_status -> plan FINALIZATION_STATUSES
_FINALIZATION_MAP = {
    "completed": "completed",
    "success": "completed",
    "ok": "completed",
    "timeout": "timeout",
    "policy_violation": "policy_violation",
    "interaction_required": "policy_violation",
    "output_limit": "output_limit",
    "progress_abort": "progress_abort",
    "isolation_error": "isolation_error",
}


def _map_finalization(status: Optional[str], success: bool) -> str:
    if status:
        mapped = _FINALIZATION_MAP.get(str(status).lower())
        if mapped:
            return mapped
    return "completed" if success else "infra_nonresult"


def _normalize_usage(usage: dict | None) -> TokenUsage:
    """Best-effort cross-vendor usage normalization.  Falls back to
    extract_total_tokens (v1) when per-field split is unavailable."""
    usage = usage if isinstance(usage, dict) else {}

    def _first(*keys: str) -> int:
        for k in keys:
            v = usage.get(k)
            if isinstance(v, (int, float)):
                return int(v)
        return 0

    inp = _first("input_tokens", "prompt_tokens", "input")
    out = _first("output_tokens", "completion_tokens", "output")
    cached = _first("cache_read_input_tokens", "cached_input_tokens", "cached_input", "cache_read_tokens")
    reasoning = _first("reasoning_tokens", "reasoning")
    cache_creation = _first("cache_creation_input_tokens", "cache_creation_tokens", "cache_write_tokens")
    if inp == 0 and out == 0:
        try:
            from apex.core.cli_backend import extract_total_tokens
            out = int(extract_total_tokens(usage))
        except Exception:
            out = 0
    return TokenUsage(input=inp, output=out, cached_input=cached, reasoning=reasoning, cache_creation=cache_creation)


def _git_diff(cwd: str, baseline_ref: str = "HEAD") -> str:
    """Worktree diff vs the base commit (filesystem-as-truth).  intent-to-add so
    new files (the common commit0 case) appear in the diff.  Never raises."""
    try:
        subprocess.run(["git", "-C", cwd, "add", "-N", "."], capture_output=True, text=True, timeout=120)
        out = subprocess.run(
            ["git", "-C", cwd, "diff", "--no-color", "--no-ext-diff", baseline_ref],
            capture_output=True, text=True, timeout=120,
        )
        return out.stdout or ""
    except Exception:
        return ""


def build_llm_config(spec: dict) -> Any:
    """Build a v1 LLMConfig from a vendor/model spec via the known-good ApexConfig
    loader (avoids guessing the LLMConfig constructor).  ``spec`` keys mirror the
    ``llm_configs[*]`` entries in the benchmark configs."""
    from apex.core.config import ApexConfig
    entry = {k: v for k, v in spec.items() if v is not None}
    entry.setdefault("backend", "codex_cli")
    cfg = ApexConfig.from_dict({"llm_configs": [entry]})
    return cfg.llm_configs[0]


@dataclass
class _SpawnSpec:
    vendor: str
    model: str
    cli_command: Optional[str] = None
    cli_model_id: Optional[str] = None
    cli_timeout: Optional[int] = None
    cli_hard_timeout_seconds: Optional[int] = None
    cli_disable_osx_sandbox: bool = True
    cli_permission_mode: Optional[str] = None


class V1Session:
    """A spawned vendor worker bound to a worktree."""

    def __init__(self, worktree_cwd: str, vendor: str, model: str, cli_version: str, llm_config: Any,
                 *, baseline_ref: str = "HEAD"):
        self.cwd = str(worktree_cwd)
        self.vendor = vendor
        self.model = model
        self.cli_version = cli_version
        self.baseline_ref = baseline_ref
        self._llm_config = llm_config
        self._client = None  # lazily constructed

    def _ensure_client(self):
        if self._client is None:
            from apex.core.cli_backend import CLIModelClient
            self._client = CLIModelClient(self._llm_config)
        return self._client

    def observe(self) -> str:
        return _git_diff(self.cwd, self.baseline_ref)

    # alias matching the plan's Session.observe_diff name
    def observe_diff(self) -> str:
        return self.observe()

    def run(self, task: ScopedTask) -> ExecResult:
        """Run one scoped task.  NEVER raises (typed-failure contract)."""
        start = time.monotonic()
        allow_edits = task.sandbox != "read-only"
        try:
            client = self._ensure_client()
            res = client.run_structured_prompt(
                task.prompt,
                working_dir=self.cwd,
                schema=task.schema,
                allow_edits=allow_edits,
                internet_enabled=bool(task.internet),
                hard_timeout_seconds=task.timeout_seconds,
            )
        except Exception as exc:
            return ExecResult(
                ok=False, finalization_status="infra_nonresult",
                error=f"{type(exc).__name__}: {exc}", vendor=self.vendor, model=self.model,
                cli_version=self.cli_version, latency_seconds=time.monotonic() - start,
                fs_diff=self.observe(),
            )

        diff = self.observe()
        usage = _normalize_usage(getattr(res, "usage", None))
        finalization = _map_finalization(getattr(res, "finalization_status", None), bool(getattr(res, "success", False)))
        session_id = None
        diag = getattr(res, "backend_diagnostics", None) or {}
        if isinstance(diag, dict):
            session_id = diag.get("session_id") or diag.get("thread_id")
        return ExecResult(
            final_message=getattr(res, "text", "") or "",
            structured_output=getattr(res, "parsed_json", None),
            usage=usage,
            session_id=session_id,
            raw_events=[],  # telemetry; intentionally not journaled
            fs_diff=diff,
            vendor=self.vendor,
            model=self.model,
            cli_version=self.cli_version,
            ok=bool(getattr(res, "success", False)),
            finalization_status=finalization,
            error=getattr(res, "error", None),
            latency_seconds=time.monotonic() - start,
        )


class V1Executor:
    """Normalized Executor over the v1 CLI backends.  Vendor-neutral: pass any of
    codex_cli / claude_cli / gemini_cli / opencode_cli / metacode_cli."""

    def __init__(self, *, probe_versions: bool = True, provision_auth: bool = True):
        self.probe_versions = probe_versions
        self._cap_cache: dict[tuple[str, str, str], CapabilityProfile] = {}
        if provision_auth:
            # Point the vendor CLIs at the Meta gateway in host mode (setdefault;
            # operator env always wins). Makes codex authenticate in-process.
            from .auth_env import ensure_vendor_auth_env
            ensure_vendor_auth_env()

    def negotiate(self, vendor: str, model: str, version: str = "") -> CapabilityProfile:
        key = (vendor, model, version)
        if key not in self._cap_cache:
            self._cap_cache[key] = _negotiate(vendor, model, version, probe=self.probe_versions)
        return self._cap_cache[key]

    def spawn(self, worktree_cwd: str, vendor: str, model: str, version: str = "",
              *, spec: Optional[dict] = None, baseline_ref: str = "HEAD") -> V1Session:
        profile = self.negotiate(vendor, model, version)
        full_spec = {"backend": vendor, "model": model}
        if spec:
            full_spec.update({k: v for k, v in spec.items() if v is not None})
        # This sandbox denies the codex/opencode OS sandbox; disable it by default
        # (operator/config value wins).
        full_spec.setdefault("cli_disable_osx_sandbox", True)
        llm_config = build_llm_config(full_spec)
        return V1Session(
            worktree_cwd, vendor, model, profile.cli_version, llm_config, baseline_ref=baseline_ref
        )
