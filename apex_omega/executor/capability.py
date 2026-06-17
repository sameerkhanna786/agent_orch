"""ACP-style capability negotiation (Fusion Ledger A10; plan §03).

The hard rule is *degrade, do not crash*.  APEX's STATIC_CAPABILITY_TABLE WINS
on conflict; vendor self-report is advisory telemetry only.  Actual degradation
is largely executed inside v1's ``run_structured_prompt`` (schema-embed +
post-parse when ``native_schema`` is absent) and by APEX's own worktree+fcntl
isolation floor (so even a full-access vendor is confined to its own worktree).
This module records *which* capabilities exist so the manifest/controller can
reason about them and so an interception gap is logged, never silently ignored.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Optional

from ..types import CapabilityProfile


# vendor -> declared capability surface (from the recon of v1 cli_backend._build_command)
STATIC_CAPABILITY_TABLE: dict[str, dict] = {
    "codex_cli": {
        "native_schema": True,              # --output-schema + --output-last-message
        "sandbox_levels": ("read-only", "workspace-write", "danger-full-access"),
        "internet": True,                   # --dangerously-enable-internet-mode --search
        "thinking": True,
        "bidirectional_stream": True,       # --json NDJSON
        "mcp": True,
        "effort_levels": ("low", "medium", "high", "xhigh", "max"),
        "resume": True,
    },
    "claude_cli": {
        "native_schema": True,              # --json-schema
        "sandbox_levels": ("read-only", "workspace-write"),  # via --permission-mode/--allowedTools
        "internet": True,
        "thinking": True,
        "bidirectional_stream": True,       # --output-format stream-json
        "mcp": True,
        "effort_levels": ("low", "medium", "high"),
        "resume": True,
    },
    "gemini_cli": {
        "native_schema": False,             # JSON via --output-format json, APEX post-parses
        "sandbox_levels": ("workspace-write",),  # --approval-mode
        "internet": True,
        "thinking": False,
        "bidirectional_stream": True,
        "mcp": False,
        "effort_levels": (),
        "resume": False,
    },
    "opencode_cli": {
        "native_schema": False,
        "sandbox_levels": ("workspace-write",),  # --yolo
        "internet": True,
        "thinking": False,
        "bidirectional_stream": True,
        "mcp": False,
        "effort_levels": (),
        "resume": False,
    },
    "metacode_cli": {
        "native_schema": False,
        "sandbox_levels": ("workspace-write",),
        "internet": True,
        "thinking": False,
        "bidirectional_stream": True,
        "mcp": False,
        "effort_levels": (),
        "resume": False,
    },
    "openai_api": {
        "native_schema": True,
        "sandbox_levels": (),               # API has no sandbox; APEX worktree floor applies
        "internet": False,
        "thinking": True,
        "bidirectional_stream": False,
        "mcp": False,
        "effort_levels": ("low", "medium", "high"),
        "resume": False,
    },
}

# resolved CLI command per vendor (mirrors v1 config.resolved_cli_command)
_CLI_COMMAND = {
    "codex_cli": "codex",
    "claude_cli": "claude",
    "gemini_cli": "gemini",
    "opencode_cli": "opencode",
    "metacode_cli": "metacode",
}


def _probe_cli_version(vendor: str, *, timeout: float = 5.0) -> str:
    cmd = _CLI_COMMAND.get(vendor)
    if not cmd or not shutil.which(cmd):
        return "unknown"
    try:
        out = subprocess.run(
            [cmd, "--version"], capture_output=True, text=True, timeout=timeout
        )
        line = (out.stdout or out.stderr or "").strip().splitlines()
        return f"{cmd}@{line[0].strip()}" if line else "unknown"
    except Exception:
        return "unknown"


def negotiate(vendor: str, model: str, version: str = "", *, probe: bool = True) -> CapabilityProfile:
    """Return the declared-wins capability profile for ``(vendor, model)``.

    ``probe=True`` best-effort captures the real CLI version for the RunManifest
    (drift detection / journal cli_version component).  Never raises — an
    unknown vendor degrades to a conservative no-capability profile."""
    declared = STATIC_CAPABILITY_TABLE.get(vendor, {
        "native_schema": False, "sandbox_levels": (), "internet": False,
        "thinking": False, "bidirectional_stream": False, "mcp": False,
        "effort_levels": (), "resume": False,
    })
    cli_version = version or (_probe_cli_version(vendor) if probe else "unknown")
    return CapabilityProfile(
        vendor=vendor,
        model=model,
        cli_version=cli_version,
        internet=bool(declared.get("internet", False)),
        native_schema=bool(declared.get("native_schema", False)),
        sandbox_levels=tuple(declared.get("sandbox_levels", ())),
        thinking=bool(declared.get("thinking", False)),
        bidirectional_stream=bool(declared.get("bidirectional_stream", False)),
        mcp=bool(declared.get("mcp", False)),
        effort_levels=tuple(declared.get("effort_levels", ())),
    )
