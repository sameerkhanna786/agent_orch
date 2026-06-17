"""Command-domain classification for host/target runtime routing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class CommandDomain(str, Enum):
    HOST_ORCHESTRATION = "host_orchestration"
    TARGET_WORKSPACE = "target_workspace"
    TARGET_EVALUATION_RUNTIME = "target_evaluation_runtime"
    ARTIFACT_SPACE = "artifact_space"
    UNKNOWN = "unknown"


READ_ONLY_OR_SETUP_COMMANDS = {
    "git",
    "rg",
    "grep",
    "find",
    "fd",
    "fdfind",
    "ls",
    "tree",
    "du",
    "diff",
    "patch",
    "shasum",
    "sha1sum",
    "sha256sum",
    "md5sum",
    "curl",
    "wget",
    "docker",
}

TARGET_EVALUATION_COMMANDS = {
    "python",
    "python3",
    "python3.10",
    "python3.11",
    "python3.12",
    "python3.13",
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
}

SHELL_COMMANDS = {"bash", "sh", "zsh", "env", "xargs", "timeout", "gtimeout"}


@dataclass(frozen=True)
class CommandDomainDecision:
    domain: CommandDomain
    command_name: str
    allowed_host_bypass: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain.value,
            "command_name": self.command_name,
            "allowed_host_bypass": bool(self.allowed_host_bypass),
            "reason": self.reason,
        }


def classify_command_domain(tokens: list[str]) -> CommandDomainDecision:
    if not tokens:
        return CommandDomainDecision(
            domain=CommandDomain.UNKNOWN,
            command_name="",
            allowed_host_bypass=False,
            reason="empty command",
        )
    command_name = Path(str(tokens[0] or "")).name
    if command_name in READ_ONLY_OR_SETUP_COMMANDS:
        return CommandDomainDecision(
            domain=CommandDomain.HOST_ORCHESTRATION,
            command_name=command_name,
            allowed_host_bypass=True,
            reason="read/setup host command",
        )
    if command_name in TARGET_EVALUATION_COMMANDS:
        return CommandDomainDecision(
            domain=CommandDomain.TARGET_EVALUATION_RUNTIME,
            command_name=command_name,
            allowed_host_bypass=False,
            reason="target evaluation command must use configured runtime",
        )
    if command_name in SHELL_COMMANDS:
        payload = " ".join(tokens[1:])
        lowered = payload.lower()
        if any(f"/{name}" in lowered for name in TARGET_EVALUATION_COMMANDS):
            return CommandDomainDecision(
                domain=CommandDomain.TARGET_EVALUATION_RUNTIME,
                command_name=command_name,
                allowed_host_bypass=False,
                reason="shell payload invokes target evaluation command",
            )
        return CommandDomainDecision(
            domain=CommandDomain.HOST_ORCHESTRATION,
            command_name=command_name,
            allowed_host_bypass=True,
            reason="shell command without target-evaluation payload",
        )
    return CommandDomainDecision(
        domain=CommandDomain.UNKNOWN,
        command_name=command_name,
        allowed_host_bypass=False,
        reason="unknown command domain",
    )
