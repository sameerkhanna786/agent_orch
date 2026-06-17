"""Benchmark target-runtime tool shims.

Benchmark authoring agents should not accidentally learn from or execute
against the host Apex environment when the task has a target runtime. This
module builds a PATH prefix containing controlled tool shims:

* source-inspection tools are delegated to the real target runtime with the
  source-of-truth workdir mounted as the working tree;
* dynamic tools run in the same configured target runtime (host env or Docker);
  and
* if no target runner exists, tools fail closed.
"""

from __future__ import annotations

import json
import os
import secrets
import shlex
import signal
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

STATIC_READ_ONLY_TOOLS: tuple[str, ...] = (
    "cat",
    "diff",
    "find",
    "grep",
    "head",
    "ls",
    "md5sum",
    "pwd",
    "rg",
    "sed",
    "sha1sum",
    "sha256sum",
    "shasum",
    "tail",
    "wc",
)

DYNAMIC_TOOL_NAMES: tuple[str, ...] = (
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
    "php",
    "phpunit",
    "dotnet",
    "swift",
)

_HOST_ENV_RUNTIME_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        "PATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
        "PIP_DISABLE_PIP_VERSION_CHECK",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "NO_COLOR",
        "TERM",
    }
)
_HOST_ENV_RUNTIME_PATH_ENV_KEYS: frozenset[str] = frozenset(
    {
        "HOME",
        "TMPDIR",
        "TMP",
        "TEMP",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
    }
)
_HOST_ENV_RUNTIME_ENV_PREFIX_ALLOWLIST: tuple[str, ...] = ("LC_",)
_DEFAULT_TARGET_TOOL_OUTPUT_CAPTURE_MAX_CHARS = 131_072


@dataclass(frozen=True)
class TargetRuntimeSpec:
    """Description of the runtime agent-invoked dynamic tools should use."""

    kind: str
    env: dict[str, str] = field(default_factory=dict)
    docker_image: str = ""
    docker_workdir: str = "/workspace"
    docker_platform: str = ""
    docker_network: str = "host"
    docker_env: dict[str, str] = field(default_factory=dict)
    docker_mounts: list[dict[str, str]] = field(default_factory=list)
    docker_container_name: str = ""
    docker_host_workdir_root: str = ""
    docker_container_workdir_root: str = "/workspace"
    docker_bin: str = "docker"
    docker_host_env: dict[str, str] = field(default_factory=dict)
    docker_user: str = ""
    docker_root_setup_script: str = ""
    docker_control_shell: str = ""
    description: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "env": dict(self.env),
            "docker_image": self.docker_image,
            "docker_workdir": self.docker_workdir,
            "docker_platform": self.docker_platform,
            "docker_network": self.docker_network,
            "docker_env": dict(self.docker_env),
            "docker_mounts": [dict(item) for item in self.docker_mounts],
            "docker_container_name": self.docker_container_name,
            "docker_host_workdir_root": self.docker_host_workdir_root,
            "docker_container_workdir_root": self.docker_container_workdir_root,
            "docker_bin": self.docker_bin,
            "docker_host_env": dict(self.docker_host_env),
            "docker_user": self.docker_user,
            "docker_root_setup_script": self.docker_root_setup_script,
            "docker_control_shell": self.docker_control_shell,
            "description": self.description,
        }


def host_env_runtime(
    env: dict[str, str],
    *,
    description: str = "host_target_env",
) -> TargetRuntimeSpec:
    runtime_env = {str(key): str(value) for key, value in dict(env or {}).items()}
    runtime_env = _sanitize_host_runtime_env(runtime_env)
    return TargetRuntimeSpec(
        kind="host_env",
        env=runtime_env,
        description=description,
    )


def docker_image_runtime(
    *,
    image: str,
    docker_workdir: str,
    docker_platform: str = "",
    docker_network: str = "host",
    docker_bin: str = "docker",
    docker_user: str = "",
    docker_host_env: dict[str, str] | None = None,
    docker_env: dict[str, str] | None = None,
    docker_mounts: list[dict[str, str]] | None = None,
    docker_root_setup_script: str = "",
    description: str = "docker_target_env",
) -> TargetRuntimeSpec:
    return TargetRuntimeSpec(
        kind="docker_image",
        docker_image=str(image or ""),
        docker_workdir=str(docker_workdir or "/workspace"),
        docker_platform=str(docker_platform or ""),
        docker_network=str(docker_network or "host"),
        docker_bin=str(docker_bin or "docker"),
        docker_user=str(docker_user or ""),
        docker_host_env={
            str(key): str(value) for key, value in dict(docker_host_env or {}).items()
        },
        docker_env={str(key): str(value) for key, value in dict(docker_env or {}).items()},
        docker_mounts=[
            {str(k): str(v) for k, v in dict(item).items()}
            for item in list(docker_mounts or [])
            if isinstance(item, dict)
        ],
        docker_root_setup_script=str(docker_root_setup_script or ""),
        description=description,
    )


def docker_exec_runtime(
    *,
    container_name: str,
    host_workdir_root: str | Path,
    container_workdir_root: str,
    docker_bin: str = "docker",
    docker_env: dict[str, str] | None = None,
    docker_host_env: dict[str, str] | None = None,
    docker_user: str = "",
    docker_control_shell: str = "",
    description: str = "docker_exec_target_env",
) -> TargetRuntimeSpec:
    return TargetRuntimeSpec(
        kind="docker_exec",
        docker_container_name=str(container_name or ""),
        docker_host_workdir_root=str(Path(host_workdir_root).expanduser().resolve(strict=False)),
        docker_container_workdir_root=str(container_workdir_root or "/workspace"),
        docker_bin=str(docker_bin or "docker"),
        docker_env={str(key): str(value) for key, value in dict(docker_env or {}).items()},
        docker_host_env={
            str(key): str(value) for key, value in dict(docker_host_env or {}).items()
        },
        docker_user=str(docker_user or ""),
        docker_control_shell=str(docker_control_shell or ""),
        description=description,
    )


def fail_closed_runtime(*, description: str = "no_target_command_runner") -> TargetRuntimeSpec:
    return TargetRuntimeSpec(kind="fail_closed", description=description)


def target_tool_env_overrides(
    *,
    workdir: Path,
    output_dir: Path,
    timeout_seconds: float,
    agent_command_timeout_seconds: float | None = None,
    output_capture_max_chars: int | None = None,
    runtime: TargetRuntimeSpec | None,
    label: str,
    blocked_command_patterns: list[dict[str, str]] | None = None,
    command_policy_blocks: bool | None = None,
    git_history_policy: str = "blocked",
    source_network_policy: str = "unspecified",
    filesystem_boundary_policy: str = "policy_enforced",
) -> tuple[dict[str, str], dict[str, Any]]:
    """Create target-runtime tool shims and return env overrides for agents."""

    if runtime is None:
        runtime = fail_closed_runtime()
    workdir = Path(workdir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser()
    shim_dir = output_dir / f"{_safe_label(label)}_target_tool_shims"
    shim_dir.mkdir(parents=True, exist_ok=True)

    host_path = _agent_visible_host_path(os.environ.get("PATH", ""))
    context_path = shim_dir / "context.json"
    outer_timeout = max(1, int(timeout_seconds or 1))
    agent_timeout = max(
        1,
        int(
            agent_command_timeout_seconds
            if agent_command_timeout_seconds is not None
            else outer_timeout
        ),
    )
    output_max_chars = max(
        0,
        int(
            output_capture_max_chars
            if output_capture_max_chars is not None
            else _DEFAULT_TARGET_TOOL_OUTPUT_CAPTURE_MAX_CHARS
        ),
    )
    normalized_blocked_command_patterns = _normalize_blocked_command_patterns(
        blocked_command_patterns
    )
    command_policy_blocks_enabled = (
        bool(normalized_blocked_command_patterns)
        if command_policy_blocks is None
        else bool(command_policy_blocks)
    )
    if not command_policy_blocks_enabled:
        normalized_blocked_command_patterns = []
    context_payload: dict[str, Any] = {
        "label": str(label or "benchmark"),
        "status": "configured",
        "mode": runtime.kind,
        "generated_by_apex": True,
        "artifact_role": "target_runtime_tool_shim",
        "runtime": runtime.to_payload(),
        "workdir": str(workdir),
        "shim_dir": str(shim_dir),
        "context_path": str(context_path),
        "timeout_seconds": outer_timeout,
        "agent_command_timeout_seconds": agent_timeout,
        "output_capture_max_chars": output_max_chars,
        "git_history_policy": str(git_history_policy or "blocked"),
        "source_network_policy": str(source_network_policy or "unspecified"),
        "filesystem_boundary_policy": str(filesystem_boundary_policy or "policy_enforced"),
        "host_path": host_path,
        "command_policy_blocks": command_policy_blocks_enabled,
        "blocked_command_patterns": normalized_blocked_command_patterns,
    }
    context_path.write_text(json.dumps(context_payload, indent=2) + "\n", encoding="utf-8")

    runner_path = shim_dir / "apex_target_tool.py"
    runner_path.write_text(_runner_source(), encoding="utf-8")
    runner_path.chmod(0o755)
    marker_path = shim_dir / ".apex_generated_artifact.json"
    marker_path.write_text(
        json.dumps(
            {
                "generated_by_apex": True,
                "artifact_role": "target_runtime_tool_shim",
                "context_path": str(context_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for tool_name in (*STATIC_READ_ONLY_TOOLS, *DYNAMIC_TOOL_NAMES):
        target = shim_dir / tool_name
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            target.symlink_to(runner_path.name)
        except OSError:
            shutil.copy2(runner_path, target)
        target.chmod(0o755)

    env = {
        "PATH": str(shim_dir) + (os.pathsep + host_path if host_path else ""),
        "APEX_TARGET_TOOL_CONTEXT": str(context_path),
        "APEX_HOST_DYNAMIC_TOOLS": "disabled",
        "APEX_HOST_PATH": host_path,
    }
    return env, {
        "status": "configured",
        "mode": runtime.kind,
        "generated_by_apex": True,
        "artifact_role": "target_runtime_tool_shim",
        "shim_dir": str(shim_dir),
        "workdir": str(workdir),
        "timeout_seconds": outer_timeout,
        "agent_command_timeout_seconds": agent_timeout,
        "output_capture_max_chars": output_max_chars,
        "git_history_policy": str(git_history_policy or "blocked"),
        "source_network_policy": str(source_network_policy or "unspecified"),
        "filesystem_boundary_policy": str(filesystem_boundary_policy or "policy_enforced"),
        "runtime_description": runtime.description,
        "source_tools": list(STATIC_READ_ONLY_TOOLS),
        "dynamic_tools": list(DYNAMIC_TOOL_NAMES),
        "command_policy_blocks": command_policy_blocks_enabled,
        "blocked_command_patterns": list(context_payload["blocked_command_patterns"]),
    }


def apply_target_tool_env_to_apex_config(config: Any, env: dict[str, str]) -> None:
    """Apply target-runtime environment authority to CLI and ACI agents."""

    if not env:
        return
    for llm_config in list(getattr(config, "llm_configs", []) or []):
        merged = dict(getattr(llm_config, "cli_env_overrides", {}) or {})
        merged.update(env)
        llm_config.cli_env_overrides = merged
    aci_config = getattr(config, "aci", None)
    if aci_config is not None:
        merged_aci_env = dict(getattr(aci_config, "runtime_env_overrides", {}) or {})
        merged_aci_env.update(env)
        aci_config.runtime_env_overrides = merged_aci_env


def _safe_label(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value or ""))
    return text.strip("._-") or "benchmark"


def _normalize_blocked_command_patterns(
    patterns: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for raw_rule in list(patterns or []):
        if not isinstance(raw_rule, dict):
            continue
        pattern = str(raw_rule.get("pattern") or "").strip()
        if not pattern:
            continue
        message = str(raw_rule.get("message") or "command is blocked by target-runtime policy")
        tools = raw_rule.get("tools")
        normalized_tools: list[str] = []
        if isinstance(tools, (list, tuple)):
            normalized_tools = [str(tool).strip() for tool in tools if str(tool).strip()]
        elif isinstance(tools, str) and tools.strip():
            normalized_tools = [tools.strip()]
        normalized.append({"pattern": pattern, "message": message, "tools": normalized_tools})
    return normalized


def _agent_visible_host_path(host_path: str) -> str:
    """Return host PATH fallback entries safe to expose behind target shims."""

    entries: list[str] = []
    for raw_entry in str(host_path or "").split(os.pathsep):
        entry = raw_entry.strip()
        if (
            not entry
            or _path_entry_is_python_env_bin(entry)
            or _path_entry_is_host_local_tooling(entry)
        ):
            continue
        entries.append(entry)
    return os.pathsep.join(entries)


def _sanitize_host_runtime_env(env: dict[str, str]) -> dict[str, str]:
    """Keep only target-runtime execution variables in shim metadata."""

    runtime_roots = _declared_runtime_roots(env)
    cleaned: dict[str, str] = {}
    for raw_key, raw_value in dict(env or {}).items():
        key = str(raw_key)
        value = str(raw_value)
        if key in _HOST_ENV_RUNTIME_ENV_ALLOWLIST or key.startswith(
            _HOST_ENV_RUNTIME_ENV_PREFIX_ALLOWLIST
        ):
            cleaned[key] = value
            continue
        if key in _HOST_ENV_RUNTIME_PATH_ENV_KEYS and _path_value_is_under_any_root(
            value,
            runtime_roots,
        ):
            cleaned[key] = value
    path = str(env.get("PATH") or "")
    if path:
        cleaned["PATH"] = _agent_visible_runtime_path(path, cleaned)
    return cleaned


def _declared_runtime_roots(env: dict[str, str]) -> tuple[Path, ...]:
    roots: list[Path] = []
    for key in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        value = str(env.get(key) or "").strip()
        if not value:
            continue
        try:
            path = Path(value).expanduser().resolve(strict=False)
        except OSError:
            continue
        roots.append(path)
        roots.append(path.parent)
    return tuple(roots)


def _path_value_is_under_any_root(value: str, roots: tuple[Path, ...]) -> bool:
    if not roots:
        return False
    try:
        path = Path(value).expanduser().resolve(strict=False)
    except OSError:
        return False
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _agent_visible_runtime_path(host_path: str, env: dict[str, str]) -> str:
    """Return runtime PATH entries that do not advertise host-local tooling."""

    declared_env_bins = _declared_runtime_env_bins(env)
    entries: list[str] = []
    for raw_entry in str(host_path or "").split(os.pathsep):
        entry = raw_entry.strip()
        if not entry:
            continue
        if entry in declared_env_bins:
            entries.append(entry)
            continue
        if _path_entry_is_python_env_bin(entry) or _path_entry_is_host_local_tooling(entry):
            continue
        if _path_entry_is_system_bin(entry):
            entries.append(entry)
            continue
        entries.append(entry)
    return os.pathsep.join(entries)


def _declared_runtime_env_bins(env: dict[str, str]) -> set[str]:
    bins: set[str] = set()
    for key in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        root = str(env.get(key) or "").strip()
        if not root:
            continue
        try:
            root_path = Path(root).expanduser()
        except OSError:
            continue
        for dirname in ("bin", "Scripts"):
            bins.add(str(root_path / dirname))
    return bins


def _path_entry_is_python_env_bin(path_entry: str) -> bool:
    try:
        path = Path(path_entry).expanduser()
    except OSError:
        return False
    if path.name.lower() not in {"bin", "scripts"}:
        return False
    env_root = path.parent
    if env_root.name in {".venv", "venv", ".env", "env"}:
        return True
    try:
        if (env_root / "pyvenv.cfg").exists():
            return True
        if (env_root / "conda-meta").is_dir():
            return True
    except OSError:
        return False
    return False


def _path_entry_is_host_local_tooling(path_entry: str) -> bool:
    try:
        path = Path(path_entry).expanduser()
    except OSError:
        return True
    text = str(path)
    if text in {"/usr/local/bin", "/usr/local/sbin", "/opt/homebrew/bin", "/opt/homebrew/sbin"}:
        return True
    host_tool_fragments = (
        f"{os.sep}.codex{os.sep}",
        f"{os.sep}.claude{os.sep}",
        f"{os.sep}claude_code{os.sep}",
        f"{os.sep}codex.system{os.sep}",
        f"{os.sep}.local{os.sep}bin",
        f"{os.sep}opt{os.sep}facebook{os.sep}",
        f"{os.sep}opt{os.sep}fbcode{os.sep}",
    )
    return any(fragment in text for fragment in host_tool_fragments)


def _path_entry_is_system_bin(path_entry: str) -> bool:
    try:
        path = Path(path_entry).expanduser()
    except OSError:
        return False
    text = str(path)
    return text in {
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
        "/System/Cryptexes/App/usr/bin",
        "/Library/Apple/usr/bin",
    }


def _docker_exec_workdir_for_raw(context: dict[str, Any], raw: str) -> str:
    runtime = dict(context.get("runtime") or {})
    raw = str(raw or ".")
    host_root = str(runtime.get("docker_host_workdir_root") or "").rstrip("/")
    container_root = str(runtime.get("docker_container_workdir_root") or "/workspace").rstrip("/")
    if host_root and (raw == host_root or raw.startswith(host_root + "/")):
        suffix = raw[len(host_root) :].lstrip("/")
        return container_root + (("/" + suffix) if suffix else "")
    return raw


def _docker_image_workdir_for_raw(context: dict[str, Any], raw: str) -> str:
    runtime = dict(context.get("runtime") or {})
    raw = str(raw or ".")
    host_workdir = str(context.get("workdir") or "").rstrip("/")
    container_root = str(
        runtime.get("docker_workdir")
        or runtime.get("docker_container_workdir_root")
        or "/workspace"
    ).rstrip("/")
    if host_workdir and (raw == host_workdir or raw.startswith(host_workdir + "/")):
        suffix = raw[len(host_workdir) :].lstrip("/")
        return container_root + (("/" + suffix) if suffix else "")
    return raw


def _host_path_for_container_raw(
    *,
    raw: str,
    host_root: str,
    container_root: str,
) -> str:
    raw = str(raw or "")
    host_root = str(host_root or "").rstrip("/")
    container_root = str(container_root or "").rstrip("/")
    if (
        host_root
        and container_root
        and (raw == container_root or raw.startswith(container_root + "/"))
    ):
        suffix = raw[len(container_root) :].lstrip("/")
        return host_root + (("/" + suffix) if suffix else "")
    return raw


def _docker_exec_host_path_for_raw(context: dict[str, Any], raw: str) -> str:
    runtime = dict(context.get("runtime") or {})
    return _host_path_for_container_raw(
        raw=str(raw or ""),
        host_root=str(runtime.get("docker_host_workdir_root") or ""),
        container_root=str(runtime.get("docker_container_workdir_root") or "/workspace"),
    )


def _docker_image_host_path_for_raw(context: dict[str, Any], raw: str) -> str:
    runtime = dict(context.get("runtime") or {})
    container_root = str(
        runtime.get("docker_workdir")
        or runtime.get("docker_container_workdir_root")
        or "/workspace"
    )
    return _host_path_for_container_raw(
        raw=str(raw or ""),
        host_root=str(context.get("workdir") or ""),
        container_root=container_root,
    )


def _docker_exec_control_shell(runtime: dict[str, Any]) -> str:
    return str(runtime.get("docker_control_shell") or "").strip() or "sh"


def _docker_exec_control_command(
    *,
    docker_bin: str,
    container_name: str,
    shell: str,
    script: str,
    argv0: str,
    args: list[str],
    exec_args: list[str] | None = None,
) -> list[str]:
    return [
        docker_bin,
        "exec",
        *(exec_args or []),
        container_name,
        shell,
        "-c",
        script,
        argv0,
        *args,
    ]


def _run_docker_exec_control_script(
    *,
    runtime: dict[str, Any],
    docker_bin: str,
    container_name: str,
    script: str,
    argv0: str,
    args: list[str],
    env: dict[str, str],
    timeout: float,
    exec_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    control_shell = _docker_exec_control_shell(runtime)
    explicit_control_shell = bool(str(runtime.get("docker_control_shell") or "").strip())
    first = subprocess.run(
        _docker_exec_control_command(
            docker_bin=docker_bin,
            container_name=container_name,
            shell=control_shell,
            script=script,
            argv0=argv0,
            args=args,
            exec_args=exec_args,
        ),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )
    if explicit_control_shell or first.returncode not in {113, 126, 127}:
        return first
    fallback_shell = "/bin/sh.apex-real"
    if control_shell == fallback_shell:
        return first
    fallback = subprocess.run(
        _docker_exec_control_command(
            docker_bin=docker_bin,
            container_name=container_name,
            shell=fallback_shell,
            script=script,
            argv0=argv0,
            args=args,
            exec_args=exec_args,
        ),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )
    if fallback.returncode in {126, 127} and not (fallback.stdout or "").strip():
        return first
    return fallback


def cleanup_target_runtime_processes(
    env: dict[str, Any],
    *,
    signum: int = signal.SIGTERM,
) -> set[int]:
    """Kill container-side target-runtime processes for a cancelled worktree."""

    context_path = str((env or {}).get("APEX_TARGET_TOOL_CONTEXT") or "").strip()
    if not context_path:
        return set()
    try:
        context = json.loads(Path(context_path).read_text(encoding="utf-8"))
    except Exception:
        return set()
    runtime = dict(context.get("runtime") or {})
    if str(runtime.get("kind") or context.get("mode") or "") != "docker_exec":
        return set()
    container_name = str(runtime.get("docker_container_name") or "").strip()
    if not container_name:
        return set()
    raw_workdir = str((env or {}).get("APEX_TARGET_TOOL_WORKDIR") or context.get("workdir") or "")
    container_workdir = _docker_exec_workdir_for_raw(context, raw_workdir).rstrip("/")
    if not container_workdir or container_workdir in {"/", "/workspace"}:
        return set()
    invocation_id = str((env or {}).get("APEX_CLI_INVOCATION_ID") or "").strip()
    docker_bin = str(runtime.get("docker_bin") or "docker")
    docker_client_env = dict(os.environ)
    docker_client_env.update(
        {str(k): str(v) for k, v in dict(runtime.get("docker_host_env") or {}).items()}
    )
    script = r"""
target=$1
sig=$2
invocation=$3
self_pid=$$
matches=" "
children_file="${TMPDIR:-/tmp}/apex-target-cleanup-$$.pids"
trap 'rm -f "$children_file"' EXIT
: > "$children_file"
target_parent=${target%/*}
target_name=${target##*/}
runtime_prefix_dash=""
runtime_prefix_dir=""
if [ -n "$target_parent" ] && [ "$target_parent" != "$target" ] && [ -n "$target_name" ]; then
  runtime_prefix_dash="$target_parent/.apex_agent_runtime/$target_name-"
  runtime_prefix_dir="$target_parent/.apex_agent_runtime/$target_name/"
fi
for proc in /proc/[0-9]*; do
  pid=${proc##*/}
  [ "$pid" = "1" ] && continue
  [ "$pid" = "$self_pid" ] && continue
  ppid=$(grep '^PPid:' "$proc/status" 2>/dev/null | { read _ value; printf '%s' "${value:-0}"; } || true)
  case "$ppid" in *[!0-9]*|"") ppid=0 ;; esac
  printf '%s %s\n' "$pid" "$ppid" >> "$children_file"
  cwd=$(readlink "$proc/cwd" 2>/dev/null || true)
  cwd=${cwd% (deleted)}
  match=0
  cmdline=$(cat "$proc/cmdline" 2>/dev/null | tr '\000' ' ' || true)
  environ=$(cat "$proc/environ" 2>/dev/null | tr '\000' '\n' || true)
  if [ -n "$invocation" ]; then
    case "$environ" in
      *"APEX_CLI_INVOCATION_ID=$invocation"*) match=1 ;;
    esac
  fi
  if [ "$match" != "1" ]; then
    if [ "$cwd" = "$target" ]; then
      match=1
    fi
    case "$cwd" in
      "$target"/*) match=1 ;;
    esac
    case "$cmdline" in
      *"$target"*) match=1 ;;
    esac
    if [ -n "$runtime_prefix_dash" ]; then
      case "$cmdline" in
        *"$runtime_prefix_dash"*|*"$runtime_prefix_dir"*) match=1 ;;
      esac
      case "$environ" in
        *"$runtime_prefix_dash"*|*"$runtime_prefix_dir"*) match=1 ;;
      esac
    fi
    case "$environ" in
      *"APEX_TARGET_TOOL_WORKDIR=$target"*) match=1 ;;
    esac
  fi
  if [ "$match" = "1" ]; then
    matches="$matches$pid "
  fi
done
changed=1
while [ "$changed" = "1" ]; do
  changed=0
  while read child parent; do
    [ -n "$child" ] || continue
    case "$matches" in
      *" $child "*) continue ;;
    esac
    case "$matches" in
      *" $parent "*)
        matches="$matches$child "
        changed=1
        ;;
    esac
  done < "$children_file"
done
for pid in $matches; do
  [ "$pid" = "$self_pid" ] && continue
  kill -"$sig" "$pid" 2>/dev/null || true
  printf '%s\n' "$pid"
done
"""
    try:
        completed = _run_docker_exec_control_script(
            runtime=runtime,
            docker_bin=docker_bin,
            container_name=container_name,
            script=script,
            argv0="apex-target-cleanup",
            args=[container_workdir, str(int(signum)), invocation_id],
            env=docker_client_env,
            timeout=10,
        )
    except Exception:
        return set()
    cleaned: set[int] = set()
    for line in (completed.stdout or "").splitlines():
        line = line.strip()
        if line.isdigit():
            cleaned.add(int(line))
    return cleaned


def read_target_runtime_file_text(
    env: dict[str, Any],
    path: str | Path,
    *,
    max_bytes: int = 8_000_000,
) -> Optional[str]:
    """Read a workdir-scoped file from a configured target runtime."""

    context_path = str((env or {}).get("APEX_TARGET_TOOL_CONTEXT") or "").strip()
    if not context_path:
        return None
    try:
        context = json.loads(Path(context_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    runtime = dict(context.get("runtime") or {})
    if str(runtime.get("kind") or context.get("mode") or "") != "docker_exec":
        return None
    container_name = str(runtime.get("docker_container_name") or "").strip()
    if not container_name:
        return None
    raw_workdir = str((env or {}).get("APEX_TARGET_TOOL_WORKDIR") or context.get("workdir") or "")
    container_workdir = _docker_exec_workdir_for_raw(context, raw_workdir).rstrip("/")
    if not container_workdir or container_workdir in {"/", "/workspace"}:
        return None
    raw_path = str(path or "").strip()
    if not raw_path:
        return None
    container_path = _docker_exec_workdir_for_raw(context, raw_path)
    if not container_path.startswith("/"):
        container_path = container_workdir.rstrip("/") + "/" + container_path.lstrip("/")
    docker_bin = str(runtime.get("docker_bin") or "docker")
    docker_client_env = dict(os.environ)
    docker_client_env.update(
        {str(k): str(v) for k, v in dict(runtime.get("docker_host_env") or {}).items()}
    )
    script = r"""
path=$1
limit=$2
[ -f "$path" ] || exit 3
head -c "$limit" "$path"
"""
    try:
        completed = _run_docker_exec_control_script(
            runtime=runtime,
            docker_bin=docker_bin,
            container_name=container_name,
            script=script,
            argv0="apex-target-read-file",
            args=[container_path, str(max(1, int(max_bytes)))],
            env=docker_client_env,
            timeout=10,
        )
    except Exception:
        return None
    if int(completed.returncode or 0) != 0:
        return None
    return completed.stdout


def target_runtime_path_for_file(env: dict[str, Any], path: str | Path) -> str:
    """Return the path a target-runtime command should use for a host file."""

    raw_path = str(path or "").strip()
    if not raw_path:
        return raw_path
    context_path = str((env or {}).get("APEX_TARGET_TOOL_CONTEXT") or "").strip()
    if not context_path:
        return raw_path
    try:
        context = json.loads(Path(context_path).read_text(encoding="utf-8"))
    except Exception:
        return raw_path
    runtime = dict(context.get("runtime") or {})
    mode = str(runtime.get("kind") or context.get("mode") or "")
    if mode == "docker_exec":
        mapped = _docker_exec_workdir_for_raw(context, raw_path)
        return mapped or raw_path
    if mode == "docker_image":
        mapped = _docker_image_workdir_for_raw(context, raw_path)
        return mapped or raw_path
    return raw_path


def target_runtime_host_path_for_file(env: dict[str, Any], path: str | Path) -> str:
    """Return the host path that backs a target-runtime visible path."""

    raw_path = str(path or "").strip()
    if not raw_path:
        return raw_path
    context_path = str((env or {}).get("APEX_TARGET_TOOL_CONTEXT") or "").strip()
    if not context_path:
        return raw_path
    try:
        context = json.loads(Path(context_path).read_text(encoding="utf-8"))
    except Exception:
        return raw_path
    runtime = dict(context.get("runtime") or {})
    mode = str(runtime.get("kind") or context.get("mode") or "")
    if mode == "docker_exec":
        mapped = _docker_exec_host_path_for_raw(context, raw_path)
        return mapped or raw_path
    if mode == "docker_image":
        mapped = _docker_image_host_path_for_raw(context, raw_path)
        return mapped or raw_path
    return raw_path


def target_runtime_process_activity(env: dict[str, Any]) -> dict[str, Any]:
    """Return workdir-scoped target-runtime process liveness for watchdogs."""

    context_path = str((env or {}).get("APEX_TARGET_TOOL_CONTEXT") or "").strip()
    if not context_path:
        return {}
    try:
        context = json.loads(Path(context_path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    runtime = dict(context.get("runtime") or {})
    if str(runtime.get("kind") or context.get("mode") or "") != "docker_exec":
        return {}
    container_name = str(runtime.get("docker_container_name") or "").strip()
    if not container_name:
        return {}
    raw_workdir = str((env or {}).get("APEX_TARGET_TOOL_WORKDIR") or context.get("workdir") or "")
    container_workdir = _docker_exec_workdir_for_raw(context, raw_workdir).rstrip("/")
    if not container_workdir or container_workdir in {"/", "/workspace"}:
        return {}
    invocation_id = str((env or {}).get("APEX_CLI_INVOCATION_ID") or "").strip()
    docker_bin = str(runtime.get("docker_bin") or "docker")
    docker_client_env = dict(os.environ)
    docker_client_env.update(
        {str(k): str(v) for k, v in dict(runtime.get("docker_host_env") or {}).items()}
    )
    sampler_nonce = secrets.token_hex(16)
    script = r"""
target=$1
invocation=$2
sampler_nonce=$3
self_pid=$$
matches=" "
records_file="${TMPDIR:-/tmp}/apex-target-activity-$$.pids"
details_file="${TMPDIR:-/tmp}/apex-target-activity-$$.details"
trap 'rm -f "$records_file" "$details_file"' EXIT
: > "$records_file"
: > "$details_file"
_apex_target_runtime_real_tool() {
  _apex_tool_name=$1
  for _apex_tool_path in \
    "/usr/bin/${_apex_tool_name}.apex-real" \
    "/bin/${_apex_tool_name}.apex-real" \
    "/usr/local/bin/${_apex_tool_name}.apex-real" \
    "/usr/sbin/${_apex_tool_name}.apex-real" \
    "/sbin/${_apex_tool_name}.apex-real"; do
    if [ -x "$_apex_tool_path" ]; then
      printf '%s\n' "$_apex_tool_path"
      return 0
    fi
  done
  command -v "$_apex_tool_name" 2>/dev/null || printf '%s\n' "$_apex_tool_name"
}
apex_cat=$(_apex_target_runtime_real_tool cat)
apex_grep=$(_apex_target_runtime_real_tool grep)
target_parent=${target%/*}
target_name=${target##*/}
runtime_prefix_dash=""
runtime_prefix_dir=""
if [ -n "$target_parent" ] && [ "$target_parent" != "$target" ] && [ -n "$target_name" ]; then
  runtime_prefix_dash="$target_parent/.apex_agent_runtime/$target_name-"
  runtime_prefix_dir="$target_parent/.apex_agent_runtime/$target_name/"
fi
for proc in /proc/[0-9]*; do
  pid=${proc##*/}
  [ "$pid" = "1" ] && continue
  [ "$pid" = "$self_pid" ] && continue
  ppid=$("$apex_grep" '^PPid:' "$proc/status" 2>/dev/null | { read _ value; printf '%s' "${value:-0}"; } || true)
  case "$ppid" in *[!0-9]*|"") ppid=0 ;; esac
  cwd=$(readlink "$proc/cwd" 2>/dev/null || true)
  cwd=${cwd% (deleted)}
  match=0
  cmdargv=$("$apex_cat" "$proc/cmdline" 2>/dev/null | tr '\000\011\012' '\037  ' || true)
  cmdline=$(printf '%s' "$cmdargv" | tr '\037' ' ')
  environ=$("$apex_cat" "$proc/environ" 2>/dev/null | tr '\000' '\n' || true)
  if [ -n "$sampler_nonce" ]; then
    case "$environ" in
      *"APEX_TARGET_ACTIVITY_SAMPLER_NONCE=$sampler_nonce"*)
        case "$cmdline" in
          *"apex-target-activity"*|*"records_file="*"details_file="*"/proc/[0-9]*"*) continue ;;
        esac
        ;;
    esac
  fi
  if [ -n "$invocation" ]; then
    case "$environ" in
      *"APEX_CLI_INVOCATION_ID=$invocation"*) match=1 ;;
    esac
  fi
  if [ "$match" != "1" ]; then
    if [ "$cwd" = "$target" ]; then
      match=1
    fi
    case "$cwd" in
      "$target"/*) match=1 ;;
    esac
    case "$cmdline" in
      *"$target"*) match=1 ;;
    esac
    if [ -n "$runtime_prefix_dash" ]; then
      case "$cmdline" in
        *"$runtime_prefix_dash"*|*"$runtime_prefix_dir"*) match=1 ;;
      esac
      case "$environ" in
        *"$runtime_prefix_dash"*|*"$runtime_prefix_dir"*) match=1 ;;
      esac
    fi
    case "$environ" in
      *"APEX_TARGET_TOOL_WORKDIR=$target"*) match=1 ;;
    esac
  fi
  stat=$("$apex_cat" "$proc/stat" 2>/dev/null || true)
  rest=${stat#*) }
  set -- $rest
  state=${1:-}
  [ "$state" = "Z" ] && continue
  utime=${12:-0}
  stime=${13:-0}
  case "$utime$stime" in
    *[!0-9]*|"") ticks=0 ;;
    *) ticks=$((utime + stime)) ;;
  esac
  printf '%s %s %s\n' "$pid" "$ppid" "$ticks" >> "$records_file"
  safe_cwd=$(printf '%s' "$cwd" | tr '\t\n' '  ')
  printf '%s\t%s\t%s\t%s\n' "$pid" "$safe_cwd" "$cmdline" "$cmdargv" >> "$details_file"
  if [ "$match" = "1" ]; then
    matches="$matches$pid "
  fi
done
changed=1
while [ "$changed" = "1" ]; do
  changed=0
  while read child parent ticks; do
    [ -n "$child" ] || continue
    case "$matches" in
      *" $child "*) continue ;;
    esac
    case "$matches" in
      *" $parent "*)
        matches="$matches$child "
        changed=1
        ;;
    esac
  done < "$records_file"
done
while read pid ppid ticks; do
  case "$matches" in
    *" $pid "*)
      found_cwd=""
      found_cmd=""
      found_argv=""
      while IFS='	' read dpid dcwd dcmd dargv; do
        [ "$dpid" = "$pid" ] || continue
        found_cwd=$dcwd
        found_cmd=$dcmd
        found_argv=$dargv
        break
      done < "$details_file"
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$pid" "$ppid" "$ticks" "$found_cwd" "$found_cmd" "$found_argv"
      ;;
  esac
done < "$records_file"
"""
    try:
        completed = _run_docker_exec_control_script(
            runtime=runtime,
            docker_bin=docker_bin,
            container_name=container_name,
            script=script,
            argv0="apex-target-activity",
            args=[container_workdir, invocation_id, sampler_nonce],
            env=docker_client_env,
            timeout=10,
            exec_args=["-e", f"APEX_TARGET_ACTIVITY_SAMPLER_NONCE={sampler_nonce}"],
        )
    except Exception:
        return {}
    pids: list[int] = []
    cpu_ticks = 0.0
    process_entries: dict[int, dict[str, Any]] = {}
    for line in (completed.stdout or "").splitlines():
        parts = line.rstrip("\n").split("\t", 5)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            ticks = float(parts[2])
        except ValueError:
            continue
        cwd = parts[3] if len(parts) > 3 else ""
        command = parts[4] if len(parts) > 4 else ""
        argv_text = parts[5] if len(parts) > 5 else ""
        argv = [part for part in argv_text.split("\x1f") if part] if argv_text else []
        if not argv and "\x1f" in command:
            # qemu/docker ps rows can shift cmdargv into the command slot; keep
            # argv populated so shell/git policy sees the real payload.
            argv = [part for part in command.split("\x1f") if part]
            command = " ".join(argv)
        if not argv and "\x1f" in cwd:
            argv = [part for part in cwd.split("\x1f") if part]
            command = command or " ".join(argv)
            cwd = ""
        if cwd and _target_runtime_activity_cwd_looks_like_command(cwd):
            if not argv:
                try:
                    argv = shlex.split(cwd, posix=True)
                except ValueError:
                    argv = cwd.split()
            command = command or " ".join(argv) or cwd
            cwd = ""
        if _target_runtime_activity_entry_is_sampler(
            command=command,
            argv=argv,
            sampler_nonce=sampler_nonce,
        ) or _target_runtime_activity_entry_is_control_helper(
            command=command,
            argv=argv,
        ):
            continue
        pids.append(pid)
        cpu_ticks += max(0.0, ticks)
        process_entries[pid] = {
            "pid": pid,
            "ppid": ppid,
            "cpu_seconds": ticks,
            "cwd": cwd,
            "command": command,
            "argv": argv,
        }
    policy_violations = _target_runtime_policy_violations(
        context=context,
        workdir=container_workdir,
        invocation_id=invocation_id,
    )
    policy_payload = {
        "git_history_policy": str(context.get("git_history_policy") or "blocked"),
        "source_network_policy": str(context.get("source_network_policy") or "unspecified"),
        "filesystem_boundary_policy": str(
            context.get("filesystem_boundary_policy") or "policy_enforced"
        ),
    }
    if not pids and not policy_violations:
        return {"process_count": 0, "pids": [], "cpu_ticks": 0.0, **policy_payload}
    activity = {
        "process_count": len(pids),
        "pids": sorted(pids),
        "cpu_ticks": cpu_ticks,
        "workdir": container_workdir,
        "process_entries": process_entries,
        **policy_payload,
    }
    if policy_violations:
        activity["policy_violations"] = policy_violations
    return activity


def _target_runtime_activity_cwd_looks_like_command(cwd: str) -> bool:
    raw = str(cwd or "").strip()
    if not raw:
        return False
    try:
        tokens = shlex.split(raw, posix=True)
    except ValueError:
        tokens = raw.split()
    if len(tokens) < 2:
        return False
    command_name = Path(tokens[0]).name
    if command_name.startswith("qemu-"):
        return True
    tool_names = set(STATIC_READ_ONLY_TOOLS) | set(DYNAMIC_TOOL_NAMES)
    return raw.startswith(("/bin/", "/usr/bin/", "/usr/local/bin/", "/opt/", "/testbed/")) and (
        command_name in tool_names
    )


def _target_runtime_activity_entry_is_sampler(
    *,
    command: str,
    argv: list[str],
    sampler_nonce: str,
) -> bool:
    tokens = [str(token or "") for token in argv]
    text = "\n".join(tokens + [str(command or "")])
    try:
        marker_index = tokens.index("apex-target-activity")
    except ValueError:
        return False
    if marker_index + 3 >= len(tokens):
        return False
    workdir_token = tokens[marker_index + 1]
    invocation_token = tokens[marker_index + 2]
    nonce_token = tokens[marker_index + 3]
    if not workdir_token.startswith("/workspace/"):
        return False
    if not invocation_token or any(char.isspace() or char == "/" for char in invocation_token):
        return False
    if len(nonce_token) != 32 or any(char not in "0123456789abcdef" for char in nonce_token):
        return False
    return (
        "records_file=" in text
        and "details_file=" in text
        and "/proc/[0-9]*" in text
        and "target=$1 invocation=$2 sampler_nonce=$3" in text
    )


def _target_runtime_activity_entry_is_control_helper(
    *,
    command: str,
    argv: list[str],
) -> bool:
    tokens = [str(token or "") for token in argv]
    text = "\n".join(tokens + [str(command or "")])
    token_names = {Path(token).name for token in tokens}
    if "apex-target-cleanup" in token_names:
        return (
            "target=$1" in text
            and "sig=$2" in text
            and "invocation=$3" in text
            and "children_file=" in text
            and "/proc/[0-9]*" in text
            and 'kill -"$sig"' in text
        )
    if "apex-target-activity" in token_names:
        return (
            "target=$1" in text
            and "invocation=$2" in text
            and "sampler_nonce=$3" in text
            and "records_file=" in text
            and "details_file=" in text
            and "/proc/[0-9]*" in text
        )
    return False


def _target_runtime_policy_violations(
    *,
    context: dict[str, Any],
    workdir: str,
    invocation_id: str,
) -> list[dict[str, Any]]:
    shim_dir = str(context.get("shim_dir") or "").strip()
    if not shim_dir:
        return []
    marker_dir = Path(shim_dir) / "policy_violations"
    if not marker_dir.is_dir():
        return []
    violations: list[dict[str, Any]] = []
    target_workdir = str(workdir or "").rstrip("/")
    target_invocation = str(invocation_id or "").strip()
    for marker in sorted(marker_dir.glob("*.json"))[-50:]:
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        marker_invocation = str(payload.get("invocation_id") or "").strip()
        marker_workdir = str(payload.get("workdir") or "").rstrip("/")
        matches_invocation = bool(target_invocation and marker_invocation == target_invocation)
        matches_workdir = bool(
            target_workdir
            and marker_workdir
            and (
                marker_workdir == target_workdir or marker_workdir.startswith(target_workdir + "/")
            )
        )
        if not matches_invocation and not matches_workdir:
            continue
        normalized = dict(payload)
        normalized["marker_path"] = str(marker)
        violations.append(normalized)
    return violations


def _runner_source() -> str:
    return "#!" + _runner_interpreter() + "\n" + _RUNNER_BODY


def _runner_interpreter() -> str:
    override = str(os.environ.get("APEX_TARGET_TOOL_RUNNER_PYTHON") or "").strip()
    candidates = [override, "/usr/bin/python3"]
    system_python = shutil.which("python3", path=os.pathsep.join(["/usr/bin", "/bin"]))
    if system_python:
        candidates.append(system_python)
    candidates.append(sys.executable)
    for candidate in candidates:
        if not candidate:
            continue
        try:
            path = Path(candidate).expanduser()
            if path.exists() and os.access(path, os.X_OK):
                return str(path)
        except OSError:
            continue
    return sys.executable


_RUNNER_BODY = r"""
from __future__ import annotations

import json
import os
import re
import select
import shlex
import signal
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


STATIC_READ_ONLY_TOOLS = {
    "cat",
    "diff",
    "find",
    "grep",
    "head",
    "ls",
    "md5sum",
    "pwd",
    "rg",
    "sed",
    "sha1sum",
    "sha256sum",
    "shasum",
    "tail",
    "wc",
}
GIT_HISTORY_SUBCOMMANDS = {
    "blame",
    "cat-file",
    "log",
    "ls-tree",
    "reflog",
    "rev-list",
    "show",
}
GIT_GLOBAL_OPTIONS_WITH_VALUE = {
    "-C",
    "-c",
    "--git-dir",
    "--namespace",
    "--super-prefix",
    "--work-tree",
}
GIT_GLOBAL_OPTIONS = {
    "--bare",
    "--no-optional-locks",
    "--no-pager",
    "--paginate",
    "--version",
}

_ABSOLUTE_DYNAMIC_TOOL_TERMINATOR = r"(?=$|[\s;&|`'\"),])"
_ABSOLUTE_HOST_DYNAMIC_TOOL_RE = re.compile(
    r"(^|[\s;&|])/(?:usr|opt|bin|sbin|usr/local|opt/homebrew)/[^\s;&|]*"
    r"(?:bash|sh|zsh|env|xargs|git|docker|curl|wget|perl|awk|make|timeout|gtimeout|patch|diff|shasum|sha1sum|sha256sum|md5sum|python(?:[0-9.]+)?|pytest|py\.test|pip(?:[0-9.]+)?|uv|poetry|hatch|tox|nox|coverage|"
    r"django-admin|node|npm|npx|pnpm|yarn|bun|deno|go|cargo|mvn|gradle|java|ruby|bundle|php|dotnet|swift)"
    + _ABSOLUTE_DYNAMIC_TOOL_TERMINATOR
)
_ABSOLUTE_DYNAMIC_PATH_RE = re.compile(
    r"/(?:usr/local|opt/homebrew|usr|opt|bin|sbin)/(?:[^\s;&|]*/)*"
    r"(?P<tool>bash|sh|zsh|env|xargs|git|docker|curl|wget|perl|awk|make|timeout|gtimeout|patch|diff|shasum|sha1sum|sha256sum|md5sum|python(?:[0-9.]+)?|pytest|py\.test|pip(?:[0-9.]+)?|uv|poetry|hatch|tox|nox|coverage|"
    r"django-admin|node|npm|npx|pnpm|yarn|bun|deno|go|cargo|mvn|gradle|java|ruby|bundle|php|dotnet|swift)"
    + _ABSOLUTE_DYNAMIC_TOOL_TERMINATOR
)
_DOCKER_TARGET_ABSOLUTE_WRAPPER_TOOLS = {"bash", "sh", "zsh", "env"}
_DOCKER_EXEC_WRAPPER = r'''
tool=${1##*/}
case "$tool" in
  zsh|bash)
    if command -v "$1" >/dev/null 2>&1; then
      exec "$@"
    fi
    original_tool=$1
    shift
    has_c=0
    for arg in "$@"; do
      case "$arg" in
        --)
          continue
          ;;
        -*)
          case "$arg" in
            *c*)
              has_c=1
              break
              ;;
          esac
          ;;
        *)
          break
          ;;
      esac
    done
    if [ "$has_c" = "1" ]; then
      shift
      if [ "$tool" = "zsh" ] && command -v bash >/dev/null 2>&1; then
        exec bash "$@"
      fi
      if command -v sh >/dev/null 2>&1; then
        exec sh "$@"
      fi
    fi
    set -- "$original_tool" "$@"
    ;;
esac
exec "$@"
'''.strip()


def _docker_root_setup_wrapper(base_wrapper: str) -> str:
    quoted_base = shlex.quote(str(base_wrapper or "exec \"$@\""))
    prelude = rf'''
if [ -z "${{APEX_TARGET_RUNTIME_AFTER_ROOT_SETUP:-}}" ]; then
  if [ -n "${{APEX_TARGET_RUNTIME_ROOT_SETUP_SCRIPT:-}}" ]; then
    if [ "$(id -u)" != "0" ]; then
      echo "target runtime root setup requested but container is not running as root" >&2
      exit 126
    fi
    /bin/sh -eu -c "$APEX_TARGET_RUNTIME_ROOT_SETUP_SCRIPT"
  fi
  if [ -n "${{APEX_TARGET_RUNTIME_RUN_AS_USER:-}}" ] && [ "$(id -u)" = "0" ]; then
    apex_run_user="$APEX_TARGET_RUNTIME_RUN_AS_USER"
    apex_uid="${{apex_run_user%%:*}}"
    apex_gid="${{apex_run_user#*:}}"
    if [ "$apex_gid" = "$apex_run_user" ]; then
      apex_gid="$apex_uid"
    fi
    case "$apex_uid" in
      ""|*[!0-9]*)
        apex_resolved_uid="$(id -u "$apex_uid" 2>/dev/null || true)"
        if [ -z "$apex_resolved_uid" ]; then
          echo "target runtime cannot resolve docker user uid: $apex_uid" >&2
          exit 126
        fi
        apex_uid="$apex_resolved_uid"
        ;;
    esac
    case "$apex_gid" in
      ""|*[!0-9]*)
        apex_resolved_gid="$(getent group "$apex_gid" 2>/dev/null | awk -F: '{{print $3}}' || true)"
        if [ -z "$apex_resolved_gid" ]; then
          echo "target runtime cannot resolve docker user gid: $apex_gid" >&2
          exit 126
        fi
        apex_gid="$apex_resolved_gid"
        ;;
    esac
    if command -v setpriv >/dev/null 2>&1; then
      exec env APEX_TARGET_RUNTIME_AFTER_ROOT_SETUP=1 APEX_TARGET_RUNTIME_ROOT_SETUP_SCRIPT= APEX_TARGET_RUNTIME_RUN_AS_USER= \
        setpriv --reuid "$apex_uid" --regid "$apex_gid" --clear-groups /bin/sh -c {quoted_base} "$0" "$@"
    fi
    echo "target runtime cannot drop from root to $APEX_TARGET_RUNTIME_RUN_AS_USER: setpriv not found" >&2
    exit 126
  fi
fi
'''
    return (prelude.strip() + "\n" + str(base_wrapper or "exec \"$@\"")).strip()


def _absolute_dynamic_arg_allowed_in_target_runtime(context: dict, arg: str) -> bool:
    runtime = dict(context.get("runtime") or {})
    mode = str(runtime.get("kind") or context.get("mode") or "")
    if mode not in {"docker_image", "docker_exec"}:
        return False
    matches = list(_ABSOLUTE_DYNAMIC_PATH_RE.finditer(str(arg or "")))
    if not matches:
        return False
    return all(
        str(match.group("tool") or "").split(".")[0] in _DOCKER_TARGET_ABSOLUTE_WRAPPER_TOOLS
        for match in matches
    )


def _shell_command_after_options(args: list[str], start: int = 0) -> str:
    cursor = start
    while cursor < len(args):
        option = str(args[cursor] or "")
        if option == "--":
            cursor += 1
            continue
        if option.startswith("-"):
            if "c" in option and cursor + 1 < len(args):
                return str(args[cursor + 1] or "")
            cursor += 1
            continue
        break
    return ""


def _embedded_shell_command(tool_name: str, args: list[str]) -> str:
    if Path(str(tool_name or "")).name in {"bash", "sh", "zsh"}:
        direct = _shell_command_after_options(args)
        if direct:
            return direct
    for index, token in enumerate(args):
        if Path(str(token or "")).name not in {"bash", "sh", "zsh"}:
            continue
        nested = _shell_command_after_options(args, start=index + 1)
        if nested:
            return nested
    return ""


def _policy_violation(context: dict, tool_name: str, args: list[str]) -> str:
    filesystem_structural = _filesystem_boundary_policy_is_structural(context)
    git_history_violation = _git_history_policy_violation(context, tool_name, args)
    if git_history_violation:
        return git_history_violation
    if _command_policy_blocks_enabled(context):
        blocked_policy = _blocked_command_policy_violation(context, tool_name, args)
        if blocked_policy:
            return blocked_policy
    if not filesystem_structural and tool_name == "find" and any(
        token in {"-exec", "-execdir", "-ok", "-okdir", "-delete"} for token in args
    ):
        return "unsupported mutating find invocation"
    if not filesystem_structural:
        for arg in args:
            if _ABSOLUTE_HOST_DYNAMIC_TOOL_RE.search(str(arg)):
                if _absolute_dynamic_arg_allowed_in_target_runtime(context, str(arg)):
                    continue
                return "absolute host dynamic tool paths are disabled; use PATH-resolved target tools"
    if not filesystem_structural and tool_name in STATIC_READ_ONLY_TOOLS:
        path_violation = _static_path_policy_violation(args)
        if path_violation:
            return path_violation
    return ""


def _filesystem_boundary_policy_is_structural(context: dict) -> bool:
    policy = str(context.get("filesystem_boundary_policy") or "").strip().lower()
    return policy in {
        "structurally_isolated",
        "structurally-isolated",
        "container_isolated",
        "container-isolated",
        "sandbox_isolated",
        "sandbox-isolated",
    }


def _command_policy_blocks_enabled(context: dict) -> bool:
    if "command_policy_blocks" in context:
        return bool(context.get("command_policy_blocks"))
    return bool(context.get("blocked_command_patterns") or [])


def _record_policy_violation(context: dict, tool_name: str, args: list[str], reason: str) -> None:
    shim_dir = str(os.environ.get("APEX_TARGET_TOOL_SHIM_DIR") or context.get("shim_dir") or "")
    if not shim_dir:
        return
    try:
        marker_dir = Path(shim_dir) / "policy_violations"
        marker_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": time.time(),
            "pid": os.getpid(),
            "tool": str(tool_name or ""),
            "args": [str(arg or "") for arg in args],
            "reason": str(reason or ""),
            "invocation_id": os.environ.get("APEX_CLI_INVOCATION_ID", ""),
            "workdir": os.environ.get("APEX_TARGET_TOOL_WORKDIR", ""),
        }
        marker = marker_dir / f"violation-{os.getpid()}-{time.time_ns()}.json"
        marker.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        return


def _tokenize_command_text(command_text: str) -> list[str]:
    raw = str(command_text or "").strip()
    if not raw:
        return []
    try:
        return shlex.split(raw, posix=True)
    except ValueError:
        return raw.split()


def _git_history_subcommand(tokens: list[str]) -> str:
    if not tokens or Path(str(tokens[0] or "")).name != "git":
        return ""
    cursor = 1
    while cursor < len(tokens):
        current = str(tokens[cursor] or "")
        if not current:
            cursor += 1
            continue
        if current == "--":
            cursor += 1
            if cursor < len(tokens):
                subcommand = Path(str(tokens[cursor] or "")).name
                return subcommand if subcommand in GIT_HISTORY_SUBCOMMANDS else ""
            return ""
        if current in GIT_GLOBAL_OPTIONS_WITH_VALUE:
            cursor += 2
            continue
        if any(current.startswith(prefix + "=") for prefix in GIT_GLOBAL_OPTIONS_WITH_VALUE):
            cursor += 1
            continue
        if current in GIT_GLOBAL_OPTIONS:
            cursor += 1
            continue
        if current.startswith("-"):
            cursor += 1
            continue
        subcommand = Path(current).name
        return subcommand if subcommand in GIT_HISTORY_SUBCOMMANDS else ""
    return ""


def _git_history_policy_violation(context: dict, tool_name: str, args: list[str]) -> str:
    policy = str(context.get("git_history_policy") or "blocked").strip().lower()
    if policy in {"structurally_erased", "structurally-erased", "allow", "allowed"}:
        return ""
    command_tokens = [[str(tool_name or ""), *(str(arg or "") for arg in args)]]
    embedded_shell = _embedded_shell_command(tool_name, args)
    if embedded_shell:
        command_tokens.append(_tokenize_command_text(embedded_shell))
    for tokens in command_tokens:
        subcommand = _git_history_subcommand(tokens)
        if subcommand:
            return (
                "git history/object discovery is blocked by target-runtime policy: "
                f"git {subcommand}; use the current worktree, visible tests, "
                "and working-tree diff only."
            )
    return ""


def _blocked_command_policy_violation(context: dict, tool_name: str, args: list[str]) -> str:
    command_texts = [" ".join([str(tool_name or ""), *(str(arg or "") for arg in args)])]
    embedded_shell = _embedded_shell_command(tool_name, args)
    if embedded_shell:
        command_texts.append(embedded_shell)
    for raw_rule in list(context.get("blocked_command_patterns") or []):
        if not isinstance(raw_rule, dict):
            continue
        tools = {str(tool).strip() for tool in list(raw_rule.get("tools") or []) if str(tool).strip()}
        if tools and str(tool_name or "") not in tools:
            continue
        pattern = str(raw_rule.get("pattern") or "")
        if not pattern:
            continue
        try:
            matched = any(re.search(pattern, command_text) for command_text in command_texts)
        except re.error:
            continue
        if matched:
            message = str(raw_rule.get("message") or "command is blocked by target-runtime policy")
            return f"blocked target-runtime command: {message}"
    return ""


def _static_path_policy_violation(args: list[str]) -> str:
    context_path = os.environ.get("APEX_TARGET_TOOL_CONTEXT", "")
    if not context_path:
        return ""
    try:
        context = json.loads(Path(context_path).read_text(encoding="utf-8"))
        workdir = _effective_workdir(context)
    except Exception:
        return "target runtime context could not be read for static path policy"
    runtime = dict(context.get("runtime") or {})
    if (
        os.environ.get("APEX_AGENT_CONTAINER") == "1"
        and str(runtime.get("kind") or context.get("mode") or "") in {"docker_exec", "docker_image"}
    ):
        # Docker target runtime: absolute paths are container paths, not host-Apex paths.
        return ""
    for raw in args:
        arg = str(raw or "")
        if not arg or arg == "--" or arg.startswith("-"):
            continue
        if not _looks_path_like(arg, workdir):
            continue
        try:
            resolved = (
                Path(arg).expanduser().resolve()
                if os.path.isabs(arg) or arg.startswith("~")
                else (workdir / arg).resolve()
            )
            resolved.relative_to(workdir)
        except Exception:
            return f"static tool path escapes target workspace: {arg}"
    return ""


def _looks_path_like(arg: str, workdir: Path) -> bool:
    if arg in {".", ".."}:
        return True
    if arg.startswith(("/", "./", "../", "~")):
        return True
    if "/" in arg:
        return True
    try:
        return (workdir / arg).exists()
    except OSError:
        return False


def _effective_workdir(context: dict) -> Path:
    override = str(os.environ.get("APEX_TARGET_TOOL_WORKDIR") or "").strip()
    raw = override or str(context.get("workdir") or ".")
    return Path(raw).expanduser().resolve()


def _stdin_if_ready() -> str | None:
    try:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read()
    except Exception:
        return None
    return None


def _bridge_descriptor() -> dict:
    descriptor_file = os.environ.get("APEX_TARGET_TOOL_BRIDGE_FILE", "")
    if descriptor_file:
        try:
            loaded = json.loads(Path(descriptor_file).read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            return {}
    url = os.environ.get("APEX_TARGET_TOOL_BRIDGE_URL", "")
    token = os.environ.get("APEX_TARGET_TOOL_BRIDGE_TOKEN", "")
    if url:
        return {"url": url, "token": token}
    return {}


def _run_bridge(
    context: dict,
    tool_name: str,
    args: list[str],
    stdin_data: str | None,
) -> int | None:
    if os.environ.get("APEX_TARGET_TOOL_BRIDGE_LOCAL") == "1":
        return None
    descriptor = _bridge_descriptor()
    urls = [
        str(descriptor.get(key) or "")
        for key in ("url", "local_url")
        if str(descriptor.get(key) or "")
    ]
    if not urls:
        return None
    token = str(descriptor.get("token") or "")
    payload = {
        "tool": str(tool_name or ""),
        "args": [str(arg or "") for arg in args],
        "stdin": stdin_data or "",
        "context_path": os.environ.get("APEX_TARGET_TOOL_CONTEXT", ""),
    }
    raw_payload = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    result = None
    for url in urls:
        request = urllib.request.Request(
            url,
            data=raw_payload,
            headers={
                "Content-Type": "application/json",
                "X-APEX-Target-Tool-Bridge-Token": token,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=max(1, min(int(context.get("timeout_seconds") or 60), 60)),
            ) as response:
                raw = response.read().decode("utf-8", errors="replace")
            result = json.loads(raw)
            break
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
            continue
    if result is None:
        print(f"APEX target tool bridge failed for {tool_name}: {last_error}", file=sys.stderr)
        return 113
    sys.stdout.write(_bounded_stream_text(context, result.get("stdout") or "", "stdout"))
    sys.stderr.write(_bounded_stream_text(context, result.get("stderr") or "", "stderr"))
    try:
        return int(result.get("returncode"))
    except (TypeError, ValueError):
        return 113


def _command_for_tool(tool_name: str, args: list[str]) -> list[str]:
    if tool_name in {"python", "python3", "python3.10", "python3.11", "python3.12", "python3.13"}:
        return ["__APEX_TARGET_PYTHON__", *args]
    if tool_name in {"pytest", "py.test"}:
        return ["__APEX_TARGET_PYTHON__", "-m", "pytest", *args]
    if tool_name in {"pip", "pip3"}:
        return ["__APEX_TARGET_PYTHON__", "-m", "pip", *args]
    if tool_name == "coverage":
        return ["__APEX_TARGET_PYTHON__", "-m", "coverage", *args]
    return [tool_name, *args]


def _target_tool_timeout_seconds(context: dict, *, agent_command: bool = False) -> int:
    key = "agent_command_timeout_seconds" if agent_command else "timeout_seconds"
    try:
        value = int(context.get(key) or context.get("timeout_seconds") or 60)
    except (TypeError, ValueError):
        value = 60
    return max(1, value)


def _target_runtime_command_is_agent_cli_launch(command: list[str]) -> bool:
    tokens = [str(item or "") for item in command]
    if not tokens:
        return False
    has_apex_runtime_marker = any(
        ".apex_agent_runtime" in token
        or "/opt/apex-agent-cli/" in token
        or token == "apex-claude-stdin"
        for token in tokens
    )
    if not has_apex_runtime_marker:
        return False
    for index, token in enumerate(tokens):
        name = Path(token).name
        tail = tokens[index + 1 :]
        if (
            name == "codex"
            and "exec" in tail
            and "--output-last-message" in tail
            and any(".apex_agent_runtime" in item for item in tail)
        ):
            return True
        if (
            name == "claude"
            and ("--system-prompt-file" in tail or "apex-claude-stdin" in tokens)
            and any(".apex_agent_runtime" in item for item in tail + tokens[:index])
        ):
            return True
    return False


def _target_tool_timeout_seconds_for_command(
    context: dict,
    command: list[str],
    *,
    agent_command: bool = False,
) -> int:
    if agent_command and _target_runtime_command_is_agent_cli_launch(command):
        return _target_tool_timeout_seconds(context, agent_command=False)
    return _target_tool_timeout_seconds(context, agent_command=agent_command)


def _target_tool_output_capture_max_chars(context: dict) -> int:
    try:
        return max(0, int(context.get("output_capture_max_chars") or 0))
    except (TypeError, ValueError):
        return 0


def _bounded_stream_text(context: dict, value, stream_name: str) -> str:
    text = _timeout_stream_text(value)
    limit = _target_tool_output_capture_max_chars(context)
    if limit <= 0 or len(text) <= limit:
        return text
    head_limit = limit // 2
    tail_limit = limit - head_limit
    omitted = max(0, len(text) - limit)
    marker = (
        f"\n...[apex target-runtime truncated {omitted} chars "
        f"from {stream_name} output]...\n"
    )
    return text[:head_limit] + marker + (text[-tail_limit:] if tail_limit else "")


def _timeout_stream_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _write_completed_process_output(context: dict, completed: subprocess.CompletedProcess) -> None:
    sys.stdout.write(_bounded_stream_text(context, completed.stdout or "", "stdout"))
    sys.stderr.write(_bounded_stream_text(context, completed.stderr or "", "stderr"))


def _report_target_runtime_timeout(
    context: dict,
    command: list[str],
    exc: subprocess.TimeoutExpired,
) -> int:
    sys.stdout.write(
        _bounded_stream_text(
            context,
            getattr(exc, "stdout", None) or getattr(exc, "output", None),
            "stdout",
        )
    )
    sys.stderr.write(_bounded_stream_text(context, getattr(exc, "stderr", None), "stderr"))
    try:
        rendered = shlex.join([str(item) for item in command[:8]])
    except Exception:
        rendered = " ".join(str(item) for item in command[:8])
    if len(command) > 8:
        rendered += " ..."
    _cleanup_docker_exec_timeout_processes(context)
    print(
        f"target runtime command timed out after {int(float(exc.timeout or 0))}s: {rendered}",
        file=sys.stderr,
    )
    return 124


def _cleanup_docker_exec_timeout_processes(context: dict) -> None:
    runtime = dict(context.get("runtime") or {})
    if str(runtime.get("kind") or context.get("mode") or "") != "docker_exec":
        return
    if os.environ.get("APEX_AGENT_CONTAINER") == "1":
        return
    container_name = str(runtime.get("docker_container_name") or "").strip()
    if not container_name:
        return
    container_workdir = _docker_exec_workdir(context).rstrip("/")
    if not container_workdir or container_workdir in {"/", "/workspace"}:
        return
    cleaned = _cleanup_docker_exec_timeout_processes_with_signal(
        context,
        runtime,
        container_name,
        container_workdir,
        int(signal.SIGTERM),
    )
    if cleaned:
        time.sleep(0.25)
        _cleanup_docker_exec_timeout_processes_with_signal(
            context,
            runtime,
            container_name,
            container_workdir,
            int(signal.SIGKILL),
        )


def _cleanup_docker_exec_timeout_processes_with_signal(
    context: dict,
    runtime: dict,
    container_name: str,
    container_workdir: str,
    signum: int,
) -> set[int]:
    invocation_id = str(os.environ.get("APEX_CLI_INVOCATION_ID") or "").strip()
    docker_bin = str(runtime.get("docker_bin") or "docker").strip() or "docker"
    docker_client_env = dict(os.environ)
    docker_client_env.update(
        {str(k): str(v) for k, v in dict(runtime.get("docker_host_env") or {}).items()}
    )
    script = r'''
target=$1
sig=$2
invocation=$3
self_pid=$$
matches=" "
children_file="${TMPDIR:-/tmp}/apex-target-cleanup-$$.pids"
trap 'rm -f "$children_file"' EXIT
: > "$children_file"
target_parent=${target%/*}
target_name=${target##*/}
runtime_prefix_dash=""
runtime_prefix_dir=""
if [ -n "$target_parent" ] && [ "$target_parent" != "$target" ] && [ -n "$target_name" ]; then
  runtime_prefix_dash="$target_parent/.apex_agent_runtime/$target_name-"
  runtime_prefix_dir="$target_parent/.apex_agent_runtime/$target_name/"
fi
for proc in /proc/[0-9]*; do
  pid=${proc##*/}
  [ "$pid" = "1" ] && continue
  [ "$pid" = "$self_pid" ] && continue
  ppid=$(grep '^PPid:' "$proc/status" 2>/dev/null | { read _ value; printf '%s' "${value:-0}"; } || true)
  case "$ppid" in *[!0-9]*|"") ppid=0 ;; esac
  printf '%s %s\n' "$pid" "$ppid" >> "$children_file"
  cwd=$(readlink "$proc/cwd" 2>/dev/null || true)
  cwd=${cwd% (deleted)}
  match=0
  cmdline=$(cat "$proc/cmdline" 2>/dev/null | tr '\000' ' ' || true)
  environ=$(cat "$proc/environ" 2>/dev/null | tr '\000' '\n' || true)
  if [ -n "$invocation" ]; then
    case "$environ" in
      *"APEX_CLI_INVOCATION_ID=$invocation"*) match=1 ;;
    esac
  fi
  if [ "$match" != "1" ]; then
    if [ "$cwd" = "$target" ]; then
      match=1
    fi
    case "$cwd" in
      "$target"/*) match=1 ;;
    esac
    case "$cmdline" in
      *"$target"*) match=1 ;;
    esac
    if [ -n "$runtime_prefix_dash" ]; then
      case "$cmdline" in
        *"$runtime_prefix_dash"*|*"$runtime_prefix_dir"*) match=1 ;;
      esac
      case "$environ" in
        *"$runtime_prefix_dash"*|*"$runtime_prefix_dir"*) match=1 ;;
      esac
    fi
    case "$environ" in
      *"APEX_TARGET_TOOL_WORKDIR=$target"*) match=1 ;;
    esac
  fi
  if [ "$match" = "1" ]; then
    matches="$matches$pid "
  fi
done
changed=1
while [ "$changed" = "1" ]; do
  changed=0
  while read child parent; do
    [ -n "$child" ] || continue
    case "$matches" in
      *" $child "*) continue ;;
    esac
    case "$matches" in
      *" $parent "*)
        matches="$matches$child "
        changed=1
        ;;
    esac
  done < "$children_file"
done
for pid in $matches; do
  [ "$pid" = "$self_pid" ] && continue
  kill -"$sig" "$pid" 2>/dev/null || true
  printf '%s\n' "$pid"
done
'''
    command = [
        docker_bin,
        "exec",
        container_name,
        "sh",
        "-c",
        script,
        "apex-target-cleanup",
        container_workdir,
        str(int(signum)),
        invocation_id,
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=docker_client_env,
        )
    except Exception:
        return set()
    if int(completed.returncode or 0) in {126, 127}:
        fallback = list(command)
        fallback[3] = "/bin/sh.apex-real"
        try:
            completed = subprocess.run(
                fallback,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                env=docker_client_env,
            )
        except Exception:
            return set()
    cleaned: set[int] = set()
    for line in (completed.stdout or "").splitlines():
        line = line.strip()
        if line.isdigit():
            cleaned.add(int(line))
    return cleaned


def _shell_wrapper_uses_c(args: list[str]) -> bool:
    cursor = 0
    while cursor < len(args):
        option = str(args[cursor] or "")
        if option == "--":
            cursor += 1
            continue
        if option.startswith("-"):
            if "c" in option:
                return True
            cursor += 1
            continue
        break
    return False


def _resolve_shell_wrapper_fallback(executable: str, args: list[str], target_env: dict[str, str]) -> str:
    shell_name = Path(str(executable or "")).name
    if shell_name not in {"bash", "zsh"} or not _shell_wrapper_uses_c(args):
        return ""
    search_path = target_env.get("PATH", "")
    fallback_order = ("bash", "sh") if shell_name == "zsh" else ("sh",)
    for candidate in fallback_order:
        resolved = shutil.which(candidate, path=search_path)
        if resolved:
            return resolved
    return ""


def _host_env_command(command: list[str], target_env: dict[str, str]) -> list[str]:
    resolved = []
    for item in command:
        if item == "__APEX_TARGET_PYTHON__":
            python_path = shutil.which("python", path=target_env.get("PATH", ""))
            python_path = python_path or shutil.which("python3", path=target_env.get("PATH", ""))
            if not python_path:
                raise FileNotFoundError(
                    "target runtime PATH has no python interpreter for python/pytest/pip"
                )
            resolved.append(python_path)
        else:
            resolved.append(item)
    executable = resolved[0] if resolved else ""
    if executable and not os.path.isabs(executable):
        resolved_executable = shutil.which(executable, path=target_env.get("PATH", ""))
        if resolved_executable:
            resolved[0] = resolved_executable
        else:
            fallback = _resolve_shell_wrapper_fallback(executable, resolved[1:], target_env)
            if fallback:
                resolved[0] = fallback
    elif executable and os.path.isabs(executable) and not os.path.exists(executable):
        fallback = _resolve_shell_wrapper_fallback(executable, resolved[1:], target_env)
        if fallback:
            resolved[0] = fallback
    return resolved


def _run_host_env(context: dict, command: list[str], stdin_data: str | None) -> int:
    runtime = dict(context.get("runtime") or {})
    runtime_env = {str(k): str(v) for k, v in dict(runtime.get("env") or {}).items()}
    target_env = dict(runtime_env)
    target_env["PATH"] = str(runtime_env.get("PATH") or os.environ.get("APEX_HOST_PATH") or "")
    # NOTE: deliberately do NOT prepend the per-task shim dir here. ``_run_host_env``
    # is the EXECUTOR that runs the already-resolved target command, so prepending
    # the shim dir would make the shims intercept the executor's own python/pytest
    # (the test runner) and refuse it. Host-path source-download egress is covered
    # by the Part-C vendored-source sanitizer + the official audit instead.
    try:
        resolved_command = _host_env_command(command, target_env)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 113
    try:
        timeout_seconds = _target_tool_timeout_seconds_for_command(
            context,
            resolved_command,
            agent_command=os.environ.get("APEX_AGENT_CONTAINER") == "1",
        )
        completed = subprocess.run(
            resolved_command,
            cwd=str(_effective_workdir(context)),
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=target_env,
        )
    except FileNotFoundError as exc:
        print(f"target runtime command not found: {exc}", file=sys.stderr)
        return 113
    except subprocess.TimeoutExpired as exc:
        return _report_target_runtime_timeout(context, resolved_command, exc)
    _write_completed_process_output(context, completed)
    return int(completed.returncode or 0)


def _run_docker(context: dict, command: list[str], stdin_data: str | None) -> int:
    runtime = dict(context.get("runtime") or {})
    image = str(runtime.get("docker_image") or "")
    if not image:
        print("target runtime has no docker image", file=sys.stderr)
        return 113
    network = str(runtime.get("docker_network") or "host").strip() or "host"
    docker_bin = str(runtime.get("docker_bin") or "docker").strip() or "docker"
    docker_cmd = [docker_bin, "run", "--rm", "--network", network]
    docker_user = str(runtime.get("docker_user") or "").strip()
    root_setup_script = str(runtime.get("docker_root_setup_script") or "").strip()
    if docker_user and not root_setup_script:
        docker_cmd.extend(["-u", docker_user])
    platform = str(runtime.get("docker_platform") or "")
    if platform:
        docker_cmd.extend(["--platform", platform])
    workdir = _effective_workdir(context)
    container_workdir = str(runtime.get("docker_workdir") or "/workspace")
    docker_cmd.extend(["-v", f"{workdir}:{container_workdir}:rw"])
    for raw_mount in list(runtime.get("docker_mounts") or []):
        if not isinstance(raw_mount, dict):
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
        docker_cmd.extend(["--mount", option])
    docker_cmd.extend(["-w", container_workdir])
    for key, value in dict(runtime.get("docker_env") or {}).items():
        docker_cmd.extend(["-e", f"{key}={value}"])
    if root_setup_script:
        docker_cmd.extend(["-e", f"APEX_TARGET_RUNTIME_ROOT_SETUP_SCRIPT={root_setup_script}"])
        if docker_user:
            docker_cmd.extend(["-e", f"APEX_TARGET_RUNTIME_RUN_AS_USER={docker_user}"])
    rendered = [
        "python" if item == "__APEX_TARGET_PYTHON__" else item
        for item in command
    ]
    docker_cmd.extend(
        [
            "--entrypoint",
            "/bin/sh",
            image,
            "-c",
            _docker_root_setup_wrapper(_DOCKER_EXEC_WRAPPER) if root_setup_script else _DOCKER_EXEC_WRAPPER,
            "apex-target-tool",
        ]
    )
    docker_cmd.extend(rendered)
    try:
        completed = subprocess.run(
            docker_cmd,
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=_target_tool_timeout_seconds(context),
            check=False,
            env={**os.environ, **dict(runtime.get("docker_host_env") or {})},
        )
    except subprocess.TimeoutExpired as exc:
        return _report_target_runtime_timeout(context, docker_cmd, exc)
    _write_completed_process_output(context, completed)
    return int(completed.returncode or 0)


def _docker_exec_workdir(context: dict) -> str:
    runtime = dict(context.get("runtime") or {})
    raw = str(os.environ.get("APEX_TARGET_TOOL_WORKDIR") or context.get("workdir") or ".")
    host_root = str(runtime.get("docker_host_workdir_root") or "").rstrip("/")
    container_root = str(runtime.get("docker_container_workdir_root") or "/workspace").rstrip("/")
    if host_root and (raw == host_root or raw.startswith(host_root + "/")):
        suffix = raw[len(host_root):].lstrip("/")
        return container_root + (("/" + suffix) if suffix else "")
    return raw


def _docker_exec_container_path(context: dict, raw_path: str) -> str:
    runtime = dict(context.get("runtime") or {})
    raw = str(raw_path or "")
    if not raw:
        return ""
    host_root = str(runtime.get("docker_host_workdir_root") or "").rstrip("/")
    container_root = str(runtime.get("docker_container_workdir_root") or "/workspace").rstrip("/")
    if host_root and (raw == host_root or raw.startswith(host_root + "/")):
        suffix = raw[len(host_root):].lstrip("/")
        return container_root + (("/" + suffix) if suffix else "")
    return raw


def _docker_exec_inner_agent_env(context: dict, runtime_env: dict[str, str]) -> dict[str, str]:
    env = {str(k): str(v) for k, v in dict(runtime_env or {}).items()}
    base_path = str(env.get("PATH") or os.environ.get("PATH") or "")
    shim_dir = _docker_exec_container_path(context, str(context.get("shim_dir") or ""))
    context_path = _docker_exec_container_path(
        context,
        str(context.get("context_path") or os.environ.get("APEX_TARGET_TOOL_CONTEXT") or ""),
    )
    if shim_dir:
        env["PATH"] = shim_dir + (os.pathsep + base_path if base_path else "")
        env["APEX_TARGET_TOOL_SHIM_DIR"] = shim_dir
    elif base_path:
        env["PATH"] = base_path
    if context_path:
        env["APEX_TARGET_TOOL_CONTEXT"] = context_path
    env["APEX_AGENT_CONTAINER"] = "1"
    env["APEX_HOST_DYNAMIC_TOOLS"] = "disabled"
    env["APEX_TARGET_TOOL_WORKDIR"] = _docker_exec_workdir(context)
    env["APEX_HOST_PATH"] = base_path
    return env


def _run_docker_exec_direct(context: dict, command: list[str], stdin_data: str | None) -> int:
    runtime = dict(context.get("runtime") or {})
    runtime_env = {str(k): str(v) for k, v in dict(runtime.get("docker_env") or {}).items()}
    target_env = dict(os.environ)
    target_env.update(runtime_env)
    target_env["PATH"] = str(runtime_env.get("PATH") or os.environ.get("PATH") or "")
    try:
        resolved_command = _host_env_command(command, target_env)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 113
    try:
        timeout_seconds = _target_tool_timeout_seconds_for_command(
            context,
            resolved_command,
            agent_command=True,
        )
        completed = subprocess.run(
            resolved_command,
            cwd=_docker_exec_workdir(context),
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=target_env,
        )
    except FileNotFoundError as exc:
        print(f"target runtime command not found: {exc}", file=sys.stderr)
        return 113
    except subprocess.TimeoutExpired as exc:
        return _report_target_runtime_timeout(context, resolved_command, exc)
    _write_completed_process_output(context, completed)
    return int(completed.returncode or 0)


def _run_docker_exec(context: dict, command: list[str], stdin_data: str | None) -> int:
    runtime = dict(context.get("runtime") or {})
    if os.environ.get("APEX_AGENT_CONTAINER") == "1":
        return _run_docker_exec_direct(context, command, stdin_data)
    container_name = str(runtime.get("docker_container_name") or "")
    if not container_name:
        print("target runtime has no docker exec container", file=sys.stderr)
        return 113
    docker_bin = str(runtime.get("docker_bin") or "docker")
    docker_cmd = [docker_bin, "exec", "-i", "-w", _docker_exec_workdir(context)]
    inner_env = _docker_exec_inner_agent_env(
        context,
        {str(k): str(v) for k, v in dict(runtime.get("docker_env") or {}).items()},
    )
    for key, value in inner_env.items():
        docker_cmd.extend(["-e", f"{key}={value}"])
    rendered = [
        "python" if item == "__APEX_TARGET_PYTHON__" else item
        for item in command
    ]
    docker_cmd.extend([container_name, "/bin/sh", "-c", _DOCKER_EXEC_WRAPPER, "apex-target-tool"])
    docker_cmd.extend(rendered)
    docker_client_env = dict(os.environ)
    docker_client_env.update(
        {str(k): str(v) for k, v in dict(runtime.get("docker_host_env") or {}).items()}
    )
    try:
        completed = subprocess.run(
            docker_cmd,
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=_target_tool_timeout_seconds(context),
            check=False,
            env=docker_client_env,
        )
    except subprocess.TimeoutExpired as exc:
        return _report_target_runtime_timeout(context, docker_cmd, exc)
    _write_completed_process_output(context, completed)
    return int(completed.returncode or 0)


def main() -> int:
    context_path = os.environ.get("APEX_TARGET_TOOL_CONTEXT", "")
    tool_name = Path(sys.argv[0]).name
    if not context_path:
        print(f"APEX target runtime context missing; refusing host {tool_name}", file=sys.stderr)
        return 113
    context = json.loads(Path(context_path).read_text(encoding="utf-8"))
    stdin_data = _stdin_if_ready()
    runtime = dict(context.get("runtime") or {})
    mode = str(runtime.get("kind") or context.get("mode") or "fail_closed")
    violation = _policy_violation(context, tool_name, sys.argv[1:])
    if violation:
        _record_policy_violation(context, tool_name, sys.argv[1:], violation)
        print(violation, file=sys.stderr)
        return 113
    bridged = _run_bridge(context, tool_name, sys.argv[1:], stdin_data)
    if bridged is not None:
        return bridged
    if mode == "fail_closed":
        print(
            f"APEX target runtime has no command runner for {tool_name}; "
            "refusing to execute host dynamic tooling.",
            file=sys.stderr,
        )
        return 113
    command = _command_for_tool(tool_name, sys.argv[1:])
    if mode == "host_env":
        return _run_host_env(context, command, stdin_data)
    if mode == "docker_image":
        return _run_docker(context, command, stdin_data)
    if mode == "docker_exec":
        return _run_docker_exec(context, command, stdin_data)
    print(f"unsupported target runtime mode: {mode}", file=sys.stderr)
    return 113


if __name__ == "__main__":
    raise SystemExit(main())
"""
