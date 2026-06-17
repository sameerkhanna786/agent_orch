"""
Helpers for parsing and rebuilding pytest shell commands.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_SHELL_OPERATORS = {"&&", "||", ";", "|", "&"}
_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
_OPTIONS_WITH_VALUES = {
    "-c",
    "-k",
    "-m",
    "-n",
    "-o",
    "-p",
    "-r",
    "-W",
    "--basetemp",
    "--benchmark-autosave",
    "--benchmark-columns",
    "--benchmark-compare",
    "--benchmark-compare-fail",
    "--benchmark-cprofile",
    "--benchmark-group-by",
    "--benchmark-histogram",
    "--benchmark-name",
    "--benchmark-save",
    "--benchmark-sort",
    "--benchmark-storage",
    "--benchmark-timer",
    "--benchmark-warmup",
    "--benchmark-warmup-iterations",
    "--capture",
    "--color",
    "--confcutdir",
    "--cov",
    "--cov-config",
    "--cov-context",
    "--cov-fail-under",
    "--cov-report",
    "--deselect",
    "--doctest-report",
    "--durations",
    "--dist",
    "--ignore",
    "--ignore-glob",
    "--import-mode",
    "--json-report-file",
    "--junit-prefix",
    "--junitxml",
    "--log-auto-indent",
    "--log-cli-date-format",
    "--log-cli-format",
    "--log-cli-level",
    "--log-date-format",
    "--log-file",
    "--log-file-date-format",
    "--log-file-format",
    "--log-file-level",
    "--log-format",
    "--log-level",
    "--maxschedchunk",
    "--maxfail",
    "--override-ini",
    "--pythonwarnings",
    "--reportchars",
    "--reruns",
    "--reruns-delay",
    "--rootdir",
    "--tb",
    "--timeout",
    "--timeout-method",
    "--tx",
}
_PLUGIN_AUTOLOAD_SENSITIVE_OPTIONS = {
    "-n",
    "--benchmark",
    "--benchmark-autosave",
    "--benchmark-columns",
    "--benchmark-compare",
    "--benchmark-compare-fail",
    "--benchmark-cprofile",
    "--benchmark-disable",
    "--benchmark-enable",
    "--benchmark-group-by",
    "--benchmark-histogram",
    "--benchmark-name",
    "--benchmark-only",
    "--benchmark-save",
    "--benchmark-session",
    "--benchmark-skip",
    "--benchmark-sort",
    "--benchmark-storage",
    "--benchmark-timer",
    "--benchmark-warmup",
    "--benchmark-warmup-iterations",
    "--cov",
    "--cov-append",
    "--cov-branch",
    "--cov-config",
    "--cov-context",
    "--cov-fail-under",
    "--cov-report",
    "--dist",
    "--json-report",
    "--json-report-file",
    "--maxschedchunk",
    "--memray",
    "--no-cov",
    "--no-cov-on-fail",
    "--reruns",
    "--reruns-delay",
    "--timeout",
    "--timeout-method",
    "--tx",
}
_PYTEST_CONFIG_FILENAMES = (
    "pyproject.toml",
    "pytest.ini",
    "setup.cfg",
    "tox.ini",
)


@dataclass(frozen=True)
class ParsedPytestCommand:
    """Structured view of a pytest command embedded in a shell command."""

    shell_prefix_tokens: tuple[str, ...]
    env_prefix_tokens: tuple[str, ...]
    invocation_tokens: tuple[str, ...]
    option_tokens: tuple[str, ...]
    target_tokens: tuple[str, ...]


def parse_pytest_command(command: str) -> Optional[ParsedPytestCommand]:
    """Parse a direct or shell-wrapped pytest command."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    if not tokens:
        return None

    shell_prefix: list[str] = []
    index = 0
    while index < len(tokens):
        segment_end = index
        while segment_end < len(tokens) and tokens[segment_end] not in _SHELL_OPERATORS:
            segment_end += 1
        segment = tokens[index:segment_end]
        parsed_segment = _parse_pytest_segment(segment)
        if parsed_segment is not None:
            env_prefix_tokens, invocation_tokens, remaining_tokens = parsed_segment
            option_tokens, target_tokens = _split_pytest_args(remaining_tokens)
            return ParsedPytestCommand(
                shell_prefix_tokens=tuple(shell_prefix),
                env_prefix_tokens=tuple(env_prefix_tokens),
                invocation_tokens=tuple(invocation_tokens),
                option_tokens=tuple(option_tokens),
                target_tokens=tuple(target_tokens),
            )

        shell_prefix.extend(segment)
        if segment_end < len(tokens):
            shell_prefix.append(tokens[segment_end])
        index = segment_end + 1

    return None


def is_pytest_command(command: str) -> bool:
    return parse_pytest_command(command) is not None


def normalize_pytest_command(
    command: str,
    *,
    force_verbose: bool = False,
    disable_plugin_autoload: bool = True,
) -> str:
    parsed = parse_pytest_command(command)
    if parsed is None:
        return command
    return render_pytest_command(
        parsed,
        force_verbose=force_verbose,
        disable_plugin_autoload=disable_plugin_autoload,
    )


def build_targeted_pytest_command(
    command: str,
    selected_tests: list[str],
    *,
    force_verbose: bool = False,
    disable_plugin_autoload: bool = True,
) -> Optional[str]:
    parsed = parse_pytest_command(command)
    if parsed is None:
        return None
    return render_pytest_command(
        parsed,
        selected_tests=selected_tests,
        force_verbose=force_verbose,
        disable_plugin_autoload=disable_plugin_autoload,
    )


def build_expected_id_filtered_pytest_command(
    command: str,
    selected_tests: list[str],
    *,
    expected_ids_file: str,
    filter_plugin: str = "_apex_expected_ids_filter",
) -> Optional[str]:
    """Build a pytest command that targets files and filters expected nodeids.

    Pytest exits with usage rc=4 when any explicit nodeid target no longer
    resolves. Expected-id inventories can contain parametrized display strings
    that drift across collection, so file targets plus the expected-id filter
    preserve runnable signal while still reporting missing IDs.
    """
    parsed = parse_pytest_command(command)
    if parsed is None or not selected_tests:
        return None
    selected_files = _collapse_targets_to_files(
        [test_id for test_id in selected_tests if test_id]
    )
    if not selected_files:
        return None

    env_prefix_tokens = [
        token
        for token in parsed.env_prefix_tokens
        if not token.startswith("APEX_EXPECTED_IDS_FILE=")
    ]
    env_prefix_tokens.append(
        f"APEX_EXPECTED_IDS_FILE={shlex.quote(str(expected_ids_file))}"
    )

    option_tokens = list(parsed.option_tokens)
    if filter_plugin and not _pytest_command_loads_plugin(option_tokens, filter_plugin):
        option_tokens.extend(["-p", filter_plugin])

    rewritten = ParsedPytestCommand(
        shell_prefix_tokens=parsed.shell_prefix_tokens,
        env_prefix_tokens=tuple(env_prefix_tokens),
        invocation_tokens=parsed.invocation_tokens,
        option_tokens=tuple(option_tokens),
        target_tokens=(),
    )
    return render_pytest_command(
        rewritten,
        selected_tests=selected_files,
        disable_plugin_autoload=False,
    )


def build_ephemeral_pytest_command(
    command: str | None,
    test_target: str,
    *,
    force_verbose: bool = False,
    disable_plugin_autoload: bool = True,
) -> Optional[str]:
    if not command:
        return None
    return build_targeted_pytest_command(
        command,
        [test_target],
        force_verbose=force_verbose,
        disable_plugin_autoload=disable_plugin_autoload,
    )


def build_runtime_python_command(
    command: str | None,
    script_path: str,
) -> Optional[str]:
    if not command:
        return None
    parsed = parse_pytest_command(command)
    if parsed is None:
        return None

    python_invocation = _derive_python_invocation(parsed.invocation_tokens)
    if python_invocation is None:
        return None

    rendered_tokens = (
        list(parsed.shell_prefix_tokens)
        + list(parsed.env_prefix_tokens)
        + python_invocation
        + [script_path]
    )
    return " ".join(_render_shell_token(token) for token in rendered_tokens)


def rewrite_pytest_command_with_python(
    command: str | None,
    python_executable: str,
    *,
    disable_plugin_autoload: bool = False,
) -> Optional[str]:
    if not command:
        return None
    parsed = parse_pytest_command(command)
    if parsed is None:
        return None
    rewritten = ParsedPytestCommand(
        shell_prefix_tokens=parsed.shell_prefix_tokens,
        env_prefix_tokens=parsed.env_prefix_tokens,
        invocation_tokens=(python_executable, "-m", "pytest"),
        option_tokens=parsed.option_tokens,
        target_tokens=parsed.target_tokens,
    )
    return render_pytest_command(
        rewritten,
        disable_plugin_autoload=disable_plugin_autoload,
    )


def build_pytest_recovery_commands(
    command: str | None,
    *,
    repo_root: str | Path | None = None,
) -> list[str]:
    if not command:
        return []

    candidates: list[str] = []
    for python_executable in _discover_repo_python_executables(repo_root):
        repo_python = rewrite_pytest_command_with_python(
            command,
            python_executable,
            disable_plugin_autoload=False,
        )
        if repo_python:
            candidates.append(repo_python)

    if importlib.util.find_spec("pytest") is not None:
        current_python = rewrite_pytest_command_with_python(
            command,
            sys.executable,
            disable_plugin_autoload=False,
        )
        if current_python:
            candidates.append(current_python)

    uv_command = _build_uv_pytest_command(command, repo_root=repo_root)
    if uv_command:
        candidates.append(uv_command)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = candidate.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _discover_repo_python_executables(
    repo_root: str | Path | None,
) -> list[str]:
    if repo_root is None:
        return []
    root = Path(repo_root)
    if not root.exists():
        return []

    candidates = [
        root / ".venv" / "bin" / "python",
        root / ".venv" / "bin" / "python3",
        root / "venv" / "bin" / "python",
        root / "venv" / "bin" / "python3",
        root / "env" / "bin" / "python",
        root / "env" / "bin" / "python3",
        root / ".venv" / "Scripts" / "python.exe",
        root / "venv" / "Scripts" / "python.exe",
        root / "env" / "Scripts" / "python.exe",
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_text = str(candidate)
        if candidate_text in seen or not candidate.is_file():
            continue
        seen.add(candidate_text)
        deduped.append(candidate_text)
    return deduped


def output_indicates_missing_pytest(output: str) -> bool:
    lowered = str(output or "").lower()
    return (
        "no module named pytest" in lowered
        or "no module named 'pytest'" in lowered
        or 'no module named "pytest"' in lowered
    )


def should_disable_pytest_plugin_autoload(
    command: str,
    *,
    repo_root: str | Path | None = None,
) -> bool:
    parsed = parse_pytest_command(command)
    if parsed is None:
        return False
    if _tokens_require_plugin_autoload(parsed.option_tokens):
        return False

    config_text = _load_pytest_config_text(repo_root)
    if config_text and _text_requires_plugin_autoload(config_text):
        return False
    return True


def infer_additional_pytest_packages(
    command: str,
    *,
    repo_root: str | Path | None = None,
) -> list[str]:
    text = "\n".join(
        part
        for part in [
            str(command or ""),
            _load_pytest_config_text(repo_root),
        ]
        if part
    ).lower()
    if not text:
        return []

    inferred: list[str] = []
    if _text_contains_any(text, ("pytest-xdist", "--dist", " -n ", "'-n'", '"-n"')):
        inferred.append("pytest-xdist")
    if _text_contains_any(text, ("pytest-memray", "--memray")):
        inferred.append("pytest-memray")
    if _text_contains_any(text, ("pytest-timeout", "--timeout")):
        inferred.append("pytest-timeout")
    if _text_contains_any(text, ("pytest-rerunfailures", "--reruns")):
        inferred.append("pytest-rerunfailures")
    if _text_contains_any(text, ("pytest-cov", "--cov")):
        inferred.append("pytest-cov")

    benchmark_markers_present = _text_contains_benchmark_markers(text)
    if "pytest-benchmark" in text or benchmark_markers_present:
        inferred.append("pytest-benchmark")
    if "pytest-codspeed" in text or ("codspeed" in text and benchmark_markers_present):
        inferred.append("pytest-codspeed")

    return list(dict.fromkeys(inferred))


# Argv-length backstop. Passing thousands of node-ids as command-line targets can
# exceed the OS argument limit (E2BIG / "Argument list too long: 'bash'") — observed
# when a large expected-id subset (e.g. a decomposition module group carrying
# thousands of ids) is rendered as individual targets. When a selection is large by
# count OR by rendered size, collapse node-ids to their unique file paths so the argv
# stays bounded for ANY selection size. Callers that need exact per-id execution route
# through APEX_EXPECTED_IDS_FILE / _apex_run_expected_ids.py (which deselects
# non-expected ids); this is purely an overflow guard and a strict no-op for normal
# small selections. General (Layer A): "never emit an argv longer than the OS allows".
_ARGV_TARGET_COLLAPSE_COUNT = 256


def _argv_byte_budget() -> int:
    """Conservative ceiling for the rendered target portion of a pytest argv."""
    limit = 0
    try:
        limit = os.sysconf("SC_ARG_MAX")
    except (ValueError, OSError, AttributeError):
        limit = 0
    if not limit or limit < 0:
        limit = 256 * 1024  # POSIX-minimum floor when the platform won't say.
    # Halve it to leave headroom for the env block, invocation, and option tokens.
    return max(64 * 1024, int(limit * 0.5))


def _collapse_targets_to_files(targets: list[str]) -> list[str]:
    """Reduce pytest node-ids to their unique file paths, order-preserving.

    ``pkg/test_x.py::TestA::test_a`` -> ``pkg/test_x.py``. Tokens without a ``::``
    node separator (plain files / dirs / glob targets) pass through unchanged.
    """
    seen: set[str] = set()
    collapsed: list[str] = []
    for token in targets:
        head = token.split("::", 1)[0] if "::" in token else token
        if head not in seen:
            seen.add(head)
            collapsed.append(head)
    return collapsed


def render_pytest_command(
    parsed: ParsedPytestCommand,
    *,
    selected_tests: Optional[list[str]] = None,
    force_verbose: bool = False,
    disable_plugin_autoload: bool = True,
) -> str:
    option_tokens = list(parsed.option_tokens)
    if force_verbose and not any(
        token == "-v" or token.startswith("-v") for token in option_tokens
    ):
        option_tokens.append("-vv")
    if force_verbose and not any(
        token == "--tb=no" or token.startswith("--tb=") for token in option_tokens
    ):
        option_tokens.append("--tb=no")

    env_prefix_tokens = list(parsed.env_prefix_tokens)
    if disable_plugin_autoload and not any(
        token.startswith("PYTEST_DISABLE_PLUGIN_AUTOLOAD=") for token in env_prefix_tokens
    ):
        env_prefix_tokens.append("PYTEST_DISABLE_PLUGIN_AUTOLOAD=1")
        option_tokens = _strip_plugin_autoload_sensitive_options(option_tokens)

    target_tokens = list(parsed.target_tokens if selected_tests is None else selected_tests)
    # Argv-length backstop: a selection that is large by count OR by rendered byte
    # size collapses to unique file paths so the command can never exceed the OS
    # argv limit. No-op for normal small selections (and for file/dir targets,
    # which have no ``::`` to collapse).
    if target_tokens and (
        len(target_tokens) > _ARGV_TARGET_COLLAPSE_COUNT
        or sum(len(token) + 1 for token in target_tokens) > _argv_byte_budget()
    ):
        target_tokens = _collapse_targets_to_files(target_tokens)
    rendered_tokens = (
        list(parsed.shell_prefix_tokens)
        + env_prefix_tokens
        + list(parsed.invocation_tokens)
        + option_tokens
        + target_tokens
    )
    return " ".join(_render_shell_token(token) for token in rendered_tokens)


def _strip_plugin_autoload_sensitive_options(option_tokens: list[str]) -> list[str]:
    stripped: list[str] = []
    index = 0
    while index < len(option_tokens):
        token = option_tokens[index]
        if _is_plugin_autoload_sensitive_option(token):
            normalized = token.split("=", 1)[0]
            index += 1
            if (
                "=" not in token
                and normalized in _OPTIONS_WITH_VALUES
                and index < len(option_tokens)
            ):
                index += 1
            continue
        stripped.append(token)
        index += 1
    return stripped


def _parse_pytest_segment(
    segment: list[str],
) -> Optional[tuple[list[str], list[str], list[str]]]:
    if not segment:
        return None

    index = 0
    if segment[index] == "env":
        index += 1
        while index < len(segment) and _is_env_assignment(segment[index]):
            index += 1
    else:
        while index < len(segment) and _is_env_assignment(segment[index]):
            index += 1

    if index >= len(segment):
        return None

    executable = Path(segment[index]).name
    if executable.startswith("pytest"):
        return segment[:index], [segment[index]], segment[index + 1 :]
    if (
        executable.startswith("python")
        and index + 2 < len(segment)
        and segment[index + 1] == "-m"
        and segment[index + 2] == "pytest"
    ):
        return segment[:index], segment[index : index + 3], segment[index + 3 :]
    return None


def _split_pytest_args(remaining_tokens: list[str]) -> tuple[list[str], list[str]]:
    option_tokens: list[str] = []
    target_tokens: list[str] = []
    index = 0
    while index < len(remaining_tokens):
        token = remaining_tokens[index]
        if token == "--":
            target_tokens.extend(remaining_tokens[index + 1 :])
            break
        if token.startswith("-"):
            option_tokens.append(token)
            if "=" in token:
                index += 1
                continue
            if token in _OPTIONS_WITH_VALUES and index + 1 < len(remaining_tokens):
                option_tokens.append(remaining_tokens[index + 1])
                index += 2
                continue
            index += 1
            continue
        target_tokens.append(token)
        index += 1
    return option_tokens, target_tokens


def _derive_python_invocation(
    invocation_tokens: tuple[str, ...],
) -> Optional[list[str]]:
    if (
        len(invocation_tokens) >= 3
        and Path(invocation_tokens[0]).name.startswith("python")
        and invocation_tokens[1] == "-m"
        and invocation_tokens[2] == "pytest"
    ):
        return [invocation_tokens[0]]

    if len(invocation_tokens) != 1:
        return None

    executable = invocation_tokens[0]
    executable_name = Path(executable).name
    if not executable_name.startswith("pytest"):
        return None

    resolved = shutil.which(executable) or executable
    pytest_path = Path(resolved)
    candidates = _python_sibling_candidates(pytest_path.name)
    for candidate_name in candidates:
        candidate = pytest_path.with_name(candidate_name)
        if candidate.exists():
            return [str(candidate)]
    if pytest_path.is_absolute():
        return [str(pytest_path.with_name("python"))]
    return None


def _python_sibling_candidates(pytest_name: str) -> list[str]:
    suffix = pytest_name[len("pytest") :]
    candidates = []
    if suffix:
        candidates.extend([f"python{suffix}", f"python3{suffix}"])
        if suffix.startswith("-"):
            version = suffix[1:]
            candidates.extend([f"python{version}", f"python3.{version}"])
    candidates.extend(["python", "python3"])

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _build_uv_pytest_command(
    command: str,
    *,
    repo_root: str | Path | None = None,
) -> Optional[str]:
    parsed = parse_pytest_command(command)
    if parsed is None:
        return None

    uv_invocation = _resolve_uv_command_tokens()
    if uv_invocation is None:
        return None

    rendered_tokens = list(parsed.shell_prefix_tokens) + list(parsed.env_prefix_tokens)
    rendered_tokens.extend(uv_invocation)
    rendered_tokens.extend(["run", "--isolated", "--no-project"])
    for package in ["pytest", *infer_additional_pytest_packages(command, repo_root=repo_root)]:
        rendered_tokens.extend(["--with", package])
    rendered_tokens.extend(["python", "-m", "pytest"])
    rendered_tokens.extend(parsed.option_tokens)
    rendered_tokens.extend(parsed.target_tokens)
    return " ".join(_render_shell_token(token) for token in rendered_tokens)


def _resolve_uv_command_tokens() -> Optional[list[str]]:
    uv_binary = shutil.which("uv")
    if uv_binary:
        return [uv_binary]

    module_cmd = [sys.executable, "-m", "uv"]
    try:
        result = subprocess.run(
            [*module_cmd, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode == 0:
        return module_cmd
    return None


def _is_env_assignment(token: str) -> bool:
    return bool(_ENV_ASSIGNMENT_RE.match(token))


def _is_plugin_autoload_sensitive_option(token: str) -> bool:
    normalized = token.split("=", 1)[0]
    if normalized in _PLUGIN_AUTOLOAD_SENSITIVE_OPTIONS:
        return True
    return normalized.startswith("--benchmark-")


def _tokens_require_plugin_autoload(option_tokens: tuple[str, ...] | list[str]) -> bool:
    return any(
        token.startswith("-") and _is_plugin_autoload_sensitive_option(token)
        for token in option_tokens
    )


def _pytest_command_loads_plugin(
    option_tokens: tuple[str, ...] | list[str],
    plugin_name: str,
) -> bool:
    index = 0
    while index < len(option_tokens):
        token = str(option_tokens[index])
        if token == "-p" and index + 1 < len(option_tokens):
            if str(option_tokens[index + 1]) == plugin_name:
                return True
            index += 2
            continue
        if token.startswith("-p") and token[2:] == plugin_name:
            return True
        index += 1
    return False


def _load_pytest_config_text(repo_root: str | Path | None) -> str:
    if repo_root is None:
        return ""
    root = Path(repo_root)
    if not root.exists():
        return ""

    chunks: list[str] = []
    for filename in _PYTEST_CONFIG_FILENAMES:
        path = root / filename
        if not path.is_file():
            continue
        try:
            chunks.append(path.read_text(errors="replace"))
        except OSError:
            continue
    return "\n".join(chunks)


def _text_requires_plugin_autoload(text: str) -> bool:
    lowered = text.lower()
    return _text_contains_any(
        lowered,
        (
            "--cov",
            "--json-report",
            "--memray",
            "--reruns",
            "--timeout",
            "--tx",
            "pytest-benchmark",
            "pytest-codspeed",
            "pytest-cov",
            "pytest-json-report",
            "pytest-memray",
            "pytest-rerunfailures",
            "pytest-timeout",
            "pytest-xdist",
            "--dist",
            " -n ",
            "'-n'",
            '"-n"',
        ),
    ) or _text_contains_benchmark_markers(lowered)


def _text_contains_benchmark_markers(text: str) -> bool:
    return _text_contains_any(
        text,
        (
            "--benchmark",
            "--benchmark-",
            "pytest-benchmark",
            "pytest-codspeed",
        ),
    )


def _text_contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _render_shell_token(token: str) -> str:
    if token in _SHELL_OPERATORS:
        return token
    if _is_env_assignment(token):
        return token
    return shlex.quote(token)
