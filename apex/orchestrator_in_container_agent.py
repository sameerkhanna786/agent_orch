"""
APEX V5 target-runtime agent loop.

This module implements the planned-but-not-built V5 target-runtime agent
loop`` documented in memory ``project_agentic_in_container_loop.md``.

WHY A SEPARATE MODULE
---------------------
The existing :mod:`apex.orchestrator` performs single-shot patch generation:
plan -> rollouts -> selection -> ``ApexResult.patch``. The V5 loop is a
DIFFERENT shape: it wraps an LLM as an iterative agent that lives in the
workspace/target runtime, executing shell commands ("tools") and observing their output for
up to ``max_turns`` turns before either submitting a unified diff or giving
up. This file is intentionally separated so the loop can be evolved
independently of the existing orchestrator (which has many existing
consumers).

The plan's intended path was ``apex/orchestrator/in_container_agent.py``,
but ``apex/orchestrator.py`` already exists as a module file and the Phase 5
constraint explicitly forbids touching it. Putting the new code at
``apex/orchestrator_in_container_agent.py`` avoids the file-vs-package
collision while preserving the "separate module, independently evolvable"
intent. See ``tools/SWE_EVO_NOTES.md`` and the Phase 5 commit body for the
full rationale.

V1 LIMITATIONS (DOCUMENTED PROMINENTLY)
---------------------------------------
- **Target-runtime shims, not true container isolation.** The default
  SWE-EVO path routes tool commands through benchmark target-runtime shims
  when configured, but this module is not itself a Docker supervisor.
  Official SWE-EVO scoring remains the only authoritative success signal.
- **Hard timeout per shell call** is enforced with process-group teardown.
- **Environment scrubbing** is applied before caller-supplied runtime
  overrides are merged.

PUBLIC INTERFACE
----------------
- :class:`InContainerAgent`
- :func:`solve_in_container_agent`
- :class:`AgentTurn` (per-turn record kept for caller introspection)
- :class:`ToolCall` (parsed LLM-issued action)
- Constants ``TOOL_RUN_IN_CONTAINER``, ``TOOL_SUBMIT_PATCH``,
  ``TOOL_GIVE_UP``, ``DEFAULT_MAX_TURNS``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:  # pragma: no cover — type-only import
    from .core.container_supervisor import ContainerSupervisor

logger = logging.getLogger("apex.in_container_agent")


# ---------------------------------------------------------------------------
# Tool schema constants
# ---------------------------------------------------------------------------

TOOL_RUN_IN_CONTAINER = "run_in_container"
TOOL_SUBMIT_PATCH = "submit_patch"
TOOL_GIVE_UP = "give_up"

VALID_TOOLS = frozenset({TOOL_RUN_IN_CONTAINER, TOOL_SUBMIT_PATCH, TOOL_GIVE_UP})

DEFAULT_MAX_TURNS = 8
# Per-tool (single in-container command) budget for the V5 agent loop. A tool
# call can be a full test-suite run or repo analysis, so a 60s cap subordinated
# the agent to command overhead and killed legitimate long commands. Floor at
# 30min per agentic step; the per-task wallclock budget is the outer bound.
DEFAULT_TURN_TIMEOUT_SECONDS = 1800
DEFAULT_MAX_OUTPUT_BYTES_PER_TOOL = 16_000
DEFAULT_MAX_TOKENS_PER_TURN = 8000
# V5-TRANSCRIPT: how many of the most-recent completed turns are rendered
# VERBATIM in the working-memory window. Older turns degrade by deterministic
# whole-turn deletion / value-preserving compaction — NEVER LLM prose
# summarization (which, per arxiv 2605.26302 "compression aging", drops exact
# values and gives a finite ~7-17 session half-life vs append-only's infinite).
DEFAULT_RECENT_VERBATIM_TURNS = 3
# V5-NOPROGRESS: emit a one-time nudge after this many equivalent (command, rc)
# repeats, and terminate as no_progress once repeats reach the terminate cap.
DEFAULT_STALL_REPEAT_THRESHOLD = 3
DEFAULT_STALL_TERMINATE_CAP = 5
_TARGET_RUNTIME_ABSOLUTE_DYNAMIC_RE = re.compile(
    r"(^|[\s;&|])/(?:usr|opt|bin|sbin|usr/local|opt/homebrew)/[^\s;&|]*"
    r"(?:bash|sh|zsh|env|xargs|git|docker|curl|wget|perl|awk|make|python(?:[0-9.]+)?|pytest|py\.test|pip(?:[0-9.]+)?|uv|poetry|hatch|tox|nox|coverage|"
    r"django-admin|node|npm|npx|pnpm|yarn|bun|deno|go|cargo|mvn|gradle|java|ruby|bundle|php|dotnet|swift)\b"
)

# Common shell-wrapper prefixes that LLMs spontaneously emit (e.g.
# "/bin/zsh -c 'sed -i ...'"). Under target-runtime enforcement these
# trigger ``_TARGET_RUNTIME_ABSOLUTE_DYNAMIC_RE`` because of the absolute
# path to ``/bin/zsh``. Since the V1 host shim already wraps every
# command in ``bash -lc <command>``, the outer ``/bin/<shell> -c '...'``
# is redundant — we can safely strip it and pass the inner command
# straight through. This avoids a deadly false-positive policy violation
# without weakening the regex (which still catches absolute paths to
# *meaningful* tools like ``/usr/bin/python``).
_SHELL_WRAPPER_PREFIX_RE = re.compile(
    r"^\s*(?:/(?:usr/)?(?:local/)?bin/(?:bash|sh|zsh)|/opt/homebrew/bin/(?:bash|sh|zsh))"
    r"\s+-l?c\s+",
    re.IGNORECASE,
)


# V5-TAIL-OUTPUT: lines worth preserving verbatim even when output is over the
# byte cap — pytest puts the decisive FAILED/assert/Traceback lines at the TAIL,
# so a head-slice (the old behaviour) discarded exactly the evidence the agent
# needs. Keeping the tail + extracted error/failure lines is value-preserving
# write-stage discipline (cf. arxiv 2605.26302 "compression aging": lossy
# truncation that drops exact values like error strings causes capability decay).
_TOOL_OUTPUT_ERROR_LINE_RE = re.compile(
    r"(FAILED|ERROR\b|Error\b|Traceback|Exception|AssertionError|^E\s|^E$|"
    r"\bassert\b|\bfailed\b|error:|panic:|SIGSEGV|segmentation fault)",
    re.IGNORECASE | re.MULTILINE,
)


def _truncate_tool_output(text: str, max_bytes: int) -> tuple[str, bool]:
    """Truncate ``text`` to ~``max_bytes`` preserving the TAIL and any
    error/failure lines, returning ``(text, truncated)``.

    Unlike a head slice, this keeps the end of the output (where test runners
    report failures) plus error-regex lines extracted from the full body, so a
    long, noisy command still surfaces its decisive evidence. A small head is
    retained for the command echo/context.
    """
    if not max_bytes or len(text) <= max_bytes:
        return text, False
    head_budget = min(1024, max(0, max_bytes // 4))
    head = text[:head_budget]
    error_lines = [
        line for line in text.splitlines() if _TOOL_OUTPUT_ERROR_LINE_RE.search(line)
    ]
    error_blob = "\n".join(error_lines)
    error_budget = max(0, max_bytes // 3)
    if len(error_blob) > error_budget:
        error_blob = error_blob[-error_budget:]
    tail_budget = max(0, max_bytes - len(head) - len(error_blob) - 256)
    tail = text[-tail_budget:] if tail_budget > 0 else ""
    elided = max(0, len(text) - len(head) - len(tail))
    parts = [head, f"\n…[{elided} bytes elided — tail + error lines preserved]…\n"]
    # Only inject the extracted error section when those lines are not already
    # fully visible in the retained tail (avoid duplicating a small tail).
    if error_blob and error_blob not in tail:
        parts.append("--- extracted error/failure lines ---\n")
        parts.append(error_blob)
        parts.append("\n--- end extracted error/failure lines ---\n")
    parts.append(tail)
    return "".join(parts), True


def _strip_redundant_shell_wrapper(command: str) -> str:
    """Strip a leading ``/bin/<shell> -c '...'`` wrapper from ``command``.

    LLMs operating in shell-tool mode often produce commands of the form
    ``/bin/zsh -c "sed -i 's/foo/bar/' file.py"``. The host shim already
    runs every tool call through ``bash -lc <command>``, so the outer
    wrapper is redundant. Removing it lets the inner command be evaluated
    by the policy regex on its own merits (a bare ``sed`` is allowed; the
    ``/bin/zsh`` prefix is what trips the absolute-path rule).

    If the wrapper is absent or the inner argument can't be safely
    unquoted, ``command`` is returned unchanged.
    """
    if not command or not isinstance(command, str):
        return command
    match = _SHELL_WRAPPER_PREFIX_RE.match(command)
    if match is None:
        return command
    remainder = command[match.end() :].strip()
    if not remainder:
        return command
    # The remainder should be a single shell-quoted argument (the inner
    # command). ``shlex.split`` on the whole rest is the safest way to
    # collapse ``"foo bar"`` -> ``foo bar`` without re-evaluating any
    # nested expansions.
    try:
        parts = shlex.split(remainder, posix=True)
    except ValueError:
        return command
    if len(parts) != 1 or not parts[0].strip():
        return command
    return parts[0]


# JSON schema we hand the LLM. Kept tight — three flat tools, no nested
# structures. ``CLIModelClient.run_structured_prompt`` will serialise this
# as a structured-output constraint on backends that support it; on
# backends that ignore it we still parse the model's JSON manually.
TOOL_CALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool": {
            "type": "string",
            "enum": sorted(VALID_TOOLS),
        },
        "args": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600},
                "patch": {"type": "string"},
                "reason": {"type": "string"},
            },
            "additionalProperties": True,
        },
    },
    "required": ["tool", "args"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """One parsed LLM-issued tool call."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""

    def is_terminal(self) -> bool:
        return self.tool in (TOOL_SUBMIT_PATCH, TOOL_GIVE_UP)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Optional["ToolCall"]:
        if not isinstance(data, dict):
            return None
        tool = data.get("tool")
        if not tool:
            return None
        args = data.get("args")
        return cls(tool=str(tool), args=args if isinstance(args, dict) else {})


@dataclass
class ToolResult:
    """Result of executing a ``run_in_container`` tool call."""

    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    timed_out: bool = False
    duration_seconds: float = 0.0
    truncated: bool = False
    error: Optional[str] = None  # populated on internal harness errors

    def to_summary_for_prompt(self) -> str:
        parts = [
            f"return_code: {self.return_code}",
            f"duration_seconds: {self.duration_seconds:.2f}",
            f"timed_out: {self.timed_out}",
        ]
        if self.error:
            parts.append(f"harness_error: {self.error}")
        if self.truncated:
            parts.append("output_truncated: true (per-tool byte cap hit)")
        body = "\n".join(parts)
        return f"{body}\n\n--- stdout ---\n{self.stdout}\n--- stderr ---\n{self.stderr}"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Optional["ToolResult"]:
        if not isinstance(data, dict):
            return None
        return cls(
            stdout=str(data.get("stdout") or ""),
            stderr=str(data.get("stderr") or ""),
            return_code=int(data.get("return_code") or 0),
            timed_out=bool(data.get("timed_out")),
            duration_seconds=float(data.get("duration_seconds") or 0.0),
            truncated=bool(data.get("truncated")),
            error=data.get("error"),
        )


@dataclass
class AgentTurn:
    """One turn (LLM call + tool dispatch)."""

    turn_index: int
    prompt: str
    llm_raw_response: str = ""
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None
    parse_error: Optional[str] = None
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "prompt_chars": len(self.prompt),
            "llm_raw_response_chars": len(self.llm_raw_response),
            "tool": self.tool_call.tool if self.tool_call else None,
            "tool_args_keys": (sorted(self.tool_call.args.keys()) if self.tool_call else []),
            "return_code": (self.tool_result.return_code if self.tool_result else None),
            "timed_out": (self.tool_result.timed_out if self.tool_result else None),
            "parse_error": self.parse_error,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
        }

    def to_transcript_record(self) -> dict[str, Any]:
        """1B: LOSSLESS durable record for the transcript sink (superset of
        ``to_dict``). Unlike ``to_dict`` (telemetry only — prompt_chars, no
        tool bodies), this preserves the exact command/patch and the tool-result
        body so ``preload_transcript`` can faithfully rehydrate working memory
        after a crash/restart (no compression aging on resume)."""
        record: dict[str, Any] = {
            "v": 1,
            "turn_index": self.turn_index,
            "tool": self.tool_call.tool if self.tool_call else None,
            "parse_error": self.parse_error,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
        }
        if self.tool_call is not None:
            record["command"] = self.tool_call.args.get("command")
            if self.tool_call.tool == TOOL_SUBMIT_PATCH:
                record["patch"] = self.tool_call.args.get("patch")
            record["tool_args"] = dict(self.tool_call.args)
        if self.tool_result is not None:
            record["tool_result"] = {
                "stdout": self.tool_result.stdout,
                "stderr": self.tool_result.stderr,
                "return_code": self.tool_result.return_code,
                "timed_out": self.tool_result.timed_out,
                "duration_seconds": self.tool_result.duration_seconds,
                "truncated": self.tool_result.truncated,
                "error": self.tool_result.error,
            }
        return record

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Optional["AgentTurn"]:
        """Rehydrate an AgentTurn from a ``to_transcript_record`` dict."""
        if not isinstance(data, dict):
            return None
        try:
            turn_index = int(data.get("turn_index"))
        except (TypeError, ValueError):
            return None
        tool = data.get("tool")
        tool_call: Optional[ToolCall] = None
        if tool:
            args = data.get("tool_args")
            if not isinstance(args, dict):
                args = {}
                if data.get("command") is not None:
                    args["command"] = data.get("command")
                if data.get("patch") is not None:
                    args["patch"] = data.get("patch")
            tool_call = ToolCall(tool=str(tool), args=args)
        tool_result = ToolResult.from_dict(data["tool_result"]) if data.get("tool_result") else None
        return cls(
            turn_index=turn_index,
            prompt="",
            tool_call=tool_call,
            tool_result=tool_result,
            parse_error=data.get("parse_error"),
            elapsed_seconds=float(data.get("elapsed_seconds") or 0.0),
        )


@dataclass
class PatchVerification:
    """V5-VERIFY-GATE: outcome of applying + re-running tests on a submitted patch.

    ``passed`` is the authoritative signal; ``failure_excerpt`` carries the
    decisive (tail-biased) failure lines fed back to the agent so it keeps
    iterating instead of the harness trusting a remembered "it should pass".
    """

    applied: bool
    passed: bool
    failure_excerpt: str = ""


@dataclass
class AgentRunSummary:
    """Caller-facing summary of one ``InContainerAgent.solve`` invocation."""

    final_patch: Optional[str]
    terminated_reason: (
        str  # "submit_patch" | "give_up" | "max_turns" | "parse_failure" | "llm_failure"
    )
    give_up_reason: Optional[str] = None
    turns: list[AgentTurn] = field(default_factory=list)
    total_elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_patch_chars": len(self.final_patch or ""),
            "terminated_reason": self.terminated_reason,
            "give_up_reason": self.give_up_reason,
            "turn_count": len(self.turns),
            "total_elapsed_seconds": round(self.total_elapsed_seconds, 4),
            "turns": [t.to_dict() for t in self.turns],
        }


# ---------------------------------------------------------------------------
# Defaults: a no-op LLM caller used by tests that supply their own driver.
# ---------------------------------------------------------------------------


# Type alias: an LLMCaller takes (prompt, schema) -> raw_response_text.
LLMCaller = Callable[[str, dict[str, Any]], str]


def _build_default_llm_caller(llm_config: Any, *, working_dir: str) -> LLMCaller:
    """Build an LLMCaller backed by ``CLIModelClient.run_structured_prompt``.

    Imported lazily so this module can be unit-tested without pulling in
    the full CLI backend stack. Tests that mock the LLM should pass their
    own ``llm_caller`` to :class:`InContainerAgent`.
    """
    if llm_config is None:
        raise ValueError("InContainerAgent requires an llm_config or explicit llm_caller")

    from .core.cli_backend import CLIModelClient  # local import on purpose

    client = CLIModelClient(llm_config)

    def _call(prompt: str, schema: dict[str, Any]) -> str:
        result = client.run_structured_prompt(
            prompt=prompt,
            working_dir=working_dir,
            schema=schema,
            allow_edits=False,
            internet_enabled=False,
        )
        if not result.success:
            raise RuntimeError(f"in_container_agent: LLM call failed: {result.error or 'unknown'}")
        if isinstance(result.parsed_json, dict):
            return json.dumps(result.parsed_json)
        return result.text or result.raw_output or ""

    return _call


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    """Find the first JSON object in ``text``.

    Tolerates Markdown code fences, surrounding prose, and trailing
    punctuation. Returns ``None`` on parse failure.
    """
    if not text:
        return None
    candidates: list[str] = []
    # 1) explicit fenced blocks
    for match in _FENCE_RE.finditer(text):
        candidates.append(match.group(1).strip())
    # 2) full text + first/last brace slice
    candidates.append(text.strip())
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(text[first : last + 1])
    seen: set[str] = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _parse_tool_call_from_llm_response(raw: str) -> tuple[Optional[ToolCall], Optional[str]]:
    """Parse an LLM response into a :class:`ToolCall`.

    Returns ``(tool_call, error_message)``. On parse failure, ``tool_call``
    is ``None`` and ``error_message`` describes the issue (suitable for
    feeding back into the next turn so the LLM can self-correct).
    """
    obj = _extract_json_object(raw)
    if obj is None:
        return None, (
            "Could not extract a JSON object from your response. "
            "Reply with a single JSON object matching the tool schema."
        )
    tool = obj.get("tool")
    args = obj.get("args")
    if not isinstance(tool, str) or tool not in VALID_TOOLS:
        return None, (f"Field 'tool' must be one of {sorted(VALID_TOOLS)}; got {tool!r}.")
    if not isinstance(args, dict):
        args = {}
    return ToolCall(tool=tool, args=args, raw_response=raw), None


# ---------------------------------------------------------------------------
# Sandbox shell execution (V1: cwd-only, no real container)
# ---------------------------------------------------------------------------


def _execute_in_workspace(
    command: str,
    *,
    workspace_dir: str,
    timeout_seconds: int,
    max_output_bytes: int,
    env_overrides: Optional[dict[str, str]] = None,
    container_supervisor: Optional["ContainerSupervisor"] = None,
) -> ToolResult:
    """Execute ``command`` inside ``workspace_dir`` with a hard timeout.

    When ``container_supervisor`` is provided, the command is dispatched
    through ``supervisor.run_in_container()`` and runs inside a real
    docker container with the workspace bind-mounted at ``/workspace``.
    Otherwise (V1 mode) we fall back to ``bash -lc`` on the host with the
    ``workspace_dir`` pinned as cwd.

    Uses ``shell=True`` to mirror the LLM's natural shell-string output. The
    cwd is pinned to ``workspace_dir``; the LLM may still ``cd`` elsewhere
    via shell — this is acceptable for V1 since the host has no isolation
    anyway. See module docstring for V1 limitations.
    """
    if not command or not isinstance(command, str):
        return ToolResult(error="empty_or_non_string_command")
    # Strip redundant ``/bin/<shell> -c '...'`` wrappers the LLM emits.
    # The host shim re-wraps in ``bash -lc`` so removing the inner one is
    # behaviour-preserving but avoids a false-positive policy violation
    # under target-runtime enforcement (the absolute-path regex would
    # otherwise reject the unstripped form).
    command = _strip_redundant_shell_wrapper(command)
    if container_supervisor is not None:
        return _execute_via_container_supervisor(
            command,
            supervisor=container_supervisor,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            env_overrides=env_overrides,
        )
    if not os.path.isdir(workspace_dir):
        return ToolResult(error=f"workspace_dir does not exist: {workspace_dir}")
    if env_overrides and env_overrides.get("APEX_TARGET_TOOL_CONTEXT"):
        if _TARGET_RUNTIME_ABSOLUTE_DYNAMIC_RE.search(command):
            return ToolResult(
                error=(
                    "target_runtime_policy_violation: absolute host dynamic tool "
                    "paths are disabled; use bare command names like 'sed', "
                    "'grep', 'python' so PATH resolution picks the target shim"
                )
            )

    from .core.cli_backend import redact_host_secrets
    from .core.subprocess_utils import terminate_process_tree

    start = time.time()
    env, _removed = redact_host_secrets(os.environ.copy())
    if env_overrides:
        env.update({str(k): str(v) for k, v in env_overrides.items()})
    process: Optional[subprocess.Popen[str]] = None
    timed_out = False
    stdout_text = ""
    stderr_text = ""
    return_code = -1
    try:
        process = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=workspace_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        try:
            stdout_text, stderr_text = process.communicate(timeout=timeout_seconds)
            return_code = process.returncode if process.returncode is not None else -1
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                stdout_text, stderr_text = terminate_process_tree(process)
            except Exception:
                logger.warning(
                    "in_container_agent: failed to terminate timed-out subprocess tree pid=%s",
                    getattr(process, "pid", None),
                )
                stdout_text = stdout_text or ""
                stderr_text = stderr_text or ""
            return_code = -9
    except OSError as exc:
        return ToolResult(
            error=f"subprocess_spawn_error: {exc}",
            duration_seconds=time.time() - start,
        )

    stdout_text, stdout_trunc = _truncate_tool_output(stdout_text, max_output_bytes)
    stderr_text, stderr_trunc = _truncate_tool_output(stderr_text, max_output_bytes)
    truncated = stdout_trunc or stderr_trunc

    return ToolResult(
        stdout=stdout_text,
        stderr=stderr_text,
        return_code=return_code,
        timed_out=timed_out,
        duration_seconds=time.time() - start,
        truncated=truncated,
    )


def _execute_via_container_supervisor(
    command: str,
    *,
    supervisor: "ContainerSupervisor",
    timeout_seconds: int,
    max_output_bytes: int,
    env_overrides: Optional[dict[str, str]] = None,
) -> ToolResult:
    """Run ``command`` inside the supervised container.

    The supervisor handles namespace isolation, network policy, and
    cleanup. We mirror the truncation / timeout / error semantics of the
    V1 host shim so callers see a uniform :class:`ToolResult` shape.
    """
    start = time.time()
    try:
        completed = supervisor.run_in_container(
            command,
            timeout=float(timeout_seconds),
            env=env_overrides,
        )
    except Exception as exc:  # pragma: no cover — defensive
        return ToolResult(
            error=f"container_supervisor_error: {exc}",
            duration_seconds=time.time() - start,
        )
    stdout_text = completed.stdout or ""
    stderr_text = completed.stderr or ""
    return_code = int(completed.returncode)
    timed_out = return_code == -9
    stdout_text, stdout_trunc = _truncate_tool_output(stdout_text, max_output_bytes)
    stderr_text, stderr_trunc = _truncate_tool_output(stderr_text, max_output_bytes)
    truncated = stdout_trunc or stderr_trunc
    return ToolResult(
        stdout=stdout_text,
        stderr=stderr_text,
        return_code=return_code,
        timed_out=timed_out,
        duration_seconds=time.time() - start,
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# The main agent class
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = (
    "You are an autonomous coding agent working INSIDE a software repository. "
    "Each turn you must reply with EXACTLY ONE JSON object selecting one tool:\n"
    f"  - {TOOL_RUN_IN_CONTAINER}: run a shell command in the workspace; "
    '   args = {"command": str, "timeout_seconds": int (optional)}\n'
    f"  - {TOOL_SUBMIT_PATCH}: emit your final unified-diff patch; "
    '   args = {"patch": str (full unified diff)}\n'
    f'  - {TOOL_GIVE_UP}: abort with a reason; args = {{"reason": str}}\n'
    "\n"
    "SHELL COMMAND RULES:\n"
    "- Use bare command names (e.g. 'sed -i ...', 'grep -rn ...', 'python -m pytest'). "
    "Do NOT prefix commands with absolute interpreter paths like '/bin/zsh -c', "
    "'/bin/bash -c', '/usr/bin/sed', or '/usr/local/bin/python'. PATH resolution "
    "is already correct and absolute host paths are blocked under target-runtime "
    "enforcement.\n"
    "- The command string is executed by the host shim; you do not need to add an "
    "outer shell wrapper.\n"
    "\n"
    "Reply with ONLY the JSON object, no commentary, no Markdown fence."
)


class InContainerAgent:
    """Iterative LLM-as-agent loop with ``run_in_container`` tool access.

    For each turn:
      1. Render context (current state of workspace, last command output,
         problem statement).
      2. LLM produces either a tool call (``run_in_container`` with a
         shell command) or a final patch.
      3. Execute the tool call, capture stdout/stderr/rc, append to
         context.
      4. Repeat up to ``max_turns`` (default 8) or until the LLM emits a
         final patch / give-up.

    Returns a unified-diff patch on success, ``None`` on giveup or
    irrecoverable parse failure.
    """

    def __init__(
        self,
        *,
        llm_config: Any = None,
        workspace_dir: str,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_tokens_per_turn: int = DEFAULT_MAX_TOKENS_PER_TURN,
        per_tool_timeout_seconds: int = DEFAULT_TURN_TIMEOUT_SECONDS,
        max_tool_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES_PER_TOOL,
        llm_caller: Optional[LLMCaller] = None,
        env_overrides: Optional[dict[str, str]] = None,
        container_supervisor: Optional["ContainerSupervisor"] = None,
        recent_verbatim_turns: int = DEFAULT_RECENT_VERBATIM_TURNS,
        stall_repeat_threshold: int = DEFAULT_STALL_REPEAT_THRESHOLD,
        stall_terminate_cap: int = DEFAULT_STALL_TERMINATE_CAP,
        patch_verifier: Optional[Callable[[str], "PatchVerification"]] = None,
        patch_verifier_reject_cap: int = 3,
        transcript_sink: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if not workspace_dir:
            raise ValueError("workspace_dir is required")
        self.llm_config = llm_config
        self.workspace_dir = workspace_dir
        self.max_turns = int(max_turns)
        self.max_tokens_per_turn = int(max_tokens_per_turn)
        self.per_tool_timeout_seconds = int(per_tool_timeout_seconds)
        self.max_tool_output_bytes = int(max_tool_output_bytes)
        self.env_overrides = dict(env_overrides or {})
        self._llm_caller = llm_caller  # lazy-built below if None
        self._turns: list[AgentTurn] = []
        self.recent_verbatim_turns = max(1, int(recent_verbatim_turns))
        self.stall_repeat_threshold = max(2, int(stall_repeat_threshold))
        self.stall_terminate_cap = max(self.stall_repeat_threshold, int(stall_terminate_cap))
        # V5-VERIFY-GATE: optional callback that applies a submitted patch and
        # re-runs the tests, returning a PatchVerification. When set, submit_patch
        # is NOT accepted on trust — the patch's effect is RECOMPUTED (the fix for
        # revision/utilization aging, Findings III/IV of arxiv 2605.26302).
        self.patch_verifier = patch_verifier
        self.patch_verifier_reject_cap = max(1, int(patch_verifier_reject_cap))
        # L2: optional durable transcript sink (e.g. append-only JSONL writer);
        # each completed turn is tee'd so the working memory survives a crash.
        self.transcript_sink = transcript_sink
        self._patch_reject_count = 0
        self._pending_feedback: Optional[str] = None
        # Optional true-container supervisor. When provided, every
        # ``run_in_container`` tool call dispatches through ``docker exec``
        # rather than the V1 host bash shim. See the module docstring's
        # V1 LIMITATIONS section for context.
        self.container_supervisor = container_supervisor
        if container_supervisor is not None:
            logger.info(
                "InContainerAgent: container supervisor active (image=%s, network=%s)",
                container_supervisor.image,
                container_supervisor.network,
            )
        else:
            logger.info(
                "InContainerAgent: no container supervisor; falling back to "
                "host bash shim (V1 mode — not real isolation)"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, problem_statement: str) -> Optional[str]:
        """Run the loop. Returns the final unified-diff patch or ``None``."""
        return self.solve_with_summary(problem_statement).final_patch

    def solve_with_summary(self, problem_statement: str) -> AgentRunSummary:
        """Run the loop and return the full summary (introspectable)."""
        if not isinstance(problem_statement, str) or not problem_statement.strip():
            raise ValueError("problem_statement must be a non-empty string")

        caller = self._llm_caller or _build_default_llm_caller(
            self.llm_config, working_dir=self.workspace_dir
        )

        run_started = time.time()
        last_tool_result: Optional[ToolResult] = None
        last_tool_call: Optional[ToolCall] = None
        last_parse_error: Optional[str] = None

        # 1B: resume after any preloaded turns (durable transcript restored after
        # a crash/reap). When self._turns is empty this is a strict no-op (start
        # at turn 1), so the non-preload path is unchanged.
        resume_from = len(self._turns) + 1
        for turn_index in range(resume_from, self.max_turns + 1):
            turn_started = time.time()
            stall_nudge = self._max_command_repeat() >= self.stall_repeat_threshold
            prompt = self._render_turn_prompt(
                problem_statement=problem_statement,
                turn_index=turn_index,
                last_parse_error=last_parse_error,
                stall_nudge=stall_nudge,
                verification_feedback=self._pending_feedback,
            )
            self._pending_feedback = None
            try:
                raw_response = caller(prompt, TOOL_CALL_SCHEMA)
            except Exception as exc:
                logger.warning(
                    "in_container_agent: LLM call raised on turn %d: %s",
                    turn_index,
                    exc,
                )
                turn = AgentTurn(
                    turn_index=turn_index,
                    prompt=prompt,
                    parse_error=f"llm_call_exception: {exc}",
                    elapsed_seconds=time.time() - turn_started,
                )
                self._turns.append(turn)
                return AgentRunSummary(
                    final_patch=None,
                    terminated_reason="llm_failure",
                    give_up_reason=str(exc),
                    turns=list(self._turns),
                    total_elapsed_seconds=time.time() - run_started,
                )

            tool_call, parse_error = _parse_tool_call_from_llm_response(raw_response)
            turn = AgentTurn(
                turn_index=turn_index,
                prompt=prompt,
                llm_raw_response=raw_response,
                tool_call=tool_call,
                parse_error=parse_error,
            )
            self._turns.append(turn)

            if tool_call is None:
                last_parse_error = parse_error
                last_tool_call = None
                last_tool_result = None
                turn.elapsed_seconds = time.time() - turn_started
                # Give the LLM another chance to self-correct on the next turn,
                # unless we're already out of turns. Otherwise drop through.
                if turn_index >= self.max_turns:
                    return AgentRunSummary(
                        final_patch=None,
                        terminated_reason="parse_failure",
                        give_up_reason=parse_error,
                        turns=list(self._turns),
                        total_elapsed_seconds=time.time() - run_started,
                    )
                continue

            last_parse_error = None  # we got a clean parse this turn

            if tool_call.tool == TOOL_SUBMIT_PATCH:
                patch = tool_call.args.get("patch", "")
                if not isinstance(patch, str) or not patch.strip():
                    last_parse_error = "submit_patch.args.patch must be a non-empty string"
                    last_tool_call = tool_call
                    last_tool_result = None
                    turn.elapsed_seconds = time.time() - turn_started
                    if turn_index >= self.max_turns:
                        return AgentRunSummary(
                            final_patch=None,
                            terminated_reason="parse_failure",
                            give_up_reason=last_parse_error,
                            turns=list(self._turns),
                            total_elapsed_seconds=time.time() - run_started,
                        )
                    continue
                # V5-VERIFY-GATE: do not accept a submitted patch on trust —
                # RECOMPUTE its effect by applying it and re-running the tests.
                # This is the forced re-read / derived-state recomputation that
                # fixes revision/utilization aging (Findings III/IV): a "confident
                # but wrong" submit is caught and fed back for another iteration.
                if self.patch_verifier is not None:
                    try:
                        verification = self.patch_verifier(patch)
                    except Exception as exc:  # noqa: BLE001 - verifier must not crash the loop
                        verification = PatchVerification(
                            applied=False, passed=False, failure_excerpt=f"verifier_error: {exc}"
                        )
                    turn.tool_result = ToolResult(
                        stdout=verification.failure_excerpt,
                        return_code=0 if verification.passed else 1,
                        error=None if verification.passed else "patch_verification_failed",
                    )
                    turn.elapsed_seconds = time.time() - turn_started
                    self._emit_transcript(turn)
                    if verification.passed:
                        return AgentRunSummary(
                            final_patch=patch,
                            terminated_reason="submit_patch_verified",
                            turns=list(self._turns),
                            total_elapsed_seconds=time.time() - run_started,
                        )
                    self._patch_reject_count += 1
                    if (
                        self._patch_reject_count >= self.patch_verifier_reject_cap
                        or turn_index >= self.max_turns
                    ):
                        return AgentRunSummary(
                            final_patch=None,
                            terminated_reason="verification_failed",
                            give_up_reason=(verification.failure_excerpt or "")[:500],
                            turns=list(self._turns),
                            total_elapsed_seconds=time.time() - run_started,
                        )
                    self._pending_feedback = (
                        "Your submitted patch did NOT pass the tests. Do not resubmit the same "
                        "patch; inspect the failures below and keep fixing.\n"
                        + (verification.failure_excerpt or "")
                    )
                    continue
                turn.elapsed_seconds = time.time() - turn_started
                return AgentRunSummary(
                    final_patch=patch,
                    terminated_reason="submit_patch",
                    turns=list(self._turns),
                    total_elapsed_seconds=time.time() - run_started,
                )

            if tool_call.tool == TOOL_GIVE_UP:
                reason = str(tool_call.args.get("reason") or "")
                turn.elapsed_seconds = time.time() - turn_started
                return AgentRunSummary(
                    final_patch=None,
                    terminated_reason="give_up",
                    give_up_reason=reason,
                    turns=list(self._turns),
                    total_elapsed_seconds=time.time() - run_started,
                )

            # tool_call.tool == TOOL_RUN_IN_CONTAINER
            tool_result = self._execute_tool_call(tool_call)
            turn.tool_result = tool_result
            last_tool_call = tool_call
            last_tool_result = tool_result
            turn.elapsed_seconds = time.time() - turn_started
            self._emit_transcript(turn)
            # V5-NOPROGRESS: if the agent has run an equivalent (command, rc)
            # enough times, it is stuck — terminate rather than burn the remaining
            # budget thrashing. The nudge (rendered next turn) fires earlier.
            if self._max_command_repeat() >= self.stall_terminate_cap:
                return AgentRunSummary(
                    final_patch=None,
                    terminated_reason="no_progress",
                    give_up_reason=(
                        "repeated an equivalent command "
                        f"{self.stall_terminate_cap}+ times with no change"
                    ),
                    turns=list(self._turns),
                    total_elapsed_seconds=time.time() - run_started,
                )

        # Loop exited without a terminal action: cap hit.
        return AgentRunSummary(
            final_patch=None,
            terminated_reason="max_turns",
            turns=list(self._turns),
            total_elapsed_seconds=time.time() - run_started,
        )

    # ------------------------------------------------------------------
    # Internal helpers (exposed for unit-test introspection).
    # ------------------------------------------------------------------

    def _execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        if tool_call.tool != TOOL_RUN_IN_CONTAINER:
            return ToolResult(error=f"unsupported_tool: {tool_call.tool}")
        command = tool_call.args.get("command")
        timeout_seconds = tool_call.args.get("timeout_seconds", self.per_tool_timeout_seconds)
        try:
            timeout_seconds = int(timeout_seconds)
        except (TypeError, ValueError):
            timeout_seconds = self.per_tool_timeout_seconds
        if timeout_seconds <= 0 or timeout_seconds > 600:
            timeout_seconds = self.per_tool_timeout_seconds
        return _execute_in_workspace(
            str(command or ""),
            workspace_dir=self.workspace_dir,
            timeout_seconds=timeout_seconds,
            max_output_bytes=self.max_tool_output_bytes,
            env_overrides=self.env_overrides,
            container_supervisor=self.container_supervisor,
        )

    # ------------------------------------------------------------------
    # V5-TRANSCRIPT: decay-resistant working memory
    # ------------------------------------------------------------------

    def _completed_tool_turns(self) -> list[AgentTurn]:
        return [t for t in self._turns if t.tool_call is not None and t.tool_result is not None]

    def _command_signature(self, turn: AgentTurn) -> Optional[str]:
        if turn.tool_call is None or turn.tool_call.tool != TOOL_RUN_IN_CONTAINER:
            return None
        command = str(turn.tool_call.args.get("command") or "").strip()
        if not command:
            return None
        rc = turn.tool_result.return_code if turn.tool_result else None
        timed_out = turn.tool_result.timed_out if turn.tool_result else None
        return f"{command}\x00{rc}\x00{timed_out}"

    def _max_command_repeat(self) -> int:
        """Highest repeat count of any (command, rc, timed_out) signature."""
        counts: dict[str, int] = {}
        for turn in self._turns:
            sig = self._command_signature(turn)
            if sig is None:
                continue
            counts[sig] = counts.get(sig, 0) + 1
        return max(counts.values()) if counts else 0

    def _derived_state_sidecar(self) -> list[str]:
        """L3: typed, machine-maintained derived state (revision-aging fix).

        These exact counters are kept verbatim rather than left for the model to
        re-derive from prose (which drifts). Rendered every turn.
        """
        completed = [
            t
            for t in self._completed_tool_turns()
            if t.tool_call is not None and t.tool_call.tool == TOOL_RUN_IN_CONTAINER
        ]
        distinct = len({self._command_signature(t) for t in completed if self._command_signature(t)})
        last_rc = completed[-1].tool_result.return_code if completed else None
        lines = [
            "## Progress state (machine-maintained, exact — trust over memory)",
            f"commands_run: {len(completed)}",
            f"distinct_commands: {distinct}",
            f"last_return_code: {last_rc}",
        ]
        repeat = self._max_command_repeat()
        if repeat >= 2:
            lines.append(f"max_identical_command_repeats: {repeat}")
        return lines

    def _render_completed_turn(self, turn: AgentTurn, *, verbatim: bool) -> str:
        tool = turn.tool_call.tool if turn.tool_call else ""
        if tool == TOOL_SUBMIT_PATCH:
            ok = bool(turn.tool_result and turn.tool_result.return_code == 0)
            status = "PASSED" if ok else "FAILED verification"
            body = (turn.tool_result.stdout if turn.tool_result else "") or ""
            return f"### Turn {turn.turn_index}: submit_patch -> {status}\n{body}".rstrip()
        command = str(turn.tool_call.args.get("command") or "") if turn.tool_call else ""
        rc = turn.tool_result.return_code if turn.tool_result else None
        if verbatim:
            return f"### Turn {turn.turn_index}\n$ {command}\n{turn.tool_result.to_summary_for_prompt()}"
        # Value-preserving COMPACT line (no prose summary): exact command + rc +
        # first error/failure line. The full output stays recoverable by re-running.
        timed = ", timed_out" if (turn.tool_result and turn.tool_result.timed_out) else ""
        err = ""
        if turn.tool_result is not None:
            combined = (turn.tool_result.stdout or "") + "\n" + (turn.tool_result.stderr or "")
            for line in combined.splitlines():
                if _TOOL_OUTPUT_ERROR_LINE_RE.search(line):
                    err = " | " + line.strip()[:200]
                    break
        first = command.splitlines()[0] if command else ""
        return f"### Turn {turn.turn_index} (compacted): $ {first} -> rc={rc}{timed}{err}"

    def _render_transcript_window(self) -> str:
        """Append-only, value-preserving working-memory window.

        Most-recent ``recent_verbatim_turns`` turns render verbatim; older turns
        render as value-preserving compact lines. When the window exceeds its
        budget we DELETE whole older turns (oldest first) — never LLM-summarize —
        so exact identifiers (paths, line numbers, error strings) are kept
        verbatim or dropped cleanly, never genericized (the compression-aging fix).
        """
        completed = self._completed_tool_turns()
        if not completed:
            return ""
        recent = self.recent_verbatim_turns
        recent_turns = completed[-recent:]
        older_turns = completed[:-recent] if len(completed) > recent else []
        verbatim_text = "\n\n".join(
            self._render_completed_turn(t, verbatim=True) for t in recent_turns
        )
        budget = max(2048, int(self.max_tokens_per_turn * 3))
        remaining = budget - len(verbatim_text)
        kept_older: list[str] = []
        dropped = 0
        for turn in reversed(older_turns):  # newest-older first
            line = self._render_completed_turn(turn, verbatim=False)
            if remaining - len(line) - 1 <= 0:
                dropped += 1
                continue
            kept_older.append(line)
            remaining -= len(line) + 1
        kept_older.reverse()
        parts = [
            "## Working memory — your prior turns (append-only; exact commands & "
            "errors preserved; re-run any command to re-observe its full output)"
        ]
        if dropped:
            parts.append(
                f"[{dropped} earlier turn(s) elided to fit the window — their commands are "
                "recoverable by re-running; NOTHING was summarized or genericized]"
            )
        parts.extend(kept_older)
        if verbatim_text:
            parts.append(verbatim_text)
        return "\n".join(parts)

    def _emit_transcript(self, turn: AgentTurn) -> None:
        if self.transcript_sink is None:
            return
        try:
            # Lossless superset (command/patch + tool-result body) so a restart
            # can faithfully rehydrate working memory via preload_transcript.
            self.transcript_sink(turn.to_transcript_record())
        except Exception:  # noqa: BLE001 - durable tee is best-effort
            logger.debug("in_container_agent: transcript_sink raised", exc_info=True)

    def reset_run_state(self) -> None:
        """3G: clear per-run state so the agent can be re-used for an independent
        task/subtask without the prior run's turns bleeding into working memory,
        stall detection, or the verify-gate reject counter. Used by
        HierarchicalAgent, which reuses one agent across decomposed subtasks."""
        self._turns = []
        self._patch_reject_count = 0
        self._pending_feedback = None

    def preload_transcript(self, path: Optional[str]) -> int:
        """1B: rehydrate prior turns from a durable JSONL transcript on restart.

        Tolerant parse (skips blank/unparseable/version-mismatch lines). Appends
        each rehydrated AgentTurn to ``self._turns`` (so the solve loop resumes
        after them and working memory survives a crash/container reap) and
        recomputes ``self._patch_reject_count`` from any failed submit_patch
        records so the reject cap is not reset by the restart. Returns the number
        of turns preloaded. No-op (returns 0) when ``path`` is missing/empty.
        """
        if not path:
            return 0
        try:
            file_path = Path(path)
            if not file_path.exists():
                return 0
            text = file_path.read_text(encoding="utf-8")
        except OSError:
            return 0
        loaded = 0
        for raw in text.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or int(record.get("v") or 0) != 1:
                continue
            turn = AgentTurn.from_dict(record)
            if turn is None:
                continue
            self._turns.append(turn)
            loaded += 1
            if (
                turn.tool_call is not None
                and turn.tool_call.tool == TOOL_SUBMIT_PATCH
                and turn.tool_result is not None
                and turn.tool_result.return_code != 0
            ):
                self._patch_reject_count += 1
        return loaded

    def _render_turn_prompt(
        self,
        *,
        problem_statement: str,
        turn_index: int,
        last_parse_error: Optional[str] = None,
        stall_nudge: bool = False,
        verification_feedback: Optional[str] = None,
    ) -> str:
        # Reserved (NEVER-truncated) sections: system prompt, goal recitation,
        # derived-state sidecar, and the response instructions. Only the
        # working-memory transcript window is budgeted/degraded, and only by
        # deterministic deletion — never positional middle-splice of the whole
        # prompt (the old bug) and never LLM summarization.
        head: list[str] = [
            _SYSTEM_PROMPT,
            "",
            "## Problem statement (re-read every turn — do not drift from this goal)",
            problem_statement.strip(),
            "",
            "## Workspace",
            f"cwd: {self.workspace_dir}",
            "",
        ]
        head.extend(self._derived_state_sidecar())
        head_text = "\n".join(head)

        transcript = self._render_transcript_window()

        tail: list[str] = [f"## Turn {turn_index} of {self.max_turns}"]
        if last_parse_error:
            tail.append("")
            tail.append("## Previous response could not be parsed")
            tail.append(last_parse_error)
        if verification_feedback:
            tail.append("")
            tail.append("## Your last submitted patch FAILED the tests")
            tail.append(verification_feedback)
        if stall_nudge:
            tail.append("")
            tail.append("## Note: you appear to be stuck")
            tail.append(
                "You have run an equivalent command several times with the same result. "
                "Change approach, inspect a different file, or call give_up — do not repeat it."
            )
        tail.append("")
        tail.append(
            "You can re-run any earlier command to re-observe its full output; do not rely on "
            "remembered output for exact values."
        )
        tail.append(
            "Reply with EXACTLY ONE JSON object matching the tool schema. No prose, no Markdown."
        )
        tail_text = "\n".join(tail)

        # Absolute safety net: if the whole prompt still exceeds a hard cap, keep
        # the head + tail intact and trim the OLDEST end of the transcript window
        # (most recent turns survive) — deterministic deletion, not middle-splice.
        hard_cap = max(4096, self.max_tokens_per_turn * 4)
        reserved = len(head_text) + len(tail_text) + 16
        transcript_budget = max(0, hard_cap - reserved)
        if transcript and len(transcript) > transcript_budget:
            kept = transcript[-transcript_budget:]
            transcript = (
                "…[older working-memory turns elided to fit context — re-run commands "
                "to re-observe; nothing summarized]…\n" + kept
            )
        return "\n\n".join(part for part in (head_text, transcript, tail_text) if part.strip())

    # Compatibility alias the spec mentioned by name.
    def _extract_tool_call_from_llm_response(
        self, raw: str
    ) -> tuple[Optional[ToolCall], Optional[str]]:
        return _parse_tool_call_from_llm_response(raw)


# ---------------------------------------------------------------------------
# Convenience function (matches spec signature)
# ---------------------------------------------------------------------------


def solve_in_container_agent(
    *,
    llm_config: Any = None,
    workspace_dir: str,
    problem_statement: str,
    max_turns: int = DEFAULT_MAX_TURNS,
    per_tool_timeout_seconds: int = DEFAULT_TURN_TIMEOUT_SECONDS,
    max_tool_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES_PER_TOOL,
    llm_caller: Optional[LLMCaller] = None,
    env_overrides: Optional[dict[str, str]] = None,
    container_supervisor: Optional["ContainerSupervisor"] = None,
    patch_verifier: Optional[Callable[[str], "PatchVerification"]] = None,
    patch_verifier_reject_cap: int = 3,
    recent_verbatim_turns: int = DEFAULT_RECENT_VERBATIM_TURNS,
    stall_repeat_threshold: int = DEFAULT_STALL_REPEAT_THRESHOLD,
    stall_terminate_cap: int = DEFAULT_STALL_TERMINATE_CAP,
    transcript_sink: Optional[Callable[[dict[str, Any]], None]] = None,
    transcript_preload_path: Optional[str] = None,
) -> Optional[str]:
    """Run a single :class:`InContainerAgent` and return the final patch.

    Convenience wrapper around ``InContainerAgent(...).solve(...)``. Returns
    ``None`` on give-up, max-turns, verification-failed, or any failure.
    Forwards the V5 activation knobs (verify-gate, decay-resistant memory, stall
    detection, durable transcript) so production callers are not silently
    dropping them.
    """
    agent = InContainerAgent(
        llm_config=llm_config,
        workspace_dir=workspace_dir,
        max_turns=max_turns,
        per_tool_timeout_seconds=per_tool_timeout_seconds,
        max_tool_output_bytes=max_tool_output_bytes,
        llm_caller=llm_caller,
        env_overrides=env_overrides,
        container_supervisor=container_supervisor,
        patch_verifier=patch_verifier,
        patch_verifier_reject_cap=patch_verifier_reject_cap,
        recent_verbatim_turns=recent_verbatim_turns,
        stall_repeat_threshold=stall_repeat_threshold,
        stall_terminate_cap=stall_terminate_cap,
        transcript_sink=transcript_sink,
    )
    if transcript_preload_path:
        try:
            agent.preload_transcript(transcript_preload_path)
        except Exception:  # noqa: BLE001 - preload is best-effort
            logger.debug("solve_in_container_agent: transcript preload failed", exc_info=True)
    return agent.solve(problem_statement)


# ---------------------------------------------------------------------------
# Helpers re-exported for the SWE-EVO driver
# ---------------------------------------------------------------------------

__all__ = [
    "AgentRunSummary",
    "AgentTurn",
    "DEFAULT_MAX_OUTPUT_BYTES_PER_TOOL",
    "DEFAULT_MAX_TOKENS_PER_TURN",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_TURN_TIMEOUT_SECONDS",
    "InContainerAgent",
    "LLMCaller",
    "TOOL_CALL_SCHEMA",
    "TOOL_GIVE_UP",
    "TOOL_RUN_IN_CONTAINER",
    "TOOL_SUBMIT_PATCH",
    "ToolCall",
    "ToolResult",
    "VALID_TOOLS",
    "solve_in_container_agent",
    # internals exposed for tests
    "_extract_json_object",
    "_execute_in_workspace",
    "_execute_via_container_supervisor",
    "_parse_tool_call_from_llm_response",
    "_strip_redundant_shell_wrapper",
]
