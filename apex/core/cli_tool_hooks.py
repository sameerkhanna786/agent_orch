"""Native CLI tool-hook contracts for agentic tool-call review.

The contracts here are intentionally declarative. APEX can use them to install
the same reviewer behavior into different agentic CLIs without pretending that
all CLIs expose the same hook names, timeout units, or denial payloads.
"""

from __future__ import annotations

import json
import shlex
import sys
from dataclasses import dataclass
from typing import Any, Optional


_CLI_BACKEND_FAMILIES: dict[str, str] = {
    "claude_cli": "claude",
    "gemini_cli": "gemini",
    "codex_cli": "codex",
    "opencode_cli": "opencode",
    "metacode_cli": "opencode",
}


@dataclass(frozen=True)
class CLIToolHookSupport:
    """Documented pre-tool hook behavior for one agentic CLI backend."""

    backend: str
    family: str
    supports_direct_pre_tool_hook: bool
    pre_tool_event: str
    matcher_all: str
    timeout_unit: str
    config_location: str
    denial_feedback_reaches_agent: bool
    supports_argument_rewrite: bool
    source_url: str
    intercepted_tool_scopes: tuple[str, ...] = ()
    known_interception_gaps: tuple[str, ...] = ()
    trust_strategy: str = ""
    notes: str = ""


_HOOK_SUPPORT: dict[str, CLIToolHookSupport] = {
    "codex_cli": CLIToolHookSupport(
        backend="codex_cli",
        family="codex",
        supports_direct_pre_tool_hook=True,
        pre_tool_event="PreToolUse",
        matcher_all="*",
        timeout_unit="seconds",
        config_location="CODEX_HOME/hooks.json or .codex/hooks.json",
        denial_feedback_reaches_agent=True,
        supports_argument_rewrite=True,
        source_url="https://developers.openai.com/codex/hooks",
        intercepted_tool_scopes=("Bash", "apply_patch", "MCP tools"),
        known_interception_gaps=(
            "unified_exec shell calls",
            "WebSearch",
            "non-shell non-MCP tools",
        ),
        trust_strategy=(
            "Non-managed hooks require trust; vetted automation may pass "
            "--dangerously-bypass-hook-trust."
        ),
        notes=(
            "Codex documents PreToolUse for Bash, apply_patch, and MCP tools; "
            "it is a guardrail, not a complete enforcement boundary."
        ),
    ),
    "claude_cli": CLIToolHookSupport(
        backend="claude_cli",
        family="claude",
        supports_direct_pre_tool_hook=True,
        pre_tool_event="PreToolUse",
        matcher_all="*",
        timeout_unit="seconds",
        config_location="settings JSON via --settings or .claude/settings.json",
        denial_feedback_reaches_agent=True,
        supports_argument_rewrite=True,
        source_url="https://code.claude.com/docs/en/hooks",
        intercepted_tool_scopes=(
            "Bash",
            "Edit",
            "Write",
            "Read",
            "Glob",
            "Grep",
            "Agent",
            "WebFetch",
            "WebSearch",
            "AskUserQuestion",
            "ExitPlanMode",
            "MCP tools",
        ),
        known_interception_gaps=(),
        trust_strategy="Settings/project/plugin hooks follow Claude Code hook trust policy.",
        notes="Claude Code documents PreToolUse as firing before every tool call.",
    ),
    "gemini_cli": CLIToolHookSupport(
        backend="gemini_cli",
        family="gemini",
        supports_direct_pre_tool_hook=True,
        pre_tool_event="BeforeTool",
        matcher_all=".*",
        timeout_unit="milliseconds",
        config_location="~/.gemini/settings.json or .gemini/settings.json",
        denial_feedback_reaches_agent=True,
        supports_argument_rewrite=True,
        source_url="https://geminicli.com/docs/hooks/reference/",
        intercepted_tool_scopes=("built-in tools", "MCP tools"),
        known_interception_gaps=(),
        trust_strategy="Project hooks are fingerprinted and warned before execution.",
        notes="Gemini CLI documents BeforeTool denial reasons as tool errors to the agent.",
    ),
    "opencode_cli": CLIToolHookSupport(
        backend="opencode_cli",
        family="opencode",
        supports_direct_pre_tool_hook=True,
        pre_tool_event="tool.execute.before",
        matcher_all="*",
        timeout_unit="reviewer subprocess seconds",
        config_location="OPENCODE_CONFIG_DIR/plugins/*.ts or .opencode/plugins/*.ts",
        denial_feedback_reaches_agent=True,
        supports_argument_rewrite=True,
        source_url="https://opencode.ai/docs/plugins/",
        intercepted_tool_scopes=("built-in tools", "custom tools", "MCP tools"),
        known_interception_gaps=(),
        trust_strategy="Local plugins in the configured plugin directory are loaded at startup.",
        notes=(
            "OpenCode/MetaCode documents a plugin-level tool.execute.before hook; "
            "throwing from the hook rejects the tool call as a tool error."
        ),
    ),
    "metacode_cli": CLIToolHookSupport(
        backend="metacode_cli",
        family="opencode",
        supports_direct_pre_tool_hook=True,
        pre_tool_event="tool.execute.before",
        matcher_all="*",
        timeout_unit="reviewer subprocess seconds",
        config_location="OPENCODE_CONFIG plugin[] file URI",
        denial_feedback_reaches_agent=True,
        supports_argument_rewrite=True,
        source_url="https://www.internalfb.com/intern/staticdocs/metacode/configure/plugins",
        intercepted_tool_scopes=("built-in tools", "custom tools", "MCP tools"),
        known_interception_gaps=(),
        trust_strategy=(
            "APEX writes an isolated opencode.json containing only its local "
            "tool-review plugin."
        ),
        notes=(
            "MetaCode documents SDK plugins loaded from the merged opencode "
            "config and tool.execute.before hooks that can reject tool calls."
        ),
    ),
}


def normalize_cli_backend(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip().lower()


def cli_backend_family(value: Any) -> str:
    return _CLI_BACKEND_FAMILIES.get(normalize_cli_backend(value), "")


def get_cli_tool_hook_support(value: Any) -> CLIToolHookSupport:
    backend = normalize_cli_backend(value)
    support = _HOOK_SUPPORT.get(backend)
    if support is not None:
        return support
    return CLIToolHookSupport(
        backend=backend,
        family="",
        supports_direct_pre_tool_hook=False,
        pre_tool_event="",
        matcher_all="",
        timeout_unit="",
        config_location="",
        denial_feedback_reaches_agent=False,
        supports_argument_rewrite=False,
        source_url="",
        intercepted_tool_scopes=(),
        known_interception_gaps=("unknown backend",),
        trust_strategy="",
        notes="Unknown CLI backend; no native pre-tool hook contract is registered.",
    )


def independent_cli_reviewer_error(
    *,
    actor_backend: Any,
    reviewer_backend: Any,
) -> Optional[str]:
    actor = normalize_cli_backend(actor_backend)
    reviewer = normalize_cli_backend(reviewer_backend)
    actor_family = cli_backend_family(actor)
    reviewer_family = cli_backend_family(reviewer)
    if not actor_family:
        return None
    if not reviewer_family:
        return (
            f"CLI actor '{actor}' requires an independent CLI reviewer, but "
            f"reviewer backend '{reviewer}' is not a known CLI backend."
        )
    if actor_family == reviewer_family:
        return (
            f"CLI actor '{actor}' and reviewer '{reviewer}' are the same CLI "
            "family. Use a different agentic CLI family as reviewer."
        )
    return None


def require_independent_cli_reviewer(
    *,
    actor_backend: Any,
    reviewer_backend: Any,
) -> None:
    error = independent_cli_reviewer_error(
        actor_backend=actor_backend,
        reviewer_backend=reviewer_backend,
    )
    if error:
        raise ValueError(error)


def build_apex_tool_review_hook_command(
    *,
    actor_backend: Any,
    reviewer_backend: Any,
    reviewer_command: str,
    timeout_seconds: int,
    python_executable: Optional[str] = None,
    reviewer_env_file: Optional[str] = None,
) -> str:
    require_independent_cli_reviewer(
        actor_backend=actor_backend,
        reviewer_backend=reviewer_backend,
    )
    args = [
        python_executable or sys.executable,
        "-m",
        "apex.tools.cli_tool_review_hook",
        "--actor-backend",
        normalize_cli_backend(actor_backend),
        "--reviewer-backend",
        normalize_cli_backend(reviewer_backend),
        "--reviewer-command",
        str(reviewer_command),
        "--timeout-seconds",
        str(max(1, int(timeout_seconds))),
    ]
    if reviewer_env_file:
        args.extend(["--reviewer-env-file", str(reviewer_env_file)])
    return shlex.join(args)


def build_cli_tool_review_hook_config(
    *,
    actor_backend: Any,
    hook_command: str,
    timeout_seconds: int = 60,
    matcher: Optional[str] = None,
) -> dict[str, Any]:
    support = get_cli_tool_hook_support(actor_backend)
    if not support.supports_direct_pre_tool_hook:
        raise ValueError(
            f"Backend '{support.backend}' does not expose a registered direct "
            "pre-tool hook contract."
        )
    timeout = max(1, int(timeout_seconds))
    if support.backend in {"codex_cli", "claude_cli"}:
        return {
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": matcher or support.matcher_all,
                        "hooks": [
                            {
                                "type": "command",
                                "command": hook_command,
                                "timeout": timeout,
                                "statusMessage": "Reviewing tool call",
                            }
                        ],
                    }
                ]
            }
        }
    if support.backend == "gemini_cli":
        return {
            "hooks": {
                "BeforeTool": [
                    {
                        "matcher": matcher or support.matcher_all,
                        "sequential": True,
                        "hooks": [
                            {
                                "name": "apex-tool-call-reviewer",
                                "type": "command",
                                "command": hook_command,
                                "timeout": timeout * 1000,
                                "description": "Agentic review before tool execution",
                            }
                        ],
                    }
                ]
            }
        }
    if support.backend in {"opencode_cli", "metacode_cli"}:
        raise ValueError(
            "OpenCode/MetaCode tool-call review uses a local plugin file, not a "
            "JSON hook config."
        )
    raise ValueError(f"Unsupported native hook backend '{support.backend}'.")


def build_opencode_tool_review_plugin_source(*, hook_command: str) -> str:
    """Render the OpenCode/MetaCode plugin that delegates to APEX's reviewer."""

    command_literal = json.dumps(str(hook_command))
    return f"""const HOOK_COMMAND = {command_literal};

async function readText(stream) {{
  if (!stream) {{
    return "";
  }}
  return await new Response(stream).text();
}}

function denialReason(output) {{
  if (!output || typeof output !== "object") {{
    return "";
  }}
  if (output.decision === "deny" || output.decision === "block") {{
    return String(output.reason || output.feedback || "Rejected by APEX tool-call reviewer.");
  }}
  const specific = output.hookSpecificOutput;
  if (specific && specific.permissionDecision === "deny") {{
    return String(
      specific.permissionDecisionReason ||
        "Rejected by APEX tool-call reviewer.",
    );
  }}
  return "";
}}

async function runApexReview(payload) {{
  const proc = Bun.spawn(["/bin/sh", "-lc", HOOK_COMMAND], {{
    stdin: "pipe",
    stdout: "pipe",
    stderr: "pipe",
    env: process.env,
    cwd: payload.cwd || process.cwd(),
  }});
  const writer = proc.stdin.getWriter();
  await writer.write(new TextEncoder().encode(JSON.stringify(payload)));
  await writer.close();
  const [stdout, stderr] = await Promise.all([
    readText(proc.stdout),
    readText(proc.stderr),
    proc.exited,
  ]);
  const text = String(stdout || "").trim();
  if (!text) {{
    if (stderr) {{
      console.error(stderr);
    }}
    return {{}};
  }}
  try {{
    return JSON.parse(text);
  }} catch (error) {{
    console.error("APEX tool-call reviewer returned malformed JSON", error, stderr);
    return {{}};
  }}
}}

export const ApexToolCallReviewer = async (ctx) => {{
  return {{
    "tool.execute.before": async (input, output) => {{
      const args = output?.args ?? input?.args ?? input?.input ?? {{}};
      const cwd = ctx?.directory || process.cwd();
      let verdict = {{}};
      try {{
        verdict = await runApexReview({{
          hook_event_name: "tool.execute.before",
          cwd,
          tool_name: input?.tool || input?.name || "",
          tool_input: args,
          original_request_name: input?.tool || input?.name || "",
        }});
      }} catch (error) {{
        console.error("APEX tool-call reviewer unavailable; allowing tool call", error);
        return;
      }}
      const reason = denialReason(verdict);
      if (reason) {{
        throw new Error(reason);
      }}
    }},
  }};
}};
"""


def render_cli_tool_review_hook_output(
    *,
    actor_backend: Any,
    approved: bool,
    feedback: str = "",
) -> dict[str, Any]:
    backend = normalize_cli_backend(actor_backend)
    reason = str(feedback or "").strip()
    if backend in {"codex_cli", "claude_cli"}:
        event = "PreToolUse"
        decision = "allow" if approved else "deny"
        output: dict[str, Any] = {
            "hookSpecificOutput": {
                "hookEventName": event,
                "permissionDecision": decision,
            }
        }
        if not approved:
            output["hookSpecificOutput"]["permissionDecisionReason"] = (
                reason or "Rejected by APEX tool-call reviewer."
            )
        return output
    if backend in {"gemini_cli", "opencode_cli", "metacode_cli"}:
        if approved:
            return {"decision": "allow"}
        return {
            "decision": "deny",
            "reason": reason or "Rejected by APEX tool-call reviewer.",
        }
    return {"decision": "allow" if approved else "deny", "reason": reason}
