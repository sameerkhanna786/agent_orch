"""APEX hook entrypoint for agentic CLI tool-call review.

The hook receives a native CLI hook payload on stdin, asks an independent
reviewer command for a JSON verdict, and prints the native allow/deny payload
expected by the actor CLI. Reviewer infrastructure failures fail open so a
flaky reviewer does not make useful agents strictly worse.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from apex.core.cli_tool_hooks import (
    independent_cli_reviewer_error,
    render_cli_tool_review_hook_output,
)


_REVIEWER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "feedback": {"type": "string"},
    },
    "required": ["approved", "feedback"],
    "additionalProperties": False,
}

_MAX_RAW_PAYLOAD_STRING_CHARS = 12000
_MAX_RAW_PAYLOAD_LIST_ITEMS = 80
_MAX_RAW_PAYLOAD_DICT_ITEMS = 120

_READ_ONLY_COMMAND_RE = re.compile(
    r"^\s*(?:"
    r"pwd|ls\b|find\b|rg\b|grep\b|cat\b|head\b|tail\b|sed\s+-n\b|"
    r"git\s+(?:status|diff|show|log|rev-parse|branch)\b|"
    r"(?:uv\s+run\s+)?(?:python\s+-m\s+)?pytest\b|"
    r"tox\b|make\s+(?:test|check|lint)\b|npm\s+test\b|yarn\s+test\b|pnpm\s+test\b"
    r")",
    flags=re.IGNORECASE,
)
_DESTRUCTIVE_COMMAND_RE = re.compile(
    r"\b(?:"
    r"rm\s+-(?:[A-Za-z]*r[A-Za-z]*f|[A-Za-z]*f[A-Za-z]*r)|"
    r"git\s+(?:reset|clean|checkout|restore)\b|"
    r"docker\s+(?:rm|rmi|system\s+prune|volume\s+rm)\b|"
    r"chmod\s+-R\b|chown\s+-R\b"
    r")",
    flags=re.IGNORECASE,
)
_DEPENDENCY_COMMAND_RE = re.compile(
    r"\b(?:"
    r"(?:python\s+-m\s+)?pip\s+(?:install|uninstall|sync)|"
    r"uv\s+(?:add|remove|sync|pip\s+(?:install|uninstall|sync))|"
    r"poetry\s+(?:add|remove|install|update)|"
    r"npm\s+(?:install|update|add|remove|ci)|"
    r"yarn\s+(?:add|remove|install|upgrade)|"
    r"pnpm\s+(?:add|remove|install|update)|"
    r"apt(?:-get)?\s+(?:install|remove|upgrade)|"
    r"brew\s+(?:install|uninstall|upgrade)|"
    r"conda\s+(?:install|remove|update)"
    r")\b",
    flags=re.IGNORECASE,
)
_SHELL_WRITE_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:sed\s+-i\b|perl\s+-[^;\n]*i\b)|(?:^|[^<])>{1,2}\s*[^&\s]",
    flags=re.IGNORECASE,
)
_ABSOLUTE_HOST_TOOL_RE = re.compile(
    r"(?:^|\s)(?:/Users/|/tmp/|/private/tmp/|/usr/local/bin/|/opt/homebrew/bin/)[^\s;&|]+",
    flags=re.IGNORECASE,
)
_RISKY_PATH_RE = re.compile(
    r"(?:^|[\\/\s'\"=:])(?:"
    r"tests?[\\/][^\\/\s'\":]+|"
    r"[^\\/\s'\":]*expected[-_]?test[-_]?ids?[^\\/\s'\":]*|"
    r"harness(?:es)?[\\/]|benchmark(?:s)?[\\/]|evaluation[\\/]|configs?[\\/]benchmark"
    r")",
    flags=re.IGNORECASE,
)
_WRITE_TOOL_MARKERS = ("write", "edit", "patch", "apply_patch", "multiedit")
_READ_TOOL_MARKERS = ("read", "grep", "glob", "list", "search")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2)


def _bounded_prompt_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 8:
        return "<truncated: maximum nesting depth exceeded>"
    if isinstance(value, str):
        if len(value) <= _MAX_RAW_PAYLOAD_STRING_CHARS:
            return value
        return (
            value[:_MAX_RAW_PAYLOAD_STRING_CHARS]
            + f"... <truncated {len(value) - _MAX_RAW_PAYLOAD_STRING_CHARS} chars>"
        )
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        limited = [
            _bounded_prompt_value(item, depth=depth + 1)
            for item in value[:_MAX_RAW_PAYLOAD_LIST_ITEMS]
        ]
        if len(value) > _MAX_RAW_PAYLOAD_LIST_ITEMS:
            limited.append(f"<truncated {len(value) - _MAX_RAW_PAYLOAD_LIST_ITEMS} items>")
        return limited
    if isinstance(value, dict):
        bounded: dict[str, Any] = {}
        items = list(value.items())
        for key, item in items[:_MAX_RAW_PAYLOAD_DICT_ITEMS]:
            bounded[str(key)] = _bounded_prompt_value(item, depth=depth + 1)
        if len(items) > _MAX_RAW_PAYLOAD_DICT_ITEMS:
            bounded["<truncated>"] = f"{len(items) - _MAX_RAW_PAYLOAD_DICT_ITEMS} fields"
        return bounded
    return str(value)


def _tool_payload_text(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def _tool_command(payload: dict[str, Any]) -> str:
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        tool_input = payload.get("input") if isinstance(payload.get("input"), dict) else {}
    for key in ("command", "cmd", "script", "shell_command"):
        value = tool_input.get(key) if isinstance(tool_input, dict) else None
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _tool_target_paths(payload: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        tool_input = payload.get("input") if isinstance(payload.get("input"), dict) else {}
    if isinstance(tool_input, dict):
        for key in (
            "file_path",
            "path",
            "target_file",
            "target_path",
            "notebook_path",
        ):
            value = tool_input.get(key)
            if isinstance(value, str) and value.strip():
                paths.append(value)
        for key in ("files", "paths", "file_paths"):
            value = tool_input.get(key)
            if isinstance(value, list):
                paths.extend(str(item) for item in value if str(item).strip())
    return paths


def classify_tool_call_risk(payload: dict[str, Any]) -> dict[str, Any]:
    """Return whether this native CLI hook payload needs agentic review."""

    tool_name = str(
        payload.get("tool_name")
        or payload.get("name")
        or payload.get("tool")
        or payload.get("original_request_name")
        or ""
    ).strip()
    normalized_tool = tool_name.lower()
    command = _tool_command(payload)
    paths = _tool_target_paths(payload)
    text = " ".join([command, " ".join(paths), _tool_payload_text(payload.get("tool_input"))])
    reasons: list[str] = []

    if command:
        if _DESTRUCTIVE_COMMAND_RE.search(command):
            reasons.append("destructive_shell")
        if _DEPENDENCY_COMMAND_RE.search(command):
            reasons.append("dependency_or_environment_change")
        if _SHELL_WRITE_RE.search(command):
            reasons.append("shell_write")
        if _ABSOLUTE_HOST_TOOL_RE.search(command):
            reasons.append("absolute_host_tool_path")
    if _RISKY_PATH_RE.search(text):
        reasons.append("protected_test_or_harness_path")

    write_like = any(marker in normalized_tool for marker in _WRITE_TOOL_MARKERS)
    if write_like:
        patch_text = str(
            (payload.get("tool_input") or {}).get("patch")
            if isinstance(payload.get("tool_input"), dict)
            else ""
        )
        touched_files = set(re.findall(r"^\s*(?:\+\+\+|---|\*\*\* (?:Update|Add|Delete) File:)\s+(.+)$", patch_text, flags=re.MULTILINE))
        if len(touched_files) >= 8 or len(patch_text) >= 20000:
            reasons.append("broad_write")
        if normalized_tool == "write" and not paths:
            reasons.append("unknown_write_target")

    read_like = any(marker in normalized_tool for marker in _READ_TOOL_MARKERS)
    command_is_read_only = bool(command and _READ_ONLY_COMMAND_RE.search(command))
    if not reasons and (read_like or command_is_read_only):
        return {
            "requires_review": False,
            "risk_reasons": [],
            "risk_level": "low",
            "tool_name": tool_name,
        }
    return {
        "requires_review": bool(reasons),
        "risk_reasons": sorted(set(reasons)),
        "risk_level": "review_required" if reasons else "low",
        "tool_name": tool_name,
    }


def _append_review_metric(event: dict[str, Any]) -> None:
    path_text = os.environ.get("APEX_TOOL_CALL_REVIEW_METRICS_PATH", "").strip()
    if not path_text:
        return
    try:
        path = Path(path_text).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(event)
        payload.setdefault("timestamp", time.time())
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    except OSError:
        return


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _decode_transcript_bytes(content: bytes) -> str:
    return content.decode("utf-8", errors="replace")


def _read_transcript_excerpt(
    path_text: str,
    *,
    max_bytes: int = 60000,
    head_bytes: int = 24000,
    tail_bytes: int = 36000,
) -> dict[str, Any]:
    empty = {"head": "", "tail": "", "truncated": False}
    if not path_text:
        return empty
    try:
        path = Path(path_text).expanduser()
        if not path.is_file():
            return empty
        size = path.stat().st_size
        if size <= max_bytes:
            return {
                "head": path.read_text(encoding="utf-8", errors="replace"),
                "tail": "",
                "truncated": False,
            }
        with path.open("rb") as handle:
            head = handle.read(max(1, int(head_bytes)))
            handle.seek(max(0, size - max(1, int(tail_bytes))))
            tail = handle.read(max(1, int(tail_bytes)))
        return {
            "head": _decode_transcript_bytes(head),
            "tail": _decode_transcript_bytes(tail),
            "truncated": True,
        }
    except OSError:
        return empty


def _load_reviewer_env_file(path_text: str) -> dict[str, str]:
    if not path_text:
        return {}
    try:
        path = Path(path_text).expanduser()
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in loaded.items()
        if key and value is not None
    }


def _extract_json_object(text: str, *, depth: int = 0) -> Optional[dict[str, Any]]:
    if depth >= 6:
        return None
    content = str(text or "").strip()
    if not content:
        return None
    candidates: list[str] = []
    candidates.extend(line.strip() for line in content.splitlines() if line.strip())
    candidates.append(content)
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            nested = _coerce_review_verdict(parsed, depth=depth)
            if nested is not None:
                return nested
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return _coerce_review_verdict(parsed, depth=depth)
    return None


def _coerce_review_verdict(value: dict[str, Any], *, depth: int = 0) -> Optional[dict[str, Any]]:
    if isinstance(value.get("approved"), bool):
        return {
            "approved": bool(value["approved"]),
            "feedback": str(value.get("feedback") or ""),
        }
    if depth >= 6:
        return None
    for key in ("result", "response", "message", "content", "text", "output"):
        nested = value.get(key)
        extracted = _coerce_nested_review_verdict(nested, depth=depth + 1)
        if extracted is not None:
            return extracted
    for nested in value.values():
        extracted = _coerce_nested_review_verdict(nested, depth=depth + 1)
        if extracted is not None:
            return extracted
    return None


def _coerce_nested_review_verdict(value: Any, *, depth: int) -> Optional[dict[str, Any]]:
    if isinstance(value, str):
        return _extract_json_object(value, depth=depth)
    if isinstance(value, dict):
        return _coerce_review_verdict(value, depth=depth)
    if isinstance(value, list):
        for item in reversed(value):
            extracted = _coerce_nested_review_verdict(item, depth=depth + 1)
            if extracted is not None:
                return extracted
    return None


def _build_reviewer_prompt(
    *,
    payload: dict[str, Any],
    actor_backend: str,
    reviewer_backend: str,
) -> str:
    transcript_excerpt = _read_transcript_excerpt(
        str(payload.get("transcript_path") or "")
    )
    risk = classify_tool_call_risk(payload)
    review_policy = [
        "Approve valid exploratory, diagnostic, and implementation tool calls.",
        (
            "Do not reject a call merely because it is broad or outside an old "
            "localization guess; reject only if the call is harmful to reaching "
            "the final objective or violates explicit constraints."
        ),
        (
            "If rejecting, provide feedback specific enough for the original "
            "agent to choose a better next tool call."
        ),
        (
            "You are reviewing while the actor CLI is paused in its native "
            "pre-tool hook. approved=true executes the original call; "
            "approved=false denies it and sends feedback back as the tool error."
        ),
    ]
    context = {
        "actor_backend": actor_backend,
        "reviewer_backend": reviewer_backend,
        "review_policy": review_policy,
        "hook_event_name": payload.get("hook_event_name"),
        "cwd": payload.get("cwd"),
        "session_id": payload.get("session_id"),
        "model": payload.get("model"),
        "permission_mode": payload.get("permission_mode"),
        "tool_name": payload.get("tool_name"),
        "tool_input": payload.get("tool_input"),
        "mcp_context": payload.get("mcp_context"),
        "original_request_name": payload.get("original_request_name"),
        "risk_classification": risk,
        "transcript_excerpt": transcript_excerpt,
        "raw_hook_payload": _bounded_prompt_value(payload),
    }
    return (
        "You are the independent APEX tool-call reviewer for an agentic CLI.\n"
        "Review the proposed tool call before it executes. You may use your own "
        "agentic CLI tools to inspect the workspace when the provided context is "
        "not enough. Approve unless the call is objectively invalid, violates "
        "the task or environment constraints, risks corrupting useful progress, "
        "or is harmful to the agent reaching its final objective. Do not reject "
        "only because a call touches files outside a prior localization guess "
        "when the call can advance the objective. If rejecting, provide concrete "
        "feedback the original agent can act on in its next tool call.\n\n"
        "Return only JSON matching this schema:\n"
        f"{_json_dumps(_REVIEWER_JSON_SCHEMA)}\n\n"
        "Review context:\n"
        f"{_json_dumps(context)}\n"
    )


def _pass_through_with_message(message: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if message:
        output["systemMessage"] = message
    return output


def review_hook_payload(
    *,
    payload: dict[str, Any],
    actor_backend: str,
    reviewer_backend: str,
    reviewer_command: str,
    timeout_seconds: int,
    reviewer_env_file: str = "",
) -> dict[str, Any]:
    risk = classify_tool_call_risk(payload)
    if _env_flag_enabled("APEX_TOOL_CALL_REVIEWER_ACTIVE"):
        _append_review_metric(
            {
                "event": "nested_passthrough",
                "actor_backend": actor_backend,
                "reviewer_backend": reviewer_backend,
                "risk": risk,
            }
        )
        return _pass_through_with_message(
            "APEX tool-call review is already active in an upstream reviewer; "
            "allowing this nested reviewer tool call.",
        )
    if not risk.get("requires_review"):
        _append_review_metric(
            {
                "event": "skipped_low_risk",
                "actor_backend": actor_backend,
                "reviewer_backend": reviewer_backend,
                "risk": risk,
                "outcome_status": "not_reviewed_low_risk",
            }
        )
        return {}
    family_error = independent_cli_reviewer_error(
        actor_backend=actor_backend,
        reviewer_backend=reviewer_backend,
    )
    if family_error:
        _append_review_metric(
            {
                "event": "review_rejected_misconfiguration",
                "actor_backend": actor_backend,
                "reviewer_backend": reviewer_backend,
                "risk": risk,
                "approved": False,
                "feedback": family_error,
                "outcome_status": "prevented_same_family_review",
            }
        )
        return render_cli_tool_review_hook_output(
            actor_backend=actor_backend,
            approved=False,
            feedback=family_error,
        )
    prompt = _build_reviewer_prompt(
        payload=payload,
        actor_backend=actor_backend,
        reviewer_backend=reviewer_backend,
    )
    env = os.environ.copy()
    env.update(_load_reviewer_env_file(reviewer_env_file))
    env["APEX_TOOL_CALL_REVIEWER_ACTIVE"] = "1"
    try:
        completed = subprocess.run(
            reviewer_command,
            input=prompt,
            capture_output=True,
            text=True,
            shell=True,
            timeout=max(1, int(timeout_seconds)),
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _append_review_metric(
            {
                "event": "reviewer_unavailable_fail_open",
                "actor_backend": actor_backend,
                "reviewer_backend": reviewer_backend,
                "risk": risk,
                "approved": True,
                "feedback": str(exc),
                "outcome_status": "reviewer_unavailable",
            }
        )
        return _pass_through_with_message(
            f"APEX tool-call reviewer unavailable; allowing tool call: {exc}",
        )
    verdict = _extract_json_object((completed.stdout or "") + "\n" + (completed.stderr or ""))
    if verdict is None:
        _append_review_metric(
            {
                "event": "reviewer_malformed_fail_open",
                "actor_backend": actor_backend,
                "reviewer_backend": reviewer_backend,
                "risk": risk,
                "approved": True,
                "feedback": "reviewer returned no valid verdict",
                "outcome_status": "reviewer_malformed",
            }
        )
        return _pass_through_with_message(
            "APEX tool-call reviewer returned no valid verdict; allowing tool call.",
        )
    approved = bool(verdict.get("approved"))
    feedback = str(verdict.get("feedback") or "")
    _append_review_metric(
        {
            "event": "reviewer_verdict",
            "actor_backend": actor_backend,
            "reviewer_backend": reviewer_backend,
            "risk": risk,
            "approved": approved,
            "feedback": feedback,
            "outcome_status": "pending_post_tool_outcome" if approved else "blocked_pre_execution",
        }
    )
    return render_cli_tool_review_hook_output(
        actor_backend=actor_backend,
        approved=approved,
        feedback=feedback,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-backend", required=True)
    parser.add_argument("--reviewer-backend", required=True)
    parser.add_argument("--reviewer-command", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--reviewer-env-file", default="")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as exc:
        output = _pass_through_with_message(
            f"APEX tool-call hook received malformed input; allowing tool call: {exc}",
        )
    else:
        if not isinstance(payload, dict):
            payload = {}
        output = review_hook_payload(
            payload=payload,
            actor_backend=args.actor_backend,
            reviewer_backend=args.reviewer_backend,
            reviewer_command=args.reviewer_command,
            timeout_seconds=args.timeout_seconds,
            reviewer_env_file=args.reviewer_env_file,
        )
    print(json.dumps(output))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
