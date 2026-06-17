"""
APEX LLM abstraction layer.

The runtime keeps the interface deliberately small:
- message serialization
- tool schema conversion
- chat completion requests
- trajectory capture
- agent-loop execution with structured completion submissions
"""

from __future__ import annotations

import importlib
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .config import ContextConfig, LLMConfig
from .llm_routing import record_llm_backend_failure

logger = logging.getLogger("apex.llm")


def _load_openai_client_class() -> Any:
    """Import the OpenAI SDK lazily so CLI-only runs do not need API deps at startup."""

    try:
        module = importlib.import_module("openai")
    except Exception as exc:  # pragma: no cover - exercised through patched imports in tests.
        raise RuntimeError(
            "The OpenAI Python SDK could not be imported. "
            "Install the API dependencies or use a CLI backend-only configuration."
        ) from exc

    client_class = getattr(module, "OpenAI", None)
    if client_class is None:
        raise RuntimeError("The installed openai package does not expose the OpenAI client.")
    return client_class


@dataclass
class Message:
    """A single conversation message."""

    role: str
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        if self.name:
            payload["name"] = self.name
        return payload


@dataclass
class ToolDefinition:
    """Schema for a callable tool."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCall:
    """A parsed tool call returned by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """LLM response payload."""

    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class AgentSubmission:
    """Structured completion payload captured from a submit_* tool."""

    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolCallVerification:
    """Pre-execution verdict for an LLM-issued tool call."""

    allowed: bool
    message: str = ""


@dataclass
class ToolCallReviewContext:
    """Full local context for an agentic pre-execution tool-call reviewer."""

    tool_call: ToolCall
    tool_definition: ToolDefinition
    available_tools: list[ToolDefinition]
    messages: list[Message]
    iteration: int
    max_iterations: int
    finish_tool_names: set[str]
    recent_tool_calls: list[str]
    actor_backend: str = ""
    actor_model: str = ""
    reviewer_backend: str = ""
    system_prompt: str = ""
    task_description: str = ""
    reason: str = "pre_execution"
    schema_verification: ToolCallVerification = field(
        default_factory=lambda: ToolCallVerification(allowed=True)
    )


ToolCallReviewer = Callable[[ToolCallReviewContext], Optional[ToolCallVerification]]


@dataclass
class StateMachineResult:
    """Result from an explicit revise-or-approve state machine."""

    output: str
    iterations: int
    approved: bool
    submission: Optional[AgentSubmission] = None


def _schema_path_child(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _schema_path_index(path: str, index: int) -> str:
    return f"{path}[{index}]"


def _schema_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _schema_type_matches(value: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return True


def _verify_value_against_tool_schema(
    value: Any,
    schema: Any,
    *,
    path: str,
) -> Optional[str]:
    if not isinstance(schema, dict):
        return None
    expected = schema.get("type")
    expected_types = (
        [str(item) for item in expected if str(item)]
        if isinstance(expected, list)
        else ([str(expected)] if isinstance(expected, str) and expected else [])
    )
    if expected_types and not any(_schema_type_matches(value, item) for item in expected_types):
        label = path or "arguments"
        return (
            f"{label} must be {('/'.join(expected_types))}; "
            f"got {_schema_type_name(value)}."
        )

    if "const" in schema and value != schema.get("const"):
        label = path or "arguments"
        return f"{label} must equal {schema.get('const')!r}."
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values and value not in enum_values:
        label = path or "arguments"
        return f"{label} must be one of {enum_values!r}."

    schema_type = expected_types[0] if expected_types else ""
    if schema_type == "object" or (not expected_types and isinstance(value, dict)):
        if not isinstance(value, dict):
            return None
        required = schema.get("required")
        if isinstance(required, list):
            for raw_key in required:
                key = str(raw_key)
                if key not in value:
                    return f"missing required argument: {key}."
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for raw_key, child_schema in properties.items():
                key = str(raw_key)
                if key in value:
                    child_error = _verify_value_against_tool_schema(
                        value[key],
                        child_schema,
                        path=_schema_path_child(path, key),
                    )
                    if child_error:
                        return child_error
        return None

    if schema_type == "array" or (not expected_types and isinstance(value, list)):
        if not isinstance(value, list):
            return None
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                child_error = _verify_value_against_tool_schema(
                    item,
                    item_schema,
                    path=_schema_path_index(path or "arguments", index),
                )
                if child_error:
                    return child_error
        return None

    return None


def verify_tool_call_against_definition(
    tool: ToolCall,
    tool_definition: ToolDefinition,
) -> ToolCallVerification:
    """Validate a provisional tool call before executing it.

    This verifier only enforces objective schema constraints: the arguments must
    be an object, required arguments must exist, and present values must satisfy
    declared primitive/object/array/enum/const types. It deliberately avoids
    subjective judging so valid tool calls are not blocked by an over-skeptical
    reviewer.
    """

    error = _verify_value_against_tool_schema(
        tool.arguments,
        tool_definition.parameters,
        path="arguments",
    )
    if not error:
        return ToolCallVerification(allowed=True)
    return ToolCallVerification(
        allowed=False,
        message=(
            f"Error: Tool call '{tool.name}' failed pre-execution verification: "
            f"{error} Revise the call using the offered tool schema."
        ),
    )


_CLI_BACKEND_FAMILIES: dict[str, str] = {
    "claude_cli": "claude",
    "gemini_cli": "gemini",
    "codex_cli": "codex",
    "opencode_cli": "opencode",
    "metacode_cli": "opencode",
}


_TOOL_CALL_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "feedback": {"type": "string"},
    },
    "required": ["approved", "feedback"],
    "additionalProperties": False,
}


def _identity_text(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip().lower()


def _llm_backend_identity(llm: Any) -> str:
    config = getattr(llm, "config", None)
    return _identity_text(getattr(config, "backend", ""))


def _llm_model_identity(llm: Any) -> str:
    config = getattr(llm, "config", None)
    model = getattr(config, "resolved_cli_model", None)
    if model:
        return str(model)
    return str(getattr(config, "model", "") or "")


def _cli_backend_family(backend: str) -> str:
    return _CLI_BACKEND_FAMILIES.get(_identity_text(backend), "")


def _distinct_cli_reviewer_verdict(
    *,
    actor_backend: str,
    reviewer_backend: str,
) -> ToolCallVerification:
    actor_family = _cli_backend_family(actor_backend)
    reviewer_family = _cli_backend_family(reviewer_backend)
    if not actor_family:
        return ToolCallVerification(allowed=True)
    if not reviewer_family:
        return ToolCallVerification(
            allowed=False,
            message=(
                "Error: agentic tool-call review is misconfigured: a CLI actor "
                f"({actor_backend}) requires a reviewer from a different CLI family, "
                "but the reviewer backend is not a known CLI backend."
            ),
        )
    if actor_family == reviewer_family:
        return ToolCallVerification(
            allowed=False,
            message=(
                "Error: agentic tool-call review is misconfigured: the reviewer "
                f"backend `{reviewer_backend}` is the same CLI family as the actor "
                f"`{actor_backend}`. Use an independent CLI reviewer."
            ),
        )
    return ToolCallVerification(allowed=True)


def _extract_json_object_payload(text: str) -> Optional[dict[str, Any]]:
    content = str(text or "").strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        parsed = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _bounded_jsonable_message(message: Message, *, max_chars: int) -> dict[str, Any]:
    payload = message.to_dict()
    content = str(payload.get("content") or "")
    if len(content) > max_chars:
        payload["content"] = content[: max_chars - len("\n[... truncated ...]")].rstrip()
        payload["content"] += "\n[... truncated ...]"
    return payload


class AgenticToolCallReviewer:
    """LLM-backed pre-execution reviewer for provisional tool calls.

    The reviewer is intentionally opt-in. It receives the full
    :class:`ToolCallReviewContext`, renders the task, available tools,
    conversation, proposed call, and deterministic schema verdict, then asks a
    reviewer model for an approve/reject JSON decision. Malformed reviewer
    responses fail open to avoid introducing reviewer-caused regressions.
    """

    def __init__(
        self,
        llm: Any,
        *,
        reviewer_backend: str = "",
        working_dir: str = ".",
        max_context_messages: Optional[int] = None,
        max_message_chars: int = 1200,
    ) -> None:
        self.llm = llm
        self.reviewer_backend = _identity_text(reviewer_backend) or _llm_backend_identity(llm)
        self.reviewer_model = _llm_model_identity(llm)
        self.working_dir = str(working_dir or ".")
        self.max_context_messages = max_context_messages
        self.max_message_chars = max(200, int(max_message_chars or 1200))

    def __call__(self, context: ToolCallReviewContext) -> ToolCallVerification:
        reviewer_backend = context.reviewer_backend or self.reviewer_backend
        identity_verdict = _distinct_cli_reviewer_verdict(
            actor_backend=context.actor_backend,
            reviewer_backend=reviewer_backend,
        )
        if not identity_verdict.allowed:
            return identity_verdict
        messages = self._review_messages(context)
        try:
            response_payload = self._invoke_reviewer(messages)
        except Exception as exc:  # pragma: no cover - defensive fail-open
            logger.warning(
                "Agentic tool-call reviewer failed for %s: %s",
                context.tool_call.name,
                exc,
            )
            return context.schema_verification
        payload = _extract_json_object_payload(response_payload)
        if not payload:
            return context.schema_verification
        approved = bool(payload.get("approved", payload.get("allow", True)))
        feedback = str(payload.get("feedback") or payload.get("reason") or "").strip()
        if approved:
            return ToolCallVerification(allowed=True, message=feedback)
        return ToolCallVerification(
            allowed=False,
            message=feedback
            or (
                f"Error: Tool call '{context.tool_call.name}' was rejected by the "
                "agentic tool-call reviewer."
            ),
        )

    def _invoke_reviewer(self, messages: list[Message]) -> str:
        if hasattr(self.llm, "chat"):
            response = self.llm.chat(messages, tools=None)
            return str(getattr(response, "content", "") or "")
        if hasattr(self.llm, "run_structured_prompt"):
            system_prompt = messages[0].content if messages else ""
            prompt = messages[-1].content if messages else ""
            result = self.llm.run_structured_prompt(
                prompt,
                working_dir=self.working_dir,
                schema=_TOOL_CALL_REVIEW_SCHEMA,
                system_prompt=system_prompt,
                allow_edits=False,
            )
            parsed = getattr(result, "parsed_json", None)
            if isinstance(parsed, dict):
                return json.dumps(parsed)
            return str(getattr(result, "text", "") or getattr(result, "raw_output", "") or "")
        raise TypeError("AgenticToolCallReviewer requires an llm with chat or run_structured_prompt")

    def _review_messages(self, context: ToolCallReviewContext) -> list[Message]:
        rendered_messages = list(context.messages)
        omitted = 0
        if (
            self.max_context_messages is not None
            and len(rendered_messages) > self.max_context_messages
        ):
            omitted = len(rendered_messages) - self.max_context_messages
            rendered_messages = rendered_messages[-self.max_context_messages :]
        available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in context.available_tools
        ]
        payload = {
            "task_description": context.task_description,
            "system_prompt": context.system_prompt,
            "iteration": context.iteration,
            "max_iterations": context.max_iterations,
            "reason": context.reason,
            "finish_tool_names": sorted(context.finish_tool_names),
            "recent_tool_calls": list(context.recent_tool_calls),
            "actor_backend": context.actor_backend,
            "actor_model": context.actor_model,
            "reviewer_backend": context.reviewer_backend or self.reviewer_backend,
            "reviewer_model": self.reviewer_model,
            "schema_verification": {
                "allowed": context.schema_verification.allowed,
                "message": context.schema_verification.message,
            },
            "available_tools": available_tools,
            "proposed_tool_call": {
                "name": context.tool_call.name,
                "arguments": context.tool_call.arguments,
            },
            "omitted_older_message_count": omitted,
            "conversation": [
                _bounded_jsonable_message(message, max_chars=self.max_message_chars)
                for message in rendered_messages
            ],
        }
        return [
            Message(
                role="system",
                content=(
                    "You are an agentic pre-execution tool-call reviewer for a coding "
                    "agent. Decide whether the proposed tool call should execute now. "
                    "Use the task, conversation, available tool schemas, recent tool "
                    "history, and schema verdict. Be low-harm: approve valid calls, "
                    "tool-only responses, and ordinary exploratory actions. Reject only "
                    "when the call is clearly unavailable, schema-invalid, stale relative "
                    "to the current task state, destructive, or contrary to explicit "
                    "constraints. Do not reject a tool call merely because it is broad "
                    "or outside a prior localization guess when it can advance the final "
                    "objective. Return only JSON: {\"approved\": true|false, "
                    "\"feedback\": \"short actionable reason\"}."
                ),
            ),
            Message(
                role="user",
                content=json.dumps(payload, indent=2, sort_keys=True, default=str),
            ),
        ]


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat-completions API."""

    def __init__(self, config: LLMConfig, temperature_override: Optional[float] = None):
        self.config = config
        self.temperature = (
            temperature_override if temperature_override is not None else config.temperature
        )
        openai_client_class = _load_openai_client_class()
        self.client = openai_client_class(api_key=config.api_key, base_url=config.base_url)
        self.total_tokens_used = 0
        self.trajectory: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        temp = self.temperature if temperature is None else temperature
        start_time = time.time()

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [message.to_dict() for message in messages],
            "temperature": temp,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
        }
        if tools:
            kwargs["tools"] = [tool.to_openai_schema() for tool in tools]
            kwargs["tool_choice"] = "auto"

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            record_llm_backend_failure(self.config, exc)
            raise
        choice = response.choices[0]
        latency_ms = (time.time() - start_time) * 1000

        parsed_tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            for tool_call in choice.message.tool_calls:
                raw_args = tool_call.function.arguments or "{}"
                try:
                    arguments = json.loads(raw_args)
                except json.JSONDecodeError:
                    arguments = {"raw": raw_args}
                parsed_tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=arguments,
                    )
                )

        usage: dict[str, int] = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            self.total_tokens_used += response.usage.total_tokens

        result = LLMResponse(
            content=choice.message.content,
            tool_calls=parsed_tool_calls,
            usage=usage,
            latency_ms=latency_ms,
        )

        self.trajectory.append(
            {
                "timestamp": time.time(),
                "model": self.config.model,
                "temperature": temp,
                "messages_count": len(messages),
                "response_content": result.content,
                "tool_calls": [
                    {"name": tool.name, "arguments": tool.arguments} for tool in parsed_tool_calls
                ],
                "usage": usage,
                "latency_ms": latency_ms,
            }
        )
        return result

    def get_trajectory(self) -> list[dict[str, Any]]:
        return list(self.trajectory)

    def reset_trajectory(self) -> None:
        self.trajectory.clear()
        self.total_tokens_used = 0


class ContextPruner:
    """Prune long-horizon agent histories while protecting critical state."""

    TASK_LIST_MARKER = "## Task List"
    CONTEXT_SUMMARY_MARKER = "[Context Summary]"
    SYSTEM_REMINDER_MARKER = "[System Reminder]"

    def __init__(self, config: ContextConfig, llm: Optional[LLMClient] = None):
        self.config = config
        self.llm = llm

    def estimate_tokens(self, messages: list[Message]) -> int:
        total = 0
        for message in messages:
            total += max(int(len(message.content or "") / 3.5), 1)
        return total

    def prune_if_needed(self, messages: list[Message]) -> list[Message]:
        if self.estimate_tokens(messages) <= self.config.max_context_tokens:
            return list(messages)

        head_count = min(self.config.protected_head_messages, len(messages))
        tail_count = min(
            self.config.protected_tail_messages,
            max(len(messages) - head_count, 0),
        )
        head = list(messages[:head_count])
        middle = (
            list(messages[head_count:-tail_count]) if tail_count else list(messages[head_count:])
        )
        tail = list(messages[-tail_count:]) if tail_count else []

        middle, protected_middle = self._extract_protected_middle(middle)
        tail = protected_middle + tail

        if self.config.prune_tool_outputs_first:
            middle = self._truncate_tool_outputs(middle)
            middle = self._prune_coding_patterns(middle)
            if self._fits(head, middle, tail):
                return head + middle + tail

        middle = self._deduplicate_tool_results(middle)
        if self._fits(head, middle, tail):
            return head + middle + tail

        middle = self._compress_assistant_messages(middle)
        if self._fits(head, middle, tail):
            return head + middle + tail

        if self.config.enable_periodic_summary and middle:
            summary = self._summarize_middle(middle)
            middle = [Message(role="system", content=f"{self.CONTEXT_SUMMARY_MARKER}\n{summary}")]
        elif middle:
            middle = middle[-1:]

        if self._fits(head, middle, tail):
            return head + middle + tail

        return head + tail + middle

    def _fits(
        self,
        head: list[Message],
        middle: list[Message],
        tail: list[Message],
    ) -> bool:
        return self.estimate_tokens(head + middle + tail) <= self.config.target_context_tokens

    def _extract_protected_middle(
        self,
        middle: list[Message],
    ) -> tuple[list[Message], list[Message]]:
        protected_indices: set[int] = set()
        protected_indices.update(
            self._find_latest_index(middle, marker)
            for marker in (
                self.TASK_LIST_MARKER,
                self.CONTEXT_SUMMARY_MARKER,
                self.SYSTEM_REMINDER_MARKER,
            )
        )
        protected_indices.discard(None)  # type: ignore[arg-type]
        ordered = sorted(protected_indices)
        extracted = [middle[index] for index in ordered]
        filtered = [
            message for index, message in enumerate(middle) if index not in protected_indices
        ]
        return filtered, extracted

    def _find_latest_index(
        self,
        messages: list[Message],
        marker: str,
    ) -> Optional[int]:
        for index in range(len(messages) - 1, -1, -1):
            if marker in (messages[index].content or ""):
                return index
        return None

    def _truncate_tool_outputs(self, messages: list[Message]) -> list[Message]:
        max_chars = max(self.config.tool_output_max_tokens * 4, 1)
        truncated: list[Message] = []
        for message in messages:
            if message.role != "tool" or len(message.content or "") <= max_chars:
                truncated.append(message)
                continue
            content = (message.content or "")[:max_chars].rstrip()
            truncated.append(
                self._copy_message(
                    message,
                    content=f"{content}\n[... truncated ...]",
                )
            )
        return truncated

    def _prune_coding_patterns(self, messages: list[Message]) -> list[Message]:
        result = [self._copy_message(message) for message in messages]
        tool_metadata = self._tool_metadata(messages)
        latest_view_index: dict[str, int] = {}
        failed_edits: dict[str, list[int]] = {}

        for tool_index, (tool_name, arguments) in tool_metadata.items():
            message = result[tool_index]
            if tool_name == "view_file":
                path = str(arguments.get("path") or "").strip()
                if path:
                    previous_index = latest_view_index.get(path)
                    if previous_index is not None:
                        result[previous_index] = self._copy_message(
                            result[previous_index],
                            content=f"[Earlier view of {path} omitted; a newer view is kept.]",
                        )
                    latest_view_index[path] = tool_index
                continue

            if tool_name == "edit_file":
                key = self._normalize_args(arguments)
                if self._is_failed_edit_output(message.content):
                    failed_edits.setdefault(key, []).append(tool_index)
                    continue
                if key in failed_edits:
                    for previous_index in failed_edits.pop(key):
                        result[previous_index] = self._copy_message(
                            result[previous_index],
                            content="[Earlier edit attempt failed before a later successful edit.]",
                        )
                continue

            if tool_name == "bash":
                lines = (message.content or "").splitlines()
                if len(lines) > 30:
                    result[tool_index] = self._copy_message(
                        message,
                        content=self._truncate_lines(lines),
                    )

        return result

    def _deduplicate_tool_results(self, messages: list[Message]) -> list[Message]:
        result = [self._copy_message(message) for message in messages]
        tool_metadata = self._tool_metadata(messages)
        seen: dict[tuple[str, str], int] = {}

        for tool_index, (tool_name, arguments) in tool_metadata.items():
            key = (tool_name, self._normalize_args(arguments))
            previous_index = seen.get(key)
            if previous_index is not None:
                result[previous_index] = self._copy_message(
                    result[previous_index],
                    content=f"[Duplicate {tool_name} output omitted; a newer identical call is kept.]",
                )
            seen[key] = tool_index

        return result

    def _compress_assistant_messages(self, messages: list[Message]) -> list[Message]:
        compressed: list[Message] = []
        for message in messages:
            if message.role != "assistant" or len(message.content or "") <= 500:
                compressed.append(message)
                continue
            paragraphs = [item for item in (message.content or "").split("\n\n") if item.strip()]
            if len(paragraphs) <= 2:
                compressed.append(message)
                continue
            content = f"{paragraphs[0]}\n\n[... reasoning compressed ...]\n\n{paragraphs[-1]}"
            compressed.append(self._copy_message(message, content=content))
        return compressed

    def _summarize_middle(self, messages: list[Message]) -> str:
        """Summarize the middle section, using the LLM when available.

        Falls back to mechanical extraction when no LLM is configured or the
        LLM call fails.
        """
        if self.llm is not None:
            llm_summary = self._llm_summarize_middle(messages)
            if llm_summary:
                return llm_summary
        return self._mechanical_summarize_middle(messages)

    def _llm_summarize_middle(self, messages: list[Message]) -> str:
        """Use the LLM to produce a concise summary of the middle section."""
        text_parts: list[str] = []
        for message in messages:
            content = (message.content or "").strip()
            if not content:
                continue
            text_parts.append(f"[{message.role}] {content[:200]}")
        text = "\n".join(text_parts)
        try:
            response = self.llm.chat(
                [
                    Message(
                        role="system",
                        content=(
                            "Summarize the following agent interaction history. "
                            "Focus on: discoveries made, files examined, edits attempted, "
                            "test results, and current hypotheses. Be concise (max 500 words)."
                        ),
                    ),
                    Message(role="user", content=text[:8000]),
                ]
            )
            summary = (response.content or "").strip()
            if summary:
                return summary
        except Exception:
            pass
        return ""

    def _mechanical_summarize_middle(self, messages: list[Message]) -> str:
        """Mechanical fallback: extract key lines without using the LLM."""
        summary_lines: list[str] = []
        for message in messages:
            content = (message.content or "").strip()
            if not content:
                continue
            line = content.splitlines()[0][:180]
            if message.role == "tool":
                lowered = content.lower()
                if any(
                    token in lowered
                    for token in ("edit", "error", "fail", "pass", "traceback", "selected")
                ):
                    summary_lines.append(f"- Tool: {line}")
            elif message.role == "assistant":
                summary_lines.append(f"- Assistant: {line}")
        if not summary_lines:
            summary_lines.append("- Earlier exploratory context was compacted to preserve budget.")
        return "\n".join(summary_lines[:10])

    def _tool_metadata(self, messages: list[Message]) -> dict[int, tuple[str, dict[str, Any]]]:
        metadata: dict[int, tuple[str, dict[str, Any]]] = {}
        pending: dict[str, tuple[str, dict[str, Any]]] = {}

        for index, message in enumerate(messages):
            if message.role == "assistant" and message.tool_calls:
                for tool_call in message.tool_calls:
                    function = tool_call.get("function", {})
                    raw_arguments = function.get("arguments", {})
                    arguments = raw_arguments
                    if isinstance(raw_arguments, str):
                        try:
                            arguments = json.loads(raw_arguments or "{}")
                        except json.JSONDecodeError:
                            arguments = {"raw": raw_arguments}
                    pending[tool_call.get("id", "")] = (
                        function.get("name", ""),
                        arguments if isinstance(arguments, dict) else {"raw": arguments},
                    )
                continue
            if message.role == "tool" and message.tool_call_id:
                resolved = pending.get(message.tool_call_id)
                if resolved is not None:
                    metadata[index] = resolved

        return metadata

    def _truncate_lines(self, lines: list[str]) -> str:
        if len(lines) <= 20:
            return "\n".join(lines)
        omitted = len(lines) - 20
        return "\n".join(lines[:10] + [f"[... {omitted} lines omitted ...]"] + lines[-10:])

    def _normalize_args(self, arguments: dict[str, Any]) -> str:
        try:
            return json.dumps(arguments, sort_keys=True, default=str)
        except TypeError:
            return str(sorted(arguments.items()))

    def _is_failed_edit_output(self, content: str) -> bool:
        lowered = (content or "").lower()
        return "edit failed" in lowered or "edit rejected" in lowered

    def _copy_message(self, message: Message, *, content: Optional[str] = None) -> Message:
        return Message(
            role=message.role,
            content=message.content if content is None else content,
            tool_call_id=message.tool_call_id,
            tool_calls=list(message.tool_calls) if message.tool_calls else None,
            name=message.name,
        )


_SUBMIT_INVESTIGATION_TOOL = ToolDefinition(
    name="submit_investigation",
    description="Submit a concise summary of the subtask findings.",
    parameters={
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    },
)


def _folded_subtask_tools(
    tools: list[ToolDefinition],
    finish_tool_names: set[str],
) -> list[ToolDefinition]:
    filtered: list[ToolDefinition] = []
    for tool in tools:
        if tool.name in {"approve", "revise", "investigate"}:
            continue
        if tool.name in finish_tool_names or tool.name.startswith("submit_"):
            continue
        filtered.append(tool)
    filtered.append(_SUBMIT_INVESTIGATION_TOOL)
    return filtered


def _bounded_summary(text: str, max_chars: int = 1200) -> str:
    content = (text or "").strip()
    if len(content) <= max_chars:
        return content
    return content[: max_chars - len("\n[... truncated ...]")].rstrip() + "\n[... truncated ...]"


def _spawn_folded_llm(llm: Any) -> Any:
    if isinstance(llm, LLMClient):
        return LLMClient(llm.config, temperature_override=llm.temperature)
    return llm


def _extract_folded_summary(
    loop: "AgentLoop",
    submission: Optional[AgentSubmission],
) -> str:
    if submission is not None:
        summary = submission.arguments.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
    for message in reversed(loop.messages):
        if message.role == "assistant" and (message.content or "").strip():
            return message.content.strip()
    return "Investigation completed without a concise summary."


def _run_folded_subtask(
    *,
    parent_llm: Any,
    tools: list[ToolDefinition],
    finish_tool_names: set[str],
    tool_executor: Optional[Callable[[str, dict[str, Any]], str]],
    context_config: ContextConfig,
    subtask_prompt: str,
    max_iterations: int,
) -> str:
    if tool_executor is None:
        return "Investigation is unavailable because no tool executor is attached."

    sub_llm = _spawn_folded_llm(parent_llm)
    sub_loop = AgentLoop(
        llm=sub_llm,
        system_prompt=(
            "You are a focused research sub-agent. Investigate the question using the "
            "available tools. Prefer reading, searching, tracing symbols, and short "
            "commands. Do not summarize speculation. When you have enough evidence, "
            "call submit_investigation with a concise factual summary of at most 200 words."
        ),
        tools=_folded_subtask_tools(tools, finish_tool_names),
        tool_executor=tool_executor,
        max_iterations=max_iterations,
        finish_tool_names={"submit_investigation"},
    )
    sub_loop.set_context_config(context_config)
    submission = sub_loop.run(subtask_prompt)
    summary = _bounded_summary(_extract_folded_summary(sub_loop, submission))

    if (
        isinstance(parent_llm, LLMClient)
        and isinstance(sub_llm, LLMClient)
        and sub_llm is not parent_llm
    ):
        parent_llm.total_tokens_used += sub_llm.total_tokens_used
        parent_llm.trajectory.append(
            {
                "timestamp": time.time(),
                "folded_subtask": True,
                "subtask_prompt": subtask_prompt[:400],
                "summary": summary,
                "usage": {"total_tokens": sub_llm.total_tokens_used},
            }
        )

    return summary


class AgentLoop:
    """
    Iterative tool-using agent loop.

    Completion is driven by a structured submit_* tool instead of free-form text.
    """

    def __init__(
        self,
        llm: LLMClient,
        system_prompt: str,
        tools: list[ToolDefinition],
        tool_executor: Callable[[str, dict[str, Any]], str],
        max_iterations: int = 30,
        finish_tool_names: Optional[set[str]] = None,
        dynamic_context_provider: Optional[Callable[[int], str | list[str] | None]] = None,
        tool_call_reviewer: Optional[ToolCallReviewer] = None,
        tool_call_actor_backend: str = "",
        tool_call_actor_model: str = "",
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools
        self._tool_names = {tool.name for tool in tools}
        self._tool_definitions = {tool.name: tool for tool in tools}
        self.tool_executor = tool_executor
        self.max_iterations = max_iterations
        self.finish_tool_names = finish_tool_names or set()
        self.dynamic_context_provider = dynamic_context_provider
        self.tool_call_reviewer = tool_call_reviewer
        self.tool_call_actor_backend = (
            _identity_text(tool_call_actor_backend) or _llm_backend_identity(llm)
        )
        self.tool_call_actor_model = str(tool_call_actor_model or _llm_model_identity(llm))
        self.messages: list[Message] = [Message(role="system", content=system_prompt)]
        self.iteration = 0
        self.finished = False
        self.submission: Optional[AgentSubmission] = None
        self._last_dynamic_context: list[str] = []
        self.context_config = ContextConfig()
        self.context_pruner = ContextPruner(self.context_config, llm)
        self.anchor_interval_iterations = 10
        self.repetition_warning_threshold = 3
        self._recent_tool_calls: deque[str] = deque(maxlen=10)
        self._last_repetition_warning: Optional[str] = None
        self.original_task_description = ""

    def add_observation(self, content: str) -> None:
        if not self.original_task_description:
            self.original_task_description = content
        self.messages.append(Message(role="user", content=content))

    def _is_completion_tool(self, tool_name: str) -> bool:
        return tool_name in self._tool_names and (
            tool_name in self.finish_tool_names or tool_name.startswith("submit_")
        )

    def _tool_is_available(self, tool_name: str) -> bool:
        return tool_name in self._tool_names

    def _tool_unavailable_message(self, tool_name: str) -> str:
        return f"Error: Tool '{tool_name}' is not available in this context."

    def _review_tool_call(self, tool: ToolCall) -> ToolCallVerification:
        tool_definition = self._tool_definitions.get(tool.name)
        if tool_definition is None:
            return ToolCallVerification(
                allowed=False,
                message=self._tool_unavailable_message(tool.name),
            )
        schema_verdict = verify_tool_call_against_definition(tool, tool_definition)
        if not schema_verdict.allowed:
            return schema_verdict
        if self.tool_call_reviewer is None:
            return schema_verdict
        context = ToolCallReviewContext(
            tool_call=tool,
            tool_definition=tool_definition,
            available_tools=list(self.tools),
            messages=list(self.messages),
            iteration=self.iteration,
            max_iterations=self.max_iterations,
            finish_tool_names=set(self.finish_tool_names),
            recent_tool_calls=list(self._recent_tool_calls),
            actor_backend=self.tool_call_actor_backend,
            actor_model=self.tool_call_actor_model,
            system_prompt=self.system_prompt,
            task_description=self.original_task_description,
            schema_verification=schema_verdict,
        )
        try:
            reviewer_verdict = self.tool_call_reviewer(context)
        except Exception as exc:  # pragma: no cover - defensive fail-open
            logger.warning("Tool-call reviewer failed for %s: %s", tool.name, exc)
            return schema_verdict
        if reviewer_verdict is None:
            return schema_verdict
        if reviewer_verdict.allowed:
            return reviewer_verdict
        return ToolCallVerification(
            allowed=False,
            message=reviewer_verdict.message
            or (
                f"Error: Tool call '{tool.name}' was rejected by the pre-execution "
                "reviewer. Revise the call with the available context."
            ),
        )

    def set_context_config(self, config: ContextConfig) -> None:
        self.context_config = config
        self.context_pruner = ContextPruner(config, self.llm)

    def fold_subtask(self, subtask_prompt: str, max_iterations: int = 5) -> str:
        return _run_folded_subtask(
            parent_llm=self.llm,
            tools=self.tools,
            finish_tool_names=self.finish_tool_names,
            tool_executor=self.tool_executor,
            context_config=self.context_config,
            subtask_prompt=subtask_prompt,
            max_iterations=max_iterations,
        )

    def _prune_messages_if_needed(self) -> None:
        self.messages = self.context_pruner.prune_if_needed(self.messages)

    def _inject_periodic_anchor(self) -> None:
        if (
            self.anchor_interval_iterations <= 0
            or self.iteration <= 0
            or self.iteration % self.anchor_interval_iterations != 0
        ):
            return
        remaining = max(self.max_iterations - self.iteration, 0)
        self.messages.append(
            Message(
                role="system",
                content=(
                    "[System Reminder]\n"
                    f"Original task:\n{self.original_task_description[:1200]}\n\n"
                    f"You have {remaining} iterations remaining. Focus on the original task, "
                    "avoid unrelated edits, and change approach if progress is stalling."
                ),
            )
        )

    def _inject_dynamic_context(self) -> None:
        if self.dynamic_context_provider is None:
            return
        context = self.dynamic_context_provider(self.iteration)
        if not context:
            return
        messages = [context] if isinstance(context, str) else list(context)
        if messages == self._last_dynamic_context:
            return
        for message in messages:
            if message:
                self.messages.append(Message(role="system", content=message))
        self._last_dynamic_context = messages

    def _record_tool_call(self, tool: ToolCall) -> None:
        if tool.name in {"approve", "revise"} or self._is_completion_tool(tool.name):
            return
        signature = f"{tool.name}:{self.context_pruner._normalize_args(tool.arguments)}"
        self._recent_tool_calls.append(signature)
        count = sum(1 for item in self._recent_tool_calls if item == signature)
        if (
            count >= self.repetition_warning_threshold
            and self._last_repetition_warning != signature
        ):
            self.messages.append(
                Message(
                    role="user",
                    content=(
                        "[System Warning] You have repeated the same tool call several times "
                        f"recently: `{tool.name}`. This suggests the current approach may be "
                        "stalled. Try a different file, a different test, or a different strategy."
                    ),
                )
            )
            self._last_repetition_warning = signature

    def _bind_runtime(self) -> tuple[Any | None, Any | None]:
        owner = getattr(self.tool_executor, "__self__", None)
        if owner is not None and hasattr(owner, "set_agent_runtime"):
            previous_runtime = (
                owner.get_agent_runtime()
                if hasattr(owner, "get_agent_runtime")
                else getattr(owner, "_agent_runtime", None)
            )
            owner.set_agent_runtime(self)
            return owner, previous_runtime
        return None, None

    def step(self) -> bool:
        if self.finished or self.iteration >= self.max_iterations:
            return False

        self.iteration += 1
        logger.info("Agent loop iteration %s/%s", self.iteration, self.max_iterations)
        self._prune_messages_if_needed()
        self._inject_periodic_anchor()
        self._inject_dynamic_context()

        response = self.llm.chat(self.messages, self.tools)

        if response.has_tool_calls:
            assistant_message = Message(
                role="assistant",
                content=response.content or "",
                tool_calls=[
                    {
                        "id": tool.id,
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "arguments": json.dumps(tool.arguments),
                        },
                    }
                    for tool in response.tool_calls
                ],
            )
            self.messages.append(assistant_message)

            for tool in response.tool_calls:
                verification = self._review_tool_call(tool)
                if not verification.allowed:
                    execution_result = verification.message
                else:
                    try:
                        execution_result = self.tool_executor(tool.name, tool.arguments)
                    except Exception as exc:  # pragma: no cover - defensive
                        execution_result = f"Error executing {tool.name}: {exc}"

                self.messages.append(
                    Message(
                        role="tool",
                        content=execution_result,
                        tool_call_id=tool.id,
                    )
                )
                self._record_tool_call(tool)

                if verification.allowed and self._is_completion_tool(tool.name):
                    self.submission = AgentSubmission(
                        tool_name=tool.name,
                        arguments=tool.arguments,
                    )
                    self.finished = True
                    break

            return not self.finished

        self.messages.append(Message(role="assistant", content=response.content or ""))
        if response.content:
            self.messages.append(
                Message(
                    role="user",
                    content=(
                        "Continue by using tools. Call the appropriate submit_* tool "
                        "when the task is complete."
                    ),
                )
            )
        return True

    def run(self, initial_observation: str) -> Optional[AgentSubmission]:
        self.add_observation(initial_observation)
        bound_owner, previous_runtime = self._bind_runtime()
        try:
            while self.step():
                pass
        finally:
            if bound_owner is not None:
                bound_owner.set_agent_runtime(previous_runtime)
        return self.submission


class AgentStateMachine:
    """
    Iterative agent runner with explicit revise-or-approve termination.

    The agent can still use regular tools, but it also gains `approve` and
    `revise` so the caller can provide structured feedback between iterations.
    """

    def __init__(
        self,
        llm: LLMClient,
        initial_prompt: str,
        initial_task: str,
        feedback_generator: Callable[[str], str],
        max_iterations: int = 8,
        tools: Optional[list[ToolDefinition]] = None,
        tool_executor: Optional[Callable[[str, dict[str, Any]], str]] = None,
        finish_tool_names: Optional[set[str]] = None,
        dynamic_context_provider: Optional[Callable[[int], str | list[str] | None]] = None,
        tool_call_reviewer: Optional[ToolCallReviewer] = None,
        tool_call_actor_backend: str = "",
        tool_call_actor_model: str = "",
    ):
        self.llm = llm
        self.initial_prompt = initial_prompt
        self.initial_task = initial_task
        self.feedback_generator = feedback_generator
        self.max_iterations = max_iterations
        self.tool_executor = tool_executor
        self.finish_tool_names = finish_tool_names or set()
        self.dynamic_context_provider = dynamic_context_provider
        self.tool_call_reviewer = tool_call_reviewer
        self.tool_call_actor_backend = (
            _identity_text(tool_call_actor_backend) or _llm_backend_identity(llm)
        )
        self.tool_call_actor_model = str(tool_call_actor_model or _llm_model_identity(llm))
        self.tools = list(tools or [])
        self.tools.extend(
            [
                ToolDefinition(
                    name="approve",
                    description="Approve your current work and terminate.",
                    parameters={
                        "type": "object",
                        "properties": {"summary": {"type": "string"}},
                        "required": ["summary"],
                    },
                ),
                ToolDefinition(
                    name="revise",
                    description="Request another iteration with updated feedback.",
                    parameters={
                        "type": "object",
                        "properties": {"plan": {"type": "string"}},
                        "required": ["plan"],
                    },
                ),
            ]
        )
        self._tool_names = {tool.name for tool in self.tools}
        self._tool_definitions = {tool.name: tool for tool in self.tools}
        self.messages: list[Message] = []
        self._last_dynamic_context: list[str] = []
        self.context_config = ContextConfig()
        self.context_pruner = ContextPruner(self.context_config, llm)
        self.anchor_interval_iterations = 10
        self.repetition_warning_threshold = 3
        self._recent_tool_calls: deque[str] = deque(maxlen=10)
        self._last_repetition_warning: Optional[str] = None

    def _tool_is_available(self, tool_name: str) -> bool:
        return tool_name in self._tool_names

    def _tool_unavailable_message(self, tool_name: str) -> str:
        return f"Error: Tool '{tool_name}' is not available in this context."

    def _review_tool_call(self, tool: ToolCall, iteration: int) -> ToolCallVerification:
        tool_definition = self._tool_definitions.get(tool.name)
        if tool_definition is None:
            return ToolCallVerification(
                allowed=False,
                message=self._tool_unavailable_message(tool.name),
            )
        schema_verdict = verify_tool_call_against_definition(tool, tool_definition)
        if not schema_verdict.allowed:
            return schema_verdict
        if self.tool_call_reviewer is None:
            return schema_verdict
        context = ToolCallReviewContext(
            tool_call=tool,
            tool_definition=tool_definition,
            available_tools=list(self.tools),
            messages=list(self.messages),
            iteration=iteration,
            max_iterations=self.max_iterations,
            finish_tool_names=set(self.finish_tool_names),
            recent_tool_calls=list(self._recent_tool_calls),
            actor_backend=self.tool_call_actor_backend,
            actor_model=self.tool_call_actor_model,
            system_prompt=self.initial_prompt,
            task_description=self.initial_task,
            schema_verification=schema_verdict,
        )
        try:
            reviewer_verdict = self.tool_call_reviewer(context)
        except Exception as exc:  # pragma: no cover - defensive fail-open
            logger.warning("Tool-call reviewer failed for %s: %s", tool.name, exc)
            return schema_verdict
        if reviewer_verdict is None:
            return schema_verdict
        if reviewer_verdict.allowed:
            return reviewer_verdict
        return ToolCallVerification(
            allowed=False,
            message=reviewer_verdict.message
            or (
                f"Error: Tool call '{tool.name}' was rejected by the pre-execution "
                "reviewer. Revise the call with the available context."
            ),
        )

    def set_context_config(self, config: ContextConfig) -> None:
        self.context_config = config
        self.context_pruner = ContextPruner(config, self.llm)

    def fold_subtask(self, subtask_prompt: str, max_iterations: int = 5) -> str:
        return _run_folded_subtask(
            parent_llm=self.llm,
            tools=self.tools,
            finish_tool_names=self.finish_tool_names,
            tool_executor=self.tool_executor,
            context_config=self.context_config,
            subtask_prompt=subtask_prompt,
            max_iterations=max_iterations,
        )

    def _prune_messages_if_needed(self) -> None:
        self.messages = self.context_pruner.prune_if_needed(self.messages)

    def _inject_periodic_anchor(self, iteration: int) -> None:
        if (
            self.anchor_interval_iterations <= 0
            or iteration <= 0
            or iteration % self.anchor_interval_iterations != 0
        ):
            return
        remaining = max(self.max_iterations - iteration, 0)
        self.messages.append(
            Message(
                role="system",
                content=(
                    "[System Reminder]\n"
                    f"Original task:\n{self.initial_task[:1200]}\n\n"
                    f"You have {remaining} iterations remaining. Focus on the original task, "
                    "avoid unrelated edits, and change approach if progress is stalling."
                ),
            )
        )

    def _inject_dynamic_context(self, iteration: int) -> None:
        if self.dynamic_context_provider is None:
            return
        context = self.dynamic_context_provider(iteration)
        if not context:
            return
        rendered = [context] if isinstance(context, str) else list(context)
        if rendered == self._last_dynamic_context:
            return
        for item in rendered:
            if item:
                self.messages.append(Message(role="system", content=item))
        self._last_dynamic_context = rendered

    def _record_tool_call(self, tool: ToolCall) -> None:
        if (
            tool.name in {"approve", "revise"}
            or tool.name in self.finish_tool_names
            or tool.name.startswith("submit_")
        ):
            return
        signature = f"{tool.name}:{self.context_pruner._normalize_args(tool.arguments)}"
        self._recent_tool_calls.append(signature)
        count = sum(1 for item in self._recent_tool_calls if item == signature)
        if (
            count >= self.repetition_warning_threshold
            and self._last_repetition_warning != signature
        ):
            self.messages.append(
                Message(
                    role="user",
                    content=(
                        "[System Warning] You have repeated the same tool call several times "
                        f"recently: `{tool.name}`. This suggests the current approach may be "
                        "stalled. Try a different file, a different test, or a different strategy."
                    ),
                )
            )
            self._last_repetition_warning = signature

    def _bind_runtime(self) -> tuple[Any | None, Any | None]:
        owner = getattr(self.tool_executor, "__self__", None)
        if owner is not None and hasattr(owner, "set_agent_runtime"):
            previous_runtime = (
                owner.get_agent_runtime()
                if hasattr(owner, "get_agent_runtime")
                else getattr(owner, "_agent_runtime", None)
            )
            owner.set_agent_runtime(self)
            return owner, previous_runtime
        return None, None

    def run(self) -> StateMachineResult:
        self.messages = [
            Message(role="system", content=self.initial_prompt),
            Message(role="user", content=self.initial_task),
        ]
        last_output = self.initial_task
        bound_owner, previous_runtime = self._bind_runtime()
        try:
            for iteration in range(1, self.max_iterations + 1):
                self._prune_messages_if_needed()
                self._inject_periodic_anchor(iteration)
                self._inject_dynamic_context(iteration)

                response = self.llm.chat(self.messages, self.tools)
                last_output = response.content or last_output

                if response.has_tool_calls:
                    assistant_message = Message(
                        role="assistant",
                        content=response.content or "",
                        tool_calls=[
                            {
                                "id": tool.id,
                                "type": "function",
                                "function": {
                                    "name": tool.name,
                                    "arguments": json.dumps(tool.arguments),
                                },
                            }
                            for tool in response.tool_calls
                        ],
                    )
                    self.messages.append(assistant_message)

                    revise_requested = False
                    for tool in response.tool_calls:
                        verification = self._review_tool_call(tool, iteration)
                        if not verification.allowed:
                            self.messages.append(
                                Message(
                                    role="tool",
                                    content=verification.message,
                                    tool_call_id=tool.id,
                                )
                            )
                            self._record_tool_call(tool)
                            continue
                        if tool.name == "approve":
                            summary = tool.arguments.get("summary", response.content or "")
                            return StateMachineResult(
                                output=summary,
                                iterations=iteration,
                                approved=True,
                                submission=AgentSubmission(
                                    tool_name="approve", arguments=tool.arguments
                                ),
                            )
                        if tool.name in self.finish_tool_names or tool.name.startswith("submit_"):
                            return StateMachineResult(
                                output=response.content or json.dumps(tool.arguments, indent=2),
                                iterations=iteration,
                                approved=True,
                                submission=AgentSubmission(
                                    tool_name=tool.name, arguments=tool.arguments
                                ),
                            )
                        if tool.name == "revise":
                            revise_requested = True
                            self.messages.append(
                                Message(
                                    role="tool",
                                    content="Revision plan noted.",
                                    tool_call_id=tool.id,
                                )
                            )
                            continue

                        if self.tool_executor is None:
                            execution_result = f"Tool '{tool.name}' is unavailable."
                        else:
                            try:
                                execution_result = self.tool_executor(tool.name, tool.arguments)
                            except Exception as exc:  # pragma: no cover - defensive
                                execution_result = f"Error executing {tool.name}: {exc}"

                        self.messages.append(
                            Message(
                                role="tool",
                                content=execution_result,
                                tool_call_id=tool.id,
                            )
                        )
                        self._record_tool_call(tool)

                    feedback = self.feedback_generator(response.content or "")
                    if revise_requested or feedback:
                        self.messages.append(
                            Message(role="user", content=feedback or "Revise your work.")
                        )
                    continue

                self.messages.append(Message(role="assistant", content=response.content or ""))
                feedback = self.feedback_generator(response.content or "")
                if feedback:
                    self.messages.append(Message(role="user", content=feedback))
        finally:
            if bound_owner is not None:
                bound_owner.set_agent_runtime(previous_runtime)

        return StateMachineResult(
            output=last_output,
            iterations=self.max_iterations,
            approved=False,
            submission=None,
        )
