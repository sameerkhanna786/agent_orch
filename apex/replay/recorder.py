"""Append-only JSONL recorder for APEX rollouts.

Hooks into two production callsites via runtime monkey-patching:

* ``apex.core.cli_backend.CLIModelClient.run_structured_prompt`` — every
  LLM call is recorded as ``{"type": "llm_call", ...}``.
* ``apex.tools.aci.ACIToolExecutor.execute`` — every tool invocation
  is recorded as ``{"type": "tool_call", ...}``.

The recorder is intentionally a context manager so that the patch is
installed and reversed in a single ``with`` block — there is no mutation
of constructor signatures or class attributes that survives the
context. The recorder is reentrant-safe per recorder instance: nested
``record_to`` blocks at module scope are forbidden (the second one
raises) so that we never silently merge logs.

Record format
-------------

One JSON object per line. Common fields:

* ``turn`` (int): monotonically increasing index assigned in the order
  callbacks fire on this recorder. Replay matches by turn first, then
  by ``prompt_hash`` / ``args_hash``.
* ``type`` (str): ``"llm_call"`` or ``"tool_call"``.

LLM call fields: ``model``, ``prompt``, ``system_prompt``, ``response``,
``parsed_json``, ``tokens_used``, ``success``, ``prompt_hash``.

Tool call fields: ``tool_name``, ``args``, ``result``, ``args_hash``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger("apex.replay.recorder")


# A module-level guard: only one ReplayRecorder may have an active patch
# at a time, because the patch targets are module-level class methods.
_ACTIVE_RECORDER_LOCK = threading.Lock()
_ACTIVE_RECORDER: Optional["ReplayRecorder"] = None


def hash_prompt(prompt: str, *, model: Optional[str] = None) -> str:
    """SHA-256 of ``model || "\\x1f" || prompt`` (first 16 hex chars).

    Replay match keys must be deterministic across runs. Model is
    included so that the same prompt routed to two different models is
    not incorrectly matched.
    """
    h = hashlib.sha256()
    h.update((model or "").encode("utf-8"))
    h.update(b"\x1f")
    h.update((prompt or "").encode("utf-8"))
    return h.hexdigest()[:16]


def hash_tool_args(tool_name: str, arguments: dict[str, Any]) -> str:
    """SHA-256 of canonical-JSON of ``(tool_name, arguments)``."""
    canonical = json.dumps(
        {"tool": tool_name, "args": arguments or {}},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class _RecorderState:
    """Mutable per-recorder state held inside the patch closures."""

    turn: int = 0
    file_handle: Optional[Any] = None
    write_lock: threading.Lock = field(default_factory=threading.Lock)


class ReplayRecorder:
    """Append-only JSONL recorder of LLM + tool invocations.

    Parameters
    ----------
    record_path
        Destination JSONL file. Parent directories are created on
        ``__enter__``. The file is opened in ``"a"`` mode so concurrent
        appends from a single process are safe; cross-process recording
        is not supported.
    """

    def __init__(self, record_path: Path | str) -> None:
        self.record_path = Path(record_path)
        self._state = _RecorderState()
        self._original_run_structured_prompt: Any = None
        self._original_aci_execute: Any = None
        self._installed = False

    # ------------------------------------------------------------------
    # Context manager entry/exit
    # ------------------------------------------------------------------

    def __enter__(self) -> "ReplayRecorder":
        global _ACTIVE_RECORDER
        with _ACTIVE_RECORDER_LOCK:
            if _ACTIVE_RECORDER is not None:
                raise RuntimeError(
                    "Another ReplayRecorder is already active; nested "
                    "recording is not supported because patches are "
                    "global."
                )
            _ACTIVE_RECORDER = self
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        # Append-only by contract.
        self._state.file_handle = open(
            self.record_path,
            "a",
            encoding="utf-8",
            buffering=1,  # line-buffered
        )
        self._install_patches()
        self._installed = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        global _ACTIVE_RECORDER
        try:
            if self._installed:
                self._uninstall_patches()
                self._installed = False
        finally:
            if self._state.file_handle is not None:
                try:
                    self._state.file_handle.flush()
                    self._state.file_handle.close()
                finally:
                    self._state.file_handle = None
            with _ACTIVE_RECORDER_LOCK:
                if _ACTIVE_RECORDER is self:
                    _ACTIVE_RECORDER = None

    # ------------------------------------------------------------------
    # Public class helpers
    # ------------------------------------------------------------------

    @classmethod
    @contextlib.contextmanager
    def record_to(cls, path: Path | str) -> Iterator["ReplayRecorder"]:
        """Convenience context manager: ``with ReplayRecorder.record_to(p): ...``."""
        recorder = cls(path)
        with recorder as active:
            yield active

    # ------------------------------------------------------------------
    # Direct write helpers — used by tests and by patches.
    # ------------------------------------------------------------------

    def write_llm_call(
        self,
        *,
        model: str,
        prompt: str,
        response: str,
        system_prompt: Optional[str] = None,
        parsed_json: Optional[dict[str, Any]] = None,
        tokens_used: int = 0,
        success: bool = True,
        extra: Optional[dict[str, Any]] = None,
    ) -> int:
        """Record one LLM call. Returns the assigned turn index."""
        with self._state.write_lock:
            turn = self._state.turn
            self._state.turn += 1
        record = {
            "turn": turn,
            "type": "llm_call",
            "model": model,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "response": response,
            "parsed_json": parsed_json,
            "tokens_used": int(tokens_used or 0),
            "success": bool(success),
            "prompt_hash": hash_prompt(prompt, model=model),
        }
        if extra:
            record["extra"] = dict(extra)
        self._append(record)
        return turn

    def write_tool_call(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
        extra: Optional[dict[str, Any]] = None,
    ) -> int:
        """Record one tool invocation. Returns the assigned turn index."""
        with self._state.write_lock:
            turn = self._state.turn
            self._state.turn += 1
        record = {
            "turn": turn,
            "type": "tool_call",
            "tool_name": tool_name,
            "args": _safe_dict(arguments),
            "result": result,
            "args_hash": hash_tool_args(tool_name, arguments or {}),
        }
        if extra:
            record["extra"] = dict(extra)
        self._append(record)
        return turn

    @property
    def turn_count(self) -> int:
        return self._state.turn

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _append(self, record: dict[str, Any]) -> None:
        if self._state.file_handle is None:
            raise RuntimeError("Recorder is not active.")
        line = json.dumps(record, sort_keys=True, default=_json_default)
        with self._state.write_lock:
            self._state.file_handle.write(line + "\n")
            self._state.file_handle.flush()

    def _install_patches(self) -> None:
        # Imported lazily so that importing the replay module does not
        # force-load cli_backend or aci at package-import time.
        from apex.core import cli_backend as _cli_mod
        from apex.tools import aci as _aci_mod

        self._original_run_structured_prompt = _cli_mod.CLIModelClient.run_structured_prompt
        self._original_aci_execute = _aci_mod.ACIToolExecutor.execute

        recorder = self
        original_llm = self._original_run_structured_prompt
        original_tool = self._original_aci_execute

        def _patched_run_structured_prompt(self, prompt, working_dir, *args, **kwargs):  # type: ignore[no-untyped-def]
            system_prompt = kwargs.get("system_prompt")
            result = original_llm(self, prompt, working_dir, *args, **kwargs)
            try:
                model = getattr(self.config, "model", "unknown")
                response_text = getattr(result, "text", "") or ""
                parsed_json = getattr(result, "parsed_json", None)
                usage = getattr(result, "usage", {}) or {}
                tokens_used = 0
                if isinstance(usage, dict):
                    tokens_used = int(usage.get("total_tokens") or usage.get("tokens_used") or 0)
                recorder.write_llm_call(
                    model=str(model),
                    prompt=prompt,
                    system_prompt=system_prompt,
                    response=response_text,
                    parsed_json=parsed_json if isinstance(parsed_json, dict) else None,
                    tokens_used=tokens_used,
                    success=bool(getattr(result, "success", True)),
                )
            except Exception:  # pragma: no cover - defensive
                logger.exception("ReplayRecorder failed to log llm_call")
            return result

        def _patched_aci_execute(self, tool_name, arguments):  # type: ignore[no-untyped-def]
            result = original_tool(self, tool_name, arguments)
            try:
                recorder.write_tool_call(
                    tool_name=tool_name,
                    arguments=arguments or {},
                    result=str(result) if result is not None else "",
                )
            except Exception:  # pragma: no cover - defensive
                logger.exception("ReplayRecorder failed to log tool_call")
            return result

        _cli_mod.CLIModelClient.run_structured_prompt = _patched_run_structured_prompt
        _aci_mod.ACIToolExecutor.execute = _patched_aci_execute

    def _uninstall_patches(self) -> None:
        try:
            from apex.core import cli_backend as _cli_mod
            from apex.tools import aci as _aci_mod
        except Exception:  # pragma: no cover - defensive
            return
        if self._original_run_structured_prompt is not None:
            _cli_mod.CLIModelClient.run_structured_prompt = self._original_run_structured_prompt
            self._original_run_structured_prompt = None
        if self._original_aci_execute is not None:
            _aci_mod.ACIToolExecutor.execute = self._original_aci_execute
            self._original_aci_execute = None


def _safe_dict(arguments: Any) -> dict[str, Any]:
    """Coerce arguments into a JSON-safe dict for the record."""
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return {str(key): _json_safe(value) for key, value in arguments.items()}
    return {"_value": _json_safe(arguments)}


def _json_safe(value: Any) -> Any:
    """Best-effort JSON normalisation."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return repr(value)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return repr(value)


__all__ = [
    "ReplayRecorder",
    "hash_prompt",
    "hash_tool_args",
]
