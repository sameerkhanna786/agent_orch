"""Replay JSONL recordings produced by :class:`ReplayRecorder`.

Two modes:

* **Pure replay** — ``with ReplayPlayer.replay(path) as p:`` substitutes
  recorded responses for every LLM and tool call. If a live call is
  attempted that does not match any remaining recorded turn, the player
  raises :class:`LiveCallDuringReplayError`. This is the strict
  determinism mode used by ``apex replay --verify``.

* **Mutation** — ``with ReplayPlayer.replay(path, mutate={...}) as p:``
  rewrites the prompt at one or more recorded turns. Once a mutated
  turn is reached, the recorded response is treated as stale: subsequent
  prompts no longer match the recording, so the player falls back to
  the original (live) implementation by calling through to the
  monkey-patched-out method via ``recorder.live_passthrough=True``.

The player is purely read-only on the recording — it never mutates the
file on disk.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from .recorder import hash_prompt, hash_tool_args

logger = logging.getLogger("apex.replay.player")


_ACTIVE_PLAYER_LOCK = threading.Lock()
_ACTIVE_PLAYER: Optional["ReplayPlayer"] = None


class ReplayDivergenceError(RuntimeError):
    """Raised by ``--verify`` when re-recording produces a different log."""


class LiveCallDuringReplayError(RuntimeError):
    """Raised in strict replay when a call has no matching recorded turn."""


@dataclass
class _ReplayState:
    """Per-player mutable state."""

    cursor: int = 0
    diverged: bool = False
    last_match_turn: int = -1
    mutations_applied: list[int] = field(default_factory=list)
    live_calls_made: list[dict[str, Any]] = field(default_factory=list)


class ReplayPlayer:
    """Replay a JSONL recording produced by :class:`ReplayRecorder`.

    Parameters
    ----------
    record_path
        Path to the JSONL recording.
    mutate
        Optional dict of one-off mutations. Keys understood:

        * ``"turn_<N>_prompt"`` (str): replace the prompt at turn ``N``.
          Once a mutated turn fires, subsequent calls fall back to the
          live implementation.

    strict
        If ``True``, calls that do not match any remaining record raise
        :class:`LiveCallDuringReplayError` instead of falling back to
        the live implementation. Defaults to ``False``; set to ``True``
        for ``--verify`` mode.
    """

    def __init__(
        self,
        record_path: Path | str,
        *,
        mutate: Optional[dict[str, str]] = None,
        strict: bool = False,
    ) -> None:
        self.record_path = Path(record_path)
        self.mutate = dict(mutate or {})
        self.strict = bool(strict)
        self._records: list[dict[str, Any]] = []
        self._state = _ReplayState()
        self._installed = False
        self._original_run_structured_prompt: Any = None
        self._original_aci_execute: Any = None

    # ------------------------------------------------------------------
    # Public class helpers
    # ------------------------------------------------------------------

    @classmethod
    @contextlib.contextmanager
    def replay(
        cls,
        path: Path | str,
        *,
        mutate: Optional[dict[str, str]] = None,
        strict: bool = False,
    ) -> Iterator["ReplayPlayer"]:
        """Convenience: ``with ReplayPlayer.replay(path) as player: ...``."""
        player = cls(path, mutate=mutate, strict=strict)
        with player as active:
            yield active

    # ------------------------------------------------------------------
    # Context entry / exit
    # ------------------------------------------------------------------

    def __enter__(self) -> "ReplayPlayer":
        global _ACTIVE_PLAYER
        with _ACTIVE_PLAYER_LOCK:
            if _ACTIVE_PLAYER is not None:
                raise RuntimeError(
                    "Another ReplayPlayer is already active; nested replay is not supported."
                )
            _ACTIVE_PLAYER = self
        self._records = self._load_records(self.record_path)
        self._install_patches()
        self._installed = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        global _ACTIVE_PLAYER
        try:
            if self._installed:
                self._uninstall_patches()
                self._installed = False
        finally:
            with _ACTIVE_PLAYER_LOCK:
                if _ACTIVE_PLAYER is self:
                    _ACTIVE_PLAYER = None

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def records(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the loaded records."""
        return list(self._records)

    @property
    def diverged(self) -> bool:
        return self._state.diverged

    @property
    def cursor(self) -> int:
        return self._state.cursor

    @property
    def live_calls(self) -> list[dict[str, Any]]:
        return list(self._state.live_calls_made)

    @property
    def mutations_applied(self) -> list[int]:
        return list(self._state.mutations_applied)

    # ------------------------------------------------------------------
    # Replay lookup helpers
    # ------------------------------------------------------------------

    def lookup_llm(self, model: str, prompt: str) -> Optional[dict[str, Any]]:
        """Return the next matching ``llm_call`` record, or ``None``.

        Match is by ``prompt_hash`` first; if multiple records share the
        same hash, we return the earliest one whose turn is at or after
        the current cursor.
        """
        target_hash = hash_prompt(prompt, model=model)
        for idx in range(self._state.cursor, len(self._records)):
            record = self._records[idx]
            if record.get("type") != "llm_call":
                continue
            if record.get("prompt_hash") == target_hash:
                self._state.cursor = idx + 1
                self._state.last_match_turn = int(record.get("turn", idx))
                return record
        return None

    def lookup_tool(self, tool_name: str, arguments: dict[str, Any]) -> Optional[dict[str, Any]]:
        target_hash = hash_tool_args(tool_name, arguments or {})
        for idx in range(self._state.cursor, len(self._records)):
            record = self._records[idx]
            if record.get("type") != "tool_call":
                continue
            if record.get("tool_name") == tool_name and record.get("args_hash") == target_hash:
                self._state.cursor = idx + 1
                self._state.last_match_turn = int(record.get("turn", idx))
                return record
        return None

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_against(self, candidate_path: Path | str) -> bool:
        """Compare ``candidate_path`` to the recording for this player.

        Returns ``True`` if the two recordings match in the
        determinism-relevant fields: turn, type, model/tool_name,
        prompt_hash/args_hash and response/result.

        Raises :class:`ReplayDivergenceError` with the first divergence
        otherwise.
        """
        actual = self._load_records(Path(candidate_path))
        if len(actual) != len(self._records):
            raise ReplayDivergenceError(
                f"record-length mismatch: expected {len(self._records)}, got {len(actual)}"
            )
        for i, (expected, got) in enumerate(zip(self._records, actual)):
            for key in ("turn", "type"):
                if expected.get(key) != got.get(key):
                    raise ReplayDivergenceError(
                        f"divergence at index {i}: {key} "
                        f"expected={expected.get(key)!r} "
                        f"actual={got.get(key)!r}"
                    )
            if expected.get("type") == "llm_call":
                comparable = ("prompt_hash", "model", "response")
            else:
                comparable = ("args_hash", "tool_name", "result")
            for key in comparable:
                if expected.get(key) != got.get(key):
                    raise ReplayDivergenceError(
                        f"divergence at turn {expected.get('turn')}: {key} mismatch"
                    )
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load_records(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"replay record not found: {path}")
        records: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
                records.append(record)
        return records

    def _resolve_mutated_prompt(self, original_prompt: str, current_turn: int) -> tuple[str, bool]:
        """Return ``(prompt, mutated)`` honouring ``self.mutate``."""
        key = f"turn_{current_turn}_prompt"
        if key in self.mutate:
            return self.mutate[key], True
        return original_prompt, False

    def _install_patches(self) -> None:
        from apex.core import cli_backend as _cli_mod
        from apex.tools import aci as _aci_mod

        self._original_run_structured_prompt = _cli_mod.CLIModelClient.run_structured_prompt
        self._original_aci_execute = _aci_mod.ACIToolExecutor.execute
        original_llm = self._original_run_structured_prompt
        original_tool = self._original_aci_execute
        player = self

        def _patched_run_structured_prompt(self, prompt, working_dir, *args, **kwargs):  # type: ignore[no-untyped-def]
            current_turn = player._predict_next_turn()
            effective_prompt, mutated = player._resolve_mutated_prompt(prompt, current_turn)
            if mutated:
                player._state.diverged = True
                player._state.mutations_applied.append(current_turn)
                player._state.live_calls_made.append(
                    {
                        "turn": current_turn,
                        "type": "llm_call",
                        "reason": "mutation",
                    }
                )
                return original_llm(self, effective_prompt, working_dir, *args, **kwargs)

            model = getattr(self.config, "model", "unknown")
            record = player.lookup_llm(model=str(model), prompt=prompt)
            if record is None:
                if player.strict:
                    raise LiveCallDuringReplayError(
                        f"no recorded llm_call matches model={model!r} "
                        f"at cursor={player._state.cursor}"
                    )
                player._state.diverged = True
                player._state.live_calls_made.append(
                    {
                        "turn": current_turn,
                        "type": "llm_call",
                        "reason": "no_match",
                    }
                )
                return original_llm(self, prompt, working_dir, *args, **kwargs)

            return _build_cli_result_from_record(record, _cli_mod)

        def _patched_aci_execute(self, tool_name, arguments):  # type: ignore[no-untyped-def]
            current_turn = player._predict_next_turn()
            record = player.lookup_tool(tool_name, arguments or {})
            if record is None:
                if player.strict:
                    raise LiveCallDuringReplayError(
                        f"no recorded tool_call matches "
                        f"tool={tool_name!r} at cursor={player._state.cursor}"
                    )
                player._state.diverged = True
                player._state.live_calls_made.append(
                    {
                        "turn": current_turn,
                        "type": "tool_call",
                        "reason": "no_match",
                    }
                )
                return original_tool(self, tool_name, arguments)
            return record.get("result", "")

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

    def _predict_next_turn(self) -> int:
        """Best-guess turn index for the *next* call about to be made.

        Used purely for mutation matching and live-call diagnostics.
        Returns the turn of the next unread record, or
        ``last_match_turn + 1`` if we have run past the recording.
        """
        if self._state.cursor < len(self._records):
            return int(self._records[self._state.cursor].get("turn", self._state.cursor))
        return self._state.last_match_turn + 1


def _build_cli_result_from_record(record: dict[str, Any], cli_mod: Any) -> Any:
    """Re-hydrate a ``CLIModelResult`` from a recorded ``llm_call`` row."""
    result_cls = getattr(cli_mod, "CLIModelResult")
    parsed_json = record.get("parsed_json")
    return result_cls(
        success=bool(record.get("success", True)),
        text=str(record.get("response", "")),
        parsed_json=parsed_json if isinstance(parsed_json, dict) else None,
        usage={"total_tokens": int(record.get("tokens_used", 0) or 0)},
        raw_output=str(record.get("response", "")),
        duration_seconds=0.0,
        error=None,
    )


__all__ = [
    "ReplayPlayer",
    "ReplayDivergenceError",
    "LiveCallDuringReplayError",
]
