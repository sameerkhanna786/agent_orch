"""Deterministic-replay debugging for APEX rollouts (Phase 6.6).

Two cooperating components:

* :class:`apex.replay.recorder.ReplayRecorder` — context-managed recorder
  that monkey-patches the LLM CLI client and the ACI tool executor so
  that every (prompt -> response) and (tool_invocation -> tool_result)
  pair is appended to a JSONL file. The patch is non-invasive and
  reversed on context exit.

* :class:`apex.replay.player.ReplayPlayer` — context-managed player that
  reads the recorded JSONL and substitutes its responses for live calls.
  Supports a ``mutate=`` parameter for one-prompt-perturbation
  divergence experiments and a ``verify=True`` mode that re-asserts the
  recorded response sequence is reproducible.

The recording format is one JSON object per line. Two record types are
emitted today:

* ``{"turn": int, "type": "llm_call", "model": str, "prompt": str,
  "response": str, "tokens_used": int, "prompt_hash": str, ...}``
* ``{"turn": int, "type": "tool_call", "tool_name": str,
  "args": dict, "result": str, "args_hash": str}``

Recording is append-only; replay is read-only. The player never
mutates the recording on disk.
"""

from __future__ import annotations

from .player import (
    LiveCallDuringReplayError,
    ReplayDivergenceError,
    ReplayPlayer,
)
from .recorder import ReplayRecorder, hash_prompt, hash_tool_args

__all__ = [
    "LiveCallDuringReplayError",
    "ReplayDivergenceError",
    "ReplayPlayer",
    "ReplayRecorder",
    "hash_prompt",
    "hash_tool_args",
]
