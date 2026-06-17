"""Streaming turn-boundary detector for CLI agent stdout/stderr.

Phase B.5 (Decisive-Edge): the CLI agent runs as an opaque subprocess
whose stdout/stderr we tail line by line. To enable mid-turn course
corrections, we need to know when a "turn" boundary lands so the
calling rollout engine can run an observer that decides whether to
inject a system-level correction or kill the process.

A "turn" here is one logical agent step — i.e. one tool-use sequence
followed by an LLM response. CLIs differ on how they delimit these:

* ``codex``     emits ``## Step N`` lines and JSON tool-use blocks.
* ``claude``    emits ``tool_use`` JSON blocks with a ``name`` field.
* ``gemini``    emits ``Iteration N`` markers.
* ``opencode``  emits step markers like ``[step N]``; best effort —
  we fall back to a generic "blank-line-delimited block" heuristic.

The parser is intentionally tolerant: when no boundary is detected the
whole stream collapses to one big terminal turn at process exit. The
:class:`Turn` dataclass captures the per-turn payload (number, raw
content, files-touched extracted from tool-use call payloads, optional
token usage, and the raw line list).

The parser does NOT mutate the underlying stream — callers feed it
lines and consume :class:`Turn` objects. This module is pure-Python
and has no external dependencies, so unit tests can drive it with a
list of strings.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Optional

logger = logging.getLogger("apex.cli_turn_parser")


# --- Public dataclasses ---


@dataclass
class Turn:
    """One logical agent turn extracted from the CLI's stdout/stderr.

    ``number`` is 1-indexed for human-friendly logging. ``content``
    is the joined raw text of the turn (newline-separated). ``files_touched``
    is the union of file paths the agent appears to have read or written
    in this turn, extracted heuristically from tool-use payloads.
    ``tokens_used`` is best-effort — many CLIs only emit token counts at
    end-of-turn (or never). ``raw_lines`` is the unmodified line slice.
    """

    number: int
    content: str
    files_touched: set[str] = field(default_factory=set)
    tokens_used: Optional[int] = None
    raw_lines: list[str] = field(default_factory=list)


# --- Boundary patterns per CLI ---


# A small DSL: each entry is a list of compiled regex objects; if ANY
# of them matches a non-empty stripped line, the line is treated as the
# START of a new turn (the previous accumulator is flushed).
_BOUNDARY_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "codex": [
        re.compile(r"^##\s*step\s+\d+\b", re.IGNORECASE),
        re.compile(r"^\[step\s+\d+\]", re.IGNORECASE),
        re.compile(r"^\[turn\s+\d+\]", re.IGNORECASE),
        re.compile(r"^\[reasoning\]\s*step\s+\d+", re.IGNORECASE),
    ],
    "codex_cli": [
        re.compile(r"^##\s*step\s+\d+\b", re.IGNORECASE),
        re.compile(r"^\[step\s+\d+\]", re.IGNORECASE),
        re.compile(r"^\[turn\s+\d+\]", re.IGNORECASE),
    ],
    "claude": [
        # Claude Code emits one JSONL message per chunk; tool_use blocks
        # carry "type":"tool_use" — each one is the start of a logical turn.
        re.compile(r'"type"\s*:\s*"tool_use"', re.IGNORECASE),
        re.compile(r'"role"\s*:\s*"assistant"', re.IGNORECASE),
        re.compile(r"^---\s*turn\s+\d+", re.IGNORECASE),
    ],
    "claude_cli": [
        re.compile(r'"type"\s*:\s*"tool_use"', re.IGNORECASE),
        re.compile(r'"role"\s*:\s*"assistant"', re.IGNORECASE),
    ],
    "gemini": [
        re.compile(r"^iteration\s+\d+\b", re.IGNORECASE),
        re.compile(r"^---\s*iteration\s+\d+", re.IGNORECASE),
        re.compile(r"^\[gemini\]\s*step\s+\d+", re.IGNORECASE),
    ],
    "gemini_cli": [
        re.compile(r"^iteration\s+\d+\b", re.IGNORECASE),
    ],
    "opencode": [
        re.compile(r"^\[step\s+\d+\]", re.IGNORECASE),
        re.compile(r"^\[turn\s+\d+\]", re.IGNORECASE),
        re.compile(r"^---\s*step\s+\d+", re.IGNORECASE),
    ],
    "opencode_cli": [
        re.compile(r"^\[step\s+\d+\]", re.IGNORECASE),
    ],
    "metacode": [
        re.compile(r"^\[step\s+\d+\]", re.IGNORECASE),
        re.compile(r"^\[turn\s+\d+\]", re.IGNORECASE),
        re.compile(r"^---\s*step\s+\d+", re.IGNORECASE),
    ],
    "metacode_cli": [
        re.compile(r"^\[step\s+\d+\]", re.IGNORECASE),
    ],
}


# Generic fall-back patterns — used when the named CLI isn't recognised.
# Conservative on purpose: only the most distinctive markers, so a noisy
# transcript without explicit step lines collapses to a single turn.
_GENERIC_BOUNDARY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^##\s*step\s+\d+\b", re.IGNORECASE),
    re.compile(r"^iteration\s+\d+\b", re.IGNORECASE),
    re.compile(r"^\[step\s+\d+\]", re.IGNORECASE),
    re.compile(r"^\[turn\s+\d+\]", re.IGNORECASE),
]


# --- File-touch extraction heuristics ---


# JSON tool-use blocks generally include a "path" or "file_path" or
# "filename" key. We pull each match; this is over-inclusive on purpose
# (the observer's allowlist is what enforces scope, and false positives
# at this level are cheap).
_FILE_FIELD_KEYS: tuple[str, ...] = (
    "path",
    "file_path",
    "filename",
    "target_file",
    "target_path",
    "file",
)

_JSON_FILE_FIELD_RE = re.compile(
    r'"(?:'
    + "|".join(re.escape(key) for key in _FILE_FIELD_KEYS)
    + r')"\s*:\s*"([^"\\]+(?:\\.[^"\\]*)*)"',
    re.IGNORECASE,
)


# Inline tool-call markers used by some CLIs (codex prints lines like
# ``> edit_file: pkg/module.py``). The trailing token is the candidate path.
_INLINE_PATH_RE = re.compile(
    r"\b(?:edit_file|write_file|read_file|view|edit|cat|apply_patch|patch)\s*:\s*"
    r"(\S+\.(?:py|js|ts|go|rs|java|c|cc|cpp|h|hpp|rb|swift|kt|scala|yaml|yml|json|toml|md|txt))",
    re.IGNORECASE,
)


# Token-usage markers — best effort.
_TOKENS_RE = re.compile(
    r'"(?:total_tokens|tokens|usage_total)"\s*:\s*(\d+)',
    re.IGNORECASE,
)


def _extract_files_from_line(line: str) -> set[str]:
    """Return the set of file paths heuristically referenced on ``line``."""
    out: set[str] = set()
    for match in _JSON_FILE_FIELD_RE.finditer(line):
        out.add(match.group(1).strip())
    for match in _INLINE_PATH_RE.finditer(line):
        out.add(match.group(1).strip())
    # Cheap second pass: if the line looks like a JSON object, try to
    # parse it and pull nested file fields the regex missed (e.g.
    # ``"input": {"path": "..."}``).
    stripped = line.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
            out.update(_walk_for_file_keys(obj))
        except (json.JSONDecodeError, ValueError):
            pass
    return out


def _walk_for_file_keys(node: Any) -> set[str]:
    """Recursively collect any string value under a known file-field key."""
    out: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str) and key.lower() in _FILE_FIELD_KEYS:
                if isinstance(value, str) and value.strip():
                    out.add(value.strip())
            out.update(_walk_for_file_keys(value))
    elif isinstance(node, list):
        for item in node:
            out.update(_walk_for_file_keys(item))
    return out


def _extract_tokens_from_line(line: str) -> Optional[int]:
    match = _TOKENS_RE.search(line)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


# --- Parser ---


class CLITurnParser:
    """Detect turn boundaries from the CLI agent's combined stdout/stderr.

    Two entry points:

    * :meth:`parse_stream` — generator that consumes lines lazily and
      yields one :class:`Turn` per detected boundary. The terminal turn
      is yielded when the iterator is exhausted (or :meth:`finalize`
      is called explicitly).
    * :meth:`parse_completed` — convenience for unit tests; takes a
      complete transcript string and returns the list of turns.

    The parser is single-threaded by construction; callers that read
    stdout and stderr from separate threads should serialise into one
    queue first (the rollout engine's existing reader pattern already
    does this).
    """

    def __init__(self, cli_name: str):
        self.cli_name = (cli_name or "").strip().lower()
        self._boundaries = self._compile_boundary_patterns(self.cli_name)
        self._current_lines: list[str] = []
        self._current_files: set[str] = set()
        self._current_tokens: Optional[int] = None
        self._turn_number = 0
        self._finalized = False

    # ------------------------------------------------------------------
    # Boundary helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compile_boundary_patterns(cli_name: str) -> list[re.Pattern[str]]:
        """Return the boundary regex list for ``cli_name``.

        Falls back to the generic pattern set when the name doesn't match.
        """
        if not cli_name:
            return list(_GENERIC_BOUNDARY_PATTERNS)
        # Try the exact name first, then strip the ``_cli`` suffix the
        # apex backends use.
        if cli_name in _BOUNDARY_PATTERNS:
            return list(_BOUNDARY_PATTERNS[cli_name])
        short = cli_name.removesuffix("_cli").removesuffix("-cli")
        if short in _BOUNDARY_PATTERNS:
            return list(_BOUNDARY_PATTERNS[short])
        return list(_GENERIC_BOUNDARY_PATTERNS)

    def _line_starts_new_turn(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        for pattern in self._boundaries:
            if pattern.search(stripped):
                return True
        return False

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------

    def feed_line(self, line: str) -> Optional[Turn]:
        """Feed one line; return the just-completed Turn, if any.

        A boundary line CLOSES the previous turn (if any) and STARTS the
        new one. The closed turn is what's returned; it includes the
        boundary line of the prior turn, NOT the new boundary line.
        """
        if self._finalized:
            # Reset implicitly so a parser instance can be reused for
            # follow-up rounds (rare, but cheaper than constructing a
            # fresh one).
            self._reset_internals()

        if self._line_starts_new_turn(line):
            completed: Optional[Turn] = None
            if self._current_lines:
                completed = self._flush_current_turn()
            self._start_new_turn(line)
            return completed

        self._append_line(line)
        return None

    def feed_lines(self, lines: Iterable[str]) -> Iterator[Turn]:
        """Convenience: feed many lines, yield turns as they close."""
        for line in lines:
            turn = self.feed_line(line)
            if turn is not None:
                yield turn

    def finalize(self) -> Optional[Turn]:
        """Flush any pending lines as the terminal Turn.

        Idempotent: a second call returns ``None``.
        """
        if self._finalized:
            return None
        self._finalized = True
        if not self._current_lines:
            return None
        return self._flush_current_turn(closing=True)

    def parse_stream(self, lines: Iterable[str]) -> Iterator[Turn]:
        """Yield turns lazily as ``lines`` is consumed.

        Implicit finalize at end-of-iteration so callers don't need to
        remember to call :meth:`finalize` themselves.
        """
        yield from self.feed_lines(lines)
        terminal = self.finalize()
        if terminal is not None:
            yield terminal

    def parse_completed(self, full_output: str) -> list[Turn]:
        """Parse a complete transcript string into a list of Turns.

        Convenience for tests and post-hoc analysis. Constructs and
        consumes a fresh internal generator so the parser instance is
        reset cleanly between calls.
        """
        self._reset_internals()
        # ``splitlines(keepends=False)`` so the boundary detector sees
        # canonical stripped lines but ``raw_lines`` keeps them too. We
        # don't need keepends for the turn payload; rejoin with "\n".
        lines = full_output.splitlines()
        return list(self.parse_stream(lines))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reset_internals(self) -> None:
        self._current_lines = []
        self._current_files = set()
        self._current_tokens = None
        self._turn_number = 0
        self._finalized = False

    def _start_new_turn(self, opening_line: str) -> None:
        self._current_lines = [opening_line]
        self._current_files = set(_extract_files_from_line(opening_line))
        self._current_tokens = _extract_tokens_from_line(opening_line)

    def _append_line(self, line: str) -> None:
        self._current_lines.append(line)
        files = _extract_files_from_line(line)
        if files:
            self._current_files.update(files)
        if self._current_tokens is None:
            tokens = _extract_tokens_from_line(line)
            if tokens is not None:
                self._current_tokens = tokens

    def _flush_current_turn(self, *, closing: bool = False) -> Turn:
        self._turn_number += 1
        turn = Turn(
            number=self._turn_number,
            content="\n".join(self._current_lines),
            files_touched=set(self._current_files),
            tokens_used=self._current_tokens,
            raw_lines=list(self._current_lines),
        )
        # Reset accumulator unless the caller is finalizing (in which
        # case _finalized=True already prevents further appends).
        self._current_lines = []
        self._current_files = set()
        self._current_tokens = None
        return turn


__all__ = [
    "CLITurnParser",
    "Turn",
]
