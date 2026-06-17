"""Default turn observers for mid-stream CLI course correction.

Phase B.5 (Decisive-Edge): the rollout engine wires a
:class:`apex.core.cli_turn_parser.CLITurnParser` against the CLI
agent's stdout stream and feeds each detected :class:`Turn` to one or
more :data:`ObserverFn` callables. An observer either returns ``None``
(let the agent continue) or a :class:`CourseCorrection` describing
either a soft mid-stream injection (``abort=False``) or a hard kill
(``abort=True``).

Default observer set
--------------------

* :func:`localizer_scope_observer` — fires when the agent edits files
  outside the localizer hypothesis (and the rollout is configured with
  ``localizer_enforcement != "advisory"``). The observer's value is to
  ask the agent for evidence before broad edits accumulate; it does not
  treat the localizer file list as the task boundary.

Future observers can be added without changing the engine wiring;
the engine concatenates them via :func:`compose_observers`.

Design notes
------------

Observers are pure functions: input is a Turn + small context, output
is an ``Optional[CourseCorrection]``. They MUST NOT block (the engine
calls them on the stream-reader thread) and MUST NOT mutate the Turn.
Logging is encouraged; raising is treated as "no correction" to avoid
crashing the rollout on observer bugs.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from ..core.cli_turn_parser import Turn

logger = logging.getLogger("apex.turn_observers")


# --- Public dataclasses & types ---


@dataclass
class CourseCorrection:
    """Outcome of a single observer firing.

    ``message`` is a human-readable instruction injected into the
    agent's next turn (CLI-specific transport — see
    ``CLIModelClient`` for which CLIs support mid-stream injection
    versus log-and-continue).

    ``abort=True`` requests termination of the agent subprocess via the
    rollout-scoped registry. Observers should reserve this for hard safety
    violations. Localizer scope is diagnostic pressure, not a process abort
    or candidate-drop signal by itself.
    """

    message: str
    abort: bool = False
    source: str = ""  # observer identifier for diagnostics
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentContext:
    """Lightweight context passed to observers.

    Kept intentionally small — observers should only reach for the few
    fields they actually need so unit tests can construct one cheaply
    via SimpleNamespace if desired.
    """

    rollout_id: Any = None
    stage_name: str = ""
    cli_name: str = ""
    worktree_path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


ObserverFn = Callable[[Turn, AgentContext], Optional[CourseCorrection]]


# --- Default observers ---


def _matches_any_glob(path: str, patterns: Iterable[str]) -> bool:
    """True iff ``path`` matches any glob in ``patterns``.

    The localizer artifact files are concrete repo-relative paths
    (e.g. ``pkg/engine.py``); the allowlist often uses globs
    (``tests/**``, ``setup.py``). We try literal-equality first (fast
    path) then glob.
    """
    if not patterns:
        return False
    path_norm = path.strip().lstrip("./")
    for raw in patterns:
        if not raw:
            continue
        pattern = str(raw).strip().lstrip("./")
        if not pattern:
            continue
        if path_norm == pattern:
            return True
        if fnmatch.fnmatch(path_norm, pattern):
            return True
        # Common case: ``tests/**`` should also match ``tests/foo.py``.
        # fnmatch handles ``tests/*`` but not the recursive form, so
        # treat trailing ``/**`` as a prefix match.
        if pattern.endswith("/**") and path_norm.startswith(pattern[:-3] + "/"):
            return True
    return False


def localizer_scope_observer(
    *,
    localizer_files: Iterable[str],
    allowlist: Iterable[str] = (),
    enforcement: str = "warning",
) -> ObserverFn:
    """Return an observer that flags edits outside the localizer hypothesis.

    * In ``advisory`` mode the observer is a no-op (returns ``None``).
    * In ``warning`` and ``hard_constraint`` modes the observer emits a
      soft :class:`CourseCorrection` (``abort=False``) the first time a
      turn touches a file outside both the localizer files set and the
      allowlist. Subsequent turns that re-touch the same off-target
      files don't re-fire (we attach a small de-dup memory keyed on
      observer id, captured via closure).

    Localization is a search prior, not a repository edit boundary. The
    observer should preserve useful exploration while making the agent
    explain and verify broader edits.
    """
    enforcement_norm = (enforcement or "advisory").strip().lower()
    localizer_set = {str(p).strip().lstrip("./") for p in localizer_files if p}
    allowlist_list = [str(p) for p in allowlist if p]

    seen_violators: set[str] = set()

    def observer(turn: Turn, ctx: AgentContext) -> Optional[CourseCorrection]:
        if enforcement_norm == "advisory":
            return None
        if not localizer_set:
            # No scope to enforce — silently no-op, matching the
            # post-validation hook's degrade-to-advisory behaviour.
            return None
        if not turn.files_touched:
            return None
        # Normalise paths the same way the localizer set was normalised.
        touched_norm = {str(p).strip().lstrip("./") for p in turn.files_touched}
        novel_violators: list[str] = []
        for path in sorted(touched_norm):
            if path in localizer_set:
                continue
            if _matches_any_glob(path, allowlist_list):
                continue
            if path in seen_violators:
                continue
            novel_violators.append(path)
            seen_violators.add(path)
        if not novel_violators:
            return None
        if enforcement_norm == "hard_constraint":
            guidance = (
                "This rollout is configured with high-severity localizer auditing; "
                "explain why the broader edit advances the objective, keep it "
                "minimal, and verify it before submitting."
            )
        else:
            guidance = (
                "Treat the localized files as a starting hypothesis, not a task "
                "boundary: explain why the broader edit is necessary, keep it "
                "minimal, and verify the behavior with tests."
            )
        message = (
            "[apex] Mid-turn localization note: you touched files outside the "
            "current localization hypothesis: {violators}. The localized files "
            "were: {expected}. {guidance}"
        ).format(
            violators=sorted(novel_violators),
            expected=sorted(localizer_set),
            guidance=guidance,
        )
        return CourseCorrection(
            message=message,
            abort=False,
            source="localizer_scope_observer",
            extra={
                "violators": sorted(novel_violators),
                "localizer_files": sorted(localizer_set),
                "enforcement": enforcement_norm,
                "turn_number": turn.number,
            },
        )

    return observer


# --- Composition helpers ---


def compose_observers(observers: Iterable[ObserverFn]) -> ObserverFn:
    """Combine multiple observers into one.

    Returns the FIRST non-None correction; observers later in the list
    are skipped once one fires. ``abort=True`` short-circuits the same
    way. Empty input yields a no-op observer.

    Errors raised inside an individual observer are logged and treated
    as "no correction" so a buggy observer cannot crash the rollout.
    """
    obs_list = [obs for obs in observers if obs is not None]
    if not obs_list:
        return lambda turn, ctx: None

    def _composed(turn: Turn, ctx: AgentContext) -> Optional[CourseCorrection]:
        for obs in obs_list:
            try:
                result = obs(turn, ctx)
            except Exception as exc:  # noqa: BLE001 - never crash the rollout
                logger.warning(
                    "Observer %r raised %s: %s; treating as no-correction.",
                    getattr(obs, "__name__", obs),
                    type(exc).__name__,
                    exc,
                )
                continue
            if result is not None:
                return result
        return None

    return _composed


__all__ = [
    "AgentContext",
    "CourseCorrection",
    "ObserverFn",
    "compose_observers",
    "localizer_scope_observer",
]
