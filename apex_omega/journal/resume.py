"""resume_or_run wiring for ExecResult-shaped agent() calls (plan §15.7 #5).

The journal is wired INTO ``agent()`` itself (not per-stage callers), so
``parallel`` / ``pipeline`` / speculative trees become resumable for free.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from ..types import ExecResult
from .wal import Journal, RESULT_INFRA_NONRESULT, RESULT_OK


def _serialize_exec_result(result: ExecResult) -> tuple[dict, str, str, dict]:
    status = RESULT_OK if result.ok else RESULT_INFRA_NONRESULT
    return result.to_dict(), (result.fs_diff or ""), status, result.usage.to_dict()


def _deserialize_exec_result(d: dict, fs_diff_text: str) -> ExecResult:
    res = ExecResult.from_dict(d)
    if fs_diff_text:
        res.fs_diff = fs_diff_text
    return res


def resume_or_run_exec(
    journal: Journal,
    components: dict,
    runner: Callable[[], ExecResult],
    *,
    node_id: str = "",
    attempt: int = 0,
    materialize: Optional[Callable[[str], None]] = None,
) -> tuple[ExecResult, bool]:
    """Journaled execution of an ``agent()`` call returning an ExecResult.

    Returns ``(result, was_cache_hit)``.  On a HIT the recorded diff is
    optionally materialized back into the worktree.
    """
    return journal.get_or_run(
        components,
        runner,
        serialize=_serialize_exec_result,
        deserialize=_deserialize_exec_result,
        kind="agent",
        node_id=node_id,
        attempt=attempt,
        materialize=materialize,
    )


def resume_or_run_json(
    journal: Journal,
    components: dict,
    runner: Callable[[], Any],
    *,
    kind: str = "stage",
    node_id: str = "",
    attempt: int = 0,
    status_fn: Optional[Callable[[Any], str]] = None,
) -> tuple[Any, bool]:
    """Journaled execution of an arbitrary JSON-serializable step (pipeline /
    parallel stage results, controller decisions).  The result is stored
    verbatim as ``structured_result`` under a ``{"value": ...}`` envelope.

    ``status_fn(value) -> result_status`` classifies the outcome; a non-OK status
    (e.g. RESULT_INFRA_NONRESULT for a failed eval cell) is recorded for audit but
    is NOT a cache hit, so a later run re-runs it.  Defaults to always-OK."""

    def _ser(value: Any) -> tuple[dict, str, str, dict]:
        status = status_fn(value) if status_fn is not None else RESULT_OK
        return {"value": value}, "", status, {}

    def _de(d: dict, _fs: str) -> Any:
        return d.get("value")

    return journal.get_or_run(
        components, runner, serialize=_ser, deserialize=_de,
        kind=kind, node_id=node_id, attempt=attempt,
    )
