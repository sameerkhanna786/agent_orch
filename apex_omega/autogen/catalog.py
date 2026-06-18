"""Name -> orchestration-source registry for ``ctx.workflow()`` composition (dynamic-workflows
parity). An authored ``orchestrate(ctx)`` composes another workflow inline by NAME
(``ctx.workflow("decompose")``) or by reference (``ctx.workflow({"scriptPath": "..."})``).

This resolver runs HOST-SIDE (it is a ``ctx`` method, not authored code), so it may read the
catalog / a file — the orchestrator sandbox boundary is unaffected: authored code only ever
passes a plain string/dict literal to ``ctx.workflow``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .templates import (
    AUDIT_ORCHESTRATION,
    BEST_OF_N_ORCHESTRATION,
    DECOMPOSE_EXEMPLAR,
    DEFAULT_ORCHESTRATION,
    RALPH_ORCHESTRATION,
)

# Built-in, name-addressable workflows. Each value is a frozen ``orchestrate(ctx)`` source.
#
# ``default-best-of-n`` resolves to the CHEAP escalating best-of-N + repair path (the old
# default) — NOT the new convergence default — because the convergence default falls THROUGH to
# it for easy/single-module repos via ctx.workflow("default-best-of-n"). ``converge`` is the new
# decompose->fan-out->reduce->loop-until-dry default (DEFAULT_ORCHESTRATION).
BUILTIN_WORKFLOWS: dict[str, str] = {
    "default-best-of-n": BEST_OF_N_ORCHESTRATION,
    "converge": DEFAULT_ORCHESTRATION,
    "decompose": DECOMPOSE_EXEMPLAR,
    "ralph": RALPH_ORCHESTRATION,
    "audit": AUDIT_ORCHESTRATION,
}


def known_workflows() -> list[str]:
    return sorted(BUILTIN_WORKFLOWS)


def resolve_workflow(name_or_ref: Any) -> str:
    """Resolve a workflow NAME (catalog) or a by-REF dict ({"scriptPath": path}) to its
    ``orchestrate(ctx)`` source string. Raises KeyError (unknown name / bad ref) or OSError
    (unreadable scriptPath). The returned source is re-linted by ``run_orchestration``."""
    if isinstance(name_or_ref, dict):
        path = name_or_ref.get("scriptPath") or name_or_ref.get("script_path")
        if not path:
            raise KeyError(f"workflow ref missing 'scriptPath': {name_or_ref!r}")
        return Path(path).read_text(encoding="utf-8")
    key = str(name_or_ref)
    if key in BUILTIN_WORKFLOWS:
        return BUILTIN_WORKFLOWS[key]
    raise KeyError(f"unknown workflow {key!r} (known: {known_workflows()})")
