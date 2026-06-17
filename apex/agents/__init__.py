"""Agent implementations for issue solving."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "AgentResult",
    "BaseAgent",
    "FullSolverAgent",
    "HeuristicRepairAgent",
    "LocalizerAgent",
    "PatcherAgent",
    "ReproducerAgent",
]


if TYPE_CHECKING:
    from .heuristic import HeuristicRepairAgent
    from .solver import (
        AgentResult,
        BaseAgent,
        FullSolverAgent,
        LocalizerAgent,
        PatcherAgent,
        ReproducerAgent,
    )


def __getattr__(name: str) -> Any:
    if name == "HeuristicRepairAgent":
        from .heuristic import HeuristicRepairAgent

        return HeuristicRepairAgent
    if name in {
        "AgentResult",
        "BaseAgent",
        "FullSolverAgent",
        "LocalizerAgent",
        "PatcherAgent",
        "ReproducerAgent",
    }:
        from .solver import (
            AgentResult,
            BaseAgent,
            FullSolverAgent,
            LocalizerAgent,
            PatcherAgent,
            ReproducerAgent,
        )

        exports = {
            "AgentResult": AgentResult,
            "BaseAgent": BaseAgent,
            "FullSolverAgent": FullSolverAgent,
            "LocalizerAgent": LocalizerAgent,
            "PatcherAgent": PatcherAgent,
            "ReproducerAgent": ReproducerAgent,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
