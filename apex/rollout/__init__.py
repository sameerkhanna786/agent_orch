"""Parallel rollout execution engine."""

from __future__ import annotations

from typing import Any

# Phase A.4: strategy-axis diversity. Re-exported eagerly so callers can
# `from apex.rollout import STRATEGY_AXES, assign_strategy` without
# triggering the lazy engine import below.
from .diversity_strategies import (
    STRATEGY_AXES,
    STRATEGY_PROMPT_PREFIXES,
    apply_strategy_prefix,
    assign_strategy,
    get_prompt_prefix,
)

# Decisive-Edge C.5: worktree pool. Re-exported eagerly so callers can
# `from apex.rollout import WorktreePool, WorktreePoolExhausted` without
# triggering the lazy engine import below.
from .worktree_pool import (
    PooledWorktree,
    WorktreePool,
    WorktreePoolExhausted,
)

__all__ = [
    "ConcurrentWorktreeError",
    "GitWorktreeManager",
    "RolloutEngine",
    "RolloutResult",
    # Strategy diversity
    "STRATEGY_AXES",
    "STRATEGY_PROMPT_PREFIXES",
    "apply_strategy_prefix",
    "assign_strategy",
    "get_prompt_prefix",
    # Worktree pool (C.5)
    "PooledWorktree",
    "WorktreePool",
    "WorktreePoolExhausted",
]


_LAZY_ENGINE_EXPORTS = {
    "ConcurrentWorktreeError",
    "GitWorktreeManager",
    "RolloutEngine",
    "RolloutResult",
}


# Backwards-compat: re-export common names eagerly via __all__ but keep
# the heavy ``engine.py`` import lazy through ``__getattr__``.


def __getattr__(name: str) -> Any:
    if name in _LAZY_ENGINE_EXPORTS:
        from .engine import (
            ConcurrentWorktreeError,
            GitWorktreeManager,
            RolloutEngine,
            RolloutResult,
        )

        exports = {
            "ConcurrentWorktreeError": ConcurrentWorktreeError,
            "GitWorktreeManager": GitWorktreeManager,
            "RolloutEngine": RolloutEngine,
            "RolloutResult": RolloutResult,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
