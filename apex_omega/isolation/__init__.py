"""Per-rollout worktree isolation (lock-before-touch; Cardinal-safe release)."""

from .worktree import WorktreeHandle, WorktreeProvider, apply_diff

__all__ = ["WorktreeProvider", "WorktreeHandle", "apply_diff"]
