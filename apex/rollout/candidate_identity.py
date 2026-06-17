"""Stable candidate identifiers shared by benchmark adapters."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..core.subprocess_utils import run_process_command


def patch_hash(patch_text: str) -> str:
    return hashlib.sha256(str(patch_text or "").encode("utf-8")).hexdigest()


def worktree_patch_hash(worktree_path: str | Path | None) -> str:
    if not worktree_path:
        return ""
    path = Path(worktree_path)
    if not path.exists():
        return ""
    try:
        result = run_process_command(
            ["git", "diff", "--binary", "HEAD"],
            cwd=path,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    text = result.stdout or ""
    if not text.strip():
        try:
            status = run_process_command(
                ["git", "status", "--porcelain"],
                cwd=path,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            status = None
        text = status.stdout if status is not None else ""
    return patch_hash(text) if text.strip() else ""


def candidate_id(
    *,
    task_id: str,
    origin_rollout_id: Optional[int],
    patch_id: str = "",
    stage: str = "candidate",
) -> str:
    parts = [
        str(task_id or "task"),
        str(origin_rollout_id if origin_rollout_id is not None else "none"),
        str(patch_id or "nopatch")[:16],
        str(stage or "candidate"),
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"cand-{digest}"


@dataclass
class CandidateIdentity:
    candidate_id: str
    origin_rollout_id: Optional[int]
    patch_id: str = ""
    workspace_id: str = ""
    generation_wave: str = "initial"
    parent_candidate_ids: list[str] = field(default_factory=list)
    selection_stage: str = "candidate"

    @classmethod
    def from_worktree(
        cls,
        *,
        task_id: str,
        origin_rollout_id: Optional[int],
        worktree_path: str | Path | None,
        generation_wave: str = "initial",
        selection_stage: str = "candidate",
        parent_candidate_ids: Optional[list[str]] = None,
    ) -> "CandidateIdentity":
        patch_id = worktree_patch_hash(worktree_path)
        cid = candidate_id(
            task_id=task_id,
            origin_rollout_id=origin_rollout_id,
            patch_id=patch_id,
            stage=selection_stage,
        )
        workspace_id = ""
        if worktree_path:
            try:
                workspace_id = Path(worktree_path).name
            except TypeError:
                workspace_id = str(worktree_path)
        return cls(
            candidate_id=cid,
            origin_rollout_id=origin_rollout_id,
            patch_id=patch_id,
            workspace_id=workspace_id,
            generation_wave=generation_wave,
            parent_candidate_ids=list(parent_candidate_ids or []),
            selection_stage=selection_stage,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "origin_rollout_id": self.origin_rollout_id,
            "patch_id": self.patch_id,
            "workspace_id": self.workspace_id,
            "generation_wave": self.generation_wave,
            "parent_candidate_ids": list(self.parent_candidate_ids),
            "selection_stage": self.selection_stage,
        }
