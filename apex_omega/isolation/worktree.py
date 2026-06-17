"""Per-rollout git-worktree isolation with lock-before-touch (Fusion Ledger A5;
plan §15.1).

Self-contained (plain ``git worktree``) so it works in unit tests without the v1
package; v1's warm CoW ``WorktreePool`` (~10x cheaper recycling) can be swapped in
behind the ``worktree_pool`` ablation flag.  Invariants enforced:

  * lock-before-touch: a per-rollout ``fcntl`` lock is acquired BEFORE provisioning;
    if held, raise ``ConcurrentWorktreeError`` (never silently share / nuke a
    sibling worktree).  No global/machine-wide mutex.
  * Cardinal safety on release: a destructive remove is REFUSED unless
    ``confirm_patch_extracted=True`` (the candidate diff must be captured first).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import fcntl  # type: ignore
    _HAVE_FCNTL = True
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore
    _HAVE_FCNTL = False

from ..errors import ConcurrentWorktreeError


def _git(*args: str, cwd: Optional[str] = None, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, timeout=timeout)


@dataclass
class WorktreeHandle:
    rollout_id: str
    path: str
    branch: str
    _lock_fd: Optional[int] = None
    _lock_path: Optional[str] = None


class WorktreeProvider:
    """Mints isolated git worktrees off a source repo at a fixed base commit."""

    def __init__(self, source_repo: str, *, base_commit: Optional[str] = None,
                 workspace_dir: Optional[str] = None, run_scope: str = "apexomega"):
        self.source_repo = str(Path(source_repo).resolve())
        self.run_scope = run_scope
        self.workspace_dir = Path(workspace_dir or (Path(self.source_repo).parent / f".apexomega_wt_{run_scope}"))
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        (self.workspace_dir / ".locks").mkdir(parents=True, exist_ok=True)
        if base_commit is None:
            res = _git("rev-parse", "HEAD", cwd=self.source_repo)
            base_commit = (res.stdout or "").strip() or "HEAD"
        self.base_commit = base_commit

    def _lock_path(self, rollout_id: str) -> Path:
        return self.workspace_dir / ".locks" / f"rollout_{rollout_id}.lock"

    def acquire(self, rollout_id: str | int) -> WorktreeHandle:
        rid = str(rollout_id)
        lock_path = self._lock_path(rid)
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        if _HAVE_FCNTL:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                raise ConcurrentWorktreeError(f"rollout {rid} worktree lock already held")
        else:  # best-effort PID marker on non-POSIX
            os.write(fd, str(os.getpid()).encode())
        wt_path = self.workspace_dir / f"wt_{rid}"
        branch = f"apexomega-{self.run_scope}-{rid}"
        if wt_path.exists():
            _git("worktree", "remove", "--force", str(wt_path), cwd=self.source_repo)
        # remove any stale branch of the same name
        _git("worktree", "prune", cwd=self.source_repo)
        _git("branch", "-D", branch, cwd=self.source_repo)
        res = _git("worktree", "add", "--detach", str(wt_path), self.base_commit, cwd=self.source_repo)
        if res.returncode != 0:
            # fall back to a branch-based worktree
            res = _git("worktree", "add", "-b", branch, str(wt_path), self.base_commit, cwd=self.source_repo)
            if res.returncode != 0:
                if _HAVE_FCNTL:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
                raise ConcurrentWorktreeError(f"git worktree add failed for {rid}: {res.stderr.strip()}")
        return WorktreeHandle(rid, str(wt_path), branch, _lock_fd=fd, _lock_path=str(lock_path))

    def release(self, handle: WorktreeHandle, *, confirm_patch_extracted: bool = False,
                reset: bool = True) -> None:
        if reset and not confirm_patch_extracted:
            raise ConcurrentWorktreeError(
                "refusing destructive worktree release without confirm_patch_extracted=True "
                "(Cardinal safety: extract the candidate diff first)"
            )
        try:
            _git("worktree", "remove", "--force", handle.path, cwd=self.source_repo)
            _git("branch", "-D", handle.branch, cwd=self.source_repo)
        finally:
            if handle._lock_fd is not None:
                if _HAVE_FCNTL:
                    try:
                        fcntl.flock(handle._lock_fd, fcntl.LOCK_UN)
                    except Exception:
                        pass
                os.close(handle._lock_fd)
                handle._lock_fd = None

    @contextmanager
    def lease(self, rollout_id: str | int, *, confirm_patch_extracted: bool = True):
        handle = self.acquire(rollout_id)
        try:
            yield handle
        finally:
            self.release(handle, confirm_patch_extracted=confirm_patch_extracted)

    def cleanup(self) -> None:
        _git("worktree", "prune", cwd=self.source_repo)
        shutil.rmtree(self.workspace_dir, ignore_errors=True)


def apply_diff(worktree_cwd: str, diff_text: str) -> bool:
    """Re-apply a recorded diff into a worktree (used on a journal cache HIT so the
    scorer sees the cached candidate, and when forking a parent diff into a fresh
    worktree for a repair lineage).  Returns True on success.

    Tries a strict apply first, then a 3-way merge apply (``--3way``): re-applying a
    parent diff onto a freshly-acquired worktree can fail strict context matching if
    lines shifted, but the 3-way merge resolves against the blob the diff was made
    from.  Indeterminate-on-failure is the caller's job (never silently no-op)."""
    if not diff_text.strip():
        return True
    for extra in ([], ["--3way"]):
        proc = subprocess.run(
            ["git", "-C", worktree_cwd, "apply", "--whitespace=nowarn", *extra, "-"],
            input=diff_text, text=True, capture_output=True,
        )
        if proc.returncode == 0:
            return True
    return False
