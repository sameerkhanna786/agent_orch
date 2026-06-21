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

    def _marker_pid_alive(self, fd: int) -> bool:
        """NEW-I8: True iff the lock-file PID marker names a LIVE process. An empty / garbage / dead
        marker is treated as stale (False) so a lock left by a crashed/killed provider can be
        reclaimed on resume instead of blocking forever. A live foreign (or same-process) holder is a
        genuine conflict (True)."""
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            raw = os.read(fd, 32).decode(errors="ignore").strip()
        except OSError:
            return False
        if not raw.isdigit():
            return False
        pid = int(raw)
        if pid == os.getpid():
            return True
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False

    def _reclaim_worktree_path(self, wt_path: Path) -> bool:
        """NEW-I1/I4: hard-clear a stale wt_<rid> (a registered worktree OR a crash orphan dir) so a
        resume can re-acquire it. `git worktree remove --force` -> prune -> rmtree (incl. a symlink)
        -> prune again, then verify the path is gone. Returns True iff clear afterward."""
        _git("worktree", "remove", "--force", str(wt_path), cwd=self.source_repo)
        _git("worktree", "prune", cwd=self.source_repo)
        try:
            if wt_path.is_symlink():
                wt_path.unlink()
            elif wt_path.exists():
                shutil.rmtree(wt_path, ignore_errors=True)
        except OSError:
            pass
        _git("worktree", "prune", cwd=self.source_repo)
        return not (wt_path.exists() or wt_path.is_symlink())

    def acquire(self, rollout_id: str | int) -> WorktreeHandle:
        rid = str(rollout_id)
        lock_path = self._lock_path(rid)
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        if _HAVE_FCNTL:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError):
                # NEW-I8: a held lock from a DEAD pid (crash/kill) is reclaimed; a LIVE holder conflicts.
                if self._marker_pid_alive(fd):
                    os.close(fd)
                    raise ConcurrentWorktreeError(f"rollout {rid} worktree lock already held (live)")
                os.close(fd)
                fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except (BlockingIOError, OSError):
                    os.close(fd)
                    raise ConcurrentWorktreeError(f"rollout {rid} worktree lock contended after reclaim")
        else:  # non-POSIX: PID marker with the same dead-PID reclamation semantics
            if self._marker_pid_alive(fd):
                os.close(fd)
                raise ConcurrentWorktreeError(f"rollout {rid} worktree lock already held (live)")
        # stamp our own PID so a future resume can reason about liveness (NEW-I8)
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, str(os.getpid()).encode())
            os.fsync(fd)
        except OSError:
            pass
        wt_path = self.workspace_dir / f"wt_{rid}"
        branch = f"apexomega-{self.run_scope}-{rid}"
        # NEW-I1/I4: hard-reclaim a stale leftover worktree before add; FAIL LOUD if it can't be
        # cleared (the old code git-worktree-removed only if it existed and continued regardless,
        # causing "worktree already exists" + building onto a dirty tree on resume).
        if wt_path.exists() or wt_path.is_symlink():
            if not self._reclaim_worktree_path(wt_path):
                if _HAVE_FCNTL:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
                raise ConcurrentWorktreeError(
                    f"rollout {rid}: stale worktree {wt_path} could not be cleared")
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


def _diff_target_files(diff_text: str) -> set[str]:
    """Repo-relative paths a unified diff writes to (the ``+++ b/<path>`` side), excluding deletions."""
    files: set[str] = set()
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            p = line[4:].strip()
            if p.startswith("b/"):
                p = p[2:]
            if p and p != "/dev/null":
                files.add(p)
    return files


def _has_conflict_markers(worktree_cwd: str, files: set[str]) -> bool:
    """NEW-I3/I9: True iff any patched file carries the FULL git conflict triplet
    (``<<<<<<<`` + ``=======`` + ``>>>>>>>``). A ``--3way`` apply can return success while leaving
    such markers; scoring that poisoned (un-parseable) tree silently corrupts the candidate, so the
    caller must treat marker-presence as an apply FAILURE (indeterminate), never a clean apply."""
    for f in files:
        fp = os.path.join(worktree_cwd, f)
        try:
            with open(fp, encoding="utf-8", errors="replace") as fh:
                t = fh.read()
        except OSError:
            continue
        if ("<<<<<<<" in t) and ("=======" in t) and (">>>>>>>" in t):
            return True
    return False


def apply_diff(worktree_cwd: str, diff_text: str) -> bool:
    """Re-apply a recorded diff into a worktree (used on a journal cache HIT so the
    scorer sees the cached candidate, and when forking a parent diff into a fresh
    worktree for a repair lineage).  Returns True on success.

    Tries a strict apply first, then a 3-way merge apply (``--3way``): re-applying a
    parent diff onto a freshly-acquired worktree can fail strict context matching if
    lines shifted, but the 3-way merge resolves against the blob the diff was made
    from.  Indeterminate-on-failure is the caller's job (never silently no-op).

    NEW-I3/I9: a ``--3way`` success that LEFT conflict markers is NOT a clean apply — we report
    failure so the caller treats it as indeterminate rather than scoring a poisoned tree."""
    if not diff_text.strip():
        return True
    targets = _diff_target_files(diff_text)
    for extra in ([], ["--3way"]):
        proc = subprocess.run(
            ["git", "-C", worktree_cwd, "apply", "--whitespace=nowarn", *extra, "-"],
            input=diff_text, text=True, capture_output=True,
        )
        if proc.returncode == 0:
            if extra and _has_conflict_markers(worktree_cwd, targets):
                return False  # 3way merged but left conflict markers -> poisoned, not clean
            return True
    return False


def apply_diff_partial(worktree_cwd: str, diff_text: str) -> dict:
    """HUNK-LEVEL partial apply for the merge/reduce step (merge-reduce-overhaul #1).

    Tries strict then ``--3way`` first — the cheap CLEAN path, identical to ``apply_diff`` (zero
    extra cost for a disjoint module). Only when the whole patch can't apply does it fall back to
    ``git apply --reject``, which lands EVERY hunk that DOES apply and writes the truly-conflicting
    hunks to ``*.rej`` — so a module that collides on one shared hunk keeps its other (often ~80-90%)
    hunks instead of being dropped wholesale (the dominant source of the converge<<ralph gap on
    tightly-coupled repos). The ``*.rej`` files are immediately DELETED so they can never pollute the
    scored/merged worktree diff (Cardinal-Contract hygiene). Returns
    ``{"clean", "applied_any", "rejected_hunks"}``. NEVER raises. SAFE by construction: the merged
    tree is always re-scored on the full gold suite, and the no-silent-loss floor reverts any partial
    graft that lowers the score — so a partial apply can never fake a pass, only (at worst) be discarded."""
    import glob as _glob
    if not diff_text.strip():
        return {"clean": True, "applied_any": False, "rejected_hunks": 0}
    targets = _diff_target_files(diff_text)
    for extra in ([], ["--3way"]):
        proc = subprocess.run(
            ["git", "-C", worktree_cwd, "apply", "--whitespace=nowarn", *extra, "-"],
            input=diff_text, text=True, capture_output=True,
        )
        if proc.returncode == 0:
            # NEW-I3/I9: a 3way that left conflict markers is NOT clean; surface it so the
            # no-silent-loss floor re-scores + reverts the poisoned graft instead of banking it.
            if extra and _has_conflict_markers(worktree_cwd, targets):
                return {"clean": False, "applied_any": True, "rejected_hunks": 1}
            return {"clean": True, "applied_any": True, "rejected_hunks": 0}
    # PARTIAL: land every applicable hunk; conflicting hunks -> *.rej (then deleted).
    try:
        proc = subprocess.run(
            ["git", "-C", worktree_cwd, "apply", "--reject", "--whitespace=nowarn", "-"],
            input=diff_text, text=True, capture_output=True,
        )
    except Exception:
        return {"clean": False, "applied_any": False, "rejected_hunks": 0}
    n_rej = 0
    for r in _glob.glob(os.path.join(worktree_cwd, "**", "*.rej"), recursive=True):
        try:
            with open(r, encoding="utf-8", errors="replace") as fh:
                n_rej += max(1, fh.read().count("@@ "))   # ~1 per rejected hunk
        except OSError:
            n_rej += 1
        try:
            os.remove(r)
        except OSError:
            pass
    # `git apply --reject` returns 0 (all hunks applied — rare here) or 1 (some rejected, rest applied).
    applied_any = proc.returncode in (0, 1)
    return {"clean": False, "applied_any": applied_any, "rejected_hunks": n_rej}
