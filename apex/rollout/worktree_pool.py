"""Decisive-Edge C.5 — Worktree pool for cross-rollout reuse.

Background
----------
For a single task running ``K`` rollouts, the legacy
:class:`~apex.rollout.engine.GitWorktreeManager` creates a fresh
``rollout_<id>`` worktree per rollout (and tears it down afterwards).
Each ``git worktree add`` round-trips against the source repo (~0.5s)
and any per-task warmup (``pip install -e .``, dependency build, etc.)
runs from scratch — adding ~4s per rollout on commit0-style benchmarks.

When the rollouts on a given task run *sequentially* on a single worker
slot (the common case for K > parallel_workers), the warmup is pure
duplication: a clean checkout at the same baseline commit is functionally
identical between rollouts. The :class:`WorktreePool` exploits this by
pre-warming a small pool of worktrees per ``(task, base_commit)`` and
recycling pool entries between rollouts via ``git checkout .`` / ``git
clean -fdx``.

Safety contract
---------------
* The pool is **per-task** — it must NOT span tasks because the source
  repo / baseline differs.
* Acquire / release is governed by a ``threading.Semaphore`` sized at
  the pool's capacity, so rollouts running concurrently on the same
  pool block cleanly when the pool is exhausted.
* ``acquire(timeout)`` raises :class:`WorktreePoolExhausted` rather
  than blocking forever — the caller can fall back to a per-rollout
  worktree (the legacy code path).
* Reset uses ``git reset --hard {base_commit}`` followed by
  ``git clean -fdx`` so untracked test files / build artifacts don't
  leak between rollouts. When the reset fails the entry is marked
  ``unhealthy`` and replaced lazily on the next acquire.
* Pool teardown removes every entry's worktree and prunes its branch
  via ``git worktree remove --force`` + ``git branch -D`` — same
  surface as :meth:`GitWorktreeManager.cleanup_all`. The pool is a
  context manager so cleanup runs on exception too.

Branch naming
-------------
Each pool entry takes branch ``apex-pool-{run_scope}-{idx}`` so:

* a stale local branch from a prior crashed solve doesn't collide with
  ours (``-D`` runs eagerly during ``_create_pool_worktree``).
* concurrent solves with the same workspace_dir but different
  ``run_scope`` don't fight over the same branch namespace (mirrors
  ``GitWorktreeManager``'s ``apex-{run_scope}-rollout-{id}`` scheme).
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..core.git_utils import (
    is_ignored_change_path,
    parse_porcelain_path,
    relativize_git_worktree_metadata,
    run_git_with_lock_recovery,
    sync_git_submodules,
)

logger = logging.getLogger("apex.rollout.worktree_pool")

_BENCHMARK_HELPER_FILENAMES = (
    ".apex_expected_test_ids.txt",
    "_apex_expected_test_ids.txt",
    "_apex_expected_ids_filter.py",
    "_apex_run_expected_ids.py",
)


__all__ = [
    "WorktreePoolExhausted",
    "PooledWorktree",
    "WorktreePool",
]


class WorktreePoolExhausted(RuntimeError):
    """Raised when ``WorktreePool.acquire`` blocks past its timeout."""

    def __init__(self, *, pool_size: int, timeout: float):
        super().__init__(f"WorktreePool exhausted (pool_size={pool_size}, timeout={timeout}s)")
        self.pool_size = pool_size
        self.timeout = timeout


@dataclass
class PooledWorktree:
    """A worktree available for reuse across rollouts on the same task.

    ``in_use`` is updated under the pool's internal lock; readers
    outside the pool should treat the field as advisory.
    ``last_reset`` is a wall-clock timestamp (``time.time()``) of the
    last successful reset — useful for diagnostics and for skipping a
    redundant reset on the very first acquire after construction.
    """

    path: Path
    branch: str
    in_use: bool = False
    last_reset: float = 0.0
    healthy: bool = True
    rollout_id: Optional[str] = None  # advisory: who currently holds it
    pool_index: int = -1


@dataclass
class _PoolStats:
    """Diagnostics surface for tests and observability."""

    acquires: int = 0
    releases: int = 0
    timeouts: int = 0
    resets: int = 0
    reset_failures: int = 0
    create_failures: int = 0
    cleanup_failures: int = 0


class WorktreePool:
    """Maintains N pre-warmed worktrees per (task, base_commit).

    The pool is consumed via :meth:`acquire` / :meth:`release`, or via
    the convenience context manager :meth:`lease`. Each pool entry is
    a ``git worktree`` at ``pool_dir/pool_<idx>`` on branch
    ``apex-pool-{run_scope}-{idx}``.

    The pool is also a context manager itself — entering pre-warms
    every entry, exiting tears them down (always, even on exception).
    """

    def __init__(
        self,
        source_repo: Path | str,
        pool_size: int,
        run_scope: str,
        *,
        base_commit: Optional[str] = None,
        pool_dir: Optional[Path | str] = None,
        eager_warmup: bool = False,
    ) -> None:
        self.source_repo = Path(source_repo).resolve()
        self.pool_size = max(1, int(pool_size))
        scope = (str(run_scope or "").strip() or "default")[:32]
        self.run_scope = scope
        self.base_commit = (str(base_commit).strip() or None) if base_commit else None
        # Default to a sibling directory of the source repo so the pool
        # disappears with the workspace_dir cleanup. Tests may pass an
        # explicit pool_dir under ``tmp_path``.
        if pool_dir is not None:
            self.pool_dir = Path(pool_dir).resolve()
        else:
            self.pool_dir = (self.source_repo.parent / f".apex_pool_{self.run_scope}").resolve()
        self._semaphore = threading.Semaphore(self.pool_size)
        self._lock = threading.Lock()
        # Index -> entry. Lazily populated on first acquire when
        # eager_warmup is False (the default for fast unit tests).
        self._entries: dict[int, PooledWorktree] = {}
        self.stats = _PoolStats()
        self._closed = False
        # Run a defensive ``git worktree prune`` at most once per pool
        # lifetime (on the first acquire) to clear administrative records for
        # worktrees a crashed prior solve left behind; guarded so it never
        # serializes the hot acquire path.
        self._pruned_stale_worktrees = False
        if eager_warmup:
            self._warmup()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "WorktreePool":
        if not self._entries:
            self._warmup()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self.cleanup()

    # ------------------------------------------------------------------
    # Pre-warm + create pool entries
    # ------------------------------------------------------------------

    def _warmup(self) -> None:
        """Eagerly construct every pool entry up front.

        Failures during warmup do NOT crash the constructor: the entry
        slot is left empty and ``acquire`` will retry on demand. This
        matches the engine's "best-effort, fall back to per-rollout"
        contract.
        """
        for idx in range(self.pool_size):
            with self._lock:
                if idx in self._entries:
                    continue
            try:
                entry = self._create_pool_worktree(idx)
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                self.stats.create_failures += 1
                logger.warning(
                    "WorktreePool failed to pre-warm slot %s (%s: %s); will retry on demand.",
                    idx,
                    type(exc).__name__,
                    exc,
                )
                continue
            with self._lock:
                self._entries[idx] = entry

    def _branch_name(self, idx: int) -> str:
        return f"apex-pool-{self.run_scope}-{idx}"

    def _create_pool_worktree(self, idx: int) -> PooledWorktree:
        """Initial pool entry: clones via ``git worktree add``.

        Mirrors :meth:`GitWorktreeManager.create_worktree`'s native
        worktree path. Snapshot fallback isn't supported here — the
        engine falls back to the legacy per-rollout worktree if the
        pool can't materialize an entry.
        """
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        worktree_path = (self.pool_dir / f"pool_{idx}").resolve()
        if worktree_path.exists():
            shutil.rmtree(worktree_path, ignore_errors=True)
        branch_name = self._branch_name(idx)
        self._remove_stale_branch_worktrees(branch_name)
        # Drop a stale branch left behind by a prior crashed solve;
        # ``-D`` is idempotent so a missing branch is fine.
        subprocess.run(
            ["git", "branch", "-D", branch_name],
            cwd=str(self.source_repo),
            capture_output=True,
            check=False,
        )
        worktree_args = ["git", "worktree", "add", "-b", branch_name]
        if self.base_commit:
            worktree_args.extend([str(worktree_path), self.base_commit])
        else:
            worktree_args.append(str(worktree_path))
        result = subprocess.run(
            worktree_args,
            cwd=str(self.source_repo),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"git worktree add failed for pool slot {idx}: {err or 'no stderr'}")
        relativize_git_worktree_metadata(worktree_path)
        submodules = sync_git_submodules(worktree_path)
        if submodules is not None and submodules.returncode != 0:
            err = (submodules.stderr or submodules.stdout or "").strip()
            raise RuntimeError(
                f"git submodule update failed for pool slot {idx}: {err or 'no stderr'}"
            )
        self._sync_benchmark_helper_files(worktree_path)
        # Resolve the actual baseline commit once so reset can be
        # deterministic even if ``base_commit`` was None.
        resolved_baseline = self.base_commit or _resolve_head_commit(worktree_path)
        if not self.base_commit and resolved_baseline:
            self.base_commit = resolved_baseline
        entry = PooledWorktree(
            path=worktree_path,
            branch=branch_name,
            in_use=False,
            last_reset=time.time(),
            healthy=True,
            pool_index=idx,
        )
        return entry

    def _remove_stale_branch_worktrees(self, branch_name: str) -> None:
        """Remove crashed-run worktrees that still hold ``branch_name``.

        ``git branch -D`` cannot delete a branch that is checked out in any
        worktree. A prior interrupted benchmark can leave exactly that state,
        causing the next pool acquire to fall back for every colliding slot.
        """

        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=str(self.source_repo),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return
        current_path: Optional[str] = None
        current_branch: Optional[str] = None

        def flush() -> None:
            nonlocal current_path, current_branch
            if current_path and current_branch == f"refs/heads/{branch_name}":
                subprocess.run(
                    ["git", "worktree", "remove", "--force", current_path],
                    cwd=str(self.source_repo),
                    capture_output=True,
                    check=False,
                )
            current_path = None
            current_branch = None

        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                flush()
                continue
            if line.startswith("worktree "):
                current_path = line[len("worktree ") :]
            elif line.startswith("branch "):
                current_branch = line[len("branch ") :]
        flush()

    # ------------------------------------------------------------------
    # Reset between rollouts
    # ------------------------------------------------------------------

    def _log_dirty_fingerprint_before_reset(self, pooled: PooledWorktree) -> None:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=str(pooled.path),
            capture_output=True,
            text=True,
            check=False,
        )
        diff = subprocess.run(
            ["git", "diff", "--binary", "--no-ext-diff"],
            cwd=str(pooled.path),
            capture_output=True,
            text=True,
            check=False,
        )
        if status.returncode != 0 or diff.returncode != 0:
            return
        status_lines = [
            line
            for line in (status.stdout or "").splitlines()
            if line.strip() and not is_ignored_change_path(parse_porcelain_path(line))
        ]
        status_text = "\n".join(status_lines)
        diff_text = diff.stdout or ""
        payload = f"{status_text}\n{diff_text}"
        if not payload.strip():
            return
        fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        dirty_entries = len(status_lines)
        logger.warning(
            "WorktreePool dirty worktree before reset: path=%s fingerprint=%s entries=%s",
            pooled.path,
            fingerprint,
            dirty_entries,
        )

    def _reset_worktree(self, pooled: PooledWorktree) -> bool:
        """Bring worktree back to clean baseline (between rollouts).

        Returns ``True`` on a clean reset, ``False`` when the worktree
        is unsalvageable. The pool replaces unhealthy entries on the
        next acquire.
        """
        try:
            self._log_dirty_fingerprint_before_reset(pooled)
            # Stage 1: drop tracked changes. ``git checkout .`` is the
            # cheapest "revert tracked files to HEAD" command and works
            # even when HEAD is detached. Route through lock-recovery so a
            # stale ``.git/worktrees/pool_N/index.lock`` (left by a crashed
            # sibling writer) is cleared and retried instead of flipping the
            # entry unhealthy and burning a warm slot.
            checkout = run_git_with_lock_recovery(
                ["git", "checkout", "."],
                cwd=str(pooled.path),
            )
            if checkout.returncode != 0:
                logger.warning(
                    "WorktreePool reset (`git checkout .`) failed for %s: %s",
                    pooled.path,
                    (checkout.stderr or checkout.stdout or "").strip(),
                )
                self.stats.reset_failures += 1
                return False
            # Stage 2: drop untracked files / dirs / ignored
            # (build artefacts, ``__pycache__``, generated tests).
            clean = run_git_with_lock_recovery(
                ["git", "clean", "-fdx"],
                cwd=str(pooled.path),
            )
            if clean.returncode != 0:
                logger.warning(
                    "WorktreePool reset (`git clean -fdx`) failed for %s: %s",
                    pooled.path,
                    (clean.stderr or clean.stdout or "").strip(),
                )
                self.stats.reset_failures += 1
                return False
            # Stage 3: hard-reset to the pinned baseline so any
            # baseline drift (e.g. a prior rollout that committed)
            # snaps back. Only meaningful when we know the baseline.
            if self.base_commit:
                reset = run_git_with_lock_recovery(
                    ["git", "reset", "--hard", self.base_commit],
                    cwd=str(pooled.path),
                )
                if reset.returncode != 0:
                    logger.warning(
                        "WorktreePool reset (`git reset --hard %s`) failed for %s: %s",
                        self.base_commit,
                        pooled.path,
                        (reset.stderr or reset.stdout or "").strip(),
                    )
                    self.stats.reset_failures += 1
                    return False
            submodules = sync_git_submodules(pooled.path)
            if submodules is not None and submodules.returncode != 0:
                logger.warning(
                    "WorktreePool reset (`git submodule update --init --recursive`) failed for %s: %s",
                    pooled.path,
                    (submodules.stderr or submodules.stdout or "").strip(),
                )
                self.stats.reset_failures += 1
                return False
            self._sync_benchmark_helper_files(pooled.path)
            pooled.last_reset = time.time()
            self.stats.resets += 1
            return True
        except Exception as exc:  # noqa: BLE001 - never crash the caller
            logger.warning(
                "WorktreePool reset raised for %s (%s: %s); marking unhealthy.",
                pooled.path,
                type(exc).__name__,
                exc,
            )
            self.stats.reset_failures += 1
            return False

    def _sync_benchmark_helper_files(self, destination: Path) -> None:
        """Copy ignored benchmark helper files from the source checkout.

        Commit0 stages its public expected-test inventory and pytest wrapper
        as ignored, untracked files. ``git worktree add`` and ``git clean
        -fdx`` do not preserve them, so the pool mirrors them after creation
        and every reset while still leaving them excluded from submitted diffs.
        """

        for filename in _BENCHMARK_HELPER_FILENAMES:
            source = self.source_repo / filename
            target = destination / filename
            try:
                if source.exists() or source.is_symlink():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists() or target.is_symlink():
                        if target.is_dir() and not target.is_symlink():
                            shutil.rmtree(target, ignore_errors=True)
                        else:
                            target.unlink()
                    shutil.copy2(source, target, follow_symlinks=False)
                elif target.exists() or target.is_symlink():
                    if target.is_dir() and not target.is_symlink():
                        shutil.rmtree(target, ignore_errors=True)
                    else:
                        target.unlink()
            except OSError as exc:
                logger.debug(
                    "Failed to sync benchmark helper %s into %s: %s",
                    filename,
                    destination,
                    exc,
                )

    # ------------------------------------------------------------------
    # Acquire / release
    # ------------------------------------------------------------------

    def acquire(
        self,
        rollout_id: str | int = "",
        timeout: float = 60.0,
    ) -> PooledWorktree:
        """Block until a worktree is available, or until ``timeout``.

        Resets state before returning so the caller always observes a
        clean baseline. Raises :class:`WorktreePoolExhausted` when the
        pool can't service the request inside the timeout window.
        """
        if self._closed:
            raise WorktreePoolExhausted(pool_size=self.pool_size, timeout=timeout)
        self._maybe_prune_stale_worktrees()
        timeout = max(0.0, float(timeout or 0.0))
        # ``acquire(timeout=0)`` is effectively non-blocking; ``Semaphore``
        # treats ``False`` as the only blocking-disable signal so the
        # value vs. wait semantics need a small adapter.
        if timeout <= 0.0:
            acquired = self._semaphore.acquire(blocking=False)
        else:
            acquired = self._semaphore.acquire(blocking=True, timeout=timeout)
        if not acquired:
            self.stats.timeouts += 1
            raise WorktreePoolExhausted(pool_size=self.pool_size, timeout=timeout)
        try:
            entry = self._claim_or_create_entry(rollout_id=str(rollout_id or ""))
        except Exception:
            # Failed to materialize an entry — give the slot back so
            # other rollouts aren't starved.
            self._semaphore.release()
            raise
        self.stats.acquires += 1
        return entry

    def _maybe_prune_stale_worktrees(self) -> None:
        """Best-effort ``git worktree prune`` once per pool lifetime.

        Cheap (a single porcelain-free git admin command against the source
        repo) and idempotent. Failures are swallowed — pruning is a hygiene
        step, never a correctness requirement. Guarded by a flag so the hot
        acquire path doesn't shell out on every call.
        """

        with self._lock:
            if self._pruned_stale_worktrees:
                return
            self._pruned_stale_worktrees = True
        try:
            run_git_with_lock_recovery(
                ["git", "worktree", "prune"],
                cwd=str(self.source_repo),
            )
        except Exception:  # noqa: BLE001 - hygiene step must never block acquire
            logger.debug(
                "WorktreePool defensive `git worktree prune` failed for %s",
                self.source_repo,
                exc_info=True,
            )

    def _claim_or_create_entry(self, *, rollout_id: str) -> PooledWorktree:
        """Atomically pick a free entry, creating one on demand.

        Marks the chosen entry ``in_use`` and resets it before
        returning. Replaces unhealthy entries lazily.
        """
        with self._lock:
            # Prefer an existing healthy free slot; fall back to creating
            # the lowest-index missing slot so the pool grows
            # deterministically.
            free_entry: Optional[PooledWorktree] = None
            for idx in range(self.pool_size):
                entry = self._entries.get(idx)
                if entry is None:
                    continue
                if entry.in_use:
                    continue
                if not entry.healthy:
                    continue
                free_entry = entry
                free_entry.in_use = True
                free_entry.rollout_id = rollout_id
                break
            create_idx: Optional[int] = None
            replace_idx: Optional[int] = None
            if free_entry is None:
                for idx in range(self.pool_size):
                    entry = self._entries.get(idx)
                    if entry is None:
                        create_idx = idx
                        break
                if create_idx is None:
                    for idx in range(self.pool_size):
                        entry = self._entries.get(idx)
                        if entry is None:
                            continue
                        if not entry.in_use and not entry.healthy:
                            replace_idx = idx
                            break
        if free_entry is not None:
            # Reset outside the lock — subprocess calls can take 100ms+
            # and would serialize concurrent acquires otherwise.
            ok = self._reset_worktree(free_entry)
            if not ok:
                with self._lock:
                    free_entry.healthy = False
                    free_entry.in_use = False
                    free_entry.rollout_id = None
                # Recurse to surface the next free slot (or create one).
                return self._claim_or_create_entry(rollout_id=rollout_id)
            return free_entry
        if create_idx is not None:
            idx = create_idx
        elif replace_idx is not None:
            idx = replace_idx
        else:
            # Should be unreachable — semaphore guarantees pool_size slots.
            raise WorktreePoolExhausted(pool_size=self.pool_size, timeout=0.0)
        # Tear down the unhealthy entry first so ``_create_pool_worktree``
        # finds a clean slot.
        with self._lock:
            stale = self._entries.pop(idx, None)
        if stale is not None:
            self._destroy_entry(stale)
        new_entry = self._create_pool_worktree(idx)
        with self._lock:
            self._entries[idx] = new_entry
            new_entry.in_use = True
            new_entry.rollout_id = rollout_id
        return new_entry

    def release(
        self,
        pooled: PooledWorktree,
        *,
        reset: bool = True,
        confirm_patch_extracted: bool = False,
    ) -> None:
        """Return worktree to pool.

        ``reset=True`` (the default) runs ``git reset --hard`` and
        ``git clean -fdx`` so the next acquire sees a clean baseline.
        ``reset=False`` is useful when the caller knows the worktree is
        about to be destroyed (e.g. after a fatal exception).
        ``confirm_patch_extracted=True`` is required before destructive reset
        so candidate edits cannot be wiped before the caller persists them.
        """
        if pooled is None:
            return
        try:
            if reset:
                if not confirm_patch_extracted:
                    logger.warning(
                        "WorktreePool refused reset without patch extraction "
                        "confirmation: path=%s rollout_id=%s",
                        pooled.path,
                        pooled.rollout_id,
                    )
                    pooled.healthy = False
                    self.stats.reset_failures += 1
                else:
                    ok = self._reset_worktree(pooled)
                    if not ok:
                        pooled.healthy = False
        finally:
            with self._lock:
                pooled.in_use = False
                pooled.rollout_id = None
            self.stats.releases += 1
            self._semaphore.release()

    # ------------------------------------------------------------------
    # Convenience context manager for `with pool.lease(...) as entry:`
    # ------------------------------------------------------------------

    def lease(
        self,
        rollout_id: str | int = "",
        timeout: float = 60.0,
        *,
        confirm_patch_extracted: bool = False,
    ) -> "_PoolLease":
        return _PoolLease(
            self,
            rollout_id=str(rollout_id or ""),
            timeout=timeout,
            confirm_patch_extracted=confirm_patch_extracted,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Remove every pool worktree and prune branches.

        Idempotent: safe to call multiple times. Marks the pool closed
        so subsequent ``acquire`` calls fail fast.
        """
        if self._closed:
            return
        self._closed = True
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            self._destroy_entry(entry)
        # Drop the pool dir if it's empty.
        try:
            if self.pool_dir.exists() and not any(self.pool_dir.iterdir()):
                self.pool_dir.rmdir()
        except OSError:
            pass
        # Final ``git worktree prune`` to clean up any administrative
        # state left behind in the source repo.
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=str(self.source_repo),
            capture_output=True,
            check=False,
        )

    def _destroy_entry(self, entry: PooledWorktree) -> None:
        """Tear down a single pool entry; never raises."""
        try:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(entry.path)],
                cwd=str(self.source_repo),
                capture_output=True,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001
            self.stats.cleanup_failures += 1
            logger.warning(
                "WorktreePool failed to remove worktree %s (%s: %s)",
                entry.path,
                type(exc).__name__,
                exc,
            )
        try:
            subprocess.run(
                ["git", "branch", "-D", entry.branch],
                cwd=str(self.source_repo),
                capture_output=True,
                check=False,
            )
        except Exception:  # noqa: BLE001
            pass
        if entry.path.exists():
            try:
                shutil.rmtree(entry.path, ignore_errors=True)
            except OSError:
                self.stats.cleanup_failures += 1


@dataclass
class _PoolLease:
    """Context-managed acquire/release sugar for ``pool.lease(...)``."""

    pool: WorktreePool
    rollout_id: str
    timeout: float
    confirm_patch_extracted: bool = False
    _entry: Optional[PooledWorktree] = field(default=None, init=False)

    def __enter__(self) -> PooledWorktree:
        self._entry = self.pool.acquire(rollout_id=self.rollout_id, timeout=self.timeout)
        return self._entry

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._entry is not None:
            # On exception, skip the reset so the unhealthy entry is
            # caught by the next acquire's reset attempt.
            self.pool.release(
                self._entry,
                reset=exc_type is None,
                confirm_patch_extracted=self.confirm_patch_extracted,
            )


def _resolve_head_commit(repo_path: Path) -> str:
    """Return ``git rev-parse HEAD`` for ``repo_path`` or ``""``."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return ""
