"""Persistent, file-backed episodic memory across solves (Phase 6 item 6.2).

The in-memory ``EpisodicMemoryBus`` (``apex.rollout.engine``) is per-
solve: parallel rollouts within ONE solve see each other's discoveries,
but the next solve on the same task starts fresh. ``EpisodicStore``
extends that bus to a JSONL-backed log keyed by a *task signature*
(typically the per-repo signature from
:func:`apex.persistence.repo_memory.repo_signature_for_path` plus a
caller-supplied ``task_id``).

Design notes
------------

* **Append-only JSONL** — episodes accumulate, never get rewritten in
  place. This is intentionally a write-amplifying log so we can reason
  about ordering and recover from concurrent writers easily.
* **Atomic appends via fcntl.flock** — multiple processes (e.g.
  parallel benchmark workers solving the same task on shards) may
  write concurrently. We hold an exclusive flock on the JSONL file for
  the duration of one append. Reads do NOT take the lock — they are
  best-effort snapshots and tolerant of partial trailing lines.
* **Schema versioning** — each line carries a ``v`` field; readers
  ignore lines with unsupported versions and log a warning.
* **No automatic decay** — unlike ``RepoMemoryStore`` we don't decay
  episodes. Callers downstream may decay confidences when synthesising
  hypotheses; the store itself stays raw so downstream consumers can
  reinterpret history as policies evolve.
* **Storage layout** — ``~/.apex/episodic/<task_signature>/episodes.jsonl``.
  The default root is overridable via ``directory=`` for tests and for
  callers that want a per-benchmark scratch dir.

Public types
------------

* :class:`Episode` — one record (rollout id, episode type, payload).
* :class:`EpisodicStore` — broadcast / query API.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("apex.persistence.episodic_store")


# fcntl is POSIX-only; the storage layer degrades to a thread lock with a
# logged warning on Windows. Benchmark workers always run on Linux/macOS.
try:
    import fcntl  # type: ignore[unused-ignore]

    _FCNTL_AVAILABLE = True
except ImportError:  # pragma: no cover — Windows fallback
    fcntl = None  # type: ignore[assignment]
    _FCNTL_AVAILABLE = False


_EPISODIC_SCHEMA_VERSION = 1
_DEFAULT_STORE_DIRNAME = ".apex/episodic"
_LOCK_TIMEOUT_SECONDS = 30.0


def _now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _resolve_store_root(directory: Optional[str]) -> Path:
    if directory:
        return Path(directory).expanduser().resolve()
    return Path(os.path.expanduser("~")) / _DEFAULT_STORE_DIRNAME


def _sanitize_task_signature(task_signature: str) -> str:
    """Defensive cleanup so a malicious task signature can't escape the root.

    Episodic store paths are joined as ``root / task_signature /
    episodes.jsonl``; an attacker-supplied signature like ``../../etc``
    must not be allowed. We drop slashes and parent-dir tokens, and
    cap length so we can't blow past PATH_MAX.
    """
    raw = (task_signature or "").strip() or "unknown"
    # strip directory separators and parent-dir tokens
    parts = raw.replace("\\", "/").split("/")
    safe_parts = [p for p in parts if p not in ("", ".", "..")]
    safe = "_".join(safe_parts) if safe_parts else "unknown"
    return safe[:128]


@dataclass
class Episode:
    """One episodic record carried across solves."""

    task_signature: str
    rollout_id: str
    episode_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    timestamp_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "v": _EPISODIC_SCHEMA_VERSION,
            "task_signature": str(self.task_signature),
            "rollout_id": str(self.rollout_id),
            "episode_type": str(self.episode_type),
            "payload": dict(self.payload),
            "timestamp": float(self.timestamp),
            "timestamp_utc": str(self.timestamp_utc),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Episode":
        return cls(
            task_signature=str(data.get("task_signature") or ""),
            rollout_id=str(data.get("rollout_id") or ""),
            episode_type=str(data.get("episode_type") or ""),
            payload=dict(data.get("payload") or {}),
            timestamp=float(data.get("timestamp") or 0.0),
            timestamp_utc=str(data.get("timestamp_utc") or ""),
        )


class EpisodicStore:
    """JSONL-backed cross-solve episodic memory.

    Threadsafe within a process via ``self._lock`` and process-safe via
    ``fcntl.flock`` on POSIX. The default storage root is
    ``~/.apex/episodic``; tests pass ``directory=tmp_path`` to keep
    state out of the user home.
    """

    def __init__(
        self,
        *,
        directory: Optional[str] = None,
        max_age_seconds: Optional[float] = None,
    ) -> None:
        self._root = _resolve_store_root(directory)
        # Default retention: keep everything. Callers that want a TTL
        # (e.g. benchmark sweeps that don't want week-old priors leaking
        # into a fresh run) pass ``max_age_seconds``. ``None`` means
        # "no TTL".
        self._default_max_age_seconds = (
            float(max_age_seconds) if max_age_seconds is not None else None
        )
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def store_path(self, task_signature: str) -> Path:
        safe = _sanitize_task_signature(task_signature)
        return self._root / safe / "episodes.jsonl"

    # ------------------------------------------------------------------
    # Mutating API
    # ------------------------------------------------------------------

    def broadcast(
        self,
        *,
        task_signature: str,
        rollout_id: str,
        episode_type: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> Episode:
        """Append one episode to the store.

        Returns the persisted ``Episode`` so callers can chain. Raises
        ``ValueError`` for empty ``episode_type`` (defensive: an empty
        type collapses indexing).
        """
        if not episode_type or not str(episode_type).strip():
            raise ValueError("episode_type must be non-empty")
        episode = Episode(
            task_signature=str(task_signature or ""),
            rollout_id=str(rollout_id or ""),
            episode_type=str(episode_type).strip(),
            payload=dict(payload or {}),
            timestamp=time.time(),
            timestamp_utc=_now_utc_iso(),
        )
        path = self.store_path(task_signature)
        line = json.dumps(episode.to_dict(), sort_keys=True) + "\n"

        with self._lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning(
                    "EpisodicStore: failed to create parent dir %s: %s",
                    path.parent,
                    exc,
                )
                return episode

            try:
                # ``ab`` so concurrent appends don't truncate. fcntl.flock
                # serialises across processes; the per-instance lock
                # serialises within a process. Open inside the with-block
                # so we can flock the file descriptor and release on
                # close even if the write raises.
                fd = open(path, "ab")
            except OSError as exc:
                logger.warning(
                    "EpisodicStore: failed to open %s for append: %s",
                    path,
                    exc,
                )
                return episode

            try:
                self._acquire_flock_locked(fd)
                try:
                    fd.write(line.encode("utf-8"))
                    fd.flush()
                    try:
                        os.fsync(fd.fileno())
                    except OSError:
                        # fsync can fail on some fs (e.g. tmpfs in CI);
                        # the data is still in the page cache so we
                        # accept the risk and keep going.
                        pass
                finally:
                    self._release_flock_locked(fd)
            finally:
                fd.close()
        return episode

    # ------------------------------------------------------------------
    # Reading API
    # ------------------------------------------------------------------

    def query(
        self,
        *,
        task_signature: str,
        episode_type: Optional[str] = None,
        max_age_seconds: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> list[Episode]:
        """Return episodes for ``task_signature`` in append order.

        ``episode_type`` filters by type (case-sensitive after strip).
        ``max_age_seconds`` filters out episodes older than the given
        TTL relative to ``time.time()``. ``limit`` caps the returned
        list to the most recent N episodes (still in chronological
        order).

        Reads do NOT take the cross-process lock. Concurrent appends
        may produce a partial trailing line; we tolerate that by
        skipping any line that fails to parse (with a debug log) so
        the caller never sees an exception from a half-flushed write.
        """
        path = self.store_path(task_signature)
        if not path.is_file():
            return []
        # Resolve TTL: explicit arg wins over instance default.
        ttl = max_age_seconds if max_age_seconds is not None else self._default_max_age_seconds
        cutoff = (time.time() - float(ttl)) if ttl is not None else None
        normalized_type = (episode_type or "").strip() if episode_type else None

        episodes: list[Episode] = []
        try:
            with open(path, "rb") as fd:
                # Take a SHARED flock so we don't see a partial mid-line
                # write from a concurrent broadcaster. POSIX open(rb)
                # supports fcntl.LOCK_SH. On Windows we just read.
                self._acquire_flock_locked(fd, exclusive=False)
                try:
                    raw = fd.read().decode("utf-8", errors="replace")
                finally:
                    self._release_flock_locked(fd)
        except OSError as exc:
            logger.warning("EpisodicStore: failed to read %s: %s", path, exc)
            return []

        for line_no, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                logger.debug(
                    "EpisodicStore: skipping unparseable line %d in %s",
                    line_no,
                    path,
                )
                continue
            if not isinstance(obj, dict):
                continue
            version = int(obj.get("v") or 0)
            if version != _EPISODIC_SCHEMA_VERSION:
                logger.debug(
                    "EpisodicStore: skipping line %d in %s with version %r",
                    line_no,
                    path,
                    obj.get("v"),
                )
                continue
            episode = Episode.from_dict(obj)
            if normalized_type and episode.episode_type != normalized_type:
                continue
            if cutoff is not None and episode.timestamp < cutoff:
                continue
            episodes.append(episode)
        if limit is not None and limit >= 0:
            episodes = episodes[-int(limit) :]
        return episodes

    def task_signatures(self) -> list[str]:
        """List all task signatures currently in the store."""
        if not self._root.is_dir():
            return []
        out: list[str] = []
        try:
            for entry in self._root.iterdir():
                if entry.is_dir() and (entry / "episodes.jsonl").is_file():
                    out.append(entry.name)
        except OSError as exc:  # pragma: no cover — defensive
            logger.warning("EpisodicStore: failed to list root %s: %s", self._root, exc)
            return []
        return sorted(out)

    def clear(self, task_signature: str) -> bool:
        """Delete the episodes file for ``task_signature``. Returns True if it existed."""
        path = self.store_path(task_signature)
        if not path.is_file():
            return False
        try:
            path.unlink()
            return True
        except OSError as exc:
            logger.warning("EpisodicStore: failed to delete %s: %s", path, exc)
            return False

    # ------------------------------------------------------------------
    # Locking
    # ------------------------------------------------------------------

    def _acquire_flock_locked(self, fd: Any, *, exclusive: bool = True) -> None:
        """Best-effort cross-process flock with a deadline.

        We retry on EWOULDBLOCK with a small sleep so we don't spin
        burn CPU on contention. The deadline guards against forgotten
        locks (e.g. a benchmark worker that crashed without releasing
        the file). On non-POSIX hosts this is a no-op.
        """
        if not _FCNTL_AVAILABLE or fcntl is None:
            return
        flag = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        deadline = time.time() + _LOCK_TIMEOUT_SECONDS
        while True:
            try:
                fcntl.flock(fd.fileno(), flag | fcntl.LOCK_NB)
                return
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                    logger.warning(
                        "EpisodicStore: flock unexpected error %s; proceeding without lock",
                        exc,
                    )
                    return
                if time.time() > deadline:
                    logger.warning(
                        "EpisodicStore: flock timeout after %.1fs; proceeding without lock",
                        _LOCK_TIMEOUT_SECONDS,
                    )
                    return
                time.sleep(0.05)

    def _release_flock_locked(self, fd: Any) -> None:
        if not _FCNTL_AVAILABLE or fcntl is None:
            return
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        except OSError:  # pragma: no cover — defensive
            pass


def task_signature_for(repo_signature: str, task_id: str) -> str:
    """Compose a task signature from a repo signature and a task id.

    The convention is ``"<repo_sig>::<task_id>"`` so two tasks on the
    same repo never collide and the per-task store can be inspected
    by humans (the repo signature is a stable hash; the task id is
    the benchmark/instance id).
    """
    repo = (repo_signature or "").strip() or "unknown"
    task = (task_id or "").strip() or "unknown"
    return f"{repo}::{task}"


__all__ = [
    "Episode",
    "EpisodicStore",
    "task_signature_for",
]
