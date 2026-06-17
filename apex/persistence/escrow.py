"""Confirmed-Candidate Escrow + Write-Ahead Durability Fabric (CCEDF).

This module makes a *confirmed-correct* candidate (one that already produced an
authoritative full-scope / F2P pass) **durable at the instant of confirmation**
and **exactly-once recoverable** across any cancel, OOM, preemption, container
reap, or process restart.

Motivation
----------
Published SWE orchestrators (Agentless, SWE-agent, Trae, CodeMonkeys, Moatless /
SWE-Search, OpenHands) keep the winning candidate in process memory until the
task returns. A preemption *near the finish line* — a wall-clock cancel during
finalization, an OOM kill of the worker, a scheduler stop-on-result that fires
just after a sibling confirmed a pass — therefore discards an already-correct
solution. The dominant Commit0 gold-suite loss had exactly this shape: a rollout
reached ``pass_rate == 1.0`` but was dropped to ``scheduler_cancelled`` with no
recorded result.

CCEDF lifts two well-understood distributed-systems primitives into the SWE
rollout layer:

* **Write-ahead log + commit-then-publish** (ARIES; Kafka exactly-once
  semantics, Confluent 2017): the confirmed candidate is appended to an
  append-only, fsync-durable WAL *before* any sibling-preempt / teardown / reap,
  so the win survives a crash between confirmation and publication.
* **Idempotent exactly-once apply**: every record carries a stable idempotency
  key ``(task_id, candidate_id)``; replay deduplicates by key so re-running a
  recovered rollout never double-counts and a duplicated append is harmless.

The store is intentionally **benchmark-agnostic**: it knows nothing about repos,
languages, pytest, or Docker. Callers attach an opaque payload (patch text,
worktree path, quick-verification summary, score) and CCEDF guarantees only that
a confirmed payload, once escrowed, is never lost and is recoverable exactly
once per key.

Design
------
* **Append-only JSONL WAL** at ``<run_dir>/escrow/confirmed_wal.jsonl`` — records
  accumulate, never rewritten in place, so ordering is reconstructable and a
  crash leaves at worst a partial trailing line (tolerated on read).
* **Atomic appends under ``fcntl.flock``** — multiple parallel benchmark workers
  may escrow concurrently; an exclusive lock is held for the duration of one
  append. Each record is a single ``write`` of one line terminated by ``\n`` so a
  torn write is detectable (unparseable trailing line) and skipped on replay.
* **Replay dedups by idempotency key, latest-wins** — and, per task, prefers the
  highest-scoring confirmed record.
* **No wall-clock dependence for correctness** — ordering uses a per-append
  sequence number; an optional caller-supplied timestamp is recorded for humans
  only.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:  # pragma: no cover - platform guard; Apex targets POSIX runners
    import fcntl

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]
    _HAVE_FCNTL = False

logger = logging.getLogger(__name__)

ESCROW_WAL_SCHEMA_VERSION = 1
_ESCROW_DIRNAME = "escrow"
_ESCROW_WAL_FILENAME = "confirmed_wal.jsonl"


def _idempotency_key(task_id: str, candidate_id: str) -> str:
    return f"{str(task_id).strip()}::{str(candidate_id).strip()}"


@dataclass
class EscrowRecord:
    """One escrowed confirmed-candidate WAL record."""

    task_id: str
    candidate_id: str
    kind: str = "confirmed_full_scope_pass"
    score: float = 1.0
    payload: dict[str, Any] = field(default_factory=dict)
    seq: int = 0
    timestamp: Optional[float] = None

    @property
    def idempotency_key(self) -> str:
        return _idempotency_key(self.task_id, self.candidate_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "v": ESCROW_WAL_SCHEMA_VERSION,
            "seq": int(self.seq),
            "task_id": self.task_id,
            "candidate_id": self.candidate_id,
            "idempotency_key": self.idempotency_key,
            "kind": self.kind,
            "score": float(self.score),
            "timestamp": self.timestamp,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Optional["EscrowRecord"]:
        if not isinstance(data, dict):
            return None
        if int(data.get("v") or 0) != ESCROW_WAL_SCHEMA_VERSION:
            return None
        task_id = str(data.get("task_id") or "").strip()
        candidate_id = str(data.get("candidate_id") or "").strip()
        if not task_id or not candidate_id:
            return None
        try:
            score = float(data.get("score") if data.get("score") is not None else 1.0)
        except (TypeError, ValueError):
            score = 1.0
        timestamp = data.get("timestamp")
        try:
            timestamp = float(timestamp) if timestamp is not None else None
        except (TypeError, ValueError):
            timestamp = None
        payload = data.get("payload")
        return cls(
            task_id=task_id,
            candidate_id=candidate_id,
            kind=str(data.get("kind") or "confirmed_full_scope_pass"),
            score=score,
            payload=payload if isinstance(payload, dict) else {},
            seq=int(data.get("seq") or 0),
            timestamp=timestamp,
        )


class EscrowStore:
    """Run-scoped, fsync-durable, idempotent escrow for confirmed candidates."""

    def __init__(self, run_dir: str | Path) -> None:
        self._wal_path = Path(run_dir) / _ESCROW_DIRNAME / _ESCROW_WAL_FILENAME

    @property
    def wal_path(self) -> Path:
        return self._wal_path

    # ------------------------------------------------------------------ #
    # Write path
    # ------------------------------------------------------------------ #
    def escrow_confirmed_candidate(
        self,
        *,
        task_id: str,
        candidate_id: str,
        payload: dict[str, Any],
        kind: str = "confirmed_full_scope_pass",
        score: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> Optional[EscrowRecord]:
        """Append a confirmed-candidate record to the WAL, durably (fsync).

        Returns the written :class:`EscrowRecord` (with its assigned sequence
        number) or ``None`` if the write could not be performed. The append is
        idempotent at the key level: re-escrowing the same ``(task_id,
        candidate_id)`` simply appends another record and replay keeps the latest,
        so a retried confirmation never corrupts state.
        """
        task_id = str(task_id).strip()
        candidate_id = str(candidate_id).strip()
        if not task_id or not candidate_id:
            return None
        record = EscrowRecord(
            task_id=task_id,
            candidate_id=candidate_id,
            kind=kind,
            score=float(score),
            payload=payload if isinstance(payload, dict) else {},
            timestamp=timestamp,
        )
        try:
            self._wal_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.debug("Escrow WAL dir create failed for %s", self._wal_path, exc_info=True)
            return None
        line: Optional[str] = None
        try:
            # Open in append mode and hold an exclusive lock for the whole
            # read-seq + write so concurrent workers assign distinct sequence
            # numbers and never interleave a partial line.
            with open(self._wal_path, "a+", encoding="utf-8") as handle:
                if _HAVE_FCNTL:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    record.seq = self._next_seq_locked()
                    line = json.dumps(record.to_dict(), default=repr)
                    handle.write(line + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    if _HAVE_FCNTL:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            logger.debug("Escrow WAL append failed for %s", self.wal_path, exc_info=True)
            return None
        logger.info(
            "Escrowed confirmed candidate task=%s candidate=%s kind=%s seq=%s",
            task_id,
            candidate_id,
            kind,
            record.seq,
        )
        return record

    def _next_seq_locked(self) -> int:
        """Return the next monotonic sequence number. Caller holds the lock."""
        records = self._read_records()
        if not records:
            return 1
        return max(int(rec.seq) for rec in records) + 1

    # ------------------------------------------------------------------ #
    # Read / replay path
    # ------------------------------------------------------------------ #
    def _read_records(self) -> list[EscrowRecord]:
        if not self._wal_path.exists():
            return []
        records: list[EscrowRecord] = []
        try:
            text = self._wal_path.read_text(encoding="utf-8")
        except OSError:
            return []
        for raw in text.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # Tolerate a torn trailing line from a crashed append.
                continue
            record = EscrowRecord.from_dict(data)
            if record is not None:
                records.append(record)
        return records

    def replay(self) -> dict[str, EscrowRecord]:
        """Return the best confirmed escrow record per task, exactly once.

        Deduplicates by idempotency key (latest sequence wins for a repeated key)
        and then, per task, keeps the highest-scoring confirmed record (sequence
        breaks ties). This is the recovery entry point at run start: callers
        prefer an escrowed confirmed candidate over re-running residual work.
        """
        by_key: dict[str, EscrowRecord] = {}
        for record in self._read_records():
            existing = by_key.get(record.idempotency_key)
            if existing is None or record.seq >= existing.seq:
                by_key[record.idempotency_key] = record
        best_by_task: dict[str, EscrowRecord] = {}
        for record in by_key.values():
            current = best_by_task.get(record.task_id)
            if current is None or (record.score, record.seq) > (current.score, current.seq):
                best_by_task[record.task_id] = record
        return best_by_task

    def confirmed_record(self, task_id: str) -> Optional[EscrowRecord]:
        """Return the escrowed confirmed record for ``task_id`` if any."""
        return self.replay().get(str(task_id).strip())

    def has_confirmed(self, task_id: str) -> bool:
        return self.confirmed_record(task_id) is not None
