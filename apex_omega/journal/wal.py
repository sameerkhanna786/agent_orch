"""Durable, input-hash-keyed write-ahead journal (Fusion Ledger A9; plan §15).

This is one of only two genuinely net-new builds in Phase 0 (the other is
``pipeline()``).  It promotes v1's unused ``ReplayRecorder`` + the narrow escrow
WAL into a per-``agent()``-call write-ahead log that survives a full process
restart (``kill -9``), so resume re-runs ONLY edited/new calls.

Discipline (CCEDF, plan §15.3):
  * append-only JSONL at ``<run_dir>/journal/calls_wal.jsonl``
  * each append is flock-guarded (cross-process) and fsync-durable BEFORE the
    result is returned to the caller (true write-ahead)
  * ``seq`` is a monotonic counter (the order key) — NEVER wall-clock, so replay
    order is a pure function of recorded sequence, not timing
  * idempotent + latest-wins by ``seq`` on lookup
  * produced git diffs are stored as content-addressed blobs under
    ``journal/diffs/<sha>.diff`` and referenced by ``fs_diff_ref``

Cache-validity rule (§15.3.2): a lookup HIT replays the recorded artifact; it
does not re-derive.  An entry left ``in_flight`` (crashed mid-call) is NOT a
valid hit and re-runs.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

try:  # POSIX advisory locking; degrades to a threading lock on non-POSIX hosts.
    import fcntl  # type: ignore
    _HAVE_FCNTL = True
except Exception:  # pragma: no cover - Windows
    fcntl = None  # type: ignore
    _HAVE_FCNTL = False

from .key import canonical_key, sha256_hex


# Result status — whether the recorded call is a usable cache hit.
RESULT_OK = "ok"
RESULT_INFRA_NONRESULT = "infra_nonresult"   # transport failure; not a valid hit, re-run
RESULT_ABSTAIN = "abstain"

# Entry lifecycle status (Sec 02 variant, folded in for crash detection).
STATUS_IN_FLIGHT = "in_flight"
STATUS_COMMITTED = "committed"
STATUS_FAILED = "failed"


@dataclass(frozen=True)
class JournalEntry:
    """One WAL record (plan §15.86-105 authoritative build spec)."""

    seq: int
    input_hash: str
    kind: str                       # agent | hedge_decision | cancel | node_expand | node_prune | stage
    prompt_canonical: str
    model_id: str
    vendor: str
    cli_version: str
    scoped_inputs_hash: str
    result_status: str              # ok | infra_nonresult | abstain
    structured_result: dict         # JSON-serializable payload (e.g. ExecResult.to_dict())
    fs_diff_ref: str                # content-addressed blob ref ('' if none)
    usage: dict
    idempotency_key: str            # f"{run_id}::{node_id}::{attempt}"
    status: str = STATUS_COMMITTED  # in_flight | committed | failed
    ts_logical: int = 0             # monotonic logical clock == seq; NEVER wall-clock

    def to_json(self) -> str:
        return json.dumps(
            {
                "seq": self.seq,
                "input_hash": self.input_hash,
                "kind": self.kind,
                "prompt_canonical": self.prompt_canonical,
                "model_id": self.model_id,
                "vendor": self.vendor,
                "cli_version": self.cli_version,
                "scoped_inputs_hash": self.scoped_inputs_hash,
                "result_status": self.result_status,
                "structured_result": self.structured_result,
                "fs_diff_ref": self.fs_diff_ref,
                "usage": self.usage,
                "idempotency_key": self.idempotency_key,
                "status": self.status,
                "ts_logical": self.ts_logical,
            },
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @classmethod
    def from_json(cls, line: str) -> "JournalEntry":
        d = json.loads(line)
        return cls(
            seq=int(d["seq"]),
            input_hash=d["input_hash"],
            kind=d.get("kind", "agent"),
            prompt_canonical=d.get("prompt_canonical", ""),
            model_id=d.get("model_id", ""),
            vendor=d.get("vendor", ""),
            cli_version=d.get("cli_version", ""),
            scoped_inputs_hash=d.get("scoped_inputs_hash", ""),
            result_status=d.get("result_status", RESULT_OK),
            structured_result=d.get("structured_result", {}),
            fs_diff_ref=d.get("fs_diff_ref", ""),
            usage=d.get("usage", {}),
            idempotency_key=d.get("idempotency_key", ""),
            status=d.get("status", STATUS_COMMITTED),
            ts_logical=int(d.get("ts_logical", d.get("seq", 0))),
        )


class Journal:
    """Append-only, restart-survivable call journal."""

    def __init__(self, run_dir: str | Path, *, run_id: str = "run", materialize_diffs: bool = True):
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.materialize_diffs = materialize_diffs
        self.journal_dir = self.run_dir / "journal"
        self.diffs_dir = self.journal_dir / "diffs"
        self.wal_path = self.journal_dir / "calls_wal.jsonl"
        self.diffs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # latest-wins committed index, by input_hash
        self._index: dict[str, JournalEntry] = {}
        # input_hashes currently in_flight without a later terminal record
        self._in_flight: dict[str, JournalEntry] = {}
        # Backbone 0.3 (review-fix): count EVERY committed kind=="agent" record, OK or
        # infra_nonresult — the live per-dispatch increment in Engine._safe_runner fires for
        # failed dispatches too, so the resume rehydration must include them or the per-RUN
        # agent ceiling drifts past its backstop across relaunches.
        self._committed_agents = 0
        self._next_seq = 0
        self._recover()

    # -- recovery ---------------------------------------------------------
    def _recover(self) -> None:
        """Rebuild the in-memory index from the WAL tail on startup.  This is the
        once-per-process O(n) scan that makes subsequent appends O(1)."""
        if not self.wal_path.exists():
            return
        with self.wal_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = JournalEntry.from_json(line)
                except Exception:
                    continue  # tolerate a torn final line from a hard crash
                self._next_seq = max(self._next_seq, entry.seq + 1)
                if entry.status == STATUS_IN_FLIGHT:
                    self._in_flight[entry.input_hash] = entry
                else:
                    # terminal record clears any earlier in_flight marker
                    self._in_flight.pop(entry.input_hash, None)
                    if entry.status == STATUS_COMMITTED:
                        if entry.kind == "agent":
                            self._committed_agents += 1   # OK or infra_nonresult both count
                        if entry.result_status == RESULT_OK:
                            prev = self._index.get(entry.input_hash)
                            if prev is None or entry.seq >= prev.seq:
                                self._index[entry.input_hash] = entry

    # -- low-level durable append ----------------------------------------
    def _append(self, entry: JournalEntry) -> None:
        line = entry.to_json() + "\n"
        # Hold the in-process lock for the whole critical section; flock guards
        # cross-process appenders.  fsync before returning == write-ahead.
        with self.wal_path.open("a", encoding="utf-8") as fh:
            if _HAVE_FCNTL:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            finally:
                if _HAVE_FCNTL:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def _alloc_seq(self) -> int:
        seq = self._next_seq
        self._next_seq += 1
        return seq

    # -- diff blob store --------------------------------------------------
    def store_diff(self, diff_text: str) -> str:
        if not diff_text:
            return ""
        ref = sha256_hex(diff_text)
        blob = self.diffs_dir / f"{ref}.diff"
        if not blob.exists():
            tmp = blob.with_suffix(".diff.tmp")
            tmp.write_text(diff_text, encoding="utf-8")
            os.replace(tmp, blob)  # atomic
        return ref

    def load_diff(self, ref: str) -> str:
        if not ref:
            return ""
        blob = self.diffs_dir / f"{ref}.diff"
        return blob.read_text(encoding="utf-8") if blob.exists() else ""

    # -- public lifecycle -------------------------------------------------
    def lookup(self, input_hash: str) -> Optional[JournalEntry]:
        """Return the latest committed OK entry for this input-hash, or None.
        An in_flight-only entry (crashed mid-call) is intentionally NOT a hit."""
        with self._lock:
            return self._index.get(input_hash)

    def begin(self, *, input_hash: str, kind: str, prompt_canonical: str, model_id: str,
              vendor: str, cli_version: str, scoped_inputs_hash: str,
              node_id: str = "", attempt: int = 0) -> int:
        with self._lock:
            seq = self._alloc_seq()
            entry = JournalEntry(
                seq=seq, input_hash=input_hash, kind=kind,
                prompt_canonical=prompt_canonical, model_id=model_id, vendor=vendor,
                cli_version=cli_version, scoped_inputs_hash=scoped_inputs_hash,
                result_status=RESULT_OK, structured_result={}, fs_diff_ref="", usage={},
                idempotency_key=f"{self.run_id}::{node_id or input_hash[:12]}::{attempt}",
                status=STATUS_IN_FLIGHT, ts_logical=seq,
            )
            self._append(entry)
            self._in_flight[input_hash] = entry
            return seq

    def commit(self, *, input_hash: str, kind: str, prompt_canonical: str, model_id: str,
               vendor: str, cli_version: str, scoped_inputs_hash: str, result_status: str,
               structured_result: dict, fs_diff_text: str = "", usage: Optional[dict] = None,
               node_id: str = "", attempt: int = 0) -> JournalEntry:
        with self._lock:
            seq = self._alloc_seq()
            fs_diff_ref = self.store_diff(fs_diff_text) if (self.materialize_diffs and fs_diff_text) else ""
            entry = JournalEntry(
                seq=seq, input_hash=input_hash, kind=kind,
                prompt_canonical=prompt_canonical, model_id=model_id, vendor=vendor,
                cli_version=cli_version, scoped_inputs_hash=scoped_inputs_hash,
                result_status=result_status, structured_result=structured_result,
                fs_diff_ref=fs_diff_ref, usage=usage or {},
                idempotency_key=f"{self.run_id}::{node_id or input_hash[:12]}::{attempt}",
                status=STATUS_COMMITTED, ts_logical=seq,
            )
            self._append(entry)
            self._in_flight.pop(input_hash, None)
            if kind == "agent":
                self._committed_agents += 1   # matches _safe_runner's per-dispatch increment
            if result_status == RESULT_OK:
                self._index[input_hash] = entry
            return entry

    # -- the central chokepoint (resume_or_run) --------------------------
    def get_or_run(
        self,
        components: dict,
        fn: Callable[[], Any],
        *,
        serialize: Callable[[Any], tuple[dict, str, str, dict]],
        deserialize: Callable[[dict, str], Any],
        kind: str = "agent",
        node_id: str = "",
        attempt: int = 0,
        materialize: Optional[Callable[[str], None]] = None,
    ) -> tuple[Any, bool]:
        """Resume-or-run a journaled unit of work.

        ``serialize(result) -> (structured_result_dict, fs_diff_text, result_status, usage)``
        ``deserialize(structured_result_dict, fs_diff_text) -> result``
        ``materialize(fs_diff_text)`` (optional) restores a cached diff into the
        worktree on a HIT (config ``journal.materialize_diffs``).

        Returns ``(result, was_cache_hit)``.
        """
        input_hash = canonical_key(components)
        hit = self.lookup(input_hash)
        if hit is not None:
            diff_text = self.load_diff(hit.fs_diff_ref) if hit.fs_diff_ref else ""
            if materialize is not None and diff_text:
                materialize(diff_text)
            return deserialize(hit.structured_result, diff_text), True

        prompt_canonical = str(components.get("prompt_canonical") or components.get("prompt", ""))
        from .key import canonicalize_prompt, scoped_inputs_hash as _sih
        prompt_canonical = canonicalize_prompt(prompt_canonical)
        si_hash = components.get("scoped_inputs_hash") or _sih(components.get("scoped_inputs"))

        self.begin(
            input_hash=input_hash, kind=kind, prompt_canonical=prompt_canonical,
            model_id=str(components.get("model", "")), vendor=str(components.get("vendor", "")),
            cli_version=str(components.get("cli_version", "")), scoped_inputs_hash=si_hash,
            node_id=node_id, attempt=attempt,
        )
        result = fn()
        structured_result, fs_diff_text, result_status, usage = serialize(result)
        self.commit(
            input_hash=input_hash, kind=kind, prompt_canonical=prompt_canonical,
            model_id=str(components.get("model", "")), vendor=str(components.get("vendor", "")),
            cli_version=str(components.get("cli_version", "")), scoped_inputs_hash=si_hash,
            result_status=result_status, structured_result=structured_result,
            fs_diff_text=fs_diff_text, usage=usage, node_id=node_id, attempt=attempt,
        )
        return result, False

    def fresh_agent_count(self) -> int:
        """Count of committed ``kind=="agent"`` records — OK AND infra_nonresult — i.e. the
        cross-process count of FRESH (non-replayed) agent dispatches. A HIT replays an
        existing committed entry and never re-begins, so this is the true per-RUN fresh tally
        to rehydrate the engine's agent backstop on resume (Backbone 0.3). It counts failed
        dispatches too, because ``Engine._safe_runner`` increments the live tally before the
        work runs — so excluding them would let the per-RUN ceiling drift across relaunches."""
        with self._lock:
            return self._committed_agents

    # -- diagnostics ------------------------------------------------------
    def stats(self) -> dict:
        with self._lock:
            return {
                "committed_ok": len(self._index),
                "in_flight": len(self._in_flight),
                "next_seq": self._next_seq,
                "wal_path": str(self.wal_path),
            }
