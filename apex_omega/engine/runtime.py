"""The vendor-neutral orchestration-as-code engine (Fusion Ledger A1; plan §02/§08).

Exposes the five paradigm primitives — ``agent`` / ``parallel`` / ``pipeline`` /
``phase`` / ``log`` (+ ``budget``) — over a normalized worker contract.  State
lives in script variables + the durable journal, never a conversation window
(Property 2: context isolation).

Design choices that keep the invariants:
  * ``agent()`` is SYNCHRONOUS and journaled at the call boundary (resume wired
    into agent() itself, §15.7 #5).  Budget is charged ONLY on fresh (non-cached)
    work, so resume is free.
  * ``parallel()`` / ``pipeline()`` use a fresh local thread pool per call, so
    nesting one inside the other can never deadlock a shared bounded pool.
  * narration (``phase``/``log``) is FAIL-OPEN: it may never block or crash a run,
    and uses a monotonic LOGICAL clock, never wall-clock (determinism, §02).
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ..errors import FailLoud
from ..journal.wal import Journal
from ..journal.resume import resume_or_run_exec
from ..types import ExecResult, ScopedTask
from .budget import Budget
from .pipeline import Stage, run_pipeline


# A worker turns a ScopedTask into an ExecResult by running a vendor CLI in a
# worktree.  It must NEVER raise (typed-failure contract); the executor layer
# provides it.  Isolation (worktree) is the worker's responsibility.
Runner = Callable[[ScopedTask], ExecResult]


class Engine:
    def __init__(
        self,
        run_dir: str | Path,
        *,
        run_id: str = "run",
        budget: Optional[Budget] = None,
        max_workers: Optional[int] = None,
        journal: Optional[Journal] = None,
        materialize_diffs: bool = True,
        max_concurrent: Optional[int] = None,
        max_total_agents: int = 1000,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.journal = journal or Journal(self.run_dir, run_id=run_id, materialize_diffs=materialize_diffs)
        self.budget = budget or Budget()
        cpu = os.cpu_count() or 4
        self.max_workers = max(1, max_workers if max_workers is not None else min(16, cpu - 2))
        # Global concurrency gate on the *expensive* worker call (the CLI subprocess),
        # so that even a generated orchestrator that fans out 1000s of agents — across
        # nested parallel/pipeline — runs at most `max_concurrent` at once. Total agents
        # over the whole run can reach the thousands (queued); `max_total_agents` is the
        # runaway backstop. Raise both when running on bigger infra.
        self.max_concurrent = max(1, max_concurrent if max_concurrent is not None else self.max_workers)
        self.max_total_agents = max_total_agents
        self._sem = threading.BoundedSemaphore(self.max_concurrent)
        # Backbone 0.3: the agent backstop is PER-RUN, not per-process. Rehydrate the
        # fresh-agent tally from the journal so a relaunch/resume continues toward the
        # SAME 1000 ceiling instead of resetting it (which would allow R x 1000).
        self._total_agents = self.journal.fresh_agent_count()
        self._total_lock = threading.Lock()
        self._phase = "init"
        self._logical = 0
        self._narration_lock = threading.Lock()
        self._narration_path = self.run_dir / "narration.jsonl"

    def agents_used(self) -> int:
        """Fresh (non-cached) worker calls dispatched so far this run."""
        with self._total_lock:
            return self._total_agents

    # -- narration (fail-open, logical-clock-only) ------------------------
    def _next_logical(self) -> int:
        with self._narration_lock:
            self._logical += 1
            return self._logical

    def _narrate(self, record: dict) -> None:
        try:
            record = {"seq": self._next_logical(), "phase": self._phase, **record}
            with self._narration_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
        except Exception:
            # narration must never crash a run (plan §02 phase/log discipline)
            pass

    def phase(self, title: str) -> None:
        """Start a new phase; boundary coincides with a journal checkpoint."""
        self._phase = str(title)
        self._narrate({"event": "phase", "title": str(title)})

    def log(self, msg: str) -> None:
        self._narrate({"event": "log", "msg": str(msg)})

    # -- agent (journaled atom) ------------------------------------------
    def agent(
        self,
        task: ScopedTask,
        runner: Runner,
        *,
        node_id: str = "",
        cli_version: str = "",
        extra_scoped_inputs: Optional[dict] = None,
        materialize: Optional[Callable[[str], None]] = None,
        agent_type: str = "",
        phase: str = "",
        label: str = "",
    ) -> ExecResult:
        """Run one journaled worker call.  On a cache HIT the recorded artifact
        (diff) is replayed/materialized and NO worker is spawned and NO budget is
        charged.  ``runner`` must return a typed ExecResult and never raise."""
        scoped = dict(task.scoped_inputs)
        if extra_scoped_inputs:
            scoped.update(extra_scoped_inputs)
        components = {
            "kind": "agent",
            "prompt": task.prompt,
            "schema": task.schema,
            "model": task.model,
            "vendor": task.vendor,
            "cli_version": cli_version,
            "effort": task.effort,
            "agentType": agent_type,
            "sandbox": task.sandbox,
            "allowed_tools": sorted(task.allowed_tools),
            "scoped_inputs": scoped,
        }

        def _safe_runner() -> ExecResult:
            # runaway backstop: a generated orchestrator can request thousands of
            # agents, but never more than the configured ceiling.
            with self._total_lock:
                self._total_agents += 1
                n = self._total_agents
            if self.max_total_agents and n > self.max_total_agents:
                return ExecResult(ok=False, finalization_status="infra_nonresult",
                                  error=f"max_total_agents ({self.max_total_agents}) exceeded")
            def _call() -> ExecResult:
                try:
                    res = runner(task)
                    if not isinstance(res, ExecResult):
                        return ExecResult(ok=False, finalization_status="infra_nonresult",
                                          error=f"runner returned {type(res).__name__}, expected ExecResult")
                    return res
                except Exception as exc:  # enforce never-raises at the engine boundary
                    return ExecResult(ok=False, finalization_status="infra_nonresult",
                                      error=f"{type(exc).__name__}: {exc}")

            self._sem.acquire()  # global concurrency gate on the expensive worker call
            owns_slot = True
            try:
                # Backbone 0.2c: per-agent watchdog. The vendor's cli_strict_hard_timeout is
                # the REAL Popen kill (at task.timeout_seconds); this only ABANDONS the wait
                # for SELECTION (infra_nonresult -> excluded + re-runs on resume) so a hung/
                # non-vendor agent can never block the cell forever. When no wall is set the
                # agent runs to COMPLETION (default-unbounded model).
                wall = task.heartbeat_timeout_seconds
                if wall is None and task.timeout_seconds:
                    wall = int(task.timeout_seconds) + 60   # engine abandons just after the vendor cap
                if not wall or wall <= 0:
                    return _call()
                box: dict = {}
                th = threading.Thread(target=lambda: box.__setitem__("r", _call()), daemon=True)
                th.start()
                th.join(wall)
                if th.is_alive():
                    # The wait is abandoned for SELECTION, but the worker (its vendor
                    # subprocess) is still running and still occupies a real concurrency slot.
                    # Reclaim the semaphore ONLY after the worker truly finishes, via a reaper
                    # — releasing it here (the old finally) would let actual concurrency exceed
                    # max_concurrent in a watchdog-heavy fan-out. (finalization_status is the
                    # canonical infra_nonresult so the repair-gate + journal treat it correctly.)
                    owns_slot = False  # ownership transferred to the reaper below
                    sem = self._sem

                    def _reap(_th=th, _sem=sem):
                        _th.join()
                        _sem.release()

                    threading.Thread(target=_reap, daemon=True, name="apexω-reap").start()
                    return ExecResult(ok=False, finalization_status="infra_nonresult",
                                      error=f"heartbeat_timeout: per-agent watchdog {wall}s exceeded "
                                            "(vendor cap is the real kill)")
                return box.get("r") or ExecResult(ok=False, finalization_status="infra_nonresult",
                                                  error="watchdog: runner produced no result")
            finally:
                if owns_slot:
                    self._sem.release()

        result, hit = resume_or_run_exec(
            self.journal, components, _safe_runner, node_id=node_id, materialize=materialize
        )
        if not hit:
            self.budget.add_usage(result.usage)
        # Per-agent phase/label/agentType are narration-only (NOT part of the journal key
        # `components`), so they group/label agents in the UI (dynamic-workflows agent() opts)
        # without ever affecting resume. A per-call `phase` overrides the global phase for
        # THIS record only (it spreads after self._phase in _narrate).
        rec = {
            "event": "agent", "node_id": node_id or "", "vendor": task.vendor,
            "model": task.model, "cache_hit": hit, "ok": result.ok,
            "finalization_status": result.finalization_status,
            "tokens": result.usage.total, "diff_bytes": len(result.fs_diff or ""),
        }
        if agent_type:
            rec["agentType"] = agent_type
        if label:
            rec["label"] = label
        if phase:
            rec["phase"] = phase
        self._narrate(rec)
        return result

    # -- parallel (barrier fan-out, null-on-fail) ------------------------
    def parallel(self, thunks: Sequence[Callable[[], Any]]) -> list[Any]:
        """Run thunks concurrently; AWAIT ALL (barrier).  A thunk that raises (or
        whose agent errors) resolves to ``None`` — the call itself never raises.
        Caller MUST filter ``None`` before use (fail-loud accounting)."""
        thunks = list(thunks)
        if not thunks:
            return []
        if len(thunks) > 4096:  # Backbone 0.4: mirror the workflow tool's per-call fan-out cap
            raise FailLoud(f"parallel() given {len(thunks)} thunks; cap is 4096 per call")
        n_workers = max(1, min(self.max_workers, len(thunks)))
        out: list[Any] = [None] * len(thunks)
        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="apexω-par") as pool:
            fut_to_idx = {pool.submit(t): i for i, t in enumerate(thunks)}
            for fut in as_completed(fut_to_idx):
                i = fut_to_idx[fut]
                try:
                    out[i] = fut.result()
                except Exception as exc:
                    out[i] = None
                    self.log(f"parallel: thunk[{i}] raised {type(exc).__name__}: {exc}")
        n_null = sum(1 for x in out if x is None)
        if n_null:
            self._narrate({"event": "parallel_nulls", "null_count": n_null, "total": len(thunks)})
        return out

    # -- pipeline (net-new per-item streaming) ---------------------------
    def pipeline(self, items: Sequence[Any], *stages: Any, **kwargs) -> list[Any]:
        return run_pipeline(self, items, *stages, **kwargs)

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        self._narrate({"event": "close", "budget": self.budget.to_dict(), "journal": self.journal.stats()})

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
