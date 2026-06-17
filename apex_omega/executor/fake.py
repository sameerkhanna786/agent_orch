"""Deterministic in-process Executor for unit tests (no v1, no network, no cost).

Lets the engine/journal/pipeline invariants be tested without paid vendor calls.
A global call counter proves resume skips re-runs (cache hits never invoke the
responder).
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

from ..types import CapabilityProfile, ExecResult, ScopedTask, TokenUsage


class FakeSession:
    def __init__(self, cwd: str, vendor: str, model: str, cli_version: str,
                 responder: Callable[[ScopedTask, "FakeSession"], ExecResult], counter):
        self.cwd = cwd
        self.vendor = vendor
        self.model = model
        self.cli_version = cli_version
        self._responder = responder
        self._counter = counter

    def observe(self) -> str:
        return ""

    def observe_diff(self) -> str:
        return ""

    def run(self, task: ScopedTask) -> ExecResult:
        with self._counter["lock"]:
            self._counter["calls"] += 1
        try:
            res = self._responder(task, self)
        except Exception as exc:
            return ExecResult(ok=False, finalization_status="infra_nonresult",
                              error=f"{type(exc).__name__}: {exc}", vendor=self.vendor, model=self.model)
        res.vendor = res.vendor or self.vendor
        res.model = res.model or self.model
        res.cli_version = res.cli_version or self.cli_version
        return res


def _default_responder(task: ScopedTask, session: FakeSession) -> ExecResult:
    msg = f"fake[{session.vendor}:{session.model}] -> {task.prompt[:60]}"
    return ExecResult(
        final_message=msg,
        structured_output=({"echo": task.prompt} if task.schema else None),
        usage=TokenUsage(input=len(task.prompt), output=max(1, len(task.prompt) // 2)),
        fs_diff=f"--- fake.diff for {task.prompt[:24]} ---\n",
        ok=True,
        finalization_status="completed",
    )


class FakeExecutor:
    def __init__(self, responder: Optional[Callable[[ScopedTask, FakeSession], ExecResult]] = None):
        self.responder = responder or _default_responder
        self._counter = {"calls": 0, "lock": threading.Lock()}

    @property
    def calls(self) -> int:
        return self._counter["calls"]

    def reset_calls(self) -> None:
        with self._counter["lock"]:
            self._counter["calls"] = 0

    def negotiate(self, vendor: str, model: str, version: str = "") -> CapabilityProfile:
        return CapabilityProfile(vendor=vendor, model=model, cli_version=version or "fake@0",
                                 native_schema=True, sandbox_levels=("read-only", "workspace-write"))

    def spawn(self, worktree_cwd: str, vendor: str, model: str, version: str = "", **_) -> FakeSession:
        return FakeSession(worktree_cwd, vendor, model, version or "fake@0", self.responder, self._counter)
