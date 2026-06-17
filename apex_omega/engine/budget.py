"""First-class cost budget primitive (plan §02.6 / §08.3 / Principle 8).

Semantics (non-negotiable invariants):
  * DEFAULTS UNBOUNDED — an operator must *opt in* to bound cost (preserves v1's
    "never optimize for cost" default for headline runs).
  * Budget exhaustion governs only WHETHER TO START new work; it must NEVER abort
    an in-flight succeeding rollout, and never suppress a candidate that already
    has execution evidence.  ``can_start()`` is the only gate.
  * ``spent()`` / ``remaining()`` mirror the Workflow-tool contract so the same
    loop-until-budget idioms apply.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Optional

from ..types import TokenUsage


class Budget:
    def __init__(self, total: Optional[int] = None):
        # total is a token ceiling; None == unbounded (the default).
        self.total: Optional[int] = total
        self._spent: int = 0
        self._lock = threading.Lock()

    def spent(self) -> int:
        with self._lock:
            return self._spent

    def remaining(self) -> float:
        with self._lock:
            if self.total is None:
                return math.inf
            return max(0, self.total - self._spent)

    def add(self, tokens: int) -> None:
        if tokens <= 0:
            return
        with self._lock:
            self._spent += int(tokens)

    def add_usage(self, usage: TokenUsage) -> None:
        self.add(usage.total)

    def can_start(self, *, reserve: int = 1) -> bool:
        """May a NEW unit of work be dispatched?  Always True when unbounded.
        When bounded, requires at least ``reserve`` tokens of headroom.  Note:
        this never affects work already running (Principle 8)."""
        if self.total is None:
            return True
        return self.remaining() >= max(1, reserve)

    @property
    def bounded(self) -> bool:
        return self.total is not None

    def to_dict(self) -> dict:
        return {"total": self.total, "spent": self.spent(), "remaining": (None if self.total is None else self.remaining())}
