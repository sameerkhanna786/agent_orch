"""Coverage-guided target selection for one-test-at-a-time generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CoverageTarget:
    symbol_name: str
    reason: str
    priority: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoverageMask:
    covered_symbols: set[str] = field(default_factory=set)
    uncovered_branches: dict[str, int] = field(default_factory=dict)
    covered_lines: set[int] = field(default_factory=set)


def pick_uncovered_target(
    api_probe: Any,
    coverage_mask: CoverageMask | None = None,
) -> CoverageTarget | None:
    mask = coverage_mask or CoverageMask()
    symbols = [
        symbol
        for symbol in list(getattr(api_probe, "symbols", []) or [])
        if str(getattr(symbol, "name", "") or "")
    ]
    if not symbols:
        return None
    branch_candidates = [
        symbol
        for symbol in symbols
        if int(mask.uncovered_branches.get(str(getattr(symbol, "name", "")), 0) or 0) > 0
    ]
    if branch_candidates:
        symbol = sorted(
            branch_candidates,
            key=lambda item: int(mask.uncovered_branches.get(str(getattr(item, "name", "")), 0)),
            reverse=True,
        )[0]
        return CoverageTarget(
            symbol_name=str(getattr(symbol, "name")),
            reason="uncovered_branches",
            priority=100 + int(mask.uncovered_branches.get(str(getattr(symbol, "name")), 0)),
        )
    for symbol in symbols:
        name = str(getattr(symbol, "name"))
        if name not in mask.covered_symbols:
            return CoverageTarget(symbol_name=name, reason="uncovered_symbol", priority=50)
    return None
