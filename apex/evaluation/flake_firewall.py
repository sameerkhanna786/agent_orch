"""Non-Deterministic-Failure Firewall (NDFF) for an authoritative scoring oracle.

In ``gold_suite_visible`` mode the project's own gold test suite IS the scoring
oracle, so a *flaky* gold test directly corrupts the published number: a
teardown/finalizer leak (Twisted ``DirtyReactorAggregateError``, an unclosed
asyncio loop, an atexit/threadpool finalizer), an order-dependent test, or a
test whose new failure has nothing to do with the candidate's change can flip a
correct candidate to a miss. SWE-bench filters flaky instances *once at
construction time*; no live SWE orchestrator runs a continuous, change-coverage-
grounded flake firewall over its own authoritative oracle during the solve loop.

NDFF closes that with three benchmark-agnostic, execution-grounded checks:

1. **Teardown-leak signature** — a green-except-ERRORS outcome whose error text
   matches a known cross-test teardown/finalizer leak is non-deterministic by
   construction (the failure leaked from another test's teardown, not from the
   candidate's code).

2. **DeFlaker change-coverage** (Bell et al., ICSE 2018) — a *newly failing*
   gold test whose executed coverage does not intersect the candidate's changed
   files cannot have been broken by the candidate; the failure is flaky. APEX can
   run this almost for free because it already knows the candidate diff.

3. **Per-node flakiness rate** — an observed history of pass/fail for a node id
   sizes the rerun budget to a confidence target instead of a flat constant.

This module is pure and dependency-light (only the failure taxonomy) so it is
unit-testable without any container or coverage backend, and is the single source
of truth for the teardown-leak marker set shared with the official-audit
re-audit budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from ..core.failure_classifier import FailureClass

# Canonical cross-test teardown / finalizer / event-loop leak signatures. A
# scored ERROR whose captured output matches one of these did not originate in
# the candidate's code — it leaked from another test's teardown — so it is
# non-deterministic. Shared with the official-audit transient re-audit budget.
TEARDOWN_LEAK_MARKERS: frozenset[str] = frozenset(
    {
        "dirtyreactoraggregateerror",
        "reactor was unclean",
        "reactoralreadyrunning",
        "reactornotrestartable",
        "delayedcall",
        "twisted.internet.error",
        "unhandled error in deferred",
        "error in atexit",
        "task was destroyed but it is pending",
        "event loop is closed",
        "no running event loop",
        "threadpool",
        "errored during teardown",
        "error during teardown",
        "error during finalizer",
        "fixture finalizer",
        "during sessionfinish",
        "during sessionstart",
        "exception ignored in",
        "resourcewarning",
    }
)


def output_has_teardown_leak_signature(output: Optional[str]) -> bool:
    """True if ``output`` matches a known cross-test teardown/finalizer leak."""
    lowered = str(output or "").lower()
    return any(marker in lowered for marker in TEARDOWN_LEAK_MARKERS)


def _normalize_files(files: Optional[Iterable[str]]) -> set[str]:
    out: set[str] = set()
    for value in files or ():
        text = str(value or "").strip().replace("\\", "/").lstrip("./")
        if text:
            out.add(text)
    return out


def failure_is_change_disjoint(
    *,
    failing_test_covered_files: Optional[Iterable[str]],
    changed_files: Optional[Iterable[str]],
) -> bool:
    """DeFlaker file-level check: a newly-failing test whose executed coverage
    does NOT intersect the candidate's changed files is flaky by construction.

    Returns ``False`` (cannot conclude flaky) when coverage is unknown/empty, so
    the firewall is conservative — it only declares a flake when it has positive
    evidence that the failing test never touched the change.
    """
    covered = _normalize_files(failing_test_covered_files)
    changed = _normalize_files(changed_files)
    if not covered or not changed:
        return False
    return covered.isdisjoint(changed)


@dataclass(frozen=True)
class FlakeVerdict:
    """Verdict for one oracle failure under the firewall."""

    is_flaky: bool
    kind: str = ""
    reason: str = ""
    failure_class: FailureClass = FailureClass.NON_DETERMINISTIC

    def to_dict(self) -> dict[str, object]:
        return {
            "is_flaky": bool(self.is_flaky),
            "kind": self.kind,
            "reason": self.reason,
            "failure_class": self.failure_class.value if self.is_flaky else "",
        }


def classify_oracle_failure(
    *,
    failed: int,
    errors: int,
    passed: int,
    output: Optional[str] = None,
    coverage_preserved: Optional[bool] = None,
    failing_test_covered_files: Optional[Iterable[str]] = None,
    changed_files: Optional[Iterable[str]] = None,
) -> FlakeVerdict:
    """Classify whether an oracle failure is non-deterministic (flaky).

    Strict by construction so it can never excuse a real regression:

    * a teardown-leak ERROR is flaky only when there are NO scored ``failed``
      tests, at least one ``errors``, some ``passed`` tests, coverage is not
      known to be broken, and the output matches a leak signature;
    * a DeFlaker change-disjoint failure is flaky only with positive coverage
      evidence that the failing test never executed any changed file.

    A real ``AssertionError`` / scored ``failed`` whose coverage intersects the
    change is therefore never declared flaky.
    """
    if int(failed) == 0 and int(errors) > 0 and int(passed) > 0:
        if coverage_preserved is not False and output_has_teardown_leak_signature(output):
            return FlakeVerdict(
                is_flaky=True,
                kind="teardown_leak",
                reason="green-except-ERRORS with a cross-test teardown/finalizer leak signature",
            )
    if failure_is_change_disjoint(
        failing_test_covered_files=failing_test_covered_files,
        changed_files=changed_files,
    ):
        return FlakeVerdict(
            is_flaky=True,
            kind="change_disjoint_coverage",
            reason="newly-failing test coverage does not intersect the candidate's changed files",
        )
    return FlakeVerdict(is_flaky=False)


@dataclass
class NodeFlakinessTracker:
    """Per-nodeid pass/fail history -> flakiness rate and a rerun budget.

    A node observed to flip between pass and fail is flaky; the rerun budget is
    sized to reach a confidence target rather than using a flat constant. Backed
    in-memory here; callers may persist the counts via the episodic store across
    runs (interface kept storage-agnostic on purpose).
    """

    passes: dict[str, int] = field(default_factory=dict)
    fails: dict[str, int] = field(default_factory=dict)

    def record(self, node_id: str, *, passed: bool) -> None:
        node_id = str(node_id).strip()
        if not node_id:
            return
        bucket = self.passes if passed else self.fails
        bucket[node_id] = bucket.get(node_id, 0) + 1

    def flakiness_rate(self, node_id: str) -> float:
        node_id = str(node_id).strip()
        p = self.passes.get(node_id, 0)
        f = self.fails.get(node_id, 0)
        total = p + f
        if total == 0:
            return 0.0
        # Flakiness peaks at a 50/50 flip rate; 1.0 - |p-f|/total maps a pure
        # pass-or-fail history to 0 and an even flip history to 1.
        return 1.0 - abs(p - f) / total

    def is_known_flaky(self, node_id: str) -> bool:
        node_id = str(node_id).strip()
        return self.passes.get(node_id, 0) > 0 and self.fails.get(node_id, 0) > 0

    def suggested_rerun_budget(
        self,
        node_id: str,
        *,
        floor: int = 1,
        ceiling: int = 5,
    ) -> int:
        """Size the rerun budget to the observed flakiness: a node never seen
        flaky gets ``floor`` reruns; a maximally-flaky node gets ``ceiling``."""
        rate = self.flakiness_rate(node_id)
        budget = floor + round(rate * (ceiling - floor))
        return max(floor, min(int(budget), ceiling))
