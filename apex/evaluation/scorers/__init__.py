"""Per-benchmark scorer implementations conforming to
:class:`apex.core.fairness_audit.BenchmarkScorerProtocol`.

Each benchmark provides one or two scorers:

* A *private* scorer using the APEX-internal scoring rules.
* An *upstream* scorer using the canonical published rules.

For SWT-Bench there is only one scorer (the upstream Docker harness)
because APEX has no private rewrite path for it. ``run_fairness_audit``
should be invoked with that scorer as both ``private_scorer`` and
``upstream_scorer``; the resulting delta is by-construction zero, which
is the correct, honest outcome.
"""

from __future__ import annotations

from .commit0_private import Commit0PrivateScorer, make_commit0_private_scorer
from .commit0_upstream import Commit0UpstreamScorer, make_commit0_upstream_scorer
from .swtbench_upstream import SWTBenchUpstreamScorer, make_swtbench_upstream_scorer
from .testgeneval_private import (
    TestGenEvalPrivateScorer,
    make_testgeneval_private_scorer,
)
from .testgeneval_upstream import (
    TestGenEvalUpstreamScorer,
    make_testgeneval_upstream_scorer,
)

__all__: list[str] = [
    "Commit0PrivateScorer",
    "Commit0UpstreamScorer",
    "SWTBenchUpstreamScorer",
    "TestGenEvalPrivateScorer",
    "TestGenEvalUpstreamScorer",
    "make_commit0_private_scorer",
    "make_commit0_upstream_scorer",
    "make_swtbench_upstream_scorer",
    "make_testgeneval_private_scorer",
    "make_testgeneval_upstream_scorer",
]
