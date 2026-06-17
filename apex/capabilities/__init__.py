"""APEX Phase 6 novelty capabilities.

These modules add NEW behaviors on top of the mature codegen / testgen /
joint pipelines without changing those pipelines' default semantics:

* :mod:`apex.capabilities.self_play` — adversarial test-vs-patch
  tournament selection (item 6.1).
* :mod:`apex.capabilities.episodic_memory` — high-level facade over the
  cross-run episodic store (item 6.2). The persistent store itself
  lives in :mod:`apex.persistence.episodic_store`.
* :mod:`apex.capabilities.active_learning` — mutation-targeted active
  learning loop that grows test suites by attacking surviving mutants
  (item 6.4).

The capabilities are intentionally orthogonal: each can be enabled
independently and they share no mutable state.
"""

from .active_learning import (
    ActiveLearningStop,
    EnhancedTestSuite,
    MutationActiveLearner,
)
from .episodic_memory import (
    Hypothesis,
    learn_from_prior_run,
    record_outcome,
)
from .self_play import (
    SelfPlayResult,
    SelfPlayTournament,
    score_pair,
    select_best_pair,
)

__all__ = [
    "ActiveLearningStop",
    "EnhancedTestSuite",
    "Hypothesis",
    "MutationActiveLearner",
    "SelfPlayResult",
    "SelfPlayTournament",
    "learn_from_prior_run",
    "record_outcome",
    "score_pair",
    "select_best_pair",
]
