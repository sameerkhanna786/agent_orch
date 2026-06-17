"""APEX orchestration package (Phase 3.2 decomposition).

The legacy ``apex/orchestrator.py`` monolith (4,005+ lines) lives
here split across responsibility-aligned modules:

    * :mod:`apex.orchestration.solver` — ``ApexOrchestrator`` class +
      ``ApexResult``. ``solve()`` is now a thin coordinator.
    * :mod:`apex.orchestration.waves` — wave-execution helpers
      (``execute_with_dynamic_transitions``,
      ``execute_progressive_rollout_plan``).
    * :mod:`apex.orchestration.followups` — abstract ``FollowupLoop``
      base + four concrete subclasses (NearMiss, StructuralRecovery,
      CoverageGap, Selection).
    * :mod:`apex.orchestration.recovery` — orchestration glue that
      strings the four follow-up loops together (was inline in the
      old ``solve()``).
    * :mod:`apex.orchestration.acceptance` — pure helpers like
      ``selected_result_is_accepted`` and
      ``rollout_has_strong_progressive_signal``.
    * :mod:`apex.orchestration.escalation` — strategy-identity helper
      driving the dynamic-transition loop guard.

The ``apex/orchestrator.py`` path remains as a thin shim re-exporting
all previously-importable names, so ``from apex.orchestrator import X``
continues to work for every existing X.
"""

from __future__ import annotations

from .abstention import (
    DEFAULT_ABSTENTION_THRESHOLD,
    DEFAULT_ABSTENTION_WEIGHTS,
    ConfidenceBreakdown,
    ConfidenceScorer,
    ParetoFrontier,
    ParetoPoint,
    compute_pareto_frontier,
    emit_pareto_artifacts,
)
from .acceptance import (
    rollout_has_authoritative_completion_signal,
    rollout_has_expected_coverage_gap,
    rollout_has_local_full_suite_completion_signal,
    rollout_has_strong_progressive_signal,
    selected_result_is_accepted,
)
from .budget_planner import (
    DEFAULT_PER_SUBTASK_TURNS,
    DEFAULT_REBALANCE_STRATEGY,
    BudgetPlanner,
    TurnBudget,
)
from .escalation import strategy_identity_for_loop_guard
from .followups import (
    CoverageGapFollowup,
    FollowupLoop,
    NearMissFollowup,
    SelectionFollowup,
    StructuralRecoveryFollowup,
)
from .hierarchical_agent import (
    DEFAULT_MAX_SUBTASKS,
    HierarchicalAgent,
    HierarchicalRunSummary,
    default_decompose,
)
from .solver import ApexOrchestrator, ApexResult
from .verification_amplifier import (
    AmplificationResult,
    DiscriminatingTest,
    DiscriminationMatrix,
    VerificationAmplifier,
)
from .waves import (
    execute_progressive_rollout_plan,
    execute_with_dynamic_transitions,
)

__all__ = [
    "AmplificationResult",
    "ApexOrchestrator",
    "ApexResult",
    "BudgetPlanner",
    "ConfidenceBreakdown",
    "ConfidenceScorer",
    "CoverageGapFollowup",
    "DEFAULT_ABSTENTION_THRESHOLD",
    "DEFAULT_ABSTENTION_WEIGHTS",
    "DEFAULT_MAX_SUBTASKS",
    "DEFAULT_PER_SUBTASK_TURNS",
    "DEFAULT_REBALANCE_STRATEGY",
    "DiscriminatingTest",
    "DiscriminationMatrix",
    "FollowupLoop",
    "HierarchicalAgent",
    "HierarchicalRunSummary",
    "NearMissFollowup",
    "ParetoFrontier",
    "ParetoPoint",
    "SelectionFollowup",
    "StructuralRecoveryFollowup",
    "TurnBudget",
    "VerificationAmplifier",
    "compute_pareto_frontier",
    "default_decompose",
    "emit_pareto_artifacts",
    "execute_progressive_rollout_plan",
    "execute_with_dynamic_transitions",
    "rollout_has_authoritative_completion_signal",
    "rollout_has_expected_coverage_gap",
    "rollout_has_local_full_suite_completion_signal",
    "rollout_has_strong_progressive_signal",
    "selected_result_is_accepted",
    "strategy_identity_for_loop_guard",
]
