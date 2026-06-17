"""Adversarial test-vs-patch self-play (Phase 6 item 6.1).

APEX already produces tests AND patches. Today they are generated in a
single coupled flow (testgen → codegen against tests, or codegen → F2P
against tests) which makes the produced pair correlated by construction.
Self-play instead generates K patches and M tests INDEPENDENTLY (no
hand-off, no shared scratchpad), then evaluates the full K x M cross-
product to find the (patch, test_suite) pair that maximises mutual
confidence.

Selection criterion
-------------------

For verdict matrix ``V`` (K rows of patches by M columns of test suites)
where ``V[i, j] = 1`` iff test suite ``j`` PASSES under patch ``i``:

* ``per_patch_survival_rate[i] = mean_j V[i, j]`` — how often patch i
  survives each test suite. A patch that survives most test suites is a
  more confident patch.
* ``per_test_kill_rate[j] = 1 - mean_i V[i, j]`` — how often test j
  KILLS each patch. A test that kills bad patches is a more confident
  test.
* ``confidence(i, j) = (1 - survival_rate_against_other_tests_i)
                       * kill_rate_against_other_patches_j``

The "against other" framing avoids self-reference: the patch's
confidence comes from how it does on test suites OTHER THAN ``j``, and
the test suite's confidence comes from how it does on patches OTHER
THAN ``i``. We then pick the pair that maximises ``V[i, j] *
confidence(i, j)`` so that the selected pair is also internally
consistent (the chosen tests pass under the chosen patch).

Cost
----

Default ``K=4, M=4`` => 16 patch-vs-test evaluations per task. Each
evaluation is just a test run (NOT a fresh LLM call) — the K patches
and M test suites are generated up front and then re-used across the
matrix. Total LLM cost is therefore K + M generations + K*M test
executions, an order of magnitude below "K*M independent rollouts".

Public API
----------

The high-level entry point is
:func:`apex.modes.run_generate_both_self_play` which calls
``run_codegen_with_tests`` K times and ``run_testgen_with_fix`` M
times in parallel, then runs :class:`SelfPlayTournament` over the
resulting candidates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# --- Defaults ---

DEFAULT_K_PATCHES = 4
DEFAULT_M_TESTS = 4
DEFAULT_PARALLELISM = 4


# Type aliases for pluggable callables. The defaults call into the real
# evaluation pipeline; tests inject fakes.
PatchCandidate = dict[str, Any]
TestCandidate = dict[str, Any]
VerdictFn = Callable[[PatchCandidate, TestCandidate], int]


@dataclass
class SelfPlayResult:
    """Outcome of one K x M tournament.

    The matrix uses the convention ``verdict_matrix[i, j] = 1`` when
    test suite ``j`` PASSES under patch ``i``. Survival is therefore a
    row mean and kill rate is ``1 - column mean``.
    """

    verdict_matrix: np.ndarray  # shape (K, M), int8
    per_patch_survival_rate: list[float]
    per_test_kill_rate: list[float]
    selected_patch_index: int
    selected_test_index: int
    mutual_confidence: float
    selected_patch: Optional[PatchCandidate] = None
    selected_test: Optional[TestCandidate] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict_matrix": self.verdict_matrix.tolist(),
            "per_patch_survival_rate": list(self.per_patch_survival_rate),
            "per_test_kill_rate": list(self.per_test_kill_rate),
            "selected_patch_index": int(self.selected_patch_index),
            "selected_test_index": int(self.selected_test_index),
            "mutual_confidence": float(self.mutual_confidence),
            "diagnostics": dict(self.diagnostics),
        }


# ---------------------------------------------------------------------------
# Selection criterion
# ---------------------------------------------------------------------------


def _row_mean_excluding(matrix: np.ndarray, row: int, col: int) -> float:
    """Mean of row ``row`` excluding column ``col``.

    When the row has only one entry the "excluding" set is empty; we
    return the value at (row, col) so the score reduces gracefully.
    """
    K, M = matrix.shape
    if M <= 1:
        return float(matrix[row, col])
    mask = np.ones(M, dtype=bool)
    mask[col] = False
    return float(matrix[row, mask].mean())


def _col_mean_excluding(matrix: np.ndarray, row: int, col: int) -> float:
    K, M = matrix.shape
    if K <= 1:
        return float(matrix[row, col])
    mask = np.ones(K, dtype=bool)
    mask[row] = False
    return float(matrix[mask, col].mean())


def score_pair_mutual_confidence(survival_excl: float, kill_excl: float) -> float:
    """Original Phase 6.1 score: ``(1 - survival_excl) * kill_excl``.

    Phase 6A flagged this as perverse: a "good" patch that survives
    every other test gets driven to score 0 because (1 - survival)
    collapses. Retained for back-compat and ablation studies.
    """
    return float((1.0 - float(survival_excl)) * float(kill_excl))


def score_pair_survival_x_kill(survival_excl: float, kill_excl: float) -> float:
    """Plain product: rewards patches that survive AND tests that kill.

    The most intuitive scoring: a strong patch should pass tests AND
    a strong test should reject other (presumably weaker) patches.
    Doesn't penalise the highest-quality patches the way
    ``mutual_confidence`` does.
    """
    return float(float(survival_excl) * float(kill_excl))


def score_pair_harmonic_mean(survival_excl: float, kill_excl: float) -> float:
    """Harmonic mean of survival and kill rate — Phase B.7 default.

    Penalises imbalance more aggressively than ``survival_x_kill``
    (a pair that's strong in one dimension but weak in the other
    cannot dominate). The harmonic mean is widely used as a balanced
    consensus score (F1 between precision and recall is the same
    formula). Returns 0 when both inputs are 0 to avoid divide-by-zero.
    """
    s = float(survival_excl)
    k = float(kill_excl)
    denom = s + k
    if denom <= 0.0:
        return 0.0
    return float(2.0 * s * k / denom)


def score_pair_borda_count(matrix: np.ndarray, i: int, j: int) -> float:
    """Borda-count style rank aggregation across the matrix.

    For pair (i, j), rank patch i within column j (how many other
    patches it beats on this test) and rank test j within row i (how
    many other tests it kills on this patch). The score is the sum
    of normalised ranks. Discrete, deterministic, robust to ties.
    """
    K, M = matrix.shape
    if K <= 1 and M <= 1:
        return float(matrix[i, j])
    # Patch rank within column j: count strictly-lower patches plus
    # half of ties (excluding i itself).
    col = matrix[:, j]
    below = int(np.sum(col < col[i]))
    ties = int(np.sum(col == col[i])) - 1
    patch_rank = (below + 0.5 * max(ties, 0)) / max(K - 1, 1)
    # Test rank within row i: a "winning" test is one whose verdict
    # equals 0 (kills the patch) for OTHER patches, but here we want
    # to credit tests that DISAGREE with the row's mean — that's the
    # discrimination signal.
    row = matrix[i, :]
    # We want tests that KILL more patches than others. ``row[j]`` is
    # the verdict on patch i; for "discriminating" we look at the
    # OTHER columns of THIS row to see if test j stands out as
    # discriminative. Approximate: rank by 1 - row mean excluding j.
    if M > 1:
        mask = np.ones(M, dtype=bool)
        mask[j] = False
        test_disc = 1.0 - float(row[mask].mean())
    else:
        test_disc = 0.0
    return float(0.5 * patch_rank + 0.5 * test_disc)


# Map of name -> two-arg scorer (survival_excl, kill_excl). Borda is
# matrix-aware so it lives outside this map; ``score_pair`` dispatches.
_SCALAR_SCORING_STRATEGIES: dict[str, Any] = {
    "mutual_confidence": score_pair_mutual_confidence,
    "survival_x_kill": score_pair_survival_x_kill,
    "harmonic_mean": score_pair_harmonic_mean,
}


# Phase B.7 default. The original ``mutual_confidence`` formula was
# flagged as perverse during Phase 6A review; harmonic mean is a
# balanced consensus score that doesn't penalise high-quality patches
# the way ``mutual_confidence`` does.
DEFAULT_SCORING_STRATEGY = "harmonic_mean"


def score_pair(
    matrix: np.ndarray,
    i: int,
    j: int,
    *,
    strategy: str = DEFAULT_SCORING_STRATEGY,
) -> float:
    """Joint confidence of pair (patch_i, test_j) under ``strategy``.

    The "against other" framing is preserved for the scalar strategies:
    each input is computed excluding the pair's own row/column. Borda
    operates on the full matrix so it's dispatched separately.

    Strategies:

    * ``"mutual_confidence"`` — original ``(1 - survival_excl) * kill_excl``.
      Flagged as perverse (penalises strong patches); kept for back-compat.
    * ``"survival_x_kill"``   — ``survival_excl * kill_excl``.
    * ``"harmonic_mean"``     — DEFAULT; harmonic mean of the two.
    * ``"borda"``             — rank-aggregation across the full matrix.
    """
    strat = (strategy or DEFAULT_SCORING_STRATEGY).strip().lower()
    if strat == "borda":
        return score_pair_borda_count(matrix, i, j)
    survival_excl = _row_mean_excluding(matrix, i, j)
    kill_excl = 1.0 - _col_mean_excluding(matrix, i, j)
    fn = _SCALAR_SCORING_STRATEGIES.get(strat)
    if fn is None:
        raise ValueError(
            f"Unknown self-play scoring strategy {strategy!r}; "
            f"valid: {sorted(_SCALAR_SCORING_STRATEGIES) + ['borda']}"
        )
    return float(fn(survival_excl, kill_excl))


def select_best_pair(
    matrix: np.ndarray,
    *,
    strategy: str = DEFAULT_SCORING_STRATEGY,
) -> tuple[int, int, float]:
    """Pick the (patch_i, test_j) pair maximising ``score_pair``.

    Internal consistency tie-break: among pairs with equal score we
    prefer ones where ``V[i, j] == 1`` (the chosen test passes under
    the chosen patch). The motivation is operational: a (patch, test)
    pair where the test fails on the patch is internally inconsistent
    and would surface as a regression at promotion time.

    Returns ``(patch_index, test_index, score)``. On a 0xN, Nx0, or
    empty matrix we return ``(0, 0, 0.0)``.

    ``strategy`` defaults to ``DEFAULT_SCORING_STRATEGY`` (Phase B.7
    harmonic mean). Pass ``"mutual_confidence"`` for the original
    Phase 6.1 formula (kept for ablation).
    """
    K, M = matrix.shape
    if K == 0 or M == 0:
        return 0, 0, 0.0
    best_i = 0
    best_j = 0
    best_score = float("-inf")
    best_consistency = -1
    for i in range(K):
        for j in range(M):
            score = score_pair(matrix, i, j, strategy=strategy)
            consistency = int(matrix[i, j])
            # Strictly better score wins; equal score with better
            # consistency wins; equal consistency keeps the earlier pair
            # for determinism.
            if score > best_score or (score == best_score and consistency > best_consistency):
                best_i = i
                best_j = j
                best_score = score
                best_consistency = consistency
    return best_i, best_j, max(0.0, best_score)


# ---------------------------------------------------------------------------
# Tournament driver
# ---------------------------------------------------------------------------


@dataclass
class SelfPlayTournament:
    """Cross-evaluate K patches against M test suites.

    Inputs to :meth:`run` are the already-generated patch candidates
    and test suite candidates. The tournament does NOT call the LLM
    itself — it expects a ``verdict_fn`` that returns 1 if the given
    test suite passes under the given patch and 0 otherwise (the
    default uses :func:`evaluate_tdd_iteration` internally; tests
    inject a fake).

    ``scoring_strategy`` (Phase B.7) selects the pair-scoring formula.
    Defaults to ``"harmonic_mean"`` — the Phase 6A review flagged the
    original ``mutual_confidence`` formula as perverse (it penalised
    patches that survived too many tests). Operators wanting the
    legacy behaviour can pass ``scoring_strategy="mutual_confidence"``.
    Valid values: ``"harmonic_mean"``, ``"survival_x_kill"``,
    ``"mutual_confidence"``, ``"borda"``.
    """

    K_patches: int = DEFAULT_K_PATCHES
    M_tests: int = DEFAULT_M_TESTS
    parallelism: int = DEFAULT_PARALLELISM
    scoring_strategy: str = DEFAULT_SCORING_STRATEGY

    def __post_init__(self) -> None:
        self.K_patches = max(1, int(self.K_patches))
        self.M_tests = max(1, int(self.M_tests))
        self.parallelism = max(1, int(self.parallelism))
        self.scoring_strategy = (self.scoring_strategy or DEFAULT_SCORING_STRATEGY).strip().lower()
        valid = set(_SCALAR_SCORING_STRATEGIES) | {"borda"}
        if self.scoring_strategy not in valid:
            raise ValueError(
                f"Unknown SelfPlayTournament scoring_strategy "
                f"{self.scoring_strategy!r}; valid: {sorted(valid)}"
            )

    def run(
        self,
        *,
        patch_candidates: list[PatchCandidate],
        test_candidates: list[TestCandidate],
        verdict_fn: VerdictFn,
    ) -> SelfPlayResult:
        """Build the K x M verdict matrix and select the best pair.

        ``patch_candidates`` and ``test_candidates`` may be shorter
        than the configured K / M (e.g. one rollout failed); we use
        whatever is supplied. Empty inputs short-circuit to a degenerate
        result with selection ``(0, 0)`` and confidence 0.0.
        """
        K = len(patch_candidates)
        M = len(test_candidates)
        if K == 0 or M == 0:
            return SelfPlayResult(
                verdict_matrix=np.zeros((max(K, 1), max(M, 1)), dtype=np.int8),
                per_patch_survival_rate=[0.0] * max(K, 1),
                per_test_kill_rate=[0.0] * max(M, 1),
                selected_patch_index=0,
                selected_test_index=0,
                mutual_confidence=0.0,
                diagnostics={
                    "K_patches": K,
                    "M_tests": M,
                    "degenerate": True,
                    "reason": "empty patch or test candidate set",
                },
            )

        matrix = np.zeros((K, M), dtype=np.int8)

        # We could parallelise the K*M evaluations with a ThreadPool,
        # but most verdict_fn implementations under the hood are already
        # sandboxed test runs that hold the GIL release through
        # subprocess. Keep it serial here to avoid surprising the
        # caller's parallelism budget; callers that want parallel
        # verdict eval can pass a verdict_fn that internally batches.
        eval_errors: list[dict[str, Any]] = []
        for i in range(K):
            for j in range(M):
                try:
                    verdict = int(verdict_fn(patch_candidates[i], test_candidates[j]))
                    matrix[i, j] = 1 if verdict else 0
                except Exception as exc:  # pragma: no cover — defensive
                    matrix[i, j] = 0
                    eval_errors.append(
                        {
                            "patch_index": i,
                            "test_index": j,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    logger.warning(
                        "self_play: verdict eval (i=%d, j=%d) raised %s",
                        i,
                        j,
                        exc,
                    )

        per_patch_survival = [float(matrix[i, :].mean()) for i in range(K)]
        per_test_kill = [float(1.0 - matrix[:, j].mean()) for j in range(M)]
        sel_i, sel_j, score = select_best_pair(matrix, strategy=self.scoring_strategy)

        diagnostics: dict[str, Any] = {
            "K_patches": K,
            "M_tests": M,
            "degenerate": False,
            "all_patches_failed": all(s == 0.0 for s in per_patch_survival),
            "all_tests_trivial": all(k == 0.0 for k in per_test_kill),
            "selection_internally_consistent": bool(matrix[sel_i, sel_j] == 1),
            "scoring_strategy": str(self.scoring_strategy),
        }
        if eval_errors:
            diagnostics["eval_errors"] = eval_errors[:8]
            diagnostics["eval_error_count"] = len(eval_errors)

        return SelfPlayResult(
            verdict_matrix=matrix,
            per_patch_survival_rate=per_patch_survival,
            per_test_kill_rate=per_test_kill,
            selected_patch_index=sel_i,
            selected_test_index=sel_j,
            mutual_confidence=score,
            selected_patch=patch_candidates[sel_i],
            selected_test=test_candidates[sel_j],
            diagnostics=diagnostics,
        )
