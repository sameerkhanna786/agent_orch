#!/usr/bin/env python3
"""Phase 4.2 calibration script for testgen candidate ranking weights.

Given a directory of historical testgen runs (each containing an
``apex_result.json`` and the candidate-level metadata produced by
``apex.evaluation.multi_candidate.summarize_candidate_selection``), fit
a weighted-composite ranking by:

  1. Loading every candidate's signal vector (pass_at_1, mutation_score,
     coverage_delta, oracle_grounding, assertion_effect, dual_state_score,
     log1p(meaningful_test_count)) along with a binary outcome label
     (``1`` if the candidate was the post-hoc best — defined as the one
     with the highest oracle/F2P-validated mutation score, ``0`` otherwise).
  2. Searching for weights that maximize the rank-1 selection accuracy
     against the post-hoc label. Two backends are supported:
        * ``"grid"`` (default): exhaustive grid search over a coarse
          simplex of 7-component weight vectors.
        * ``"linreg"``: closed-form linear regression on the (signal →
          outcome) pairs, projected onto the simplex.
  3. Emitting:
        * A recommended weights dict on stdout as JSON.
        * A side-by-side comparison report (rank-1 accuracy under the
          default vs. recommended weights) on stderr.

The fitted weights are written nowhere by this script — operators paste
them into ``ApexConfig.selection.testgen_ranking_weights`` (global) or
``ApexConfig.benchmark.testgen_ranking_weights_override`` (per-benchmark)
themselves, so the calibration is auditable.

Usage:
    python -m apex.scripts.calibrate_testgen_ranking \\
        --runs-dir /path/to/historical/runs \\
        --backend grid \\
        --grid-step 0.05

This script is build-only for the Phase 4.2 milestone — there is no
historical data to run against yet. The smoke-test path (``--self-test``)
exercises the pipeline on synthetic data so future operators have a
known-good starting point.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# Local import keeps the script runnable both as a module and as a
# standalone file (``python apex/scripts/calibrate_testgen_ranking.py``).
try:
    from apex.evaluation.multi_candidate import (
        DEFAULT_TESTGEN_RANKING_WEIGHTS,
    )
except ImportError:  # pragma: no cover — script-mode fallback
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from apex.evaluation.multi_candidate import (
        DEFAULT_TESTGEN_RANKING_WEIGHTS,
    )


_WEIGHT_KEYS: tuple[str, ...] = (
    "pass_at_1",
    "mutation_score",
    "coverage_delta",
    "oracle_grounding",
    "assertion_effect",
    "dual_state_score",
    "meaningful_test_count_log",
)


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationRecord:
    """One candidate's signal vector + post-hoc outcome label.

    Attributes:
        run_id: Identifier of the parent run (one run = one task with N
            candidates). Used for grouping during accuracy evaluation —
            we only count "the selected candidate matched the
            post-hoc-best candidate" once per run.
        candidate_id: Identifier of the candidate within the run.
        signals: Dict keyed by the weight component name with the raw
            signal value. ``meaningful_test_count_log`` is precomputed
            here so the fitter can stay linear.
        is_best: Post-hoc label — True iff this candidate had the
            strongest objective signal in the run (we use F2P pass rate
            on the held-out oracle if present, falling back to the
            mutation score in the parent run JSON).
    """

    run_id: str
    candidate_id: str
    signals: dict[str, float]
    is_best: bool


def _signals_from_candidate(candidate_dict: dict[str, Any]) -> dict[str, float]:
    return {
        "pass_at_1": float(candidate_dict.get("unfiltered_pass_at_1") or 0.0),
        "mutation_score": float(candidate_dict.get("mutation_score") or 0.0),
        "coverage_delta": float(candidate_dict.get("coverage_delta") or 0.0),
        "oracle_grounding": float(candidate_dict.get("oracle_grounding_score") or 0.0),
        "assertion_effect": float(candidate_dict.get("assertion_effect_score") or 0.0),
        "dual_state_score": float(candidate_dict.get("dual_state_score") or 0.0),
        "meaningful_test_count_log": math.log1p(
            max(0, int(candidate_dict.get("meaningful_test_count") or 0))
        ),
    }


def _label_post_hoc_best(
    candidates: list[dict[str, Any]],
) -> Optional[str]:
    """Pick the post-hoc-best candidate id from a run.

    Preference order (descending):
      1. Highest ``unfiltered_pass_at_1`` — the official benchmark metric.
      2. Highest ``mutation_score`` — the discrimination signal.
      3. Highest ``oracle_grounding_score`` — for cases where pass_at_1
         and mutation are tied at zero.
    Returns ``None`` if the run has no candidates.
    """
    if not candidates:
        return None

    def key(c: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(c.get("unfiltered_pass_at_1") or 0.0),
            float(c.get("mutation_score") or 0.0),
            float(c.get("oracle_grounding_score") or 0.0),
        )

    best = max(candidates, key=key)
    return str(best.get("candidate_id") or "")


def load_records_from_runs_dir(runs_dir: Path) -> list[CalibrationRecord]:
    """Walk ``runs_dir`` and load every candidate-level record.

    Expected layout (matches what
    :func:`apex.evaluation.multi_candidate.summarize_candidate_selection`
    emits to ``apex_result.json``):

        runs_dir/<run_id>/apex_result.json
            {
                "candidate_selection": {
                    "candidates": [
                        {"candidate_id": ..., "unfiltered_pass_at_1": ..., ...},
                        ...
                    ],
                    "selected_candidate": "...",
                },
                ...
            }
    """
    records: list[CalibrationRecord] = []
    for result_path in sorted(runs_dir.glob("*/apex_result.json")):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_id = result_path.parent.name
        candidates = (payload.get("candidate_selection") or {}).get("candidates") or []
        post_hoc_best = _label_post_hoc_best(candidates)
        for c in candidates:
            cid = str(c.get("candidate_id") or "")
            if not cid:
                continue
            records.append(
                CalibrationRecord(
                    run_id=run_id,
                    candidate_id=cid,
                    signals=_signals_from_candidate(c),
                    is_best=(cid == post_hoc_best),
                )
            )
    return records


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------


def composite_score(
    signals: dict[str, float],
    weights: dict[str, float],
) -> float:
    return sum(weights.get(key, 0.0) * signals.get(key, 0.0) for key in _WEIGHT_KEYS)


def rank_one_accuracy(
    records: Iterable[CalibrationRecord],
    weights: dict[str, float],
) -> float:
    """Fraction of runs where argmax(composite_score) matches the
    post-hoc-best candidate. A perfect ranking returns 1.0."""

    by_run: dict[str, list[CalibrationRecord]] = {}
    for r in records:
        by_run.setdefault(r.run_id, []).append(r)
    if not by_run:
        return 0.0
    hits = 0
    for run_records in by_run.values():
        scored = [(composite_score(r.signals, weights), r) for r in run_records]
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored and scored[0][1].is_best:
            hits += 1
    return hits / len(by_run)


# ---------------------------------------------------------------------------
# Fitters
# ---------------------------------------------------------------------------


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, w) for w in weights.values())
    if total <= 0.0:
        return dict(DEFAULT_TESTGEN_RANKING_WEIGHTS)
    return {k: max(0.0, v) / total for k, v in weights.items()}


def grid_search(
    records: list[CalibrationRecord],
    *,
    step: float = 0.10,
) -> dict[str, float]:
    """Coarse simplex grid search over the 7-key weight vector.

    Step ``s`` enumerates ``round(1/s)+1``-ary points per axis and
    keeps only those summing to ~1 (within ``s/2``). For ``s=0.10`` this
    is ~8k candidates; for ``s=0.05`` ~84k. Costly but parallelizable.
    """
    levels = [round(i * step, 6) for i in range(int(round(1.0 / step)) + 1)]
    best_weights = dict(DEFAULT_TESTGEN_RANKING_WEIGHTS)
    best_score = rank_one_accuracy(records, best_weights)
    for combo in itertools.product(levels, repeat=len(_WEIGHT_KEYS)):
        if abs(sum(combo) - 1.0) > step / 2:
            continue
        weights = {key: float(value) for key, value in zip(_WEIGHT_KEYS, combo)}
        score = rank_one_accuracy(records, weights)
        if score > best_score:
            best_score = score
            best_weights = weights
    return best_weights


def linreg_fit(records: list[CalibrationRecord]) -> dict[str, float]:
    """Closed-form least-squares fit on (signal vector → label).

    We solve ``argmin_w || Xw - y ||^2`` via the normal equations
    ``w = (X^T X)^-1 X^T y`` using a tiny pure-Python Gauss-Jordan
    inverter (no numpy dependency at calibration time). The fitted
    coefficients are clamped to >= 0 and renormalized to a simplex so
    the result is a valid weight vector for the composite scorer.
    """
    if not records:
        return dict(DEFAULT_TESTGEN_RANKING_WEIGHTS)
    n_features = len(_WEIGHT_KEYS)
    xtx = [[0.0] * n_features for _ in range(n_features)]
    xty = [0.0] * n_features
    for r in records:
        x = [r.signals.get(k, 0.0) for k in _WEIGHT_KEYS]
        y = 1.0 if r.is_best else 0.0
        for i in range(n_features):
            xty[i] += x[i] * y
            for j in range(n_features):
                xtx[i][j] += x[i] * x[j]
    # Tikhonov: add a small ridge to make the inverse always defined.
    for i in range(n_features):
        xtx[i][i] += 1e-6
    inv = _invert_matrix(xtx)
    if inv is None:
        return dict(DEFAULT_TESTGEN_RANKING_WEIGHTS)
    coeffs = [sum(inv[i][j] * xty[j] for j in range(n_features)) for i in range(n_features)]
    fit = {key: max(0.0, value) for key, value in zip(_WEIGHT_KEYS, coeffs)}
    return _normalize(fit)


def _invert_matrix(matrix: list[list[float]]) -> Optional[list[list[float]]]:
    n = len(matrix)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]
    for i in range(n):
        pivot = aug[i][i]
        if abs(pivot) < 1e-12:
            for k in range(i + 1, n):
                if abs(aug[k][i]) > 1e-12:
                    aug[i], aug[k] = aug[k], aug[i]
                    pivot = aug[i][i]
                    break
            else:
                return None
        for j in range(2 * n):
            aug[i][j] /= pivot
        for k in range(n):
            if k == i:
                continue
            factor = aug[k][i]
            for j in range(2 * n):
                aug[k][j] -= factor * aug[i][j]
    return [row[n:] for row in aug]


# ---------------------------------------------------------------------------
# Self-test (synthetic data)
# ---------------------------------------------------------------------------


def _synthetic_records(seed: int = 0, n_runs: int = 30) -> list[CalibrationRecord]:
    """Generate synthetic candidate signals where mutation_score is the
    only true signal. The fitter should recover a heavy weight on
    mutation_score (and small weight on the other axes)."""
    import random as _random

    rng = _random.Random(seed)
    records: list[CalibrationRecord] = []
    for run_idx in range(n_runs):
        n_candidates = rng.randint(2, 5)
        cands: list[CalibrationRecord] = []
        for cand_idx in range(n_candidates):
            mut = rng.random()
            signals = {
                "pass_at_1": rng.random(),
                "mutation_score": mut,
                "coverage_delta": rng.random(),
                "oracle_grounding": rng.random(),
                "assertion_effect": rng.random(),
                "dual_state_score": rng.random(),
                "meaningful_test_count_log": math.log1p(rng.randint(0, 10)),
            }
            cands.append(
                CalibrationRecord(
                    run_id=f"run_{run_idx}",
                    candidate_id=f"c_{cand_idx}",
                    signals=signals,
                    is_best=False,
                )
            )
        # The "best" candidate is the one with the highest mutation_score.
        best_idx = max(range(len(cands)), key=lambda i: cands[i].signals["mutation_score"])
        for i, c in enumerate(cands):
            records.append(
                CalibrationRecord(
                    run_id=c.run_id,
                    candidate_id=c.candidate_id,
                    signals=c.signals,
                    is_best=(i == best_idx),
                )
            )
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Decisive-Edge D.2 — critic reranker calibration
#
# The SelectionCritic in ``apex.selection.selector`` scores each candidate
# cluster as a weighted sum of named features (issue_alignment,
# risk_coverage, ..., conflict_discipline). The literature prior lives in
# ``apex.selection.selector.DEFAULT_CRITIC_WEIGHTS``. This section refits
# those weights against the same per-task historical run records used for
# the testgen ranking calibration, then writes
# ``apex/configs/critic_weights_calibrated.json`` (the path that
# SelectionCritic loads at construction time).
#
# Reuse: the per-feature signal vector for each candidate is taken from
# ``cluster_critic_features`` if present (the selector emits these into
# ``apex_result.json`` post-rollout). When that block is missing we fall
# back to the same raw signals used for ranking calibration, mapped to
# the closest critic feature name. The mapping is deliberately
# conservative — when no signal maps to a feature we keep the literature
# prior weight for that key.
# ---------------------------------------------------------------------------


# Critic feature keys (must match
# ``apex.selection.selector.DEFAULT_CRITIC_FEATURE_KEYS``).
_CRITIC_FEATURE_KEYS: tuple[str, ...] = (
    "issue_alignment",
    "risk_coverage",
    "localization_alignment",
    "consensus_alignment",
    "source_change_quality",
    "patch_focus",
    "test_alignment",
    "obligation_coverage",
    "hypothesis_alignment",
    "task_state_focus_alignment",
    "artifact_confidence",
    "outcome_signal",
    "test_edit_discipline",
    "conflict_discipline",
)


def _critic_default_weights() -> dict[str, float]:
    """Fetch the literature-prior critic weights without forcing the
    selector module to be importable at script-startup time (the
    selector pulls in heavy planning / git deps)."""
    try:
        from apex.selection.selector import DEFAULT_CRITIC_WEIGHTS

        return dict(DEFAULT_CRITIC_WEIGHTS)
    except ImportError:  # pragma: no cover — script-mode fallback
        # Hard-coded mirror of the literature prior. Kept in sync with
        # apex/selection/selector.py:DEFAULT_CRITIC_WEIGHTS.
        return {
            "issue_alignment": 0.10,
            "risk_coverage": 0.06,
            "localization_alignment": 0.14,
            "consensus_alignment": 0.11,
            "source_change_quality": 0.06,
            "patch_focus": 0.04,
            "test_alignment": 0.05,
            "obligation_coverage": 0.09,
            "hypothesis_alignment": 0.05,
            "task_state_focus_alignment": 0.04,
            "artifact_confidence": 0.05,
            "outcome_signal": 0.08,
            "test_edit_discipline": 0.07,
            "conflict_discipline": 0.06,
        }


def _default_critic_weights_output_path() -> Path:
    """``apex/configs/critic_weights_calibrated.json`` next to the
    ``apex`` package — same layout SelectionCritic loads from."""
    return Path(__file__).resolve().parents[1] / "configs" / "critic_weights_calibrated.json"


@dataclass(frozen=True)
class CriticCalibrationRecord:
    """One critic-feature vector + post-hoc binary label.

    ``signals`` is keyed by critic feature name. Missing keys score 0.0
    in the composite, just like the ranking calibrator. ``is_best`` is
    the same post-hoc label used for testgen ranking — the candidate
    in this run with the strongest objective signal.
    """

    run_id: str
    candidate_id: str
    signals: dict[str, float]
    is_best: bool


def _critic_signals_from_candidate(candidate_dict: dict[str, Any]) -> dict[str, float]:
    """Extract critic feature scores from one candidate-record JSON.

    First tries ``cluster_critic_features`` (the dict the selector emits
    when ``enable_critic_reranking`` is on). Falls back to a manual
    mapping from the raw ranking signals — coverage_delta as a proxy
    for source_change_quality, mutation_score for patch_focus, etc.
    """
    raw = candidate_dict.get("critic_features") or candidate_dict.get("cluster_critic_features")
    if isinstance(raw, dict) and raw:
        return {key: float(raw.get(key) or 0.0) for key in _CRITIC_FEATURE_KEYS}
    # Fallback: derive a low-fidelity vector from the ranking-side
    # signals so a calibration run on a runs-dir without explicit critic
    # features still produces *some* signal. Keys without a sensible
    # mapping default to 0.0 (zero signal — the prior weight wins).
    pass_at_1 = float(candidate_dict.get("unfiltered_pass_at_1") or 0.0)
    mutation = float(candidate_dict.get("mutation_score") or 0.0)
    coverage = float(candidate_dict.get("coverage_delta") or 0.0)
    oracle = float(candidate_dict.get("oracle_grounding_score") or 0.0)
    assertion = float(candidate_dict.get("assertion_effect_score") or 0.0)
    dual = float(candidate_dict.get("dual_state_score") or 0.0)
    return {
        "issue_alignment": coverage,
        "risk_coverage": coverage,
        "localization_alignment": oracle,
        "consensus_alignment": dual,
        "source_change_quality": coverage,
        "patch_focus": mutation,
        "test_alignment": pass_at_1,
        "obligation_coverage": pass_at_1,
        "hypothesis_alignment": oracle,
        "task_state_focus_alignment": oracle,
        "artifact_confidence": assertion,
        "outcome_signal": pass_at_1,
        "test_edit_discipline": 0.0,
        "conflict_discipline": 0.0,
    }


def load_critic_records_from_runs_dir(runs_dir: Path) -> list[CriticCalibrationRecord]:
    """Mirror of :func:`load_records_from_runs_dir` that emits the
    critic-feature vector instead of the ranking-signal vector.
    """
    records: list[CriticCalibrationRecord] = []
    for result_path in sorted(runs_dir.glob("*/apex_result.json")):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_id = result_path.parent.name
        candidates = (payload.get("candidate_selection") or {}).get("candidates") or []
        post_hoc_best = _label_post_hoc_best(candidates)
        for c in candidates:
            cid = str(c.get("candidate_id") or "")
            if not cid:
                continue
            records.append(
                CriticCalibrationRecord(
                    run_id=run_id,
                    candidate_id=cid,
                    signals=_critic_signals_from_candidate(c),
                    is_best=(cid == post_hoc_best),
                )
            )
    return records


def critic_composite_score(
    signals: dict[str, float],
    weights: dict[str, float],
) -> float:
    return sum(weights.get(key, 0.0) * signals.get(key, 0.0) for key in _CRITIC_FEATURE_KEYS)


def critic_rank_one_accuracy(
    records: Iterable[CriticCalibrationRecord],
    weights: dict[str, float],
) -> float:
    by_run: dict[str, list[CriticCalibrationRecord]] = {}
    for r in records:
        by_run.setdefault(r.run_id, []).append(r)
    if not by_run:
        return 0.0
    hits = 0
    for run_records in by_run.values():
        scored = [(critic_composite_score(r.signals, weights), r) for r in run_records]
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored and scored[0][1].is_best:
            hits += 1
    return hits / len(by_run)


def _normalize_critic(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, w) for w in weights.values())
    if total <= 0.0:
        return _critic_default_weights()
    return {k: max(0.0, v) / total for k, v in weights.items()}


def critic_linreg_fit(records: list[CriticCalibrationRecord]) -> dict[str, float]:
    """Closed-form least-squares fit on (critic feature vector → label).

    Same ridge-regularized normal equations used for the ranking
    calibrator. Returns a normalized non-negative weight dict.
    """
    if not records:
        return _critic_default_weights()
    n_features = len(_CRITIC_FEATURE_KEYS)
    xtx = [[0.0] * n_features for _ in range(n_features)]
    xty = [0.0] * n_features
    for r in records:
        x = [r.signals.get(k, 0.0) for k in _CRITIC_FEATURE_KEYS]
        y = 1.0 if r.is_best else 0.0
        for i in range(n_features):
            xty[i] += x[i] * y
            for j in range(n_features):
                xtx[i][j] += x[i] * x[j]
    for i in range(n_features):
        xtx[i][i] += 1e-6
    inv = _invert_matrix(xtx)
    if inv is None:
        return _critic_default_weights()
    coeffs = [sum(inv[i][j] * xty[j] for j in range(n_features)) for i in range(n_features)]
    fit = {key: max(0.0, value) for key, value in zip(_CRITIC_FEATURE_KEYS, coeffs)}
    return _normalize_critic(fit)


def _expected_calibration_error(
    records: Iterable[CriticCalibrationRecord],
    weights: dict[str, float],
    *,
    bins: int = 10,
) -> float:
    """Expected calibration error of the critic composite as a probability.

    Standard bucketed ECE: bucket records by their composite score in
    [0, 1] and compare bucket mean confidence to bucket-empirical
    accuracy. Used purely as a diagnostic — no impact on the fitted
    weights.
    """
    by_record = list(records)
    if not by_record:
        return 0.0
    # Snap composite into [0, 1] for the bucketing.
    raw = [(critic_composite_score(r.signals, weights), r.is_best) for r in by_record]
    if not raw:
        return 0.0
    lo = min(score for score, _ in raw)
    hi = max(score for score, _ in raw)
    span = hi - lo
    if span <= 0.0:
        return 0.0
    norms = [((score - lo) / span, label) for score, label in raw]
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for score, label in norms:
        idx = min(bins - 1, int(score * bins))
        buckets[idx].append((score, label))
    total = len(norms)
    ece = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        avg_conf = sum(s for s, _ in bucket) / len(bucket)
        avg_acc = sum(1 for _, lbl in bucket if lbl) / len(bucket)
        ece += (len(bucket) / total) * abs(avg_conf - avg_acc)
    return ece


def _synthetic_critic_records(seed: int = 0, n_runs: int = 30) -> list[CriticCalibrationRecord]:
    """Synthetic critic records where ``localization_alignment`` is the
    only true signal. The fitter should recover a heavy weight on it."""
    import random as _random

    rng = _random.Random(seed)
    records: list[CriticCalibrationRecord] = []
    for run_idx in range(n_runs):
        n_candidates = rng.randint(2, 5)
        cands: list[CriticCalibrationRecord] = []
        for cand_idx in range(n_candidates):
            signals = {key: rng.random() for key in _CRITIC_FEATURE_KEYS}
            cands.append(
                CriticCalibrationRecord(
                    run_id=f"run_{run_idx}",
                    candidate_id=f"c_{cand_idx}",
                    signals=signals,
                    is_best=False,
                )
            )
        best_idx = max(
            range(len(cands)),
            key=lambda i: cands[i].signals["localization_alignment"],
        )
        for i, c in enumerate(cands):
            records.append(
                CriticCalibrationRecord(
                    run_id=c.run_id,
                    candidate_id=c.candidate_id,
                    signals=c.signals,
                    is_best=(i == best_idx),
                )
            )
    return records


def _build_critic_weights_payload(
    *,
    fitted: dict[str, float],
    n_records: int,
    n_runs: int,
    rank1_accuracy: float,
    ece: float,
    backend: str,
    synthetic: bool,
) -> dict[str, Any]:
    """Build the JSON payload SelectionCritic expects.

    Schema mirrors the placeholder shipped at
    ``apex/configs/critic_weights_calibrated.json``.
    """
    return {
        "_schema_version": "1",
        "_comment": (
            "Calibrated SelectionCritic reranker weights. Generated by "
            "apex/scripts/calibrate_testgen_ranking.py --target critic. "
            "Loaded by apex.selection.selector.SelectionCritic."
        ),
        "policy_version": "calibrated-v1" if not synthetic else "calibrated-v0-synthetic",
        "weights": {key: float(fitted.get(key, 0.0)) for key in _CRITIC_FEATURE_KEYS},
        "training_metadata": {
            "n_examples": int(n_records),
            "n_runs": int(n_runs),
            "ece": float(ece),
            "rank1_accuracy": float(rank1_accuracy),
            "synthetic": bool(synthetic),
            "backend": str(backend),
        },
    }


def calibrate_critic_weights(
    *,
    runs_dir: Optional[Path],
    backend: str,
    self_test: bool,
    output_path: Optional[Path],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Run the critic calibration pipeline.

    Returns ``(fitted_weights, payload)``. Writes the payload to
    ``output_path`` (or the default path next to apex/configs) when the
    fitted weights are non-trivially different from the prior.
    """
    if self_test:
        records = _synthetic_critic_records()
        synthetic = True
    elif runs_dir is not None:
        records = load_critic_records_from_runs_dir(runs_dir)
        synthetic = False
    else:
        records = []
        synthetic = False

    default = _critic_default_weights()
    if not records:
        # Graceful no-op: log a warning and return the prior. Caller
        # decides whether to emit the placeholder JSON or skip.
        logger.warning(
            "No critic-calibration records loaded (runs_dir=%s, self_test=%s); "
            "returning literature-prior weights without writing output.",
            runs_dir,
            self_test,
        )
        payload = _build_critic_weights_payload(
            fitted=default,
            n_records=0,
            n_runs=0,
            rank1_accuracy=0.0,
            ece=0.0,
            backend=backend,
            synthetic=True,
        )
        return default, payload

    if backend == "linreg":
        fitted = critic_linreg_fit(records)
    elif backend == "grid":
        # Grid for 14 features at any reasonable step is intractable
        # (~10**14 candidates); we deliberately fall back to linreg for
        # the critic. Operators wanting a true grid search should
        # extend this branch with a coordinate-descent variant.
        logger.warning(
            "Grid backend requested for critic calibration but the 14-key "
            "simplex is too large; falling back to linreg fit.",
        )
        fitted = critic_linreg_fit(records)
    else:
        fitted = critic_linreg_fit(records)

    default_acc = critic_rank_one_accuracy(records, default)
    fitted_acc = critic_rank_one_accuracy(records, fitted)
    ece = _expected_calibration_error(records, fitted)
    payload = _build_critic_weights_payload(
        fitted=fitted,
        n_records=len(records),
        n_runs=len({r.run_id for r in records}),
        rank1_accuracy=fitted_acc,
        ece=ece,
        backend=backend,
        synthetic=synthetic,
    )
    sys.stderr.write(
        "critic_calibration_report:\n"
        f"  records:                 {len(records)}\n"
        f"  runs:                    {len({r.run_id for r in records})}\n"
        f"  backend:                 {backend}\n"
        f"  default_rank1_accuracy:  {default_acc:.4f}\n"
        f"  fitted_rank1_accuracy:   {fitted_acc:.4f}\n"
        f"  improvement:             {(fitted_acc - default_acc):+.4f}\n"
        f"  expected_calibration_err: {ece:.4f}\n"
    )
    if output_path is not None:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            sys.stderr.write(f"  wrote critic weights -> {output_path}\n")
        except OSError as exc:
            logger.warning("Failed to write critic weights to %s: %s", output_path, exc)
    return fitted, payload


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        help="Directory containing historical testgen runs. "
        "Each run is expected at <runs-dir>/<run_id>/apex_result.json.",
    )
    parser.add_argument(
        "--backend",
        choices=("grid", "linreg"),
        default="grid",
        help="Calibration backend (default: grid).",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.10,
        help="Step size for the grid backend (smaller = finer = slower).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run on synthetic data instead of --runs-dir. Useful as a "
        "smoke test that the fitter pipeline is wired correctly.",
    )
    parser.add_argument(
        "--target",
        choices=("ranking", "critic", "both"),
        default="both",
        help="Which weights to calibrate. 'ranking' = the testgen "
        "weighted-composite ranking (legacy behavior). 'critic' = the "
        "SelectionCritic reranker weights (new in Decisive-Edge D.2). "
        "'both' (default) runs both passes against the same runs-dir.",
    )
    parser.add_argument(
        "--critic-output",
        type=Path,
        default=None,
        help="Output path for the calibrated critic weights JSON. "
        "Defaults to apex/configs/critic_weights_calibrated.json.",
    )
    args = parser.parse_args(argv)

    target = args.target
    needs_runs = (not args.self_test) and (args.runs_dir is None)
    if needs_runs:
        parser.error("Must supply --runs-dir or --self-test.")
        return 2

    exit_code = 0

    if target in {"ranking", "both"}:
        if args.self_test:
            records = _synthetic_records()
        else:
            records = load_records_from_runs_dir(args.runs_dir)

        if not records:
            sys.stderr.write("No ranking records loaded; skipping ranking fit.\n")
            if target == "ranking":
                return 1
            exit_code = max(exit_code, 1) if target == "ranking" else exit_code
        else:
            default_weights = dict(DEFAULT_TESTGEN_RANKING_WEIGHTS)
            if args.backend == "grid":
                fitted = grid_search(records, step=float(args.grid_step))
            else:
                fitted = linreg_fit(records)

            default_acc = rank_one_accuracy(records, default_weights)
            fitted_acc = rank_one_accuracy(records, fitted)

            sys.stderr.write(
                "calibration_report:\n"
                f"  records:              {len(records)}\n"
                f"  runs:                 {len({r.run_id for r in records})}\n"
                f"  backend:              {args.backend}\n"
                f"  default_rank1_accuracy: {default_acc:.4f}\n"
                f"  fitted_rank1_accuracy:  {fitted_acc:.4f}\n"
                f"  improvement:            {(fitted_acc - default_acc):+.4f}\n"
            )
            sys.stdout.write(json.dumps(fitted, indent=2, sort_keys=True) + "\n")

    if target in {"critic", "both"}:
        if args.self_test and args.critic_output is None:
            with tempfile.TemporaryDirectory(prefix="apex-critic-calibration-") as tmp:
                calibrate_critic_weights(
                    runs_dir=args.runs_dir,
                    backend=args.backend,
                    self_test=args.self_test,
                    output_path=Path(tmp) / "critic_weights_calibrated.json",
                )
        else:
            critic_output = args.critic_output or _default_critic_weights_output_path()
            calibrate_critic_weights(
                runs_dir=args.runs_dir,
                backend=args.backend,
                self_test=args.self_test,
                output_path=critic_output,
            )

    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
