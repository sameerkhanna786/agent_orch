"""Phase 6.3 — Calibrated abstention as a first-class outcome.

Building on Phase 2C's salvage-as-success removal (the canonical
``Status`` enum now has ``ABSTAINED`` as a peer of ``SOLVED``), this
module makes the *confidence* behind that decision explicit.

The orchestrator already exposes a ``Status`` enum, but the *score*
that drove the decision was implicit — a mishmash of the verifier's
``accepted`` boolean, ``quick_verification`` thresholds, and (via
Phase 3.4) a calibrated controller policy state. Phase 6.3 collects
those signals into a single, weighted, normalized ``ConfidenceBreakdown``
so:

  * the acceptance gate can use a *calibrated threshold* instead of a
    cliff-edged ``accepted == True`` check,
  * benchmark reports can sweep the threshold and emit a Pareto curve
    of (precision, abstention_rate),
  * downstream consumers can introspect WHY the run was accepted /
    abstained (per-component contributions) rather than guessing.

Public surface
--------------

  * :class:`ConfidenceBreakdown` — dataclass returned by the scorer.
  * :class:`ConfidenceScorer` — instance configurable with weights /
    threshold; ``score(...)`` returns a ``ConfidenceBreakdown``.
  * :func:`compute_pareto_frontier` — module-level analysis helper that
    sweeps a threshold over a list of pre-scored
    ``(ConfidenceBreakdown, ground_truth_solved)`` pairs and returns a
    Pareto curve of (precision, abstention_rate).
  * :func:`emit_pareto_artifacts` — writes ``pareto_frontier.json`` and
    ``PARETO_REPORT.md`` to a target directory.

Calibration weights
-------------------

The default per-component weights are literature-informed (mirror the
Phase 4.2 ranking weights) and intentionally conservative — they sum to
1.0 with ``salvage_penalty`` subtracted so a salvage-only patch never
out-scores an honestly-verified one:

    verifier_strength            0.30
    cluster_consensus            0.20
    controller_policy_certainty  0.20
    mutation_kill_rate           0.15
    f2p_consensus_rate           0.10
    salvage_penalty              0.05  (subtracted)

Operators tune the weights and the threshold via ``OrchestrationConfig``
(``abstention_threshold``, ``abstention_weights``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence

__all__ = [
    "ConfidenceBreakdown",
    "ConfidenceScorer",
    "ParetoFrontier",
    "ParetoPoint",
    "compute_pareto_frontier",
    "compute_pareto_frontier_per_benchmark",
    "emit_pareto_artifacts",
    "load_calibrated_abstention_thresholds",
    "resolve_abstention_threshold",
    "DEFAULT_ABSTENTION_WEIGHTS",
    "DEFAULT_ABSTENTION_THRESHOLD",
    "BENCHMARK_ID_ALIASES",
]


# Decisive-Edge C.2: callers stamp ``benchmark_metadata["benchmark_name"]``
# with values like "swtbench" / "commit0" / "swebench_pro". The calibrator
# normalizes those to "swt_bench" / "commit0" / "swebench_pro" / etc. when
# writing the per-benchmark thresholds JSON. Mirror the calibrator's alias
# table here so threshold lookups resolve regardless of which spelling the
# caller used.
BENCHMARK_ID_ALIASES: dict[str, str] = {
    "commit0_lite": "commit0",
    "commit0": "commit0",
    "swtbench": "swt_bench",
    "swtbench_lite": "swt_bench",
    "swt_bench": "swt_bench",
    "swt_bench_lite": "swt_bench",
    "testgeneval": "testgeneval",
    "testgeneval_lite": "testgeneval",
    "swebench_pro": "swebench_pro",
    "swebench_pro_testgen": "swebench_pro_testgen",
    "swe_evo": "swe_evo",
}


def _normalize_benchmark_id(benchmark_id: Optional[str]) -> str:
    text = (benchmark_id or "").strip().lower()
    if not text:
        return ""
    return BENCHMARK_ID_ALIASES.get(text, text)


DEFAULT_ABSTENTION_THRESHOLD: float = 0.50

# Weights are non-negative and (verifier + cluster + controller + mutation +
# f2p) sum to 0.95; ``salvage_penalty`` (0.05) is subtracted from the sum so
# the total is in [-0.05, 1.0]. We clamp to [0, 1] in score().
DEFAULT_ABSTENTION_WEIGHTS: dict[str, float] = {
    "verifier_strength": 0.30,
    "cluster_consensus": 0.20,
    "controller_policy_certainty": 0.20,
    "mutation_kill_rate": 0.15,
    "f2p_consensus_rate": 0.10,
    "salvage_penalty": 0.05,
}


# Decisive-Edge C.2: in-process LRU cache so the scorer doesn't re-read the
# JSON on every score() call. The cache key is the resolved file path; pass
# ``cache=False`` to ``load_calibrated_abstention_thresholds`` to bypass.
_CALIBRATED_THRESHOLDS_CACHE: dict[str, dict[str, float]] = {}


def _default_calibrated_thresholds_path() -> Path:
    """Resolve ``apex/configs/abstention_thresholds_per_benchmark.json``.

    This walks two parents up from this module — ``apex/orchestration/`` →
    ``apex/`` — then into ``configs/``. The path is computed lazily so
    test suites can monkey-patch the env without import-time side effects.
    """
    return (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "abstention_thresholds_per_benchmark.json"
    )


def load_calibrated_abstention_thresholds(
    path: Optional[str | Path] = None,
    *,
    cache: bool = True,
) -> dict[str, float]:
    """Load the per-benchmark calibrated abstention thresholds JSON.

    Parameters
    ----------
    path
        Optional path to the JSON file. Defaults to
        ``apex/configs/abstention_thresholds_per_benchmark.json`` resolved
        relative to this module's package root.
    cache
        When True (default) the resolved file is read once per process
        and cached by absolute path. Pass ``cache=False`` from tests
        that mutate the file between calls.

    Returns
    -------
    dict[str, float]
        Mapping ``{benchmark_id: threshold}``. Always returns an empty
        dict when the file is missing or unparseable — callers should
        fall through to the global default in that case.
    """
    resolved = Path(path) if path is not None else _default_calibrated_thresholds_path()
    cache_key = str(resolved)
    if cache and cache_key in _CALIBRATED_THRESHOLDS_CACHE:
        return dict(_CALIBRATED_THRESHOLDS_CACHE[cache_key])
    table: dict[str, float] = {}
    try:
        if resolved.exists():
            with resolved.open(encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                for key, value in data.items():
                    if str(key).startswith("_"):
                        continue
                    try:
                        table[str(key)] = float(value)
                    except (TypeError, ValueError):
                        continue
    except (OSError, json.JSONDecodeError):
        table = {}
    if cache:
        _CALIBRATED_THRESHOLDS_CACHE[cache_key] = dict(table)
    return table


def resolve_abstention_threshold(
    *,
    benchmark_id: Optional[str] = None,
    benchmark_override: Optional[float] = None,
    global_default: float = DEFAULT_ABSTENTION_THRESHOLD,
    calibrated_table: Optional[Mapping[str, float]] = None,
    calibrated_path: Optional[str | Path] = None,
) -> float:
    """Resolve the effective abstention threshold for one run.

    Priority (highest first):
      1. Per-benchmark calibrated threshold (from
         ``apex/configs/abstention_thresholds_per_benchmark.json``).
      2. Per-benchmark override (``benchmark_override``).
      3. Global default (``global_default``).

    Both the raw and the normalized benchmark id are looked up in the
    calibrated table — operators that wrote either spelling get a hit.

    Parameters
    ----------
    benchmark_id
        Benchmark identifier (e.g. "commit0", "swtbench", "testgeneval").
        ``None`` skips the per-benchmark lookups and returns the override
        or the global default.
    benchmark_override
        ``BenchmarkConfig.abstention_threshold_override`` for this run.
    global_default
        ``OrchestrationConfig.abstention_threshold`` for this run.
    calibrated_table
        Optional pre-loaded calibrated table. When ``None`` the table is
        loaded from disk via :func:`load_calibrated_abstention_thresholds`.
    calibrated_path
        Optional path passed through to the loader.
    """
    if benchmark_id:
        if calibrated_table is None:
            calibrated_table = load_calibrated_abstention_thresholds(path=calibrated_path)
        # Try the raw key first, then the normalized alias.
        for candidate in (str(benchmark_id), _normalize_benchmark_id(benchmark_id)):
            if not candidate:
                continue
            if candidate in calibrated_table:
                try:
                    return float(calibrated_table[candidate])
                except (TypeError, ValueError):
                    pass
    if benchmark_override is not None:
        try:
            return float(benchmark_override)
        except (TypeError, ValueError):
            pass
    return float(global_default)


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out:  # NaN check without importing math
        return default
    return out


@dataclass
class ConfidenceBreakdown:
    """Calibrated, introspectable confidence for one APEX run.

    ``overall`` is the weighted aggregate, clamped to [0, 1].
    ``breakdown`` records the *raw* per-component scores BEFORE weighting
    so reports can show "the verifier said 0.9 but cluster consensus was
    only 0.4". ``recommended_action`` is computed against
    ``threshold_used``.
    """

    overall: float
    breakdown: dict[str, float] = field(default_factory=dict)
    recommended_action: Literal["accept", "abstain"] = "abstain"
    threshold_used: float = DEFAULT_ABSTENTION_THRESHOLD
    weights_used: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": round(float(self.overall), 6),
            "breakdown": {str(k): round(float(v), 6) for k, v in self.breakdown.items()},
            "recommended_action": self.recommended_action,
            "threshold_used": round(float(self.threshold_used), 6),
            "weights_used": {str(k): round(float(v), 6) for k, v in self.weights_used.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConfidenceBreakdown":
        action = str(data.get("recommended_action") or "abstain")
        if action not in ("accept", "abstain"):
            action = "abstain"
        return cls(
            overall=_safe_float(data.get("overall"), 0.0),
            breakdown={
                str(k): _safe_float(v, 0.0) for k, v in dict(data.get("breakdown") or {}).items()
            },
            recommended_action=action,  # type: ignore[arg-type]
            threshold_used=_safe_float(data.get("threshold_used"), DEFAULT_ABSTENTION_THRESHOLD),
            weights_used={
                str(k): _safe_float(v, 0.0) for k, v in dict(data.get("weights_used") or {}).items()
            },
        )


class ConfidenceScorer:
    """Aggregate per-component signals into a calibrated confidence score.

    The scorer is intentionally tolerant of missing data: each component
    falls back to 0.0 when its source signal is absent. That keeps the
    scorer usable in early phases where (e.g.) mutation testing hasn't
    run yet, without mis-scoring the result.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_ABSTENTION_THRESHOLD,
        weights: Optional[Mapping[str, float]] = None,
        *,
        benchmark_threshold_override: Optional[float] = None,
        calibrated_thresholds: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.threshold = float(threshold)
        merged = dict(DEFAULT_ABSTENTION_WEIGHTS)
        if weights:
            for key, value in weights.items():
                if key not in merged:
                    raise ValueError(
                        f"unknown abstention weight key: {key!r}; valid keys = {sorted(merged)}"
                    )
                merged[key] = float(value)
        self.weights: dict[str, float] = merged
        # Decisive-Edge C.2: per-benchmark resolution lives on the scorer
        # so callers don't have to re-pass these on every score() call.
        # ``benchmark_threshold_override`` is the
        # :attr:`BenchmarkConfig.abstention_threshold_override` for the
        # current run; ``calibrated_thresholds`` is the pre-loaded
        # per-benchmark table (defaults to disk read on first lookup).
        self.benchmark_threshold_override: Optional[float] = (
            float(benchmark_threshold_override)
            if benchmark_threshold_override is not None
            else None
        )
        self.calibrated_thresholds: Optional[dict[str, float]] = (
            {str(k): float(v) for k, v in calibrated_thresholds.items()}
            if calibrated_thresholds is not None
            else None
        )

    def resolve_threshold(self, benchmark_id: Optional[str] = None) -> float:
        """Resolve the effective threshold for the supplied benchmark.

        See :func:`resolve_abstention_threshold` for the priority order.
        Without ``benchmark_id`` this returns the scorer's own
        ``self.threshold`` (set at construction from
        ``OrchestrationConfig.abstention_threshold``).
        """
        if not benchmark_id:
            if self.benchmark_threshold_override is not None:
                return float(self.benchmark_threshold_override)
            return float(self.threshold)
        return resolve_abstention_threshold(
            benchmark_id=benchmark_id,
            benchmark_override=self.benchmark_threshold_override,
            global_default=self.threshold,
            calibrated_table=self.calibrated_thresholds,
        )

    # ------------------------------------------------------------------
    # Component extractors
    # ------------------------------------------------------------------

    @staticmethod
    def _verification_has_clean_full_scope_pass(verification: Mapping[str, Any]) -> bool:
        """Return True for an accepted zero-failure full-scope test signal."""
        candidates: list[Mapping[str, Any]] = [verification]
        for key in ("test_result", "quick_verification"):
            nested = verification.get(key)
            if isinstance(nested, Mapping):
                candidates.append(nested)
        for signal in candidates:
            try:
                passed = int(signal.get("passed") or 0)
                failed = int(signal.get("failed") or 0)
                errors = int(signal.get("errors") or 0)
                missing_expected = int(signal.get("missing_expected_test_count") or 0)
            except (TypeError, ValueError):
                continue
            coverage_preserved = signal.get("expected_coverage_preserved")
            if coverage_preserved is None:
                coverage_preserved = signal.get("coverage_preserved")
            if (
                passed > 0
                and failed == 0
                and errors == 0
                and missing_expected == 0
                and coverage_preserved is not False
            ):
                return True
        return False

    @staticmethod
    def _rollout_is_consensus_candidate(rollout: Any) -> bool:
        """Only successful/accepted candidates should affect consensus."""
        if bool(getattr(rollout, "success", False)):
            return True
        if bool(getattr(rollout, "internally_accepted", False)):
            return True
        verification = getattr(rollout, "verification", None)
        if isinstance(verification, Mapping) and verification.get("accepted") is True:
            return True
        qv = getattr(rollout, "quick_verification", None)
        if isinstance(qv, Mapping):
            try:
                passed = int(qv.get("passed") or 0)
                failed = int(qv.get("failed") or 0)
                errors = int(qv.get("errors") or 0)
                missing_expected = int(qv.get("missing_expected_test_count") or 0)
            except (TypeError, ValueError):
                return False
            coverage_preserved = qv.get("coverage_preserved")
            if (
                passed > 0
                and failed == 0
                and errors == 0
                and missing_expected == 0
                and coverage_preserved is not False
            ):
                return True
        return False

    @staticmethod
    def _extract_verifier_strength(apex_result: Any) -> tuple[float, bool]:
        """Extract the verifier's confidence in [0,1].

        Priority: ``verification_summary["overall_score"]`` if present;
        else 1.0 if ``internally_accepted``; else 0.0.

        Returns ``(score, present)``. ``present == False`` only when the
        verifier produced no signal at all (no overall_score, no
        accepted, no internally_accepted=True).
        """
        verification = getattr(apex_result, "verification_summary", None)
        if isinstance(verification, dict):
            clean_full_scope_pass = verification.get(
                "accepted"
            ) is True and ConfidenceScorer._verification_has_clean_full_scope_pass(verification)
            score = verification.get("overall_score")
            if isinstance(score, (int, float)):
                if clean_full_scope_pass:
                    return max(_clamp01(float(score)), 0.95), True
                return _clamp01(float(score)), True
            if verification.get("accepted") is True:
                return 1.0, True
            if verification.get("accepted") is False:
                return 0.0, True
        if bool(getattr(apex_result, "internally_accepted", False)):
            return 1.0, True
        return 0.0, False

    @staticmethod
    def _extract_cluster_consensus(
        apex_result: Any,
        rollout_results: Optional[Sequence[Any]] = None,
    ) -> tuple[float, bool]:
        """Fraction of rollout clusters agreeing on the winning patch.

        We treat ``selected_changed_files`` as the cluster signature
        (canonical sorted tuple of file paths). The consensus rate is
        ``count_matching_winner / total_with_signature``.

        Returns ``(rate, present)``. ``present == False`` when there are
        no rollouts to compare against (e.g., V5 single-shot agent).
        """
        winner_files = tuple(sorted(getattr(apex_result, "selected_changed_files", []) or []))
        rollouts = list(rollout_results or [])
        if not rollouts or not winner_files:
            return 0.0, False
        total = 0
        agreeing = 0
        selected_id = getattr(apex_result, "selected_rollout_id", None)
        for r in rollouts:
            if not ConfidenceScorer._rollout_is_consensus_candidate(r):
                continue
            files = getattr(r, "changed_files", None)
            if files is None:
                continue
            sig = tuple(sorted(files))
            if not sig:
                continue
            total += 1
            if sig == winner_files or (
                selected_id is not None and getattr(r, "rollout_id", None) == selected_id
            ):
                agreeing += 1
        if total == 0:
            return 0.0, False
        return _clamp01(agreeing / total), True

    @staticmethod
    def _extract_controller_policy_certainty(
        controller_action: Any,
    ) -> tuple[float, bool]:
        """Max calibrated state probability from the Phase 3.4 policy.

        Accepts:
          * an object with ``state_probabilities: dict[str, float]``
            (a TaskRegimeProfile or similar)
          * a dict containing ``state_probabilities`` or ``probabilities``
          * a dict mapping state-name -> probability directly
          * a numeric scalar (already-aggregated certainty)

        Returns ``(certainty, present)``. ``present == False`` only when
        no controller action / probability data was supplied.
        """
        if controller_action is None:
            return 0.0, False
        if isinstance(controller_action, (int, float)):
            return _clamp01(float(controller_action)), True
        # Dataclass / object path
        probs = getattr(controller_action, "state_probabilities", None)
        if isinstance(probs, dict) and probs:
            try:
                return _clamp01(max(float(v) for v in probs.values())), True
            except ValueError:
                return 0.0, False
        if isinstance(controller_action, dict):
            cand = controller_action.get("state_probabilities") or controller_action.get(
                "probabilities"
            )
            if isinstance(cand, dict) and cand:
                try:
                    return _clamp01(max(float(v) for v in cand.values())), True
                except ValueError:
                    return 0.0, False
            # Treat the dict itself as a probability mapping if all values are numeric.
            try:
                values = [float(v) for v in controller_action.values()]
                if values:
                    return _clamp01(max(values)), True
            except (TypeError, ValueError):
                return 0.0, False
        return 0.0, False

    @staticmethod
    def _extract_mutation_kill_rate(
        apex_result: Any,
        rollout_results: Optional[Sequence[Any]] = None,
    ) -> tuple[float, bool]:
        """Mutation kill rate from the verification summary, when available.

        Returns ``(rate, present)``.
        """
        verification = getattr(apex_result, "verification_summary", None)
        if isinstance(verification, dict):
            for key in ("mutation_kill_rate", "mutation_kill_score", "mutation_score"):
                if key in verification and isinstance(verification[key], (int, float)):
                    return _clamp01(float(verification[key])), True
            mut = verification.get("mutation_summary") or verification.get("mutation")
            if isinstance(mut, dict):
                for key in ("kill_rate", "kill_score", "score"):
                    if key in mut and isinstance(mut[key], (int, float)):
                        return _clamp01(float(mut[key])), True
        # Fallback: scan rollouts for the same key on their verification.
        for r in rollout_results or []:
            ver = getattr(r, "verification", None)
            if isinstance(ver, dict):
                for key in ("mutation_kill_rate", "mutation_kill_score"):
                    if key in ver and isinstance(ver[key], (int, float)):
                        return _clamp01(float(ver[key])), True
        return 0.0, False

    @staticmethod
    def _extract_f2p_consensus_rate(
        apex_result: Any,
        rollout_results: Optional[Sequence[Any]] = None,
    ) -> tuple[float, bool]:
        """Fraction of expected fail-to-pass tests that the patch flips.

        Pulls from ``verification_summary`` first (orchestrator-level
        rollup) then falls back to the selected rollout's
        ``quick_verification.expected_coverage_ratio``. Returns
        ``(rate, present)``.
        """
        verification = getattr(apex_result, "verification_summary", None)
        if isinstance(verification, dict):
            for key in (
                "f2p_consensus_rate",
                "f2p_pass_rate",
                "expected_coverage_ratio",
            ):
                if key in verification and isinstance(verification[key], (int, float)):
                    return _clamp01(float(verification[key])), True
            qv = verification.get("quick_verification")
            if isinstance(qv, dict):
                ratio = qv.get("expected_coverage_ratio")
                if isinstance(ratio, (int, float)):
                    return _clamp01(float(ratio)), True
        # Last-ditch: find the selected rollout in rollout_results and
        # peek at its quick_verification.
        selected_id = getattr(apex_result, "selected_rollout_id", None)
        if selected_id is not None:
            for r in rollout_results or []:
                if getattr(r, "rollout_id", None) == selected_id:
                    qv = getattr(r, "quick_verification", None)
                    if isinstance(qv, dict):
                        ratio = qv.get("expected_coverage_ratio")
                        if isinstance(ratio, (int, float)):
                            return _clamp01(float(ratio)), True
        return 0.0, False

    @staticmethod
    def _extract_salvage_penalty(apex_result: Any) -> tuple[float, bool]:
        """Returns ``(penalty, present)``. ``present`` is True iff the
        run *was* on the salvage path; non-salvage runs report
        ``(0.0, False)`` so the salvage component doesn't down-weight a
        clean accept just by being included in the average.
        """
        if bool(getattr(apex_result, "salvaged", False)):
            return 1.0, True
        if bool(getattr(apex_result, "salvaged_for_external_scoring", False)):
            return 1.0, True
        return 0.0, False

    # ------------------------------------------------------------------
    # Public scoring
    # ------------------------------------------------------------------

    def score(
        self,
        apex_result: Any,
        controller_action: Any = None,
        rollout_results: Optional[Sequence[Any]] = None,
        *,
        benchmark_id: Optional[str] = None,
    ) -> ConfidenceBreakdown:
        """Aggregate per-component signals into a calibrated confidence.

        Aggregation:

        * Each per-component extractor returns ``(value, present)`` where
          ``present == False`` means the underlying signal wasn't
          available for this run (not "available and equal to zero").
        * The aggregate is the **weight-normalized average over the
          PRESENT positive components** (verifier / cluster / controller
          / mutation / f2p), then the salvage_penalty fraction is
          subtracted (only when present). Renormalising over present
          components prevents missing-data signals from silently dragging
          a verifier-confident run below the threshold.
        * If no positive component is present the aggregate is 0.0 and
          the run is recommended to abstain.

        The returned ``ConfidenceBreakdown.overall`` is in [0, 1].
        ``recommended_action`` is ``"accept"`` iff ``overall >= threshold``,
        where ``threshold`` is the per-benchmark resolved threshold when
        ``benchmark_id`` is supplied (calibrated > override > global) —
        see :meth:`resolve_threshold`.
        """
        effective_threshold = self.resolve_threshold(benchmark_id)
        components = {
            "verifier_strength": self._extract_verifier_strength(apex_result),
            "cluster_consensus": self._extract_cluster_consensus(apex_result, rollout_results),
            "controller_policy_certainty": (
                self._extract_controller_policy_certainty(controller_action)
            ),
            "mutation_kill_rate": self._extract_mutation_kill_rate(apex_result, rollout_results),
            "f2p_consensus_rate": self._extract_f2p_consensus_rate(apex_result, rollout_results),
            "salvage_penalty": self._extract_salvage_penalty(apex_result),
        }
        breakdown: dict[str, float] = {k: v[0] for k, v in components.items()}

        positive_keys = (
            "verifier_strength",
            "cluster_consensus",
            "controller_policy_certainty",
            "mutation_kill_rate",
            "f2p_consensus_rate",
        )
        weight_sum_present = 0.0
        weighted_present = 0.0
        for key in positive_keys:
            value, present = components[key]
            if not present:
                continue
            weight = float(self.weights.get(key, 0.0))
            weight_sum_present += weight
            weighted_present += value * weight
        if weight_sum_present > 0.0:
            base_score = weighted_present / weight_sum_present
        else:
            base_score = 0.0
        # Salvage penalty is a fractional discount applied multiplicatively
        # only when the salvage path was actually used. The fractional
        # discount equals the salvage_penalty WEIGHT (default 0.05).
        salvage_value, salvage_present = components["salvage_penalty"]
        salvage_weight = float(self.weights.get("salvage_penalty", 0.0))
        if salvage_present and salvage_value > 0.0:
            overall = _clamp01(base_score * (1.0 - salvage_weight * salvage_value))
        else:
            overall = _clamp01(base_score)

        recommendation: Literal["accept", "abstain"] = (
            "accept" if overall >= effective_threshold else "abstain"
        )
        return ConfidenceBreakdown(
            overall=overall,
            breakdown=breakdown,
            recommended_action=recommendation,
            threshold_used=effective_threshold,
            weights_used=dict(self.weights),
        )


# ---------------------------------------------------------------------------
# Pareto-frontier reporting
# ---------------------------------------------------------------------------


@dataclass
class ParetoPoint:
    """One point on the (precision, abstention_rate) curve."""

    threshold: float
    precision: float
    abstention_rate: float
    accepted: int
    abstained: int
    true_positives: int
    false_positives: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": round(float(self.threshold), 6),
            "precision": round(float(self.precision), 6),
            "abstention_rate": round(float(self.abstention_rate), 6),
            "accepted": int(self.accepted),
            "abstained": int(self.abstained),
            "true_positives": int(self.true_positives),
            "false_positives": int(self.false_positives),
        }


@dataclass
class ParetoFrontier:
    """Threshold sweep + pareto-optimal subset."""

    points: list[ParetoPoint] = field(default_factory=list)
    n_runs: int = 0
    n_solved_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_runs": int(self.n_runs),
            "n_solved_total": int(self.n_solved_total),
            "points": [p.to_dict() for p in self.points],
        }


def _result_is_solved(result: Any) -> bool:
    """Treat ``Status.SOLVED`` (string ``"solved"``) as the success label.

    Accepts an ApexResult, a Status enum, a status string, or a dict.
    """
    # Direct Status / string
    if hasattr(result, "value") and isinstance(getattr(result, "value"), str):
        return result.value == "solved"
    if isinstance(result, str):
        return result == "solved"
    # ApexResult / object
    status = getattr(result, "status", None)
    if status is not None:
        if hasattr(status, "value"):
            return status.value == "solved"
        if isinstance(status, str):
            return status == "solved"
    if isinstance(result, dict):
        st = result.get("status")
        if hasattr(st, "value"):
            return st.value == "solved"
        if isinstance(st, str):
            return st == "solved"
    if hasattr(result, "success"):
        return bool(result.success)
    return False


def _result_confidence(result: Any) -> Optional[ConfidenceBreakdown]:
    cb = getattr(result, "confidence", None)
    if isinstance(cb, ConfidenceBreakdown):
        return cb
    if isinstance(cb, dict):
        return ConfidenceBreakdown.from_dict(cb)
    return None


def compute_pareto_frontier(
    results: Sequence[Any],
    *,
    thresholds: Optional[Iterable[float]] = None,
    confidence_extractor: Optional[Any] = None,
    ground_truth_extractor: Optional[Any] = None,
) -> ParetoFrontier:
    """Sweep an abstention threshold and return the (precision, abstention_rate)
    curve.

    Args:
        results: a sequence of ApexResult-like items. Each must expose
            ``.confidence`` (a ``ConfidenceBreakdown``) and either a
            ``Status``-valued ``.status`` or a ``.success`` bool.
        thresholds: which thresholds to evaluate. Defaults to 0.0, 0.05,
            0.10, ..., 1.0 (21 points) — fine enough to draw a smooth
            curve; coarse enough to keep the JSON small.
        confidence_extractor: optional callable ``result -> ConfidenceBreakdown``
            for non-ApexResult inputs.
        ground_truth_extractor: optional callable ``result -> bool`` indicating
            "is this result actually a true positive (i.e. the patch is
            correct)?". Defaults to ``Status.SOLVED``-based mapping. Use
            this hook to plug in the *external* benchmark grader rather
            than the internal acceptance gate.

    Returns:
        A ``ParetoFrontier`` with one ``ParetoPoint`` per threshold.

    Definitions:
        * accepted = result.confidence.overall >= threshold
        * true_positive = accepted AND ground_truth_solved
        * false_positive = accepted AND NOT ground_truth_solved
        * precision = TP / (TP + FP), or 1.0 when denominator is 0
        * abstention_rate = abstained / total
    """
    if thresholds is None:
        thresholds = [round(i * 0.05, 4) for i in range(0, 21)]
    threshold_list = sorted({round(float(t), 6) for t in thresholds})
    extractor = confidence_extractor or _result_confidence
    truth_fn = ground_truth_extractor or _result_is_solved

    items: list[tuple[float, bool]] = []
    for r in results:
        cb = extractor(r)
        if cb is None:
            # Skip results without a confidence — they're not on the
            # frontier (legacy results from before 6.3 wiring).
            continue
        items.append((float(cb.overall), bool(truth_fn(r))))

    n_runs = len(items)
    n_solved_total = sum(1 for _, s in items if s)
    points: list[ParetoPoint] = []
    for thr in threshold_list:
        accepted = 0
        true_positives = 0
        false_positives = 0
        for score, solved in items:
            if score >= thr:
                accepted += 1
                if solved:
                    true_positives += 1
                else:
                    false_positives += 1
        abstained = n_runs - accepted
        denom = true_positives + false_positives
        precision = (true_positives / denom) if denom > 0 else 1.0
        abstention_rate = (abstained / n_runs) if n_runs > 0 else 0.0
        points.append(
            ParetoPoint(
                threshold=thr,
                precision=precision,
                abstention_rate=abstention_rate,
                accepted=accepted,
                abstained=abstained,
                true_positives=true_positives,
                false_positives=false_positives,
            )
        )
    return ParetoFrontier(
        points=points,
        n_runs=n_runs,
        n_solved_total=n_solved_total,
    )


def _result_benchmark_id(result: Any) -> Optional[str]:
    """Best-effort extraction of the benchmark id from a result-like object.

    Looks at, in order:
      * ``result.benchmark_id`` (string attr)
      * ``result.diagnostics["benchmark_id"]``
      * ``result.diagnostics["benchmark_name"]``
      * ``result["benchmark_id"]`` / ``result["benchmark_name"]`` for dicts

    Returns ``None`` when no benchmark identifier is present.
    """
    direct = getattr(result, "benchmark_id", None)
    if isinstance(direct, str) and direct.strip():
        return direct
    diagnostics = getattr(result, "diagnostics", None)
    if isinstance(diagnostics, dict):
        for key in ("benchmark_id", "benchmark_name"):
            value = diagnostics.get(key)
            if isinstance(value, str) and value.strip():
                return value
    if isinstance(result, dict):
        for key in ("benchmark_id", "benchmark_name"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value
        diag = result.get("diagnostics")
        if isinstance(diag, dict):
            for key in ("benchmark_id", "benchmark_name"):
                value = diag.get(key)
                if isinstance(value, str) and value.strip():
                    return value
    return None


def compute_pareto_frontier_per_benchmark(
    results: Sequence[Any],
    *,
    thresholds: Optional[Iterable[float]] = None,
    confidence_extractor: Optional[Any] = None,
    ground_truth_extractor: Optional[Any] = None,
    benchmark_extractor: Optional[Any] = None,
    normalize_benchmark_ids: bool = True,
) -> dict[str, ParetoFrontier]:
    """Group ``results`` by benchmark_id and run :func:`compute_pareto_frontier`
    on each group.

    The returned dict is keyed by the (optionally normalized) benchmark
    id. Results without a benchmark id land under the empty-string
    bucket so callers can still see the un-tagged subset rather than
    silently dropping it.

    Different benchmarks have different confidence-score distributions
    (testgeneval skews high because the gold oracle is exact; commit0
    skews lower because cluster_consensus is noisier on full-repo
    rewrites). Sweeping the threshold per-benchmark lets operators read
    the optimal threshold straight off each curve rather than averaging
    them into one curve that's optimal for nothing.
    """
    extract_benchmark = benchmark_extractor or _result_benchmark_id
    grouped: dict[str, list[Any]] = {}
    for r in results:
        bench = extract_benchmark(r) or ""
        if normalize_benchmark_ids and bench:
            bench = _normalize_benchmark_id(bench) or bench
        grouped.setdefault(bench, []).append(r)
    out: dict[str, ParetoFrontier] = {}
    for bench, bench_results in grouped.items():
        out[bench] = compute_pareto_frontier(
            bench_results,
            thresholds=thresholds,
            confidence_extractor=confidence_extractor,
            ground_truth_extractor=ground_truth_extractor,
        )
    return out


def emit_pareto_artifacts(
    frontier: ParetoFrontier,
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write ``pareto_frontier.json`` and ``PARETO_REPORT.md`` to *output_dir*.

    Returns a mapping ``{"json": <path>, "markdown": <path>}`` with the
    absolute paths of the emitted files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "pareto_frontier.json"
    md_path = out / "PARETO_REPORT.md"

    json_path.write_text(json.dumps(frontier.to_dict(), indent=2, sort_keys=True))

    md_lines: list[str] = []
    md_lines.append("# Calibrated Abstention — Pareto Frontier")
    md_lines.append("")
    md_lines.append(
        f"**Runs scored:** {frontier.n_runs}  **Ground-truth solved:** {frontier.n_solved_total}"
    )
    md_lines.append("")
    md_lines.append("| threshold | precision | abstention_rate | accepted | abstained | TP | FP |")
    md_lines.append("|----------:|----------:|----------------:|---------:|----------:|---:|---:|")
    for point in frontier.points:
        md_lines.append(
            "| {thr:.2f} | {prec:.3f} | {abst:.3f} | {acc} | {ab} | {tp} | {fp} |".format(
                thr=point.threshold,
                prec=point.precision,
                abst=point.abstention_rate,
                acc=point.accepted,
                ab=point.abstained,
                tp=point.true_positives,
                fp=point.false_positives,
            )
        )
    md_lines.append("")
    md_path.write_text("\n".join(md_lines) + "\n")
    return {"json": str(json_path), "markdown": str(md_path)}
