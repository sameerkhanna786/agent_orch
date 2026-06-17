"""Execution-Grounded Trajectory Critic + auto-labeled step PRM (EG-Critic).

A learned, execution-grounded value head for candidate selection and (offline)
for the frontier search prior. Two ideas, both grounded in published work, fused
on a benchmark whose oracle is authoritative:

* **Execution-grounded trajectory critic** — a calibrated model that scores a
  whole candidate trajectory from execution-grounded features (local full-scope
  pass, fail->pass strength, expected-coverage ratio, mutation kills, process
  quality, backend-anomaly penalty). Trained offline on APEX's own harvested
  ``(trajectory -> gold F2P pass/fail)`` outcomes — the same TD-critic idea as
  OpenHands' critic-32b, but on a gold oracle.

* **Math-Shepherd-style auto-labeled step PRM** — propagate the terminal gold
  reward back over a trajectory's steps with a discount (no human step labels),
  giving a per-step value that can later seed the PUCT prior in
  ``frontier_search`` (AlphaZero-style policy/value heads).

CRITICAL SAFETY CONTRACT (mirrors the design-review open risks):

* The critic **never overrides execution evidence**. It only re-ranks among
  candidates that are *already execution-tied* (e.g. all full-scope passing or
  all near-miss at the same tier). It can break ties and order the search
  frontier; it can never promote an unverified candidate above a verified one.
* It is **offline-first and default-off**: the untrained critic returns the
  heuristic floor score, so enabling the module without a fitted model is a
  no-op. Live wiring into ``frontier_search``/``SelectionCritic`` is gated behind
  a ``component_ablation`` arm so the published score cannot silently move.

This module is pure Python (no heavy ML deps): the "model" is a transparent,
calibrated linear head whose weights are either the hand-set prior (the floor) or
fitted offline by least-squares on harvested outcomes. That keeps it auditable,
fast, and trivially testable, while leaving the interface ready for a heavier
learned head later.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

# Execution-grounded feature names, in a fixed order. All are in [0, 1] (penalty
# features are stored as their complement so "higher is better" holds uniformly).
FEATURE_NAMES: tuple[str, ...] = (
    "full_scope_pass",
    "fail_to_pass_strength",
    "expected_coverage_ratio",
    "mutation_kill_ratio",
    "process_quality",
    "no_backend_anomaly",
)

# Hand-set prior weights (the floor). Execution-tied signals dominate; soft
# signals (process quality) contribute least. Normalized to sum to 1.0.
_PRIOR_WEIGHTS: dict[str, float] = {
    "full_scope_pass": 0.40,
    "fail_to_pass_strength": 0.25,
    "expected_coverage_ratio": 0.15,
    "mutation_kill_ratio": 0.10,
    "process_quality": 0.05,
    "no_backend_anomaly": 0.05,
}


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def extract_execution_features(candidate: Any, verification: Any = None) -> dict[str, float]:
    """Map a candidate + its verification into the execution-grounded feature
    vector. Tolerant of dict or attribute access and of missing fields (absent
    signals read as 0.0, i.e. no evidence)."""
    quick = _as_dict(getattr(candidate, "quick_verification", None))
    verification = verification if verification is not None else getattr(
        candidate, "verification", None
    )
    vdict = _as_dict(verification if isinstance(verification, dict) else getattr(verification, "__dict__", None))

    def _qv(key: str) -> Any:
        return quick.get(key)

    passed = _qv("passed")
    failed = _qv("failed")
    errors = _qv("errors")
    full_scope = 0.0
    if str(_qv("scope") or "") == "full_test_command":
        rc = _qv("returncode")
        if (rc in (0, None)) and (failed in (0, None)) and (errors in (0, None)):
            pr = _qv("pass_rate")
            full_scope = 1.0 if (pr is None or _clamp01(pr) >= 0.999) else _clamp01(pr)

    f2p = _qv("fail_to_pass_strength")
    if f2p is None:
        # Derive a coarse F->P strength from passed/total when not provided.
        total = sum(int(v) for v in (passed, failed, errors) if isinstance(v, int))
        f2p = (int(passed) / total) if (isinstance(passed, int) and total > 0) else 0.0

    coverage = _qv("expected_coverage_ratio")
    if coverage is None:
        coverage = _qv("coverage_ratio")

    mutation = vdict.get("mutation_kill_ratio")
    if mutation is None:
        mutation = getattr(candidate, "mutation_kill_ratio", None)

    process = _as_dict(getattr(candidate, "selection_diagnostics", None)).get("process_quality")
    process_score = _as_dict(process).get("score") if process is not None else None

    backend_anomaly = bool(_as_dict(getattr(candidate, "search_metadata", None)).get("backend_anomaly"))

    return {
        "full_scope_pass": _clamp01(full_scope),
        "fail_to_pass_strength": _clamp01(f2p),
        "expected_coverage_ratio": _clamp01(coverage),
        "mutation_kill_ratio": _clamp01(mutation),
        "process_quality": _clamp01(process_score) if process_score is not None else 0.5,
        "no_backend_anomaly": 0.0 if backend_anomaly else 1.0,
    }


@dataclass
class ExecutionGroundedCritic:
    """Calibrated execution-grounded value head. Untrained == heuristic floor."""

    weights: dict[str, float] = field(default_factory=lambda: dict(_PRIOR_WEIGHTS))
    bias: float = 0.0
    fitted: bool = False

    def score_features(self, features: dict[str, float]) -> float:
        total = self.bias
        for name in FEATURE_NAMES:
            total += float(self.weights.get(name, 0.0)) * _clamp01(features.get(name, 0.0))
        return _clamp01(total)

    def score_candidate(self, candidate: Any, verification: Any = None) -> float:
        return self.score_features(extract_execution_features(candidate, verification))

    def rank_among_tied(self, candidates: Sequence[Any]) -> list[Any]:
        """Stable re-ranking of an *execution-tied* candidate set by learned
        value (descending). Callers must only pass candidates that already share
        the same execution tier — the critic breaks ties, it does not cross tiers.
        """
        scored = [
            (self.score_candidate(candidate), index, candidate)
            for index, candidate in enumerate(candidates)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [candidate for _, _, candidate in scored]

    # ---------------------------------------------------------------- #
    # Offline fitting (least-squares ridge on harvested outcomes)
    # ---------------------------------------------------------------- #
    def fit(
        self,
        samples: Sequence[tuple[dict[str, float], float]],
        *,
        l2: float = 1e-3,
        learning_rate: float = 0.5,
        epochs: int = 400,
    ) -> "ExecutionGroundedCritic":
        """Fit weights from ``(features, gold_label)`` pairs (label in [0,1],
        typically 1.0 if the trajectory's patch passed the gold F2P oracle else
        0.0). Uses simple ridge-regularized gradient descent so there is no numpy
        dependency. Falls back to the prior weights if there are no samples."""
        rows = [(dict(f), _clamp01(y)) for f, y in samples if isinstance(f, dict)]
        if not rows:
            return self
        weights = {name: 0.0 for name in FEATURE_NAMES}
        bias = 0.0
        n = len(rows)
        for _ in range(max(1, int(epochs))):
            grad = {name: 0.0 for name in FEATURE_NAMES}
            grad_bias = 0.0
            for features, label in rows:
                pred = bias + sum(
                    weights[name] * _clamp01(features.get(name, 0.0)) for name in FEATURE_NAMES
                )
                err = pred - label
                for name in FEATURE_NAMES:
                    grad[name] += err * _clamp01(features.get(name, 0.0))
                grad_bias += err
            for name in FEATURE_NAMES:
                weights[name] -= learning_rate * (grad[name] / n + l2 * weights[name])
            bias -= learning_rate * (grad_bias / n)
        self.weights = weights
        self.bias = bias
        self.fitted = True
        return self

    def selection_accuracy(
        self,
        groups: Sequence[Sequence[tuple[dict[str, float], float]]],
    ) -> float:
        """Offline metric: fraction of execution-tied groups in which the critic's
        top-scored member is a gold-passing candidate. This is the falsifiable
        signal — it must beat the heuristic floor on a held-out split before the
        learned head is allowed to move live rankings."""
        if not groups:
            return 0.0
        correct = 0
        evaluated = 0
        for group in groups:
            rows = [(dict(f), _clamp01(y)) for f, y in group if isinstance(f, dict)]
            if not rows or not any(y >= 0.5 for _, y in rows):
                continue
            evaluated += 1
            best = max(rows, key=lambda row: self.score_features(row[0]))
            if best[1] >= 0.5:
                correct += 1
        return (correct / evaluated) if evaluated else 0.0

    # ---------------------------------------------------------------- #
    # Persistence (WS2C): ship the offline-fit head as a JSON artifact
    # ---------------------------------------------------------------- #
    def to_payload(self) -> dict[str, Any]:
        return {
            "_schema_version": "1",
            "fitted": bool(self.fitted),
            "bias": float(self.bias),
            "weights": {name: float(self.weights.get(name, 0.0)) for name in FEATURE_NAMES},
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ExecutionGroundedCritic":
        if not isinstance(payload, dict):
            return cls()
        weights = dict(_PRIOR_WEIGHTS)
        raw_weights = payload.get("weights")
        if isinstance(raw_weights, dict):
            for name in FEATURE_NAMES:
                if name in raw_weights:
                    try:
                        weights[name] = float(raw_weights[name])
                    except (TypeError, ValueError):
                        pass
        try:
            bias = float(payload.get("bias") or 0.0)
        except (TypeError, ValueError):
            bias = 0.0
        return cls(weights=weights, bias=bias, fitted=bool(payload.get("fitted")))


def _default_eg_critic_weights_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "eg_critic_weights.json"


def save_eg_critic(critic: ExecutionGroundedCritic, path: "str | Path") -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(critic.to_payload(), indent=2), encoding="utf-8")
    return destination


def load_eg_critic(path: "str | Path | None" = None) -> Optional[ExecutionGroundedCritic]:
    """Load a persisted EG-critic. Returns ``None`` when the artifact is missing
    or unreadable (so the live wiring stays a no-op). A loaded-but-unfitted
    artifact yields a floor critic whose ``.fitted`` is False — callers must
    check ``.fitted`` before letting it move rankings."""
    resolved = Path(path) if path else _default_eg_critic_weights_path()
    try:
        if not resolved.exists():
            return None
        data = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return ExecutionGroundedCritic.from_payload(data)
    except (OSError, ValueError):
        logger.debug("load_eg_critic failed for %s", resolved, exc_info=True)
        return None


def auto_label_trajectory_steps(
    num_steps: int,
    terminal_reward: float,
    *,
    gamma: float = 0.99,
) -> list[float]:
    """Math-Shepherd-style automatic step labels: propagate the terminal gold
    reward back over ``num_steps`` with discount ``gamma`` (TD return). No human
    step annotation required; the gold F2P oracle supplies the terminal reward.

    Returns one value per step, earliest first. The last step carries the full
    terminal reward; earlier steps are discounted toward it.
    """
    num_steps = max(0, int(num_steps))
    reward = _clamp01(terminal_reward)
    if num_steps == 0:
        return []
    # Step i (0-indexed) gets gamma^(num_steps-1-i) * reward.
    return [_clamp01((gamma ** (num_steps - 1 - i)) * reward) for i in range(num_steps)]
