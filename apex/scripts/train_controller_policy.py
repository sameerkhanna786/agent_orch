#!/usr/bin/env python3
"""Train calibrated controller policies for the four task regimes.

The trainer prefers historical ``controller_decisions.jsonl`` traces when they
exist on disk: it joins each ``policy_evaluation`` entry that targets a
``regime.<state>`` model to the run's ``task_result.json`` overall_score and
fits a per-state logistic regression.  When no historical traces are available
(common during early development) the trainer falls back to a hand-crafted
synthetic dataset that mirrors the heuristic baseline; the resulting weights
ship as a calibrated approximation that can be improved as live data arrives.

Outputs one ``<regime>.json`` per state into ``--output``.  Each bundle stores
its weights, calibrated transform parameters, validation ECE, and a
``policy_version`` tag so consumers can tell synthetic-fit weights from
production-fit ones.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

# Allow ``python apex/scripts/train_controller_policy.py`` invocations.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apex.controller_models import (  # noqa: E402  (sys.path shim)
    calibrated_weights_dir,
    reset_calibrated_library_cache,
)
from apex.controller_policy import TASK_REGIME_STATES  # noqa: E402

# Per-state synthetic feature priors. Each entry is the regime name mapped to
# its ground-truth feature signature: features that should drive the
# probability up (positive coefficient) and features that should pull it down
# (negative coefficient).  The synthetic generator samples around these priors
# so the resulting logistic model approximates the heuristic adapter signals.
_SYNTHETIC_PRIORS: dict[str, dict[str, dict[str, float]]] = {
    "importability_blocker": {
        "positive": {
            "obs__zero_passing_with_traceback": 1.0,
            "obs__terminal_source_focus": 0.6,
            "obs__collection_error_cluster": 0.9,
            "terminal_source_count": 0.5,
            "failing_collection_file_count": 0.5,
            "evidence_count": 0.4,
            "heuristic_score": 1.4,
        },
        "negative": {
            "passing_test_count": 0.5,
            "obs__incomplete_source_scaffold": 0.4,
            "obs__failing_test_breadth": 0.3,
        },
    },
    "contract_gap": {
        "positive": {
            "obs__completion_pattern": 1.0,
            "obs__incomplete_source_scaffold": 0.9,
            "obs__incomplete_test_scaffold": 0.6,
            "obs__public_api_contract": 0.6,
            "incomplete_source_count": 0.5,
            "incomplete_test_count": 0.4,
            "evidence_count": 0.3,
            "heuristic_score": 1.4,
        },
        "negative": {
            "obs__zero_passing_with_traceback": 0.5,
            "obs__failing_test_breadth": 0.3,
        },
    },
    "broad_regression": {
        "positive": {
            "obs__failing_test_breadth": 1.0,
            "obs__mixed_pass_fail_surface": 0.6,
            "obs__relevant_file_breadth": 0.6,
            "obs__coverage_preservation_invariant": 0.5,
            "failing_test_count": 0.4,
            "relevant_file_count": 0.3,
            "evidence_count": 0.3,
            "heuristic_score": 1.4,
        },
        "negative": {
            "obs__terminal_source_focus": 0.4,
            "obs__completion_pattern": 0.3,
        },
    },
    "high_interface_risk": {
        "positive": {
            "obs__interface_symbol_signal": 1.0,
            "obs__multi_module_focus": 0.7,
            "obs__public_api_pattern": 0.8,
            "interface_symbol_count": 0.5,
            "source_focus_count": 0.4,
            "evidence_count": 0.3,
            "heuristic_score": 1.4,
        },
        "negative": {
            "obs__zero_passing_with_traceback": 0.3,
        },
    },
}


@dataclass
class TrainingExample:
    """One labelled (features, label) pair for a specific regime state."""

    regime: str
    features: dict[str, float] = field(default_factory=dict)
    label: float = 0.0
    source: str = ""  # "historical" or "synthetic"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FittedRegimeModel:
    """Result of fitting + calibrating one regime model."""

    regime: str
    feature_names: list[str]
    weights: list[float]
    intercept: float
    calibration: dict[str, Any]
    training_metadata: dict[str, Any]
    policy_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "regime": self.regime,
            "feature_names": list(self.feature_names),
            "weights": [float(w) for w in self.weights],
            "intercept": float(self.intercept),
            "transform": "logistic",
            "blend": 1.0,
            "lower": 0.0,
            "upper": 1.0,
            "calibration": dict(self.calibration),
            "training_metadata": dict(self.training_metadata),
        }


# ---------------------------------------------------------------------------
# Historical trace harvesting
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            out.append(payload)
    return out


def discover_trace_paths(roots: Iterable[str]) -> list[Path]:
    """Locate every controller_decisions.jsonl-style trace under the inputs."""

    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        for matched in glob.glob(root, recursive=True):
            base = Path(matched)
            if base.is_file() and base.name.endswith(".jsonl"):
                resolved = base.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    out.append(resolved)
                continue
            if not base.is_dir():
                continue
            for child in base.rglob("controller_decisions.jsonl"):
                resolved = child.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    out.append(resolved)
    return out


def _join_outcome(trace_path: Path) -> float:
    """Best-effort join of the trace to the run's task outcome score."""

    candidates = [
        trace_path.parent / "task_result.json",
        trace_path.parent / "apex_result.json",
        trace_path.parent.parent / "task_result.json",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        final = payload.get("final") if isinstance(payload, dict) else None
        if isinstance(final, dict):
            for key in ("pass_rate", "required_pass_rate", "score"):
                value = final.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
        if isinstance(payload, dict):
            for key in ("overall_score", "score"):
                value = payload.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
            if isinstance(payload.get("success"), bool):
                return 1.0 if payload.get("success") else 0.0
    return 0.0


def harvest_historical_examples(trace_paths: Iterable[Path]) -> list[TrainingExample]:
    """Walk traces and emit (regime, features, label) examples."""

    examples: list[TrainingExample] = []
    for path in trace_paths:
        outcome = _join_outcome(path)
        records = _load_jsonl(path)
        for record in records:
            evaluations = record.get("policy_evaluations") or []
            for entry in evaluations:
                if not isinstance(entry, dict):
                    continue
                model_name = str(entry.get("model_name") or "").strip()
                if not model_name.startswith("regime."):
                    continue
                regime = model_name.split(".", 1)[1]
                if regime not in TASK_REGIME_STATES:
                    continue
                features_payload = entry.get("features") or {}
                if not isinstance(features_payload, dict):
                    continue
                features = {
                    str(name): float(value)
                    for name, value in features_payload.items()
                    if isinstance(value, (int, float))
                }
                # Treat the model's own raw output as the soft label fallback.
                label_value = (
                    1.0
                    if outcome >= 0.5 and float(features.get(f"model_target__{regime}", 0.0)) >= 0.5
                    else 0.0
                )
                examples.append(
                    TrainingExample(
                        regime=regime,
                        features=features,
                        label=label_value,
                        source="historical",
                        metadata={
                            "trace_path": str(path),
                            "outcome": outcome,
                            "policy_version": str(
                                entry.get("metadata", {}).get("policy_version") or ""
                            ),
                        },
                    )
                )
    return examples


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------


def _sample_synthetic_example(
    regime: str,
    *,
    rng: random.Random,
    is_positive: bool,
) -> TrainingExample:
    prior = _SYNTHETIC_PRIORS[regime]
    features: dict[str, float] = {}
    # Draw observation-level features.
    for name, scale in prior["positive"].items():
        if is_positive:
            value = abs(rng.gauss(scale, 0.25))
        else:
            value = abs(rng.gauss(0.0, 0.15))
        features[name] = round(value, 4)
    for name, scale in prior["negative"].items():
        if is_positive:
            value = abs(rng.gauss(0.0, 0.15))
        else:
            value = abs(rng.gauss(scale, 0.25))
        features[name] = round(value, 4)
    # Counters get a coarse mapping from observation strength.
    counters = {
        "failing_test_count": ("obs__failing_test_breadth", 4.0),
        "passing_test_count": ("obs__failing_test_breadth", 0.0),
        "terminal_source_count": ("obs__terminal_source_focus", 1.0),
        "source_focus_count": ("obs__multi_module_focus", 2.0),
        "incomplete_source_count": ("obs__incomplete_source_scaffold", 1.0),
        "incomplete_test_count": ("obs__incomplete_test_scaffold", 1.0),
        "relevant_file_count": ("obs__relevant_file_breadth", 4.0),
        "interface_symbol_count": ("obs__interface_symbol_signal", 1.0),
        "exception_count": ("obs__zero_passing_with_traceback", 1.0),
        "failing_collection_file_count": ("obs__collection_error_cluster", 2.0),
    }
    for counter_name, (driver, base) in counters.items():
        driver_value = float(features.get(driver, 0.0))
        if driver_value > 0.0:
            features[counter_name] = round(base + driver_value * 2.0 + rng.uniform(0.0, 1.5), 2)
        else:
            features[counter_name] = round(rng.uniform(0.0, 1.0), 2)
    # Adapter / language toggles.
    features["adapter_is_python_pytest"] = 1.0 if rng.random() < 0.55 else 0.0
    features["adapter_is_generic"] = 1.0 - features["adapter_is_python_pytest"]
    features["python_pytest_command"] = features["adapter_is_python_pytest"]
    features["preserve_collected_test_coverage"] = 1.0 if rng.random() < 0.35 else 0.0
    features["has_test_command"] = 1.0
    features[f"state__{regime}"] = round(
        sum(value for name, value in features.items() if name.startswith("obs__")),
        4,
    )
    features[f"model_target__{regime}"] = 1.0
    # Heuristic baseline approximates a clamped sum of the regime-relevant signals.
    heuristic_score = max(
        0.0,
        min(
            1.0,
            features[f"state__{regime}"] / 2.5
            + (0.15 if is_positive else -0.05)
            + rng.uniform(-0.05, 0.05),
        ),
    )
    features["heuristic_score"] = round(heuristic_score, 4)
    features["evidence_count"] = round(
        sum(1.0 for name in prior["positive"] if features.get(name, 0.0) > 0.0),
        2,
    )
    label = 1.0 if is_positive else 0.0
    return TrainingExample(
        regime=regime,
        features=features,
        label=label,
        source="synthetic",
    )


def build_synthetic_dataset(
    *,
    examples_per_regime: int = 320,
    seed: int = 17,
) -> list[TrainingExample]:
    rng = random.Random(seed)
    out: list[TrainingExample] = []
    for regime in TASK_REGIME_STATES:
        # 50/50 positive/negative class balance, plus small drift per regime to
        # avoid identical decision boundaries.
        for _ in range(examples_per_regime // 2):
            out.append(_sample_synthetic_example(regime, rng=rng, is_positive=True))
        for _ in range(examples_per_regime // 2):
            out.append(_sample_synthetic_example(regime, rng=rng, is_positive=False))
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Logistic regression fit (numpy only, gradient descent with L2)
# ---------------------------------------------------------------------------


def _gather_feature_names(examples: list[TrainingExample]) -> list[str]:
    names: set[str] = set()
    for example in examples:
        for name in example.features.keys():
            if not name:
                continue
            # Skip per-regime target indicators (used for joining only).
            if name.startswith("model_target__"):
                continue
            names.add(name)
    return sorted(names)


def _example_matrix(
    examples: list[TrainingExample],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(examples), len(feature_names)), dtype=np.float64)
    labels = np.zeros(len(examples), dtype=np.float64)
    index_lookup = {name: idx for idx, name in enumerate(feature_names)}
    for row_idx, example in enumerate(examples):
        for name, value in example.features.items():
            col_idx = index_lookup.get(name)
            if col_idx is None:
                continue
            matrix[row_idx, col_idx] = float(value)
        labels[row_idx] = float(example.label)
    return matrix, labels


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    learning_rate: float = 0.1,
    iterations: int = 1500,
    l2: float = 0.01,
) -> tuple[np.ndarray, float]:
    n_features = features.shape[1]
    weights = np.zeros(n_features, dtype=np.float64)
    intercept = 0.0
    n = max(1, features.shape[0])
    for _ in range(iterations):
        logits = features @ weights + intercept
        predictions = _sigmoid(logits)
        error = predictions - labels
        grad_weights = (features.T @ error) / n + l2 * weights
        grad_intercept = float(np.mean(error))
        weights -= learning_rate * grad_weights
        intercept -= learning_rate * grad_intercept
    return weights, intercept


def _platt_scale(
    raw_logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Platt scaling: fit a 1-D logistic regression on the raw logits."""

    weights, intercept = _fit_logistic_regression(
        raw_logits.reshape(-1, 1),
        labels,
        learning_rate=0.1,
        iterations=1500,
        l2=0.01,
    )
    return float(weights[0]), float(intercept)


def _expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    if probabilities.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(probabilities.size)
    ece = 0.0
    for i in range(bins):
        lower = edges[i]
        upper = edges[i + 1]
        if i == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        bin_size = float(np.sum(mask))
        if bin_size == 0:
            continue
        bin_conf = float(np.mean(probabilities[mask]))
        bin_acc = float(np.mean(labels[mask]))
        ece += (bin_size / total) * abs(bin_conf - bin_acc)
    return float(ece)


def _train_val_split(
    examples: list[TrainingExample],
    *,
    val_ratio: float = 0.25,
    seed: int = 13,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    cut = max(1, int(round(len(shuffled) * (1.0 - val_ratio))))
    return shuffled[:cut], shuffled[cut:]


def fit_regime_model(
    examples: list[TrainingExample],
    *,
    regime: str,
    policy_version: str,
    val_ratio: float = 0.25,
    seed: int = 13,
) -> FittedRegimeModel:
    regime_examples = [item for item in examples if item.regime == regime]
    if not regime_examples:
        raise ValueError(f"no training examples for regime={regime}")
    train, val = _train_val_split(regime_examples, val_ratio=val_ratio, seed=seed)
    if not val:
        val = train[: max(1, len(train) // 4)]

    feature_names = _gather_feature_names(regime_examples)
    train_x, train_y = _example_matrix(train, feature_names)
    val_x, val_y = _example_matrix(val, feature_names)

    weights, intercept = _fit_logistic_regression(train_x, train_y)

    val_logits = val_x @ weights + intercept
    val_probs = _sigmoid(val_logits)
    pre_ece = _expected_calibration_error(val_probs, val_y)
    platt_a, platt_b = _platt_scale(val_logits, val_y)
    # Compose Platt scaling into the linear model so inference stays a single dot product.
    composed_weights = weights * platt_a
    composed_intercept = intercept * platt_a + platt_b
    calibrated_probs = _sigmoid(val_x @ composed_weights + composed_intercept)
    post_ece = _expected_calibration_error(calibrated_probs, val_y)

    train_logits = train_x @ composed_weights + composed_intercept
    train_probs = _sigmoid(train_logits)
    train_acc = float(np.mean((train_probs >= 0.5) == (train_y >= 0.5)))
    val_acc = float(np.mean((calibrated_probs >= 0.5) == (val_y >= 0.5)))

    metadata = {
        "n_examples": int(len(regime_examples)),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "synthetic": all(item.source == "synthetic" for item in regime_examples),
        "ece": round(post_ece, 4),
        "ece_uncalibrated": round(pre_ece, 4),
        "train_accuracy": round(train_acc, 4),
        "val_accuracy": round(val_acc, 4),
        "positive_fraction": round(float(np.mean(train_y)), 4),
    }
    calibration = {
        "method": "platt",
        "params": {"a": round(float(platt_a), 6), "b": round(float(platt_b), 6)},
    }
    return FittedRegimeModel(
        regime=regime,
        feature_names=feature_names,
        weights=[float(w) for w in composed_weights],
        intercept=float(composed_intercept),
        calibration=calibration,
        training_metadata=metadata,
        policy_version=policy_version,
    )


def write_regime_bundle(
    model: FittedRegimeModel,
    *,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{model.regime}.json"
    target.write_text(
        json.dumps(model.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _summarize_dataset(examples: list[TrainingExample]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_examples": len(examples),
        "by_regime": {},
        "synthetic_examples": sum(1 for item in examples if item.source == "synthetic"),
        "historical_examples": sum(1 for item in examples if item.source == "historical"),
    }
    for regime in TASK_REGIME_STATES:
        regime_examples = [item for item in examples if item.regime == regime]
        positives = sum(1 for item in regime_examples if item.label >= 0.5)
        summary["by_regime"][regime] = {
            "count": len(regime_examples),
            "positives": positives,
            "negatives": len(regime_examples) - positives,
        }
    return summary


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train calibrated controller policy weights.")
    parser.add_argument(
        "--traces",
        action="append",
        default=[],
        help=(
            "Directory or glob pattern containing controller_decisions.jsonl files. "
            "Repeat to provide multiple sources."
        ),
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Always include synthetic examples (used as the only source if no traces are found).",
    )
    parser.add_argument(
        "--examples-per-regime",
        type=int,
        default=320,
        help="Number of synthetic examples to draw per regime (default: 320).",
    )
    parser.add_argument(
        "--policy-version",
        default="",
        help="Override the policy_version stamp on the resulting bundles.",
    )
    parser.add_argument(
        "--output",
        default=str(calibrated_weights_dir()),
        help="Output directory for the per-regime weight bundles.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for synthetic data + train/val splits (default: 17).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.25,
        help="Validation split ratio (default: 0.25).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print the dataset summary and recommended split without fitting models.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output).resolve()
    historical_examples: list[TrainingExample] = []
    trace_paths: list[Path] = []
    if args.traces:
        trace_paths = discover_trace_paths(args.traces)
        historical_examples = harvest_historical_examples(trace_paths)
    use_synthetic = bool(args.synthetic or not historical_examples)
    synthetic_examples: list[TrainingExample] = []
    if use_synthetic:
        synthetic_examples = build_synthetic_dataset(
            examples_per_regime=int(args.examples_per_regime),
            seed=int(args.seed),
        )
    examples = historical_examples + synthetic_examples
    summary = _summarize_dataset(examples)
    summary["trace_paths"] = [str(path) for path in trace_paths]
    summary["recommended_val_ratio"] = float(args.val_ratio)

    if not args.quiet:
        print(json.dumps({"dataset_summary": summary}, indent=2, sort_keys=True))

    if args.summary_only:
        return {"dataset_summary": summary, "models": {}}

    policy_version = args.policy_version or (
        "calibrated-v1-synthetic" if not historical_examples else "calibrated-v1"
    )
    fitted_models: dict[str, FittedRegimeModel] = {}
    for regime in TASK_REGIME_STATES:
        regime_examples = [item for item in examples if item.regime == regime]
        if not regime_examples:
            continue
        fitted = fit_regime_model(
            examples,
            regime=regime,
            policy_version=policy_version,
            val_ratio=float(args.val_ratio),
            seed=int(args.seed),
        )
        write_regime_bundle(fitted, output_dir=output_dir)
        fitted_models[regime] = fitted
        if not args.quiet:
            md = fitted.training_metadata
            print(
                f"  regime={regime} "
                f"n={md['n_examples']} "
                f"train={md['n_train']} val={md['n_val']} "
                f"ece={md['ece']:.4f} acc={md['val_accuracy']:.3f}"
            )

    # Drop any cached library so subsequent in-process consumers see fresh weights.
    reset_calibrated_library_cache()

    return {
        "dataset_summary": summary,
        "models": {regime: model.to_dict() for regime, model in fitted_models.items()},
        "policy_version": policy_version,
        "output_dir": str(output_dir),
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    payload = run(args)
    if not args.quiet and not args.summary_only:
        # Keep the final summary machine-readable for downstream tooling.
        print(
            json.dumps(
                {
                    "policy_version": payload.get("policy_version"),
                    "output_dir": payload.get("output_dir"),
                    "ece_per_regime": {
                        regime: model["training_metadata"]["ece"]
                        for regime, model in payload.get("models", {}).items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
