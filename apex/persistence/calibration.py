"""
Calibration analysis for APEX scoring components.

Two NeurIPS-relevant questions about a coding-agent orchestrator:

1. **Are the verification / critic / cluster scores actually calibrated
   to acceptance probability?** A high cluster ``combined_score`` should
   imply a high empirical acceptance rate. If it doesn't, the multi-stage
   selector is making confident wrong choices.
2. **Are calibrated controller policies improving over the heuristic
   baseline?** The controller logs both ``baseline_value`` and the
   blended ``value`` for every model evaluation. We can directly compare
   their reliability.

This module walks a directory of past APEX runs (``apex_result.json``
files) and computes:

- per-bin reliability tables for any ``(predicted_score, observed_label)``
  pair the caller registers,
- a Brier score, and
- the expected calibration error (ECE).

It is deliberately dependency-free (no ``numpy``) so the analysis can run
on any host that already has APEX installed. For NeurIPS tables we want
deterministic, reproducible numbers — Python ``statistics`` is enough.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger("apex.persistence.calibration")


@dataclass
class CalibrationBin:
    """One reliability-table bin."""

    bin_index: int
    lower: float
    upper: float
    count: int = 0
    mean_predicted: float = 0.0
    mean_observed: float = 0.0
    sum_predicted: float = 0.0
    sum_observed: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "bin_index": int(self.bin_index),
            "lower": round(float(self.lower), 4),
            "upper": round(float(self.upper), 4),
            "count": int(self.count),
            "mean_predicted": round(float(self.mean_predicted), 6),
            "mean_observed": round(float(self.mean_observed), 6),
            "abs_gap": round(abs(float(self.mean_predicted) - float(self.mean_observed)), 6),
        }


@dataclass
class CalibrationReport:
    """Aggregate calibration metrics for one (score, label) extractor."""

    score_name: str
    label_name: str
    sample_count: int = 0
    brier_score: float = 0.0
    expected_calibration_error: float = 0.0
    base_rate: float = 0.0
    mean_predicted: float = 0.0
    bin_count: int = 0
    bins: list[CalibrationBin] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_name": self.score_name,
            "label_name": self.label_name,
            "sample_count": int(self.sample_count),
            "brier_score": round(float(self.brier_score), 6),
            "expected_calibration_error": round(float(self.expected_calibration_error), 6),
            "base_rate": round(float(self.base_rate), 6),
            "mean_predicted": round(float(self.mean_predicted), 6),
            "bin_count": int(self.bin_count),
            "bins": [item.to_dict() for item in self.bins],
        }


@dataclass
class CalibrationDataset:
    """Raw (predicted_score, observed_label) pairs collected from runs."""

    score_name: str
    label_name: str
    samples: list[tuple[float, float]] = field(default_factory=list)

    def add(self, predicted: float, observed: float) -> None:
        self.samples.append((float(predicted), float(observed)))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _equal_width_bin_index(value: float, *, bin_count: int) -> int:
    if bin_count <= 0:
        return 0
    clamped = _clamp01(value)
    if clamped >= 1.0:
        return bin_count - 1
    return min(bin_count - 1, int(clamped * bin_count))


def _build_report(
    dataset: CalibrationDataset,
    *,
    bin_count: int = 10,
) -> CalibrationReport:
    samples = list(dataset.samples)
    sample_count = len(samples)
    if sample_count == 0:
        return CalibrationReport(
            score_name=dataset.score_name,
            label_name=dataset.label_name,
            bin_count=bin_count,
        )
    brier_total = 0.0
    sum_predicted = 0.0
    sum_observed = 0.0
    bins: dict[int, CalibrationBin] = {}
    for predicted, observed in samples:
        clamped_predicted = _clamp01(predicted)
        clamped_observed = _clamp01(observed)
        brier_total += (clamped_predicted - clamped_observed) ** 2
        sum_predicted += clamped_predicted
        sum_observed += clamped_observed
        index = _equal_width_bin_index(clamped_predicted, bin_count=bin_count)
        if index not in bins:
            bins[index] = CalibrationBin(
                bin_index=index,
                lower=index / bin_count,
                upper=min(1.0, (index + 1) / bin_count),
            )
        bin_record = bins[index]
        bin_record.count += 1
        bin_record.sum_predicted += clamped_predicted
        bin_record.sum_observed += clamped_observed

    ece_total = 0.0
    ordered_bins: list[CalibrationBin] = []
    for index in range(bin_count):
        record = bins.get(index)
        if record is None:
            ordered_bins.append(
                CalibrationBin(
                    bin_index=index,
                    lower=index / bin_count,
                    upper=min(1.0, (index + 1) / bin_count),
                )
            )
            continue
        record.mean_predicted = record.sum_predicted / max(record.count, 1)
        record.mean_observed = record.sum_observed / max(record.count, 1)
        ece_total += (record.count / sample_count) * abs(
            record.mean_predicted - record.mean_observed
        )
        ordered_bins.append(record)

    return CalibrationReport(
        score_name=dataset.score_name,
        label_name=dataset.label_name,
        sample_count=sample_count,
        brier_score=brier_total / sample_count,
        expected_calibration_error=ece_total,
        base_rate=sum_observed / sample_count,
        mean_predicted=sum_predicted / sample_count,
        bin_count=bin_count,
        bins=ordered_bins,
    )


def _walk_apex_result_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.name == "apex_result.json":
            yield root
        return
    if not root.is_dir():
        return
    for path in root.rglob("apex_result.json"):
        if path.is_file():
            yield path


SampleExtractor = Callable[[dict[str, Any]], list[tuple[float, float]]]


def cluster_combined_vs_acceptance(payload: dict[str, Any]) -> list[tuple[float, float]]:
    """Pair each rollout's verification ``overall_score`` with acceptance.

    The selector's ``combined_score`` blends size, verification, and the
    critic; verification.overall_score is the cleanest published proxy
    for "would this be accepted." Acceptance is the result-level
    ``success`` value (one row per *winning* rollout) — for non-winning
    rollouts we use the per-rollout verification's ``accepted`` field
    when available.
    """

    samples: list[tuple[float, float]] = []
    rollouts = payload.get("rollout_summaries") or []
    if not isinstance(rollouts, list):
        return samples
    selected_id = payload.get("selected_rollout_id")
    success = bool(payload.get("success"))
    for rollout in rollouts:
        if not isinstance(rollout, dict):
            continue
        verification = rollout.get("verification") or {}
        if not isinstance(verification, dict):
            continue
        score = verification.get("overall_score")
        if not isinstance(score, (int, float)):
            continue
        accepted = verification.get("accepted")
        if accepted is None and selected_id is not None:
            accepted = success and rollout.get("rollout_id") == selected_id
        samples.append((float(score), 1.0 if bool(accepted) else 0.0))
    return samples


def critic_score_vs_acceptance(payload: dict[str, Any]) -> list[tuple[float, float]]:
    """Pair each rollout's selection-critic score with acceptance."""

    samples: list[tuple[float, float]] = []
    rollouts = payload.get("rollout_summaries") or []
    if not isinstance(rollouts, list):
        return samples
    selected_id = payload.get("selected_rollout_id")
    success = bool(payload.get("success"))
    for rollout in rollouts:
        if not isinstance(rollout, dict):
            continue
        diagnostics = rollout.get("selection_diagnostics") or {}
        critic = diagnostics.get("critic") if isinstance(diagnostics, dict) else None
        if not isinstance(critic, dict):
            continue
        score = critic.get("score")
        if not isinstance(score, (int, float)):
            continue
        verification = rollout.get("verification") or {}
        accepted = verification.get("accepted") if isinstance(verification, dict) else None
        if accepted is None and selected_id is not None:
            accepted = success and rollout.get("rollout_id") == selected_id
        samples.append((float(score), 1.0 if bool(accepted) else 0.0))
    return samples


_DEFAULT_EXTRACTORS: dict[tuple[str, str], SampleExtractor] = {
    ("verification_overall_score", "rollout_accepted"): cluster_combined_vs_acceptance,
    ("selection_critic_score", "rollout_accepted"): critic_score_vs_acceptance,
}


def collect_calibration_datasets(
    root: Path,
    *,
    extractors: Optional[dict[tuple[str, str], SampleExtractor]] = None,
) -> list[CalibrationDataset]:
    """Walk ``root`` for apex_result.json files and assemble datasets."""

    chosen_extractors = dict(extractors or _DEFAULT_EXTRACTORS)
    datasets = {
        key: CalibrationDataset(score_name=key[0], label_name=key[1]) for key in chosen_extractors
    }
    for path in _walk_apex_result_files(root):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("Failed to load %s: %s", path, exc)
            continue
        if not isinstance(payload, dict):
            continue
        for key, extractor in chosen_extractors.items():
            try:
                pairs = extractor(payload)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Extractor %r failed on %s: %s", key, path, exc)
                pairs = []
            dataset = datasets[key]
            for predicted, observed in pairs:
                dataset.add(predicted, observed)
    return list(datasets.values())


def build_calibration_reports(
    root: Path,
    *,
    bin_count: int = 10,
    extractors: Optional[dict[tuple[str, str], SampleExtractor]] = None,
) -> list[CalibrationReport]:
    datasets = collect_calibration_datasets(root, extractors=extractors)
    return [_build_report(dataset, bin_count=bin_count) for dataset in datasets]


def render_reliability_markdown(report: CalibrationReport) -> str:
    """Render a Markdown reliability table for the report."""

    lines = [
        f"## Calibration: {report.score_name} → {report.label_name}",
        "",
        f"- Samples: {report.sample_count}",
        f"- Base rate: {report.base_rate:.3f}",
        f"- Mean predicted: {report.mean_predicted:.3f}",
        f"- Brier score: {report.brier_score:.4f}",
        f"- Expected calibration error: {report.expected_calibration_error:.4f}",
        "",
        "| bin | range | count | mean_predicted | mean_observed | abs_gap |",
        "| --- | --- | ---:| ---: | ---: | ---: |",
    ]
    for bin_record in report.bins:
        lines.append(
            "| {idx} | [{lower:.2f}, {upper:.2f}) | {count} | {mp:.3f} | {mo:.3f} | {gap:.3f} |".format(
                idx=bin_record.bin_index,
                lower=bin_record.lower,
                upper=bin_record.upper,
                count=bin_record.count,
                mp=bin_record.mean_predicted,
                mo=bin_record.mean_observed,
                gap=abs(bin_record.mean_predicted - bin_record.mean_observed),
            )
        )
    return "\n".join(lines)
