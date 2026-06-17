"""TestGenEval upstream-canonical scorer.

Implements :class:`apex.core.fairness_audit.BenchmarkScorerProtocol` for
the TestGenEval scoring path that uses the *unpatched* upstream harness
plus only the defensive ``baseline_covs`` KeyError fix (so
``generate_report.py`` doesn't crash on lite-subset rows). The headline
``pass@1`` value is read from the upstream
``official_reports/<model_name>_full.json`` artifact.

This scorer is the ``upstream_scorer`` half of the fairness-audit pair.
The ``private_scorer`` half is :class:`TestGenEvalPrivateScorer` in
``testgeneval_private.py``.

Inputs to :meth:`TestGenEvalUpstreamScorer.score_task`:

* ``task``: anything with an ``instance_id`` / ``task_id`` attribute.
* ``apex_artifacts``: ignored except for an optional
  ``"official_reports_dir"`` override.

Outputs (flat metric mapping; numeric fields participate in the
fairness delta):

* ``pass_at_1`` (float in {0.0, 1.0}): from the upstream
  ``full_pass_at_1`` field (canonical TestGenEval headline).
* ``pass_at_1_unfiltered`` (float): from
  ``full_unfiltered_pass_at_1`` if present, else 0.
* ``mutation_score`` (float): from ``full_av_mutation_score``.
* ``coverage`` (float): from ``full_av_coverage`` (raw upstream
  number, not improvement-over-baseline).
* ``status`` (str, non-numeric): coarse status derived from pass@1.
* ``score_source`` (str, non-numeric): always
  ``"upstream_canonical_full_json"``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("apex.scorers.testgeneval_upstream")


@dataclass
class TestGenEvalUpstreamScorer:
    """Score a TestGenEval task with the upstream-canonical rules."""

    # Tell pytest this is not a test class despite the ``Test`` prefix.
    __test__ = False

    log_dir: Path
    model_name: str
    dataset_name: str = "kjain14/testgenevallite"
    split: str = "test"
    official_reports_dir: Path | None = None
    name: str = "testgeneval_upstream"

    # Cache the loaded full report so we only parse the JSON once per
    # audit. The cache is reset whenever the resolved path changes.
    _cached_report: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _cached_report_path: Path | None = field(default=None, init=False, repr=False)

    def score_task(self, task: Any, apex_artifacts: Any) -> dict[str, Any]:
        instance_id = _resolve_instance_id(task)
        report = self._load_full_report(apex_artifacts)
        if report is None:
            return _no_report_payload("upstream_full_report_missing")
        instance_payload = (
            report.get(instance_id) or report.get("results", {}).get(instance_id) or {}
        )
        if not isinstance(instance_payload, dict) or not instance_payload:
            return _no_report_payload("upstream_instance_missing", instance_id)
        # The upstream report nests the metrics under per-setting keys
        # ("full", "first", "last", "extra"). The "full" setting is the
        # canonical one for TestGenEval pass@1 / mutation / coverage.
        return {
            "pass_at_1": _coerce_pass_at_1(instance_payload, "full"),
            "pass_at_1_unfiltered": _coerce_unfiltered_pass_at_1(instance_payload, "full"),
            "mutation_score": _coerce_metric(instance_payload, "full_av_mutation_score"),
            "coverage": _coerce_metric(instance_payload, "full_av_coverage"),
            "status": _coarse_status(instance_payload, "full"),
            "score_source": "upstream_canonical_full_json",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_full_report(self, apex_artifacts: Any) -> dict[str, Any] | None:
        path = self._resolve_full_report_path(apex_artifacts)
        if path is None:
            return None
        if self._cached_report is not None and self._cached_report_path == path:
            return self._cached_report
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("upstream full report unreadable %s: %s", path, exc)
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("upstream full report not JSON %s: %s", path, exc)
            return None
        if not isinstance(payload, dict):
            return None
        self._cached_report = payload
        self._cached_report_path = path
        return payload

    def _resolve_full_report_path(self, apex_artifacts: Any) -> Path | None:
        candidate_dir = self.official_reports_dir
        if isinstance(apex_artifacts, dict):
            override = apex_artifacts.get("official_reports_dir")
            if override:
                candidate_dir = Path(override)
        if candidate_dir is None:
            return None
        exact = candidate_dir / f"{self.model_name}_full.json"
        if exact.exists():
            return exact
        # Fall back to any ``*_full.json`` if the model_name suffix
        # differs from what the harness wrote.
        candidates = sorted(candidate_dir.glob("*_full.json"))
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            logger.warning(
                "ambiguous upstream full reports in %s: %s",
                candidate_dir,
                ", ".join(p.name for p in candidates[:8]),
            )
        return None


def _resolve_instance_id(task: Any) -> str:
    if hasattr(task, "instance_id"):
        return str(getattr(task, "instance_id"))
    if hasattr(task, "task_id"):
        return str(getattr(task, "task_id"))
    if isinstance(task, dict):
        return str(task.get("instance_id") or task.get("task_id") or task.get("id") or "")
    return str(task)


def _coerce_pass_at_1(payload: dict[str, Any], setting: str) -> float:
    """Read the canonical pass@1 field; default 0 if absent."""
    for key in (f"{setting}_pass_at_1", f"{setting}_avg_pass_at_1"):
        if key in payload:
            return _to_float(payload[key])
    return 0.0


def _coerce_unfiltered_pass_at_1(payload: dict[str, Any], setting: str) -> float:
    for key in (
        f"{setting}_unfiltered_pass_at_1",
        f"{setting}_unfiltered_avg_pass_at_1",
    ):
        if key in payload:
            return _to_float(payload[key])
    return 0.0


def _coerce_metric(payload: dict[str, Any], key: str) -> float:
    if key in payload:
        return _to_float(payload[key])
    return -1.0


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _coarse_status(payload: dict[str, Any], setting: str) -> str:
    p = _coerce_pass_at_1(payload, setting)
    if p > 0.0:
        return "filtered_pass"
    return "failed"


def _no_report_payload(reason: str, instance_id: str = "") -> dict[str, Any]:
    return {
        "pass_at_1": 0.0,
        "pass_at_1_unfiltered": 0.0,
        "mutation_score": -1.0,
        "coverage": -1.0,
        "status": "no_log",
        "score_source": reason if not instance_id else f"{reason}:{instance_id}",
    }


def make_testgeneval_upstream_scorer(
    *,
    log_dir: Path,
    model_name: str,
    dataset_name: str = "kjain14/testgenevallite",
    split: str = "test",
    official_reports_dir: Path | None = None,
) -> TestGenEvalUpstreamScorer:
    """Factory mirroring the other benchmark scorer factories."""
    return TestGenEvalUpstreamScorer(
        log_dir=Path(log_dir),
        model_name=model_name,
        dataset_name=dataset_name,
        split=split,
        official_reports_dir=Path(official_reports_dir) if official_reports_dir else None,
    )


__all__ = [
    "TestGenEvalUpstreamScorer",
    "make_testgeneval_upstream_scorer",
]
