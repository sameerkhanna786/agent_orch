"""APEX fairness-audit module.

This module implements a *side-by-side scorer audit* so that APEX benchmark
runs can be reported with both the APEX-private scoring (which today uses
several private shortcuts -- a private exit-code rewrite for Commit0,
log-marker re-aggregation for TestGenEval, and a TestGenEval container
harness patch) AND the upstream-canonical scoring (the scoring that the
benchmark authors publish results with). The goal is to produce a per-task
``FairnessDelta`` so we can publish the deltas openly instead of obscuring
them.

Per ``BENCHMARK_FAIRNESS_CHECKLIST.md`` the *harness* owns scoring policy.
This module is the harness-level audit hook: it is intentionally
orchestrator-agnostic and contains no benchmark-private metadata.

Design notes
------------
* This module is Phase 0: it builds the data model, aggregator, and the
  ``run_fairness_audit`` driver. It does *not* yet wire into the benchmark
  runners (Commit0BenchmarkRunner, TestGenEvalBenchmarkRunner,
  SWTBenchBenchmarkRunner). That is Phase 1.
* The Phase 1 wiring will define concrete scorers
  ``Commit0PrivateScorer``, ``Commit0UpstreamScorer``,
  ``TestGenEvalPrivateScorer``, ``TestGenEvalUpstreamScorer``, and
  ``SWTBenchUpstreamScorer`` that conform to ``BenchmarkScorerProtocol``.
  SWT-Bench has only one scoring path (the upstream Docker harness) and
  therefore only one scorer; ``run_fairness_audit`` MAY be invoked with
  ``private_scorer is upstream_scorer`` in that case and the resulting
  delta will be all zeros, which is the correct, honest outcome.
* The two scorers passed to ``run_fairness_audit`` MUST share the same
  Docker registry / image versions so the delta isolates *scoring*
  differences and not *image* differences. Enforcement of that constraint
  belongs to Phase 1 wiring -- here we only document it and provide the
  ``shared_image_digest`` field on ``FairnessDelta`` so it can be recorded.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------


class FairnessAuditMode(str, Enum):
    """How the harness should treat scoring during a benchmark run.

    Members
    -------
    OFF
        Business as usual. Only the APEX-private scorer runs; no audit
        artifact is emitted. This is the default for production runs that
        are not being published.
    PARALLEL
        Run *both* scorers on every task, compute per-metric deltas, and
        emit ``fairness_delta.json`` / ``FAIRNESS_REPORT.md``. The
        APEX-private number remains the headline number, but the delta is
        published alongside it.
    UPSTREAM_ONLY
        Only the upstream-canonical scorer is treated as authoritative.
        The APEX-private scorer may still run but its output is recorded
        as diagnostic-only and never used for the headline number.
    """

    OFF = "off"
    PARALLEL = "parallel"
    UPSTREAM_ONLY = "upstream_only"


# ---------------------------------------------------------------------------
# Sentinel values
# ---------------------------------------------------------------------------


#: Marker recorded in a per-metric delta when one scorer reported the
#: metric and the other did not (or vice versa). We cannot meaningfully
#: subtract numerics in that case, so we record NaN and additionally append
#: a "scorer disagreement" note so it is visible in the markdown report.
SCORER_DISAGREEMENT_DELTA = float("nan")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class FairnessDelta:
    """Per-task fairness audit record.

    Attributes
    ----------
    task_id
        Stable identifier for the task (e.g. Commit0 instance_id).
    apex_private_scores
        The dict returned by the APEX-private scorer. Numeric fields will
        participate in the delta computation. Non-numeric fields (e.g.
        ``"exitcode_used": "report.exitcode"``) are preserved verbatim and
        will be reported but not differenced.
    upstream_canonical_scores
        The dict returned by the upstream-canonical scorer.
    delta
        Per-metric numeric delta computed as
        ``apex_private - upstream_canonical`` for every key that is
        numeric in *both* dicts. Keys that exist in only one side are
        recorded with value :data:`SCORER_DISAGREEMENT_DELTA` (NaN).
    notes
        Free-form notes about why the two scorers diverged on this task
        (e.g. "APEX-private exit-code rewrite triggered: report=0,
        shell=1"). Phase 1 wiring will populate these from scorer
        diagnostics.
    shared_image_digest
        Digest / tag of the Docker image used for *both* scorings. This
        is the audit anchor proving the delta isolates the scoring
        difference and not an image difference. May be ``None`` for unit
        tests or for benchmarks that do not use a Docker harness.
    """

    task_id: str
    apex_private_scores: dict[str, Any] = field(default_factory=dict)
    upstream_canonical_scores: dict[str, Any] = field(default_factory=dict)
    delta: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    shared_image_digest: str | None = None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (NaN -> ``"NaN"`` string).

        We use the string sentinel ``"NaN"`` because vanilla
        ``json.dumps`` emits the literal ``NaN`` token which is *not*
        valid JSON per RFC 8259 and will be rejected by strict parsers.
        """
        payload = asdict(self)
        payload["delta"] = {
            key: ("NaN" if isinstance(value, float) and math.isnan(value) else value)
            for key, value in self.delta.items()
        }
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FairnessDelta":
        """Inverse of :meth:`to_dict`. Restores NaN sentinels."""
        raw_delta = dict(payload.get("delta", {}))
        delta: dict[str, float] = {}
        for key, value in raw_delta.items():
            if isinstance(value, str) and value == "NaN":
                delta[key] = SCORER_DISAGREEMENT_DELTA
            else:
                delta[key] = float(value)
        return cls(
            task_id=payload["task_id"],
            apex_private_scores=dict(payload.get("apex_private_scores", {})),
            upstream_canonical_scores=dict(payload.get("upstream_canonical_scores", {})),
            delta=delta,
            notes=list(payload.get("notes", [])),
            shared_image_digest=payload.get("shared_image_digest"),
        )

    @classmethod
    def from_json(cls, blob: str) -> "FairnessDelta":
        return cls.from_dict(json.loads(blob))


# ---------------------------------------------------------------------------
# Scorer protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BenchmarkScorerProtocol(Protocol):
    """Protocol every benchmark scorer must implement.

    A scorer takes a ``task`` description (the same task object the
    benchmark runner already knows about, e.g. a ``Commit0Task``) and the
    ``apex_artifacts`` produced by the orchestrator for that task (patch
    file, generated test files, container logs, etc.) and returns a flat
    mapping of metric name to value. Numeric values participate in delta
    computation; non-numeric values are recorded for provenance but not
    differenced.

    Phase 1 wiring will define the following concrete scorers conforming
    to this Protocol:

    * ``Commit0PrivateScorer``      -- uses the APEX private exit-code
      rewrite path and report.exitcode aggregation.
    * ``Commit0UpstreamScorer``     -- uses the canonical
      ``commit0 evaluate`` shell exit code only.
    * ``TestGenEvalPrivateScorer``  -- uses APEX log-marker re-aggregation
      and the APEX TestGenEval container harness patch.
    * ``TestGenEvalUpstreamScorer`` -- uses the canonical TestGenEval
      grader output unmodified.
    * ``SWTBenchUpstreamScorer``    -- the only SWT-Bench scorer; the
      benchmark has no APEX-private equivalent. Audit runs for SWT-Bench
      should pass this scorer as both ``private_scorer`` and
      ``upstream_scorer``; the resulting delta will legitimately be all
      zeros.
    """

    def score_task(
        self, task: Any, apex_artifacts: Any
    ) -> dict[str, Any]:  # pragma: no cover - Protocol
        ...


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _is_numeric(value: Any) -> bool:
    """Return True iff ``value`` is a finite numeric we can subtract.

    Booleans are intentionally excluded -- ``True - False`` works in
    Python but it almost always indicates a categorical metric rather
    than a numeric one, and silently coercing it produces misleading
    deltas.
    """
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value))


def _compute_delta(
    apex_private_scores: dict[str, Any],
    upstream_canonical_scores: dict[str, Any],
) -> tuple[dict[str, float], list[str]]:
    """Return ``(delta_map, disagreement_notes)``.

    A "disagreement" is recorded whenever:

    * A metric key appears on one side but not the other.
    * A metric key appears on both sides but only one side is numeric.
    """
    notes: list[str] = []
    delta: dict[str, float] = {}

    all_keys = set(apex_private_scores) | set(upstream_canonical_scores)
    for key in sorted(all_keys):
        in_apex = key in apex_private_scores
        in_upstream = key in upstream_canonical_scores
        apex_value = apex_private_scores.get(key)
        upstream_value = upstream_canonical_scores.get(key)

        if not in_apex:
            delta[key] = SCORER_DISAGREEMENT_DELTA
            notes.append(
                f"scorer disagreement on metric '{key}': missing from "
                f"apex-private; upstream={upstream_value!r}"
            )
            continue
        if not in_upstream:
            delta[key] = SCORER_DISAGREEMENT_DELTA
            notes.append(
                f"scorer disagreement on metric '{key}': missing from "
                f"upstream-canonical; apex-private={apex_value!r}"
            )
            continue

        apex_numeric = _is_numeric(apex_value)
        upstream_numeric = _is_numeric(upstream_value)

        if apex_numeric and upstream_numeric:
            delta[key] = float(apex_value) - float(upstream_value)
            continue

        if apex_numeric != upstream_numeric:
            delta[key] = SCORER_DISAGREEMENT_DELTA
            notes.append(
                f"scorer disagreement on metric '{key}': "
                f"apex-private={apex_value!r} ({type(apex_value).__name__}), "
                f"upstream-canonical={upstream_value!r} "
                f"({type(upstream_value).__name__}); cannot subtract"
            )
            continue

        # Both non-numeric. Don't put it in delta but flag if they differ.
        if apex_value != upstream_value:
            notes.append(
                f"non-numeric metric '{key}' differs: "
                f"apex-private={apex_value!r}, upstream-canonical={upstream_value!r}"
            )

    return delta, notes


def run_fairness_audit(
    task: Any,
    apex_artifacts: Any,
    private_scorer: BenchmarkScorerProtocol,
    upstream_scorer: BenchmarkScorerProtocol,
    *,
    shared_image_digest: str | None = None,
    extra_notes: Iterable[str] | None = None,
) -> FairnessDelta:
    """Run both scorers on the same task and return the per-task delta.

    Parameters
    ----------
    task
        The benchmark task object. Passed through opaquely to both
        scorers.
    apex_artifacts
        The orchestrator artifacts (patch, generated tests, container
        logs, etc.) to score. Passed through opaquely to both scorers.
    private_scorer
        Scorer using the APEX-private rules.
    upstream_scorer
        Scorer using the upstream-canonical rules.
    shared_image_digest
        The digest/tag of the Docker image both scorers are expected to
        run inside. **Phase 1 wiring is responsible for ensuring this is
        actually shared**; this function only records it for audit.
    extra_notes
        Optional caller-provided notes to attach (e.g. "TestGenEval
        harness patch active").

    Returns
    -------
    FairnessDelta
        Combined record. The returned object's ``task_id`` is taken from
        ``task.task_id`` if present, else ``task.instance_id`` if
        present, else ``str(task)``.
    """
    apex_scores = dict(private_scorer.score_task(task, apex_artifacts))
    upstream_scores = dict(upstream_scorer.score_task(task, apex_artifacts))
    delta, disagreement_notes = _compute_delta(apex_scores, upstream_scores)

    task_id = getattr(task, "task_id", None) or getattr(task, "instance_id", None) or str(task)

    notes: list[str] = []
    if extra_notes:
        notes.extend(str(note) for note in extra_notes)
    notes.extend(disagreement_notes)

    return FairnessDelta(
        task_id=str(task_id),
        apex_private_scores=apex_scores,
        upstream_canonical_scores=upstream_scores,
        delta=delta,
        notes=notes,
        shared_image_digest=shared_image_digest,
    )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


#: A task is "flagged" when any single numeric metric's |delta| exceeds
#: 2 percentage points (0.02 in fractional form, or 2.0 in
#: percentage-point form). The aggregator treats both representations the
#: same way -- it just tests ``abs(delta) > FLAG_THRESHOLD`` -- so callers
#: must pick a consistent representation in their scorers.
FLAG_THRESHOLD = 0.02


class FairnessAuditAggregator:
    """Accumulate per-task ``FairnessDelta`` records and summarize.

    The aggregator is deliberately small: the Phase 1 benchmark runners
    will own one of these per benchmark, call :meth:`add_task` for each
    completed task, and call :meth:`write_to` once at the end of the run.
    """

    def __init__(self) -> None:
        self._tasks: list[FairnessDelta] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_task(self, delta: FairnessDelta) -> None:
        """Record one task's delta. Order is preserved for reporting."""
        if not isinstance(delta, FairnessDelta):
            raise TypeError(
                f"FairnessAuditAggregator.add_task expected FairnessDelta, "
                f"got {type(delta).__name__}"
            )
        self._tasks.append(delta)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @property
    def tasks(self) -> list[FairnessDelta]:
        """Return a shallow copy of the recorded tasks."""
        return list(self._tasks)

    def __len__(self) -> int:
        return len(self._tasks)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Aggregate stats over all recorded tasks.

        Returns a dict with:

        * ``num_tasks`` -- total tasks recorded.
        * ``num_flagged_tasks`` -- count of tasks where |delta| >
          :data:`FLAG_THRESHOLD` on at least one numeric metric.
        * ``num_disagreement_tasks`` -- count of tasks where any metric
          could not be differenced (NaN delta).
        * ``per_metric`` -- per-metric stats: ``mean_delta``,
          ``max_abs_delta``, ``num_tasks_with_metric``,
          ``num_disagreements``.
        * ``flagged_task_ids`` -- list of task_ids in the flagged set,
          in insertion order.
        """
        per_metric: dict[str, dict[str, Any]] = {}
        flagged_task_ids: list[str] = []
        num_disagreement_tasks = 0

        for record in self._tasks:
            task_flagged = False
            task_has_disagreement = False
            for metric, value in record.delta.items():
                bucket = per_metric.setdefault(
                    metric,
                    {
                        "values": [],  # numeric only; NaN excluded
                        "num_tasks_with_metric": 0,
                        "num_disagreements": 0,
                    },
                )
                bucket["num_tasks_with_metric"] += 1
                if isinstance(value, float) and math.isnan(value):
                    bucket["num_disagreements"] += 1
                    task_has_disagreement = True
                    continue
                bucket["values"].append(float(value))
                if abs(float(value)) > FLAG_THRESHOLD:
                    task_flagged = True

            if task_flagged:
                flagged_task_ids.append(record.task_id)
            if task_has_disagreement:
                num_disagreement_tasks += 1

        # Reduce
        reduced: dict[str, dict[str, Any]] = {}
        for metric, bucket in per_metric.items():
            values = bucket["values"]
            if values:
                mean_delta = sum(values) / len(values)
                max_abs_delta = max(abs(v) for v in values)
            else:
                mean_delta = SCORER_DISAGREEMENT_DELTA
                max_abs_delta = SCORER_DISAGREEMENT_DELTA
            reduced[metric] = {
                "mean_delta": mean_delta,
                "max_abs_delta": max_abs_delta,
                "num_tasks_with_metric": bucket["num_tasks_with_metric"],
                "num_disagreements": bucket["num_disagreements"],
            }

        return {
            "num_tasks": len(self._tasks),
            "num_flagged_tasks": len(flagged_task_ids),
            "num_disagreement_tasks": num_disagreement_tasks,
            "flag_threshold": FLAG_THRESHOLD,
            "per_metric": reduced,
            "flagged_task_ids": flagged_task_ids,
        }

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def write_to(self, path: Path) -> dict[str, Path]:
        """Write ``fairness_audit.json`` and ``FAIRNESS_REPORT.md``.

        ``path`` is the *directory* to write into. It will be created if
        missing. Returns a dict mapping artifact name to absolute path.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        json_path = path / "fairness_audit.json"
        md_path = path / "FAIRNESS_REPORT.md"

        json_payload = {
            "summary": _jsonify(self.summary()),
            "tasks": [task.to_dict() for task in self._tasks],
        }
        json_path.write_text(json.dumps(json_payload, sort_keys=True, indent=2))
        md_path.write_text(self._render_markdown())

        return {"json": json_path.resolve(), "markdown": md_path.resolve()}

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def _render_markdown(self) -> str:
        summary = self.summary()
        lines: list[str] = []
        lines.append("# APEX Fairness Audit Report")
        lines.append("")
        lines.append(
            "This report shows per-metric deltas between the APEX-private "
            "scorer and the upstream-canonical scorer. A non-zero delta "
            "means the two scoring paths disagreed on this task. Per "
            "`BENCHMARK_FAIRNESS_CHECKLIST.md` the upstream-canonical "
            "number is the publishable number; the delta is the audit "
            "trail."
        )
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append(f"* Total tasks audited: {summary['num_tasks']}")
        lines.append(
            f"* Flagged tasks (|delta| > {summary['flag_threshold']:g} "
            f"on any metric): {summary['num_flagged_tasks']}"
        )
        lines.append(
            f"* Tasks with scorer disagreement (NaN delta on at least "
            f"one metric): {summary['num_disagreement_tasks']}"
        )
        lines.append("")

        lines.append("## Per-metric deltas")
        lines.append("")
        per_metric = summary["per_metric"]
        if not per_metric:
            lines.append("_No metrics recorded._")
        else:
            lines.append("| metric | mean delta | max |delta| | tasks | disagreements |")
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for metric in sorted(per_metric):
                stats = per_metric[metric]
                mean_str = _fmt_float(stats["mean_delta"])
                max_str = _fmt_float(stats["max_abs_delta"])
                lines.append(
                    f"| `{metric}` | {mean_str} | {max_str} | "
                    f"{stats['num_tasks_with_metric']} | "
                    f"{stats['num_disagreements']} |"
                )
        lines.append("")

        if summary["flagged_task_ids"]:
            lines.append("## Flagged tasks")
            lines.append("")
            for tid in summary["flagged_task_ids"]:
                lines.append(f"* `{tid}`")
            lines.append("")

        lines.append("## Per-task detail")
        lines.append("")
        if not self._tasks:
            lines.append("_No tasks recorded._")
        for record in self._tasks:
            lines.append(f"### `{record.task_id}`")
            lines.append("")
            if record.shared_image_digest:
                lines.append(f"* Shared image digest: `{record.shared_image_digest}`")
            lines.append(
                f"* APEX-private scores: `{json.dumps(record.apex_private_scores, sort_keys=True)}`"
            )
            lines.append(
                f"* Upstream-canonical scores: `{json.dumps(record.upstream_canonical_scores, sort_keys=True)}`"
            )
            if record.delta:
                lines.append("* Delta:")
                for metric in sorted(record.delta):
                    lines.append(f"    * `{metric}`: {_fmt_float(record.delta[metric])}")
            else:
                lines.append("* Delta: _empty_")
            if record.notes:
                lines.append("* Notes:")
                for note in record.notes:
                    lines.append(f"    * {note}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_float(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return f"{value:+.4f}"


def _jsonify(obj: Any) -> Any:
    """Replace NaN floats with the string ``"NaN"`` recursively."""
    if isinstance(obj, float) and math.isnan(obj):
        return "NaN"
    if isinstance(obj, dict):
        return {key: _jsonify(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(value) for value in obj]
    return obj


__all__ = [
    "BenchmarkScorerProtocol",
    "FairnessAuditAggregator",
    "FairnessAuditMode",
    "FairnessDelta",
    "FLAG_THRESHOLD",
    "SCORER_DISAGREEMENT_DELTA",
    "run_fairness_audit",
]
