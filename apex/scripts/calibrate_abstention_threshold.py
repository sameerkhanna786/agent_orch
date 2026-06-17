#!/usr/bin/env python3
"""Phase A.5 / Prep for C.2 — per-benchmark abstention threshold calibrator.

The orchestrator's calibrated abstention scorer
(:class:`apex.orchestration.abstention.ConfidenceScorer`) maps each task
to a single confidence value in ``[0, 1]``. The current default
threshold is the global ``OrchestrationConfig.abstention_threshold =
0.50``. That number was chosen literature-blind; per-benchmark distributions
of confidence scores differ enough that a single threshold leaves real
F1 on the table.

This script:

  1. Walks a benchmark sweep root and joins each task's
     ``apex_result.json`` confidence score (overall) with the task's
     ground-truth outcome (``status == "solved"`` post-hoc, or
     ``officially_accepted`` when present).
  2. Buckets tasks by benchmark (commit0 / swt_bench / testgeneval / ...).
  3. For each benchmark, sweeps a threshold over ``[0.0, 1.0]`` in
     ``--threshold-step`` increments (default 0.05) and computes the F1
     score of the "accept" decision (precision = TP / (TP + FP), recall =
     TP / (TP + FN)).
  4. Picks the threshold maximising F1 — ties broken by higher precision
     then by lower threshold so the chosen value is the most conservative
     among the optima.
  5. Writes ``apex/configs/abstention_thresholds_per_benchmark.json``::

         {
           "commit0": 0.40,
           "swt_bench": 0.45,
           "testgeneval": 0.65,
           "_metadata": {
             "calibration_run": "...",
             "n_tasks_per_benchmark": {"commit0": 30, ...},
             "f1_per_benchmark": {"commit0": 0.83, ...},
             "threshold_step": 0.05
           }
         }

The calibration is intentionally diagnostic-only: it never silently
edits ``OrchestrationConfig`` defaults. Operators wire the per-benchmark
table into ``BenchmarkConfig.abstention_threshold_override`` (or copy
the ``commit0`` entry into ``orchestration.abstention_threshold``)
themselves so the change shows up in the run manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class TaskRecord:
    """One per-task (benchmark, confidence, ground-truth) tuple."""

    benchmark: str
    task_id: str
    confidence: float
    solved: bool
    source_path: Path


_BENCHMARK_DIR_ALIASES = {
    "commit0_lite": "commit0",
    "commit0": "commit0",
    "swtbench_lite": "swt_bench",
    "swtbench": "swt_bench",
    "swt_bench_lite": "swt_bench",
    "testgeneval_lite": "testgeneval",
    "testgeneval": "testgeneval",
    "swebench_pro": "swebench_pro",
    "swe_evo": "swe_evo",
}


def _normalize_benchmark(name: str) -> str:
    text = (name or "").strip().lower()
    if not text:
        return ""
    return _BENCHMARK_DIR_ALIASES.get(text, text)


def _infer_benchmark(result_path: Path) -> str:
    skip = {"apex_output", "outputs", "output", "run", "."}
    parent = result_path.parent
    while parent.name in skip and parent.parent != parent:
        parent = parent.parent
    benchmark_dir = parent.parent
    return _normalize_benchmark(benchmark_dir.name)


def _infer_task_id(result_path: Path) -> str:
    skip = {"apex_output", "outputs", "output", "run", "."}
    parent = result_path.parent
    while parent.name in skip and parent.parent != parent:
        parent = parent.parent
    return parent.name


def _confidence_overall(payload: dict[str, Any]) -> Optional[float]:
    cb = payload.get("confidence")
    if isinstance(cb, dict):
        for key in ("overall", "score"):
            value = cb.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    # Some runners stash the value at the top level when there is no
    # full breakdown (fast fallback path).
    for key in ("confidence_overall", "confidence_score"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _solved_label(payload: dict[str, Any]) -> bool:
    """Ground-truth: did the task actually solve?

    Preference: ``officially_accepted`` (upstream harness verdict) >
    ``status == "solved"`` > ``success`` boolean.
    """
    official = payload.get("officially_accepted")
    if isinstance(official, bool):
        return official
    status = payload.get("status")
    if hasattr(status, "value"):
        return getattr(status, "value") == "solved"
    if isinstance(status, str):
        return status == "solved"
    return bool(payload.get("success"))


def _iter_apex_results(root: Path) -> Iterable[Path]:
    if not root.exists():
        return iter(())
    if root.is_file() and root.name == "apex_result.json":
        return iter([root])
    return root.rglob("apex_result.json")


def load_task_records(root: Path) -> list[TaskRecord]:
    out: list[TaskRecord] = []
    seen: set[Path] = set()
    for result_path in _iter_apex_results(root):
        resolved = result_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        confidence = _confidence_overall(payload)
        if confidence is None:
            # Skip results missing a confidence score; they predate Phase
            # 6.3 wiring (or the orchestrator bailed early).
            continue
        out.append(
            TaskRecord(
                benchmark=_infer_benchmark(result_path),
                task_id=_infer_task_id(result_path),
                confidence=float(confidence),
                solved=_solved_label(payload),
                source_path=resolved,
            )
        )
    return out


# ---------------------------------------------------------------------------
# F1 sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdPoint:
    threshold: float
    accepted: int
    abstained: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": round(self.threshold, 6),
            "accepted": int(self.accepted),
            "abstained": int(self.abstained),
            "true_positives": int(self.true_positives),
            "false_positives": int(self.false_positives),
            "false_negatives": int(self.false_negatives),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
        }


def _evaluate_threshold(
    records: list[TaskRecord],
    threshold: float,
) -> ThresholdPoint:
    accepted = 0
    tp = 0
    fp = 0
    fn = 0
    for record in records:
        if record.confidence >= threshold:
            accepted += 1
            if record.solved:
                tp += 1
            else:
                fp += 1
        elif record.solved:
            # Abstained but the task was solvable — false negative.
            fn += 1
    abstained = len(records) - accepted
    denom_p = tp + fp
    precision = (tp / denom_p) if denom_p > 0 else 1.0
    denom_r = tp + fn
    recall = (tp / denom_r) if denom_r > 0 else 0.0
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return ThresholdPoint(
        threshold=float(threshold),
        accepted=accepted,
        abstained=abstained,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _threshold_grid(step: float) -> list[float]:
    if step <= 0.0 or step > 1.0:
        raise ValueError(f"threshold step out of range: {step!r}")
    n_steps = int(round(1.0 / step))
    return [round(i * step, 6) for i in range(0, n_steps + 1)]


def calibrate_one_benchmark(
    records: list[TaskRecord],
    *,
    step: float = 0.05,
) -> tuple[float, ThresholdPoint, list[ThresholdPoint]]:
    """Pick the F1-maximising threshold; ties broken by precision then by
    lowest threshold (most conservative)."""
    if not records:
        # Degenerate: no data → fall back to the legacy default.
        fallback = ThresholdPoint(
            threshold=0.50,
            accepted=0,
            abstained=0,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            precision=1.0,
            recall=0.0,
            f1=0.0,
        )
        return 0.50, fallback, [fallback]

    sweep = [_evaluate_threshold(records, t) for t in _threshold_grid(step)]

    def sort_key(point: ThresholdPoint) -> tuple[float, float, float]:
        # Maximise F1, then precision; minimise threshold.
        return (-point.f1, -point.precision, point.threshold)

    best = min(sweep, key=sort_key)
    return float(best.threshold), best, sweep


def calibrate_thresholds(
    records: Iterable[TaskRecord],
    *,
    step: float = 0.05,
) -> dict[str, Any]:
    by_benchmark: dict[str, list[TaskRecord]] = {}
    for record in records:
        if not record.benchmark:
            continue
        by_benchmark.setdefault(record.benchmark, []).append(record)

    thresholds: dict[str, float] = {}
    f1s: dict[str, float] = {}
    counts: dict[str, int] = {}
    sweeps: dict[str, list[dict[str, Any]]] = {}
    for benchmark, bench_records in sorted(by_benchmark.items()):
        threshold, best, sweep = calibrate_one_benchmark(bench_records, step=step)
        thresholds[benchmark] = round(threshold, 6)
        f1s[benchmark] = round(best.f1, 6)
        counts[benchmark] = len(bench_records)
        sweeps[benchmark] = [point.to_dict() for point in sweep]

    return {
        "thresholds": thresholds,
        "f1_per_benchmark": f1s,
        "n_tasks_per_benchmark": counts,
        "sweep_per_benchmark": sweeps,
        "threshold_step": float(step),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_output_payload(
    summary: dict[str, Any],
    *,
    calibration_run: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    payload.update(summary["thresholds"])
    payload["_metadata"] = {
        "calibration_run": calibration_run,
        "n_tasks_per_benchmark": summary["n_tasks_per_benchmark"],
        "f1_per_benchmark": summary["f1_per_benchmark"],
        "threshold_step": summary["threshold_step"],
    }
    return payload


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Benchmark sweep root containing per-benchmark / per-task apex_result.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the per-benchmark thresholds JSON.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.05,
        help="Sweep step in [0, 1] (default: 0.05 → 21 thresholds).",
    )
    parser.add_argument(
        "--calibration-run",
        default="",
        help="Optional label persisted in the output _metadata block (defaults to --runs-dir basename).",
    )
    parser.add_argument(
        "--full-sweep",
        action="store_true",
        help="Also persist the per-benchmark threshold sweep curve in the output JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    records = load_task_records(args.runs_dir)
    summary = calibrate_thresholds(records, step=float(args.threshold_step))

    calibration_run = (args.calibration_run or "").strip() or args.runs_dir.name
    payload = _build_output_payload(summary, calibration_run=calibration_run)

    if args.full_sweep:
        payload["_metadata"]["sweep_per_benchmark"] = summary["sweep_per_benchmark"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Console summary so `bash` orchestration scripts get a one-line
    # status per benchmark.
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    sys.stderr.write(
        "abstention_calibration_summary:\n"
        f"  total_records:       {len(records)}\n"
        f"  benchmarks:          {sorted(summary['thresholds'].keys())}\n"
    )
    for benchmark, threshold in sorted(summary["thresholds"].items()):
        f1 = summary["f1_per_benchmark"].get(benchmark, 0.0)
        n = summary["n_tasks_per_benchmark"].get(benchmark, 0)
        sys.stderr.write(f"  {benchmark:14s} threshold={threshold:.2f} f1={f1:.4f} n={n}\n")
    return 0 if records else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
