#!/usr/bin/env python3
"""A/B harness comparing the heuristic baseline to the calibrated controller.

Two operating modes:

* ``--mode pairwise`` (default): for every input task the harness invokes the
  orchestrator twice — once with the calibrated library, once with a stripped
  library that forces the heuristic baseline. Captures the controller trace
  and apex_result.json for each side, then aggregates per-task win rate and
  per-regime calibration accuracy.

* ``--mode shadow``: runs the orchestrator once with the calibrated policy and
  re-evaluates each ``regime.<state>`` decision through the heuristic baseline
  in-process. Lighter weight (one orchestrator call per task) and useful when
  the only goal is detecting policy disagreement.

The orchestrator entry point is parameterised via ``--orchestrator-callable``
so the harness stays unit-testable; the default targets
``apex.orchestrator:run_task_dict`` which mirrors the production CLI.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# Allow ``python apex/scripts/ab_controller_policy.py`` invocations.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apex.controller_models import (  # noqa: E402  (sys.path shim)
    ControllerModelLibraryConfig,
    LinearPolicyModelConfig,
    calibrated_weights_dir,
    evaluate_heuristic_baseline,
    reset_calibrated_library_cache,
)
from apex.controller_policy import TASK_REGIME_STATES  # noqa: E402


def _resolve_callable(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError(f"orchestrator callable must be 'module:attr', got {spec!r}")
    module_name, attr_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    target = getattr(module, attr_name, None)
    if target is None or not callable(target):
        raise ValueError(f"orchestrator callable {spec!r} is not callable")
    return target


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


def _outcome_score(payload: Any) -> float:
    if not isinstance(payload, dict):
        return 0.0
    final = payload.get("final")
    if isinstance(final, dict):
        for key in ("pass_rate", "required_pass_rate", "score"):
            value = final.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    for key in ("overall_score", "score"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    if isinstance(payload.get("success"), bool):
        return 1.0 if payload.get("success") else 0.0
    return 0.0


@dataclass
class ArmResult:
    arm: str
    output_dir: Path
    duration_seconds: float
    apex_result: dict[str, Any]
    trace_records: list[dict[str, Any]]
    score: float = 0.0

    def regime_evaluations(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for record in self.trace_records:
            for entry in record.get("policy_evaluations") or []:
                if not isinstance(entry, dict):
                    continue
                model_name = str(entry.get("model_name") or "").strip()
                if model_name.startswith("regime."):
                    out.append(entry)
        return out


@dataclass
class TaskComparison:
    task_id: str
    heuristic: Optional[ArmResult] = None
    calibrated: Optional[ArmResult] = None
    shadow_disagreements: list[dict[str, Any]] = field(default_factory=list)

    def delta(self) -> float:
        if self.heuristic is None or self.calibrated is None:
            return 0.0
        return self.calibrated.score - self.heuristic.score

    def winner(self) -> str:
        delta = self.delta()
        if delta > 1e-6:
            return "calibrated"
        if delta < -1e-6:
            return "heuristic"
        return "tie"


def _heuristic_library() -> ControllerModelLibraryConfig:
    """Return an empty library so evaluate_policy_model bypasses the disk weights."""

    library = ControllerModelLibraryConfig(policy_version="heuristic-bootstrap-v1")
    # Add a sentinel disabled model per regime so the disk loader is bypassed
    # entirely (evaluate_policy_model returns the heuristic baseline for any
    # explicitly-disabled model).
    for state in TASK_REGIME_STATES:
        library.models[f"regime.{state}"] = LinearPolicyModelConfig(enabled=False)
    return library


def _run_arm(
    *,
    arm: str,
    task_payload: dict[str, Any],
    orchestrator: Callable[..., Any],
    work_root: Path,
    use_calibrated: bool,
) -> ArmResult:
    output_dir = work_root / arm
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(task_payload)
    payload["output_dir"] = str(output_dir)
    payload.setdefault("controller_models", {})
    if use_calibrated:
        # Drop any pre-baked overrides; let the on-disk calibrated weights win.
        payload.pop("controller_models", None)
        payload["controller_models"] = {
            "policy_version": "calibrated-v1-synthetic",
            "models": {},
        }
        reset_calibrated_library_cache()
    else:
        payload["controller_models"] = _heuristic_library().to_dict()
    start = time.monotonic()
    apex_result = orchestrator(payload) or {}
    duration = time.monotonic() - start

    if not isinstance(apex_result, dict):
        apex_result = {"raw": apex_result}

    result_path = output_dir / "apex_result.json"
    if not result_path.exists() and isinstance(apex_result, dict) and apex_result:
        result_path.write_text(json.dumps(apex_result, indent=2, sort_keys=True), encoding="utf-8")

    trace_path = output_dir / "controller_decisions.jsonl"
    if not trace_path.exists():
        # Some orchestrators write under a nested location; pick the first match.
        candidates = list(output_dir.rglob("controller_decisions.jsonl"))
        if candidates:
            trace_path = candidates[0]
    trace_records = _load_jsonl(trace_path)

    return ArmResult(
        arm=arm,
        output_dir=output_dir,
        duration_seconds=duration,
        apex_result=apex_result,
        trace_records=trace_records,
        score=_outcome_score(apex_result),
    )


def _shadow_disagreements(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompute heuristic baseline values for regime.* evaluations and flag deltas."""

    out: list[dict[str, Any]] = []
    for record in records:
        for entry in record.get("policy_evaluations") or []:
            if not isinstance(entry, dict):
                continue
            model_name = str(entry.get("model_name") or "").strip()
            if not model_name.startswith("regime."):
                continue
            features = entry.get("features") or {}
            baseline_value = float(entry.get("baseline_value") or 0.0)
            heuristic_eval = evaluate_heuristic_baseline(
                model_name=model_name,
                features=features,
                baseline_value=baseline_value,
            )
            calibrated_value = float(entry.get("value") or baseline_value)
            disagreement = abs(calibrated_value - heuristic_eval.value)
            if disagreement >= 0.10:
                out.append(
                    {
                        "model_name": model_name,
                        "calibrated_value": round(calibrated_value, 4),
                        "heuristic_value": round(float(heuristic_eval.value), 4),
                        "delta": round(disagreement, 4),
                    }
                )
    return out


def run_pairwise(
    *,
    tasks: list[dict[str, Any]],
    orchestrator: Callable[..., Any],
    work_root: Path,
) -> list[TaskComparison]:
    comparisons: list[TaskComparison] = []
    for task in tasks:
        task_id = str(task.get("task_id") or task.get("instance_id") or task.get("name") or "task")
        task_root = work_root / task_id
        task_root.mkdir(parents=True, exist_ok=True)
        heuristic = _run_arm(
            arm="heuristic",
            task_payload=task,
            orchestrator=orchestrator,
            work_root=task_root,
            use_calibrated=False,
        )
        calibrated = _run_arm(
            arm="calibrated",
            task_payload=task,
            orchestrator=orchestrator,
            work_root=task_root,
            use_calibrated=True,
        )
        comparisons.append(
            TaskComparison(
                task_id=task_id,
                heuristic=heuristic,
                calibrated=calibrated,
            )
        )
    return comparisons


def run_shadow(
    *,
    tasks: list[dict[str, Any]],
    orchestrator: Callable[..., Any],
    work_root: Path,
) -> list[TaskComparison]:
    comparisons: list[TaskComparison] = []
    for task in tasks:
        task_id = str(task.get("task_id") or task.get("instance_id") or task.get("name") or "task")
        task_root = work_root / task_id
        task_root.mkdir(parents=True, exist_ok=True)
        calibrated = _run_arm(
            arm="calibrated",
            task_payload=task,
            orchestrator=orchestrator,
            work_root=task_root,
            use_calibrated=True,
        )
        disagreements = _shadow_disagreements(calibrated.trace_records)
        comparisons.append(
            TaskComparison(
                task_id=task_id,
                calibrated=calibrated,
                shadow_disagreements=disagreements,
            )
        )
    return comparisons


def aggregate(comparisons: list[TaskComparison]) -> dict[str, Any]:
    deltas: list[float] = []
    wins = {"heuristic": 0, "calibrated": 0, "tie": 0}
    per_regime_calibration: dict[str, list[float]] = {state: [] for state in TASK_REGIME_STATES}
    per_regime_disagreement: dict[str, int] = {state: 0 for state in TASK_REGIME_STATES}
    shadow_disagreement_total = 0
    has_pairwise = False

    for comparison in comparisons:
        if comparison.heuristic is not None and comparison.calibrated is not None:
            has_pairwise = True
            deltas.append(comparison.delta())
            wins[comparison.winner()] += 1
            for entry in comparison.calibrated.regime_evaluations():
                model_name = str(entry.get("model_name") or "")
                regime = model_name.split(".", 1)[1] if "." in model_name else ""
                if regime not in TASK_REGIME_STATES:
                    continue
                value = float(entry.get("value") or 0.0)
                target = comparison.calibrated.score
                per_regime_calibration[regime].append(abs(value - target))
        for entry in comparison.shadow_disagreements:
            shadow_disagreement_total += 1
            model_name = str(entry.get("model_name") or "")
            regime = model_name.split(".", 1)[1] if "." in model_name else ""
            if regime in per_regime_disagreement:
                per_regime_disagreement[regime] += 1

    summary: dict[str, Any] = {
        "task_count": len(comparisons),
        "wins": wins,
        "win_rate_calibrated": (
            wins["calibrated"] / max(1, wins["calibrated"] + wins["heuristic"])
            if has_pairwise
            else 0.0
        ),
        "delta_overall_score": {
            "mean": float(statistics.fmean(deltas)) if deltas else 0.0,
            "median": float(statistics.median(deltas)) if deltas else 0.0,
            "n": len(deltas),
        },
        "regime_calibration_mae": {
            state: round(float(statistics.fmean(values)), 4) if values else None
            for state, values in per_regime_calibration.items()
        },
        "shadow_disagreements_total": shadow_disagreement_total,
        "shadow_disagreements_per_regime": dict(per_regime_disagreement),
    }
    return summary


def render_report(summary: dict[str, Any], comparisons: list[TaskComparison]) -> str:
    lines: list[str] = []
    lines.append("# Controller Policy A/B Report")
    lines.append("")
    lines.append(f"- Tasks compared: **{summary['task_count']}**")
    pairwise_n = summary["delta_overall_score"]["n"]
    if pairwise_n:
        lines.append(
            f"- Pairwise tasks: **{pairwise_n}** "
            f"(calibrated wins {summary['wins']['calibrated']}, "
            f"heuristic wins {summary['wins']['heuristic']}, "
            f"ties {summary['wins']['tie']})"
        )
        lines.append(f"- Calibrated win rate: **{summary['win_rate_calibrated']:.3f}**")
        lines.append(
            f"- Overall score delta (calibrated - heuristic): mean "
            f"**{summary['delta_overall_score']['mean']:.4f}**, median "
            f"**{summary['delta_overall_score']['median']:.4f}**"
        )
    if summary.get("shadow_disagreements_total"):
        lines.append(f"- Shadow disagreements logged: **{summary['shadow_disagreements_total']}**")
    lines.append("")
    lines.append("## Per-regime calibration MAE")
    lines.append("")
    for regime, value in summary["regime_calibration_mae"].items():
        if value is None:
            lines.append(f"- `{regime}`: n/a")
        else:
            lines.append(f"- `{regime}`: {value:.4f}")
    if summary.get("shadow_disagreements_per_regime"):
        lines.append("")
        lines.append("## Shadow disagreements per regime")
        lines.append("")
        for regime, count in summary["shadow_disagreements_per_regime"].items():
            lines.append(f"- `{regime}`: {count}")
    lines.append("")
    lines.append("## Per-task")
    lines.append("")
    lines.append("| Task | Heuristic | Calibrated | Delta | Winner | Shadow disagreements |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for comparison in comparisons:
        h = f"{comparison.heuristic.score:.4f}" if comparison.heuristic else "n/a"
        c = f"{comparison.calibrated.score:.4f}" if comparison.calibrated else "n/a"
        delta = (
            f"{comparison.delta():+.4f}"
            if comparison.heuristic and comparison.calibrated
            else "n/a"
        )
        winner = comparison.winner() if comparison.heuristic and comparison.calibrated else "shadow"
        shadow_count = len(comparison.shadow_disagreements)
        lines.append(
            f"| `{comparison.task_id}` | {h} | {c} | {delta} | {winner} | {shadow_count} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A/B harness for the controller policy.")
    parser.add_argument(
        "--tasks",
        required=True,
        help="Path to a JSON file containing a list of task payloads to send to the orchestrator.",
    )
    parser.add_argument(
        "--mode",
        choices=("pairwise", "shadow"),
        default="pairwise",
        help="pairwise (default): run heuristic and calibrated arms. shadow: run calibrated only and log heuristic deltas.",
    )
    parser.add_argument(
        "--shadow-mode",
        action="store_true",
        help="Convenience alias for --mode shadow.",
    )
    parser.add_argument(
        "--orchestrator-callable",
        default="apex.orchestrator:run_task_dict",
        help="Python callable that runs a single task. Signature: callable(task_dict) -> apex_result.",
    )
    parser.add_argument(
        "--work-root",
        default="",
        help="Directory for per-task outputs (default: a temp directory).",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Path to write the AB_REPORT.md (default: <work_root>/AB_REPORT.md).",
    )
    parser.add_argument(
        "--summary",
        default="",
        help="Optional path to write the JSON aggregate summary.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    tasks_path = Path(args.tasks)
    tasks_payload = json.loads(tasks_path.read_text(encoding="utf-8"))
    if isinstance(tasks_payload, dict):
        tasks = list(tasks_payload.get("tasks") or [])
    elif isinstance(tasks_payload, list):
        tasks = tasks_payload
    else:
        raise SystemExit(
            f"--tasks must point at a list or {{'tasks': [...]}}, got {type(tasks_payload)}"
        )
    if not tasks:
        raise SystemExit("no tasks supplied")

    orchestrator = _resolve_callable(args.orchestrator_callable)
    work_root = (
        Path(args.work_root) if args.work_root else Path(tempfile.mkdtemp(prefix="apex_ab_"))
    )
    work_root.mkdir(parents=True, exist_ok=True)

    mode = "shadow" if (args.shadow_mode or args.mode == "shadow") else "pairwise"
    if mode == "pairwise":
        comparisons = run_pairwise(tasks=tasks, orchestrator=orchestrator, work_root=work_root)
    else:
        comparisons = run_shadow(tasks=tasks, orchestrator=orchestrator, work_root=work_root)

    summary = aggregate(comparisons)
    summary["mode"] = mode
    summary["work_root"] = str(work_root)
    summary["calibrated_weights_dir"] = str(calibrated_weights_dir())

    report_path = Path(args.report) if args.report else work_root / "AB_REPORT.md"
    report_path.write_text(render_report(summary, comparisons), encoding="utf-8")
    summary["report_path"] = str(report_path)
    if args.summary:
        Path(args.summary).write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        summary["summary_path"] = str(args.summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
