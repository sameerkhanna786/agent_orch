"""
Paired comparison utilities for benchmark report artifacts.
"""

from __future__ import annotations

import copy
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_EPSILON = 1e-9


def load_benchmark_report(path: str | Path) -> dict[str, Any]:
    """Load one benchmark report JSON payload."""

    report_path = Path(path).resolve()
    with report_path.open() as handle:
        return json.load(handle)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def infer_benchmark_family(report: dict[str, Any]) -> str:
    """Infer the benchmark family from a stored report payload."""

    report_kind = str(report.get("report_kind") or "").lower()
    harness_name = str(report.get("harness_name") or "").lower()
    task_sample = next(
        (task for task in report.get("tasks", []) if isinstance(task, dict)),
        {},
    )
    final_sample = dict(task_sample.get("final") or {})

    if "testgen" in report_kind or "testgen" in harness_name:
        return "swebench_pro_testgen"
    if "commit0" in report_kind or "commit0" in harness_name:
        return "commit0"
    if "swebench" in report_kind or "swe-bench" in harness_name or "swebench" in harness_name:
        return "swebench_pro"
    if "required_pass_rate" in final_sample or "required_tests" in final_sample:
        return "swebench_pro"
    if "pass_rate" in final_sample or "instance_id" in task_sample:
        return "commit0"
    return "local"


def _canonical_task_id(task: dict[str, Any]) -> str:
    for key in ("instance_id", "repo", "task_name"):
        value = str(task.get(key) or "").strip()
        if value:
            return value
    return "unknown_task"


def _task_display_name(task: dict[str, Any], benchmark_family: str) -> str:
    if benchmark_family in {"swebench_pro", "swebench_pro_testgen"}:
        return (
            str(task.get("instance_id") or "").strip()
            or str(task.get("task_name") or "").strip()
            or _canonical_task_id(task)
        )
    return (
        str(task.get("task_name") or "").strip()
        or str(task.get("repo") or "").strip()
        or str(task.get("instance_id") or "").strip()
        or _canonical_task_id(task)
    )


def _extract_task_score(task: dict[str, Any], benchmark_family: str) -> float:
    if benchmark_family == "commit0":
        return _safe_float((task.get("final") or {}).get("pass_rate"), 0.0)
    if benchmark_family == "swebench_pro":
        return _safe_float((task.get("final") or {}).get("required_pass_rate"), 0.0)
    if benchmark_family == "swebench_pro_testgen":
        metadata = dict(task.get("execution_metadata") or {})
        f2p_summary = dict(metadata.get("f2p_summary") or {})
        if f2p_summary:
            return _safe_float(
                f2p_summary.get("f2p_rate"),
                1.0 if bool(f2p_summary.get("any_f2p")) else 0.0,
            )
        coverage_summary = dict(task.get("coverage_summary") or {})
        return _safe_float(
            coverage_summary.get("overall_contract_axis_recall"),
            0.0,
        )
    return 1.0 if task.get("final_tests_passed") else 0.0


def _extract_baseline_score(task: dict[str, Any], benchmark_family: str) -> Optional[float]:
    if benchmark_family == "commit0":
        return _safe_float((task.get("baseline") or {}).get("pass_rate"), 0.0)
    if benchmark_family == "swebench_pro":
        return _safe_float((task.get("baseline") or {}).get("required_pass_rate"), 0.0)
    return None


def _extract_task_solved(task: dict[str, Any], benchmark_family: str) -> bool:
    if benchmark_family == "swebench_pro_testgen":
        metadata = dict(task.get("execution_metadata") or {})
        f2p_summary = dict(metadata.get("f2p_summary") or {})
        if f2p_summary:
            return bool(f2p_summary.get("any_f2p"))
        return (
            bool(task.get("success"))
            and _extract_task_score(task, benchmark_family) >= 1.0 - _EPSILON
        )
    return bool(task.get("final_tests_passed", False))


def classify_failure_type(
    task: dict[str, Any],
    benchmark_family: str,
    *,
    score: Optional[float] = None,
    baseline_score: Optional[float] = None,
) -> str:
    """Assign a coarse failure bucket for report-level analysis."""

    if bool(task.get("skipped", False)):
        category = str(task.get("skip_category") or "unknown").strip() or "unknown"
        return f"skipped:{category}"
    if _extract_task_solved(task, benchmark_family):
        return "solved"

    if benchmark_family == "swebench_pro_testgen":
        metadata = dict(task.get("execution_metadata") or {})
        f2p_summary = dict(metadata.get("f2p_summary") or {})
        if f2p_summary:
            status = str(f2p_summary.get("status") or "").strip()
            if status.startswith("skip_"):
                return status
            if status.startswith("error"):
                return "testgen_oracle_error"
            if bool(task.get("success")):
                return "no_f2p_signal"
        comparison_status = str(metadata.get("comparison_status") or "").strip()
        if comparison_status and comparison_status != "ok":
            return comparison_status
        if bool(task.get("success")):
            return "no_f2p_signal"

    failure_reason = str(task.get("failure_reason") or "")
    final_output = str((task.get("final") or {}).get("output") or "")
    combined_text = f"{failure_reason}\n{final_output}".lower()
    returncode = int(_safe_float((task.get("final") or {}).get("returncode"), 1))

    if "timeout" in combined_text or "timed out" in combined_text or returncode == 124:
        return "timeout"

    environment_markers = (
        "module not found",
        "modulenotfounderror",
        "no module named",
        "command not found",
        "docker",
        "network",
        "connection reset",
        "connection refused",
        "failed to build wheel",
        "pip install",
        "permission denied",
    )
    if any(marker in combined_text for marker in environment_markers):
        return "environment"

    if "coverage failure" in combined_text or "fail-under" in combined_text:
        return "evaluation_gate"

    if "selector" in combined_text or "selection" in combined_text:
        return "selector_miss"

    effective_score = score if score is not None else _extract_task_score(task, benchmark_family)
    effective_baseline = (
        baseline_score
        if baseline_score is not None
        else _extract_baseline_score(task, benchmark_family)
    )
    if effective_baseline is not None:
        if effective_score + _EPSILON < effective_baseline:
            return "regression"
        if effective_score > effective_baseline + _EPSILON:
            return "partial_implementation"
    if effective_score > _EPSILON and effective_score < 1.0 - _EPSILON:
        return "partial_implementation"

    if bool(task.get("success")) or bool(task.get("agent_success")):
        return "selector_miss"
    if bool(task.get("orchestrator_success")) or bool(task.get("candidate_found")):
        return "selector_miss"
    return "no_progress"


def normalize_benchmark_report(
    path: str | Path,
    *,
    label: Optional[str] = None,
) -> dict[str, Any]:
    """Normalize one report into a task-indexed representation."""

    report_path = Path(path).resolve()
    report = load_benchmark_report(report_path)
    benchmark_family = infer_benchmark_family(report)
    tasks = [task for task in report.get("tasks", []) if isinstance(task, dict)]

    normalized_tasks: dict[str, dict[str, Any]] = {}
    task_order: list[str] = []
    for task in tasks:
        task_id = _canonical_task_id(task)
        task_order.append(task_id)
        score = _extract_task_score(task, benchmark_family)
        baseline_score = _extract_baseline_score(task, benchmark_family)
        failure_type = classify_failure_type(
            task,
            benchmark_family,
            score=score,
            baseline_score=baseline_score,
        )
        normalized_tasks[task_id] = {
            "task_id": task_id,
            "display_name": _task_display_name(task, benchmark_family),
            "task_name": str(task.get("task_name") or "").strip(),
            "repo": str(task.get("repo") or "").strip(),
            "instance_id": str(task.get("instance_id") or "").strip(),
            "score": round(score, 6),
            "score_percent": round(100.0 * score, 2),
            "baseline_score": _round_or_none(baseline_score, 6),
            "baseline_score_percent": (
                round(100.0 * baseline_score, 2) if baseline_score is not None else None
            ),
            "solved": _extract_task_solved(task, benchmark_family),
            "success": bool(task.get("success", False) or task.get("agent_success", False)),
            "tokens": int(_safe_float(task.get("total_tokens"), 0)),
            "duration_seconds": round(_safe_float(task.get("duration_seconds"), 0.0), 4),
            "failure_reason": str(task.get("failure_reason") or "").strip(),
            "failure_type": failure_type,
            "skipped": bool(task.get("skipped", False)),
            "skip_category": str(task.get("skip_category") or "").strip() or None,
            "execution_metadata": copy.deepcopy(task.get("execution_metadata") or {}),
        }

    return {
        "label": label or report_path.stem,
        "path": str(report_path),
        "report_paths": [str(report_path)],
        "report_kind": str(report.get("report_kind") or "unknown"),
        "harness_name": str(report.get("harness_name") or "unknown"),
        "harness_version": report.get("harness_version"),
        "benchmark_family": benchmark_family,
        "config_source": report.get("config_source"),
        "model_config": copy.deepcopy(report.get("model_config") or []),
        "ablation_config": copy.deepcopy(report.get("ablation_config") or {}),
        "tasks": normalized_tasks,
        "task_order": task_order,
        "raw_report": report,
    }


def _merge_normalized_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("Expected at least one normalized report to merge.")
    benchmark_families = {report["benchmark_family"] for report in reports}
    if len(benchmark_families) != 1:
        raise ValueError("Cannot merge report shards from different benchmark families.")

    merged = copy.deepcopy(reports[0])
    merged["report_paths"] = [
        path for report in reports for path in report.get("report_paths", [report["path"]])
    ]
    merged["raw_reports"] = [copy.deepcopy(report["raw_report"]) for report in reports]

    merged_tasks: dict[str, dict[str, Any]] = {}
    merged_order: list[str] = []
    for report in reports:
        for task_id in report["task_order"]:
            if task_id not in merged_tasks:
                merged_order.append(task_id)
            merged_tasks[task_id] = copy.deepcopy(report["tasks"][task_id])

    merged["tasks"] = merged_tasks
    merged["task_order"] = merged_order
    if len({report["report_kind"] for report in reports}) > 1:
        merged["report_kind"] = "merged"
    return merged


def _summarize_task_set(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(task["score"]) for task in tasks]
    baselines = [
        float(task["baseline_score"])
        for task in tasks
        if isinstance(task.get("baseline_score"), (int, float))
    ]
    solved_tasks = sum(1 for task in tasks if task["solved"])
    total_tokens = sum(int(task["tokens"]) for task in tasks)
    total_duration_seconds = sum(float(task["duration_seconds"]) for task in tasks)
    failure_taxonomy = Counter(task["failure_type"] for task in tasks)

    average_score = _mean(scores)
    average_baseline = _mean(baselines) if baselines else None
    average_baseline_delta = (
        average_score - average_baseline if average_baseline is not None else None
    )
    hours = total_duration_seconds / 3600.0

    return {
        "average_score": round(average_score, 6),
        "average_score_percent": round(100.0 * average_score, 2),
        "average_baseline_score": _round_or_none(average_baseline, 6),
        "average_baseline_score_percent": (
            round(100.0 * average_baseline, 2) if average_baseline is not None else None
        ),
        "average_score_delta_vs_baseline": _round_or_none(average_baseline_delta, 6),
        "average_score_delta_vs_baseline_percent": (
            round(100.0 * average_baseline_delta, 2) if average_baseline_delta is not None else None
        ),
        "solve_rate": round(solved_tasks / len(tasks), 6) if tasks else 0.0,
        "solve_rate_percent": round(100.0 * solved_tasks / len(tasks), 2) if tasks else 0.0,
        "solved_tasks": solved_tasks,
        "total_tokens": total_tokens,
        "total_duration_seconds": round(total_duration_seconds, 4),
        "average_tokens": round(total_tokens / len(tasks), 2) if tasks else 0.0,
        "average_duration_seconds": round(total_duration_seconds / len(tasks), 2) if tasks else 0.0,
        "tokens_per_solve": round(total_tokens / solved_tasks, 2) if solved_tasks else None,
        "hours_per_solve": round(hours / solved_tasks, 4) if solved_tasks else None,
        "solves_per_million_tokens": (
            round((solved_tasks / total_tokens) * 1_000_000.0, 4) if total_tokens else None
        ),
        "solves_per_hour": round(solved_tasks / hours, 4) if hours > 0 else None,
        "failure_taxonomy": dict(sorted(failure_taxonomy.items())),
    }


def _bootstrap_mean_ci(
    values: list[float],
    *,
    sample_count: int = 2000,
    confidence: float = 0.95,
) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        value = round(values[0], 6)
        return [value, value]

    rng = random.Random(0)
    means: list[float] = []
    for _ in range(max(sample_count, 100)):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(_mean(sample))
    means.sort()
    lower_index = int(((1.0 - confidence) / 2.0) * len(means))
    upper_index = int((1.0 - ((1.0 - confidence) / 2.0)) * len(means)) - 1
    lower_index = max(0, min(lower_index, len(means) - 1))
    upper_index = max(0, min(upper_index, len(means) - 1))
    return [round(means[lower_index], 6), round(means[upper_index], 6)]


def _format_signed_percent(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def _compare_runs(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    task_ids: list[str],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    score_deltas: list[float] = []
    solve_deltas: list[float] = []
    token_deltas: list[float] = []
    duration_deltas: list[float] = []
    changed_task_rows: list[dict[str, Any]] = []
    failure_transitions: Counter[tuple[str, str]] = Counter()

    score_wins = 0
    score_losses = 0
    score_ties = 0
    solve_wins = 0
    solve_losses = 0
    solve_ties = 0

    for task_id in task_ids:
        reference_task = reference["tasks"][task_id]
        candidate_task = candidate["tasks"][task_id]
        score_delta = float(candidate_task["score"]) - float(reference_task["score"])
        solve_delta = int(candidate_task["solved"]) - int(reference_task["solved"])
        token_delta = int(candidate_task["tokens"]) - int(reference_task["tokens"])
        duration_delta = float(candidate_task["duration_seconds"]) - float(
            reference_task["duration_seconds"]
        )

        score_deltas.append(score_delta)
        solve_deltas.append(float(solve_delta))
        token_deltas.append(float(token_delta))
        duration_deltas.append(duration_delta)
        failure_transitions[(reference_task["failure_type"], candidate_task["failure_type"])] += 1

        if score_delta > _EPSILON:
            score_wins += 1
        elif score_delta < -_EPSILON:
            score_losses += 1
        else:
            score_ties += 1

        if solve_delta > 0:
            solve_wins += 1
        elif solve_delta < 0:
            solve_losses += 1
        else:
            solve_ties += 1

        if abs(score_delta) > _EPSILON or solve_delta != 0:
            changed_task_rows.append(
                {
                    "task_id": task_id,
                    "task_name": reference_task["display_name"],
                    "reference_score": round(float(reference_task["score"]), 6),
                    "reference_score_percent": round(float(reference_task["score_percent"]), 2),
                    "candidate_score": round(float(candidate_task["score"]), 6),
                    "candidate_score_percent": round(float(candidate_task["score_percent"]), 2),
                    "score_delta": round(score_delta, 6),
                    "score_delta_percent": round(100.0 * score_delta, 2),
                    "solve_delta": solve_delta,
                    "reference_failure": reference_task["failure_type"],
                    "candidate_failure": candidate_task["failure_type"],
                }
            )

    changed_task_rows.sort(
        key=lambda row: (
            -abs(int(row["solve_delta"])),
            -abs(float(row["score_delta"])),
            row["task_name"],
        )
    )

    average_score_delta = _mean(score_deltas)
    average_solve_delta = _mean(solve_deltas)
    average_token_delta = _mean(token_deltas)
    average_duration_delta = _mean(duration_deltas)
    score_ci = _bootstrap_mean_ci(score_deltas, sample_count=bootstrap_samples)
    solve_ci = _bootstrap_mean_ci(solve_deltas, sample_count=bootstrap_samples)

    return {
        "reference_label": reference["label"],
        "candidate_label": candidate["label"],
        "compared_task_count": len(task_ids),
        "average_score_delta": round(average_score_delta, 6),
        "average_score_delta_percent": round(100.0 * average_score_delta, 2),
        "average_score_delta_ci": score_ci,
        "average_score_delta_ci_percent": [round(100.0 * value, 2) for value in score_ci],
        "solve_rate_delta": round(average_solve_delta, 6),
        "solve_rate_delta_percent": round(100.0 * average_solve_delta, 2),
        "solve_rate_delta_ci": solve_ci,
        "solve_rate_delta_ci_percent": [round(100.0 * value, 2) for value in solve_ci],
        "average_token_delta": round(average_token_delta, 2),
        "average_duration_delta_seconds": round(average_duration_delta, 2),
        "score_wins": score_wins,
        "score_losses": score_losses,
        "score_ties": score_ties,
        "solve_wins": solve_wins,
        "solve_losses": solve_losses,
        "solve_ties": solve_ties,
        "changed_task_count": len(changed_task_rows),
        "task_deltas": changed_task_rows,
        "failure_transitions": [
            {
                "from": source,
                "to": target,
                "count": count,
            }
            for (source, target), count in sorted(
                failure_transitions.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
        ],
    }


def compare_benchmark_reports(
    report_paths: list[str | Path],
    *,
    labels: Optional[list[str]] = None,
    comparison_name: Optional[str] = None,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    """Compare two or more benchmark reports on their shared task set."""

    if len(report_paths) < 2:
        raise ValueError("Expected at least two benchmark reports to compare.")
    if labels is not None and len(labels) != len(report_paths):
        raise ValueError("labels must match the number of report_paths")

    normalized_reports = [
        normalize_benchmark_report(
            path,
            label=(labels[index] if labels is not None else None),
        )
        for index, path in enumerate(report_paths)
    ]
    grouped_reports: list[dict[str, Any]] = []
    grouped_report_map: dict[str, list[dict[str, Any]]] = {}
    for report in normalized_reports:
        grouped_report_map.setdefault(report["label"], []).append(report)
    seen_labels: set[str] = set()
    for report in normalized_reports:
        label = report["label"]
        if label in seen_labels:
            continue
        seen_labels.add(label)
        grouped_reports.append(_merge_normalized_reports(grouped_report_map[label]))

    benchmark_families = {report["benchmark_family"] for report in grouped_reports}
    if len(benchmark_families) != 1:
        raise ValueError("All reports must belong to the same benchmark family.")

    reference = grouped_reports[0]
    reference_task_ids = list(reference["task_order"])
    common_task_ids = [
        task_id
        for task_id in reference_task_ids
        if all(task_id in report["tasks"] for report in grouped_reports)
    ]
    if not common_task_ids:
        raise ValueError("No shared tasks were found across the provided reports.")

    run_summaries: list[dict[str, Any]] = []
    for report in grouped_reports:
        matched_tasks = [report["tasks"][task_id] for task_id in common_task_ids]
        matched_summary = _summarize_task_set(matched_tasks)
        run_summaries.append(
            {
                "label": report["label"],
                "report_path": report["path"],
                "report_paths": list(report.get("report_paths", [report["path"]])),
                "report_kind": report["report_kind"],
                "benchmark_family": report["benchmark_family"],
                "report_task_count": len(report["tasks"]),
                "source_report_count": len(report.get("report_paths", [report["path"]])),
                "matched_task_count": len(matched_tasks),
                "missing_task_ids": [
                    task_id for task_id in reference_task_ids if task_id not in report["tasks"]
                ],
                "extra_task_ids": [
                    task_id for task_id in report["task_order"] if task_id not in reference["tasks"]
                ],
                "matched_average_score": matched_summary["average_score"],
                "matched_average_score_percent": matched_summary["average_score_percent"],
                "matched_average_baseline_score": matched_summary["average_baseline_score"],
                "matched_average_baseline_score_percent": matched_summary[
                    "average_baseline_score_percent"
                ],
                "matched_average_score_delta_vs_baseline": matched_summary[
                    "average_score_delta_vs_baseline"
                ],
                "matched_average_score_delta_vs_baseline_percent": matched_summary[
                    "average_score_delta_vs_baseline_percent"
                ],
                "matched_solve_rate": matched_summary["solve_rate"],
                "matched_solve_rate_percent": matched_summary["solve_rate_percent"],
                "matched_solved_tasks": matched_summary["solved_tasks"],
                "matched_total_tokens": matched_summary["total_tokens"],
                "matched_total_duration_seconds": matched_summary["total_duration_seconds"],
                "matched_average_tokens": matched_summary["average_tokens"],
                "matched_average_duration_seconds": matched_summary["average_duration_seconds"],
                "matched_tokens_per_solve": matched_summary["tokens_per_solve"],
                "matched_hours_per_solve": matched_summary["hours_per_solve"],
                "matched_solves_per_million_tokens": matched_summary["solves_per_million_tokens"],
                "matched_solves_per_hour": matched_summary["solves_per_hour"],
                "failure_taxonomy": matched_summary["failure_taxonomy"],
            }
        )

    pairwise_comparisons = [
        _compare_runs(reference, candidate, common_task_ids, bootstrap_samples=bootstrap_samples)
        for candidate in grouped_reports[1:]
    ]

    per_task: dict[str, dict[str, Any]] = {}
    for task_id in common_task_ids:
        per_task[task_id] = {
            "task_name": reference["tasks"][task_id]["display_name"],
            "runs": {
                report["label"]: copy.deepcopy(report["tasks"][task_id])
                for report in grouped_reports
            },
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparison_name": comparison_name or "Benchmark comparison",
        "benchmark_family": reference["benchmark_family"],
        "reference_label": reference["label"],
        "reference_report_path": reference["path"],
        "common_task_count": len(common_task_ids),
        "common_task_ids": list(common_task_ids),
        "runs": run_summaries,
        "pairwise_comparisons": pairwise_comparisons,
        "per_task": per_task,
    }


def render_benchmark_comparison_markdown(payload: dict[str, Any]) -> str:
    """Render a human-readable markdown summary for a comparison payload."""

    lines = [
        "# Benchmark Comparison",
        "",
        f"- Comparison: {payload.get('comparison_name') or 'Benchmark comparison'}",
        f"- Benchmark family: {payload['benchmark_family']}",
        f"- Reference run: {payload['reference_label']}",
        f"- Matched tasks: {payload['common_task_count']}",
        f"- Generated at: {payload['generated_at']}",
        "",
        "## Runs",
        "",
        "| Run | Report Tasks | Matched Tasks | Avg Score | Solve Rate | Tokens | Duration (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for run in payload["runs"]:
        lines.append(
            "| {label} | {report_tasks} | {matched_tasks} | {avg_score:.2f}% | {solve_rate:.2f}% | {tokens} | {duration:.2f} |".format(
                label=run["label"],
                report_tasks=run["report_task_count"],
                matched_tasks=run["matched_task_count"],
                avg_score=run["matched_average_score_percent"],
                solve_rate=run["matched_solve_rate_percent"],
                tokens=run["matched_total_tokens"],
                duration=run["matched_total_duration_seconds"],
            )
        )

    lines.extend(["", "## Failure Taxonomy", ""])
    failure_labels: list[str] = []
    for run in payload["runs"]:
        for label in run.get("failure_taxonomy", {}):
            if label not in failure_labels:
                failure_labels.append(label)
    if failure_labels:
        lines.append("| Run | " + " | ".join(failure_labels) + " |")
        lines.append("| --- | " + " | ".join("---:" for _ in failure_labels) + " |")
        for run in payload["runs"]:
            lines.append(
                "| {label} | {counts} |".format(
                    label=run["label"],
                    counts=" | ".join(
                        str((run.get("failure_taxonomy") or {}).get(failure_label, 0))
                        for failure_label in failure_labels
                    ),
                )
            )
    else:
        lines.append("- No failures to classify on the shared task set.")

    for comparison in payload["pairwise_comparisons"]:
        lines.extend(
            [
                "",
                f"## Pairwise vs {comparison['reference_label']}",
                "",
                f"### {comparison['candidate_label']}",
                "",
                (
                    "- Average score delta: "
                    f"{_format_signed_percent(comparison['average_score_delta_percent'])} "
                    f"(95% CI {_format_signed_percent(comparison['average_score_delta_ci_percent'][0])} "
                    f"to {_format_signed_percent(comparison['average_score_delta_ci_percent'][1])})"
                ),
                (
                    "- Solve rate delta: "
                    f"{_format_signed_percent(comparison['solve_rate_delta_percent'])} "
                    f"(95% CI {_format_signed_percent(comparison['solve_rate_delta_ci_percent'][0])} "
                    f"to {_format_signed_percent(comparison['solve_rate_delta_ci_percent'][1])})"
                ),
                (
                    "- Score wins/losses/ties: "
                    f"{comparison['score_wins']} / {comparison['score_losses']} / {comparison['score_ties']}"
                ),
                (
                    "- Solve wins/losses/ties: "
                    f"{comparison['solve_wins']} / {comparison['solve_losses']} / {comparison['solve_ties']}"
                ),
                (f"- Average token delta per task: {comparison['average_token_delta']:+.2f}"),
                (
                    "- Average duration delta per task: "
                    f"{comparison['average_duration_delta_seconds']:+.2f}s"
                ),
            ]
        )

        if comparison["failure_transitions"]:
            top_transitions = comparison["failure_transitions"][:5]
            lines.append(
                "- Top failure transitions: "
                + ", ".join(
                    f"{item['from']} -> {item['to']} ({item['count']})" for item in top_transitions
                )
            )

        if comparison["task_deltas"]:
            lines.extend(
                [
                    "",
                    "| Task | {reference} | {candidate} | Score Delta | Solve Delta | {reference} Failure | {candidate} Failure |".format(
                        reference=comparison["reference_label"],
                        candidate=comparison["candidate_label"],
                    ),
                    "| --- | ---: | ---: | ---: | ---: | --- | --- |",
                ]
            )
            for task in comparison["task_deltas"]:
                lines.append(
                    "| {task_name} | {reference_score:.2f}% | {candidate_score:.2f}% | {score_delta:+.2f}% | {solve_delta:+d} | {reference_failure} | {candidate_failure} |".format(
                        task_name=task["task_name"],
                        reference_score=task["reference_score_percent"],
                        candidate_score=task["candidate_score_percent"],
                        score_delta=task["score_delta_percent"],
                        solve_delta=int(task["solve_delta"]),
                        reference_failure=task["reference_failure"],
                        candidate_failure=task["candidate_failure"],
                    )
                )
        else:
            lines.extend(["", "- No task-level score or solve changes on the matched set."])

    return "\n".join(lines).rstrip() + "\n"
