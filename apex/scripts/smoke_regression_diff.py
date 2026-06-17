#!/usr/bin/env python3
"""Decisive-Edge B.9 — smoke regression diff.

Compares two benchmark report directories (typically the pre- and
post-remediation smoke runs from ``scripts/run_benchmark_sweep.sh``)
and produces ``REGRESSION_DIFF.md`` with:

  * A per-task table of score deltas (matched by ``instance_id`` →
    ``task_name`` → ``repo`` in that fallback order).
  * A regressions section flagging any task whose score dropped by more
    than the ``--threshold-pp`` percentage points (default 3pp).
  * An attribution summary that bins each regression by likely cause:

      - strategy_axis_change          : the rollout's
        ``diversity_strategy_axis`` differs between runs
      - salvage_to_abstention         : status went SOLVED → ABSTAINED /
        FAILED
      - localizer_hard_constraint     : ``off_target_patches`` appeared
        in diagnostics in the post-run
      - selection_winner_changed      : ``selected_rollout_id`` differs
      - amplifier_kicked_in           : ``amplifier_used: true`` appeared
        in the post-run
      - tasks_added                   : present in post but not pre
      - tasks_removed                 : present in pre but not post
      - unattributed                  : score regressed but none of the
        heuristics fired

Inputs are the per-benchmark output dirs that
``scripts/run_benchmark_sweep.sh`` writes (e.g. ``commit0_lite/``,
``swtbench_lite/``, ``testgeneval_lite/``). The runner emits a
``benchmark_report.json`` per benchmark whose ``tasks`` field is the
per-task list. We auto-discover that file under each input dir.

Example::

    python apex/scripts/smoke_regression_diff.py \\
        --pre runs/sweep_pre/ \\
        --post runs/sweep_post/ \\
        --output runs/diff/REGRESSION_DIFF.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Allow ``python apex/scripts/smoke_regression_diff.py`` invocations.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Discovery + parsing
# ---------------------------------------------------------------------------


_REPORT_FILENAMES: tuple[str, ...] = (
    "benchmark_report.json",
    "RUN_REPORT.json",
)


def _discover_report_files(root: Path) -> dict[str, Path]:
    """Walk ``root`` and pull out per-benchmark report JSONs.

    Keys are inferred from the report's ``report_kind`` /
    ``harness_name`` field, falling back to the parent directory name.
    Multiple report files of the same kind get suffixed by parent dir.
    """
    if not root.exists():
        return {}
    if root.is_file():
        return {root.stem: root}
    found: dict[str, Path] = {}
    for path in root.rglob("*.json"):
        if path.name not in _REPORT_FILENAMES:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if "tasks" not in payload:
            # not a benchmark_report-shaped file
            continue
        kind = (
            str(payload.get("report_kind") or "").strip().lower()
            or str(payload.get("harness_name") or "").strip().lower()
            or path.parent.name.lower()
            or "benchmark"
        )
        # Disambiguate when the same kind appears under multiple parents
        # (e.g. swtbench_lite vs swtbench_full both report_kind=swtbench).
        if kind in found:
            kind = f"{kind}__{path.parent.name}"
        found[kind] = path
    return found


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_task_id(task: dict[str, Any]) -> str:
    for key in ("instance_id", "task_name", "repo"):
        value = str(task.get(key) or "").strip()
        if value:
            return value
    return "unknown_task"


def _task_status(task: dict[str, Any]) -> str:
    """Best-effort classification of a task into SOLVED / ABSTAINED / FAILED.

    The benchmark schema doesn't expose a single ``status`` column; we
    derive it from the standard fields the harness writes.
    """
    if bool(task.get("skipped")):
        return "SKIPPED"
    if "abstention_outcome" in task:
        outcome = str(task.get("abstention_outcome") or "").strip().lower()
        if outcome and outcome != "selected":
            return "ABSTAINED"
    if bool(task.get("final_tests_passed")) or bool(task.get("success")):
        return "SOLVED"
    return "FAILED"


def _task_score(task: dict[str, Any]) -> float:
    """Per-task score; matches the precedence used in the harness.

    Looks at common shapes across commit0 / swtbench / testgeneval /
    swebench_pro. Returns 0.0 when no score is found; callers decide
    how to treat that.
    """
    final = task.get("final")
    if isinstance(final, dict):
        for key in ("pass_rate", "overall_score", "required_pass_rate", "score"):
            value = final.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    for key in ("score", "overall_score"):
        value = task.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    if bool(task.get("final_tests_passed")) or bool(task.get("success")):
        return 1.0
    return 0.0


def _task_diagnostics(task: dict[str, Any]) -> dict[str, Any]:
    """Pull diagnostics out of either the top-level or final.diagnostics.

    Different harnesses stash them in different places; we union both.
    """
    out: dict[str, Any] = {}
    top = task.get("diagnostics")
    if isinstance(top, dict):
        out.update(top)
    final = task.get("final")
    if isinstance(final, dict):
        nested = final.get("diagnostics")
        if isinstance(nested, dict):
            for k, v in nested.items():
                out.setdefault(k, v)
    exec_md = task.get("execution_metadata")
    if isinstance(exec_md, dict):
        for k, v in exec_md.items():
            out.setdefault(k, v)
    return out


def _task_strategy_axis(task: dict[str, Any]) -> Optional[str]:
    """Pull the diversity strategy axis from the task payload."""
    diagnostics = _task_diagnostics(task)
    for key in ("diversity_strategy_axis", "strategy_axis"):
        value = diagnostics.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    sm = task.get("search_metadata")
    if isinstance(sm, dict):
        value = sm.get("diversity_strategy_axis")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _task_selected_rollout_id(task: dict[str, Any]) -> Any:
    return task.get("selected_rollout_id") or task.get("orchestrator_selected_rollout_id")


def _task_off_target_patches(task: dict[str, Any]) -> int:
    """Count off-target patches from diagnostics (B-α localizer enforcer)."""
    diagnostics = _task_diagnostics(task)
    raw = diagnostics.get("off_target_patches")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, list):
        return len(raw)
    if isinstance(raw, dict):
        return len(raw)
    return 0


def _task_amplifier_used(task: dict[str, Any]) -> bool:
    diagnostics = _task_diagnostics(task)
    for key in ("amplifier_used", "verification_amplifier_used"):
        value = diagnostics.get(key)
        if isinstance(value, bool):
            return value
    return False


# ---------------------------------------------------------------------------
# Diff data structures
# ---------------------------------------------------------------------------


@dataclass
class TaskDiff:
    benchmark: str
    task_id: str
    pre_score: Optional[float]
    post_score: Optional[float]
    pre_status: Optional[str]
    post_status: Optional[str]
    score_delta: float
    pre_task: Optional[dict[str, Any]] = None
    post_task: Optional[dict[str, Any]] = None
    attributions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Attribution heuristics
# ---------------------------------------------------------------------------


def attribute_regression(diff: TaskDiff) -> list[str]:
    """Return a list of attribution labels for a regressed task.

    Each heuristic is independent; a regression can be attributed to
    multiple causes (e.g. a strategy axis change AND a selection winner
    change are both worth flagging). The list is empty when no
    heuristic fires (the diff is then ``unattributed``).
    """
    out: list[str] = []
    pre = diff.pre_task or {}
    post = diff.post_task or {}

    # 1. Strategy-axis change.
    pre_axis = _task_strategy_axis(pre)
    post_axis = _task_strategy_axis(post)
    if pre_axis and post_axis and pre_axis != post_axis:
        out.append("strategy_axis_change")

    # 2. Salvage → abstention.
    if diff.pre_status == "SOLVED" and diff.post_status in {"ABSTAINED", "FAILED"}:
        out.append("salvage_to_abstention")

    # 3. Localizer hard_constraint rejection.
    pre_off = _task_off_target_patches(pre)
    post_off = _task_off_target_patches(post)
    if post_off > pre_off:
        out.append("localizer_hard_constraint")

    # 4. Selection winner change.
    if (
        _task_selected_rollout_id(pre) is not None
        and _task_selected_rollout_id(post) is not None
        and _task_selected_rollout_id(pre) != _task_selected_rollout_id(post)
    ):
        out.append("selection_winner_changed")

    # 5. Verification amplifier kicked in (post run only).
    if _task_amplifier_used(post) and not _task_amplifier_used(pre):
        out.append("amplifier_kicked_in")

    return out


# ---------------------------------------------------------------------------
# Top-level diff
# ---------------------------------------------------------------------------


def diff_benchmark(
    *,
    benchmark: str,
    pre_report: dict[str, Any],
    post_report: dict[str, Any],
    threshold_pp: float,
) -> dict[str, Any]:
    """Compute the per-task diff between two benchmark reports."""
    pre_tasks = {
        _canonical_task_id(task): task
        for task in (pre_report.get("tasks") or [])
        if isinstance(task, dict)
    }
    post_tasks = {
        _canonical_task_id(task): task
        for task in (post_report.get("tasks") or [])
        if isinstance(task, dict)
    }
    all_ids = sorted(set(pre_tasks) | set(post_tasks))
    diffs: list[TaskDiff] = []
    added: list[str] = []
    removed: list[str] = []
    threshold_fraction = float(threshold_pp) / 100.0
    for task_id in all_ids:
        pre = pre_tasks.get(task_id)
        post = post_tasks.get(task_id)
        if pre is None and post is not None:
            added.append(task_id)
            diffs.append(
                TaskDiff(
                    benchmark=benchmark,
                    task_id=task_id,
                    pre_score=None,
                    post_score=_task_score(post),
                    pre_status=None,
                    post_status=_task_status(post),
                    score_delta=0.0,
                    pre_task=None,
                    post_task=post,
                    attributions=["tasks_added"],
                )
            )
            continue
        if post is None and pre is not None:
            removed.append(task_id)
            diffs.append(
                TaskDiff(
                    benchmark=benchmark,
                    task_id=task_id,
                    pre_score=_task_score(pre),
                    post_score=None,
                    pre_status=_task_status(pre),
                    post_status=None,
                    score_delta=0.0,
                    pre_task=pre,
                    post_task=None,
                    attributions=["tasks_removed"],
                )
            )
            continue
        pre_score = _task_score(pre)  # type: ignore[arg-type]
        post_score = _task_score(post)  # type: ignore[arg-type]
        delta = post_score - pre_score
        diff = TaskDiff(
            benchmark=benchmark,
            task_id=task_id,
            pre_score=pre_score,
            post_score=post_score,
            pre_status=_task_status(pre),  # type: ignore[arg-type]
            post_status=_task_status(post),  # type: ignore[arg-type]
            score_delta=delta,
            pre_task=pre,
            post_task=post,
        )
        if delta < -threshold_fraction:
            diff.attributions = attribute_regression(diff) or ["unattributed"]
        diffs.append(diff)
    regressions = [
        d for d in diffs if d.score_delta < -threshold_fraction and d.pre_task and d.post_task
    ]
    return {
        "benchmark": benchmark,
        "diffs": diffs,
        "added": added,
        "removed": removed,
        "regressions": regressions,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def render_report(
    *,
    per_benchmark: list[dict[str, Any]],
    threshold_pp: float,
) -> str:
    lines: list[str] = [
        "# Smoke Regression Diff",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        f"Regression threshold: any task with Δ(score) < -{threshold_pp:.1f}pp.",
        "",
    ]
    overall_attribution: dict[str, int] = {}
    for entry in per_benchmark:
        benchmark = entry["benchmark"]
        diffs: list[TaskDiff] = entry["diffs"]
        added: list[str] = entry["added"]
        removed: list[str] = entry["removed"]
        regressions: list[TaskDiff] = entry["regressions"]
        lines.append(f"## Benchmark: {benchmark}")
        lines.append("")
        lines.append(f"- matched tasks: **{len(diffs) - len(added) - len(removed)}**")
        lines.append(f"- added (post only): **{len(added)}**")
        lines.append(f"- removed (pre only): **{len(removed)}**")
        lines.append(f"- flagged regressions (Δ < -{threshold_pp:.1f}pp): **{len(regressions)}**")
        lines.append("")
        lines.append(
            "| task_id | pre_status | post_status | pre_score | post_score | Δ | attributions |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for d in diffs:
            attribs = ", ".join(d.attributions) if d.attributions else ""
            pre_score = "-" if d.pre_score is None else f"{d.pre_score:.3f}"
            post_score = "-" if d.post_score is None else f"{d.post_score:.3f}"
            lines.append(
                f"| {d.task_id} | {d.pre_status or '-'} | "
                f"{d.post_status or '-'} | {pre_score} | {post_score} | "
                f"{d.score_delta:+.3f} | {attribs} |"
            )
        if regressions:
            lines.append("")
            lines.append(f"### Regressions in {benchmark}")
            lines.append("")
            for d in regressions:
                lines.append(
                    f"- **{d.task_id}**: "
                    f"{d.pre_status} → {d.post_status}, "
                    f"score {d.pre_score:.3f} → {d.post_score:.3f} "
                    f"({d.score_delta:+.3f}). "
                    f"Attribution: {', '.join(d.attributions) or 'unattributed'}."
                )
                for attr in d.attributions:
                    overall_attribution[attr] = overall_attribution.get(attr, 0) + 1
        for attr in (
            "tasks_added" if added else None,
            "tasks_removed" if removed else None,
        ):
            if attr is not None:
                overall_attribution[attr] = overall_attribution.get(attr, 0) + (
                    len(added) if attr == "tasks_added" else len(removed)
                )
        lines.append("")
    lines.append("## Attribution summary")
    lines.append("")
    if overall_attribution:
        lines.append("| attribution | count |")
        lines.append("|---|---|")
        for attr, count in sorted(overall_attribution.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"| {attr} | {count} |")
    else:
        lines.append("_No regressions or task-set changes detected on this slice._")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decisive-Edge B.9 — smoke regression diff. Compare two "
            "benchmark report directories produced by "
            "scripts/run_benchmark_sweep.sh and emit REGRESSION_DIFF.md."
        ),
    )
    parser.add_argument(
        "--pre",
        type=Path,
        required=True,
        help="Pre-remediation benchmark output dir (or single JSON report).",
    )
    parser.add_argument(
        "--post",
        type=Path,
        required=True,
        help="Post-remediation benchmark output dir (or single JSON report).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=("Path to write REGRESSION_DIFF.md. Defaults to <post>/REGRESSION_DIFF.md."),
    )
    parser.add_argument(
        "--threshold-pp",
        type=float,
        default=3.0,
        help=(
            "Regression threshold in percentage points. Tasks whose score "
            "drops by more than this are flagged. Default 3.0."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    pre_reports = _discover_report_files(args.pre)
    post_reports = _discover_report_files(args.post)
    if not pre_reports:
        sys.stderr.write(
            f"[smoke_regression_diff] no benchmark_report.json found under {args.pre}\n"
        )
        return 2
    if not post_reports:
        sys.stderr.write(
            f"[smoke_regression_diff] no benchmark_report.json found under {args.post}\n"
        )
        return 2
    benchmarks = sorted(set(pre_reports) | set(post_reports))
    per_benchmark: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        pre_path = pre_reports.get(benchmark)
        post_path = post_reports.get(benchmark)
        if pre_path is None or post_path is None:
            sys.stderr.write(
                f"[smoke_regression_diff] benchmark {benchmark!r} present in "
                f"only one side; skipping diff.\n"
            )
            continue
        pre_report = _load_report(pre_path)
        post_report = _load_report(post_path)
        per_benchmark.append(
            diff_benchmark(
                benchmark=benchmark,
                pre_report=pre_report,
                post_report=post_report,
                threshold_pp=args.threshold_pp,
            )
        )
    if not per_benchmark:
        sys.stderr.write("[smoke_regression_diff] no overlapping benchmarks; nothing to diff.\n")
        return 2
    report_text = render_report(
        per_benchmark=per_benchmark,
        threshold_pp=args.threshold_pp,
    )
    output_path = args.output
    if output_path is None:
        post_root = args.post if args.post.is_dir() else args.post.parent
        output_path = post_root / "REGRESSION_DIFF.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    print(f"[smoke_regression_diff] wrote {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
