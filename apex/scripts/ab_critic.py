#!/usr/bin/env python3
"""Decisive-Edge C.4 — A/B harness for the LLM critic vs. verifier-only ranking.

Compares ``SelectionConfig.use_critic = True`` (the historic default that
shipped with the published 86.3% Commit0-Lite headline) against
``use_critic = False`` (verifier-only ranking: pass-rate then lowest
test-edit count). Mirrors the wiring of ``apex/scripts/ab_prompts.py`` and
``apex/scripts/ab_masai_in_cli.py`` so operators can run the three A/Bs in
parallel on the same task slice.

For every task the harness invokes the configured orchestrator twice —
once per arm — under separate ``output_dir`` roots, captures each arm's
``apex_result.json``, and aggregates per-task and overall deltas:

  * Per-task win rate (which arm produced a higher overall score).
  * Per-task score delta.
  * Top-line recommendation: keep the critic if it beats verifier-only by
    >=3pp on win rate AND non-negative mean score delta.

Like the other A/Bs, this harness uses CLI agents (claude / codex /
gemini / opencode) — there is no API key check. The orchestrator
callable already routes through the configured CLI backend; we only set
``SelectionConfig.use_critic`` per arm.

Example::

    python apex/scripts/ab_critic.py \\
        --task-id babel \\
        --output-dir runs/critic_ab_20260514

    python apex/scripts/ab_critic.py --dry-run    # preview the planned A/B

The script writes ``CRITIC_AB_REPORT.md`` to ``--output-dir`` and prints
the recommended winner. It does NOT execute the A/B itself in this commit
— the operator runs it under their preferred conditions.
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

# Allow ``python apex/scripts/ab_critic.py`` invocations.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Defaults — kept in lockstep with apex/scripts/ab_prompts.py and
# apex/scripts/ab_masai_in_cli.py so the three A/Bs share a comparable
# 30-task Commit0-Lite slice.
# ---------------------------------------------------------------------------


_DEFAULT_COMMIT0_LITE_SLICE: tuple[str, ...] = (
    "babel",
    "tinydb",
    "wcwidth",
    "voluptuous",
    "imapclient",
    "joblib",
    "pypdf",
    "minitorch",
    "click",
    "marshmallow",
    "jinja2",
    "rich",
    "fabric",
    "loguru",
    "more-itertools",
    "pandas-stubs",
    "pendulum",
    "python-progressbar",
    "requests",
    "structlog",
    "tabulate",
    "tomli",
    "typer",
    "urllib3",
    "websockets",
    "xlsxwriter",
    "yarl",
    "zarr",
    "zipp",
    "asyncpg",
)


_AB_ARMS: tuple[str, ...] = ("verifier_only", "critic")
# Critic must beat verifier-only by this many percentage points (win rate)
# AND show a non-negative mean score delta to be retained as default.
_KEEP_CRITIC_WIN_RATE_MARGIN_PP: float = 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_callable(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError(f"orchestrator callable must be 'module:attr', got {spec!r}")
    module_name, attr_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    target = getattr(module, attr_name, None)
    if target is None or not callable(target):
        raise ValueError(f"orchestrator callable {spec!r} is not callable")
    return target


def _load_apex_result(output_dir: Path) -> dict[str, Any]:
    candidate = output_dir / "apex_result.json"
    if not candidate.exists():
        return {}
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _outcome_score(payload: Any) -> float:
    if not isinstance(payload, dict):
        return 0.0
    final = payload.get("final")
    if isinstance(final, dict):
        for key in ("overall_score", "pass_rate", "required_pass_rate", "score"):
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


def _critic_was_called(payload: Any) -> bool:
    """Read selection_diagnostics → critic on the selected rollout summary.

    Used to verify that the verifier_only arm really ran without the
    critic (the diagnostics block omits the ``critic`` sub-object when
    the gate is off). Returns False when the diagnostics block is
    missing entirely.
    """
    if not isinstance(payload, dict):
        return False
    summaries = payload.get("rollout_summaries") or payload.get("rollouts") or []
    if not isinstance(summaries, list):
        return False
    selected_id = payload.get("selected_rollout_id")
    selected_summary: Optional[dict[str, Any]] = None
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        if selected_id is not None and summary.get("rollout_id") == selected_id:
            selected_summary = summary
            break
    if selected_summary is None and summaries:
        candidate = summaries[0]
        if isinstance(candidate, dict):
            selected_summary = candidate
    if not isinstance(selected_summary, dict):
        return False
    diagnostics = selected_summary.get("selection_diagnostics")
    if not isinstance(diagnostics, dict):
        return False
    return "critic" in diagnostics


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ArmResult:
    arm: str
    output_dir: Path
    duration_seconds: float
    apex_result: dict[str, Any]
    score: float = 0.0
    critic_observed: bool = False


@dataclass
class TaskComparison:
    task_id: str
    arms: dict[str, ArmResult] = field(default_factory=dict)

    def score_delta(self) -> float:
        """``critic - verifier_only``; positive = critic helped."""
        critic = self.arms.get("critic")
        baseline = self.arms.get("verifier_only")
        if critic is None or baseline is None:
            return 0.0
        return critic.score - baseline.score

    def winner(self) -> str:
        delta = self.score_delta()
        if delta > 1e-6:
            return "critic"
        if delta < -1e-6:
            return "verifier_only"
        return "tie"


# ---------------------------------------------------------------------------
# Arm execution
# ---------------------------------------------------------------------------


def _build_task_payload(task_id: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "benchmark": "commit0_lite",
    }


def _run_arm(
    *,
    arm: str,
    task_payload: dict[str, Any],
    orchestrator: Callable[..., Any],
    work_root: Path,
) -> ArmResult:
    output_dir = work_root / arm
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(task_payload)
    payload["output_dir"] = str(output_dir)
    selection_overrides = dict(payload.get("selection") or {})
    selection_overrides["use_critic"] = arm == "critic"
    payload["selection"] = selection_overrides
    started = time.time()
    try:
        orchestrator(payload)
    except Exception as exc:  # noqa: BLE001 - capture but don't crash A/B
        sys.stderr.write(
            f"[ab_critic] arm={arm} task={task_payload.get('task_id')} "
            f"raised {type(exc).__name__}: {exc}\n"
        )
    duration = time.time() - started
    apex_result = _load_apex_result(output_dir)
    return ArmResult(
        arm=arm,
        output_dir=output_dir,
        duration_seconds=duration,
        apex_result=apex_result,
        score=_outcome_score(apex_result),
        critic_observed=_critic_was_called(apex_result),
    )


def _run_task(
    *,
    task_id: str,
    task_payload: dict[str, Any],
    orchestrator: Callable[..., Any],
    work_root: Path,
) -> TaskComparison:
    comparison = TaskComparison(task_id=task_id)
    for arm in _AB_ARMS:
        arm_root = work_root / task_id / arm
        arm_root.mkdir(parents=True, exist_ok=True)
        comparison.arms[arm] = _run_arm(
            arm=arm,
            task_payload=task_payload,
            orchestrator=orchestrator,
            work_root=arm_root,
        )
    return comparison


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _aggregate(comparisons: list[TaskComparison]) -> dict[str, Any]:
    deltas = [c.score_delta() for c in comparisons]
    winners = [c.winner() for c in comparisons]
    n = len(comparisons)
    critic_wins = sum(1 for w in winners if w == "critic")
    verifier_wins = sum(1 for w in winners if w == "verifier_only")
    ties = sum(1 for w in winners if w == "tie")
    return {
        "task_count": n,
        "critic_wins": critic_wins,
        "verifier_only_wins": verifier_wins,
        "ties": ties,
        "critic_win_rate": (critic_wins / n) if n else 0.0,
        "verifier_only_win_rate": (verifier_wins / n) if n else 0.0,
        "tie_rate": (ties / n) if n else 0.0,
        "mean_score_delta": (statistics.fmean(deltas) if deltas else 0.0),
        "median_score_delta": (statistics.median(deltas) if deltas else 0.0),
    }


def _recommend_winner(summary: dict[str, Any]) -> str:
    """Top-line recommendation: keep_critic / disable_critic / inconclusive.

    ``keep_critic`` requires:
      * critic_wins beats verifier_only_wins by ``_KEEP_CRITIC_WIN_RATE_MARGIN_PP``
        percentage points or more
      * mean_score_delta is non-negative

    ``disable_critic`` fires when verifier_only outright wins on absolute
    count OR mean_score_delta dips below -2pp.
    """
    if summary["task_count"] == 0:
        return "no_data"
    n = max(int(summary["task_count"]), 1)
    advantage_pp = 100.0 * (summary["critic_wins"] - summary["verifier_only_wins"]) / n
    delta = float(summary["mean_score_delta"])
    if advantage_pp >= _KEEP_CRITIC_WIN_RATE_MARGIN_PP and delta >= -1e-3:
        return "keep_critic"
    if advantage_pp <= -1e-6 or delta < -0.02:
        return "disable_critic"
    return "inconclusive"


def _render_report(
    *,
    comparisons: list[TaskComparison],
    summary: dict[str, Any],
    output_dir: Path,
) -> Path:
    recommended = _recommend_winner(summary)
    lines = [
        "# LLM Critic vs Verifier-Only A/B Report",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## Summary",
        "",
        f"- Tasks compared: **{summary['task_count']}**",
        f"- critic wins: **{summary['critic_wins']}** ({summary['critic_win_rate']:.1%})",
        f"- verifier_only wins: **{summary['verifier_only_wins']}** "
        f"({summary['verifier_only_win_rate']:.1%})",
        f"- ties: **{summary['ties']}** ({summary['tie_rate']:.1%})",
        f"- mean Δ(score) [critic − verifier_only]: **{summary['mean_score_delta']:+.4f}**",
        f"- median Δ(score): **{summary['median_score_delta']:+.4f}**",
        "",
        f"## Top-line recommendation: `{recommended}`",
        "",
    ]
    if recommended == "keep_critic":
        lines.append(
            "Critic helps on this slice (advantage ≥ "
            f"{_KEEP_CRITIC_WIN_RATE_MARGIN_PP:.0f}pp on win rate AND mean Δ "
            "≥ 0). Keep ``SelectionConfig.use_critic = True`` (the current "
            "default)."
        )
    elif recommended == "disable_critic":
        lines.append(
            "Verifier-only wins. Set "
            "``SelectionConfig.use_critic = False`` in your config to "
            "skip the LLM critic call (cost win) and rely on the "
            "verifier alone."
        )
    elif recommended == "inconclusive":
        lines.append(
            "Inconclusive — the critic neither helps nor hurts measurably. "
            "Re-run with a larger task slice or stratify by repo to see "
            "if the critic helps specific repo families."
        )
    else:
        lines.append("No data — re-run the harness with at least one task.")
    lines.extend(
        [
            "",
            "## Per-task breakdown",
            "",
            "| task_id | winner | score(verifier_only) | score(critic) | Δ | critic_observed |",
            "|---|---|---|---|---|---|",
        ]
    )
    for comparison in comparisons:
        baseline = comparison.arms.get("verifier_only")
        critic = comparison.arms.get("critic")
        baseline_score = baseline.score if baseline else float("nan")
        critic_score = critic.score if critic else float("nan")
        critic_observed = critic.critic_observed if critic else False
        lines.append(
            f"| {comparison.task_id} | {comparison.winner()} | "
            f"{baseline_score:.3f} | {critic_score:.3f} | "
            f"{comparison.score_delta():+.3f} | {'yes' if critic_observed else 'no'} |"
        )
    report_path = output_dir / "CRITIC_AB_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decisive-Edge C.4 A/B harness comparing the LLM critic "
            "against verifier-only ranking. Produces CRITIC_AB_REPORT.md."
        ),
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help=(
            "Commit0 task id to evaluate. Repeat for each task. When "
            "omitted, the default 30-task Commit0-Lite slice is used."
        ),
    )
    parser.add_argument(
        "--task-list-file",
        type=Path,
        default=None,
        help=(
            "Newline-delimited file of task ids; combined with --task-id. "
            "Lines starting with '#' are treated as comments."
        ),
    )
    parser.add_argument(
        "--task-payload-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON file overriding the {task_id, benchmark} payload "
            "— useful when targeting a non-Commit0 benchmark."
        ),
    )
    parser.add_argument(
        "--orchestrator-callable",
        default="apex.orchestrator:run_task_dict",
        help=("module:attr orchestrator entry point. Defaults to apex.orchestrator:run_task_dict."),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CRITIC_AB_REPORT.md and per-arm artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Resolve the orchestrator + task list and print the planned "
            "execution without invoking the orchestrator."
        ),
    )
    return parser.parse_args(argv)


def _resolve_task_ids(args: argparse.Namespace) -> list[str]:
    ids: list[str] = list(args.task_id or [])
    if args.task_list_file is not None:
        text = args.task_list_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                ids.append(stripped)
    if not ids:
        ids = list(_DEFAULT_COMMIT0_LITE_SLICE)
    seen: set[str] = set()
    deduped: list[str] = []
    for task_id in ids:
        if task_id not in seen:
            seen.add(task_id)
            deduped.append(task_id)
    return deduped


def _resolve_task_payload(
    args: argparse.Namespace,
    task_id: str,
) -> dict[str, Any]:
    if args.task_payload_file is None:
        return _build_task_payload(task_id)
    template = json.loads(args.task_payload_file.read_text(encoding="utf-8"))
    if not isinstance(template, dict):
        raise ValueError(
            f"--task-payload-file must contain a JSON object; got {type(template).__name__}"
        )
    payload = dict(template)
    payload["task_id"] = task_id
    payload.setdefault("benchmark", "commit0_lite")
    return payload


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    task_ids = _resolve_task_ids(args)
    if args.output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="apex_critic_ab_"))
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ab_critic] arms={list(_AB_ARMS)} tasks={len(task_ids)} output_dir={output_dir}")
    if args.dry_run:
        print(f"[ab_critic] dry-run: orchestrator={args.orchestrator_callable}")
        for task_id in task_ids:
            print(f"  - {task_id}")
        print("[ab_critic] dry-run complete. Drop --dry-run to execute the A/B.")
        return 0
    orchestrator = _resolve_callable(args.orchestrator_callable)
    comparisons: list[TaskComparison] = []
    for task_id in task_ids:
        print(f"[ab_critic] task={task_id}")
        payload = _resolve_task_payload(args, task_id)
        comparison = _run_task(
            task_id=task_id,
            task_payload=payload,
            orchestrator=orchestrator,
            work_root=output_dir,
        )
        comparisons.append(comparison)
    summary = _aggregate(comparisons)
    report_path = _render_report(
        comparisons=comparisons,
        summary=summary,
        output_dir=output_dir,
    )
    print(f"[ab_critic] report: {report_path}")
    print("[ab_critic] recommended winner: " + _recommend_winner(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
