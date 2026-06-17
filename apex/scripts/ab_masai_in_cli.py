#!/usr/bin/env python3
"""Phase 3.5 — A/B harness for the CLI MASAI pre-round mode.

Compares ``RolloutConfig.cli_agent_use_masai_preround = "off"`` (the
historic CLI behaviour where the agent goes straight to patching) against
``"structured_prompt"`` (the new default introduced in Phase 3.5, where
Reproducer + Localizer pre-rounds run before the patcher and inject a
``# Grounded Context`` YAML block into the patcher's prompt).

For every task the harness invokes the configured orchestrator twice —
once per arm — under separate ``output_dir`` roots, captures each side's
``apex_result.json``, and aggregates:

  * Per-task win rate (which arm produced a higher overall score).
  * Per-task delta on overall score.
  * Mean turns saved (drops in patcher iterations / rounds_used) when the
    pre-round eliminates work the patcher would have spent localising.
  * Pre-round latency cost (extra wall-clock seconds spent on Reproducer +
    Localizer in the structured arm).

The script writes ``MASAI_AB_REPORT.md`` to ``--output-dir`` and prints
the recommended winner. It does NOT execute the A/B itself in this commit
— it provides the scaffold + a default 30-task Commit0-Lite slice; the
operator runs the A/B under the conditions of their choosing.

Example::

    python apex/scripts/ab_masai_in_cli.py \\
        --task-id matplotlib__matplotlib-26011 \\
        --task-id django__django-15400 \\
        --output-dir runs/masai_ab_20260514

By default the harness targets ``apex.orchestrator:run_task_dict`` (the
production CLI entry point). Override with ``--orchestrator-callable`` for
testing.
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

# Allow ``python apex/scripts/ab_masai_in_cli.py`` invocations.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# A 30-task Commit0-Lite slice — covers babel / joblib / pypdf /
# imapclient / voluptuous (env-overrides-required) plus a handful of
# stable / well-trodden tasks. Operator can swap with --task-id /
# --task-list-file flags. Order is intentionally diverse to amortise
# preround cost across short / long tasks.
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


_AB_ARMS: tuple[str, ...] = ("off", "structured_prompt")


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
    """Extract the comparison score from an apex_result.json payload.

    Mirrors :pyfunc:`apex.scripts.ab_controller_policy._outcome_score` so
    the two A/B harnesses surface comparable numbers.
    """
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


def _patcher_turns(payload: Any) -> int:
    """Heuristic turn-count: rounds + iterations across patcher trajectories."""
    if not isinstance(payload, dict):
        return 0
    total = 0
    rollouts = payload.get("rollouts") or payload.get("rollout_results") or []
    if not isinstance(rollouts, list):
        return 0
    for rollout in rollouts:
        if not isinstance(rollout, dict):
            continue
        traj = rollout.get("trajectory") or rollout.get("rollout_trajectory") or []
        if not isinstance(traj, list):
            continue
        for entry in traj:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("stage") or "").strip() != "patcher":
                continue
            rounds = entry.get("rounds_used")
            if isinstance(rounds, (int, float)) and rounds > 0:
                total += int(rounds)
                continue
            iters = entry.get("iterations_used")
            if isinstance(iters, (int, float)) and iters > 0:
                total += int(iters)
                continue
            iterations = entry.get("iterations")
            if isinstance(iterations, list):
                total += len(iterations)
    return total


def _preround_latency_seconds(payload: Any) -> float:
    """Sum the structured-arm pre-round latency from rollout trajectories."""
    if not isinstance(payload, dict):
        return 0.0
    total = 0.0
    rollouts = payload.get("rollouts") or payload.get("rollout_results") or []
    if not isinstance(rollouts, list):
        return 0.0
    for rollout in rollouts:
        if not isinstance(rollout, dict):
            continue
        traj = rollout.get("trajectory") or rollout.get("rollout_trajectory") or []
        if not isinstance(traj, list):
            continue
        for entry in traj:
            if not isinstance(entry, dict):
                continue
            preround = entry.get("masai_preround")
            if not isinstance(preround, dict):
                continue
            for key in ("reproducer_duration_seconds", "localizer_duration_seconds"):
                value = preround.get(key)
                if isinstance(value, (int, float)) and value > 0:
                    total += float(value)
    return total


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
    patcher_turns: int = 0
    preround_latency_seconds: float = 0.0


@dataclass
class TaskComparison:
    task_id: str
    arms: dict[str, ArmResult] = field(default_factory=dict)

    def score_delta(self) -> float:
        off = self.arms.get("off")
        structured = self.arms.get("structured_prompt")
        if off is None or structured is None:
            return 0.0
        return structured.score - off.score

    def winner(self) -> str:
        delta = self.score_delta()
        if delta > 1e-6:
            return "structured_prompt"
        if delta < -1e-6:
            return "off"
        return "tie"

    def turns_saved(self) -> int:
        off = self.arms.get("off")
        structured = self.arms.get("structured_prompt")
        if off is None or structured is None:
            return 0
        return off.patcher_turns - structured.patcher_turns


# ---------------------------------------------------------------------------
# Arm execution
# ---------------------------------------------------------------------------


def _build_task_payload(task_id: str) -> dict[str, Any]:
    """Build the orchestrator input payload for one Commit0 task.

    The default shape matches ``apex.orchestrator.run_task_dict`` —
    customise via ``--task-payload-file`` when running against a non-
    Commit0 benchmark.
    """
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
    rollout_overrides = dict(payload.get("rollout") or {})
    rollout_overrides["cli_agent_use_masai_preround"] = arm
    payload["rollout"] = rollout_overrides
    started = time.time()
    try:
        orchestrator(payload)
    except Exception as exc:  # noqa: BLE001 - capture but don't crash the A/B
        sys.stderr.write(
            f"[ab_masai_in_cli] arm={arm} task={task_payload.get('task_id')} "
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
        patcher_turns=_patcher_turns(apex_result),
        preround_latency_seconds=_preround_latency_seconds(apex_result),
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
    turns_saved = [c.turns_saved() for c in comparisons]
    preround_latencies = [
        (
            c.arms["structured_prompt"].preround_latency_seconds
            if "structured_prompt" in c.arms
            else 0.0
        )
        for c in comparisons
    ]
    winners = [c.winner() for c in comparisons]
    n = len(comparisons)
    structured_wins = sum(1 for w in winners if w == "structured_prompt")
    off_wins = sum(1 for w in winners if w == "off")
    ties = sum(1 for w in winners if w == "tie")
    return {
        "task_count": n,
        "structured_wins": structured_wins,
        "off_wins": off_wins,
        "ties": ties,
        "structured_win_rate": (structured_wins / n) if n else 0.0,
        "off_win_rate": (off_wins / n) if n else 0.0,
        "tie_rate": (ties / n) if n else 0.0,
        "mean_score_delta": (statistics.fmean(deltas) if deltas else 0.0),
        "median_score_delta": (statistics.median(deltas) if deltas else 0.0),
        "mean_turns_saved": (statistics.fmean(turns_saved) if turns_saved else 0.0),
        "mean_preround_latency_seconds": (
            statistics.fmean(preround_latencies) if preround_latencies else 0.0
        ),
        "max_preround_latency_seconds": (max(preround_latencies) if preround_latencies else 0.0),
    }


def _recommend_winner(summary: dict[str, Any]) -> str:
    if summary["task_count"] == 0:
        return "no_data"
    structured_advantage = summary["structured_wins"] - summary["off_wins"]
    delta = summary["mean_score_delta"]
    # Recommend structured_prompt only when it beats "off" by a clear
    # margin AND the mean score delta is non-negative. The pre-round
    # latency penalty is already priced into the score (longer rollouts
    # get fewer patcher iterations under a wallclock budget).
    if structured_advantage > 0 and delta >= -1e-3:
        return "structured_prompt"
    if structured_advantage < 0 or delta < -0.02:
        return "off"
    return "inconclusive"


def _render_report(
    *,
    comparisons: list[TaskComparison],
    summary: dict[str, Any],
    output_dir: Path,
) -> Path:
    recommended = _recommend_winner(summary)
    lines = [
        "# MASAI-in-CLI A/B Report",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## Summary",
        "",
        f"- Tasks compared: **{summary['task_count']}**",
        f"- structured_prompt wins: **{summary['structured_wins']}** ({summary['structured_win_rate']:.1%})",
        f"- off wins: **{summary['off_wins']}** ({summary['off_win_rate']:.1%})",
        f"- ties: **{summary['ties']}** ({summary['tie_rate']:.1%})",
        f"- mean Δ(score) [structured − off]: **{summary['mean_score_delta']:+.4f}**",
        f"- median Δ(score): **{summary['median_score_delta']:+.4f}**",
        f"- mean turns saved by structured arm: **{summary['mean_turns_saved']:+.2f}**",
        f"- mean pre-round latency cost: **{summary['mean_preround_latency_seconds']:.1f}s**",
        f"- max pre-round latency cost: **{summary['max_preround_latency_seconds']:.1f}s**",
        "",
        f"## Recommendation: `{recommended}`",
        "",
    ]
    if recommended == "structured_prompt":
        lines.append(
            "Pre-rounds win on this slice. Keep the Phase 3.5 default "
            '(`RolloutConfig.cli_agent_use_masai_preround = "structured_prompt"`).'
        )
    elif recommended == "off":
        lines.append(
            "Off mode wins on this slice. Flip the default by setting "
            '`RolloutConfig.cli_agent_use_masai_preround = "off"` in your config.'
        )
    elif recommended == "inconclusive":
        lines.append(
            "Inconclusive — neither arm shows a clear advantage. Re-run with a "
            "larger task slice or stratify by repo to see if the pre-round helps "
            "specific repo families."
        )
    else:
        lines.append("No data — re-run the harness with at least one task.")
    lines.extend(
        [
            "",
            "## Per-task breakdown",
            "",
            "| task_id | winner | score(off) | score(structured) | Δ | turns_off | turns_structured | preround_latency_s |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for comparison in comparisons:
        off = comparison.arms.get("off")
        structured = comparison.arms.get("structured_prompt")
        off_score = off.score if off else float("nan")
        struct_score = structured.score if structured else float("nan")
        off_turns = off.patcher_turns if off else 0
        struct_turns = structured.patcher_turns if structured else 0
        latency = structured.preround_latency_seconds if structured else 0.0
        lines.append(
            f"| {comparison.task_id} | {comparison.winner()} | "
            f"{off_score:.3f} | {struct_score:.3f} | "
            f"{comparison.score_delta():+.3f} | {off_turns} | {struct_turns} | "
            f"{latency:.1f} |"
        )
    report_path = output_dir / "MASAI_AB_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 3.5 A/B harness comparing CLI MASAI pre-round modes "
            "(off vs. structured_prompt). Produces MASAI_AB_REPORT.md."
        ),
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help=(
            "Commit0 task id to evaluate. Repeat the flag for each task. "
            "When omitted, the default 30-task Commit0-Lite slice is used."
        ),
    )
    parser.add_argument(
        "--task-list-file",
        type=Path,
        default=None,
        help=("Path to a newline-delimited file of task ids. Combined with any --task-id values."),
    )
    parser.add_argument(
        "--task-payload-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON file overriding the default {task_id, benchmark} "
            "task payload — useful when targeting a non-Commit0 benchmark."
        ),
    )
    parser.add_argument(
        "--orchestrator-callable",
        default="apex.orchestrator:run_task_dict",
        help=(
            "module:attr spec for the orchestrator entry point. Defaults to "
            "apex.orchestrator:run_task_dict (the production CLI). Override "
            "with a stub for harness testing."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to emit MASAI_AB_REPORT.md plus per-arm "
            "apex_result.json copies. Defaults to a tempdir."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Resolve the orchestrator + task list and print the planned "
            "execution without invoking the orchestrator. Useful for "
            "verifying the scaffold."
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
    # Deduplicate while preserving order.
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
        output_dir = Path(tempfile.mkdtemp(prefix="apex_masai_ab_"))
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ab_masai_in_cli] arms={list(_AB_ARMS)} tasks={len(task_ids)} output_dir={output_dir}")
    if args.dry_run:
        print(f"[ab_masai_in_cli] dry-run: orchestrator={args.orchestrator_callable}")
        for task_id in task_ids:
            print(f"  - {task_id}")
        print("[ab_masai_in_cli] dry-run complete. Drop --dry-run to execute the A/B.")
        return 0
    orchestrator = _resolve_callable(args.orchestrator_callable)
    comparisons: list[TaskComparison] = []
    for task_id in task_ids:
        print(f"[ab_masai_in_cli] task={task_id}")
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
    print(f"[ab_masai_in_cli] report: {report_path}")
    print("[ab_masai_in_cli] recommended winner: " + _recommend_winner(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
