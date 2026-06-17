#!/usr/bin/env python3
"""Decisive-Edge B.8 — A/B harness for the v1 vs v2 agent prompt module.

Compares ``RolloutConfig.prompts_version = "v1"`` (the historic free-prose
prompts in :pymod:`apex.agents.prompts`) against ``"v2"`` (the structured
rewrite in :pymod:`apex.agents.prompts_v2`). Mirrors the wiring of
``apex/scripts/ab_masai_in_cli.py`` so operators can run both A/Bs in
parallel, with the same default 30-task Commit0-Lite slice.

For every task the harness invokes the configured orchestrator twice —
once per arm — under separate ``output_dir`` roots, captures each arm's
``apex_result.json``, and aggregates per-task and per-agent deltas:

  * Per-task win rate (which arm produced a higher overall score).
  * Per-task score delta.
  * Per-agent (Reproducer / Localizer / Patcher / TestWriter) mean turn
    count delta — extracted from the rollout trajectories.
  * Per-agent verdict: keep v1 / promote v2 / inconclusive.

The script writes ``PROMPTS_AB_REPORT.md`` to ``--output-dir`` and prints
the recommended winner. It does NOT execute the A/B itself in this commit
— the operator runs it under their preferred conditions. See ``--dry-run``
for a no-op preview.

Like ``ab_masai_in_cli.py``, this harness uses CLI agents (claude / codex
/ gemini / opencode) — there is no API key check. The orchestrator
callable already routes through the configured CLI backend; we only set
``RolloutConfig.prompts_version`` per arm.

Example::

    python apex/scripts/ab_prompts.py \\
        --task-id matplotlib__matplotlib-26011 \\
        --output-dir runs/prompts_ab_20260514

    python apex/scripts/ab_prompts.py --dry-run   # preview the planned A/B
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

# Allow ``python apex/scripts/ab_prompts.py`` invocations.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Defaults — kept in lockstep with apex/scripts/ab_masai_in_cli.py so the
# two A/Bs can be cross-compared on the same task slice.
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


_AB_ARMS: tuple[str, ...] = ("v1", "v2")
_AGENTS: tuple[str, ...] = ("reproducer", "localizer", "patcher", "test_writer")


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


def _per_agent_turn_counts(payload: Any) -> dict[str, int]:
    """Per-stage turn count from rollout trajectories.

    Mirrors :pyfunc:`apex.scripts.ab_masai_in_cli._patcher_turns` but
    bucketised by stage so we can compare per-agent token / turn deltas.
    """
    out = {agent: 0 for agent in _AGENTS}
    if not isinstance(payload, dict):
        return out
    rollouts = payload.get("rollouts") or payload.get("rollout_results") or []
    if not isinstance(rollouts, list):
        return out
    for rollout in rollouts:
        if not isinstance(rollout, dict):
            continue
        traj = rollout.get("trajectory") or rollout.get("rollout_trajectory") or []
        if not isinstance(traj, list):
            continue
        for entry in traj:
            if not isinstance(entry, dict):
                continue
            stage = str(entry.get("stage") or "").strip().lower()
            if stage not in out:
                continue
            rounds = entry.get("rounds_used")
            if isinstance(rounds, (int, float)) and rounds > 0:
                out[stage] += int(rounds)
                continue
            iters = entry.get("iterations_used")
            if isinstance(iters, (int, float)) and iters > 0:
                out[stage] += int(iters)
                continue
            iterations = entry.get("iterations")
            if isinstance(iterations, list):
                out[stage] += len(iterations)
    return out


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
    per_agent_turns: dict[str, int] = field(default_factory=dict)


@dataclass
class TaskComparison:
    task_id: str
    arms: dict[str, ArmResult] = field(default_factory=dict)

    def score_delta(self) -> float:
        v1 = self.arms.get("v1")
        v2 = self.arms.get("v2")
        if v1 is None or v2 is None:
            return 0.0
        return v2.score - v1.score

    def winner(self) -> str:
        delta = self.score_delta()
        if delta > 1e-6:
            return "v2"
        if delta < -1e-6:
            return "v1"
        return "tie"

    def per_agent_turn_delta(self) -> dict[str, int]:
        v1 = self.arms.get("v1")
        v2 = self.arms.get("v2")
        if v1 is None or v2 is None:
            return {agent: 0 for agent in _AGENTS}
        return {
            agent: v2.per_agent_turns.get(agent, 0) - v1.per_agent_turns.get(agent, 0)
            for agent in _AGENTS
        }


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
    rollout_overrides = dict(payload.get("rollout") or {})
    rollout_overrides["prompts_version"] = arm
    payload["rollout"] = rollout_overrides
    started = time.time()
    try:
        orchestrator(payload)
    except Exception as exc:  # noqa: BLE001 - capture but don't crash A/B
        sys.stderr.write(
            f"[ab_prompts] arm={arm} task={task_payload.get('task_id')} "
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
        per_agent_turns=_per_agent_turn_counts(apex_result),
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
    v2_wins = sum(1 for w in winners if w == "v2")
    v1_wins = sum(1 for w in winners if w == "v1")
    ties = sum(1 for w in winners if w == "tie")
    per_agent_deltas: dict[str, list[int]] = {agent: [] for agent in _AGENTS}
    for c in comparisons:
        delta = c.per_agent_turn_delta()
        for agent in _AGENTS:
            per_agent_deltas[agent].append(delta[agent])
    per_agent_summary: dict[str, dict[str, float]] = {}
    for agent, vals in per_agent_deltas.items():
        per_agent_summary[agent] = {
            "mean_turn_delta": (statistics.fmean(vals) if vals else 0.0),
            "median_turn_delta": (statistics.median(vals) if vals else 0.0),
        }
    return {
        "task_count": n,
        "v2_wins": v2_wins,
        "v1_wins": v1_wins,
        "ties": ties,
        "v2_win_rate": (v2_wins / n) if n else 0.0,
        "v1_win_rate": (v1_wins / n) if n else 0.0,
        "tie_rate": (ties / n) if n else 0.0,
        "mean_score_delta": (statistics.fmean(deltas) if deltas else 0.0),
        "median_score_delta": (statistics.median(deltas) if deltas else 0.0),
        "per_agent": per_agent_summary,
    }


def _recommend_winner(summary: dict[str, Any]) -> str:
    """Top-line recommendation: keep_v1 / promote_v2 / inconclusive."""
    if summary["task_count"] == 0:
        return "no_data"
    advantage = summary["v2_wins"] - summary["v1_wins"]
    delta = summary["mean_score_delta"]
    if advantage > 0 and delta >= -1e-3:
        return "promote_v2"
    if advantage < 0 or delta < -0.02:
        return "keep_v1"
    return "inconclusive"


def _per_agent_recommendations(summary: dict[str, Any]) -> dict[str, str]:
    """Per-agent: did v2 cut turns substantially?

    A reduction in turn count is taken as a positive signal for v2 — the
    more-structured prompt should drive the agent to its submission tool
    sooner. A material increase in turn count without a paired score
    bump signals v2 is verbose without payoff.
    """
    out: dict[str, str] = {}
    per_agent = summary.get("per_agent") or {}
    score_advantage = summary.get("v2_wins", 0) - summary.get("v1_wins", 0)
    for agent in _AGENTS:
        stats = per_agent.get(agent) or {}
        delta = float(stats.get("mean_turn_delta") or 0.0)
        # Convergence vs. wandering. -1 turn = clear v2 win for this agent.
        if delta <= -1.0 and score_advantage >= 0:
            out[agent] = "promote_v2"
        elif delta >= 1.0 and score_advantage <= 0:
            out[agent] = "keep_v1"
        else:
            out[agent] = "inconclusive"
    return out


def _render_report(
    *,
    comparisons: list[TaskComparison],
    summary: dict[str, Any],
    output_dir: Path,
) -> Path:
    recommended = _recommend_winner(summary)
    per_agent_recs = _per_agent_recommendations(summary)
    lines = [
        "# Prompts v1 vs v2 A/B Report",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## Summary",
        "",
        f"- Tasks compared: **{summary['task_count']}**",
        f"- v2 wins: **{summary['v2_wins']}** ({summary['v2_win_rate']:.1%})",
        f"- v1 wins: **{summary['v1_wins']}** ({summary['v1_win_rate']:.1%})",
        f"- ties: **{summary['ties']}** ({summary['tie_rate']:.1%})",
        f"- mean Δ(score) [v2 − v1]: **{summary['mean_score_delta']:+.4f}**",
        f"- median Δ(score): **{summary['median_score_delta']:+.4f}**",
        "",
        f"## Top-line recommendation: `{recommended}`",
        "",
    ]
    if recommended == "promote_v2":
        lines.append(
            "v2 wins on this slice. Flip "
            '`RolloutConfig.prompts_version = "v2"` in your default config.'
        )
    elif recommended == "keep_v1":
        lines.append(
            'v1 holds. Keep `RolloutConfig.prompts_version = "v1"` '
            "(the current default) and audit the v2 deltas before re-running."
        )
    elif recommended == "inconclusive":
        lines.append(
            "Inconclusive. Re-run with a larger task slice or stratify "
            "by repo to see if v2 helps specific repo families."
        )
    else:
        lines.append("No data — re-run the harness with at least one task.")
    lines.extend(
        [
            "",
            "## Per-agent recommendations",
            "",
            "| agent | recommendation | mean Δ(turns) | median Δ(turns) |",
            "|---|---|---|---|",
        ]
    )
    per_agent = summary.get("per_agent") or {}
    for agent in _AGENTS:
        stats = per_agent.get(agent) or {}
        lines.append(
            f"| {agent} | `{per_agent_recs.get(agent, 'inconclusive')}` | "
            f"{stats.get('mean_turn_delta', 0.0):+.2f} | "
            f"{stats.get('median_turn_delta', 0.0):+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Per-task breakdown",
            "",
            "| task_id | winner | score(v1) | score(v2) | Δ | turns(v1→v2) |",
            "|---|---|---|---|---|---|",
        ]
    )
    for comparison in comparisons:
        v1 = comparison.arms.get("v1")
        v2 = comparison.arms.get("v2")
        v1_score = v1.score if v1 else float("nan")
        v2_score = v2.score if v2 else float("nan")
        v1_total = sum((v1.per_agent_turns or {}).values()) if v1 else 0
        v2_total = sum((v2.per_agent_turns or {}).values()) if v2 else 0
        lines.append(
            f"| {comparison.task_id} | {comparison.winner()} | "
            f"{v1_score:.3f} | {v2_score:.3f} | "
            f"{comparison.score_delta():+.3f} | {v1_total}→{v2_total} |"
        )
    report_path = output_dir / "PROMPTS_AB_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decisive-Edge B.8 A/B harness comparing prompts_version "
            "(v1 vs v2). Produces PROMPTS_AB_REPORT.md."
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
        help="Newline-delimited file of task ids; combined with --task-id.",
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
        help="Directory for PROMPTS_AB_REPORT.md and per-arm artifacts.",
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
        output_dir = Path(tempfile.mkdtemp(prefix="apex_prompts_ab_"))
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ab_prompts] arms={list(_AB_ARMS)} tasks={len(task_ids)} output_dir={output_dir}")
    if args.dry_run:
        print(f"[ab_prompts] dry-run: orchestrator={args.orchestrator_callable}")
        for task_id in task_ids:
            print(f"  - {task_id}")
        print("[ab_prompts] dry-run complete. Drop --dry-run to execute the A/B.")
        return 0
    orchestrator = _resolve_callable(args.orchestrator_callable)
    comparisons: list[TaskComparison] = []
    for task_id in task_ids:
        print(f"[ab_prompts] task={task_id}")
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
    print(f"[ab_prompts] report: {report_path}")
    print("[ab_prompts] recommended winner: " + _recommend_winner(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
