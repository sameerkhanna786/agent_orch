#!/usr/bin/env python3
"""Decisive-Edge C.7 — A/B harness: APEX MASAI scaffolding vs raw native CoT.

Modern frontier models (Claude 4.7 Opus, GPT-5.5, Gemini 2.5 Pro) all
have native chain-of-thought built into the model. APEX's MASAI
scaffolding (Reproducer + Localizer + Patcher pipeline) was conceived
when models needed external scaffolding to keep their reasoning
consistent across multi-step tasks. C.7 asks the empirical question:

  "Does APEX's MASAI scaffolding still help, or does the model's
  native CoT subsume it?"

Per-model decision: it's plausible Claude 4.7 Opus benefits from MASAI
more than GPT-5.5 (or vice versa). The harness sweeps across a
configurable list of models and emits a per-model recommendation.

For every (model, task) pair the harness invokes the configured
orchestrator twice:

  * Run A — APEX MASAI scaffolded: ``cli_agent_use_masai_preround =
    "structured_prompt"`` + the Reproducer / Localizer / Patcher
    pipeline as currently shipped.
  * Run B — Raw native CoT: ``cli_agent_use_masai_preround = "off"``
    AND a single-shot prompt that asks the model to reason through
    reproduction, localization, fix planning, and patch generation in
    one pass and submit the unified diff via the submit_patch tool.
    The single-shot prompt template is embedded below.

Per-model recommendation:
  * MASAI wins if it beats raw by ≥3pp on win rate AND mean Δ ≥ 0.
  * Raw wins if it beats MASAI on absolute count AND mean Δ ≤ 0.
  * Otherwise inconclusive.

Like the other A/Bs, this harness uses CLI agents (claude / codex /
gemini / opencode) — there is no API key check. The orchestrator
callable already routes through the configured CLI backend; we set the
``model`` (per-arm prefix) and the relevant rollout flags per arm.

Example::

    python apex/scripts/ab_native_cot.py \\
        --model claude-4.7-opus \\
        --model gpt-5.5 \\
        --task-id babel \\
        --output-dir runs/cot_ab_20260514

    python apex/scripts/ab_native_cot.py --dry-run    # preview the planned A/B

The script writes ``NATIVE_COT_AB_REPORT.md`` to ``--output-dir`` and
prints the per-model recommendation. It does NOT execute the A/B itself
in this commit — the operator runs it under their preferred conditions.
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

# Allow ``python apex/scripts/ab_native_cot.py`` invocations.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Defaults — kept in lockstep with the other Decisive-Edge A/B harnesses.
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

# When the operator omits --model, we sweep the three frontier CLI models
# APEX currently routes to. Matches the model strings the agent backends
# accept; override with --model for per-task tuning.
_DEFAULT_MODELS: tuple[str, ...] = (
    "claude-4.7-opus",
    "gpt-5.5",
    "gemini-2.5-pro",
)

_AB_ARMS: tuple[str, ...] = ("masai", "raw_cot")
# MASAI must beat raw native CoT by this many percentage points (win rate)
# AND show a non-negative mean score delta to be retained as default.
_KEEP_MASAI_WIN_RATE_MARGIN_PP: float = 3.0


# The single-shot raw native CoT prompt. Inspired by B-γ's prompts_v2.py
# work — the prompt reads the issue, asks the model to reason through
# reproduction / localization / fix-planning / patch generation in ONE
# pass, and submit a unified diff via submit_patch. The model's native
# chain-of-thought is the only structuring mechanism — there are no
# pre-rounds, no intermediate submission tools, and no MASAI handoffs.
RAW_NATIVE_COT_PROMPT_TEMPLATE: str = """\
You are solving a real software engineering bug end-to-end. Your model has
native chain-of-thought reasoning; use it.

# Workflow (apply ALL stages, in ORDER, in your own reasoning trace):

  1. REPRODUCE — Read the issue. Identify the failing entry point
     (function / API / CLI subcommand / HTTP route). Run the existing
     test suite or a minimal command that exhibits the bug NOW.

  2. LOCALIZE — Trace from the failing entry point to the source files
     and symbols that own the broken behavior. Rank the candidates;
     name the function whose contract is violated.

  3. PLAN THE FIX — State, in your reasoning trace, the smallest
     edit that will make the failing tests pass without breaking the
     passing tests. Name the strategy axis: minimal_fix, refactor,
     defensive, isolated_helper, inverted_logic, two_step_decompose,
     or test_first_red_green.

  4. PATCH — Apply the edit. Re-run the targeted failing tests. If
     they pass, run the broader visible suite. If those pass, submit
     the unified diff via the ``submit_patch`` tool.

# Non-negotiables (the patch is rejected if any are violated):

  * The patch must compile — run ``python -m py_compile <file>`` (or
    the language equivalent) on every edited file. SyntaxError is
    worse than no patch.
  * Do NOT edit ``conftest.py``, ``pytest.ini``, ``tox.ini``,
    ``setup.cfg``, or any file under ``tests/`` / ``test/`` UNLESS
    the path is explicitly listed as ``incomplete_test_files`` in
    the issue.
  * Do NOT delete or weaken visible tests to make them pass.
  * Do NOT create scratch files at the repo root.

# Output envelope — when calling ``submit_patch``, include:

  * ``summary``: 1-3 sentences naming the bug + the fix.
  * ``changed_files``: list of paths you actually edited.
  * ``tests_run``: list of the test commands you executed.
  * ``confidence``: float in [0, 1].

# Issue

{issue_description}

# Test command (when provided)

{test_command_or_empty}

Begin reasoning now. Do all four stages in one continuous trace.
"""


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


def build_raw_native_cot_prompt(
    *,
    issue_description: str,
    test_command: Optional[str] = None,
) -> str:
    """Render the C.7 raw-native-CoT single-shot prompt.

    Exposed at module scope so tests + the harness share the same
    template. See ``RAW_NATIVE_COT_PROMPT_TEMPLATE``.
    """
    return RAW_NATIVE_COT_PROMPT_TEMPLATE.format(
        issue_description=str(issue_description or "").strip(),
        test_command_or_empty=str(test_command or "").strip() or "(none)",
    )


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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ArmResult:
    arm: str
    model: str
    output_dir: Path
    duration_seconds: float
    apex_result: dict[str, Any]
    score: float = 0.0


@dataclass
class TaskComparison:
    """One (model, task) pair across both arms."""

    task_id: str
    model: str
    arms: dict[str, ArmResult] = field(default_factory=dict)

    def score_delta(self) -> float:
        """``masai - raw_cot``; positive = MASAI helped."""
        masai = self.arms.get("masai")
        raw = self.arms.get("raw_cot")
        if masai is None or raw is None:
            return 0.0
        return masai.score - raw.score

    def winner(self) -> str:
        delta = self.score_delta()
        if delta > 1e-6:
            return "masai"
        if delta < -1e-6:
            return "raw_cot"
        return "tie"


# ---------------------------------------------------------------------------
# Arm execution
# ---------------------------------------------------------------------------


def _build_task_payload(task_id: str, model: str) -> dict[str, Any]:
    """Per-(model, task) orchestrator input.

    The model is propagated via ``llm_overrides.model`` so the
    orchestrator's CLI backend resolution picks the right binary; it
    falls back to the orchestrator-level default when the operator runs
    a stub orchestrator that ignores the override.
    """
    return {
        "task_id": task_id,
        "benchmark": "commit0_lite",
        "model": model,
        "llm_overrides": {"model": model},
    }


def _run_arm(
    *,
    arm: str,
    model: str,
    task_payload: dict[str, Any],
    orchestrator: Callable[..., Any],
    work_root: Path,
) -> ArmResult:
    output_dir = work_root / arm
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(task_payload)
    payload["output_dir"] = str(output_dir)
    rollout_overrides = dict(payload.get("rollout") or {})
    if arm == "masai":
        # MASAI arm: pre-rounds active + the existing scaffolded pipeline.
        rollout_overrides["cli_agent_use_masai_preround"] = "structured_prompt"
        rollout_overrides["use_native_cot_only"] = False
    else:
        # raw_cot arm: pre-rounds OFF + flag the orchestrator to use the
        # single-shot raw-CoT prompt template. ``use_native_cot_only``
        # is consumed by the orchestrator (or its stub in tests) — the
        # production orchestrator is expected to substitute the patcher
        # prompt with ``build_raw_native_cot_prompt``. Stub
        # orchestrators ignore the flag.
        rollout_overrides["cli_agent_use_masai_preround"] = "off"
        rollout_overrides["use_native_cot_only"] = True
        rollout_overrides["raw_native_cot_prompt_template"] = RAW_NATIVE_COT_PROMPT_TEMPLATE
    payload["rollout"] = rollout_overrides
    started = time.time()
    try:
        orchestrator(payload)
    except Exception as exc:  # noqa: BLE001 - capture but don't crash A/B
        sys.stderr.write(
            f"[ab_native_cot] arm={arm} model={model} task={task_payload.get('task_id')} "
            f"raised {type(exc).__name__}: {exc}\n"
        )
    duration = time.time() - started
    apex_result = _load_apex_result(output_dir)
    return ArmResult(
        arm=arm,
        model=model,
        output_dir=output_dir,
        duration_seconds=duration,
        apex_result=apex_result,
        score=_outcome_score(apex_result),
    )


def _run_task(
    *,
    task_id: str,
    model: str,
    task_payload: dict[str, Any],
    orchestrator: Callable[..., Any],
    work_root: Path,
) -> TaskComparison:
    comparison = TaskComparison(task_id=task_id, model=model)
    for arm in _AB_ARMS:
        arm_root = work_root / model / task_id / arm
        arm_root.mkdir(parents=True, exist_ok=True)
        comparison.arms[arm] = _run_arm(
            arm=arm,
            model=model,
            task_payload=task_payload,
            orchestrator=orchestrator,
            work_root=arm_root,
        )
    return comparison


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _aggregate_per_model(
    comparisons: list[TaskComparison],
) -> dict[str, dict[str, Any]]:
    """Bucket comparisons by model and aggregate per model."""
    buckets: dict[str, list[TaskComparison]] = {}
    for c in comparisons:
        buckets.setdefault(c.model, []).append(c)
    out: dict[str, dict[str, Any]] = {}
    for model, items in buckets.items():
        deltas = [c.score_delta() for c in items]
        winners = [c.winner() for c in items]
        n = len(items)
        masai_wins = sum(1 for w in winners if w == "masai")
        raw_wins = sum(1 for w in winners if w == "raw_cot")
        ties = sum(1 for w in winners if w == "tie")
        out[model] = {
            "task_count": n,
            "masai_wins": masai_wins,
            "raw_cot_wins": raw_wins,
            "ties": ties,
            "masai_win_rate": (masai_wins / n) if n else 0.0,
            "raw_cot_win_rate": (raw_wins / n) if n else 0.0,
            "tie_rate": (ties / n) if n else 0.0,
            "mean_score_delta": (statistics.fmean(deltas) if deltas else 0.0),
            "median_score_delta": (statistics.median(deltas) if deltas else 0.0),
        }
    return out


def _recommend_per_model(per_model: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Per-model decision: keep_masai / drop_masai / inconclusive."""
    out: dict[str, str] = {}
    for model, summary in per_model.items():
        if summary["task_count"] == 0:
            out[model] = "no_data"
            continue
        n = max(int(summary["task_count"]), 1)
        advantage_pp = 100.0 * (summary["masai_wins"] - summary["raw_cot_wins"]) / n
        delta = float(summary["mean_score_delta"])
        if advantage_pp >= _KEEP_MASAI_WIN_RATE_MARGIN_PP and delta >= -1e-3:
            out[model] = "keep_masai"
        elif advantage_pp <= -1e-6 and delta <= 1e-3:
            out[model] = "drop_masai"
        else:
            out[model] = "inconclusive"
    return out


def _render_report(
    *,
    comparisons: list[TaskComparison],
    per_model: dict[str, dict[str, Any]],
    per_model_recs: dict[str, str],
    output_dir: Path,
) -> Path:
    lines = [
        "# Native CoT vs MASAI A/B Report",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## Per-model recommendations",
        "",
        "| model | recommendation | tasks | masai_wins | raw_cot_wins | ties | mean Δ |",
        "|---|---|---|---|---|---|---|",
    ]
    for model in sorted(per_model.keys()):
        s = per_model[model]
        lines.append(
            f"| {model} | `{per_model_recs.get(model, 'no_data')}` | "
            f"{s['task_count']} | {s['masai_wins']} | "
            f"{s['raw_cot_wins']} | {s['ties']} | "
            f"{s['mean_score_delta']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Per-(model, task) breakdown",
            "",
            "| model | task_id | winner | score(masai) | score(raw_cot) | Δ |",
            "|---|---|---|---|---|---|",
        ]
    )
    for c in comparisons:
        masai = c.arms.get("masai")
        raw = c.arms.get("raw_cot")
        masai_score = masai.score if masai else float("nan")
        raw_score = raw.score if raw else float("nan")
        lines.append(
            f"| {c.model} | {c.task_id} | {c.winner()} | "
            f"{masai_score:.3f} | {raw_score:.3f} | "
            f"{c.score_delta():+.3f} |"
        )
    lines.extend(
        [
            "",
            "## How to act on the per-model recommendations",
            "",
            "* `keep_masai` — Keep `cli_agent_use_masai_preround = "
            '"structured_prompt"` for this model.',
            "* `drop_masai` — For this model, switch to "
            '`cli_agent_use_masai_preround = "off"` and rely on the '
            "model's native CoT. Big cost win when MASAI doesn't help.",
            "* `inconclusive` — Re-run with a larger task slice or a stratified by-repo cut.",
        ]
    )
    report_path = output_dir / "NATIVE_COT_AB_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decisive-Edge C.7 A/B harness: APEX MASAI scaffolding vs "
            "raw native CoT. Per-model recommendation. Produces "
            "NATIVE_COT_AB_REPORT.md."
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
        help=("Newline-delimited file of task ids; combined with --task-id."),
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Model id to sweep. Repeat for multiple models. When "
            "omitted, sweeps the default frontier model set "
            "(claude-4.7-opus, gpt-5.5, gemini-2.5-pro)."
        ),
    )
    parser.add_argument(
        "--task-payload-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON file overriding the {task_id, benchmark, model} "
            "payload — useful when targeting a non-Commit0 benchmark."
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
        help="Directory for NATIVE_COT_AB_REPORT.md and per-arm artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Resolve the orchestrator + (model, task) plan and print the "
            "planned execution without invoking the orchestrator."
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


def _resolve_models(args: argparse.Namespace) -> list[str]:
    models = list(args.model or [])
    if not models:
        models = list(_DEFAULT_MODELS)
    seen: set[str] = set()
    deduped: list[str] = []
    for m in models:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
    return deduped


def _resolve_task_payload(
    args: argparse.Namespace,
    task_id: str,
    model: str,
) -> dict[str, Any]:
    if args.task_payload_file is None:
        return _build_task_payload(task_id, model)
    template = json.loads(args.task_payload_file.read_text(encoding="utf-8"))
    if not isinstance(template, dict):
        raise ValueError(
            f"--task-payload-file must contain a JSON object; got {type(template).__name__}"
        )
    payload = dict(template)
    payload["task_id"] = task_id
    payload.setdefault("benchmark", "commit0_lite")
    payload["model"] = model
    payload.setdefault("llm_overrides", {})
    if isinstance(payload["llm_overrides"], dict):
        payload["llm_overrides"]["model"] = model
    return payload


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    task_ids = _resolve_task_ids(args)
    models = _resolve_models(args)
    if args.output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="apex_native_cot_ab_"))
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[ab_native_cot] arms={list(_AB_ARMS)} models={models} "
        f"tasks={len(task_ids)} output_dir={output_dir}"
    )
    if args.dry_run:
        print(f"[ab_native_cot] dry-run: orchestrator={args.orchestrator_callable}")
        for model in models:
            for task_id in task_ids:
                print(f"  - model={model} task={task_id}")
        print("[ab_native_cot] dry-run complete. Drop --dry-run to execute the A/B.")
        return 0
    orchestrator = _resolve_callable(args.orchestrator_callable)
    comparisons: list[TaskComparison] = []
    for model in models:
        for task_id in task_ids:
            print(f"[ab_native_cot] model={model} task={task_id}")
            payload = _resolve_task_payload(args, task_id, model)
            comparison = _run_task(
                task_id=task_id,
                model=model,
                task_payload=payload,
                orchestrator=orchestrator,
                work_root=output_dir,
            )
            comparisons.append(comparison)
    per_model = _aggregate_per_model(comparisons)
    per_model_recs = _recommend_per_model(per_model)
    report_path = _render_report(
        comparisons=comparisons,
        per_model=per_model,
        per_model_recs=per_model_recs,
        output_dir=output_dir,
    )
    print(f"[ab_native_cot] report: {report_path}")
    for model, rec in per_model_recs.items():
        print(f"[ab_native_cot] model={model} recommendation={rec}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
