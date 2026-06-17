"""SWE-Bench codegen evaluation runner.

Single CLI entry that drives APEX through one of the three public
SWE-Bench dataset shapes (classic / Verified / Multilingual). The Pro
codegen workflow lives in ``swebench_pro_codegen_eval.py`` because
the Pro container shape needs the Pro harness rather than the public
``swebench`` package.

Per-task workflow:

1. Load the dataset row via :class:`SWEBenchHarness.discover_tasks`.
   The :class:`SWEBenchTask` constructor scrubs the gold ``patch`` and
   ``test_patch`` immediately — the orchestrator never sees them.
2. Prepare an empty workspace (the harness's docker image owns the
   actual repo).
3. Invoke ``ApexOrchestrator.solve(repo_path=workspace,
   issue_description=task.build_issue_description())``.
4. Extract ``result.patch`` and append a JSONL row to the predictions
   file.
5. After all tasks finish (or were skipped), shell the harness exactly
   once across the full predictions file. Per-task try/except keeps a
   single bad task from aborting the loop.
6. Parse the per-instance ``report.json`` files into
   :class:`ScoreReport` records and write a summary.

Per-task try/except is non-negotiable — we apply the lesson from
``staged-weaving-sky.md`` that one bad row should never break a 2,294-row
sweep.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .checkpointing import atomic_write_json, atomic_write_text
from .swebench_benchmark import (
    SWEBENCH_DEFAULT_SPLIT,
    SWEBENCH_HARNESS_INSTALL_HINT,
    SWEBENCH_HARNESS_MODE_CLASSIC,
    SWEBENCH_HARNESS_MODES,
    SWEBENCH_VERIFIED_DATASET_NAME,
    ScoreReport,
    SWEBenchHarness,
    SWEBenchPredictionRecord,
    SWEBenchTask,
)
from .target_runtime import (
    apply_target_tool_env_to_apex_config,
    docker_image_runtime,
    target_tool_env_overrides,
)

logger = logging.getLogger("apex.evaluation.swebench_codegen")


_DEFAULT_RUN_ID = "apex-swebench"


@dataclass(frozen=True)
class SWEBenchCodegenConfig:
    """Operator-tunable configuration for the codegen runner."""

    output_dir: str
    dataset_name: str = SWEBENCH_VERIFIED_DATASET_NAME
    split: str = SWEBENCH_DEFAULT_SPLIT
    harness_mode: str = SWEBENCH_HARNESS_MODE_CLASSIC
    model_name: str = "apex"
    parallelism: int = 1
    limit: int = 0
    task_ids: list[str] = field(default_factory=list)
    repos: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    skip_existing: bool = False
    predictions_path: str = ""
    run_id: str = _DEFAULT_RUN_ID
    log_dir: str = ""
    apex_config_path: str = ""
    config_path: str = ""
    swebench_runner_args: list[str] = field(default_factory=list)
    harness_max_workers: int = 4
    harness_timeout_seconds: float = 14400.0
    harness_python_executable: str = ""
    harness_cache_level: str = "instance"
    harness_force_rebuild: bool = False
    skip_harness_invocation: bool = False
    generation_timeout_seconds: float = 1800.0


# Internal hook for tests. ``None`` falls through to the real ApexOrchestrator;
# tests assign a callable returning a stub result so we never spawn an
# actual orchestrator (which would call into LLM CLIs).
_orchestrator_factory_override: Optional[Callable[..., Any]] = None
_harness_factory_override: Optional[Callable[..., Any]] = None


def set_orchestrator_factory_override(
    factory: Optional[Callable[..., Any]],
) -> None:
    """Test hook: install/remove a stub orchestrator factory."""

    global _orchestrator_factory_override
    _orchestrator_factory_override = factory


def set_harness_factory_override(
    factory: Optional[Callable[..., Any]],
) -> None:
    """Test hook: install/remove a stub harness factory."""

    global _harness_factory_override
    _harness_factory_override = factory


def _build_orchestrator(
    config_path: str,
    *,
    target_tool_env: Optional[dict[str, str]] = None,
) -> Any:
    """Construct the ApexOrchestrator (or test stub).

    Loading ApexConfig is deferred to keep ``--help`` cheap and to let
    tests inject a stub via :func:`set_orchestrator_factory_override`
    without dragging in the entire orchestrator import graph.
    """

    if _orchestrator_factory_override is not None:
        orchestrator = _orchestrator_factory_override(config_path)
        if hasattr(orchestrator, "config"):
            apply_target_tool_env_to_apex_config(orchestrator.config, target_tool_env or {})
        return orchestrator
    from ..core.config import ApexConfig
    from ..orchestrator import ApexOrchestrator

    if config_path:
        apex_config = ApexConfig.from_file(config_path)
    else:
        apex_config = ApexConfig()
    apply_target_tool_env_to_apex_config(apex_config, target_tool_env or {})
    return ApexOrchestrator(apex_config)


def _build_harness(config: SWEBenchCodegenConfig) -> SWEBenchHarness:
    """Construct the SWEBenchHarness (or test stub)."""

    if _harness_factory_override is not None:
        return _harness_factory_override(config)
    return SWEBenchHarness(
        output_dir=config.output_dir,
        dataset_name=config.dataset_name,
        split=config.split,
        harness_mode=config.harness_mode,
        max_workers=config.harness_max_workers,
        cache_level=config.harness_cache_level,
        force_rebuild=config.harness_force_rebuild,
        timeout_seconds=config.harness_timeout_seconds,
        python_executable=(
            config.harness_python_executable or None  # type: ignore[arg-type]
        ),
    )


def _process_one_task(
    *,
    task: SWEBenchTask,
    workspace: Path,
    orchestrator: Any,
    config: SWEBenchCodegenConfig,
) -> tuple[SWEBenchPredictionRecord, dict[str, Any]]:
    """Solve one task with the orchestrator.

    Returns ``(record, diagnostics)``. ``diagnostics`` is a small JSON-able
    dict suitable for per-task summary writing — not the full
    ``ApexResult.to_dict()`` blob, which can be megabytes per task and
    blows up the summary file.
    """

    issue_description = task.build_issue_description()
    started = time.time()
    try:
        result = orchestrator.solve(
            repo_path=str(workspace),
            issue_description=issue_description,
        )
    except Exception as exc:
        elapsed = time.time() - started
        logger.exception(
            "orchestrator.solve raised for %s after %.1fs",
            task.instance_id,
            elapsed,
        )
        return (
            SWEBenchPredictionRecord(
                instance_id=task.instance_id,
                model_name_or_path=config.model_name,
                model_patch="",
            ),
            {
                "instance_id": task.instance_id,
                "status": "errored",
                "error": f"{type(exc).__name__}: {exc}",
                "duration_seconds": elapsed,
            },
        )
    elapsed = time.time() - started
    patch = str(getattr(result, "patch", "") or "")
    success = bool(getattr(result, "success", False))
    diagnostics = {
        "instance_id": task.instance_id,
        "status": "ok" if success and patch else "no_patch",
        "duration_seconds": elapsed,
        "selected_rollout_id": getattr(result, "selected_rollout_id", None),
        "total_rollouts": getattr(result, "total_rollouts", None),
        "successful_rollouts": getattr(result, "successful_rollouts", None),
        "total_tokens": getattr(result, "total_tokens", None),
        "patch_length": len(patch),
        "explanation": (getattr(result, "explanation", None) or "")[:512],
    }
    record = SWEBenchPredictionRecord(
        instance_id=task.instance_id,
        model_name_or_path=config.model_name,
        model_patch=patch,
    )
    return record, diagnostics


def _filter_existing_records(
    tasks: list[SWEBenchTask],
    records_dir: Path,
) -> list[SWEBenchTask]:
    """Drop tasks whose per-instance record already exists on disk."""

    existing = {p.stem for p in records_dir.glob("*.json")}
    if not existing:
        return tasks
    keep = [t for t in tasks if t.instance_id not in existing]
    skipped = len(tasks) - len(keep)
    if skipped:
        logger.warning(
            "skip_existing: filtered %d -> %d tasks (skipped %d already in %s)",
            len(tasks),
            len(keep),
            skipped,
            records_dir,
        )
    return keep


def run_codegen_eval(config: SWEBenchCodegenConfig) -> dict[str, Any]:
    """End-to-end driver. Loads dataset, solves, scores, writes summary."""

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    log_dir = (
        Path(config.log_dir).expanduser().resolve()
        if config.log_dir
        else (output_dir / "harness_logs")
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    preds_path = (
        Path(config.predictions_path).expanduser().resolve()
        if config.predictions_path
        else output_dir / "predictions" / f"{config.model_name}__{config.run_id}.jsonl"
    )
    preds_path.parent.mkdir(parents=True, exist_ok=True)

    harness = _build_harness(config)
    tasks = harness.discover_tasks(
        instances=list(config.task_ids) or None,
        repos=list(config.repos) or None,
        languages=list(config.languages) or None,
        limit=int(config.limit) or None,
    )
    if config.skip_existing:
        tasks = _filter_existing_records(tasks, records_dir)

    manifest = {
        "started_at": time.time(),
        "model_name": config.model_name,
        "dataset_name": config.dataset_name,
        "split": config.split,
        "harness_mode": config.harness_mode,
        "task_count": len(tasks),
        "parallelism": config.parallelism,
        "predictions_path": str(preds_path),
        "log_dir": str(log_dir),
        "run_id": config.run_id,
        "skip_harness_invocation": config.skip_harness_invocation,
        "config_path": config.config_path,
        "apex_config_path": config.apex_config_path,
    }
    atomic_write_json(output_dir / "run_manifest.json", manifest)

    records: list[SWEBenchPredictionRecord] = []
    diagnostics_by_id: dict[str, dict[str, Any]] = {}

    def _solve_task(task: SWEBenchTask) -> None:
        try:
            workspace = harness.prepare_workspace(task)
        except Exception as exc:
            logger.exception(
                "prepare_workspace failed for %s; recording empty patch",
                task.instance_id,
            )
            diagnostics_by_id[task.instance_id] = {
                "instance_id": task.instance_id,
                "status": "prepare_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            records.append(
                SWEBenchPredictionRecord(
                    instance_id=task.instance_id,
                    model_name_or_path=config.model_name,
                    model_patch="",
                )
            )
            return
        try:
            image_uri = (
                harness.target_image_uri(task) if hasattr(harness, "target_image_uri") else ""
            )
            target_tool_env, target_tool_diag = target_tool_env_overrides(
                workdir=workspace,
                output_dir=output_dir / "target_runtime_tools" / task.instance_id,
                timeout_seconds=max(1, int(config.generation_timeout_seconds or 1)),
                runtime=(
                    docker_image_runtime(
                        image=image_uri,
                        docker_workdir="/testbed",
                        description="swebench_official_task_image",
                    )
                    if image_uri
                    else None
                ),
                label=f"swebench_codegen_{task.instance_id}",
            )
            orchestrator = _build_orchestrator(
                config.apex_config_path,
                target_tool_env=target_tool_env,
            )
            record, diag = _process_one_task(
                task=task,
                workspace=workspace,
                orchestrator=orchestrator,
                config=config,
            )
            diag["target_runtime_tools"] = target_tool_diag
        except Exception as exc:
            logger.exception(
                "_process_one_task raised for %s; recording empty patch",
                task.instance_id,
            )
            record = SWEBenchPredictionRecord(
                instance_id=task.instance_id,
                model_name_or_path=config.model_name,
                model_patch="",
            )
            diag = {
                "instance_id": task.instance_id,
                "status": "errored",
                "error": f"{type(exc).__name__}: {exc}",
            }
        records.append(record)
        diagnostics_by_id[task.instance_id] = diag
        atomic_write_json(records_dir / f"{task.instance_id}.json", diag)

    parallelism = max(1, int(config.parallelism))
    if parallelism == 1:
        for task in tasks:
            _solve_task(task)
    else:
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            futures = {pool.submit(_solve_task, task): task for task in tasks}
            for future in as_completed(futures):
                # _solve_task already swallows per-task errors; re-raise
                # only for truly unexpected failures.
                try:
                    future.result()
                except BaseException as exc:  # pragma: no cover - defensive
                    task = futures[future]
                    logger.exception(
                        "future raised for %s outside _solve_task",
                        task.instance_id,
                    )
                    records.append(
                        SWEBenchPredictionRecord(
                            instance_id=task.instance_id,
                            model_name_or_path=config.model_name,
                            model_patch="",
                        )
                    )
                    diagnostics_by_id[task.instance_id] = {
                        "instance_id": task.instance_id,
                        "status": "errored",
                        "error": f"{type(exc).__name__}: {exc}",
                    }

    records.sort(key=lambda r: r.instance_id)
    harness.write_predictions_file(preds_path, records)

    harness_outcome: dict[str, Any] = {
        "invoked": False,
        "command": [],
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
    }
    score_reports: list[ScoreReport] = []
    if config.skip_harness_invocation:
        # Construct the would-be command for the operator's records.
        command_kwargs = {
            "predictions_path": preds_path,
            "run_id": config.run_id,
            "log_dir": log_dir,
            "instance_ids": [t.instance_id for t in tasks],
            "extra_args": list(config.swebench_runner_args or []),
        }
        try:
            harness_outcome["command"] = harness.build_run_evaluation_command(**command_kwargs)
        except TypeError as exc:
            if "extra_args" not in str(exc):
                raise
            command_kwargs.pop("extra_args", None)
            harness_outcome["command"] = harness.build_run_evaluation_command(**command_kwargs)
        harness_outcome["skipped_reason"] = "skip_harness_invocation"
    elif not records:
        harness_outcome["skipped_reason"] = "no_predictions"
    else:
        try:
            run_kwargs = {
                "predictions_path": preds_path,
                "run_id": config.run_id,
                "log_dir": log_dir,
                "instance_ids": [t.instance_id for t in tasks],
                "extra_args": list(config.swebench_runner_args or []),
            }
            try:
                completed = harness.run_evaluation(**run_kwargs)
            except TypeError as exc:
                if "extra_args" not in str(exc):
                    raise
                run_kwargs.pop("extra_args", None)
                completed = harness.run_evaluation(**run_kwargs)
        except SystemExit as exc:
            harness_outcome["error"] = str(exc)
            logger.error(
                "swebench harness invocation failed: %s. %s",
                exc,
                SWEBENCH_HARNESS_INSTALL_HINT,
            )
            completed = None
        except subprocess.TimeoutExpired:
            harness_outcome["error"] = (
                f"swebench harness exceeded {config.harness_timeout_seconds}s timeout"
            )
            logger.error(harness_outcome["error"])
            completed = None
        if completed is not None:
            harness_outcome["invoked"] = True
            harness_outcome["command"] = list(completed.args)
            harness_outcome["returncode"] = completed.returncode
            harness_outcome["stdout_tail"] = (completed.stdout or "")[-2048:]
            harness_outcome["stderr_tail"] = (completed.stderr or "")[-2048:]
        for task in tasks:
            try:
                report = harness.parse_report_for_task(
                    task,
                    run_id=config.run_id,
                    log_dir=log_dir,
                    model_name=config.model_name,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception(
                    "report parsing raised for %s",
                    task.instance_id,
                )
                report = ScoreReport(
                    instance_id=task.instance_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
            score_reports.append(report)
            atomic_write_json(
                records_dir / f"{task.instance_id}.report.json",
                report.to_dict(),
            )

    finished = time.time()
    solved = sum(1 for r in score_reports if r.solved)
    summary = {
        "status": ("failed" if len(tasks) > 0 and not records else "ok"),
        "model_name": config.model_name,
        "dataset_name": config.dataset_name,
        "split": config.split,
        "harness_mode": config.harness_mode,
        "task_count": len(tasks),
        "predictions_path": str(preds_path),
        "log_dir": str(log_dir),
        "records_dir": str(records_dir),
        "started_at": manifest["started_at"],
        "finished_at": finished,
        "elapsed_seconds": finished - manifest["started_at"],
        "harness": harness_outcome,
        "solved_count": solved,
        "score": (solved / len(score_reports)) if score_reports else None,
        "task_diagnostics": diagnostics_by_id,
        "score_reports": [r.to_dict() for r in score_reports],
    }
    atomic_write_json(output_dir / "generation_summary.json", summary)
    atomic_write_text(
        output_dir / "harness_command.txt",
        " ".join(str(token) for token in harness_outcome.get("command") or []),
    )
    return summary


def _parse_args(argv: list[str]) -> SWEBenchCodegenConfig:
    parser = argparse.ArgumentParser(
        prog="python -m apex.evaluation.swebench_codegen_eval",
        description=(
            "Drive APEX through SWE-Bench / Verified / Multilingual codegen. "
            "Produces a JSONL predictions file and shells the official "
            "swebench harness once across all tasks. The Pro variant is "
            "served by swebench_pro_codegen_eval.py."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default=SWEBENCH_VERIFIED_DATASET_NAME)
    parser.add_argument("--split", default=SWEBENCH_DEFAULT_SPLIT)
    parser.add_argument(
        "--harness-mode",
        default=SWEBENCH_HARNESS_MODE_CLASSIC,
        choices=list(SWEBENCH_HARNESS_MODES),
    )
    parser.add_argument("--model-name", default="apex")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--repo", action="append", default=[])
    parser.add_argument("--language", action="append", default=[])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--predictions-path",
        default="",
        help=(
            "Override the JSONL predictions path. Defaults to "
            "<output-dir>/predictions/<model>__<run_id>.jsonl."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=_DEFAULT_RUN_ID,
        help=(
            "Run identifier passed to swebench harness as --run_id. "
            "Determines the report.json directory layout."
        ),
    )
    parser.add_argument(
        "--log-dir",
        default="",
        help=("Override the harness log directory. Defaults to <output-dir>/harness_logs."),
    )
    parser.add_argument(
        "--apex-config-path",
        default="",
        help="Path to the ApexConfig JSON used to construct ApexOrchestrator.",
    )
    parser.add_argument(
        "--config-path",
        default="",
        help=(
            "Path to a benchmark-runner config JSON (for operator-side "
            "audit and reproducibility). Currently informational only; "
            "all knobs are exposed as CLI flags."
        ),
    )
    parser.add_argument(
        "--swebench-runner-arg",
        action="append",
        default=[],
        help=(
            "Extra arg passed verbatim to the swebench harness CLI. "
            "Repeatable. Note: these are appended after the standard "
            "args; conflicting values may be overridden silently by the "
            "harness."
        ),
    )
    parser.add_argument("--harness-max-workers", type=int, default=4)
    parser.add_argument("--harness-timeout-seconds", type=float, default=14400.0)
    parser.add_argument("--harness-python-executable", default="")
    parser.add_argument(
        "--harness-cache-level",
        default="instance",
        choices=("none", "base", "env", "instance"),
    )
    parser.add_argument("--harness-force-rebuild", action="store_true")
    parser.add_argument(
        "--skip-harness-invocation",
        action="store_true",
        help=(
            "Generate predictions and write them to disk, but do NOT shell "
            "the swebench harness. Useful for dry-runs and for splitting "
            "generation from scoring across machines."
        ),
    )
    parser.add_argument(
        "--generation-timeout-seconds",
        type=float,
        default=1800.0,
        help=(
            "Per-task wall-clock budget for orchestrator.solve. Currently "
            "informational only — enforcement lives inside the orchestrator's "
            "own rollout timeouts."
        ),
    )
    args = parser.parse_args(argv)
    return SWEBenchCodegenConfig(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        split=args.split,
        harness_mode=args.harness_mode,
        model_name=args.model_name,
        parallelism=int(args.parallelism or 1),
        limit=int(args.limit or 0),
        task_ids=list(args.task_id or []),
        repos=list(args.repo or []),
        languages=list(args.language or []),
        skip_existing=bool(args.skip_existing),
        predictions_path=args.predictions_path,
        run_id=args.run_id,
        log_dir=args.log_dir,
        apex_config_path=args.apex_config_path,
        config_path=args.config_path,
        swebench_runner_args=list(args.swebench_runner_arg or []),
        harness_max_workers=int(args.harness_max_workers or 4),
        harness_timeout_seconds=float(args.harness_timeout_seconds or 14400.0),
        harness_python_executable=str(args.harness_python_executable or ""),
        harness_cache_level=str(args.harness_cache_level),
        harness_force_rebuild=bool(args.harness_force_rebuild),
        skip_harness_invocation=bool(args.skip_harness_invocation),
        generation_timeout_seconds=float(args.generation_timeout_seconds or 1800.0),
    )


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=os.environ.get("APEX_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = _parse_args(list(argv if argv is not None else sys.argv[1:]))
    summary = run_codegen_eval(config)
    print(json.dumps(_summary_for_stdout(summary), indent=2, sort_keys=True))
    return 0 if summary.get("status") == "ok" else 1


def _summary_for_stdout(summary: dict[str, Any]) -> dict[str, Any]:
    """Lightweight summary suitable for terminal stdout."""

    return {
        "status": summary.get("status"),
        "model_name": summary.get("model_name"),
        "dataset_name": summary.get("dataset_name"),
        "split": summary.get("split"),
        "harness_mode": summary.get("harness_mode"),
        "task_count": summary.get("task_count"),
        "predictions_path": summary.get("predictions_path"),
        "harness_invoked": (summary.get("harness") or {}).get("invoked", False),
        "harness_returncode": (summary.get("harness") or {}).get("returncode"),
        "solved_count": summary.get("solved_count"),
        "score": summary.get("score"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
