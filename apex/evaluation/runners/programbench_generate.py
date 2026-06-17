"""ProgramBench prediction generation driver.

Bridges ProgramBench's per-task ``task.yaml + tests.json`` schema to the APEX
orchestrator: per task, materialize an empty workspace + spec, hand it to
``ApexOrchestrator.solve`` so APEX edits the workspace in place, then ask the
``ProgramBenchHarness`` to package + score the result.

Mirrors ``apex.evaluation.runners.testgenevallite_generate``'s argparse shape
and per-task isolation discipline so a single misbehaving program never aborts
the loop. Every per-task failure is captured into the run summary with the
exception type + message.

The orchestrator is instantiated lazily — a ``--dry-run`` flag plus the
``ApexOrchestratorFactory`` injection point in ``run_generate`` keep the
unit tests free of real LLM calls.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from apex.evaluation.checkpointing import atomic_write_json, atomic_write_text
from apex.evaluation.programbench_benchmark import (
    PROGRAMBENCH_DEFAULT_EVAL_TIMEOUT_SECONDS,
    PROGRAMBENCH_DEFAULT_HF_DATASET,
    PROGRAMBENCH_DEFAULT_IMAGE_TAG,
    ProgramBenchHarness,
    ProgramBenchScoreReport,
    ProgramBenchTask,
    write_prediction_record,
)
from apex.evaluation.target_runtime import (
    apply_target_tool_env_to_apex_config,
    docker_image_runtime,
    target_tool_env_overrides,
)

logger = logging.getLogger("apex.programbench_generate")


@dataclass(frozen=True)
class ProgramBenchGenerateConfig:
    output_dir: str
    model_name: str = "apex"
    dataset_name: str = PROGRAMBENCH_DEFAULT_HF_DATASET
    spec_dir: str = ""
    hidden_tests_dir: str = ""
    task_ids: list[str] = field(default_factory=list)
    limit: int = 0
    parallelism: int = 1
    apex_config_path: str = ""
    image_tag: str = PROGRAMBENCH_DEFAULT_IMAGE_TAG
    docker_org: str = "programbench"
    cli_executable: str = "programbench"
    eval_timeout_seconds: int = PROGRAMBENCH_DEFAULT_EVAL_TIMEOUT_SECONDS
    skip_existing: bool = False
    dry_run: bool = False  # skip orchestrator + harness; emit empty solutions
    skip_evaluation: bool = False  # run agent but skip the harness eval step
    allow_synthetic_specs: bool = False


# Type aliases for injectable factories used by the unit tests.
ApexOrchestratorFactory = Callable[[ProgramBenchGenerateConfig], Any]
HarnessFactory = Callable[[ProgramBenchGenerateConfig], ProgramBenchHarness]


def _default_orchestrator_factory(config: ProgramBenchGenerateConfig) -> Any:
    """Lazily import + construct an ``ApexOrchestrator`` from the config path.

    Imported lazily so unit tests + ``--help`` don't pay the cost of pulling
    in the entire APEX dep graph.
    """

    from apex.core.config import ApexConfig
    from apex.orchestrator import ApexOrchestrator

    if config.apex_config_path:
        apex_config = ApexConfig.from_file(config.apex_config_path)
    else:
        apex_config = ApexConfig()
    return ApexOrchestrator(apex_config)


def _default_harness_factory(config: ProgramBenchGenerateConfig) -> ProgramBenchHarness:
    return ProgramBenchHarness(
        cli_executable=config.cli_executable,
        docker_org=config.docker_org,
        image_tag=config.image_tag,
        eval_timeout_seconds=int(config.eval_timeout_seconds),
    )


def _filter_tasks(
    tasks: list[ProgramBenchTask],
    *,
    task_ids: Iterable[str],
    limit: int,
) -> list[ProgramBenchTask]:
    wanted = {str(tid) for tid in task_ids if str(tid)}
    if wanted:
        tasks = [t for t in tasks if t.instance_id in wanted]
    if limit and limit > 0:
        tasks = tasks[:limit]
    return tasks


def _process_one(
    *,
    task: ProgramBenchTask,
    config: ProgramBenchGenerateConfig,
    orchestrator_factory: ApexOrchestratorFactory,
    harness: ProgramBenchHarness,
    output_dir: Path,
) -> dict[str, Any]:
    """Run APEX + harness on one task. NEVER raises for the per-task loop."""

    started = time.time()
    workspace_root = output_dir / "workspaces"
    workspace_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / "harness_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    apex_diagnostics: dict[str, Any] = {
        "started_at": started,
        "dry_run": bool(config.dry_run),
    }
    score: Optional[ProgramBenchScoreReport] = None
    try:
        workspace = harness.prepare_workspace(task, workspace_root)
        apex_diagnostics["workspace"] = str(workspace)
        if config.dry_run:
            apex_diagnostics["solver"] = "skipped_dry_run"
        else:
            try:
                orchestrator = orchestrator_factory(config)
                target_image = task.image_name or ""
                if target_image and ":" not in target_image.rsplit("/", 1)[-1]:
                    target_image = f"{target_image}:{task.image_tag or config.image_tag}"
                runtime = (
                    docker_image_runtime(
                        image=target_image,
                        docker_workdir="/workspace",
                        description="programbench_official_task_image",
                    )
                    if target_image
                    else None
                )
                target_tool_env, target_tool_diag = target_tool_env_overrides(
                    workdir=workspace,
                    output_dir=output_dir / "target_runtime_tools" / task.instance_id,
                    timeout_seconds=config.eval_timeout_seconds,
                    runtime=runtime,
                    label=f"programbench_{task.instance_id}",
                )
                apex_diagnostics["target_runtime_tools"] = target_tool_diag
                if hasattr(orchestrator, "config"):
                    apply_target_tool_env_to_apex_config(orchestrator.config, target_tool_env)
                result = orchestrator.solve(
                    repo_path=str(workspace),
                    issue_description=task.spec_text,
                    benchmark_metadata={
                        "benchmark": "programbench",
                        "instance_id": task.instance_id,
                        "language": task.language,
                        "image_name": task.image_name,
                        "image_tag": task.image_tag,
                    },
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("orchestrator.solve raised for %s", task.instance_id)
                apex_diagnostics["solver"] = "errored"
                apex_diagnostics["solver_error"] = f"{type(exc).__name__}: {exc}"
                apex_diagnostics["solver_traceback"] = traceback.format_exc()
            else:
                apex_diagnostics["solver"] = "completed"
                apex_diagnostics["solver_success"] = bool(getattr(result, "success", False))
                apex_diagnostics["selected_rollout_id"] = getattr(
                    result, "selected_rollout_id", None
                )
                patch = getattr(result, "patch", None)
                if patch:
                    apex_diagnostics["patch_excerpt"] = str(patch)[:4000]
        if config.skip_evaluation:
            apex_diagnostics["harness"] = "skipped"
            score = ProgramBenchScoreReport(
                program_id=task.instance_id,
                total_tests=task.total_active_tests(),
                error_code="harness_skipped",
                error_details="--skip-evaluation passed",
            )
        else:
            try:
                score = harness.evaluate_solution(
                    task,
                    workspace,
                    run_dir=run_dir,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("harness.evaluate_solution raised for %s", task.instance_id)
                score = ProgramBenchScoreReport(
                    program_id=task.instance_id,
                    total_tests=task.total_active_tests(),
                    error_code="harness_exception",
                    error_details=f"{type(exc).__name__}: {exc}",
                )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("workspace prep raised for %s", task.instance_id)
        apex_diagnostics["workspace_error"] = f"{type(exc).__name__}: {exc}"
        score = ProgramBenchScoreReport(
            program_id=task.instance_id,
            total_tests=task.total_active_tests(),
            error_code="workspace_prep_failed",
            error_details=f"{type(exc).__name__}: {exc}",
        )
    finished = time.time()
    apex_diagnostics["finished_at"] = finished
    apex_diagnostics["elapsed_seconds"] = finished - started
    if score is None:
        score = ProgramBenchScoreReport(
            program_id=task.instance_id,
            total_tests=task.total_active_tests(),
            error_code="unknown_failure",
            error_details="no score produced",
        )
    return write_prediction_record(
        task=task,
        score=score,
        apex_diagnostics=apex_diagnostics,
        model_name=config.model_name,
    )


def run_generate(
    config: ProgramBenchGenerateConfig,
    *,
    orchestrator_factory: ApexOrchestratorFactory = _default_orchestrator_factory,
    harness_factory: HarnessFactory = _default_harness_factory,
) -> dict[str, Any]:
    """Execute the ProgramBench generation loop and write a JSONL preds file.

    Returns a summary dict; also writes ``generation_summary.json`` and per-task
    record JSONs alongside the JSONL preds file. Per-task failures are isolated
    so one broken program never aborts the run.
    """

    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = output_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_path = preds_dir / (
        f"{config.model_name}__{Path(config.dataset_name).name}__predictions.jsonl"
    )

    harness = harness_factory(config)
    tasks = harness.discover_tasks(
        config.dataset_name,
        spec_dir=config.spec_dir or None,
        hidden_tests_dir=config.hidden_tests_dir or None,
    )
    tasks = _filter_tasks(tasks, task_ids=config.task_ids, limit=config.limit)

    synthetic_spec_tasks = [
        task.instance_id for task in tasks if getattr(task, "spec_source", "") == "synthetic"
    ]
    if synthetic_spec_tasks and not (config.allow_synthetic_specs or config.dry_run):
        summary = {
            "status": "failed",
            "error": "programbench_real_specs_required",
            "message": (
                "ProgramBench benchmark runs require a real --spec-dir. "
                "Use --allow-synthetic-specs only for smoke runs."
            ),
            "synthetic_spec_task_count": len(synthetic_spec_tasks),
            "synthetic_spec_task_examples": synthetic_spec_tasks[:10],
            "prediction_path": str(preds_path),
            "records_dir": str(records_dir),
            "task_count": len(tasks),
            "predictions_written": 0,
            "started_at": time.time(),
            "finished_at": time.time(),
            "elapsed_seconds": 0.0,
        }
        atomic_write_json(output_dir / "generation_summary.json", summary)
        return summary

    if config.skip_existing:
        existing_ids = {p.stem for p in records_dir.glob("*.json")}
        before = len(tasks)
        tasks = [t for t in tasks if t.instance_id not in existing_ids]
        if before != len(tasks):
            logger.warning(
                "skip_existing: filtered %d → %d tasks (skipped %d already in %s)",
                before,
                len(tasks),
                before - len(tasks),
                records_dir,
            )

    manifest = {
        "started_at": time.time(),
        "model_name": config.model_name,
        "dataset": config.dataset_name,
        "total_tasks": len(tasks),
        "parallelism": config.parallelism,
        "apex_config_path": config.apex_config_path,
        "image_tag": config.image_tag,
        "cli_executable": config.cli_executable,
        "dry_run": config.dry_run,
        "skip_evaluation": config.skip_evaluation,
        "prediction_path": str(preds_path),
    }
    atomic_write_json(output_dir / "run_manifest.json", manifest)

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    parallelism = max(1, int(config.parallelism))

    def _record_one(task: ProgramBenchTask) -> dict[str, Any]:
        return _process_one(
            task=task,
            config=config,
            orchestrator_factory=orchestrator_factory,
            harness=harness,
            output_dir=output_dir,
        )

    if parallelism == 1:
        for task in tasks:
            try:
                record = _record_one(task)
                atomic_write_json(records_dir / f"{record['instance_id']}.json", record)
                completed.append(record)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Task %s raised at top level", task.instance_id)
                failures.append(
                    {"instance_id": task.instance_id, "error": f"{type(exc).__name__}: {exc}"}
                )
    else:
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            futures = {pool.submit(_record_one, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    record = future.result()
                    atomic_write_json(records_dir / f"{record['instance_id']}.json", record)
                    completed.append(record)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("Task %s raised at top level", task.instance_id)
                    failures.append(
                        {"instance_id": task.instance_id, "error": f"{type(exc).__name__}: {exc}"}
                    )

    completed.sort(key=lambda r: str(r.get("instance_id") or ""))
    atomic_write_text(
        preds_path,
        "".join(json.dumps(rec, sort_keys=True) + "\n" for rec in completed),
    )

    finished = time.time()
    summary = {
        "status": (
            "failed" if len(tasks) > 0 and not completed else ("ok" if not failures else "partial")
        ),
        "prediction_path": str(preds_path),
        "records_dir": str(records_dir),
        "task_count": len(tasks),
        "predictions_written": len(completed),
        "failures": failures,
        "started_at": manifest["started_at"],
        "finished_at": finished,
        "elapsed_seconds": finished - manifest["started_at"],
    }
    atomic_write_json(output_dir / "generation_summary.json", summary)
    return summary


def _parse_args(argv: list[str]) -> ProgramBenchGenerateConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ProgramBench predictions through APEX. Materializes a "
            "workspace per task, hands it to ApexOrchestrator.solve, then "
            "packages + scores the result via the official `programbench eval` CLI."
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="apex")
    parser.add_argument(
        "--dataset-name",
        default=PROGRAMBENCH_DEFAULT_HF_DATASET,
        help=(
            "HuggingFace dataset id OR path to a local directory of per-task "
            "subdirs (each with task.yaml + tests.json). Defaults to "
            f"{PROGRAMBENCH_DEFAULT_HF_DATASET}."
        ),
    )
    parser.add_argument(
        "--spec-dir",
        default="",
        help=(
            "Optional directory holding per-instance specification markdown "
            "(searched as <spec-dir>/<instance_id>/spec.md or "
            "<spec-dir>/<instance_id>.md)."
        ),
    )
    parser.add_argument(
        "--hidden-tests-dir",
        default="",
        help=(
            "Optional directory holding per-instance hidden test bundles "
            "(<hidden-tests-dir>/<instance_id>/...) for offline smoke runs."
        ),
    )
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0, help="Cap task count (0 = no limit).")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument(
        "--apex-config-path",
        default="",
        help="Path to an APEX JSON config (e.g. configs/benchmark_programbench_max.json).",
    )
    parser.add_argument(
        "--image-tag",
        default=PROGRAMBENCH_DEFAULT_IMAGE_TAG,
        help=(
            "Docker image tag passed to programbench eval (default "
            f"{PROGRAMBENCH_DEFAULT_IMAGE_TAG})."
        ),
    )
    parser.add_argument(
        "--docker-org",
        default="programbench",
        help="Docker Hub org for ProgramBench images (default 'programbench').",
    )
    parser.add_argument(
        "--cli-executable",
        default="programbench",
        help="Override the upstream `programbench` CLI binary path.",
    )
    parser.add_argument(
        "--eval-timeout-seconds",
        type=int,
        default=PROGRAMBENCH_DEFAULT_EVAL_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Resume a partial run by skipping tasks whose record JSON already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip the orchestrator entirely; useful for smoke-validating "
            "the workspace + harness packaging path."
        ),
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help=(
            "Run APEX but skip the upstream `programbench eval` shell-out. "
            "Useful when no Linux x86_64 box is available for scoring."
        ),
    )
    parser.add_argument(
        "--allow-synthetic-specs",
        action="store_true",
        help=(
            "Allow ProgramBench's sparse fallback spec text. This is for smoke "
            "runs only; benchmark runs fail closed without real specs."
        ),
    )
    args = parser.parse_args(argv)
    return ProgramBenchGenerateConfig(
        output_dir=args.output_dir,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        spec_dir=args.spec_dir,
        hidden_tests_dir=args.hidden_tests_dir,
        task_ids=list(args.task_id or []),
        limit=int(args.limit or 0),
        parallelism=int(args.parallelism or 1),
        apex_config_path=args.apex_config_path,
        image_tag=args.image_tag,
        docker_org=args.docker_org,
        cli_executable=args.cli_executable,
        eval_timeout_seconds=int(args.eval_timeout_seconds),
        skip_existing=bool(args.skip_existing),
        dry_run=bool(args.dry_run),
        skip_evaluation=bool(args.skip_evaluation),
        allow_synthetic_specs=bool(args.allow_synthetic_specs),
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("APEX_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    summary = run_generate(_parse_args(list(argv or sys.argv[1:])))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
