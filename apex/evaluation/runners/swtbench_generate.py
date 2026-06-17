"""SWT-Bench prediction generation driver.

Bridges the SWT-Bench HuggingFace datasets (Lite / Verified / Full, all
in the ``nmuendler/SWT-Bench_*_bm25_27k_zsb`` family) to APEX's V5
test-generation + voting pipeline, then writes a JSONL predictions file
in the format the official ``swt_bench`` harness consumes.

Architecture: this runner is intentionally a thin wrapper around
:func:`apex.evaluation.swtbench_benchmark.evaluate_swtbench_task_with_default_generator`,
which in turn re-uses ``evaluate_testgeneval_task_with_default_generator``
and the V5 voting layer unchanged. The whole point of Phase 3 is that
APEX already implements TEX-T (the SWT-Bench winner) — this runner just
wires SWT-Bench rows into that machinery.

Differences from the TestGenEvalLite runner:

  * ``_task_from_row`` reads SWE-bench-shape fields, not TestGenEval
    fields (delegated to ``swtbench_benchmark.task_from_row``).
  * ``_record_from_result`` wraps the chosen test artifact into a
    ``model_patch`` git diff against the buggy ``test_file``, not raw
    ``preds:{"full":[src]}``.
  * The dual-state oracle is *always* engaged (every SWT-Bench row
    ships a gold patch).
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from apex.core.docker_pinning import resolve_image
from apex.core.failure_classifier import classify_failure
from apex.core.fairness_audit import (
    FairnessAuditAggregator,
    FairnessAuditMode,
    run_fairness_audit,
)
from apex.core.run_manifest import RunManifest, detect_upstream_harness_versions
from apex.evaluation.checkpointing import atomic_write_json, atomic_write_text
from apex.evaluation.prediction_telemetry import reconcile_authoritative_validation
from apex.evaluation.scorers.swtbench_upstream import SWTBenchUpstreamScorer
from apex.evaluation.swtbench_benchmark import (
    SWTBenchTask,
    evaluate_swtbench_task_with_default_generator,
    load_tasks_from_json,
    scrub_row_for_agent_prompt,
    task_from_row,
)
from apex.evaluation.testgeneval_benchmark import TestGenEvalTaskResult

logger = logging.getLogger("apex.swtbench_generate")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SWTBenchGenerateConfig:
    output_dir: str
    model_name: str = "apex"
    dataset_name: str = "nmuendler/SWT-Bench_Lite_bm25_27k_zsb"
    split: str = "test"
    from_json: str = ""
    task_ids: list[str] = field(default_factory=list)
    limit: int = 0
    parallelism: int = 1
    generation_timeout_seconds: float = 300.0
    pytest_timeout_seconds: float = 120.0
    max_repair_attempts: int = 3
    candidate_count: int = 4
    agent_models: list[str] = field(default_factory=lambda: ["codex", "claude", "gemini"])
    measure_mutation: bool = False
    measure_coverage: bool = False
    measure_assertion_effect: bool = False
    skip_existing: bool = False
    # Docker adapter wiring (used when V5 voting needs a real
    # dual-state oracle and for the optional final-acceptance gate).
    swtbench_docker_namespace: str = "aorwall"
    swtbench_python: str = ""
    # Phase 1c: subprocess retry budget for the SWT-Bench docker harness.
    # Real APEX failures are NEVER retried; only env_* and HARNESS_BUG
    # classifications are. Set to 1 to disable retry entirely.
    subprocess_max_attempts: int = 3
    # Phase 1c: fairness-audit mode for SWT-Bench. SWT-Bench has only one
    # scoring path (the upstream harness), so we pass the same scorer as
    # both private and upstream — the resulting delta is by-construction
    # zero, which lets the audit framework still emit a per-task
    # "comparable" entry rather than omitting SWT-Bench entirely.
    fairness_audit_mode: str = "off"


# ---------------------------------------------------------------------------
# Row -> task and result -> JSONL record adapters
# ---------------------------------------------------------------------------


def _task_from_row(row: dict[str, Any]) -> Optional[SWTBenchTask]:
    """Map a SWT-Bench BM25 dataset row -> SWTBenchTask.

    Thin wrapper around ``swtbench_benchmark.task_from_row`` so callers
    have a single entry point in this module.
    """

    return task_from_row(row)


def _record_from_result(
    *,
    row: dict[str, Any],
    task: SWTBenchTask,
    result: TestGenEvalTaskResult,
    model_name: str,
) -> dict[str, Any]:
    """Build the JSONL prediction record matching the SWT-Bench harness.

    Format: ``{"instance_id": ..., "model_name_or_path": ...,
    "model_patch": "<unified diff adding/modifying tests>"}``. The diff
    is built with ``difflib.unified_diff`` against the buggy
    ``test_file`` content (no git required). When the agent emits
    multiple test artifacts the runner picks the first non-empty one
    (V5 voting has already chosen the winner inside the testgeneval
    pipeline).
    """

    # Phase 1c item 1.10: bootstrap the selection flag (Phase 2 will own
    # real wiring at SELECTION TIME) so the per-artifact submission check
    # has a non-empty selected set.
    _ensure_selection_flag(result)

    artifact_text = _select_primary_artifact_text(result)
    artifact_path = _select_primary_artifact_path(result, fallback=task.focal_test_file_path)

    model_patch = _build_unified_diff(
        artifact_text=artifact_text,
        artifact_path=artifact_path or task.focal_test_file_path,
        baseline_test_source=task.baseline_test_source,
    )

    diagnostics = dict(result.diagnostics or {})
    apex_generation = dict(diagnostics.get("generation") or {})
    apex_validation = dict(diagnostics.get("apex_validation") or {})
    apex_validation.setdefault("schema_version", 1)
    apex_validation.setdefault(
        "prediction_quality",
        _infer_quality(result, apex_validation),
    )
    apex_validation = reconcile_authoritative_validation(apex_validation)
    return {
        "id": str(row.get("id") or task.instance_id),
        "instance_id": task.instance_id,
        "model_name_or_path": model_name,
        "model_patch": model_patch,
        "apex_generation": apex_generation,
        "apex_generation_duration_seconds": float(result.duration_seconds or 0.0),
        "apex_validation": apex_validation,
    }


def _collect_artifacts(
    result: TestGenEvalTaskResult,
) -> list[dict[str, Any]]:
    """Flatten generated_/shipped_/final_artifacts into a single list."""
    diagnostics = dict(result.diagnostics or {})
    out: list[dict[str, Any]] = []
    for key in ("generated_artifacts", "shipped_artifacts", "final_artifacts"):
        value = diagnostics.get(key)
        if isinstance(value, list):
            out.extend(item for item in value if isinstance(item, dict))
    return out


def _selected_for_submission(artifact: dict[str, Any]) -> bool:
    """Item 1.10: respect the explicit selection flag.

    Phase 2 will wire the selection flag at SELECTION TIME (selector.py
    will mark the chosen winner). For now, callers default it to True on
    the first non-empty artifact so the new filter is functionally a
    no-op — but the field plumbing is in place so flipping it later is
    a one-line change.
    """
    return bool(artifact.get("selected_for_submission", False))


def _select_primary_artifact_text(result: TestGenEvalTaskResult) -> str:
    """Return the source text of the artifact tagged for submission.

    Per item 1.10 of the Phase 1 plan:

      * Filter artifacts on ``selected_for_submission == True``.
      * If exactly one matches, return its content.
      * If >1 match, use the first selected non-empty artifact and log a
        warning so the duplicate selection is visible in run records.
      * If zero match, fall back to the historical "first non-empty"
        behavior.
    """
    artifacts = _collect_artifacts(result)
    selected = [a for a in artifacts if _selected_for_submission(a)]
    selected_non_empty = [
        str(a.get("content") or "") for a in selected if str(a.get("content") or "").strip()
    ]
    if len(selected) == 1:
        if selected_non_empty:
            return selected_non_empty[0]
        logger.warning(
            "_select_primary_artifact_text: selected artifact was empty; "
            "falling back to first non-empty."
        )
    if len(selected) > 1:
        logger.warning(
            "_select_primary_artifact_text: %d artifacts tagged "
            "selected_for_submission=True; using first selected non-empty. "
            "Phase 2 selector wiring should mark exactly one.",
            len(selected),
        )
        if selected_non_empty:
            return selected_non_empty[0]
    elif artifacts:
        # 0 selected. Phase 2 will guarantee >=1 selected; until then this
        # is the expected hot-path because the field defaults to False
        # except for the first artifact (set by `_record_from_result`).
        logger.debug(
            "_select_primary_artifact_text: no artifact tagged "
            "selected_for_submission; using first non-empty (legacy path)."
        )
    for artifact in artifacts:
        text = str(artifact.get("content") or "")
        if text.strip():
            return text
    return ""


def _select_primary_artifact_path(result: TestGenEvalTaskResult, *, fallback: str) -> str:
    """Return the path of the artifact tagged for submission.

    Mirrors :func:`_select_primary_artifact_text`. See item 1.10.
    """
    artifacts = _collect_artifacts(result)
    selected = [a for a in artifacts if _selected_for_submission(a)]
    if len(selected) == 1:
        path = str(selected[0].get("path") or "")
        if path:
            return path
        logger.warning(
            "_select_primary_artifact_path: selected artifact had no path; "
            "falling back to first non-empty."
        )
    elif len(selected) > 1:
        logger.warning(
            "_select_primary_artifact_path: %d artifacts tagged "
            "selected_for_submission=True; using first selected path. "
            "Phase 2 selector wiring should mark exactly one.",
            len(selected),
        )
        for artifact in selected:
            if not str(artifact.get("content") or "").strip():
                continue
            path = str(artifact.get("path") or "")
            if path:
                return path
    for artifact in artifacts:
        text = str(artifact.get("content") or "")
        if not text.strip():
            continue
        path = str(artifact.get("path") or "")
        if path:
            return path
    return fallback


def _ensure_selection_flag(result: TestGenEvalTaskResult) -> None:
    """Default ``selected_for_submission=True`` on the first non-empty
    artifact when no artifact carries the flag.

    This is the Phase-1c shim: Phase 2 owns the *real* selection wiring
    (selector.py will set this from the V5 winner). Until then we
    bootstrap the field so consumers can trust it.

    The mutation is in-place on the diagnostics dict; that dict is the
    same object the rest of the runner reads.
    """
    artifacts = _collect_artifacts(result)
    if not artifacts:
        return
    if any(_selected_for_submission(a) for a in artifacts):
        return
    for artifact in artifacts:
        text = str(artifact.get("content") or "")
        if text.strip():
            artifact["selected_for_submission"] = True
            return


def _build_unified_diff(
    *,
    artifact_text: str,
    artifact_path: str,
    baseline_test_source: str,
) -> str:
    """Render a model_patch unified diff the swt_bench harness accepts.

    Uses ``difflib.unified_diff`` so we don't need a real git checkout.
    Falls back to a creation-only diff (against an empty buggy file)
    when the dataset row didn't ship a baseline test source.
    """

    if not artifact_text.strip():
        return ""
    rel_path = (artifact_path or "tests/test_apex_swtbench.py").lstrip("/")
    a_lines = (baseline_test_source or "").splitlines(keepends=True)
    b_lines = artifact_text.splitlines(keepends=True)
    if a_lines and not a_lines[-1].endswith("\n"):
        a_lines[-1] += "\n"
    if b_lines and not b_lines[-1].endswith("\n"):
        b_lines[-1] += "\n"
    diff_lines = list(
        difflib.unified_diff(
            a_lines,
            b_lines,
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            n=3,
        )
    )
    if not diff_lines:
        return ""
    header = f"diff --git a/{rel_path} b/{rel_path}\n"
    return header + "".join(diff_lines)


def _infer_quality(result: TestGenEvalTaskResult, validation: dict[str, Any]) -> str:
    if validation.get("prediction_quality"):
        return str(validation["prediction_quality"])
    if not result.success:
        return "failed"
    if float(result.all_pass_at_1 or 0.0) >= 1.0:
        return "clean"
    if float(result.pass_at_1 or 0.0) > 0.0:
        return "filtered_only"
    return "unknown"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_tasks_from_huggingface(
    dataset_name: str,
    split: str,
) -> list[tuple[dict[str, Any], SWTBenchTask]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment-only
        raise SystemExit(
            "the 'datasets' package is required to load SWT-Bench from HF.\n"
            "install with: pip install datasets"
        ) from exc

    dataset = load_dataset(dataset_name, split=split)
    rows: list[tuple[dict[str, Any], SWTBenchTask]] = []
    for row in dataset:
        row_dict = dict(row)
        task = _task_from_row(row_dict)
        if task is None:
            logger.warning(
                "skipping malformed row instance_id=%s",
                row_dict.get("instance_id"),
            )
            continue
        rows.append((row_dict, task))
    return rows


def _filter_rows(
    rows: list[tuple[dict[str, Any], SWTBenchTask]],
    *,
    task_ids: Iterable[str],
    limit: int,
) -> list[tuple[dict[str, Any], SWTBenchTask]]:
    wanted = {str(tid) for tid in task_ids if str(tid)}
    if wanted:
        rows = [
            (row, task)
            for row, task in rows
            if str(row.get("id") or "") in wanted or task.instance_id in wanted
        ]
    if limit and limit > 0:
        rows = rows[:limit]
    return rows


# ---------------------------------------------------------------------------
# Per-task processing
# ---------------------------------------------------------------------------


def _process_one(
    *,
    row: dict[str, Any],
    task: SWTBenchTask,
    output_dir: Path,
    config: SWTBenchGenerateConfig,
    manifest: Optional[RunManifest] = None,
) -> dict[str, Any]:
    instance_dir = output_dir / "generation" / str(row.get("id") or task.instance_id)
    instance_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # Phase 1c: capture per-task manifest snapshot if not provided. The
    # caller (run_generate) typically provides a single shared manifest
    # for the whole run; the per-task fallback keeps direct callers of
    # _process_one (and tests) honest.
    if manifest is None:
        manifest = RunManifest.capture(seed=None)

    # Hygiene: scrub gold-leak fields BEFORE the task ever touches the
    # generation pipeline. The driver shim does not currently render
    # ``row`` into a prompt — but defense-in-depth is the rule for
    # SWE-bench-style rows (cf. Phase 2 plan's contamination note).
    row = scrub_row_for_agent_prompt(row)

    docker_token = None
    try:
        from apex.evaluation.docker_acceptance_adapter import (
            DockerTaskContext,
            reset_docker_task_context,
            set_docker_task_context,
        )
        from apex.evaluation.swtbench_docker_adapter import (
            DEFAULT_SUBPROCESS_TIMEOUT_SECONDS,
            make_swtbench_docker_adapter,
        )

        task_row = task.to_huggingface_row()
        log_dir = output_dir / "swtbench_docker_logs"
        # Phase 1c item 1.5: pin the docker namespace via the registry
        # before the adapter shells out so the manifest carries the
        # resolved digest.
        try:
            resolve_image(
                f"{config.swtbench_docker_namespace}/sweb.eval.x86_64:latest",
                record_to_manifest=manifest,
            )
        except Exception:  # pragma: no cover - best-effort
            pass
        adapter = make_swtbench_docker_adapter(
            task_instance=task_row,
            model_name=config.model_name,
            dataset_name=config.dataset_name,
            log_dir=log_dir,
            namespace=config.swtbench_docker_namespace,
            swt_bench_python=config.swtbench_python,
            timeout_seconds=DEFAULT_SUBPROCESS_TIMEOUT_SECONDS,
            max_attempts=int(config.subprocess_max_attempts),
        )
        docker_token = set_docker_task_context(
            DockerTaskContext(
                task_instance=task_row,
                namespace=config.swtbench_docker_namespace,
                log_dir=log_dir,
                adapter=adapter,
            )
        )
        result = evaluate_swtbench_task_with_default_generator(
            task=task,
            output_dir=instance_dir,
            generation_timeout_seconds=config.generation_timeout_seconds,
            pytest_timeout_seconds=config.pytest_timeout_seconds,
            max_repair_attempts=config.max_repair_attempts,
            candidate_count=config.candidate_count,
            agent_models=list(config.agent_models or []),
            measure_mutation=config.measure_mutation,
            measure_coverage=config.measure_coverage,
            measure_assertion_effect=config.measure_assertion_effect,
        )
    except Exception as exc:  # pragma: no cover - defensive
        # Phase 1c: classify the exception text via the orchestrator-wide
        # taxonomy so the per-task record carries failure_class.
        classification = classify_failure(
            stderr=f"{type(exc).__name__}: {exc}",
            stdout="",
            returncode=1,
            context={"phase": "test_execution"},
        )
        result = TestGenEvalTaskResult(
            instance_id=task.instance_id,
            success=False,
            pass_at_1=0.0,
            error=f"{type(exc).__name__}: {exc}",
            duration_seconds=time.time() - started,
            diagnostics={
                "apex_validation": {
                    "failure_class": classification.failure_class.value,
                    "failure_classification": classification.to_dict(),
                },
            },
        )
    finally:
        if docker_token is not None:
            reset_docker_task_context(docker_token)
    record = _record_from_result(
        row=row,
        task=task,
        result=result,
        model_name=config.model_name,
    )
    # Phase 1c: attach the orchestrator-wide failure_class to the record.
    diag = dict(result.diagnostics or {})
    validation = dict(diag.get("apex_validation") or {})
    if validation.get("failure_class"):
        record.setdefault("apex_validation", {})
        record["apex_validation"]["failure_class"] = validation["failure_class"]
        if validation.get("failure_classification"):
            record["apex_validation"]["failure_classification"] = validation[
                "failure_classification"
            ]
    # Stash the result on the record so run_generate can run the
    # fairness scorer without re-evaluating the task.
    record["_apex_internal_result"] = result
    return record


def run_generate(config: SWTBenchGenerateConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.from_json:
        tasks = load_tasks_from_json(config.from_json)
        rows = [(t.to_huggingface_row(), t) for t in tasks]
    else:
        rows = _load_tasks_from_huggingface(config.dataset_name, config.split)
    rows = _filter_rows(rows, task_ids=config.task_ids, limit=config.limit)

    preds_dir = output_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_path = preds_dir / (
        f"{config.model_name}__{Path(config.dataset_name).name}__0__{config.split}.jsonl"
    )
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    if config.skip_existing:
        existing_ids = {p.stem for p in records_dir.glob("*.json")}
        before = len(rows)
        rows = [
            (row, task)
            for row, task in rows
            if str(row.get("id") or task.instance_id) not in existing_ids
        ]
        if before != len(rows):
            logger.warning(
                "skip_existing: filtered %d -> %d rows (skipped %d already in %s)",
                before,
                len(rows),
                before - len(rows),
                records_dir,
            )

    # Phase 1c: capture a RunManifest snapshot up front so reviewers can
    # reproduce this run byte-for-byte.
    started_at_ts = time.time()
    run_manifest = RunManifest.capture(
        seed=None,
        additional_metadata={
            "benchmark": "swtbench",
            "model_name": config.model_name,
            "dataset": config.dataset_name,
            "split": config.split,
            "total_tasks": len(rows),
            "parallelism": config.parallelism,
            "started_at": started_at_ts,
            "prediction_path": str(preds_path),
            "generation": {
                "generation_timeout_seconds": config.generation_timeout_seconds,
                "pytest_timeout_seconds": config.pytest_timeout_seconds,
                "max_repair_attempts": config.max_repair_attempts,
                "candidate_count": config.candidate_count,
                "agent_models": list(config.agent_models),
            },
            "subprocess_max_attempts": int(config.subprocess_max_attempts),
            "fairness_audit_mode": str(config.fairness_audit_mode),
        },
    )
    # Record upstream harness versions (swt_bench, etc.) on the manifest.
    for harness_name, harness_version in detect_upstream_harness_versions().items():
        run_manifest.add_upstream_harness(harness_name, harness_version)
    # Pin the docker namespace coarsely (per-instance images resolve at
    # adapter time inside _process_one).
    try:
        resolve_image(
            f"{config.swtbench_docker_namespace}/sweb.eval.x86_64:latest",
            record_to_manifest=run_manifest,
        )
    except Exception:  # pragma: no cover - best-effort
        pass

    # Phase 1c: optional fairness audit.
    fairness_aggregator: Optional[FairnessAuditAggregator] = None
    fairness_mode = FairnessAuditMode(config.fairness_audit_mode)
    if fairness_mode in {FairnessAuditMode.PARALLEL, FairnessAuditMode.UPSTREAM_ONLY}:
        fairness_aggregator = FairnessAuditAggregator()

    # Persist the manifest early so a crashed run still leaves a manifest.
    run_manifest.write_to(output_dir)

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    parallelism = max(1, int(config.parallelism))

    def _record_one_failure(row_dict: dict[str, Any], exc: BaseException) -> None:
        rid = str(row_dict.get("id") or row_dict.get("instance_id") or "unknown")
        logger.exception("Task %s raised; isolating failure: %s", rid, exc)
        failures.append({"id": rid, "error": f"{type(exc).__name__}: {exc}"})

    def _finalize_record(record: dict[str, Any], task: SWTBenchTask) -> dict[str, Any]:
        """Strip the in-memory result handle and run the fairness audit."""
        result_obj = record.pop("_apex_internal_result", None)
        if fairness_aggregator is not None:
            scorer = SWTBenchUpstreamScorer()
            # SWT-Bench has only one scoring path -- pass the same scorer
            # as both private and upstream so the by-construction-zero
            # delta is recorded honestly. See module docstring.
            try:
                delta = run_fairness_audit(
                    task=task,
                    apex_artifacts=result_obj if result_obj is not None else record,
                    private_scorer=scorer,
                    upstream_scorer=scorer,
                    extra_notes=[
                        "SWT-Bench has a single upstream scorer; delta is by-construction zero."
                    ],
                )
                fairness_aggregator.add_task(delta)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("fairness audit failed for %s: %s", task.instance_id, exc)
        return record

    if parallelism == 1:
        for row, task in rows:
            try:
                record = _process_one(
                    row=row,
                    task=task,
                    output_dir=output_dir,
                    config=config,
                    manifest=run_manifest,
                )
                record = _finalize_record(record, task)
                atomic_write_json(records_dir / f"{record['id']}.json", record)
                completed.append(record)
            except Exception as exc:
                _record_one_failure(row, exc)
    else:
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            futures = {
                pool.submit(
                    _process_one,
                    row=row,
                    task=task,
                    output_dir=output_dir,
                    config=config,
                    manifest=run_manifest,
                ): (row, task)
                for row, task in rows
            }
            for future in as_completed(futures):
                row_dict, task_obj = futures[future]
                try:
                    record = future.result()
                    record = _finalize_record(record, task_obj)
                    atomic_write_json(records_dir / f"{record['id']}.json", record)
                    completed.append(record)
                except Exception as exc:
                    _record_one_failure(row_dict, exc)

    # Persist the fairness aggregator output if it was active.
    if fairness_aggregator is not None and len(fairness_aggregator) > 0:
        try:
            fairness_aggregator.write_to(output_dir / "fairness")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("failed to write fairness audit output: %s", exc)

    # Re-write the manifest at the end so any docker images / harnesses
    # added during the run are persisted.
    run_manifest.write_to(output_dir)

    completed.sort(key=lambda r: str(r.get("id") or ""))
    atomic_write_text(
        preds_path,
        "".join(json.dumps(rec, sort_keys=True) + "\n" for rec in completed),
    )

    finished = time.time()
    summary = {
        "status": (
            "failed" if len(rows) > 0 and not completed else ("ok" if not failures else "partial")
        ),
        "prediction_path": str(preds_path),
        "records_dir": str(records_dir),
        "task_count": len(rows),
        "predictions_written": len(completed),
        "failures": failures,
        "started_at": started_at_ts,
        "finished_at": finished,
        "elapsed_seconds": finished - started_at_ts,
    }
    if fairness_aggregator is not None:
        summary["fairness_audit"] = {
            "mode": fairness_mode.value,
            "tasks_audited": len(fairness_aggregator),
            "summary": fairness_aggregator.summary(),
        }
    atomic_write_json(output_dir / "generation_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> SWTBenchGenerateConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SWT-Bench predictions through APEX's V5 test-gen + "
            "voting pipeline. Writes a JSONL preds file consumable by the "
            "official swt_bench harness."
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="apex")
    parser.add_argument(
        "--dataset-name",
        default="nmuendler/SWT-Bench_Lite_bm25_27k_zsb",
        help=(
            "HuggingFace dataset name. Defaults to SWT-Bench Lite. Use "
            "nmuendler/SWT-Bench_Verified_bm25_27k_zsb for Verified or "
            "nmuendler/SWT-Bench_bm25_27k_zsb for Full."
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--from-json",
        default="",
        help="If set, load tasks from a local JSON file instead of HuggingFace.",
    )
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0, help="Cap task count (0 = no limit).")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--generation-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--pytest-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-repair-attempts", type=int, default=3)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--measure-mutation", action="store_true")
    parser.add_argument("--measure-coverage", action="store_true")
    parser.add_argument("--measure-assertion-effect", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--swtbench-docker-namespace",
        default="aorwall",
        help=(
            "Docker image namespace for the swt_bench harness. Defaults "
            "to upstream's 'aorwall' family."
        ),
    )
    parser.add_argument(
        "--swtbench-python",
        default="",
        help=(
            "Python interpreter that has the swt-bench package installed. "
            "Empty (default) = same as the runner's interpreter; the "
            "harness wrapper will invoke the resolved SWT-Bench module on "
            "this binary."
        ),
    )
    parser.add_argument(
        "--agent",
        action="append",
        default=None,
        choices=("codex", "claude", "gemini", "opencode", "metacode"),
        help=(
            "Add an agent to the multi-agent ensemble (repeatable). "
            "DEFAULT is codex+claude+gemini; pass an OpenCode-family agent "
            "explicitly for ablations."
        ),
    )
    args = parser.parse_args(argv)
    return SWTBenchGenerateConfig(
        output_dir=args.output_dir,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        split=args.split,
        from_json=args.from_json,
        task_ids=list(args.task_id or []),
        limit=int(args.limit or 0),
        parallelism=int(args.parallelism or 1),
        generation_timeout_seconds=float(args.generation_timeout_seconds),
        pytest_timeout_seconds=float(args.pytest_timeout_seconds),
        max_repair_attempts=int(args.max_repair_attempts),
        candidate_count=int(args.candidate_count),
        measure_mutation=bool(args.measure_mutation),
        measure_coverage=bool(args.measure_coverage),
        measure_assertion_effect=bool(args.measure_assertion_effect),
        skip_existing=bool(args.skip_existing),
        swtbench_docker_namespace=str(args.swtbench_docker_namespace),
        swtbench_python=str(args.swtbench_python or ""),
        agent_models=list(args.agent)
        if args.agent is not None
        else ["codex", "claude", "gemini"],
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
