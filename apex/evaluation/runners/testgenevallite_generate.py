"""TestGenEvalLite prediction generation driver.

Bridges the kjain14/testgenevallite HuggingFace dataset to the V4-equipped
generation pipeline (W1 final acceptance gate, W2 atomic acceptance via
isolation, W3 AST roundtrip, W5 preflights, W8 deterministic repair, W10
reporting), then writes a JSONL predictions file in the format the official
``run_evaluation.py`` harness consumes. Optionally invokes the harness via
``apex.evaluation.runners.testgenevallite``.

Generation, validation, repair, oracle synthesis, and emission contain zero
benchmark-conditional code paths. Everything benchmark-specific is bounded
by the dataset row -> ``TestGenEvalTask`` mapping in ``_task_from_row`` and
the prediction emission format in ``_record_from_result``.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import signal
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from apex.core.parallelism import default_task_parallelism, resolve_task_parallelism
from apex.evaluation.checkpointing import atomic_write_json, atomic_write_text, load_json_if_exists
from apex.evaluation.prediction_telemetry import reconcile_authoritative_validation
from apex.evaluation.testgeneval_benchmark import (
    TestGenEvalTask,
    TestGenEvalTaskResult,
    evaluate_testgeneval_task_with_default_generator,
    load_tasks_from_json,
)

logger = logging.getLogger("apex.testgenevallite_generate")


@dataclass(frozen=True)
class TestGenEvalLiteGenerateConfig:
    output_dir: str
    model_name: str = "apex"
    dataset_name: str = "kjain14/testgenevallite"
    split: str = "test"
    from_json: str = ""
    task_ids: list[str] = field(default_factory=list)
    limit: int = 0
    # Default to host-CPU/Docker-aware parallelism (capped at 4 — matches
    # ``scripts/launch_*.sh``). Explicit ``--parallelism N`` wins.
    parallelism: int = field(default_factory=default_task_parallelism)
    generation_timeout_seconds: float = 300.0
    pytest_timeout_seconds: float = 120.0
    max_repair_attempts: int = 3
    candidate_count: int = 3
    # Multi-agent ensemble: one candidate per named agent. SOTA-aligned default
    # is the full 4-agent heterogeneous ensemble (codex+claude+gemini+opencode):
    # heterogeneous-agent diversity is the strongest test-time-scaling setting
    # (Trae/CodeMonkeys-style), so we never downgrade it by default. Single-agent
    # is opt-in via --agent X. Passing an empty list keeps candidate_count and
    # uses the default single-agent config.
    agent_models: list[str] = field(
        default_factory=lambda: ["codex", "claude", "gemini", "opencode"]
    )
    measure_mutation: bool = False
    measure_coverage: bool = False
    measure_assertion_effect: bool = False
    # Target-environment execution. When an official docker repo is supplied,
    # bind the task's benchmark adapter before generation so validation,
    # repair, and final acceptance all run in the same environment.
    target_environment_enabled: bool = True
    target_environment_required: bool = False
    # Post-generation docker-based final acceptance gate (W1 ground-truth).
    docker_gate_enabled: bool = False
    docker_official_repo: str = ""
    docker_namespace: str = "kdjain"
    docker_gate_iterations: int = 3
    docker_gate_timeout_seconds: int = 600
    docker_gate_keep_minimum: int = 1
    # V5 cross-candidate voting: patch-as-oracle dual-version verifier +
    # anti-hack ledger + (optional) LLM critic + (optional) mutation
    # tiebreak. Capability-gated on an explicit dual-state oracle; this is
    # intentionally independent of the final Docker acceptance gate.
    v5_voting_enabled: bool = True
    # Capability-driven: when a row supplies the buggy/fixed dual-state
    # signals (repo + version + base_commit + non-empty patch + docker
    # repo path), V5 patch-as-oracle voting engages. Rows that don't
    # supply those signals fall through to the no-op skip path with a
    # clear diagnostic. We default ON so users don't have to know which
    # of their benchmarks happen to support it.
    v5_dual_state_oracle_enabled: bool = True
    # Cap on patch_surrogate's parallel CLI fan-out. Default 1 because
    # spawning 4 simultaneous agentic CLI subprocesses (one per model)
    # while generation is also active reliably crashes the orchestrator
    # on macOS (silent process death; tracked as the "patch fan-out
    # silent exit" bug). Raise per-host once the underlying signal
    # interaction is fully diagnosed.
    v5_patch_parallelism: int = 1
    # Whether to actually run the patch_surrogate fan-out. When False,
    # V5 voting uses the dual_version_verifier's fallback path
    # ("fails-on-buggy" weak proxy + mutation tiebreak + anti-hack
    # downweight), which is degraded but still better than no V5. We
    # default ON; operators on hosts that hit the patch_surrogate
    # crash should pass --no-v5-patch-surrogate.
    v5_use_patch_surrogate: bool = True
    # When True, rows whose record file already exists in records_dir are
    # skipped. Used to resume a partially-completed run without redoing
    # work.
    skip_existing: bool = False
    v5_use_anti_hack: bool = True
    v5_use_llm_critic: bool = True
    v5_critic_agent: str = "claude"
    v5_use_mutation_tiebreak: bool = True
    v5_mutation_n: int = 5


def _task_from_row(row: dict[str, Any]) -> Optional[TestGenEvalTask]:
    """Map a TestGenEvalLite dataset row onto APEX's TestGenEvalTask."""

    instance_id = str(row.get("instance_id") or "").strip()
    code_file = str(row.get("code_file") or "").strip()
    code_src = row.get("code_src") or ""
    if not instance_id or not code_file or not code_src:
        return None
    return TestGenEvalTask(
        instance_id=instance_id,
        focal_method_path=code_file,
        focal_method_source=str(code_src),
        existing_test_path=str(row.get("test_file") or "").strip(),
        existing_test_source=str(row.get("test_src") or ""),
        problem_statement=str(row.get("problem_statement") or ""),
        language="python",
        repo_path=None,
        metadata={
            "benchmark": "testgenevallite",
            "source_repo": str(row.get("repo") or ""),
            "version": str(row.get("version") or ""),
            "base_commit": str(row.get("base_commit") or ""),
            "instance_id": instance_id,
        },
    )


def _record_from_result(
    *,
    row: dict[str, Any],
    task: TestGenEvalTask,
    result: TestGenEvalTaskResult,
    model_name: str,
) -> dict[str, Any]:
    """Build the JSONL prediction record matching the kjain14 harness format."""

    _ensure_selection_flag(result)
    artifact_text = _select_primary_artifact_text(result)
    diagnostics = dict(result.diagnostics or {})
    apex_generation = dict(diagnostics.get("generation") or {})
    apex_validation = dict(diagnostics.get("apex_validation") or {})
    apex_validation.setdefault("schema_version", 1)
    apex_validation.setdefault(
        "prediction_quality",
        _infer_quality(result, apex_validation),
    )
    apex_validation = reconcile_authoritative_validation(apex_validation)
    # Audit C1: persist EVERY remaining diagnostic key so V5 voting,
    # mock-path / attribute-chain findings, test-dedup, repair-attempt
    # counts, and any future diagnostic survive to disk. Without this
    # block all 160 records in the v5_full_20260509 run had empty
    # `diagnostics: {}` and we lost root-cause data for every failure.
    # We use the existing cycle scrubber so a self-referential dict
    # doesn't blow up the line writer (Audit C5).
    apex_diagnostics: dict[str, Any] = {}
    _CARRIED_KEYS_HANDLED_ELSEWHERE = {"generation", "apex_validation"}
    for key, value in diagnostics.items():
        if key in _CARRIED_KEYS_HANDLED_ELSEWHERE:
            continue
        apex_diagnostics[key] = value
    if apex_diagnostics:
        try:
            from apex.evaluation.checkpointing import _safe_jsonable

            apex_diagnostics = _safe_jsonable(apex_diagnostics)
        except Exception:  # pragma: no cover - defensive
            apex_diagnostics = {}
    return {
        "id": str(row.get("id") or task.instance_id),
        "instance_id": task.instance_id,
        "model_name_or_path": model_name,
        "preds": {"full": [artifact_text]},
        "apex_generation": apex_generation,
        "apex_generation_duration_seconds": float(result.duration_seconds or 0.0),
        "apex_validation": apex_validation,
        "apex_diagnostics": apex_diagnostics,
    }


def _collect_artifacts(result: TestGenEvalTaskResult) -> list[dict[str, Any]]:
    diagnostics = dict(result.diagnostics or {})
    out: list[dict[str, Any]] = []
    for key in ("generated_artifacts", "shipped_artifacts", "final_artifacts"):
        value = diagnostics.get(key)
        if isinstance(value, list):
            out.extend(item for item in value if isinstance(item, dict))
    if not out:
        gate_diag = dict(diagnostics.get("final_acceptance_gate") or {})
        post = dict(gate_diag.get("post_result") or {})
        post_diag = dict(post.get("diagnostics") or {})
        gen = post_diag.get("generated_artifacts")
        if isinstance(gen, list):
            out.extend(item for item in gen if isinstance(item, dict))
    return out


def _selected_for_submission(artifact: dict[str, Any]) -> bool:
    return bool(artifact.get("selected_for_submission", False))


def _select_primary_artifact_text(result: TestGenEvalTaskResult) -> str:
    """Pull the source text of the artifact tagged for submission."""

    candidate_artifacts = _collect_artifacts(result)
    selected = [a for a in candidate_artifacts if _selected_for_submission(a)]
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
            "selected_for_submission=True; using first selected non-empty.",
            len(selected),
        )
        if selected_non_empty:
            return selected_non_empty[0]
    elif candidate_artifacts:
        logger.debug(
            "_select_primary_artifact_text: no artifact tagged "
            "selected_for_submission; using first non-empty."
        )
    for artifact in candidate_artifacts:
        text = str(artifact.get("content") or "")
        if text.strip():
            return text
    return ""


def _ensure_selection_flag(result: TestGenEvalTaskResult) -> None:
    """Default selected_for_submission=True on the first non-empty artifact."""

    artifacts = _collect_artifacts(result)
    if not artifacts:
        return
    if any(_selected_for_submission(a) for a in artifacts):
        return
    for artifact in artifacts:
        if str(artifact.get("content") or "").strip():
            artifact["selected_for_submission"] = True
            return


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


def _load_tasks_from_huggingface(
    dataset_name: str,
    split: str,
) -> list[tuple[dict[str, Any], TestGenEvalTask]]:
    """Lazy-import the ``datasets`` package; raise a clear error if missing."""

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment-only
        raise SystemExit(
            "the 'datasets' package is required to load TestGenEvalLite from HF.\n"
            "install with: pip install datasets"
        ) from exc

    dataset = load_dataset(dataset_name, split=split)
    rows: list[tuple[dict[str, Any], TestGenEvalTask]] = []
    for row in dataset:
        row_dict = dict(row)
        task = _task_from_row(row_dict)
        if task is None:
            logger.warning("skipping malformed row instance_id=%s", row_dict.get("instance_id"))
            continue
        rows.append((row_dict, task))
    return rows


def _filter_rows(
    rows: list[tuple[dict[str, Any], TestGenEvalTask]],
    *,
    task_ids: Iterable[str],
    limit: int,
) -> list[tuple[dict[str, Any], TestGenEvalTask]]:
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


def _is_complete_generation_record(path: Path, expected_id: str) -> bool:
    payload = load_json_if_exists(path)
    if not isinstance(payload, dict):
        return False
    record_id = str(payload.get("id") or payload.get("instance_id") or "").strip()
    if record_id != str(expected_id).strip():
        return False
    preds = payload.get("preds")
    if not isinstance(preds, dict) or not isinstance(preds.get("full"), list):
        return False
    validation = payload.get("apex_validation")
    if isinstance(validation, dict):
        quality = str(validation.get("prediction_quality") or "").strip().lower()
        if quality in {"failed", "filtered_only", "partial", "clean", "fallback_last_valid"}:
            return True
        failure_reason = str(validation.get("failure_reason") or "").strip()
        if failure_reason:
            return True
    return "apex_generation_duration_seconds" in payload


def _process_one(
    *,
    row: dict[str, Any],
    task: TestGenEvalTask,
    output_dir: Path,
    config: TestGenEvalLiteGenerateConfig,
) -> dict[str, Any]:
    instance_dir = output_dir / "generation" / str(row.get("id") or task.instance_id)
    instance_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    # Bind a per-task docker context before generation so every dynamic
    # validation/repair/gate step executes in the target project environment.
    docker_token = None
    if config.target_environment_enabled and config.docker_official_repo:
        from apex.evaluation.docker_acceptance_adapter import (
            DockerTaskContext,
            make_docker_testgenevallite_adapter,
            set_docker_task_context,
        )

        log_dir = output_dir / "docker_gate_logs"
        adapter = make_docker_testgenevallite_adapter(
            task_instance=row,
            model_name=config.model_name,
            namespace=config.docker_namespace,
            log_dir=log_dir,
            official_repo=Path(config.docker_official_repo),
            timeout_seconds=config.docker_gate_timeout_seconds,
        )
        ctx = DockerTaskContext(
            task_instance=dict(row),
            namespace=config.docker_namespace,
            official_repo=Path(config.docker_official_repo).expanduser().resolve(),
            log_dir=log_dir,
            adapter=adapter,
        )
        docker_token = set_docker_task_context(ctx)
    elif config.target_environment_required:
        return _record_from_result(
            row=row,
            task=task,
            result=TestGenEvalTaskResult(
                instance_id=task.instance_id,
                success=False,
                pass_at_1=0.0,
                error="target environment required but docker_official_repo is not configured",
                duration_seconds=time.time() - started,
                diagnostics={
                    "apex_validation": {
                        "prediction_quality": "failed",
                        "target_environment_required": True,
                        "target_environment_status": "missing_docker_official_repo",
                    }
                },
            ),
            model_name=config.model_name,
        )
    try:
        result = evaluate_testgeneval_task_with_default_generator(
            task=task,
            output_dir=instance_dir,
            generation_timeout_seconds=config.generation_timeout_seconds,
            pytest_timeout_seconds=config.pytest_timeout_seconds,
            measure_mutation=config.measure_mutation,
            measure_coverage=config.measure_coverage,
            measure_assertion_effect=config.measure_assertion_effect,
            measure_stability=False,
            install_repo=False,
            candidate_count=config.candidate_count,
            max_repair_attempts=config.max_repair_attempts,
            agent_models=list(config.agent_models or []),
        )
    except Exception as exc:  # pragma: no cover - defensive
        result = TestGenEvalTaskResult(
            instance_id=task.instance_id,
            success=False,
            pass_at_1=0.0,
            error=f"{type(exc).__name__}: {exc}",
            duration_seconds=time.time() - started,
        )
    finally:
        if docker_token is not None:
            from apex.evaluation.docker_acceptance_adapter import reset_docker_task_context

            reset_docker_task_context(docker_token)
    if config.v5_voting_enabled:
        v5_ready, v5_skip = _v5_dual_state_capability(row=row, config=config)
        if len(config.agent_models or []) < 2:
            v5_ready = False
            v5_skip = {
                "status": "skipped_insufficient_agents",
                "agent_count": len(config.agent_models or []),
            }
        if v5_ready:
            try:
                result = _apply_v5_voting(
                    result=result,
                    row=row,
                    task=task,
                    output_dir=output_dir,
                    config=config,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception(
                    "V5 voting raised for %s; preserving pre-V5 result",
                    task.instance_id,
                )
                result.diagnostics["v5_voting"] = {
                    "status": "errored",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            result.diagnostics["v5_voting"] = dict(v5_skip)
            logger.warning(
                "V5 voting requested for %s but skipped: %s",
                task.instance_id,
                v5_skip.get("reason") or v5_skip.get("status"),
            )
    if config.docker_gate_enabled:
        try:
            result = _apply_docker_gate(
                result=result,
                row=row,
                output_dir=output_dir,
                config=config,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(
                "Docker gate raised for %s; preserving pre-gate result",
                task.instance_id,
            )
            result.diagnostics.setdefault("apex_validation", {})["docker_final_acceptance_gate"] = {
                "status": "errored",
                "error": f"{type(exc).__name__}: {exc}",
            }
    return _record_from_result(
        row=row,
        task=task,
        result=result,
        model_name=config.model_name,
    )


def _v5_candidate_records_from_result(
    result: TestGenEvalTaskResult,
) -> list[dict[str, Any]]:
    """Return every candidate artifact available to V5 voting.

    ``generated_artifacts`` contains only the selected suite after the default
    generator ranks candidates. Prefer the explicit candidate bundle when it is
    present, then fall back to the selected artifacts for older diagnostics.
    """

    diagnostics = dict(result.diagnostics or {})
    records: list[dict[str, Any]] = []
    bundle = list(diagnostics.get("candidate_artifact_bundle") or [])
    for bundle_index, entry in enumerate(bundle):
        if not isinstance(entry, dict):
            continue
        candidate_id = str(entry.get("candidate_id") or f"candidate_{bundle_index + 1}")
        artifacts = list(entry.get("artifacts") or [])
        generation = dict(entry.get("generation") or {})
        agent = str(
            generation.get("agent")
            or generation.get("model")
            or generation.get("generator")
            or candidate_id
        )
        for artifact_index, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict):
                continue
            content = str(artifact.get("content") or "")
            if not content.strip():
                continue
            raw_metadata = artifact.get("metadata")
            metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
            records.append(
                {
                    "test_id": str(
                        metadata.get("test_id")
                        or artifact.get("test_id")
                        or f"{candidate_id}:{artifact_index}"
                    ),
                    "agent": str(metadata.get("agent") or artifact.get("agent") or agent),
                    "candidate_id": candidate_id,
                    "artifact_path": str(artifact.get("path") or "tests/test_apex.py"),
                    "artifact_content": content,
                }
            )
    if records:
        return records

    artifacts_raw = list(diagnostics.get("generated_artifacts") or [])
    for index, artifact in enumerate(artifacts_raw):
        if not isinstance(artifact, dict):
            continue
        content = str(artifact.get("content") or "")
        if not content.strip():
            continue
        raw_metadata = artifact.get("metadata")
        metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        records.append(
            {
                "test_id": str(metadata.get("agent") or artifact.get("agent") or f"test_{index}"),
                "agent": str(metadata.get("agent") or artifact.get("agent") or f"agent_{index}"),
                "candidate_id": str(artifact.get("candidate_id") or f"selected_{index}"),
                "artifact_path": str(artifact.get("path") or "tests/test_apex.py"),
                "artifact_content": content,
            }
        )
    return records


def _v5_dual_state_capability(
    *,
    row: dict[str, Any],
    config: TestGenEvalLiteGenerateConfig,
) -> tuple[bool, dict[str, Any]]:
    """Return whether V5 patch voting has a valid dual-state oracle.

    This is capability-based, not benchmark-name based. Patch-as-oracle voting
    is valid only when callers explicitly opt in (or the row advertises the
    capability), Docker harness metadata is present, and a non-empty patch can
    represent the fixed state.
    """

    enabled = bool(
        config.v5_dual_state_oracle_enabled
        or row.get("apex_dual_state_oracle")
        or row.get("dual_state_oracle")
    )
    if not enabled:
        return False, {
            "status": "skipped_no_dual_state_oracle",
            "reason": "v5_dual_state_oracle_not_enabled",
        }
    if not str(config.docker_official_repo or "").strip():
        return False, {
            "status": "skipped_no_docker_repo",
            "reason": "docker_official_repo_required_for_dual_state_voting",
        }
    missing = [
        key for key in ("repo", "version", "base_commit") if not str(row.get(key) or "").strip()
    ]
    if missing:
        return False, {
            "status": "skipped_missing_dual_state_fields",
            "reason": "missing_required_task_fields",
            "missing_fields": missing,
        }
    if not str(row.get("patch") or "").strip():
        return False, {
            "status": "skipped_missing_patch",
            "reason": "non_empty_patch_required_for_fixed_state",
        }
    return True, {"status": "available"}


def _v5_captured_oracle_values(result: TestGenEvalTaskResult) -> dict[str, Any]:
    diagnostics = dict(result.diagnostics or {})
    payloads: list[dict[str, Any]] = []
    generation = diagnostics.get("generation")
    if isinstance(generation, dict):
        oracle = generation.get("oracle_grounding")
        if isinstance(oracle, dict):
            payloads.append(oracle)
    for entry in list(diagnostics.get("candidate_artifact_bundle") or []):
        if not isinstance(entry, dict):
            continue
        gen = entry.get("generation")
        if isinstance(gen, dict) and isinstance(gen.get("oracle_grounding"), dict):
            payloads.append(dict(gen["oracle_grounding"]))
    values: dict[str, Any] = {}
    for payload in payloads:
        # Prefer the pre-summarized {repr_key: value} dict written by the
        # producer side (apex.evaluation.oracle_capture
        # .summarize_captures_for_diagnostics). This is the canonical
        # ledger-friendly shape; fall back to walking captures only when
        # the payload pre-dates the producer wiring.
        captured_values = payload.get("captured_values")
        if isinstance(captured_values, dict) and captured_values:
            for key, value in captured_values.items():
                if isinstance(key, str) and key:
                    values[key] = value
        for capture in list(payload.get("captures") or []):
            if not isinstance(capture, dict):
                continue
            for key in ("repr_text", "exc_type", "exc_message"):
                value = capture.get(key)
                if value:
                    values[str(value)] = value
            if "value" in capture:
                value = capture.get("value")
                values[repr(value)] = value
                if isinstance(value, (str, int, float, bool)):
                    values[str(value)] = value
    return values


def _v5_focal_signature_summary(task: TestGenEvalTask) -> str:
    import ast

    try:
        tree = ast.parse(task.focal_method_source or "")
    except SyntaxError:
        return ""
    lines: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines.append(f"def {node.name}{_ast_signature(node)}")
        elif isinstance(node, ast.ClassDef):
            lines.append(f"class {node.name}")
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    lines.append(f"  def {child.name}{_ast_signature(child)}")
        if len(lines) >= 24:
            break
    return "\n".join(lines)


def _ast_signature(node: Any) -> str:
    try:
        args = ast.unparse(node.args)
    except Exception:
        args = ""
    return f"({args})"


def _apply_v5_voting(
    *,
    result: TestGenEvalTaskResult,
    row: dict[str, Any],
    task: TestGenEvalTask,
    output_dir: Path,
    config: TestGenEvalLiteGenerateConfig,
) -> TestGenEvalTaskResult:
    """Run the V5 cross-candidate voting layer on the generated artifacts.

    Pipeline:
      1. ``anti_hack_ledger`` — drop tests with no execution-grounded
         oracle and bookkeep hack_score for surviving tests.
      2. ``patch_surrogate`` — fan out one patch-generation request per
         agent; the diffs become oracle proxies.
      3. ``dual_version_verifier`` — build the (test × patch) F→P
         matrix in the project's docker container.
      4. ``llm_critic`` (optional) — re-label F→P cells as right_reason
         vs. wrong_reason to penalize brittle matches.
      5. ``cross_candidate_voter`` — pick the test with the highest
         critic-adjusted oracle score, with optional mutation-killing
         tiebreak.

    Replaces ``result.diagnostics["generated_artifacts"][primary]``
    with the winner before downstream emission.
    """

    artifacts_raw = list(result.diagnostics.get("generated_artifacts") or [])
    candidates = _v5_candidate_records_from_result(result)
    if len(candidates) < 2:
        # Nothing to vote on; preserve result as-is.
        result.diagnostics["v5_voting"] = {
            "status": "skipped_insufficient_candidates",
            "candidate_count": len(candidates),
        }
        return result

    from apex.evaluation.anti_hack_ledger import (
        build_ledger,
        downweight_oracle_score,
    )
    from apex.evaluation.cross_candidate_voter import (
        make_local_mutation_scorer,
        select_winner,
    )
    from apex.evaluation.docker_acceptance_adapter import (
        make_docker_testgenevallite_adapter,
    )
    from apex.evaluation.dual_version_verifier import verify_tests_against_patches
    from apex.evaluation.llm_critic import (
        adjusted_oracle_scores,
        critique_test_rows,
        make_default_critic_caller,
    )
    from apex.evaluation.patch_surrogate import generate_candidate_patches

    log_dir = output_dir / "v5_voting"
    log_dir.mkdir(parents=True, exist_ok=True)
    workdir = output_dir / "v5_workdirs" / str(row.get("id") or task.instance_id)
    workdir.mkdir(parents=True, exist_ok=True)
    adapter = make_docker_testgenevallite_adapter(
        task_instance=row,
        model_name=config.model_name,
        namespace=config.docker_namespace,
        log_dir=log_dir / "docker_logs",
        official_repo=Path(config.docker_official_repo),
        timeout_seconds=config.docker_gate_timeout_seconds,
    )

    diag_v5: dict[str, Any] = {
        "status": "ok",
        "candidate_count": len(candidates),
        "candidate_source": (
            "candidate_artifact_bundle"
            if result.diagnostics.get("candidate_artifact_bundle")
            else "generated_artifacts"
        ),
    }
    logger.warning(
        "V5: starting voting for %s with %d candidates", task.instance_id, len(candidates)
    )

    # Step 1: anti-hack ledger.
    surviving: list[dict[str, Any]] = []
    ledger_reports: list[dict[str, Any]] = []
    if config.v5_use_anti_hack:
        captured_oracle_values = _v5_captured_oracle_values(result)
        focal_signature_summary = _v5_focal_signature_summary(task)
        for candidate in candidates:
            report = build_ledger(
                test_id=candidate["test_id"],
                test_source=candidate["artifact_content"],
                captured_oracle_values=captured_oracle_values,
                existing_test_source=task.existing_test_source or "",
                focal_signature_summary=focal_signature_summary,
            )
            ledger_reports.append(report.to_dict())
            candidate["_hack_score"] = report.hack_score
            if not report.rejected:
                surviving.append(candidate)
        diag_v5["anti_hack_ledger"] = {
            "input_count": len(candidates),
            "surviving_count": len(surviving),
            "reports": ledger_reports,
            "captured_oracle_value_count": len(captured_oracle_values),
            "signature_summary_present": bool(focal_signature_summary),
        }
        if not surviving:
            # Every candidate flagged as hack: keep the candidates but
            # preserve the hack scores so selection can downweight instead
            # of losing the signal entirely.
            surviving = candidates
            diag_v5["anti_hack_ledger"]["fallback_mode"] = "downweight_only"
    else:
        surviving = candidates
        for c in surviving:
            c.setdefault("_hack_score", 0.0)

    logger.warning("V5: anti-hack ledger done; %d/%d surviving", len(surviving), len(candidates))

    # Step 1.5 (P1 step 7): cross-candidate test-level dedup. Removes
    # AST-shape duplicates across the bundle so the voter sees the bundle's
    # honest diversity, not artifacts of training-distribution overlap.
    try:
        from apex.evaluation.test_dedup import dedup_candidate_bundle

        dedup_outcome = dedup_candidate_bundle(surviving)
        if dedup_outcome.changed:
            surviving = dedup_outcome.candidates
        diag_v5["test_dedup"] = dedup_outcome.to_dict()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("V5 test_dedup skipped: %s", exc)
        diag_v5["test_dedup"] = {"status": "skipped_exception", "error": str(exc)}

    # Step 2: patch surrogate fan-out (optional).
    patch_candidates: list[dict[str, Any]] = []
    if config.v5_use_patch_surrogate:
        logger.warning(
            "V5: starting patch surrogate (parallelism=%d, agents=%s)",
            max(1, int(config.v5_patch_parallelism)),
            list(config.agent_models or []),
        )
        patch_result = generate_candidate_patches(
            task=task,
            agent_models=list(config.agent_models or []),
            output_dir=log_dir / "patches",
            generation_timeout_seconds=config.generation_timeout_seconds,
            request_parallelism=max(1, int(config.v5_patch_parallelism)),
            bug_description=getattr(task, "problem_statement", "") or "",
        )
        diag_v5["patch_surrogate"] = patch_result.to_dict()
        patch_candidates = [
            {"patch_id": c.agent, "origin_agent": c.agent, "diff": c.diff}
            for c in patch_result.usable
        ]
        logger.warning(
            "V5: patch surrogate done; %d/%d usable patches",
            len(patch_candidates),
            len(patch_result.candidates),
        )
    else:
        diag_v5["patch_surrogate"] = {"status": "disabled_via_config"}
        logger.warning("V5: patch surrogate skipped (v5_use_patch_surrogate=False)")

    # Step 3: dual-version F→P matrix.
    logger.warning("V5: starting dual-version verifier")
    dvv_report = verify_tests_against_patches(
        test_candidates=surviving,
        patch_candidates=patch_candidates,
        benchmark_adapter=adapter,
        workdir=workdir,
        focal_path=task.focal_method_path,
    )
    diag_v5["dual_version_verifier"] = dvv_report.to_dict()

    # Step 4 (optional): LLM critic for right-reason labelling.
    critic_verdicts = []
    if config.v5_use_llm_critic and patch_candidates:
        try:
            caller = make_default_critic_caller(
                judge_agent=config.v5_critic_agent,
                timeout_seconds=int(config.docker_gate_timeout_seconds),
                working_dir=Path(
                    task.repo_path
                    or (getattr(task, "metadata", {}) or {}).get("source_truth_workdir")
                    or workdir
                ),
            )
            critic_verdicts = critique_test_rows(
                test_candidates=surviving,
                dual_version_rows=dvv_report.rows,
                llm_caller=caller,
                focal_path=task.focal_method_path,
                focal_source_excerpt=(task.focal_method_source or "")[:1800],
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("V5 critic skipped: %s", exc)
            critic_verdicts = []
    diag_v5["llm_critic"] = [v.to_dict() for v in critic_verdicts]

    # Compute adjusted oracle scores from critic + apply hack-score downweight.
    adj_scores = adjusted_oracle_scores(
        dual_version_rows=dvv_report.rows,
        critic_verdicts=critic_verdicts,
    )
    hack_by_id = {c["test_id"]: float(c.get("_hack_score") or 0.0) for c in surviving}

    # Build a synthetic row set carrying the adjusted score so the voter
    # picks based on (critic ∘ anti_hack ∘ raw_oracle).
    @dataclass
    class _VoterRow:
        test_id: str
        oracle_score: float
        original_score: float

    voter_rows = [
        _VoterRow(
            test_id=row.test_id,
            oracle_score=float(
                downweight_oracle_score(
                    raw_oracle_score=adj_scores.get(row.test_id, row.oracle_score),
                    hack_score=hack_by_id.get(row.test_id, 0.0),
                )
            ),
            original_score=float(row.oracle_score),
        )
        for row in dvv_report.rows
    ]

    # Step 5: cross-candidate voter (with optional mutation tiebreak).
    mutation_scorer = None
    if config.v5_use_mutation_tiebreak:
        try:
            # Audit M8: pass the focal-module path so the scorer can
            # materialize the focal file in the (typically empty)
            # v5_workdirs/<id>/ tmp clone, instead of failing silently.
            mutation_scorer = make_local_mutation_scorer(
                focal_module_source=task.focal_method_source or "",
                workdir=workdir,
                benchmark_adapter=adapter,
                artifact_path=str(surviving[0].get("artifact_path") or "tests/test_apex.py"),
                n_mutants=int(config.v5_mutation_n),
                focal_module_relpath=str(task.focal_method_path or "") or None,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("V5 mutation scorer skipped: %s", exc)
            mutation_scorer = None

    # Audit H6: capture the pre-V5 baseline pick BEFORE the voter runs.
    # The baseline is the first non-empty artifact in the upstream
    # ``generated_artifacts`` list (mirrors what `_select_primary_artifact_text`
    # would have returned without V5).
    baseline_text = ""
    for a in artifacts_raw:
        if isinstance(a, dict):
            text = str(a.get("content") or "").strip()
            if text:
                baseline_text = text
                break

    winner, selection_diag = select_winner(
        test_candidates=surviving,
        dual_version_rows=voter_rows,
        mutation_killing_scorer=mutation_scorer,
    )
    diag_v5["selection"] = selection_diag.to_dict()

    # Plug winner back into result.diagnostics["generated_artifacts"].
    # Audit H5: when the voter abstains (winner=None) we DO NOT clobber
    # the baseline. The selection_path on the diagnostic captures why.
    selection_diff: dict[str, Any] = {
        "baseline_present": bool(baseline_text),
        "v5_winner_id": getattr(selection_diag, "winner_id", None),
        "v5_abstained": getattr(selection_diag, "abstained", False),
        "changed_pick": False,
    }
    if winner is not None:
        winner_id = str(winner.get("test_id") or "")
        winner_text = str(winner.get("artifact_content") or "")
        winner_path = str(winner.get("artifact_path") or "tests/test_apex.py")
        selection_diff["changed_pick"] = bool(winner_text) and winner_text.strip() != baseline_text
        new_artifacts: list[dict[str, Any]] = []
        replaced = False
        for a in artifacts_raw:
            if not isinstance(a, dict):
                new_artifacts.append(a)
                continue
            if not replaced and (a.get("content") or "").strip():
                updated = dict(a)
                updated["content"] = winner_text
                updated["path"] = winner_path
                updated["selected_for_submission"] = True
                new_artifacts.append(updated)
                replaced = True
            else:
                updated = dict(a)
                updated["selected_for_submission"] = False
                new_artifacts.append(updated)
        if not replaced:
            new_artifacts.append(
                {
                    "path": winner_path,
                    "content": winner_text,
                    "selected_for_submission": True,
                    "source_test_id": winner_id,
                }
            )
        result.diagnostics["generated_artifacts"] = new_artifacts
    else:
        _ensure_selection_flag(result)

    diag_v5["selection_diff"] = selection_diff
    result.diagnostics["v5_voting"] = diag_v5
    return result


def _apply_docker_gate(
    *,
    result: TestGenEvalTaskResult,
    row: dict[str, Any],
    output_dir: Path,
    config: TestGenEvalLiteGenerateConfig,
) -> TestGenEvalTaskResult:
    """Run the W1 final-acceptance gate using a docker-based adapter.

    The local validator can't run Django/sympy/Flask tests; the docker
    adapter hands the artifact to the same image the official harness will
    use. Drops failing tests until the artifact passes or stops shrinking.
    """

    from apex.evaluation.docker_acceptance_adapter import (
        make_docker_testgenevallite_adapter,
    )
    from apex.evaluation.final_acceptance_gate import (
        GeneratedArtifact as _Artifact,
    )
    from apex.evaluation.final_acceptance_gate import (
        ship_acceptance,
    )

    artifacts = list(result.diagnostics.get("generated_artifacts") or [])
    if not artifacts or not config.docker_official_repo:
        return result
    primary = next(
        (a for a in artifacts if isinstance(a, dict) and (a.get("content") or "").strip()),
        None,
    )
    if primary is None:
        return result
    log_dir = output_dir / "docker_gate_logs"
    workdir = output_dir / "docker_gate_workdirs" / str(row.get("id") or result.instance_id)
    workdir.mkdir(parents=True, exist_ok=True)
    adapter = make_docker_testgenevallite_adapter(
        task_instance=row,
        model_name=config.model_name,
        namespace=config.docker_namespace,
        log_dir=log_dir,
        official_repo=Path(config.docker_official_repo),
        timeout_seconds=config.docker_gate_timeout_seconds,
    )
    gate_result = ship_acceptance(
        _Artifact(
            path=str(primary.get("path") or "tests/test_apex_generated.py"),
            content=str(primary.get("content") or ""),
        ),
        benchmark_adapter=adapter,
        workdir=workdir,
        keep_minimum=config.docker_gate_keep_minimum,
        max_drop_iterations=config.docker_gate_iterations,
    )
    diag = result.diagnostics.setdefault("apex_validation", {})
    diag["docker_final_acceptance_gate"] = {
        "status": gate_result.status,
        "iterations": gate_result.iterations,
        "dropped_tests": list(gate_result.dropped_tests),
        "note": gate_result.note,
    }
    if gate_result.shipped and gate_result.artifact.content:
        diag.update(reconcile_authoritative_validation(diag))
        # Replace the primary artifact with the gate's accepted version.
        new_artifacts: list[dict[str, Any]] = []
        replaced = False
        for art in artifacts:
            if not isinstance(art, dict):
                new_artifacts.append(art)
                continue
            if not replaced and (art.get("content") or "").strip():
                updated = dict(art)
                updated["content"] = gate_result.artifact.content
                new_artifacts.append(updated)
                replaced = True
            else:
                new_artifacts.append(art)
        result.diagnostics["generated_artifacts"] = new_artifacts
    return result


def _sweep_orphan_subprocesses() -> None:
    """Reap any codex CLI subprocesses or kdjain docker containers spawned
    by this run. Belt-and-suspenders cleanup for the per-call hooks in
    ``apex.core.cli_backend`` which can race under heavy parallel load
    and leave stragglers (we observed 9355 codex orphans after a
    sequence of killed runs)."""

    import subprocess as _subprocess

    own_pid = os.getpid()
    # Find every codex process that is a descendant of our process tree.
    try:
        own_tree: set[int] = {own_pid}
        ps = _subprocess.run(
            ["ps", "-axo", "pid=,ppid="],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        children_by_parent: dict[int, list[int]] = {}
        for line in (ps.stdout or "").splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                pid, ppid = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            children_by_parent.setdefault(ppid, []).append(pid)
        stack = [own_pid]
        while stack:
            p = stack.pop()
            for c in children_by_parent.get(p, []):
                if c not in own_tree:
                    own_tree.add(c)
                    stack.append(c)
        # Match codex children only — never kill our own python.
        codex_pids: list[int] = []
        ps2 = _subprocess.run(
            ["ps", "-axo", "pid=,command="],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        for line in (ps2.stdout or "").splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) != 2:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            if pid in own_tree and pid != own_pid and "codex" in parts[1]:
                codex_pids.append(pid)
        for pid in codex_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
        time.sleep(1.0)
        for pid in codex_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue
    except Exception:  # pragma: no cover - best-effort
        pass

    # Kill any kdjain/swe-bench docker containers that may have outlived
    # the gate's own kill path.
    try:
        ps = _subprocess.run(
            ["docker", "ps", "--format", "{{.ID}} {{.Image}}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        ids: list[str] = []
        for line in (ps.stdout or "").splitlines():
            cid, _, image = line.strip().partition(" ")
            if "kdjain/swe-bench" in image:
                ids.append(cid)
        for cid in ids:
            _subprocess.run(["docker", "kill", cid], capture_output=True, timeout=15, check=False)
    except Exception:  # pragma: no cover - best-effort
        pass


def run_generate(config: TestGenEvalLiteGenerateConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Default the upstream harness's file-mount mode for docker eval calls.
    # Without ``SWEBENCH_DOCKER_FORK_DIR`` set, ``swebench_docker.run_docker``
    # passes the prediction payload as a base64 ``-e INSTANCE=...`` env var,
    # which silently fails with ``argument list too long`` on macOS for
    # large predictions (we lost ~46/160 predictions to this in the V5
    # full eval). Setting it makes the harness mount the prediction as a
    # file at ``/home/swe-bench/task_instance.json`` instead. Honor an
    # operator-set value if already exported.
    #
    # Audit H9: capture the original value so we can restore at the end
    # of run_generate. Two concurrent run_generate calls (uncommon but
    # possible) used to clobber each other's setting silently.
    docker_repo = str(config.docker_official_repo or "").strip()
    _original_swebench_fork_dir = os.environ.get("SWEBENCH_DOCKER_FORK_DIR")
    _set_swebench_fork_dir = False
    if docker_repo and not (os.environ.get("SWEBENCH_DOCKER_FORK_DIR") or "").strip():
        os.environ["SWEBENCH_DOCKER_FORK_DIR"] = str(Path(docker_repo).expanduser().resolve())
        _set_swebench_fork_dir = True
        logger.info(
            "set SWEBENCH_DOCKER_FORK_DIR=%s for file-mount harness mode",
            os.environ["SWEBENCH_DOCKER_FORK_DIR"],
        )
    # P2.2 fix: env restore must happen on every exit path, including
    # exceptions during dataset load / worker pool / atomic_write_text.
    # The previous code only restored at the bottom of the function,
    # leaving the env mutated under any failure.
    try:
        return _run_generate_body(
            config=config,
            output_dir=output_dir,
        )
    finally:
        if _set_swebench_fork_dir:
            if _original_swebench_fork_dir is None:
                os.environ.pop("SWEBENCH_DOCKER_FORK_DIR", None)
            else:
                os.environ["SWEBENCH_DOCKER_FORK_DIR"] = _original_swebench_fork_dir


def _per_task_parent_deadline_seconds(config: TestGenEvalLiteGenerateConfig) -> float:
    """Best-effort watchdog threshold for one active worker.

    This is intentionally above the inner CLI / pytest / Docker timeouts. The
    parent loop cannot kill a running thread safely; it can only flag suspicious
    drain time. Keep this as a late warning, not a second competing timeout.
    """

    generation_timeout = max(1.0, float(config.generation_timeout_seconds or 0.0))
    repair_attempts = max(0, int(config.max_repair_attempts or 0))
    generation_budget = generation_timeout * max(1, repair_attempts + 1)
    pytest_budget = max(0.0, float(config.pytest_timeout_seconds or 0.0))
    docker_budget = (
        max(0.0, float(config.docker_gate_timeout_seconds or 0.0))
        if config.docker_gate_enabled
        or (
            bool(getattr(config, "target_environment_enabled", False))
            and bool(str(config.docker_official_repo or "").strip())
        )
        else 0.0
    )
    slack = min(900.0, max(120.0, generation_budget * 0.25))
    return max(1.0, generation_budget + pytest_budget + docker_budget + slack)


def _run_generate_body(
    *, config: TestGenEvalLiteGenerateConfig, output_dir: Path
) -> dict[str, Any]:
    """Body of ``run_generate`` extracted so the env-restore wrapper
    above can guarantee cleanup via try/finally regardless of how this
    function exits. P2.2 fix.
    """

    if config.from_json:
        tasks = load_tasks_from_json(config.from_json)
        rows = [({"instance_id": t.instance_id, "id": t.instance_id}, t) for t in tasks]
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
        existing_ids = {
            p.stem for p in records_dir.glob("*.json") if _is_complete_generation_record(p, p.stem)
        }
        before = len(rows)
        rows = [
            (row, task)
            for row, task in rows
            if str(row.get("id") or task.instance_id) not in existing_ids
        ]
        if before != len(rows):
            logger.warning(
                "skip_existing: filtered %d → %d rows (skipped %d already in %s)",
                before,
                len(rows),
                before - len(rows),
                records_dir,
            )

    manifest = {
        "started_at": time.time(),
        "model_name": config.model_name,
        "dataset": config.dataset_name,
        "split": config.split,
        "total_tasks": len(rows),
        "parallelism": config.parallelism,
        "config": {
            "generation_timeout_seconds": config.generation_timeout_seconds,
            "pytest_timeout_seconds": config.pytest_timeout_seconds,
            "max_repair_attempts": config.max_repair_attempts,
            "candidate_count": config.candidate_count,
        },
        "prediction_path": str(preds_path),
    }
    atomic_write_json(output_dir / "run_manifest.json", manifest)

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    # Reconcile the operator's choice (or the host-aware default) against
    # the actual task count: spawning more workers than tasks wastes
    # nothing, but is misleading in the manifest. Honour an explicit cap.
    parallelism = resolve_task_parallelism(int(config.parallelism), task_count=len(rows))
    per_task_deadline_seconds = _per_task_parent_deadline_seconds(config)

    def _record_one_failure(row_dict: dict[str, Any], exc: BaseException) -> None:
        """Record a per-task failure without aborting the run.

        Audit C4: also write a stub record to ``records/`` so the failed
        task isn't a silent dead-end on disk. The previous behavior left
        no trace under records/, which made post-mortem impossible — the
        sympy-24909 case in v5_full_20260509 had ``apex_generation: {}``
        and 348s duration but no failure_reason, no docker workdir, no
        explanation. We now emit a record carrying the exception type
        and message so the failure is traceable.
        """

        rid = str(row_dict.get("id") or row_dict.get("instance_id") or "unknown")
        logger.exception("Task %s raised; isolating failure: %s", rid, exc)
        failures.append({"id": rid, "error": f"{type(exc).__name__}: {exc}"})
        try:
            stub_record = {
                "id": rid,
                "instance_id": str(row_dict.get("instance_id") or rid),
                "model_name_or_path": config.model_name,
                "preds": {"full": [""]},
                "apex_generation": {},
                "apex_generation_duration_seconds": 0.0,
                "apex_validation": {
                    "schema_version": 1,
                    "prediction_quality": "failed",
                    "failure_class": "orchestrator_error",
                    "failure_reason": f"{type(exc).__name__}: {exc}"[:1000],
                },
                "apex_diagnostics": {
                    "orchestrator_failure": {
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc)[:2000],
                    }
                },
            }
            atomic_write_json(records_dir / f"{rid}.json", stub_record)
        except Exception:  # pragma: no cover - last-resort failure handler
            logger.exception("Could not write stub failure record for %s", rid)

    try:
        if parallelism == 1:
            for row, task in rows:
                try:
                    record = _process_one(row=row, task=task, output_dir=output_dir, config=config)
                    atomic_write_json(records_dir / f"{record['id']}.json", record)
                    completed.append(record)
                except Exception as exc:
                    _record_one_failure(row, exc)
        else:
            pool = ThreadPoolExecutor(max_workers=parallelism)
            try:
                row_iter = iter(rows)
                futures: dict[Any, dict[str, Any]] = {}
                pending: set[Any] = set()

                def _submit_next() -> bool:
                    try:
                        row, task = next(row_iter)
                    except StopIteration:
                        return False
                    future = pool.submit(
                        _process_one,
                        row=row,
                        task=task,
                        output_dir=output_dir,
                        config=config,
                    )
                    futures[future] = {
                        "row": row,
                        "started_at": time.time(),
                        "deadline_logged": False,
                    }
                    pending.add(future)
                    return True

                for _ in range(min(parallelism, len(rows))):
                    _submit_next()
                while pending:
                    done, pending = wait(
                        pending,
                        timeout=1.0,
                        return_when=FIRST_COMPLETED,
                    )
                    now = time.time()
                    for future in list(pending):
                        meta = futures[future]
                        if now - float(meta["started_at"]) < per_task_deadline_seconds:
                            continue
                        if meta.get("deadline_logged"):
                            continue
                        meta["deadline_logged"] = True
                        row_dict = dict(meta["row"])
                        rid = str(row_dict.get("id") or row_dict.get("instance_id") or "unknown")
                        logger.error(
                            "Task %s exceeded parent deadline of %.1fs; "
                            "waiting for worker-level timeout cleanup",
                            rid,
                            per_task_deadline_seconds,
                        )
                    for future in done:
                        row_dict = dict(futures[future]["row"])
                        try:
                            record = future.result()
                            atomic_write_json(records_dir / f"{record['id']}.json", record)
                            completed.append(record)
                        except Exception as exc:
                            _record_one_failure(row_dict, exc)
                        finally:
                            futures.pop(future, None)
                            _submit_next()
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
    finally:
        # Sweep any leaked codex / docker processes spawned by this run so
        # the next run starts on a clean machine. With parallel CLI calls
        # under load the per-call cleanup hooks can race; this sweep is the
        # definitive backstop.
        _sweep_orphan_subprocesses()
    completed.sort(key=lambda r: str(r.get("id") or ""))
    # Audit C5: cycle-scrub before line-writing. A self-referential dict
    # in any record killed json.dumps in the v5_full_20260509 run and we
    # lost the run summary. ``_safe_jsonable`` replaces cycles with the
    # ``"<circular>"`` marker so a single bad record doesn't abort the
    # writer for the other 159.
    from apex.evaluation.checkpointing import _safe_jsonable as _scrub_cycles

    atomic_write_text(
        preds_path,
        "".join(
            json.dumps(_scrub_cycles(rec), sort_keys=True, default=repr) + "\n" for rec in completed
        ),
    )

    finished = time.time()
    summary = {
        "status": "ok",
        "prediction_path": str(preds_path),
        "records_dir": str(records_dir),
        "task_count": len(rows),
        "predictions_written": len(completed),
        "started_at": manifest["started_at"],
        "finished_at": finished,
        "elapsed_seconds": finished - manifest["started_at"],
    }
    atomic_write_json(output_dir / "generation_summary.json", summary)
    return summary


def _parse_args(argv: list[str]) -> TestGenEvalLiteGenerateConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Generate TestGenEvalLite predictions through the V4 pipeline. "
            "Writes a JSONL preds file consumable by the official harness."
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="apex")
    parser.add_argument("--dataset-name", default="kjain14/testgenevallite")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--from-json",
        default="",
        help="If set, load tasks from a local JSON file instead of HuggingFace.",
    )
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0, help="Cap task count (0 = no limit).")
    parser.add_argument(
        "--parallelism",
        type=int,
        default=0,
        help=(
            "Worker count for parallel generation. Defaults to "
            "min(task_count, host_cpu_or_docker_cpu, 4) when 0/unset."
        ),
    )
    parser.add_argument("--generation-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--pytest-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-repair-attempts", type=int, default=3)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--measure-mutation", action="store_true")
    parser.add_argument("--measure-coverage", action="store_true")
    parser.add_argument("--measure-assertion-effect", action="store_true")
    parser.add_argument(
        "--no-target-environment",
        dest="target_environment_enabled",
        action="store_false",
        help=(
            "Disable target-environment execution during generation. "
            "Default is ON when --docker-official-repo is supplied."
        ),
    )
    parser.add_argument(
        "--require-target-environment",
        action="store_true",
        help=(
            "Fail records instead of falling back to host dynamic execution "
            "when no target environment can be bound."
        ),
    )
    parser.add_argument(
        "--docker-gate-enabled",
        action="store_true",
        help="Run the W1 final-acceptance gate via the project's docker image after generation.",
    )
    parser.add_argument("--docker-official-repo", default="")
    parser.add_argument("--docker-namespace", default="kdjain")
    parser.add_argument("--docker-gate-iterations", type=int, default=3)
    parser.add_argument("--docker-gate-timeout-seconds", type=int, default=600)
    parser.add_argument("--docker-gate-keep-minimum", type=int, default=1)
    parser.add_argument(
        "--no-v5-voting",
        dest="v5_voting_enabled",
        action="store_false",
        help="Disable V5 cross-candidate voting (defaults to ON).",
    )
    parser.add_argument(
        "--no-v5-dual-state-oracle",
        dest="v5_dual_state_oracle_enabled",
        action="store_false",
        help=(
            "Disable V5 patch-as-oracle voting. Default is ON; the "
            "capability check inside _v5_dual_state_capability skips "
            "voting per-row when the row lacks the required buggy/fixed "
            "metadata, so leaving it on is safe across benchmarks."
        ),
    )
    parser.add_argument("--no-v5-anti-hack", dest="v5_use_anti_hack", action="store_false")
    parser.add_argument("--no-v5-llm-critic", dest="v5_use_llm_critic", action="store_false")
    parser.add_argument(
        "--v5-critic-agent",
        default="claude",
        choices=("codex", "claude", "gemini", "opencode", "metacode"),
    )
    parser.add_argument(
        "--no-v5-mutation-tiebreak",
        dest="v5_use_mutation_tiebreak",
        action="store_false",
    )
    parser.add_argument("--v5-mutation-n", type=int, default=5)
    parser.add_argument(
        "--v5-patch-parallelism",
        type=int,
        default=1,
        help=(
            "Cap on the patch_surrogate parallel CLI fan-out. Default 1 "
            "(serialized) because parallel fan-out reliably crashes the "
            "orchestrator on macOS. Raise once the signal interaction is "
            "diagnosed."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip rows whose record file already exists in the output "
            "records dir. Use this to resume a partial run."
        ),
    )
    parser.add_argument(
        "--no-v5-patch-surrogate",
        dest="v5_use_patch_surrogate",
        action="store_false",
        help=(
            "Skip the patch_surrogate fan-out entirely. V5 voting falls "
            "through to the dual-version verifier's fallback path "
            "(fails-on-buggy + mutation tiebreak), which is degraded but "
            "still better than no V5. Workaround for the macOS-specific "
            "subprocess crash during sequential CLI patch agent spawns."
        ),
    )
    parser.set_defaults(v5_use_patch_surrogate=True)
    parser.set_defaults(
        v5_voting_enabled=True,
        v5_dual_state_oracle_enabled=True,
        v5_use_anti_hack=True,
        v5_use_llm_critic=True,
        v5_use_mutation_tiebreak=True,
    )
    parser.add_argument(
        "--agent",
        action="append",
        default=None,  # None means "use the dataclass default"
        choices=("codex", "claude", "gemini", "opencode", "metacode"),
        help=(
            "Add an agent to the multi-agent ensemble (repeatable). "
            "Each candidate uses a different agent — TEX-T-style "
            "diversity from cross-model differences. "
            "DEFAULT is codex+claude+gemini. Pass an OpenCode-family agent "
            "explicitly for ablations."
        ),
    )
    args = parser.parse_args(argv)
    parallelism_arg = int(args.parallelism or 0)
    # When the operator passed --parallelism N (N>=1) we forward it; when
    # they omitted it we forward 0 so that the dataclass'
    # default_factory=default_task_parallelism kicks in.
    extra_kwargs: dict[str, Any] = {}
    if parallelism_arg >= 1:
        extra_kwargs["parallelism"] = parallelism_arg
    return TestGenEvalLiteGenerateConfig(
        output_dir=args.output_dir,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        split=args.split,
        from_json=args.from_json,
        task_ids=list(args.task_id or []),
        limit=int(args.limit or 0),
        generation_timeout_seconds=float(args.generation_timeout_seconds),
        pytest_timeout_seconds=float(args.pytest_timeout_seconds),
        max_repair_attempts=int(args.max_repair_attempts),
        candidate_count=int(args.candidate_count),
        measure_mutation=bool(args.measure_mutation),
        measure_coverage=bool(args.measure_coverage),
        measure_assertion_effect=bool(args.measure_assertion_effect),
        target_environment_enabled=bool(args.target_environment_enabled),
        target_environment_required=bool(args.require_target_environment),
        docker_gate_enabled=bool(args.docker_gate_enabled),
        docker_official_repo=args.docker_official_repo,
        docker_namespace=args.docker_namespace,
        docker_gate_iterations=int(args.docker_gate_iterations),
        docker_gate_timeout_seconds=int(args.docker_gate_timeout_seconds),
        docker_gate_keep_minimum=int(args.docker_gate_keep_minimum),
        # When --agent was passed at all (even once), use exactly those.
        # When --agent was NOT passed (args.agent is None), let the
        # dataclass default kick in (the strong three).
        agent_models=list(args.agent)
        if args.agent is not None
        else ["codex", "claude", "gemini"],
        v5_voting_enabled=bool(args.v5_voting_enabled),
        v5_dual_state_oracle_enabled=bool(args.v5_dual_state_oracle_enabled),
        v5_use_anti_hack=bool(args.v5_use_anti_hack),
        v5_use_llm_critic=bool(args.v5_use_llm_critic),
        v5_critic_agent=str(args.v5_critic_agent),
        v5_use_mutation_tiebreak=bool(args.v5_use_mutation_tiebreak),
        v5_mutation_n=int(args.v5_mutation_n),
        v5_patch_parallelism=int(args.v5_patch_parallelism),
        v5_use_patch_surrogate=bool(args.v5_use_patch_surrogate),
        skip_existing=bool(args.skip_existing),
        **extra_kwargs,
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
