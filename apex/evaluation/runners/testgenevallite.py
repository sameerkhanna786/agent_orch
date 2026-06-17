"""Checked-in TestGenEvalLite runner wrapper.

This module is intentionally thin: Apex owns generation, validation, repair,
and telemetry; the official TestGenEval checkout owns final scoring.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from apex.core.fairness_audit import FairnessAuditAggregator, FairnessAuditMode
from apex.core.parallelism import default_task_parallelism
from apex.core.run_manifest import (
    RunManifest,
    detect_upstream_harness_versions,
)
from apex.evaluation.checkpointing import atomic_write_json, atomic_write_text
from apex.evaluation.prediction_telemetry import enrich_testgeneval_prediction_jsonl
from apex.evaluation.runners._active_manifest import (
    set_active_manifest,
)
from apex.evaluation.runners._preflight import pre_flight_for_testgen
from apex.evaluation.upstream_patches import apply_testgeneval_upstream_patches


@dataclass(frozen=True)
class TestGenEvalLiteRunConfig:
    __test__ = False

    official_repo: str
    predictions_jsonl: str
    output_dir: str
    model_name: str = "apex"
    split: str = "testgenevallite"
    # Default to host-CPU/Docker-aware parallelism (capped at 4 — matches
    # ``scripts/launch_*.sh``). Operators passing ``--task-parallelism N``
    # still win — see ``resolve_task_parallelism`` at run time.
    task_parallelism: int = field(default_factory=default_task_parallelism)
    timeout_seconds: int = 300
    mutation_timeout_seconds: int = 3600
    skip_existing: bool = True
    skip_mutation: bool = False
    dataset_name: str = "kjain14/testgenevallite"
    docker_namespace: str = "kdjain"
    task_ids: list[str] = field(default_factory=list)
    extra_args: list[str] = field(default_factory=list)
    run_preflight: bool = True
    force_preflight: bool = False
    baseline_run_dir: str = ""
    apply_upstream_patches: bool = False
    docker_platform: str = ""
    # Phase 1b fairness audit. "off" / "parallel" / "upstream_only".
    # When "parallel", the runner runs both the patched-harness scoring
    # and the unpatched-harness scoring (with only the defensive
    # baseline_covs fix applied) and emits a per-task delta into
    # ``fairness_audit.json`` / ``FAIRNESS_REPORT.md``.
    fairness_audit_mode: str = "off"
    # Optional seed value to record into the run manifest. Does not
    # affect orchestration; pure provenance.
    run_seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_testgenevallite_command(config: TestGenEvalLiteRunConfig) -> list[str]:
    """Build the command that runs the official TestGenEvalLite evaluator.

    Prefers ``run_evaluation.py`` (the canonical kjain14/testgeneval entry
    point) when it exists in the official repo. Falls back to a Makefile
    target or the ``testgeneval.evaluate`` module so callers shipping a
    different harness layout still get a working command.
    """

    repo = Path(config.official_repo).expanduser()
    output_dir = Path(config.output_dir).expanduser().resolve()
    log_dir = output_dir / "official_eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = str(Path(config.predictions_jsonl).expanduser().resolve())
    run_evaluation = repo / "run_evaluation.py"
    if run_evaluation.exists():
        python = sys.executable
        command = [
            python,
            str(run_evaluation),
            "--predictions_path",
            predictions_path,
            "--log_dir",
            str(log_dir),
            "--swe_bench_tasks",
            config.dataset_name,
            "--namespace",
            config.docker_namespace,
            "--timeout",
            str(int(config.timeout_seconds)),
            "--num_processes",
            str(int(config.task_parallelism)),
            "--mutation_timeout",
            str(int(config.mutation_timeout_seconds)),
        ]
        if config.skip_existing:
            command.append("--skip_existing")
        if config.skip_mutation:
            command.append("--skip_mutation")
        command.extend(config.extra_args)
        return command

    makefile = repo / "Makefile.testgenevallite"
    if makefile.exists():
        command = [
            "make",
            "-f",
            str(makefile),
            "evaluate",
            f"PREDICTIONS={config.predictions_jsonl}",
            f"OUTPUT_DIR={config.output_dir}",
            f"MODEL_NAME={config.model_name}",
            f"TASK_PARALLELISM={int(config.task_parallelism)}",
            f"TIMEOUT={int(config.timeout_seconds)}",
            f"MUTATION_TIMEOUT={int(config.mutation_timeout_seconds)}",
        ]
        if config.skip_existing:
            command.append("SKIP_EXISTING=1")
        if config.task_ids:
            command.append("TASK_IDS=" + ",".join(config.task_ids))
        command.extend(config.extra_args)
        return command

    return [
        sys.executable,
        "-m",
        "testgeneval.evaluate",
        "--split",
        config.split,
        "--predictions",
        config.predictions_jsonl,
        "--output-dir",
        config.output_dir,
        "--model-name",
        config.model_name,
        "--task-parallelism",
        str(int(config.task_parallelism)),
        "--timeout",
        str(int(config.timeout_seconds)),
        "--mutation-timeout",
        str(int(config.mutation_timeout_seconds)),
        *(["--skip-existing"] if config.skip_existing else []),
        *sum((["--task-id", task_id] for task_id in config.task_ids), []),
        *config.extra_args,
    ]


def run_testgenevallite(
    config: TestGenEvalLiteRunConfig,
    *,
    monitor_interval_seconds: float = 30.0,
) -> dict[str, Any]:
    output = Path(config.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    if config.run_preflight:
        try:
            preflight = pre_flight_for_testgen(
                output,
                docker_platform=config.docker_platform or None,
            )
        except TypeError as exc:
            if "docker_platform" not in str(exc):
                raise
            preflight = pre_flight_for_testgen(output)
        atomic_write_json(output / "preflight.json", preflight.to_dict())
        if not preflight.passed and not config.force_preflight:
            return {
                "status": "preflight_failed",
                "config": config.to_dict(),
                "preflight": preflight.to_dict(),
            }

    fairness_mode = _coerce_fairness_audit_mode(config.fairness_audit_mode)
    if config.apply_upstream_patches:
        # Per Phase 1.3 the patch directory is a divergence vs. the
        # published kjain14/testgeneval baseline (even after the
        # memory-swappiness hunk was removed). Keep this opt-in so
        # publishable comparisons do not accidentally score on a modified
        # harness.
        warnings.warn(
            "TestGenEval upstream patches were explicitly enabled; "
            "this is a divergence vs. the published kjain14/testgeneval "
            "baseline. See "
            "apex/evaluation/upstream_patches/testgeneval/UPSTREAM_PR_PLAN.md "
            "for the upstream-merge plan.",
            DeprecationWarning,
            stacklevel=2,
        )
        upstream_patch_status = apply_testgeneval_upstream_patches(config.official_repo)
    elif fairness_mode in {FairnessAuditMode.PARALLEL, FairnessAuditMode.UPSTREAM_ONLY}:
        # When the audit is enabled but the user disabled the full
        # patch set, still apply the defensive baseline_covs fix so
        # ``generate_report.py`` does not crash on lite-subset rows.
        upstream_patch_status = apply_testgeneval_upstream_patches(
            config.official_repo,
            baseline_covs_only=True,
        )
    else:
        upstream_patch_status = {"status": "disabled", "patch_count": 0}

    manifest = _capture_manifest_resilient(
        config=config,
        fairness_mode=fairness_mode,
        upstream_patch_status=upstream_patch_status,
    )
    # Register the manifest as the active one so deep-stack docker
    # invocations (e.g. apex.evaluation.docker_subprocess_runner._docker_image_for)
    # can record image digests without threading the manifest through
    # their call sites. Cleared in the ``finally`` at the end of the
    # function so concurrent runs don't leak each other's state.
    set_active_manifest(manifest)
    try:
        return _run_testgenevallite_inner(
            config=config,
            output=output,
            monitor_interval_seconds=monitor_interval_seconds,
            fairness_mode=fairness_mode,
            upstream_patch_status=upstream_patch_status,
            manifest=manifest,
        )
    finally:
        set_active_manifest(None)


def _run_testgenevallite_inner(
    *,
    config: TestGenEvalLiteRunConfig,
    output: Path,
    monitor_interval_seconds: float,
    fairness_mode: FairnessAuditMode,
    upstream_patch_status: dict[str, Any],
    manifest: RunManifest,
) -> dict[str, Any]:
    prediction_enrichment = enrich_testgeneval_prediction_jsonl(config.predictions_jsonl)
    resume_state = _write_resume_state(output, skip_existing=config.skip_existing)
    command = build_testgenevallite_command(config)
    resume_launcher = _write_resume_launcher(output, config, command)
    status_path = output / "runner_status.json"
    atomic_write_json(
        status_path,
        {
            "status": "running",
            "command": command,
            "config": config.to_dict(),
            "started_at": time.time(),
            "upstream_patches": upstream_patch_status,
            "prediction_enrichment": prediction_enrichment,
            "resume_state": resume_state,
            "resume_launcher": resume_launcher,
            "resume_env": str(output / "resume_env.sh"),
            "resume_environment": _resume_environment_snapshot(),
            "live_subprocess_count": _live_test_subprocess_count(),
        },
    )
    # Force the docker invocation into the "fork" branch so our patched
    # swebench_docker code (W11 mutation_timeout, etc.) is mounted into each
    # container instead of the image's stale built-in copy.
    child_env = dict(os.environ)
    child_env.setdefault(
        "SWEBENCH_DOCKER_FORK_DIR",
        str(Path(config.official_repo).expanduser().resolve()),
    )
    process = subprocess.Popen(
        command,
        cwd=str(Path(config.official_repo).expanduser()),
        start_new_session=True,
        env=child_env,
    )
    try:
        while process.poll() is None:
            atomic_write_json(
                status_path,
                {
                    "status": "running",
                    "pid": process.pid,
                    "command": command,
                    "config": config.to_dict(),
                    "upstream_patches": upstream_patch_status,
                    "resume_state": _write_resume_state(
                        output,
                        skip_existing=config.skip_existing,
                    ),
                    "resume_launcher": resume_launcher,
                    "resume_env": str(output / "resume_env.sh"),
                    "resume_environment": _resume_environment_snapshot(),
                    "updated_at": time.time(),
                    "live_subprocess_count": _live_test_subprocess_count(),
                },
            )
            time.sleep(max(1.0, float(monitor_interval_seconds)))
    except KeyboardInterrupt:
        os.killpg(process.pid, signal.SIGTERM)
        raise
    returncode = process.wait()
    report_status = _run_official_report_generator(
        official_repo=Path(config.official_repo).expanduser(),
        predictions_jsonl=Path(config.predictions_jsonl).expanduser().resolve(),
        log_dir=Path(config.output_dir).expanduser().resolve() / "official_eval_logs",
        output_dir=Path(config.output_dir).expanduser().resolve() / "official_reports",
        dataset_name=config.dataset_name,
        env=child_env,
    )
    run_report = _write_official_run_report(output, model_name=config.model_name)
    if isinstance(run_report, dict):
        run_report["report_generator"] = report_status

    # Fairness audit: when PARALLEL we score every task with both the
    # APEX-private scorer (parallel log-marker aggregator + patched
    # harness) and the upstream-canonical scorer (unpatched harness +
    # baseline_covs defensive fix only). The audit emits per-task
    # deltas into ``fairness_audit.json`` / ``FAIRNESS_REPORT.md``.
    fairness_audit_payload: dict[str, Any] | None = None
    if fairness_mode == FairnessAuditMode.PARALLEL:
        fairness_audit_payload = _run_fairness_audit_parallel(
            output_dir=output,
            config=config,
        )
    elif fairness_mode == FairnessAuditMode.UPSTREAM_ONLY:
        fairness_audit_payload = {
            "mode": fairness_mode.value,
            "note": (
                "upstream_only mode: only the upstream-canonical scorer "
                "is treated as authoritative; APEX-private aggregator "
                "output is recorded as diagnostic-only."
            ),
        }

    # Persist the run manifest so reviewers can reproduce this run.
    manifest_path = manifest.write_to(output)

    payload = {
        "status": "succeeded" if returncode == 0 else "failed",
        "returncode": returncode,
        "command": command,
        "config": config.to_dict(),
        "upstream_patches": upstream_patch_status,
        "prediction_enrichment": prediction_enrichment,
        "resume_state": _write_resume_state(
            output,
            skip_existing=config.skip_existing,
        ),
        "resume_launcher": resume_launcher,
        "resume_env": str(output / "resume_env.sh"),
        "run_report": run_report,
        "resume_environment": _resume_environment_snapshot(),
        "finished_at": time.time(),
        "live_subprocess_count": _live_test_subprocess_count(),
        "run_manifest": str(manifest_path),
        "fairness_audit": fairness_audit_payload,
    }
    if config.baseline_run_dir:
        payload["run_compare"] = _write_run_compare(
            baseline_run_dir=config.baseline_run_dir,
            candidate_run_dir=output,
        )
    atomic_write_json(status_path, payload)
    return payload


def _capture_manifest_resilient(
    *,
    config: TestGenEvalLiteRunConfig,
    fairness_mode: FairnessAuditMode,
    upstream_patch_status: dict[str, Any],
) -> RunManifest:
    """Create a :class:`RunManifest`, falling back gracefully when the
    subprocess shims (e.g. test-mocked ``subprocess.Popen``) interfere
    with ``git`` / ``importlib.metadata`` introspection. The fallback
    returns a manifest with the ``apex_git_sha`` set to ``"unknown"``
    so the rest of the run still produces a manifest artifact.
    """
    apex_config_payload = {
        "fairness_audit_mode": fairness_mode.value,
        "apply_upstream_patches": bool(config.apply_upstream_patches),
        "model_name": config.model_name,
        "dataset_name": config.dataset_name,
        "docker_namespace": config.docker_namespace,
        "split": config.split,
        "skip_mutation": bool(config.skip_mutation),
        "skip_existing": bool(config.skip_existing),
        "task_parallelism": int(config.task_parallelism),
        "timeout_seconds": int(config.timeout_seconds),
        "mutation_timeout_seconds": int(config.mutation_timeout_seconds),
        "task_ids": list(config.task_ids or []),
    }
    try:
        manifest = RunManifest.capture(
            apex_config=apex_config_payload,
            seed=config.run_seed,
        )
    except Exception:  # noqa: BLE001 — defensive; capture must never abort the run
        manifest = RunManifest(
            apex_git_sha="unknown",
            apex_git_dirty=False,
            python_version=sys.version.split()[0],
            platform="unknown",
            started_at="",
            seed=config.run_seed,
            apex_config=apex_config_payload,
        )
    try:
        for harness, version in detect_upstream_harness_versions().items():
            manifest.add_upstream_harness(harness, version)
    except Exception:  # noqa: BLE001 — defensive
        pass
    manifest.add_model("testgeneval_predictions_model", config.model_name)
    manifest.additional_metadata["docker_namespace"] = config.docker_namespace
    manifest.additional_metadata["upstream_patch_status"] = upstream_patch_status
    return manifest


def _coerce_fairness_audit_mode(value: Any) -> FairnessAuditMode:
    if isinstance(value, FairnessAuditMode):
        return value
    text = str(value or "off").strip().lower()
    try:
        return FairnessAuditMode(text)
    except ValueError:
        return FairnessAuditMode.OFF


def _run_fairness_audit_parallel(
    *,
    output_dir: Path,
    config: TestGenEvalLiteRunConfig,
) -> dict[str, Any]:
    """Score every task with both scorers and emit a fairness audit.

    The actual scorer implementations live in
    ``apex.evaluation.scorers.testgeneval_private`` and
    ``apex.evaluation.scorers.testgeneval_upstream``. This thin
    orchestrator iterates the per-task eval logs the harness already
    wrote, builds a synthetic ``task`` dict (instance_id only), and
    calls ``run_fairness_audit`` on each. The aggregated output is
    written under ``output_dir / "fairness_audit"``.
    """
    from apex.core.fairness_audit import run_fairness_audit
    from apex.evaluation.scorers.testgeneval_private import (
        TestGenEvalPrivateScorer,
    )
    from apex.evaluation.scorers.testgeneval_upstream import (
        TestGenEvalUpstreamScorer,
    )

    log_dir = output_dir / "official_eval_logs"
    if not log_dir.exists():
        return {
            "mode": FairnessAuditMode.PARALLEL.value,
            "status": "skipped",
            "reason": "official_eval_logs missing",
        }

    private_scorer = TestGenEvalPrivateScorer(
        log_dir=log_dir,
        model_name=config.model_name,
        dataset_name=config.dataset_name,
        split="test",
    )
    upstream_scorer = TestGenEvalUpstreamScorer(
        log_dir=log_dir,
        model_name=config.model_name,
        dataset_name=config.dataset_name,
        split="test",
        official_reports_dir=output_dir / "official_reports",
    )

    legacy_diagnostic = _legacy_aggregator_diagnostic(
        log_dir=log_dir,
        model_name=config.model_name,
        dataset_name=config.dataset_name,
    )
    legacy_note = f"parallel_log_marker_legacy_aggregator: {legacy_diagnostic}"

    aggregator = FairnessAuditAggregator()
    apex_artifacts = {"output_dir": str(output_dir)}
    instance_ids = _enumerate_audit_instance_ids(log_dir, config.model_name)
    for instance_id in instance_ids:
        task_obj = _AuditTask(instance_id)
        delta = run_fairness_audit(
            task=task_obj,
            apex_artifacts=apex_artifacts,
            private_scorer=private_scorer,
            upstream_scorer=upstream_scorer,
            extra_notes=[legacy_note],
        )
        aggregator.add_task(delta)

    audit_dir = output_dir / "fairness_audit"
    artifacts = aggregator.write_to(audit_dir)
    return {
        "mode": FairnessAuditMode.PARALLEL.value,
        "status": "written",
        "num_tasks": len(aggregator),
        "summary": aggregator.summary(),
        "json_path": str(artifacts["json"]),
        "markdown_path": str(artifacts["markdown"]),
        "legacy_pass_at_1_diagnostic": legacy_diagnostic,
    }


def _legacy_aggregator_diagnostic(
    *,
    log_dir: Path,
    model_name: str,
    dataset_name: str,
) -> dict[str, Any]:
    """Best-effort legacy aggregator output for the audit notes.

    Returns a small dict suitable to embed in the per-task audit
    ``notes``. Falls back to a sentinel when the dataset cannot be
    loaded (offline / no HF token) so the audit still emits.
    """
    try:
        from apex.evaluation.runners.testgenevallite_aggregate import (
            aggregate_eval_logs,
        )

        aggregate, _per_task = aggregate_eval_logs(
            log_dir=log_dir,
            model_name=model_name,
            dataset_name=dataset_name,
            split="test",
        )
        return aggregate.as_legacy_diagnostic()
    except Exception as exc:  # pragma: no cover - diagnostics, never fatal
        return {
            "scored_via": "parallel_log_marker_aggregator",
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        }


@dataclass(frozen=True)
class _AuditTask:
    """Minimal task container expected by ``run_fairness_audit``."""

    instance_id: str

    @property
    def task_id(self) -> str:
        return self.instance_id


def _enumerate_audit_instance_ids(log_dir: Path, model_name: str) -> list[str]:
    """Return the instance_ids reachable from ``log_dir`` for ``model_name``.

    We use the per-model ``.eval.log`` files written by the harness.
    Suffix is ``.<model_name>.full.eval.log`` per the upstream
    convention.
    """
    suffix = f".{model_name}.full.eval.log"
    ids: list[str] = []
    seen: set[str] = set()
    for path in sorted(log_dir.iterdir()):
        if not path.name.endswith(suffix):
            continue
        token = path.name[: -len(suffix)]
        if token in seen:
            continue
        seen.add(token)
        ids.append(token)
    return ids


def _run_official_report_generator(
    *,
    official_repo: Path,
    predictions_jsonl: Path,
    log_dir: Path,
    output_dir: Path,
    dataset_name: str,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run the upstream ``generate_report.py`` so ``official_reports/<model>_full.json``
    exists for the W10 join. Idempotent; safe to call after a partial harness
    run."""

    script = official_repo / "generate_report.py"
    if not script.exists():
        return {"status": "missing_generate_report"}
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script),
        "--predictions_path",
        str(predictions_jsonl),
        "--log_dir",
        str(log_dir),
        "--swe_bench_tasks",
        dataset_name,
        "--output_dir",
        str(output_dir),
    ]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(official_repo),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "command": cmd}
    except OSError as exc:
        return {"status": "exception", "error": f"{type(exc).__name__}: {exc}"}
    return {
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stdout_tail": (completed.stdout or "")[-2000:],
        "stderr_tail": (completed.stderr or "")[-2000:],
    }


def _live_test_subprocess_count() -> int:
    try:
        probe = subprocess.run(
            ["ps", "-axo", "command="],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0
    if probe.returncode != 0:
        return 0
    count = 0
    for command in probe.stdout.splitlines():
        lowered = command.lower()
        if any(token in lowered for token in ("runtests.py", "pytest", "cosmic-ray")):
            count += 1
    return count


def _summarize_resume_state(output: Path, *, skip_existing: bool) -> dict[str, Any]:
    log_dir = output / "official_eval_logs"
    logs = sorted(log_dir.glob("*.eval.log")) if log_dir.exists() else []
    completed = 0
    partial = 0
    tasks: list[dict[str, Any]] = []
    for log in logs:
        try:
            text = log.read_text(encoding="utf-8", errors="ignore")[-4000:]
        except OSError:
            partial += 1
            continue
        if ">>>>> All Tests Passed" in text or ">>>>> Some Tests Failed" in text:
            completed += 1
            status = "completed"
            last_completed_tier = "official_eval"
        else:
            partial += 1
            status = "partial"
            last_completed_tier = "unknown"
        task_id = log.name
        if task_id.endswith(".eval.log"):
            task_id = task_id[: -len(".eval.log")]
        tasks.append(
            {
                "task_id": task_id,
                "status": status,
                "last_completed_tier": last_completed_tier,
                "log_path": str(log),
            }
        )
    return {
        "eval_log_count": len(logs),
        "completed_eval_log_count": completed,
        "partial_eval_log_count": partial,
        "skip_existing": bool(skip_existing),
        "resumable": bool(skip_existing),
        "state_path": str(output / "resume_state.json"),
        "tasks": tasks,
    }


def _write_resume_state(output: Path, *, skip_existing: bool) -> dict[str, Any]:
    state = _summarize_resume_state(output, skip_existing=skip_existing)
    atomic_write_json(output / "resume_state.json", state)
    return state


def _write_resume_launcher(
    output: Path,
    config: TestGenEvalLiteRunConfig,
    command: list[str],
) -> str:
    script = output / "resume_testgenevallite.sh"
    env_script = output / "resume_env.sh"
    env = _resume_environment_snapshot()
    env_lines = [
        "#!/usr/bin/env bash",
        "# Non-secret environment captured when the TestGenEvalLite run started.",
    ]
    for name in ("SWEBENCH_DOCKER_FORK_DIR", "DOCKER_HOST"):
        value = str(env.get(name) or "")
        if value:
            env_lines.append(f"export {name}={_shell_quote(value)}")
    env_lines.append(
        "# HF tokens are intentionally not written here; keep them in the shell environment."
    )
    atomic_write_text(env_script, "\n".join(env_lines) + "\n")
    env_script.chmod(0o755)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"source {str(env_script)!r}",
        f"cd {str(Path(config.official_repo).expanduser())!r}",
    ]
    lines.extend(
        [
            "# Secret tokens are intentionally not written here; keep them in the shell environment.",
            " ".join(_shell_quote(part) for part in command),
            "",
        ]
    )
    atomic_write_text(script, "\n".join(lines))
    script.chmod(0o755)
    return str(script)


def _select_official_full_json(output: Path, model_name: str = "") -> Path | None:
    reports = output / "official_reports"
    if not reports.exists():
        return None
    if model_name:
        exact = reports / f"{model_name}_full.json"
        if exact.exists():
            return exact
    candidates = sorted(reports.glob("*_full.json"))
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise RuntimeError(
            "ambiguous official full reports: " + ", ".join(path.name for path in candidates[:8])
        )
    return None


def _write_official_run_report(output: Path, model_name: str = "") -> dict[str, Any]:
    try:
        from apex.evaluation.run_artifacts import (
            join_harness_results_into_records,
            write_testgen_run_report,
        )
        from apex.evaluation.run_compare import _load_run_records

        join_report = None
        records_dir = output / "records"
        full_json = _select_official_full_json(output, model_name=model_name)
        if records_dir.exists() and full_json is not None:
            join_report = join_harness_results_into_records(records_dir, full_json).to_dict()
        records = _load_run_records(output)
        tasks = list(dict(records.get("tasks") or {}).values())
        summary = dict(records.get("summary") or {})
        if not tasks and not summary:
            return {"status": "no_records"}
        path = write_testgen_run_report(
            output,
            summary=summary,
            task_records=tasks,
        )
        return {"status": "written", "path": str(path), "join_report": join_report}
    except Exception as exc:  # pragma: no cover - diagnostics should not fail run cleanup
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def _resume_environment_snapshot() -> dict[str, Any]:
    return {
        "SWEBENCH_DOCKER_FORK_DIR": os.environ.get("SWEBENCH_DOCKER_FORK_DIR", ""),
        "HF_TOKEN_present": bool(os.environ.get("HF_TOKEN")),
        "HUGGING_FACE_HUB_TOKEN_present": bool(os.environ.get("HUGGING_FACE_HUB_TOKEN")),
        "DOCKER_HOST": os.environ.get("DOCKER_HOST", ""),
    }


def _shell_quote(value: Any) -> str:
    text = str(value)
    if not text:
        return "''"
    if all(ch.isalnum() or ch in "@%_+=:,./-" for ch in text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def _write_run_compare(
    *,
    baseline_run_dir: str,
    candidate_run_dir: Path,
) -> dict[str, Any]:
    try:
        from apex.evaluation.checkpointing import atomic_write_text
        from apex.evaluation.run_compare import (
            compare_testgen_runs,
            render_testgen_run_comparison,
        )

        comparison = compare_testgen_runs(baseline_run_dir, candidate_run_dir)
        atomic_write_json(candidate_run_dir / "RUN_COMPARE.json", comparison)
        atomic_write_text(
            candidate_run_dir / "RUN_COMPARE.md",
            render_testgen_run_comparison(comparison),
        )
        return {
            "status": "written",
            "baseline_run_dir": baseline_run_dir,
            "json_path": str(candidate_run_dir / "RUN_COMPARE.json"),
            "markdown_path": str(candidate_run_dir / "RUN_COMPARE.md"),
        }
    except Exception as exc:  # pragma: no cover - diagnostics should not fail run cleanup
        return {
            "status": "failed",
            "baseline_run_dir": baseline_run_dir,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _parse_args(argv: list[str]) -> TestGenEvalLiteRunConfig:
    parser = argparse.ArgumentParser(description="Run official TestGenEvalLite scoring.")
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--predictions-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="apex")
    parser.add_argument("--split", default="testgenevallite")
    # Default 0 -> let ``resolve_task_parallelism`` pick the host budget.
    # Operators passing ``--task-parallelism N`` (N>=1) still win.
    parser.add_argument(
        "--task-parallelism",
        type=int,
        default=0,
        help=(
            "Worker count for parallel docker runs. Defaults to "
            "min(task_count, host_cpu_or_docker_cpu, 4) when 0/unset."
        ),
    )
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--mutation-timeout-seconds", type=int, default=3600)
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--no-preflight", action="store_true")
    parser.add_argument("--force-preflight", action="store_true")
    parser.add_argument("--baseline-run-dir", default="")
    parser.add_argument("--apply-upstream-patches", action="store_true")
    parser.add_argument("--no-apply-upstream-patches", action="store_true")
    parser.add_argument("--skip-mutation", action="store_true")
    parser.add_argument("--dataset", default="kjain14/testgenevallite")
    parser.add_argument("--docker-namespace", default="kdjain")
    parser.add_argument("--docker-platform", default=os.environ.get("APEX_DOCKER_PLATFORM", ""))
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    kwargs: dict[str, Any] = dict(
        official_repo=args.official_repo,
        predictions_jsonl=args.predictions_jsonl,
        output_dir=args.output_dir,
        model_name=args.model_name,
        split=args.split,
        timeout_seconds=args.timeout_seconds,
        mutation_timeout_seconds=args.mutation_timeout_seconds,
        skip_existing=not args.no_skip_existing,
        skip_mutation=args.skip_mutation,
        dataset_name=args.dataset,
        docker_namespace=args.docker_namespace,
        task_ids=list(args.task_id or []),
        extra_args=list(args.extra_args or []),
        run_preflight=not args.no_preflight,
        force_preflight=args.force_preflight,
        baseline_run_dir=args.baseline_run_dir,
        apply_upstream_patches=(
            bool(args.apply_upstream_patches) and not bool(args.no_apply_upstream_patches)
        ),
        docker_platform=args.docker_platform,
    )
    # Operator passed 0 (or omitted): drop the field so the dataclass'
    # ``default_factory=default_task_parallelism`` runs and we get the
    # host-aware default. Otherwise honour their explicit number.
    if int(args.task_parallelism) >= 1:
        kwargs["task_parallelism"] = int(args.task_parallelism)
    return TestGenEvalLiteRunConfig(**kwargs)


def main(argv: list[str] | None = None) -> int:
    result = run_testgenevallite(_parse_args(list(argv or sys.argv[1:])))
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
