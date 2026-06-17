"""
Operational helpers for doctor, status, cleanup, resume/retry, watch, replay,
experiment matrices, and run archiving.
"""

from __future__ import annotations

import copy
import gzip
import json
import math
import os
import re
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Optional

from .core.config import ApexConfig
from .evaluation.benchmark import BenchmarkRunner
from .evaluation.commit0_benchmark import Commit0BenchmarkRunner
from .evaluation.compare import compare_benchmark_reports
from .evaluation.run_artifacts import (
    compare_run_directories,
    doctor_summary,
    inspect_run_directory,
    render_status_table,
    update_run_manifest,
)
from .evaluation.swebench_pro_benchmark import SWEBenchProBenchmarkRunner


def render_doctor_report(payload: dict[str, Any]) -> str:
    lines = [
        f"Doctor success: {'yes' if payload.get('success') else 'no'}",
        f"Config hash: {payload.get('config_hash') or 'unknown'}",
    ]
    git = payload.get("git") or {}
    if git.get("sha"):
        lines.append(f"Git SHA: {git.get('sha')}")
    backends = payload.get("backend_health", []) or []
    healthy_count = payload.get("healthy_backend_count")
    if healthy_count is None:
        healthy_count = sum(1 for item in backends if bool(item.get("healthy")))
    lines.extend(
        [
            "",
            "Backend health: {healthy}/{total} CLI agent(s) healthy "
            "(>=1 required for success).".format(
                healthy=int(healthy_count),
                total=len(backends),
            ),
        ]
    )
    for backend in backends:
        version = ((backend.get("version") or {}).get("version")) or "unknown"
        # Per-backend status: "working" when healthy, "unavailable" when
        # the CLI isn't installed or its launcher returned an OS error,
        # "unhealthy" when it's installed but the probe failed (timeout,
        # nonzero exit). The bucket is informative; only the aggregate
        # ">=1 healthy" gates overall success.
        status_bucket = "working"
        if not bool(backend.get("healthy")):
            reason = str(backend.get("unavailable_reason") or "").lower()
            if "not installed" in reason or "no such" in reason or "not on path" in reason:
                status_bucket = "unavailable"
            else:
                status_bucket = "unhealthy"
        lines.append(
            "- {backend} model={model} status={status} version={version}".format(
                backend=backend.get("backend"),
                model=backend.get("model"),
                status=status_bucket,
                version=version,
            )
        )
        reason = backend.get("unavailable_reason")
        if reason:
            lines.append(f"  reason: {reason}")
    if payload.get("backend_smoke_tests"):
        lines.extend(["", "Structured-output smoke tests:"])
        for smoke in payload["backend_smoke_tests"]:
            lines.append(
                "- {backend} success={success} duration={duration:.1f}s".format(
                    backend=smoke.get("backend"),
                    success=smoke.get("success"),
                    duration=float(smoke.get("duration_seconds") or 0.0),
                )
            )
            if smoke.get("error"):
                lines.append(f"  error: {smoke['error']}")
    if payload.get("backend_tool_smoke_tests"):
        lines.extend(["", "Tool-call smoke tests:"])
        for smoke in payload["backend_tool_smoke_tests"]:
            lines.append(
                "- {backend} success={success} duration={duration:.1f}s".format(
                    backend=smoke.get("backend"),
                    success=smoke.get("success"),
                    duration=float(smoke.get("duration_seconds") or 0.0),
                )
            )
            if smoke.get("error"):
                lines.append(f"  error: {smoke['error']}")
    if payload.get("benchmark_env_parity"):
        lines.extend(["", "Benchmark env parity:"])
        for check in payload["benchmark_env_parity"]:
            label = "required" if check.get("required", True) else "optional"
            lines.append(
                f"- {check.get('name')} ({label}): {'ok' if check.get('success') else 'failed'}"
            )
            if check.get("note"):
                lines.append(f"  note: {check['note']}")
            if check.get("error"):
                lines.append(f"  error: {check['error']}")
    if payload.get("command_checks"):
        lines.extend(["", "Command checks:"])
        for check in payload["command_checks"]:
            lines.append(
                f"- {check.get('command')}: {'present' if check.get('available') else 'missing'}"
            )
    return "\n".join(lines)


def render_run_compare(payload: dict[str, Any]) -> str:
    lines = [
        f"Left run: {payload.get('left_run')}",
        f"Right run: {payload.get('right_run')}",
        f"Benchmark family: {payload.get('benchmark_family') or 'unknown'}",
        f"Score delta: {float(payload.get('score_delta_percent') or 0.0):+.2f}%",
        f"Solve delta: {float(payload.get('solve_delta_percent') or 0.0):+.2f}%",
        f"Config hash changed: {'yes' if payload.get('config_hash_changed') else 'no'}",
        f"Git SHA changed: {'yes' if payload.get('git_sha_changed') else 'no'}",
    ]
    if "prompt_template_hash_changed" in payload:
        lines.append(
            f"Prompt template hash changed: {'yes' if payload.get('prompt_template_hash_changed') else 'no'}"
        )
    return "\n".join(lines)


def _iter_processes() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    processes = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, command = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        processes.append({"pid": pid, "command": command.strip()})
    return processes


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _process_cwd(pid: int) -> Optional[Path]:
    if not shutil_which("lsof"):
        return None
    result = subprocess.run(
        ["lsof", "-a", "-d", "cwd", "-Fn", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.startswith("n"):
            return Path(line[1:]).resolve()
    return None


def shutil_which(command: str) -> Optional[str]:
    return (
        subprocess.run(
            ["which", command],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or None
    )


def _path_within(path: Optional[Path], roots: list[Path]) -> bool:
    if path is None:
        return False
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _looks_like_apex_worker_command(command: str) -> bool:
    normalized = command.lower()
    return any(
        marker in normalized
        for marker in (
            "codex exec",
            "claude -p",
            "gemini -p",
            "opencode run",
            "pytest",
            "python -m pytest",
            "python3 -m pytest",
        )
    )


def _terminate_pid(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.time() + 1.5
    while time.time() < deadline:
        if not _process_exists(pid):
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return


def _normalize_benchmark_family(manifest: dict[str, Any]) -> str:
    family = str(manifest.get("benchmark_family") or "").strip().lower()
    if family:
        return family
    entrypoint = str(((manifest.get("execution") or {}).get("entrypoint")) or "").strip().lower()
    if entrypoint == "benchmark":
        return "local"
    if entrypoint == "commit0-benchmark":
        return "commit0"
    if entrypoint == "swebench-pro-benchmark":
        return "swebench_pro"
    raise RuntimeError("Run manifest is missing benchmark family metadata.")


def _load_config_from_manifest(manifest: dict[str, Any]) -> ApexConfig:
    config_payload = manifest.get("config_payload")
    if isinstance(config_payload, dict):
        return ApexConfig._from_dict(copy.deepcopy(config_payload))
    config_source = manifest.get("config_source")
    if config_source:
        return ApexConfig.from_file(str(config_source))
    settings = dict(manifest.get("settings") or {})
    model_config = list(manifest.get("model_config") or [])
    fallback_payload = {
        "llm_configs": model_config or ApexConfig().to_dict().get("llm_configs", []),
        "rollout": {
            "num_rollouts": settings.get("num_rollouts", 5),
            "min_rollouts": settings.get("min_rollouts", 1),
            "max_rollouts": settings.get("max_rollouts", 16),
            "parallel_workers": settings.get("parallel_workers", 3),
            "llm_profiles": list(settings.get("rollout_profiles") or []),
        },
        "planning": {
            "planner_model": settings.get("planner_model"),
        },
        "search": {
            "mode": settings.get("search_mode", "off"),
        },
        "selection": {
            "strategy": settings.get("selection_strategy", "multi_stage"),
        },
        "benchmark": {
            "task_parallelism": settings.get("task_parallelism", 1),
        },
        "output_dir": str(manifest.get("output_dir") or ApexConfig().output_dir),
    }
    try:
        return ApexConfig._from_dict(fallback_payload)
    except Exception as exc:
        raise RuntimeError(
            "Run manifest does not contain a config snapshot or config source, "
            "and fallback reconstruction failed."
        ) from exc


def _execution_args_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    execution = dict(manifest.get("execution") or {})
    return dict(execution.get("args") or {})


def _create_runner_for_family(
    *,
    family: str,
    config: ApexConfig,
    output_dir: str | Path,
    args: dict[str, Any],
    config_source: Optional[str],
) -> Any:
    normalized_family = family.lower()
    if normalized_family == "local":
        fixtures_dir = args.get("fixtures_dir")
        if not fixtures_dir:
            raise RuntimeError("Local benchmark manifest is missing fixtures_dir.")
        runner = BenchmarkRunner(
            config=config,
            fixtures_dir=str(fixtures_dir),
            output_dir=str(output_dir),
        )
        runner.config_source = config_source
        return runner
    if normalized_family == "commit0":
        runner = Commit0BenchmarkRunner(
            config=config,
            output_dir=str(output_dir),
            dataset_name=str(args.get("dataset_name") or "wentingzhao/commit0_combined"),
            dataset_split=str(args.get("dataset_split") or "test"),
            split=str(args.get("split") or "lite"),
        )
        runner.config_source = config_source
        return runner
    if normalized_family == "swebench_pro":
        runner = SWEBenchProBenchmarkRunner(
            config=config,
            output_dir=str(output_dir),
            dataset_name=str(args.get("dataset_name") or "ScaleAI/SWE-bench_Pro"),
            dataset_split=str(args.get("dataset_split") or "test"),
            dockerhub_username=str(args.get("dockerhub_username") or "jefzda"),
            scripts_cache_dir=args.get("scripts_cache_dir"),
            docker_platform=args.get("docker_platform"),
            block_network=bool(args.get("block_network", False)),
            agent_visibility_mode=str(args.get("agent_visibility_mode") or "published_parity"),
            rollout_selection_policy=str(args.get("rollout_selection_policy") or "orchestrator"),
        )
        runner.config_source = config_source
        return runner
    raise RuntimeError(f"Unsupported benchmark family for replay/resume: {family}")


def _run_runner_for_family(
    *,
    family: str,
    runner: Any,
    args: dict[str, Any],
    subset_task_ids: Optional[list[str]] = None,
) -> Any:
    normalized_family = family.lower()
    if normalized_family == "local":
        task_names = (
            list(subset_task_ids)
            if subset_task_ids is not None
            else list(args.get("task_names") or [])
        )
        return runner.run(task_names=task_names or None)
    if normalized_family == "commit0":
        repos = (
            list(subset_task_ids) if subset_task_ids is not None else list(args.get("repos") or [])
        )
        return runner.run(
            repos=repos or None,
            limit=None if subset_task_ids is not None else args.get("limit"),
        )
    if normalized_family == "swebench_pro":
        if subset_task_ids is not None:
            return runner.run(instances=list(subset_task_ids))
        return runner.run(
            instances=list(args.get("instances") or []) or None,
            repos=list(args.get("repos") or []) or None,
            languages=list(args.get("languages") or []) or None,
            limit=args.get("limit"),
        )
    raise RuntimeError(f"Unsupported benchmark family: {family}")


def _run_recorded_benchmark(
    *,
    run_dir: str | Path,
    manifest: dict[str, Any],
    subset_task_ids: Optional[list[str]] = None,
) -> Any:
    family = _normalize_benchmark_family(manifest)
    config = _load_config_from_manifest(manifest)
    config.output_dir = str(Path(run_dir).resolve())
    args = _execution_args_from_manifest(manifest)
    runner = _create_runner_for_family(
        family=family,
        config=config,
        output_dir=run_dir,
        args=args,
        config_source=manifest.get("config_source"),
    )
    return _run_runner_for_family(
        family=family,
        runner=runner,
        args=args,
        subset_task_ids=subset_task_ids,
    )


def _report_summary(report: Any, family: str, *, run_dir: str | Path) -> dict[str, Any]:
    normalized_family = family.lower()
    if normalized_family == "commit0":
        return {
            "primary_metric_name": "average_pass_rate_percent",
            "primary_metric_percent": float(getattr(report, "average_pass_rate_percent", 0.0)),
            "solved_rate_percent": float(getattr(report, "solved_rate_percent", 0.0)),
            "total_tasks": int(getattr(report, "total_tasks", 0)),
            "report_path": str(Path(run_dir) / "benchmark_report.json"),
        }
    if normalized_family == "swebench_pro":
        return {
            "primary_metric_name": "score_percent",
            "primary_metric_percent": float(getattr(report, "score_percent", 0.0)),
            "solved_rate_percent": float(getattr(report, "score_percent", 0.0)),
            "total_tasks": int(getattr(report, "total_tasks", 0)),
            "report_path": str(Path(run_dir) / "benchmark_report.json"),
        }
    total_tasks = int(getattr(report, "total_tasks", 0))
    resolved_tasks = int(getattr(report, "resolved_tasks", 0))
    resolved_rate_percent = 100.0 * (resolved_tasks / total_tasks) if total_tasks else 0.0
    return {
        "primary_metric_name": "resolved_rate_percent",
        "primary_metric_percent": resolved_rate_percent,
        "solved_rate_percent": resolved_rate_percent,
        "total_tasks": total_tasks,
        "report_path": str(Path(run_dir) / "benchmark_report.json"),
    }


def _active_task_ids(status: dict[str, Any]) -> list[str]:
    return [
        str(task.get("task_id"))
        for task in status.get("tasks", [])
        if task.get("health") == "running"
    ]


def _assert_run_not_active(status: dict[str, Any], *, force: bool) -> None:
    active_task_ids = _active_task_ids(status)
    if active_task_ids and not force:
        raise RuntimeError(
            "Run still appears active. Re-run with --force if you want to restart it anyway. "
            f"Active tasks: {', '.join(active_task_ids[:8])}"
        )


def _select_retry_task_ids(
    status: dict[str, Any],
    *,
    failed_only: bool = False,
    suspicious_only: bool = False,
    task_ids: Optional[list[str]] = None,
) -> list[str]:
    tasks = {str(task.get("task_id")): task for task in status.get("tasks", [])}
    if task_ids:
        unknown = [task_id for task_id in task_ids if task_id not in tasks]
        if unknown:
            raise RuntimeError(f"Unknown task ids: {', '.join(unknown)}")
        selected = list(task_ids)
    else:
        selected = []
        for task_id, task in tasks.items():
            health = str(task.get("health") or "")
            if failed_only and health == "failed":
                selected.append(task_id)
            elif suspicious_only and health == "suspicious":
                selected.append(task_id)
            elif not failed_only and not suspicious_only and health in {"failed", "suspicious"}:
                selected.append(task_id)
    if task_ids and (failed_only or suspicious_only):
        selected = [
            task_id
            for task_id in selected
            if (
                (failed_only and tasks[task_id].get("health") == "failed")
                or (suspicious_only and tasks[task_id].get("health") == "suspicious")
                or (
                    failed_only
                    and suspicious_only
                    and tasks[task_id].get("health") in {"failed", "suspicious"}
                )
            )
        ]
    return list(dict.fromkeys(selected))


def _resume_task_ids(status: dict[str, Any]) -> list[str]:
    return [
        str(task.get("task_id"))
        for task in status.get("tasks", [])
        if str(task.get("health") or "") in {"running", "suspicious", "pending"}
    ]


def _purge_task_artifacts(
    status: dict[str, Any],
    *,
    task_ids: list[str],
    dry_run: bool = False,
) -> list[str]:
    selected = set(task_ids)
    root = Path(status["run_dir"]).resolve()
    removed: list[str] = []
    for task in status.get("tasks", []):
        task_id = str(task.get("task_id"))
        if task_id not in selected:
            continue
        targets = [
            Path(task.get("path") or root / task_id),
            root / "workspaces" / task_id,
            root / ".runtime" / task_id,
        ]
        for target in targets:
            if not target.exists():
                continue
            removed.append(str(target))
            if not dry_run:
                subprocess.run(["rm", "-rf", str(target)], check=False)
    return removed


def resume_run(
    run_dir: str | Path,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    status = inspect_run_directory(run_dir)
    _assert_run_not_active(status, force=force)
    manifest = dict(status.get("manifest") or {})
    if not manifest:
        raise RuntimeError("Run manifest not found; this run cannot be resumed.")
    selected_task_ids = _resume_task_ids(status)
    payload = {
        "action": "resume",
        "run_dir": str(Path(run_dir).resolve()),
        "benchmark_family": status.get("benchmark_family"),
        "selected_task_ids": selected_task_ids,
        "dry_run": dry_run,
        "no_op": not bool(selected_task_ids),
    }
    if dry_run or not selected_task_ids:
        return payload
    report = _run_recorded_benchmark(run_dir=run_dir, manifest=manifest)
    payload.update(
        _report_summary(report, status.get("benchmark_family") or "local", run_dir=run_dir)
    )
    return payload


def retry_run(
    run_dir: str | Path,
    *,
    failed_only: bool = False,
    suspicious_only: bool = False,
    task_ids: Optional[list[str]] = None,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    status = inspect_run_directory(run_dir)
    _assert_run_not_active(status, force=force)
    manifest = dict(status.get("manifest") or {})
    if not manifest:
        raise RuntimeError("Run manifest not found; this run cannot be retried.")
    selected_task_ids = _select_retry_task_ids(
        status,
        failed_only=failed_only,
        suspicious_only=suspicious_only,
        task_ids=task_ids,
    )
    removed_paths = _purge_task_artifacts(
        status,
        task_ids=selected_task_ids,
        dry_run=dry_run,
    )
    payload = {
        "action": "retry",
        "run_dir": str(Path(run_dir).resolve()),
        "benchmark_family": status.get("benchmark_family"),
        "selected_task_ids": selected_task_ids,
        "removed_paths": removed_paths,
        "dry_run": dry_run,
        "no_op": not bool(selected_task_ids),
    }
    if dry_run or not selected_task_ids:
        return payload
    report = _run_recorded_benchmark(run_dir=run_dir, manifest=manifest)
    payload.update(
        _report_summary(report, status.get("benchmark_family") or "local", run_dir=run_dir)
    )
    return payload


def render_watch_frame(status: dict[str, Any]) -> str:
    lines = [
        f"APEX watch: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}",
        "",
        render_status_table(status),
    ]
    manifest = status.get("manifest_summary") or {}
    if manifest.get("backends"):
        lines.extend(["", "Manifest backend snapshot:"])
        for backend in manifest.get("backends") or []:
            lines.append(
                "- {backend} model={model} healthy={healthy} version={version}".format(
                    backend=backend.get("backend"),
                    model=backend.get("model"),
                    healthy=backend.get("healthy"),
                    version=backend.get("version") or "unknown",
                )
            )
    failure_clusters = status.get("failure_clusters") or []
    if failure_clusters:
        lines.extend(["", "Failure clusters:"])
        for cluster in failure_clusters[:5]:
            lines.append(
                "- {bucket}: {count}".format(
                    bucket=cluster.get("bucket"),
                    count=cluster.get("count"),
                )
            )
    return "\n".join(lines)


def watch_run(
    run_dir: str | Path,
    *,
    refresh_seconds: float = 2.0,
    iterations: Optional[int] = None,
    no_clear: bool = False,
) -> dict[str, Any]:
    remaining = None if iterations is None else max(1, int(iterations))
    latest_status: dict[str, Any] = {}
    while remaining is None or remaining > 0:
        latest_status = inspect_run_directory(run_dir)
        frame = render_watch_frame(latest_status)
        if not no_clear:
            sys.stdout.write("\033[2J\033[H")
        sys.stdout.write(frame + ("\n" if not frame.endswith("\n") else ""))
        sys.stdout.flush()
        if remaining is not None:
            remaining -= 1
            if remaining <= 0:
                break
        time.sleep(max(0.1, float(refresh_seconds)))
    return latest_status


def replay_failure(
    run_dir: str | Path,
    *,
    task_id: Optional[str] = None,
    cluster: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if bool(task_id) == bool(cluster):
        raise RuntimeError("Specify exactly one of task_id or cluster for replay.")

    status = inspect_run_directory(run_dir)
    manifest = dict(status.get("manifest") or {})
    if not manifest:
        raise RuntimeError("Run manifest not found; this run cannot be replayed.")

    if task_id:
        selected_task_ids = [task_id]
    else:
        selected_task_ids = [
            str(task.get("task_id"))
            for task in status.get("tasks", [])
            if str(task.get("failure_root") or "") == str(cluster)
        ]
    if not selected_task_ids:
        raise RuntimeError("No failed tasks matched the requested replay target.")

    label = task_id or cluster or "replay"
    destination = Path(
        output_dir or (Path(run_dir) / "replays" / f"{label}-{int(time.time())}")
    ).resolve()
    payload = {
        "action": "replay",
        "source_run_dir": str(Path(run_dir).resolve()),
        "output_dir": str(destination),
        "benchmark_family": status.get("benchmark_family"),
        "selected_task_ids": selected_task_ids,
        "dry_run": dry_run,
    }
    if dry_run:
        return payload

    destination.mkdir(parents=True, exist_ok=True)
    report = _run_recorded_benchmark(
        run_dir=destination,
        manifest=manifest,
        subset_task_ids=selected_task_ids,
    )
    update_run_manifest(
        destination,
        extra_updates={
            "replay_source_run_dir": str(Path(run_dir).resolve()),
            "replay_source_cluster": cluster,
            "replay_selected_task_ids": list(selected_task_ids),
        },
    )
    payload.update(
        _report_summary(report, status.get("benchmark_family") or "local", run_dir=destination)
    )
    return payload


def _sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return slug or "variant"


def _deep_merge_dict(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict) and "." not in str(key):
            merged[key] = _deep_merge_dict(dict(merged[key]), value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _apply_override_path(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    cursor: Any = payload
    parts = [part for part in dotted_path.split(".") if part]
    if not parts:
        return
    for index, part in enumerate(parts[:-1]):
        next_part = parts[index + 1]
        if isinstance(cursor, list):
            cursor = cursor[int(part)]
            continue
        if part not in cursor or not isinstance(cursor[part], (dict, list)):
            cursor[part] = [] if next_part.isdigit() else {}
        cursor = cursor[part]
    leaf = parts[-1]
    if isinstance(cursor, list):
        cursor[int(leaf)] = copy.deepcopy(value)
    else:
        cursor[leaf] = copy.deepcopy(value)


def _apply_config_overrides(
    base_payload: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    payload = copy.deepcopy(base_payload)
    for key, value in overrides.items():
        if "." in str(key):
            _apply_override_path(payload, str(key), value)
        elif isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key] = _deep_merge_dict(dict(payload[key]), value)
        else:
            payload[key] = copy.deepcopy(value)
    return payload


def _sign_test_p_value(wins: int, losses: int) -> Optional[float]:
    trials = int(wins) + int(losses)
    if trials <= 0:
        return None
    dominant = max(int(wins), int(losses))
    tail = 0.0
    for count in range(dominant, trials + 1):
        tail += math.comb(trials, count) / (2**trials)
    return min(1.0, 2.0 * tail)


def _primary_metric_from_report_payload(
    report: dict[str, Any], family: str
) -> tuple[str, float, float]:
    normalized_family = family.lower()
    if normalized_family == "commit0":
        return (
            "average_pass_rate_percent",
            float(report.get("average_pass_rate_percent") or 0.0),
            float(report.get("solved_rate_percent") or 0.0),
        )
    if normalized_family == "swebench_pro":
        return (
            "score_percent",
            float(report.get("score_percent") or 0.0),
            float(report.get("score_percent") or 0.0),
        )
    total_tasks = int(report.get("total_tasks") or 0)
    resolved_tasks = int(report.get("resolved_tasks") or 0)
    solved_rate_percent = 100.0 * (resolved_tasks / total_tasks) if total_tasks else 0.0
    return ("resolved_rate_percent", solved_rate_percent, solved_rate_percent)


def render_matrix_report(payload: dict[str, Any]) -> str:
    lines = [
        f"Experiment matrix: {payload.get('spec_path')}",
        f"Benchmark family: {payload.get('benchmark_family')}",
        f"Output root: {payload.get('output_root')}",
        "",
        "Variants:",
    ]
    for variant in payload.get("variants", []):
        lines.append(
            "- {name}: {metric}={score:.2f}% stdev={stdev:.2f} solved={solved:.2f}% repeats={repeats}".format(
                name=variant.get("name"),
                metric=variant.get("primary_metric_name"),
                score=float(variant.get("mean_primary_metric_percent") or 0.0),
                stdev=float(variant.get("stdev_primary_metric_percent") or 0.0),
                solved=float(variant.get("mean_solved_rate_percent") or 0.0),
                repeats=int(variant.get("repeat_count") or 0),
            )
        )
    pairwise = payload.get("pairwise_significance") or []
    if pairwise:
        lines.extend(["", "Pairwise significance vs baseline:"])
        for item in pairwise:
            lines.append(
                "- {candidate}: delta={delta:+.2f}% wins/losses/ties={wins}/{losses}/{ties} p={pvalue}".format(
                    candidate=item.get("candidate"),
                    delta=float(item.get("mean_score_delta_percent") or 0.0),
                    wins=int(item.get("score_wins") or 0),
                    losses=int(item.get("score_losses") or 0),
                    ties=int(item.get("score_ties") or 0),
                    pvalue=(
                        f"{float(item['sign_test_pvalue']):.4f}"
                        if item.get("sign_test_pvalue") is not None
                        else "n/a"
                    ),
                )
            )
    return "\n".join(lines)


def run_experiment_matrix(
    spec_path: str | Path,
    *,
    output: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
) -> dict[str, Any]:
    spec_file = Path(spec_path).resolve()
    spec = json.loads(spec_file.read_text())
    benchmark = dict(spec.get("benchmark") or {})
    family = str(benchmark.get("family") or "").strip().lower()
    if not family:
        raise RuntimeError("Experiment matrix spec requires benchmark.family.")
    variants = list(spec.get("variants") or [])
    if not variants:
        raise RuntimeError("Experiment matrix spec requires at least one variant.")
    repeats = max(1, int(spec.get("repeats", 1)))

    if isinstance(spec.get("config"), dict):
        base_config_payload = copy.deepcopy(spec["config"])
        config_source = None
    elif spec.get("base_config"):
        base_config_path = Path(spec["base_config"]).resolve()
        base_config_payload = ApexConfig.from_file(base_config_path).to_dict()
        config_source = str(base_config_path)
    else:
        base_config_payload = ApexConfig().to_dict()
        config_source = None

    resolved_output_dir = output_dir if output_dir is not None else output
    output_root = Path(
        resolved_output_dir
        or spec.get("output_dir")
        or (Path.cwd() / f".apex_matrix_{int(time.time())}")
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_payloads: list[dict[str, Any]] = []
    variant_summaries: list[dict[str, Any]] = []
    for variant_index, variant in enumerate(variants):
        name = str(variant.get("name") or f"variant_{variant_index + 1}")
        variant_slug = _sanitize_slug(name)
        config_overrides = dict(variant.get("overrides") or {})
        benchmark_overrides = dict(variant.get("benchmark_overrides") or {})
        variant_benchmark = _deep_merge_dict(benchmark, benchmark_overrides)
        family = str(variant_benchmark.get("family") or family).strip().lower()

        report_paths: list[str] = []
        run_dirs: list[str] = []
        metric_name = "score_percent"
        primary_metrics: list[float] = []
        solved_rates: list[float] = []
        for repeat_index in range(repeats):
            config_payload = _apply_config_overrides(base_config_payload, config_overrides)
            config = ApexConfig._from_dict(config_payload)
            run_dir = output_root / variant_slug / f"repeat_{repeat_index + 1:02d}"
            runner = _create_runner_for_family(
                family=family,
                config=config,
                output_dir=run_dir,
                args=variant_benchmark,
                config_source=config_source,
            )
            _run_runner_for_family(
                family=family,
                runner=runner,
                args=variant_benchmark,
                subset_task_ids=None,
            )
            report_path = run_dir / "benchmark_report.json"
            report_payload = json.loads(report_path.read_text())
            metric_name, primary_metric_percent, solved_rate_percent = (
                _primary_metric_from_report_payload(
                    report_payload,
                    family,
                )
            )
            update_run_manifest(
                run_dir,
                extra_updates={
                    "matrix_variant_name": name,
                    "matrix_variant_index": variant_index,
                    "matrix_repeat_index": repeat_index,
                    "matrix_spec_path": str(spec_file),
                },
            )
            run_payloads.append(
                {
                    "variant": name,
                    "variant_slug": variant_slug,
                    "repeat_index": repeat_index,
                    "run_dir": str(run_dir),
                    "report_path": str(report_path),
                    "primary_metric_name": metric_name,
                    "primary_metric_percent": primary_metric_percent,
                    "solved_rate_percent": solved_rate_percent,
                }
            )
            report_paths.append(str(report_path))
            run_dirs.append(str(run_dir))
            primary_metrics.append(primary_metric_percent)
            solved_rates.append(solved_rate_percent)

        mean_primary = sum(primary_metrics) / len(primary_metrics) if primary_metrics else 0.0
        mean_solved = sum(solved_rates) / len(solved_rates) if solved_rates else 0.0
        stdev_primary = 0.0
        if len(primary_metrics) > 1:
            variance = sum((value - mean_primary) ** 2 for value in primary_metrics) / (
                len(primary_metrics) - 1
            )
            stdev_primary = math.sqrt(max(0.0, variance))
        variant_summaries.append(
            {
                "name": name,
                "variant_slug": variant_slug,
                "repeat_count": len(primary_metrics),
                "primary_metric_name": metric_name,
                "mean_primary_metric_percent": mean_primary,
                "stdev_primary_metric_percent": stdev_primary,
                "mean_solved_rate_percent": mean_solved,
                "run_dirs": run_dirs,
                "report_paths": report_paths,
            }
        )

    pairwise_significance: list[dict[str, Any]] = []
    if variant_summaries:
        baseline = variant_summaries[0]
        for candidate in variant_summaries[1:]:
            compare_count = min(
                len(baseline.get("report_paths") or []),
                len(candidate.get("report_paths") or []),
            )
            score_deltas: list[float] = []
            solve_deltas: list[float] = []
            score_wins = 0
            score_losses = 0
            score_ties = 0
            common_task_count = 0
            for index in range(compare_count):
                comparison_payload = compare_benchmark_reports(
                    [
                        baseline["report_paths"][index],
                        candidate["report_paths"][index],
                    ],
                    labels=[baseline["name"], candidate["name"]],
                )
                comparison = comparison_payload["pairwise_comparisons"][0]
                score_deltas.append(float(comparison.get("average_score_delta_percent") or 0.0))
                solve_deltas.append(float(comparison.get("solve_rate_delta_percent") or 0.0))
                score_wins += int(comparison.get("score_wins") or 0)
                score_losses += int(comparison.get("score_losses") or 0)
                score_ties += int(comparison.get("score_ties") or 0)
                common_task_count += int(comparison_payload.get("common_task_count") or 0)
            pairwise_significance.append(
                {
                    "baseline": baseline["name"],
                    "candidate": candidate["name"],
                    "comparison_count": compare_count,
                    "common_task_count": common_task_count,
                    "mean_score_delta_percent": (
                        sum(score_deltas) / len(score_deltas) if score_deltas else 0.0
                    ),
                    "mean_solve_delta_percent": (
                        sum(solve_deltas) / len(solve_deltas) if solve_deltas else 0.0
                    ),
                    "score_wins": score_wins,
                    "score_losses": score_losses,
                    "score_ties": score_ties,
                    "sign_test_pvalue": _sign_test_p_value(score_wins, score_losses),
                }
            )

    payload = {
        "spec_path": str(spec_file),
        "benchmark_family": family,
        "output_root": str(output_root),
        "variants": variant_summaries,
        "runs": run_payloads,
        "pairwise_significance": pairwise_significance,
    }
    summary_json = output_root / "matrix_summary.json"
    summary_md = output_root / "matrix_summary.md"
    summary_json.write_text(json.dumps(payload, indent=2))
    summary_md.write_text(render_matrix_report(payload))
    payload["summary_json"] = str(summary_json)
    payload["summary_markdown"] = str(summary_md)
    return payload


def _replace_symlink(path: Path, target: Path) -> None:
    if path.is_symlink() or path.exists():
        path.unlink()
    path.symlink_to(target.name)


def _compress_log_file(path: Path) -> Optional[Path]:
    if path.suffix == ".gz" or not path.is_file():
        return None
    target = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as source, gzip.open(target, "wb") as destination:
        destination.write(source.read())
    path.unlink()
    return target


def archive_runs(
    run_dirs: list[str | Path],
    *,
    archive_root: str | Path,
    prune_workspaces: bool = False,
    prune_runtime: bool = False,
    compress_logs: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    normalized_runs = [Path(run_dir).resolve() for run_dir in run_dirs]
    archive_dir = Path(archive_root).resolve()
    if not dry_run:
        archive_dir.mkdir(parents=True, exist_ok=True)

    archives: list[dict[str, Any]] = []
    pruned_paths: list[str] = []
    compressed_logs_out: list[str] = []
    symlinks: list[str] = []

    for run_dir in normalized_runs:
        status = inspect_run_directory(run_dir)
        _assert_run_not_active(status, force=force)
        benchmark_family = str(status.get("benchmark_family") or "unknown")
        archive_path = archive_dir / f"{run_dir.name}.tar.gz"
        archives.append(
            {
                "run_dir": str(run_dir),
                "archive_path": str(archive_path),
                "benchmark_family": benchmark_family,
            }
        )
        if dry_run:
            continue

        with tarfile.open(archive_path, "w:gz") as handle:
            handle.add(run_dir, arcname=run_dir.name)

        if prune_workspaces:
            target = run_dir / "workspaces"
            if target.exists():
                subprocess.run(["rm", "-rf", str(target)], check=False)
                pruned_paths.append(str(target))
        if prune_runtime:
            target = run_dir / ".runtime"
            if target.exists():
                subprocess.run(["rm", "-rf", str(target)], check=False)
                pruned_paths.append(str(target))
        if compress_logs:
            for path in run_dir.rglob("*.log"):
                compressed = _compress_log_file(path)
                if compressed is not None:
                    compressed_logs_out.append(str(compressed))

        latest_link = archive_dir / "latest"
        _replace_symlink(latest_link, archive_path)
        symlinks.append(str(latest_link))
        family_link = archive_dir / f"latest-{benchmark_family}"
        _replace_symlink(family_link, archive_path)
        symlinks.append(str(family_link))

    return {
        "run_dirs": [str(path) for path in normalized_runs],
        "archive_root": str(archive_dir),
        "archives": archives,
        "pruned_paths": pruned_paths,
        "compressed_logs": compressed_logs_out,
        "latest_symlinks": symlinks,
        "dry_run": dry_run,
    }


def cleanup_runs(
    run_dirs: list[str | Path],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    normalized_runs = [Path(run_dir).resolve() for run_dir in run_dirs]
    stale_roots: list[Path] = []
    active_pids: set[int] = set()
    for run_dir in normalized_runs:
        status = inspect_run_directory(run_dir)
        run_has_active_worker = False
        for task in status.get("tasks", []):
            payloads = [task.get("live_state") or {}] + list(task.get("rollouts") or [])
            for payload in payloads:
                pid = payload.get("process_pid")
                if isinstance(pid, int) and _process_exists(pid):
                    active_pids.add(pid)
                    run_has_active_worker = True
        if not run_has_active_worker:
            stale_roots.append(run_dir)

    workspace_roots = [root / "workspaces" for root in stale_roots] + [
        root / ".runtime" for root in stale_roots
    ]
    killed_processes: list[dict[str, Any]] = []
    for process in _iter_processes():
        pid = int(process["pid"])
        if pid == os.getpid() or pid in active_pids:
            continue
        if not _looks_like_apex_worker_command(process["command"]):
            continue
        cwd = _process_cwd(pid)
        if not _path_within(cwd, workspace_roots + stale_roots):
            if not any(str(root) in process["command"] for root in stale_roots):
                continue
        killed_processes.append(
            {
                "pid": pid,
                "command": process["command"],
                "cwd": str(cwd) if cwd else None,
            }
        )
        if not dry_run:
            _terminate_pid(pid)

    removed_directories: list[str] = []
    for root in stale_roots:
        for target in (root / "workspaces", root / ".runtime"):
            if not target.exists():
                continue
            removed_directories.append(str(target))
            if not dry_run:
                subprocess.run(["rm", "-rf", str(target)], check=False)

    return {
        "run_dirs": [str(path) for path in normalized_runs],
        "stale_run_dirs": [str(path) for path in stale_roots],
        "killed_processes": killed_processes,
        "removed_directories": removed_directories,
        "dry_run": dry_run,
    }


__all__ = [
    "archive_runs",
    "cleanup_runs",
    "compare_run_directories",
    "doctor_summary",
    "inspect_run_directory",
    "render_doctor_report",
    "render_matrix_report",
    "render_run_compare",
    "render_status_table",
    "render_watch_frame",
    "replay_failure",
    "resume_run",
    "retry_run",
    "run_experiment_matrix",
    "watch_run",
]
