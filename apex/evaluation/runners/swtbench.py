"""Checked-in SWT-Bench harness wrapper.

This module is intentionally thin: APEX owns generation, validation,
voting, and prediction emission; the upstream ``swt-bench`` package
owns final scoring (F2P / dual-state verdict).

The wrapper mirrors :mod:`apex.evaluation.runners.testgenevallite` but
shells out to the installed SWT-Bench module rather than the
TestGenEval ``run_evaluation.py`` script. Predictions are git patches
(``model_patch``) rather than raw test source.

Pre-req: ``pip install swt-bench`` on the python interpreter passed via
``--swtbench-python`` (defaults to the same interpreter running this
wrapper). When the package is missing, the wrapper exits cleanly with
an install hint instead of crashing.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from apex.core.docker_pinning import resolve_image
from apex.core.failure_classifier import classify_failure
from apex.core.parallelism import default_task_parallelism
from apex.core.run_manifest import RunManifest
from apex.core.subprocess_retry import DEFAULT_RETRY_ON
from apex.evaluation.checkpointing import atomic_write_json
from apex.evaluation.swtbench_harness import (
    make_swtbench_harness_env,
    resolve_swtbench_module,
)


@dataclass(frozen=True)
class SWTBenchRunConfig:
    __test__ = False

    predictions_jsonl: str
    output_dir: str
    dataset_name: str = "nmuendler/SWT-Bench_Lite_bm25_27k_zsb"
    model_name: str = "apex"
    run_id: str = ""
    # Default to host-CPU/Docker-aware parallelism (capped at 4 — matches
    # ``scripts/launch_*.sh``). Explicit ``--task-parallelism N`` wins.
    task_parallelism: int = field(default_factory=default_task_parallelism)
    timeout_seconds: int = 1800
    skip_existing: bool = True
    docker_namespace: str = "aorwall"
    swtbench_python: str = ""
    task_ids: list[str] = field(default_factory=list)
    extra_args: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_python(config: SWTBenchRunConfig) -> str:
    """Pick the interpreter that has swt-bench installed.

    Resolution order:
      1. Explicit ``--swtbench-python``.
      2. ``APEX_SWTBENCH_PYTHON`` env var.
      3. The interpreter currently running this wrapper (sys.executable).
    """

    if config.swtbench_python:
        return config.swtbench_python
    env_override = os.environ.get("APEX_SWTBENCH_PYTHON", "").strip()
    if env_override:
        return env_override
    return sys.executable


def _swt_bench_installed(python: str) -> tuple[bool, str]:
    """Check whether a supported SWT-Bench entrypoint is importable.

    Returns ``(ok, diagnostic)`` so the caller can surface a clean
    install hint instead of letting the harness blow up halfway through.
    """

    module, diagnostic = resolve_swtbench_module(python)
    return (bool(module), diagnostic)


def build_swtbench_command(config: SWTBenchRunConfig) -> list[str]:
    """Build the command that runs the official SWT-Bench evaluator.

    Mirrors the documented invocation, while accepting both package layouts
    observed from upstream:

        python -m swt_bench.main \\
            --dataset_name <hf-name> \\
            --predictions_path <jsonl> \\
            --filter_swt --max_workers <n> --run_id <id>
    """

    python = _resolve_python(config)
    predictions_path = str(Path(config.predictions_jsonl).expanduser().resolve())
    run_id = config.run_id or f"apex_swtbench_{int(time.time())}"
    swtbench_module, _ = resolve_swtbench_module(python)
    command = [
        python,
        "-m",
        swtbench_module or "swt_bench.main",
        "--dataset_name",
        config.dataset_name,
        "--predictions_path",
        predictions_path,
        "--max_workers",
        str(int(config.task_parallelism)),
        "--run_id",
        run_id,
        "--filter_swt",
        "--namespace",
        config.docker_namespace,
        "--timeout",
        str(int(config.timeout_seconds)),
    ]
    for task_id in config.task_ids:
        command.extend(["--instance_ids", str(task_id)])
    command.extend(config.extra_args)
    return command


def _run_one_attempt(
    *,
    command: list[str],
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    status_path: Path,
    config: SWTBenchRunConfig,
    monitor_interval_seconds: float,
) -> tuple[int, Optional[str]]:
    """Run the swt_bench harness once and stream output to log files.

    Returns ``(returncode, spawn_error)``. ``spawn_error`` is non-None
    only when ``Popen`` itself failed (binary missing, etc.).
    """
    stdout_handle = stdout_path.open("ab", buffering=0)
    stderr_handle = stderr_path.open("ab", buffering=0)
    try:
        try:
            process = subprocess.Popen(
                command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
                env=env,
            )
        except OSError as exc:
            return -1, f"{type(exc).__name__}: {exc}"

        try:
            while process.poll() is None:
                atomic_write_json(
                    status_path,
                    {
                        "status": "running",
                        "pid": process.pid,
                        "command": command,
                        "config": config.to_dict(),
                        "updated_at": time.time(),
                        "stdout_log": str(stdout_path),
                        "stderr_log": str(stderr_path),
                    },
                )
                time.sleep(max(1.0, float(monitor_interval_seconds)))
        except KeyboardInterrupt:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
            raise

        return process.wait(), None
    finally:
        stdout_handle.close()
        stderr_handle.close()


def _read_log_tail(path: Path, max_bytes: int = 32_000) -> str:
    """Read the last *max_bytes* bytes of *path* for failure classification."""
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    if size == 0:
        return ""
    try:
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def run_swtbench(
    config: SWTBenchRunConfig,
    *,
    monitor_interval_seconds: float = 30.0,
    max_attempts: int = 3,
    backoff: str = "exponential",
    manifest: Optional[RunManifest] = None,
) -> dict[str, Any]:
    """Run the upstream swt_bench harness with classification-driven retry.

    On non-zero exit, the stdout/stderr tail is fed through
    :func:`apex.core.failure_classifier.classify_failure`. If the failure
    classifies as env_* or HARNESS_BUG and ``max_attempts > 1``, the run
    is retried with exponential backoff (1s / 2s / 4s). APEX_MISS and
    UNCLASSIFIED failures surface immediately.

    The active :class:`RunManifest` (auto-created when not supplied) is
    populated with the resolved docker namespace so the published
    headline number can be reproduced byte-for-byte.
    """
    output = Path(config.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    # Capture (or accept) a RunManifest so reviewers can reproduce this run.
    if manifest is None:
        manifest = RunManifest.capture(seed=None)

    # Record the docker image namespace via docker_pinning so the manifest
    # carries the digest (when available). SWT-Bench resolves per-instance
    # images dynamically inside the harness, so we pin the namespace itself
    # as a coarse audit anchor.
    namespace_tag = f"{config.docker_namespace}/sweb.eval.x86_64:latest"
    try:
        resolve_image(namespace_tag, record_to_manifest=manifest)
    except Exception:  # pragma: no cover - best-effort
        pass

    python = _resolve_python(config)
    installed, diagnostic = _swt_bench_installed(python)
    if not installed:
        payload = {
            "status": "swt_bench_not_installed",
            "python": python,
            "diagnostic": diagnostic,
            "install_hint": (
                f"Install with: {python} -m pip install swt-bench. "
                "Or pass --swtbench-python pointing at an interpreter "
                "that already has it."
            ),
        }
        atomic_write_json(output / "runner_status.json", payload)
        _write_run_manifest(output, manifest)
        return payload

    command = build_swtbench_command(config)
    status_path = output / "runner_status.json"
    log_dir = output / "swtbench_eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    attempt_diagnostics: list[dict[str, Any]] = []
    attempts_budget = max(1, int(max_attempts))
    final_returncode = -1
    final_spawn_error: Optional[str] = None
    final_classification = None

    started_at = time.time()
    atomic_write_json(
        status_path,
        {
            "status": "running",
            "command": command,
            "config": config.to_dict(),
            "started_at": started_at,
        },
    )

    for attempt_idx in range(1, attempts_budget + 1):
        # Per-attempt log files so retry diagnostics are separable.
        suffix = "" if attempt_idx == 1 else f".attempt{attempt_idx}"
        stdout_path = log_dir / f"swt_bench{suffix}.stdout.log"
        stderr_path = log_dir / f"swt_bench{suffix}.stderr.log"

        attempt_started = time.time()
        returncode, spawn_error = _run_one_attempt(
            command=command,
            env=make_swtbench_harness_env(),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            status_path=status_path,
            config=config,
            monitor_interval_seconds=monitor_interval_seconds,
        )
        attempt_duration = time.time() - attempt_started

        if spawn_error is not None:
            # Spawn failure (binary missing, etc.) is not retryable.
            final_returncode = -1
            final_spawn_error = spawn_error
            attempt_diagnostics.append(
                {
                    "attempt": attempt_idx,
                    "spawn_error": spawn_error,
                    "duration_seconds": round(attempt_duration, 3),
                }
            )
            break

        if returncode == 0:
            attempt_diagnostics.append(
                {
                    "attempt": attempt_idx,
                    "returncode": 0,
                    "duration_seconds": round(attempt_duration, 3),
                }
            )
            final_returncode = 0
            break

        # Classify the failure to decide retry.
        stderr_tail = _read_log_tail(stderr_path)
        stdout_tail = _read_log_tail(stdout_path)
        classification = classify_failure(
            stderr=stderr_tail,
            stdout=stdout_tail,
            returncode=int(returncode),
            context={"phase": "scoring"},
        )
        final_returncode = int(returncode)
        final_classification = classification
        attempt_diagnostics.append(
            {
                "attempt": attempt_idx,
                "returncode": int(returncode),
                "duration_seconds": round(attempt_duration, 3),
                "classification": classification.to_dict(),
            }
        )

        if attempt_idx >= attempts_budget:
            break
        if classification.failure_class not in DEFAULT_RETRY_ON:
            break

        if backoff == "none":
            wait = 0.0
        else:
            wait = float(2 ** (attempt_idx - 1))
        if wait > 0:
            time.sleep(wait)

    # Use the LAST attempt's logs as the canonical pointers in the status payload.
    suffix = "" if len(attempt_diagnostics) <= 1 else f".attempt{len(attempt_diagnostics)}"
    canonical_stdout = log_dir / f"swt_bench{suffix}.stdout.log"
    canonical_stderr = log_dir / f"swt_bench{suffix}.stderr.log"

    if final_spawn_error is not None:
        status_label = "spawn_failed"
    elif final_returncode == 0:
        status_label = "succeeded"
    else:
        status_label = "failed"

    payload: dict[str, Any] = {
        "status": status_label,
        "returncode": final_returncode,
        "command": command,
        "config": config.to_dict(),
        "stdout_log": str(canonical_stdout),
        "stderr_log": str(canonical_stderr),
        "finished_at": time.time(),
        "attempts": attempt_diagnostics,
    }
    if final_spawn_error is not None:
        payload["error"] = final_spawn_error
    if final_classification is not None and final_returncode != 0:
        payload["failure_classification"] = final_classification.to_dict()
        payload["failure_class"] = final_classification.failure_class.value
    atomic_write_json(status_path, payload)
    _write_run_manifest(output, manifest)
    return payload


def _write_run_manifest(output_dir: Path, manifest: RunManifest) -> None:
    """Write the RunManifest to *output_dir/run_manifest.json* atomically."""
    try:
        manifest.write_to(output_dir)
    except Exception:  # pragma: no cover - best-effort
        pass


def _parse_args(argv: list[str]) -> SWTBenchRunConfig:
    parser = argparse.ArgumentParser(
        description="Run the official SWT-Bench harness on a JSONL of predictions."
    )
    parser.add_argument("--predictions-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dataset-name",
        default="nmuendler/SWT-Bench_Lite_bm25_27k_zsb",
    )
    parser.add_argument("--model-name", default="apex")
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--task-parallelism",
        type=int,
        default=0,
        help=(
            "Worker count for parallel docker runs. Defaults to "
            "min(task_count, host_cpu_or_docker_cpu, 4) when 0/unset."
        ),
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--docker-namespace", default="aorwall")
    parser.add_argument("--swtbench-python", default="")
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    kwargs: dict[str, Any] = dict(
        predictions_jsonl=args.predictions_jsonl,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        run_id=args.run_id,
        timeout_seconds=args.timeout_seconds,
        skip_existing=not args.no_skip_existing,
        docker_namespace=args.docker_namespace,
        swtbench_python=args.swtbench_python,
        task_ids=list(args.task_id or []),
        extra_args=list(args.extra_args or []),
    )
    if int(args.task_parallelism) >= 1:
        kwargs["task_parallelism"] = int(args.task_parallelism)
    return SWTBenchRunConfig(**kwargs)


def main(argv: list[str] | None = None) -> int:
    result = run_swtbench(_parse_args(list(argv or sys.argv[1:])))
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
