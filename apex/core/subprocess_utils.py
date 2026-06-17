"""Shared subprocess helpers for verification and benchmark execution."""

from __future__ import annotations

import contextlib
import contextvars
import os
import re
import selectors
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from .pytest_utils import (
    build_pytest_recovery_commands,
    output_indicates_missing_pytest,
    should_disable_pytest_plugin_autoload,
)

_PYTEST_COMMAND_RE = re.compile(r"(^|[^a-z])pytest([^a-z]|$)", re.IGNORECASE)
_PYTEST_PLUGIN_CONFLICT_MARKERS = (
    "pluggyteardownraisedwarning",
    "pytest_cmdline_parse",
    "pytest11",
    "error importing plugin",
    "importerror while importing plugin",
    "cannot import name 'fixturedef' from 'pytest'",
    "pytest_asyncio",
)
_POST_EXIT_PIPE_DRAIN_TIMEOUT_SECONDS = 5.0
_STRUCTURED_COMPLETION_GRACE_SECONDS = 5.0
_STRUCTURED_COMPLETION_PROBE_INTERVAL_SECONDS = 1.0
_PIPE_READ_CHUNK_SIZE = 65536
# Progress-based liveness (K3): cadence at which the per-test-run poll loop
# samples process-tree CPU. A CPU-active test (no stdout, e.g. a long compiled
# computation) refreshes the liveness clock so it is never killed as a stall.
_CPU_PROBE_INTERVAL_SECONDS = 1.0
# Minimum CPU-second delta that counts as forward progress (matches the CLI
# watchdog so busy children on oversubscribed hosts still register).
_CPU_PROGRESS_DELTA_SECONDS = 0.01
StructuredCompletionProbe = Callable[[], Optional[int]]

# Progress-based liveness (K3): the engine sets the active quick-verification
# stall window around a verifier command via ``quick_verification_stall_window``.
# ``run_shell_command`` / ``run_process_command`` default ``stall_window`` from
# this contextvar when the caller did not pass one explicitly — so the verifier
# (which forwards ``timeout=None`` under liveness) gets stall-based liveness
# WITHOUT any change to ``selection/verifier.py``. The contextvar is read on the
# same thread that runs the command, so no cross-thread propagation is needed.
_QV_STALL_WINDOW_CONTEXT: contextvars.ContextVar[Optional[float]] = contextvars.ContextVar(
    "apex_quick_verification_stall_window", default=None
)
# Sentinel so callers can distinguish "no stall_window argument supplied" (and
# therefore fall back to the contextvar) from an explicit ``stall_window=None``.
_STALL_WINDOW_UNSET = object()


@contextlib.contextmanager
def quick_verification_stall_window(stall_window: Optional[float]) -> Iterator[None]:
    """Bind the active per-test-run stall window for the current context.

    The engine wraps a quick-verification command launch in this manager so the
    downstream ``run_shell_command`` poll loop applies a stall-based (not fixed
    wall-clock) liveness window. Fail open: a non-positive / None value clears
    the binding (no stall window) and the call uses any explicitly-passed value.
    """
    value = (
        float(stall_window)
        if isinstance(stall_window, (int, float)) and stall_window > 0
        else None
    )
    token = _QV_STALL_WINDOW_CONTEXT.set(value)
    try:
        yield
    finally:
        _QV_STALL_WINDOW_CONTEXT.reset(token)


def _resolve_stall_window(stall_window: Any) -> Optional[float]:
    """Resolve the effective stall window for a call.

    An explicit ``stall_window`` argument wins; otherwise fall back to the
    active ``quick_verification_stall_window`` contextvar (None when unset).
    """
    if stall_window is _STALL_WINDOW_UNSET:
        ctx_value = _QV_STALL_WINDOW_CONTEXT.get()
        return float(ctx_value) if isinstance(ctx_value, (int, float)) and ctx_value > 0 else None
    if isinstance(stall_window, (int, float)) and stall_window > 0:
        return float(stall_window)
    return None


def _process_tree_cpu_seconds(root_pid: int) -> float:
    """Best-effort cumulative CPU-seconds across a process tree.

    Mirrors ``cli_backend._process_tree_cpu_seconds``. Fails open: any sampling
    error returns 0.0, which (because the caller compares against the previous
    sample) registers as *no* CPU growth — it can only DELAY a kill via a
    spuriously-fresh later sample, never accelerate one.
    """
    try:
        tracked = _collect_process_tree_pids(root_pid)
    except Exception:  # noqa: BLE001 - never let sampling crash the poll loop
        return 0.0
    if not tracked:
        return 0.0
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,time="],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return 0.0
    if result.returncode != 0:
        return 0.0
    total = 0.0
    for line in result.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid not in tracked:
            continue
        total += _parse_ps_cpu_time(parts[1])
    return total


def _parse_ps_cpu_time(token: str) -> float:
    """Parse a ``ps`` TIME column (``[[DD-]HH:]MM:SS[.ss]``) to seconds."""
    token = token.strip()
    if not token:
        return 0.0
    days = 0.0
    if "-" in token:
        day_part, _, token = token.partition("-")
        try:
            days = float(day_part)
        except ValueError:
            days = 0.0
    pieces = token.split(":")
    try:
        nums = [float(piece) for piece in pieces]
    except ValueError:
        return 0.0
    seconds = 0.0
    for value in nums:
        seconds = seconds * 60.0 + value
    return days * 86400.0 + seconds


class TaskProcessRegistry:
    """Best-effort task-scoped registry for subprocess tree cleanup."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pids_by_task: dict[str, set[int]] = {}
        self._metadata_by_task_pid: dict[str, dict[int, dict[str, Any]]] = {}

    def register(
        self,
        task_id: str,
        pid: int,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if not task_id or not isinstance(pid, int) or pid <= 0:
            return
        with self._lock:
            normalized_task_id = str(task_id)
            self._pids_by_task.setdefault(normalized_task_id, set()).add(pid)
            if metadata:
                self._metadata_by_task_pid.setdefault(normalized_task_id, {})[pid] = dict(metadata)
        self.reap()

    def unregister(self, task_id: str, pid: int) -> None:
        if not task_id or not isinstance(pid, int) or pid <= 0:
            return
        with self._lock:
            normalized_task_id = str(task_id)
            pids = self._pids_by_task.get(normalized_task_id)
            if not pids:
                return
            pids.discard(pid)
            if not pids:
                self._pids_by_task.pop(normalized_task_id, None)
            metadata = self._metadata_by_task_pid.get(normalized_task_id)
            if metadata is not None:
                metadata.pop(pid, None)
                if not metadata:
                    self._metadata_by_task_pid.pop(normalized_task_id, None)

    def reap(self) -> None:
        with self._lock:
            for task_id, pids in list(self._pids_by_task.items()):
                live = {pid for pid in pids if _pid_exists(pid)}
                if live:
                    self._pids_by_task[task_id] = live
                    metadata = self._metadata_by_task_pid.get(task_id)
                    if metadata is not None:
                        for pid in list(metadata):
                            if pid not in live:
                                metadata.pop(pid, None)
                        if not metadata:
                            self._metadata_by_task_pid.pop(task_id, None)
                else:
                    self._pids_by_task.pop(task_id, None)
                    self._metadata_by_task_pid.pop(task_id, None)

    def kill(self, task_id: str, *, signum: int = signal.SIGTERM) -> set[int]:
        with self._lock:
            roots = set(self._pids_by_task.get(str(task_id), set()))
            metadata_by_pid = dict(self._metadata_by_task_pid.get(str(task_id), {}))
        tracked: set[int] = set()
        for metadata in metadata_by_pid.values():
            _cleanup_registered_target_runtime(metadata, signum=signum)
        for pid in roots:
            tracked.update(_collect_process_tree_pids(pid) or {pid})
        _signal_process_tree(tracked, signum)
        return tracked


PROCESS_REGISTRY = TaskProcessRegistry()


@dataclass(frozen=True)
class ProcessTelemetry:
    root_pid: int
    pids: list[int] = field(default_factory=list)
    oldest_child_age_seconds: Optional[float] = None
    top_cpu_pid: Optional[int] = None
    top_cpu_percent: Optional[float] = None
    top_cpu_command: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_pid": self.root_pid,
            "pids": list(self.pids),
            "oldest_child_age_seconds": self.oldest_child_age_seconds,
            "top_cpu_pid": self.top_cpu_pid,
            "top_cpu_percent": self.top_cpu_percent,
            "top_cpu_command": self.top_cpu_command,
        }


def terminate_process_tree(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Terminate a subprocess tree and drain output with bounded timeouts."""

    return _terminate_process_tree(process)


def build_command_env(overrides: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Build the environment used for shell commands."""

    from .cli_backend import redact_host_secrets

    env, _removed = redact_host_secrets(os.environ.copy())
    if overrides:
        env.update(overrides)
    # Shell startup hooks can re-source host dotfiles after secret redaction
    # and target-runtime env overrides, so verifier commands remove them.
    for key in ("BASH_ENV", "ENV", "ZDOTDIR"):
        env.pop(key, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_shell_command(
    command: str,
    cwd: str | Path,
    *,
    env: Optional[dict[str, str]] = None,
    timeout: Optional[int] = None,
    task_id: str | None = None,
    completion_probe: Optional[StructuredCompletionProbe] = None,
    completion_grace_seconds: float = _STRUCTURED_COMPLETION_GRACE_SECONDS,
    stall_window: Any = _STALL_WINDOW_UNSET,
) -> subprocess.CompletedProcess[str]:
    """
    Run a shell command with light-weight pytest recovery.

    If pytest startup fails because unrelated host plugins are incompatible with
    the active pytest version, retry once with plugin auto-loading disabled.

    Progress-based liveness (K3): when a stall window is in effect (passed
    explicitly OR via the active ``quick_verification_stall_window`` context)
    the command is killed only after that many seconds with no output AND no CPU
    growth (a deadlock); a CPU-active or output-streaming suite runs to
    completion.
    """

    stall_window = _resolve_stall_window(stall_window)
    merged_env = build_command_env(env)
    active_env = merged_env
    result = _run_once(
        command,
        cwd,
        active_env,
        timeout,
        task_id=task_id,
        completion_probe=completion_probe,
        completion_grace_seconds=completion_grace_seconds,
        stall_window=stall_window,
    )
    if _should_retry_without_pytest_plugin_autoload(command, cwd, result, active_env):
        retry_env = dict(merged_env)
        retry_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        retry_result = _run_once(
            command,
            cwd,
            retry_env,
            timeout,
            task_id=task_id,
            completion_probe=completion_probe,
            completion_grace_seconds=completion_grace_seconds,
            stall_window=stall_window,
        )
        active_env = retry_env
        if retry_result.returncode == 0:
            return retry_result
        result = retry_result

    recovery_result = _retry_with_pytest_recovery(
        command,
        cwd,
        active_env,
        timeout,
        result,
        task_id=task_id,
        completion_probe=completion_probe,
        completion_grace_seconds=completion_grace_seconds,
        stall_window=stall_window,
    )
    if recovery_result is not None:
        return recovery_result
    return result


def run_process_command(
    args: list[str],
    cwd: str | Path | None = None,
    *,
    env: Optional[dict[str, str]] = None,
    timeout: Optional[int] = None,
    task_id: str | None = None,
    stall_window: Any = _STALL_WINDOW_UNSET,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess argv list with timeout cleanup for child processes."""

    stall_window = _resolve_stall_window(stall_window)
    merged_env = build_command_env(env)
    return _run_with_timeout_cleanup(
        args, cwd, merged_env, timeout, task_id=task_id, stall_window=stall_window
    )


def _run_once(
    command: str,
    cwd: str | Path,
    env: dict[str, str],
    timeout: Optional[int],
    *,
    task_id: str | None = None,
    completion_probe: Optional[StructuredCompletionProbe] = None,
    completion_grace_seconds: float = _STRUCTURED_COMPLETION_GRACE_SECONDS,
    stall_window: Optional[float] = None,
) -> subprocess.CompletedProcess[str]:
    return _run_with_timeout_cleanup(
        ["bash", "--noprofile", "--norc", "-c", command],
        cwd,
        env,
        timeout,
        task_id=task_id,
        completion_probe=completion_probe,
        completion_grace_seconds=completion_grace_seconds,
        stall_window=stall_window,
    )


def _run_with_timeout_cleanup(
    args: list[str],
    cwd: str | Path | None,
    env: dict[str, str],
    timeout: Optional[int],
    *,
    task_id: str | None = None,
    completion_probe: Optional[StructuredCompletionProbe] = None,
    completion_grace_seconds: float = _STRUCTURED_COMPLETION_GRACE_SECONDS,
    stall_window: Optional[float] = None,
) -> subprocess.CompletedProcess[str]:
    registry_metadata = _process_registry_metadata(cwd=cwd, env=env)
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        start_new_session=True,
    )
    if task_id:
        PROCESS_REGISTRY.register(task_id, process.pid, metadata=registry_metadata)
    try:
        stdout, stderr, returncode_override = _communicate_with_timeout_and_post_exit_cleanup(
            process,
            args=args,
            timeout=timeout,
            registry_metadata=registry_metadata,
            completion_probe=completion_probe,
            completion_grace_seconds=completion_grace_seconds,
            stall_window=stall_window,
        )
    except subprocess.TimeoutExpired as original_timeout:
        telemetry = collect_process_telemetry(process.pid)
        _cleanup_registered_target_runtime(registry_metadata, signum=signal.SIGTERM)
        stdout_tail, stderr_tail = _terminate_process_tree(process)
        _cleanup_registered_target_runtime(registry_metadata, signum=signal.SIGKILL)
        stdout = _ensure_text(original_timeout.output) + _ensure_text(stdout_tail)
        stderr = _ensure_text(original_timeout.stderr) + _ensure_text(stderr_tail)
        # Surface the actual trip value (stall window when set, else the legacy
        # fixed timeout) so callers/log labels reflect the stall semantics.
        timeout_value = float(getattr(original_timeout, "timeout", 0.0) or 0.0)
        if not timeout_value:
            timeout_value = float(timeout) if timeout is not None else 0.0
        exc = subprocess.TimeoutExpired(args, timeout_value, output=stdout, stderr=stderr)
        exc.apex_process_telemetry = telemetry.to_dict()  # type: ignore[attr-defined]
        raise exc
    finally:
        if task_id:
            PROCESS_REGISTRY.unregister(task_id, process.pid)

    return subprocess.CompletedProcess(
        args=args,
        returncode=(
            int(returncode_override)
            if returncode_override is not None
            else int(process.returncode or 0)
        ),
        stdout=stdout,
        stderr=stderr,
    )


def _communicate_with_timeout_and_post_exit_cleanup(
    process: subprocess.Popen[bytes],
    *,
    args: list[str],
    timeout: Optional[int],
    registry_metadata: dict[str, Any],
    completion_probe: Optional[StructuredCompletionProbe] = None,
    completion_grace_seconds: float = _STRUCTURED_COMPLETION_GRACE_SECONDS,
    stall_window: Optional[float] = None,
) -> tuple[str, str, Optional[int]]:
    """Read subprocess output without hanging on pipe-owning descendants.

    Some benchmark commands finish and write their structured report, but
    subprocesses spawned by the test suite keep stdout/stderr file descriptors
    open. ``Popen.communicate(timeout=None)`` then waits forever even though the
    root command is done. Some verifiers also write an authoritative structured
    report before their process tree exits. These cleanup paths fire only after
    the root has exited or a structured completion probe stays complete for a
    short grace period.

    Progress-based liveness (K3): when ``stall_window`` is set, the per-test-run
    process is killed only after ``stall_window`` of *no meaningful progress*
    (no new output AND no process-tree CPU growth). A CPU-active or
    output-streaming suite never trips; a deadlocked test (0 output + 0 CPU)
    trips after the window. The legacy fixed ``timeout`` (``deadline``) kill is
    retained only as a fallback when ``stall_window`` is None.
    """

    if process.stdout is None or process.stderr is None:
        stdout, stderr = process.communicate(timeout=timeout)
        return _ensure_text(stdout), _ensure_text(stderr), None

    selector = selectors.DefaultSelector()
    chunks: dict[str, list[bytes]] = {"stdout": [], "stderr": []}
    streams: dict[int, str] = {}
    for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
        fd = stream.fileno()
        os.set_blocking(fd, False)
        selector.register(fd, selectors.EVENT_READ, data=name)
        streams[fd] = name

    stall_window_seconds = (
        float(stall_window)
        if isinstance(stall_window, (int, float)) and stall_window > 0
        else None
    )
    deadline = (
        time.monotonic() + float(timeout)
        if timeout is not None and stall_window_seconds is None
        else None
    )
    start_monotonic = time.monotonic()
    last_output_at = start_monotonic
    last_cpu_at = start_monotonic
    last_cpu_seconds = _process_tree_cpu_seconds(process.pid) if stall_window_seconds else 0.0
    next_cpu_probe_at = start_monotonic + _CPU_PROBE_INTERVAL_SECONDS
    exited_at: Optional[float] = None
    structured_completed_at: Optional[float] = None
    structured_completion_returncode: Optional[int] = None
    next_completion_probe_at = 0.0
    post_exit_pipe_drain_timed_out = False
    structured_completion_timed_out = False
    hard_timeout_raised = False
    try:
        while streams or process.poll() is None:
            now = time.monotonic()
            # K3 stall kill: a still-running process with no new output AND no
            # CPU growth for the whole window is a deadlock. CPU is sampled on
            # the existing probe cadence; ``last_output_at`` is bumped whenever
            # ``os.read`` returns bytes (below). Fail open: a failed CPU sample
            # returns 0.0 (no growth) — it can only DELAY the kill via a later
            # spuriously-fresh sample, never accelerate it.
            if stall_window_seconds is not None and process.poll() is None:
                if now >= next_cpu_probe_at:
                    next_cpu_probe_at = now + _CPU_PROBE_INTERVAL_SECONDS
                    current_cpu_seconds = _process_tree_cpu_seconds(process.pid)
                    if current_cpu_seconds > last_cpu_seconds + _CPU_PROGRESS_DELTA_SECONDS:
                        last_cpu_at = now
                    last_cpu_seconds = current_cpu_seconds
                if (now - max(last_output_at, last_cpu_at)) >= stall_window_seconds:
                    hard_timeout_raised = True
                    raise subprocess.TimeoutExpired(
                        args,
                        float(stall_window_seconds),
                        output=_decode_chunks(chunks["stdout"]),
                        stderr=_decode_chunks(chunks["stderr"]),
                    )
            if deadline is not None and now >= deadline:
                hard_timeout_raised = True
                raise subprocess.TimeoutExpired(
                    args,
                    float(timeout),
                    output=_decode_chunks(chunks["stdout"]),
                    stderr=_decode_chunks(chunks["stderr"]),
                )

            returncode = process.poll()
            if returncode is not None and exited_at is None:
                exited_at = now
            if (
                returncode is None
                and completion_probe is not None
                and now >= next_completion_probe_at
            ):
                next_completion_probe_at = now + _STRUCTURED_COMPLETION_PROBE_INTERVAL_SECONDS
                probed_returncode = _call_structured_completion_probe(completion_probe)
                if probed_returncode is None:
                    structured_completed_at = None
                    structured_completion_returncode = None
                elif structured_completed_at is None:
                    structured_completed_at = now
                    structured_completion_returncode = int(probed_returncode)
                elif now - structured_completed_at >= max(
                    0.0,
                    float(completion_grace_seconds),
                ):
                    structured_completion_timed_out = True
                    break
            if exited_at is not None and now - exited_at >= _POST_EXIT_PIPE_DRAIN_TIMEOUT_SECONDS:
                post_exit_pipe_drain_timed_out = True
                break

            wait_timeout = 0.1
            if deadline is not None:
                wait_timeout = min(wait_timeout, max(0.0, deadline - now))
            if stall_window_seconds is not None:
                # Keep the loop waking on the CPU-probe cadence so a stall is
                # detected promptly even when no output ever arrives.
                wait_timeout = min(wait_timeout, max(0.0, next_cpu_probe_at - now))
            if exited_at is not None:
                remaining = _POST_EXIT_PIPE_DRAIN_TIMEOUT_SECONDS - (now - exited_at)
                wait_timeout = min(wait_timeout, max(0.0, remaining))
            if structured_completed_at is not None:
                remaining = max(0.0, float(completion_grace_seconds)) - (
                    now - structured_completed_at
                )
                wait_timeout = min(wait_timeout, max(0.0, remaining))

            if not streams:
                time.sleep(wait_timeout)
                continue

            events = selector.select(wait_timeout)
            if not events:
                continue
            for key, _mask in events:
                fd = int(key.fd)
                name = str(key.data)
                try:
                    data = os.read(fd, _PIPE_READ_CHUNK_SIZE)
                except BlockingIOError:
                    continue
                except OSError:
                    data = b""
                if data:
                    chunks[name].append(data)
                    # K3: new output bytes refresh the liveness clock (S1).
                    last_output_at = time.monotonic()
                    continue
                selector.unregister(fd)
                streams.pop(fd, None)

        if structured_completion_timed_out:
            _cleanup_lingering_post_exit_processes(
                process,
                registry_metadata,
                signum=signal.SIGTERM,
            )
            _drain_available_registered_streams(selector, streams, chunks)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                _cleanup_lingering_post_exit_processes(
                    process,
                    registry_metadata,
                    signum=signal.SIGKILL,
                )
                _drain_available_registered_streams(selector, streams, chunks)
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            chunks["stderr"].append(
                b"\nAPEX subprocess cleanup: structured completion was observed "
                b"but the process did not exit; cleaned up lingering processes.\n"
            )
        elif post_exit_pipe_drain_timed_out:
            _cleanup_lingering_post_exit_processes(
                process,
                registry_metadata,
                signum=signal.SIGTERM,
            )
            _drain_available_registered_streams(selector, streams, chunks)
            _cleanup_lingering_post_exit_processes(
                process,
                registry_metadata,
                signum=signal.SIGKILL,
            )
            _drain_available_registered_streams(selector, streams, chunks)
            chunks["stderr"].append(
                b"\nAPEX subprocess cleanup: root command exited but descendant "
                b"processes kept output pipes open; cleaned up lingering processes.\n"
            )
    finally:
        selector.close()
        if not hard_timeout_raised:
            for stream in (process.stdout, process.stderr):
                try:
                    stream.close()
                except OSError:
                    pass

    return (
        _decode_chunks(chunks["stdout"]),
        _decode_chunks(chunks["stderr"]),
        structured_completion_returncode if structured_completion_timed_out else None,
    )


def _call_structured_completion_probe(
    completion_probe: StructuredCompletionProbe,
) -> Optional[int]:
    try:
        returncode = completion_probe()
    except Exception:
        return None
    if isinstance(returncode, bool) or returncode is None:
        return None
    try:
        return int(returncode)
    except (TypeError, ValueError):
        return None


def _drain_available_registered_streams(
    selector: selectors.BaseSelector,
    streams: dict[int, str],
    chunks: dict[str, list[bytes]],
) -> None:
    deadline = time.monotonic() + 1.0
    while streams and time.monotonic() < deadline:
        events = selector.select(0.05)
        if not events:
            continue
        for key, _mask in events:
            fd = int(key.fd)
            name = str(key.data)
            try:
                data = os.read(fd, _PIPE_READ_CHUNK_SIZE)
            except BlockingIOError:
                continue
            except OSError:
                data = b""
            if data:
                chunks[name].append(data)
                continue
            selector.unregister(fd)
            streams.pop(fd, None)


def _decode_chunks(chunks: list[bytes]) -> str:
    return b"".join(chunks).decode("utf-8", errors="replace")


def _ensure_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _cleanup_lingering_post_exit_processes(
    process: subprocess.Popen[Any],
    registry_metadata: dict[str, Any],
    *,
    signum: int,
) -> None:
    _cleanup_registered_target_runtime(registry_metadata, signum=signum)
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass
    except PermissionError:
        pass
    except OSError:
        pass


def _process_registry_metadata(
    *,
    cwd: str | Path | None,
    env: dict[str, str],
) -> dict[str, Any]:
    target_env_keys = (
        "APEX_TARGET_TOOL_CONTEXT",
        "APEX_TARGET_TOOL_WORKDIR",
        "APEX_AGENT_CONTAINER",
    )
    target_env = {
        key: str(env.get(key) or "") for key in target_env_keys if str(env.get(key) or "").strip()
    }
    metadata: dict[str, Any] = {}
    if cwd is not None:
        metadata["cwd"] = str(cwd)
    if target_env:
        metadata["env"] = target_env
    return metadata


def _cleanup_registered_target_runtime(
    metadata: dict[str, Any],
    *,
    signum: int,
) -> None:
    env = metadata.get("env") if isinstance(metadata, dict) else None
    if not isinstance(env, dict) or not env.get("APEX_TARGET_TOOL_CONTEXT"):
        return
    try:
        from apex.evaluation.target_runtime import cleanup_target_runtime_processes

        cleanup_target_runtime_processes(env, signum=signum)
    except Exception:
        return


def _terminate_process_tree(process: subprocess.Popen[Any]) -> tuple[str, str]:
    tracked_pids = _collect_process_tree_pids(process.pid)
    _signal_process_tree(tracked_pids, signal.SIGTERM)

    try:
        stdout, stderr = process.communicate(timeout=5)
        return _ensure_text(stdout), _ensure_text(stderr)
    except subprocess.TimeoutExpired:
        tracked_pids.update(_collect_process_tree_pids(process.pid))
        _signal_process_tree(tracked_pids, signal.SIGKILL)
        try:
            stdout, stderr = process.communicate(timeout=2)
            return _ensure_text(stdout), _ensure_text(stderr)
        except subprocess.TimeoutExpired:
            for stream in (process.stdout, process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            return (
                _ensure_text(stdout),
                _ensure_text(stderr) + "\nbounded post-kill drain timed out",
            )


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _collect_process_tree_pids(root_pid: int) -> set[int]:
    tracked = {root_pid}
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return tracked

    children_by_parent: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        children_by_parent.setdefault(ppid, []).append(pid)

    stack = [root_pid]
    while stack:
        parent = stack.pop()
        for child in children_by_parent.get(parent, []):
            if child in tracked:
                continue
            tracked.add(child)
            stack.append(child)
    return tracked


def collect_process_telemetry(root_pid: int) -> ProcessTelemetry:
    tracked = _collect_process_tree_pids(root_pid)
    if not tracked:
        return ProcessTelemetry(root_pid=root_pid)
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,etimes=,pcpu=,command="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ProcessTelemetry(root_pid=root_pid, pids=sorted(tracked))

    oldest_age: Optional[float] = None
    top_cpu_pid: Optional[int] = None
    top_cpu_percent: Optional[float] = None
    top_cpu_command = ""
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        try:
            pid = int(parts[0])
            age = float(parts[2])
            cpu = float(parts[3])
        except ValueError:
            continue
        if pid not in tracked:
            continue
        if oldest_age is None or age > oldest_age:
            oldest_age = age
        if top_cpu_percent is None or cpu > top_cpu_percent:
            top_cpu_pid = pid
            top_cpu_percent = cpu
            top_cpu_command = parts[4]
    return ProcessTelemetry(
        root_pid=root_pid,
        pids=sorted(tracked),
        oldest_child_age_seconds=oldest_age,
        top_cpu_pid=top_cpu_pid,
        top_cpu_percent=top_cpu_percent,
        top_cpu_command=top_cpu_command,
    )


def _signal_process_tree(pids: set[int], signum: int) -> None:
    if not pids:
        return

    pgids: set[int] = set()
    for pid in pids:
        try:
            pgids.add(os.getpgid(pid))
        except ProcessLookupError:
            continue

    if hasattr(os, "killpg"):
        for pgid in pgids:
            try:
                os.killpg(pgid, signum)
            except ProcessLookupError:
                continue

    for pid in sorted(pids, reverse=True):
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            continue


def _should_retry_without_pytest_plugin_autoload(
    command: str,
    cwd: str | Path,
    result: subprocess.CompletedProcess[str],
    env: dict[str, str],
) -> bool:
    if result.returncode == 0:
        return False
    if env.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
        return False
    if not _PYTEST_COMMAND_RE.search(command):
        return False
    if not should_disable_pytest_plugin_autoload(command, repo_root=cwd):
        return False

    output = ((result.stdout or "") + (result.stderr or "")).lower()
    return any(marker in output for marker in _PYTEST_PLUGIN_CONFLICT_MARKERS)


def _retry_with_pytest_recovery(
    command: str,
    cwd: str | Path,
    env: dict[str, str],
    timeout: Optional[int],
    result: subprocess.CompletedProcess[str],
    *,
    task_id: str | None = None,
    completion_probe: Optional[StructuredCompletionProbe] = None,
    completion_grace_seconds: float = _STRUCTURED_COMPLETION_GRACE_SECONDS,
    stall_window: Optional[float] = None,
) -> Optional[subprocess.CompletedProcess[str]]:
    if result.returncode == 0:
        return None
    if not _PYTEST_COMMAND_RE.search(command):
        return None
    output = (result.stdout or "") + (result.stderr or "")
    if not output_indicates_missing_pytest(output):
        return None

    attempted = False
    latest = result
    original = command.strip()
    for recovery_command in build_pytest_recovery_commands(command, repo_root=cwd):
        if recovery_command.strip() == original:
            continue
        attempted = True
        candidate_result = _run_once(
            recovery_command,
            cwd,
            env,
            timeout,
            task_id=task_id,
            completion_probe=completion_probe,
            completion_grace_seconds=completion_grace_seconds,
            stall_window=stall_window,
        )
        latest = candidate_result
        candidate_output = (candidate_result.stdout or "") + (candidate_result.stderr or "")
        if candidate_result.returncode == 0 or not output_indicates_missing_pytest(
            candidate_output
        ):
            return candidate_result
    return latest if attempted else None
