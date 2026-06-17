"""
Shared benchmark checkpoint helpers.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

TASK_RESULT_FILENAME = "task_result.json"
RUN_STATE_FILENAME = "benchmark_state.json"


def atomic_write_text(path: str | Path, content: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    return destination


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write ``payload`` as JSON to ``path`` atomically, scrubbing
    circular references and unserializable types so a single bad
    diagnostic never aborts the write. Cycles are replaced with the
    sentinel ``"<circular>"``; non-JSON-serializable leaves with
    ``repr(value)``."""

    return atomic_write_text(
        path,
        json.dumps(_safe_jsonable(payload), indent=2, default=repr),
    )


def _safe_jsonable(value: Any, _seen: Optional[set[int]] = None) -> Any:
    """Return a JSON-safe copy of ``value`` with cycles replaced by
    a string marker. Detects cycles via id() tracking on the active
    descent path; non-recursive references to the same shared dict /
    list are preserved (only true cycles become the marker)."""

    if _seen is None:
        _seen = set()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    obj_id = id(value)
    if obj_id in _seen:
        return "<circular>"
    if isinstance(value, dict):
        _seen.add(obj_id)
        try:
            return {str(k): _safe_jsonable(v, _seen) for k, v in value.items()}
        finally:
            _seen.discard(obj_id)
    if isinstance(value, (list, tuple, set, frozenset)):
        _seen.add(obj_id)
        try:
            return [_safe_jsonable(v, _seen) for v in value]
        finally:
            _seen.discard(obj_id)
    # Everything else (Path, dataclass instance, datetime, ...): let
    # ``default=repr`` in atomic_write_json handle it.
    return value


def load_json_if_exists(path: str | Path) -> Optional[dict[str, Any]]:
    candidate = Path(path)
    if not candidate.exists():
        return None
    try:
        payload = json.loads(candidate.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


_TERMINAL_CHECKPOINT_STATUSES = {
    "completed",
    "failed",
    "skipped",
    "timeout",
    "error",
}


def validate_task_checkpoint_payload(payload: Any) -> tuple[bool, str]:
    if not isinstance(payload, dict) or not payload:
        return False, "not_object"
    task_id = (
        payload.get("instance_id")
        or payload.get("task_id")
        or payload.get("task_name")
        or payload.get("id")
    )
    if not str(task_id or "").strip():
        return False, "missing_task_id"
    if (
        "success" not in payload
        and "final_tests_passed" not in payload
        and "skipped" not in payload
    ):
        return False, "missing_terminal_result"
    status = str(payload.get("status") or payload.get("terminal_state") or "").lower()
    if status and status not in _TERMINAL_CHECKPOINT_STATUSES:
        return False, f"non_terminal_status:{status}"
    return True, "ok"


def load_valid_task_checkpoint(task_output_dir: str | Path) -> Optional[dict[str, Any]]:
    payload = load_json_if_exists(task_result_path(task_output_dir))
    valid, _ = validate_task_checkpoint_payload(payload)
    return payload if valid else None


def task_result_path(task_output_dir: str | Path) -> Path:
    return Path(task_output_dir) / TASK_RESULT_FILENAME


def has_task_checkpoint(task_output_dir: str | Path) -> bool:
    return load_valid_task_checkpoint(task_output_dir) is not None


def reset_incomplete_directory(directory: str | Path) -> Path:
    target = Path(directory)
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    return target


def ensure_clean_directory_for_task(
    directory: str | Path,
    *,
    completed: bool,
) -> Path:
    target = Path(directory)
    if completed:
        target.mkdir(parents=True, exist_ok=True)
        return target
    return reset_incomplete_directory(target)


def write_task_checkpoint(task_output_dir: str | Path, payload: dict[str, Any]) -> Path:
    output_dir = Path(task_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return atomic_write_json(task_result_path(output_dir), payload)


def build_run_state(
    *,
    report_kind: str,
    harness_name: str,
    harness_version: str,
    started_at: float,
    requested_task_ids: list[str],
    completed_task_ids: list[str],
    successful_tasks: Optional[int] = None,
    failed_tasks: Optional[int] = None,
    completed: Optional[bool] = None,
    current_task_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    total_tasks = len(requested_task_ids)
    completed_tasks = len(completed_task_ids)
    remaining_tasks = max(0, total_tasks - completed_tasks)
    metadata_payload = dict(metadata or {})
    active_task_ids = [
        str(task_id)
        for task_id in (metadata_payload.get("active_task_ids") or [])
        if str(task_id or "").strip()
    ]
    active_tasks = metadata_payload.get("active_tasks") or {}
    if not isinstance(active_tasks, dict):
        active_tasks = {}
    if bool(completed):
        current_task_id = None
        active_task_ids = []
        active_tasks = {}
    elif current_task_id is None:
        completed_lookup = {task_id for task_id in completed_task_ids}
        current_task_id = next(
            (task_id for task_id in requested_task_ids if task_id not in completed_lookup),
            None,
        )
    payload = {
        "version": 1,
        "report_kind": report_kind,
        "harness_name": harness_name,
        "harness_version": harness_version,
        "started_at": started_at,
        "updated_at": time.time(),
        "requested_task_ids": list(requested_task_ids),
        "completed_task_ids": list(completed_task_ids),
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "successful_tasks": (int(successful_tasks) if successful_tasks is not None else None),
        "failed_tasks": (int(failed_tasks) if failed_tasks is not None else None),
        "remaining_tasks": remaining_tasks,
        "completed_task_count": completed_tasks,
        "failed_task_count": (int(failed_tasks) if failed_tasks is not None else None),
        "active_task_ids": active_task_ids,
        "active_tasks": active_tasks,
        "queued_task_count": max(0, remaining_tasks - len(active_task_ids)),
        "cancelled_task_count": int(metadata_payload.get("cancelled_task_count") or 0),
        "status": (
            "completed"
            if bool(completed) or (total_tasks > 0 and completed_tasks >= total_tasks)
            else "in_progress"
        ),
        "current_task": current_task_id,
    }
    if metadata_payload:
        metadata_payload["active_task_ids"] = active_task_ids
        metadata_payload["active_tasks"] = active_tasks
        payload["metadata"] = metadata_payload
    return payload
