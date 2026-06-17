"""Run-level process and container ownership helpers.

The full benchmark runner still owns task scheduling, but shared supervisor
helpers keep resource ownership explicit and testable.  In particular Docker
containers launched for target runtimes must be labeled with run metadata so a
stop/cleanup path can distinguish Apex-owned resources from unrelated user
containers.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

APEX_DOCKER_LABEL_PREFIX = "apex."
APEX_RUN_ID_LABEL = "apex.run_id"
APEX_TASK_ID_LABEL = "apex.task_id"
APEX_OWNER_PID_LABEL = "apex.owner_pid"
APEX_BENCHMARK_LABEL = "apex.benchmark"
APEX_CREATED_AT_LABEL = "apex.created_at"


@dataclass(frozen=True)
class ApexDockerResource:
    """A Docker container that appears to be owned by Apex."""

    container_id: str
    name: str
    labels: dict[str, str] = field(default_factory=dict)
    status: str = ""

    @property
    def run_id(self) -> str:
        return self.labels.get(APEX_RUN_ID_LABEL, "")

    @property
    def benchmark(self) -> str:
        return self.labels.get(APEX_BENCHMARK_LABEL, "")

    @property
    def is_apex_owned(self) -> bool:
        return self.name.startswith("apex-") or any(
            key.startswith(APEX_DOCKER_LABEL_PREFIX) for key in self.labels
        )


def apex_docker_labels(
    *,
    run_id: str,
    task_id: Optional[str] = None,
    benchmark: Optional[str] = None,
    owner_pid: Optional[int] = None,
    created_at: Optional[float] = None,
) -> dict[str, str]:
    """Return normalized labels for an Apex-owned Docker resource."""

    labels = {
        APEX_RUN_ID_LABEL: str(run_id or "unknown"),
        APEX_OWNER_PID_LABEL: str(owner_pid if owner_pid is not None else os.getpid()),
        APEX_CREATED_AT_LABEL: str(created_at if created_at is not None else time.time()),
    }
    if task_id:
        labels[APEX_TASK_ID_LABEL] = str(task_id)
    if benchmark:
        labels[APEX_BENCHMARK_LABEL] = str(benchmark)
    return labels


def docker_label_args(labels: dict[str, str]) -> list[str]:
    """Render Docker ``--label`` arguments in stable order."""

    args: list[str] = []
    for key in sorted(labels):
        value = labels[key]
        if not key or value is None:
            continue
        args.extend(["--label", f"{key}={value}"])
    return args


def parse_docker_label_string(label_text: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for raw_part in str(label_text or "").split(","):
        part = raw_part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        if key:
            labels[key] = value.strip()
    return labels


def _docker_ps_rows(
    *,
    docker_env: Optional[dict[str, str]] = None,
    filters: Optional[list[str]] = None,
) -> list[str]:
    command = [
        "docker",
        "ps",
        "-a",
        "--format",
        "{{.ID}}\t{{.Names}}\t{{.Labels}}\t{{.Status}}",
    ]
    for item in filters or []:
        command.extend(["--filter", item])
    result = subprocess.run(
        command,
        env=docker_env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def list_apex_docker_containers(
    *,
    docker_env: Optional[dict[str, str]] = None,
) -> list[ApexDockerResource]:
    """List Docker containers that are labeled or named as Apex resources."""

    rows = []
    rows.extend(_docker_ps_rows(docker_env=docker_env, filters=[f"label={APEX_RUN_ID_LABEL}"]))
    rows.extend(_docker_ps_rows(docker_env=docker_env, filters=["name=apex-"]))
    resources: dict[str, ApexDockerResource] = {}
    for row in rows:
        parts = row.split("\t", 3)
        if len(parts) < 4:
            continue
        container_id, name, labels_text, status = parts
        resource = ApexDockerResource(
            container_id=container_id.strip(),
            name=name.strip(),
            labels=parse_docker_label_string(labels_text),
            status=status.strip(),
        )
        if resource.container_id and resource.is_apex_owned:
            resources[resource.container_id] = resource
    return sorted(resources.values(), key=lambda item: (item.run_id, item.name, item.container_id))


def cleanup_apex_docker_containers(
    *,
    run_id: Optional[str] = None,
    benchmark: Optional[str] = None,
    dry_run: bool = False,
    docker_env: Optional[dict[str, str]] = None,
) -> list[ApexDockerResource]:
    """Remove Apex-owned Docker containers, optionally scoped to a run.

    When ``run_id`` is provided, only labeled containers for that run are
    removed.  Unlabeled legacy ``apex-*`` containers are removed only by an
    unscoped cleanup call.
    """

    selected: list[ApexDockerResource] = []
    for resource in list_apex_docker_containers(docker_env=docker_env):
        if run_id is not None and resource.run_id != str(run_id):
            continue
        if run_id is not None and not resource.run_id:
            continue
        if benchmark is not None and resource.benchmark != str(benchmark):
            continue
        selected.append(resource)
    if dry_run or not selected:
        return selected
    subprocess.run(
        ["docker", "rm", "-f", *[resource.container_id for resource in selected]],
        env=docker_env,
        capture_output=True,
        text=True,
        check=False,
    )
    return selected


@dataclass
class RunSupervisor:
    """Small durable manifest for run-owned resources.

    The benchmark-specific scheduler can embed this class without moving all
    execution into a new abstraction.  It records enough state for cleanup
    tools and interrupted-run diagnostics.
    """

    run_id: str
    manifest_path: Path
    benchmark: str = ""
    parent_pid: int = field(default_factory=os.getpid)
    child_pids: set[int] = field(default_factory=set)
    containers: dict[str, dict[str, Any]] = field(default_factory=dict)
    cancel_requested: bool = False
    cancel_reason: str = ""

    def register_child(self, pid: int) -> None:
        if pid > 0:
            self.child_pids.add(int(pid))
            self.write_manifest()

    def register_container(self, name: str, *, task_id: Optional[str] = None) -> None:
        if name:
            self.containers[name] = {
                "name": name,
                "task_id": task_id or "",
                "registered_at": time.time(),
            }
            self.write_manifest()

    def request_cancel(self, reason: str = "") -> None:
        self.cancel_requested = True
        self.cancel_reason = reason
        self.write_manifest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "benchmark": self.benchmark,
            "parent_pid": self.parent_pid,
            "child_pids": sorted(self.child_pids),
            "containers": dict(self.containers),
            "cancel_requested": self.cancel_requested,
            "cancel_reason": self.cancel_reason,
            "updated_at": time.time(),
        }

    def write_manifest(self) -> None:
        from apex.evaluation.checkpointing import atomic_write_json

        atomic_write_json(self.manifest_path, self.to_dict())
