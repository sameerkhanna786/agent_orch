"""Default-parallelism helpers for benchmark runners.

The runner dataclasses (`TestGenEvalLiteRunConfig`, `SWTBenchRunConfig`,
`TestGenEvalLiteGenerateConfig`) historically defaulted ``task_parallelism``
to ``1``. Operators then had to override to ``--task-parallelism N`` on
every CLI invocation; forgetting collapses smoke and validate runs to
serial execution.

This module exposes:

* :func:`max_concurrent_docker_jobs` — the upper bound we'll allow,
  computed from the host's CPU count and the Docker daemon's reported
  CPU budget (when available), capped at ``DEFAULT_PARALLELISM_CAP``
  (currently 4 — matches the cap operators use in the launch scripts).

* :func:`default_task_parallelism` — the value the runners use as a
  ``field(default_factory=...)`` default. When ``task_count`` is known,
  it returns ``min(task_count, max_concurrent_docker_jobs())``; when
  unknown (config-construction time), it returns
  ``max_concurrent_docker_jobs()`` so a freshly-constructed config is
  parallel out of the box.

* :func:`resolve_task_parallelism` — the helper runners call at run
  time once the task count is known, so a config with the default still
  honours an operator's explicit override.

The helpers degrade gracefully when ``docker`` is unavailable or when
``os.cpu_count()`` returns ``None`` (sandboxed environments). Both
fall back to the conservative parallelism of 2.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_PARALLELISM_CAP = 4
"""Hard ceiling — matches the value baked into ``scripts/launch_*.sh``."""

DEFAULT_PARALLELISM_FLOOR = 2
"""When CPU/Docker probes fail, never collapse to serial unless the caller asks."""


def _detected_cpu_count() -> int:
    """Return ``os.cpu_count()`` (or ``2`` when the host hides it)."""
    cpu = os.cpu_count()
    if cpu is None or cpu < 1:
        return DEFAULT_PARALLELISM_FLOOR
    return cpu


@functools.lru_cache(maxsize=1)
def _docker_cpu_budget() -> Optional[int]:
    """Best-effort CPU budget reported by the Docker daemon.

    Returns ``None`` when ``docker`` is missing, ``docker info`` fails,
    or any other unexpected error fires (e.g. tests have monkey-patched
    ``subprocess.Popen`` with a sentinel that doesn't implement the
    context-manager protocol). Older daemons may not include ``NCPU``
    in the JSON payload — we return ``None`` in that case rather than
    guessing.

    Cached for the life of the process so dataclass-construction
    bursts (one per runner config) don't re-shell-out each time.
    """
    if os.environ.get("APEX_PARALLELISM_DISABLE_DOCKER_PROBE", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return None
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return None
    try:
        completed = subprocess.run(  # noqa: S603 - trusted bin path
            [docker_bin, "info", "--format", "{{json .}}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - intentionally broad
        # ``subprocess.TimeoutExpired``, ``OSError``, AND tests that
        # patch ``Popen`` with a non-context-manager sentinel — silently
        # downgrade to "no docker info" so callers fall back to CPU count.
        logger.debug("docker info probe failed: %s", exc)
        return None
    if completed.returncode != 0:
        return None
    try:
        info = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return None
    ncpu = info.get("NCPU")
    if isinstance(ncpu, int) and ncpu > 0:
        return ncpu
    return None


def max_concurrent_docker_jobs(*, cap: int = DEFAULT_PARALLELISM_CAP) -> int:
    """Return the maximum number of docker jobs we'll launch in parallel.

    Caps at :data:`DEFAULT_PARALLELISM_CAP` (matching the launch scripts'
    historical ``--parallelism 4``). Honours the Docker daemon's
    ``NCPU`` budget when reachable, else falls back to ``os.cpu_count()``.
    """
    if cap < 1:
        cap = 1
    cpu = _detected_cpu_count()
    docker_cpu = _docker_cpu_budget()
    candidate = cpu if docker_cpu is None else min(cpu, docker_cpu)
    candidate = max(DEFAULT_PARALLELISM_FLOOR, candidate)
    return max(1, min(cap, candidate))


def default_task_parallelism(
    *,
    task_count: Optional[int] = None,
    cap: int = DEFAULT_PARALLELISM_CAP,
) -> int:
    """Default value for ``task_parallelism`` runner fields.

    When ``task_count`` is known, never exceed it (no point spawning
    more workers than tasks). When unknown (config-construction time
    before tasks are loaded), return the host's parallelism budget so
    that the freshly-constructed config is parallel.
    """
    budget = max_concurrent_docker_jobs(cap=cap)
    if task_count is None:
        return budget
    if task_count < 1:
        return 1
    return max(1, min(task_count, budget))


def resolve_task_parallelism(
    requested: int,
    *,
    task_count: Optional[int] = None,
    cap: int = DEFAULT_PARALLELISM_CAP,
) -> int:
    """Reconcile a user-supplied ``--task-parallelism`` with the host budget.

    * Negative or zero ``requested`` -> fall back to
      :func:`default_task_parallelism`.
    * Positive ``requested`` -> respect the operator's choice but never
      exceed ``task_count`` when known. We DO NOT silently cap at
      ``cap`` here: an operator passing ``--task-parallelism 16`` on a
      box that can take it must be honoured. The cap is only a default;
      explicit user intent wins.
    """
    if requested is None or int(requested) < 1:
        return default_task_parallelism(task_count=task_count, cap=cap)
    requested_int = int(requested)
    if task_count is not None and task_count > 0:
        return max(1, min(requested_int, task_count))
    return requested_int


__all__ = [
    "DEFAULT_PARALLELISM_CAP",
    "DEFAULT_PARALLELISM_FLOOR",
    "default_task_parallelism",
    "max_concurrent_docker_jobs",
    "resolve_task_parallelism",
]
