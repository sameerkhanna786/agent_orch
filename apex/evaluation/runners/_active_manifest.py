"""Process-global registry for the active :class:`RunManifest`.

A benchmark runner sets the active manifest at run start via
:func:`set_active_manifest` and clears it at the end. Code paths deep
in the call stack — for instance
:func:`apex.evaluation.docker_subprocess_runner._docker_image_for` —
read the active manifest via :func:`get_active_manifest` and record
docker image digests into it without having to thread a manifest
parameter through every helper.

This is intentionally a process-global rather than a contextvar: APEX's
benchmark runners spawn subprocess workers and we only ever care about
the parent-process manifest (subprocess workers run independently and
emit their own provenance).

Use :func:`active_manifest` as a context manager to scope the
registration safely.
"""

from __future__ import annotations

from contextlib import contextmanager
from threading import RLock
from typing import Any, Iterator, Optional

_lock = RLock()
_active: Optional[Any] = None


def set_active_manifest(manifest: Any) -> None:
    """Register *manifest* as the active per-process manifest.

    Pass ``None`` to clear. Replaces any previously-active manifest
    without warning — callers that want exclusive scoping should use
    :func:`active_manifest` instead.
    """
    global _active
    with _lock:
        _active = manifest


def get_active_manifest() -> Optional[Any]:
    """Return the currently-active manifest or ``None`` if unset."""
    with _lock:
        return _active


@contextmanager
def active_manifest(manifest: Any) -> Iterator[Any]:
    """Context manager: register *manifest* for the body, then clear.

    Restores the previous active manifest on exit (so nested runners
    don't accidentally leak state).
    """
    global _active
    with _lock:
        previous = _active
        _active = manifest
    try:
        yield manifest
    finally:
        with _lock:
            _active = previous


__all__ = [
    "active_manifest",
    "get_active_manifest",
    "set_active_manifest",
]
