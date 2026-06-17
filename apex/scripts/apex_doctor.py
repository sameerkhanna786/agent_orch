"""Phase 5.7: Docker container, worktree, and result-dir cleanup audit.

The ``apex doctor`` subcommand exposed in :mod:`apex.cli` delegates to
this module. Three subcommands:

* ``apex doctor scan`` — list ``apex_*`` named docker containers and
  worktree / ``.apex_*`` result directories older than ``--age`` days
  (default 7). Reports total disk usage. Read-only.
* ``apex doctor clean --confirm`` — actually remove them. Without
  ``--confirm`` it stays in dry-run mode.
* ``apex doctor verify-manifest <run_dir>`` — re-parse the run's
  ``run_manifest.json`` and re-run ``docker inspect`` to confirm image
  digests still match the recorded values (catches image drift).

Subprocess invocations are isolated through small helpers that tests
can monkey-patch. The CI integration step in ``apex doctor scan`` is
designed to assert ``leak_count == 0`` after the test suite finishes.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger("apex.scripts.apex_doctor")


# Names matching this prefix are considered apex-owned and eligible for
# scan/clean. We intentionally restrict to ``apex_*`` (and the in-container
# ``.apex_*`` for result dirs) so we never accidentally scrub user
# containers / dirs that share the workspace.
_APEX_CONTAINER_NAME_PREFIXES = ("apex_", "apex-")
_APEX_WORKTREE_DIR_PREFIXES = ("apex_worktree_", ".apex_worktree_")
_APEX_RESULT_DIR_PREFIXES = (".apex_",)


@dataclass
class DockerContainerEntry:
    """One apex-owned docker container surfaced by ``docker ps -a``."""

    container_id: str
    name: str
    created_unix: float
    state: str
    image: str

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - float(self.created_unix or 0.0))


@dataclass
class FilesystemEntry:
    """One on-disk worktree / result directory surfaced by the scan."""

    path: Path
    size_bytes: int
    mtime_unix: float
    kind: str  # "worktree" | "result_dir"

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - float(self.mtime_unix or 0.0))


@dataclass
class DoctorScanReport:
    containers: list[DockerContainerEntry] = field(default_factory=list)
    filesystem: list[FilesystemEntry] = field(default_factory=list)
    total_disk_bytes: int = 0
    age_threshold_seconds: float = 0.0
    docker_available: bool = True
    docker_error: Optional[str] = None

    @property
    def leak_count(self) -> int:
        """How many entries exceed the age threshold (the leak signal)."""
        return sum(1 for c in self.containers if c.age_seconds >= self.age_threshold_seconds) + sum(
            1 for f in self.filesystem if f.age_seconds >= self.age_threshold_seconds
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "docker_available": self.docker_available,
            "docker_error": self.docker_error,
            "age_threshold_seconds": float(self.age_threshold_seconds),
            "total_disk_bytes": int(self.total_disk_bytes),
            "leak_count": self.leak_count,
            "containers": [
                {
                    "container_id": c.container_id,
                    "name": c.name,
                    "image": c.image,
                    "state": c.state,
                    "created_unix": float(c.created_unix),
                    "age_seconds": c.age_seconds,
                }
                for c in self.containers
            ],
            "filesystem": [
                {
                    "path": str(f.path),
                    "size_bytes": int(f.size_bytes),
                    "mtime_unix": float(f.mtime_unix),
                    "age_seconds": f.age_seconds,
                    "kind": f.kind,
                }
                for f in self.filesystem
            ],
        }


@dataclass
class DoctorCleanReport:
    removed_containers: list[str] = field(default_factory=list)
    removed_paths: list[str] = field(default_factory=list)
    failed_containers: list[dict[str, str]] = field(default_factory=list)
    failed_paths: list[dict[str, str]] = field(default_factory=list)
    bytes_reclaimed: int = 0
    confirmed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmed": self.confirmed,
            "removed_containers": list(self.removed_containers),
            "removed_paths": list(self.removed_paths),
            "failed_containers": list(self.failed_containers),
            "failed_paths": list(self.failed_paths),
            "bytes_reclaimed": int(self.bytes_reclaimed),
        }


@dataclass
class ManifestVerifyReport:
    manifest_path: str = ""
    images_checked: list[dict[str, Any]] = field(default_factory=list)
    drift_detected: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "images_checked": list(self.images_checked),
            "drift_detected": self.drift_detected,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Subprocess helpers — isolated so tests can monkey-patch them.
# ---------------------------------------------------------------------------


def _docker_ps(*, runner: Any = subprocess.run) -> tuple[bool, list[dict[str, Any]], Optional[str]]:
    """List all docker containers as JSON-decoded dicts.

    Returns ``(ok, entries, error)``. ``ok=False`` means docker is not
    available; ``entries`` is the list of parsed JSON lines from
    ``docker ps -a --format '{{json .}}'``. The format is line-delimited
    JSON, one container per line.
    """
    if shutil.which("docker") is None:
        return False, [], "docker binary not on PATH"
    try:
        result = runner(
            ["docker", "ps", "-a", "--format", "{{json .}}"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, [], f"{type(exc).__name__}: {exc}"
    if result.returncode != 0:
        return False, [], (result.stderr or "").strip() or f"docker ps exited {result.returncode}"
    entries: list[dict[str, Any]] = []
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            entries.append(parsed)
    return True, entries, None


def _docker_rm(container_id: str, *, runner: Any = subprocess.run) -> tuple[bool, str]:
    """Remove a container by id; return (ok, message)."""
    try:
        result = runner(
            ["docker", "rm", "-f", container_id],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if result.returncode != 0:
        return False, (result.stderr or "").strip() or f"docker rm exited {result.returncode}"
    return True, ""


def _docker_inspect_digest(
    image: str,
    *,
    runner: Any = subprocess.run,
) -> tuple[bool, Optional[str], Optional[str]]:
    """Return ``(ok, digest, error)`` for the named image.

    ``digest`` is the first ``RepoDigests`` entry (``repo@sha256:...``)
    if available, otherwise the image's ``Id`` (``sha256:...``).
    """
    if shutil.which("docker") is None:
        return False, None, "docker binary not on PATH"
    try:
        result = runner(
            ["docker", "inspect", "--format", "{{json .}}", image],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, None, f"{type(exc).__name__}: {exc}"
    if result.returncode != 0:
        return (
            False,
            None,
            (result.stderr or "").strip() or f"docker inspect exited {result.returncode}",
        )
    try:
        payload = json.loads((result.stdout or "").strip())
    except json.JSONDecodeError as exc:
        return False, None, f"failed to parse docker inspect json: {exc}"
    repo_digests = payload.get("RepoDigests") or []
    if isinstance(repo_digests, list) and repo_digests:
        return True, str(repo_digests[0]), None
    image_id = payload.get("Id")
    if image_id:
        return True, str(image_id), None
    return True, None, "image inspected but no digest field present"


def _parse_container_entry(entry: dict[str, Any]) -> Optional[DockerContainerEntry]:
    name = str(entry.get("Names") or entry.get("name") or "").split(",")[0].strip()
    if not name or not name.startswith(_APEX_CONTAINER_NAME_PREFIXES):
        return None
    container_id = str(entry.get("ID") or entry.get("Id") or entry.get("id") or "")
    image = str(entry.get("Image") or "")
    state = str(entry.get("State") or entry.get("Status") or "")
    created_unix = _parse_docker_created(entry)
    return DockerContainerEntry(
        container_id=container_id,
        name=name,
        created_unix=created_unix,
        state=state,
        image=image,
    )


def _parse_docker_created(entry: dict[str, Any]) -> float:
    """Parse the ``CreatedAt`` field from ``docker ps -a`` output.

    Docker's default format is ``2025-01-15 12:34:56 +0000 UTC``. We
    accept either that, the ``Created`` numeric epoch field (preferred
    when present), or fall back to ``time.time()`` (so a missing field
    doesn't make the entry look ancient).
    """
    created_numeric = entry.get("Created")
    if isinstance(created_numeric, (int, float)) and created_numeric > 0:
        return float(created_numeric)
    raw = str(entry.get("CreatedAt") or "").strip()
    if not raw:
        return time.time()
    # Strip trailing timezone label like " UTC" that strptime can't parse.
    cleaned = re.sub(r"\s+UTC\s*$", "", raw)
    for fmt in (
        "%Y-%m-%d %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    ):
        try:
            parsed = time.strptime(cleaned, fmt)
            return time.mktime(parsed)
        except (ValueError, OverflowError):
            continue
    return time.time()


# ---------------------------------------------------------------------------
# Filesystem scan helpers
# ---------------------------------------------------------------------------


def _iter_filesystem_candidates(
    roots: Iterable[Path],
) -> Iterable[FilesystemEntry]:
    """Yield ``FilesystemEntry`` records for apex-owned dirs under ``roots``."""
    for root in roots:
        try:
            root_path = Path(root).expanduser().resolve()
        except OSError:
            continue
        if not root_path.is_dir():
            continue
        try:
            children = list(root_path.iterdir())
        except OSError:
            continue
        for entry in children:
            if not entry.is_dir():
                continue
            name = entry.name
            kind: Optional[str] = None
            if name.startswith(_APEX_WORKTREE_DIR_PREFIXES):
                kind = "worktree"
            elif name.startswith(_APEX_RESULT_DIR_PREFIXES):
                kind = "result_dir"
            if kind is None:
                continue
            try:
                stat = entry.stat()
            except OSError:
                continue
            yield FilesystemEntry(
                path=entry,
                size_bytes=_dir_size_bytes(entry),
                mtime_unix=float(stat.st_mtime),
                kind=kind,
            )


def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        for sub in path.rglob("*"):
            try:
                if sub.is_file():
                    total += sub.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


# ---------------------------------------------------------------------------
# Public API: scan / clean / verify-manifest
# ---------------------------------------------------------------------------


def scan(
    *,
    age_days: float = 7.0,
    filesystem_roots: Optional[list[Path]] = None,
    runner: Any = subprocess.run,
) -> DoctorScanReport:
    """Scan for apex-owned docker containers and on-disk leaks."""
    age_seconds = max(0.0, float(age_days) * 86400.0)
    report = DoctorScanReport(age_threshold_seconds=age_seconds)
    ok, entries, error = _docker_ps(runner=runner)
    report.docker_available = ok
    report.docker_error = error
    if ok:
        for raw in entries:
            parsed = _parse_container_entry(raw)
            if parsed is not None:
                report.containers.append(parsed)
    roots = filesystem_roots or [Path.cwd()]
    for fs_entry in _iter_filesystem_candidates(roots):
        report.filesystem.append(fs_entry)
        report.total_disk_bytes += int(fs_entry.size_bytes)
    return report


def clean(
    *,
    age_days: float = 7.0,
    filesystem_roots: Optional[list[Path]] = None,
    confirm: bool = False,
    runner: Any = subprocess.run,
) -> DoctorCleanReport:
    """Remove apex-owned docker containers and on-disk leaks.

    Without ``confirm=True`` this is a dry-run: it reports what would be
    cleaned but does NOT touch docker or the filesystem. Returned
    ``DoctorCleanReport`` always reflects what *would* (or did) be
    removed.
    """
    report = DoctorCleanReport(confirmed=bool(confirm))
    scan_result = scan(
        age_days=age_days,
        filesystem_roots=filesystem_roots,
        runner=runner,
    )
    age_threshold = scan_result.age_threshold_seconds
    for container in scan_result.containers:
        if container.age_seconds < age_threshold:
            continue
        if not confirm:
            report.removed_containers.append(container.container_id or container.name)
            continue
        ok, error = _docker_rm(container.container_id, runner=runner)
        if ok:
            report.removed_containers.append(container.container_id or container.name)
        else:
            report.failed_containers.append(
                {"id": container.container_id, "name": container.name, "error": error}
            )
    for fs_entry in scan_result.filesystem:
        if fs_entry.age_seconds < age_threshold:
            continue
        if not confirm:
            report.removed_paths.append(str(fs_entry.path))
            report.bytes_reclaimed += int(fs_entry.size_bytes)
            continue
        try:
            shutil.rmtree(fs_entry.path)
            report.removed_paths.append(str(fs_entry.path))
            report.bytes_reclaimed += int(fs_entry.size_bytes)
        except OSError as exc:
            report.failed_paths.append({"path": str(fs_entry.path), "error": str(exc)})
    return report


def verify_manifest(
    run_dir: Path,
    *,
    runner: Any = subprocess.run,
) -> ManifestVerifyReport:
    """Re-run ``docker inspect`` against the digests recorded in the manifest.

    Looks for ``run_manifest.json`` directly inside ``run_dir`` and reads
    the ``docker_images`` mapping. For each ``image_tag -> digest`` entry,
    we re-inspect the local image and compare. Drift is reported with
    the recorded vs. observed digests.
    """
    report = ManifestVerifyReport(manifest_path="")
    run_path = Path(run_dir).expanduser().resolve()
    manifest_path = run_path / "run_manifest.json"
    report.manifest_path = str(manifest_path)
    if not manifest_path.is_file():
        report.error = f"run_manifest.json not found at {manifest_path}"
        return report
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        report.error = f"failed to read manifest: {exc}"
        return report
    docker_images = payload.get("docker_images") or {}
    if not isinstance(docker_images, dict):
        report.error = "docker_images field missing or wrong type"
        return report
    for tag, recorded_digest in sorted(docker_images.items()):
        if not tag:
            continue
        ok, observed, error = _docker_inspect_digest(str(tag), runner=runner)
        entry = {
            "image": str(tag),
            "recorded_digest": str(recorded_digest or ""),
            "observed_digest": str(observed or "") if ok else None,
            "ok": bool(ok),
            "drift": False,
            "error": error if not ok else None,
        }
        if ok and recorded_digest and observed:
            if str(recorded_digest) != str(observed):
                entry["drift"] = True
                report.drift_detected = True
        report.images_checked.append(entry)
    return report


# ---------------------------------------------------------------------------
# Rendering helpers — used by cli.py
# ---------------------------------------------------------------------------


def render_scan(report: DoctorScanReport) -> str:
    lines: list[str] = []
    lines.append(
        f"Docker available: {'yes' if report.docker_available else 'no'}"
        + (f" ({report.docker_error})" if report.docker_error else "")
    )
    lines.append(f"Age threshold: {report.age_threshold_seconds:.0f}s")
    lines.append(f"Total leak disk usage: {report.total_disk_bytes} bytes")
    lines.append(f"Leak count: {report.leak_count}")
    if report.containers:
        lines.append("")
        lines.append(f"apex_* containers ({len(report.containers)}):")
        for c in report.containers:
            lines.append(
                f"  - {c.name} [{c.container_id[:12]}] state={c.state} age={c.age_seconds:.0f}s image={c.image}"
            )
    if report.filesystem:
        lines.append("")
        lines.append(f"apex worktree/result dirs ({len(report.filesystem)}):")
        for f in report.filesystem:
            lines.append(f"  - {f.path} ({f.kind}) age={f.age_seconds:.0f}s size={f.size_bytes}B")
    return "\n".join(lines)


def render_clean(report: DoctorCleanReport) -> str:
    lines = [
        f"Confirmed: {'yes' if report.confirmed else 'no (dry run)'}",
        f"Removed containers: {len(report.removed_containers)}",
        f"Removed paths: {len(report.removed_paths)}",
        f"Bytes reclaimed: {report.bytes_reclaimed}",
        f"Failed containers: {len(report.failed_containers)}",
        f"Failed paths: {len(report.failed_paths)}",
    ]
    return "\n".join(lines)


def render_verify_manifest(report: ManifestVerifyReport) -> str:
    lines = [
        f"Manifest: {report.manifest_path}",
        f"Drift detected: {'yes' if report.drift_detected else 'no'}",
    ]
    if report.error:
        lines.append(f"Error: {report.error}")
    for entry in report.images_checked:
        marker = "DRIFT" if entry.get("drift") else "ok" if entry.get("ok") else "fail"
        lines.append(
            f"  - [{marker}] {entry.get('image')} recorded={entry.get('recorded_digest')!s} observed={entry.get('observed_digest')!s}"
        )
    return "\n".join(lines)


__all__ = [
    "DockerContainerEntry",
    "DoctorCleanReport",
    "DoctorScanReport",
    "FilesystemEntry",
    "ManifestVerifyReport",
    "clean",
    "render_clean",
    "render_scan",
    "render_verify_manifest",
    "scan",
    "verify_manifest",
]
