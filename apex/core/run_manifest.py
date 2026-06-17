"""
Per-run reproducibility manifest for APEX benchmark runs.

The :class:`RunManifest` is a JSON-serialisable record of every fact a
downstream reviewer needs in order to reproduce a published headline number:

* APEX git SHA (and whether the working tree was dirty)
* Python and platform versions
* Resolved :class:`apex.core.config.ApexConfig` as JSON
* Every ``APEX_*`` environment variable, with secrets redacted
* Model id / alias mappings (filled in by callers as they invoke models)
* Docker image -> sha256 digest mappings (filled in by callers)
* Upstream harness versions (swebench, swt_bench, testgeneval,
  swebench_docker) with git-SHA detection for editable installs

This module is intentionally side-effect free at import time and degrades
gracefully when:

* the working directory is not a git repo
* the ``docker`` CLI is unavailable
* an upstream harness is not installed

Wire-up into the actual benchmark runners is performed in a later phase.
"""

from __future__ import annotations

import json
import logging
import os
import platform as _platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("apex.run_manifest")

# Substrings that mark an env var value as secret-bearing. Matched
# case-insensitively against the *name* of the env var.
_SECRET_NAME_MARKERS: tuple[str, ...] = ("KEY", "TOKEN", "SECRET")

# Sentinel value written in place of a redacted secret so reviewers can see
# that the variable was set without leaking its contents.
_REDACTED: str = "<redacted>"

# Sentinel value written for docker images we could not resolve.
_DOCKER_UNAVAILABLE: str = "unavailable"

# Default location of the APEX repo root. Used by :meth:`RunManifest.capture`
# to resolve the git SHA when no explicit path is given.
_DEFAULT_APEX_REPO: Path = Path(__file__).resolve().parents[2]

# Names of upstream harness distributions we know how to detect.
_UPSTREAM_HARNESS_NAMES: tuple[str, ...] = (
    "swebench",
    "swt_bench",
    "testgeneval",
    "swebench_docker",
)


def _run_git(args: list[str], cwd: Path) -> Optional[str]:
    """Run ``git`` with *args* in *cwd* and return ``stdout`` stripped.

    Returns ``None`` if git is missing, the command fails, or *cwd* is not a
    git work tree. Never raises.
    """
    git_bin = shutil.which("git")
    if git_bin is None:
        return None
    try:
        result = subprocess.run(
            [git_bin, *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("git %s failed in %s: %s", " ".join(args), cwd, exc)
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _detect_apex_git_state(repo_path: Path) -> tuple[str, bool]:
    """Return ``(sha, dirty)`` for the APEX repo at *repo_path*.

    On any failure (missing git, not a repo, etc.) returns
    ``("unknown", False)`` so manifest capture never raises.
    """
    sha = _run_git(["rev-parse", "HEAD"], repo_path)
    if not sha:
        return ("unknown", False)
    porcelain = _run_git(["status", "--porcelain"], repo_path)
    dirty = bool(porcelain)
    return (sha, dirty)


def _detect_platform() -> str:
    """Return a short platform identifier such as ``darwin-arm64``."""
    system = _platform.system().lower()
    machine = _platform.machine().lower()
    if not system:
        system = sys.platform
    if not machine:
        machine = "unknown"
    return f"{system}-{machine}"


def _is_secret_name(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in _SECRET_NAME_MARKERS)


def _capture_apex_env_vars(env: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Return every ``APEX_*`` env var, redacting secret-looking ones."""
    source = os.environ if env is None else env
    captured: dict[str, str] = {}
    for name, value in source.items():
        if not name.startswith("APEX_"):
            continue
        if _is_secret_name(name):
            captured[name] = _REDACTED
        else:
            captured[name] = value
    return dict(sorted(captured.items()))


def _safe_apex_config_dict(apex_config: Any) -> dict[str, Any]:
    """Best-effort conversion of an ``ApexConfig``-like object to a dict.

    Accepts:

    * an object with a ``to_dict()`` method (e.g. ``apex.core.config.ApexConfig``)
    * a plain ``dict``
    * ``None`` (returns ``{}``)
    """
    if apex_config is None:
        return {}
    if isinstance(apex_config, dict):
        return dict(apex_config)
    to_dict = getattr(apex_config, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ApexConfig.to_dict() failed: %s", exc)
            return {"__to_dict_failed__": str(exc)}
        if isinstance(payload, dict):
            return payload
    # Fall back to a string repr so we never lose the reference entirely.
    return {"__repr__": repr(apex_config)}


def _docker_repo_digest(tag: str) -> str:
    """Return the sha256 digest for *tag* via ``docker inspect``.

    Returns :data:`_DOCKER_UNAVAILABLE` when docker is missing, the image is
    not present, or the call fails. Never raises.
    """
    docker_bin = shutil.which("docker")
    if docker_bin is None:
        logger.warning("docker CLI not available; cannot resolve digest for %s", tag)
        return _DOCKER_UNAVAILABLE
    try:
        result = subprocess.run(
            [docker_bin, "inspect", "--format", "{{index .RepoDigests 0}}", tag],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("docker inspect failed for %s: %s", tag, exc)
        return _DOCKER_UNAVAILABLE
    if result.returncode != 0:
        logger.warning(
            "docker inspect returned %s for %s: %s",
            result.returncode,
            tag,
            (result.stderr or "").strip(),
        )
        return _DOCKER_UNAVAILABLE
    digest = result.stdout.strip()
    if not digest:
        logger.warning("docker inspect produced empty digest for %s", tag)
        return _DOCKER_UNAVAILABLE
    return digest


def _editable_install_location(distribution: Any) -> Optional[Path]:
    """Return the source directory of *distribution* if installed editable.

    Reads PEP 610 ``direct_url.json`` metadata. Returns ``None`` if the
    distribution is not an editable install or the metadata is missing /
    malformed.
    """
    try:
        raw = distribution.read_text("direct_url.json")
    except Exception:
        return None
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    dir_info = payload.get("dir_info") or {}
    if not dir_info.get("editable"):
        return None
    url = payload.get("url", "")
    if not url.startswith("file://"):
        return None
    path = Path(url[len("file://") :])
    if not path.exists():
        return None
    return path


def _detect_one_harness_version(name: str) -> Optional[str]:
    """Return the version string for *name*, or ``None`` if not installed.

    Editable installs report ``editable@<git-sha>`` (or ``editable@unknown``
    when the source dir is not a git repo). Standard installs report the
    PEP 440 version string.
    """
    try:
        import importlib.metadata as _md
    except ImportError:  # pragma: no cover - Python < 3.8
        return None
    try:
        dist = _md.distribution(name)
    except _md.PackageNotFoundError:
        return None
    editable_path = _editable_install_location(dist)
    if editable_path is not None:
        sha = _run_git(["rev-parse", "HEAD"], editable_path)
        return f"editable@{sha or 'unknown'}"
    try:
        return dist.version
    except Exception:  # pragma: no cover - defensive
        return None


def detect_upstream_harness_versions() -> dict[str, str]:
    """Return a mapping of known upstream harness names to version strings.

    Harnesses that are not installed are omitted. Editable installs report
    ``editable@<git-sha>``; pip-installed harnesses report the PEP 440
    version.
    """
    found: dict[str, str] = {}
    for name in _UPSTREAM_HARNESS_NAMES:
        version = _detect_one_harness_version(name)
        if version is not None:
            found[name] = version
    return found


@dataclass
class RunManifest:
    """Per-run reproducibility manifest.

    Use :meth:`capture` to build one at the start of a run, then mutate it
    via :meth:`add_docker_image`, :meth:`add_model`, and
    :meth:`add_upstream_harness` as the run progresses. Persist with
    :meth:`write_to`; reload with :meth:`read_from`.
    """

    apex_git_sha: str
    apex_git_dirty: bool
    python_version: str
    platform: str
    started_at: str
    seed: Optional[int]
    apex_config: dict[str, Any] = field(default_factory=dict)
    apex_env_vars: dict[str, str] = field(default_factory=dict)
    model_versions: dict[str, str] = field(default_factory=dict)
    docker_images: dict[str, str] = field(default_factory=dict)
    upstream_harness_versions: dict[str, str] = field(default_factory=dict)
    additional_metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def capture(
        cls,
        *,
        apex_config: Any = None,
        seed: Optional[int] = None,
        apex_repo_path: Optional[Path] = None,
        env: Optional[dict[str, str]] = None,
        additional_metadata: Optional[dict[str, Any]] = None,
    ) -> "RunManifest":
        """Snapshot everything we can detect locally.

        ``docker_images``, ``model_versions``, and
        ``upstream_harness_versions`` are intentionally left empty -
        callers populate them via :meth:`add_docker_image`,
        :meth:`add_model`, and :meth:`add_upstream_harness` (or by
        copying from :func:`detect_upstream_harness_versions`).
        """
        repo = Path(apex_repo_path) if apex_repo_path else _DEFAULT_APEX_REPO
        sha, dirty = _detect_apex_git_state(repo)
        return cls(
            apex_git_sha=sha,
            apex_git_dirty=dirty,
            python_version=sys.version.split()[0],
            platform=_detect_platform(),
            started_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            seed=seed,
            apex_config=_safe_apex_config_dict(apex_config),
            apex_env_vars=_capture_apex_env_vars(env),
            model_versions={},
            docker_images={},
            upstream_harness_versions={},
            additional_metadata=dict(additional_metadata or {}),
        )

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def add_docker_image(self, tag: str) -> str:
        """Resolve and record the sha256 digest for *tag*.

        Returns the recorded value (digest or :data:`_DOCKER_UNAVAILABLE`).
        Safe to call when the docker CLI is unavailable - the entry is
        recorded as ``"unavailable"`` and a warning is logged.
        """
        digest = _docker_repo_digest(tag)
        self.docker_images[tag] = digest
        return digest

    def add_model(self, alias: str, model_id: str) -> None:
        """Record the resolved *model_id* used for a logical *alias*."""
        self.model_versions[alias] = model_id

    def add_upstream_harness(self, name: str, version: str) -> None:
        """Record the version of an upstream harness used by this run.

        For editable installs, callers are encouraged to record
        ``editable@<git-sha>`` - :func:`detect_upstream_harness_versions`
        does this automatically.
        """
        self.upstream_harness_versions[name] = version

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the manifest."""
        return {
            "apex_git_sha": self.apex_git_sha,
            "apex_git_dirty": self.apex_git_dirty,
            "python_version": self.python_version,
            "platform": self.platform,
            "started_at": self.started_at,
            "seed": self.seed,
            "apex_config": self.apex_config,
            "apex_env_vars": dict(self.apex_env_vars),
            "model_versions": dict(self.model_versions),
            "docker_images": dict(self.docker_images),
            "upstream_harness_versions": dict(self.upstream_harness_versions),
            "additional_metadata": dict(self.additional_metadata),
        }

    def write_to(self, path: Path) -> Path:
        """Atomically write the manifest as JSON.

        If *path* is a directory (or does not exist but has no ``.json``
        suffix), the file is written as ``run_manifest.json`` inside it.
        Returns the absolute path actually written.
        """
        target = Path(path)
        if target.is_dir():
            target = target / "run_manifest.json"
        elif target.suffix == "" and not target.exists():
            # Treat extension-less, non-existing paths as directories.
            target.mkdir(parents=True, exist_ok=True)
            target = target / "run_manifest.json"
        target.parent.mkdir(parents=True, exist_ok=True)

        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)
        # Atomic write: tmp file in the same directory + os.replace.
        fd, tmp_name = tempfile.mkstemp(
            prefix=".run_manifest-",
            suffix=".json.tmp",
            dir=str(target.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, target)
        except Exception:
            # Best-effort cleanup on failure.
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        return target.resolve()

    @classmethod
    def read_from(cls, path: Path) -> "RunManifest":
        """Load a manifest previously written by :meth:`write_to`.

        If *path* is a directory, looks for ``run_manifest.json`` inside.
        """
        target = Path(path)
        if target.is_dir():
            target = target / "run_manifest.json"
        with open(target, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return cls(
            apex_git_sha=payload.get("apex_git_sha", "unknown"),
            apex_git_dirty=bool(payload.get("apex_git_dirty", False)),
            python_version=payload.get("python_version", ""),
            platform=payload.get("platform", ""),
            started_at=payload.get("started_at", ""),
            seed=payload.get("seed"),
            apex_config=dict(payload.get("apex_config", {}) or {}),
            apex_env_vars=dict(payload.get("apex_env_vars", {}) or {}),
            model_versions=dict(payload.get("model_versions", {}) or {}),
            docker_images=dict(payload.get("docker_images", {}) or {}),
            upstream_harness_versions=dict(payload.get("upstream_harness_versions", {}) or {}),
            additional_metadata=dict(payload.get("additional_metadata", {}) or {}),
        )


__all__ = [
    "RunManifest",
    "detect_upstream_harness_versions",
]
