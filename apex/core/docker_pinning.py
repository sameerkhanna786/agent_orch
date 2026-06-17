"""Docker image digest pinning helper.

Centralizes the resolution of docker image tags (e.g.
``aorwall/sweb.eval.x86_64.django__django-11999:latest``) to a pinned
``sha256`` digest. Pinning is mandatory per the BENCHMARK_FAIRNESS_CHECKLIST:
the tag a benchmark "thinks" it ran against is mutable, so a published
headline number is only reproducible if the manifest records the digest.

Resolution order for a given tag:

  1. ``configs/docker_image_digests.json`` (the *pinned* registry).
  2. ``docker inspect --format '{{index .RepoDigests 0}}' <tag>``.
  3. The bare tag, with a logged warning that the run is unpinned.

The pinned registry is JSON of shape::

    {
      "tag": "sha256:...",
      "tag2": "sha256:...",
      ...
    }

This module is benchmark-agnostic. Each benchmark wires it in via
:func:`resolve_image`, optionally records the resolution into the active
:class:`apex.core.run_manifest.RunManifest`, and uses the resolved
``image_ref`` (digest-pinned when possible) for ``docker run`` / ``docker
pull``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Optional

logger = logging.getLogger("apex.docker_pinning")


# Default location of the pinned digest registry. Can be overridden via the
# APEX_DOCKER_DIGEST_REGISTRY env var (useful for tests).
_DEFAULT_REGISTRY_PATH: Path = (
    Path(__file__).resolve().parents[1] / "configs" / "docker_image_digests.json"
)


# Sentinel digest meaning "we tried docker inspect but it failed".
_DOCKER_INSPECT_FAILED = "docker_inspect_failed"


@dataclass(frozen=True)
class ResolvedImage:
    """Result of :func:`resolve_image`.

    Attributes:
        original_tag: The tag the caller asked to resolve.
        digest: ``sha256:...`` if known; ``None`` if neither the registry
            nor ``docker inspect`` could provide one.
        image_ref: The ref the caller should pass to ``docker run`` /
            ``docker pull``. Equals ``"<repo>@<digest>"`` when
            ``digest`` is set, else equals ``original_tag``.
        source: ``"registry"`` | ``"docker_inspect"`` | ``"unpinned"``.
            For audit / manifest annotation.
    """

    original_tag: str
    digest: Optional[str]
    image_ref: str
    source: str


# ---------------------------------------------------------------------------
# Registry loading (cached)
# ---------------------------------------------------------------------------


_registry_lock = RLock()
_registry_cache: Optional[dict[str, str]] = None
_registry_path_cache: Optional[Path] = None


def _registry_path() -> Path:
    override = os.environ.get("APEX_DOCKER_DIGEST_REGISTRY", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _DEFAULT_REGISTRY_PATH


def _load_registry() -> dict[str, str]:
    global _registry_cache, _registry_path_cache
    with _registry_lock:
        path = _registry_path()
        if _registry_cache is not None and _registry_path_cache == path:
            return _registry_cache
        if not path.exists():
            logger.debug("docker digest registry missing at %s", path)
            _registry_cache = {}
            _registry_path_cache = path
            return _registry_cache
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("failed to read docker digest registry %s: %s", path, exc)
            _registry_cache = {}
            _registry_path_cache = path
            return _registry_cache
        if not isinstance(payload, dict):
            logger.warning(
                "docker digest registry %s is not a JSON object; ignoring",
                path,
            )
            _registry_cache = {}
            _registry_path_cache = path
            return _registry_cache
        cleaned: dict[str, str] = {}
        for key, value in payload.items():
            if isinstance(key, str) and isinstance(value, str) and value:
                cleaned[key] = value
        _registry_cache = cleaned
        _registry_path_cache = path
        return _registry_cache


def reset_registry_cache() -> None:
    """Clear the in-memory registry cache. Used in tests."""
    global _registry_cache, _registry_path_cache
    with _registry_lock:
        _registry_cache = None
        _registry_path_cache = None


# ---------------------------------------------------------------------------
# Docker inspect fallback
# ---------------------------------------------------------------------------


def _docker_inspect_digest(tag: str) -> Optional[str]:
    """Return the sha256 digest for *tag* via ``docker inspect``.

    Returns ``None`` when docker is missing, the image is not present, or
    the call fails. Never raises.
    """
    docker_bin = shutil.which("docker")
    if docker_bin is None:
        logger.debug("docker CLI not available; cannot resolve digest for %s", tag)
        return None
    try:
        result = subprocess.run(
            [
                docker_bin,
                "inspect",
                "--format",
                "{{index .RepoDigests 0}}",
                tag,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("docker inspect failed for %s: %s", tag, exc)
        return None
    if result.returncode != 0:
        logger.debug(
            "docker inspect returned %s for %s: %s",
            result.returncode,
            tag,
            (result.stderr or "").strip(),
        )
        return None
    raw = result.stdout.strip()
    if not raw:
        return None
    # ``RepoDigests[0]`` looks like ``repo@sha256:...``. We want only the
    # ``sha256:...`` portion as the digest, but the full ref is what the
    # caller will pass to docker run.
    if "@" in raw:
        return raw.split("@", 1)[1]
    if raw.startswith("sha256:"):
        return raw
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _split_repo_tag(tag: str) -> tuple[str, str]:
    """Return ``(repo, tag_part)`` for a docker tag.

    For ``foo/bar:1.2`` -> ``("foo/bar", "1.2")``.
    For ``foo/bar`` -> ``("foo/bar", "latest")``.
    Handles registry-prefixed tags (``ghcr.io/foo/bar:1``) correctly by
    splitting only on the LAST ``:`` after the last ``/``.
    """
    last_slash = tag.rfind("/")
    last_colon = tag.rfind(":")
    if last_colon > last_slash and last_colon != -1:
        return tag[:last_colon], tag[last_colon + 1 :]
    return tag, "latest"


def resolve_image(
    tag: str,
    *,
    record_to_manifest: Any = None,
) -> ResolvedImage:
    """Resolve *tag* to a digest-pinned :class:`ResolvedImage`.

    Resolution order:

      1. Pinned registry (``configs/docker_image_digests.json``).
      2. ``docker inspect``.
      3. Bare tag (logged as unpinned).

    Args:
        tag: The docker image tag (e.g.
            ``aorwall/sweb.eval.x86_64.django__django-11999:latest``).
        record_to_manifest: Optional :class:`apex.core.run_manifest.RunManifest`
            instance. When provided, the resolution is recorded via
            ``manifest.docker_images[tag] = digest_or_unpinned``.

    Returns:
        :class:`ResolvedImage`. Always returns; never raises.
    """
    if not tag or not isinstance(tag, str):
        raise ValueError(f"resolve_image: invalid tag {tag!r}")

    # Pre-pinned references like ``python@sha256:abcd...`` must be
    # returned as-is. Without this guard ``_split_repo_tag`` would split
    # the SHA on the colon (since there is no slash to bound it),
    # ``_docker_inspect_digest`` would resolve the digest of that
    # already-pinned ref, and the result would be the malformed
    # ``python@sha256@sha256:abcd...`` reference that ``docker run``
    # rejects with "invalid reference format". Observed during the
    # commit0 V5 smoke run on cachetools where the benchmark layer
    # passes a pre-resolved digest into ``ContainerSupervisor`` which in
    # turn re-calls ``resolve_image`` inside ``__enter__``.
    if "@sha256:" in tag:
        digest_pos = tag.rindex("@sha256:")
        digest = tag[digest_pos + 1 :]  # "sha256:..."
        if digest.startswith("sha256:") and len(digest) > len("sha256:"):
            resolved = ResolvedImage(
                original_tag=tag,
                digest=digest,
                image_ref=tag,
                source="prepinned",
            )
            _maybe_record_manifest(record_to_manifest, tag, digest)
            return resolved

    registry = _load_registry()
    if tag in registry:
        digest = registry[tag]
        # Sentinel ``"unpinned"`` (or empty string) means "tag tracked but
        # digest not yet resolved on the build host". Don't construct a
        # bogus ``repo@unpinned`` ref — fall through to docker_inspect /
        # bare-tag and emit the standard unpinned warning.
        if digest and digest != "unpinned" and digest.startswith("sha256:"):
            repo, _ = _split_repo_tag(tag)
            image_ref = f"{repo}@{digest}"
            resolved = ResolvedImage(
                original_tag=tag,
                digest=digest,
                image_ref=image_ref,
                source="registry",
            )
            _maybe_record_manifest(record_to_manifest, tag, digest)
            return resolved
        # Tag is tracked-but-unpinned; emit the audit signal and fall
        # through to docker inspect.
        logger.info(
            "docker image %s is tracked in the digest registry as 'unpinned' "
            "— attempting docker inspect fallback before declaring the run "
            "unpinned.",
            tag,
        )

    digest = _docker_inspect_digest(tag)
    if digest is not None:
        repo, _ = _split_repo_tag(tag)
        image_ref = f"{repo}@{digest}" if digest.startswith("sha256:") else tag
        resolved = ResolvedImage(
            original_tag=tag,
            digest=digest,
            image_ref=image_ref,
            source="docker_inspect",
        )
        _maybe_record_manifest(record_to_manifest, tag, digest)
        return resolved

    logger.warning(
        "docker image %s is not pinned in the registry and docker inspect "
        "could not resolve a digest; running unpinned. Add it to %s for "
        "reproducible builds.",
        tag,
        _registry_path(),
    )
    resolved = ResolvedImage(
        original_tag=tag,
        digest=None,
        image_ref=tag,
        source="unpinned",
    )
    _maybe_record_manifest(record_to_manifest, tag, "unpinned")
    return resolved


def _maybe_record_manifest(manifest: Any, tag: str, digest: str) -> None:
    """Record (tag -> digest) on the manifest if one was provided.

    Tolerates duck-typed manifest objects: anything with a writable
    ``docker_images`` mapping works. We do NOT call ``add_docker_image``
    because that re-runs ``docker inspect``; we already have the answer.
    """
    if manifest is None:
        return
    images = getattr(manifest, "docker_images", None)
    if images is None or not hasattr(images, "__setitem__"):
        return
    try:
        images[tag] = digest
    except Exception:  # pragma: no cover - defensive
        logger.debug("failed to record digest for %s on manifest", tag)


__all__ = [
    "ResolvedImage",
    "resolve_image",
    "reset_registry_cache",
]
