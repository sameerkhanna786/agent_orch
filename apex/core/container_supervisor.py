"""Docker container supervisor for the V5 in-container agent loop.

This module is the V2 lift over the V5 ``InContainerAgent``'s V1
``bash -lc cwd=workspace_dir`` shim: it owns the lifecycle of a real
Docker container that the agent operates inside. The agent's
``run_in_container`` tool dispatches through ``docker exec`` instead of
spawning host processes, so the workspace is genuinely isolated.

Design constraints:

- **No external Python docker SDK dependency.** We invoke the ``docker``
  CLI via :mod:`subprocess`. This keeps APEX's runtime deps minimal and
  avoids wedging an extra import path for users who already have
  ``docker`` on their ``PATH``.
- **Network is denied by default** (``--network none``). V5 oracle work
  doesn't need internet; some agentic tasks do, and callers can override.
- **Image refs are digest-pinned** through
  :func:`apex.core.docker_pinning.resolve_image`. The pinned digest is
  recorded into the supplied :class:`apex.core.run_manifest.RunManifest`
  so reviewers can reproduce the run.
- **Cleanup is mandatory.** ``__exit__`` always issues ``docker rm -f``,
  even on a raised exception, so tests (and real runs) never leak
  containers.

Usage::

    from apex.core.container_supervisor import ContainerSupervisor

    with ContainerSupervisor(
        image="aorwall/sweb.eval.x86_64.django__django-11999:latest",
        workspace_dir=Path("/tmp/ws"),
        manifest=manifest,
    ) as supervisor:
        proc = supervisor.run_in_container("pytest -q tests/test_x.py")
        assert proc.returncode == 0

The :class:`InContainerAgent` accepts an optional supervisor via its
``container_supervisor`` constructor argument; when provided, every
``run_in_container`` tool call routes through ``docker exec`` instead of
the V1 host-side bash shim.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from .docker_pinning import ResolvedImage, resolve_image

logger = logging.getLogger("apex.container_supervisor")


_DEFAULT_WORKSPACE_MOUNT = "/workspace"
_DEFAULT_NETWORK: "_NetworkMode" = "none"

# Type alias for the network knob. ``host`` is host-namespace, ``none`` is
# fully isolated, ``bridge`` is the docker default.
_NetworkMode = Literal["host", "none", "bridge"]


class ContainerSupervisorError(RuntimeError):
    """Raised on a hard supervisor failure (container could not be
    created, docker CLI missing, etc.). Recoverable per-command failures
    surface via ``CompletedProcess.returncode`` instead.
    """


@dataclass
class _SupervisorState:
    """Mutable runtime state owned by one :class:`ContainerSupervisor`."""

    container_id: Optional[str] = None
    started: bool = False
    resolved_image: Optional[ResolvedImage] = None
    docker_bin: Optional[str] = None
    last_exec_returncode: Optional[int] = None
    cleanup_attempts: int = 0


@dataclass(frozen=True)
class ContainerSecurityOptions:
    """Typed Docker sandbox knobs; avoids raw `docker run` escape hatches."""

    user: Optional[str] = None
    no_new_privileges: bool = True
    pids_limit: Optional[int] = 512
    cap_drop: tuple[str, ...] = ()
    read_only_rootfs: bool = False
    tmpfs_mounts: tuple[str, ...] = ()
    readonly_mounts: tuple[tuple[Path, str], ...] = ()

    @staticmethod
    def default() -> "ContainerSecurityOptions":
        return ContainerSecurityOptions()


class ContainerSupervisor:
    """Owns the lifecycle of a docker container the V5 agent operates inside.

    Lifecycle:

      * ``__init__``  — record config; do not touch docker.
      * ``__enter__`` — resolve image digest, ``docker run -d`` the
        container with ``sleep infinity`` as the entrypoint, bind-mount
        the workspace at ``/workspace``.
      * ``run_in_container`` — ``docker exec`` arbitrary shell commands
        with a hard timeout. Returns
        :class:`subprocess.CompletedProcess`.
      * ``copy_into`` / ``copy_out`` — ``docker cp`` shims for files the
        agent needs to drop into / pull out of the container.
      * ``__exit__`` — ``docker rm -f`` the container. Always runs, even
        on a raised exception.

    Args:
        image: Docker image tag (e.g.
            ``aorwall/sweb.eval.x86_64.django__django-11999:latest``).
            Resolved through :func:`apex.core.docker_pinning.resolve_image`
            so that the pinned digest is used at ``docker run`` time and
            recorded into the supplied manifest.
        workspace_dir: Host directory that will be bind-mounted at
            ``/workspace`` inside the container (rw).
        name: Optional container name. If ``None``, an auto-generated
            ``apex-v5-<uuid>`` name is used.
        network: Docker network mode. Default ``none`` (no internet).
            Callers that need outbound connectivity (e.g. some package
            installs) can pass ``"host"`` or ``"bridge"``.
        mem_limit: Optional ``--memory`` flag value (e.g. ``"4g"``).
        cpu_limit: Optional ``--cpus`` flag value (e.g. ``"2"``).
        env: Optional environment variables to set inside the container
            via ``-e KEY=VALUE`` flags.
        manifest: Optional :class:`apex.core.run_manifest.RunManifest`
            that receives the resolved image digest.
        docker_bin: Optional explicit path to the ``docker`` CLI. By
            default we resolve via ``shutil.which("docker")``.
        security: Typed Docker security/mount options. Raw ``docker run``
            escape hatches are intentionally rejected.
    """

    def __init__(
        self,
        image: str,
        workspace_dir: Path,
        *,
        name: Optional[str] = None,
        network: _NetworkMode = _DEFAULT_NETWORK,
        mem_limit: Optional[str] = None,
        cpu_limit: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        manifest: Any = None,
        docker_bin: Optional[str] = None,
        security: Optional[ContainerSecurityOptions] = None,
        run_extra_args: Optional[list[str]] = None,
    ) -> None:
        if not image or not isinstance(image, str):
            raise ValueError("image must be a non-empty string")
        if not isinstance(workspace_dir, Path):
            workspace_dir = Path(workspace_dir)
        if not workspace_dir.exists() or not workspace_dir.is_dir():
            raise ValueError(f"workspace_dir must be an existing directory: {workspace_dir}")
        if network not in ("host", "none", "bridge"):
            raise ValueError(f"network must be one of 'host', 'none', 'bridge'; got {network!r}")
        if run_extra_args:
            raise ValueError(
                "run_extra_args is not supported; use typed ContainerSecurityOptions instead"
            )
        self.image = image
        self.workspace_dir = workspace_dir.resolve()
        self.name = name or f"apex-v5-{uuid.uuid4().hex[:12]}"
        self.network: _NetworkMode = network
        self.mem_limit = mem_limit
        self.cpu_limit = cpu_limit
        self.env = dict(env or {})
        self.manifest = manifest
        self._explicit_docker_bin = docker_bin
        self.security = security or ContainerSecurityOptions.default()
        self.run_extra_args: list[str] = []
        self._state = _SupervisorState()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def container_id(self) -> str:
        """The running container id (short form). Empty before ``__enter__``."""
        return self._state.container_id or ""

    @property
    def resolved_image(self) -> Optional[ResolvedImage]:
        """The :class:`ResolvedImage` returned by :func:`resolve_image`."""
        return self._state.resolved_image

    @property
    def started(self) -> bool:
        return self._state.started

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "ContainerSupervisor":
        self._state.docker_bin = self._explicit_docker_bin or shutil.which("docker")
        if not self._state.docker_bin:
            raise ContainerSupervisorError(
                "docker CLI is not available on PATH; cannot launch container"
            )

        # Resolve & pin image digest. ``resolve_image`` records into the
        # manifest if one was supplied. Failures here are non-fatal: we
        # fall through to running with the bare tag (resolve_image already
        # logs the unpinned warning).
        try:
            resolved = resolve_image(self.image, record_to_manifest=self.manifest)
        except ValueError as exc:
            raise ContainerSupervisorError(f"resolve_image rejected {self.image!r}: {exc}") from exc
        self._state.resolved_image = resolved
        image_ref = resolved.image_ref

        run_args = [
            self._state.docker_bin,
            "run",
            "-d",
            "--rm",
            "--name",
            self.name,
            "--network",
            self.network,
            "--mount",
            "type=bind,"
            f"source={self.workspace_dir},"
            f"target={_DEFAULT_WORKSPACE_MOUNT}",
            "-w",
            _DEFAULT_WORKSPACE_MOUNT,
        ]
        if self.security.user:
            run_args.extend(["--user", self.security.user])
        if self.security.no_new_privileges:
            run_args.extend(["--security-opt", "no-new-privileges=true"])
        if self.security.pids_limit is not None:
            run_args.extend(["--pids-limit", str(int(self.security.pids_limit))])
        for capability in self.security.cap_drop:
            run_args.extend(["--cap-drop", str(capability)])
        if self.security.read_only_rootfs:
            run_args.append("--read-only")
        for mount_path in self.security.tmpfs_mounts:
            run_args.extend(["--tmpfs", str(mount_path)])
        for source, target in self.security.readonly_mounts:
            run_args.extend(
                [
                    "--mount",
                    "type=bind,"
                    f"source={Path(source).expanduser().resolve(strict=False)},"
                    f"target={target},readonly",
                ]
            )
        if self.mem_limit:
            run_args.extend(["--memory", str(self.mem_limit)])
        if self.cpu_limit:
            run_args.extend(["--cpus", str(self.cpu_limit)])
        for key, value in self.env.items():
            run_args.extend(["-e", f"{key}={value}"])
        run_args.extend(self.run_extra_args)
        run_args.append(image_ref)
        # The image's CMD might exit immediately; pin it to a sleep loop
        # so the container stays alive across multiple ``docker exec``s.
        # Using ``sh -c "tail -f /dev/null"`` is more portable than
        # ``sleep infinity`` (busybox sleep doesn't accept ``infinity``).
        run_args.extend(["sh", "-c", "tail -f /dev/null"])

        logger.info(
            "ContainerSupervisor: launching container %s from %s (digest_source=%s, network=%s)",
            self.name,
            image_ref,
            resolved.source,
            self.network,
        )
        try:
            result = subprocess.run(
                run_args,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
        except subprocess.SubprocessError as exc:
            raise ContainerSupervisorError(f"docker run failed to start: {exc}") from exc
        if result.returncode != 0:
            raise ContainerSupervisorError(
                "docker run returned non-zero "
                f"(rc={result.returncode}); stderr={result.stderr.strip()!r}"
            )
        container_id = (result.stdout or "").strip()
        if not container_id:
            raise ContainerSupervisorError(
                f"docker run returned no container id; stderr={result.stderr.strip()!r}"
            )
        # Short-form is fine; docker exec accepts both full and short ids
        # as well as the name we set.
        self._state.container_id = container_id[:12]
        self._state.started = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Always attempt cleanup even on exception.
        self._teardown()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_in_container(
        self,
        command: str,
        *,
        timeout: float = 60.0,
        cwd: str = _DEFAULT_WORKSPACE_MOUNT,
        env: Optional[dict[str, str]] = None,
    ) -> subprocess.CompletedProcess:
        """Execute *command* inside the container via ``docker exec``.

        The command is wrapped in ``sh -c`` so the agent can use shell
        operators (``|``, ``&&``, redirections). The container is the
        only allowed namespace; we never fall back to host shell.

        Returns a :class:`subprocess.CompletedProcess` whose
        ``stdout`` / ``stderr`` are captured strings and ``returncode`` is
        the exit status of the command (or ``-9`` on timeout).
        """
        if not self._state.started or not self._state.container_id:
            raise ContainerSupervisorError(
                "run_in_container called before __enter__ (or after __exit__)"
            )
        if not command or not isinstance(command, str):
            raise ValueError("command must be a non-empty string")
        docker_bin = self._state.docker_bin
        if not docker_bin:
            raise ContainerSupervisorError("docker CLI is not available for docker exec")

        exec_args = [
            docker_bin,
            "exec",
            "-w",
            cwd,
        ]
        for key, value in (env or {}).items():
            exec_args.extend(["-e", f"{key}={value}"])
        exec_args.extend([self._state.container_id, "sh", "-c", command])

        try:
            result = subprocess.run(
                exec_args,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            self._state.last_exec_returncode = -9
            return subprocess.CompletedProcess(
                args=exec_args, returncode=-9, stdout=stdout, stderr=stderr
            )
        self._state.last_exec_returncode = int(result.returncode)
        return result

    def copy_into(self, src: Path, dst: str) -> None:
        """``docker cp <src> <container>:<dst>``.

        ``dst`` is interpreted inside the container. ``src`` is a host
        path. Raises :class:`ContainerSupervisorError` on failure.
        """
        if not self._state.started or not self._state.container_id:
            raise ContainerSupervisorError("copy_into called before __enter__")
        docker_bin = self._state.docker_bin
        if not docker_bin:
            raise ContainerSupervisorError("docker CLI is not available for docker cp")
        src_path = Path(src)
        if not src_path.exists():
            raise ValueError(f"copy_into source path does not exist: {src_path}")
        cmd = [
            docker_bin,
            "cp",
            str(src_path),
            f"{self._state.container_id}:{dst}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60)
        if result.returncode != 0:
            raise ContainerSupervisorError(
                f"docker cp host->container failed (rc={result.returncode}): "
                f"{result.stderr.strip()!r}"
            )

    def copy_out(self, src: str, dst: Path) -> None:
        """``docker cp <container>:<src> <dst>``.

        ``src`` is interpreted inside the container. ``dst`` is a host
        path. Raises :class:`ContainerSupervisorError` on failure.
        """
        if not self._state.started or not self._state.container_id:
            raise ContainerSupervisorError("copy_out called before __enter__")
        docker_bin = self._state.docker_bin
        if not docker_bin:
            raise ContainerSupervisorError("docker CLI is not available for docker cp")
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            docker_bin,
            "cp",
            f"{self._state.container_id}:{src}",
            str(dst_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60)
        if result.returncode != 0:
            raise ContainerSupervisorError(
                f"docker cp container->host failed (rc={result.returncode}): "
                f"{result.stderr.strip()!r}"
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _teardown(self) -> None:
        """Best-effort container removal. Idempotent."""
        if not self._state.docker_bin or not self._state.container_id:
            self._state.started = False
            return
        docker_bin = self._state.docker_bin
        cmd = [
            docker_bin,
            "rm",
            "-f",
            self._state.container_id,
        ]
        self._state.cleanup_attempts += 1
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60)
        except subprocess.SubprocessError as exc:
            logger.warning(
                "ContainerSupervisor: docker rm raised on cleanup of %s: %s",
                self._state.container_id,
                exc,
            )
        else:
            if result.returncode != 0:
                # Container already gone is fine; anything else is logged.
                stderr = (result.stderr or "").strip()
                if "no such container" not in stderr.lower():
                    logger.warning(
                        "ContainerSupervisor: docker rm returned rc=%s for %s: %s",
                        result.returncode,
                        self._state.container_id,
                        stderr,
                    )
        self._state.container_id = None
        self._state.started = False


__all__ = [
    "ContainerSupervisor",
    "ContainerSupervisorError",
    "ContainerSecurityOptions",
]
