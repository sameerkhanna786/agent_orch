"""One-shot Python driver execution in a project's docker container.

Used by the V4 W4 oracle-capture path so ``import django.template`` /
``import sympy`` / etc. resolve against the project's installed conda env
instead of our local apex venv (which doesn't have those deps). Each call
spins up a fresh container, mounts the driver source, runs ``conda run -n
<env> python /tmp/apex_driver.py`` with cwd at the in-container repo dir,
and returns the captured stdout / stderr / returncode.

The image / repo-dir / conda-env naming follows the same conventions
``swebench_docker.run_docker.run_docker_evaluation`` uses, so any task that
the official harness can score is also one this runner can probe.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DockerSubprocessResult:
    stdout: str
    stderr: str
    returncode: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_python_in_project_container(
    *,
    task_instance: dict[str, Any],
    driver_source: str,
    namespace: str,
    official_repo: Path,
    log_dir: Path,
    timeout_seconds: int = 60,
    project_mount: Path | None = None,
) -> DockerSubprocessResult:
    """Execute ``driver_source`` inside the docker image for the project.

    The driver runs as ``conda run -n <env> --no-capture-output python
    /tmp/apex_driver.py`` with the working directory set to the in-container
    repo path. The driver script lives in ``log_dir`` so it's host-shareable
    on macOS Docker Desktop (where ``/var/folders`` cannot be bind-mounted
    as a single file).
    """

    import sys as _sys

    repo_str = str(Path(official_repo).expanduser().resolve())
    # Audit H9: insert under try/finally so the path doesn't leak to
    # subsequent task invocations (which may use a different repo).
    _path_inserted = False
    if repo_str not in _sys.path:
        _sys.path.insert(0, repo_str)
        _path_inserted = True
    try:
        from swebench_docker.constants import MAP_VERSION_TO_INSTALL  # noqa: WPS433
    finally:
        if _path_inserted:
            try:
                _sys.path.remove(repo_str)
            except ValueError:  # pragma: no cover - defensive
                pass

    repo = str(task_instance["repo"])
    version = str(task_instance["version"])
    repo_name = repo.replace("/", "_")
    spec = MAP_VERSION_TO_INSTALL[repo][version]

    image = _docker_image_for(task_instance, spec, namespace=namespace)
    repo_dir_in_container = _container_repo_dir(repo, repo_name, version, spec)
    conda_env = f"{repo_name}__{version}"

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix="apex_driver_",
        dir=str(log_dir),
        delete=False,
    ) as fh:
        fh.write(driver_source)
        driver_path = Path(fh.name)

    try:
        # Phase 5.6 security: oracle driver execution is purely a read-only
        # ``import X; X.api()`` probe. It does NOT need network access (no
        # pip installs, no remote fetches), so we run with --network=none to
        # eliminate the data-exfiltration surface that --network host opens
        # up. The long-lived runtime container in commit0_benchmark.py keeps
        # --network host because it might genuinely need pip installs at
        # bootstrap time.
        cmd = [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--platform",
            "linux/amd64",
            "-v",
            f"{driver_path}:/tmp/apex_driver.py:ro",
        ]
        if project_mount is not None:
            cmd.extend(
                [
                    "-v",
                    f"{Path(project_mount).expanduser().resolve()}:{repo_dir_in_container}:rw",
                ]
            )
        cmd.extend(
            [
                "-w",
                repo_dir_in_container,
                image,
            ]
        )
        cmd.extend(
            [
                "conda",
                "run",
                "-n",
                conda_env,
                "--no-capture-output",
                "python",
                "/tmp/apex_driver.py",
            ]
        )
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=float(timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return DockerSubprocessResult(
                stdout="", stderr="docker subprocess timed out", returncode=124
            )
        except OSError as exc:
            return DockerSubprocessResult(
                stdout="",
                stderr=f"{type(exc).__name__}: {exc}",
                returncode=1,
            )
        return DockerSubprocessResult(
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            returncode=int(completed.returncode or 0),
        )
    finally:
        try:
            driver_path.unlink()
        except OSError:
            pass


def export_project_source_from_container(
    *,
    task_instance: dict[str, Any],
    namespace: str,
    official_repo: Path,
    destination: Path,
    timeout_seconds: int = 180,
) -> DockerSubprocessResult:
    """Copy the benchmark image's project checkout into ``destination``.

    Authoring agents need the same source tree the benchmark harness will run.
    This exports the repository directory from the official SWE-bench-style
    image instead of asking agents to reason from Apex's synthetic focal-file
    workdir.
    """

    import sys as _sys

    repo_str = str(Path(official_repo).expanduser().resolve())
    _path_inserted = False
    if repo_str not in _sys.path:
        _sys.path.insert(0, repo_str)
        _path_inserted = True
    try:
        from swebench_docker.constants import MAP_VERSION_TO_INSTALL  # noqa: WPS433
    finally:
        if _path_inserted:
            try:
                _sys.path.remove(repo_str)
            except ValueError:  # pragma: no cover - defensive
                pass

    try:
        repo = str(task_instance["repo"])
        version = str(task_instance["version"])
        repo_name = repo.replace("/", "_")
        spec = MAP_VERSION_TO_INSTALL[repo][version]
    except Exception as exc:
        return DockerSubprocessResult(
            stdout="",
            stderr=f"failed to resolve benchmark install spec: {type(exc).__name__}: {exc}",
            returncode=1,
        )

    image = _docker_image_for(task_instance, spec, namespace=namespace)
    repo_dir_in_container = _container_repo_dir(repo, repo_name, version, spec)
    destination = Path(destination)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_destination = parent / f".{destination.name}.apex_source_export_tmp"
    if tmp_destination.exists():
        shutil.rmtree(tmp_destination, ignore_errors=True)
    tmp_destination.mkdir(parents=True, exist_ok=True)

    container_id = ""
    try:
        create = subprocess.run(
            ["docker", "create", "--platform", "linux/amd64", image],
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
            check=False,
        )
        if create.returncode != 0:
            shutil.rmtree(tmp_destination, ignore_errors=True)
            return DockerSubprocessResult(
                stdout=create.stdout or "",
                stderr=create.stderr or "",
                returncode=int(create.returncode or 1),
            )
        container_id = (create.stdout or "").strip().splitlines()[-1].strip()
        if not container_id:
            shutil.rmtree(tmp_destination, ignore_errors=True)
            return DockerSubprocessResult(
                stdout=create.stdout or "",
                stderr="docker create did not return a container id",
                returncode=1,
            )
        copy = subprocess.run(
            ["docker", "cp", f"{container_id}:{repo_dir_in_container}/.", str(tmp_destination)],
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
            check=False,
        )
        if copy.returncode != 0:
            shutil.rmtree(tmp_destination, ignore_errors=True)
            return DockerSubprocessResult(
                stdout=copy.stdout or "",
                stderr=copy.stderr or "",
                returncode=int(copy.returncode or 1),
            )
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        tmp_destination.rename(destination)
        return DockerSubprocessResult(
            stdout=f"exported {image}:{repo_dir_in_container} to {destination}",
            stderr="",
            returncode=0,
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp_destination, ignore_errors=True)
        return DockerSubprocessResult(
            stdout="",
            stderr="docker source export timed out",
            returncode=124,
        )
    except OSError as exc:
        shutil.rmtree(tmp_destination, ignore_errors=True)
        return DockerSubprocessResult(
            stdout="",
            stderr=f"{type(exc).__name__}: {exc}",
            returncode=1,
        )
    finally:
        if container_id:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )


def _docker_image_for(
    task_instance: dict[str, Any], spec: dict[str, Any], *, namespace: str
) -> str:
    repo_name = str(task_instance["repo"]).replace("/", "_")
    image_prefix = "swe-bench"
    if spec.get("instance_image", False):
        tag = f"{namespace}/{image_prefix}-{repo_name}-instance:{task_instance['instance_id']}"
    else:
        tag = f"{namespace}/{image_prefix}-{repo_name}-testbed:{task_instance['version']}"
    # Phase 1.5: centralize digest pinning + record into the active
    # RunManifest. Best-effort; falls back to the mutable tag only when
    # no registry/inspect digest is available.
    return _record_image_to_active_manifest(tag)


def _record_image_to_active_manifest(tag: str) -> str:
    """Resolve *tag* via docker_pinning and record it into any active
    :class:`apex.core.run_manifest.RunManifest` registered for this
    process. Pure best-effort: return *tag* when the image cannot be
    resolved.
    """
    try:
        from apex.core.docker_pinning import resolve_image
        from apex.evaluation.runners._active_manifest import get_active_manifest
    except Exception:  # pragma: no cover - defensive
        return tag
    manifest = get_active_manifest()
    if tag in getattr(manifest, "docker_images", {}) and manifest.docker_images[tag] not in (
        "",
        None,
        "unpinned",
    ):
        recorded = str(manifest.docker_images[tag] or "")
        repo = tag.rsplit(":", 1)[0]
        return f"{repo}@{recorded}" if recorded.startswith("sha256:") else tag
    try:
        return resolve_image(tag, record_to_manifest=manifest).image_ref
    except Exception:  # pragma: no cover - defensive
        return tag


def _container_repo_dir(repo: str, repo_name: str, version: str, spec: dict[str, Any]) -> str:
    """Mirror swebench_docker/run_docker.py's logic for project source location."""

    if (
        ("packages" in spec and spec["packages"] == "environment.yml")
        or repo == "mwaskom/seaborn"
        or (repo == "sympy/sympy" and _version_at_least(version, "1.6"))
    ):
        return f"/home/swe-bench/{repo_name}"
    return f"/opt/{repo_name}"


def _version_at_least(version: str, minimum: str) -> bool:
    def parse(value: str) -> tuple[int, ...]:
        return tuple(int(part) for part in value.split(".") if part.isdigit())

    return parse(version) >= parse(minimum)
