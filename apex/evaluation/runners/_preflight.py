"""Pre-flight checks for long-running test-generation benchmark runs."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreflightCheck:
    __test__ = False

    name: str
    status: str
    detail: str = ""
    required: bool = True

    @property
    def passed(self) -> bool:
        return self.status in {"pass", "warn"} or not self.required

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TestgenPreflightReport:
    __test__ = False

    run_dir: str
    checks: list[PreflightCheck] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "passed": self.passed,
            "started_at": self.started_at,
            "checks": [check.to_dict() for check in self.checks],
        }


def pre_flight_for_testgen(
    run_dir: str | Path,
    *,
    required_docker_free_gb: float = 80.0,
    kill_orphans: bool = False,
    prune: bool = False,
    check_network: bool = True,
    require_docker: bool = True,
    require_hf_token: bool = True,
    require_swebench_docker_fork_dir: bool = True,
    docker_platform: str | None = None,
) -> TestgenPreflightReport:
    """Run deterministic operator checks before a TestGenEval-style run."""

    root = Path(run_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    checks: list[PreflightCheck] = []
    checks.extend(_command_checks(require_docker=require_docker))
    checks.append(_writable_check(root))
    checks.append(_disk_check(root, required_gb=required_docker_free_gb))
    checks.append(_docker_check(prune=prune, required=require_docker))
    checks.append(
        _swebench_docker_fork_dir_check(
            required=require_swebench_docker_fork_dir,
        )
    )
    checks.append(
        _docker_bind_mount_probe(
            root,
            required=require_docker,
            docker_platform=docker_platform,
        )
    )
    checks.extend(_python_dependency_checks())
    checks.append(_hf_token_check(required=require_hf_token))
    if check_network:
        checks.append(_network_check("huggingface.co", 443))
    checks.append(_orphan_process_check(kill=kill_orphans))
    return TestgenPreflightReport(run_dir=str(root), checks=checks)


def _command_checks(*, require_docker: bool) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    for command in ("git", "python3", "docker"):
        available = shutil.which(command) is not None
        checks.append(
            PreflightCheck(
                name=f"command:{command}",
                status="pass" if available else "fail",
                detail=shutil.which(command) or "not found on PATH",
                required=command != "docker" or require_docker,
            )
        )
    return checks


def _writable_check(root: Path) -> PreflightCheck:
    probe = root / ".apex_preflight_write_probe"
    try:
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        return PreflightCheck("run_dir_writable", "fail", str(exc))
    return PreflightCheck("run_dir_writable", "pass", str(root))


def _disk_check(root: Path, *, required_gb: float) -> PreflightCheck:
    usage = shutil.disk_usage(root)
    free_gb = usage.free / (1024**3)
    status = "pass" if free_gb >= required_gb else "fail"
    return PreflightCheck(
        "disk_free",
        status,
        f"{free_gb:.1f} GB free; required {required_gb:.1f} GB",
    )


def _docker_check(*, prune: bool, required: bool) -> PreflightCheck:
    if shutil.which("docker") is None:
        return PreflightCheck("docker", "fail", "docker not found on PATH", required=required)
    if prune:
        # Audit H2: ``docker system prune`` can hang indefinitely on a
        # stuck daemon; cap so the whole preflight isn't blocked.
        try:
            subprocess.run(
                ["docker", "system", "prune", "-f"],
                check=False,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return PreflightCheck(
                "docker",
                "fail",
                "docker system prune timed out (daemon stuck?)",
                required=required,
            )
    probe = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if probe.returncode != 0:
        return PreflightCheck(
            "docker",
            "fail",
            (probe.stderr or probe.stdout or "docker info failed").strip()[-4000:],
            required=required,
        )
    return PreflightCheck("docker", "pass", probe.stdout.strip(), required=required)


def _swebench_docker_fork_dir_check(*, required: bool) -> PreflightCheck:
    value = os.environ.get("SWEBENCH_DOCKER_FORK_DIR", "").strip()
    if value:
        return PreflightCheck(
            "swebench_docker_fork_dir",
            "pass",
            value,
            required=required,
        )
    return PreflightCheck(
        "swebench_docker_fork_dir",
        "fail",
        "SWEBENCH_DOCKER_FORK_DIR not set; resume scripts must source the original environment",
        required=required,
    )


def _docker_bind_mount_probe(
    root: Path,
    *,
    required: bool,
    docker_platform: str | None = None,
) -> PreflightCheck:
    if shutil.which("docker") is None:
        return PreflightCheck(
            "docker_bind_mount_probe",
            "fail",
            "docker not found on PATH",
            required=required,
        )
    probe_dir = root / ".apex_docker_mount_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    (probe_dir / "probe.txt").write_text("ok\n", encoding="utf-8")
    try:
        command = ["docker", "run", "--rm"]
        if docker_platform:
            command.extend(["--platform", str(docker_platform)])
        command.extend(
            [
                "-v",
                f"{probe_dir}:/probe:rw",
                "alpine:latest",
                "cat",
                "/probe/probe.txt",
            ]
        )
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return PreflightCheck(
            "docker_bind_mount_probe",
            "fail",
            f"{type(exc).__name__}: {exc}",
            required=required,
        )
    if completed.returncode != 0:
        return PreflightCheck(
            "docker_bind_mount_probe",
            "fail",
            (completed.stderr or completed.stdout or "probe failed").strip()[-1000:],
            required=required,
        )
    return PreflightCheck(
        "docker_bind_mount_probe",
        "pass",
        completed.stdout.strip(),
        required=required,
    )


def _python_dependency_checks() -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    for module in ("dotenv", "requests"):
        probe = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        checks.append(
            PreflightCheck(
                name=f"python_import:{module}",
                status="pass" if probe.returncode == 0 else "fail",
                detail=(probe.stderr or probe.stdout or "").strip()[-1000:],
                required=True,
            )
        )
    return checks


def _hf_token_check(*, required: bool) -> PreflightCheck:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return PreflightCheck("hf_token", "pass", "token present", required=required)
    return PreflightCheck("hf_token", "fail", "HF token not detected", required=required)


def _network_check(host: str, port: int) -> PreflightCheck:
    try:
        with socket.create_connection((host, port), timeout=5.0):
            return PreflightCheck("network:huggingface", "pass", f"{host}:{port}", required=False)
    except OSError as exc:
        return PreflightCheck("network:huggingface", "warn", str(exc), required=False)


def _orphan_process_check(*, kill: bool) -> PreflightCheck:
    probe = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if probe.returncode != 0:
        return PreflightCheck("orphan_test_processes", "warn", "ps failed", required=False)
    current_pid = os.getpid()
    matches: list[str] = []
    for line in probe.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        command_lower = command.lower()
        if any(token in command_lower for token in ("runtests.py", "pytest", "cosmic-ray")):
            matches.append(stripped)
            if kill:
                try:
                    os.kill(pid, 15)
                except OSError:
                    pass
    if matches:
        status = "warn" if kill else "fail"
        suffix = "sent SIGTERM" if kill else "rerun with kill_orphans=True to terminate"
        return PreflightCheck(
            "orphan_test_processes",
            status,
            "\n".join(matches[:20]) + f"\n{suffix}",
            required=not kill,
        )
    return PreflightCheck("orphan_test_processes", "pass", "none detected")
