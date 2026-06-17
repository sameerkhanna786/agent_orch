"""Docker-based BenchmarkAdapter for TestGenEvalLite.

Invokes the official ``swebench_docker.run_docker.run_docker_evaluation``
function with ``skip_mutation=True`` so the W1 final-acceptance gate has a
real validation surface for projects that need their own conda env (Django,
sympy, Flask, ...). Local pytest in the apex venv can't import these
projects' deps, so the local default adapter returns ``harness_error`` and
the gate has no signal. The docker adapter hands the artifact to the same
docker image the official harness will use for scoring; the per-test pass/
fail extracted from the resulting eval log is the ground-truth signal we
need.

Cost: ~30-60s per ``run_unfiltered`` call. With max 5 gate iterations per
task and parallelism 4, this adds ~5 min per parallel slot. Acceptable
inside the existing 300s generation budget.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

from .benchmark_adapters import BenchmarkAdapter
from .final_acceptance_gate import FinalAcceptanceRun, GeneratedArtifact
from .splice_simulator import SpliceMode

# Audit H10: serialize the brief env-mutation window in ``run_unfiltered``
# so two concurrent threads don't trample each other's
# SWEBENCH_DOCKER_FORK_DIR setting. The lock is held only across the
# narrow asyncio.run window — not across the docker subprocess itself.
_DOCKER_ADAPTER_ENV_LOCK = threading.Lock()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DockerTaskContext:
    """Per-task docker context shared across W4 oracle capture, W7 gap-fill,
    and W1 final acceptance — so each can invoke the project's container
    instead of falling back to the local apex venv (which lacks the project's
    deps for Django/sympy/Flask)."""

    task_instance: dict[str, Any] = field(default_factory=dict)
    namespace: str = "kdjain"
    official_repo: Optional[Path] = None
    log_dir: Optional[Path] = None
    adapter: Optional[BenchmarkAdapter] = None


_current_docker_context: contextvars.ContextVar[Optional[DockerTaskContext]] = (
    contextvars.ContextVar("apex_docker_task_context", default=None)
)


def set_docker_task_context(context: DockerTaskContext | None):
    """Bind the docker context for the current async/thread context. Returns
    a token that the caller must pass to ``reset_docker_task_context``."""

    return _current_docker_context.set(context)


def reset_docker_task_context(token) -> None:
    _current_docker_context.reset(token)


def get_docker_task_context() -> Optional[DockerTaskContext]:
    return _current_docker_context.get()


_FAILED_LINE_RE = re.compile(r"^(FAILED|ERROR)\s+([^\s]+::test_[^\s\[]+)")
_TAIL_FAILED_RE = re.compile(r"(FAILED|ERROR)\s+([^\s]+::test_[^\s\[]+)")
_RESULT_MARKERS = (
    "Unfiltered Tests Passed",
    "Unfiltered Tests Failed",
    "All Tests Passed",
    "Some Tests Failed",
)


def _safe_log_token(value: Any) -> str:
    """Return a filesystem-safe token for per-cell docker log names."""

    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")[:160]


@dataclass(frozen=True)
class DockerTestGenEvalLiteAdapter(BenchmarkAdapter):
    """A ``BenchmarkAdapter`` whose ``run_unfiltered`` runs the artifact in
    the project's docker image with mutation skipped.

    Construct via :func:`make_docker_testgenevallite_adapter`; the dataclass
    fields below are populated there. The adapter is frozen because
    ``BenchmarkAdapter`` is frozen — we side-channel the per-task config via
    instance attributes set in ``__post_init__``-equivalent (here the
    factory).
    """

    task_instance: dict[str, Any]
    model_name: str
    namespace: str
    log_dir: Path
    official_repo: Path
    timeout_seconds: int = 600
    mutation_timeout_seconds: int = 600  # ignored for skip_mutation=True

    def with_patch_override(
        self,
        patch_diff: str,
        run_suffix: str = "",
    ) -> "DockerTestGenEvalLiteAdapter":
        """Return a copy whose docker task uses ``patch_diff`` as the fix.

        Dual-version verification needs to run the same generated test against
        both buggy code and a candidate surrogate patch. The official docker
        harness derives the checked-out code state from the task instance's
        ``patch`` field, so filesystem workdir mutations are invisible here.
        """

        task_instance = dict(self.task_instance)
        task_instance["_apex_patch_override"] = str(patch_diff or "")
        safe_suffix = _safe_log_token(run_suffix)
        if safe_suffix:
            task_instance["_apex_log_suffix"] = safe_suffix
        return replace(self, task_instance=task_instance)

    def with_run_suffix(self, run_suffix: str) -> "DockerTestGenEvalLiteAdapter":
        """Return a copy whose official-harness log name is unique."""

        task_instance = dict(self.task_instance)
        safe_suffix = _safe_log_token(run_suffix)
        if safe_suffix:
            task_instance["_apex_log_suffix"] = safe_suffix
        else:
            task_instance.pop("_apex_log_suffix", None)
        return replace(self, task_instance=task_instance)

    def _effective_model_name(self) -> str:
        suffix = _safe_log_token(self.task_instance.get("_apex_log_suffix"))
        if not suffix:
            return self.model_name
        return f"{self.model_name}.{suffix}"

    def run_unfiltered(
        self,
        artifact: GeneratedArtifact | dict[str, Any] | str,
        workdir: Path,
        *,
        timeout_seconds: float | None = None,
        python_executable: str | None = None,
    ) -> FinalAcceptanceRun:
        # Ensure swebench_docker is importable BEFORE any helper that uses it
        # (e.g. ``_build_task_instance``). Audit H9: we used to leak the
        # ``official_repo`` path into sys.path forever; on a multi-repo run
        # this accumulated dead entries and could shadow imports for later
        # tasks. We now insert under try/finally so the path is restored
        # before the call returns. The cost is one re-import of
        # ``swebench_docker`` per call (negligible vs the docker run).
        repo_str = str(self.official_repo.resolve())
        _path_inserted = False
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
            _path_inserted = True

        try:
            return self._run_unfiltered_inner(
                artifact,
                workdir,
                timeout_seconds=timeout_seconds,
                python_executable=python_executable,
            )
        finally:
            if _path_inserted:
                try:
                    sys.path.remove(repo_str)
                except ValueError:  # pragma: no cover - defensive
                    pass

    def _run_unfiltered_inner(
        self,
        artifact: GeneratedArtifact | dict[str, Any] | str,
        workdir: Path,
        *,
        timeout_seconds: float | None = None,
        python_executable: str | None = None,
    ) -> FinalAcceptanceRun:
        item = GeneratedArtifact.from_any(artifact)
        text = item.content or ""
        if not text.strip():
            return FinalAcceptanceRun(status="harness_error", diagnostic="empty artifact")
        try:
            spliced = self._build_task_instance(text)
        except Exception as exc:
            return FinalAcceptanceRun(
                status="harness_error",
                diagnostic=f"task_instance build failed: {type(exc).__name__}: {exc}",
            )
        instance_id = str(spliced.get("id") or spliced.get("instance_id") or "")
        if not instance_id:
            return FinalAcceptanceRun(
                status="harness_error",
                diagnostic="task_instance has neither id nor instance_id",
            )
        model_name = str(
            spliced.get("model_name_or_path")
            or spliced.get("model")
            or self._effective_model_name()
        )
        log_path = self.log_dir / f"{instance_id}.{model_name}.full.eval.log"
        if log_path.exists():
            log_path.unlink()  # ensure we read a fresh log
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # The upstream ``run_docker_evaluation`` calls ``await
        # process.communicate()`` with no asyncio-level timeout. If a
        # container hangs (Docker Desktop arch mismatch, conda init
        # deadlock, etc.) the worker thread blocks forever and stalls the
        # whole pipeline. We wrap the call with ``asyncio.wait_for`` and a
        # generous safety margin; on timeout we also nuke any matching
        # container by image name so the docker daemon recovers.
        #
        # Audit H10: only mutate the SPECIFIC env keys we need, restore
        # them by hand. The previous ``os.environ.clear()`` + ``update()``
        # pattern was unsafe under thread parallelism (one thread's
        # ``clear()`` would blow away another thread's mutations) and
        # the lock below pins the brief mutation window.
        wait_timeout = int(timeout_seconds or self.timeout_seconds) + 60
        _required_env = {
            "SWEBENCH_DOCKER_FORK_DIR": str(self.official_repo.resolve()),
        }
        with _DOCKER_ADAPTER_ENV_LOCK:
            previous_env: dict[str, Optional[str]] = {
                key: os.environ.get(key) for key in _required_env
            }
            for key, value in _required_env.items():
                if not (os.environ.get(key) or "").strip():
                    os.environ[key] = value
            try:
                try:
                    _run_async_with_clean_loop(
                        self._invoke_docker_async_with_timeout(
                            spliced,
                            timeout=int(timeout_seconds or self.timeout_seconds),
                            wait_timeout=wait_timeout,
                        ),
                        timeout_seconds=wait_timeout + 5,
                    )
                finally:
                    for key, prior in previous_env.items():
                        if prior is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = prior
            except asyncio.TimeoutError:
                self._kill_dangling_containers_for(spliced)
                return FinalAcceptanceRun(
                    status="harness_error",
                    diagnostic=(f"docker subprocess exceeded {wait_timeout}s — container nuked"),
                )
            except Exception as exc:  # pragma: no cover - defensive
                return FinalAcceptanceRun(
                    status="harness_error",
                    diagnostic=(f"docker invocation failed: {type(exc).__name__}: {exc}"),
                )
        # Wait briefly for the log to flush in case the container exited
        # right before fsync returned.
        deadline = time.time() + 5.0
        while not log_path.exists() and time.time() < deadline:
            time.sleep(0.2)
        if not log_path.exists():
            return FinalAcceptanceRun(
                status="harness_log_missing",
                diagnostic=f"no eval log produced at {log_path}",
                failure_taxonomy="harness_log_missing",
            )
        return _parse_eval_log(log_path)

    def _build_task_instance(self, artifact_text: str) -> dict[str, Any]:
        """Mirror ``run_evaluation.py``'s preprocessing: copy the dataset row,
        derive ``test_cmd`` / ``test_directives`` for the project, splice the
        artifact in as ``preds.full[0]``. Without ``test_cmd`` the in-container
        ``run_tests_task`` raises ``KeyError`` immediately."""

        from swebench_docker.constants import (  # noqa: WPS433
            KEY_ID,
            KEY_INSTANCE_ID,
            KEY_MODEL,
            KEY_PREDICTIONS,
            MAP_REPO_TO_TEST_FRAMEWORK,
        )
        from swebench_docker.swebench_utils import get_test_directives  # noqa: WPS433

        src = self.task_instance
        repo = str(src["repo"])
        test_type = MAP_REPO_TO_TEST_FRAMEWORK[repo]
        model_name = self._effective_model_name()
        # Build a temporary task instance with our preds so get_test_directives
        # has the same shape it'd see in the official flow.
        scratch = {
            **src,
            "test_directives": [],
            "test_cmd": "",
            KEY_MODEL: model_name,
            KEY_PREDICTIONS: {"full": artifact_text},
        }
        test_directives = get_test_directives(scratch)
        test_cmd = f"{test_type} {' '.join(test_directives)}"
        patch_value = (
            src.get("_apex_patch_override") if "_apex_patch_override" in src else src.get("patch")
        )
        return {
            "repo": src["repo"],
            "version": src["version"],
            "base_commit": src["base_commit"],
            KEY_ID: src.get(KEY_ID) or src.get("id"),
            KEY_INSTANCE_ID: src.get(KEY_INSTANCE_ID) or src.get("instance_id"),
            KEY_MODEL: model_name,
            KEY_PREDICTIONS: {"full": [artifact_text]},
            "preds_context": src.get("preds_context") or {},
            "test_patch": src.get("test_patch") or "",
            "test_file": src.get("test_file") or "",
            "code_file": src.get("code_file") or "",
            "patch": str(patch_value or ""),
            "test_directives": test_directives,
            "test_cmd": test_cmd,
        }

    async def _invoke_docker_async_with_timeout(
        self,
        task_instance: dict[str, Any],
        *,
        timeout: int,
        wait_timeout: int,
    ) -> None:
        await asyncio.wait_for(
            self._invoke_docker_async(task_instance, timeout=timeout),
            timeout=float(wait_timeout),
        )

    async def _invoke_docker_async(self, task_instance: dict[str, Any], *, timeout: int) -> None:
        from swebench_docker.run_docker import run_docker_evaluation  # noqa: WPS433

        # The harness emits one log per (id, setting). We invoke for the
        # "full" setting only (skip_mutation=True keeps it short).
        await run_docker_evaluation(
            task_instance,
            self.namespace,
            str(self.log_dir),
            "full",
            0,
            timeout=int(timeout),
            verbose=False,
            base64_instance=True,
            only_baseline=False,
            skip_mutation=True,
            mutation_timeout=int(self.mutation_timeout_seconds),
        )

    def _kill_dangling_containers_for(self, task_instance: dict[str, Any]) -> None:
        """Best-effort kill of any docker container running the project's
        image. Called when the host-side wait_for times out."""

        try:
            import subprocess  # local import to keep cold-start cheap

            repo = str(task_instance.get("repo") or "").replace("/", "_")
            instance_id = str(task_instance.get("instance_id") or "")
            patterns = [
                f"swe-bench-{repo}-testbed",
                f"swe-bench-{repo}-instance:{instance_id}" if instance_id else "",
            ]
            patterns = [p for p in patterns if p]
            if not patterns:
                return
            ps = subprocess.run(
                ["docker", "ps", "--format", "{{.ID}} {{.Image}}"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            ids: list[str] = []
            for line in (ps.stdout or "").splitlines():
                container_id, _, image = line.strip().partition(" ")
                if any(p in image for p in patterns):
                    ids.append(container_id)
            for cid in ids:
                subprocess.run(
                    ["docker", "kill", cid], capture_output=True, timeout=15, check=False
                )
        except Exception:  # pragma: no cover - best-effort cleanup
            return


def _run_async_with_clean_loop(coro: Any, *, timeout_seconds: float) -> Any:
    """Run a coroutine with explicit cancellation and loop cleanup."""

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        task = loop.create_task(coro)
        try:
            return loop.run_until_complete(asyncio.wait_for(task, timeout=float(timeout_seconds)))
        except BaseException:
            if not task.done():
                task.cancel()
                loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
            raise
        finally:
            pending = [item for item in asyncio.all_tasks(loop) if not item.done()]
            for item in pending:
                item.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            if hasattr(loop, "shutdown_default_executor"):
                loop.run_until_complete(loop.shutdown_default_executor())
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def make_docker_testgenevallite_adapter(
    *,
    task_instance: dict[str, Any],
    model_name: str,
    namespace: str,
    log_dir: Path,
    official_repo: Path,
    timeout_seconds: int = 600,
) -> DockerTestGenEvalLiteAdapter:
    return DockerTestGenEvalLiteAdapter(
        name="testgenevallite_docker",
        splice_mode=SpliceMode.REPLACE,
        task_instance=dict(task_instance),
        model_name=str(model_name),
        namespace=str(namespace),
        log_dir=Path(log_dir).expanduser().resolve(),
        official_repo=Path(official_repo).expanduser().resolve(),
        timeout_seconds=int(timeout_seconds),
    )


def _parse_eval_log(log_path: Path) -> FinalAcceptanceRun:
    """Extract per-test pass/fail and overall status from an eval log."""

    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return FinalAcceptanceRun(
            status="harness_error",
            diagnostic=f"unreadable log: {type(exc).__name__}: {exc}",
        )
    per_test: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        m = _FAILED_LINE_RE.match(stripped)
        if m:
            verdict = "fail" if m.group(1) == "FAILED" else "error"
            per_test[m.group(2)] = verdict
            continue
        # Pytest also emits "PASSED tests/foo.py::test_bar" lines in -v mode;
        # if present, capture them so the gate can distinguish.
        m2 = re.match(r"^PASSED\s+([^\s]+::test_[^\s\[]+)", stripped)
        if m2:
            per_test.setdefault(m2.group(1), "pass")
    # Whole-suite verdict from harness summary lines
    overall = ""
    for marker in _RESULT_MARKERS:
        if marker in text:
            overall = marker
            break
    if not overall:
        # No summary marker → harness probably crashed
        return FinalAcceptanceRun(
            status="harness_error",
            per_test_status=per_test,
            stdout_tail=text[-4000:],
            diagnostic="no harness verdict marker in log",
        )
    if overall in ("Unfiltered Tests Passed", "All Tests Passed"):
        status = "pass"
    else:
        status = "fail"
    if not per_test:
        # Harness verdict says fail but we couldn't extract per-test rows
        # (e.g., collection error). Surface a single sentinel so the gate
        # treats the whole artifact as failing.
        if status == "fail":
            per_test["__suite__::collection_or_setup"] = "error"
    return FinalAcceptanceRun(
        status=("collection_failed" if "__suite__::collection_or_setup" in per_test else status),
        per_test_status=per_test,
        stdout_tail=text[-4000:],
        stderr_tail="",
        returncode=0 if status == "pass" else 1,
        diagnostic=overall,
        failure_taxonomy=(
            "collection_failed"
            if "__suite__::collection_or_setup" in per_test
            else ("artifact_failed" if status == "fail" else "")
        ),
    )
