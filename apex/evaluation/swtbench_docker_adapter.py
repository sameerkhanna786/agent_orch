"""Docker-based BenchmarkAdapter for SWT-Bench.

Mirrors :mod:`apex.evaluation.docker_acceptance_adapter` but invokes the
SWT-Bench harness's docker entrypoint instead of swebench_docker. SWT-Bench
ships its own image namespace (the SWE-bench-eval family, distinct from
the kdjain TestGenEval namespace) and its own ``swt_bench`` python package
that drives the docker run loop.

The SWT-Bench harness is invoked as a subprocess (``python -m`` the resolved
upstream entrypoint with a single-row predictions JSONL) rather than as an
in-process import — that decouples APEX from a swt_bench python install on
the local interpreter and lets us run the harness inside its own
hermetic docker image without polluting the apex venv. The trade-off is
~2-3s of subprocess startup per run_unfiltered call, which is dwarfed by
the per-container 30-60s evaluation cost.

This adapter is what the V5 dual-state verifier hands each (test ×
patch) cell off to. The voting machinery is benchmark-agnostic — it
asks the adapter to "run this test artifact against the buggy /
patched commit and return a verdict" — so plugging SWT-Bench into V5
is a one-adapter change.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

from apex.core.docker_pinning import resolve_image
from apex.core.subprocess_retry import (
    DEFAULT_RETRY_ON,
    RetryDiagnostics,
    run_with_classification,
)

from .benchmark_adapters import BenchmarkAdapter
from .final_acceptance_gate import FinalAcceptanceRun, GeneratedArtifact
from .splice_simulator import SpliceMode
from .swtbench_harness import make_swtbench_harness_env, resolve_swtbench_module

logger = logging.getLogger(__name__)


# Default image namespace SWT-Bench publishes its eval images under. The
# upstream swt_bench package also defaults to this — leave overridable
# per-call so operators can point at a private mirror.
DEFAULT_SWTBENCH_NAMESPACE = "aorwall"

# Default timeout for the swt_bench subprocess. Generous because cold
# image pulls dominate first-call latency on a fresh host.
DEFAULT_SUBPROCESS_TIMEOUT_SECONDS = 1800


def _safe_log_token(value: Any) -> str:
    """Return a filesystem-safe token for per-call log file names."""

    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")[:160]


@dataclass(frozen=True)
class SWTBenchDockerAdapter(BenchmarkAdapter):
    """A ``BenchmarkAdapter`` that runs an artifact through SWT-Bench's
    docker harness.

    Construct via :func:`make_swtbench_docker_adapter`. ``run_unfiltered``
    writes the artifact as a unified-diff prediction JSONL, shells out to
    the resolved SWT-Bench module, then parses the per-instance log the
    harness drops into ``log_dir``.
    """

    task_instance: dict[str, Any]
    model_name: str
    namespace: str
    log_dir: Path
    dataset_name: str
    swt_bench_python: str = ""
    timeout_seconds: int = DEFAULT_SUBPROCESS_TIMEOUT_SECONDS
    run_id_prefix: str = "apex_swtbench"
    # Phase 1c: subprocess retry budget. Real APEX failures are NEVER
    # retried; only env_* and HARNESS_BUG failures trigger a retry. Set
    # to 1 to disable retry entirely.
    max_attempts: int = 3

    # ------------------------------------------------------------------
    # V5 dual-state hooks
    # ------------------------------------------------------------------

    def with_patch_override(
        self,
        patch_diff: str,
        run_suffix: str = "",
    ) -> "SWTBenchDockerAdapter":
        """Return a copy whose docker task uses ``patch_diff`` as the fix.

        Required by V5's ``dual_version_verifier``. The verifier asks the
        adapter to grade a candidate test against a *candidate* fix patch
        (from the patch_surrogate fan-out), not just against the gold
        ``patch`` shipped with the dataset row.
        """

        task_instance = dict(self.task_instance)
        task_instance["_apex_patch_override"] = str(patch_diff or "")
        safe_suffix = _safe_log_token(run_suffix)
        if safe_suffix:
            task_instance["_apex_log_suffix"] = safe_suffix
        return replace(self, task_instance=task_instance)

    def with_run_suffix(self, run_suffix: str) -> "SWTBenchDockerAdapter":
        """Return a copy whose harness log name is unique."""

        task_instance = dict(self.task_instance)
        safe_suffix = _safe_log_token(run_suffix)
        if safe_suffix:
            task_instance["_apex_log_suffix"] = safe_suffix
        else:
            task_instance.pop("_apex_log_suffix", None)
        return replace(self, task_instance=task_instance)

    # ------------------------------------------------------------------
    # Adapter contract
    # ------------------------------------------------------------------

    def _effective_model_name(self) -> str:
        suffix = _safe_log_token(self.task_instance.get("_apex_log_suffix"))
        if not suffix:
            return self.model_name
        return f"{self.model_name}.{suffix}"

    def _effective_run_id(self) -> str:
        suffix = _safe_log_token(self.task_instance.get("_apex_log_suffix"))
        if not suffix:
            return self.run_id_prefix
        return f"{self.run_id_prefix}_{suffix}"

    def run_unfiltered(
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

        instance_id = str(
            self.task_instance.get("instance_id") or self.task_instance.get("id") or ""
        )
        if not instance_id:
            return FinalAcceptanceRun(
                status="harness_error",
                diagnostic="task_instance has neither id nor instance_id",
            )

        # Build the JSONL prediction the swt_bench harness consumes.
        model_patch = _build_model_patch_from_artifact(
            artifact_text=text,
            artifact_path=str(
                item.path or self.task_instance.get("test_file") or "tests/test_apex_swtbench.py"
            ),
            base_commit=str(self.task_instance.get("base_commit") or ""),
            existing_test_source=str(self.task_instance.get("test_src") or ""),
        )
        if not model_patch:
            return FinalAcceptanceRun(
                status="harness_error",
                diagnostic="failed to build a unified diff from artifact",
            )

        record = {
            "instance_id": instance_id,
            "model_name_or_path": self._effective_model_name(),
            "model_patch": model_patch,
        }

        # Working area for this single-row invocation. Don't pollute the
        # caller's workdir; SWT-Bench writes its own logs into ``log_dir``.
        scratch = Path(tempfile.mkdtemp(prefix="apex_swtbench_", dir=str(workdir)))
        preds_path = scratch / "predictions.jsonl"
        preds_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        run_id = f"{self._effective_run_id()}_{instance_id}_{int(time.time() * 1000)}"
        log_dir = self.log_dir / run_id
        log_dir.mkdir(parents=True, exist_ok=True)

        cmd = self._build_command(
            preds_path=preds_path,
            run_id=run_id,
        )

        timeout = int(timeout_seconds or self.timeout_seconds)
        env = make_swtbench_harness_env()
        # If the caller pre-staged a docker namespace override, honor it.
        env.setdefault("SWTBENCH_NAMESPACE", self.namespace)

        # Pin the namespace tag via docker_pinning so any active manifest
        # records the digest it ran against. SWT-Bench resolves per-instance
        # images dynamically inside the harness; we anchor the namespace.
        try:
            resolve_image(f"{self.namespace}/sweb.eval.x86_64:latest")
        except Exception:  # pragma: no cover - best-effort
            pass

        # Retry on env_* / HARNESS_BUG; never on APEX_MISS.
        retry_diag = RetryDiagnostics()
        try:
            completed = run_with_classification(
                cmd,
                max_attempts=int(self.max_attempts),
                backoff="exponential",
                retry_on=DEFAULT_RETRY_ON,
                classifier_context={"phase": "scoring"},
                diagnostics_sink=retry_diag,
                timeout=timeout + 60,
                env=env,
            )
        except FileNotFoundError as exc:
            # Belt-and-suspenders: run_with_classification already converts
            # spawn failures to a CompletedProcess(returncode=127), so this
            # branch is defensive only.
            return FinalAcceptanceRun(
                status="harness_error",
                diagnostic=(
                    f"swt_bench harness binary not found: {exc}. "
                    "Install with `pip install swt-bench`, or pass "
                    "swt_bench_python pointing at an interpreter that has it."
                ),
                failure_taxonomy="harness_bug",
            )

        # Treat the synthesised timeout-rc=124 as a timeout for cleanup.
        if completed.returncode == 124:
            self._kill_dangling_containers(run_id)
            cls = (
                retry_diag.final_classification.failure_class.value
                if retry_diag.final_classification
                else "env_timeout"
            )
            return FinalAcceptanceRun(
                status="harness_error",
                stdout_tail=(completed.stdout or "")[-4000:],
                stderr_tail=(completed.stderr or "")[-4000:],
                returncode=completed.returncode,
                diagnostic=f"swt_bench subprocess exceeded {timeout + 60}s",
                failure_taxonomy=cls,
            )

        if completed.returncode != 0:
            cls_name = (
                retry_diag.final_classification.failure_class.value
                if retry_diag.final_classification
                else ""
            )
            return FinalAcceptanceRun(
                status="harness_error",
                stdout_tail=(completed.stdout or "")[-4000:],
                stderr_tail=(completed.stderr or "")[-4000:],
                returncode=completed.returncode,
                diagnostic=(
                    f"swt_bench returncode={completed.returncode}; "
                    "see stderr_tail for detail" + (f" [{cls_name}]" if cls_name else "")
                ),
                failure_taxonomy=cls_name,
            )

        return _parse_swtbench_log(
            log_dir=log_dir,
            run_id=run_id,
            instance_id=instance_id,
            stdout_tail=(completed.stdout or "")[-4000:],
            stderr_tail=(completed.stderr or "")[-4000:],
        )

    def _build_command(self, *, preds_path: Path, run_id: str) -> list[str]:
        python = self.swt_bench_python or _default_python()
        swtbench_module, _ = resolve_swtbench_module(python)
        return [
            python,
            "-m",
            swtbench_module or "swt_bench.main",
            "--dataset_name",
            self.dataset_name,
            "--predictions_path",
            str(preds_path),
            "--max_workers",
            "1",
            "--run_id",
            run_id,
            "--filter_swt",
            "--namespace",
            self.namespace,
            "--instance_ids",
            str(self.task_instance.get("instance_id") or ""),
        ]

    def _kill_dangling_containers(self, run_id: str) -> None:
        """Best-effort kill of any container tagged with ``run_id``."""

        try:
            ps = subprocess.run(
                ["docker", "ps", "--format", "{{.ID}} {{.Names}}"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            ids: list[str] = []
            for line in (ps.stdout or "").splitlines():
                cid, _, name = line.strip().partition(" ")
                if run_id in name or "swt-bench" in name.lower():
                    ids.append(cid)
            for cid in ids:
                subprocess.run(
                    ["docker", "kill", cid],
                    capture_output=True,
                    timeout=15,
                    check=False,
                )
        except Exception:  # pragma: no cover - best-effort cleanup
            return


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_python() -> str:
    """Resolve the python the swt_bench package is installed under.

    Honors ``APEX_SWTBENCH_PYTHON`` so operators can point at a separate
    interpreter (avoids forcing apex.venv to install swt-bench).
    """

    override = os.environ.get("APEX_SWTBENCH_PYTHON", "").strip()
    if override:
        return override
    # Default to the same python the caller is running under.
    return shutil.which("python") or shutil.which("python3") or "python"


def _build_model_patch_from_artifact(
    *,
    artifact_text: str,
    artifact_path: str,
    base_commit: str,
    existing_test_source: str,
) -> str:
    """Build a unified-diff ``model_patch`` against the buggy test file.

    Uses ``difflib.unified_diff`` so we don't need git in the path.
    Returns an empty string if the input is empty.
    """

    if not artifact_text.strip():
        return ""
    import difflib

    rel_path = artifact_path.lstrip("/")
    a_lines = (existing_test_source or "").splitlines(keepends=True)
    b_lines = artifact_text.splitlines(keepends=True)
    if a_lines and not a_lines[-1].endswith("\n"):
        a_lines[-1] += "\n"
    if b_lines and not b_lines[-1].endswith("\n"):
        b_lines[-1] += "\n"
    diff_lines = list(
        difflib.unified_diff(
            a_lines,
            b_lines,
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            n=3,
        )
    )
    if not diff_lines:
        # Identical buggy/new content — produce a no-op header so the
        # harness still has a syntactically valid diff to evaluate.
        return ""
    header = f"diff --git a/{rel_path} b/{rel_path}\n"
    return header + "".join(diff_lines)


_PASS_MARKERS = (
    "All Tests Passed",
    "Unfiltered Tests Passed",
    "Tests Passed",
    "FAIL_TO_PASS_PASSED",
)
_FAIL_MARKERS = (
    "All Tests Failed",
    "Some Tests Failed",
    "Unfiltered Tests Failed",
)


def _parse_swtbench_log(
    *,
    log_dir: Path,
    run_id: str,
    instance_id: str,
    stdout_tail: str,
    stderr_tail: str,
) -> FinalAcceptanceRun:
    """Find and parse the per-instance log SWT-Bench drops in ``log_dir``.

    Schema differs from swebench_docker but the upstream package also
    writes an instance-level JSON report (``report.json``) into a
    per-run subdirectory. Prefer that when present; fall back to a log
    text scan + the captured stdout.
    """

    # 1) Try the report.json shape first.
    candidates = list(log_dir.rglob(f"*{run_id}*/{instance_id}*/report.json"))
    if not candidates:
        candidates = list(log_dir.rglob(f"*{instance_id}*/report.json"))
    for report_path in candidates:
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        verdict = _verdict_from_report(payload)
        if verdict is not None:
            return verdict

    # 2) Fall back to scanning eval logs for verdict markers.
    #    Cover three layouts: file name contains instance_id, OR the log
    #    sits in a directory whose name contains instance_id, OR a
    #    well-known eval-log naming pattern from the run.
    text_blobs: list[str] = []
    seen: set[Path] = set()

    def _slurp(path: Path) -> None:
        if path in seen or not path.is_file() or path.suffix not in {".log", ".txt"}:
            return
        seen.add(path)
        try:
            text_blobs.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            return

    for path in log_dir.rglob(f"*{instance_id}*"):
        _slurp(path)
    for path in log_dir.rglob("*"):
        if instance_id in str(path.parent):
            _slurp(path)
    if stdout_tail:
        text_blobs.append(stdout_tail)
    text = "\n".join(text_blobs)

    if any(marker in text for marker in _PASS_MARKERS):
        return FinalAcceptanceRun(
            status="pass",
            stdout_tail=stdout_tail or text[-4000:],
            stderr_tail=stderr_tail,
            returncode=0,
            diagnostic="swt_bench: pass marker present",
        )
    if any(marker in text for marker in _FAIL_MARKERS):
        return FinalAcceptanceRun(
            status="fail",
            stdout_tail=stdout_tail or text[-4000:],
            stderr_tail=stderr_tail,
            returncode=1,
            diagnostic="swt_bench: fail marker present",
        )
    return FinalAcceptanceRun(
        status="harness_error",
        stdout_tail=stdout_tail or text[-4000:],
        stderr_tail=stderr_tail,
        diagnostic="swt_bench: no verdict marker found in logs",
    )


def _verdict_from_report(payload: Any) -> Optional[FinalAcceptanceRun]:
    """Map a swt_bench report.json payload onto a FinalAcceptanceRun."""

    if not isinstance(payload, dict):
        return None
    # The swt_bench harness exposes per-instance result dicts. Common keys
    # observed: ``resolved`` (bool), ``f2p_success`` (bool), ``status``
    # (str). Be permissive.
    if isinstance(payload.get("resolved"), bool):
        status = "pass" if payload["resolved"] else "fail"
    elif isinstance(payload.get("f2p_success"), bool):
        status = "pass" if payload["f2p_success"] else "fail"
    elif isinstance(payload.get("status"), str):
        raw = payload["status"].lower()
        if raw in {"pass", "passed", "resolved"}:
            status = "pass"
        elif raw in {"fail", "failed", "unresolved"}:
            status = "fail"
        else:
            return None
    else:
        return None
    return FinalAcceptanceRun(
        status=status,
        diagnostic=f"swt_bench report: {payload}",
        returncode=0 if status == "pass" else 1,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_swtbench_docker_adapter(
    *,
    task_instance: dict[str, Any],
    model_name: str,
    dataset_name: str,
    log_dir: Path,
    namespace: str = DEFAULT_SWTBENCH_NAMESPACE,
    swt_bench_python: str = "",
    timeout_seconds: int = DEFAULT_SUBPROCESS_TIMEOUT_SECONDS,
    run_id_prefix: str = "apex_swtbench",
    max_attempts: int = 3,
) -> SWTBenchDockerAdapter:
    """Build a SWTBenchDockerAdapter wired into the V5 voting layer.

    ``dataset_name`` MUST match the HF dataset the caller is running
    against (e.g., ``nmuendler/SWT-Bench_Lite_bm25_27k_zsb``); the
    swt_bench harness uses it to look up per-task setup metadata.
    """

    return SWTBenchDockerAdapter(
        name="swtbench_docker",
        splice_mode=SpliceMode.REPLACE,
        task_instance=dict(task_instance),
        model_name=str(model_name),
        namespace=str(namespace),
        log_dir=Path(log_dir).expanduser().resolve(),
        dataset_name=str(dataset_name),
        swt_bench_python=str(swt_bench_python or ""),
        timeout_seconds=int(timeout_seconds),
        run_id_prefix=str(run_id_prefix),
        max_attempts=int(max_attempts),
    )
