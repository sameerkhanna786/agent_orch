"""Benchmark adapter contracts shared by test-generation validators."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .final_acceptance_gate import FinalAcceptanceRun, GeneratedArtifact
from .splice_simulator import AppendSpliceSimulator, SpliceMode, TestGenEvalSpliceSimulator


@dataclass(frozen=True)
class BenchmarkAdapter:
    name: str
    splice_mode: SpliceMode

    def splice_simulator(self) -> TestGenEvalSpliceSimulator:
        if self.splice_mode == SpliceMode.APPEND:
            return AppendSpliceSimulator()
        return TestGenEvalSpliceSimulator(splice_mode=self.splice_mode)

    def run_unfiltered(
        self,
        artifact: GeneratedArtifact | dict[str, Any] | str,
        workdir: Path,
        *,
        timeout_seconds: float = 30.0,
        python_executable: str | None = None,
    ) -> FinalAcceptanceRun:
        """Run a generated artifact as a whole file.

        Benchmark-specific adapters can override this. The default is a local
        Python/pytest runner used by unit tests and non-official smoke probes.
        """

        item = GeneratedArtifact.from_any(artifact)
        target = Path(workdir) / item.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(item.content, encoding="utf-8")
        executable = python_executable or sys.executable
        command = [
            executable,
            "-m",
            "pytest",
            "-q",
            "-vv",
            "--tb=short",
            "-p",
            "no:cacheprovider",
            str(target),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=float(timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return FinalAcceptanceRun(
                status="harness_error",
                diagnostic="unfiltered run timed out",
            )
        except OSError as exc:
            return FinalAcceptanceRun(
                status="harness_error",
                diagnostic=f"{type(exc).__name__}: {exc}",
            )
        text = (completed.stdout or "") + "\n" + (completed.stderr or "")
        per_test = _parse_pytest_statuses(text)
        status = "pass" if completed.returncode == 0 else "fail"
        return FinalAcceptanceRun(
            status=status,
            per_test_status=per_test,
            stdout_tail=(completed.stdout or "")[-4000:],
            stderr_tail=(completed.stderr or "")[-4000:],
            returncode=completed.returncode,
            diagnostic="" if completed.returncode == 0 else text[-4000:],
        )


TESTGENEVAL_ADAPTER = BenchmarkAdapter("testgeneval", SpliceMode.REPLACE)
TESTGENEVAL_LITE_ADAPTER = BenchmarkAdapter("testgenevallite", SpliceMode.REPLACE)
COMMIT0_ADAPTER = BenchmarkAdapter("commit0", SpliceMode.APPEND)
SWEBENCH_PRO_TESTGEN_ADAPTER = BenchmarkAdapter("swebench_pro_testgen", SpliceMode.PATCH)

_ADAPTERS = {
    adapter.name: adapter
    for adapter in (
        TESTGENEVAL_ADAPTER,
        TESTGENEVAL_LITE_ADAPTER,
        COMMIT0_ADAPTER,
        SWEBENCH_PRO_TESTGEN_ADAPTER,
    )
}


def get_benchmark_adapter(name: str) -> BenchmarkAdapter:
    normalized = str(name or "").strip().lower().replace("-", "_")
    return _ADAPTERS.get(normalized, BenchmarkAdapter(normalized or "unknown", SpliceMode.APPEND))


def _parse_pytest_statuses(text: str) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for line in str(text or "").splitlines():
        stripped = line.strip()
        match = re.match(r"(.+?::test[^\s]+)\s+(PASSED|FAILED|ERROR)\b", stripped)
        if match:
            statuses[match.group(1)] = match.group(2).lower().replace("ed", "")
            continue
        match = re.match(r"(FAILED|ERROR)\s+(.+?::test[^\s]+)", stripped)
        if match:
            statuses[match.group(2)] = "fail" if match.group(1) == "FAILED" else "error"
    return statuses
