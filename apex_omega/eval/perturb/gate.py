"""Validation gate — the LINCHPIN — + regenerated gold expected-id set.

For each repo: apply the rename(+neutralization) to the FULL REFERENCE (gold)
implementation, install it into a fresh isolated venv, run its gold test suite,
and REQUIRE 100% pass (zero collection errors).  This (a) PROVES the rename is
semantics-preserving and (b) yields the regenerated gold expected-id set (the new
bz2).  If the perturbed reference does NOT pass 100%, the rename is UNSOUND for
that repo -> the caller SKIPs it.  We never emit a variant that fails its own
gold tests.

Runs in any interpreter with ``pip``/``venv`` (no rope/libcst needed here); the
gate venv is isolated per-repo (the perturbed package keeps vanilla's top-level
import name, so isolation is mandatory).
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GateResult:
    passed: bool
    collected: int = 0
    n_passed: int = 0
    n_failed: int = 0
    n_errors: int = 0
    expected_ids: list[str] = field(default_factory=list)
    detail: str = ""
    venv_python: str = ""


def _run(cmd: list[str], cwd: Optional[Path] = None, timeout: int = 1800, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True,
        timeout=timeout, check=False, env=env,
    )


def make_isolated_venv(venv_dir: Path, python_exe: str) -> Path:
    """Create a fresh venv at *venv_dir* and return its python path."""
    import shutil
    if venv_dir.exists():
        shutil.rmtree(venv_dir, ignore_errors=True)
    _run([python_exe, "-m", "venv", str(venv_dir)], timeout=300)
    vpy = venv_dir / "bin" / "python"
    _run([str(vpy), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"], timeout=600)
    return vpy


def _collect_passed_ids(json_report_path: Path) -> tuple[list[str], int, int, int, int]:
    """Parse a pytest --json-report file -> (passed_ids, collected, passed, failed, errors)."""
    data = json.loads(json_report_path.read_text(encoding="utf-8"))
    tests = data.get("tests") or []
    passed_ids = sorted(t["nodeid"] for t in tests if t.get("outcome") == "passed")
    n_passed = sum(1 for t in tests if t.get("outcome") == "passed")
    n_failed = sum(1 for t in tests if t.get("outcome") in ("failed", "error"))
    summary = data.get("summary") or {}
    collected = int(summary.get("collected", len(tests)))
    n_errors = int(summary.get("error", 0)) + int(summary.get("errors", 0))
    return passed_ids, collected, n_passed, n_failed, n_errors


def run_gate(
    perturbed_reference_root: Path,
    *,
    test_dir: str,
    test_cmd: str,
    python_exe: str,
    venv_dir: Path,
    install_command: str = "pip install -e .",
    extra_pip: tuple[str, ...] = (),
    timeout: int = 2400,
    double_run: bool = True,
) -> GateResult:
    """Install the perturbed reference and run its gold suite under json-report.

    GATE = zero collection errors AND 100% of collected tests PASS.

    Returns a :class:`GateResult`; ``passed`` is the soundness verdict and
    ``expected_ids`` is the regenerated gold inventory (only valid if passed).
    """
    perturbed_reference_root = perturbed_reference_root.resolve()
    vpy = make_isolated_venv(venv_dir, python_exe)
    # install json-report plugin + the repo's own deps (editable) + extras
    pj = _run([str(vpy), "-m", "pip", "install", "pytest", "pytest-json-report", *extra_pip], timeout=900)
    if pj.returncode != 0:
        return GateResult(False, detail=f"pip install pytest/json-report failed: {pj.stderr[-800:]}", venv_python=str(vpy))
    # editable install of the perturbed package
    inst_cmd = install_command.replace("pip install", f"{vpy} -m pip install", 1) \
        if install_command.startswith("pip install") else f"{vpy} -m pip install -e ."
    inst = _run(["bash", "-lc", inst_cmd], cwd=perturbed_reference_root, timeout=1800)
    if inst.returncode != 0:
        return GateResult(False, detail=f"editable install failed: {inst.stderr[-1200:]}", venv_python=str(vpy))

    def _one_run(tag: str) -> GateResult:
        report = perturbed_reference_root / f".gate_report_{tag}.json"
        target = test_dir if test_dir else "."
        cmd = [
            str(vpy), "-m", "pytest", target,
            "--json-report", f"--json-report-file={report}",
            "--continue-on-collection-errors", "-p", "no:cacheprovider", "-q",
        ]
        proc = _run(cmd, cwd=perturbed_reference_root, timeout=timeout)
        if not report.exists():
            return GateResult(False, detail=f"[{tag}] no json report; stdout tail:\n{proc.stdout[-1500:]}\n{proc.stderr[-800:]}", venv_python=str(vpy))
        ids, collected, n_pass, n_fail, n_err = _collect_passed_ids(report)
        ok = (n_fail == 0 and n_err == 0 and collected > 0 and n_pass == collected)
        return GateResult(
            passed=ok, collected=collected, n_passed=n_pass, n_failed=n_fail,
            n_errors=n_err, expected_ids=ids, venv_python=str(vpy),
            detail=f"[{tag}] collected={collected} passed={n_pass} failed={n_fail} errors={n_err}",
        )

    r1 = _one_run("a")
    if not r1.passed or not double_run:
        return r1
    # double-run (cache-cleared) to catch pickle/order nondeterminism before freezing
    r2 = _one_run("b")
    if not r2.passed:
        return GateResult(False, detail=f"double-run mismatch: run-a OK but run-b: {r2.detail}", venv_python=str(vpy))
    if set(r1.expected_ids) != set(r2.expected_ids):
        return GateResult(False, detail="double-run id-set mismatch (nondeterministic)", venv_python=str(vpy))
    r1.detail += " | double-run OK"
    return r1


def write_expected_ids_bz2(expected_ids: list[str], bz2_path: Path) -> None:
    """Write the regenerated gold ids to a bz2 in the commit0 get_pytest_ids format
    (newline-joined node ids)."""
    import bz2 as _bz2
    bz2_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(expected_ids)
    with _bz2.open(bz2_path, "wt", encoding="utf-8") as fh:
        fh.write(payload)


def read_expected_ids_bz2(bz2_path: Path) -> list[str]:
    import bz2 as _bz2
    with _bz2.open(bz2_path, "rt", encoding="utf-8") as fh:
        return [x for x in fh.read().split("\n") if x.strip()]
