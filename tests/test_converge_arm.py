"""Phase-3 A/B wiring: the converge orchestration selector (architect freeze + catalog) and the
run_ladder Arm-A/Arm-B definitions. Offline, no codex burn."""

from __future__ import annotations

import os
import tempfile

import pytest

from apex_omega.autogen.architect import author_orchestration
from apex_omega.autogen.catalog import known_workflows, resolve_workflow
from apex_omega.autogen.templates import (
    BEST_OF_N_ORCHESTRATION,
    CONVERGE_EXEMPLAR,
    DEFAULT_ORCHESTRATION,
)
from apex_omega.engine.runtime import Engine
from apex_omega.executor.fake import FakeExecutor
from apex_omega.workflows.best_of_n import WorkerSpec


def _eng():
    return Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=8)


@pytest.mark.parametrize("selector", ["converge", "rebuild"])
def test_author_freezes_converge(monkeypatch, selector):
    monkeypatch.setenv("APEX_OMEGA_ORCHESTRATION", selector)
    fw = author_orchestration(_eng(), executor=FakeExecutor(),
                              worker_specs=[WorkerSpec("codex_cli", "m")],
                              repo_map={"difficulty": "medium"}, author=True)
    assert fw.origin == "converge"
    assert fw.source == DEFAULT_ORCHESTRATION     # frozen convergence default even with author=True
    assert fw.lint_ok


def test_catalog_registers_converge():
    assert "converge" in known_workflows()
    assert resolve_workflow("converge") == DEFAULT_ORCHESTRATION
    # default-best-of-n now resolves to the CHEAP path (not the convergence default)
    assert resolve_workflow("default-best-of-n") == BEST_OF_N_ORCHESTRATION


def test_converge_exemplar_lints():
    from apex_omega.autogen.sandbox import lint_source
    assert lint_source(CONVERGE_EXEMPLAR).ok
    assert lint_source(DEFAULT_ORCHESTRATION).ok
    assert lint_source(BEST_OF_N_ORCHESTRATION).ok


def test_run_ladder_ab_arms_present():
    """run_ladder reads LADDER_ARMS at IMPORT time, so test the selector in a fresh subprocess
    (a clean module-state import) — avoids reload-under-pytest import-resolution issues."""
    import json
    import subprocess
    import sys

    code = (
        "import json, scripts.run_ladder as rl;"
        "print(json.dumps([[a[0], (a[2] if len(a) > 2 else {})] for a in rl.ARMS]))"
    )
    env = dict(os.environ)
    env["LADDER_ARMS"] = "omega_flips_unbounded,omega_converge_unbounded"
    env["PYTHONPATH"] = os.getcwd()
    proc = subprocess.run([sys.executable, "-c", code], env=env, text=True,
                          capture_output=True, cwd=os.getcwd())
    assert proc.returncode == 0, proc.stderr[-2000:]
    arms = json.loads(proc.stdout.strip().splitlines()[-1])
    by_label = {a[0]: a[1] for a in arms}
    assert [a[0] for a in arms] == ["omega_flips_unbounded", "omega_converge_unbounded"]
    # Arm A = flips only (no orchestration switch).
    assert by_label["omega_flips_unbounded"].get("APEX_OMEGA_REPAIR_ITERS") == "2"
    assert "APEX_OMEGA_ORCHESTRATION" not in by_label["omega_flips_unbounded"]
    # Arm B = converge + flips.
    assert by_label["omega_converge_unbounded"].get("APEX_OMEGA_ORCHESTRATION") == "converge"
    assert by_label["omega_converge_unbounded"].get("APEX_OMEGA_REPAIR_ITERS") == "2"
