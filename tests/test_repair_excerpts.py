"""F1: thread failure_excerpts into the base-loop repair waves via ctx.repair_excerpts(red), gated by
APEX_OMEGA_REPAIR_EXCERPTS_LOOP (default off => byte-identical). The cheap "earlier helps" win: the
base loop-until-dry / run_phase repair previously discarded the excerpts reduce_residuals computes.

Must NOT collide with the pre-existing APEX_OMEGA_REPAIR_EXCERPTS (the repair_attempt Reflexion path,
default ON), and must NOT affect SARP's direct excerpt-passing.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from apex_omega.autogen.context import OrchestrationContext
from apex_omega.engine.runtime import Engine
from apex_omega.executor.fake import FakeExecutor
from apex_omega.kernel.verify import VerificationResult
from apex_omega.workflows.best_of_n import WorkerSpec


@pytest.fixture(autouse=True)
def _clear_env():
    keys = ["APEX_OMEGA_REPAIR_EXCERPTS_LOOP", "APEX_OMEGA_REPAIR_EXCERPTS", "APEX_OMEGA_SARP"]
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _ctx():
    d = Path(tempfile.mkdtemp()) / "r"
    d.mkdir()
    (d / "f.py").write_text("x = 0\n")
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-qm", "b"]):
        subprocess.run(c, cwd=d, check=True, capture_output=True)
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=8)
    return OrchestrationContext(
        eng, executor=FakeExecutor(None), worker_specs=[WorkerSpec("codex_cli", "m")],
        source_repo=str(d), base_commit=None, score_fn=lambda wt: VerificationResult(),
        prompt_builder=lambda c, i, s: "x", max_agents=8, initial_agents=1)


_RED = {"failure_excerpts": "assert c() == 3  ->  assert 0 == 3", "residual_failing_ids": ["t::c"]}


def test_repair_excerpts_off_is_empty_byte_identical():
    ctx = _ctx()
    assert ctx.repair_excerpts(_RED) == ""        # default off -> "" -> repair call byte-identical
    assert ctx.repair_excerpts(None) == ""


def test_repair_excerpts_on_returns_excerpts():
    os.environ["APEX_OMEGA_REPAIR_EXCERPTS_LOOP"] = "1"
    ctx = _ctx()
    assert ctx.repair_excerpts(_RED) == "assert c() == 3  ->  assert 0 == 3"
    assert ctx.repair_excerpts({}) == ""          # no excerpts -> ""
    assert ctx.repair_excerpts(None) == ""


def test_repair_excerpts_loop_flag_is_distinct_from_repair_excerpts():
    """The new LOOP flag must be independent of the pre-existing APEX_OMEGA_REPAIR_EXCERPTS
    (context.py:1449, the Reflexion redaction path). Setting the OLD flag must not enable the LOOP
    path, and vice-versa."""
    ctx = _ctx()
    os.environ["APEX_OMEGA_REPAIR_EXCERPTS"] = "1"      # OLD flag ON
    os.environ.pop("APEX_OMEGA_REPAIR_EXCERPTS_LOOP", None)   # LOOP flag OFF
    assert ctx.repair_excerpts(_RED) == ""             # OLD flag does NOT enable the LOOP helper
    os.environ["APEX_OMEGA_REPAIR_EXCERPTS"] = "0"      # OLD flag OFF
    os.environ["APEX_OMEGA_REPAIR_EXCERPTS_LOOP"] = "1"  # LOOP flag ON
    assert ctx.repair_excerpts(_RED) == "assert c() == 3  ->  assert 0 == 3"  # LOOP independent of OLD


def test_repair_excerpts_off_values():
    ctx = _ctx()
    for off in ("0", "false", "no", "off", ""):
        os.environ["APEX_OMEGA_REPAIR_EXCERPTS_LOOP"] = off
        assert ctx.repair_excerpts(_RED) == "", off


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
