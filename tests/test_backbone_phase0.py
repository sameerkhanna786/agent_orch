"""Backbone Phase 0 — safety floor (budget default-unbounded, per-RUN agent backstop
with journal rehydration, parallel fan-out cap, frozen-.py strip fix)."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import pytest

from apex_omega.engine.runtime import Engine
from apex_omega.errors import FailLoud
from apex_omega.journal.wal import RESULT_OK, Journal
from apex_omega.types import ExecResult, ScopedTask

_RUN_LADDER = Path(__file__).resolve().parents[1] / "scripts" / "run_ladder.py"


def _load_run_ladder():
    spec = importlib.util.spec_from_file_location("run_ladder", _RUN_LADDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- 0.6 frozen-.py strip fix (verified live bug) --------------------------- #
def test_strip_keeps_frozen_py():
    rl = _load_run_ladder()
    rundir = Path(tempfile.mkdtemp()) / "cell"
    orch = rundir / "journal" / "orchestrator"
    orch.mkdir(parents=True)
    frozen_py = orch / "bb4ba79.py"
    frozen_py.write_text("def orchestrate(ctx):\n    return None\n")
    (orch / "frozen.json").write_text('{"source_path": "bb4ba79.py"}')
    bulky = rundir / "src" / "mod.so"
    bulky.parent.mkdir(parents=True)
    bulky.write_bytes(b"\x00" * 64)
    rl._strip_checkout(rundir)
    assert frozen_py.exists(), "frozen orchestration .py was stripped -> resume would crash"
    assert (orch / "frozen.json").exists()
    assert not bulky.exists()


# --- 0.1 token budget default UNBOUNDED (opt-in only) ----------------------- #
def test_budget_default_unbounded(monkeypatch):
    monkeypatch.delenv("APEX_OMEGA_TOKEN_CEILING", raising=False)
    from apex_omega.eval.commit0_driver import Commit0EvalDriver
    d = Commit0EvalDriver(run_dir=tempfile.mkdtemp(), base_config={})
    assert d.engine.budget.total is None, "default must be UNBOUNDED (never optimize for cost)"
    monkeypatch.setenv("APEX_OMEGA_TOKEN_CEILING", "12345")
    d2 = Commit0EvalDriver(run_dir=tempfile.mkdtemp(), base_config={})
    assert d2.engine.budget.total == 12345, "ceiling must be opt-in via env"


# --- 0.3 per-RUN agent backstop + journal rehydration ----------------------- #
def _commit_agent(j, i):
    j.commit(input_hash=f"h{i}", kind="agent", prompt_canonical="p", model_id="m",
             vendor="v", cli_version="", scoped_inputs_hash=f"s{i}", result_status=RESULT_OK,
             structured_result={}, fs_diff_text="", usage={})


def test_journal_fresh_agent_count_counts_committed_ok_agents():
    j = Journal(tempfile.mkdtemp(), run_id="t")
    _commit_agent(j, 0)
    _commit_agent(j, 1)
    # a non-agent kind must NOT be counted
    j.commit(input_hash="hx", kind="score", prompt_canonical="", model_id="", vendor="",
             cli_version="", scoped_inputs_hash="sx", result_status=RESULT_OK,
             structured_result={}, fs_diff_text="", usage={})
    assert j.fresh_agent_count() == 2


def test_agent_backstop_is_per_run_rehydrated():
    rd = tempfile.mkdtemp()
    j = Journal(rd, run_id="t")
    _commit_agent(j, 0)
    _commit_agent(j, 1)                                   # simulate a prior process: 2 fresh agents
    eng = Engine(rd, run_id="t", max_total_agents=3)
    assert eng.agents_used() == 2, "engine must rehydrate the per-run agent tally from the journal"

    def ok_runner(task):
        return ExecResult(ok=True, finalization_status="completed")

    r1 = eng.agent(ScopedTask(prompt="x", scoped_inputs={"n": "A"}), ok_runner, node_id="A")
    assert r1.ok                                          # 3rd overall == ceiling, still admitted
    r2 = eng.agent(ScopedTask(prompt="y", scoped_inputs={"n": "B"}), ok_runner, node_id="B")
    assert r2.finalization_status == "infra_nonresult" and "max_total_agents" in (r2.error or "")


# --- 0.4 parallel fan-out cap (4096) ---------------------------------------- #
def test_parallel_4096_cap():
    eng = Engine(tempfile.mkdtemp(), run_id="t")
    assert eng.parallel([lambda: 1] * 8) == [1] * 8       # under the cap: fine
    with pytest.raises(FailLoud):
        eng.parallel([lambda: 1] * 4097)


# --- 0.2 per-agent wall DECOUPLED from the cell wall ------------------------ #
import subprocess  # noqa: E402

from apex_omega.autogen import autosolve  # noqa: E402
from apex_omega.autogen.context import OrchestrationContext  # noqa: E402
from apex_omega.executor.fake import FakeExecutor  # noqa: E402
from apex_omega.kernel.verify import VerificationResult  # noqa: E402
from apex_omega.workflows.best_of_n import WorkerSpec  # noqa: E402


def _git_repo() -> str:
    d = Path(tempfile.mkdtemp()) / "repo"
    d.mkdir()
    (d / "mod.py").write_text("def f():\n    return 1\n")
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-qm", "base"]):
        subprocess.run(c, cwd=d, check=True, capture_output=True)
    return str(d)


def _mk_ctx(timeout_seconds, difficulty):
    return OrchestrationContext(
        Engine(tempfile.mkdtemp(), run_id="t"), executor=FakeExecutor(),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")], source_repo=_git_repo(),
        base_commit=None, score_fn=lambda wt: None, prompt_builder=lambda c, i, s: "x",
        repo_map={"difficulty": difficulty}, timeout_seconds=timeout_seconds,
    )


def test_per_agent_timeout_decoupled_from_cell():
    ctx = _mk_ctx(3600, "medium")
    assert ctx.timeout_seconds == 3600
    assert ctx.per_agent_timeout_seconds == 2400 and ctx.per_agent_timeout_seconds < ctx.timeout_seconds
    assert _mk_ctx(3600, "hard").per_agent_timeout_seconds == 3000


def test_per_agent_timeout_none_when_cell_unbounded():
    assert _mk_ctx(None, "medium").per_agent_timeout_seconds is None   # unbounded -> agents run to completion


def test_per_agent_watchdog_excludes_and_reruns():
    # 0.2c: a hung/non-vendor agent is abandoned for SELECTION (heartbeat_timeout ->
    # infra_nonresult -> excluded, NOT a journal hit -> re-runs), without killing the cell.
    import time as _time
    eng = Engine(tempfile.mkdtemp(), run_id="t")
    calls = {"n": 0}

    def slow(task):
        calls["n"] += 1
        _time.sleep(3)                                    # > the watchdog wall
        return ExecResult(ok=True, finalization_status="completed")

    r1 = eng.agent(ScopedTask(prompt="x", heartbeat_timeout_seconds=0.3, scoped_inputs={"k": "v"}),
                   slow, node_id="A")
    # canonical infra_nonresult (review-fix #11: not the off-enum "heartbeat_timeout"); the
    # watchdog detail is carried in .error.
    assert r1.finalization_status == "infra_nonresult" and not r1.ok and "heartbeat_timeout" in (r1.error or "")
    r2 = eng.agent(ScopedTask(prompt="x", heartbeat_timeout_seconds=0.3, scoped_inputs={"k": "v"}),
                   slow, node_id="A")
    assert calls["n"] == 2 and r2.finalization_status == "infra_nonresult"   # re-ran (not a cache hit)


# --- 0.5 floor-probe always banks; RESCUE gated (default OFF = stands alone) - #
def _autosolve_abstaining(monkeypatch, rescue_env):
    if rescue_env is None:
        monkeypatch.delenv("APEX_OMEGA_FLOOR_RESCUE", raising=False)
    else:
        monkeypatch.setenv("APEX_OMEGA_FLOOR_RESCUE", rescue_env)
    eng = Engine(tempfile.mkdtemp(), run_id="t")
    return autosolve(
        eng, source_repo=_git_repo(), executor=FakeExecutor(),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")],
        score_fn=lambda wt: VerificationResult(accepted=True, score=1.0, passed=1, total=1, pass_rate=1.0),
        prompt_builder=lambda c, i, s: "fix", author=True,
        author_fn=lambda rm: "def orchestrate(ctx):\n    return ctx.select([])\n",  # clean abstain
        scout_agents=0,
    )


def test_floor_rescue_off_by_default_autogen_stands_alone(monkeypatch):
    r = _autosolve_abstaining(monkeypatch, None)
    assert r["solved"] is False and r["floor_rescued"] is False   # abstained authored plan stands alone


def test_floor_rescue_on_when_enabled(monkeypatch):
    r = _autosolve_abstaining(monkeypatch, "1")
    assert r["solved"] is True and r["floor_rescued"] is True      # verified floor rescues the abstain
