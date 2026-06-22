"""SARP — State-Aware Adaptive Replanning (the last-mile fix). Gated APEX_OMEGA_SARP.

Covers: gate-off inertness (byte-identical), the governor ADAPT-BEFORE-CUT pre-check (hard cuts never
defused), diagnose_residual fact-check, the frontier-nontrivial threshold, bounded termination under
persistent sterility (G1/G2), stuck-stops, and the load-bearing offline near-solve repro (a sterile
near-solve that OFF abstains but ON diagnoses + re-aims with excerpts and advances to a full solve).

Offline: real git worktree + real pytest score_fn + FakeExecutor. No codex burn.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from apex_omega.autogen.context import OrchestrationContext
from apex_omega.engine.governor import RunGovernor
from apex_omega.engine.runtime import Engine
from apex_omega.executor.fake import FakeExecutor
from apex_omega.kernel.verify import VerificationResult
from apex_omega.types import ExecResult, TokenUsage
from apex_omega.workflows.best_of_n import WorkerSpec


@pytest.fixture(autouse=True)
def _clear_sarp_env():
    keys = [k for k in os.environ if k.startswith("APEX_OMEGA_SARP")]
    saved = {k: os.environ[k] for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    yield
    for k in list(os.environ):
        if k.startswith("APEX_OMEGA_SARP"):
            os.environ.pop(k, None)
    os.environ.update(saved)


# ---------------------------------------------------------------- near-solve repo (2/3 pass on base)
def _near_solve_repo() -> str:
    d = Path(tempfile.mkdtemp()) / "repo"
    d.mkdir()
    (d / "mod.py").write_text("def a():\n    return 1\n\ndef b():\n    return 2\n\ndef c():\n    return 0  # BUG\n")
    (d / "test_mod.py").write_text(
        "from mod import a, b, c\n\n"
        "def test_a():\n    assert a() == 1\n\n"
        "def test_b():\n    assert b() == 2\n\n"
        "def test_c():\n    assert c() == 3\n")
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-qm", "base"]):
        subprocess.run(cmd, cwd=d, check=True, capture_output=True)
    return str(d)


def _real_pytest_score(node_ids):
    def _score(wt: str) -> VerificationResult:
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
               "--no-header", *node_ids]
        proc = subprocess.run(cmd, cwd=wt, text=True, capture_output=True,
                              env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PATH": os.environ["PATH"]}, timeout=120)
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        passed = int((re.search(r"(\d+) passed", out) or [0, 0])[1]) if re.search(r"(\d+) passed", out) else 0
        failed = int((re.search(r"(\d+) failed", out) or [0, 0])[1]) if re.search(r"(\d+) failed", out) else 0
        failing = [f[1].split(" ")[0] for f in re.findall(r"^(FAILED|ERROR)\s+(\S+)", out, re.MULTILINE)]
        accepted = failed == 0 and passed == len(node_ids)
        return VerificationResult(accepted=accepted, score=1.0 if accepted else passed / max(1, len(node_ids)),
                                  passed=passed, failed=failed, total=len(node_ids),
                                  pass_rate=passed / max(1, len(node_ids)), failing_nodeids=failing)
    return _score


_IDS = ["test_mod.py::test_a", "test_mod.py::test_b", "test_mod.py::test_c"]


def _ctx(engine, repo, responder=None):
    return OrchestrationContext(
        engine, executor=FakeExecutor(responder),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")],
        source_repo=repo, base_commit=None, score_fn=_real_pytest_score(_IDS),
        prompt_builder=lambda c, i, s: "solve", max_agents=64, initial_agents=1,
        repo_map={"modules": ["mod"]})


# ---------------------------------------------------------------- gate OFF inertness
def test_sarp_off_is_inert():
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, _near_solve_repo())
    assert ctx._sarp_on() is False
    red = {"residual_failing_ids": ["test_mod.py::test_c"], "advanced": False,
           "accepted": False, "gold_total": 3, "failure_excerpts": "x"}
    assert ctx.sarp_step(red, [{"module": "mod", "gold_test_ids": _IDS}]) is None
    assert ctx.diagnose_residual(["test_mod.py::test_c"]) == {}
    assert ctx._sarp_wave_state_extra() == {"sarp_enabled": False}
    assert eng.agents_used() == 0


def test_sarp_off_governor_keys_inert():
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, _near_solve_repo())
    st = ctx._wave_state()
    assert st.get("sarp_enabled") is False  # merged into _wave_state, inert when off


# ---------------------------------------------------------------- governor unit
def _gov():
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=999)
    return RunGovernor(engine=eng, agent_ceiling=999, plateau_k_dry=2, harness_stall_cut=24)


def test_governor_hard_cuts_never_defused_by_sarp():
    g = _gov()
    hold = {"sarp_enabled": True, "sarp_frontier_nontrivial": True, "sarp_stuck": False,
            "sarp_rungs_remaining": 5, "sarp_total_budget_remaining": 5}
    # nonresult-streak (hard) still cuts even with SARP holding
    cont, reason = g.verdict({**hold, "nonresult_streak": 999})
    assert cont is False and reason == "cut:nonresult-streak"
    # harness-stall (hard) still cuts
    cont, reason = g.verdict({**hold, "indeterminate_streak": 999})
    assert cont is False and reason == "cut:harness-stall"


def test_governor_soft_cut_held_by_sarp_then_fires():
    g = _gov()
    sterile = {"sterile_streak": 999}
    hold = {"sarp_enabled": True, "sarp_frontier_nontrivial": True, "sarp_stuck": False,
            "sarp_rungs_remaining": 3, "sarp_total_budget_remaining": 8}
    cont, reason = g.verdict({**sterile, **hold})
    assert cont is True and reason == "continue:sarp-adapt"      # deferred
    # once rungs exhausted -> the sterile cut fires
    cont, reason = g.verdict({**sterile, **hold, "sarp_rungs_remaining": 0})
    assert cont is False and reason == "cut:sterile-diff-streak"
    # once stuck -> cut fires
    cont, reason = g.verdict({**sterile, **hold, "sarp_stuck": True})
    assert cont is False and reason == "cut:sterile-diff-streak"
    # trivial frontier -> not held
    cont, reason = g.verdict({**sterile, **hold, "sarp_frontier_nontrivial": False})
    assert cont is False and reason == "cut:sterile-diff-streak"


# ---------------------------------------------------------------- helpers
def test_frontier_nontrivial_threshold():
    os.environ["APEX_OMEGA_SARP"] = "1"
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, _near_solve_repo())
    assert ctx._sarp_frontier_nontrivial(6146, 6159) is True     # near-solve
    assert ctx._sarp_frontier_nontrivial(2, 3) is True           # 0.66 >= 0.5
    assert ctx._sarp_frontier_nontrivial(1, 100) is False        # 0.01 trivial
    assert ctx._sarp_frontier_nontrivial(0, 0) is False


def test_residual_set_sha_is_order_invariant():
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, _near_solve_repo())
    assert ctx.residual_set_sha(["a", "b"]) == ctx.residual_set_sha(["b", "a"])
    assert ctx.residual_set_sha(["a"]) != ctx.residual_set_sha(["a", "b"])


# ---------------------------------------------------------------- diagnose_residual fact-check
def test_diagnose_residual_factcheck_downgrades_ungrounded():
    os.environ["APEX_OMEGA_SARP"] = "1"

    def responder(task, session):
        props = (task.schema or {}).get("properties") or {}
        if "root_cause_class" in props:
            return ExecResult(structured_output={
                "root_cause_class": "unsolvable",          # ungrounded -> must downgrade
                "direction": "give up",
                "target_ids": ["NOT::a::real::id"],        # not in residual -> dropped
                "stuck": True,                              # ungrounded -> not honored
                "evidence_ids": []}, ok=True, finalization_status="completed",
                usage=TokenUsage(input=1, output=1))
        return ExecResult(final_message="x", ok=True)

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, _near_solve_repo(), responder=responder)
    diag = ctx.diagnose_residual(["test_mod.py::test_c"], excerpts="assert 0 == 3", n=1)
    assert diag["root_cause_class"] == "semantic_logic_bug"   # downgraded from unsolvable (ungrounded)
    assert diag["stuck"] is False                              # ungrounded stuck not honored
    assert "NOT::a::real::id" not in diag["target_ids"]        # hallucinated target dropped


# ---------------------------------------------------------------- termination (G1/G2)
def test_sarp_terminates_under_persistent_sterility():
    os.environ["APEX_OMEGA_SARP"] = "1"
    os.environ["APEX_OMEGA_SARP_TOTAL_RUNG_BUDGET"] = "3"

    def responder(task, session):
        props = (task.schema or {}).get("properties") or {}
        if "root_cause_class" in props:
            return ExecResult(structured_output={"root_cause_class": "semantic_logic_bug",
                              "direction": "x", "target_ids": ["test_mod.py::test_c"],
                              "evidence_ids": ["test_mod.py::test_c"]},
                              ok=True, finalization_status="completed", usage=TokenUsage(input=1, output=1))
        return ExecResult(final_message="noop", ok=True)   # repair writes nothing -> stays sterile

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=999)
    ctx = _ctx(eng, _near_solve_repo(), responder=responder)
    ctx._best_gold_passed = 2
    red = {"residual_failing_ids": ["test_mod.py::test_c"], "advanced": False, "accepted": False,
           "gold_total": 3, "failure_excerpts": "assert 0 == 3", "merged_diff": ""}
    mods = [{"module": "mod", "gold_test_ids": _IDS}]
    # drive the controller repeatedly (simulating the loop); it MUST terminate (set stuck) within budget
    for _ in range(20):
        ctx.sarp_step(red, mods)
        if ctx._sarp_stuck:
            break
    assert ctx._sarp_stuck is True
    assert ctx._sarp_total_used <= 3                       # G1: never exceeds the per-run ceiling
    assert eng.agents_used() < 999                         # terminated strictly before the ceiling


def test_sarp_stuck_diagnosis_stops_immediately():
    os.environ["APEX_OMEGA_SARP"] = "1"

    def responder(task, session):
        props = (task.schema or {}).get("properties") or {}
        if "root_cause_class" in props:
            return ExecResult(structured_output={"root_cause_class": "unsolvable", "direction": "x",
                              "stuck": True, "target_ids": ["test_mod.py::test_c"],
                              "evidence_ids": ["test_mod.py::test_c"]},
                              ok=True, finalization_status="completed", usage=TokenUsage(input=1, output=1))
        return ExecResult(final_message="noop", ok=True)

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=999)
    ctx = _ctx(eng, _near_solve_repo(), responder=responder)
    ctx._best_gold_passed = 2
    red = {"residual_failing_ids": ["test_mod.py::test_c"], "advanced": False, "accepted": False,
           "gold_total": 3, "failure_excerpts": "assert 0 == 3", "merged_diff": ""}
    out = ctx.sarp_step(red, [{"module": "mod", "gold_test_ids": _IDS}])
    assert out is None and ctx._sarp_stuck is True
    assert ctx._sarp_total_used == 0                       # stuck before spending any agent rung


# ---------------------------------------------------------------- the load-bearing efficacy repro
def test_sarp_targeted_reaim_advances_near_solve():
    """OFF the near-solve abstains at 2/3 (sterile); ON, diagnose_residual + the targeted re-aim rung
    (WITH excerpts + direction) fix the last test -> 3/3 accepted. The last-mile gap closes."""
    os.environ["APEX_OMEGA_SARP"] = "1"
    seen = {"excerpts": False, "sarp_prompt": False}

    def responder(task, session):
        props = (task.schema or {}).get("properties") or {}
        if "root_cause_class" in props:       # the residual-diagnosis scout
            return ExecResult(structured_output={
                "root_cause_class": "semantic_logic_bug",
                "direction": "c() must return 3", "target_symbol": "c",
                "target_ids": ["test_mod.py::test_c"], "evidence_ids": ["test_mod.py::test_c"]},
                ok=True, finalization_status="completed", usage=TokenUsage(input=1, output=1))
        # the targeted-repair agent: confirm it received the excerpts + SARP re-aim, then FIX c()
        p = task.prompt or ""
        if "SARP RE-AIM" in p or "DIAGNOSED ROOT CAUSE" in p:
            seen["sarp_prompt"] = True
        if "assert" in p or "== 3" in p:
            seen["excerpts"] = True
        Path(session.cwd, "mod.py").write_text(
            "def a():\n    return 1\n\ndef b():\n    return 2\n\ndef c():\n    return 3\n")
        return ExecResult(final_message="fixed c", ok=True, finalization_status="completed",
                          usage=TokenUsage(input=1, output=1))

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=999)
    ctx = _ctx(eng, _near_solve_repo(), responder=responder)
    ctx._best_gold_passed = 2
    red = {"residual_failing_ids": ["test_mod.py::test_c"], "advanced": False, "accepted": False,
           "gold_total": 3, "failure_excerpts": "assert c() == 3  ->  assert 0 == 3", "merged_diff": ""}
    out = ctx.sarp_step(red, [{"module": "mod", "gold_test_ids": _IDS}])
    assert out is not None, "SARP did not act on the near-solve sterile round"
    assert out.get("accepted") is True, "targeted re-aim did not reach a full solve"
    assert seen["sarp_prompt"] is True, "the repair agent did not get the SARP re-aim brief"
    assert seen["excerpts"] is True, "failure excerpts were not threaded to the repair agent"


def test_reduce_residuals_exposes_gold_total():
    """REGRESSION (the live-run bug): reduce_residuals' RESULT dict must expose gold_total — without it
    SARP's _sarp_frontier_nontrivial saw 0 and never engaged at the 6151/6159 plateau."""
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, _near_solve_repo())
    red = ctx.reduce_residuals([], carry_diff="", scope_ids=_IDS)
    assert "gold_total" in red and red["gold_total"] == 3
    assert red["gold_passed"] == 2          # 2/3 pass on the base
    assert "failure_excerpts" in red        # the other SARP plumbing field


def test_sarp_fires_via_real_reduce_path():
    """INTEGRATION: drive SARP off the REAL reduce_residuals output (not a hand-crafted red), so the
    gold_total / advanced / excerpts plumbing is exercised end-to-end. A sterile near-solve round must
    open an episode, diagnose, re-aim, and reach a full solve."""
    os.environ["APEX_OMEGA_SARP"] = "1"

    def responder(task, session):
        props = (task.schema or {}).get("properties") or {}
        if "root_cause_class" in props:
            return ExecResult(structured_output={
                "root_cause_class": "semantic_logic_bug", "direction": "c() must return 3",
                "target_ids": ["test_mod.py::test_c"], "evidence_ids": ["test_mod.py::test_c"]},
                ok=True, finalization_status="completed", usage=TokenUsage(input=1, output=1))
        # the targeted-repair agent fixes c()
        Path(session.cwd, "mod.py").write_text(
            "def a():\n    return 1\n\ndef b():\n    return 2\n\ndef c():\n    return 3\n")
        return ExecResult(final_message="fixed", ok=True, finalization_status="completed",
                          usage=TokenUsage(input=1, output=1))

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=999)
    ctx = _ctx(eng, _near_solve_repo(), responder=responder)
    mods = [{"module": "mod", "gold_test_ids": _IDS}]
    r1 = ctx.reduce_residuals([], carry_diff="", scope_ids=_IDS)   # frontier 0->2 (advanced)
    assert r1["advanced"] is True
    r2 = ctx.reduce_residuals([], carry_diff="", scope_ids=_IDS)   # 2->2 (STERILE near-solve)
    assert r2["advanced"] is False and r2["gold_total"] == 3 and r2["gold_passed"] == 2
    # The load-bearing regression guard for the live-run bug: SARP must ENGAGE off the real reduce
    # output (gold_total present -> non-trivial frontier recognized). out is not None == a rung ran;
    # before the gold_total fix this returned None (inert) despite the sterile near-solve.
    out = ctx.sarp_step(r2, mods)
    assert out is not None, "SARP did not fire on a sterile near-solve from the real reduce path"
    assert ctx._sarp_total_used >= 1, "SARP returned a result but spent no adaptation rung"
    # (the full-solve OUTCOME is covered by test_sarp_targeted_reaim_advances_near_solve; the
    # FakeExecutor does not faithfully capture the repair diff, so we assert ENGAGEMENT here.)


def test_sarp_pre_opens_episode_to_defer_cut_before_engage():
    """REGRESSION (the 2nd live-run bug): when the no-progress/sterile streak is ALREADY at the cut
    threshold on loop entry (inherited from the un-hooked fan-out), the governor cut fires at the loop
    TOP before sarp_step (loop BOTTOM) opens an episode -> SARP never engages. should_continue_waves()
    must pre-open the episode at a sterile non-trivial plateau so the cut is deferred."""
    os.environ["APEX_OMEGA_SARP"] = "1"
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=999)
    ctx = _ctx(eng, _near_solve_repo())
    ctx._best_gold_passed = 6151
    ctx._sterile_streak = 999                 # governor would fire cut:sterile-diff-streak
    ctx._sarp_last = {"residual": ["test_mod.py::test_c"], "gold_total": 6159,
                      "advanced": False, "indeterminate": False}
    assert ctx._sarp_state is None
    cont = ctx.should_continue_waves()
    assert cont is True, "SARP did not defer the cut at a sterile near-solve plateau"
    assert ctx._sarp_state is not None, "SARP episode was not pre-opened"


def test_sarp_pre_open_skips_trivial_and_advancing():
    """_sarp_maybe_open must NOT open on a trivial frontier or an advancing/indeterminate last reduce."""
    os.environ["APEX_OMEGA_SARP"] = "1"
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=999)
    ctx = _ctx(eng, _near_solve_repo())
    ctx._sterile_streak = 999
    # trivial frontier -> no open -> cut fires
    ctx._best_gold_passed = 10
    ctx._sarp_last = {"residual": ["x"], "gold_total": 6159, "advanced": False, "indeterminate": False}
    assert ctx.should_continue_waves() is False and ctx._sarp_state is None
    # advancing last reduce -> no open
    ctx._best_gold_passed = 6151
    ctx._sarp_last = {"residual": ["x"], "gold_total": 6159, "advanced": True, "indeterminate": False}
    assert ctx.should_continue_waves() is False and ctx._sarp_state is None


def test_sarp_episode_opens_in_wave_state_chokepoint():
    """REGRESSION (the 3rd live-run bug): the governor cut often fires during the un-hooked FAN-OUT
    (ctx.parallel -> _wave_verdict -> _halted) before the loop runs. The SARP pre-open MUST live in
    _wave_state() — the chokepoint BOTH ctx.parallel and should_continue_waves flow through — so SARP
    defers the cut wherever it is evaluated. This asserts _wave_state itself opens the episode."""
    os.environ["APEX_OMEGA_SARP"] = "1"
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=999)
    ctx = _ctx(eng, _near_solve_repo())
    ctx._best_gold_passed = 6151
    ctx._sarp_last = {"residual": ["test_mod.py::test_c"], "gold_total": 6159,
                      "advanced": False, "indeterminate": False}
    assert ctx._sarp_state is None
    st = ctx._wave_state()                         # the shared chokepoint (parallel + loop verdict)
    assert ctx._sarp_state is not None, "SARP episode not opened in _wave_state chokepoint"
    assert st.get("sarp_enabled") is True and st.get("sarp_frontier_nontrivial") is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
