"""Phase-2 convergence TEMPLATE (the rebuilt DEFAULT_ORCHESTRATION) end-to-end, offline.

Runs the ACTUAL frozen orchestration string through the restricted sandbox (run_orchestration)
over real git worktrees + a real pytest score_fn + the FakeExecutor, proving the new default
DECOMPOSES -> FANS-OUT per module -> REDUCES -> LOOPS-UNTIL-DRY -> accepts deterministically, and
that the SKIP-DECOMPOSITION gate keeps easy/single-module repos on the cheap best-of-N path (no
over-spawn). No codex burn.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from apex_omega.autogen.context import OrchestrationContext
from apex_omega.autogen.sandbox import run_orchestration
from apex_omega.autogen.templates import BEST_OF_N_ORCHESTRATION, DEFAULT_ORCHESTRATION
from apex_omega.engine.runtime import Engine
from apex_omega.executor.fake import FakeExecutor
from apex_omega.kernel.verify import VerificationResult
from apex_omega.types import ExecResult, TokenUsage
from apex_omega.workflows.best_of_n import WorkerSpec


# --------------------------------------------------------------------------- fixtures
def _git(d: Path) -> None:
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-qm", "base"]):
        subprocess.run(c, cwd=d, check=True, capture_output=True)


def _two_module_repo() -> str:
    d = Path(tempfile.mkdtemp()) / "repo"
    d.mkdir()
    (d / "mod_a.py").write_text("def a():\n    return 0  # BUG\n")
    (d / "mod_b.py").write_text("def b():\n    return 0  # BUG\n")
    (d / "test_a.py").write_text("from mod_a import a\n\ndef test_a():\n    assert a() == 1\n")
    (d / "test_b.py").write_text("from mod_b import b\n\ndef test_b():\n    assert b() == 2\n")
    _git(d)
    return str(d)


def _one_module_repo() -> str:
    d = Path(tempfile.mkdtemp()) / "repo"
    d.mkdir()
    (d / "only.py").write_text("def f():\n    return 0  # BUG\n")
    (d / "test_only.py").write_text("from only import f\n\ndef test_f():\n    assert f() == 1\n")
    _git(d)
    return str(d)


def _real_pytest_score(node_ids):
    import os

    def _score(wt: str) -> VerificationResult:
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
               "--no-header", *node_ids]
        proc = subprocess.run(cmd, cwd=wt, text=True, capture_output=True,
                              env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PATH": os.environ["PATH"]},
                              timeout=120)
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        passed = int((re.search(r"(\d+) passed", out) or [0, 0])[1]) if re.search(r"(\d+) passed", out) else 0
        failed = int((re.search(r"(\d+) failed", out) or [0, 0])[1]) if re.search(r"(\d+) failed", out) else 0
        failing = re.findall(r"^(FAILED|ERROR)\s+(\S+)", out, re.MULTILINE)
        failing_ids = [f[1].split(" ")[0] for f in failing]
        if not failing_ids and failed:
            failing_ids = re.findall(r"\b\S+::\S+\b", out)
        total = passed + failed
        accepted = failed == 0 and passed == len(node_ids)
        return VerificationResult(
            accepted=accepted, score=1.0 if accepted else (passed / max(1, total)),
            passed=passed, failed=failed, total=len(node_ids),
            pass_rate=passed / max(1, len(node_ids)), failing_nodeids=failing_ids)
    return _score


def _ctx(engine, repo, score_fn, responder, repo_map):
    return OrchestrationContext(
        engine, executor=FakeExecutor(responder),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")],
        source_repo=repo, base_commit=None, score_fn=score_fn,
        prompt_builder=lambda c, i, s: "solve", max_agents=32, initial_agents=1,
        repo_map=repo_map)


# --------------------------------------------------------------------------- decompose -> converge
def test_default_template_decomposes_fanout_reduces_to_accept():
    """The rebuilt DEFAULT_ORCHESTRATION over a medium 2-module repo: decompose -> per-module
    fan-out -> reduce -> accept, run through the real restricted sandbox."""
    repo = _two_module_repo()
    phases = []

    def responder(task, session):
        if task.schema is not None:    # decompose
            return ExecResult(structured_output={
                "modules": [
                    {"module": "mod_a", "gold_test_ids": ["test_a.py::test_a"], "depends_on": []},
                    {"module": "mod_b", "gold_test_ids": ["test_b.py::test_b"], "depends_on": []},
                ], "order": ["mod_a", "mod_b"]}, ok=True, finalization_status="completed")
        mod = (task.scoped_inputs or {}).get("module", "")
        if mod == "mod_a":
            Path(session.cwd, "mod_a.py").write_text("def a():\n    return 1\n")
        elif mod == "mod_b":
            Path(session.cwd, "mod_b.py").write_text("def b():\n    return 2\n")
        diff = subprocess.run(["git", "-C", session.cwd, "diff"], text=True,
                              capture_output=True).stdout
        return ExecResult(final_message=f"patched {mod}", fs_diff=diff, ok=True,
                          finalization_status="completed", usage=TokenUsage(input=1, output=1))

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    eng.phase = lambda title, _p=phases, _o=eng.phase: (_p.append(title), _o(title))[1]
    score = _real_pytest_score(["test_a.py::test_a", "test_b.py::test_b"])
    ctx = _ctx(eng, repo, score, responder, {"difficulty": "medium", "modules": ["mod_a", "mod_b"]})

    winner = run_orchestration(DEFAULT_ORCHESTRATION, ctx)
    assert winner is not None and winner.accepted          # decompose+fanout+reduce reached green
    assert "scope" in phases and "fanout" in phases and "reduce" in phases   # the shape ran
    # the per-module fan-out spawned a distinct agent per module (plus the read-only decompose).
    module_cands = [c for c in ctx.all_candidates() if (c.meta or {}).get("module")]
    assert {(c.meta or {}).get("module") for c in module_cands} == {"mod_a", "mod_b"}


def test_default_template_loops_until_dry_on_residual():
    """A module that needs TWO passes: the first fan-out greens mod_a only; mod_b is solved by the
    loop-until-dry residual-repair round. Proves the loop iterates on the EXACT residual and
    carries the best partial forward (off-by-K close)."""
    repo = _two_module_repo()
    state = {"b_round": 0}

    def responder(task, session):
        if task.schema is not None:
            return ExecResult(structured_output={
                "modules": [
                    {"module": "mod_a", "gold_test_ids": ["test_a.py::test_a"], "depends_on": []},
                    {"module": "mod_b", "gold_test_ids": ["test_b.py::test_b"], "depends_on": []},
                ], "order": ["mod_a", "mod_b"]}, ok=True, finalization_status="completed")
        # mod_a always solved; mod_b only solved on the residual-repair round (residual_repair=True)
        is_residual = bool((task.scoped_inputs or {}).get("residual_ids"))
        mod = (task.scoped_inputs or {}).get("module", "")
        if mod == "mod_a":
            Path(session.cwd, "mod_a.py").write_text("def a():\n    return 1\n")
        elif mod == "mod_b" and not is_residual:
            pass  # first fan-out: mod_b agent fails to fix it (leaves the bug)
        if is_residual:
            # residual repair on the live merged tree: mod_a already fixed (carry), fix mod_b.
            assert "return 1" in Path(session.cwd, "mod_a.py").read_text()  # carry present
            Path(session.cwd, "mod_b.py").write_text("def b():\n    return 2\n")
            state["b_round"] += 1
        diff = subprocess.run(["git", "-C", session.cwd, "diff"], text=True,
                              capture_output=True).stdout
        return ExecResult(final_message=f"patched {mod}", fs_diff=diff, ok=True,
                          finalization_status="completed", usage=TokenUsage(input=1, output=1))

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    score = _real_pytest_score(["test_a.py::test_a", "test_b.py::test_b"])
    ctx = _ctx(eng, repo, score, responder, {"difficulty": "medium", "modules": ["mod_a", "mod_b"]})

    winner = run_orchestration(DEFAULT_ORCHESTRATION, ctx)
    assert winner is not None and winner.accepted
    assert state["b_round"] >= 1     # the loop-until-dry residual round actually ran and closed b


# --------------------------------------------------------------------------- SPFG+ honest stop
def test_unsolvable_residual_stops_without_faking_pass(monkeypatch):
    """A deliberately-unsolvable residual (the repair agent never fixes mod_b): the loop-until-dry
    must TERMINATE on the governor's plateau cut (no progress) and ABSTAIN — never fake a pass.
    Patience is lowered so the test is fast."""
    monkeypatch.setenv("APEX_OMEGA_PLATEAU_PATIENCE", "3")
    repo = _two_module_repo()
    rounds = {"n": 0}

    def responder(task, session):
        if task.schema is not None:
            return ExecResult(structured_output={
                "modules": [
                    {"module": "mod_a", "gold_test_ids": ["test_a.py::test_a"], "depends_on": []},
                    {"module": "mod_b", "gold_test_ids": ["test_b.py::test_b"], "depends_on": []},
                ], "order": ["mod_a", "mod_b"]}, ok=True, finalization_status="completed")
        mod = (task.scoped_inputs or {}).get("module", "")
        is_residual = bool((task.scoped_inputs or {}).get("residual_ids"))
        if mod == "mod_a":
            Path(session.cwd, "mod_a.py").write_text("def a():\n    return 1\n")
        if is_residual:
            rounds["n"] += 1   # the repair agent runs but NEVER fixes mod_b (unsolvable residual)
        diff = subprocess.run(["git", "-C", session.cwd, "diff"], text=True,
                              capture_output=True).stdout
        return ExecResult(final_message="noop", fs_diff=diff, ok=True,
                          finalization_status="completed", usage=TokenUsage(input=1, output=1))

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    score = _real_pytest_score(["test_a.py::test_a", "test_b.py::test_b"])
    ctx = _ctx(eng, repo, score, responder, {"difficulty": "medium", "modules": ["mod_a", "mod_b"]})

    # the orchestration raises CutLosses/PlateauStop once the governor halts (caught by the host
    # autosolve normally; here we assert it stops and never accepts).
    from apex_omega.errors import PlateauStop
    try:
        winner = run_orchestration(DEFAULT_ORCHESTRATION, ctx)
        accepted = winner is not None and winner.accepted
    except PlateauStop:
        accepted = False
    assert accepted is False                 # honest abstain — no fake pass on an unsolvable residual
    assert rounds["n"] >= 1                   # the loop did iterate before giving up
    # the best banked candidate is a real near-solve (mod_a green), never accepted.
    best = ctx.select(ctx.all_candidates())
    assert best is None or not best.accepted


# --------------------------------------------------------------------------- skip-decomposition gate
def test_easy_repo_skips_decomposition_no_overspawn():
    """An easy repo MUST skip decomposition and fall through to best-of-N — no decompose agent is
    ever spawned (the over-spawn cost guard)."""
    repo = _one_module_repo()
    spawned_schema = {"n": 0}

    def responder(task, session):
        if task.schema is not None:
            spawned_schema["n"] += 1     # a decompose/ask agent — must NOT happen on easy
            return ExecResult(structured_output={"modules": []}, ok=True)
        Path(session.cwd, "only.py").write_text("def f():\n    return 1\n")
        diff = subprocess.run(["git", "-C", session.cwd, "diff"], text=True,
                              capture_output=True).stdout
        return ExecResult(final_message="patched", fs_diff=diff, ok=True,
                          finalization_status="completed", usage=TokenUsage(input=1, output=1))

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    score = _real_pytest_score(["test_only.py::test_f"])
    ctx = _ctx(eng, repo, score, responder, {"difficulty": "easy", "modules": ["only"]})

    winner = run_orchestration(DEFAULT_ORCHESTRATION, ctx)
    assert spawned_schema["n"] == 0          # decomposition was SKIPPED (no read-only scope agent)
    assert winner is not None and winner.accepted


def test_single_module_repo_skips_decomposition():
    """A medium repo that decomposes to <=1 module also falls through to best-of-N (no fan-out)."""
    repo = _one_module_repo()

    def responder(task, session):
        if task.schema is not None:      # decompose returns a single module -> skip-gate trips
            return ExecResult(structured_output={
                "modules": [{"module": "only", "gold_test_ids": ["test_only.py::test_f"]}],
                "order": ["only"]}, ok=True, finalization_status="completed")
        Path(session.cwd, "only.py").write_text("def f():\n    return 1\n")
        diff = subprocess.run(["git", "-C", session.cwd, "diff"], text=True,
                              capture_output=True).stdout
        return ExecResult(final_message="patched", fs_diff=diff, ok=True,
                          finalization_status="completed", usage=TokenUsage(input=1, output=1))

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    score = _real_pytest_score(["test_only.py::test_f"])
    ctx = _ctx(eng, repo, score, responder, {"difficulty": "medium", "modules": ["only"]})

    winner = run_orchestration(DEFAULT_ORCHESTRATION, ctx)
    assert winner is not None and winner.accepted
    # no module-scoped fan-out candidate was produced (we fell through to best-of-N solve attempts)
    assert not any((c.meta or {}).get("module") for c in ctx.all_candidates())


# --------------------------------------------------------------------------- best-of-n still works
def test_best_of_n_workflow_resolves_and_runs():
    """ctx.workflow('default-best-of-n') still resolves to the cheap path and accepts."""
    repo = _one_module_repo()

    def responder(task, session):
        Path(session.cwd, "only.py").write_text("def f():\n    return 1\n")
        diff = subprocess.run(["git", "-C", session.cwd, "diff"], text=True,
                              capture_output=True).stdout
        return ExecResult(final_message="patched", fs_diff=diff, ok=True,
                          finalization_status="completed", usage=TokenUsage(input=1, output=1))

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    score = _real_pytest_score(["test_only.py::test_f"])
    ctx = _ctx(eng, repo, score, responder, {"difficulty": "easy"})
    winner = run_orchestration(BEST_OF_N_ORCHESTRATION, ctx)
    assert winner is not None and winner.accepted
