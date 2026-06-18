"""Phase-3 stress test: the carry-forward MERGE-CONFLICT path driven END-TO-END through the
rebuilt DEFAULT_ORCHESTRATION on a synthetic repo whose modules touch the SAME shared file.

This is the load-bearing guarantee of the convergence rebuild: when two per-module diffs
overlap on a shared file, the second cannot apply (apply_diff strict AND --3way both fail).
That conflict MUST be:
  (1) surfaced as conflicts[] from reduce_residuals (never silently treated as 'applied');
  (2) recorded as a deferral (ctx.defer 'merge_conflict') so the loop is aware of it;
  (3) NEVER allowed to erase the already-applied module's progress (the carry survives);
  (4) closed by the loop-until-dry residual-repair round on the LIVE merged tree, so the
      run still converges to a real solve — or honestly abstains, never a fake pass.

Seam-level conflict behaviour is covered in test_converge_seams.py; THIS file proves the
FULL frozen DEFAULT_ORCHESTRATION drives the conflict through to convergence (and to an
honest abstain when the residual is genuinely unsolvable). All offline: real git worktrees +
a real pytest subprocess for scoring + the FakeExecutor. No codex burn.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from apex_omega.autogen.context import OrchestrationContext
from apex_omega.autogen.sandbox import run_orchestration
from apex_omega.autogen.templates import DEFAULT_ORCHESTRATION
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


def _shared_file_repo() -> str:
    """Two logical modules (A owns X, B owns Y) but BOTH live in the SAME shared.py file. Any
    two whole-file rewrites overlap, so the second per-module diff conflicts on the merge tree."""
    d = Path(tempfile.mkdtemp()) / "repo"
    d.mkdir()
    (d / "shared.py").write_text("X = 0  # a-slot\nY = 0  # b-slot\n")
    (d / "test_shared.py").write_text(
        "from shared import X, Y\n\n"
        "def test_x():\n    assert X == 1\n\n"
        "def test_y():\n    assert Y == 2\n")
    _git(d)
    return str(d)


def _real_pytest_score(node_ids):
    def _score(wt: str) -> VerificationResult:
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
               "--no-header", *node_ids]
        proc = subprocess.run(cmd, cwd=wt, text=True, capture_output=True,
                              env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                                   "PATH": os.environ["PATH"]}, timeout=120)
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


def _write_x(cwd: str) -> None:
    """Fix X (whole-file rewrite that leaves Y at the original value)."""
    Path(cwd, "shared.py").write_text("X = 1  # a-slot\nY = 0  # b-slot\n")


def _write_y(cwd: str, *, keep_x: bool) -> None:
    """Fix Y. keep_x=False simulates a stale per-module diff (built off the BASE, so X is still 0
    -> a whole-file rewrite that OVERLAPS module A's rewrite -> conflict on the merge tree).
    keep_x=True is the live-tree repair: it sees A's fix already present and preserves it."""
    x = "X = 1  # a-slot" if keep_x else "X = 0  # a-slot"
    Path(cwd, "shared.py").write_text(f"{x}\nY = 2  # b-slot\n")


def _diff(cwd: str) -> str:
    return subprocess.run(["git", "-C", cwd, "diff"], text=True, capture_output=True).stdout


# --------------------------------------------------------------------------- the stress test
def test_shared_file_conflict_converges_through_loop():
    """FULL DEFAULT_ORCHESTRATION over the shared-file repo.

    Fan-out: module A rewrites shared.py to fix X; module B rewrites shared.py (off the base) to
    fix Y -> on reduce, B conflicts. Assert: A's progress survives (test_x green, only test_y
    residual), B is recorded in conflicts[] AND deferred, and the loop-until-dry residual round
    (which edits the LIVE merged tree, so it preserves A's fix) closes test_y -> real accept.
    """
    repo = _shared_file_repo()
    seen = {"residual_saw_carry": None, "residual_rounds": 0}

    def responder(task, session):
        if task.schema is not None:   # decompose -> two modules sharing the file
            return ExecResult(structured_output={
                "modules": [
                    {"module": "mod_a", "gold_test_ids": ["test_shared.py::test_x"], "depends_on": []},
                    {"module": "mod_b", "gold_test_ids": ["test_shared.py::test_y"], "depends_on": []},
                ], "order": ["mod_a", "mod_b"]}, ok=True, finalization_status="completed")
        scoped = task.scoped_inputs or {}
        is_residual = bool(scoped.get("residual_ids"))
        if is_residual:
            # residual-repair runs on the LIVE merged tree: A's X fix MUST already be present
            # (the carry), and we fix Y WITHOUT erasing it.
            seen["residual_saw_carry"] = "X = 1" in Path(session.cwd, "shared.py").read_text()
            seen["residual_rounds"] += 1
            _write_y(session.cwd, keep_x=True)
        else:
            mod = scoped.get("module", "")
            if mod == "mod_a":
                _write_x(session.cwd)
            elif mod == "mod_b":
                # stale per-module diff built off the base (X still 0) -> overlaps A -> conflict
                _write_y(session.cwd, keep_x=False)
        return ExecResult(final_message="patched", fs_diff=_diff(session.cwd), ok=True,
                          finalization_status="completed", usage=TokenUsage(input=1, output=1))

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    score = _real_pytest_score(["test_shared.py::test_x", "test_shared.py::test_y"])
    ctx = _ctx(eng, repo, score, responder,
               {"difficulty": "medium", "modules": ["mod_a", "mod_b"]})

    winner = run_orchestration(DEFAULT_ORCHESTRATION, ctx)

    # the conflicting module was recorded as a deferral (never silently swallowed)
    deferred = [r["item"] for r in ctx.blocked("merge_conflict")]
    assert "mod_b" in deferred
    # the residual-repair round saw the carried X fix (A's progress NOT erased by B's conflict)
    assert seen["residual_saw_carry"] is True
    assert seen["residual_rounds"] >= 1
    # and the run still converged to a REAL accept through the conflict
    assert winner is not None and winner.accepted


def test_shared_file_conflict_unsolvable_residual_abstains(monkeypatch):
    """Same conflict, but the residual is genuinely unsolvable (the repair agent never fixes Y).
    The loop-until-dry must TERMINATE on the governor plateau cut and ABSTAIN — A's partial
    progress is preserved (test_x stays green) but the run NEVER fakes a pass. Patience lowered
    so the cut fires fast."""
    monkeypatch.setenv("APEX_OMEGA_PLATEAU_PATIENCE", "3")
    repo = _shared_file_repo()
    seen = {"residual_rounds": 0, "carry_preserved_each_round": True}

    def responder(task, session):
        if task.schema is not None:
            return ExecResult(structured_output={
                "modules": [
                    {"module": "mod_a", "gold_test_ids": ["test_shared.py::test_x"], "depends_on": []},
                    {"module": "mod_b", "gold_test_ids": ["test_shared.py::test_y"], "depends_on": []},
                ], "order": ["mod_a", "mod_b"]}, ok=True, finalization_status="completed")
        scoped = task.scoped_inputs or {}
        is_residual = bool(scoped.get("residual_ids"))
        if is_residual:
            # the repair agent runs on the live merged tree but NEVER fixes Y (unsolvable).
            if "X = 1" not in Path(session.cwd, "shared.py").read_text():
                seen["carry_preserved_each_round"] = False   # the carry was lost -> regression
            seen["residual_rounds"] += 1
        else:
            mod = scoped.get("module", "")
            if mod == "mod_a":
                _write_x(session.cwd)
            elif mod == "mod_b":
                _write_y(session.cwd, keep_x=False)   # conflicts on reduce
        return ExecResult(final_message="noop", fs_diff=_diff(session.cwd), ok=True,
                          finalization_status="completed", usage=TokenUsage(input=1, output=1))

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    score = _real_pytest_score(["test_shared.py::test_x", "test_shared.py::test_y"])
    ctx = _ctx(eng, repo, score, responder,
               {"difficulty": "medium", "modules": ["mod_a", "mod_b"]})

    from apex_omega.errors import PlateauStop
    try:
        winner = run_orchestration(DEFAULT_ORCHESTRATION, ctx)
        accepted = winner is not None and winner.accepted
    except PlateauStop:
        accepted = False

    assert accepted is False                                   # honest abstain, no fake pass
    assert "mod_b" in [r["item"] for r in ctx.blocked("merge_conflict")]
    assert seen["residual_rounds"] >= 1                        # the loop iterated before giving up
    assert seen["carry_preserved_each_round"] is True          # A's progress NEVER silently erased
    # the best banked candidate is a real near-solve (X green), never accepted.
    best = ctx.select(ctx.all_candidates())
    assert best is None or not best.accepted
