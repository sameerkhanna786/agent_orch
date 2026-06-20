"""Ralph-wiggum baseline + the cut-losses detector applied to it.

Ralph is a vanilla iterate-until-done loop: ONE sequential lineage re-running the IDENTICAL
prompt each turn (NO failing-test feedback / excerpts / diff-paste — that would be Reflexion)
in a PERSISTENT workspace (the accumulated diff is pre-applied each turn), NO scout/author/
patterns. It shares the SAME governor cut-losses detector as omega, so naive persistence stops
the instant it stops making progress."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from apex_omega.autogen.architect import author_orchestration
from apex_omega.autogen.context import OrchestrationContext
from apex_omega.engine.runtime import Engine
from apex_omega.executor.fake import FakeExecutor
from apex_omega.kernel.verify import VerificationResult
from apex_omega.types import ExecResult, TokenUsage
from apex_omega.workflows.best_of_n import WorkerSpec


def _git_repo() -> str:
    d = Path(tempfile.mkdtemp()) / "repo"
    d.mkdir()
    (d / "mod.py").write_text("def f():\n    return 0\n")
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-qm", "base"]):
        subprocess.run(c, cwd=d, check=True, capture_output=True)
    return str(d)


def _specs():
    return [WorkerSpec("codex_cli", "gpt-5.5")]


def _ctx(score_fn, responder, **kw):
    return OrchestrationContext(
        Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=200),
        executor=FakeExecutor(responder), worker_specs=_specs(), source_repo=_git_repo(),
        base_commit=None, score_fn=score_fn, prompt_builder=lambda c, i, s: "x", **kw)


def _diff(n: int) -> str:
    # a REAL git diff vs the base mod.py (return 0). Faithful ralph PRE-APPLIES the carried diff
    # each turn, so the test must hand it a valid applyable diff (not a fake string). Every _diff(n)
    # is base->return-n, so it always applies cleanly onto a fresh base worktree.
    return ("diff --git a/mod.py b/mod.py\n--- a/mod.py\n+++ b/mod.py\n"
            "@@ -1,2 +1,2 @@\n def f():\n-    return 0\n+    return %d\n" % n)


def test_ralph_orchestration_is_frozen_directly(monkeypatch):
    # APEX_OMEGA_ORCHESTRATION=ralph freezes the fixed ralph workflow (no scout, no author).
    monkeypatch.setenv("APEX_OMEGA_ORCHESTRATION", "ralph")
    eng = Engine(tempfile.mkdtemp(), run_id="t")
    fw = author_orchestration(eng, executor=FakeExecutor(), worker_specs=_specs(),
                              repo_map={}, author=False)
    assert fw.origin == "ralph"
    assert "ralph_loop" in fw.source and "orchestrate(ctx)" in fw.source


def test_ralph_loop_iterates_until_accept():
    # score improves across iterations and accepts on the 3rd -> ralph persists and SOLVES
    # (it is not cut while still improving). Each iteration writes a DISTINCT diff.
    seq = iter([
        VerificationResult(accepted=False, score=0.4, passed=2, failed=3, total=5, pass_rate=0.4),
        VerificationResult(accepted=False, score=0.6, passed=3, failed=2, total=5, pass_rate=0.6),
        VerificationResult(accepted=True, score=1.0, passed=5, failed=0, total=5, pass_rate=1.0),
    ])
    last = {"v": None}

    def score_fn(wt):
        try:
            last["v"] = next(seq)
        except StopIteration:
            pass
        return last["v"]

    n = {"i": 0}

    def responder(task, session):
        n["i"] += 1
        Path(session.cwd, "mod.py").write_text(f"def f():\n    return {n['i']}\n")
        # distinct REAL diff each turn -> distinct score-journal key (the score sequence advances)
        # AND a valid diff the NEXT iteration can pre-apply as the carried workspace.
        return ExecResult(final_message="patched", ok=True, finalization_status="completed",
                          fs_diff=_diff(n["i"]), usage=TokenUsage(input=1, output=1))

    ctx = _ctx(score_fn, responder)
    winner = ctx.ralph_loop()
    assert winner is not None and winner.accepted is True
    assert ctx.agents_used() == 3                 # 3 sequential iterations, no premature cut
    assert ctx._halted is False                   # solved before any cut


def test_ralph_loop_cut_when_stuck_and_does_not_run_forever():
    # never accepts AND emits an identical diff every turn -> ralph is cut for non-progress
    # (the budget-aware plateau / sterile streak) instead of burning the whole budget.
    stuck = VerificationResult(accepted=False, score=0.0, passed=0, failed=5, total=5, pass_rate=0.0)

    def score_fn(wt):
        return stuck

    def responder(task, session):
        Path(session.cwd, "mod.py").write_text("def f():\n    return 99\n")
        # IDENTICAL real diff every turn -> applies cleanly as the carry AND repeats -> sterile-diff
        # hard cut (the agent keeps producing the same non-progressing patch).
        return ExecResult(final_message="x", ok=True, finalization_status="completed",
                          fs_diff=_diff(99), usage=TokenUsage(input=1, output=1))

    ctx = _ctx(score_fn, responder, max_agents=100)
    winner = ctx.ralph_loop()
    assert winner is None or winner.accepted is False
    assert ctx._halted is True
    assert ctx._halt_reason.startswith("cut:")     # a genuine non-progress FAILURE
    assert ctx.agents_used() <= 10                  # cut early, NOT run to the 100 ceiling
