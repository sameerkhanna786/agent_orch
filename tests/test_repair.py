"""Test-driven repair lineage (additional work beyond a flat rollout).

``solve_and_repair`` runs a base attempt then, if it is genuine-but-incomplete,
repair passes seeded by the failing tests — stopping on accept / plateau / no-signal
/ cap. With ``max_iters=0`` it is exactly ``solve_attempt`` (never worse than flat
best-of-N). The anti-fetch policy is always appended to solver prompts.
"""

from __future__ import annotations

import subprocess
import tempfile
import threading
from pathlib import Path

from apex_omega.autogen.context import OrchestrationContext
from apex_omega.engine.runtime import Engine
from apex_omega.executor.fake import FakeExecutor
from apex_omega.kernel.verify import VerificationResult
from apex_omega.types import ExecResult, TokenUsage
from apex_omega.workflows.best_of_n import WorkerSpec


def _git_repo() -> str:
    d = Path(tempfile.mkdtemp()) / "repo"
    d.mkdir()
    (d / "mod.py").write_text("def f():\n    pass\n")
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-qm", "base"]):
        subprocess.run(c, cwd=d, check=True, capture_output=True)
    return str(d)


_RESP_N = {"n": 0}


def _responder(task, session):
    # Distinct fs_diff per call: real attempts produce distinct patches, so each gets its
    # own journaled score. (A constant diff would correctly cache-HIT under Backbone 1.1
    # score-journaling and make a repair reuse the base's score.)
    _RESP_N["n"] += 1
    return ExecResult(final_message="edit",
                      fs_diff=f"--- a/mod.py\n+++ b/mod.py\n@@ -1 +1 @@\n+attempt{_RESP_N['n']}\n",
                      ok=True, finalization_status="completed", usage=TokenUsage(input=1, output=1))


def _ctx(engine, score_fn, repair_iters=2):
    return OrchestrationContext(
        engine, executor=FakeExecutor(_responder),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")],
        source_repo=_git_repo(), base_commit=None, score_fn=score_fn,
        prompt_builder=lambda c, i, s: "fix", max_agents=8, initial_agents=1,
        repair_iters=repair_iters,
    )


def test_repair_lineage_fixes_incomplete_base():
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=200)
    state = {"n": 0, "lock": threading.Lock()}

    def score_fn(wt):
        with state["lock"]:
            state["n"] += 1
            n = state["n"]
        if n == 1:  # base attempt: genuine-but-incomplete -> repairable
            return VerificationResult(accepted=False, score=0.5, passed=1, failed=1, total=2,
                                      pass_rate=0.5, failing_nodeids=["test_b"])
        return VerificationResult(accepted=True, score=1.0, passed=2, total=2, pass_rate=1.0)

    winner = _ctx(eng, score_fn).solve_and_repair(attempt_id=0, max_iters=2)
    assert winner is not None and winner.accepted          # the repair pass solved it
    assert state["n"] == 2                                  # base + ONE repair (stopped on accept)
    assert winner.candidate_id.startswith("r")             # winner is the repair candidate


def test_repair_skips_when_base_has_no_signal():
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=200)
    state = {"n": 0, "lock": threading.Lock()}

    def score_fn(wt):
        with state["lock"]:
            state["n"] += 1
        return VerificationResult(accepted=False, score=0.0, passed=0, total=2, pass_rate=0.0)

    winner = _ctx(eng, score_fn).solve_and_repair(attempt_id=0, max_iters=2)
    assert state["n"] == 1                  # pass_rate==0 -> nothing to build on -> NO repair fired
    assert winner is not None and not winner.accepted


def test_repair_stops_on_plateau():
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=200)
    state = {"n": 0, "lock": threading.Lock()}

    def score_fn(wt):
        with state["lock"]:
            state["n"] += 1
        # always 0.5, never improving -> one repair, then plateau stop (no endless spend)
        return VerificationResult(accepted=False, score=0.5, passed=1, total=2, pass_rate=0.5,
                                  failing_nodeids=["test_b"])

    winner = _ctx(eng, score_fn).solve_and_repair(attempt_id=0, max_iters=5)
    assert state["n"] == 2                  # base + ONE repair, then plateau (not 6)
    assert winner is not None and not winner.accepted


def test_max_iters_zero_equals_single_attempt():
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=200)
    state = {"n": 0, "lock": threading.Lock()}

    def score_fn(wt):
        with state["lock"]:
            state["n"] += 1
        return VerificationResult(accepted=False, score=0.5, passed=1, total=2, pass_rate=0.5,
                                  failing_nodeids=["test_b"])

    winner = _ctx(eng, score_fn).solve_and_repair(attempt_id=0, max_iters=0)
    assert state["n"] == 1                  # exactly one attempt, no repair (== solve_attempt)
    assert winner is not None


def test_repair_iters_ceiling_clamps_repair_off():
    # Tier-P0b: repair is OFF by default — the repair_iters ceiling (0) clamps any
    # solve_and_repair request down to a single attempt (flat best-of-N).
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=200)
    state = {"n": 0, "lock": threading.Lock()}

    def score_fn(wt):
        with state["lock"]:
            state["n"] += 1
        return VerificationResult(accepted=False, score=0.5, passed=1, total=2, pass_rate=0.5,
                                  failing_nodeids=["test_b"])

    # repair_iters=0 -> even an explicit max_iters=5 is clamped to 0 (no repair)
    _ctx(eng, score_fn, repair_iters=0).solve_and_repair(attempt_id=0, max_iters=5)
    assert state["n"] == 1


def test_acceptance_checkpoint_written_on_accept():
    # Tier-1.1: a verified-accepted attempt banks a checkpoint to engine.run_dir the
    # instant it passes, so a later cell-wall kill cannot discard the solve.
    import json as _json
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=200)
    ctx = _ctx(eng, lambda wt: VerificationResult(accepted=True, score=1.0, passed=1,
                                                  total=1, pass_rate=1.0))
    cand = ctx.solve_attempt(attempt_id=0)
    assert cand is not None and cand.accepted
    cp = Path(eng.run_dir) / "accepted_checkpoint.json"
    assert cp.exists(), "no acceptance checkpoint written on accept"
    rec = _json.loads(cp.read_text())
    assert rec["accepted"] is True and rec["candidate_id"] == "a0"


def test_no_checkpoint_when_not_accepted():
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=200)
    ctx = _ctx(eng, lambda wt: VerificationResult(accepted=False, score=0.5, passed=1,
                                                  total=2, pass_rate=0.5))
    ctx.solve_attempt(attempt_id=0)
    assert not (Path(eng.run_dir) / "accepted_checkpoint.json").exists()


def test_solve_and_repair_accepts_prompt_and_tolerates_stray_kwargs():
    # Regression for the run-4 break: an AUTHORED orchestrator called
    # ctx.solve_and_repair(..., prompt=...) and crashed the whole cell with TypeError.
    # solve_and_repair must accept solve_attempt's kwargs (incl. prompt) AND ignore
    # unknown kwargs rather than raise.
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=200)
    seen = {"prompts": []}

    def responder(task, session):
        seen["prompts"].append(task.prompt)
        return ExecResult(final_message="edit", fs_diff="--- a/mod.py\n+++ b/mod.py\n",
                          ok=True, finalization_status="completed", usage=TokenUsage(input=1, output=1))

    ctx = OrchestrationContext(
        eng, executor=FakeExecutor(responder),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")], source_repo=_git_repo(),
        base_commit=None,
        score_fn=lambda wt: VerificationResult(accepted=True, score=1.0, passed=1, total=1, pass_rate=1.0),
        prompt_builder=lambda c, i, s: "default", max_agents=8, initial_agents=1,
    )
    winner = ctx.solve_and_repair(attempt_id=0, prompt="CUSTOM-PROMPT", max_iters=0,
                                  foo="bar", lens=2)  # stray kwargs must NOT raise
    assert winner is not None and winner.accepted
    assert any("CUSTOM-PROMPT" in p for p in seen["prompts"])              # custom prompt used
    # (no anti-fetch suffix anymore — sandbox-not-prompt policy; cheating is prevented
    # structurally by worktree-shadowing, not by limiting the model's prompt)
