"""Regression tests for the 14 confirmed adversarial-review findings (Phases 0-3).

Each test pins a specific defect the review surfaced so it cannot silently return."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from apex_omega.autogen.context import OrchestrationContext
from apex_omega.autogen.sandbox import lint_source
from apex_omega.engine.runtime import Engine
from apex_omega.executor.fake import FakeExecutor
from apex_omega.kernel.select import Candidate
from apex_omega.kernel.verify import VerificationResult
from apex_omega.types import ExecResult, ScopedTask, TokenUsage
from apex_omega.workflows.best_of_n import WorkerSpec


def _git_repo() -> str:
    d = Path(tempfile.mkdtemp()) / "repo"
    d.mkdir()
    (d / "mod.py").write_text("def f():\n    return 1\n")
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-qm", "b"]):
        subprocess.run(c, cwd=d, check=True, capture_output=True)
    return str(d)


def _schema_responder(payload):
    def r(task, session):
        return ExecResult(final_message="ok",
                          structured_output=(payload if task.schema else None),
                          ok=True, finalization_status="completed",
                          fs_diff="d\n", usage=TokenUsage(input=1, output=1))
    return r


def _ctx_with(responder, **kw):
    return OrchestrationContext(
        Engine(tempfile.mkdtemp(), run_id="t"), executor=FakeExecutor(responder),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")], source_repo=_git_repo(),
        base_commit=None, score_fn=lambda wt: None, prompt_builder=lambda c, i, s: "x", **kw)


# --- #1 CRITICAL: lint blocks execution-gate binds at ANY nesting depth -------- #
def test_lint_blocks_nested_and_loop_accept_binds():
    bad = [
        "def orchestrate(ctx):\n    c=None\n    c.accepted, _ = True, 0\n    return c\n",        # tuple
        "def orchestrate(ctx):\n    c=None\n    [c.accepted, _] = [True, 0]\n    return c\n",     # list
        "def orchestrate(ctx):\n    c=None\n    c.accepted, *r = True, 0\n    return c\n",         # starred
        "def orchestrate(ctx):\n    c=None\n    for c.accepted in [True]:\n        pass\n    return c\n",  # for-target
        "def orchestrate(ctx):\n    c=None\n    x=[0 for c.accepted in [True]]\n    return c\n",   # comprehension
        "def orchestrate(ctx):\n    c=None\n    c.combined_score, _ = 999.0, 0\n    return c\n",   # rank-key
        "def orchestrate(ctx):\n    c=None\n    c['accepted'] = True\n    return c\n",             # subscript
    ]
    for src in bad:
        assert not lint_source(src).ok, src

    # reads + the sanctioned soft-write seam still pass
    ok_src = ("def orchestrate(ctx):\n"
              "    cands = ctx.parallel([ctx.make_attempt(0)])\n"
              "    w = ctx.select(cands)\n"
              "    keep = w.accepted if w else False\n"   # READ is fine
              "    return w\n")
    assert lint_source(ok_src).ok


# --- #2 HIGH: per-run agent tally counts FAILED dispatches (resume parity) ------ #
def test_fresh_agent_count_includes_failed_dispatches():
    eng = Engine(tempfile.mkdtemp(), run_id="t")

    def ok(t):
        return ExecResult(ok=True, finalization_status="completed")

    def bad(t):
        return ExecResult(ok=False, finalization_status="infra_nonresult")

    eng.agent(ScopedTask(prompt="a", scoped_inputs={"i": 1}), ok, node_id="a")
    eng.agent(ScopedTask(prompt="b", scoped_inputs={"i": 2}), ok, node_id="b")
    eng.agent(ScopedTask(prompt="c", scoped_inputs={"i": 3}), bad, node_id="c")
    eng.agent(ScopedTask(prompt="d", scoped_inputs={"i": 4}), bad, node_id="d")
    assert eng.agents_used() == 4
    assert eng.journal.fresh_agent_count() == 4              # failed dispatches counted
    # a resumed Engine rehydrates the FULL tally (not just the 2 OK) -> no R*ceiling drift
    assert Engine(eng.run_dir, run_id="t").agents_used() == 4


# --- #3 HIGH + #11 LOW: watchdog -> canonical infra_nonresult, sem not deadlocked #
def test_watchdog_status_and_semaphore_reclaim():
    import time as _time
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_concurrent=1)

    def slow(t):
        _time.sleep(2)
        return ExecResult(ok=True, finalization_status="completed")

    r = eng.agent(ScopedTask(prompt="x", heartbeat_timeout_seconds=0.2, scoped_inputs={"k": 1}),
                  slow, node_id="A")
    assert r.finalization_status == "infra_nonresult" and "heartbeat_timeout" in (r.error or "")
    # the slot must still be acquirable (a fast follow-up agent completes) despite the
    # abandoned worker — i.e. the reaper, not a premature release, owns the slot.
    r2 = eng.agent(ScopedTask(prompt="y", scoped_inputs={"k": 2}),
                   lambda t: ExecResult(ok=True, finalization_status="completed"), node_id="B")
    assert r2.ok


# --- #4 MED: repair-lineage ids are a deterministic function of the base id ------ #
def test_repair_id_deterministic_from_base():
    def score(wt):
        return VerificationResult(accepted=False, score=0.5, pass_rate=0.5, passed=1, failed=1, total=2)

    ctx = OrchestrationContext(
        Engine(tempfile.mkdtemp(), run_id="t"), executor=FakeExecutor(),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")], source_repo=_git_repo(),
        base_commit=None, score_fn=score, prompt_builder=lambda c, i, s: "x", repair_iters=1)
    ctx.solve_and_repair(attempt_id=5, max_iters=1)
    ids = {c.candidate_id for c in ctx.all_candidates()}
    assert "a5" in ids and "r700500" in ids   # 700000 + (5 % 1000)*100 + 0, NOT a call-time draw


# --- #5 MED: ctx.ask auto-id is deterministic -> replays on resume -------------- #
def test_ask_auto_id_deterministic_across_resume():
    rd = tempfile.mkdtemp()
    repo = _git_repo()
    calls = {"n": 0}

    def responder(task, session):
        calls["n"] += 1
        return ExecResult(final_message="ok", structured_output=({"x": 1} if task.schema else None),
                          ok=True, finalization_status="completed", usage=TokenUsage(input=1, output=1))

    def mk():
        return OrchestrationContext(
            Engine(rd, run_id="t"), executor=FakeExecutor(responder),
            worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")], source_repo=repo, base_commit=None,
            score_fn=lambda wt: None, prompt_builder=lambda c, i, s: "x")

    mk().ask("same question", schema={"type": "object"})    # auto-id (no agent_id)
    assert calls["n"] == 1
    mk().ask("same question", schema={"type": "object"})    # resume: SAME derived id -> HIT
    assert calls["n"] == 1                                   # not re-run


# --- #7 MED: inconclusive scoring -> indeterminate (re-runs, not cached fail) ---- #
def test_scoring_inconclusive_maps_to_indeterminate():
    from apex_omega.eval.scoring import verification_from_commit0_evaluation

    class _Inconclusive:
        passed = failed = errors = total = missing_expected = 0
        pass_rate = 0.0
        scored_success = False
        evaluation_status = "audit_inconclusive"
        verification_taxonomy = ""

    vr = verification_from_commit0_evaluation(_Inconclusive())
    assert vr.indeterminate is True and vr.accepted is False

    class _GenuineFail:
        passed = 1
        failed = 2
        errors = 0
        total = 3
        missing_expected = 0
        pass_rate = 0.33
        scored_success = False
        evaluation_status = "scored"
        verification_taxonomy = ""

    vr2 = verification_from_commit0_evaluation(_GenuineFail())
    assert vr2.indeterminate is False and vr2.accepted is False   # a real failure still caches


# --- #9 LOW: adversarial_verify uses a per-candidate key space (no cross-hit) ---- #
def test_adversarial_verify_distinct_keyspace_per_candidate():
    calls = {"n": 0}

    def responder(task, session):
        calls["n"] += 1
        return ExecResult(structured_output=({"refuted": False} if task.schema else None),
                          ok=True, finalization_status="completed", usage=TokenUsage(input=1, output=1))

    ctx = _ctx_with(responder)
    shared = "X" * 7000                                       # identical first 6000 bytes
    a = Candidate(candidate_id="a", accepted=True, public_signal_score=1.0,
                  diff=shared + "A", content_sha="aaaa1111")
    b = Candidate(candidate_id="b", accepted=True, public_signal_score=1.0,
                  diff=shared + "B", content_sha="bbbb2222")
    ctx.adversarial_verify(a, n=2)
    after_a = calls["n"]
    ctx.adversarial_verify(b, n=2)
    assert calls["n"] == after_a * 2                          # B ran fresh, not replayed from A


# --- #10 LOW: a failed verifier cannot LOWER the refute bar (refute_if='all') ---- #
def test_adversarial_verify_all_not_lowered_by_failed_verifier():
    def responder(task, session):
        if "CHEATED the visible suite" in (task.prompt or ""):
            return ExecResult(structured_output=({"refuted": True} if task.schema else None),
                              ok=True, finalization_status="completed", usage=TokenUsage(input=1, output=1))
        return ExecResult(ok=False, finalization_status="infra_nonresult", usage=TokenUsage(input=1, output=1))

    ctx = _ctx_with(responder)
    c = Candidate(candidate_id="a", accepted=True, public_signal_score=1.0, diff="d", content_sha="00ff")
    out = ctx.adversarial_verify(c, n=3, refute_if="all")
    assert out.accepted is True            # need=n=3; only 1 of 3 refuted -> NOT refuted


# --- sandbox-not-prompt policy: NO anti-fetch prompt limiter; attempts run with internet -- #
def test_no_anti_fetch_prompt_limiter():
    # the prompt the worker receives must NOT contain an anti-fetch instruction: cheating is
    # prevented structurally (worktree shadows site-packages), never by limiting the model.
    import apex_omega.autogen.context as ctxmod
    assert not hasattr(ctxmod, "ANTI_FETCH_POLICY")            # constant removed
    seen = {}

    def responder(task, session):
        seen["prompt"] = task.prompt
        seen["internet"] = task.internet
        return ExecResult(final_message="m", fs_diff="d\n", ok=True,
                          finalization_status="completed", usage=TokenUsage(input=1, output=1))

    ctx = OrchestrationContext(
        Engine(tempfile.mkdtemp(), run_id="t"), executor=FakeExecutor(responder),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")], source_repo=_git_repo(),
        base_commit=None, score_fn=lambda wt: VerificationResult(accepted=False, pass_rate=0.0),
        prompt_builder=lambda c, i, s: "implement the package")
    ctx.solve_attempt(attempt_id=0)
    assert "do not download" not in seen["prompt"].lower()      # no fetch prohibition
    assert "out of scope" not in seen["prompt"].lower()
    assert seen["internet"] is False                           # internet OFF (iron-tight: no egress)


# --- escape/cheat telemetry: RECORD, never kill -------------------------------- #
def test_integrity_classifier_detects_escape_fetch_and_cheat():
    from apex_omega.autogen.context import classify_attempt_integrity
    esc = classify_attempt_integrity(ExecResult(
        ok=False, finalization_status="policy_violation",
        error="CLI subprocess attempted repository discovery outside the root"))
    assert esc["attempted"] and any(s["kind"] == "sandbox_escape" for s in esc["signals"])
    fetch = classify_attempt_integrity(ExecResult(
        ok=True, finalization_status="completed", error="tried to pip install upstream from pypi"))
    assert any(s["kind"] == "fetch_attempt" for s in fetch["signals"])
    cheat = classify_attempt_integrity(ExecResult(
        ok=True, finalization_status="completed",
        fs_diff="--- a/tests/test_x.py\n+++ b/tests/test_x.py\n"))
    assert any(s["kind"] == "test_modification" for s in cheat["signals"])
    clean = classify_attempt_integrity(ExecResult(
        ok=True, finalization_status="completed", fs_diff="--- a/mod.py\n+++ b/mod.py\n"))
    assert clean["attempted"] is False                         # a normal attempt records nothing


def test_accept_path_independent_of_prompt():
    # acceptance/scoring must be keyed on the candidate DIFF, never the prompt — so prompt
    # content (incl. any guidance) can never influence whether a cell is judged solved.
    diff = "--- constant.diff ---\n"

    def responder(task, session):
        return ExecResult(final_message="m", fs_diff=diff, ok=True,
                          finalization_status="completed", usage=TokenUsage(input=1, output=1))

    calls = {"n": 0}

    def score(wt):
        calls["n"] += 1
        return VerificationResult(accepted=True, score=1.0, pass_rate=1.0, passed=1, total=1)

    ctx = OrchestrationContext(
        Engine(tempfile.mkdtemp(), run_id="t"), executor=FakeExecutor(responder),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")], source_repo=_git_repo(),
        base_commit=None, score_fn=score,
        prompt_builder=lambda c, i, s: "PROMPT-" + str(i))   # DIFFERENT prompt per attempt
    c1 = ctx.solve_attempt(attempt_id=1)
    c2 = ctx.solve_attempt(attempt_id=2)                      # different prompt, SAME diff
    assert c1.accepted and c2.accepted
    assert calls["n"] == 1                                    # score keyed on diff -> 2nd is a HIT


def test_authored_string_ids_do_not_crash_the_cell():
    # the LLM-authored orchestrator may pass an id of ANY type (observed live:
    # ctx.ask(agent_id="mimesis-scout") -> int("mimesis-scout") ValueError crashed the cell).
    from apex_omega.autogen.context import _as_int_id
    assert _as_int_id(5) == 5 and _as_int_id("7") == 7
    assert isinstance(_as_int_id("mimesis-scout"), int)                  # no crash
    assert _as_int_id("mimesis-scout") == _as_int_id("mimesis-scout")    # deterministic
    # ctx.ask tolerates a string agent_id end-to-end
    ctx = _ctx_with(_schema_responder({"x": 1}))
    out = ctx.ask("q", schema={"type": "object"}, agent_id="mimesis-scout")
    assert isinstance(out, dict)
    # solve_attempt tolerates a string attempt_id (it indexes worker specs via aid % n)
    ctx2 = OrchestrationContext(
        Engine(tempfile.mkdtemp(), run_id="t"), executor=FakeExecutor(_schema_responder(None)),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")], source_repo=_git_repo(), base_commit=None,
        score_fn=lambda wt: VerificationResult(accepted=False, pass_rate=0.0),
        prompt_builder=lambda c, i, s: "x")
    cand = ctx2.solve_attempt(attempt_id="weird-id", strategy="minimal")
    assert cand is not None                                              # ran, did not raise


def test_escape_attempt_is_recorded_but_attempt_still_scored():
    def responder(task, session):
        return ExecResult(ok=False, finalization_status="policy_violation",
                          error="attempted repository discovery outside the root",
                          fs_diff="--- a/mod.py\n+++ b/mod.py\n", usage=TokenUsage(input=1, output=1))

    eng = Engine(tempfile.mkdtemp(), run_id="t")
    ctx = OrchestrationContext(
        eng, executor=FakeExecutor(responder),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")], source_repo=_git_repo(), base_commit=None,
        score_fn=lambda wt: VerificationResult(accepted=False, pass_rate=0.4, passed=2, total=5),
        prompt_builder=lambda c, i, s: "x")
    cand = ctx.solve_attempt(attempt_id=0)
    assert cand is not None                                     # NOT killed/excluded for trying
    assert cand.meta.get("integrity", {}).get("attempted") is True
    log = Path(eng.run_dir) / "integrity_log.jsonl"
    assert log.exists() and "sandbox_escape" in log.read_text()  # recorded for cross-eval analysis
