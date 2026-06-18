"""Phase 0/1 of the dynamic-workflow rebuild: the harness INDETERMINATE
classification fix + the three flag flips.

All offline, no codex burn.

(1) HARNESS scoring fix — a pytest plugin-abort (rc=4 usage error before
    collection), a collection error, or a native interpreter crash (segfault /
    abort / signal: rc<0 or 134-139) must be classified INDETERMINATE
    (environment/harness failure), never scored as a real 0. KEEP the
    all-gold-ids accept gate (a real partial stays a real residual).

(2) THREE FLAG FLIPS — repair_iters default 0->2, repair excerpts ON by default,
    model_reasoning_effort pinned per-exec as an explicit codex ``-c`` flag
    (xhigh edit / high read-only).
"""

from __future__ import annotations

import subprocess
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from apex_omega.autogen.context import OrchestrationContext
from apex_omega.eval.scoring import verification_from_commit0_evaluation
from apex_omega.engine.runtime import Engine
from apex_omega.executor.fake import FakeExecutor
from apex_omega.kernel.verify import VerificationResult
from apex_omega.types import ExecResult, TokenUsage
from apex_omega.workflows.best_of_n import WorkerSpec


# --------------------------------------------------------------------------- #
# helpers (mirror tests/test_repair.py)
# --------------------------------------------------------------------------- #
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
    _RESP_N["n"] += 1
    return ExecResult(final_message="edit",
                      fs_diff=f"--- a/mod.py\n+++ b/mod.py\n@@ -1 +1 @@\n+attempt{_RESP_N['n']}\n",
                      ok=True, finalization_status="completed", usage=TokenUsage(input=1, output=1))


# --------------------------------------------------------------------------- #
# (1) HARNESS classification — INDETERMINATE on plugin-abort/collection/crash
# --------------------------------------------------------------------------- #
def _fake_eval(**kw):
    """A duck-typed Commit0Evaluation stand-in (scoring reads via getattr)."""
    base = dict(
        passed=0, failed=0, errors=0, total_tests=10, missing_expected=0,
        pass_rate=0.0, returncode=1, scored_success=False,
        evaluation_status="unsolved", verification_taxonomy="",
        scoring_source="commit0_test_ids", diagnostics={},
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_native_crash_returncode_is_indeterminate_not_zero():
    # A segfault (rc=139) / abort (134) / signal-kill (rc<0) must NOT be scored as
    # a genuine 0 — it is an environment failure.
    for rc in (134, 137, 138, 139, -11, -9):
        vr = verification_from_commit0_evaluation(_fake_eval(returncode=rc, passed=0, failed=10))
        assert vr.indeterminate is True, f"rc={rc} should be indeterminate"
        assert vr.accepted is False


def test_harness_failure_diagnostic_is_indeterminate():
    vr = verification_from_commit0_evaluation(
        _fake_eval(returncode=4, diagnostics={"harness_failure": True})
    )
    assert vr.indeterminate is True and vr.accepted is False


def test_parser_error_diagnostic_is_indeterminate():
    # rc=4 (pytest usage error / plugin-abort before collection) -> parser_error.
    vr = verification_from_commit0_evaluation(
        _fake_eval(returncode=4, diagnostics={"parser_error": "pytest_json_report_missing"})
    )
    assert vr.indeterminate is True and vr.accepted is False


def test_inconclusive_status_is_indeterminate():
    vr = verification_from_commit0_evaluation(
        _fake_eval(evaluation_status="audit_inconclusive")
    )
    assert vr.indeterminate is True and vr.accepted is False


def test_real_partial_stays_a_real_residual_not_indeterminate():
    # The whole point: a genuine near-solve (babel 4598/4607 class) is a REAL
    # residual the convergence loop must keep iterating on — NOT neutralized.
    vr = verification_from_commit0_evaluation(
        _fake_eval(passed=9, failed=1, total_tests=10, pass_rate=0.9, returncode=1)
    )
    assert vr.indeterminate is False
    assert vr.accepted is False           # all-gold-ids accept gate KEPT
    assert vr.passed == 9 and vr.pass_rate == pytest.approx(0.9)


def test_full_green_still_accepts():
    vr = verification_from_commit0_evaluation(
        _fake_eval(passed=10, failed=0, total_tests=10, pass_rate=1.0,
                   returncode=0, scored_success=True)
    )
    assert vr.accepted is True and vr.indeterminate is False


def test_benchmark_native_crash_helper():
    from apex.evaluation.commit0_benchmark import _commit0_returncode_is_native_crash
    assert _commit0_returncode_is_native_crash(139) is True
    assert _commit0_returncode_is_native_crash(134) is True
    assert _commit0_returncode_is_native_crash(-11) is True
    assert _commit0_returncode_is_native_crash(0) is False
    assert _commit0_returncode_is_native_crash(1) is False     # a real test failure
    assert _commit0_returncode_is_native_crash(4) is False     # usage error (parser path)


# --------------------------------------------------------------------------- #
# (2a) FLAG FLIP — repair_iters default 0 -> 2
# --------------------------------------------------------------------------- #
def test_repair_iters_default_is_two():
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=200)
    ctx = OrchestrationContext(
        eng, executor=FakeExecutor(_responder),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")],
        source_repo=_git_repo(), base_commit=None,
        score_fn=lambda wt: VerificationResult(accepted=False, score=0.5, passed=1,
                                               total=2, pass_rate=0.5),
        prompt_builder=lambda c, i, s: "fix", max_agents=8, initial_agents=1,
        # NOTE: repair_iters intentionally NOT passed -> exercise the new default.
    )
    assert ctx.repair_iters == 2


def test_repair_lineage_runs_by_default():
    # With the default flip, a genuine-but-incomplete base now gets a repair pass
    # WITHOUT any opt-in env var.
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=200)
    state = {"n": 0, "lock": threading.Lock()}

    def score_fn(wt):
        with state["lock"]:
            state["n"] += 1
            n = state["n"]
        if n == 1:
            return VerificationResult(accepted=False, score=0.5, passed=1, failed=1,
                                      total=2, pass_rate=0.5, failing_nodeids=["test_b"])
        return VerificationResult(accepted=True, score=1.0, passed=2, total=2, pass_rate=1.0)

    ctx = OrchestrationContext(
        eng, executor=FakeExecutor(_responder),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")],
        source_repo=_git_repo(), base_commit=None, score_fn=score_fn,
        prompt_builder=lambda c, i, s: "fix", max_agents=8, initial_agents=1,
    )
    winner = ctx.solve_and_repair(attempt_id=0, max_iters=2)
    assert winner is not None and winner.accepted
    assert state["n"] == 2                         # base + ONE repair (default ON)


def test_commit0_autogen_repair_iters_env_default_is_two():
    # The as-run eval value: APEX_OMEGA_REPAIR_ITERS now defaults to "2".
    import os
    prior = os.environ.pop("APEX_OMEGA_REPAIR_ITERS", None)
    try:
        assert int(os.environ.get("APEX_OMEGA_REPAIR_ITERS", "2") or 2) == 2
    finally:
        if prior is not None:
            os.environ["APEX_OMEGA_REPAIR_ITERS"] = prior


# --------------------------------------------------------------------------- #
# (2b) FLAG FLIP — repair excerpts ON by default (sanitized)
# --------------------------------------------------------------------------- #
def _repair_ctx(eng, seen):
    def responder(task, session):
        seen.append(task.prompt)
        return ExecResult(final_message="edit",
                          fs_diff="--- a/mod.py\n+++ b/mod.py\n@@ -1 +1 @@\n+x\n",
                          ok=True, finalization_status="completed", usage=TokenUsage(input=1, output=1))

    return OrchestrationContext(
        eng, executor=FakeExecutor(responder),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")],
        source_repo=_git_repo(), base_commit=None,
        score_fn=lambda wt: VerificationResult(accepted=True, score=1.0, passed=1, total=1, pass_rate=1.0),
        prompt_builder=lambda c, i, s: "base-issue", max_agents=8, initial_agents=1,
    )


def _fake_parent(ctx):
    cand = ctx.solve_attempt(attempt_id=0)
    cand.meta = dict(cand.meta or {})
    cand.meta["failing_nodeids"] = ["tests/test_x.py::test_one"]
    cand.meta["failure_excerpts"] = (
        "tests/test_x.py::test_one FAILED\nE  assert 0 == 42\n"
    )
    return cand


def test_repair_excerpts_on_by_default(monkeypatch):
    monkeypatch.delenv("APEX_OMEGA_REPAIR_EXCERPTS", raising=False)
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=200)
    seen: list[str] = []
    ctx = _repair_ctx(eng, seen)
    parent = _fake_parent(ctx)
    seen.clear()
    ctx.repair_attempt(parent, attempt_id=1)
    repair_prompt = next((p for p in seen if "REPAIR PASS" in p), "")
    assert repair_prompt, "repair prompt not captured"
    # sanitized node-id block present by default; the raw answer value (42) is NOT.
    assert "sanitized" in repair_prompt.lower()
    assert "== 42" not in repair_prompt


def test_repair_excerpts_off_when_explicitly_zero(monkeypatch):
    monkeypatch.setenv("APEX_OMEGA_REPAIR_EXCERPTS", "0")
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=200)
    seen: list[str] = []
    ctx = _repair_ctx(eng, seen)
    parent = _fake_parent(ctx)
    seen.clear()
    ctx.repair_attempt(parent, attempt_id=1)
    repair_prompt = next((p for p in seen if "REPAIR PASS" in p), "")
    assert repair_prompt
    assert "sanitized" not in repair_prompt.lower()   # excerpt block suppressed


# --------------------------------------------------------------------------- #
# (2c) FLAG FLIP — model_reasoning_effort pinned per-exec in the codex argv
# --------------------------------------------------------------------------- #
def _cfg(cli_args=None):
    return SimpleNamespace(cli_args=list(cli_args or []))


def test_codex_effort_xhigh_for_edit_turns(monkeypatch):
    from apex.core.cli_backend import _codex_cli_reasoning_effort_args
    monkeypatch.delenv("APEX_CODEX_EFFORT_EDIT", raising=False)
    args = _codex_cli_reasoning_effort_args(_cfg(), allow_edits=True)
    assert args == ["-c", "model_reasoning_effort=xhigh"]


def test_codex_effort_high_for_readonly_turns(monkeypatch):
    from apex.core.cli_backend import _codex_cli_reasoning_effort_args
    monkeypatch.delenv("APEX_CODEX_EFFORT_READONLY", raising=False)
    args = _codex_cli_reasoning_effort_args(_cfg(), allow_edits=False)
    assert args == ["-c", "model_reasoning_effort=high"]


def test_codex_effort_not_duplicated_when_operator_pinned():
    from apex.core.cli_backend import _codex_cli_reasoning_effort_args
    cfg = _cfg(["-c", "model_reasoning_effort=medium"])
    assert _codex_cli_reasoning_effort_args(cfg, allow_edits=True) == []


def test_codex_effort_suppressed_when_off(monkeypatch):
    from apex.core.cli_backend import _codex_cli_reasoning_effort_args
    monkeypatch.setenv("APEX_CODEX_EFFORT_EDIT", "off")
    assert _codex_cli_reasoning_effort_args(_cfg(), allow_edits=True) == []


def test_codex_effort_lands_in_built_codex_argv(monkeypatch):
    # End-to-end: the pinned -c flag must actually appear in the constructed codex
    # exec argv (so it reaches the JSONL launch event, not just the config fallback).
    from apex.core.cli_backend import CLIModelClient
    from apex.core.config import LLMConfig, LLMBackend

    monkeypatch.delenv("APEX_CODEX_EFFORT_EDIT", raising=False)
    monkeypatch.delenv("APEX_CODEX_EFFORT_READONLY", raising=False)
    cfg = LLMConfig(backend=LLMBackend.CODEX_CLI, model="gpt-5.5")
    client = CLIModelClient(cfg)
    workdir = tempfile.mkdtemp()
    command, _temps = client._build_command(
        prompt="do the task", working_dir=workdir, schema=None,
        system_prompt=None, allow_edits=True,
    )
    joined = " ".join(command)
    assert "model_reasoning_effort=xhigh" in joined
    # NO --output-schema on a coding turn (codex per-turn collapse bug).
    assert "--output-schema" not in joined
