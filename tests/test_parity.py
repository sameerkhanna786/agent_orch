"""Dynamic-workflows parity: ctx.args, ctx.workflow() nesting, the workflow catalog, and the
determinism residual guards (hash/non-deterministic-call lint)."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from apex_omega.autogen.catalog import known_workflows, resolve_workflow
from apex_omega.autogen.context import OrchestrationContext
from apex_omega.autogen.sandbox import lint_source
from apex_omega.engine.runtime import Engine
from apex_omega.errors import FailLoud
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


def _solver_responder():
    n = {"i": 0}

    def r(task, session):
        n["i"] += 1
        Path(session.cwd, "mod.py").write_text(f"def f():\n    return {n['i']}\n")
        return ExecResult(final_message="patched", ok=True, finalization_status="completed",
                          fs_diff=f"--- d{n['i']} ---", usage=TokenUsage(input=1, output=1))
    return r


def _ctx(*, args=None, responder=None, accept=True):
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=200)
    score = (lambda wt: VerificationResult(accepted=accept, score=1.0 if accept else 0.0,
                                           passed=1 if accept else 0, total=1,
                                           pass_rate=1.0 if accept else 0.0))
    return OrchestrationContext(
        eng, executor=FakeExecutor(responder or _solver_responder()),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")], source_repo=_git_repo(),
        base_commit=None, score_fn=score, prompt_builder=lambda c, i, s: "fix", args=args)


# --- determinism residual guards (0.6) -----------------------------------------
def test_lint_rejects_nondeterminism_but_allows_stable_idioms():
    assert not lint_source("def orchestrate(ctx):\n    return hash('x')").ok        # hash() removed
    assert not lint_source("def orchestrate(ctx):\n    return ctx.x.now()").ok        # clock
    assert not lint_source("def orchestrate(ctx):\n    return ctx.x.uuid4()").ok      # uuid
    assert not lint_source("def orchestrate(ctx):\n    return ctx.x.random()").ok     # rng
    # stable, deterministic idioms still pass
    assert lint_source("def orchestrate(ctx):\n    return ctx.select(sorted([]))").ok


# --- workflow catalog (0.3) ----------------------------------------------------
def test_catalog_resolves_names_and_refs_and_raises_on_unknown():
    assert {"default-best-of-n", "decompose", "ralph"} <= set(known_workflows())
    assert "orchestrate(ctx)" in resolve_workflow("default-best-of-n")
    with pytest.raises(KeyError):
        resolve_workflow("does-not-exist")
    d = Path(tempfile.mkdtemp())
    (d / "wf.py").write_text("def orchestrate(ctx):\n    return 1\n")
    assert "orchestrate" in resolve_workflow({"scriptPath": str(d / "wf.py")})
    with pytest.raises(KeyError):
        resolve_workflow({"foo": "bar"})


# --- ctx.args (0.2) ------------------------------------------------------------
def test_ctx_args_explicit_and_repo_map_fallback():
    assert _ctx(args={"q": 1}).args == {"q": 1}
    c = _ctx()
    c.repo_map = {"args": {"r": 2}}
    assert c.args == {"r": 2}                       # falls back to repo_map['args']
    assert _ctx().args is None


# --- ctx.workflow() nesting (0.1) ----------------------------------------------
def test_ctx_workflow_runs_named_child_on_shared_engine():
    ctx = _ctx()
    winner = ctx.workflow("default-best-of-n")     # compose the verified best-of-N inline
    assert winner is not None and winner.accepted is True
    # the child shared THIS engine (its agents count against the same run)
    assert ctx.agents_used() >= 1


def test_ctx_workflow_is_one_level_deep():
    ctx = _ctx()
    child = ctx._spawn_child()
    assert child._nesting_depth == 1 and child._node_ns == "w1_"
    with pytest.raises(FailLoud):
        child.workflow("default-best-of-n")        # a child cannot nest further


def test_child_journal_nodes_are_namespaced_no_collision():
    # the child's attempt/candidate ids carry the namespace so they never collide with the
    # parent's on resume (the root stays un-prefixed for back-compat).
    ctx = _ctx()
    parent_c = ctx.solve_attempt(attempt_id=0)
    child = ctx._spawn_child()
    child_c = child.solve_attempt(attempt_id=0)
    assert parent_c.candidate_id == "a0"
    assert child_c.candidate_id == "w1_a0" and child_c.candidate_id != parent_c.candidate_id


# --- IOU / defer (1.2) ---------------------------------------------------------
def test_ctx_defer_and_blocked():
    ctx = _ctx()
    ctx.defer("imports", "pkg.mod", "circular import")
    ctx.defer("api", "Foo.bar")
    assert len(ctx.blocked()) == 2
    only = ctx.blocked(scope="imports")
    assert len(only) == 1 and only[0]["item"] == "pkg.mod" and only[0]["reason"] == "circular import"


# --- loop_until_dry SEEN item dedup (1.3) --------------------------------------
def test_loop_until_dry_converges_on_no_new_keys():
    # identical candidate every round -> after k_dry rounds with no NEW key, the loop stops
    # (dedupe-vs-SEEN convergence), well before max_rounds.
    def fixed_responder():
        def r(task, session):
            Path(session.cwd, "mod.py").write_text("def f():\n    return 7\n")
            return ExecResult(final_message="x", ok=True, finalization_status="completed",
                              fs_diff="--- fixed ---", usage=TokenUsage(input=1, output=1))
        return r
    ctx = _ctx(responder=fixed_responder(), accept=False)
    rounds = {"n": 0}

    def make_round(i):
        rounds["n"] += 1
        return [lambda i=i: ctx.solve_attempt(attempt_id=1000 + i)]

    produced = ctx.loop_until_dry(make_round, k_dry=2, max_rounds=64,
                                  key=lambda c: c.content_sha)
    assert rounds["n"] <= 4          # converged early (round0 new key, then 2 dry -> stop)
    assert produced                  # but did produce the candidates it saw


# --- adversarial_filter admit gate (1.1) ---------------------------------------
def _filter_responder():
    def r(task, session):
        if getattr(task, "schema", None):              # a read-only ask
            refuted = "REFUTE-ME" in (task.prompt or "")
            return ExecResult(final_message="", structured_output={"refuted": refuted},
                              ok=True, finalization_status="completed", usage=TokenUsage(input=1, output=1))
        return ExecResult(final_message="x", ok=True, finalization_status="completed",
                          usage=TokenUsage(input=1, output=1))
    return r


def test_adversarial_filter_admits_only_survivors():
    ctx = _ctx(responder=_filter_responder())
    kept = ctx.adversarial_filter(["keep this finding", "REFUTE-ME false positive"], votes=3)
    assert kept == ["keep this finding"]              # the majority-refuted item is dropped
    # votes<=0 is identity; never touches Candidate.accepted (plain-data only)
    assert ctx.adversarial_filter(["a", "b"], votes=0) == ["a", "b"]
