"""TREE-SEARCH v1 (LATS-style) — gated by APEX_OMEGA_TREE_SEARCH (default OFF).

Covers:
  * OFF byte-identical: with the flag unset, the new tree methods early-return inert (None/empty),
    no _tree* state is allocated, and a plain converge-style run is unchanged.
  * ENGAGES (FakeExecutor): with the flag on, ctx.tree_search expands multiple nodes seeded by
    parent diffs, builds a host-side tree (>1 node, parent pointers set), and returns a ctx.select
    result. The responder writes a real fix into the worktree so scoring is execution-grounded.
  * Cardinal-safe: no tree code sets Candidate.accepted; only ctx.select accepts.
  * UCT determinism: uct_select returns the SAME pick given the same node stats (no random/time).

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
from apex_omega.engine.runtime import Engine
from apex_omega.executor.fake import FakeExecutor
from apex_omega.kernel.verify import VerificationResult
from apex_omega.types import ExecResult, TokenUsage
from apex_omega.workflows.best_of_n import WorkerSpec


@pytest.fixture(autouse=True)
def _clear_tree_env():
    keys = [k for k in os.environ if k.startswith("APEX_OMEGA_TREE_SEARCH")]
    saved = {k: os.environ[k] for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    yield
    for k in list(os.environ):
        if k.startswith("APEX_OMEGA_TREE_SEARCH"):
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


# ---------------------------------------------------------------- OFF byte-identical / inert
def test_tree_off_is_inert():
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, _near_solve_repo())
    assert ctx._tree_on() is False
    # every tree method early-returns inert before touching _tree* state.
    with pytest.raises(TypeError):
        ctx.tree_search()  # budget_nodes is MANDATORY (no default)
    assert ctx.tree_search(budget_nodes=4) is None
    assert ctx.uct_select() == {}
    assert ctx.expand_node({"id": "x", "diff": ""}) is None
    assert ctx.tree_state() == {"nodes": [], "best": None, "uct_pick": None}
    # no _tree* state mutated, no agents dispatched.
    assert ctx._tree == {} and ctx._tree_root is None and ctx._tree_children == {}
    assert ctx._tree_expansions == 0
    assert eng.agents_used() == 0


def test_tree_state_allocated_empty_at_init():
    """The host-side tree containers exist but stay empty when off (lazy/inert => byte-identical)."""
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, _near_solve_repo())
    assert ctx._tree == {}
    assert ctx._tree_root is None
    assert ctx._tree_children == {}
    assert ctx._tree_expansions == 0


# ---------------------------------------------------------------- ENGAGES (builds a tree, solves)
def test_tree_search_engages_and_solves():
    os.environ["APEX_OMEGA_TREE_SEARCH"] = "1"
    seen = {"carry_grafts": 0}

    def responder(task, session):
        # the tree expansion / root agent: build on whatever is in the workspace, fix c() so the
        # near-solve reaches 3/3 (execution-grounded — scoring runs the real pytest after).
        p = task.prompt or ""
        # a non-root expansion is carry-grafted (parent diff pre-applied); detect by the brief.
        if "TREE-SEARCH EXPANSION" in p:
            seen["carry_grafts"] += 1
        Path(session.cwd, "mod.py").write_text(
            "def a():\n    return 1\n\ndef b():\n    return 2\n\ndef c():\n    return 3\n")
        return ExecResult(final_message="fixed c", ok=True, finalization_status="completed",
                          usage=TokenUsage(input=1, output=1))

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=999)
    ctx = _ctx(eng, _near_solve_repo(), responder=responder)
    assert ctx._tree_on() is True
    winner = ctx.tree_search(budget_nodes=4, branch=2, c_uct=1.4)
    # a host-side tree was built (root + at least the registered nodes) with parent pointers set.
    assert len(ctx._tree) >= 1
    assert ctx._tree_root is not None and ctx._tree_root in ctx._tree
    # the root has no parent; any child points back to a real node.
    assert ctx._tree[ctx._tree_root]["parent"] is None
    for nid, node in ctx._tree.items():
        if node["parent"] is not None:
            assert node["parent"] in ctx._tree, "child parent pointer must reference a real node"
    # tree_search returns the ctx.select result (an accepted winner once c() is fixed to 3/3).
    assert winner is not None
    assert winner.accepted is True
    assert eng.agents_used() >= 1


def test_tree_search_expands_multiple_nodes_seeded_by_parent_diffs():
    """A near-solve that STAYS 2/3 (the agent writes nothing useful) forces multiple expansions, each
    seeded by its parent's diff — proving the tree builds (>1 node) and carry-grafts the lineage."""
    os.environ["APEX_OMEGA_TREE_SEARCH"] = "1"
    seen = {"expansions": 0}

    def responder(task, session):
        p = task.prompt or ""
        if "TREE-SEARCH EXPANSION" in p:
            seen["expansions"] += 1
        # write a tiny harmless new file so each attempt has a NON-empty (but non-solving) diff,
        # so nodes are registered with diffs and the lineage carry-grafts.
        Path(session.cwd, f"note_{seen['expansions']}.txt").write_text("x\n")
        return ExecResult(final_message="noop", ok=True, finalization_status="completed",
                          usage=TokenUsage(input=1, output=1))

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=999)
    ctx = _ctx(eng, _near_solve_repo(), responder=responder)
    winner = ctx.tree_search(budget_nodes=4, branch=2, c_uct=1.4)
    # multiple nodes registered (root + expansions), parent pointers set on the children.
    assert len(ctx._tree) > 1, "tree-search should build more than the root node"
    children = [n for n in ctx._tree.values() if n["parent"] is not None]
    assert len(children) >= 1
    assert seen["expansions"] >= 1, "expand_node should run carry-grafted expansions"
    # budget honored: never more expansions than budget_nodes.
    assert ctx._tree_expansions <= 4
    # the near-solve never reaches 3/3 -> ctx.select abstains (never fakes a pass).
    assert winner is None or winner.accepted is False


# ---------------------------------------------------------------- Cardinal Contract
def test_tree_code_never_sets_accepted():
    """No tree path sets Candidate.accepted; acceptance is select-owned (execution-grounded)."""
    os.environ["APEX_OMEGA_TREE_SEARCH"] = "1"

    def responder(task, session):
        # never solve (stay 2/3) so any 'accepted' would have to be a forbidden soft promotion.
        Path(session.cwd, "note.txt").write_text("x\n")
        return ExecResult(final_message="noop", ok=True, finalization_status="completed",
                          usage=TokenUsage(input=1, output=1))

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=999)
    ctx = _ctx(eng, _near_solve_repo(), responder=responder)
    ctx.tree_search(budget_nodes=4)
    # every banked candidate from a non-solving run is unaccepted (no soft promotion).
    for c in ctx.all_candidates():
        assert c.accepted is False, "tree code must never set Candidate.accepted"
    # tree nodes carry no 'accepted' key at all (soft, host-side stats only).
    for node in ctx._tree.values():
        assert "accepted" not in node


def test_tree_search_source_passes_frozen_lint_no_accepted_binding():
    """The frozen TREE_SEARCH_ORCHESTRATION must pass the frozen-template lint (no getattr/os.environ/
    imports/binding .accepted) — the Cardinal Contract at the authored-code layer."""
    from apex_omega.autogen.sandbox import lint_source
    from apex_omega.autogen.templates import TREE_SEARCH_ORCHESTRATION
    r = lint_source(TREE_SEARCH_ORCHESTRATION)
    assert r.ok is True, f"frozen tree-search lint violations: {r.violations}"
    assert ".accepted" not in TREE_SEARCH_ORCHESTRATION


# ---------------------------------------------------------------- UCT determinism
def test_uct_select_is_deterministic():
    os.environ["APEX_OMEGA_TREE_SEARCH"] = "1"
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, _near_solve_repo())
    # hand-build a small tree with distinct stats (no agents dispatched).
    ctx._tree_expansions = 5
    ctx._tree = {
        "n1": {"id": "n1", "parent": None, "diff": "d1", "gold_passed": 2, "gold_total": 3,
               "visits": 0, "value_sum": 0.0, "since_improve": 0, "_branch": 2},
        "n2": {"id": "n2", "parent": "n1", "diff": "d2", "gold_passed": 1, "gold_total": 3,
               "visits": 0, "value_sum": 0.0, "since_improve": 3, "_branch": 2},
    }
    ctx._tree_root = "n1"
    p1 = ctx.uct_select(c_uct=1.4)
    p2 = ctx.uct_select(c_uct=1.4)
    assert p1 == p2, "uct_select must be deterministic given identical node stats"
    assert p1 and p1.get("id") in ("n1", "n2")


def test_uct_select_skips_exhausted_nodes():
    """A non-root node whose visits >= branch is NOT expandable; the root always is."""
    os.environ["APEX_OMEGA_TREE_SEARCH"] = "1"
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, _near_solve_repo())
    ctx._tree_expansions = 3
    ctx._tree = {
        "r": {"id": "r", "parent": None, "diff": "d", "gold_passed": 3, "gold_total": 3,
              "visits": 5, "value_sum": 5.0, "since_improve": 0, "_branch": 2},   # exhausted but ROOT
        "c": {"id": "c", "parent": "r", "diff": "d", "gold_passed": 0, "gold_total": 3,
              "visits": 9, "value_sum": 0.0, "since_improve": 0, "_branch": 2},   # exhausted child -> skipped
    }
    ctx._tree_root = "r"
    pick = ctx.uct_select(c_uct=1.4)
    assert pick.get("id") == "r", "root stays expandable even past its branch count"


def test_tree_state_introspection_reports_best_and_pick():
    os.environ["APEX_OMEGA_TREE_SEARCH"] = "1"
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, _near_solve_repo())
    ctx._tree_expansions = 2
    ctx._tree = {
        "r": {"id": "r", "parent": None, "diff": "d", "gold_passed": 1, "gold_total": 3,
              "visits": 0, "value_sum": 0.0, "since_improve": 0, "_branch": 2},
        "c": {"id": "c", "parent": "r", "diff": "d", "gold_passed": 2, "gold_total": 3,
              "visits": 0, "value_sum": 0.0, "since_improve": 0, "_branch": 2},
    }
    ctx._tree_root = "r"
    st = ctx.tree_state()
    assert {n["id"] for n in st["nodes"]} == {"r", "c"}
    assert st["best"]["id"] == "c"          # highest gold_passed
    assert st["uct_pick"] is not None
