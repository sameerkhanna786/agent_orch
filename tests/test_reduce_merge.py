"""Reduce/merge + no-silent-loss convergence fixes (the jinja/s0 abstain class).

Covers the mitigation for the converge-arm collapse bug: fan-out emits COMPETING WHOLE-REPO
candidates (not disjoint slices) -> reduce_residuals (a textual 3-way patch-stack) conflicts them
all out -> empty/zero merged tree -> the old `while ... and residual:` loop ran ZERO recovery
rounds (failing_nodeids is empty on a collection error) -> abstain. The fixes, grounded in the
Anthropic dynamic-workflow paradigm (independence; SELECT-don't-merge competing fulls; reduce-by-
key; NO SILENT LOSS):

  * loop guard keyed on `not red["accepted"]` (+ module_gold_ids targets when residual is empty),
  * a COLLAPSE fallback that SELECTs among competing fulls then falls back to whole-repo best-of-N,
  * diff-hygiene: harness scaffolding (.apex_seatbelt/) excluded/stripped so it never creates a
    spurious cross-module merge conflict,
  * modules_overlap / module_gold_ids / _strip_scaffold_hunks pure helpers.

All offline: real git worktrees + a real-or-scripted score_fn + the FakeExecutor. No codex burn.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from apex_omega.autogen.context import (
    OrchestrationContext,
    _diff_touched_paths,
    _strip_scaffold_hunks,
)
from apex_omega.autogen.sandbox import run_orchestration
from apex_omega.autogen.templates import DEFAULT_ORCHESTRATION
from apex_omega.engine.runtime import Engine
from apex_omega.executor.fake import FakeExecutor
from apex_omega.kernel.verify import VerificationResult, candidate_from_verification
from apex_omega.types import ExecResult, TokenUsage
from apex_omega.workflows.best_of_n import WorkerSpec


# --------------------------------------------------------------------------- fixtures
def _git(d: Path) -> None:
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-qm", "base"]):
        subprocess.run(c, cwd=d, check=True, capture_output=True)


def _competing_repo() -> str:
    """One source file (three functions) + three tests — the from-scratch shape where each module
    agent rewrites the WHOLE file, so module diffs OVERLAP (the jinja collapse class)."""
    d = Path(tempfile.mkdtemp()) / "repo"
    d.mkdir()
    (d / "lib.py").write_text("def f():\n    return 0\ndef g():\n    return 0\ndef h():\n    return 0\n")
    (d / "test_lib.py").write_text(
        "from lib import f, g, h\n\n"
        "def test_f():\n    assert f() == 1\n\n"
        "def test_g():\n    assert g() == 2\n\n"
        "def test_h():\n    assert h() == 3\n")
    _git(d)
    return str(d)


GOLD = ["test_lib.py::test_f", "test_lib.py::test_g", "test_lib.py::test_h"]
PLAN3 = {"modules": [{"module": "m1", "gold_test_ids": ["test_lib.py::test_f"]},
                     {"module": "m2", "gold_test_ids": ["test_lib.py::test_g"]},
                     {"module": "m3", "gold_test_ids": ["test_lib.py::test_h"]}],
         "order": ["m1", "m2", "m3"]}


def _real_pytest_score(node_ids):
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


def _cand(cid, diff, *, passed=0, total=2, accepted=False, module=None, indeterminate=False,
          carry_conflict=False):
    meta = {}
    if module:
        meta["module"] = module
    if indeterminate:
        meta["indeterminate"] = True
    if carry_conflict:
        meta["carry_conflict"] = True
    return candidate_from_verification(
        candidate_id=cid, diff=diff,
        vr=VerificationResult(accepted=accepted, passed=passed, total=total,
                              pass_rate=passed / max(1, total), indeterminate=indeterminate),
        meta=meta)


def _phase_recorder(eng):
    phases = []
    eng.phase = lambda title, _p=phases, _o=eng.phase: (_p.append(title), _o(title))[1]
    return phases


# --------------------------------------------------------------------------- pure helpers
def test_strip_scaffold_hunks_git_format():
    diff = (
        "diff --git a/.apex_seatbelt/read_jail.sb b/.apex_seatbelt/read_jail.sb\n"
        "new file mode 100644\n--- /dev/null\n+++ b/.apex_seatbelt/read_jail.sb\n@@ -0,0 +1,1 @@\n+(version 1)\n"
        "diff --git a/src/lib.py b/src/lib.py\n--- a/src/lib.py\n+++ b/src/lib.py\n@@ -1 +1 @@\n-x\n+y\n")
    out = _strip_scaffold_hunks(diff)
    assert ".apex_seatbelt" not in out          # scaffolding section dropped
    assert "src/lib.py" in out and "+y" in out  # real edit preserved


def test_strip_scaffold_hunks_minimal_format():
    diff = (
        "--- /dev/null\n+++ b/.apex_seatbelt/read_jail.sb\n@@ -0,0 +1 @@\n+(version 1)\n"
        "--- a/mod.py\n+++ b/mod.py\n@@ -1 +1 @@\n-a\n+b\n")
    out = _strip_scaffold_hunks(diff)
    assert ".apex_seatbelt" not in out
    assert "mod.py" in out and "+b" in out


def test_diff_touched_paths():
    diff = "--- a/x.py\n+++ b/x.py\n@@\n--- a/pkg/y.py\n+++ b/pkg/y.py\n@@\n--- /dev/null\n+++ b/new.py\n@@\n"
    assert _diff_touched_paths(diff) == {"x.py", "pkg/y.py", "new.py"}


def test_module_gold_ids_unions_and_sorts():
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=8)
    ctx = _ctx(eng, _competing_repo(), lambda wt: VerificationResult(), None, {})
    modules = [{"module": "a", "gold_test_ids": ["t::b", "t::a"]},
               {"module": "b", "gold_test_ids": ["t::a", "t::c"]},
               {"module": "c"}, None]
    assert ctx.module_gold_ids(modules) == ["t::a", "t::b", "t::c"]   # sorted union, no dups, fail-soft
    assert ctx.module_gold_ids([]) == []


def test_modules_overlap_true_for_competing_fulls():
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=8)
    ctx = _ctx(eng, _competing_repo(), lambda wt: VerificationResult(), None, {})
    whole1 = _cand("a", "--- a/lib.py\n+++ b/lib.py\n@@\n--- a/util.py\n+++ b/util.py\n@@\n", module="m1")
    whole2 = _cand("b", "--- a/lib.py\n+++ b/lib.py\n@@\n--- a/util.py\n+++ b/util.py\n@@\n", module="m2")
    assert ctx.modules_overlap([whole1, whole2]) is True


def test_modules_overlap_false_for_disjoint():
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=8)
    ctx = _ctx(eng, _competing_repo(), lambda wt: VerificationResult(), None, {})
    a = _cand("a", "--- a/mod_a.py\n+++ b/mod_a.py\n@@\n", module="m1")
    b = _cand("b", "--- a/mod_b.py\n+++ b/mod_b.py\n@@\n", module="m2")
    assert ctx.modules_overlap([a, b]) is False
    assert ctx.modules_overlap([a, None]) is False     # singletons / None never "overlap"


# --------------------------------------------------------------------------- reduce_residuals
def test_reduce_returns_gold_passed():
    """reduce now surfaces gold_passed so the orchestrator can tell a climbing partial from a
    total collapse."""
    repo = _competing_repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=8)
    ctx = _ctx(eng, repo, _real_pytest_score(GOLD), None, {})
    # one whole-file candidate that greens ONLY test_f (1/3) -> gold_passed == 1, not accepted.
    diff = ("--- a/lib.py\n+++ b/lib.py\n@@ -1,6 +1,6 @@\n"
            "-def f():\n-    return 0\n-def g():\n-    return 0\n-def h():\n-    return 0\n"
            "+def f():\n+    return 1\n+def g():\n+    return 0\n+def h():\n+    return 0\n")
    red = ctx.reduce_residuals([_cand("p", diff, passed=1, total=3, module="m1")])
    assert red["gold_passed"] == 1 and red["accepted"] is False


def test_reduce_handles_none_empty_indeterminate_candidates():
    """reduce never raises on None / empty-diff / indeterminate entries; the one real diff lands."""
    repo = _competing_repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=8)
    ctx = _ctx(eng, repo, _real_pytest_score(GOLD), None, {})
    good = ("--- a/lib.py\n+++ b/lib.py\n@@ -1,6 +1,6 @@\n"
            "-def f():\n-    return 0\n-def g():\n-    return 0\n-def h():\n-    return 0\n"
            "+def f():\n+    return 1\n+def g():\n+    return 0\n+def h():\n+    return 0\n")
    cands = [None,
             _cand("empty", "", module="m_empty"),
             _cand("indet", "--- a/lib.py\n+++ b/lib.py\n@@\n+junk\n", module="m_ind", indeterminate=True),
             _cand("good", good, passed=1, total=3, module="m1")]
    red = ctx.reduce_residuals(cands)            # must not raise
    assert red["gold_passed"] == 1               # the one real diff applied + scored
    assert "m_ind" in red["conflicts"]           # indeterminate recorded, not applied


def test_reduce_strips_scaffolding_so_disjoint_modules_merge_clean():
    """Two genuinely-disjoint module diffs that EACH also carry the per-worktree .apex_seatbelt
    new-file: with scaffolding stripped they merge CLEANLY (no spurious conflict) and the merged
    artifact is scaffolding-free."""
    repo = _two_module_repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=8)
    ctx = _ctx(eng, repo, _real_pytest_score(["test_a.py::test_a", "test_b.py::test_b"]), None, {})
    sb = ("diff --git a/.apex_seatbelt/read_jail.sb b/.apex_seatbelt/read_jail.sb\n"
          "new file mode 100644\n--- /dev/null\n+++ b/.apex_seatbelt/read_jail.sb\n@@ -0,0 +1 @@\n")
    diff_a = (sb + "+(profile A)\n"
              "diff --git a/mod_a.py b/mod_a.py\n--- a/mod_a.py\n+++ b/mod_a.py\n@@ -1,2 +1,2 @@\n"
              "-def a():\n-    return 0  # BUG\n+def a():\n+    return 1\n")
    diff_b = (sb + "+(profile B is different)\n"   # DIFFERENT scaffolding bytes -> would conflict
              "diff --git a/mod_b.py b/mod_b.py\n--- a/mod_b.py\n+++ b/mod_b.py\n@@ -1,2 +1,2 @@\n"
              "-def b():\n-    return 0  # BUG\n+def b():\n+    return 2\n")
    red = ctx.reduce_residuals([_cand("ma", diff_a, passed=1, total=2, module="mod_a"),
                                _cand("mb", diff_b, passed=1, total=2, module="mod_b")])
    assert red["conflicts"] == []                       # NO scaffolding-only conflict
    assert red["accepted"] is True                      # both real edits merged -> green
    assert ".apex_seatbelt" not in red["merged_diff"]   # artifact is scaffolding-free


def test_carry_only_scaffolding_is_noop():
    """A carry diff that is ONLY scaffolding strips to empty -> not applied, never a __carry__
    conflict."""
    repo = _competing_repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=8)
    ctx = _ctx(eng, repo, _real_pytest_score(GOLD), None, {})
    sb_only = ("--- /dev/null\n+++ b/.apex_seatbelt/read_jail.sb\n@@ -0,0 +1 @@\n+(version 1)\n")
    red = ctx.reduce_residuals([], carry_diff=sb_only)
    assert "__carry__" not in red["conflicts"]


def _two_module_repo() -> str:
    d = Path(tempfile.mkdtemp()) / "repo"
    d.mkdir()
    (d / "mod_a.py").write_text("def a():\n    return 0  # BUG\n")
    (d / "mod_b.py").write_text("def b():\n    return 0  # BUG\n")
    (d / "test_a.py").write_text("from mod_a import a\n\ndef test_a():\n    assert a() == 1\n")
    (d / "test_b.py").write_text("from mod_b import b\n\ndef test_b():\n    assert b() == 2\n")
    _git(d)
    return str(d)


# --------------------------------------------------------------------------- collapse -> best-of-N
def _competing_responder():
    """decompose -> 3 modules; each module agent emits a COMPETING whole-file BROKEN rewrite
    (overlap -> conflict, 0 pass); the best-of-N fallback agent (no module/residual) writes the
    CORRECT whole-file solution."""
    def responder(task, session):
        if task.schema is not None:
            return ExecResult(structured_output=PLAN3, ok=True, finalization_status="completed")
        si = task.scoped_inputs or {}
        mod = si.get("module")
        if mod:
            wrong = {"m1": "10", "m2": "20", "m3": "30"}.get(mod, "99")
            Path(session.cwd, "lib.py").write_text(
                f"# {mod}\ndef f():\n    return {wrong}\ndef g():\n    return {wrong}\ndef h():\n    return {wrong}\n")
        else:   # best-of-N fallback attempt -> the correct whole-repo solution
            Path(session.cwd, "lib.py").write_text(
                "def f():\n    return 1\ndef g():\n    return 2\ndef h():\n    return 3\n")
        diff = subprocess.run(["git", "-C", session.cwd, "diff"], text=True, capture_output=True).stdout
        return ExecResult(final_message=f"patched {mod}", fs_diff=diff, ok=True,
                          finalization_status="completed", usage=TokenUsage(input=1, output=1))
    return responder


def test_collapse_competing_fulls_falls_back_to_best_of_n():
    """THE headline regression: competing whole-repo candidates all-conflict -> the run does NOT
    abstain (the jinja/s0 bug) but routes to the whole-repo best-of-N path and returns an accepted
    winner. No-silent-loss."""
    repo = _competing_repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=32)
    phases = _phase_recorder(eng)
    ctx = _ctx(eng, repo, _real_pytest_score(GOLD), _competing_responder(),
               {"difficulty": "medium", "modules": ["lib"]})
    winner = run_orchestration(DEFAULT_ORCHESTRATION, ctx)
    assert winner is not None and winner.accepted          # NOT abstain
    assert "reduce" in phases                              # the collapse was detected after reduce
    assert "autosolve" in phases                           # best-of-N fallback actually ran
    assert "verify" not in phases                          # collapse returned before loop/verify
    assert not (winner.meta or {}).get("module")           # winner came from best-of-N, not a module


def test_collapse_judge_abstains_then_best_of_n_no_zero_shipped():
    """When every competing full scores 0, judge_select MUST abstain (cardinal contract — no 0%
    candidate is ever promoted); the run then proceeds to best-of-N and the accepted winner is a
    real (passing) one, never a shipped zero."""
    repo = _competing_repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=32)
    ctx = _ctx(eng, repo, _real_pytest_score(GOLD), _competing_responder(),
               {"difficulty": "medium", "modules": ["lib"]})
    winner = run_orchestration(DEFAULT_ORCHESTRATION, ctx)
    # no module candidate was acceptable (all broke) -> nothing unverified shipped.
    module_cands = [c for c in ctx.all_candidates() if (c.meta or {}).get("module")]
    assert module_cands and not any(c.accepted for c in module_cands)
    assert winner is not None and winner.accepted and winner.public_signal_score >= 1.0


def test_passing_full_among_conflicts_is_selected_not_best_of_n():
    """If ONE competing full candidate actually PASSES, the collapse SELECT (judge_select over all
    banked candidates) returns it WITHOUT running best-of-N — competing fulls are SELECTed."""
    repo = _competing_repo()

    def responder(task, session):
        if task.schema is not None:
            return ExecResult(structured_output=PLAN3, ok=True, finalization_status="completed")
        mod = (task.scoped_inputs or {}).get("module")
        if mod == "m2":        # m2 is the fully-correct whole-repo solution (accepted)
            Path(session.cwd, "lib.py").write_text(
                "def f():\n    return 1\ndef g():\n    return 2\ndef h():\n    return 3\n")
        elif mod:              # m1/m3 are competing broken whole-file rewrites (conflict)
            wrong = {"m1": "10", "m3": "30"}[mod]
            Path(session.cwd, "lib.py").write_text(
                f"# {mod}\ndef f():\n    return {wrong}\ndef g():\n    return {wrong}\ndef h():\n    return {wrong}\n")
        else:
            raise AssertionError("best-of-N must NOT run when a passing full exists")
        diff = subprocess.run(["git", "-C", session.cwd, "diff"], text=True, capture_output=True).stdout
        return ExecResult(final_message=f"patched {mod}", fs_diff=diff, ok=True,
                          finalization_status="completed", usage=TokenUsage(input=1, output=1))

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=32)
    phases = _phase_recorder(eng)
    ctx = _ctx(eng, repo, _real_pytest_score(GOLD), responder, {"difficulty": "medium", "modules": ["lib"]})
    winner = run_orchestration(DEFAULT_ORCHESTRATION, ctx)
    assert winner is not None and winner.accepted
    assert (winner.meta or {}).get("module") == "m2"   # SELECTed the passing module, not best-of-N
    assert "autosolve" not in phases                   # best-of-N never ran


# --------------------------------------------------------------------------- loop engagement (RANK 1)
def _marker_score(ids, marker="DONE"):
    """A scripted score_fn that mimics a COLLECTION-error shape: while unsolved it reports a partial
    pass with an EMPTY failing_nodeids (the exact condition that made the old `and residual` loop run
    zero rounds); once the marker is present it accepts."""
    def _score(wt: str) -> VerificationResult:
        p = Path(wt, "lib.py")
        done = p.exists() and marker in p.read_text()
        if done:
            return VerificationResult(accepted=True, score=1.0, passed=len(ids), failed=0,
                                      total=len(ids), pass_rate=1.0, failing_nodeids=[])
        return VerificationResult(accepted=False, score=0.0, passed=1, failed=0, total=len(ids),
                                  pass_rate=1.0 / len(ids), failing_nodeids=[])   # EMPTY residual
    return _score


def test_loop_engages_on_empty_residual_collection_error():
    """RANK 1: the merged tree is unsolved but failing_nodeids is EMPTY (collection error). The new
    `not red["accepted"]` guard MUST still engage loop-until-dry (the old `and residual` ran zero
    rounds and abstained), feeding module_gold_ids as the repair target."""
    repo = _competing_repo()
    seen = {"residual_targets": None, "rounds": 0}

    def responder(task, session):
        if task.schema is not None:
            return ExecResult(structured_output=PLAN3, ok=True, finalization_status="completed")
        si = task.scoped_inputs or {}
        if si.get("residual_ids") is not None:          # loop-until-dry repair round
            seen["residual_targets"] = list(si.get("residual_ids") or [])
            seen["rounds"] += 1
            Path(session.cwd, "lib.py").write_text("# DONE\ndef f():\n    return 1\n")   # write marker -> accept
        # module solves leave lib.py at base (empty/no progress)
        diff = subprocess.run(["git", "-C", session.cwd, "diff"], text=True, capture_output=True).stdout
        return ExecResult(final_message="x", fs_diff=diff, ok=True, finalization_status="completed",
                          usage=TokenUsage(input=1, output=1))

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=32)
    ctx = _ctx(eng, repo, _marker_score(GOLD), responder, {"difficulty": "medium", "modules": ["lib"]})
    winner = run_orchestration(DEFAULT_ORCHESTRATION, ctx)
    assert seen["rounds"] >= 1                                   # the loop engaged (old guard: 0)
    assert seen["residual_targets"]                             # repair got NON-empty targets...
    assert seen["residual_targets"] == ctx.module_gold_ids(PLAN3["modules"])   # ...= module gold ids
    assert winner is not None and winner.accepted


def test_unsolvable_empty_residual_terminates_no_hang(monkeypatch):
    """The new `not red["accepted"]` guard must still TERMINATE: an unsolvable empty-residual merge
    (repair never accepts) stops on the governor's plateau cut and abstains — never an infinite
    loop, never a fake pass."""
    monkeypatch.setenv("APEX_OMEGA_PLATEAU_PATIENCE", "3")
    repo = _competing_repo()
    rounds = {"n": 0}

    def responder(task, session):
        if task.schema is not None:
            return ExecResult(structured_output=PLAN3, ok=True, finalization_status="completed")
        if (task.scoped_inputs or {}).get("residual_ids") is not None:
            rounds["n"] += 1            # repair runs but NEVER writes the marker (unsolvable)
        diff = subprocess.run(["git", "-C", session.cwd, "diff"], text=True, capture_output=True).stdout
        return ExecResult(final_message="noop", fs_diff=diff, ok=True, finalization_status="completed",
                          usage=TokenUsage(input=1, output=1))

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=32)
    ctx = _ctx(eng, repo, _marker_score(GOLD), responder, {"difficulty": "medium", "modules": ["lib"]})
    from apex_omega.errors import PlateauStop
    try:
        winner = run_orchestration(DEFAULT_ORCHESTRATION, ctx)
        accepted = winner is not None and winner.accepted
    except PlateauStop:
        accepted = False
    assert accepted is False                 # honest abstain
    assert 1 <= rounds["n"] < 100            # iterated, then terminated (bounded — no hang)


# --------------------------------------------------------------------------- no over-eager collapse
def test_disjoint_modules_do_not_collapse_clean_merge_accepts():
    """A genuinely-disjoint 2-module repo must merge cleanly and accept — the collapse fallback must
    NOT fire (guards against over-eager best-of-N routing)."""
    repo = _two_module_repo()
    phases = []

    def responder(task, session):
        if task.schema is not None:
            return ExecResult(structured_output={
                "modules": [{"module": "mod_a", "gold_test_ids": ["test_a.py::test_a"]},
                            {"module": "mod_b", "gold_test_ids": ["test_b.py::test_b"]}],
                "order": ["mod_a", "mod_b"]}, ok=True, finalization_status="completed")
        mod = (task.scoped_inputs or {}).get("module", "")
        if mod == "mod_a":
            Path(session.cwd, "mod_a.py").write_text("def a():\n    return 1\n")
        elif mod == "mod_b":
            Path(session.cwd, "mod_b.py").write_text("def b():\n    return 2\n")
        diff = subprocess.run(["git", "-C", session.cwd, "diff"], text=True, capture_output=True).stdout
        return ExecResult(final_message=f"patched {mod}", fs_diff=diff, ok=True,
                          finalization_status="completed", usage=TokenUsage(input=1, output=1))

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=32)
    phases = _phase_recorder(eng)
    ctx = _ctx(eng, repo, _real_pytest_score(["test_a.py::test_a", "test_b.py::test_b"]),
               responder, {"difficulty": "medium", "modules": ["mod_a", "mod_b"]})
    winner = run_orchestration(DEFAULT_ORCHESTRATION, ctx)
    assert winner is not None and winner.accepted
    assert "autosolve" not in phases     # clean merge accepted -> NO best-of-N fallback


def test_arm_a_easy_path_never_enters_reduce_or_collapse():
    """Arm A invariant: an easy repo skips decomposition entirely -> best-of-N -> never reduce /
    fanout / collapse. The converge fixes are a pure superset gated to the decomposition path."""
    d = Path(tempfile.mkdtemp()) / "repo"
    d.mkdir()
    (d / "only.py").write_text("def f():\n    return 0  # BUG\n")
    (d / "test_only.py").write_text("from only import f\n\ndef test_f():\n    assert f() == 1\n")
    _git(d)

    def responder(task, session):
        if task.schema is not None:
            raise AssertionError("decompose must not run on an easy repo")
        Path(session.cwd, "only.py").write_text("def f():\n    return 1\n")
        diff = subprocess.run(["git", "-C", session.cwd, "diff"], text=True, capture_output=True).stdout
        return ExecResult(final_message="patched", fs_diff=diff, ok=True,
                          finalization_status="completed", usage=TokenUsage(input=1, output=1))

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=32)
    phases = _phase_recorder(eng)
    ctx = _ctx(eng, str(d), _real_pytest_score(["test_only.py::test_f"]),
               responder, {"difficulty": "easy", "modules": ["only"]})
    winner = run_orchestration(DEFAULT_ORCHESTRATION, ctx)
    assert winner is not None and winner.accepted
    assert "reduce" not in phases and "fanout" not in phases    # never entered the converge path
    assert not any((c.meta or {}).get("module") for c in ctx.all_candidates())
