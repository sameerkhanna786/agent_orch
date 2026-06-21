"""Phase-2 convergence seams: decompose / solve_module / reduce_residuals / repair_residual
+ the carry-forward mechanism (apply the running best partial diff into each fresh worktree
BEFORE the agent runs; a 3-way merge conflict -> INDETERMINATE + re-solve, never silent
progress loss).

All offline: real git worktrees + a real pytest subprocess for scoring + the FakeExecutor
(a per-module responder writes fixes into session.cwd). No codex burn.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from apex_omega.autogen.context import DECOMPOSE_SCHEMA, OrchestrationContext
from apex_omega.engine.runtime import Engine
from apex_omega.executor.fake import FakeExecutor
from apex_omega.kernel.verify import VerificationResult
from apex_omega.types import ExecResult, TokenUsage
from apex_omega.workflows.best_of_n import WorkerSpec


# --------------------------------------------------------------------------- helpers
def _two_module_repo() -> str:
    """A synthetic 2-module repo: each module has its own stub + its own failing test."""
    d = Path(tempfile.mkdtemp()) / "repo"
    d.mkdir()
    (d / "mod_a.py").write_text("def a():\n    return 0  # BUG\n")
    (d / "mod_b.py").write_text("def b():\n    return 0  # BUG\n")
    (d / "test_a.py").write_text("from mod_a import a\n\ndef test_a():\n    assert a() == 1\n")
    (d / "test_b.py").write_text("from mod_b import b\n\ndef test_b():\n    assert b() == 2\n")
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-qm", "base"]):
        subprocess.run(c, cwd=d, check=True, capture_output=True)
    return str(d)


def _shared_file_repo() -> str:
    """A repo whose two modules BOTH edit the SAME shared file with conflicting hunks —
    the carry-forward merge-conflict stress fixture."""
    d = Path(tempfile.mkdtemp()) / "repo"
    d.mkdir()
    # one shared file with two adjacent lines each module will rewrite differently
    (d / "shared.py").write_text("X = 0  # a-slot\nY = 0  # b-slot\n")
    (d / "test_shared.py").write_text(
        "from shared import X, Y\n\n"
        "def test_x():\n    assert X == 1\n\n"
        "def test_y():\n    assert Y == 2\n")
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-qm", "base"]):
        subprocess.run(c, cwd=d, check=True, capture_output=True)
    return str(d)


def _real_pytest_score(node_ids):
    """A real-pytest score_fn that runs ONLY the given node ids and extracts failing ids."""
    def _score(wt: str) -> VerificationResult:
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
               "--no-header", *node_ids]
        proc = subprocess.run(cmd, cwd=wt, text=True, capture_output=True,
                              env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PATH": __import__("os").environ["PATH"]},
                              timeout=120)
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        passed = int((re.search(r"(\d+) passed", out) or [0, 0])[1]) if re.search(r"(\d+) passed", out) else 0
        failed = int((re.search(r"(\d+) failed", out) or [0, 0])[1]) if re.search(r"(\d+) failed", out) else 0
        # failing node ids come from the FAILED lines pytest prints in -q short summary
        failing = re.findall(r"^(FAILED|ERROR)\s+(\S+)", out, re.MULTILINE)
        failing_ids = [f[1].split(" ")[0] for f in failing]
        if not failing_ids and failed:
            # pytest -q without -rA may not print FAILED lines; fall back to "::"-bearing tokens
            failing_ids = re.findall(r"\b\S+::\S+\b", out)
        total = passed + failed
        accepted = failed == 0 and passed == len(node_ids)
        return VerificationResult(
            accepted=accepted, score=1.0 if accepted else (passed / max(1, total)),
            passed=passed, failed=failed, total=len(node_ids),
            pass_rate=passed / max(1, len(node_ids)),
            failing_nodeids=failing_ids)
    return _score


def _ctx(engine, source_repo, score_fn, responder=None, repo_map=None):
    return OrchestrationContext(
        engine, executor=FakeExecutor(responder),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")],
        source_repo=source_repo, base_commit=None, score_fn=score_fn,
        prompt_builder=lambda c, i, s: "solve", max_agents=32, initial_agents=1,
        repo_map=repo_map or {},
    )


# --------------------------------------------------------------------------- decompose
def test_decompose_returns_schema_validated_plan():
    repo = _two_module_repo()

    def responder(task, session):
        if task.schema is not None:
            return ExecResult(
                final_message="plan",
                structured_output={
                    "modules": [
                        {"module": "mod_a", "gold_test_ids": ["test_a.py::test_a"], "depends_on": []},
                        {"module": "mod_b", "gold_test_ids": ["test_b.py::test_b"], "depends_on": ["mod_a"]},
                    ],
                    "order": ["mod_a", "mod_b"],
                },
                usage=TokenUsage(input=5, output=5), ok=True, finalization_status="completed")
        return ExecResult(final_message="x", ok=True)

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, lambda wt: VerificationResult(), responder=responder)
    plan = ctx.decompose()
    assert plan is not None
    assert [m["module"] for m in plan["modules"]] == ["mod_a", "mod_b"]
    assert plan["order"] == ["mod_a", "mod_b"]
    # schema is the read-only contract
    from apex_omega.schema_validate import validate_schema
    ok, _ = validate_schema(plan, DECOMPOSE_SCHEMA)
    assert ok
    assert ctx.repo_map.get("decomposition") == plan


def test_decompose_fails_open_to_repo_map_modules():
    repo = _two_module_repo()

    def responder(task, session):
        # schema'd ask returns garbage -> miss on every nudge -> None
        if task.schema is not None:
            return ExecResult(final_message="nope", structured_output={"bad": 1}, ok=True)
        return ExecResult(final_message="x", ok=True)

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, lambda wt: VerificationResult(), responder=responder,
               repo_map={"modules": ["mod_a", "mod_b"]})
    plan = ctx.decompose()
    assert plan is not None and len(plan["modules"]) == 2          # fell open to repo_map['modules']
    assert all(m["gold_test_ids"] == [] for m in plan["modules"])  # degenerate (no gold subset)


def test_decompose_returns_none_when_undecomposable():
    repo = _two_module_repo()

    def responder(task, session):
        if task.schema is not None:
            return ExecResult(final_message="nope", structured_output=None, ok=True)
        return ExecResult(final_message="x", ok=True)

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, lambda wt: VerificationResult(), responder=responder, repo_map={})
    assert ctx.decompose() is None     # no schema reply + no repo_map['modules'] -> best-of-N fallback


# --------------------------------------------------------------------------- solve_module + carry
def test_solve_module_applies_carry_before_agent():
    repo = _two_module_repo()
    seen = {"had_carry": None}

    def responder(task, session):
        # the carry diff (mod_a fix) must be PRESENT in the worktree before this agent edits.
        seen["had_carry"] = "return 1" in Path(session.cwd, "mod_a.py").read_text()
        Path(session.cwd, "mod_b.py").write_text("def b():\n    return 2\n")
        return ExecResult(final_message="patched b", ok=True, finalization_status="completed",
                          usage=TokenUsage(input=1, output=1))

    # carry diff that fixes mod_a
    carry = ("--- a/mod_a.py\n+++ b/mod_a.py\n@@ -1,2 +1,2 @@\n"
             "-def a():\n-    return 0  # BUG\n+def a():\n+    return 1\n")
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, _real_pytest_score(["test_a.py::test_a", "test_b.py::test_b"]),
               responder=responder)
    cand = ctx.solve_module({"module": "mod_b", "gold_test_ids": ["test_b.py::test_b"]}, carry_diff=carry)
    assert seen["had_carry"] is True            # carry-forward applied BEFORE the agent ran
    assert cand is not None and cand.accepted    # carry(a) + agent(b) => full suite green


# --------------------------------------------------------------------------- reduce_residuals
def test_reduce_residuals_merges_module_diffs():
    repo = _two_module_repo()
    # Build two real candidate diffs (one per module) by hand.
    diff_a = ("--- a/mod_a.py\n+++ b/mod_a.py\n@@ -1,2 +1,2 @@\n"
              "-def a():\n-    return 0  # BUG\n+def a():\n+    return 1\n")
    diff_b = ("--- a/mod_b.py\n+++ b/mod_b.py\n@@ -1,2 +1,2 @@\n"
              "-def b():\n-    return 0  # BUG\n+def b():\n+    return 2\n")

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, _real_pytest_score(["test_a.py::test_a", "test_b.py::test_b"]))

    from apex_omega.kernel.verify import candidate_from_verification
    ca = candidate_from_verification(candidate_id="ma", diff=diff_a,
                                     vr=VerificationResult(passed=1, total=2, pass_rate=0.5),
                                     meta={"module": "mod_a"})
    cb = candidate_from_verification(candidate_id="mb", diff=diff_b,
                                     vr=VerificationResult(passed=1, total=2, pass_rate=0.5),
                                     meta={"module": "mod_b"})
    red = ctx.reduce_residuals([ca, cb])
    assert red["accepted"] is True              # both modules merged -> full suite green
    assert red["residual_failing_ids"] == []
    assert red["conflicts"] == []
    assert red["merged_diff"].strip()           # the merged artifact is captured


def test_reduce_residuals_reports_residual_when_one_module_missing():
    repo = _two_module_repo()
    diff_a = ("--- a/mod_a.py\n+++ b/mod_a.py\n@@ -1,2 +1,2 @@\n"
              "-def a():\n-    return 0  # BUG\n+def a():\n+    return 1\n")
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, _real_pytest_score(["test_a.py::test_a", "test_b.py::test_b"]))
    from apex_omega.kernel.verify import candidate_from_verification
    ca = candidate_from_verification(candidate_id="ma", diff=diff_a,
                                     vr=VerificationResult(passed=1, total=2, pass_rate=0.5),
                                     meta={"module": "mod_a"})
    red = ctx.reduce_residuals([ca])
    assert red["accepted"] is False
    assert any("test_b" in i for i in red["residual_failing_ids"])   # b still fails


# --------------------------------------------------------------------------- repair_residual
def test_repair_residual_edits_live_merged_tree():
    repo = _two_module_repo()
    seen = {"merged_present": None}

    def responder(task, session):
        # the merged carry (mod_a fixed) must be in the worktree; the repair agent edits live.
        seen["merged_present"] = "return 1" in Path(session.cwd, "mod_a.py").read_text()
        Path(session.cwd, "mod_b.py").write_text("def b():\n    return 2\n")
        return ExecResult(final_message="repaired b", ok=True, finalization_status="completed",
                          usage=TokenUsage(input=1, output=1))

    carry = ("--- a/mod_a.py\n+++ b/mod_a.py\n@@ -1,2 +1,2 @@\n"
             "-def a():\n-    return 0  # BUG\n+def a():\n+    return 1\n")
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, _real_pytest_score(["test_a.py::test_a", "test_b.py::test_b"]),
               responder=responder)
    cand = ctx.repair_residual(["test_b.py::test_b"], carry_diff=carry, round=0)
    assert seen["merged_present"] is True
    assert cand is not None and cand.accepted


# --------------------------------------------------------------------------- carry-forward conflict
def test_carry_conflict_yields_indeterminate_not_none():
    """A carry diff that cannot apply (overlapping hunks on a shared file) must produce an
    INDETERMINATE Candidate (carry_conflict=True), NOT None and NOT a silent erase."""
    repo = _shared_file_repo()

    def responder(task, session):
        # should NOT be reached when the carry conflicts (we bail before spawn)
        return ExecResult(final_message="ran", ok=True, finalization_status="completed")

    # a carry diff built against DIFFERENT context than the base -> strict + 3way both fail.
    bad_carry = ("--- a/shared.py\n+++ b/shared.py\n@@ -1,2 +1,2 @@\n"
                 "-COMPLETELY = 'different'\n-CONTEXT = 'lines'\n"
                 "+X = 1  # a-slot\n+Y = 0  # b-slot\n")
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    fx = FakeExecutor(responder)
    ctx = OrchestrationContext(
        eng, executor=fx, worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")],
        source_repo=repo, base_commit=None,
        score_fn=_real_pytest_score(["test_shared.py::test_x", "test_shared.py::test_y"]),
        prompt_builder=lambda c, i, s: "solve", max_agents=32)
    cand = ctx.solve_module({"module": "shared", "gold_test_ids": ["test_shared.py::test_x"]},
                            carry_diff=bad_carry)
    assert cand is not None                          # NOT None (distinguishable from infra)
    assert cand.meta.get("carry_conflict") is True   # explicit conflict signal
    assert cand.meta.get("indeterminate") is True
    assert fx.calls == 0                             # the agent was NEVER spawned (carry bailed first)


def test_reduce_conflict_preserves_carry_and_requeues_module():
    """Two module diffs touching the SAME shared file with conflicting hunks: the first applies,
    the second conflicts -> recorded in conflicts[], the merge still scores the first module's
    progress (never silently erased)."""
    repo = _shared_file_repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, _real_pytest_score(["test_shared.py::test_x", "test_shared.py::test_y"]))
    from apex_omega.kernel.verify import candidate_from_verification

    # module A rewrites the whole shared file (fixes X, leaves Y)
    diff_a = ("--- a/shared.py\n+++ b/shared.py\n@@ -1,2 +1,2 @@\n"
              "-X = 0  # a-slot\n-Y = 0  # b-slot\n+X = 1  # a-slot\n+Y = 0  # b-slot\n")
    # module B ALSO rewrites the same two lines (would fix Y) -> overlapping hunk, conflicts.
    diff_b = ("--- a/shared.py\n+++ b/shared.py\n@@ -1,2 +1,2 @@\n"
              "-X = 0  # a-slot\n-Y = 0  # b-slot\n+X = 0  # a-slot\n+Y = 2  # b-slot\n")
    ca = candidate_from_verification(candidate_id="ma", diff=diff_a,
                                     vr=VerificationResult(passed=1, total=2, pass_rate=0.5),
                                     meta={"module": "mod_a"})
    cb = candidate_from_verification(candidate_id="mb", diff=diff_b,
                                     vr=VerificationResult(passed=1, total=2, pass_rate=0.5),
                                     meta={"module": "mod_b"})
    red = ctx.reduce_residuals([ca, cb])
    assert "mod_b" in red["conflicts"]                       # B conflicted, re-queued
    assert red["accepted"] is False
    # A's progress survived: test_x passes (X==1), only test_y residual.
    assert any("test_y" in i for i in red["residual_failing_ids"])
    assert not any("test_x" in i for i in red["residual_failing_ids"])


# --------------------------------------------------------------------------- binary-carry regression
def test_binary_artifact_carry_reapplies_only_with_binary_capture():
    """REGRESSION (babel collapse): a carry diff that CREATES a binary artifact (agent-generated
    locale-data *.dat class) must RE-APPLY into a fresh worktree. `git diff` WITHOUT --binary records
    only 'Binary files differ' (no content) which `git apply` rejects -> every carry-forward / merge
    re-apply CONFLICTS -> indeterminate rounds -> early governor cut (converge 925->2, ralph 4458->
    1165). The fix is `git diff --binary` at capture (v1_executor._git_diff + context._merged_diff)."""
    from apex_omega.isolation.worktree import apply_diff

    def _git(args, cwd):
        return subprocess.run(["git", "-C", str(cwd), *args], text=True, capture_output=True)

    base = Path(tempfile.mkdtemp()) / "src"
    base.mkdir()
    (base / "mod.py").write_text("X = 0\n")
    for c in (["init", "-q"], ["add", "-A"],
              ["-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-qm", "base"]):
        _git(c, base)
    # an agent worktree: create a real binary artifact + edit source (mirrors the executor's add -N).
    wt = Path(tempfile.mkdtemp()) / "wt"
    subprocess.run(["git", "clone", "-q", str(base), str(wt)], check=True, capture_output=True)
    (wt / "data.bin").write_bytes(bytes(range(256)) * 8)
    (wt / "mod.py").write_text("X = 1\n")
    _git(["add", "-N", "."], wt)
    plain = _git(["diff", "--no-color", "HEAD"], wt).stdout            # old capture (BUG)
    binary = _git(["diff", "--binary", "--no-color", "HEAD"], wt).stdout   # the FIX
    assert "Binary files" in plain and "GIT binary patch" not in plain
    assert "GIT binary patch" in binary
    # fresh worktree at base: the plain diff FAILS to apply; the --binary diff SUCCEEDS + recreates it.
    f1 = Path(tempfile.mkdtemp()) / "f1"
    subprocess.run(["git", "clone", "-q", str(base), str(f1)], check=True, capture_output=True)
    assert apply_diff(str(f1), plain) is False
    f2 = Path(tempfile.mkdtemp()) / "f2"
    subprocess.run(["git", "clone", "-q", str(base), str(f2)], check=True, capture_output=True)
    assert apply_diff(str(f2), binary) is True
    assert (f2 / "data.bin").exists() and (f2 / "mod.py").read_text() == "X = 1\n"


# --------------------------------------------------------------------------- merge-reduce-overhaul
def test_apply_diff_partial_lands_clean_hunks_drops_conflicting():
    """HUNK-LEVEL partial apply (#1): a 2-hunk diff where one hunk conflicts must still land the
    OTHER hunk (the work all-or-nothing would shed), and leave NO *.rej residue."""
    import glob
    from apex_omega.isolation.worktree import apply_diff_partial
    d = Path(tempfile.mkdtemp()) / "r"
    d.mkdir()
    pads = "\n".join("# p%d" % i for i in range(1, 13))   # 12 pad lines -> A and B are SEPARATE hunks
    (d / "f.py").write_text("A = 0\n" + pads + "\nB = 0\n")
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-qm", "base"]):
        subprocess.run(c, cwd=d, check=True, capture_output=True)
    # generate a 2-hunk diff (A:0->1 far from B:0->2) via git
    (d / "f.py").write_text("A = 1\n" + pads + "\nB = 2\n")
    D = subprocess.run(["git", "-C", str(d), "diff"], text=True, capture_output=True).stdout
    subprocess.run(["git", "-C", str(d), "checkout", "f.py"], check=True, capture_output=True)
    # now make hunk-1's context STALE so it cannot apply, while hunk-2 (B) still matches base
    (d / "f.py").write_text("A = 9\n" + pads + "\nB = 0\n")
    r = apply_diff_partial(str(d), D)
    txt = (d / "f.py").read_text()
    assert r["clean"] is False and r["applied_any"] is True and r["rejected_hunks"] >= 1
    assert "B = 2" in txt          # the disjoint hunk LANDED (recovered work)
    assert "A = 9" in txt          # the conflicting hunk did NOT apply (worktree value kept)
    assert glob.glob(str(d / "**" / "*.rej"), recursive=True) == []   # no .rej residue


def test_reduce_floor_reverts_regressing_merge_to_best_coherent():
    """NO-SILENT-LOSS floor (#2): when a merge scores BELOW the best banked coherent tree, the
    reduce must carry the BEST tree forward (its diff/residual/gold), never the regression."""
    from apex_omega.kernel.verify import candidate_from_verification
    repo = _two_module_repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    # the merge re-score returns a LOW (regressed) result regardless of tree.
    low = VerificationResult(passed=1, total=10, pass_rate=0.1,
                             failing_nodeids=["t%d" % i for i in range(2, 11)])
    ctx = _ctx(eng, repo, lambda wt: low)
    # bank a STRONG coherent candidate (gold 5) and fold it into the frontier.
    strong = candidate_from_verification(
        candidate_id="strong",
        diff=("--- a/mod_a.py\n+++ b/mod_a.py\n@@ -1,2 +1,2 @@\n"
              "-def a():\n-    return 0  # BUG\n+def a():\n+    return 1\n"),
        vr=VerificationResult(passed=5, total=10, pass_rate=0.5,
                              failing_nodeids=["t6", "t7", "t8", "t9", "t10"]),
        meta={"gold_passed": 5, "gold_total": 10, "indeterminate": False, "empty_diff": False,
              "failing_nodeids": ["t6", "t7", "t8", "t9", "t10"], "finalization_status": "completed"})
    ctx._all_candidates.append(strong)
    ctx._observe([strong])
    assert ctx._best_gold_passed == 5
    # a weak module whose merged tree re-scores at gold 1 (< 5) -> floor reverts to strong.
    weak = candidate_from_verification(
        candidate_id="weak",
        diff=("--- a/mod_b.py\n+++ b/mod_b.py\n@@ -1,2 +1,2 @@\n"
              "-def b():\n-    return 0  # BUG\n+def b():\n+    return 2\n"),
        vr=VerificationResult(passed=1, total=10, pass_rate=0.1), meta={"module": "mod_b", "gold_passed": 1})
    red = ctx.reduce_residuals([weak])
    assert red["floored"] is True
    assert red["gold_passed"] == 5                         # carried the best, not the regression
    assert red["merged_diff"] == strong.diff
    assert red["residual_failing_ids"] == ["t6", "t7", "t8", "t9", "t10"]


# --------------------------------------------------------------------------- merge-reduce v2 (coupled)
def _overlap_cands():
    from apex_omega.kernel.verify import candidate_from_verification
    da = "--- a/shared.py\n+++ b/shared.py\n@@ -1 +1 @@\n-x = 0\n+x = 1\n"
    db = "--- a/shared.py\n+++ b/shared.py\n@@ -1 +1 @@\n-x = 0\n+x = 2\n"
    ca = candidate_from_verification(candidate_id="a", diff=da, vr=VerificationResult(passed=1), meta={"module": "a", "gold_passed": 1})
    cb = candidate_from_verification(candidate_id="b", diff=db, vr=VerificationResult(passed=1), meta={"module": "b", "gold_passed": 1})
    return ca, cb


def test_reduce_surfaces_coupling_telemetry():
    repo = _two_module_repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, _real_pytest_score(["test_a.py::test_a", "test_b.py::test_b"]))
    from apex_omega.kernel.verify import candidate_from_verification
    diff_a = ("--- a/mod_a.py\n+++ b/mod_a.py\n@@ -1,2 +1,2 @@\n"
              "-def a():\n-    return 0  # BUG\n+def a():\n+    return 1\n")
    ca = candidate_from_verification(candidate_id="ma", diff=diff_a,
                                     vr=VerificationResult(passed=1, total=2, pass_rate=0.5), meta={"module": "mod_a"})
    red = ctx.reduce_residuals([ca])
    # v2 telemetry keys are present + typed (clean disjoint apply -> no shedding)
    assert red["max_rejected_hunks"] == 0 and red["n_partial_merged"] == 0
    assert red["conflict_frac"] == 0.0
    assert isinstance(red["advanced"], bool)


def test_coupled_plateau_streak_and_gates(monkeypatch):
    repo = _two_module_repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, lambda wt: VerificationResult())
    ca, cb = _overlap_cands()                       # two OVERLAPPING modules (same shared.py)
    ctx._best_gold_passed = 937
    hi = {"gold_passed": 937, "conflict_frac": 1.0, "max_rejected_hunks": 50, "conflicts": ["a", "b"]}
    # gold==0 total-collapse is owned by the collapse fallback, never the integrator
    assert ctx.coupled_plateau({**hi, "gold_passed": 0, "advanced": False}, [ca, cb]) is False
    # a multi-cand high-conflict reduce that ADVANCED latches coupling but does NOT switch (climbing)
    assert ctx.coupled_plateau({**hi, "advanced": True}, [ca, cb]) is False
    # then a SUSTAINED non-advancing plateau (single-cand loop reduces) -> streak builds -> switch at 2
    flat = {"gold_passed": 937, "conflict_frac": 0.0, "max_rejected_hunks": 0, "conflicts": [], "advanced": False}
    assert ctx.coupled_plateau(flat, [ca]) is False   # streak 1
    assert ctx.coupled_plateau(flat, [ca]) is True     # streak 2 -> SWITCH
    # a frontier rise mid-streak resets it (a climbing loop is never switched)
    assert ctx.coupled_plateau({**flat, "advanced": True}, [ca]) is False
    assert ctx.coupled_plateau(flat, [ca]) is False     # streak back to 1


def test_coupled_plateau_disjoint_and_flag_off(monkeypatch):
    from apex_omega.kernel.verify import candidate_from_verification
    repo = _two_module_repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, lambda wt: VerificationResult())
    ctx._best_gold_passed = 5
    # DISJOINT clean modules (different files) -> never coupled even if it plateaus
    da = candidate_from_verification(candidate_id="a", diff="--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a=0\n+a=1\n", vr=VerificationResult(passed=1), meta={"module": "a"})
    db = candidate_from_verification(candidate_id="b", diff="--- a/y.py\n+++ b/y.py\n@@ -1 +1 @@\n-b=0\n+b=1\n", vr=VerificationResult(passed=1), meta={"module": "b"})
    clean = {"gold_passed": 5, "conflict_frac": 0.0, "max_rejected_hunks": 0, "conflicts": [], "advanced": False}
    assert ctx.coupled_plateau(clean, [da, db]) is False
    assert ctx.coupled_plateau(clean, [da, db]) is False   # never latches (disjoint)
    # flag OFF -> always False even on the coupled shape
    monkeypatch.setenv("APEX_OMEGA_COHERENT_INTEGRATOR", "0")
    ca, cb = _overlap_cands()
    assert ctx.coupled_plateau({"gold_passed": 937, "conflict_frac": 1.0, "max_rejected_hunks": 50,
                                "conflicts": ["a", "b"], "advanced": False}, [ca, cb]) is False


def test_run_phase_routes_to_integrator_on_coupled_plateau(monkeypatch):
    # the PHASED path (hybrid arms) must ALSO switch to the coherent integrator on a coupled plateau,
    # not just the converge default. Stub the heavy seams + force coupled_plateau -> verify the
    # integrator (ralph_loop) is invoked, seeded by carry_best, and its candidate becomes the result.
    repo = _two_module_repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, lambda wt: VerificationResult())
    ctx.repo_map["decomposition"] = {"modules": [{"module": "mod_a", "gold_test_ids": ["t1"]}]}
    from apex_omega.kernel.verify import candidate_from_verification
    wcand = candidate_from_verification(
        candidate_id="integ", diff="--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n",
        vr=VerificationResult(passed=3, total=5), meta={"gold_passed": 3, "failing_nodeids": ["t4", "t5"]})
    spy = {"ralph": 0, "seed": None, "brief": None}
    monkeypatch.setattr(ctx, "fanout_modules", lambda *a, **k: [wcand])
    monkeypatch.setattr(ctx, "reduce_residuals", lambda *a, **k: {
        "merged_diff": "d", "residual_failing_ids": ["t4", "t5"], "accepted": False,
        "gold_passed": 3, "conflicts": ["m1", "m2"]})
    monkeypatch.setattr(ctx, "repair_residual", lambda *a, **k: wcand)
    monkeypatch.setattr(ctx, "coupled_plateau", lambda red, cands: True)     # force the switch
    monkeypatch.setattr(ctx, "should_continue_waves", lambda: True)
    monkeypatch.setattr(ctx, "carry_best", lambda: "BEST_TREE")
    monkeypatch.setattr(ctx, "integrator_brief", lambda m, ids: "BRIEF")

    def _spy_ralph(*, id_base=0, seed_carry=None, brief=None):
        spy["ralph"] += 1
        spy["seed"] = seed_carry
        spy["brief"] = brief
        return wcand
    monkeypatch.setattr(ctx, "ralph_loop", _spy_ralph)
    res = ctx.run_phase({"modules": ["mod_a"], "acceptance_gold_ids": ["t1", "t2", "t3", "t4", "t5"],
                         "name": "p0"}, phase_index=0)
    assert spy["ralph"] == 1                             # integrator invoked from the phased path
    assert spy["seed"] == "BEST_TREE" and spy["brief"] == "BRIEF"
    assert res["candidate"] is wcand
    assert res["phase_pass_count"] == 3 and res["phase_total"] == 5   # t1,t2,t3 green; t4,t5 residual
    assert res["phase_passed"] is False


def test_reset_patience_rebases_clocks_not_frontier():
    repo = _two_module_repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, lambda wt: VerificationResult())
    ctx._best_gold_passed = 900
    ctx._valid_measurements = 20
    ctx._valid_measurements_at_best = 5      # a stale (large) plateau gap
    ctx._sterile_streak = 9
    ctx._reset_patience()
    assert ctx._valid_measurements_at_best == 20      # rebased to NOW (no plateau)
    assert ctx._sterile_streak == 0
    assert ctx._best_gold_passed == 900               # frontier UNTOUCHED (must still be beaten)


# --------------------------------------------------------------------------- carry_best
def test_carry_best_picks_highest_gold_partial():
    repo = _two_module_repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, lambda wt: VerificationResult())
    from apex_omega.kernel.verify import candidate_from_verification
    weak = candidate_from_verification(candidate_id="w", diff="--- weak\n",
                                       vr=VerificationResult(passed=1, total=4, pass_rate=0.25),
                                       meta={"gold_passed": 1})
    strong = candidate_from_verification(candidate_id="s", diff="--- strong\n",
                                         vr=VerificationResult(passed=3, total=4, pass_rate=0.75),
                                         meta={"gold_passed": 3})
    indet = candidate_from_verification(candidate_id="i", diff="--- indet\n",
                                        vr=VerificationResult(passed=9, indeterminate=True),
                                        meta={"gold_passed": 9, "indeterminate": True})
    ctx._all_candidates.extend([weak, strong, indet])
    assert ctx.carry_best() == "--- strong\n"     # highest VALID gold_passed (indeterminate ignored)


# --------------------------------------------------------------------------- prompt-contract wiring
def test_seams_use_repo_map_brief_builders():
    """solve_module + repair_residual prefer the eval-provided CONTRACT builders on
    repo_map['brief_builders'] (the live wiring) over their built-in briefs."""
    repo = _two_module_repo()
    seen = {"module_prompt": "", "residual_prompt": ""}

    def module_solve(ctx, module, gold_ids, *, carry_nonempty):
        return f"CONTRACT1 module={module} ids={list(gold_ids)} carry={carry_nonempty}"

    def residual_repair(ctx, failing_nodeids, passed, total, *, excerpts=""):
        return f"CONTRACT2 ids={list(failing_nodeids)} state={passed}/{total} ex={excerpts}"

    def responder(task, session):
        if "CONTRACT1" in task.prompt:
            seen["module_prompt"] = task.prompt
        if "CONTRACT2" in task.prompt:
            seen["residual_prompt"] = task.prompt
        Path(session.cwd, "mod_b.py").write_text("def b():\n    return 2\n")
        return ExecResult(final_message="ok", ok=True, finalization_status="completed",
                          usage=TokenUsage(input=1, output=1))

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, _real_pytest_score(["test_a.py::test_a", "test_b.py::test_b"]),
               responder=responder,
               repo_map={"brief_builders": {"module_solve": module_solve,
                                            "residual_repair": residual_repair}})
    ctx.solve_module({"module": "mod_b", "gold_test_ids": ["test_b.py::test_b"]})
    assert "CONTRACT1 module=mod_b" in seen["module_prompt"]
    assert "ids=['test_b.py::test_b']" in seen["module_prompt"]
    ctx.repair_residual(["test_b.py::test_b"], carry_diff="", round=0)
    assert "CONTRACT2 ids=['test_b.py::test_b']" in seen["residual_prompt"]


# --------------------------------------------------------------------------- end-to-end converge
def test_end_to_end_decompose_fanout_reduce_loop():
    """The full convergence shape over a real 2-module repo: decompose -> per-module fan-out
    (each agent greens its module) -> reduce merges -> accept. Deterministic via FakeExecutor."""
    repo = _two_module_repo()

    def responder(task, session):
        if task.schema is not None:   # decompose
            return ExecResult(structured_output={
                "modules": [
                    {"module": "mod_a", "gold_test_ids": ["test_a.py::test_a"], "depends_on": []},
                    {"module": "mod_b", "gold_test_ids": ["test_b.py::test_b"], "depends_on": []},
                ], "order": ["mod_a", "mod_b"]}, ok=True)
        # module solve: implement the module named in the scoped inputs
        mod = (task.scoped_inputs or {}).get("module", "")
        if mod == "mod_a":
            Path(session.cwd, "mod_a.py").write_text("def a():\n    return 1\n")
        elif mod == "mod_b":
            Path(session.cwd, "mod_b.py").write_text("def b():\n    return 2\n")
        # mirror a real session: the authoritative artifact is the worktree git diff
        diff = subprocess.run(["git", "-C", session.cwd, "diff"], text=True,
                              capture_output=True).stdout
        return ExecResult(final_message=f"patched {mod}", fs_diff=diff, ok=True,
                          finalization_status="completed", usage=TokenUsage(input=1, output=1))

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    score = _real_pytest_score(["test_a.py::test_a", "test_b.py::test_b"])
    ctx = _ctx(eng, repo, score, responder=responder)

    plan = ctx.decompose()
    assert plan is not None and len(plan["modules"]) == 2
    carry = ""
    cands = [ctx.solve_module(m, carry_diff=carry) for m in plan["modules"]]
    assert all(c is not None for c in cands)
    red = ctx.reduce_residuals(cands, carry_diff=carry)
    assert red["accepted"] is True and red["residual_failing_ids"] == []
    winner = ctx.select(ctx.all_candidates())
    assert winner is not None and winner.accepted
