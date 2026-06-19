"""Phase-planner (the Claude-Code-style HYBRID) unit + integration tests.

Covers the net-new seams: plan_phases (ordered phases with per-phase acceptance, schema-validated,
durable, fail-open), reduce_residuals(scope_ids=...) (pure-set-test phase pass, no extra pytest),
_checkpoint_phase (monotone PARTIAL bank, never solved:1), the _observe frontier-rise checkpoint,
goal_align_gate (grounded no-veer review), last_residual, run_phase (scoped converge body), and the
end-to-end architect.phase_planned_solve. All offline (real git worktrees + real pytest scoring +
FakeExecutor) — no codex burn. Mirrors tests/test_converge_seams.py conventions.
"""

from __future__ import annotations

import json
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
from apex_omega.kernel.verify import VerificationResult, candidate_from_verification
from apex_omega.types import ExecResult, TokenUsage
from apex_omega.workflows.best_of_n import WorkerSpec


# --------------------------------------------------------------------------- helpers
def _two_module_repo() -> str:
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


def _ctx(engine, source_repo, score_fn, responder=None, repo_map=None):
    return OrchestrationContext(
        engine, executor=FakeExecutor(responder),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")],
        source_repo=source_repo, base_commit=None, score_fn=score_fn,
        prompt_builder=lambda c, i, s: "solve", max_agents=32, initial_agents=1,
        repo_map=repo_map or {})


# --------------------------------------------------------------------------- plan_phases
def _plan_decomp():
    return {"modules": [
        {"module": "mod_a", "gold_test_ids": ["test_a.py::test_a"], "depends_on": []},
        {"module": "mod_b", "gold_test_ids": ["test_b.py::test_b"], "depends_on": ["mod_a"]},
    ], "order": ["mod_a", "mod_b"]}


def test_plan_phases_returns_ordered_validated_phases():
    repo = _two_module_repo()

    def responder(task, session):
        props = (task.schema or {}).get("properties") or {}
        if "phases" in props:
            return ExecResult(structured_output={"phases": [
                {"name": "models", "objective": "build mod_a", "modules": ["mod_a"],
                 "acceptance_gold_ids": ["test_a.py::test_a", "BOGUS::id"], "depends_on": []},
                {"name": "api", "objective": "build mod_b", "modules": ["mod_b"],
                 "acceptance_gold_ids": ["test_b.py::test_b"], "depends_on": ["models"]},
            ]}, ok=True, finalization_status="completed", usage=TokenUsage(input=1, output=1))
        return ExecResult(final_message="x", ok=True)

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, lambda wt: VerificationResult(), responder=responder)
    phases = ctx.plan_phases(plan=_plan_decomp(), max_phases=3)
    assert phases is not None and len(phases) == 2
    assert [p["name"] for p in phases] == ["models", "api"]
    # hallucinated acceptance id was dropped (validated against the real gold inventory)
    assert phases[0]["acceptance_gold_ids"] == ["test_a.py::test_a"]
    # durable: phase_plan.json persisted for resume
    assert (Path(eng.run_dir) / "phase_plan.json").exists()


def test_plan_phases_resume_rereads_not_replans():
    repo = _two_module_repo()
    calls = {"n": 0}

    def responder(task, session):
        props = (task.schema or {}).get("properties") or {}
        if "phases" in props:
            calls["n"] += 1
            return ExecResult(structured_output={"phases": [
                {"name": "p1", "objective": "a", "modules": ["mod_a"],
                 "acceptance_gold_ids": ["test_a.py::test_a"], "depends_on": []},
                {"name": "p2", "objective": "b", "modules": ["mod_b"],
                 "acceptance_gold_ids": ["test_b.py::test_b"], "depends_on": []},
            ]}, ok=True, finalization_status="completed", usage=TokenUsage(input=1, output=1))
        return ExecResult(final_message="x", ok=True)

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, lambda wt: VerificationResult(), responder=responder)
    p1 = ctx.plan_phases(plan=_plan_decomp(), max_phases=3)
    p2 = ctx.plan_phases(plan=_plan_decomp(), max_phases=3)   # resume: re-read, no new ask
    assert p1 == p2
    assert calls["n"] == 1                                    # the planner agent ran exactly once


def test_plan_phases_fails_open_to_none_on_degenerate():
    repo = _two_module_repo()

    def responder(task, session):
        props = (task.schema or {}).get("properties") or {}
        if "phases" in props:                                # only ONE valid phase -> degenerate
            return ExecResult(structured_output={"phases": [
                {"name": "only", "objective": "x", "modules": ["mod_a"],
                 "acceptance_gold_ids": ["test_a.py::test_a"], "depends_on": []}]},
                ok=True, finalization_status="completed", usage=TokenUsage(input=1, output=1))
        return ExecResult(final_message="x", ok=True)

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, lambda wt: VerificationResult(), responder=responder)
    assert ctx.plan_phases(plan=_plan_decomp(), max_phases=3) is None


def test_plan_phases_single_module_returns_none():
    repo = _two_module_repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, lambda wt: VerificationResult())
    one = {"modules": [{"module": "mod_a", "gold_test_ids": ["test_a.py::test_a"]}], "order": ["mod_a"]}
    assert ctx.plan_phases(plan=one, max_phases=3) is None    # <2 modules -> no phase plan


# --------------------------------------------------------------------------- reduce_residuals scope_ids
def test_reduce_scope_ids_reports_phase_pass_pure_set_test():
    repo = _two_module_repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, _real_pytest_score(["test_a.py::test_a", "test_b.py::test_b"]))
    diff_a = ("--- a/mod_a.py\n+++ b/mod_a.py\n@@ -1,2 +1,2 @@\n"
              "-def a():\n-    return 0  # BUG\n+def a():\n+    return 1\n")
    ca = candidate_from_verification(candidate_id="ma", diff=diff_a,
                                     vr=VerificationResult(passed=1, total=2, pass_rate=0.5),
                                     meta={"module": "mod_a"})
    # phase scoped to test_a only -> mod_a green -> phase_passed True (full suite still NOT accepted)
    red = ctx.reduce_residuals([ca], scope_ids=["test_a.py::test_a"])
    assert red["accepted"] is False
    assert red["phase_passed"] is True
    assert red["phase_pass_count"] == 1 and red["phase_total"] == 1
    # phase scoped to test_b -> still failing -> phase_passed False
    red2 = ctx.reduce_residuals([ca], scope_ids=["test_b.py::test_b"])
    assert red2["phase_passed"] is False


def test_reduce_without_scope_ids_unchanged():
    """The converge arm (scope_ids=None) is byte-for-byte unchanged: no phase_* keys leak in."""
    repo = _two_module_repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, _real_pytest_score(["test_a.py::test_a", "test_b.py::test_b"]))
    diff_a = ("--- a/mod_a.py\n+++ b/mod_a.py\n@@ -1,2 +1,2 @@\n"
              "-def a():\n-    return 0  # BUG\n+def a():\n+    return 1\n")
    ca = candidate_from_verification(candidate_id="ma", diff=diff_a,
                                     vr=VerificationResult(passed=1, total=2, pass_rate=0.5),
                                     meta={"module": "mod_a"})
    red = ctx.reduce_residuals([ca])
    assert "phase_passed" not in red and "phase_pass_count" not in red
    assert set(red) >= {"merged_diff", "residual_failing_ids", "accepted", "candidate", "conflicts"}


# --------------------------------------------------------------------------- _checkpoint_phase
def test_checkpoint_phase_is_monotone_and_never_solved():
    repo = _two_module_repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, lambda wt: VerificationResult())
    cand = candidate_from_verification(candidate_id="p", diff="--- d\n",
                                       vr=VerificationResult(passed=3, total=5, pass_rate=0.6),
                                       meta={"gold_passed": 3, "gold_total": 5})
    p = Path(eng.run_dir) / "phase_checkpoint.json"
    ctx._checkpoint_phase(cand, subset_passed=3, subset_total=5, phase_id="ph1")
    rec = json.loads(p.read_text())
    assert rec["accepted"] is False and rec["gold_passed"] == 3   # a PARTIAL, never a solve (C7)
    # a LOWER count must NOT overwrite (monotone)
    ctx._checkpoint_phase(cand, subset_passed=2, subset_total=5, phase_id="ph1")
    assert json.loads(p.read_text())["gold_passed"] == 3
    # a STRICT rise overwrites
    ctx._checkpoint_phase(cand, subset_passed=4, subset_total=5, phase_id="ph2")
    assert json.loads(p.read_text())["gold_passed"] == 4


def test_observe_banks_partial_frontier_on_rise():
    repo = _two_module_repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, lambda wt: VerificationResult())
    cand = candidate_from_verification(candidate_id="c", diff="--- d\n",
                                       vr=VerificationResult(passed=2, total=5, pass_rate=0.4),
                                       meta={"gold_passed": 2, "gold_total": 5})
    ctx._observe([cand])                                  # a strict frontier rise (0 -> 2)
    rec = json.loads((Path(eng.run_dir) / "phase_checkpoint.json").read_text())
    assert rec["accepted"] is False and rec["gold_passed"] == 2


# --------------------------------------------------------------------------- goal_align_gate
def _gate_ctx(repo, verdict, evidence_ids):
    def responder(task, session):
        props = (task.schema or {}).get("properties") or {}
        if "verdict" in props:
            return ExecResult(structured_output={"verdict": verdict, "reason": "r",
                                                 "evidence_ids": evidence_ids, "retarget_gold_ids": []},
                              ok=True, finalization_status="completed", usage=TokenUsage(input=1, output=1))
        return ExecResult(final_message="x", ok=True)
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    return _ctx(eng, repo, lambda wt: VerificationResult(), responder=responder)


def test_goal_gate_grounded_abort():
    ctx = _gate_ctx(_two_module_repo(), "abort", ["test_b.py::test_b"])
    g = ctx.goal_align_gate(_plan_decomp(), {"name": "p", "objective": "o", "acceptance_gold_ids": []},
                            residual_ids=["test_b.py::test_b"], stage="pre", n=1)
    assert g["verdict"] == "abort"


def test_goal_gate_ungrounded_dissent_downgraded_to_proceed():
    ctx = _gate_ctx(_two_module_repo(), "abort", ["HALLUCINATED::id"])
    g = ctx.goal_align_gate(_plan_decomp(), {"name": "p", "objective": "o", "acceptance_gold_ids": []},
                            residual_ids=["test_b.py::test_b"], stage="pre", n=1)
    assert g["verdict"] == "proceed"                     # ungrounded abort downgraded (anti-hallucination)


def test_goal_gate_disabled_spawns_no_agents(monkeypatch):
    monkeypatch.setenv("APEX_OMEGA_GOAL_GATE", "0")
    repo = _two_module_repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    fx = FakeExecutor(lambda task, session: ExecResult(final_message="x", ok=True))
    ctx = OrchestrationContext(
        eng, executor=fx, worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")],
        source_repo=repo, base_commit=None, score_fn=lambda wt: VerificationResult(),
        prompt_builder=lambda c, i, s: "solve", max_agents=32)
    g = ctx.goal_align_gate(_plan_decomp(), {"name": "p", "acceptance_gold_ids": []},
                            residual_ids=["test_b.py::test_b"], stage="pre", n=3)
    assert g["verdict"] == "proceed"
    assert fx.calls == 0                                  # gate OFF -> zero agents spent


# --------------------------------------------------------------------------- last_residual
def test_last_residual_tracks_reduce():
    repo = _two_module_repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, _real_pytest_score(["test_a.py::test_a", "test_b.py::test_b"]))
    diff_a = ("--- a/mod_a.py\n+++ b/mod_a.py\n@@ -1,2 +1,2 @@\n"
              "-def a():\n-    return 0  # BUG\n+def a():\n+    return 1\n")
    ca = candidate_from_verification(candidate_id="ma", diff=diff_a,
                                     vr=VerificationResult(passed=1, total=2, pass_rate=0.5),
                                     meta={"module": "mod_a"})
    ctx.reduce_residuals([ca])
    assert any("test_b" in i for i in ctx.last_residual())


# --------------------------------------------------------------------------- run_phase (scoped converge)
def _module_responder():
    def responder(task, session):
        props = (task.schema or {}).get("properties") or {}
        if "verdict" in props:
            return ExecResult(structured_output={"verdict": "proceed", "reason": "", "evidence_ids": [],
                                                 "retarget_gold_ids": []},
                              ok=True, finalization_status="completed", usage=TokenUsage(input=1, output=1))
        if "phases" in props:
            return ExecResult(structured_output={"phases": [
                {"name": "models", "objective": "impl mod_a", "modules": ["mod_a"],
                 "acceptance_gold_ids": ["test_a.py::test_a"], "depends_on": []},
                {"name": "api", "objective": "impl mod_b", "modules": ["mod_b"],
                 "acceptance_gold_ids": ["test_b.py::test_b"], "depends_on": ["models"]},
            ]}, ok=True, finalization_status="completed", usage=TokenUsage(input=1, output=1))
        if "modules" in props:                           # decompose
            return ExecResult(structured_output=_plan_decomp(), ok=True,
                              finalization_status="completed", usage=TokenUsage(input=1, output=1))
        mod = (task.scoped_inputs or {}).get("module", "")
        if mod == "mod_a":
            Path(session.cwd, "mod_a.py").write_text("def a():\n    return 1\n")
        elif mod == "mod_b":
            Path(session.cwd, "mod_b.py").write_text("def b():\n    return 2\n")
        # a well-behaved module agent returns ONLY its own file's diff (disjoint slice)
        diff = subprocess.run(["git", "-C", session.cwd, "diff", "--", f"{mod}.py"],
                              text=True, capture_output=True).stdout
        return ExecResult(final_message=f"patched {mod}", fs_diff=diff, ok=True,
                          finalization_status="completed", usage=TokenUsage(input=1, output=1))
    return responder


def test_run_phase_scoped_to_one_module():
    repo = _two_module_repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, _real_pytest_score(["test_a.py::test_a", "test_b.py::test_b"]),
               responder=_module_responder(),
               repo_map={"difficulty": "medium", "decomposition": _plan_decomp()})
    phase = {"name": "models", "objective": "impl mod_a", "modules": ["mod_a"],
             "acceptance_gold_ids": ["test_a.py::test_a"], "files_owned": ["mod_a.py"]}
    red = ctx.run_phase(phase, carry_diff="")
    assert red["phase_passed"] is True                    # the phase subset (test_a) is green
    assert red["accepted_full"] is False                  # full suite (test_b) not yet
    assert red["merged_diff"].strip()                     # carry to seed the next phase


# --------------------------------------------------------------------------- end-to-end phase_planned_solve
def test_phase_planned_solve_end_to_end():
    from apex_omega.autogen.architect import phase_planned_solve
    repo = _two_module_repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo, _real_pytest_score(["test_a.py::test_a", "test_b.py::test_b"]),
               responder=_module_responder(), repo_map={"difficulty": "medium"})
    winner = phase_planned_solve(eng, ctx, ctx.repo_map)
    assert winner is not None and winner.accepted          # ordered phases -> full suite green
    assert (Path(eng.run_dir) / "phase_plan.json").exists()
    # the whole-suite accepted solve is checkpointed (engine-owned accept, C7)
    assert (Path(eng.run_dir) / "accepted_checkpoint.json").exists()


def test_author_freezes_hybrid_origin(monkeypatch):
    """APEX_OMEGA_ORCHESTRATION=hybrid freezes the converge default as the fall-through body, tagged
    origin='hybrid' so autosolve runs the host-side phase planner around it."""
    from apex_omega.autogen.architect import author_orchestration
    from apex_omega.autogen.templates import DEFAULT_ORCHESTRATION
    monkeypatch.setenv("APEX_OMEGA_ORCHESTRATION", "hybrid")
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=8)
    fw = author_orchestration(eng, executor=FakeExecutor(),
                              worker_specs=[WorkerSpec("codex_cli", "m")],
                              repo_map={"difficulty": "medium"}, author=True)
    assert fw.origin == "hybrid"
    assert fw.source == DEFAULT_ORCHESTRATION and fw.lint_ok


def test_run_ladder_hybrid_arms_present():
    """The A/B arms (converge / hybrid / hybrid-nogate / hybrid-codegen / ralph) are wired with the
    right env overlays — read in a fresh subprocess (run_ladder reads LADDER_ARMS at import)."""
    import subprocess
    import sys
    code = ("import json, scripts.run_ladder as rl;"
            "print(json.dumps([[a[0], (a[2] if len(a) > 2 else {})] for a in rl.ARMS]))")
    env = dict(os.environ)
    env["LADDER_ARMS"] = "converge,hybrid,hybrid-nogate,hybrid-codegen,ralph"
    env["PYTHONPATH"] = os.getcwd()
    proc = subprocess.run([sys.executable, "-c", code], env=env, text=True,
                          capture_output=True, cwd=os.getcwd())
    assert proc.returncode == 0, proc.stderr[-2000:]
    arms = json.loads(proc.stdout.strip().splitlines()[-1])
    by = {a[0]: a[1] for a in arms}
    assert [a[0] for a in arms] == ["converge", "hybrid", "hybrid-nogate", "hybrid-codegen", "ralph"]
    assert by["converge"].get("APEX_OMEGA_ORCHESTRATION") == "converge"
    assert "APEX_OMEGA_PHASE_PLANNER" not in by["converge"]
    assert by["hybrid"].get("APEX_OMEGA_ORCHESTRATION") == "hybrid"
    assert by["hybrid"].get("APEX_OMEGA_PHASE_PLANNER") == "1"
    assert "APEX_OMEGA_GOAL_GATE" not in by["hybrid"]                 # gate ON by default
    assert by["hybrid-nogate"].get("APEX_OMEGA_GOAL_GATE") == "0"     # ablation: gate OFF
    assert by["hybrid-codegen"].get("APEX_OMEGA_PHASE_CODEGEN") == "1"
    assert by["ralph"].get("APEX_OMEGA_ORCHESTRATION") == "ralph"
    # all three hybrid/converge arms carry the SAME repair flips (apples-to-apples)
    for lab in ("converge", "hybrid", "hybrid-nogate"):
        assert by[lab].get("APEX_OMEGA_REPAIR_ITERS") == "2"


def test_phase_planned_solve_easy_returns_none():
    """Easy repos must skip the phase planner entirely (the C3 over-spawn guard)."""
    from apex_omega.autogen.architect import phase_planned_solve
    repo = _two_module_repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    fx = FakeExecutor(lambda task, session: ExecResult(final_message="x", ok=True))
    ctx = OrchestrationContext(
        eng, executor=fx, worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")],
        source_repo=repo, base_commit=None, score_fn=lambda wt: VerificationResult(),
        prompt_builder=lambda c, i, s: "solve", max_agents=32, repo_map={"difficulty": "easy"})
    assert phase_planned_solve(eng, ctx, ctx.repo_map) is None
    assert fx.calls == 0                                   # not a single agent spent on an easy repo
