"""O2/O3/O4 redesign: ctx.diagnose() STAGE-2 fusion, ctx.review_plan() advisory review, and the
Phase-0 / re-order helpers. All gated -> OFF is byte-identical to the hybrid-nogate baseline.

Offline (FakeExecutor responders mock the read-only scouts); no codex burn.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from apex_omega.autogen.architect import _reorder_plan_modules, _synthesize_phase0
from apex_omega.autogen.context import OrchestrationContext
from apex_omega.engine.runtime import Engine
from apex_omega.executor.fake import FakeExecutor
from apex_omega.kernel.verify import VerificationResult
from apex_omega.types import ExecResult, TokenUsage
from apex_omega.workflows.best_of_n import WorkerSpec


def _repo() -> str:
    d = Path(tempfile.mkdtemp()) / "repo"
    d.mkdir()
    (d / "f.py").write_text("x = 0\n")
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-qm", "base"]):
        subprocess.run(c, cwd=d, check=True, capture_output=True)
    return str(d)


def _ctx(engine, repo, responder=None, repo_map=None):
    return OrchestrationContext(
        engine, executor=FakeExecutor(responder),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")],
        source_repo=repo, base_commit=None,
        score_fn=lambda wt: VerificationResult(),
        prompt_builder=lambda c, i, s: "solve", max_agents=32, initial_agents=1,
        repo_map=repo_map or {})


@pytest.fixture(autouse=True)
def _clear_gates():
    keys = ["APEX_OMEGA_DIAG", "APEX_OMEGA_PLAN_REVIEW", "APEX_OMEGA_PHASE0"]
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# --------------------------------------------------------------------- diagnose()
def test_diagnose_off_is_noop():
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, _repo())
    assert ctx.diagnose() == {}
    assert eng.agents_used() == 0      # no scouts spawned when gated off


def test_diagnose_fuses_ast_and_scout_and_factchecks():
    os.environ["APEX_OMEGA_DIAG"] = "1"
    # AST pre-pass result is normally injected by build_repo_map; inject it directly on repo_map.
    ast_diag = {
        "collects_cleanly": False,
        "unresolved_internal": [{"module": "pkg.schema", "symbol": "GenerateSchema",
                                 "importer": "conftest.py", "reason": "missing_symbol"}],
        "unresolved_external": [], "import_depth": 1, "addopts": "",
        "suspect_plugin_addopts": [], "evidence": ["1 unresolved internal import"],
        "first_failing_import": {"module": "pkg.schema", "symbol": "GenerateSchema"},
    }

    def responder(task, session):
        props = (task.schema or {}).get("properties") or {}
        if "blocker_class" in props:
            return ExecResult(structured_output={
                "blocker_class": "collection_error",
                "must_implement_modules": ["pkg.schema", "pkg.HALLUCINATED"],  # 2nd not in AST
                "import_chain": ["conftest.py", "pkg.schema"],
                "suggested_first_fix": "implement GenerateSchema in pkg/schema.py",
                "evidence": ["conftest imports GenerateSchema"]},
                ok=True, finalization_status="completed", usage=TokenUsage(input=1, output=1))
        return ExecResult(final_message="x", ok=True)

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, _repo(), responder=responder, repo_map={"diagnosis": ast_diag, "modules": ["pkg"]})
    diag = ctx.diagnose(n=1)
    assert diag["blocker_class"] == "collection_error"
    assert diag["collects_cleanly"] is False
    assert "pkg.schema" in diag["must_implement_modules"]
    assert "pkg.HALLUCINATED" not in diag["must_implement_modules"]   # fact-checked out
    assert diag["suggested_first_fix"].startswith("implement GenerateSchema")
    # cached: a second call does not re-spawn scouts
    n_after = eng.agents_used()
    ctx.diagnose(n=1)
    assert eng.agents_used() == n_after


def test_diagnose_collection_failure_overrides_scout_classification():
    os.environ["APEX_OMEGA_DIAG"] = "1"
    ast_diag = {"collects_cleanly": False,
                "unresolved_internal": [{"module": "pkg.core", "symbol": None,
                                         "importer": "conftest.py", "reason": "missing_module"}],
                "unresolved_external": [], "import_depth": 0, "evidence": []}

    def responder(task, session):
        props = (task.schema or {}).get("properties") or {}
        if "blocker_class" in props:   # scout wrongly says implementation_gap
            return ExecResult(structured_output={"blocker_class": "implementation_gap"},
                              ok=True, finalization_status="completed", usage=TokenUsage(input=1, output=1))
        return ExecResult(final_message="x", ok=True)

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, _repo(), responder=responder, repo_map={"diagnosis": ast_diag, "modules": ["pkg"]})
    # a STATIC collection failure is execution-grounded reality -> overrides the scout vote
    assert ctx.diagnose(n=1)["blocker_class"] == "collection_error"


# ------------------------------------------------------------------- review_plan()
_PLAN = {"modules": [
    {"module": "api", "gold_test_ids": ["test_api.py::t"], "depends_on": ["core"]},
    {"module": "core", "gold_test_ids": ["test_core.py::t"], "depends_on": []},
], "order": ["api", "core"]}


def test_review_plan_off_is_noop():
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, _repo(), repo_map={"modules": ["api", "core"]})
    rv = ctx.review_plan(_PLAN, seam="decompose")
    assert rv["verdict"] == "proceed"
    assert eng.agents_used() == 0


def test_review_plan_revise_is_grounded_and_bounded():
    os.environ["APEX_OMEGA_PLAN_REVIEW"] = "1"

    def responder(task, session):
        props = (task.schema or {}).get("properties") or {}
        if "first_modules" in props:
            return ExecResult(structured_output={
                "verdict": "revise", "reason": "core is a prerequisite of api",
                "first_modules": ["core"], "missing_modules": [], "evidence": ["core"]},
                ok=True, finalization_status="completed", usage=TokenUsage(input=1, output=1))
        return ExecResult(final_message="x", ok=True)

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, _repo(), responder=responder, repo_map={"modules": ["api", "core"]})
    diag = {"blocker_class": "implementation_gap", "collects_cleanly": True,
            "must_implement_modules": ["core"]}
    rv = ctx.review_plan(_PLAN, seam="decompose", diagnosis=diag, n=1)
    assert rv["verdict"] == "revise" and "core" in rv["first_modules"]
    # bounded: the SAME seam is not reviewed twice (returns proceed without spawning)
    n_after = eng.agents_used()
    rv2 = ctx.review_plan(_PLAN, seam="decompose", diagnosis=diag, n=1)
    assert rv2["verdict"] == "proceed" and eng.agents_used() == n_after


def test_review_plan_downgrades_ungrounded_revise():
    os.environ["APEX_OMEGA_PLAN_REVIEW"] = "1"

    def responder(task, session):
        props = (task.schema or {}).get("properties") or {}
        if "first_modules" in props:   # names a module that is NOT in the plan or diagnosis
            return ExecResult(structured_output={
                "verdict": "revise", "first_modules": ["ghost"], "missing_modules": [],
                "evidence": []}, ok=True, finalization_status="completed",
                usage=TokenUsage(input=1, output=1))
        return ExecResult(final_message="x", ok=True)

    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, _repo(), responder=responder, repo_map={"modules": ["api", "core"]})
    rv = ctx.review_plan(_PLAN, seam="decompose",
                         diagnosis={"must_implement_modules": ["core"]}, n=1)
    assert rv["verdict"] == "proceed"   # ungrounded "ghost" -> downgraded


# --------------------------------------------------------------- plan helpers
def test_reorder_floats_first_modules():
    np = _reorder_plan_modules(_PLAN, ["core"])
    assert [m["module"] for m in np["modules"]] == ["core", "api"]
    assert np["order"] == ["core", "api"]
    # module set + gold ids unchanged
    assert {m["module"] for m in np["modules"]} == {"core", "api"}


def test_reorder_noop_when_no_match():
    np = _reorder_plan_modules(_PLAN, ["nope"])
    assert [m["module"] for m in np["modules"]] == ["api", "core"]


def test_synthesize_phase0_targets_must_implement_gold_ids():
    diag = {"must_implement_modules": ["core"], "collects_cleanly": False,
            "suggested_first_fix": "implement core"}
    p0 = _synthesize_phase0(diag, _PLAN)
    assert p0 is not None and p0["name"] == "phase0-collect" and p0["is_phase0"] is True
    assert p0["acceptance_gold_ids"] == ["test_core.py::t"]
    assert "COLLECT" in p0["objective"]


def test_synthesize_phase0_falls_back_to_full_suite():
    diag = {"must_implement_modules": ["unmatched"], "collects_cleanly": False}
    p0 = _synthesize_phase0(diag, _PLAN)
    assert p0 is not None
    assert set(p0["acceptance_gold_ids"]) == {"test_api.py::t", "test_core.py::t"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
