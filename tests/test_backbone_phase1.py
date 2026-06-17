"""Backbone Phase 1 — journaled score (resume never re-runs pytest for an unchanged
diff; from_dict re-derives acceptance) + the finite progress-gated relaunch helpers."""

from __future__ import annotations

import importlib.util
import subprocess
import tempfile
from pathlib import Path

from apex_omega.autogen.context import OrchestrationContext
from apex_omega.engine.runtime import Engine
from apex_omega.executor.fake import FakeExecutor
from apex_omega.kernel.verify import VerificationResult
from apex_omega.workflows.best_of_n import WorkerSpec

_RUN_LADDER = Path(__file__).resolve().parents[1] / "scripts" / "run_ladder.py"


def _load_run_ladder():
    spec = importlib.util.spec_from_file_location("run_ladder", _RUN_LADDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git_repo() -> str:
    d = Path(tempfile.mkdtemp()) / "repo"
    d.mkdir()
    (d / "mod.py").write_text("def f():\n    return 1\n")
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-qm", "base"]):
        subprocess.run(c, cwd=d, check=True, capture_output=True)
    return str(d)


class _Res:
    def __init__(self, diff):
        self.fs_diff = diff


def _ctx(score_fn):
    return OrchestrationContext(
        Engine(tempfile.mkdtemp(), run_id="t"), executor=FakeExecutor(),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")], source_repo=_git_repo(),
        base_commit=None, score_fn=score_fn, prompt_builder=lambda c, i, s: "x",
    )


# --- 1.1 journaled score: HIT for an unchanged diff -> no pytest re-run ------ #
def test_score_journaled_hit_no_rerun():
    calls = {"n": 0}

    def score_fn(wt):
        calls["n"] += 1
        return VerificationResult(accepted=True, score=1.0, passed=1, total=1, pass_rate=1.0)

    ctx = _ctx(score_fn)
    res = _Res("--- a/mod.py\n+++ b/mod.py\n+x\n")
    vr1 = ctx._scored("/tmp/wt", res)
    vr2 = ctx._scored("/tmp/wt", res)            # same diff -> journal HIT -> score_fn NOT re-run
    assert calls["n"] == 1
    assert vr1.accepted and vr2.accepted and vr2.passed == 1
    # a DIFFERENT diff is a fresh score (miss)
    ctx._scored("/tmp/wt", _Res("--- a\n+++ b\n+y\n"))
    assert calls["n"] == 2


def test_indeterminate_score_is_not_a_hit():
    calls = {"n": 0}

    def score_fn(wt):
        calls["n"] += 1
        return VerificationResult(accepted=False, score=0.0, indeterminate=True)

    ctx = _ctx(score_fn)
    res = _Res("--- a\n+++ b\n+z\n")
    ctx._scored("/tmp/wt", res)
    ctx._scored("/tmp/wt", res)                  # indeterminate -> infra_nonresult -> re-runs
    assert calls["n"] == 2


# --- 1.1 from_dict re-derives acceptance (lossless on accept fields) --------- #
def test_verificationresult_from_dict_roundtrip():
    vr = VerificationResult(accepted=True, score=0.89, passed=5, failed=2, errors=1, total=8,
                            missing_expected=3, pass_rate=0.625, indeterminate=False,
                            failing_nodeids=["t::a"], failure_excerpts="boom")
    v2 = VerificationResult.from_dict(vr.to_dict())
    assert (v2.accepted, v2.passed, v2.failed, v2.errors, v2.total, v2.missing_expected,
            v2.indeterminate) == (True, 5, 2, 1, 8, 3, False)
    assert abs(v2.pass_rate - 0.625) < 1e-9


# --- 1.3 relaunch progress helpers ------------------------------------------ #
def test_has_journal_and_progress_monotonic():
    rl = _load_run_ladder()
    rd = Path(tempfile.mkdtemp())
    assert rl._has_journal(rd) is False
    jdir = rd / "journal"
    jdir.mkdir(parents=True)
    wal = jdir / "calls_wal.jsonl"
    wal.write_text('{"seq":0,"status":"committed","result_status":"ok","kind":"agent"}\n')
    assert rl._has_journal(rd) is True
    p1 = rl._journal_progress(rd)
    assert p1 == (1, 0)
    wal.write_text(wal.read_text() + '{"seq":1,"status":"committed","result_status":"ok","kind":"agent"}\n')
    p2 = rl._journal_progress(rd)
    assert p2 > p1 and p2 == (2, 1)              # fresh committed work advances progress
