"""O1/NEW-I2/NEW-I5: lossless candidate frontier across a resume.

The headline silent-loss (networkx scored 2220 but the cell reported 13): on a mid-cell kill the
in-memory _all_candidates starts empty on resume, so carry_best()/select skip the high candidate and
the best frontier is lost. The fix banks every diff-bearing candidate as a durable kind="candidate"
journal record and rebuilds _all_candidates from those records in __init__. Here a SECOND
OrchestrationContext on the SAME run_dir simulates the resume.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from apex_omega.autogen.context import OrchestrationContext
from apex_omega.engine.runtime import Engine
from apex_omega.executor.fake import FakeExecutor
from apex_omega.kernel.verify import VerificationResult, candidate_from_verification
from apex_omega.workflows.best_of_n import WorkerSpec


def _repo() -> str:
    d = Path(tempfile.mkdtemp()) / "repo"
    d.mkdir()
    (d / "f.py").write_text("x = 0\n")
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-qm", "base"]):
        subprocess.run(c, cwd=d, check=True, capture_output=True)
    return str(d)


def _ctx(engine, repo):
    return OrchestrationContext(
        engine, executor=FakeExecutor(None),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")],
        source_repo=repo, base_commit=None,
        score_fn=lambda wt: VerificationResult(),
        prompt_builder=lambda c, i, s: "solve", max_agents=32, initial_agents=1,
    )


_HIGH_DIFF = (
    "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x = 0\n+x = 1\n")


def _high_candidate(cid="a7", gold=2220, diff=_HIGH_DIFF):
    vr = VerificationResult(accepted=False, score=0.9, passed=gold, failed=10,
                            total=gold + 10, pass_rate=0.99, failing_nodeids=[])
    return candidate_from_verification(
        candidate_id=cid, diff=diff, vr=vr, rollout_id=7, cluster_id=7,
        meta={"gold_passed": gold, "gold_total": gold + 10, "indeterminate": False})


def test_bank_then_restore_recovers_frontier():
    repo = _repo()
    run_dir = tempfile.mkdtemp()
    # --- pre-kill: a high candidate is scored + banked ---
    eng1 = Engine(run_dir, run_id="t", max_total_agents=64)
    ctx1 = _ctx(eng1, repo)
    cand = _high_candidate()
    ctx1._all_candidates.append(cand)
    ctx1._bank_candidate(cand)
    assert ctx1.carry_best() == _HIGH_DIFF  # live frontier present

    # --- resume: a fresh engine+ctx on the SAME run_dir must rebuild the frontier ---
    eng2 = Engine(run_dir, run_id="t", max_total_agents=64)
    ctx2 = _ctx(eng2, repo)
    ids = [c.candidate_id for c in ctx2.all_candidates()]
    assert "a7" in ids, "banked candidate was not restored on resume"
    assert ctx2.carry_best() == _HIGH_DIFF, "O1: best partial diff lost on resume (2220->13)"
    restored = next(c for c in ctx2.all_candidates() if c.candidate_id == "a7")
    assert int((restored.meta or {}).get("gold_passed", 0)) == 2220
    assert abs(restored.public_signal_score - 0.99) < 1e-6


def test_empty_diff_candidate_not_banked():
    """A carry-conflict / no-edit candidate carries no restorable work and must not be banked."""
    repo = _repo()
    run_dir = tempfile.mkdtemp()
    eng1 = Engine(run_dir, run_id="t", max_total_agents=64)
    ctx1 = _ctx(eng1, repo)
    empty = candidate_from_verification(
        candidate_id="z0", diff="", vr=VerificationResult(indeterminate=True),
        rollout_id=0, cluster_id=0, meta={"indeterminate": True, "carry_conflict": True})
    ctx1._bank_candidate(empty)
    eng2 = Engine(run_dir, run_id="t", max_total_agents=64)
    ctx2 = _ctx(eng2, repo)
    assert "z0" not in [c.candidate_id for c in ctx2.all_candidates()]


def test_restore_dedupes_by_candidate_id():
    """Re-banking the same candidate (idempotent key) must restore exactly one copy."""
    repo = _repo()
    run_dir = tempfile.mkdtemp()
    eng1 = Engine(run_dir, run_id="t", max_total_agents=64)
    ctx1 = _ctx(eng1, repo)
    cand = _high_candidate(cid="dup1")
    ctx1._bank_candidate(cand)
    ctx1._bank_candidate(cand)  # idempotent re-bank
    eng2 = Engine(run_dir, run_id="t", max_total_agents=64)
    ctx2 = _ctx(eng2, repo)
    assert [c.candidate_id for c in ctx2.all_candidates()].count("dup1") == 1


def test_fresh_run_no_restore_noop():
    """A fresh run (no candidate records) restores nothing -> byte-identical to pre-fix."""
    repo = _repo()
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo)
    assert ctx.all_candidates() == []
    assert ctx.carry_best() == ""


def test_select_sees_restored_accepted_candidate():
    """select() abstains on partials (Cardinal Contract) but MUST rank a restored ACCEPTED
    (full-suite verified) candidate — a kill after the green pass must not lose the solve."""
    repo = _repo()
    run_dir = tempfile.mkdtemp()
    eng1 = Engine(run_dir, run_id="t", max_total_agents=64)
    ctx1 = _ctx(eng1, repo)
    vr = VerificationResult(accepted=True, score=1.0, passed=2230, failed=0,
                            total=2230, pass_rate=1.0, failing_nodeids=[])
    won = candidate_from_verification(
        candidate_id="hi", diff=_HIGH_DIFF, vr=vr, rollout_id=1, cluster_id=1,
        meta={"gold_passed": 2230, "gold_total": 2230})
    ctx1._bank_candidate(won)
    eng2 = Engine(run_dir, run_id="t", max_total_agents=64)
    ctx2 = _ctx(eng2, repo)
    best = ctx2.select(ctx2.all_candidates())
    assert best is not None and best.candidate_id == "hi"


def test_select_still_abstains_on_restored_partial():
    """A restored PARTIAL (accepted=False) is carry-usable but select must still abstain on it."""
    repo = _repo()
    run_dir = tempfile.mkdtemp()
    eng1 = Engine(run_dir, run_id="t", max_total_agents=64)
    ctx1 = _ctx(eng1, repo)
    ctx1._bank_candidate(_high_candidate(cid="part", gold=2220))
    eng2 = Engine(run_dir, run_id="t", max_total_agents=64)
    ctx2 = _ctx(eng2, repo)
    assert ctx2.select(ctx2.all_candidates()) is None        # Cardinal: no accept on a partial
    assert ctx2.carry_best() == _HIGH_DIFF                    # ...but it IS carry-forward usable


def test_governor_cut_banks_best_partial():
    """NEW-I6: a governor CUT must bank the best-so-far partial to phase_checkpoint.json BEFORE the
    halt takes effect, so the cut never discards a frontier a resume could carry."""
    import json
    repo = _repo()
    run_dir = tempfile.mkdtemp()
    eng = Engine(run_dir, run_id="t", max_total_agents=64)
    ctx = _ctx(eng, repo)
    ctx._all_candidates.append(_high_candidate(cid="part", gold=1500))
    # force a cut verdict deterministically
    ctx.governor.verdict = lambda state: (False, "cut:no-progress")
    cont = ctx._wave_verdict(ctx._wave_state())
    assert cont is False and ctx._halted is True and ctx._halt_is_cut is True
    p = Path(run_dir) / "phase_checkpoint.json"
    assert p.exists(), "best-partial was not banked before the cut"
    rec = json.loads(p.read_text())
    assert rec["accepted"] is False and int(rec["gold_passed"]) == 1500


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
