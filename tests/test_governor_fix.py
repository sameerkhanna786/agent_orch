"""Governor large-repo fairness fixes (from orchestration_research/GOVERNOR_AUDIT.md).

Fix 1: a strict DROP in the collection-error count (more of the gold suite now collects) is genuine
implementation progress on a not-yet-passing large repo — it resets BOTH patience arms (and the
sterile streak, Fix 2) WITHOUT banking a gold solve. A genuinely flat run (errors 5091->5091) gets
no credit and is still cut. Fix 3: the in-cell governor's harness_stall_cut is unified to the shared
INDET_CEIL across tiers. All offline.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from apex_omega.autogen.context import OrchestrationContext
from apex_omega.engine.frontier import FrontierTracker, frontier_defaults
from apex_omega.engine.runtime import Engine
from apex_omega.executor.fake import FakeExecutor
from apex_omega.kernel.verify import VerificationResult, candidate_from_verification
from apex_omega.workflows.best_of_n import WorkerSpec


def _repo() -> str:
    d = Path(tempfile.mkdtemp()) / "repo"
    d.mkdir()
    (d / "m.py").write_text("x = 0\n")
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-qm", "base"]):
        subprocess.run(c, cwd=d, check=True, capture_output=True)
    return str(d)


def _ctx():
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=1000)
    return OrchestrationContext(
        eng, executor=FakeExecutor(), worker_specs=[WorkerSpec("codex_cli", "m")],
        source_repo=_repo(), base_commit=None, score_fn=lambda wt: VerificationResult(),
        prompt_builder=lambda c, i, s: "solve", max_agents=64)


def _cand(errors, cid, *, gold=0, total=5091):
    # an EMPTY-diff, VALID measurement: gold flat at `gold`, only the collection-error count varies.
    return candidate_from_verification(
        candidate_id=cid, diff="",
        vr=VerificationResult(passed=gold, errors=errors, total=total, pass_rate=gold / max(1, total)),
        meta={"gold_passed": gold, "gold_total": total, "errors": errors,
              "empty_diff": True, "indeterminate": False})


def _vcand(cid, *, errors=0, gold=0, total=100, failing=None, excerpts="", empty=True):
    # a VALID measurement carrying the full SPFG++ vector inputs (errors / failing_nodeids /
    # failure_excerpts). empty=False emits a NEW distinct diff (active, not sterile).
    failing = list(failing or [])
    return candidate_from_verification(
        candidate_id=cid, diff=("" if empty else cid),
        vr=VerificationResult(passed=gold, errors=errors, total=total,
                              pass_rate=gold / max(1, total),
                              failing_nodeids=failing, failure_excerpts=excerpts),
        meta={"gold_passed": gold, "gold_total": total, "errors": errors,
              "empty_diff": empty, "indeterminate": False,
              "failing_nodeids": failing, "failure_excerpts": excerpts})


# ----------------------------------------------------------------- Fix 3: unified harness ceiling
def test_fix3_in_cell_harness_stall_unified_to_indet_ceil():
    ctx = _ctx()
    assert ctx.governor.harness_stall_cut == frontier_defaults()[2]  # == INDET_CEIL (24), not 8


# ----------------------------------------------------------------- Fix 1+2: context._observe
def test_fix1_collection_error_drop_resets_patience_and_sterile():
    ctx = _ctx()
    ctx._frontier_vector = False                   # pin the errors-only fallback (APEX_FRONTIER_VECTOR=0 ablation)
    ctx._observe([_cand(5091, "a")])              # baseline (establishes the error frontier, NOT a rise)
    ctx._observe([_cand(5091, "b")])              # flat errors, empty diff -> no progress
    assert ctx._wave_state()["valid_measurements_since_improvement"] >= 1
    assert ctx._sterile_streak >= 1
    ctx._observe([_cand(4000, "c")])              # errors 5091->4000 = progress -> resets BOTH arms
    assert ctx._wave_state()["valid_measurements_since_improvement"] == 0
    assert ctx._sterile_streak == 0               # Fix 2: sterile reset inherits the rise
    assert ctx._best_gold_passed == 0             # NEVER banked as a gold solve (C7)
    assert ctx._best_min_errors == 4000


def test_fix1_flat_errors_still_plateaus_and_goes_sterile():
    ctx = _ctx()
    ctx._frontier_vector = False                   # errors-only fallback
    ctx._observe([_cand(5091, "a")])              # baseline
    base = ctx._sterile_streak
    for i in range(3):                            # errors FLAT at 5091, empty diffs -> no progress
        ctx._observe([_cand(5091, "flat%d" % i)])
    # no creditable progress: the valid-measurement arm advances and the sterile streak climbs.
    assert ctx._wave_state()["valid_measurements_since_improvement"] >= 3
    assert ctx._sterile_streak >= 3
    assert ctx._best_gold_passed == 0


def test_fix1_does_not_bank_solve_on_error_drop():
    ctx = _ctx()
    ctx._frontier_vector = False                   # errors-only fallback
    ctx._observe([_cand(5091, "a")])
    ctx._observe([_cand(0, "b", gold=0)])         # full collection, still 0 gold passing
    assert ctx._best_gold_passed == 0             # collection fixed != a solve
    assert ctx._best_min_errors == 0


# ----------------------------------------------------------------- SPFG++ secondary-frontier VECTOR
def test_spfgpp_failing_len_shrink_resets_patience():
    ctx = _ctx()                                   # vector ON by default
    ctx._observe([_vcand("a", failing=["t1", "t2", "t3", "t4"])])   # baseline
    ctx._observe([_vcand("b", failing=["t1", "t2", "t3", "t4"])])   # flat -> no progress
    assert ctx._wave_state()["valid_measurements_since_improvement"] >= 1
    ctx._observe([_vcand("c", failing=["t1", "t2"])])              # residual shrank -> progress
    assert ctx._wave_state()["valid_measurements_since_improvement"] == 0
    assert ctx._best_gold_passed == 0                              # NEVER a gold solve (C7)
    assert ctx._best_vec["neg_failing_len"] == -2


def test_spfgpp_import_depth_rise_resets_patience_when_errors_flat():
    # errors COUNT flat, but the failing import advances deeper (a.b -> a.b.c.d) = real collection
    # progress the errors-only secondary MISSES (the FM-2 collection-collapse case).
    ctx = _ctx()
    ctx._observe([_vcand("a", errors=50, excerpts="No module named 'a.b'")])      # baseline depth 2
    ctx._observe([_vcand("b", errors=50, excerpts="No module named 'a.b'")])      # flat -> no progress
    assert ctx._wave_state()["valid_measurements_since_improvement"] >= 1
    ctx._observe([_vcand("c", errors=50, excerpts="No module named 'a.b.c.d'")])  # deeper -> progress
    assert ctx._wave_state()["valid_measurements_since_improvement"] == 0
    assert ctx._best_vec["import_depth"] == 4
    assert ctx._best_gold_passed == 0


def test_spfgpp_flat_everything_still_plateaus():
    ctx = _ctx()
    ctx._observe([_vcand("a", errors=50, failing=["t1", "t2"], excerpts="No module named 'a.b'")])
    for i in range(3):                              # everything flat -> genuine no-progress
        ctx._observe([_vcand("f%d" % i, errors=50, failing=["t1", "t2"], excerpts="No module named 'a.b'")])
    assert ctx._wave_state()["valid_measurements_since_improvement"] >= 3
    assert ctx._sterile_streak >= 3
    assert ctx._best_gold_passed == 0


def test_spfgpp_active_new_code_resets_sterile_not_frontier():
    # an agent emitting NEW distinct diffs that improve NO test outcome: NOT sterile (still active)
    # but the no-PROGRESS frontier arm keeps ticking -> eventually cut. The deliberate refinement
    # that closes the false-CONTINUE-forever hole of a raw cum-diff frontier component.
    ctx = _ctx()
    ctx._observe([_vcand("a", errors=50, failing=["t1", "t2"], empty=False)])   # baseline
    for i in range(3):
        ctx._observe([_vcand("new%d" % i, errors=50, failing=["t1", "t2"], empty=False)])  # new diffs, flat outcomes
    assert ctx._sterile_streak == 0                                  # active -> not sterile
    assert ctx._wave_state()["valid_measurements_since_improvement"] >= 3   # but no real progress -> arm advances


def test_spfgpp_parse_import_depth():
    f = OrchestrationContext._parse_import_depth
    assert f("No module named 'a.b.c'") == 3
    assert f("cannot import name 'X' from 'pkg.sub'") == 2
    assert f("E   No module named 'a'\nE   No module named 'a.b.c.d'") == 4
    assert f("some unrelated assertion error") is None
    assert f("") is None


def test_spfgpp_ablation_off_is_errors_only():
    ctx = _ctx()
    ctx._frontier_vector = False
    # failing_len shrink alone must NOT credit when the vector is ablated (errors-only fallback).
    ctx._observe([_vcand("a", errors=50, failing=["t1", "t2", "t3"])])  # baseline
    ctx._observe([_vcand("b", errors=50, failing=["t1"])])             # residual shrank, errors flat
    assert ctx._wave_state()["valid_measurements_since_improvement"] >= 1   # NOT credited (errors flat)


# ----------------------------------------------------------------- Fix 1: FrontierTracker (ladder tier)
def test_fix1_frontier_tracker_errors_drop_resets_arms():
    t = FrontierTracker(7200, 12, 24)
    t.ingest(0, 0.0, valid=True, wall_delta=100.0, errors=5091)   # baseline
    assert t.state()["valid_measurements_since_improvement"] == 0
    t.ingest(0, 0.0, valid=True, wall_delta=100.0, errors=4000)   # progress -> reset
    assert t.state()["valid_measurements_since_improvement"] == 0
    assert t.best == -1 or t.state()["best_gold_passed"] == 0     # never a gold rise
    t.ingest(0, 0.0, valid=True, wall_delta=100.0, errors=4000)   # flat -> arm advances
    assert t.state()["valid_measurements_since_improvement"] == 1


def test_fix1_frontier_tracker_backcompat_no_errors_arg():
    # existing callers pass no errors -> behaves exactly as before (gold-only frontier).
    t = FrontierTracker(7200, 12, 24)
    t.ingest(5, 0.5, valid=True, wall_delta=10.0)
    t.ingest(5, 0.5, valid=True, wall_delta=10.0)
    assert t.state()["best_gold_passed"] == 5
    assert t.state()["valid_measurements_since_improvement"] == 1


def test_fix1_frontier_from_wal_credits_error_drop():
    import json
    rd = Path(tempfile.mkdtemp())
    (rd / "journal").mkdir()
    wal = rd / "journal" / "calls_wal.jsonl"
    def rec(errors):
        return json.dumps({"kind": "score", "status": "committed", "result_status": "ok",
                           "structured_result": {"value": {"passed": 0, "pass_rate": 0.0,
                                                            "errors": errors, "indeterminate": False}}})
    # 1 baseline + 2 error-drops + 1 flat: the last improvement is at valid idx 3, so
    # valid_measurements_since_improvement == 1 (only the flat tail), NOT 4.
    wal.write_text("\n".join([rec(5091), rec(4000), rec(3000), rec(3000)]) + "\n")
    from apex_omega.engine.frontier import frontier_from_wal
    st = frontier_from_wal(rd).as_state()
    assert st["best_gold_passed"] == 0                       # never a gold solve
    assert st["valid_measurements"] == 4
    assert st["valid_measurements_since_improvement"] == 1   # error-drops reset the arm
