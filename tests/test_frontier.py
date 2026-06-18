"""SPFG+ shared-module tests: FrontierTracker, plateau_verdict, w_meas_effective,
frontier_from_wal (Mode-C), frontier_from_rollouts (Mode-A).

Covers the design's stress cases E1-E9 + the indeterminate guard, clock-start,
BEST-not-LAST, dual-AND, determinism, and budget-scaling.
"""

import json
import os

import pytest

from apex_omega.engine.frontier import (
    DEFAULT_PARAMS,
    FrontierOutcome,
    FrontierParams,
    FrontierTracker,
    frontier_from_rollouts,
    frontier_from_wal,
    plateau_verdict,
    w_meas_effective,
)

W_TIME = 7200.0
W_MEAS = 12
INDET_CEIL = 24


def _tracker(w_time=W_TIME, w_meas=W_MEAS, indet_ceil=INDET_CEIL, w_meas_eff=None):
    return FrontierTracker(w_time, w_meas, indet_ceil, w_meas_eff=w_meas_eff)


def _verdict(t, w_meas_eff=None):
    return plateau_verdict(t.state(), w_meas_eff or t.w_meas_eff, t.w_time, t.indet_ceil)


# ---------------------------------------------------------------------------
# Clock start + indeterminate neutrality
# ---------------------------------------------------------------------------

def test_clock_unstarted_until_first_valid():
    t = _tracker()
    assert t.wall_at_best is None
    assert t.state()["seconds_since_frontier_improved"] == 0
    # indeterminate prefix burns zero patience
    for _ in range(3):
        t.ingest(0, 0.0, valid=False, wall_delta=5000.0)
    s = t.state()
    assert s["valid_measurements"] == 0
    assert s["seconds_since_frontier_improved"] == 0
    assert s["indeterminate_total"] == 3
    assert t.wall_at_best is None
    # first valid starts the clock at 0
    t.ingest(5, 0.5, valid=True, wall_delta=100.0)
    assert t.wall_at_best == 100.0  # reset to wall_accum on this improving ingest
    assert t.state()["best_gold_passed"] == 5


def test_indeterminate_neutral_to_both_arms():
    t = _tracker()
    t.ingest(5, 0.5, valid=True, wall_delta=10.0)
    base = t.state()
    for _ in range(10):
        t.ingest(0, 0.0, valid=False, wall_delta=99999.0)
    s = t.state()
    # no valid advance, no wall advance from indeterminate
    assert s["valid_measurements"] == base["valid_measurements"]
    assert s["seconds_since_frontier_improved"] == base["seconds_since_frontier_improved"]
    assert s["indeterminate_streak"] == 10
    assert s["indeterminate_total"] == 10


# ---------------------------------------------------------------------------
# E1 slow-first-solve / E2 prep
# ---------------------------------------------------------------------------

def test_E1_slow_first_solve():
    t = _tracker()
    for _ in range(3):
        t.ingest(0, 0.0, valid=False, wall_delta=2000.0)
    assert _verdict(t)[0] == FrontierOutcome.CONTINUE.value
    t.ingest(0, 0.0, valid=True, wall_delta=1.0)   # valid but no pass yet
    t.ingest(300, 0.9, valid=True, wall_delta=1.0)  # near-full jump
    assert t.state()["best_gold_passed"] == 300
    assert (2, 300) in t.history
    assert _verdict(t)[0] == FrontierOutcome.CONTINUE.value


def test_E2_prefrontier():
    t = _tracker()
    out, why = _verdict(t)
    assert out == FrontierOutcome.CONTINUE.value
    assert why == "pre-frontier"


# ---------------------------------------------------------------------------
# E3 slow-steady (each rise resets both arms)
# ---------------------------------------------------------------------------

def test_E3_slow_steady_never_cuts():
    t = _tracker()
    for i in range(20):
        t.ingest((i + 1) * 5, 0.1, valid=True, wall_delta=1200.0)
        out, _ = _verdict(t)
        assert out == FrontierOutcome.CONTINUE.value
        s = t.state()
        assert s["valid_measurements_since_improvement"] == 0
        assert s["seconds_since_frontier_improved"] == 0


# ---------------------------------------------------------------------------
# E4 true-plateau (dual-AND)
# ---------------------------------------------------------------------------

def test_E4_dual_and_requires_both_floors():
    # meas crossed but not time -> continue
    t = _tracker()
    t.ingest(10, 0.5, valid=True, wall_delta=10.0)
    for _ in range(W_MEAS + 2):
        t.ingest(10, 0.5, valid=True, wall_delta=1.0)
    s = t.state()
    assert s["valid_measurements_since_improvement"] >= W_MEAS
    assert s["seconds_since_frontier_improved"] < W_TIME
    assert _verdict(t)[0] == FrontierOutcome.CONTINUE.value

    # time crossed but not meas -> continue
    t2 = _tracker()
    t2.ingest(10, 0.5, valid=True, wall_delta=10.0)
    t2.ingest(10, 0.5, valid=True, wall_delta=99999.0)
    s2 = t2.state()
    assert s2["seconds_since_frontier_improved"] >= W_TIME
    assert s2["valid_measurements_since_improvement"] < W_MEAS
    assert _verdict(t2)[0] == FrontierOutcome.CONTINUE.value

    # both crossed -> plateau-cut
    t3 = _tracker()
    t3.ingest(10, 0.5, valid=True, wall_delta=10.0)
    for _ in range(W_MEAS + 1):
        t3.ingest(10, 0.5, valid=True, wall_delta=1000.0)
    s3 = t3.state()
    assert s3["valid_measurements_since_improvement"] >= W_MEAS
    assert s3["seconds_since_frontier_improved"] >= W_TIME
    out, why = _verdict(t3)
    assert out == FrontierOutcome.PLATEAU_CUT.value
    assert "flat at 10" in why


# ---------------------------------------------------------------------------
# E5 harness-zero -> harness-stall, not plateau
# ---------------------------------------------------------------------------

def test_E5_harness_stall_distinct_from_plateau():
    t = _tracker()
    for _ in range(INDET_CEIL):
        t.ingest(0, 0.0, valid=False, wall_delta=10000.0)
    out, _ = _verdict(t)
    assert out == FrontierOutcome.INDETERMINATE_CUT.value  # not plateau-cut

    # a valid ingest mid-stream resets indeterminate_streak
    t2 = _tracker()
    for _ in range(INDET_CEIL - 1):
        t2.ingest(0, 0.0, valid=False, wall_delta=1.0)
    t2.ingest(5, 0.5, valid=True, wall_delta=1.0)
    assert t2.state()["indeterminate_streak"] == 0
    assert _verdict(t2)[0] == FrontierOutcome.CONTINUE.value


def test_indeterminate_cut_after_first_valid():
    t = _tracker()
    t.ingest(5, 0.5, valid=True, wall_delta=1.0)
    for _ in range(INDET_CEIL):
        t.ingest(0, 0.0, valid=False, wall_delta=1.0)
    out, why = _verdict(t)
    assert out == FrontierOutcome.INDETERMINATE_CUT.value
    assert "sustained harness failure" in why


# ---------------------------------------------------------------------------
# E6 oscillation (BEST-not-LAST)
# ---------------------------------------------------------------------------

def test_E6_best_not_last():
    t = _tracker()
    for pc in [3, 1, 2, 0, 4]:
        t.ingest(pc, pc / 10.0, valid=True, wall_delta=1.0)
    assert t.state()["best_gold_passed"] == 4
    # last ingest (4 > 3) reset both arms
    assert t.state()["valid_measurements_since_improvement"] == 0
    # the dips did not create false rises in history
    assert [pc for _, pc in t.history] == [3, 4]


# ---------------------------------------------------------------------------
# E7 single-jump-then-stuck
# ---------------------------------------------------------------------------

def test_E7_early_gain_no_permanent_immunity():
    t = _tracker()
    t.ingest(50, 0.5, valid=True, wall_delta=10.0)
    for _ in range(W_MEAS + 1):
        t.ingest(50, 0.5, valid=True, wall_delta=1000.0)
    assert _verdict(t)[0] == FrontierOutcome.PLATEAU_CUT.value


# ---------------------------------------------------------------------------
# E8 fairness / budget-scaling
# ---------------------------------------------------------------------------

def test_E8_w_meas_effective():
    assert w_meas_effective(12, None) == 12
    assert w_meas_effective(12, 8) == 5
    assert w_meas_effective(12, 16) == 10
    assert w_meas_effective(12, 1) == 3


def test_E8_oneshot_never_cuttable():
    # budget=1 -> eff 3, but only 1 valid measurement ever produced.
    eff = w_meas_effective(W_MEAS, 1)
    assert eff == 3
    t = _tracker(w_meas_eff=eff)
    t.ingest(10, 0.5, valid=True, wall_delta=99999.0)  # the single 1-shot measurement
    s = t.state()
    assert s["valid_measurements_since_improvement"] < eff
    assert _verdict(t, eff)[0] == FrontierOutcome.CONTINUE.value


def test_E8_plateaued_best_of_8_is_cut():
    eff = w_meas_effective(W_MEAS, 8)  # 5
    t = _tracker(w_meas_eff=eff)
    t.ingest(10, 0.5, valid=True, wall_delta=10.0)
    for _ in range(eff + 1):
        t.ingest(10, 0.5, valid=True, wall_delta=2000.0)
    assert _verdict(t, eff)[0] == FrontierOutcome.PLATEAU_CUT.value


# ---------------------------------------------------------------------------
# E9 trickle-forever never cuts
# ---------------------------------------------------------------------------

def test_E9_trickle_forever():
    t = _tracker()
    for i in range(200):
        t.ingest(i + 1, (i + 1) / 1000.0, valid=True, wall_delta=5000.0)
        assert _verdict(t)[0] == FrontierOutcome.CONTINUE.value


# ---------------------------------------------------------------------------
# secondary pass_rate tie-break
# ---------------------------------------------------------------------------

def test_pass_rate_secondary_tiebreak_resets():
    t = _tracker()
    t.ingest(10, 0.50, valid=True, wall_delta=10.0)
    for _ in range(W_MEAS):
        t.ingest(10, 0.50, valid=True, wall_delta=1000.0)
    # same count but a strict pass_rate increase => improvement, resets clocks
    t.ingest(10, 0.60, valid=True, wall_delta=1000.0)
    s = t.state()
    assert s["valid_measurements_since_improvement"] == 0
    assert s["seconds_since_frontier_improved"] == 0
    assert _verdict(t)[0] == FrontierOutcome.CONTINUE.value


# ---------------------------------------------------------------------------
# determinism — same inputs -> same verdict
# ---------------------------------------------------------------------------

def test_determinism_same_inputs_same_verdict():
    seq = [(3, 0.1, True, 1000.0), (3, 0.1, True, 1000.0), (5, 0.2, True, 1000.0)]
    a, b = _tracker(), _tracker()
    for args in seq:
        a.ingest(*args)
        b.ingest(*args)
    assert a.state() == b.state()
    assert _verdict(a) == _verdict(b)


# ---------------------------------------------------------------------------
# plateau_verdict edge: valid==0 + indet over ceil
# ---------------------------------------------------------------------------

def test_plateau_verdict_no_valid_indet_ceil():
    s = {"valid_measurements": 0, "indeterminate_total": INDET_CEIL,
         "indeterminate_streak": INDET_CEIL, "valid_measurements_since_improvement": 0,
         "seconds_since_frontier_improved": 0, "best_gold_passed": 0}
    out, why = plateau_verdict(s, W_MEAS, W_TIME, INDET_CEIL)
    assert out == FrontierOutcome.INDETERMINATE_CUT.value
    assert "no valid measurement" in why


# ---------------------------------------------------------------------------
# Mode-C frontier_from_wal
# ---------------------------------------------------------------------------

def _wal_score(seq, status, result_status, value):
    return {"seq": seq, "ts_logical": seq, "kind": "score", "status": status,
            "result_status": result_status,
            "structured_result": ({"value": value} if value is not None else {})}


def test_frontier_from_wal(tmp_path):
    wal = tmp_path / "calls_wal.jsonl"
    lines = [
        _wal_score(1, "in_flight", "ok", None),  # empty payload, skipped
        _wal_score(2, "committed", "ok",
                   {"passed": 5, "total": 10, "pass_rate": 0.5, "indeterminate": False}),
        _wal_score(3, "committed", "infra_nonresult",
                   {"passed": 0, "total": 0, "pass_rate": 0.0, "indeterminate": True}),
        _wal_score(4, "committed", "ok",
                   {"passed": 3, "total": 10, "pass_rate": 0.3, "indeterminate": False}),  # dip
        {"seq": 5, "kind": "agent", "status": "committed"},  # non-score ignored
    ]
    wal.write_text("\n".join(json.dumps(x) for x in lines))
    fs = frontier_from_wal(tmp_path)
    assert fs.mode == "C"
    assert fs.gold_frontier == 5  # best, not last (dip to 3 ignored)
    assert fs.valid_measurements == 2
    assert fs.indeterminate_total == 1
    # determinism: read twice, identical
    fs2 = frontier_from_wal(tmp_path)
    assert fs.as_state() == fs2.as_state()


def test_frontier_from_wal_direct_path(tmp_path):
    wal = tmp_path / "calls_wal.jsonl"
    wal.write_text(json.dumps(_wal_score(
        1, "committed", "ok",
        {"passed": 7, "total": 7, "pass_rate": 1.0, "indeterminate": False})))
    fs = frontier_from_wal(wal)
    assert fs.gold_frontier == 7
    assert fs.valid_measurements == 1


# ---------------------------------------------------------------------------
# Mode-A frontier_from_rollouts
# ---------------------------------------------------------------------------

def test_frontier_from_rollouts(tmp_path):
    rs = tmp_path / "rollout_status"
    rs.mkdir()
    (rs / "rollout_0.json").write_text(json.dumps({
        "rollout_id": 0, "verification_returncode": 0, "verification_passed": 8,
        "verification_selected_test_count": 10, "verification_timed_out": False,
        "quick_verification": {"pass_rate": 0.8}, "last_progress_at": 100.0,
    }))
    (rs / "rollout_1.json").write_text(json.dumps({
        "rollout_id": 1, "failure_reason": "agent stage died", "last_progress_at": 50.0,
        # NO verification_* keys -> indeterminate
    }))
    fs = frontier_from_rollouts(tmp_path)
    assert fs.mode == "A"
    assert fs.gold_frontier == 8
    assert fs.valid_measurements == 1
    assert fs.indeterminate_total == 1


def test_frontier_from_rollouts_indeterminate_rc(tmp_path):
    rs = tmp_path / "rollout_status"
    rs.mkdir()
    (rs / "rollout_0.json").write_text(json.dumps({
        "rollout_id": 0, "verification_returncode": 2,  # not in (0,1)
        "verification_selected_test_count": 0, "verification_passed": 0,
        "last_progress_at": 10.0,
    }))
    fs = frontier_from_rollouts(tmp_path)
    assert fs.gold_frontier == 0
    assert fs.valid_measurements == 0
    assert fs.indeterminate_total == 1


def test_frontier_from_rollouts_scorecard(tmp_path):
    ev = tmp_path / "rollout_evals" / "r0"
    ev.mkdir(parents=True)
    (ev / "candidate_scorecard.json").write_text(json.dumps({
        "candidates": [
            {"rollout_id": 0, "passed": 4, "total_tests": 10, "pass_rate": 0.4,
             "evaluation_status": "ok", "evaluation": {"returncode": 0, "total_tests": 10},
             "last_progress_at": 1.0},
            {"rollout_id": 1, "passed": 9, "total_tests": 10, "pass_rate": 0.9,
             "evaluation_status": "ok", "evaluation": {"returncode": 1, "total_tests": 10},
             "last_progress_at": 2.0},
        ]
    }))
    fs = frontier_from_rollouts(tmp_path)
    assert fs.gold_frontier == 9
    assert fs.valid_measurements == 2


# ---------------------------------------------------------------------------
# params + env
# ---------------------------------------------------------------------------

def test_params_from_env(monkeypatch):
    assert DEFAULT_PARAMS.w_time == 7200.0
    assert DEFAULT_PARAMS.w_meas == 12
    assert DEFAULT_PARAMS.indet_ceil == 24
    monkeypatch.setenv("APEX_FRONTIER_PLATEAU_WALL_S", "100")
    monkeypatch.setenv("LADDER_PLATEAU_MEAS", "4")
    monkeypatch.setenv("APEX_FRONTIER_INDET_CEIL", "9")
    p = FrontierParams.from_env()
    assert p.w_time == 100.0
    assert p.w_meas == 4
    assert p.indet_ceil == 9


# ---------------------------------------------------------------------------
# Regression: Mode-A valid rollout missing both count AND pass_rate
# (must NOT crash; must be non-advancing) — confirmed must-fix.
# ---------------------------------------------------------------------------

def test_frontier_from_rollouts_no_count_no_rate_non_advancing(tmp_path):
    rs = tmp_path / "rollout_status"
    rs.mkdir()
    # A VALID verification (rc=0, sel=10, not timed out, clean class) that carries
    # NEITHER verification_passed NOR quick_verification.pass_rate.
    (rs / "rollout_0.json").write_text(json.dumps({
        "rollout_id": 0, "verification_returncode": 0,
        "verification_selected_test_count": 10, "verification_timed_out": False,
        "last_progress_at": 5.0,
    }))
    fs = frontier_from_rollouts(tmp_path)  # must NOT raise TypeError
    assert fs.mode == "A"
    assert fs.valid_measurements == 1     # it is a real (valid) measurement
    assert fs.gold_frontier == 0          # non-advancing: clamped, no count
    assert fs.history == []               # no strict gold rise recorded
    assert fs.indeterminate_total == 0


def test_frontier_from_rollouts_pass_rate_only_advances_secondary(tmp_path):
    """Valid rollouts with ONLY quick_verification.pass_rate (no count) still
    drive the secondary pass_rate tiebreak (a strict rate rise = progress)."""
    rs = tmp_path / "rollout_status"
    rs.mkdir()
    (rs / "rollout_0.json").write_text(json.dumps({
        "rollout_id": 0, "verification_returncode": 0,
        "verification_selected_test_count": 10, "verification_timed_out": False,
        "quick_verification": {"pass_rate": 0.3}, "last_progress_at": 1.0,
    }))
    (rs / "rollout_1.json").write_text(json.dumps({
        "rollout_id": 1, "verification_returncode": 0,
        "verification_selected_test_count": 10, "verification_timed_out": False,
        "quick_verification": {"pass_rate": 0.7}, "last_progress_at": 2.0,
    }))
    fs = frontier_from_rollouts(tmp_path)
    assert fs.valid_measurements == 2
    assert fs.gold_frontier == 0  # no count frontier
    # the strict rate rise (0.3 -> 0.7) advanced the secondary arm, resetting the clock
    assert fs.last_advance_epoch == 2.0


# ---------------------------------------------------------------------------
# Mode-C: multi-WAL aggregation + un-enveloped (defensive) payload branch.
# ---------------------------------------------------------------------------

def test_frontier_from_wal_multi_wal_aggregation(tmp_path):
    (tmp_path / "cellA").mkdir()
    (tmp_path / "cellB").mkdir()
    walA = tmp_path / "cellA" / "calls_wal.jsonl"
    walB = tmp_path / "cellB" / "calls_wal.jsonl"
    walA.write_text(json.dumps(_wal_score(
        1, "committed", "ok",
        {"passed": 4, "total": 10, "pass_rate": 0.4, "indeterminate": False})))
    walB.write_text(json.dumps(_wal_score(
        1, "committed", "ok",
        {"passed": 9, "total": 10, "pass_rate": 0.9, "indeterminate": False})))
    fs = frontier_from_wal(tmp_path)  # rglob across both
    assert fs.gold_frontier == 9      # max across both WALs
    assert fs.valid_measurements == 2


def test_frontier_from_wal_unenveloped_payload(tmp_path):
    """A score record whose structured_result is the payload directly (no
    {'value': ...} envelope) is tolerated by the defensive _wal_value branch."""
    wal = tmp_path / "calls_wal.jsonl"
    rec = {"seq": 1, "ts_logical": 1, "kind": "score", "status": "committed",
           "result_status": "ok",
           "structured_result": {"passed": 6, "total": 10, "pass_rate": 0.6,
                                 "indeterminate": False}}
    wal.write_text(json.dumps(rec))
    fs = frontier_from_wal(wal)
    assert fs.gold_frontier == 6
    assert fs.valid_measurements == 1
