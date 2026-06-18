"""Tier-1 SPFG+ gate: run_ladder's relaunch/continue decision is FRONTIER-based.

The activity gate (`prog<=last_prog` over committed WAL records) force-cut a cell whose
GOLD pass-count was still climbing (the minitorch case: ~177/369 over ~10h/6 relaunches).
SPFG+ replaces that with `relaunch_decision`: relaunch while the gold frontier rises (never
bound total wall-time), cut ONLY on a dual-AND no-progress plateau (cut:no-progress) or a
sustained indeterminate/harness wall (cut:harness-stall), and ALWAYS recover a banked
verified solve FIRST.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

_RUN_LADDER = Path(__file__).resolve().parents[1] / "scripts" / "run_ladder.py"


def _load_run_ladder():
    spec = importlib.util.spec_from_file_location("run_ladder", _RUN_LADDER)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _wal_score(seq, passed, total, pass_rate, *, indeterminate=False, result_status="ok"):
    return {
        "seq": seq, "kind": "score", "status": "committed",
        "result_status": result_status,
        "structured_result": {"value": {
            "passed": passed, "total": total, "pass_rate": pass_rate,
            "indeterminate": indeterminate}},
    }


def _write_wal(rundir: Path, records) -> None:
    jdir = rundir / "journal"
    jdir.mkdir(parents=True, exist_ok=True)
    wal = jdir / "calls_wal.jsonl"
    wal.write_text("\n".join(json.dumps(r) for r in records) + "\n")


# --- module-level params (env defaults) -------------------------------------- #
def test_frontier_params_loaded_from_env_defaults():
    rl = _load_run_ladder()
    assert (rl.W_TIME, rl.W_MEAS, rl.INDET_CEIL, rl.POLL_S) == (7200, 12, 24, 300)


def test_arm_budget_derivation():
    rl = _load_run_ladder()
    p = Path("/tmp")
    assert rl._arm_budget("b0", ["--arms", "B0_single_model"], p) == 1
    assert rl._arm_budget("k8", ["--arms", "baseline", "--rollouts", "8"], p) == 8
    assert rl._arm_budget("b2", ["--arms", "B2_v1_full_cap16"], p) == 16
    assert rl._arm_budget(
        "o", ["--arms", "autogen_orchestrator", "--autogen-max-agents", "1000"], p) is None
    # ralph: autogen + scout 0 is still unbounded
    assert rl._arm_budget(
        "r", ["--arms", "autogen_orchestrator", "--autogen-scout-agents", "0",
              "--autogen-max-agents", "1000"], p) is None


# --- frontier_state dispatch ------------------------------------------------- #
def test_frontier_state_mode_c_reads_wal():
    rl = _load_run_ladder()
    rd = Path(tempfile.mkdtemp())
    _write_wal(rd, [
        _wal_score(0, 5, 10, 0.5),
        _wal_score(1, 0, 0, 0.0, indeterminate=True, result_status="infra_nonresult"),
        _wal_score(2, 3, 10, 0.3),   # dip below best -> frontier stays 5
        _wal_score(3, 8, 10, 0.8),
    ])
    fs = rl.frontier_state(rd)
    assert fs.mode == "C"
    assert fs.gold_frontier == 8
    assert fs.valid_measurements == 3       # the infra_nonresult is excluded
    assert fs.indeterminate_total == 1


def test_frontier_state_mode_a_when_no_wal():
    rl = _load_run_ladder()
    rd = Path(tempfile.mkdtemp())
    rsdir = rd / "rollout_status"
    rsdir.mkdir(parents=True)
    (rsdir / "rollout_0.json").write_text(json.dumps({
        "verification_returncode": 0, "verification_selected_test_count": 8,
        "verification_passed": 6, "last_progress_at": 100.0,
        "quick_verification": {"pass_rate": 0.75}}))
    (rsdir / "rollout_1.json").write_text(json.dumps({
        "failure_reason": "agent stage died"}))   # no verification_* -> indeterminate
    fs = rl.frontier_state(rd)
    assert fs.mode == "A"
    assert fs.gold_frontier == 6
    assert fs.valid_measurements == 1
    assert fs.indeterminate_total == 1


# --- relaunch_decision: the core gate ---------------------------------------- #
def test_relaunch_when_frontier_rises_minitorch_regression_guard():
    """An arm whose gold pass-count is climbing must keep relaunching regardless of
    wall-time (the minitorch case the activity gate wrongly force-cut)."""
    rl = _load_run_ladder()
    rd = Path(tempfile.mkdtemp())
    _write_wal(rd, [_wal_score(0, 120, 369, 0.32), _wal_score(1, 177, 369, 0.48)])
    action, why, frontier, fs = rl.relaunch_decision(rd, "omega", ["--arms",
        "autogen_orchestrator", "--autogen-max-agents", "1000"], attempt=3, last_prog=120)
    assert action == "relaunch"
    assert frontier == 177


def test_first_attempt_relaunches_when_journal_exists():
    rl = _load_run_ladder()
    rd = Path(tempfile.mkdtemp())
    _write_wal(rd, [_wal_score(0, 5, 10, 0.5)])
    action, why, frontier, fs = rl.relaunch_decision(rd, "o", ["--arms",
        "autogen_orchestrator", "--autogen-max-agents", "1000"], attempt=0, last_prog=None)
    assert action == "relaunch"   # last_prog None -> treated as a rise on first kill


def test_plateau_cut_when_frontier_flat_across_both_windows():
    """A flat frontier past BOTH the measurement floor and the wall floor cuts
    cut:no-progress. We force a large wall by spacing rollout last_progress_at epochs."""
    rl = _load_run_ladder()
    rd = Path(tempfile.mkdtemp())
    rsdir = rd / "rollout_status"
    rsdir.mkdir(parents=True)
    # one advance to 4, then many flat valid rollouts spanning > W_TIME with no rise.
    (rsdir / "rollout_00.json").write_text(json.dumps({
        "verification_returncode": 0, "verification_selected_test_count": 8,
        "verification_passed": 4, "last_progress_at": 0.0,
        "quick_verification": {"pass_rate": 0.5}}))
    for i in range(1, 14):   # 13 flat valid measurements (>= w_meas_eff for unbounded=12)
        (rsdir / f"rollout_{i:02d}.json").write_text(json.dumps({
            "verification_returncode": 0, "verification_selected_test_count": 8,
            "verification_passed": 4, "last_progress_at": float(i * 1000),  # 13000s > 7200
            "quick_verification": {"pass_rate": 0.5}}))
    action, why, frontier, fs = rl.relaunch_decision(
        rd, "o", ["--arms", "autogen_orchestrator", "--autogen-max-agents", "1000"],
        attempt=2, last_prog=4)
    assert action == "plateau"
    assert "cut:no-progress" in why
    assert frontier == 4


def test_harness_stall_when_indeterminate_wall():
    rl = _load_run_ladder()
    rd = Path(tempfile.mkdtemp())
    # all-indeterminate WAL, no valid measurement, past INDET_CEIL=24.
    recs = [_wal_score(i, 0, 0, 0.0, indeterminate=True, result_status="infra_nonresult")
            for i in range(30)]
    _write_wal(rd, recs)
    action, why, frontier, fs = rl.relaunch_decision(
        rd, "o", ["--arms", "autogen_orchestrator", "--autogen-max-agents", "1000"],
        attempt=2, last_prog=0)
    assert action == "harness"
    assert "cut:harness-stall" in why


def test_within_window_warm_resumes_until_backstop():
    rl = _load_run_ladder()
    rd = Path(tempfile.mkdtemp())
    # one valid measurement, flat but well within both windows -> relaunch (warm resume).
    _write_wal(rd, [_wal_score(0, 3, 10, 0.3)])
    action, why, frontier, fs = rl.relaunch_decision(
        rd, "o", ["--arms", "autogen_orchestrator", "--autogen-max-agents", "1000"],
        attempt=2, last_prog=3)
    assert action == "relaunch"
    # at the relaunch backstop with no rise and no cut -> exhausted (not a plateau cut).
    action2, why2, _, _ = rl.relaunch_decision(
        rd, "o", ["--arms", "autogen_orchestrator", "--autogen-max-agents", "1000"],
        attempt=rl.LADDER_MAX_RELAUNCH, last_prog=3)
    assert action2 == "exhausted"


def test_b0_one_shot_never_plateau_cuttable():
    """B0 budget=1 -> w_meas_eff floor 3, but a 1-shot never reaches 3 valid measurements,
    so the gate never returns plateau."""
    rl = _load_run_ladder()
    rd = Path(tempfile.mkdtemp())
    _write_wal(rd, [_wal_score(0, 2, 10, 0.2)])   # one valid measurement only
    action, why, frontier, fs = rl.relaunch_decision(
        rd, "B0", ["--arms", "B0_single_model"], attempt=2, last_prog=2)
    assert action != "plateau"


def test_relaunch_decision_deterministic():
    rl = _load_run_ladder()
    rd = Path(tempfile.mkdtemp())
    _write_wal(rd, [_wal_score(0, 5, 10, 0.5), _wal_score(1, 7, 10, 0.7)])
    a = rl.relaunch_decision(rd, "o", ["--arms", "autogen_orchestrator",
        "--autogen-max-agents", "1000"], attempt=1, last_prog=5)
    b = rl.relaunch_decision(rd, "o", ["--arms", "autogen_orchestrator",
        "--autogen-max-agents", "1000"], attempt=1, last_prog=5)
    assert (a[0], a[2]) == (b[0], b[2])
