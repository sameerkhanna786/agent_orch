"""SPFG+ "Solve-Progress Frontier Governor" — the ONE shared definition.

This module is the single-sourced progress metric consumed identically by all three
SPFG+ enforcement tiers:

  1. ``scripts/run_ladder.py``       (Tier-1, wraps BOTH Mode-C and Mode-A subprocesses)
  2. ``apex_omega/engine/governor.py`` + ``apex_omega/autogen/context.py`` (Tier-2, Mode-C in-cell)
  3. Mode-A via the Tier-1 ladder gate (reads rollout_status / benchmark_state)

PRINCIPLE: fail/cut an arm (baseline OR orchestrator) ONLY on a GENUINE no-solve-progress
plateau, but NEVER while it is still making measurable progress.

PROGRESS METRIC = the FRONTIER F: monotonic best-so-far COUNT (integer) of gold
expected-test-ids passing in a VALID measurement (NOT pass_rate — collected counts drift).
A strict raw pass_rate increase within the same gold count is a SECONDARY tie-break that
also counts as progress (matches the existing context.py round_pass>_best_pass_rate+1e-9).
BEST-not-LAST: F = max over all valid measurements; a dip below F is a dry sample.

INDETERMINATE GUARD: a measurement contributes to F and to the patience clocks ONLY if it
is a real test outcome (excludes infra_nonresult / indeterminate harness-fail / parser-error
/ environment-failure / scoring-timeout / editable-shadow / Mode-A total==0&&rc not in (0,1)).
A wall of indeterminate => a DISTINCT outcome ``cut:harness-stall`` (after INDET_CEIL),
NOT ``cut:no-progress``.

PATIENCE = DUAL AND window: cut:no-progress fires only when BOTH
(seconds_since_frontier_improved >= W_TIME) AND
(valid_measurements_since_frontier_improved >= W_meas_effective). Any frontier rise in
EITHER window resets BOTH.

DETERMINISM (resume-safe): every decision function is a PURE function of its inputs. No
live wall-clock is read inside any decision — time is INJECTED as ``wall_delta`` and
accumulated into a JOURNALED scalar (``wall_accum``), so a journal replay is deterministic.

This file is IMPORT-ONLY for the tier builders — never re-edit it from a tier.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from math import ceil
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Frozen, env-overridable default params (NO repo identity in any decision path).
# ---------------------------------------------------------------------------

# env var names exported as module constants for tier callers / tests.
ENV_W_TIME = ("APEX_FRONTIER_PLATEAU_WALL_S", "LADDER_PLATEAU_WALL_S")
ENV_W_MEAS = ("APEX_FRONTIER_PLATEAU_MEAS", "LADDER_PLATEAU_MEAS")
ENV_INDET_CEIL = ("APEX_FRONTIER_INDET_CEIL", "LADDER_INDET_CEIL")
ENV_POLL_S = ("LADDER_POLL_S",)


def _env_first(names, default):
    for n in names:
        v = os.environ.get(n)
        if v is not None and str(v).strip():
            return v
    return default


def _env_int(names, default):
    raw = _env_first(names, None)
    if raw is None:
        return default
    try:
        return int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return default


def _env_float(names, default):
    raw = _env_first(names, None)
    if raw is None:
        return default
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class FrontierParams:
    """The frozen SPFG+ knobs. ``from_env`` applies the documented env fallbacks."""

    w_time: float = 7200.0  # seconds of VALID-measurement wall-time
    w_meas: int = 12        # VALID measurements
    indet_ceil: int = 24    # indeterminate ceiling -> cut:harness-stall
    poll_s: int = 300       # ladder daemon poll cadence (Tier-1 only)

    @classmethod
    def from_env(cls) -> "FrontierParams":
        return cls(
            w_time=float(_env_int(ENV_W_TIME, 7200)),
            w_meas=_env_int(ENV_W_MEAS, 12),
            indet_ceil=_env_int(ENV_INDET_CEIL, 24),
            poll_s=_env_int(ENV_POLL_S, 300),
        )


DEFAULT_PARAMS = FrontierParams()


def frontier_defaults() -> Tuple[int, int, int, int]:
    """(W_TIME, W_meas, INDET_CEIL, POLL_S) from env, for tier modules that want scalars."""
    p = FrontierParams.from_env()
    return (int(p.w_time), int(p.w_meas), int(p.indet_ceil), int(p.poll_s))


# ---------------------------------------------------------------------------
# Outcome taxonomy.
# ---------------------------------------------------------------------------

class FrontierOutcome(str, Enum):
    CONTINUE = "continue"
    PLATEAU_CUT = "plateau-cut"            # genuine no-progress give-up  -> cut:no-progress
    INDETERMINATE_CUT = "indeterminate-cut"  # harness/scorer wall        -> cut:harness-stall


# tier callers map the verdict token to a cut reason string.
OUTCOME_TO_CUT_REASON = {
    FrontierOutcome.PLATEAU_CUT.value: "cut:no-progress",
    FrontierOutcome.INDETERMINATE_CUT.value: "cut:harness-stall",
}


# ---------------------------------------------------------------------------
# Fairness — mode-scaled measurement window.
# ---------------------------------------------------------------------------

def w_meas_effective(global_w_meas: int, arm_attempt_budget: Optional[int]) -> int:
    """The cross-granularity fairness fix for the measurement floor.

    Unbounded arms (orchestrators / ralph; budget None) keep the global window. A
    finite-budget arm scales DOWN to ``min(global, max(3, ceil(0.6*budget)))`` so a
    genuinely plateaued best-of-8 (budget 8 -> 5) IS cuttable. A 1-shot (budget 1 -> 3)
    returns the floor 3 BUT can never accumulate >=3 valid measurements, so it is
    correctly NEVER plateau-cuttable. The wall (w_time) floor is the equalizer.
    """
    if arm_attempt_budget is None:
        return int(global_w_meas)
    return min(int(global_w_meas), max(3, ceil(0.6 * int(arm_attempt_budget))))


# ---------------------------------------------------------------------------
# FrontierTracker — the one definition; live (Mode-C) and disk-reconstructed.
# ---------------------------------------------------------------------------

class FrontierTracker:
    """Best-so-far gold-pass-COUNT frontier with a dual-AND patience window.

    The wall clock is INJECTED (``wall_delta``) and accumulated into a journaled scalar;
    no live clock is read here, so a replay that re-ingests the same records is identical.
    """

    def __init__(self, w_time: float, w_meas: int, indet_ceil: int,
                 *, w_meas_eff: Optional[int] = None) -> None:
        self.w_time = float(w_time)
        self.w_meas = int(w_meas)
        self.w_meas_eff = int(w_meas_eff) if w_meas_eff is not None else int(w_meas)
        self.indet_ceil = int(indet_ceil)

        self.best = -1               # best gold pass COUNT (-1 == none yet)
        self.best_rate = -1.0        # secondary tie-break (raw pass_rate)
        self.valid = 0               # count of VALID measurements ingested
        self.valid_at_best = 0       # self.valid when the frontier last rose
        self.wall_accum = 0.0        # accumulated VALID-measurement wall seconds (journaled)
        self.wall_at_best = None     # None => clock UNSTARTED; 0.0 on first valid; reset on rise
        self.indet_streak = 0
        self.indet_total = 0
        self.best_min_errors: Optional[int] = None  # Fix 1: secondary collection-error frontier
        self.history: List[Tuple[int, int]] = []  # (valid_idx, pass_count) at each strict rise

    def ingest(self, pass_count: int, pass_rate: float, valid: bool, wall_delta: float,
               errors: int = -1) -> None:
        if not valid:
            # INDETERMINATE: neutral to F and to BOTH patience clocks.
            self.indet_streak += 1
            self.indet_total += 1
            return
        self.indet_streak = 0
        self.valid += 1
        if self.wall_at_best is None:
            self.wall_at_best = 0.0  # first VALID ingest STARTS the clock
        self.wall_accum += max(0.0, float(wall_delta))
        pass_count = int(pass_count)
        pass_rate = float(pass_rate)
        improved = (pass_count > self.best) or (pass_rate > self.best_rate + 1e-9)
        # Fix 1 (governor audit): a strict drop in the collection-error count is genuine
        # implementation progress on a not-yet-passing large repo — it resets BOTH patience arms
        # WITHOUT advancing the gold frontier (self.best is unchanged below when only this fires).
        if errors is not None and int(errors) >= 0:
            e = int(errors)
            if self.best_min_errors is None:
                self.best_min_errors = e        # establish baseline (not itself an improvement)
            elif e < self.best_min_errors:
                self.best_min_errors = e
                improved = True
        if pass_count > self.best:
            self.best = pass_count
            self.history.append((self.valid, pass_count))
        if pass_rate > self.best_rate:
            self.best_rate = pass_rate
        if improved:
            self.valid_at_best = self.valid
            self.wall_at_best = self.wall_accum  # RESET BOTH arms

    def state(self) -> dict:
        secs = (self.wall_accum - self.wall_at_best) if self.wall_at_best is not None else 0.0
        return {
            "best_gold_passed": max(self.best, 0),
            "valid_measurements": self.valid,
            "valid_measurements_since_improvement": self.valid - self.valid_at_best,
            "seconds_since_frontier_improved": secs,
            "indeterminate_streak": self.indet_streak,
            "indeterminate_total": self.indet_total,
            "frontier_history": list(self.history),
        }


def plateau_verdict(s: dict, w_meas_eff: int, w_time: float, indet_ceil: int) -> Tuple[str, str]:
    """PURE verdict over a ``FrontierTracker.state()``-shaped dict.

    Returns ``(FrontierOutcome value, human reason)``.
    """
    if s["valid_measurements"] == 0:
        # clock UNSTARTED (prep / clone / build / indeterminate-only prefix).
        if s["indeterminate_total"] >= indet_ceil:
            return (FrontierOutcome.INDETERMINATE_CUT.value,
                    "all-harness-fail, no valid measurement")
        return (FrontierOutcome.CONTINUE.value, "pre-frontier")
    if s["indeterminate_streak"] >= indet_ceil:
        return (FrontierOutcome.INDETERMINATE_CUT.value, "sustained harness failure")
    if (s["valid_measurements_since_improvement"] >= w_meas_eff
            and s["seconds_since_frontier_improved"] >= w_time):
        return (FrontierOutcome.PLATEAU_CUT.value,
                f'frontier flat at {s["best_gold_passed"]}')
    return (FrontierOutcome.CONTINUE.value,
            f'frontier={s["best_gold_passed"]} climbing-or-within-window')


# ---------------------------------------------------------------------------
# On-disk reconstruction (Tier-1 ladder, separate process — epoch wall is fine here).
# ---------------------------------------------------------------------------

@dataclass
class FrontierState:
    """A ``plateau_verdict``-ready reconstruction from on-disk artifacts.

    ``as_state()`` returns the SAME keys ``plateau_verdict`` reads. The ladder is a
    SEPARATE process (not a journal replay), so file/epoch wall-clock is acceptable for
    ``seconds_since_frontier_improved``.
    """

    gold_frontier: int = 0
    valid_measurements: int = 0
    indeterminate_total: int = 0
    indeterminate_streak: int = 0
    latest_valid_epoch: Optional[float] = None
    last_advance_epoch: Optional[float] = None
    mode: str = "C"
    history: List[Tuple[int, int]] = field(default_factory=list)
    # Fix 1: explicit valid-measurement index of the last improvement (gold OR collection-error
    # frontier). When set it overrides the history-derived index so a non-gold collection-error
    # rise also resets the valid-measurement arm. None => fall back to history (Mode-A unchanged).
    valid_at_best_idx: Optional[int] = None

    def as_state(self) -> dict:
        if self.latest_valid_epoch is not None and self.last_advance_epoch is not None:
            secs = max(0.0, self.latest_valid_epoch - self.last_advance_epoch)
        else:
            secs = 0.0
        # valid_measurements_since_improvement is reconstructed via the index of the last
        # improvement (explicit valid_at_best_idx if set, else the last gold-rise in history).
        if self.valid_at_best_idx is not None:
            valid_at_best = self.valid_at_best_idx
        elif self.history:
            valid_at_best = self.history[-1][0]
        else:
            valid_at_best = 0
        return {
            "best_gold_passed": max(self.gold_frontier, 0),
            "valid_measurements": self.valid_measurements,
            "valid_measurements_since_improvement": self.valid_measurements - valid_at_best,
            "seconds_since_frontier_improved": secs,
            "indeterminate_streak": self.indeterminate_streak,
            "indeterminate_total": self.indeterminate_total,
            "frontier_history": list(self.history),
        }


def _iter_jsonl(path: Path):
    try:
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except (ValueError, TypeError):
                    continue
    except (OSError, IOError):
        return


def _wal_value(rec: dict) -> Optional[dict]:
    """Unwrap the resume_or_run_json {"value": {...}} envelope on a score record."""
    sr = rec.get("structured_result")
    if not isinstance(sr, dict):
        return None
    val = sr.get("value")
    if isinstance(val, dict):
        return val
    # tolerate an un-enveloped payload (defensive); else treat as empty/in-flight.
    if "passed" in sr or "indeterminate" in sr:
        return sr
    return None


def frontier_from_wal(rundir) -> FrontierState:
    """Mode-C reconstruction: scan calls_wal.jsonl for committed ``score`` records.

    Accepts a run-dir Path/str (recursively finds calls_wal.jsonl) OR a direct wal path.
    VALID iff status=="committed" and kind=="score" and result_status=="ok" and not
    value["indeterminate"]. ``passed`` is the gold COUNT; ``pass_rate`` is the secondary
    tie-break. in_flight records have an empty structured_result and are skipped.

    WAL ts_logical==seq is NOT wall-clock, so this reconstruction carries no wall epoch;
    the ladder poll thread anchors the wall arm via observation timestamps instead.
    """
    p = Path(rundir)
    if p.is_dir():
        wals = sorted(p.rglob("calls_wal.jsonl"))
    elif p.exists():
        wals = [p]
    else:
        wals = []

    fs = FrontierState(mode="C")
    best = -1
    best_rate = -1.0
    best_min_errors = None   # Fix 1: secondary collection-error frontier (Mode-C relaunch gate)
    valid_idx = 0
    for wal in wals:
        for rec in _iter_jsonl(wal):
            if rec.get("kind") != "score":
                continue
            if rec.get("status") != "committed":
                continue  # skip in_flight (empty structured_result) / failed
            rs = rec.get("result_status")
            val = _wal_value(rec)
            indet = (rs == "infra_nonresult") or (
                isinstance(val, dict) and bool(val.get("indeterminate")))
            if rs != "ok" or val is None or indet:
                fs.indeterminate_total += 1
                fs.indeterminate_streak += 1
                continue
            # VALID
            fs.indeterminate_streak = 0
            fs.valid_measurements += 1
            valid_idx += 1
            pc = int(val.get("passed", 0) or 0)
            pr = float(val.get("pass_rate", 0.0) or 0.0)
            _e = val.get("errors")
            err = int(_e) if _e is not None else -1
            improved = (pc > best) or (pr > best_rate + 1e-9)
            # Fix 1: a strict drop in collection errors is implementation progress -> resets BOTH
            # patience arms (wall via last_advance_epoch, valid via valid_at_best_idx) without
            # advancing the gold frontier (best/history unchanged when only this fires).
            if err >= 0:
                if best_min_errors is None:
                    best_min_errors = err       # establish baseline (not itself an improvement)
                elif err < best_min_errors:
                    best_min_errors = err
                    improved = True
            if pc > best:
                best = pc
                fs.history.append((valid_idx, pc))
            if pr > best_rate:
                best_rate = pr
            if improved:
                fs.last_advance_epoch = fs.latest_valid_epoch
                fs.valid_at_best_idx = valid_idx
            fs.latest_valid_epoch = fs.latest_valid_epoch  # epoch anchored by ladder poll
    fs.gold_frontier = max(best, 0)
    return fs


_MODE_A_INDET_CLASSES = ("harness_failure", "environment_failure", "parser_error")


def _rollout_valid(rec: dict) -> Tuple[bool, Optional[int], Optional[float]]:
    """Mode-A validity (mirrors best_of_n indeterminate=(total==0 and rc not in (0,1))).

    Returns (valid, gold_pass_count_or_None, pass_rate_or_None).
    A rollout that died pre-verification LACKS verification_* keys => indeterminate.
    """
    if "verification_returncode" not in rec:
        return (False, None, None)
    rc = rec.get("verification_returncode")
    sel = rec.get("verification_selected_test_count")
    timed_out = bool(rec.get("verification_timed_out"))
    fc = str(rec.get("failure_class", "") or "").lower()
    if rc not in (0, 1):
        return (False, None, None)
    if sel in (0, None):
        return (False, None, None)
    if timed_out:
        return (False, None, None)
    if any(c in fc for c in _MODE_A_INDET_CLASSES):
        return (False, None, None)
    passed = rec.get("verification_passed")
    qv = rec.get("quick_verification") or {}
    pr = qv.get("pass_rate") if isinstance(qv, dict) else None
    return (True, (int(passed) if passed is not None else None),
            (float(pr) if pr is not None else None))


def frontier_from_rollouts(rundir, *, arm_attempt_budget: Optional[int] = None) -> FrontierState:
    """Mode-A reconstruction from rollout_status/rollout_*.json (and candidate_scorecard).

    Preferred per-candidate granularity: rollout_evals/**/candidate_scorecard.json. Falls
    back to rollout_status/rollout_*.json. VALID iff verification_returncode in (0,1) and
    verification_selected_test_count not in (0,None) and not timed_out and failure_class
    not a harness/env/parser failure. gold_frontier = max verification_passed over valid.
    """
    p = Path(rundir)
    fs = FrontierState(mode="A")
    best = -1
    best_rate = -1.0
    valid_idx = 0

    records: List[Tuple[float, dict]] = []
    # candidate scorecards first (finer granularity).
    for sc in sorted(p.rglob("candidate_scorecard.json")):
        try:
            data = json.loads(Path(sc).read_text())
        except (OSError, ValueError):
            continue
        for c in (data.get("candidates") or []):
            ev = c.get("evaluation") or {}
            rec = {
                "verification_returncode": ev.get("returncode", c.get("evaluation", {}).get("returncode")),
                "verification_selected_test_count": ev.get("total_tests", c.get("total_tests")),
                "verification_passed": c.get("passed"),
                "verification_timed_out": ev.get("timed_out", False),
                "failure_class": c.get("evaluation_status", ""),
                "quick_verification": {"pass_rate": c.get("pass_rate")},
            }
            ts = c.get("last_progress_at") or c.get("updated_at")
            records.append((float(ts) if ts else float(valid_idx), rec))

    if not records:
        rollout_files = sorted(p.rglob("rollout_status/rollout_*.json"))
        if not rollout_files:
            rollout_files = sorted(p.rglob("rollout_*.json"))
        for rf in rollout_files:
            try:
                rec = json.loads(Path(rf).read_text())
            except (OSError, ValueError):
                continue
            ts = rec.get("last_progress_at") or rec.get("updated_at") or 0.0
            records.append((float(ts) if ts else 0.0, rec))

    records.sort(key=lambda t: t[0])
    for ts, rec in records:
        valid, pc, pr = _rollout_valid(rec)
        if not valid:
            fs.indeterminate_total += 1
            fs.indeterminate_streak += 1
            continue
        fs.indeterminate_streak = 0
        fs.valid_measurements += 1
        valid_idx += 1
        # No verification_passed count: do not advance the integer frontier
        # (a missing count is non-advancing). pass_rate, if present, still
        # drives the secondary tiebreak below via `rate`.
        pcv = int(pc) if pc is not None else -1
        rate = float(pr) if pr is not None else 0.0
        improved = (pcv > best) or (rate > best_rate + 1e-9)
        if pcv > best:
            best = pcv
            fs.history.append((valid_idx, pcv))
        if rate > best_rate:
            best_rate = rate
        if improved:
            fs.last_advance_epoch = ts
        fs.latest_valid_epoch = ts
    fs.gold_frontier = max(best, 0)
    return fs
