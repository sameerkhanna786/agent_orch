"""SPFG+ Tier-2 (Mode-C) governor + context integration tests.

The frontier governor must cut a GENUINE no-solve-progress plateau (cut:no-progress) only
on the DUAL-AND of a valid-measurement window AND a journaled valid-measurement wall clock,
cut a sustained harness/scorer wall distinctly (cut:harness-stall), keep indeterminate
measurements NEUTRAL to the frontier arms, preserve every existing cut reason and ordering,
and replay the same halt sequence on resume (verdict cached by position).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from apex_omega.autogen.context import OrchestrationContext
from apex_omega.engine.governor import RunGovernor
from apex_omega.engine.runtime import Engine
from apex_omega.errors import PlateauStop
from apex_omega.executor.fake import FakeExecutor
from apex_omega.kernel.select import Candidate
from apex_omega.workflows.best_of_n import WorkerSpec


def _git_repo() -> str:
    d = Path(tempfile.mkdtemp()) / "repo"
    d.mkdir()
    (d / "mod.py").write_text("def f():\n    return 1\n")
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-qm", "base"]):
        subprocess.run(c, cwd=d, check=True, capture_output=True)
    return str(d)


def _ctx(run_dir=None):
    return OrchestrationContext(
        Engine(run_dir or tempfile.mkdtemp(), run_id="t"), executor=FakeExecutor(),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")], source_repo=_git_repo(),
        base_commit=None, score_fn=lambda wt: None, prompt_builder=lambda c, i, s: "x",
    )


def _cand(pr, sha="", *, gold=0, indeterminate=False, gold_total=10):
    c = Candidate(candidate_id="x", public_signal_score=pr, content_sha=sha)
    c.meta = {"gold_passed": gold, "gold_total": gold_total, "indeterminate": indeterminate,
              "empty_diff": False, "finalization_status": "completed"}
    return c


# --- unit: RunGovernor.verdict SPFG+ arms ----------------------------------- #
def _gov():
    class _Budget:
        total = None
        def can_start(self, *, reserve=1):
            return True
    class _E:
        budget = _Budget()
        def agents_used(self):
            return 0
    g = RunGovernor(engine=_E())
    return g


def _state(**kw):
    base = {"attempts_since_improvement": 0, "agents_used": 0, "nonresult_streak": 0,
            "sterile_streak": 0, "tokens_since_improvement": 0,
            "valid_measurements": 1, "valid_measurements_since_improvement": 0,
            "seconds_since_frontier_improved": 0.0, "indeterminate_streak": 0}
    base.update(kw)
    return base


def test_dual_and_requires_both_windows():
    g = _gov()
    g.plateau_patience_meas = 5
    g.plateau_wall_seconds = 1000.0
    g.plateau_patience = 9999          # disable the legacy backstop for this isolation
    # meas crossed, wall NOT -> continue
    cont, _ = g.verdict(_state(valid_measurements_since_improvement=6,
                               seconds_since_frontier_improved=10.0))
    assert cont is True
    # wall crossed, meas NOT -> continue
    cont, _ = g.verdict(_state(valid_measurements_since_improvement=2,
                               seconds_since_frontier_improved=5000.0))
    assert cont is True
    # BOTH crossed -> cut:no-progress
    cont, reason = g.verdict(_state(valid_measurements_since_improvement=6,
                                    seconds_since_frontier_improved=5000.0))
    assert cont is False and reason == "cut:no-progress"


def test_harness_stall_distinct_and_before_soft_plateau():
    g = _gov()
    g.harness_stall_cut = 4
    g.plateau_patience_meas = 1
    g.plateau_wall_seconds = 0.0
    g.plateau_patience = 9999
    # a wall of indeterminate (no valid measurement at all) -> harness-stall, NOT no-progress
    cont, reason = g.verdict(_state(valid_measurements=0, indeterminate_streak=4,
                                    valid_measurements_since_improvement=0,
                                    seconds_since_frontier_improved=0.0))
    assert cont is False and reason == "cut:harness-stall"


def test_existing_hard_cuts_fire_first_and_unchanged():
    g = _gov()
    # nonresult streak still wins over everything
    cont, reason = g.verdict(_state(nonresult_streak=8, indeterminate_streak=99,
                                    valid_measurements_since_improvement=99,
                                    seconds_since_frontier_improved=1e9))
    assert (cont, reason) == (False, "cut:nonresult-streak")
    cont, reason = g.verdict(_state(sterile_streak=8))
    assert (cont, reason) == (False, "cut:sterile-diff-streak")


def test_legacy_attempts_backstop_still_cuts():
    g = _gov()
    g.plateau_patience = 3
    cont, reason = g.verdict(_state(attempts_since_improvement=3,
                                    valid_measurements_since_improvement=0,
                                    seconds_since_frontier_improved=0.0))
    assert cont is False and reason == "cut:no-progress"


def test_to_dict_carries_new_knobs():
    g = _gov()
    d = g.to_dict()
    for k in ("plateau_wall_seconds", "plateau_patience_meas", "harness_stall_cut"):
        assert k in d


# --- integration: context wave loop drives the frontier arms ---------------- #
def test_indeterminate_is_neutral_to_frontier_then_harness_stall():
    ctx = _ctx()
    ctx.governor.harness_stall_cut = 4
    ctx.governor.plateau_patience = 9999          # ensure the attempt arm does not pre-empt
    for i in range(4):
        if ctx._halted:
            break
        ctx.parallel([lambda i=i: _cand(0.0, sha=f"s{i}", indeterminate=True)])
    assert ctx._halted is True and ctx._halt_reason == "cut:harness-stall"
    # indeterminate measurements never advanced the frontier valid arms
    assert ctx._valid_measurements == 0
    assert ctx._wave_state()["valid_measurements"] == 0


def test_frontier_dual_and_cut_in_context():
    ctx = _ctx()
    ctx.governor.plateau_patience_meas = 3
    ctx.governor.plateau_wall_seconds = 1.0        # tiny wall so the increment crosses it
    ctx.governor.plateau_patience = 9999           # isolate the dual-AND (not the legacy arm)
    # one valid rise then flat valid measurements until BOTH windows cross
    ctx.parallel([lambda: _cand(0.5, sha="rise", gold=5)])    # frontier rises -> resets
    for i in range(4):
        if ctx._halted:
            break
        ctx.parallel([lambda i=i: _cand(0.4, sha=f"flat{i}", gold=4)])  # dips/flat, no new gold
    assert ctx._halted is True and ctx._halt_reason == "cut:no-progress"


def test_frontier_rise_resets_both_arms_no_cut():
    ctx = _ctx()
    ctx.governor.plateau_patience_meas = 3
    ctx.governor.plateau_wall_seconds = 1.0
    ctx.governor.plateau_patience = 9999
    # a strictly-rising frontier every wave (the minitorch still-progressing case) -> never cut
    for i in range(8):
        ctx.parallel([lambda i=i: _cand(0.1 * (i + 1), sha=f"r{i}", gold=i + 1)])
        assert ctx._halted is False, f"false-cut while frontier rising at wave {i}"
    assert ctx._best_gold_passed == 8


def test_wave_state_has_frontier_keys():
    ctx = _ctx()
    ctx.parallel([lambda: _cand(0.3, sha="a", gold=2)])
    s = ctx._wave_state()
    for k in ("valid_measurements", "valid_measurements_since_improvement",
              "seconds_since_frontier_improved", "indeterminate_streak"):
        assert k in s
    assert s["valid_measurements"] == 1


def test_frontier_cut_replays_deterministically_on_resume():
    rd = tempfile.mkdtemp()
    repo = _git_repo()

    def _mk():
        return OrchestrationContext(
            Engine(rd, run_id="t"), executor=FakeExecutor(),
            worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")], source_repo=repo,
            base_commit=None, score_fn=lambda wt: None, prompt_builder=lambda c, i, s: "x",
        )

    ctx1 = _mk()
    ctx1.governor.plateau_patience_meas = 3
    ctx1.governor.plateau_wall_seconds = 1.0
    ctx1.governor.plateau_patience = 9999
    ctx1.parallel([lambda: _cand(0.5, sha="rise", gold=5)])
    for i in range(4):
        if ctx1._halted:
            break
        ctx1.parallel([lambda i=i: _cand(0.4, sha=f"flat{i}", gold=4)])
    assert ctx1._halted is True and ctx1._halt_reason == "cut:no-progress"

    # resume: live knobs would NEVER halt; the journaled verdict by position must win.
    ctx2 = _mk()
    ctx2.governor.plateau_patience_meas = 9999
    ctx2.governor.plateau_wall_seconds = 1e12
    ctx2.governor.plateau_patience = 9999
    ctx2.parallel([lambda: _cand(0.5, sha="rise", gold=5)])
    for i in range(4):
        if ctx2._halted:
            break
        ctx2.parallel([lambda i=i: _cand(0.4, sha=f"flat{i}", gold=4)])
    assert ctx2._halted is True and ctx2._halt_reason == "cut:no-progress"
