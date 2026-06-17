"""Backbone Phase 2 — RunGovernor + always-on plateau (terminates a no-clock run
without discarding work) and the structural soft-write seam (patterns can never
promote acceptance)."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from apex_omega.autogen.context import OrchestrationContext
from apex_omega.engine.runtime import Engine
from apex_omega.errors import PlateauStop
from apex_omega.executor.fake import FakeExecutor
from apex_omega.kernel.select import Candidate
from apex_omega.types import ExecResult, TokenUsage
from apex_omega.workflows.best_of_n import WorkerSpec


def _git_repo() -> str:
    d = Path(tempfile.mkdtemp()) / "repo"
    d.mkdir()
    (d / "mod.py").write_text("def f():\n    return 1\n")
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-qm", "base"]):
        subprocess.run(c, cwd=d, check=True, capture_output=True)
    return str(d)


def _ctx():
    return OrchestrationContext(
        Engine(tempfile.mkdtemp(), run_id="t"), executor=FakeExecutor(),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")], source_repo=_git_repo(),
        base_commit=None, score_fn=lambda wt: None, prompt_builder=lambda c, i, s: "x",
    )


def _cand(pr, sha=""):
    return Candidate(candidate_id="x", public_signal_score=pr, content_sha=sha)


def _ask_resp(payload):
    """A FakeExecutor responder that returns ``payload`` as structured_output for any
    schema'd (read-only ask) call."""
    def r(task, session):
        return ExecResult(final_message="ok",
                          structured_output=(payload if task.schema else None),
                          ok=True, finalization_status="completed",
                          fs_diff="d\n", usage=TokenUsage(input=1, output=1))
    return r


def _ctx_with(responder):
    return OrchestrationContext(
        Engine(tempfile.mkdtemp(), run_id="t"), executor=FakeExecutor(responder),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")], source_repo=_git_repo(),
        base_commit=None, score_fn=lambda wt: None, prompt_builder=lambda c, i, s: "x",
    )


# --- 2.1/2.4 cut-losses terminates a no-clock fan-out -------------------------- #
def test_governor_plateau_raises_after_dry_rounds():
    # The soft plateau cuts on ATTEMPTS-since-improvement (schedule/arm-invariant — a wave-based
    # rule would never fire under omega's doubling schedule). Pin a small attempt-patience and
    # simulate agent growth; distinct shas so this isolates the no-progress cut, not the sterile one.
    ctx = _ctx()
    ctx.governor.plateau_patience = 3
    agents = {"n": 0}
    ctx._engine.agents_used = lambda: agents["n"]
    for i in range(3):
        agents["n"] += 1
        ctx.parallel([lambda i=i: _cand(0.0, sha=f"s{i}")])   # 3 dry attempts -> halt set
    assert ctx._halted is True and ctx._halt_reason == "cut:no-progress"
    with pytest.raises(PlateauStop):                          # CutLosses subclass -> caught
        agents["n"] += 1
        ctx.parallel([lambda: _cand(0.0, sha="sX")])          # a `while True` here terminates


def test_sterile_diff_streak_is_a_hard_cut():
    # repeating the SAME (or empty) diff with no improvement is an objectively-stuck state: the
    # sterile-diff hard cut fires once enough sterile ATTEMPTS accumulate (no agent growth needed,
    # so the soft plateau — which needs attempts — never interferes here). Pin the cut small.
    ctx = _ctx()
    ctx.governor.sterile_streak_cut = 3
    for _ in range(8):
        if ctx._halted:
            break
        ctx.parallel([lambda: _cand(0.0, sha="identical")])   # same diff every wave (width 1)
    assert ctx._halted is True and ctx._halt_reason == "cut:sterile-diff-streak"


def test_hard_cuts_are_attempt_based_not_wave_based():
    # review M1: a width-1 lineage and a width-N wave with the SAME per-attempt sterility must be
    # cut after a comparable number of ATTEMPTS, not waves. With sterile_streak_cut=6 (attempts):
    # a single width-6 all-sterile wave trips it, AND ~6 width-1 sterile waves trip it.
    wide = _ctx(); wide.governor.sterile_streak_cut = 6
    wide._seen_shas.add("dup")                                 # pre-seed so the whole wave is sterile
    wide.parallel([(lambda: _cand(0.0, sha="dup")) for _ in range(6)])   # one width-6 sterile wave
    assert wide._halted is True and wide._halt_reason == "cut:sterile-diff-streak"

    narrow = _ctx(); narrow.governor.sterile_streak_cut = 6
    narrow._seen_shas.add("dup")
    for _ in range(6):
        if narrow._halted:
            break
        narrow.parallel([lambda: _cand(0.0, sha="dup")])      # width-1, 6 sterile attempts
    assert narrow._halted is True and narrow._halt_reason == "cut:sterile-diff-streak"


def test_governor_does_not_halt_while_improving():
    ctx = _ctx()
    for p in (0.1, 0.2, 0.3, 0.4, 0.5):
        ctx.parallel([lambda p=p: _cand(p)])      # strictly improving each round
    assert ctx._halted is False and abs(ctx.best_pass_rate - 0.5) < 1e-9


def test_detector_does_not_cut_slow_winner_under_growing_waves():
    # THE critical false-cut regression guard (the jinja base-s0 counterexample: pass_rate=0 for
    # ~6 doubling waves = ~63 agents, then SOLVED at wave 6). The default attempt-patience (64)
    # must exceed those ~63 dry attempts so the slow-but-real winner is NEVER cut before it solves.
    # Distinct diffs each wave (so the sterile cut is not the subject).
    ctx = _ctx()
    agents = {"n": 0}
    ctx._engine.agents_used = lambda: agents["n"]          # simulate budget growth per wave
    for w, sz in enumerate([1, 2, 4, 8, 16, 32]):          # 6 dry waves -> 63 dry agents
        agents["n"] += sz
        ctx.parallel([lambda w=w: _cand(0.0, sha=f"w{w}")])
        assert ctx._halted is False, f"false-cut at dry wave {w} (since_improve={agents['n']})"
    agents["n"] += 1
    ctx.parallel([lambda: _cand(1.0, sha="solve")])        # the next wave finally solves
    assert ctx._halted is False and abs(ctx.best_pass_rate - 1.0) < 1e-9


def test_detector_does_not_cut_slow_winner_on_a_sequential_width_1_arm():
    # review M2: the slow-winner guard must hold for a SEQUENTIAL (1-agent-per-wave) arm too — the
    # case RALPH actually uses — not only the growing-wave omega schedule. 6 flat width-1 waves
    # then a solve on the 7th must NOT be cut. agents_used grows by exactly +1/wave (sequential).
    ctx = _ctx()
    agents = {"n": 0}
    ctx._engine.agents_used = lambda: agents["n"]
    for w in range(6):                                   # 6 flat dry waves, one attempt each
        agents["n"] += 1
        ctx.parallel([lambda w=w: _cand(0.0, sha=f"w{w}")])
        assert ctx._halted is False, f"false-cut at sequential dry wave {w} (dry={ctx._dry_rounds})"
    agents["n"] += 1
    ctx.parallel([lambda: _cand(1.0, sha="solve")])     # 7th attempt finally solves
    assert ctx._halted is False and abs(ctx.best_pass_rate - 1.0) < 1e-9


def test_detector_cuts_a_genuinely_stuck_run_within_bounded_waves():
    # a doomed repo (pass_rate 0 forever, distinct diffs so not the sterile cut) IS cut by the
    # attempt-based no-progress plateau once attempts-since-improvement reaches the patience floor
    # — not run to the ceiling. Pin patience small for a fast, deterministic check.
    ctx = _ctx()
    ctx.governor.plateau_patience = 24
    agents = {"n": 0}
    ctx._engine.agents_used = lambda: agents["n"]
    cut_wave = None
    for w in range(60):
        agents["n"] += 4
        if ctx._halted:
            cut_wave = w
            break
        ctx.parallel([lambda w=w: _cand(0.0, sha=f"d{w}")])
    assert ctx._halted is True
    assert ctx._halt_reason == "cut:no-progress"
    # 4 agents/wave, patience 24 -> cut once since-improvement >= 24, i.e. by ~wave 6.
    assert (cut_wave or 60) <= 8


def test_wave_decisions_replay_deterministically_on_resume():
    rd = tempfile.mkdtemp()
    repo = _git_repo()

    def _mk():
        return OrchestrationContext(
            Engine(rd, run_id="t"), executor=FakeExecutor(),
            worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")], source_repo=repo,
            base_commit=None, score_fn=lambda wt: None, prompt_builder=lambda c, i, s: "x",
        )

    # run 1 (pin sterile cut=2): identical diffs -> sterile hard cut. wave0 sees the sha fresh
    # (sterile 0), wave1 repeats it (sterile 1), wave2 (sterile 2) -> halt (recorded {continue:False}).
    ctx1 = _mk()
    ctx1.governor.sterile_streak_cut = 2
    ctx1.parallel([lambda: _cand(0.0, sha="dup")])   # wave0 -> fresh sha, continue
    ctx1.parallel([lambda: _cand(0.0, sha="dup")])   # wave1 -> sterile 1, continue
    assert ctx1._halted is False
    ctx1.parallel([lambda: _cand(0.0, sha="dup")])   # wave2 -> sterile 2 -> halt
    assert ctx1._halted is True

    # run 2 attaches to the SAME run-dir/journal but with a governor that LIVE would never
    # halt (cut knobs huge). The journaled verdicts must win -> identical halt sequence.
    ctx2 = _mk()
    ctx2.governor.sterile_streak_cut = 999            # live would say "continue" forever
    ctx2.governor.plateau_patience = 999
    ctx2.parallel([lambda: _cand(0.0, sha="dup")])   # replays wave0=continue (HIT)
    ctx2.parallel([lambda: _cand(0.0, sha="dup")])   # replays wave1=continue (HIT)
    assert ctx2._halted is False
    ctx2.parallel([lambda: _cand(0.0, sha="dup")])   # replays wave2=halt (HIT) despite knobs
    assert ctx2._halted is True


def test_governor_can_start_default_unbounded():
    ctx = _ctx()
    # default: no token/agent budget set -> can_start gated only by the per-run ceiling
    assert ctx.governor.token_budget is None
    assert ctx.governor.can_start() is True


# --- 2.2 ctx.ask: a read-only SIGNAL, never a Candidate ----------------------- #
def test_ask_returns_signal_not_candidate():
    ctx = _ctx()
    out = ctx.ask("what files?", schema={"type": "object"}, agent_id=1)
    assert isinstance(out, dict)              # a signal (structured_output), not a Candidate
    txt = ctx.ask("summarize", agent_id=2)
    assert isinstance(txt, str)              # no schema -> final text
    assert ctx.all_candidates() == []        # ask NEVER banks a candidate


def test_ask_forces_read_only_sandbox():
    seen = {}

    def responder(task, session):
        seen["sandbox"] = task.sandbox
        return ExecResult(final_message="ok", ok=True, finalization_status="completed",
                          usage=TokenUsage(input=1, output=1))

    ctx = OrchestrationContext(
        Engine(tempfile.mkdtemp(), run_id="t"), executor=FakeExecutor(responder),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")], source_repo=_git_repo(),
        base_commit=None, score_fn=lambda wt: None, prompt_builder=lambda c, i, s: "x",
        sandbox="workspace-write",
    )
    ctx.ask("read the repo", agent_id=7)
    assert seen["sandbox"] == "read-only"    # forced read-only regardless of ctx.sandbox


# --- 2.2 structural soft-write seam ------------------------------------------- #
def test_set_soft_cannot_touch_accepted():
    c = Candidate(candidate_id="x", accepted=True)
    c.set_soft(perspective=0.9, eg_critic=0.5)
    assert c.accepted is True                      # unchanged
    assert c.perspective_score == 0.9 and c.eg_critic_tiebreak == 0.5


def test_refute_only_downgrades():
    c = Candidate(candidate_id="x", accepted=True)
    assert c.refute().accepted is False
    assert c.refute().accepted is False            # idempotent; never promotes
    c2 = Candidate(candidate_id="y", accepted=False)
    assert c2.refute().accepted is False           # cannot become True


# --- 2.3 patterns: cannot promote acceptance; degrade to best-of-N ------------- #
def test_adversarial_verify_downgrades_accepted():
    ctx = _ctx_with(_ask_resp({"refuted": True}))
    c = Candidate(candidate_id="a", accepted=True, public_signal_score=1.0)
    assert ctx.adversarial_verify(c, n=1).accepted is False


def test_adversarial_verify_cannot_promote_or_spend_on_unaccepted():
    ctx = _ctx_with(_ask_resp({"refuted": False}))
    c = Candidate(candidate_id="a", accepted=False, public_signal_score=0.5)
    assert ctx.adversarial_verify(c, n=3).accepted is False
    assert ctx.agents_used() == 0                  # no-op on a non-accepted candidate


def test_adversarial_verify_keeps_accept_when_not_refuted():
    ctx = _ctx_with(_ask_resp({"refuted": False}))
    c = Candidate(candidate_id="a", accepted=True, public_signal_score=1.0)
    assert ctx.adversarial_verify(c, n=3).accepted is True  # majority not met -> stands


def test_judge_panel_sets_soft_not_accept():
    ctx = _ctx_with(_ask_resp({"score": 0.7}))
    c1 = Candidate(candidate_id="a", accepted=True, public_signal_score=1.0)
    c2 = Candidate(candidate_id="b", accepted=False, public_signal_score=0.4)
    ctx.judge_panel([c1, c2])
    assert abs(c1.perspective_score - 0.7) < 1e-9
    assert c1.accepted is True and c2.accepted is False     # accept untouched


def test_synthesize_degrades_to_best_without_agent():
    ctx = _ctx_with(_ask_resp({"plan": "do it"}))
    only = Candidate(candidate_id="a", accepted=False, public_signal_score=0.5)
    assert ctx.synthesize([only], attempt_id=1) is only     # single -> no extra agent
    acc = Candidate(candidate_id="b", accepted=True, public_signal_score=1.0)
    assert ctx.synthesize([only, acc], attempt_id=2) is acc # accepted present -> return it
    assert ctx.agents_used() == 0


def test_loop_until_dry_stops_on_accept():
    ctx = _ctx_with(_ask_resp(None))
    acc = Candidate(candidate_id="a", accepted=True, public_signal_score=1.0)
    rounds = {"n": 0}

    def make_round(i):
        rounds["n"] += 1
        return [lambda: acc]

    out = ctx.loop_until_dry(make_round, k_dry=2, max_rounds=10)
    assert rounds["n"] == 1 and acc in out         # completion-first: stop on accept


def test_completeness_critic_returns_signal_not_candidate():
    ctx = _ctx_with(_ask_resp({"complete": False, "gaps": ["x"]}))
    c = Candidate(candidate_id="a", accepted=False, public_signal_score=0.3)
    out = ctx.completeness_critic(c)
    assert isinstance(out, dict) and out.get("gaps") == ["x"]
    assert not isinstance(out, Candidate)


# --- 2.2 lint: authored code cannot assign the execution gate ------------------ #
def test_lint_forbids_assigning_accepted():
    from apex_omega.autogen.sandbox import lint_source
    src = ("def orchestrate(ctx):\n"
           "    cands = ctx.parallel([ctx.make_attempt(0)])\n"
           "    c = cands[0]\n"
           "    c.accepted = True\n"
           "    return c\n")
    res = lint_source(src)
    assert not res.ok and any("accepted" in v for v in res.violations)


def test_default_orchestration_still_lints_ok():
    from apex_omega.autogen.sandbox import lint_source
    from apex_omega.autogen.templates import DEFAULT_ORCHESTRATION
    assert lint_source(DEFAULT_ORCHESTRATION).ok


# --- 2.4 teaching: the quality-pattern exemplar is valid + runs end-to-end ----- #
def test_pattern_exemplar_lints_and_runs():
    from apex_omega.autogen.architect import PATTERN_EXEMPLAR
    from apex_omega.autogen.sandbox import lint_source, run_orchestration
    from apex_omega.kernel.verify import VerificationResult
    assert lint_source(PATTERN_EXEMPLAR).ok        # frozen-replayable authored code

    def score(wt):
        return VerificationResult(accepted=True, score=1.0, pass_rate=1.0, passed=1, total=1)

    ctx = OrchestrationContext(
        Engine(tempfile.mkdtemp(), run_id="t"), executor=FakeExecutor(),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5")], source_repo=_git_repo(),
        base_commit=None, score_fn=score, prompt_builder=lambda c, i, s: "impl",
    )
    winner = run_orchestration(PATTERN_EXEMPLAR, ctx)   # composes waves+synth+verify+select
    assert winner is not None and winner.accepted is True
