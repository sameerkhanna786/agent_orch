"""Agent-scout fan-out: difficulty -> initial agent count + approach plumbing."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from apex_omega.autogen import agent_scout, autosolve, difficulty_profile
from apex_omega.autogen.context import OrchestrationContext
from apex_omega.engine.runtime import Engine
from apex_omega.executor.fake import FakeExecutor
from apex_omega.kernel.verify import VerificationResult
from apex_omega.types import ExecResult, TokenUsage
from apex_omega.workflows.best_of_n import WorkerSpec


def _git_repo() -> str:
    d = Path(tempfile.mkdtemp()) / "repo"
    d.mkdir()
    (d / "mod.py").write_text("def add(a, b):\n    return a - b\n")
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-qm", "base"]):
        subprocess.run(c, cwd=d, check=True, capture_output=True)
    return str(d)


def _git_repo_n(n_files: int) -> str:
    """A repo with ``n_files`` source modules, to control the static file-count
    difficulty proxy (build_repo_map: <15 easy, <80 medium, >=80 hard)."""
    d = Path(tempfile.mkdtemp()) / "repo"
    d.mkdir()
    for i in range(n_files):
        (d / f"mod{i}.py").write_text("def add(a, b):\n    return a - b\n")
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-qm", "base"]):
        subprocess.run(c, cwd=d, check=True, capture_output=True)
    return str(d)


def _scout_responder(by_index):
    """A FakeExecutor responder that answers scout prompts with a difficulty keyed
    deterministically by the scout index (thread-safe under parallel), and patches
    the worktree on solve prompts."""
    def responder(task, session):
        props = (task.schema or {}).get("properties") or {}
        if "difficulty" in props:  # a scout call
            i = int((task.scoped_inputs or {}).get("scout", 0))
            d = by_index[i % len(by_index)]
            return ExecResult(structured_output={"difficulty": d, "approach": f"plan-{d}",
                                                  "key_files": ["mod.py"], "risks": "none"},
                              ok=True, finalization_status="completed", usage=TokenUsage(input=1, output=1))
        Path(session.cwd, "mod.py").write_text("def add(a, b):\n    return a + b\n")
        return ExecResult(final_message="patched", ok=True, finalization_status="completed",
                          usage=TokenUsage(input=1, output=1))
    return responder


def _specs():
    return [WorkerSpec("codex_cli", "gpt-5.5")]


def test_difficulty_profile_mapping_and_clamp():
    assert difficulty_profile("easy") == (1, 8)
    assert difficulty_profile("medium") == (3, 24)
    assert difficulty_profile("hard") == (8, 64)
    assert difficulty_profile(None) == (3, 24)         # unknown -> medium
    assert difficulty_profile("hard", ceiling=4) == (4, 4)  # clamped to the hard ceiling


def test_plan_waves_starts_at_initial_agents():
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=200)
    ctx = OrchestrationContext(eng, executor=FakeExecutor(), worker_specs=_specs(),
                               source_repo=_git_repo(), base_commit=None,
                               score_fn=lambda w: None, prompt_builder=lambda c, i, s: "x",
                               max_agents=64, initial_agents=8)
    waves = ctx.plan_waves()
    assert waves[0] == 8                # FIRST wave size = difficulty-driven initial agents
    assert sum(waves) <= 64             # never exceeds the soft cap
    # easy: a single agent first
    ctx2 = OrchestrationContext(eng, executor=FakeExecutor(), worker_specs=_specs(),
                                source_repo=_git_repo(), base_commit=None,
                                score_fn=lambda w: None, prompt_builder=lambda c, i, s: "x",
                                max_agents=8, initial_agents=1)
    assert ctx2.plan_waves()[0] == 1


def test_agent_scout_aggregates_median():
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=200)
    fx = FakeExecutor(_scout_responder(["easy", "hard", "hard"]))
    out = agent_scout(eng, executor=fx, worker_specs=_specs(), source_repo=_git_repo(),
                      base_commit=None, base_repo_map={"difficulty": "medium"}, n_scouts=3)
    assert out["difficulty"] == "hard"      # median of [easy, hard, hard]
    assert out["n_scouts"] == 3 and out["source"] == "agent_scout"
    assert "plan-hard" in out["approach"]


def test_agent_scout_fails_open_to_static():
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=200)

    def responder(task, session):  # no structured output -> scouts yield None
        return ExecResult(final_message="no json", ok=True, finalization_status="completed")

    out = agent_scout(eng, executor=FakeExecutor(responder), worker_specs=_specs(),
                      source_repo=_git_repo(), base_commit=None,
                      base_repo_map={"difficulty": "easy"}, n_scouts=2)
    assert out["source"] == "static_fallback" and out["difficulty"] == "easy"


def test_autosolve_scout_cannot_inflate_above_static_proxy():
    # ANTI-INFLATION (jinja fix): a static-MEDIUM repo whose scouts all say "hard" must NOT be
    # escalated to hard — inflation steers the architect toward heavy decompose/repair and (at
    # small budgets) cannibalizes solve shots. The scout still INFORMS approach/key_files and may
    # refine difficulty DOWN, but never UP past the static proxy. So difficulty stays medium.
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=500)
    fx = FakeExecutor(_scout_responder(["hard", "hard", "hard"]))
    r = autosolve(eng, source_repo=_git_repo_n(20), executor=fx, worker_specs=_specs(),
                  score_fn=lambda w: VerificationResult(accepted=True, score=1.0, passed=1,
                                                        total=1, pass_rate=1.0),
                  prompt_builder=lambda c, i, s: "fix", author=False, scout_agents=3)
    assert r["scout"]["difficulty"] == "hard"          # scout still reports its raw read
    assert r["difficulty"] == "medium"                 # but NOT inflated above the static proxy
    assert r["agent_budget"]["initial"] == 3 and r["agent_budget"]["soft_cap"] == 24
    assert r["solved"] is True


def test_autosolve_caps_scouts_at_small_budget():
    # BUDGET-AWARE scouting (jinja fix): scouts count against the same K-agent pool as solve
    # attempts, so at small budgets a 3-scout fan-out is capped (K=8 -> 1 scout) to leave the
    # majority of the budget for solve attempts. At a large/unbounded budget the cap is a no-op.
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=500)
    sc = lambda w: VerificationResult(accepted=True, score=1.0, passed=1, total=1, pass_rate=1.0)
    r = autosolve(eng, source_repo=_git_repo_n(20), executor=FakeExecutor(_scout_responder(["hard"] * 3)),
                  worker_specs=_specs(), score_fn=sc, prompt_builder=lambda c, i, s: "fix",
                  author=False, scout_agents=3, max_agents=8)
    assert r["scout"]["n_scouts"] == 1                 # 8 // 6 = 1 -> capped from 3
    r2 = autosolve(eng, source_repo=_git_repo_n(20), executor=FakeExecutor(_scout_responder(["hard"] * 3)),
                   worker_specs=_specs(), score_fn=sc, prompt_builder=lambda c, i, s: "fix",
                   author=False, scout_agents=3, max_agents=1000)
    assert r2["scout"]["n_scouts"] == 3                # unbounded -> keeps the requested 3


def test_autosolve_easy_static_clamps_scout_inflation():
    # Regression for the voluptuous over-budget (4 agents for a 1-agent task): a
    # statically-EASY (small) repo must NOT be inflated to a 64-agent budget just
    # because the scout rates it "hard". The scout still informs approach/difficulty,
    # but the budget difficulty stays easy -> a 1-agent probe first.
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=500)
    fx = FakeExecutor(_scout_responder(["hard", "hard", "hard"]))
    r = autosolve(eng, source_repo=_git_repo(), executor=fx, worker_specs=_specs(),  # 1-file -> easy
                  score_fn=lambda w: VerificationResult(accepted=True, score=1.0, passed=1,
                                                        total=1, pass_rate=1.0),
                  prompt_builder=lambda c, i, s: "fix", author=False, scout_agents=3)
    assert r["scout"]["difficulty"] == "hard"          # scout still aggregates "hard"
    assert r["difficulty"] == "easy"                   # but budget difficulty stays easy
    assert r["agent_budget"]["initial"] == 1 and r["agent_budget"]["soft_cap"] == 8
    assert r["solved"] is True


def test_autosolve_easy_uses_one_agent_first():
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=500)
    fx = FakeExecutor(_scout_responder(["easy", "easy", "easy"]))
    r = autosolve(eng, source_repo=_git_repo(), executor=fx, worker_specs=_specs(),
                  score_fn=lambda w: VerificationResult(accepted=True, score=1.0, passed=1,
                                                        total=1, pass_rate=1.0),
                  prompt_builder=lambda c, i, s: "fix", author=False, scout_agents=3)
    assert r["scout"]["difficulty"] == "easy"
    assert r["agent_budget"]["initial"] == 1   # easy -> a single attempt first (fewest agents)
    assert r["solved"] is True
