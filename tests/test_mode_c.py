"""Mode C (generated-code orchestrator on commit0) — unit tests.

Covers, WITHOUT any paid commit0 call:
  * plan_waves doubling-to-cap schedule (small cap -> few waves; cap=1000 -> escalates)
    and tuple back-compat.
  * difficulty -> soft max_agents mapping (easy=8 / medium=24 / hard=64, ceiling 1000).
  * the autogen arm routing: Commit0EvalDriver.run_cell dispatches to
    run_autogen_cell (Mode C) instead of the v1 subprocess _invoke_runner.
"""

from __future__ import annotations

import tempfile

import apex_omega.eval.commit0_driver as driver_mod
from apex_omega.ablation import build_ablation_config, get_arm
from apex_omega.autogen.context import OrchestrationContext
from apex_omega.eval import registry
from apex_omega.eval.commit0_autogen import difficulty_to_max_agents
from apex_omega.eval.commit0_driver import Commit0EvalDriver
from apex_omega.engine.runtime import Engine


# ---- plan_waves: doubling-to-cap schedule ---------------------------------
class _StubCtx:
    """Minimal stand-in exposing only what plan_waves touches."""
    def __init__(self, max_agents: int, initial_agents: int = 1):
        self.max_agents = max_agents
        self.initial_agents = initial_agents

    # bind the real implementation
    plan_waves = OrchestrationContext.plan_waves


def test_plan_waves_doubling_small_cap_few_waves():
    # small cap -> a couple/few waves, doubling, summing to exactly the cap
    waves = _StubCtx(max_agents=10).plan_waves()
    assert waves == [1, 2, 4, 3], waves  # 1+2+4=7, then 3 caps at 10
    assert sum(waves) == 10
    # most tasks: a handful of agents, not hundreds
    assert len(waves) <= 5


def test_plan_waves_doubling_escalates_toward_1000():
    waves = _StubCtx(max_agents=1000).plan_waves()
    assert sum(waves) == 1000
    # geometric escalation -> reaches the high cap in O(log) waves, not 1000 waves
    assert len(waves) < 20
    assert waves[0] == 1 and waves[1] == 2 and waves[2] == 4
    # but only escalates UP TO the cap, never beyond
    assert all(w > 0 for w in waves)


def test_plan_waves_respects_max_wave_per_wave_cap():
    waves = _StubCtx(max_agents=2000).plan_waves(max_wave=8)
    assert max(waves) == 8
    assert sum(waves) == 2000


def test_plan_waves_start_factor():
    waves = _StubCtx(max_agents=100).plan_waves(start=2, factor=3)
    # 2, 6, 18, 54, then 20 caps at 100
    assert waves[0] == 2 and waves[1] == 6 and waves[2] == 18
    assert sum(waves) == 100


def test_plan_waves_tuple_backcompat():
    # explicit tuple schedule keeps the old fixed-wave behaviour, bounded by cap
    waves = _StubCtx(max_agents=100).plan_waves((1, 3, 5, 8))
    assert waves == [1, 3, 5, 8]
    capped = _StubCtx(max_agents=6).plan_waves((1, 3, 5, 8))
    assert capped == [1, 3, 2] and sum(capped) == 6


# ---- difficulty -> soft max_agents ----------------------------------------
def test_difficulty_to_max_agents_mapping():
    assert difficulty_to_max_agents("easy") == 8
    assert difficulty_to_max_agents("medium") == 24
    assert difficulty_to_max_agents("hard") == 64
    # unknown / None -> the medium default
    assert difficulty_to_max_agents(None) == 24
    assert difficulty_to_max_agents("weird") == 24


def test_difficulty_to_max_agents_clamped_to_ceiling():
    # never exceeds the hard ceiling
    assert difficulty_to_max_agents("hard", ceiling=10) == 10
    assert difficulty_to_max_agents("easy", ceiling=4) == 4
    # default ceiling is 1000 (the backstop)
    assert difficulty_to_max_agents("hard") <= 1000


# ---- dispatch: autogen arm routes to Mode C -------------------------------
def test_autogen_arm_flips_orchestrator():
    abl = build_ablation_config(get_arm("autogen_orchestrator"))
    assert abl.orchestrator == "autogen"
    # the default best-of-N arm does NOT
    assert build_ablation_config(get_arm("baseline")).orchestrator == "best_of_n"


def _base_cfg() -> dict:
    return {"llm_configs": [{"backend": "codex_cli", "model": "gpt-5.5"}]}


def test_run_cell_routes_autogen_to_mode_c(monkeypatch):
    """The autogen arm must dispatch run_cell -> run_autogen_cell (in-process),
    NOT the v1 subprocess _invoke_runner.  Monkeypatch both to assert routing
    without any paid call."""
    rd = tempfile.mkdtemp()
    driver = Commit0EvalDriver(rd, _base_cfg(), autogen_max_agents=7, autogen_author=False)
    # the per-cell auth preflight makes a real vendor probe — stub it so this routing test stays
    # paid-call-free (auth preflight is exercised in tests/test_auth_preflight.py).
    import apex_omega.executor.auth_env as _ae
    monkeypatch.setattr(_ae, "refresh_vendor_auth", lambda v, **k: (True, "test"))

    called = {"autogen": 0, "subprocess": 0, "kwargs": None}

    def fake_autogen(engine, cfg_dict, repo, **kwargs):
        called["autogen"] += 1
        called["kwargs"] = kwargs
        return {"_report_path": str(driver.run_dir / "r.json"), "_returncode": 0,
                "total_tasks": 1, "solved_tasks": 1, "agents_used": 3,
                "average_pass_rate_percent": 100.0, "_mode": "autogen"}

    def fake_subprocess(self, *a, **k):
        called["subprocess"] += 1
        return {"_report_path": "x", "total_tasks": 1, "solved_tasks": 0, "_returncode": 0}

    # patch where _invoke_autogen looks it up (module-level import inside method)
    import apex_omega.eval.commit0_autogen as autogen_mod
    monkeypatch.setattr(autogen_mod, "run_autogen_cell", fake_autogen)
    monkeypatch.setattr(autogen_mod, "write_cell_report", lambda rep, od: rep)
    monkeypatch.setattr(Commit0EvalDriver, "_invoke_runner", fake_subprocess)

    arm = get_arm("autogen_orchestrator")
    report = driver.run_cell(arm, "voluptuous", limit=1)

    assert called["autogen"] == 1, "autogen arm should route to run_autogen_cell"
    assert called["subprocess"] == 0, "autogen arm must NOT hit the v1 subprocess"
    assert report["solved_tasks"] == 1
    # the driver's autogen knobs are passed through
    assert called["kwargs"]["max_agents"] == 7
    assert called["kwargs"]["author"] is False
    assert called["kwargs"]["agent_ceiling"] == 1000


def test_run_cell_routes_nonautogen_to_subprocess(monkeypatch):
    """A normal (best_of_n) arm must still go through the v1 subprocess path."""
    rd = tempfile.mkdtemp()
    driver = Commit0EvalDriver(rd, _base_cfg())
    import apex_omega.executor.auth_env as _ae
    monkeypatch.setattr(_ae, "refresh_vendor_auth", lambda v, **k: (True, "test"))

    called = {"autogen": 0, "subprocess": 0}

    def fake_subprocess(self, *a, **k):
        called["subprocess"] += 1
        return {"_report_path": "x", "total_tasks": 1, "solved_tasks": 0, "_returncode": 0}

    import apex_omega.eval.commit0_autogen as autogen_mod
    monkeypatch.setattr(autogen_mod, "run_autogen_cell",
                        lambda *a, **k: called.__setitem__("autogen", called["autogen"] + 1) or {})
    monkeypatch.setattr(Commit0EvalDriver, "_invoke_runner", fake_subprocess)

    driver.run_cell(get_arm("baseline"), "voluptuous", limit=1)
    assert called["subprocess"] == 1
    assert called["autogen"] == 0


def test_worker_specs_from_cfg_mixed_vendors():
    from apex_omega.eval.commit0_autogen import _worker_specs_from_cfg

    specs = _worker_specs_from_cfg({"llm_configs": [
        {"backend": "codex_cli", "model": "gpt-5.5"},
        {"backend": "claude_cli", "model": "opus", "cli_model_id": "claude-opus-4-8[1m]"},
    ]})
    assert [s.vendor for s in specs] == ["codex_cli", "claude_cli"]
    assert specs[1].extra.get("cli_model_id") == "claude-opus-4-8[1m]"
    # empty config -> safe default single codex worker
    assert _worker_specs_from_cfg({})[0].vendor == "codex_cli"
