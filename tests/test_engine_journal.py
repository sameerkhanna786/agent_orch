"""Engine + durable-journal invariants (plan §02, §15).

Run with: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 <apex venv python> -m pytest tests/ -q
(the apex venv ships third-party pytest plugins that break collection in a bare
repo; APEX-Ω's own tests need none).
"""

from __future__ import annotations

import tempfile

from apex_omega.engine import Engine, Budget, Stage
from apex_omega.executor import FakeExecutor
from apex_omega.types import ExecResult, ScopedTask, TokenUsage


def _runner(session):
    return lambda task: session.run(task)


def _task(prompt, sha="base", **kw):
    return ScopedTask(prompt=prompt, model="m", vendor="codex_cli",
                      scoped_inputs={"repo_snapshot_sha": sha}, **kw)


def test_agent_resume_same_process():
    run_dir = tempfile.mkdtemp()
    fx = FakeExecutor()
    eng = Engine(run_dir, run_id="t")
    sess = fx.spawn(run_dir, "codex_cli", "m")
    t = _task("do x")
    r1 = eng.agent(t, _runner(sess), node_id="n", cli_version="v1")
    r2 = eng.agent(t, _runner(sess), node_id="n", cli_version="v1")
    assert fx.calls == 1, "identical call must hit cache"
    assert r1.final_message == r2.final_message


def test_agent_resume_across_restart():
    run_dir = tempfile.mkdtemp()
    fx = FakeExecutor()
    Engine(run_dir, run_id="t").agent(_task("do x"), _runner(fx.spawn(run_dir, "codex_cli", "m")),
                                      node_id="n", cli_version="v1")
    assert fx.calls == 1
    fx.reset_calls()
    # new engine, same run_dir -> served from WAL, no worker call
    eng2 = Engine(run_dir, run_id="t")
    eng2.agent(_task("do x"), _runner(fx.spawn(run_dir, "codex_cli", "m")), node_id="n", cli_version="v1")
    assert fx.calls == 0, "restart resume must serve from WAL"


def test_edited_and_changed_snapshot_rerun():
    run_dir = tempfile.mkdtemp()
    fx = FakeExecutor()
    eng = Engine(run_dir, run_id="t")
    sess = fx.spawn(run_dir, "codex_cli", "m")
    eng.agent(_task("do x", sha="A"), _runner(sess), node_id="n", cli_version="v1")
    assert fx.calls == 1
    eng.agent(_task("do x DIFFERENT", sha="A"), _runner(sess), node_id="n", cli_version="v1")
    assert fx.calls == 2, "edited prompt must re-run"
    eng.agent(_task("do x", sha="B"), _runner(sess), node_id="n", cli_version="v1")
    assert fx.calls == 3, "changed repo snapshot must re-run (stale-vs-changed-code guard)"


def test_cli_version_change_reruns():
    run_dir = tempfile.mkdtemp()
    fx = FakeExecutor()
    eng = Engine(run_dir, run_id="t")
    sess = fx.spawn(run_dir, "codex_cli", "m")
    eng.agent(_task("do x"), _runner(sess), node_id="n", cli_version="v1")
    eng.agent(_task("do x"), _runner(sess), node_id="n", cli_version="v2")
    assert fx.calls == 2, "cli_version change must re-run (mid-run vendor swap)"


def test_budget_charged_on_fresh_not_cached():
    run_dir = tempfile.mkdtemp()
    fx = FakeExecutor()
    eng = Engine(run_dir, run_id="t")
    sess = fx.spawn(run_dir, "codex_cli", "m")
    t = _task("hello world")
    r1 = eng.agent(t, _runner(sess), node_id="n", cli_version="v1")
    spent_after_first = eng.budget.spent()
    assert spent_after_first == r1.usage.total > 0
    eng.agent(t, _runner(sess), node_id="n", cli_version="v1")  # cache hit
    assert eng.budget.spent() == spent_after_first, "cache hit must not charge budget"


def test_parallel_null_on_fail():
    run_dir = tempfile.mkdtemp()
    eng = Engine(run_dir, run_id="t")

    def boom():
        raise RuntimeError("x")

    out = eng.parallel([lambda: 1, boom, lambda: 3])
    assert out == [1, None, 3]


def test_pipeline_deterministic_order_and_streaming():
    run_dir = tempfile.mkdtemp()
    eng = Engine(run_dir, run_id="t")
    items = [{"id": f"i{i}", "n": i} for i in range(6)]
    res = eng.pipeline(
        items,
        Stage("double", lambda prev, it, ix: {"id": it["id"], "v": it["n"] * 2}),
        Stage("inc", lambda prev, it, ix: {"id": prev["id"], "v": prev["v"] + 1}),
        item_id=lambda it: it["id"],
    )
    # order is a pure function of input index, not completion time
    assert [r["v"] for r in res] == [1, 3, 5, 7, 9, 11]


def test_pipeline_stage_failure_drops_item():
    run_dir = tempfile.mkdtemp()
    eng = Engine(run_dir, run_id="t")
    items = [{"id": "ok"}, {"id": "bad"}]

    def s1(prev, it, ix):
        if it["id"] == "bad":
            raise ValueError("boom")
        return {"id": it["id"], "ok": True}

    res = eng.pipeline(items, Stage("s1", s1), Stage("s2", lambda p, it, ix: {**p, "s2": True}),
                       item_id=lambda it: it["id"])
    assert res[0] == {"id": "ok", "ok": True, "s2": True}
    assert res[1] is None  # failed item dropped, never faked


def test_budget_can_start():
    b = Budget(total=100)
    assert b.can_start()
    b.add(100)
    assert not b.can_start()
    assert Budget().can_start()  # unbounded


def test_pipeline_budget_preserves_completed_stage_output():
    # Regression (review finding #1): budget exhaustion between stages must NOT
    # discard a stage that already produced output (never suppress a candidate
    # with execution evidence). Only an item that never started -> None.
    run_dir = tempfile.mkdtemp()
    eng = Engine(run_dir, run_id="t", budget=Budget(total=5))
    items = [{"id": "i0"}]

    def s0(prev, it, ix):
        eng.budget.add(10)  # exhaust the budget during stage 0
        return {"id": it["id"], "done0": True}

    def s1(prev, it, ix):
        return {**prev, "done1": True}

    res = eng.pipeline(items, Stage("s0", s0), Stage("s1", s1), item_id=lambda it: it["id"])
    assert res[0] == {"id": "i0", "done0": True}, "completed stage output must survive budget exhaustion"


def test_pipeline_accepts_non_json_native_items():
    # Regression (review finding #4): raw input items need not be JSON-native;
    # stage-0 keys on the stable item id, not the raw object.
    class Doc:
        def __init__(self, n):
            self.id = f"d{n}"
            self.n = n

    run_dir = tempfile.mkdtemp()
    eng = Engine(run_dir, run_id="t")
    items = [Doc(1), Doc(2)]
    res = eng.pipeline(items, Stage("first", lambda prev, it, ix: {"id": it.id, "v": it.n}),
                       item_id=lambda it: it.id)
    assert res == [{"id": "d1", "v": 1}, {"id": "d2", "v": 2}]
