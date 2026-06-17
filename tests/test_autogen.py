"""Generated-code orchestration: sandbox/lint + freeze + fail-open (plan §7.3)."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from apex_omega.autogen import (
    DEFAULT_ORCHESTRATION,
    autosolve,
    extract_code,
    lint_source,
    run_orchestration,
)
from apex_omega.engine.runtime import Engine
from apex_omega.errors import FailLoud
from apex_omega.executor.fake import FakeExecutor
from apex_omega.kernel.verify import VerificationResult
from apex_omega.workflows.best_of_n import WorkerSpec


# ---- lint / sandbox (pure, fast) ------------------------------------------
def test_lint_rejects_imports_and_forbidden():
    assert not lint_source("import os\ndef orchestrate(ctx):\n    return None\n").ok
    assert not lint_source("def orchestrate(ctx):\n    return open('/etc/passwd')\n").ok
    assert not lint_source("def orchestrate(ctx):\n    return ctx.__class__\n").ok
    assert not lint_source("import random\ndef orchestrate(ctx):\n    return None\n").ok


def test_lint_requires_orchestrate():
    assert not lint_source("def other(ctx):\n    return 1\n").ok


def test_lint_accepts_default_template():
    assert lint_source(DEFAULT_ORCHESTRATION).ok


def test_run_orchestration_blocks_escape_and_runs_safe():
    # forbidden source is rejected at lint
    with pytest.raises(FailLoud):
        run_orchestration("import os\ndef orchestrate(ctx):\n    return None\n", object())

    # a safe script using only builtins + ctx runs
    class Ctx:
        def select(self, xs):
            return sorted(xs)[-1] if xs else None
    out = run_orchestration("def orchestrate(ctx):\n    return ctx.select([3,1,2])\n", Ctx())
    assert out == 3


def test_extract_code_from_markdown():
    txt = "sure!\n```python\ndef orchestrate(ctx):\n    return 1\n```\ndone"
    assert "def orchestrate(ctx)" in extract_code(txt)


# ---- end-to-end autosolve with a stub scorer (fast; no pytest subprocess) ---
def _git_repo() -> str:
    d = Path(tempfile.mkdtemp()) / "repo"
    d.mkdir()
    (d / "mod.py").write_text("def add(a, b):\n    return a - b\n")
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-qm", "base"]):
        subprocess.run(c, cwd=d, check=True, capture_output=True)
    return str(d)


def _accept_score(_wt):
    return VerificationResult(accepted=True, score=1.0, passed=1, total=1, pass_rate=1.0)


def _specs():
    return [WorkerSpec("codex_cli", "gpt-5.5"), WorkerSpec("claude_cli", "opus")]


def _pb(ctx, i, strat):
    return f"fix add ({strat} {i})"


def test_autosolve_template_solves():
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=32)
    r = autosolve(eng, source_repo=_git_repo(), executor=FakeExecutor(), worker_specs=_specs(),
                  score_fn=_accept_score, prompt_builder=_pb, author=False, max_agents=10)
    assert r["solved"] is True and r["orchestration"]["origin"] == "template"
    assert r["agents_used"] >= 1


def test_autosolve_fails_open_on_bad_author():
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=32)
    r = autosolve(eng, source_repo=_git_repo(), executor=FakeExecutor(), worker_specs=_specs(),
                  score_fn=_accept_score, prompt_builder=_pb, author=True,
                  author_fn=lambda rm: "import os\ndef orchestrate(ctx):\n    return None\n",
                  max_agents=10)
    # bad author lints out -> fallback template -> still solves (completion-first floor)
    assert r["solved"] is True and r["orchestration"]["origin"] == "fallback"


def test_autosolve_fails_open_on_runtime_crash():
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=32)
    r = autosolve(eng, source_repo=_git_repo(), executor=FakeExecutor(), worker_specs=_specs(),
                  score_fn=_accept_score, prompt_builder=_pb, author=True,
                  author_fn=lambda rm: "def orchestrate(ctx):\n    raise RuntimeError('boom')\n",
                  max_agents=10)
    assert r["solved"] is True  # crashed strategy -> floor -> solved


def test_autosolve_freeze_resume():
    rd = tempfile.mkdtemp()
    eng = Engine(rd, run_id="t", max_total_agents=32)
    repo = _git_repo()
    r1 = autosolve(eng, source_repo=repo, executor=FakeExecutor(), worker_specs=_specs(),
                   score_fn=_accept_score, prompt_builder=_pb, author=False, max_agents=10)
    sha1 = (Path(rd) / "orchestrator" / "frozen.json")
    assert sha1.exists()
    # second run over the same run_dir reuses the frozen orchestration (no re-author)
    eng2 = Engine(rd, run_id="t", max_total_agents=32)
    from apex_omega.autogen import load_frozen
    assert load_frozen(eng2) is not None


def test_authored_can_use_type_builtin():
    # Regression: jinja autogen failed because authored code used type() which the
    # sandbox forbade -> NameError -> 0 attempts -> 0/1. Must now run clean.
    src = ("def orchestrate(ctx):\n"
           "    ws = ctx.worker_specs\n"
           "    cs = []\n"
           "    for j in range(2):\n"
           "        w = ws[j % len(ws)]\n"
           "        if type(w) == tuple or type(ws) == list:\n"
           "            c = ctx.solve_attempt(attempt_id=j, vendor=w[0], model=w[1])\n"
           "            if c is not None:\n"
           "                cs.append(c)\n"
           "    return ctx.select(cs)\n")
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=32)
    r = autosolve(eng, source_repo=_git_repo(), executor=FakeExecutor(), worker_specs=_specs(),
                  score_fn=_accept_score, prompt_builder=_pb, author=True,
                  author_fn=lambda rm: src, max_agents=8)
    assert r["solved"] is True and r["orchestration"]["origin"] == "authored", r


def test_authored_abstain_is_a_real_failure_not_template_rescued():
    # PROPER-COMPARISON SEMANTICS (fail-open-to-template invariant intentionally
    # dropped): an authored plan that runs cleanly but abstains must be reported as
    # autogen's OWN failure (solved False, origin stays "authored") — it must NOT
    # silently inherit the best-of-N template's solve. A *malformed* orchestrator
    # (lint/crash) still falls open; abstention does not.
    src = "def orchestrate(ctx):\n    return ctx.select([])\n"   # runs clean, accepts nothing
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=32)
    r = autosolve(eng, source_repo=_git_repo(), executor=FakeExecutor(), worker_specs=_specs(),
                  score_fn=_accept_score, prompt_builder=_pb, author=True,
                  author_fn=lambda rm: src, max_agents=8)
    assert r["solved"] is False, r                               # honest autogen failure
    assert r["orchestration"]["origin"] == "authored", r["orchestration"]


def test_worker_spec_dual_access():
    # Regression: LLM-authored code accessed workers as tuples (w[0]/w[1]).
    w = WorkerSpec("codex_cli", "gpt-5.5", {"cli_model_id": "x"})
    assert w[0] == "codex_cli" and w[1] == "gpt-5.5"           # tuple-like
    assert w["vendor"] == "codex_cli" and w["model"] == "gpt-5.5"  # dict-like
    assert w.vendor == "codex_cli" and w.get("model") == "gpt-5.5"  # attr + get
    v, m = w                                                    # unpacking
    assert (v, m) == ("codex_cli", "gpt-5.5") and len(w) == 2


def test_candidate_dual_access():
    from apex_omega.kernel.verify import VerificationResult, candidate_from_verification
    c = candidate_from_verification(candidate_id="r0", diff="d",
                                    vr=VerificationResult(accepted=True, score=1.0))
    assert c["accepted"] is True and c.get("combined_score") == 1.0 and c.accepted is True


def test_autosolve_authored_subscripts_workers_runs_clean():
    # The exact pattern that crashed cookiecutter: workers[i][0]/[1]. Must now run
    # as origin=authored (not crash -> fail-open).
    src = (
        "def orchestrate(ctx):\n"
        "    ctx.phase('x')\n"
        "    workers = ctx.worker_specs\n"
        "    cands = []\n"
        "    for j in range(2):\n"
        "        w = workers[j % len(workers)]\n"
        "        vendor = w[0]\n"
        "        model = w[1]\n"
        "        c = ctx.solve_attempt(attempt_id=j, vendor=vendor, model=model)\n"
        "        if c is not None:\n"
        "            cands.append(c)\n"
        "    return ctx.select(cands)\n"
    )
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=32)
    r = autosolve(eng, source_repo=_git_repo(), executor=FakeExecutor(), worker_specs=_specs(),
                  score_fn=_accept_score, prompt_builder=_pb, author=True,
                  author_fn=lambda rm: src, max_agents=8)
    assert r["solved"] is True, r
    assert r["orchestration"]["origin"] == "authored", r["orchestration"]


def test_engine_total_agent_ceiling():
    # the runaway backstop caps fresh agents even if a strategy asks for more
    from apex_omega.types import ScopedTask
    eng = Engine(tempfile.mkdtemp(), run_id="t", max_total_agents=3)
    fx = FakeExecutor()
    sess = fx.spawn(eng.run_dir.as_posix(), "codex_cli", "m")
    oks = 0
    for i in range(10):
        res = eng.agent(ScopedTask(prompt=f"p{i}", model="m", vendor="codex_cli",
                                   scoped_inputs={"i": i}), lambda t: sess.run(t),
                        node_id=f"n{i}", cli_version="v")
        oks += 1 if res.ok else 0
    assert oks == 3, f"expected 3 within ceiling, got {oks}"
