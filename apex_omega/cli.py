"""APEX-Ω command-line interface.

Run with the v1 venv python so the kernel reuse + commit0 scoring resolve, e.g.:

    cd /Users/sameertkhanna/Documents/agent_orch
    PYTHONPATH=. /Users/sameertkhanna/Documents/apex/apex/.venv/bin/python -m apex_omega doctor

Commands:
    doctor              cheap preflight (no paid calls)
    arms                list the ablation experiment matrix
    repos               list the 15 commit0 target repos + flags
    eval                run the commit0 ablation matrix (Mode A: v1-as-worker)
    bestofn-demo        free synthetic demo of the engine-native best-of-N loop
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def _p(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


# --------------------------------------------------------------------------
def cmd_doctor(args: argparse.Namespace) -> int:
    report: dict = {"ok": True, "checks": {}}

    def check(name, fn):
        try:
            report["checks"][name] = {"ok": True, "value": fn()}
        except Exception as e:
            report["checks"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            report["ok"] = False

    check("python", lambda: sys.version.split()[0])
    check("apex_omega_import", lambda: __import__("apex_omega").__version__)
    check("apex_import", lambda: __import__("apex.core.config", fromlist=["x"]).__file__)

    def _apex_vendored():
        import apex
        repo_root = str(Path(__file__).resolve().parents[1])
        vendored = apex.__file__.startswith(repo_root + "/apex/")
        return {"path": apex.__file__, "vendored_in_repo": vendored,
                "note": "self-contained" if vendored else
                        "WARNING: using an EXTERNAL apex; run with PYTHONPATH=<repo> for the vendored copy"}
    check("apex_vendored", _apex_vendored)

    check("commit0_import", lambda: __import__("commit0").__file__)
    check("hf_offline_env", lambda: {k: os.environ.get(k) for k in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE")})

    def _docker():
        if not shutil.which("docker"):
            return "docker binary absent"
        import subprocess
        r = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)
        return "daemon up" if r.returncode == 0 else "daemon DOWN (local no-Docker path required)"
    check("docker", _docker)

    def _vendors():
        return {v: bool(shutil.which(c)) for v, c in
                {"codex_cli": "codex", "claude_cli": "claude", "gemini_cli": "gemini",
                 "opencode_cli": "opencode"}.items()}
    check("vendor_clis", _vendors)

    def _vendor_auth():
        # APEX-Ω provisions Meta-gateway host-mode auth (setdefault); report the
        # resulting state. On a Meta host this makes codex/gemini/opencode work.
        from apex_omega.executor.auth_env import ensure_vendor_auth_env
        ensure_vendor_auth_env()
        present = {k: bool(os.environ.get(k)) for k in (
            "CODEX_BASE_URL", "OPENAI_API_KEY", "APEX_TARGET_RUNTIME_CLI_AUTH_MODE",
            "ANTHROPIC_VERTEX_BASE_URL", "ANTHROPIC_API_KEY")}
        return {"present": present,
                "codex_auth_ready": present["CODEX_BASE_URL"] and present["OPENAI_API_KEY"]
                                    and present["APEX_TARGET_RUNTIME_CLI_AUTH_MODE"],
                "note": "codex needs CODEX_BASE_URL+OPENAI_API_KEY+APEX_TARGET_RUNTIME_CLI_AUTH_MODE=host_cli "
                        "(provisioned by default to plugboard); gemini(gemini-3.1-pro)/opencode self-auth; "
                        "claude 401s in this sandbox (keychain). Verify with a real probe before trusting numbers."}
    check("vendor_auth", _vendor_auth)

    def _arms():
        from apex_omega.ablation import ARMS, build_ablation_config
        from apex_omega.ablation.safety_modes import validate_safety_modes
        for aid, arm in ARMS.items():
            validate_safety_modes(build_ablation_config(arm).to_safety_modes(),
                                  dynamic_coverage_available=lambda: True)
        return f"{len(ARMS)} arms safety-validate"
    check("ablation_arms", _arms)

    def _configs():
        from apex.core.config import ApexConfig
        from apex_omega.ablation import v1_runnable_arms, get_arm
        from apex_omega.eval import load_base_config, build_arm_config_dict
        base = load_base_config(args.base_config)
        for aid in v1_runnable_arms():
            ApexConfig.from_dict(build_arm_config_dict(base, get_arm(aid), rollouts=1))
        return f"{len(v1_runnable_arms())} v1-runnable arm configs load"
    check("arm_configs_load", _configs)

    def _discover():
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from apex.core.config import ApexConfig
        from apex.evaluation.commit0_benchmark import Commit0BenchmarkRunner
        from apex_omega.eval import load_base_config
        cfg = ApexConfig.from_dict(load_base_config(args.base_config))
        r = Commit0BenchmarkRunner(config=cfg, output_dir="/tmp/apexomega_doctor", split=args.probe_repo)
        tasks = r.discover_tasks(repos=[args.probe_repo], limit=1)
        return f"discover_tasks({args.probe_repo}) -> {len(tasks)} task(s)"
    check("commit0_discover", _discover)

    _p(report)
    return 0 if report["ok"] else 1


def cmd_arms(args: argparse.Namespace) -> int:
    from apex_omega.ablation import ARMS, v1_runnable_arms
    v1ok = set(v1_runnable_arms())
    rows = []
    for aid, arm in ARMS.items():
        rows.append({"id": aid, "kind": arm.kind, "maps_to": arm.maps_to,
                     "v1_runnable": aid in v1ok, "isolates": arm.isolates})
    _p({"count": len(rows), "arms": rows})
    return 0


def cmd_repos(args: argparse.Namespace) -> int:
    from apex_omega.eval import registry
    rows = [{"name": r.name, "in_lite": r.in_lite, "python": r.python_version,
             "local_runnable": r.local_runnable, "forces_docker": r.forces_docker,
             "fallback_rev": r.dataset_fallback_revision, "notes": r.notes}
            for r in registry.TARGET_REPOS]
    _p({"count": len(rows), "local_runnable": registry.local_runnable_targets(),
        "docker_required": registry.docker_required_targets(), "repos": rows})
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from apex_omega.engine.budget import Budget
    from apex_omega.eval import Commit0EvalDriver, load_base_config, registry
    from apex_omega.ablation import get_arm

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if args.repos:
        repos = [r.strip() for r in args.repos.split(",") if r.strip()]
    else:
        repos = registry.local_runnable_targets() if args.local_only else list(registry.TARGET_NAMES)
    # validate
    for a in arms:
        get_arm(a)
    for r in repos:
        registry.get(r)
    base = load_base_config(args.base_config)
    budget = Budget(total=args.budget) if args.budget else Budget()
    driver = Commit0EvalDriver(args.run_dir, base, budget=budget, agent_mode=args.agent_mode,
                               cell_timeout_seconds=args.cell_timeout,
                               autogen_max_agents=args.autogen_max_agents,
                               autogen_author=args.autogen_author,
                               autogen_scout_agents=args.autogen_scout_agents)
    print(f"[apex_omega] eval matrix arms={arms} repos={repos} limit={args.limit} "
          f"rollouts={args.rollouts} run_dir={args.run_dir}", file=sys.stderr)
    matrix = driver.run_matrix(arms, repos, limit=args.limit, rollouts=args.rollouts)
    _p({"summary": matrix["summary"], "budget": matrix["budget"],
        "journal": matrix["journal"], "report": str(Path(args.run_dir) / "matrix_report.json")})
    return 0


def cmd_bestofn_demo(args: argparse.Namespace) -> int:
    """Free, deterministic demonstration of the engine-native best-of-N loop on a
    synthetic git repo (no paid calls): worktree isolation + journaled agent() +
    real pytest scoring + Cardinal-Contract selection + abstention."""
    import subprocess
    import tempfile
    import textwrap
    from apex_omega.engine.runtime import Engine
    from apex_omega.executor.fake import FakeExecutor
    from apex_omega.types import ExecResult, TokenUsage
    from apex_omega.workflows.best_of_n import WorkerSpec, best_of_n_solve, make_pytest_score_fn

    work = Path(tempfile.mkdtemp(prefix="apexomega_bestofn_"))
    repo = work / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("def add(a, b):\n    return a - b  # BUG\n")
    (repo / "test_mod.py").write_text("from mod import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.email=a@b.c", "-c", "user.name=apex", "commit", "-qm", "base"]):
        subprocess.run(c, cwd=repo, check=True, capture_output=True)

    # A fake worker that actually writes the fix into its worktree (so real
    # scoring passes) — exercises the full loop deterministically and for free.
    def responder(task, session):
        Path(session.cwd, "mod.py").write_text("def add(a, b):\n    return a + b  # fixed\n")
        return ExecResult(final_message="patched mod.add", usage=TokenUsage(input=10, output=20),
                          ok=True, finalization_status="completed")

    engine = Engine(work / "run", run_id="bestofn_demo")
    res = best_of_n_solve(
        engine, source_repo=str(repo), executor=FakeExecutor(responder),
        worker_specs=[WorkerSpec("codex_cli", "gpt-5.5"), WorkerSpec("claude_cli", "opus")],
        build_prompt=lambda i, wt: f"Fix add() in mod.py so tests pass (rollout {i})",
        score_fn=make_pytest_score_fn(
            f"{sys.executable} -m pytest -q test_mod.py -p no:cacheprovider -o addopts=",
            env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        ),
        k=args.k,
    )
    engine.close()
    out = res.to_dict()
    out["validated"] = (not res.abstained) and res.winner is not None and res.winner.accepted
    _p(out)
    shutil.rmtree(work, ignore_errors=True)
    return 0 if out["validated"] else 1


def cmd_autosolve_demo(args: argparse.Namespace) -> int:
    """Free, deterministic demonstration of the GENERATED-CODE orchestration path
    (scout -> author -> freeze -> sandbox exec -> verified select), with fail-open
    to the best-of-N floor. Uses a synthetic repo + FakeExecutor (no paid calls)."""
    import subprocess
    import tempfile
    from apex_omega.engine.runtime import Engine
    from apex_omega.executor.fake import FakeExecutor
    from apex_omega.types import ExecResult, TokenUsage
    from apex_omega.workflows.best_of_n import WorkerSpec, make_pytest_score_fn
    from apex_omega.autogen import autosolve

    work = Path(tempfile.mkdtemp(prefix="apexomega_autosolve_"))
    repo = work / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("def add(a, b):\n    return a - b  # BUG\n")
    (repo / "test_mod.py").write_text("from mod import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.email=a@b.c", "-c", "user.name=apex", "commit", "-qm", "base"]):
        subprocess.run(c, cwd=repo, check=True, capture_output=True)

    def responder(task, session):
        Path(session.cwd, "mod.py").write_text("def add(a, b):\n    return a + b\n")
        return ExecResult(final_message="patched", usage=TokenUsage(input=5, output=5),
                          ok=True, finalization_status="completed")

    score = make_pytest_score_fn(f"{sys.executable} -m pytest -q test_mod.py -p no:cacheprovider -o addopts=",
                                 env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"})
    specs = [WorkerSpec("codex_cli", "gpt-5.5"), WorkerSpec("claude_cli", "opus")]
    pb = lambda ctx, i, strat: f"Fix add() in mod.py ({strat}, attempt {i})"

    def _run(name, **kw):
        eng = Engine(work / f"run_{name}", run_id=name, max_total_agents=64)
        r = autosolve(eng, source_repo=str(repo), executor=FakeExecutor(responder),
                      worker_specs=specs, score_fn=score, prompt_builder=pb, max_agents=20, **kw)
        eng.close()
        return r

    # custom authored script (lints clean) ; lint-reject (import) ; runtime-crash
    custom = ("def orchestrate(ctx):\n    ctx.phase('custom')\n"
              "    cands = [c for c in ctx.parallel([ctx.make_attempt(j) for j in range(3)]) if c]\n"
              "    return ctx.select(cands)\n")
    bad_import = "import os\ndef orchestrate(ctx):\n    return ctx.select([])\n"
    crashes = "def orchestrate(ctx):\n    raise RuntimeError('boom')\n"

    scen = {
        "template_floor": _run("tpl", author=False),
        "authored_ok": _run("auth", author=True, author_fn=lambda rm: custom),
        "lint_failopen": _run("bad", author=True, author_fn=lambda rm: bad_import),
        "runtime_failopen": _run("crash", author=True, author_fn=lambda rm: crashes),
    }
    out = {k: {"solved": v["solved"], "origin": v["orchestration"]["origin"],
               "agents_used": v["agents_used"], "winner": (v["winner"] or {}).get("vendor"),
               "error": v["error"]} for k, v in scen.items()}
    out["validated"] = all(v["solved"] for v in scen.values())
    _p(out)
    shutil.rmtree(work, ignore_errors=True)
    return 0 if out["validated"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="apex_omega", description="APEX-Ω vendor-neutral dynamic-workflow engine")
    sub = p.add_subparsers(dest="command", required=True)

    base_default = str(Path(__file__).resolve().parent.parent / "configs" / "base_commit0_local.json")

    d = sub.add_parser("doctor", help="cheap preflight (no paid calls)")
    d.add_argument("--base-config", default=base_default)
    d.add_argument("--probe-repo", default="voluptuous")
    d.set_defaults(func=cmd_doctor)

    a = sub.add_parser("arms", help="list the ablation matrix")
    a.set_defaults(func=cmd_arms)

    r = sub.add_parser("repos", help="list target repos")
    r.set_defaults(func=cmd_repos)

    e = sub.add_parser("eval", help="run the commit0 ablation matrix (Mode A)")
    e.add_argument("--arms", default="baseline")
    e.add_argument("--repos", default="")
    e.add_argument("--limit", type=int, default=1)
    e.add_argument("--rollouts", type=int, default=None)
    e.add_argument("--run-dir", default="runs/commit0_eval")
    e.add_argument("--base-config", default=base_default)
    e.add_argument("--local-only", action="store_true")
    e.add_argument("--budget", type=int, default=None)
    e.add_argument("--agent-mode", default="scaffolded")
    e.add_argument("--cell-timeout", type=int, default=7200)
    # Mode C (autogen_orchestrator arm) knobs; defaults are safe.
    e.add_argument("--autogen-max-agents", type=int, default=None,
                   help="soft agent cap for the autogen arm (default: difficulty-adaptive 8/24/64)")
    e.add_argument("--autogen-author", action="store_true",
                   help="let a planner author a tailored orchestrate(ctx) (paid); default off (template)")
    e.add_argument("--autogen-scout-agents", type=int, default=3,
                   help="parallel scout agents that set difficulty -> initial agent count (0 = static heuristic)")
    e.set_defaults(func=cmd_eval)

    b = sub.add_parser("bestofn-demo", help="free synthetic engine-native best-of-N demo")
    b.add_argument("--k", type=int, default=3)
    b.set_defaults(func=cmd_bestofn_demo)

    ag = sub.add_parser("autosolve-demo", help="free synthetic generated-code orchestration demo")
    ag.set_defaults(func=cmd_autosolve_demo)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
