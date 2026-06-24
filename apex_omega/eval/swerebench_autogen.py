"""Mode C for the SWE-rebench benchmark: run the GENERATED-CODE orchestrator on a
real SWE-rebench instance.

This is a near-verbatim sibling of ``commit0_autogen.run_autogen_cell`` that swaps
the runner + gold provider for the SWE-rebench ones, discovers by instance-id, and
DROPS the commit0-only datasets/pydantic-core preflight (the SWE-rebench runner is
self-contained and never imports ``datasets``).  It reuses the EXACT score_fn body
and ``verification_from_commit0_evaluation`` UNCHANGED — the SweRebench evaluation
object is a real ``Commit0Evaluation`` and is shape-compatible with scoring.py.

Gated: only reached when ``APEX_OMEGA_BENCHMARK=='swerebench'`` (the commit0 path
stays byte-identical when the selector is unset).
"""

from __future__ import annotations

import os
import subprocess
import traceback
from pathlib import Path
from typing import Optional

from ..engine.runtime import Engine
from .scoring import verification_from_commit0_evaluation
# Reuse the helpers from commit0_autogen that are benchmark-neutral (src-layout
# detection, worker-spec building, local-config forcing for vendor fleet).
from .commit0_autogen import (
    _detect_src_pkg,
    _resolve_pkg_origin,
    _is_within,
    _worker_specs_from_cfg,
    write_cell_report,
)
from . import swerebench_registry as _registry
from .swerebench_runner import SweRebenchRunner, gold_ids_for, TASK_FRAMING_BLOCK


def run_autogen_cell(
    engine: Engine,
    cfg_dict: dict,
    repo: str,
    *,
    apex_python: Optional[str] = None,
    apex_repo: Optional[str] = None,
    max_agents: Optional[int] = None,
    agent_ceiling: int = 1000,
    author: bool = False,
    author_vendor: Optional[str] = None,
    scout_agents: int = 3,
    scout_vendor: Optional[str] = None,
    output_dir: str,
    cell_timeout_seconds: int = 7200,
    fallback_rev: Optional[str] = None,
) -> dict:
    """Run Mode C (generated-code orchestrator) on ONE SWE-rebench instance.

    ``repo`` here is a SWE-rebench INSTANCE-ID (the ladder/cli pass it via
    --repos). Returns a cell-report dict compatible with the Commit0EvalDriver
    summary. NEVER raises — returns an ``_error`` dict on failure so the matrix
    keeps going.
    """
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["APEX_CELL_ROOT"] = str(out)

    def _err(msg: str, **extra) -> dict:
        rep = {
            "_report_path": None,
            "_returncode": -1,
            "total_tasks": 0,
            "solved_tasks": 0,
            "agents_used": engine.agents_used(),
            "average_pass_rate_percent": 0.0,
            "_mode": "autogen",
            "_benchmark": "swerebench",
            "_error": msg,
        }
        rep.update(extra)
        try:
            import json
            (out / "autogen_cell_error.json").write_text(
                json.dumps(rep, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass
        return rep

    # Docker-required instances cannot prep on a Docker-down host. Curated
    # SWE-rebench instances are Docker-free by construction, but keep the guard.
    try:
        if _registry.get(repo).forces_docker:
            return _err(f"instance {repo} requires Docker; skipped in Mode C")
    except KeyError:
        return _err(f"instance {repo} not in swerebench slice registry")
    except Exception:
        pass

    try:
        from ..autogen import autosolve
        from ..executor.auth_env import preflight_vendor_auth
        from ..executor.v1_executor import V1Executor

        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        # AUTH PREFLIGHT (no pydantic/datasets preflight — the runner is self-contained).
        _vendors = [str(e.get("backend") or "codex_cli")
                    for e in (cfg_dict.get("llm_configs") or [])] or ["codex_cli"]
        _auth = preflight_vendor_auth(_vendors)
        engine.log(f"auth preflight: {_auth}")

        # --- 1) build the self-contained SWE-rebench runner --------------------
        runner = SweRebenchRunner()

        # --- 2) discover the task by instance-id ------------------------------
        tasks = runner.discover_tasks(instance_ids=[repo], limit=1)
        if not tasks:
            return _err(f"discover_tasks({repo}) returned no tasks (not in slice)")
        task = tasks[0]

        # --- 3) PREP (clone@base -> apex-base -> uv venv -> install -> clean) ---
        repo_dir = out / "repo"
        runtime_dir = out / "runtime"
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        if runtime_dir.exists():
            import shutil as _shutil
            try:
                _shutil.rmtree(runtime_dir)
                engine.log("resume: cleared stale runtime/ before re-prep (venv-idempotent)")
            except OSError as exc:
                engine.log(f"resume: could not clear runtime/: {exc}")
        engine.phase(f"swerebench:prep:{repo}")
        env = runner._prepare_repo(task, repo_dir, runtime_dir)

        # --- 4) venv python ----------------------------------------------------
        venv_python = Path(env["VIRTUAL_ENV"]) / "bin" / "python"

        # --- 5) gold ids (the pinned provider; NEVER commit0.harness.get_pytest_ids)
        expected_ids = list(gold_ids_for(repo) or task.gold_ids or [])
        test_command = getattr(task, "test_cmd", None) or "pytest"
        repo_name = getattr(task, "repo_name", repo)

        def prompt_builder(ctx, attempt_id: int, strategy: str) -> str:
            issue = task.build_issue_description(
                test_command,
                expected_test_count=(len(expected_ids) or None),
                expected_test_ids=expected_ids or None,
            )
            approach = (ctx.repo_map.get("approach") or "").strip()
            key_files = ctx.repo_map.get("key_files") or []
            plan = ""
            if approach:
                plan += f"\n\nScout completion plan:\n{approach[:2000]}"
            if key_files:
                plan += f"\n\nLikely key files: {', '.join(map(str, key_files[:20]))}"
            hint = (
                f"\n\nStrategy for this attempt ({strategy}): implement the change "
                f"with a {strategy} approach; run the test command and iterate until "
                f"the gold tests are green."
            )
            return issue + plan + hint

        def _issue_and_plan(ctx) -> str:
            issue = task.build_issue_description(
                test_command,
                expected_test_count=(len(expected_ids) or None),
                expected_test_ids=expected_ids or None,
            )
            approach = (ctx.repo_map.get("approach") or "").strip()
            key_files = ctx.repo_map.get("key_files") or []
            plan = ""
            if approach:
                plan += f"\n\nScout completion plan:\n{approach[:2000]}"
            if key_files:
                plan += f"\n\nLikely key files: {', '.join(map(str, key_files[:20]))}"
            return issue + plan

        def module_solve_brief(ctx, module: str, module_gold_ids, *, carry_nonempty: bool) -> str:
            ids = "\n".join(map(str, (module_gold_ids or [])[:60])) or "(infer from the module's tests)"
            carry = ("\nFiles partially implemented by earlier agents are PRESENT in this "
                     "workspace — build ON them, do not revert.\n" if carry_nonempty else "")
            return (
                _issue_and_plan(ctx)
                + "\n\n--- MODULE-SCOPED SOLVE ---\n"
                + f"OBJECTIVE: implement ONLY the module `{module}`. Make EXACTLY these gold tests "
                + "pass — other modules are handled by parallel agents:\n" + ids + "\n"
                + "BOUNDARIES: edit only files belonging to this module; do NOT edit/add/delete any "
                + "test file; do not touch other modules.\n"
                + carry
                + "TOOL-GUIDANCE: run the scoped subset and iterate until that subset is green.\n"
            )

        def residual_repair_brief(ctx, failing_nodeids, passed: int, total: int, *,
                                  excerpts: str = "") -> str:
            ids = "\n".join(map(str, (failing_nodeids or [])[:40])) or "(see the failing subset)"
            evidence = (("\nFailure evidence:\n" + str(excerpts)[:3000] + "\n") if (excerpts or "").strip() else "")
            return (
                _issue_and_plan(ctx)
                + "\n\n--- RESIDUAL REPAIR (live merged tree) ---\n"
                + f"STATE: {int(passed)} of {int(total)} gold tests pass. The merged implementation "
                + "is ALREADY IN THIS WORKSPACE — keep what works.\n"
                + "These EXACT gold tests still FAIL; make them pass without breaking the rest:\n"
                + ids + "\n" + evidence
                + "INSTRUCTION: make the smallest correct change to turn these specific tests green. "
                + "Do NOT edit tests. Re-run only the failing subset and iterate.\n"
            )

        brief_builders = {"module_solve": module_solve_brief, "residual_repair": residual_repair_brief}

        # --- 6) score_fn: reuse the score_fn body + scoring.py UNCHANGED -------
        eval_cap = max(300, min(1800, int(cell_timeout_seconds) // 3))
        eval_counter = {"n": 0}

        def score_fn(worktree_path: str):
            eval_counter["n"] += 1
            label = f"swerebench_{repo}_{eval_counter['n']}"
            artifacts_dir = out / "evals" / label
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            wt = Path(worktree_path)
            call_env = dict(env)
            src_pkg = _detect_src_pkg(wt)
            if src_pkg is not None:
                prior = call_env.get("PYTHONPATH", "")
                call_env["PYTHONPATH"] = str(wt / "src") + (os.pathsep + prior if prior else "")
                origin = _resolve_pkg_origin(str(venv_python), src_pkg, call_env)
                if origin is None or not _is_within(origin, wt):
                    from ..kernel.verify import VerificationResult
                    engine.log(f"src-layout editable-shadow: {src_pkg} resolves to {origin}, "
                               f"not under worktree {wt} -> indeterminate")
                    return VerificationResult(
                        accepted=False, score=0.0, indeterminate=True,
                        reason=f"editable resolution outside worktree: {origin}",
                    )
            try:
                evaluation = runner.evaluate_repo(
                    task,
                    wt,
                    artifacts_dir=artifacts_dir,
                    label=label,
                    python_executable=str(venv_python),
                    env=call_env,
                    expected_test_ids=expected_ids or None,
                    use_expected_test_scoring=True,
                    timeout_seconds=eval_cap,
                )
            except Exception as exc:
                from ..kernel.verify import VerificationResult
                engine.log(f"score_fn evaluate_repo raised: {type(exc).__name__}: {exc}")
                return VerificationResult(
                    accepted=False, score=0.0, reason=f"evaluate_repo failed: {exc}",
                    indeterminate=True,
                )
            return verification_from_commit0_evaluation(
                evaluation, expected_test_count=len(expected_ids))

        worker_specs = _worker_specs_from_cfg(cfg_dict)

        from ..autogen.architect import build_repo_map

        scout_extra = {
            "repo": repo, "test_command": test_command,
            "expected_test_count": len(expected_ids),
            "issue_description": (task.build_issue_description(test_command)
                                  if hasattr(task, "build_issue_description") else "")[:2000],
        }
        repo_map = build_repo_map(str(repo_dir), base_commit="apex-base", extra=dict(scout_extra))
        repo_map["task_framing"] = TASK_FRAMING_BLOCK
        repo_map["brief_builders"] = brief_builders

        engine.log(f"swerebench cell instance={repo} workers={[s.vendor for s in worker_specs]} "
                   f"author={author} scout_agents={scout_agents} ceiling={agent_ceiling}")

        engine.phase(f"swerebench:solve:{repo}")
        repair_iters = int(os.environ.get("APEX_OMEGA_REPAIR_ITERS", "2") or 2)
        from ..journal.key import sha256_hex
        expected_ids_sha = sha256_hex("\n".join(sorted(expected_ids))) if expected_ids else ""
        scoring_env_sha = sha256_hex(str(venv_python) + "|" + str(eval_cap))
        result = autosolve(
            engine,
            source_repo=str(repo_dir),
            base_commit="apex-base",
            executor=V1Executor(),
            worker_specs=worker_specs,
            score_fn=score_fn,
            prompt_builder=prompt_builder,
            repo_map=repo_map,
            scout_extra=scout_extra,
            scout_agents=scout_agents,
            scout_vendor=scout_vendor,
            author=author,
            author_vendor=author_vendor,
            max_agents=max_agents,
            agent_ceiling=agent_ceiling,
            run_scope=f"swerebench_{repo}",
            timeout_seconds=cell_timeout_seconds,
            repair_iters=repair_iters,
            expected_ids_sha=expected_ids_sha,
            scoring_env_sha=scoring_env_sha,
        )
        difficulty = result.get("difficulty")
        winner = result.get("winner") or {}
        solved = bool(result.get("solved"))
        pass_rate_pct = 100.0 if solved else 0.0
        return {
            "_report_path": str(out / "autogen_cell_report.json"),
            "_returncode": 0,
            "_mode": "autogen",
            "_benchmark": "swerebench",
            "total_tasks": 1,
            "solved_tasks": 1 if solved else 0,
            "agents_used": int(result.get("agents_used", engine.agents_used()) or 0),
            "average_pass_rate_percent": pass_rate_pct,
            "difficulty": difficulty,
            "agent_budget": result.get("agent_budget"),
            "scout": result.get("scout"),
            "agent_ceiling": agent_ceiling,
            "solved": solved,
            "abstained": bool(result.get("abstained")),
            "cut_losses": result.get("cut_losses"),
            "orchestration": result.get("orchestration"),
            "budget": result.get("budget"),
            "winner": winner or None,
            "_orchestration_error": result.get("error"),
            "_repo": repo,
            "_stratum": getattr(task, "stratum", ""),
        }
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}", _traceback=traceback.format_exc()[-2000:])
