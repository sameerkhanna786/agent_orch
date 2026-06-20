"""Mode C: run the GENERATED-CODE orchestrator on a real commit0 repo.

This is the third commit0 evaluation mode:

  * Mode A (``commit0_driver._invoke_runner``): v1's ``Commit0BenchmarkRunner``
    drives the whole solve in a subprocess (v1-as-worker).
  * Mode B (``workflows/best_of_n``): the engine-native fixed best-of-N loop.
  * Mode C (HERE): the APEX-Ω GENERATED-CODE orchestrator
    (``apex_omega.autogen.autosolve``) plans + authors + freezes a tailored
    ``orchestrate(ctx)`` and runs it in-process on the engine — reusing v1's
    PROVEN repo prep (clone + apex-base + uv venv + editable install) and v1's
    execution-authoritative scoring (``evaluate_repo``) verbatim.  We never
    re-implement the prep or the acceptance gate — that would risk diverging from
    the only publishable number (plan §20.1) and weaken the Cardinal gate.

The recipe (from recon, validated against
``apex/apex/apex/evaluation/commit0_benchmark.py``):

  1. Build the v1 ``Commit0BenchmarkRunner`` (force LOCAL no-Docker scoring).
  2. ``task = runner.discover_tasks(repos=[repo], limit=1)[0]``.
  3. ONE call ``env = runner._prepare_repo(task, repo_dir, runtime_dir)`` does
     EVERYTHING: clone -> ``git checkout -B apex-base <base_commit>`` -> history
     scrub (preserving apex-base) -> ``_build_runtime_env`` (uv venv + sandbox
     HOME/TMP) -> pre_install -> packages -> pip_packages -> test deps -> the
     editable ``uv pip install -e .`` -> ``git reset --hard`` + ``git clean -fdx``.
     Returns the env dict.  We do NOT call ``_build_runtime_env`` ourselves and we
     run NO separate dep-install step.
  4. ``venv_python = Path(env['VIRTUAL_ENV']) / 'bin' / 'python'``.
  5. ``expected_ids = _load_expected_test_ids(repo)``;
     ``test_command = task.test_cmd`` (metadata; evaluate_repo builds the real one).
  6. ``score_fn(worktree)`` forks each candidate from ``apex-base`` (handled by the
     autosolve WorktreeProvider) and scores its EDITED code via
     ``runner.evaluate_repo(task, worktree, python_executable=venv_python, env=env,
     expected_test_ids=expected_ids, use_expected_test_scoring=True)`` then maps to
     an APEX-Ω VerificationResult — the worktree (not base repo_dir) is scored.

NEVER raises: any failure returns an ``_error`` cell dict so the matrix continues.
"""

from __future__ import annotations

import os
import subprocess
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from ..engine.runtime import Engine
from .scoring import verification_from_commit0_evaluation


# --- P0.1: per-worktree editable resolution (src-layout false-zero fix) --------
# v1's repo prep does ``pip install -e .`` against the BASE repo_dir, pinning the
# editable importer at base/src/<pkg>.  ``score_fn`` runs pytest in the candidate
# WORKTREE but reuses that base env, so for src-layout repos ``import <pkg>``
# resolves to the base STUB and the gate scores correct candidate code as ZERO
# (this is the jinja failure: correct code, 851 collection errors / 0 pass).
# Flat-layout repos resolve via cwd and are unaffected (verified-green path).
def _detect_src_pkg(worktree: Path) -> Optional[str]:
    """Return the import package name for a src-layout repo
    (``<worktree>/src/<pkg>/__init__.py``); None for flat-layout (P0.1 no-op)."""
    src = worktree / "src"
    if not src.is_dir():
        return None
    for child in sorted(src.iterdir()):
        if child.is_dir() and (child / "__init__.py").exists():
            return child.name
    return None


def _resolve_pkg_origin(venv_python: str, pkg: str, env: dict) -> Optional[str]:
    """Where ``import pkg`` WOULD load from, WITHOUT executing the module body
    (importlib.util.find_spec), under the given interpreter + env.  Returns the
    origin path or None.  find_spec (not import) so a candidate whose code is
    broken still reports WHERE it would import from."""
    code = ("import importlib.util as u, sys;"
            "s = u.find_spec(sys.argv[1]);"
            "sys.stdout.write((s.origin or '') if s else '')")
    try:
        r = subprocess.run([venv_python, "-c", code, pkg], env=env,
                           capture_output=True, text=True, timeout=60)
        return (r.stdout or "").strip() or None
    except Exception:
        return None


def _is_within(path_str: str, root: Path) -> bool:
    try:
        Path(path_str).resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


# Soft, difficulty-adaptive agent budgets (fewest agents first; the doubling
# wave schedule escalates toward these only while unsolved).  Never exceeds the
# hard ``agent_ceiling`` (1000) backstop wired into the Engine.
_DIFFICULTY_MAX_AGENTS = {"easy": 8, "medium": 24, "hard": 64}
_DEFAULT_DIFFICULTY = "medium"


def difficulty_to_max_agents(difficulty: Optional[str], *, ceiling: int = 1000) -> int:
    """Map a repo-map difficulty bucket to a soft ``max_agents`` cap (clamped to
    the hard ceiling).  Unknown/None difficulty -> the medium default."""
    soft = _DIFFICULTY_MAX_AGENTS.get(str(difficulty or "").lower(), _DIFFICULTY_MAX_AGENTS[_DEFAULT_DIFFICULTY])
    return max(1, min(int(soft), int(ceiling)))


# NOTE: the gold-test-guided "design contract" (apex_omega/eval/design_contract.py) was REMOVED
# from the evaluated path (fairness directive 2026-06-16). The harness must provide the
# orchestrator/agents ONLY the original commit0 prompt + the gold test suite; deriving the enum/
# API/parametrization shape FROM the gold tests is the model's/orchestrator's job, not the
# harness's. The module is quarantined (unreferenced here); see its docstring.


def _force_local_config_dict(cfg_dict: dict) -> dict:
    """Force the host-local no-Docker scoring path onto a config dict so a dev mac
    (Docker down) scores cleanly via local_pytest_json_report. Also pins the GOLD
    evaluation contract and asserts gold expected-id scoring is REQUIRED (same gate as
    the v1 baseline arms) so the autogen path can never fall through to visible-suite
    (pytest_summary) acceptance."""
    from ..ablation.arms import deep_merge
    from .commit0_driver import pin_gold_scoring_contract

    local = deep_merge(cfg_dict, {
        "benchmark": {
            "commit0_primary_evaluation_backend": "local_pytest_json_report",
            "commit0_official_audit_selected": False,
            "commit0_docker_fallback_on_failure": False,
            "commit0_docker_runtime_mode": "never",
        }
    })
    return pin_gold_scoring_contract(local)


def _worker_specs_from_cfg(cfg_dict: dict):
    """Build APEX-Ω WorkerSpecs from the v1 ``llm_configs`` so Mode C runs the same
    vendor fleet (codex / claude / mixed) the arm declares."""
    from ..workflows.best_of_n import WorkerSpec

    specs = []
    for entry in (cfg_dict.get("llm_configs") or []):
        if not isinstance(entry, dict):
            continue
        backend = entry.get("backend") or "codex_cli"
        model = entry.get("model") or "gpt-5.5"
        extra = {k: v for k, v in entry.items()
                 if k not in ("backend", "model") and v is not None}
        specs.append(WorkerSpec(vendor=backend, model=model, extra=extra))
    if not specs:
        specs = [WorkerSpec(vendor="codex_cli", model="gpt-5.5")]
    return specs


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
    """Run Mode C (generated-code orchestrator) on ONE real commit0 repo.

    Reuses v1's repo prep (``_prepare_repo``) + scoring (``evaluate_repo``)
    verbatim and routes the solve through ``apex_omega.autogen.autosolve`` on the
    in-process engine (journaled, budget-accounted, resume-survivable).

    Returns a cell-report dict whose keys are compatible with the
    ``Commit0EvalDriver`` summary (``solved_tasks``, ``total_tasks``,
    ``agents_used``, ``average_pass_rate_percent``) plus orchestration metadata +
    the winner.  NEVER raises — on any failure it returns an ``_error`` dict (with
    ``_report_path`` set) so the ablation matrix keeps going.
    """
    # ABSOLUTE: v1 _prepare_repo runs `uv venv` with cwd inside the repo, so a RELATIVE
    # runtime_dir resolves against the repo dir -> the venv is created at a doubled/nested path
    # and the env smoke test can't find runtime/.venv/bin/python3 ("interpreter not found").
    # Resolving here makes repo_dir/runtime_dir absolute so the venv lands where expected,
    # regardless of whether the caller passed a relative --run-dir / LADDER_DIR.
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Cell-scoped workspace-guard severity (FM-1): export THIS cell's root so the CLI policy guard
    # treats read-only discovery anywhere under it (repo / runtime / sibling-module worktrees) as a
    # SOFT course-correction rather than a fatal abort. Other cells + planted upstream copies stay
    # FATAL. Set in this cell process's env; the in-process guard reads it via _agent_runtime_infra_roots.
    os.environ["APEX_CELL_ROOT"] = str(out)

    def _err(msg: str, **extra) -> dict:
        # Mark a NON-result so _cell_status classifies it RESULT_INFRA_NONRESULT
        # (not a cached OK) — fixing the env and re-running must retry the cell.
        rep = {
            "_report_path": None,
            "_returncode": -1,
            "total_tasks": 0,
            "solved_tasks": 0,
            "agents_used": engine.agents_used(),
            "average_pass_rate_percent": 0.0,
            "_mode": "autogen",
            "_error": msg,
        }
        rep.update(extra)
        try:  # keep a debug artifact (not the classification _report_path, which stays None)
            import json
            (out / "autogen_cell_error.json").write_text(
                json.dumps(rep, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass
        return rep

    # Docker-required repos (apt-get pre_install) cannot prep locally on a
    # Docker-down host — skip cleanly as a non-result rather than crash in prep.
    try:
        from . import registry as _registry
        if _registry.get(repo).forces_docker:
            return _err(f"repo {repo} requires Docker (apt-get pre_install); skipped in Mode C")
    except Exception:
        pass

    try:
        # --- lazy v1 imports (keep the engine importable without the apex venv) ---
        from apex.core.config import ApexConfig
        from apex.evaluation.commit0_benchmark import (
            Commit0BenchmarkRunner,
            TASK_FRAMING_BLOCK,
            _load_expected_test_ids,
        )
        from ..autogen import autosolve
        from ..executor.auth_env import preflight_vendor_auth
        from ..executor.v1_executor import V1Executor

        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        # --- 0) AUTH PREFLIGHT: refresh + validate vendor auth BEFORE prep/agents, so a stale
        # gateway token never silently burns the whole cell on 0-token infra_nonresult results
        # (it warms the host auth the sandboxed rollout agents reuse). Fails loud if auth is dead.
        _vendors = [str(e.get("backend") or "codex_cli")
                    for e in (cfg_dict.get("llm_configs") or [])] or ["codex_cli"]
        _auth = preflight_vendor_auth(_vendors)
        engine.log(f"auth preflight: {_auth}")

        # --- 1) build the v1 runner (force LOCAL no-Docker scoring) -------------
        local_cfg = _force_local_config_dict(cfg_dict)
        config = ApexConfig.from_dict(local_cfg)
        fallback_revs = [fallback_rev] if fallback_rev else None
        runner = Commit0BenchmarkRunner(
            config=config,
            output_dir=str(out / "v1_runner"),
            dataset_split="test",
            dataset_fallback_revisions=fallback_revs,
            split=repo,
        )

        # --- 2) discover the task ---------------------------------------------
        tasks = runner.discover_tasks(repos=[repo], limit=1)
        if not tasks:
            return _err(f"discover_tasks({repo}) returned no tasks")
        task = tasks[0]

        # --- 3) PREP per the recon recipe: one _prepare_repo call does it all ---
        # (clone + apex-base + scrub + uv venv + ALL installs incl. editable -e .)
        repo_dir = out / "repo"
        runtime_dir = out / "runtime"
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        # RESUME-IDEMPOTENT PREP: a RELAUNCHED cell re-enters here with a runtime venv left
        # by the prior launch; `uv venv` then aborts ("A virtual environment already exists"),
        # which crashed every relaunched cell at _prepare_repo and made it report total=0 /
        # not_run despite having banked real attempts (observed live on the mimesis cells).
        # The runtime (venv + sandbox HOME/TMP) is fully rebuildable; the durable journal in
        # the run dir carries resume. So clear it and let prep rebuild cleanly.
        if runtime_dir.exists():
            import shutil as _shutil
            try:
                _shutil.rmtree(runtime_dir)
                engine.log("resume: cleared stale runtime/ before re-prep (venv-idempotent)")
            except OSError as exc:
                engine.log(f"resume: could not clear runtime/: {exc}")
        engine.phase(f"autogen:prep:{repo}")
        env = runner._prepare_repo(task, repo_dir, runtime_dir)

        # --- 4) venv python (verbatim from commit0_benchmark.py:13474) ---------
        venv_python = Path(env["VIRTUAL_ENV"]) / "bin" / "python"

        # --- 5) expected ids + test command (metadata) -------------------------
        expected_ids = list(_load_expected_test_ids(repo) or [])
        test_command = getattr(task, "test_cmd", None) or "pytest"

        # --- prompt builder: v1's issue description + a short strategy hint -----
        repo_name = getattr(task, "repo_name", repo)

        # The worker prompt is ONLY: (1) the original commit0 prompt (v1 build_issue_description,
        # which legitimately surfaces the gold test suite — visible tests + expected-id inventory)
        # plus (2) the orchestrator's OWN runtime reasoning (the scout's approach/key_files and the
        # strategy hint). NO harness-derived design contract — figuring out the API/enum/param
        # shape from the gold tests is the model's/orchestrator's job (fairness directive).
        def prompt_builder(ctx, attempt_id: int, strategy: str) -> str:
            issue = task.build_issue_description(
                test_command,
                expected_test_count=(len(expected_ids) or None),
                expected_test_ids=expected_ids or None,
            )
            # Inject the scout's completion plan (set on ctx.repo_map by the agent-scout fan-out)
            # so workers act on the discovered approach — this is the orchestrator's own scouting.
            approach = (ctx.repo_map.get("approach") or "").strip()
            key_files = ctx.repo_map.get("key_files") or []
            plan = ""
            if approach:
                plan += f"\n\nScout completion plan:\n{approach[:2000]}"
            if key_files:
                plan += f"\n\nLikely key files: {', '.join(map(str, key_files[:20]))}"
            hint = (
                f"\n\nStrategy for this attempt ({strategy}): implement the missing "
                f"functionality with a {strategy} approach; run the test command and "
                f"iterate until the visible suite is green."
            )
            return issue + plan + hint

        # --- two convergence prompt CONTRACTS (share the issue/expected-ids closure) ----------
        # Both PREFIX the verbatim issue description (fairness firewall: TASK_FRAMING_BLOCK + gold
        # inventory must stay intact; the briefs add SCOPE + FEEDBACK, never answers). They are
        # passed via the prompt= override on the convergence seams (solve_module / repair_residual);
        # the convergence default uses the seams' built-in briefs by default, but these expose the
        # eval's richer issue text + scout plan for the as-run eval path.
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
            """CONTRACT 1 — module-scoped solve (delegation: objective / output / boundaries /
            tool-guidance). Scopes ONE agent to a single module + its gold subset."""
            ids = "\n".join(map(str, (module_gold_ids or [])[:60])) or "(infer from the module's tests)"
            carry = ("\nFiles partially implemented by earlier agents are PRESENT in this "
                     "workspace — build ON them, do not revert.\n" if carry_nonempty else "")
            return (
                _issue_and_plan(ctx)
                + "\n\n--- MODULE-SCOPED SOLVE ---\n"
                + f"OBJECTIVE: implement ONLY the module `{module}`. Make EXACTLY these gold tests "
                + "pass — other modules are handled by parallel agents, so do NOT reimplement the "
                + "whole repo:\n" + ids + "\n"
                + "BOUNDARIES: edit only files belonging to this module; do NOT edit/add/delete any "
                + "test file; do not touch other modules (note any genuinely-missing shared symbol "
                + "for the reducer instead of forking it).\n"
                + carry
                + "TOOL-GUIDANCE: run the scoped subset and iterate until that subset is green.\n"
            )

        def residual_repair_brief(ctx, failing_nodeids, passed: int, total: int, *,
                                  excerpts: str = "") -> str:
            """CONTRACT 2 — residual repair on the LIVE merged tree, scoped to the exact still-
            failing node-ids. Excerpts (real assertion tails) injected when provided."""
            ids = "\n".join(map(str, (failing_nodeids or [])[:40])) or "(see the failing subset)"
            # FM-4: bound the repair context. The carry tree is APPLIED to the worktree (not pasted),
            # ids are capped, and the assertion tail is hard-capped here so a large-suite repair turn
            # can never blow the context/time budget on a multi-thousand-test repo.
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

        # Expose the two contracts to the convergence seams (solve_module / repair_residual read
        # repo_map['brief_builders'] when present, else fall back to their built-in briefs). This is
        # the live wiring: the eval's richer issue text + scout plan flow into the module/residual
        # briefs. The key is non-JSON (functions); build_author_prompt excludes it from the author
        # repo-map dump.
        brief_builders = {"module_solve": module_solve_brief, "residual_repair": residual_repair_brief}

        # --- 6) score_fn: reuse v1 evaluate_repo on the candidate WORKTREE ------
        # (NEVER reinvent the gate — this is the Cardinal contract source.)
        # Budget-aware per-eval timeout: cap one pytest scoring run so a single slow
        # candidate cannot eat the whole cell wall-clock (run-4 jinja: one eval hit the
        # fixed 1800s cap and consumed half the 3600s cell). A scoring timeout maps to
        # indeterminate (excluded), so capping is safe.
        eval_cap = max(300, min(1800, int(cell_timeout_seconds) // 3))
        eval_counter = {"n": 0}

        def score_fn(worktree_path: str):
            eval_counter["n"] += 1
            label = f"autogen_{repo}_{eval_counter['n']}"
            artifacts_dir = out / "evals" / label
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            # P0.1: build a PER-CALL env (never mutate the shared `env` -> race-free
            # across the concurrent fan). For src-layout repos prepend <worktree>/src
            # to PYTHONPATH so the candidate's edits win over the base editable stub.
            wt = Path(worktree_path)
            call_env = dict(env)
            src_pkg = _detect_src_pkg(wt)
            if src_pkg is not None:
                prior = call_env.get("PYTHONPATH", "")
                call_env["PYTHONPATH"] = str(wt / "src") + (os.pathsep + prior if prior else "")
                origin = _resolve_pkg_origin(str(venv_python), src_pkg, call_env)
                if origin is None or not _is_within(origin, wt):
                    # An editable *finder* (PEP 660) still shadows the worktree, or the
                    # package won't resolve: return INDETERMINATE (excluded), never a
                    # false-zero scored as a real failure and never a false-accept.
                    from ..kernel.verify import VerificationResult
                    engine.log(f"P0.1 editable-shadow: {src_pkg} resolves to {origin}, "
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
            except Exception as exc:  # scoring infra failure -> indeterminate, not accepted
                from ..kernel.verify import VerificationResult
                engine.log(f"score_fn evaluate_repo raised: {type(exc).__name__}: {exc}")
                return VerificationResult(
                    accepted=False, score=0.0, reason=f"evaluate_repo failed: {exc}",
                    indeterminate=True,
                )
            # FM-3: pass the AUTHORITATIVE static gold-id count so a collapsed-collection denominator
            # (babel gold_total=10) can never falsely accept and the frontier sees the true distance.
            return verification_from_commit0_evaluation(
                evaluation, expected_test_count=len(expected_ids))

        # --- worker fleet from the arm's llm_configs ---------------------------
        worker_specs = _worker_specs_from_cfg(cfg_dict)

        # --- base repo map for the scout (the agent-scout REFINES difficulty) ---
        from ..autogen.architect import build_repo_map

        scout_extra = {
            "repo": repo, "test_command": test_command,
            "expected_test_count": len(expected_ids),
            "issue_description": (task.build_issue_description(test_command)
                                  if hasattr(task, "build_issue_description") else "")[:2000],
        }
        repo_map = build_repo_map(str(repo_dir), base_commit="apex-base", extra=dict(scout_extra))
        # (No design-contract derivation — removed for fairness; the agent figures the API/enum
        # shape out from the gold test suite itself. build_repo_map above is allowed scouting.)
        # Surface the BINDING TASK-FRAMING rules to the orchestrator + scout (single source of
        # truth = the v1 constant; workers ALSO get it unconditionally via build_issue_description).
        # The orchestrator may restate/amplify it to subagents but can never remove it.
        repo_map["task_framing"] = TASK_FRAMING_BLOCK
        # CONVERGENCE prompt contracts (live wiring): the seams pick these up via ctx.repo_map.
        repo_map["brief_builders"] = brief_builders

        engine.log(f"autogen cell repo={repo} workers={[s.vendor for s in worker_specs]} "
                   f"author={author} scout_agents={scout_agents} ceiling={agent_ceiling}")

        # --- run Mode C: scout fan-out (difficulty -> initial agents) -> author ->
        # ---            freeze -> sandbox exec -> verified select ----------------
        engine.phase(f"autogen:solve:{repo}")
        # Test-driven repair depth ceiling. Default is now 2 (ON): the orchestrator-level
        # iterate-to-convergence loop is the dynamic-workflow signature move and is SAFE by
        # default because the SPFG+ governor stops a true plateau (the run-4 budget-blowup
        # fix). Set APEX_OMEGA_REPAIR_ITERS=0 to force the old flat best-of-N behaviour.
        repair_iters = int(os.environ.get("APEX_OMEGA_REPAIR_ITERS", "2") or 2)
        # review-fix #13: content-bearing score drift keys (were documented but never set).
        from ..journal.key import sha256_hex
        expected_ids_sha = sha256_hex("\n".join(sorted(expected_ids))) if expected_ids else ""
        scoring_env_sha = sha256_hex(str(venv_python) + "|" + str(eval_cap))
        # NOTE on network: an agent was observed pip-downloading the upstream package to /tmp to
        # copy in. That cheat is BLOCKED UNIFORMLY for every arm by the v1 workspace-jail (denies
        # copying outside-workspace paths in) + the worktree-shadow (a fetched package can't be
        # imported over the candidate), and every attempt is recorded in integrity_log.jsonl. We
        # deliberately do NOT add a per-arm offline block here: it would only affect the omega
        # path (baselines prep+solve in a separate v1 subprocess), creating an unfair asymmetry,
        # and the runner-level block would break prep's dependency installs. True network isolation
        # needs an OS sandbox (unavailable here; the agent CLI also needs the network for the model
        # API), so cheats are prevented structurally + symmetrically rather than by egress blocking.
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
            max_agents=max_agents,        # None -> scout difficulty drives the cap
            agent_ceiling=agent_ceiling,
            run_scope=f"autogen_{repo}",
            timeout_seconds=cell_timeout_seconds,
            repair_iters=repair_iters,
            expected_ids_sha=expected_ids_sha,
            scoring_env_sha=scoring_env_sha,
        )
        difficulty = result.get("difficulty")

        winner = result.get("winner") or {}
        solved = bool(result.get("solved"))
        # Pull the winner's pass-rate (execution-authoritative) for the summary.
        pass_rate_pct = 100.0 if solved else 0.0
        return {
            "_report_path": str(out / "autogen_cell_report.json"),
            "_returncode": 0,
            "_mode": "autogen",
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
            # CUT-LOSSES: surface the governor's non-progress cut reason (if any) so the
            # reclassifier/ledger books a cut as a diagnosable FAILURE, not infra/timeout.
            "cut_losses": result.get("cut_losses"),
            "orchestration": result.get("orchestration"),
            "budget": result.get("budget"),
            "winner": winner or None,
            "_orchestration_error": result.get("error"),
            "_repo": repo,
        }
    except Exception as exc:  # never raise out of a cell (matrix must continue)
        return _err(f"{type(exc).__name__}: {exc}", _traceback=traceback.format_exc()[-2000:])


def write_cell_report(report: dict, output_dir: str) -> dict:
    """Persist a successful Mode C cell report to disk (so the journal status_fn
    treats it as a valid cache hit)."""
    import json

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = report.get("_report_path") or str(out / "autogen_cell_report.json")
    try:
        Path(path).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        report["_report_path"] = path
    except Exception:
        pass
    return report
