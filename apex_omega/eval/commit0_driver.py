"""Commit0 evaluation + ablation driver (Mode A: v1-as-worker).

For each (arm, repo) cell it deep-merges the arm's ``v1_overlay`` onto a base
ApexConfig, forces the local no-Docker scoring path, and runs v1's proven
``Commit0BenchmarkRunner`` — reusing ALL of v1's repo prep + solve + the
execution-authoritative scoring (the only publishable number, §20.1).  The
APEX-Ω engine wraps each cell as a journaled step so the matrix is
resume-survivable and cost-accounted, and so adding/repeating arms is cheap.

This is the workhorse that makes "test evaluation and ablation over the 15
repos" runnable today; the engine-native best-of-N workflow (workflows/) is the
Phase-0 demonstration of the new primitives.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from ..ablation.arms import ARMS, AblationArm, build_ablation_config, deep_merge, get_arm
from ..engine.budget import Budget
from ..engine.runtime import Engine
from ..journal.resume import resume_or_run_json
from ..journal.wal import RESULT_INFRA_NONRESULT, RESULT_OK
from . import registry


# APEX-Ω is self-contained: the apex kernel SOURCE is vendored at <repo>/apex, so
# Mode A's subprocess runs `python -m apex` against the VENDORED copy (cwd + PYTHONPATH
# point at the agent_orch repo root). apex_python only supplies the third-party deps
# (datasets/commit0/tree-sitter/...); override it to your venv if different.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])          # <agent_orch>
DEFAULT_APEX_REPO = _REPO_ROOT
DEFAULT_APEX_PYTHON = os.environ.get("APEX_OMEGA_PYTHON") or sys.executable


# Settings forced onto every arm so a dev mac (Docker daemon down) stays clean.
_LOCAL_NO_DOCKER = {
    "benchmark": {
        "commit0_primary_evaluation_backend": "local_pytest_json_report",
        "commit0_official_audit_selected": False,
        "commit0_docker_fallback_on_failure": False,
        "commit0_docker_runtime_mode": "never",
    }
}


def _ensure_hf_offline() -> None:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Provision Meta-gateway host-mode vendor auth (setdefault) so Mode A's
    # subprocess (env=dict(os.environ)) and Mode C in-process workers authenticate.
    from ..executor.auth_env import ensure_vendor_auth_env
    ensure_vendor_auth_env()


def load_base_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pin_gold_scoring_contract(cfg: dict) -> dict:
    """Pin the commit0 GOLD evaluation contract onto a config dict and ASSERT the
    expected-id-scoring gate is satisfied. This guarantees every arm (v1 baselines +
    autogen) scores by exact gold expected-test-id match and can NEVER fall through to
    the ``pytest_summary`` visible-suite acceptance path (which would accept a green raw
    suite without verifying any gold-id match). An empty/failed expected-id inventory
    then becomes a HARNESS FAILURE (indeterminate, re-run), not a silent false solve.
    Raises if, after pinning, the gate is still not satisfied (mis-merged config)."""
    from apex.core.config import ApexConfig
    from apex.evaluation.commit0_benchmark import (
        COMMIT0_GOLD_EVALUATION_CONTRACT,
        _commit0_expected_id_scoring_required,
    )

    cfg = deep_merge(cfg, {"benchmark": {"evaluation_contract":
                                         dict(COMMIT0_GOLD_EVALUATION_CONTRACT)}})
    if not _commit0_expected_id_scoring_required(ApexConfig.from_dict(cfg)):
        raise RuntimeError(
            "commit0 gold scoring is REQUIRED but the resolved evaluation contract does "
            "not require expected-test-id scoring after pinning the gold contract; refusing "
            "to run with a visible-suite (pytest_summary) acceptance fallback. "
            f"resolved contract={ApexConfig.from_dict(cfg).benchmark.resolved_evaluation_contract_config('commit0')}")
    return cfg


def build_arm_config_dict(base: dict, arm: AblationArm, *, force_local: bool = True,
                          rollouts: Optional[int] = None) -> dict:
    cfg = deep_merge(base, arm.v1_overlay or {})
    if force_local:
        cfg = deep_merge(cfg, _LOCAL_NO_DOCKER)
    if rollouts is not None:
        cfg = deep_merge(cfg, {"rollout": {"num_rollouts": rollouts, "min_rollouts": 1,
                                           "max_rollouts": rollouts}})
    # GOLD SCORING REQUIRED for every arm (v1 baselines + autogen), symmetric + non-negotiable.
    cfg = pin_gold_scoring_contract(cfg)
    return cfg


class Commit0EvalDriver:
    def __init__(self, run_dir: str | Path, base_config: dict, *, budget: Optional[Budget] = None,
                 engine: Optional[Engine] = None, force_local: bool = True,
                 apex_python: str = DEFAULT_APEX_PYTHON, apex_repo: str = DEFAULT_APEX_REPO,
                 agent_mode: str = "scaffolded", cell_timeout_seconds: int = 7200,
                 autogen_max_agents: Optional[int] = None, autogen_author: bool = False,
                 autogen_agent_ceiling: int = 1000, autogen_scout_agents: int = 3):
        _ensure_hf_offline()
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.base_config = base_config
        self.force_local = force_local
        self.apex_python = apex_python
        self.apex_repo = apex_repo
        self.agent_mode = agent_mode
        self.cell_timeout_seconds = cell_timeout_seconds
        # Mode C (generated-code orchestrator) knobs; defaults are safe (no author
        # call, difficulty-adaptive agent budget, hard 1000 backstop).
        self.autogen_max_agents = autogen_max_agents
        self.autogen_author = autogen_author
        self.autogen_agent_ceiling = autogen_agent_ceiling
        self.autogen_scout_agents = autogen_scout_agents
        # The shared engine carries the hard 1000-agent backstop (max_total_agents)
        # and a sane in-process concurrency (min(16, cpu-2)).  Mode A cells run via
        # subprocess (they don't touch engine.agent()); Mode C cells journal their
        # agent() calls on this engine and are bounded by the backstop + soft caps.
        sane_concurrent = max(1, min(16, (os.cpu_count() or 4) - 2))
        # Backbone 0.1: the DEFAULT IS UNBOUNDED (no token budget) — "never optimize for
        # cost" (Budget's documented invariant + the dynamic-workflow model). The token
        # budget is strictly OPT-IN via APEX_OMEGA_TOKEN_CEILING; the always-on runaway
        # guard is the per-RUN agent backstop (max_total_agents) + the resumable journal,
        # NOT a default cost ceiling. can_start() only gates STARTING new work.
        _env = os.environ.get("APEX_OMEGA_TOKEN_CEILING")
        _ceiling = int(_env) if (_env and _env.strip()) else None
        if budget is not None:
            _budget = budget
        elif _ceiling is not None:
            _budget = Budget(total=_ceiling)
        else:
            _budget = Budget()                          # total=None -> unbounded (default)
        self.engine = engine or Engine(self.run_dir, run_id="commit0_eval",
                                       budget=_budget,
                                       max_total_agents=autogen_agent_ceiling,
                                       max_concurrent=sane_concurrent)

    # -- one (arm, repo) cell -------------------------------------------
    def run_cell(self, arm: AblationArm, repo: str, *, limit: int = 1, rollouts: Optional[int] = None,
                 output_dir: Optional[str] = None) -> dict:
        spec = registry.get(repo)
        cfg_dict = build_arm_config_dict(self.base_config, arm, force_local=self.force_local, rollouts=rollouts)
        cell_out = Path(output_dir or (self.run_dir / "cells" / f"{arm.id}__{repo}"))
        cell_out.mkdir(parents=True, exist_ok=True)
        # Detect the generated-code orchestrator arm (Mode C): the AblationConfig
        # for this arm flips ``orchestrator`` to "autogen" (e.g. arm id
        # "autogen_orchestrator").  Route in-process through run_autogen_cell.
        is_autogen = build_ablation_config(arm).orchestrator == "autogen"
        # The journal key must capture EVERYTHING that determines the result: the
        # config hash AND the result-determining harness identity (agent_mode,
        # which v1 checkout/venv) — else a changed solve surface returns a stale hit.
        cfg_hash = json.dumps(cfg_dict, sort_keys=True)
        scoped_inputs = {
            "config": cfg_hash,
            "fallback_rev": spec.dataset_fallback_revision,
            "agent_mode": self.agent_mode,
            "apex_python": self.apex_python,
            "apex_repo": self.apex_repo,
        }
        if is_autogen:
            # Mode C result depends on the orchestrator identity + agent budget +
            # author toggle, not the subprocess agent_mode.
            scoped_inputs.update({
                "orchestrator": "autogen",
                "autogen_max_agents": self.autogen_max_agents,
                "autogen_author": self.autogen_author,
                "autogen_agent_ceiling": self.autogen_agent_ceiling,
                "autogen_scout_agents": self.autogen_scout_agents,
            })
        components = {
            "kind": "commit0_cell",
            "arm": arm.id,
            "repo": repo,
            "limit": limit,
            "scoped_inputs": scoped_inputs,
        }

        def _run() -> dict:
            self.engine.phase(f"{arm.id}:{repo}")
            self.engine.log(f"running commit0 cell arm={arm.id} repo={repo} limit={limit} "
                            f"local_runnable={spec.local_runnable} "
                            f"mode={'autogen' if is_autogen else 'v1_subprocess'}")
            if is_autogen:
                return self._invoke_autogen(cfg_dict, repo, output_dir=str(cell_out),
                                            fallback_rev=spec.dataset_fallback_revision)
            return self._invoke_runner(cfg_dict, repo, limit=limit, output_dir=str(cell_out),
                                       fallback_rev=spec.dataset_fallback_revision)

        def _cell_status(rep: dict) -> str:
            # An infra failure (non-zero rc with no report, or no report at all) is
            # NOT a valid cache hit — fixing the env and re-running must retry it.
            if not isinstance(rep, dict):
                return RESULT_INFRA_NONRESULT
            if rep.get("_report_path") is None:
                return RESULT_INFRA_NONRESULT
            if rep.get("_returncode", 0) != 0 and not rep.get("total_tasks"):
                return RESULT_INFRA_NONRESULT
            return RESULT_OK

        report, hit = resume_or_run_json(self.engine.journal, components, _run,
                                         kind="commit0_cell", node_id=f"{arm.id}:{repo}",
                                         status_fn=_cell_status)
        report = dict(report or {})
        report["_cache_hit"] = hit
        report["_arm"] = arm.id
        report["_repo"] = repo
        report["_arm_kind"] = arm.kind
        report["_maps_to"] = arm.maps_to
        return report

    def _invoke_runner(self, cfg_dict: dict, repo: str, *, limit: int, output_dir: str,
                       fallback_rev: Optional[str], rollouts: Optional[int] = None) -> dict:
        """Invoke the recon-validated v1 CLI path in a subprocess (robust; uses
        the exact host-local no-Docker pipeline that v1 ships).  The APEX-Ω engine
        still owns orchestration (journaled cell, budget, matrix, resume)."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        config_path = out / "arm_config.json"
        config_path.write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")
        cmd = [
            self.apex_python, "-m", "apex", "commit0-benchmark",
            "--config", str(config_path),
            "--split", repo, "--repos", repo,
            "--limit", str(limit),
            "--output", str(out),
            "--agent-mode", self.agent_mode,
        ]
        if rollouts is not None:
            cmd += ["--rollouts", str(rollouts)]
        env = dict(os.environ)
        env.setdefault("HF_DATASETS_OFFLINE", "1")
        env.setdefault("HF_HUB_OFFLINE", "1")
        # Ensure the subprocess `python -m apex` resolves the VENDORED apex (repo
        # root first on PYTHONPATH), not any external/editable install.
        env["PYTHONPATH"] = self.apex_repo + os.pathsep + env.get("PYTHONPATH", "")
        try:
            proc = subprocess.run(
                cmd, cwd=self.apex_repo, env=env, text=True,
                capture_output=True, timeout=self.cell_timeout_seconds,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            # Infra failure -> return a non-result dict (classified RESULT_INFRA_NONRESULT
            # by run_cell so it re-runs next time), never abort the matrix.
            return {"_report_path": None, "_returncode": -1, "total_tasks": 0,
                    "solved_tasks": 0, "_error": f"{type(exc).__name__}: {exc}"}
        report = _find_report(out)
        report.setdefault("_returncode", proc.returncode)
        if proc.returncode != 0 and not report.get("total_tasks"):
            report["_stderr_tail"] = (proc.stderr or "")[-2000:]
            report["_stdout_tail"] = (proc.stdout or "")[-1000:]
        return report

    def _invoke_autogen(self, cfg_dict: dict, repo: str, *, output_dir: str,
                        fallback_rev: Optional[str]) -> dict:
        """Mode C: run the generated-code orchestrator in-process on self.engine
        (journaled by the agent() calls inside autosolve).  Reuses v1's prep +
        scoring verbatim and NEVER raises (returns an _error dict on failure so the
        matrix continues)."""
        from .commit0_autogen import run_autogen_cell, write_cell_report

        report = run_autogen_cell(
            self.engine,
            cfg_dict,
            repo,
            apex_python=self.apex_python,
            apex_repo=self.apex_repo,
            max_agents=self.autogen_max_agents,
            agent_ceiling=self.autogen_agent_ceiling,
            author=self.autogen_author,
            scout_agents=self.autogen_scout_agents,
            output_dir=output_dir,
            cell_timeout_seconds=self.cell_timeout_seconds,
            fallback_rev=fallback_rev,
        )
        return write_cell_report(report, output_dir)

    # -- a full matrix ---------------------------------------------------
    def run_matrix(self, arm_ids: list[str], repos: list[str], *, limit: int = 1,
                   rollouts: Optional[int] = None) -> dict:
        cells: list[dict] = []
        for arm_id in arm_ids:
            arm = get_arm(arm_id)
            for repo in repos:
                try:
                    cells.append(self.run_cell(arm, repo, limit=limit, rollouts=rollouts))
                except Exception as exc:  # one cell's failure never aborts the matrix
                    self.engine.log(f"cell {arm_id}:{repo} raised {type(exc).__name__}: {exc}")
                    cells.append({"_arm": arm_id, "_repo": repo, "_arm_kind": arm.kind,
                                  "_error": f"{type(exc).__name__}: {exc}",
                                  "total_tasks": 0, "solved_tasks": 0})
        matrix = {
            "arms": arm_ids,
            "repos": repos,
            "limit": limit,
            "cells": cells,
            "budget": self.engine.budget.to_dict(),
            "journal": self.engine.journal.stats(),
            "summary": _summarize(cells),
        }
        out = self.run_dir / "matrix_report.json"
        out.write_text(json.dumps(matrix, indent=2, default=str), encoding="utf-8")
        return matrix


def _find_report(out_dir: Path) -> dict:
    """Locate and parse the commit0 benchmark report JSON written by the runner."""
    candidates = [out_dir / "benchmark_report.json"]
    candidates += sorted(out_dir.rglob("*benchmark_report*.json"))
    candidates += sorted(out_dir.rglob("*report*.json"))
    seen = set()
    for c in candidates:
        if c in seen or not c.exists():
            continue
        seen.add(c)
        try:
            data = json.loads(c.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            data["_report_path"] = str(c)
            # surface the common headline keys at top level for the summary
            for k in ("solved_tasks", "total_tasks", "average_pass_rate_percent",
                      "average_pass_rate", "solve_rate", "strict_solved", "runnable_solved"):
                if k in data:
                    data.setdefault(k, data[k])
            return data
    return {"_report_path": None}


def _report_to_dict(report: Any) -> dict:
    if report is None:
        return {}
    if isinstance(report, dict):
        return report
    if hasattr(report, "to_dict"):
        try:
            return report.to_dict()
        except Exception:
            pass
    # defensive attribute scrape
    out = {}
    for attr in ("solved_tasks", "total_tasks", "average_pass_rate_percent", "average_pass_rate",
                 "strict_solved", "runnable_solved", "solve_rate", "pass_rate"):
        if hasattr(report, attr):
            out[attr] = getattr(report, attr)
    return out or {"_repr": repr(report)[:500]}


def _summarize(cells: list[dict]) -> dict:
    by_arm: dict[str, dict] = {}
    for c in cells:
        arm = c.get("_arm", "?")
        agg = by_arm.setdefault(arm, {"solved": 0, "total": 0, "cells": 0})
        agg["cells"] += 1
        agg["solved"] += int(c.get("solved_tasks", 0) or 0)
        agg["total"] += int(c.get("total_tasks", 0) or 0)
    for arm, agg in by_arm.items():
        agg["solve_rate"] = (agg["solved"] / agg["total"]) if agg["total"] else None
    return by_arm
