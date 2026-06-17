#!/usr/bin/env python3
"""Capture autogen-vs-template failure evidence from the ladder run (read-only).

INVARIANT under test: autogen (scout + authored orchestrate) may cost MORE than
omega_template, but must NOT fail more often. This script collects, per repo, the
evidence needed to diagnose any autogen<template regression after the run:
  - solved/agents/difficulty/orchestration origin + error per autogen & template cell
  - authored-code RUNTIME errors surfaced in narration ("... raised <Err>: <msg>"),
    incl. the missing-builtin NameErrors that silently zero out attempts
  - static scan of each authored orchestrate(ctx) for builtins MISSING from
    SAFE_BUILTINS (the sandbox-allowlist gap that crashes authored attempts)
  - regression flags: repos where autogen failed but template solved

Run:  PYTHONPATH=. <venv> scripts/capture_autogen_evidence.py
Out:  runs/ladder/autogen_evidence.json  +  runs/ladder/autogen_evidence.md
"""

from __future__ import annotations

import ast
import builtins as B
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LADDER = REPO / "runs" / "ladder"

try:
    from apex_omega.autogen.sandbox import SAFE_BUILTINS
    SAFE = set(SAFE_BUILTINS)
except Exception:
    SAFE = set()

_RAISED = re.compile(r"raised\s+([A-Za-z_]+Error|[A-Za-z_]+Exception)\s*:?\s*([^\"]*)")


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _cell_report(cell_dir: Path) -> dict:
    for f in list(cell_dir.rglob("autogen_cell_report.json")) + list(cell_dir.rglob("benchmark_report.json")):
        d = _read_json(f)
        if isinstance(d, dict):
            return d
    return {}


def _authored_src(cell_dir: Path) -> str:
    for f in cell_dir.rglob("orchestrator/*.py"):
        try:
            return f.read_text()
        except Exception:
            pass
    return ""


def _missing_builtins(src: str) -> list[str]:
    if not src:
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return sorted(b for b in used if hasattr(B, b) and b not in SAFE)


def _classify_failure(a: dict) -> str:
    """Why did this autogen cell fail? The distinction the user cares about:
      infra_crash      -> sandbox/lint/runtime defect in OUR harness (missing builtin,
                          lint reject, orchestration exception). MUST be fixed; not a
                          real autogen weakness.
      malformed_orch   -> generated code didn't lint/compile -> fell open to template.
      genuine_abstain  -> authored plan ran clean, spent agents, accepted nothing. This
                          is autogen's REAL weakness (strategy/prompt gap) to study.
      unknown          -> couldn't classify from artifacts.
    """
    if (a.get("solved") or 0) >= 1:
        return "solved"
    origin = a.get("orchestration_origin")
    if a.get("missing_builtins") or a.get("narration_errors", {}).get("counts") or a.get("orchestration_error"):
        return "infra_crash"
    if origin in ("fallback", "authored_then_floor"):
        return "malformed_orch"      # generated code was rejected/crashed -> floored
    # Strategy chose a workspace-illegal action (fetch upstream, etc.) -> attempts
    # blocked by the sandbox before verification. BOTH a strategy defect and infra.
    if (a.get("policy_violations") or 0) > 0 or (a.get("infra_nonresults") or 0) > 0:
        return "strategy_sandbox_block"
    if origin == "authored" and (a.get("agents_used") or 0) > 0:
        return "genuine_abstain"     # ran its own plan, attempts verified-but-failed
    return "unknown"


def _narration_errors(cell_dir: Path) -> dict:
    nar = cell_dir / "narration.jsonl"
    errs: dict[str, int] = {}
    samples: list[str] = []
    if nar.exists():
        for line in nar.read_text().splitlines():
            d = _read_json_line(line)
            msg = (d or {}).get("msg", "")
            for m in _RAISED.finditer(msg):
                key = f"{m.group(1)}: {m.group(2).strip()[:60]}"
                errs[key] = errs.get(key, 0) + 1
                if len(samples) < 3:
                    samples.append(msg[:160])
    return {"counts": errs, "samples": samples}


def _read_json_line(line: str):
    try:
        return json.loads(line)
    except Exception:
        return None


def main() -> int:
    autogen, template = {}, {}
    for cell in sorted(LADDER.glob("omega_autogen_k8__*")):
        repo = cell.name.split("__", 1)[1]
        rep = _cell_report(cell)
        src = _authored_src(cell)
        autogen[repo] = {
            "solved": rep.get("solved_tasks"), "total": rep.get("total_tasks"),
            "agents_used": rep.get("agents_used"), "difficulty": rep.get("difficulty"),
            "orchestration_origin": (rep.get("orchestration") or {}).get("origin"),
            "orchestration_error": rep.get("_orchestration_error"),
            "missing_builtins": _missing_builtins(src),
            "narration_errors": _narration_errors(cell),
            "authored_chars": len(src),
        }
        _pv = _inr = 0
        for w in cell.rglob("calls_wal.jsonl"):
            try:
                _t = w.read_text().lower()
            except OSError:
                continue
            _pv += _t.count('"finalization_status":"policy_violation"')
            _inr += _t.count('"result_status":"infra_nonresult"')
        autogen[repo]["policy_violations"] = _pv
        autogen[repo]["infra_nonresults"] = _inr
        autogen[repo]["failure_class"] = _classify_failure(autogen[repo])
    for cell in sorted(LADDER.glob("omega_template_k8__*")):
        repo = cell.name.split("__", 1)[1]
        rep = _cell_report(cell)
        template[repo] = {"solved": rep.get("solved_tasks"), "total": rep.get("total_tasks"),
                          "agents_used": rep.get("agents_used")}

    # INFRA failures = our bugs to fix (must be driven to zero). UNDERPERFORMANCE =
    # autogen abstained where template solved: a REAL strategy gap (expected now that
    # fail-open-to-template is OFF), to be studied, not "fixed" by masking.
    infra_failures, underperf, missing_union, error_union = [], [], set(), {}
    by_class: dict[str, list] = {}
    for repo, a in autogen.items():
        t = template.get(repo, {})
        fc = a.get("failure_class")
        by_class.setdefault(fc, []).append(repo)
        if fc == "infra_crash" or fc == "malformed_orch":
            infra_failures.append({"repo": repo, "class": fc, "autogen": a})
        a_ok = (a.get("solved") or 0) >= 1
        t_ok = (t.get("solved") or 0) >= 1
        if t_ok and not a_ok:
            underperf.append({"repo": repo, "class": fc, "autogen": a, "template": t})
        for b in a.get("missing_builtins", []):
            missing_union.add(b)
        for k, v in a.get("narration_errors", {}).get("counts", {}).items():
            error_union[k] = error_union.get(k, 0) + v

    out = {
        "comparison_mode": "fail-open-to-template is OFF; autogen stands on its own (proper comparison)",
        "autogen_cells": autogen, "template_cells": template,
        "failure_classes": {k: sorted(v) for k, v in by_class.items()},
        "INFRA_FAILURES_must_fix": infra_failures,                     # our bugs
        "autogen_underperformance_vs_template": underperf,            # real strategy gaps
        "missing_builtins_union": sorted(missing_union),
        "authored_runtime_errors_union": error_union,
    }
    (LADDER / "autogen_evidence.json").write_text(json.dumps(out, indent=2))

    lines = ["# Autogen failure evidence (generated-code orchestration)", ""]
    lines.append("_Comparison mode: fail-open-to-template **OFF** — autogen stands on its own._")
    lines.append("")
    lines.append(f"Failure classes: **{ {k: len(v) for k, v in by_class.items()} }**")
    lines.append(f"INFRA failures (OUR bugs to fix): "
                 f"**{[f['repo']+':'+f['class'] for f in infra_failures] or 'none'}**")
    lines.append(f"SAFE_BUILTINS gap — builtins authored code used but sandbox forbids: "
                 f"**{sorted(missing_union) or 'none'}**")
    lines.append(f"Authored-code runtime errors seen: **{error_union or 'none'}**")
    lines.append(f"Autogen UNDERPERFORMED template (real strategy gap, expected): "
                 f"**{[u['repo']+':'+(u['class'] or '?') for u in underperf] or 'none'}**")
    lines.append("")
    lines.append("| repo | template | autogen | difficulty | failure class | missing builtins | authored errors |")
    lines.append("|---|---|---|---|---|---|---|")
    for repo in sorted(set(autogen) | set(template)):
        a, t = autogen.get(repo, {}), template.get(repo, {})
        lines.append(f"| {repo} | {t.get('solved')}/{t.get('total')} | "
                     f"{a.get('solved')}/{a.get('total')} (agents {a.get('agents_used')}) | "
                     f"{a.get('difficulty')} | {a.get('failure_class')} | {a.get('missing_builtins')} | "
                     f"{list(a.get('narration_errors',{}).get('counts',{}).keys())} |")
    (LADDER / "autogen_evidence.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {LADDER/'autogen_evidence.json'} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
