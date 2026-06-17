#!/usr/bin/env python3
"""Durable, ACCUMULATING template-vs-autogen signals ledger (read-only over runs).

Purpose: capture every signal that will inform a later plan to improve autogen
(generated-code orchestration) — solve-rate, AGENT EFFICIENCY (agents-per-solve),
failure-class mix, where autogen wins vs the template, and difficulty correlation —
fused with an append-only qualitative observation log that survives cell-stripping
and cell re-runs.

Inputs (all under runs/ladder):
  - per-cell reports (autogen_cell_report.json / benchmark_report.json)
  - progress.jsonl            (wall-clock per cell)
  - signals_log.jsonl         (append-only qualitative observations / hypotheses)
Outputs:
  - runs/ladder/SIGNALS_LEDGER.md     (human-readable, the thing to read later)
  - runs/ladder/signals_ledger.json   (machine-readable snapshot)

Append an observation:  python scripts/track_signals.py --note "text" [--tag T --repo R --source S]
Regenerate the ledger:  PYTHONPATH=. python scripts/track_signals.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LADDER = REPO / "runs" / "ladder"
SIGNALS_LOG = LADDER / "signals_log.jsonl"

ARM_LABELS = {
    "B0_codex_1shot": "B0 vanilla 1-shot",
    "baseline_v1_k8": "baseline_v1 (K=8)",
    "omega_template_k8": "template (K=8)",
    "omega_autogen_k8": "autogen (K=8)",
    "B2_v1_fullcap16": "B2 v1 cap16",
}
REPOS = ["voluptuous", "jinja", "mimesis", "pydantic", "networkx", "cookiecutter"]

try:
    from apex_omega.autogen.sandbox import SAFE_BUILTINS
    _SAFE = set(SAFE_BUILTINS)
except Exception:
    _SAFE = set()


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _cell_report(cell_dir: Path) -> dict:
    for f in list(cell_dir.rglob("autogen_cell_report.json")) + list(cell_dir.rglob("benchmark_report.json")):
        d = _read_json(f)
        if isinstance(d, dict) and ("solved_tasks" in d or d.get("completed")):
            return d
    return {}


def _attempt_blocks(cell_dir: Path) -> tuple[int, int]:
    """Count solver attempts the sandbox BLOCKED before verification: a strategy
    that chose a workspace-illegal action (e.g. fetch upstream into /tmp) shows up
    as finalization_status=policy_violation / result_status=infra_nonresult with
    tokens=0 and no pytest run. This is genuinely BOTH a strategy defect and an
    infra block, and must NOT be filed as a clean 'genuine_abstain'."""
    pv = inr = 0
    for w in cell_dir.rglob("calls_wal.jsonl"):
        try:
            text = w.read_text().lower()
        except OSError:
            continue
        pv += text.count('"finalization_status":"policy_violation"')
        inr += text.count('"result_status":"infra_nonresult"')
    return pv, inr


def _classify_failure(a: dict, pv: int = 0, inr: int = 0) -> str:
    if (a.get("solved") or 0) >= 1:
        return "solved"
    if a.get("total") in (0, None):
        return "not_run"
    origin = a.get("origin")
    if a.get("orchestration_error"):
        return "infra_crash"
    if origin in ("fallback", "authored_then_floor"):
        return "malformed_orch"
    # Strategy chose a sandbox-illegal action -> attempts blocked, never verified.
    if pv > 0 or inr > 0:
        return "strategy_sandbox_block"
    if origin == "authored" and (a.get("agents") or 0) > 0:
        return "genuine_abstain"
    if origin in (None, "template") and (a.get("agents") or 0) == 0:
        return "infra_or_timeout"
    return "unknown"


def _wall_by_cell() -> dict:
    out = {}
    if SIGNALS_LOG.exists():
        pass
    prog = LADDER / "progress.jsonl"
    if prog.exists():
        for line in prog.read_text().splitlines():
            d = _read_json_line(line)
            if d:
                out[(d.get("label"), d.get("repo"))] = d.get("wall_s") or d.get("dur_s")
    return out


def _read_json_line(line: str):
    try:
        return json.loads(line)
    except Exception:
        return None


def collect() -> dict:
    walls = _wall_by_cell()
    cells: dict = {}
    for cell in sorted(LADDER.glob("*__*")):
        if "__" not in cell.name or not cell.is_dir():
            continue
        label, repo = cell.name.split("__", 1)
        rep = _cell_report(cell)
        orch = rep.get("orchestration") or {}
        rec = {
            "solved": rep.get("solved_tasks"),
            "total": rep.get("total_tasks"),
            "agents": rep.get("agents_used"),
            "difficulty": rep.get("difficulty"),
            "origin": orch.get("origin"),
            "orchestration_error": rep.get("_orchestration_error"),
            "wall_s": walls.get((label, repo)),
        }
        pv, inr = _attempt_blocks(cell)
        rec["policy_violations"] = pv
        rec["infra_nonresults"] = inr
        rec["failure_class"] = _classify_failure(rec, pv, inr)
        cells[(label, repo)] = rec
    return cells


def analyze(cells: dict) -> dict:
    # solve-rate per arm
    arms = {}
    for (label, repo), r in cells.items():
        a = arms.setdefault(label, {"cells": 0, "solved": 0, "total": 0, "agents_on_solves": [], "fail_classes": {}})
        a["cells"] += 1
        a["solved"] += int(r.get("solved") or 0)
        a["total"] += int(r.get("total") or 0)
        if (r.get("solved") or 0) >= 1 and r.get("agents"):
            a["agents_on_solves"].append(r["agents"])
        fc = r.get("failure_class")
        a["fail_classes"][fc] = a["fail_classes"].get(fc, 0) + 1
    for a in arms.values():
        a["solve_rate"] = (a["solved"] / a["total"]) if a["total"] else None
        a["agents_per_solve"] = (sum(a["agents_on_solves"]) / len(a["agents_on_solves"])) if a["agents_on_solves"] else None

    # autogen vs template head-to-head (the core signal)
    h2h = []
    for repo in REPOS:
        t = cells.get(("omega_template_k8", repo))
        g = cells.get(("omega_autogen_k8", repo))
        if not t and not g:
            continue
        t_ok = (t or {}).get("solved", 0) and (t or {}).get("solved") >= 1
        g_ok = (g or {}).get("solved", 0) and (g or {}).get("solved") >= 1
        if g is None or g.get("total") in (0, None):
            verdict = "autogen_not_run"
        elif g_ok and not t_ok:
            verdict = "AUTOGEN_WON"            # the upside case that justifies autogen
        elif t_ok and not g_ok:
            verdict = "autogen_lost"           # strategy gap
        elif g_ok and t_ok:
            ta, ga = (t or {}).get("agents") or 0, (g or {}).get("agents") or 0
            verdict = "tie_autogen_costlier" if ga > ta else ("tie_autogen_cheaper" if ga < ta else "tie")
        else:
            verdict = "both_failed"
        h2h.append({
            "repo": repo, "difficulty": (g or t or {}).get("difficulty"),
            "template": {"solved": (t or {}).get("solved"), "agents": (t or {}).get("agents"), "wall_s": (t or {}).get("wall_s")},
            "autogen": {"solved": (g or {}).get("solved"), "agents": (g or {}).get("agents"),
                        "wall_s": (g or {}).get("wall_s"), "failure_class": (g or {}).get("failure_class")},
            "verdict": verdict,
        })
    return {"arms": arms, "head_to_head": h2h}


def render_md(cells: dict, an: dict, obs: list) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = [f"# Template vs Autogen — Signals Ledger", "",
         f"_Updated {now}. Comparison mode: fail-open-to-template OFF (autogen stands alone)._",
         "_Goal: accumulate signal to design a plan that improves autogen's power/capabilities._", ""]

    L.append("## Solve-rate + efficiency per arm")
    L.append("| arm | solved/total | solve-rate | agents/solve | failure mix |")
    L.append("|---|---|---|---|---|")
    for label in ["B0_codex_1shot", "baseline_v1_k8", "omega_template_k8", "omega_autogen_k8", "B2_v1_fullcap16"]:
        a = an["arms"].get(label)
        if not a:
            continue
        sr = f"{a['solve_rate']*100:.0f}%" if a["solve_rate"] is not None else "-"
        aps = f"{a['agents_per_solve']:.1f}" if a["agents_per_solve"] is not None else "-"
        fc = {k: v for k, v in a["fail_classes"].items() if k != "solved"}
        L.append(f"| {ARM_LABELS.get(label, label)} | {a['solved']}/{a['total']} | {sr} | {aps} | {fc or '-'} |")

    L.append("")
    L.append("## Autogen vs template — per-repo head-to-head (the core signal)")
    L.append("| repo | difficulty | template (solved/agents) | autogen (solved/agents) | verdict |")
    L.append("|---|---|---|---|---|")
    for h in an["head_to_head"]:
        t, g = h["template"], h["autogen"]
        L.append(f"| {h['repo']} | {h['difficulty']} | {t['solved']}/{t['agents']} | "
                 f"{g['solved']}/{g['agents']} ({g['failure_class']}) | **{h['verdict']}** |")

    # tallies that matter for the improvement plan
    v = [h["verdict"] for h in an["head_to_head"]]
    L.append("")
    L.append(f"- **autogen wins (solved where template didn't):** {v.count('AUTOGEN_WON')}  "
             f"_(the case that justifies autogen; watch this number)_")
    L.append(f"- **autogen losses (failed where template solved):** {v.count('autogen_lost')}")
    L.append(f"- **ties where autogen cost MORE agents:** {v.count('tie_autogen_costlier')}")
    L.append(f"- **ties where autogen cost fewer/equal:** {v.count('tie_autogen_cheaper') + v.count('tie')}")

    L.append("")
    L.append("## Accumulated observations & hypotheses")
    for o in obs:
        rp = f" [{o.get('repo')}]" if o.get("repo") and o.get("repo") != "*" else ""
        L.append(f"- `{o.get('tag','note')}`{rp} {o.get('observation','')}  _({o.get('source','')}, {o.get('ts','')})_")

    L.append("")
    L.append("## When is there enough signal to plan autogen improvements?")
    L.append("- Need autogen vs template on all clean repos (>= 4) AND a variance read "
             "(>= 3 repeats on >=1 hard repo) before concluding systematic vs luck.")
    L.append("- Decision metric: autogen must show >=1 clear AUTOGEN_WON (solve template misses) "
             "OR a lower agents/solve at equal solve-rate; otherwise its added complexity isn't paying off.")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--note")
    ap.add_argument("--tag", default="note")
    ap.add_argument("--repo", default="*")
    ap.add_argument("--source", default="manual")
    args = ap.parse_args()

    LADDER.mkdir(parents=True, exist_ok=True)
    if args.note:
        rec = {"ts": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "tag": args.tag,
               "repo": args.repo, "observation": args.note, "source": args.source}
        with SIGNALS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print("appended observation:", rec)

    obs = []
    if SIGNALS_LOG.exists():
        for line in SIGNALS_LOG.read_text().splitlines():
            d = _read_json_line(line)
            if d:
                obs.append(d)

    cells = collect()
    an = analyze(cells)
    (LADDER / "signals_ledger.json").write_text(json.dumps(
        {"arms": an["arms"], "head_to_head": an["head_to_head"], "observations": obs}, indent=2, default=str))
    md = render_md(cells, an, obs)
    (LADDER / "SIGNALS_LEDGER.md").write_text(md)
    print(md)
    print(f"\nwrote {LADDER/'SIGNALS_LEDGER.md'} and signals_ledger.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
