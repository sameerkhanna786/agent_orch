#!/usr/bin/env python3
"""Reclassify a commit0 ladder grid by TERMINAL CAUSE (methodology per user directive).

A cell counts as a real FAILURE only if the mode/orchestrator was *confident-but-wrong*
(a ship-arm — B0/baseline/B2 — returned its best rollout but it was not green) or *gave up*
(an orchestrated arm exhausted its agent budget / abstained with no accepted candidate).

A cell is a NON-RESULT (excluded from the solve-rate denominator and flagged to RE-RUN)
when it was blocked by something other than the strategy:
  * TIMEOUT  — the wall-clock cut it off before it used its agent budget
  * INFRA    — a harness error (e.g. venv-collision-on-resume, prep crash)
  * TOKEN    — a token ceiling stopped it (none by default; the engine is unbounded)

Usage:  python scripts/reclassify_grid.py [ladder_dir]   (default runs/ladder)
Env:    LADDER_CELL_TIMEOUT (default 3600) — the wall used to detect TIMEOUT.
Writes: <ladder_dir>/GRID_RECLASSIFIED.md and grid_reclassified.json, and prints a
        rerun list (the non-result cells, which should be re-run with the wall removed).
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

CELL_TIMEOUT = int(os.environ.get("LADDER_CELL_TIMEOUT", "3600"))
SHIP_ARMS = ("B0_", "baseline_", "B2_")   # arms that always ship their best rollout

FAILURE = {"confident_wrong", "gave_up"}
NON_RESULT = {"timeout", "infra", "token"}


def _load_report(cell_dir: Path) -> dict:
    for pat in ("autogen_cell_report.json", "benchmark_report.json"):
        hits = sorted(glob.glob(str(cell_dir / "**" / pat), recursive=True))
        for f in hits:
            try:
                return json.load(open(f))
            except Exception:
                continue
    # an error artifact (cell crashed before a real report)
    for f in sorted(glob.glob(str(cell_dir / "**" / "autogen_cell_error.json"), recursive=True)):
        try:
            return json.load(open(f))
        except Exception:
            continue
    return {}


def _integrity(cell_dir: Path) -> dict:
    """Tally recorded escape/cheat ATTEMPTS for a cell (telemetry; never affects pass/fail).
    Reads integrity_log.jsonl (one record per attempt that tried to escape the sandbox or
    cheat). These are denied structurally, so they never change the outcome — but they are
    useful cross-eval signal about model behaviour."""
    counts: dict = {}
    n = 0
    for f in glob.glob(str(cell_dir / "**" / "integrity_log.jsonl"), recursive=True) + \
            glob.glob(str(cell_dir / "integrity_log.jsonl")):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            n += 1
            for s in rec.get("signals", []):
                counts[s.get("kind")] = counts.get(s.get("kind"), 0) + 1
    return {"attempts": n, "by_kind": counts}


def classify(label: str, repo: str, rep: dict, wall_s: float) -> tuple[str, str]:
    """Return (outcome, evidence)."""
    # C7: a 'solved' must be backed by an execution-accepted candidate, not just a report field.
    # autogen cells carry a winner dict (require winner.accepted is True); v1 cells (B0/baseline/
    # B2) have no winner — their solved_tasks IS v1's execution-gated scored_success, so trust it.
    winner = rep.get("winner") or {}
    accepted_ok = (winner.get("accepted") is True) if winner else True
    solved = (int(rep.get("solved_tasks") or 0) >= 1) and accepted_ok
    err = rep.get("_error") or rep.get("_orchestration_error")
    is_ship = any(label.startswith(p) for p in SHIP_ARMS)
    near_wall = bool(wall_s and wall_s >= 0.97 * CELL_TIMEOUT)
    if solved:
        return "solved", "solved_tasks>=1"
    if err and not solved:
        return "infra", f"harness error: {str(err)[:80]}"
    if rep.get("_token_ceiling_hit"):
        return "token", "token ceiling reached"
    if is_ship:
        # B0/baseline/B2 always ship their best rollout -> a non-solve is confident-wrong,
        # unless the wall cut it off first.
        if near_wall:
            return "timeout", f"wall {wall_s:.0f}s ~ cap {CELL_TIMEOUT}s (cut before finishing)"
        return "confident_wrong", "shipped best rollout; not green"
    # orchestrated arm (template/autogen): gave up iff it used its agent budget / abstained.
    soft_cap = ((rep.get("agent_budget") or {}).get("soft_cap")) or 8
    agents = int(rep.get("agents_used") or 0)
    if agents >= soft_cap or rep.get("abstained"):
        return "gave_up", f"used {agents}/{soft_cap} agents, no accepted candidate (abstained)"
    if near_wall:
        return "timeout", f"wall {wall_s:.0f}s ~ cap {CELL_TIMEOUT}s before budget exhausted"
    return "gave_up", f"used {agents}/{soft_cap} agents, no accepted candidate"


def main() -> int:
    ladder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/ladder")
    prog_path = ladder / "progress.jsonl"
    prog = [json.loads(l) for l in open(prog_path) if l.strip()] if prog_path.exists() else []
    from collections import defaultdict
    last, relaunched = {}, set()
    for r in prog:
        seed = int(r.get("seed", 0) or 0)
        key = (r.get("label"), r.get("repo"), seed)
        if r.get("status") == "relaunch":
            relaunched.add(key)
        if r.get("status") in ("done", "skip"):
            last[key] = r

    rows = []
    for (label, repo, seed), r in sorted(last.items()):
        cell_dir = ladder / f"{label}__{repo}__s{seed}"
        if not cell_dir.exists():
            cell_dir = ladder / f"{label}__{repo}"          # single-seed back-compat naming
        rep = _load_report(cell_dir)
        wall = float(r.get("wall_s") or 0)
        outcome, evidence = classify(label, repo, rep, wall)
        integ = _integrity(cell_dir)
        rows.append({"arm": label, "repo": repo, "seed": seed, "outcome": outcome,
                     "wall_s": round(wall, 1), "agents": rep.get("agents_used"),
                     "relaunched": (label, repo, seed) in relaunched,
                     "integrity_attempts": integ["attempts"], "integrity_by_kind": integ["by_kind"],
                     "evidence": evidence})

    n_seeds = max((row["seed"] for row in rows), default=0) + 1

    # per-(arm,repo) aggregate ACROSS seeds: solved/N, pass@k (>=1 seed solved), variance.
    cellagg = defaultdict(lambda: {"seeds": 0, "solved": 0, "failures": 0, "non_results": 0, "outcomes": []})
    for row in rows:
        c = cellagg[(row["arm"], row["repo"])]
        c["seeds"] += 1
        c["outcomes"].append(f"s{row['seed']}:{row['outcome']}")
        if row["outcome"] == "solved":
            c["solved"] += 1
        elif row["outcome"] in FAILURE:
            c["failures"] += 1
        else:
            c["non_results"] += 1
    per_cell = []
    for (arm, repo), c in sorted(cellagg.items()):
        genuine = c["solved"] + c["failures"]
        per_cell.append({"arm": arm, "repo": repo, "seeds": c["seeds"], "solved": c["solved"],
                         "failures": c["failures"], "non_results": c["non_results"],
                         "pass_at_k": int(c["solved"] >= 1),
                         "solve_rate": (c["solved"] / genuine) if genuine else None,
                         "outcomes": c["outcomes"]})

    # per-arm tallies across ALL seed cells (seed-averaged) + per-arm pass@k over repos.
    arms: dict = {}
    for row in rows:
        a = arms.setdefault(row["arm"], {"solved": 0, "failures": 0, "non_results": 0})
        if row["outcome"] == "solved":
            a["solved"] += 1
        elif row["outcome"] in FAILURE:
            a["failures"] += 1
        else:
            a["non_results"] += 1
    arm_repo = defaultdict(lambda: {"repos": 0, "passk": 0})
    for cr in per_cell:
        ar = arm_repo[cr["arm"]]
        ar["repos"] += 1
        ar["passk"] += cr["pass_at_k"]
    for arm, a in arms.items():
        denom = a["solved"] + a["failures"]
        a["solve_rate_genuine"] = (a["solved"] / denom) if denom else None
        a["repos"] = arm_repo[arm]["repos"]
        a["pass_at_k_repos"] = arm_repo[arm]["passk"]

    rerun = sorted({f"{row['arm']}__{row['repo']}__s{row['seed']}"
                    for row in rows if row["outcome"] in NON_RESULT})

    # ---- emit ----
    out_json = {"cell_timeout": CELL_TIMEOUT, "n_seeds": n_seeds, "rows": rows,
                "per_cell": per_cell, "arms": arms, "rerun_targets": rerun}
    (ladder / "grid_reclassified.json").write_text(json.dumps(out_json, indent=2))

    md = [f"# Grid (reclassified by terminal cause) — n={n_seeds} seed(s)\n",
          f"Wall used for TIMEOUT detection: {CELL_TIMEOUT}s. Failures = confident_wrong + "
          "gave_up. Non-results (timeout/infra/token) are EXCLUDED from the denominator and "
          "listed to re-run with the wall removed.\n",
          "## Per-arm (across all seeds)\n"
          "| arm | solved-cells | failures | non-results | solve-rate (genuine) | repos pass@k |",
          "|---|---|---|---|---|---|"]
    for arm, a in sorted(arms.items()):
        sr = "-" if a["solve_rate_genuine"] is None else f"{a['solve_rate_genuine']*100:.0f}%"
        md.append(f"| {arm} | {a['solved']} | {a['failures']} | {a['non_results']} | {sr} | "
                  f"{a['pass_at_k_repos']}/{a['repos']} |")
    md.append(f"\n## Per (arm, repo) across {n_seeds} seeds (solved/seeds, pass@k, variance)\n"
              "| arm | repo | solved/seeds | pass@k | failures | non-results | outcomes |")
    md.append("|---|---|---|---|---|---|---|")
    for cr in per_cell:
        md.append(f"| {cr['arm']} | {cr['repo']} | {cr['solved']}/{cr['seeds']} | "
                  f"{'Y' if cr['pass_at_k'] else 'n'} | {cr['failures']} | {cr['non_results']} | "
                  f"{', '.join(cr['outcomes'])} |")
    md.append("\n## Re-run targets (non-results — re-run with wall removed)\n")
    md.append("\n".join(f"- {c}" for c in rerun) if rerun else "_(none)_")

    # escape/cheat telemetry (recorded, never penalized)
    integ_tot: dict = {}
    integ_cells = 0
    for row in rows:
        if row["integrity_attempts"]:
            integ_cells += 1
            for k, v in row["integrity_by_kind"].items():
                integ_tot[k] = integ_tot.get(k, 0) + v
    md.append("\n## Escape / cheat telemetry (recorded, NOT penalized — denied by the sandbox)\n")
    if integ_tot:
        md.append(f"Cells with >=1 recorded attempt: {integ_cells}. Totals by kind: {integ_tot}.")
        md.append("\n| arm | repo | attempts | by_kind |\n|---|---|---|---|")
        for row in rows:
            if row["integrity_attempts"]:
                md.append(f"| {row['arm']} | {row['repo']} | {row['integrity_attempts']} | "
                          f"{row['integrity_by_kind']} |")
    else:
        md.append("_No escape/cheat attempts recorded (integrity_log.jsonl absent or empty). "
                  "Note: cells run before this telemetry was added won't have it._")
    out_json["integrity_totals"] = integ_tot
    (ladder / "grid_reclassified.json").write_text(json.dumps(out_json, indent=2))
    (ladder / "GRID_RECLASSIFIED.md").write_text("\n".join(md) + "\n")

    print("\n".join(md))
    print("\nrerun_targets:", rerun)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
