#!/usr/bin/env python3
"""A/B verdict for an omega ladder run dir (e.g. /tmp/omega_ab).

Reads progress.jsonl (gold-scored solved/agents/wall per cell) and, when present,
each cell's matrix_report.json (token spend + orchestration.origin). Prints a
per-repo Arm-A-vs-Arm-B table and checks the promotion rule:

  Arm B (converge) wins ONLY if it converts the hard repos (mimesis/babel) that
  Arm A (flips) misses, WITHOUT regressing the easy/medium repos (voluptuous/jinja)
  or the wall-clock / agent budget.

Usage: python scripts/ab_verdict.py [RUN_DIR]   (default /tmp/omega_ab)
"""
import json
import os
import sys
from collections import defaultdict

RUN = sys.argv[1] if len(sys.argv) > 1 else "/tmp/omega_ab"
A = "omega_flips_unbounded"      # Arm A: flat best-of-N default + repair flips
B = "omega_converge_unbounded"   # Arm B: rebuilt decompose->fanout->reduce->loop->verify
REPO_ORDER = ["voluptuous", "jinja", "mimesis", "babel", "pydantic"]


def planned_total(run):
    """Total planned cells, parsed from the runner's first stdout line if present."""
    import re
    p = os.path.join(run, "runner_stdout.log")
    if not os.path.exists(p):
        return None
    try:
        first = open(p).readline()
        m = re.search(r"(\d+)\s+cells total", first)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def load_progress(run):
    cells = []
    p = os.path.join(run, "progress.jsonl")
    if not os.path.exists(p):
        sys.exit(f"no progress.jsonl in {run}")
    for line in open(p):
        line = line.strip()
        if line:
            cells.append(json.loads(line))
    return dedup(cells)


def dedup(cells):
    """A resumed runner appends 'skip' rows for already-done cells, so a cell can have
    several rows (done + skip[+ relaunch]). Keep ONE row per (label,repo,seed): prefer a
    row that has a real solved value, then one carrying wall_s, then the latest ts — so
    we never double-count a seed and never lose the wall_s recorded on the original 'done'."""
    best = {}
    for c in cells:
        k = (c["label"], c["repo"], c["seed"])
        score = (1 if c.get("solved") is not None else 0,
                 1 if "wall_s" in c else 0,
                 c.get("ts", 0))
        if k not in best or score > best[k][0]:
            best[k] = (score, c)
    return [v[1] for v in best.values()]


def token_spend(run, label, repo, seed):
    """Best-effort per-cell token spend + orchestration origin from matrix_report.json."""
    d = os.path.join(run, f"{label}__{repo}__s{seed}")
    mr = os.path.join(d, "matrix_report.json")
    if not os.path.exists(mr):
        return None, None
    try:
        data = json.load(open(mr))
        c = (data.get("cells") or [{}])[0]
        spent = (c.get("budget") or {}).get("spent")
        origin = (c.get("orchestration") or {}).get("origin")
        return spent, origin
    except Exception:
        return None, None


def main():
    cells = load_progress(RUN)
    # index: (label, repo) -> list of {seed, solved, agents, wall, tokens, origin}
    idx = defaultdict(list)
    seeds_seen = set()
    for c in cells:
        label, repo, seed = c["label"], c["repo"], c["seed"]
        seeds_seen.add(seed)
        tok, origin = token_spend(RUN, label, repo, seed)
        idx[(label, repo)].append({
            "seed": seed, "solved": c.get("solved", 0), "total": c.get("total", 1),
            "agents": c.get("agents"), "wall": c.get("wall_s"),
            "tokens": tok, "origin": origin, "difficulty": c.get("difficulty"),
        })

    repos = [r for r in REPO_ORDER if (A, r) in idx or (B, r) in idx]
    n_seeds_planned = max(seeds_seen) + 1 if seeds_seen else 0

    print(f"\nA/B verdict — {RUN}")
    print(f"  Arm A = {A} (flat best-of-N + repair)")
    print(f"  Arm B = {B} (rebuilt convergence)")
    print(f"  seeds observed: {sorted(seeds_seen)}  (cells done: {len(cells)})\n")

    def agg(label, repo):
        rows = sorted(idx.get((label, repo), []), key=lambda r: r["seed"])
        solved = sum(r["solved"] for r in rows)
        total = sum(r["total"] for r in rows)
        n = len(rows)
        agents = [r["agents"] for r in rows if r["agents"] is not None]
        walls = [r["wall"] for r in rows if r["wall"] is not None]
        toks = [r["tokens"] for r in rows if r["tokens"]]
        return {
            "n": n, "solved": solved, "total": total, "rows": rows,
            "agents_mean": (sum(agents) / len(agents)) if agents else None,
            "wall_mean": (sum(walls) / len(walls)) if walls else None,
            "tok_total": sum(toks) if toks else None,
            "per_seed": {r["seed"]: r["solved"] for r in rows},
        }

    hdr = f"{'repo':<12} {'diff':<7} {'A solved':>9} {'B solved':>9}  {'A agents~':>9} {'B agents~':>9}  {'A wall~':>9} {'B wall~':>9}"
    print(hdr)
    print("-" * len(hdr))
    promotion = {"regressed": [], "converted": [], "tie": []}
    for repo in repos:
        a, b = agg(A, repo), agg(B, repo)
        diff = (a["rows"] or b["rows"])[0]["difficulty"] if (a["rows"] or b["rows"]) else "?"

        def sc(x):
            return f"{x['solved']}/{x['total']}" if x["n"] else "--"

        def num(v, f="{:.0f}"):
            return f.format(v) if v is not None else "--"

        print(f"{repo:<12} {str(diff):<7} {sc(a):>9} {sc(b):>9}  "
              f"{num(a['agents_mean']):>9} {num(b['agents_mean']):>9}  "
              f"{num(a['wall_mean']):>9} {num(b['wall_mean']):>9}")

        # promotion-rule bookkeeping (only on seeds both arms have completed)
        common = set(a["per_seed"]) & set(b["per_seed"])
        for s in sorted(common):
            av, bv = a["per_seed"][s], b["per_seed"][s]
            if bv > av:
                promotion["converted"].append(f"{repo}/s{s} (A={av} B={bv})")
            elif bv < av:
                promotion["regressed"].append(f"{repo}/s{s} (A={av} B={bv})")
            else:
                promotion["tie"].append(f"{repo}/s{s}")

    # overall
    tot_a = agg_all(idx, A, repos)
    tot_b = agg_all(idx, B, repos)
    print("-" * len(hdr))
    print(f"{'TOTAL':<12} {'':<7} {tot_a['solved']:>7}/{tot_a['total']:<1} {tot_b['solved']:>7}/{tot_b['total']:<1}")

    print("\nPromotion-rule check (B wins only if it converts hard repos w/o regressing others):")
    print(f"  B converts (B>A): {promotion['converted'] or 'none'}")
    print(f"  B regresses (B<A): {promotion['regressed'] or 'none'}")
    print(f"  ties: {len(promotion['tie'])}")
    converged_origins = {(repo, r['origin']) for repo in repos for r in idx.get((B, repo), []) if r['origin']}
    print(f"\n  Arm B orchestration origins seen: {sorted(o for _, o in converged_origins) or 'n/a'}")

    if not promotion["converted"] and promotion["regressed"]:
        verdict = "LEANING A: B has not converted any hard cell and regressed somewhere."
    elif promotion["converted"] and not promotion["regressed"]:
        verdict = "LEANING B: B converts hard cells with no regressions (confirm cost budget)."
    elif promotion["converted"] and promotion["regressed"]:
        verdict = "MIXED: B both converts and regresses — weigh per-repo + cost."
    else:
        verdict = "TIE so far on completed seeds."
    print(f"\n  PRELIMINARY VERDICT: {verdict}")
    # Asymmetric matrix: A runs to its full 4-repo x N-seed target; B may be FROZEN
    # (stopped mid-run to fix the reduce/merge bug), so report each arm independently.
    n_seeds = max(n_seeds_planned, 3)  # this sweep is 3 seeds
    a_cells = sum(len(idx.get((A, r), [])) for r in repos)
    b_cells = sum(len(idx.get((B, r), [])) for r in repos)
    a_target = len(repos) * n_seeds
    print(f"  Arm A: {a_cells}/{a_target} cells done"
          + ("" if a_cells >= a_target else f" ({a_target - a_cells} pending — A still running)"))
    print(f"  Arm B: {b_cells} cells done (FROZEN — stopped for fix; not resuming)\n")


def agg_all(idx, label, repos):
    solved = total = 0
    for repo in repos:
        for r in idx.get((label, repo), []):
            solved += r["solved"]
            total += r["total"]
    return {"solved": solved, "total": total}


if __name__ == "__main__":
    main()
