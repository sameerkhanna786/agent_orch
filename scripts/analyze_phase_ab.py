#!/usr/bin/env python3
"""Scientific analysis of the phase-planner A/B ladder (pre-registered in
orchestration_research/EVAL_PROTOCOL.md).

Layers inferential statistics on top of scripts/reclassify_grid.py's C7-correct per-cell
classification (execution-gated SOLVE; infra/timeout/token NON-RESULTS excluded from denominators):

  1. per-arm solve-rate over GENUINE cells (solved + honest-fail) with a Wilson 95% CI
  2. paired McNemar exact test (two-sided binomial on discordant pairs), hybrid-vs-converge and
     hybrid-vs-hybrid-nogate, matched on (repo, seed) — per-repo and pooled
  3. paired SIGN test on the continuous best-gold FRONTIER (gold_passed/gold_total from
     phase_checkpoint.json) — graded progress that has power even when full solves are rare
  4. agents/solve (cost) distribution; pass@k; integrity (escape/cheat) telemetry
  5. an explicit power/limitations statement + a verdict against the pre-registered decision rules

Pure-stdlib (no scipy): Wilson interval, exact binomial two-sided p via math.comb.

Usage:  python scripts/analyze_phase_ab.py [ladder_dir]   (default runs/ladder)
Writes: <ladder_dir>/PHASE_AB_ANALYSIS.md and phase_ab_analysis.json
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reclassify_grid import CELL_TIMEOUT, FAILURE, NON_RESULT, _integrity, _load_report, classify  # noqa: E402

CONTROL = "converge"
TREATMENT = "hybrid"
ABLATION = "hybrid-nogate"


# ----------------------------------------------------------------- statistics (stdlib only)
def wilson_ci(k: int, n: int, z: float = 1.959963984540054):
    """Wilson score 95% CI for a binomial proportion k/n. Returns (lo, hi) or (None, None)."""
    if n == 0:
        return (None, None)
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _binom_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided p-value for k successes in n Bernoulli(0.5) trials (the small-sample test
    used by both McNemar-exact and the sign test). p = 2 * P(X <= min(k, n-k)); capped at 1.0."""
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar_exact(b: int, c: int) -> dict:
    """Paired binary test. b = #(A solved, B not); c = #(B solved, A not). Discordant n=b+c."""
    n = b + c
    return {"b": b, "c": c, "discordant": n, "p_value": _binom_two_sided_p(c, n)}


def sign_test(diffs: list) -> dict:
    """Paired sign test on continuous differences (treatment - control); ties dropped."""
    pos = sum(1 for d in diffs if d > 1e-12)
    neg = sum(1 for d in diffs if d < -1e-12)
    n = pos + neg
    return {"n_pos": pos, "n_neg": neg, "n_nonzero": n, "p_value": _binom_two_sided_p(pos, n)}


def _med_iqr(xs: list):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return (None, None, None)
    def q(p):
        if len(xs) == 1:
            return xs[0]
        i = p * (len(xs) - 1)
        lo, hi = int(math.floor(i)), int(math.ceil(i))
        return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)
    return (q(0.5), q(0.25), q(0.75))


# ----------------------------------------------------------------- frontier extraction
def _frontier(cell_dir: Path, outcome: str) -> tuple:
    """(best_gold_passed, gold_total, frac). A solved cell is full frontier; otherwise read the
    best partial banked to phase_checkpoint.json (telemetry-only; survives an outer kill)."""
    best_gp, gtot = None, None
    for f in glob.glob(str(cell_dir / "**" / "phase_checkpoint.json"), recursive=True) + \
            glob.glob(str(cell_dir / "phase_checkpoint.json")):
        try:
            d = json.load(open(f))
            gp, gt = int(d.get("gold_passed") or 0), int(d.get("gold_total") or 0)
            if best_gp is None or gp > best_gp:
                best_gp, gtot = gp, gt
        except Exception:
            continue
    if outcome == "solved":
        # an execution-accepted full pass: frontier = 1.0 (gold_total may be from a partial bank)
        return (gtot or best_gp, gtot or best_gp, 1.0)
    if best_gp is None or not gtot:
        return (best_gp, gtot, None)
    return (best_gp, gtot, best_gp / gtot)


# ----------------------------------------------------------------- collect
def collect(ladder: Path) -> list:
    prog_path = ladder / "progress.jsonl"
    if not prog_path.exists():
        return []
    last, relaunched = {}, set()
    for line in open(prog_path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
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
            cell_dir = ladder / f"{label}__{repo}"
        rep = _load_report(cell_dir)
        wall = float(r.get("wall_s") or 0)
        outcome, evidence = classify(label, repo, rep, wall)
        # ROBUSTNESS: a cell that did NO work (crashed/auth-failed before any agent ran — e.g. the
        # pilot's 1.3s auth-preflight race) must be an infra NON-RESULT, never a strategy "gave_up".
        # Guard: a failure verdict with no agents used and a sub-minute wall is a crash, not a fail.
        if outcome in FAILURE and not rep.get("agents_used") and wall < 60:
            outcome, evidence = "infra", f"no work done (wall={wall:.0f}s, 0 agents) — crash/auth, excluded"
        gp, gtot, frac = _frontier(cell_dir, outcome)
        integ = _integrity(cell_dir)
        rows.append({"arm": label, "repo": repo, "seed": seed, "outcome": outcome,
                     "solved": outcome == "solved", "genuine": outcome in (FAILURE | {"solved"}),
                     "nonresult": outcome in NON_RESULT, "wall_s": round(wall, 1),
                     "agents": rep.get("agents_used"), "gold_passed": gp, "gold_total": gtot,
                     "frontier_frac": frac, "relaunched": (label, repo, seed) in relaunched,
                     "integrity": integ["attempts"], "evidence": evidence})
    return rows


# ----------------------------------------------------------------- analyze
def analyze(rows: list) -> dict:
    arms = sorted({r["arm"] for r in rows})
    repos = sorted({r["repo"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    by = {(r["arm"], r["repo"], r["seed"]): r for r in rows}

    # per-arm solve-rate (genuine denom) + Wilson CI
    per_arm = {}
    for a in arms:
        ar = [r for r in rows if r["arm"] == a]
        genuine = [r for r in ar if r["genuine"]]
        solved = [r for r in genuine if r["solved"]]
        lo, hi = wilson_ci(len(solved), len(genuine))
        fr = _med_iqr([r["frontier_frac"] for r in ar if r["frontier_frac"] is not None])
        ag = _med_iqr([r["agents"] for r in solved if r["agents"]])
        per_arm[a] = {
            "cells": len(ar), "genuine": len(genuine), "solved": len(solved),
            "nonresults": sum(1 for r in ar if r["nonresult"]),
            "solve_rate": (len(solved) / len(genuine)) if genuine else None,
            "ci95": [lo, hi], "frontier_median": fr[0], "frontier_iqr": [fr[1], fr[2]],
            "agents_per_solve_median": ag[0], "agents_per_solve_iqr": [ag[1], ag[2]],
            "integrity_attempts": sum(r["integrity"] for r in ar),
        }

    # per (arm, repo): solved/seeds, pass@k, median frontier
    per_cell = {}
    for a in arms:
        for repo in repos:
            cs = [by[(a, repo, s)] for s in seeds if (a, repo, s) in by]
            gen = [c for c in cs if c["genuine"]]
            sol = [c for c in gen if c["solved"]]
            fr = _med_iqr([c["frontier_frac"] for c in cs if c["frontier_frac"] is not None])
            per_cell[f"{a}/{repo}"] = {
                "seeds": len(cs), "genuine": len(gen), "solved": len(sol),
                "pass_at_k": int(len(sol) >= 1), "nonresults": sum(1 for c in cs if c["nonresult"]),
                "frontier_median": fr[0],
                "outcomes": [f"s{c['seed']}:{c['outcome']}" for c in cs],
            }

    # paired tests (matched on repo, seed), per comparison
    def paired(a_ctrl, a_treat):
        per_repo, pooled_b, pooled_c, fdiffs = {}, 0, 0, []
        for repo in repos:
            b = c = 0
            for s in seeds:
                rc, rt = by.get((a_ctrl, repo, s)), by.get((a_treat, repo, s))
                if not rc or not rt or not rc["genuine"] or not rt["genuine"]:
                    continue
                if rc["solved"] and not rt["solved"]:
                    b += 1
                elif rt["solved"] and not rc["solved"]:
                    c += 1
                if rc["frontier_frac"] is not None and rt["frontier_frac"] is not None:
                    fdiffs.append(rt["frontier_frac"] - rc["frontier_frac"])
            per_repo[repo] = mcnemar_exact(b, c)
            pooled_b += b
            pooled_c += c
        return {"control": a_ctrl, "treatment": a_treat, "per_repo": per_repo,
                "pooled_mcnemar": mcnemar_exact(pooled_b, pooled_c),
                "frontier_sign_test": sign_test(fdiffs),
                "frontier_median_diff": _med_iqr(fdiffs)[0]}

    comps = {}
    if CONTROL in arms and TREATMENT in arms:
        comps["hybrid_vs_converge"] = paired(CONTROL, TREATMENT)
    if TREATMENT in arms and ABLATION in arms:
        comps["hybrid_vs_nogate"] = paired(ABLATION, TREATMENT)

    return {"n_seeds": len(seeds), "arms": arms, "repos": repos, "per_arm": per_arm,
            "per_cell": per_cell, "comparisons": comps, "rows": rows}


# ----------------------------------------------------------------- verdict + render
def _pct(x):
    return "-" if x is None else f"{x * 100:.0f}%"


def verdict(an: dict) -> list:
    out = []
    pa = an["per_arm"]
    hv = an["comparisons"].get("hybrid_vs_converge")
    if not hv:
        return ["No converge↔hybrid pairing available (missing arm)."]
    mc = hv["pooled_mcnemar"]
    st = hv["frontier_sign_test"]
    net = mc["c"] - mc["b"]                       # +ve => hybrid solves more than converge
    out.append(f"H1 (solve): hybrid net solves vs converge = {net:+d} "
               f"(discordant b={mc['b']} c={mc['c']}, McNemar exact p={mc['p_value']:.3f}). "
               + ("UNDERPOWERED — too few discordant pairs to conclude." if mc["discordant"] < 6
                  else ("hybrid SUPERIOR." if net > 0 and mc["p_value"] < 0.05
                        else ("hybrid INFERIOR." if net < 0 and mc["p_value"] < 0.05
                              else "no significant difference."))))
    out.append(f"Secondary (frontier): median Δ(hybrid−converge) = "
               f"{('%.3f' % hv['frontier_median_diff']) if hv['frontier_median_diff'] is not None else 'n/a'}, "
               f"sign test p={st['p_value']:.3f} (+{st['n_pos']}/−{st['n_neg']}). "
               + ("hybrid makes MORE graded progress." if st["n_pos"] > st["n_neg"] and st["p_value"] < 0.05
                  else "no significant frontier difference."))
    # cost
    hc = pa.get(TREATMENT, {}).get("agents_per_solve_median")
    cc = pa.get(CONTROL, {}).get("agents_per_solve_median")
    out.append(f"H2 (cost): agents/solve median hybrid={hc} converge={cc}.")
    # gate
    gn = an["comparisons"].get("hybrid_vs_nogate")
    if gn:
        gmc, gst = gn["pooled_mcnemar"], gn["frontier_sign_test"]
        out.append(f"H3 (gate value): hybrid vs hybrid-nogate net solves={gmc['c'] - gmc['b']:+d} "
                   f"(p={gmc['p_value']:.3f}); frontier sign p={gst['p_value']:.3f} "
                   f"(+{gst['n_pos']}/−{gst['n_neg']}). "
                   + ("gate ADDS value." if (gmc['c'] - gmc['b'] > 0 or gst['n_pos'] > gst['n_neg'])
                      else "gate shows NO benefit -> prefer the cheaper gate-off arm."))
    return out


def render(an: dict) -> str:
    L = [f"# Phase-planner A/B — scientific analysis (n={an['n_seeds']} seed(s))\n",
         "Pre-registered: `orchestration_research/EVAL_PROTOCOL.md`. SOLVE = execution-accepted "
         "full-suite pass (C7). Non-results (timeout/infra/token) excluded from denominators. "
         f"TIMEOUT wall = {CELL_TIMEOUT}s.\n",
         "## Per-arm (genuine denom = solved + honest-fail)\n",
         "| arm | solved/genuine | solve-rate | Wilson 95% CI | median frontier | agents/solve (med) | non-results |",
         "|---|---|---|---|---|---|---|"]
    for a in an["arms"]:
        d = an["per_arm"][a]
        ci = d["ci95"]
        ci_s = "-" if ci[0] is None else f"[{_pct(ci[0])}, {_pct(ci[1])}]"
        L.append(f"| {a} | {d['solved']}/{d['genuine']} | {_pct(d['solve_rate'])} | {ci_s} | "
                 f"{_pct(d['frontier_median'])} | {d['agents_per_solve_median'] or '-'} | {d['nonresults']} |")
    L.append("\n## Per (arm, repo) across seeds (solved/genuine, pass@k, median frontier, outcomes)\n")
    L.append("| arm/repo | solved/genuine | pass@k | median frontier | non-results | outcomes |")
    L.append("|---|---|---|---|---|---|")
    for k, c in sorted(an["per_cell"].items()):
        L.append(f"| {k} | {c['solved']}/{c['genuine']} | {'Y' if c['pass_at_k'] else 'n'} | "
                 f"{_pct(c['frontier_median'])} | {c['nonresults']} | {', '.join(c['outcomes'])} |")
    for name, cmp in an["comparisons"].items():
        L.append(f"\n## Paired comparison: {name}\n")
        mc = cmp["pooled_mcnemar"]
        L.append(f"- Pooled McNemar exact: b(control-only solved)={mc['b']}, c(treatment-only "
                 f"solved)={mc['c']}, discordant={mc['discordant']}, two-sided p={mc['p_value']:.3f}")
        fmd = cmp["frontier_median_diff"]
        st = cmp["frontier_sign_test"]
        L.append(f"- Frontier (treatment−control): median Δ="
                 f"{('%.3f' % fmd) if fmd is not None else 'n/a'}, sign test +{st['n_pos']}/"
                 f"−{st['n_neg']} (nonzero {st['n_nonzero']}), p={st['p_value']:.3f}")
        L.append("- per-repo McNemar: " + "; ".join(
            f"{repo} b={v['b']} c={v['c']} p={v['p_value']:.2f}" for repo, v in cmp["per_repo"].items()))
    L.append("\n## Verdict (against pre-registered decision rules)\n")
    L += [f"- {line}" for line in verdict(an)]
    L.append("\n## Power / limitations\n")
    hv = an["comparisons"].get("hybrid_vs_converge", {})
    disc = hv.get("pooled_mcnemar", {}).get("discordant", 0)
    L.append(f"- McNemar discordant pairs (hybrid vs converge) = {disc}. With <6 discordant pairs "
             "the binary test cannot reach p<0.05 — the continuous frontier endpoint carries the "
             "power. n=3 gives DIRECTION, not a statistically-confident ranking (a single hard-repo "
             "flip moves a per-arm rate by 1/genuine). Treat any single-cell swing as variance.")
    L.append("- Stochastic agents + single vendor (codex) + the mimesis fetch-monoculture prior are "
             "shared confounds; they bound external validity, not the within-experiment pairing.")
    return "\n".join(L) + "\n"


def main() -> int:
    ladder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/ladder")
    rows = collect(ladder)
    if not rows:
        print(f"no progress rows under {ladder}")
        return 1
    an = analyze(rows)
    (ladder / "phase_ab_analysis.json").write_text(json.dumps(an, indent=2))
    md = render(an)
    (ladder / "PHASE_AB_ANALYSIS.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
