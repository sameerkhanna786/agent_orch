#!/usr/bin/env python3
"""Careful per-cell FAILURE LEDGER for a commit0 ladder run.

For every NON-SOLVED cell it digs into the artifacts (cell report + journal WAL + integrity log)
and records the SPECIFIC reason it failed, classified so failures can be acted on:

  reason_class:
    infra:<x>           harness/env failure (venv/prep/path) — NOT a capability failure (re-run)
    timeout             wall-clock cut it before the strategy finished (re-run, wall removed)
    collection_error    pytest could not even import/collect (e.g. pydantic-core ABI) — 0 pass / N errors
    expected_id_mismatch tests pass but the EXACT gold ids don't match (e.g. enum repr) — passed>0, total==0
    partial_pass        genuine incomplete implementation — 0 < pass_rate < 1
    no_progress         every attempt produced nothing scorable (all infra/policy/empty)
    confident_wrong     a ship-arm (B0/baseline) returned its best rollout but it was not green
    gave_up             orchestrated arm abstained after using its budget
    unknown             could not determine from artifacts

Each entry also records: best pass_rate reached, passed/failed/errors/total, agents, the agent
finalization mix (completed/policy_violation/infra_nonresult/timeout), recorded escape/cheat
attempts, and a short failure excerpt.

Usage:  python scripts/failure_ledger.py [ladder_dir]   (default runs/ladder)
Env:    LADDER_CELL_TIMEOUT (default 3600) for timeout detection.
Writes: <ladder_dir>/FAILURE_LEDGER.md + failure_ledger.json. Safe on an in-progress run.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

CELL_TIMEOUT = int(os.environ.get("LADDER_CELL_TIMEOUT", "3600"))
SHIP_ARMS = ("B0_", "baseline_", "B2_")


def _load_report(cell_dir: Path) -> dict:
    for pat in ("autogen_cell_report.json", "benchmark_report.json", "autogen_cell_error.json"):
        for f in sorted(glob.glob(str(cell_dir / "**" / pat), recursive=True)) + \
                sorted(glob.glob(str(cell_dir / pat))):
            try:
                return json.load(open(f))
            except Exception:
                continue
    return {}


def _wal_stats(cell_dir: Path) -> dict:
    fin: Counter = Counter()
    scores = []  # (pass_rate, passed, failed, errors, total, indeterminate)
    excerpts = []
    failing = []
    for w in glob.glob(str(cell_dir / "**" / "calls_wal.jsonl"), recursive=True):
        for line in open(w):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("status") != "committed":
                continue
            sr = d.get("structured_result") or {}
            if d.get("kind") == "agent":
                fin[sr.get("finalization_status")] += 1
            elif d.get("kind") == "score":
                v = sr.get("value") or {}
                scores.append((v.get("pass_rate"), v.get("passed"), v.get("failed"),
                               v.get("errors"), v.get("total"), v.get("indeterminate")))
                if v.get("failure_excerpts"):
                    excerpts.append(str(v["failure_excerpts"])[:160])
                for fn in (v.get("failing_nodeids") or [])[:5]:
                    failing.append(str(fn))
    return {"finalization": dict(fin), "scores": scores, "excerpts": excerpts[:3], "failing": failing[:8]}


def _integrity(cell_dir: Path) -> dict:
    counts: Counter = Counter()
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
                counts[s.get("kind")] += 1
    return {"attempts": n, "by_kind": dict(counts)}


def _best_score(scores):
    real = [s for s in scores if not s[5] and isinstance(s[0], (int, float))]
    return max(real, key=lambda s: s[0]) if real else None


def reason_for(arm, repo, rep, wal, integ, wall) -> tuple[str, str]:
    err = rep.get("_error") or rep.get("_orchestration_error")
    is_ship = any(arm.startswith(p) for p in SHIP_ARMS)
    near_wall = bool(wall and wall >= 0.97 * CELL_TIMEOUT)
    scores = wal["scores"]
    best = _best_score(scores)

    if err:
        head = str(err).splitlines()[0][:120]
        low = head.lower()
        kind = ("infra:venv" if ("inspect python" in low or "virtual environment" in low or
                                 "interpreter not found" in low)
                else "infra:prep" if "_prepare_repo" in str(err) or "prepare" in low
                else f"infra:{head[:40]}")
        return kind, f"harness error: {head}"
    if near_wall and not best:
        return "timeout", f"wall {wall:.0f}s ~ cap {CELL_TIMEOUT}s, no scored attempt"
    if best is not None:
        pr, passed, failed, errors, total, _ind = best
        if (passed or 0) == 0 and (errors or 0) > 0 and (total or 0) == 0:
            return "collection_error", (f"pytest could not collect: {errors} errors, 0 passed "
                                        f"(package won't import). excerpt: {(wal['excerpts'] or [''])[0]}")
        if (passed or 0) > 0 and (total or 0) == 0:
            return "expected_id_mismatch", (f"{passed} tests pass but 0 gold ids matched "
                                            f"(exact parametrized-id mismatch, e.g. enum repr)")
        if 0 < (pr or 0) < 1:
            return ("confident_wrong" if is_ship else "partial_pass"), (
                f"best pass_rate={pr:.3f} ({passed}/{total} passed, {failed} failed, {errors} err); "
                f"sample failing: {', '.join(wal['failing'][:4])}")
        if (pr or 0) == 0:
            return ("confident_wrong" if is_ship else "no_progress"), (
                f"best scored attempt pass_rate=0 ({passed} passed / {total} total)")
    # v1 ship-arms (B0/baseline/B2) prep+solve in a subprocess and don't write the apex WAL —
    # fall back to the benchmark report's aggregate pass-rate so the reason is still informative.
    ppct = rep.get("average_pass_rate_percent")
    if ppct is not None and not wal["scores"]:
        pr = float(ppct) / 100.0
        cls = "confident_wrong" if is_ship else ("partial_pass" if pr > 0 else "no_progress")
        return cls, f"v1 best-of-N aggregate pass_rate={pr:.3f} ({ppct}%), not green"
    # no genuine scored attempt
    finmix = wal["finalization"]
    if finmix:
        dom = max(finmix, key=finmix.get)
        if dom == "policy_violation":
            return "no_progress", (f"all attempts blocked: {finmix} (escape/policy denied; "
                                   f"integrity={integ['by_kind']})")
        return "no_progress", f"no scored attempt; finalization mix {finmix}"
    return ("confident_wrong" if is_ship else "gave_up"), "no scored attempt and no winner"


def main() -> int:
    ladder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/ladder")
    prog_path = ladder / "progress.jsonl"
    prog = [json.loads(l) for l in open(prog_path) if l.strip()] if prog_path.exists() else []
    last = {}
    for r in prog:
        if r.get("status") in ("done", "skip"):
            last[(r.get("label"), r.get("repo"), int(r.get("seed", 0) or 0))] = r

    failures = []
    for (arm, repo, seed), r in sorted(last.items()):
        if r.get("solved"):
            continue
        cell = ladder / f"{arm}__{repo}__s{seed}"
        if not cell.exists():
            cell = ladder / f"{arm}__{repo}"
        rep = _load_report(cell)
        wal = _wal_stats(cell)
        integ = _integrity(cell)
        wall = float(r.get("wall_s") or 0)
        rclass, detail = reason_for(arm, repo, rep, wal, integ, wall)
        best = _best_score(wal["scores"])
        failures.append({
            "arm": arm, "repo": repo, "seed": seed, "reason_class": rclass, "reason": detail,
            "best_pass_rate": (round(best[0], 3) if best else None),
            "agents": rep.get("agents_used"), "wall_s": round(wall, 1),
            "finalization": wal["finalization"], "integrity": integ,
        })

    by_class = Counter(f["reason_class"] for f in failures)
    by_class_repo = defaultdict(Counter)
    for f in failures:
        by_class_repo[f["reason_class"]][f"{f['arm']}/{f['repo']}"] += 1

    out = {"cell_timeout": CELL_TIMEOUT, "n_failures": len(failures),
           "by_reason_class": dict(by_class), "failures": failures}
    (ladder / "failure_ledger.json").write_text(json.dumps(out, indent=2))

    md = [f"# Failure ledger — {len(failures)} non-solved cells\n",
          "## Failures by reason-class\n| reason_class | count | cells |", "|---|---|---|"]
    for cls, n in by_class.most_common():
        cells = ", ".join(f"{k}×{v}" for k, v in by_class_repo[cls].most_common())
        md.append(f"| **{cls}** | {n} | {cells[:140]} |")
    md.append("\n## Per-failure detail\n| arm | repo | seed | reason_class | best_pass | agents | finalization | reason |")
    md.append("|---|---|---|---|---|---|---|---|")
    for f in failures:
        md.append(f"| {f['arm']} | {f['repo']} | s{f['seed']} | **{f['reason_class']}** | "
                  f"{f['best_pass_rate']} | {f['agents']} | {f['finalization']} | {f['reason'][:160]} |")
    (ladder / "FAILURE_LEDGER.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:60]))
    print(f"\nwrote {ladder}/FAILURE_LEDGER.md (+ .json) — {len(failures)} failures, classes: {dict(by_class)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
