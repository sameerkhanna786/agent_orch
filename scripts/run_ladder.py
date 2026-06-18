#!/usr/bin/env python3
"""Resumable, bounded-parallel comparison-ladder runner for APEX-Ω commit0 eval.

Runs each (arm, repo) cell as its own `apex_omega eval` subprocess into its own
run-dir, so cells are isolated and the matrix SURVIVES a laptop close: completed
cells (a report exists) are SKIPPED on re-run; only incomplete cells re-run. Just
re-invoke this script to resume.

Order is repo-major with fast repos first, so a complete ladder *row* lands early
(an interpretable comparison) before the slow repos finish.

  Resume:   PYTHONPATH=<repo> <venv-python> scripts/run_ladder.py
  Progress: runs/ladder/progress.jsonl   (one line per cell completion)
  Summary:  runs/ladder/ladder_report.json
  Tunable:  LADDER_CONCURRENCY (default 3), APEX_OMEGA_PYTHON
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENV = os.environ.get("APEX_OMEGA_PYTHON", sys.executable)
# LADDER_DIR is overridable so a seed sweep runs each seed into its own resumable tree
# (e.g. LADDER_DIR=runs/ladder_s0, _s1, _s2 for n>=3 seeds + pass@k across them).
LADDER_DIR = Path(os.environ.get("LADDER_DIR") or (REPO / "runs" / "ladder")).resolve()
PROGRESS = LADDER_DIR / "progress.jsonl"
CONCURRENCY = int(os.environ.get("LADDER_CONCURRENCY", "2"))
# n seeds per cell: each (arm,repo) runs LADDER_SEEDS independent stochastic samples into its
# own __s{seed} run-dir, all in ONE concurrency-bounded pool (so a 5-seed sweep keeps the same
# peak load as 1 seed, just more cells). Gives per-arm variance + pass@k instead of n=1.
SEEDS = max(1, int(os.environ.get("LADDER_SEEDS", "1")))
# 3600 (was 2400): heavy single-solves (mimesis ~1430-1921s, pydantic heavier) were
# clipped at exactly 2400s under C=6 contention (B0-mimesis, baseline-pydantic in run-2).
# FAIR wall (2026-06-17): 24h, not 3600s. The 3600s wall truncated the heavy Mode-A best-of-8
# arms mid-work (B0/baseline jinja still grinding in 'patcher', pass-rate climbing) while the lean
# Mode-C arms finished in <1100s — an UNFAIR guillotine. 86400 is so generous no arm truncates
# mid-work, yet it is a FINITE NUMBER (not 0/None) so every downstream cap stays well-defined and
# BOUNDED: the per-agent watchdog (context.py min(_pa, t)=_pa), eval_cap (commit0_autogen), the
# Mode-A per-step caps, and the outer subprocess backstop (CELL_TIMEOUT+600). Each arm now runs to
# its NATURAL cap (best-of-8 completion / plateau governor); report agents/solve + wall as cost.
CELL_TIMEOUT = int(os.environ.get("LADDER_CELL_TIMEOUT", "86400"))
# Disk safety: each cell writes ~300M of per-rollout repo checkouts. On a near-full
# disk that causes ENOSPC, which corrupts cells (fast-fails that look like real
# failures). So we (a) refuse to start a cell unless there's headroom, and (b) strip
# each cell down to evidence (reports/narration/diffs) the moment it finishes.
MIN_FREE_MB = int(os.environ.get("LADDER_MIN_FREE_MB", "1200"))
# Backbone 1.3: the outer subprocess timeout becomes a PAUSE trigger, not a guillotine.
# A killed cell is RELAUNCHED against the same --run-dir (warm journal+venv) so it resumes
# from the journal — UP TO this many times, and only while it keeps making journal progress
# (so an adversarial/stuck cell still terminates). This replaces run-4's discard-on-timeout.
LADDER_MAX_RELAUNCH = int(os.environ.get("LADDER_MAX_RELAUNCH", "8"))
_KEEP_SUFFIXES = (".json", ".jsonl", ".md", ".diff", ".patch", ".txt", ".log")
_lock = threading.Lock()


def _free_mb() -> int:
    st = os.statvfs(str(LADDER_DIR))
    return int(st.f_bavail * st.f_frsize / (1024 * 1024))


def _strip_checkout(rundir: Path) -> None:
    """Delete bulky source checkouts, KEEP evidence — bounds the disk footprint to
    roughly one in-flight cell at a time so the matrix can't refill the disk.

    CRITICAL: never ``chmod`` a symlink. ``Path.chmod()`` follows the link to its
    TARGET, and a finished cell's runtime venv (``runtime/.venv/bin/python*``)
    symlinks into the SHARED uv-managed interpreter. chmod-ing those would set the
    real python to 0o600 (no execute), breaking ``_build_runtime_env`` —
    "Failed to query Python interpreter ... Permission denied (os error 13)" — for
    EVERY later cell. So symlinks are only unlinked (the link, not the target);
    only real files are chmod+unlinked. (This was a recurring, concurrency-amplified
    bug: the first cell to finish clobbered the shared interpreter for the rest.)"""
    for p in rundir.rglob("*"):
        if p.is_symlink():
            # Remove the link itself; do NOT chmod (that follows to the target).
            try:
                p.unlink()
            except OSError:
                pass
            continue
        if p.is_file():
            # Keep the ENTIRE journal/ subtree (WAL + diff blobs + the frozen
            # orchestrator <sha>.py) — it is evidence AND the basis for resume; stripping
            # the frozen .py made load_frozen() crash on resume (verified live bug).
            try:
                if "journal" in p.relative_to(rundir).parts:
                    continue
            except ValueError:
                pass
        if p.is_file() and p.suffix not in _KEEP_SUFFIXES:
            try:
                p.chmod(0o600)
                p.unlink()
            except OSError:
                pass
    for p in sorted(rundir.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if p.is_dir() and not p.is_symlink():
            try:
                p.rmdir()
            except OSError:
                pass

# Baselines stay FIXED (the v1 comparators): B0 = 1-shot, baseline = best-of-8. The OMEGA arms run
# UNBOUNDED (user directive 2026-06-17): no fixed agent cap — the run-to-completion governor decides
# when to stop (plateau-stop after k dry waves with no pass-rate improvement + the 1000-agent
# backstop). This is the backbone's default-unbounded design; K=8 was an equal-budget constraint we
# removed so omega can "spin up as many agents as it needs". NOTE: this is no longer equal-total-
# budget vs the fixed best-of-8 baseline, so agents_used (cost) is reported alongside solve-rate.
_OMEGA_MAX = "1000"   # = the hard agent ceiling; the plateau governor is the real terminator
# An ARMS entry is (label, flags) or (label, flags, env_overlay). The optional 3rd element is
# a per-cell child-env overlay (e.g. the ralph baseline's mode switch).
ARMS = [
    ("B0_codex_1shot",          ["--arms", "B0_single_model"]),
    ("baseline_v1_k8",          ["--arms", "baseline", "--rollouts", "8"]),
    # RALPH-WIGGUM baseline: a "vanilla" CLI in a dumb iterate-until-done loop (ONE sequential
    # lineage, fed the failing tests each turn, NO scout/author/patterns) with a large budget,
    # governed by the SAME cut-losses detector as omega — to see how naive persistence with a
    # big agent budget compares. Flips the autogen path to the frozen ralph workflow via env.
    ("ralph_wiggum_loop",       ["--arms", "autogen_orchestrator", "--autogen-scout-agents", "0",
                                 "--autogen-max-agents", _OMEGA_MAX],
                                {"APEX_OMEGA_ORCHESTRATION": "ralph", "APEX_OMEGA_REPAIR_ITERS": "200"}),
    ("omega_template_unbounded", ["--arms", "autogen_orchestrator", "--autogen-scout-agents", "0",
                                  "--autogen-max-agents", _OMEGA_MAX]),
    ("omega_autogen_unbounded",  ["--arms", "autogen_orchestrator", "--autogen-scout-agents", "3",
                                  "--autogen-author", "--autogen-max-agents", _OMEGA_MAX]),
]
# fast -> slow so complete rows land early. networkx (needs Docker) + cookiecutter
# (Docker/heavy) dropped per decision: the 4-repo ladder is a sufficient comparison.
REPOS = ["voluptuous", "jinja", "mimesis", "pydantic"]
# Each EXTRA cell may carry an env overlay (4th element) merged into the child env.
# (The gold-test design-contract A/B arm was removed: the contract is gone from the evaluated
# path entirely — the agent gets only the commit0 prompt + gold tests — so there is nothing to
# A/B against. The env-overlay mechanism is retained for future legitimate per-cell variants.)
EXTRA = [
    # expensive cost-pathology witness: one fast repo only
    ("B2_v1_fullcap16", ["--arms", "B2_v1_full_cap16"], ["voluptuous"], {}),
]


def _rundir(label: str, repo: str, seed: int) -> Path:
    # seed-suffixed only when running >1 seed, so single-seed runs keep back-compat naming.
    return LADDER_DIR / (f"{label}__{repo}__s{seed}" if SEEDS > 1 else f"{label}__{repo}")


def cells() -> list[tuple[str, list[str], str, dict, int]]:
    out = []
    for seed in range(SEEDS):                # seed-major: a full matrix lands per seed
        for repo in REPOS:                   # repo-major within a seed (fast -> slow)
            for entry in ARMS:               # (label, flags) or (label, flags, env_overlay)
                label, flags = entry[0], entry[1]
                env = entry[2] if len(entry) > 2 else {}
                out.append((label, flags, repo, dict(env), seed))
        for label, flags, repos, env in EXTRA:
            for repo in repos:
                out.append((label, flags, repo, dict(env), seed))
    return out


def _reports(rundir: Path) -> list[Path]:
    return list(rundir.rglob("autogen_cell_report.json")) + list(rundir.rglob("benchmark_report.json"))


def cell_done(rundir: Path) -> bool:
    for f in _reports(rundir):
        try:
            d = json.loads(f.read_text())
            # A Mode-C autogen_cell_report.json has NO "completed" key (completed=None) and is
            # written only at cell end, so solved_tasks present => done. A Mode-A
            # benchmark_report.json sets "completed" EXPLICITLY; a partial/crashed cell can leave
            # completed=False with solved_tasks=0 — that must NOT count as done (else on resume it
            # is skipped and recorded as a FALSE 0/total failure, poisoning the denominator).
            if d.get("completed") is True or ("solved_tasks" in d and d.get("completed") is None):
                return True
        except Exception:
            continue
    return False


def parse_result(rundir: Path) -> dict:
    for f in _reports(rundir):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        return {"solved": d.get("solved_tasks"), "total": d.get("total_tasks"),
                "pass_pct": d.get("average_pass_rate_percent"),
                "dur_s": round(float(d.get("duration_seconds") or 0), 1),
                "agents": d.get("agents_used"), "difficulty": d.get("difficulty")}
    return {}


def _emit(label: str, repo: str, status: str, res: dict, seed: int = 0) -> None:
    rec = {"label": label, "repo": repo, "seed": seed, "status": status,
           "ts": int(time.time()), **(res or {})}
    with _lock:
        with PROGRESS.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    print("CELL " + json.dumps(rec), flush=True)


def _recover_checkpoint(rundir: Path):
    """Tier-1.1: if a cell was killed (subprocess timeout) AFTER an attempt banked a
    verified accepted solve, recover it from the acceptance checkpoint so a real solve
    is not discarded (run-4 lost verified mimesis 6052/6052 passes to this exact
    timeout-kill). Returns the checkpoint record (accepted) or None."""
    for p in list(rundir.rglob("accepted_checkpoint.json")):
        try:
            d = json.loads(p.read_text())
            if d.get("accepted"):
                return d
        except Exception:
            continue
    return None


def _has_journal(rundir: Path) -> bool:
    return any(rundir.rglob("calls_wal.jsonl"))


def _journal_progress(rundir: Path):
    """Monotonic (committed_ok, max_seq) across the run's journals. Grows ONLY when a
    relaunch does FRESH work (a HIT replays without re-appending), so it is the
    progress signal that decides whether another relaunch is worthwhile (1.3)."""
    best = (0, 0)
    for wal in rundir.rglob("calls_wal.jsonl"):
        try:
            n_ok, last_seq = 0, 0
            for line in wal.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                last_seq = max(last_seq, int(d.get("seq", 0)))
                if d.get("status") == "committed" and d.get("result_status") == "ok":
                    n_ok += 1
            best = max(best, (n_ok, last_seq))
        except OSError:
            continue
    return best


def run_cell(label: str, flags: list[str], repo: str, env_overlay: dict | None = None,
             seed: int = 0) -> None:
    rundir = _rundir(label, repo, seed)

    def emit(status: str, res: dict) -> None:
        _emit(label, repo, status, res, seed=seed)

    if cell_done(rundir):
        emit("skip", parse_result(rundir))
        return
    free = _free_mb()
    if free < MIN_FREE_MB:
        # Refuse to run rather than corrupt the cell with an ENOSPC fast-fail.
        emit("skip_diskfull", {"free_mb": free, "min_free_mb": MIN_FREE_MB})
        return
    rundir.mkdir(parents=True, exist_ok=True)
    cmd = [VENV, "-m", "apex_omega", "eval", *flags, "--repos", repo, "--limit", "1",
           "--run-dir", str(rundir), "--cell-timeout", str(CELL_TIMEOUT)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    # 0-token blocker fix (2026-06-17, validated voluptuous 1/1): the rollout codex was
    # double-sandboxed (outer sandbox-exec read-jail -> "os error 2" ENOENT) sitting on top of
    # the Meta codex launcher's own seatbelt ("os error 1" EPERM, in-process app-server) -> every
    # agent returned 0 tokens / infra_nonresult. Run codex FULLY UNSANDBOXED for this LOCAL eval:
    # disable the outer read-jail AND take the bypass branch (which now also passes
    # --dangerously-disable-osx-sandbox, the launcher flag). The git worktree + container
    # sanitizer provide the isolation that matters for the benchmark. setdefault so an operator
    # who wants the credential read-jail back can set APEX_HOST_CLI_READ_JAIL=1.
    env.setdefault("APEX_HOST_CLI_READ_JAIL", "0")
    env.setdefault("APEX_CODEX_BYPASS_SANDBOX", "1")
    if env_overlay:                           # per-cell overlay (e.g. design-contract A/B)
        env.update(env_overlay)
    t0 = time.monotonic()
    # Backbone 1.3: PAUSE+RESUME instead of guillotine. On a kill, RELAUNCH against the
    # same warm --run-dir (the child reattaches the journal and replays the prior prefix
    # as cache HITs) — up to LADDER_MAX_RELAUNCH times and only while journal progress
    # advances. A verified solve banked before any kill is recovered, never discarded.
    last_prog = None
    completed = False
    for attempt in range(1 + LADDER_MAX_RELAUNCH):
        try:
            subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True,
                           timeout=CELL_TIMEOUT + 600)
            completed = True
            break  # child returned on its own -> parse the report
        except Exception as exc:
            if not _has_journal(rundir):
                emit("error", {"error": f"{type(exc).__name__}: {exc}"[:200]})
                return
            prog = _journal_progress(rundir)
            stuck = last_prog is not None and prog <= last_prog
            if attempt >= LADDER_MAX_RELAUNCH or stuck:
                ckpt = _recover_checkpoint(rundir)
                if ckpt:
                    emit("done", {
                        "solved": 1, "total": 1, "pass_pct": 100.0,
                        "wall_s": round(time.monotonic() - t0, 1), "recovered_from_checkpoint": True,
                        "candidate_id": ckpt.get("candidate_id"),
                        "_note": f"killed ({type(exc).__name__}) after {attempt} relaunch(es); "
                                 "verified solve recovered from journal"})
                    return
                why = "no journal progress" if stuck else f"{LADDER_MAX_RELAUNCH} relaunches exhausted"
                emit("error",
                      {"error": f"{type(exc).__name__}: {exc} ({why})"[:200]})
                return
            last_prog = prog
            emit("relaunch", {"attempt": attempt + 1, "progress": list(prog)})
            # do NOT _strip_checkout between attempts (keep the journal + venv warm for resume)
    res = parse_result(rundir)
    res["wall_s"] = round(time.monotonic() - t0, 1)
    # review-fix #8 (belt-and-suspenders): a clean child completion that reports unsolved but
    # left a verified-accept checkpoint on disk (a banked accept dropped by a post-select
    # crash) is recovered, not lost — the checkpoint used to be consulted only on a kill.
    if not res.get("solved"):
        ckpt = _recover_checkpoint(rundir)
        if ckpt:
            res.update({"solved": 1, "total": res.get("total") or 1, "pass_pct": 100.0,
                        "recovered_from_checkpoint": True, "candidate_id": ckpt.get("candidate_id"),
                        "_note": "verified accept recovered from checkpoint on clean completion"})
    # Strip the bulky checkout immediately (only once a report exists, so a
    # report-less crash keeps its artifacts for debugging).
    if cell_done(rundir):
        _strip_checkout(rundir)
        res["free_mb_after"] = _free_mb()
    emit("done", res)


def main() -> int:
    LADDER_DIR.mkdir(parents=True, exist_ok=True)
    todo = cells()
    pending = [c for c in todo if not cell_done(_rundir(c[0], c[2], c[4]))]
    print(f"ladder: {len(todo)} cells total ({SEEDS} seed(s)), {len(pending)} pending, "
          f"concurrency {CONCURRENCY}", flush=True)
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(run_cell, *c) for c in todo]
        for f in as_completed(futs):
            f.result()
    # aggregate per arm
    agg: dict[str, dict] = {}
    if PROGRESS.exists():
        for line in PROGRESS.read_text().splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            a = agg.setdefault(d["label"], {"cells": 0, "solved": 0, "total": 0})
            a["cells"] += 1
            a["solved"] += int(d.get("solved") or 0)
            a["total"] += int(d.get("total") or 0)
    for a in agg.values():
        a["solve_rate"] = (a["solved"] / a["total"]) if a["total"] else None
    (LADDER_DIR / "ladder_report.json").write_text(json.dumps(agg, indent=2))
    print("LADDER DONE " + json.dumps(agg), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
