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
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
# SPFG+ Tier-1: the relaunch/continue gate is FRONTIER-based (gold-pass-COUNT progress),
# not pure journal activity. The shared definition lives in apex_omega.engine.frontier and
# is consumed identically by all three tiers (ladder here, Mode-C governor/context next).
from apex_omega.engine.frontier import (  # noqa: E402
    FrontierOutcome,
    OUTCOME_TO_CUT_REASON,
    frontier_defaults,
    frontier_from_rollouts,
    frontier_from_wal,
    plateau_verdict,
    w_meas_effective,
)

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
# SPFG+ frozen, env-overridable knobs (single-sourced via apex_omega.engine.frontier so the
# ladder gate and the Mode-C governor read the SAME defaults). W_TIME=7200s VALID-measurement
# wall, W_MEAS=12 valid measurements, INDET_CEIL=24 indeterminate ceiling (-> cut:harness-stall),
# POLL_S=300 daemon poll cadence.
W_TIME, W_MEAS, INDET_CEIL, POLL_S = frontier_defaults()
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
    # ===== CONVERGENCE-REBUILD A/B (Phase 3; offline-validated, live run is user-driven) =====
    # Arm A = the EXISTING flat best-of-N default + the Phase-1 flag flips (repair_iters=2,
    # repair excerpts on). Arm B = the REBUILT convergence default (decompose -> fan-out ->
    # reduce -> loop-until-dry on the exact residuals, carrying the best partial forward) via
    # APEX_OMEGA_ORCHESTRATION=converge. Both run the SAME flips so the ONLY variable is the
    # orchestration shape.
    #
    # PROMOTION CRITERIA — promote Arm B over Arm A ONLY if it:
    #   (1) CONVERTS the babel/mimesis near-solves (4598/4607, 6044/6052) to real SOLVES (the
    #       off-by-K class the carry-forward + residual loop targets);
    #   (2) does NOT regress voluptuous/jinja solve-rate OR cost — no 5-6x agent over-spawn
    #       (the decomposition skip-gate keeps easy/<=1-module repos on the cheap path);
    #   (3) stays within the wall-clock budget (run-4 budget blowup must NOT recur — safe only
    #       because the SPFG+ governor stops a true plateau while letting a climbing frontier go).
    # Suggested live A/B:
    #   LADDER_ARMS=omega_flips_unbounded,omega_converge_unbounded \
    #   LADDER_REPOS=voluptuous,jinja,mimesis,babel LADDER_SEEDS=3 python scripts/run_ladder.py
    ("omega_flips_unbounded",    ["--arms", "autogen_orchestrator", "--autogen-scout-agents", "0",
                                  "--autogen-max-agents", _OMEGA_MAX],
                                 {"APEX_OMEGA_REPAIR_ITERS": "2", "APEX_OMEGA_REPAIR_EXCERPTS": "1"}),
    ("omega_converge_unbounded", ["--arms", "autogen_orchestrator", "--autogen-scout-agents", "0",
                                  "--autogen-max-agents", _OMEGA_MAX],
                                 {"APEX_OMEGA_ORCHESTRATION": "converge",
                                  "APEX_OMEGA_REPAIR_ITERS": "2", "APEX_OMEGA_REPAIR_EXCERPTS": "1"}),
    # ===== PHASE-PLANNER A/B (the Claude-Code-style hybrid; offline-validated, live run user-driven) =====
    # All three carry the SAME repair flips as omega_converge_unbounded so the ONLY variable is the
    # orchestration shape (apples-to-apples). short labels match orchestration_research/DECISION.md.
    #   A converge      (control/bar)  = the incumbent decompose->fan-out->reduce->loop-until-dry.
    #   B hybrid        (treatment)    = host-side ordered phases + per-phase scoped converge +
    #                                    partial-frontier checkpoint + adversarial goal-alignment gate.
    #   C hybrid-nogate (ablation)     = B with the goal gate OFF (does the no-veer review earn its agents?).
    #   D hybrid-codegen(ablation,opt) = B + per-phase generated orchestration (run ONLY if B/C plateau).
    ("converge",      ["--arms", "autogen_orchestrator", "--autogen-scout-agents", "0",
                       "--autogen-max-agents", _OMEGA_MAX],
                      {"APEX_OMEGA_ORCHESTRATION": "converge",
                       "APEX_OMEGA_REPAIR_ITERS": "2", "APEX_OMEGA_REPAIR_EXCERPTS": "1"}),
    ("hybrid",        ["--arms", "autogen_orchestrator", "--autogen-scout-agents", "0",
                       "--autogen-max-agents", _OMEGA_MAX],
                      {"APEX_OMEGA_ORCHESTRATION": "hybrid", "APEX_OMEGA_PHASE_PLANNER": "1",
                       "APEX_OMEGA_REPAIR_ITERS": "2", "APEX_OMEGA_REPAIR_EXCERPTS": "1"}),
    ("hybrid-nogate", ["--arms", "autogen_orchestrator", "--autogen-scout-agents", "0",
                       "--autogen-max-agents", _OMEGA_MAX],
                      {"APEX_OMEGA_ORCHESTRATION": "hybrid", "APEX_OMEGA_PHASE_PLANNER": "1",
                       "APEX_OMEGA_GOAL_GATE": "0",
                       "APEX_OMEGA_REPAIR_ITERS": "2", "APEX_OMEGA_REPAIR_EXCERPTS": "1"}),
    # ===== DIAGNOSIS REDESIGN A/B (O2/O3/O4; the user's "review at every planning stage + diagnose
    # the real cause" directive). hybrid-diag = hybrid-nogate (the OLD BEST: gate OFF) PLUS the new
    # gated redesign: ctx.diagnose() (AST collection pre-pass + fact-checked scouts), ctx.review_plan()
    # (advisory bounded plan review at every seam, grounded in the diagnosis), and the synthetic
    # make-it-collect Phase 0. The ONLY variable vs hybrid-nogate is the three redesign gates, so this
    # is the clean 2-arm test. Suggested run (no mid-cell kills):
    #   LADDER_ARMS=hybrid-nogate,hybrid-diag LADDER_REPOS=babel,mimesis,pydantic,networkx \
    #   LADDER_SEEDS=3 python scripts/run_ladder.py
    ("hybrid-diag",   ["--arms", "autogen_orchestrator", "--autogen-scout-agents", "0",
                       "--autogen-max-agents", _OMEGA_MAX],
                      {"APEX_OMEGA_ORCHESTRATION": "hybrid", "APEX_OMEGA_PHASE_PLANNER": "1",
                       "APEX_OMEGA_GOAL_GATE": "0",
                       "APEX_OMEGA_DIAG": "1", "APEX_OMEGA_PLAN_REVIEW": "1", "APEX_OMEGA_PHASE0": "1",
                       "APEX_OMEGA_REPAIR_ITERS": "2", "APEX_OMEGA_REPAIR_EXCERPTS": "1"}),
    # ===== SARP A/B (last-mile fix) — "the new version" vs the A/B winner (hybrid-diag). The ONLY
    # variable is APEX_OMEGA_SARP: on a sterile near-solve plateau it diagnoses the residual gap's
    # direction (read-only scouts) + re-aims with failure excerpts before the governor cuts. Tests
    # whether the last-mile gap closes (mimesis 6110->6159, babel 5655->5663). Suggested:
    #   LADDER_ARMS=hybrid-diag,hybrid-diag-sarp LADDER_REPOS=mimesis,babel LADDER_SEEDS=2 \
    #   python scripts/run_ladder.py
    ("hybrid-diag-sarp", ["--arms", "autogen_orchestrator", "--autogen-scout-agents", "0",
                          "--autogen-max-agents", _OMEGA_MAX],
                         {"APEX_OMEGA_ORCHESTRATION": "hybrid", "APEX_OMEGA_PHASE_PLANNER": "1",
                          "APEX_OMEGA_GOAL_GATE": "0",
                          "APEX_OMEGA_DIAG": "1", "APEX_OMEGA_PLAN_REVIEW": "1", "APEX_OMEGA_PHASE0": "1",
                          "APEX_OMEGA_SARP": "1",
                          "APEX_OMEGA_REPAIR_ITERS": "2", "APEX_OMEGA_REPAIR_EXCERPTS": "1"}),
    ("hybrid-codegen", ["--arms", "autogen_orchestrator", "--autogen-scout-agents", "0",
                        "--autogen-max-agents", _OMEGA_MAX],
                       {"APEX_OMEGA_ORCHESTRATION": "hybrid", "APEX_OMEGA_PHASE_PLANNER": "1",
                        "APEX_OMEGA_PHASE_CODEGEN": "1",
                        "APEX_OMEGA_REPAIR_ITERS": "2", "APEX_OMEGA_REPAIR_EXCERPTS": "1"}),
    # ralph alias (short label matching the DECISION eval plan): vanilla persistence floor.
    ("ralph",         ["--arms", "autogen_orchestrator", "--autogen-scout-agents", "0",
                       "--autogen-max-agents", _OMEGA_MAX],
                      {"APEX_OMEGA_ORCHESTRATION": "ralph", "APEX_OMEGA_REPAIR_ITERS": "200"}),
]
# Default 4-repo comparison set. LADDER_REPOS (comma-separated commit0 target names) overrides it
# for a custom sweep (e.g. a 12-15 repo breadth run); see apex_omega/eval/registry TARGET_NAMES.
REPOS = (
    [r.strip() for r in os.environ["LADDER_REPOS"].split(",") if r.strip()]
    if os.environ.get("LADDER_REPOS") else ["voluptuous", "jinja", "mimesis", "pydantic"]
)
# Each EXTRA cell may carry an env overlay (4th element) merged into the child env.
# (The gold-test design-contract A/B arm was removed: the contract is gone from the evaluated
# path entirely — the agent gets only the commit0 prompt + gold tests — so there is nothing to
# A/B against. The env-overlay mechanism is retained for future legitimate per-cell variants.)
EXTRA = [
    # expensive cost-pathology witness: one fast repo only
    ("B2_v1_fullcap16", ["--arms", "B2_v1_full_cap16"], ["voluptuous"], {}),
]

# LADDER_ARMS (comma-separated arm labels) selects + ORDERS a subset of ARMS for a custom run,
# e.g. "omega_autogen_unbounded" alone, or "omega_autogen_unbounded,omega_template_unbounded".
# When set, the B2 EXTRA cells are kept only if their label is named (custom runs target the
# named arms). Combine with LADDER_REPOS for a focused arm x repo sweep.
_arm_sel = os.environ.get("LADDER_ARMS")
if _arm_sel:
    _order = [a.strip() for a in _arm_sel.split(",") if a.strip()]
    _by_label = {a[0]: a for a in ARMS}
    ARMS = [_by_label[name] for name in _order if name in _by_label]
    EXTRA = [e for e in EXTRA if e[0] in _order]


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


def _recover_partial_frontier(rundir: Path):
    """Phase-planner addendum: surface the strongest PARTIAL/phase frontier banked to
    phase_checkpoint.json. TELEMETRY ONLY — a partial is NEVER a solve (accepted is always False);
    only accepted_checkpoint.json yields solved:1 (Cardinal Contract C7). Returns the best partial
    record (highest gold_passed) or None. Lets a relaunch/audit see the off-by-K progress that
    survived an outer kill without ever inflating the solve-rate."""
    best = None
    for p in list(rundir.rglob("phase_checkpoint.json")):
        try:
            d = json.loads(p.read_text())
            if d.get("accepted"):          # defensive: a partial must never claim acceptance
                continue
            if best is None or int(d.get("gold_passed", 0) or 0) > int(best.get("gold_passed", 0) or 0):
                best = d
        except Exception:
            continue
    return best


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


def _arm_budget(label: str, flags: list[str], rundir: Path) -> int | None:
    """The arm's finite ATTEMPT budget (for w_meas_effective fairness scaling), or None
    for unbounded orchestrators. Derived from the cell flags (the same source the cell was
    launched with) with a Mode-A benchmark_state fallback.

      - B0 1-shot                  -> 1   (w_meas_eff floor 3 but <3 valid measurements ever
                                            produced, so NEVER plateau-cuttable)
      - best-of-N (--rollouts N)   -> N   (best-of-8 -> w_meas_eff 5)
      - B2 full-cap16              -> 16  (-> w_meas_eff 10)
      - autogen orchestrators / ralph (--autogen-max-agents <big>) -> None (unbounded)
      - undeterminable             -> None (conservative: keeps the global window)
    """
    arms = []
    rollouts = None
    autogen = False
    it = iter(flags)
    for tok in it:
        if tok == "--arms":
            arms.append(next(it, ""))
        elif tok == "--rollouts":
            try:
                rollouts = int(next(it, ""))
            except (TypeError, ValueError):
                rollouts = None
        elif tok == "--autogen-max-agents":
            autogen = True
            next(it, None)
    arms_s = " ".join(arms)
    if autogen or "autogen_orchestrator" in arms_s:
        return None                              # unbounded omega / ralph
    if "B0_single_model" in arms_s:
        return 1
    if "B2_v1_full_cap16" in arms_s:
        return 16
    if rollouts is not None:
        return rollouts
    # Mode-A fallback: max_rollouts * candidates_per_rollout from benchmark_state metadata.
    for bs in Path(rundir).rglob("benchmark_state.json"):
        try:
            d = json.loads(bs.read_text())
        except (OSError, ValueError):
            continue
        alloc = (((d.get("metadata") or {}).get("ablation_config") or {}).get("allocator") or {})
        mr = alloc.get("max_rollouts")
        if isinstance(mr, int) and mr > 0:
            return mr
    return None


def frontier_state(rundir: Path):
    """The SPFG+ progress reconstruction for either mode (Tier-1 wraps BOTH).

    Mode-C (an in-process orchestrator run) is detected by the presence of a
    ``calls_wal.jsonl`` journal; otherwise the cell is a Mode-A best-of-N subprocess and the
    frontier is read from rollout_status / candidate_scorecard artifacts. Returns a
    ``FrontierState`` whose ``.as_state()`` feeds ``plateau_verdict``.
    """
    if _has_journal(rundir):
        return frontier_from_wal(rundir)
    return frontier_from_rollouts(rundir)


def relaunch_decision(rundir: Path, label: str, flags: list[str], attempt: int,
                      last_prog: int | None):
    """The frontier-aware relaunch/continue gate (replaces the activity prog<=last_prog cut).

    Returns ``(action, reason, frontier, fs)`` where action is one of:
      - "relaunch"   : the gold frontier ROSE since the last attempt (still-progressing,
                       e.g. the minitorch case whose pass-count was climbing) -> warm-resume.
      - "plateau"    : a GENUINE no-solve-progress plateau (cut:no-progress).
      - "harness"    : an indeterminate/harness wall (cut:harness-stall).
      - "exhausted"  : LADDER_MAX_RELAUNCH backstop reached while not plateau-cut.

    A rising frontier ALWAYS wins (relaunch) regardless of wall-time — SPFG+ never bounds
    total wall-time; it cuts only on a dual-AND plateau or a harness wall.
    """
    fs = frontier_state(rundir)
    frontier = int(fs.gold_frontier)
    rose = (last_prog is None) or (frontier > last_prog)
    budget = _arm_budget(label, flags, rundir)
    w_meas_eff = w_meas_effective(W_MEAS, budget)
    outcome, why = plateau_verdict(fs.as_state(), w_meas_eff, W_TIME, INDET_CEIL)
    if rose:
        return ("relaunch", f"frontier rose to {frontier}", frontier, fs)
    if outcome == FrontierOutcome.PLATEAU_CUT.value:
        return ("plateau", f"{OUTCOME_TO_CUT_REASON[outcome]}: {why}", frontier, fs)
    if outcome == FrontierOutcome.INDETERMINATE_CUT.value:
        return ("harness", f"{OUTCOME_TO_CUT_REASON[outcome]}: {why}", frontier, fs)
    if attempt >= LADDER_MAX_RELAUNCH:
        return ("exhausted", f"{LADDER_MAX_RELAUNCH} relaunches exhausted (frontier={frontier}, "
                f"within window: {why})", frontier, fs)
    return ("relaunch", f"within window, warm resume (frontier={frontier}; {why})", frontier, fs)


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
    # Backbone 1.3 + SPFG+: PAUSE+RESUME instead of guillotine. On a kill, RELAUNCH against
    # the same warm --run-dir (the child reattaches the journal and replays the prior prefix
    # as cache HITs). The relaunch/continue DECISION is now FRONTIER-based: relaunch while the
    # gold-pass COUNT is climbing (still-progressing, e.g. minitorch's pass-count rose over
    # 6 relaunches and should NOT have been force-cut), and cut ONLY on a genuine dual-AND
    # no-progress plateau (cut:no-progress) or a sustained indeterminate/harness wall
    # (cut:harness-stall) — never on pure inactivity. A verified solve banked before any kill
    # is recovered FIRST, never discarded.
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
            action, why, frontier, fs = relaunch_decision(
                rundir, label, flags, attempt, last_prog)
            if action == "relaunch":
                last_prog = frontier
                emit("relaunch", {"attempt": attempt + 1, "gold_frontier": frontier,
                                  "valid_measurements": fs.valid_measurements, "reason": why})
                # do NOT _strip_checkout between attempts (keep journal + venv warm for resume)
                continue
            # A cut (plateau / harness-stall) or relaunch-budget exhaustion. ALWAYS consult the
            # acceptance checkpoint FIRST so a banked verified solve is never discarded (run-4
            # data-loss bug) even when the frontier gate decided to stop.
            ckpt = _recover_checkpoint(rundir)
            if ckpt:
                emit("done", {
                    "solved": 1, "total": 1, "pass_pct": 100.0,
                    "wall_s": round(time.monotonic() - t0, 1), "recovered_from_checkpoint": True,
                    "candidate_id": ckpt.get("candidate_id"), "gold_frontier": frontier,
                    "_note": f"killed ({type(exc).__name__}) after {attempt} relaunch(es); "
                             f"verified solve recovered from journal despite {action}"})
                return
            base = {"error": f"{type(exc).__name__}: {exc} ({why})"[:200],
                    "outcome": OUTCOME_TO_CUT_REASON.get(
                        FrontierOutcome.PLATEAU_CUT.value if action == "plateau"
                        else FrontierOutcome.INDETERMINATE_CUT.value, action),
                    "gold_frontier": frontier, "valid_measurements": fs.valid_measurements,
                    "indeterminate_total": fs.indeterminate_total,
                    "seconds_since_frontier_improved": fs.as_state()[
                        "seconds_since_frontier_improved"]}
            _pf = _recover_partial_frontier(rundir)        # phase-planner PARTIAL banked (telemetry only)
            if _pf:
                base["partial_frontier"] = int(_pf.get("gold_passed", 0) or 0)
                base["partial_frontier_total"] = int(_pf.get("gold_total", 0) or 0)
            if action == "plateau":
                emit("plateau_cut", base)
            elif action == "harness":
                emit("indeterminate", base)
            else:  # exhausted relaunch backstop (frontier still within window)
                base["outcome"] = "exhausted-relaunch-backstop"
                emit("error", base)
            return
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
        else:
            _pf = _recover_partial_frontier(rundir)        # PARTIAL frontier (telemetry only; never a solve)
            if _pf:
                res["partial_frontier"] = int(_pf.get("gold_passed", 0) or 0)
                res["partial_frontier_total"] = int(_pf.get("gold_total", 0) or 0)
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
