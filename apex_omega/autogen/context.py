"""OrchestrationContext — the capability-restricted API a *generated* orchestrator
may call (plan §7.3 freeze-then-journal; §2 orchestration-as-code).

The generated code gets full Python control flow (loops, conditionals,
decomposition, 1000s of agents) but can ONLY touch this object — no filesystem,
no imports, no subprocess, and crucially **no way to mark a candidate accepted**.
Acceptance is engine-owned and execution-grounded: the generated strategy decides
*where compute goes*; the kernel decides *what passes*.  That split is what keeps
"task completion above all" honest — completion means a VERIFIED pass, reached by
escalating compute, never by lowering the bar.
"""

from __future__ import annotations

import itertools
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ..ablation.arms import AblationConfig
from ..engine.governor import RunGovernor
from ..engine.runtime import Engine
from ..errors import CutLosses, FailLoud, PlateauStop
from ..isolation.worktree import WorktreeProvider, apply_diff
from ..kernel.select import Candidate, select_best
from ..kernel.verify import VerificationResult, candidate_from_verification
from ..schema_validate import validate_schema
from ..types import ScopedTask


# Strategy hints cycled across attempts to preserve diversity (decorrelates
# hallucinations — the whole reason best-of-N works).
DEFAULT_STRATEGIES = ("minimal", "comprehensive", "test_driven", "edge_case_hardening")

# Harness scaffolding that the read-jail launcher writes into EVERY per-rollout worktree (e.g.
# `.apex_seatbelt/read_jail.sb`, the seatbelt profile). It is NOT part of any solution: it bloats
# the candidate patch and, being a per-worktree byte-divergent NEW file, turns a genuinely-disjoint
# module merge into a SPURIOUS 3-way conflict (it caused real cross-module conflicts in the jinja
# reduce collapse). Excluded at diff extraction (_merged_diff here; _git_diff in the executor) and
# defensively stripped from any already-cached diff before it is applied in reduce_residuals.
_SCAFFOLD_PREFIXES = (".apex_seatbelt/",)
# git pathspec form: keep everything under `.` EXCEPT the scaffolding dirs.
_SCAFFOLD_PATHSPEC = ("--", ".") + tuple(f":(exclude){p}" for p in _SCAFFOLD_PREFIXES)


def _diff_touched_paths(diff: str) -> set:
    """The set of target paths a unified diff edits (from its ``+++ b/<path>`` lines). Pure;
    used to DETECT competing whole-repo candidates (fan-out non-disjointness)."""
    paths = set()
    for line in (diff or "").splitlines():
        if line.startswith("+++ "):
            p = line[4:].strip()
            if p.startswith("b/"):
                p = p[2:]
            if p and p != "/dev/null":
                paths.add(p)
    return paths


def _strip_scaffold_hunks(diff: str) -> str:
    """Drop the per-file sections of a unified diff that target harness scaffolding
    (``_SCAFFOLD_PREFIXES``). Pure string transform that preserves diff framing; handles both the
    ``git diff`` form (``diff --git`` headers) and the minimal ``--- a/ / +++ b/`` form. So a
    cached candidate diff that already captured ``.apex_seatbelt/`` cannot torpedo a disjoint
    merge in reduce_residuals."""
    if not diff or not diff.strip():
        return diff
    delim = r"(?m)(?=^diff --git )" if "diff --git " in diff else r"(?m)(?=^--- )"
    out = []
    for block in re.split(delim, diff):
        if not block:
            continue
        tgt = ""
        for line in block.splitlines():
            if line.startswith("+++ "):
                tgt = line[4:].strip()
                if tgt.startswith("b/"):
                    tgt = tgt[2:]
                break
        if tgt and any(tgt.startswith(p) for p in _SCAFFOLD_PREFIXES):
            continue
        out.append(block)
    return "".join(out)


# DECOMPOSE_SCHEMA — the read-only schema'd reply contract for ctx.decompose(): a module
# breakdown of the repo (module name -> the gold test ids that module must turn green +
# topological deps), plus an explicit topological ``order``. This is a SIGNAL the convergence
# default uses to fan out per-module solve agents; the number of modules becomes the PRIMARY
# difficulty signal (file-count stays the floor). A schema-miss / undecomposable repo returns
# None (ctx.ask null terminal) -> the caller falls back to flat best-of-N.
DECOMPOSE_SCHEMA = {
    "type": "object",
    "required": ["modules"],
    "properties": {
        "modules": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["module", "gold_test_ids"],
                "properties": {
                    "module": {"type": "string"},
                    "gold_test_ids": {"type": "array", "items": {"type": "string"}},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                    # OPTIONAL (advisory): the disjoint slice of files this module owns. Lets the
                    # convergence default DETECT (not assume) fan-out disjointness; absent -> the
                    # plan is fail-open compatible (older plans / a model that omits it still work).
                    "files": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "order": {"type": "array", "items": {"type": "string"}},
    },
}

# PHASE_PLAN_SCHEMA — ctx.plan_phases() reply contract: a Claude-Code-style ORDERED list of
# phases (manageable chunks WITH objectives + per-phase acceptance), grouping the decompose
# modules in dependency order. Each phase carries an `objective` (the chunk's goal), the union
# of its modules' gold test ids as its `acceptance_gold_ids` (the per-phase acceptance predicate
# — validated against the real gold inventory; hallucinated ids are dropped), `files_owned`
# (delegation-contract boundaries), the constituent `modules`, and `depends_on`. A schema-miss /
# <2 valid phases returns None (ctx.ask null terminal) -> the caller falls back to whole-repo
# converge. This is the planner; acceptance stays engine-owned (a phase pass is a SUBSET of gold
# ids green, NEVER an accept — only ctx.select on the full suite accepts).
PHASE_PLAN_SCHEMA = {
    "type": "object",
    "required": ["phases"],
    "properties": {
        "phases": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["objective", "acceptance_gold_ids"],
                "properties": {
                    "name": {"type": "string"},
                    "objective": {"type": "string"},
                    "acceptance_gold_ids": {"type": "array", "items": {"type": "string"}},
                    "files_owned": {"type": "array", "items": {"type": "string"}},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                    "modules": {"type": "array", "items": {"type": "string"}},
                    # OPTIONAL: the planner may flag a phase as needing a bespoke per-phase
                    # orchestration script (the user's "generate code per phase"); honored only
                    # when APEX_OMEGA_PHASE_CODEGEN=1 (default OFF).
                    "needs_custom_orchestration": {"type": "boolean"},
                },
            },
        },
    },
}

# GATE_SCHEMA — ctx.goal_align_gate() per-skeptic reply contract: the adversarial GOAL-ALIGNMENT
# review that keeps a long phased run from VEERING off the goal. A skeptic returns proceed/revise/
# abort; a revise/abort MUST cite `evidence_ids` that are REAL failing gold node-ids (grounding it
# in execution reality, not the transcript — the moat over a transcript-only verifier). An
# ungrounded verdict is DOWNGRADED to proceed. read-only SIGNAL: it can re-target or stop the phase
# loop but can NEVER set acceptance (Cardinal Contract).
GATE_SCHEMA = {
    "type": "object",
    "required": ["verdict"],
    "properties": {
        "verdict": {"type": "string", "enum": ["proceed", "revise", "abort"]},
        "reason": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "retarget_gold_ids": {"type": "array", "items": {"type": "string"}},
    },
}

# SANDBOX-NOT-PROMPT POLICY (user directive 2026-06-16): we do NOT limit the model via
# prompts and we do NOT penalize/kill an attempt for trying to fetch/cheat. The CORRECTNESS
# BOUNDARY is structural — the worktree's source SHADOWS any site-packages install (cwd for
# flat repos; P0.1 PYTHONPATH-prepend + origin guard for src-layout), so a fetched upstream
# package physically cannot be imported over the candidate's edits and therefore can NEVER
# produce a false "solve" (the score always reflects the worktree code). The agent may try to
# download/clone/install the real distribution; it simply will not help it pass. So there is
# no anti-fetch prompt suffix: an attempt is judged ONLY by the visible suite running against
# its worktree, and a blocked escape is not a failure.


def classify_attempt_integrity(res) -> dict:
    """RECORD (never penalize) signals that an attempt tried to ESCAPE the sandbox or CHEAT.
    Pure telemetry for cross-eval analysis — the worktree-shadow + execution gate already make
    any of these incapable of producing a false solve, and the attempt is scored normally
    regardless. Signals:
      * sandbox_escape   — the vendor sandbox DENIED an action (policy_violation / "outside the
                           root" / permission denied): the agent tried to reach beyond its jail.
      * fetch_attempt    — the agent tried to download/clone/pip-install the upstream package.
      * test_modification — the diff edits the VISIBLE test files (a weaken-the-tests cheat;
                           harmless because acceptance is keyed on the gold expected-test ids)."""
    fs = str(getattr(res, "finalization_status", "") or "")
    err = str(getattr(res, "error", "") or "")
    diff = str(getattr(res, "fs_diff", "") or "")
    el = err.lower()
    signals = []
    if (fs == "policy_violation" or "outside the root" in el or "permission denied" in el
            # a DOWNGRADED workspace-discovery escape completes cleanly; its soft reason (folded into
            # error by the v1 adapter, T1) still names the denied out-of-workspace discovery.
            or "outside the rollout workspace" in el or "workspace_discovery" in el):
        signals.append({"kind": "sandbox_escape", "detail": (err or fs)[:200]})
    if any(t in el for t in ("download", "clone", "pip install", "uv pip", "fetch", "pypi", "upstream")):
        signals.append({"kind": "fetch_attempt", "detail": err[:200]})
    test_edits = [ln for ln in diff.splitlines()
                  if ln.startswith(("+++ ", "--- ")) and ("/test" in ln or "test_" in ln)]
    if test_edits:
        signals.append({"kind": "test_modification", "detail": "; ".join(test_edits[:5])[:200]})
    return {"attempted": bool(signals), "signals": signals}


def _as_int_id(v) -> int:
    """Coerce an attempt/agent id into a deterministic int. AUTHORED orchestrator code (LLM
    output) may pass an id of ANY type — e.g. a descriptive string like 'mimesis-scout' — and it
    is used to index worker specs (aid % n), form cluster ids, and key the journal. int() when
    possible; otherwise a stable hash (so the run stays replayable) instead of crashing the cell."""
    try:
        return int(v)
    except (ValueError, TypeError):
        from ..journal.key import sha256_hex
        return int(sha256_hex(str(v))[:12], 16)


def _materialize_cached_diff(wt: str, diff: str) -> None:
    """Re-apply a journaled diff into a fresh worktree on a resume HIT. If it FAILS to
    apply (worktree context shifted), RAISE so the attempt is treated as infra (excluded +
    re-attempted) instead of silently scoring an UNPATCHED tree as a false failure that then
    caches permanently (review-fix #12). ``apply_diff`` returns False when neither the strict
    nor the 3-way apply succeeds."""
    if diff and not apply_diff(wt, diff):
        raise RuntimeError(f"cached diff failed to re-apply on resume in {wt}")


class _MergeRes:
    """A minimal stand-in for an ExecResult so reduce_residuals can reuse the journaled
    _scored() path (which keys the cache on ``res.fs_diff``). The merge is a plain-Python
    apply-and-score with no agent, so the merged diff IS the artifact and the cache key."""

    __slots__ = ("fs_diff",)

    def __init__(self, merged_diff: str):
        self.fs_diff = merged_diff or ""


class OrchestrationContext:
    def __init__(
        self,
        engine: Engine,
        *,
        executor: Any,
        worker_specs: Sequence[Any],
        source_repo: str,
        base_commit: Optional[str],
        score_fn: Callable[[str], VerificationResult],
        prompt_builder: Callable[["OrchestrationContext", int, str], str],
        repo_map: Optional[dict] = None,
        abl: Optional[AblationConfig] = None,
        run_scope: str = "autosolve",
        max_agents: Optional[int] = None,
        initial_agents: int = 1,
        sandbox: str = "workspace-write",
        timeout_seconds: Optional[int] = None,
        strategies: Sequence[str] = DEFAULT_STRATEGIES,
        repair_iters: int = 2,
        args: Any = None,
        node_ns: str = "",
        nesting_depth: int = 0,
    ):
        self._engine = engine
        self._executor = executor
        self._worker_specs = list(worker_specs)
        self._score_fn = score_fn
        self._prompt_builder = prompt_builder
        # dynamic-workflows parity: the launch payload (ctx.args) + composition state. node_ns
        # namespaces this context's journal node-ids / worktree ids so a NESTED ctx.workflow()
        # child cannot collide with the parent on resume (default "" => unchanged for the root).
        self._args = args
        self._node_ns = str(node_ns or "")
        self._nesting_depth = int(nesting_depth)
        self._base_commit = base_commit
        self._run_scope = run_scope
        self.repo_map = dict(repo_map or {})
        self._abl = abl or AblationConfig()
        self.strategies = tuple(strategies)
        self.sandbox = sandbox
        self.timeout_seconds = timeout_seconds
        # Backbone 0.2: per-agent wall DECOUPLED from the cell wall so a single agent can
        # never eat the whole cell (the true run-4 root cause). Difficulty-derived, strictly
        # < the cell timeout; None when the cell is unbounded (agents run to completion).
        _diff = str((self.repo_map or {}).get("difficulty") or "").lower()
        _pa = {"easy": 1800, "medium": 2400, "hard": 3000}.get(_diff, 2400)
        self.per_agent_timeout_seconds = None if timeout_seconds is None else min(_pa, int(timeout_seconds))
        # HARD CEILING on test-driven repair depth. Clamps every solve_and_repair/
        # make_repairing_attempt call, so an authored orchestrator can never EXCEED the
        # configured repair budget. Default is now 2 (ON): the run-4 budget blowup is no
        # longer a risk because the SPFG+ governor (engine/frontier.py + governor.py) stops
        # a TRUE plateau (no valid-measurement improvement + no frontier rise within the
        # patience windows) while letting a climbing frontier keep going — so repair only
        # spends budget while it is actually closing the gap. Set repair_iters=0 to force
        # the old flat best-of-N behaviour.
        self.repair_iters = max(0, int(repair_iters))
        # Backbone 1.1: optional drift keys for the journaled SCORE step (set by the eval
        # harness; "" = unused). The score is keyed on the diff content + repo snapshot so
        # resume never re-runs pytest for an unchanged diff.
        self.expected_ids_sha = ""
        self.scoring_env_sha = ""
        # the orchestrator should respect this soft cap (fewest agents first);
        # the engine enforces a hard ceiling regardless.
        self.max_agents = min(max_agents or engine.max_total_agents, engine.max_total_agents)
        # Backbone 2.1: the RunGovernor is the single "may we continue" authority
        # (default UNBOUNDED; always-on guards = per-run agent ceiling + plateau). The
        # plateau state below makes a no-clock fan-out loop terminate without discarding
        # work; ctx.parallel raises PlateauStop once halted (caught by autosolve).
        # Fix 3 (governor audit): unify the harness-stall ceiling across tiers. The in-cell
        # governor defaulted harness_stall_cut=8 while the ladder tier uses INDET_CEIL (24), giving
        # two different harness walls. Source the shared frontier default so both tiers agree.
        from ..engine.frontier import frontier_defaults
        _indet_ceil = frontier_defaults()[2]
        self.governor = RunGovernor(
            engine=engine, agent_ceiling=engine.max_total_agents,
            token_budget=engine.budget.total, agent_budget=self.max_agents, plateau_k_dry=2,
            harness_stall_cut=_indet_ceil,
        )
        # Backbone 2.4 CUT-LOSSES detector state. Progress is tracked on the BEST-so-far
        # distance-to-solve (gold ids green = ``_best_gold_passed``; raw pass_rate as the
        # secondary tier), NEVER the last wave — so a refactor dip that recovers is not a
        # cut, and a high-but-flat pass_rate (no new gold) still counts as dry. Plus two
        # hard-cut streaks (all-nonresult waves; sterile/identical-diff waves).
        self._dry_rounds = 0                  # telemetry: consecutive dry WAVES (not the cut unit)
        self._best_pass_rate = 0.0
        self._best_gold_passed = 0
        # Fix 1 (governor audit): SECONDARY implementation-progress frontier = the min collection-
        # error count seen in a VALID measurement ("distance to first collect"). A strict DROP
        # (errors 5091->4000 as modules/imports get implemented) counts as progress that RESETS the
        # patience arms + the sterile streak — WITHOUT banking a gold solve (best_gold_passed stays
        # the only acceptance number). This stops the governor from cutting a large monolithic-import
        # repo (whose gold suite cannot collect until much is implemented) as sterile/no-progress
        # while it is genuinely progressing. A genuinely flat run (errors 5091->5091) is unaffected
        # and still cut correctly.
        self._best_min_errors = None
        # SPFG++ (FM-2 / stop-policy design): GENERALIZE the errors-only secondary into a small
        # VECTOR of execution-grounded TEST-OUTCOME progress signals, each oriented higher=better and
        # credited only on a STRICT rise from an established baseline (same no-credit-on-first
        # convention as gold). A rise in ANY component resets BOTH patience arms but NEVER banks a
        # gold solve. Components:
        #   neg_errors      = -collection-error count  (more of the suite now collects)
        #   neg_failing_len = -full-suite failing-id count (residual failures shrinking — captures
        #                     fail->pass beyond the EXPECTED-id gold count)
        #   import_depth    = deepest dotted module reached before a collection error (as shallow
        #                     modules get implemented the first failing import moves DEEPER -> rises;
        #                     the ONLY rising signal on an early collection-collapse repo where errors
        #                     are still flat — directly fixes the FM-2 "cut a progressing collapse").
        # Deliberate refinement of the 8-vector design: NEW-on-disk-code activity (cum diff bytes /
        # changed files) is routed to the STERILE-streak reset via `any_new_useful`, NOT this frontier
        # arm — so an agent thrashing out new-but-useless code is not "sterile" yet still eventually
        # cut on genuine no-PROGRESS (avoids the false-CONTINUE-forever hole). Ablate the whole vector
        # with APEX_FRONTIER_VECTOR=0 -> errors-only secondary (exact pre-SPFG++ behavior).
        self._best_vec: dict = {}
        self._frontier_vector = (
            os.environ.get("APEX_FRONTIER_VECTOR", "1").strip().lower()
            not in ("0", "false", "no", "off"))
        self._nonresult_streak = 0
        self._sterile_streak = 0
        self._seen_shas: set = set()
        self._tokens_at_best = 0
        self._agents_at_best = 0              # agents_used when the BEST last improved (cut unit)
        # ---- SPFG+ "Solve-Progress Frontier Governor" arms (Backbone 2.5) ----
        # The FRONTIER = the best-so-far gold-pass COUNT in a VALID measurement (pass_rate is the
        # secondary tie-break). cut:no-progress now fires only on a GENUINE no-solve-progress
        # plateau: BOTH a VALID-measurement window AND a journaled VALID-measurement wall clock
        # elapsed since the frontier last rose. Indeterminate (harness/scorer-failed) measurements
        # are NEUTRAL to both arms and instead feed the DISTINCT cut:harness-stall streak — fixing
        # the real bug where attempts_since_improvement advanced the plateau clock through a harness
        # outage. The wall scalar is a JOURNALED, replay-deterministic accumulation (a fixed nominal
        # per-VALID-measurement increment from the config per-agent timeout — NEVER a live clock),
        # so a journal replay reconstructs the exact same seconds_since_frontier_improved.
        self._valid_measurements = 0
        self._valid_measurements_at_best = 0
        self._valid_wall_accum = 0.0
        self._valid_wall_at_best = None       # None => clock UNSTARTED (no valid measurement yet)
        self._indeterminate_streak = 0
        self._indeterminate_total = 0         # cumulative (telemetry / cut_losses ledger)
        self._wall_started = False
        self._frontier_history: list = []     # [(valid_idx, gold_count)] at each strict frontier rise
        # The nominal per-VALID-measurement wall increment. Deterministic (a config scalar, not a
        # live clock) so the wall arm reconstructs identically on resume. Defaults to the per-agent
        # timeout (difficulty-derived) or a fixed nominal when the cell is unbounded.
        self._valid_wall_increment = float(self.per_agent_timeout_seconds or 2400)
        self._halt_reason = ""
        self._halt_is_cut = False
        self._halted = False
        self._all_candidates: list = []
        # the most recent full-suite residual failing node-ids (set by reduce_residuals); the
        # phase planner's goal-alignment gate grounds its skeptics in this REAL failing set.
        self._last_residual: list = []
        # IOU / blocked-on ledger (ctx.defer): a structured deferral sentinel (the paradigm's
        # todo!("blocked_on: X::Y")) so a bounded loop can record an unresolved item and TERMINATE,
        # handing it to a downstream phase instead of spinning or dropping it silently.
        self._iou: list = []
        # Repair/ralph parent-diff carry limit (chars). Default truncates (a Reflexion hint);
        # the ralph baseline raises it so its sequential lineage carries the full accumulated
        # diff (naive-persistence fidelity).
        self._repair_diff_limit = 6000
        # The scout's difficulty assessment sets the INITIAL wave size (first
        # number of agents). Easy -> 1; harder -> several. Bounded by the soft cap.
        self.initial_agents = max(1, min(initial_agents, self.max_agents))
        # source repo path kept for ctx.ask (read-only sub-questions run here, no worktree)
        self._source_repo = source_repo
        self._provider = WorktreeProvider(
            source_repo, base_commit=base_commit,
            workspace_dir=str(Path(engine.run_dir) / "worktrees"), run_scope=run_scope,
        )
        self._attempt_counter = itertools.count()
        self._counter_lock = threading.Lock()
        # Backbone 2.0: wave-decision counter. Every continue/halt verdict is journaled by
        # POSITION (kind="wave") so a resumed run replays the SAME branch sequence rather
        # than recomputing it from volatile inputs (budget-remaining, decision timing) —
        # closing the C1 control-flow-divergence hole. Decisions are taken on the main
        # orchestrator thread only (after the ctx.parallel barrier), so no lock is needed.
        self._wave_counter = itertools.count()

    # ---- narration / budget (read-mostly) ----
    def phase(self, title: str) -> None:
        self._engine.phase(str(title))

    def log(self, msg: str) -> None:
        self._engine.log(str(msg))

    @property
    def budget(self):
        return self._engine.budget

    def agents_used(self) -> int:
        return self._engine.agents_used()

    @property
    def worker_specs(self) -> list:
        return list(self._worker_specs)

    # ---- journaled SCORE (resume never re-runs pytest for an unchanged diff) ----
    def _scored(self, wt, res):
        """Journal the execution-authoritative pytest score so a resumed cell replays it
        as a cache HIT (no pytest re-run) for an unchanged candidate diff, and a kill
        AFTER green recovers the counts (Backbone 1.1). Keyed on the diff content + repo
        snapshot (+ optional env-drift shas). indeterminate -> infra_nonresult so a
        harness/launch failure is NEVER a phantom-accept and re-runs."""
        from ..journal.key import sha256_hex
        from ..journal.resume import resume_or_run_json
        from ..journal.wal import RESULT_INFRA_NONRESULT, RESULT_OK  # noqa: F401 (RESULT_OK documents intent)
        components = {"kind": "score", "scoped_inputs": {
            "diff_sha": sha256_hex(res.fs_diff or ""),
            "repo_snapshot_sha": self._provider.base_commit,
            "expected_ids_sha": self.expected_ids_sha,
            "scoring_env_sha": self.scoring_env_sha,
        }}

        def _run():
            return self._score_fn(wt).to_dict()

        def _status(v):
            return RESULT_INFRA_NONRESULT if (v or {}).get("indeterminate") else RESULT_OK

        d, _hit = resume_or_run_json(self._engine.journal, components, _run,
                                     kind="score", node_id="score", status_fn=_status)
        return VerificationResult.from_dict(d)

    # ---- acceptance checkpoint: bank a verified solve THE INSTANT it passes ----
    def _checkpoint_accepted(self, cand) -> None:
        """Persist a verified-accepted candidate to disk immediately, BEFORE the
        wave barrier (ctx.parallel) or the cell wall-clock can discard it. run-4 lost
        verified mimesis 6052/6052 passes exactly this way: an attempt computed an
        accepted solve, but the cell was killed at the outer wall while ctx.parallel
        still waited on slow sibling thunks, so ctx.select never banked the winner.
        Best-effort, atomic, idempotent (first accept wins)."""
        try:
            import json as _json
            p = Path(self._engine.run_dir) / "accepted_checkpoint.json"
            if p.exists():
                return
            rec = {"accepted": True, "candidate_id": cand.candidate_id,
                   "pass_rate": cand.public_signal_score, "score": cand.combined_score,
                   "content_sha": cand.content_sha,
                   "repo": self.repo_map.get("repo") or self.repo_map.get("source_repo")}
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(_json.dumps(rec))
            tmp.replace(p)
        except Exception:
            pass

    def _checkpoint_phase(self, cand, *, subset_passed: int, subset_total: int,
                          phase_id: str = "") -> None:
        """Bank a PARTIAL/phase frontier gain to disk the instant it appears, so an outer
        subprocess kill cannot discard the work-in-progress (the phased analogue of
        _checkpoint_accepted). Writes <run_dir>/phase_checkpoint.json — a SEPARATE file from
        accepted_checkpoint.json so a partial is NEVER reported solved:1 (Cardinal Contract C7);
        run_ladder surfaces it as partial_frontier telemetry only. MONOTONE: overwrites only on a
        STRICT gold-pass-COUNT rise. Atomic temp-write + replace. accepted is always False here —
        only the engine-owned whole-suite ctx.select accepts. Best-effort, never fatal."""
        try:
            if cand is None:
                return
            import json as _json
            p = Path(self._engine.run_dir) / "phase_checkpoint.json"
            prev = -1
            if p.exists():
                try:
                    prev = int(_json.loads(p.read_text()).get("gold_passed", -1))
                except Exception:
                    prev = -1
            if int(subset_passed) <= prev:
                return   # monotone: bank only a strict frontier rise
            rec = {"accepted": False, "gold_passed": int(subset_passed),
                   "gold_total": int(subset_total),
                   "candidate_id": getattr(cand, "candidate_id", ""),
                   "content_sha": getattr(cand, "content_sha", ""),
                   "pass_rate": getattr(cand, "public_signal_score", 0.0),
                   "phase_id": str(phase_id or ""),
                   "repo": self.repo_map.get("repo") or self.repo_map.get("source_repo")}
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(_json.dumps(rec))
            tmp.replace(p)
        except Exception:
            pass

    # ---- escape/cheat telemetry (record, never penalize) ----
    def _record_integrity(self, attempt_id: str, vendor: str, integ: dict) -> None:
        """Append escape/cheat telemetry to ``integrity_log.jsonl`` (best-effort, NEVER fatal).
        The attempt is scored normally — this records what the agent TRIED so escape/cheat
        behaviour can be analyzed across evaluations. We do not kill or down-rank the attempt;
        the sandbox simply denies the action and the agent is left to solve the real task."""
        try:
            import json as _json
            rec = {"attempt": attempt_id, "vendor": vendor,
                   "repo": self.repo_map.get("repo") or self.repo_map.get("source_repo"),
                   "signals": integ.get("signals", [])}
            p = Path(self._engine.run_dir) / "integrity_log.jsonl"
            with p.open("a", encoding="utf-8") as fh:
                fh.write(_json.dumps(rec) + "\n")
            kinds = [s.get("kind") for s in integ.get("signals", [])]
            self._engine.log(f"integrity: {attempt_id} attempted {kinds} (denied by sandbox; "
                             "scored normally, not penalized)")
        except Exception:
            pass

    # ---- fan-out primitives (journaled, concurrency-bounded by the engine) ----
    def parallel(self, thunks: Sequence[Callable[[], Any]]) -> list:
        # Backbone 2.1/2.4: always-on CUT-LOSSES governor. Once halted, raise the SPECIFIC
        # stop — CutLosses for a genuine non-progress cut, PlateauStop for an honest
        # ceiling/explored stop — so a no-clock `while True: ctx.parallel(...)` terminates and
        # the host selects the best banked candidate. This round's results are still returned;
        # only a SUBSEQUENT call after the halt-set raises.
        if self._halted:
            self._engine.log(f"governor halt: {self._halt_reason or 'plateau'}")
            exc = CutLosses if self._halt_is_cut else PlateauStop
            raise exc(self._halt_reason or "plateau: no progress")
        out = self._engine.parallel(list(thunks))
        self._observe(out)
        self._wave_verdict(self._wave_state())   # sets self._halted on a halt verdict
        return out

    def _observe(self, out: Sequence[Any]) -> None:
        """Fold a batch of returned candidates into the SPFG+ frontier + cut-losses accounting
        (frontier rise / valid-measurement + journaled-wall / nonresult + sterile streaks). This
        is the SAME accounting ctx.parallel runs, factored out so the convergence REDUCE step
        (reduce_residuals' merged full-suite candidate) can feed the frontier too — a climbing
        frontier (more residual ids green) resets BOTH patience arms, while a conflict/indeterminate
        reduce is NEUTRAL (feeds only the harness-stall streak). It does NOT call the wave verdict
        (the caller decides when to take a halt decision)."""
        # --- CUT-LOSSES accounting over the returned candidates (FRESH or CACHED, so the
        # detector state is faithfully reconstructed during a resume replay) ---
        round_gold = self._best_gold_passed
        round_pass = self._best_pass_rate
        frontier_cand = None     # the candidate that pushed the gold frontier up this batch
        round_min_errors = None  # Fix 1: lowest VALID collection-error count this batch (secondary frontier)
        round_vec: dict = {}     # SPFG++: this batch's BEST (higher=better) per secondary-frontier component
        n_attempts = len(out)
        n_nonresult = 0          # attempts that produced no usable work
        n_sterile = 0            # attempts with an empty diff OR a diff already seen this run
        any_new_useful = False   # at least one attempt produced a NEW non-empty diff
        for c in out:
            if c is None:
                n_nonresult += 1
                n_sterile += 1   # a None result is both no-work and no-useful-diff
                # SPFG+: a None result is an indeterminate (no real measurement) — NEUTRAL to the
                # frontier arms; it only feeds the harness-stall streak.
                self._indeterminate_streak += 1
                self._indeterminate_total += 1
                continue
            m = getattr(c, "meta", {}) or {}
            gp = int(m.get("gold_passed", 0) or 0)
            pr = getattr(c, "public_signal_score", None)
            pr = pr if (isinstance(pr, (int, float)) and not isinstance(pr, bool)) else 0.0
            if gp > round_gold:
                round_gold = gp
                frontier_cand = c
            if pr > round_pass:
                round_pass = pr
            # SPFG+ valid-measurement filter: a candidate contributes to the FRONTIER and to the
            # patience clocks ONLY if it is a real test outcome. An indeterminate (harness/scorer
            # failure) measurement is NEUTRAL to both frontier arms and feeds the harness-stall
            # streak instead; a valid measurement increments the valid count + the journaled wall
            # and resets the streak.
            if bool(m.get("indeterminate")):
                self._indeterminate_streak += 1
                self._indeterminate_total += 1
            else:
                self._indeterminate_streak = 0
                self._valid_measurements += 1
                if self._valid_wall_at_best is None:
                    self._valid_wall_at_best = 0.0   # first VALID measurement STARTS the clock
                    self._wall_started = True
                self._valid_wall_accum += self._valid_wall_increment
                # Fix 1: track the lowest VALID collection-error count this batch (errors =
                # erroring/uncollected gold tests). A new low across batches = real progress toward
                # the first collect/pass on a large not-yet-collecting repo.
                _err = int(m.get("errors", 0) or 0)
                round_min_errors = _err if round_min_errors is None else min(round_min_errors, _err)
                # SPFG++ secondary-frontier vector (valid measurements only), each higher=better:
                if self._frontier_vector:
                    def _bump(key, val):
                        cur = round_vec.get(key)
                        round_vec[key] = val if cur is None else max(cur, val)
                    _bump("neg_errors", -_err)
                    _bump("neg_failing_len", -len(m.get("failing_nodeids") or []))
                    _idep = self._parse_import_depth(m.get("failure_excerpts"))
                    if _idep is not None:
                        _bump("import_depth", int(_idep))
            fs = m.get("finalization_status")
            if bool(m.get("indeterminate")) or (
                    fs in ("policy_violation", "infra_nonresult", "timeout") and gp <= 0 and pr <= 0.0):
                n_nonresult += 1
            empty = bool(m.get("empty_diff"))       # F1: a no-edit attempt is sterile regardless of sha
            sha = getattr(c, "content_sha", "") or ""
            if empty or (sha and sha in self._seen_shas):
                n_sterile += 1
            elif sha:
                any_new_useful = True
            if sha:
                self._seen_shas.add(sha)
        # PROGRESS = strict improvement in BEST distance-to-solve (gold ids green PRIMARY,
        # raw pass_rate SECONDARY). BEST-not-LAST: a dip that does not beat the best is dry,
        # not a regression cut; a high-but-flat pass_rate with no new gold is also dry.
        # Fix 1 (governor audit): a strict DROP in the collection-error frontier (more of the gold
        # suite now collects) is genuine implementation progress on a not-yet-passing large repo.
        # It counts as a frontier rise for the patience/sterile clocks but is NEVER a gold solve
        # (best_gold_passed is unchanged below when only this fires). A flat run (errors 5091->5091)
        # gets no credit and is still cut correctly.
        # establish the collection-error baseline WITHOUT crediting it (the first measurement is not
        # itself progress — same convention as gold); only a STRICT DROP from an established baseline
        # counts, so a small already-collecting repo (errors==0 throughout) is wholly unaffected.
        secondary_improved = False
        if self._frontier_vector:
            # SPFG++: a strict rise in ANY test-outcome component (errors down / residual failures
            # shrinking / imports advancing deeper) is genuine implementation progress on a large
            # not-yet-passing repo. _fold_secondary_vector establishes each component's baseline
            # without crediting it (first measurement is not itself progress) and commits the new
            # best on a strict rise — NEVER banking a gold solve.
            secondary_improved = self._fold_secondary_vector(round_vec)
        elif round_min_errors is not None:
            if self._best_min_errors is None:
                self._best_min_errors = round_min_errors
            elif round_min_errors < self._best_min_errors:
                secondary_improved = True
        improved = ((round_gold > self._best_gold_passed)
                    or (round_pass > self._best_pass_rate + 1e-9)
                    or secondary_improved)
        if improved:
            # record a STRICT gold-count rise in the frontier history (telemetry / ledger).
            if round_gold > self._best_gold_passed:
                self._frontier_history.append((self._valid_measurements, int(round_gold)))
                # ACCEPTANCE-CHECKPOINT the PARTIAL frontier the instant it rises, so an outer
                # kill never discards the work-in-progress (telemetry-only; never a solve, C7).
                if frontier_cand is not None:
                    self._checkpoint_phase(
                        frontier_cand, subset_passed=int(round_gold),
                        subset_total=int((getattr(frontier_cand, "meta", {}) or {}).get("gold_total", 0) or 0),
                        phase_id="frontier")
            self._best_gold_passed = round_gold
            self._best_pass_rate = round_pass
            # Fix 1: advance the secondary collection-error frontier on a strict drop (baseline was
            # established above). Never banks a gold solve — best_gold_passed is unchanged when only
            # the secondary fired. (The SPFG++ vector path commits its component bests inside
            # _fold_secondary_vector; only the errors-only fallback commits here.)
            if secondary_improved and not self._frontier_vector:
                self._best_min_errors = round_min_errors
            self._dry_rounds = 0
            self._tokens_at_best = self._engine.budget.spent()
            self._agents_at_best = self._engine.agents_used()
            # SPFG+: a FRONTIER rise resets BOTH patience arms (valid-measurement window + wall).
            self._valid_measurements_at_best = self._valid_measurements
            self._valid_wall_at_best = self._valid_wall_accum if self._wall_started else None
        else:
            self._dry_rounds += 1
        # HARD-CUT STREAKS counted in ATTEMPTS, not waves (review M1: SIZE-INVARIANT — a width-1
        # ralph lineage and a width-N omega wave reach the cut after a comparable number of
        # ATTEMPTS, so the arm comparison stays apples-to-apples).
        # nonresult: a wave with ANY usable work resets; an all-nonresult (or empty) wave adds
        # its attempt count.
        if n_attempts == 0:
            self._nonresult_streak += 1
        elif n_nonresult == n_attempts:
            self._nonresult_streak += n_nonresult
        else:
            self._nonresult_streak = 0
        # sterile: a wave with ANY new useful diff OR any improvement resets; otherwise it adds
        # the count of sterile (empty/repeated-diff) attempts.
        if any_new_useful or improved:
            self._sterile_streak = 0
        else:
            self._sterile_streak += max(1, n_sterile)

    def _fold_secondary_vector(self, round_vec: dict) -> bool:
        """SPFG++ secondary frontier. Each component is oriented higher=better. The FIRST observation
        of a component establishes its baseline WITHOUT crediting it (same no-credit-on-first
        convention as the gold frontier); any later STRICT rise credits progress and commits the new
        component best. Returns True iff some component strictly rose. Monotone-best per component, so
        a dip that does not beat the best is dry (never a regression), and this NEVER banks a gold
        solve (best_gold_passed is untouched)."""
        improved = False
        for key, val in round_vec.items():
            if val is None:
                continue
            best = self._best_vec.get(key)
            if best is None:
                self._best_vec[key] = val          # establish baseline, no credit
            elif val > best:
                self._best_vec[key] = val           # strict rise -> credit + commit
                improved = True
        return improved

    @staticmethod
    def _parse_import_depth(excerpts) -> Optional[int]:
        """Best-effort collection-progress proxy: the deepest dotted module name implicated in a
        pytest COLLECTION import error (``No module named 'a.b.c'`` -> depth 3; ``cannot import name X
        from 'a.b'`` -> depth 2). As shallow modules get implemented the first failing import moves
        DEEPER, so the max depth RISES — the only monotone progress signal on an early collection-
        collapse repo whose error COUNT is still flat. Returns None when no import error is present
        (e.g. the suite collects cleanly), so an already-collecting repo gets no spurious credit."""
        if not excerpts:
            return None
        best: Optional[int] = None
        text = str(excerpts)
        for mm in re.finditer(r"No module named ['\"]([\w\.]+)['\"]", text):
            d = mm.group(1).count(".") + 1
            best = d if best is None else max(best, d)
        for mm in re.finditer(r"cannot import name ['\"][\w]+['\"] from ['\"]([\w\.]+)['\"]", text):
            d = mm.group(1).count(".") + 1
            best = d if best is None else max(best, d)
        return best

    def _wave_state(self) -> dict:
        """The cut-losses detector inputs at the current wave boundary. All cut signals are in
        ATTEMPTS (agents), so the rule is invariant to the wave schedule and the arm width."""
        agents = self._engine.agents_used()
        # SPFG+ wall arm: the journaled VALID-measurement wall seconds since the frontier last rose.
        # 0 until the clock has started (first valid measurement). Reconstructed deterministically
        # on resume from the journaled valid-measurement count (no live clock), so the cached
        # _wave_verdict replays identically.
        secs = ((self._valid_wall_accum - self._valid_wall_at_best)
                if (self._wall_started and self._valid_wall_at_best is not None) else 0.0)
        return {
            "attempts_since_improvement": max(0, agents - self._agents_at_best),
            "dry_rounds": self._dry_rounds,          # telemetry only (not a cut signal)
            "agents_used": agents,
            "nonresult_streak": self._nonresult_streak,
            "sterile_streak": self._sterile_streak,
            "tokens_since_improvement": max(0, self._engine.budget.spent() - self._tokens_at_best),
            # SPFG+ frontier arms (matching FrontierTracker.state() + governor.verdict reads).
            "valid_measurements": self._valid_measurements,
            "valid_measurements_since_improvement": max(0, self._valid_measurements - self._valid_measurements_at_best),
            "seconds_since_frontier_improved": secs,
            "indeterminate_streak": self._indeterminate_streak,
        }

    # ---- journaled wave decision (Backbone 2.0 determinism-under-resume) ----
    def _wave_verdict(self, state: dict) -> bool:
        """Record/replay the governor's (continue, reason) verdict by POSITION. First run:
        computes live and journals it. Resume: replays the journaled verdict+reason (cache
        HIT) so the control-flow branch AND the recorded cut reason are identical regardless
        of this process's volatile counters."""
        from ..journal.resume import resume_or_run_json
        n = next(self._wave_counter)

        def _decide():
            cont, reason = self.governor.verdict(state)
            return {"continue": bool(cont), "reason": str(reason)}

        d, _hit = resume_or_run_json(
            self._engine.journal,
            {"kind": "wave", "scoped_inputs": {"wave": (f"{self._node_ns}{n}" if self._node_ns else n)}},
            _decide, kind="wave", node_id=f"{self._node_ns}wave{n}")
        if isinstance(d, dict):
            cont = bool(d.get("continue", True))
            reason = str(d.get("reason") or "")
        else:  # back-compat: an older journal stored a bare bool
            cont = bool(d)
            reason = "" if cont else "plateau:no-progress"
        if not cont:
            self._halted = True
            self._halt_reason = reason or "plateau:no-progress"
            self._halt_is_cut = self._halt_reason.startswith("cut:")
        return cont

    def should_continue_waves(self) -> bool:
        """Journaled continue/halt verdict for an authored ``while`` wave loop — the
        resume-deterministic form of "may we keep escalating?". Returns False once the
        governor halts (cut-losses / agent ceiling / opt-in budget); on resume the SAME
        verdict sequence is replayed from the journal so the loop never diverges. Pair it
        with ctx.parallel: ``while ctx.should_continue_waves(): cands += ctx.parallel(...)``."""
        if self._halted:
            return False
        return self._wave_verdict(self._wave_state())

    # ---- read-accessors for escalation patterns ----
    def all_candidates(self) -> list:
        return [c for c in self._all_candidates if c is not None]

    @property
    def best_pass_rate(self) -> float:
        return self._best_pass_rate

    def residual_failures(self) -> list:
        best = None
        for c in self.all_candidates():
            if best is None or (c.public_signal_score or 0.0) > (best.public_signal_score or 0.0):
                best = c
        return list((best.meta or {}).get("failing_nodeids") or []) if best is not None else []

    def signals(self, thunks: Sequence[Callable[[], Any]]) -> list:
        """Run READ-ONLY signal thunks (e.g. ``ctx.ask`` fan-out) concurrently WITHOUT
        plateau accounting. Patterns that gather LLM signals use this so a verifier/judge
        fan-out is never mistaken for a solve wave that advances the plateau counter."""
        return self._engine.parallel(list(thunks))

    def pipeline(self, items: Sequence[Any], *stages: Any, **kw) -> list:
        return self._engine.pipeline(items, *stages, **kw)

    # ---- dynamic-workflows parity: launch payload (args) + nested composition (workflow) ----
    @property
    def args(self) -> Any:
        """The launch payload passed to this orchestration (== dynamic-workflows ``args``).
        Falls back to ``repo_map['args']`` so an eval harness can stash it there."""
        return self._args if self._args is not None else (self.repo_map or {}).get("args")

    def _spawn_child(self, *, args: Any = None) -> "OrchestrationContext":
        """A child context that SHARES this engine (shared agent counter / token budget /
        concurrency cap) for ctx.workflow() composition, with a NAMESPACED journal so its nodes
        never collide with the parent's on resume."""
        child = OrchestrationContext(
            self._engine, executor=self._executor, worker_specs=self._worker_specs,
            source_repo=self._source_repo, base_commit=self._base_commit,
            score_fn=self._score_fn, prompt_builder=self._prompt_builder,
            repo_map=self.repo_map, abl=self._abl, run_scope=self._run_scope,
            max_agents=self.max_agents, initial_agents=self.initial_agents,
            sandbox=self.sandbox, timeout_seconds=self.timeout_seconds,
            strategies=self.strategies, repair_iters=self.repair_iters, args=args,
            node_ns=f"{self._node_ns}w{self._nesting_depth + 1}_",
            nesting_depth=self._nesting_depth + 1,
        )
        child.expected_ids_sha = self.expected_ids_sha
        child.scoring_env_sha = self.scoring_env_sha
        return child

    def workflow(self, name_or_ref: Any, args: Any = None) -> Any:
        """Compose another orchestration inline (== dynamic-workflows ``workflow()``): resolve a
        named (catalog) or by-ref ({"scriptPath": ...}) ``orchestrate(ctx)`` source, run it in a
        CHILD context that SHARES this engine (so the agent counter, token budget, and concurrency
        cap are shared), and return its result. Limited to ONE level deep (a child cannot nest)."""
        from .catalog import resolve_workflow
        from .sandbox import run_orchestration
        if self._nesting_depth >= 1:
            raise FailLoud("ctx.workflow() nesting is limited to one level deep")
        source = resolve_workflow(name_or_ref)
        label = name_or_ref.get("scriptPath") if isinstance(name_or_ref, dict) else str(name_or_ref)
        child = self._spawn_child(args=args)
        self.log(f"workflow: running nested '{label}' (shared engine; ns={child._node_ns!r})")
        return run_orchestration(source, child)

    # ---- composable quality patterns (Backbone 2.3) ----
    # Thin host-side wrappers over apex_omega/patterns. Each degrades to plain best-of-N
    # at zero knobs and CANNOT set accepted: they steer compute via read-only ctx.ask
    # signals + the Candidate.set_soft/refute seam, or by producing a fresh
    # EXECUTION-SCORED solve_attempt. So a pattern can downgrade/re-rank/extend, never
    # promote an unverified solve.
    def adversarial_verify(self, candidate, **kw):
        from ..patterns import adversarial_verify as _f
        return _f(self, candidate, **kw)

    def adversarial_filter(self, items, **kw):
        """ADMIT-gate plain-data items (findings/claims): keep only those that SURVIVE N
        read-only skeptics. Never touches Candidate.accepted (Cardinal Contract)."""
        from ..patterns import adversarial_filter as _f
        return _f(self, items, **kw)

    def judge_panel(self, candidates, **kw):
        from ..patterns import judge_panel as _f
        return _f(self, candidates, **kw)

    def judge_select(self, candidates, **kw):
        """Judge-panel-then-SELECT: attach the soft judge score, then return the
        EXECUTION-AUTHORITATIVE winner (ctx.select). Judges can only break execution-equal
        ties — never promote an unaccepted candidate (Cardinal Contract)."""
        from ..patterns import judge_select as _f
        return _f(self, candidates, **kw)

    def tournament(self, candidates, **kw):
        """Pairwise round-robin judging -> SOFT win-rate tiebreak (re-rank later with ctx.select).
        A soft signal only: cannot promote an unaccepted candidate (Cardinal Contract)."""
        from ..patterns import tournament as _f
        return _f(self, candidates, **kw)

    def classify_and_route(self, items, *, classify, routes, **kw):
        """Classify each item with a read-only ctx.ask, then dispatch it to routes[category]
        (e.g. cheap model for easy items, stronger vendor for hard ones)."""
        from ..patterns import classify_and_route as _f
        return _f(self, items, classify=classify, routes=routes, **kw)

    def quarantined_ask(self, question, untrusted_content, **kw):
        """Analyze UNTRUSTED content with a strictly read-only, anti-injection-framed agent
        (the quarantine pattern). A SIGNAL only — never produces a Candidate."""
        from ..patterns import quarantined_ask as _f
        return _f(self, question, untrusted_content, **kw)

    def synthesize(self, candidates, *, attempt_id, **kw):
        from ..patterns import synthesize as _f
        return _f(self, candidates, attempt_id=attempt_id, **kw)

    def loop_until_dry(self, make_round, **kw):
        from ..patterns import loop_until_dry as _f
        return _f(self, make_round, **kw)

    def completeness_critic(self, candidate, **kw):
        from ..patterns import completeness_critic as _f
        return _f(self, candidate, **kw)

    # ---- IOU / blocked-on deferral (bounded-loop termination, dynamic-workflows parity) ----
    def defer(self, scope: str, item: Any, reason: str = "") -> dict:
        """Record an unresolved item as a structured IOU (== ``todo!("blocked_on: scope::item")``)
        so a bounded loop can stop instead of spinning, deferring resolution to a downstream phase.
        Returns the recorded record; read them back with ``ctx.blocked()``."""
        rec = {"scope": str(scope), "item": item, "reason": str(reason or "")}
        self._iou.append(rec)
        self.log(f"defer: blocked_on {scope}::{item}" + (f" ({reason})" if reason else ""))
        return rec

    def blocked(self, scope: Optional[str] = None) -> list:
        """The IOU/blocked-on deferrals recorded via ``ctx.defer`` (optionally filtered by scope)."""
        return [r for r in self._iou if scope is None or r.get("scope") == scope]

    # ---- the core unit of work: one verified attempt ----
    def _next_attempt_id(self) -> int:
        with self._counter_lock:
            return next(self._attempt_counter)

    def _attempt(self, *, aid: int, prefix: str, node_prefix: str, prompt: str,
                 strategy: str, vendor: Optional[str], model: Optional[str],
                 scoped_extra: Optional[dict] = None, meta_extra: Optional[dict] = None,
                 checkpoint: bool = True, agent_type: str = "",
                 phase: Optional[str] = None, label: Optional[str] = None,
                 pre_apply_diff: str = "") -> Optional[Candidate]:
        """The SHARED body of every scored attempt (Backbone 2.2 extraction). Acquire a
        fresh worktree, run ONE journaled coding agent, materialize its diff, score it by
        execution, build a ranked Candidate, bank it, and (if accepted) checkpoint. This
        is the ONLY place an attempt is scored, so solve and repair are mechanically
        identical and journal-replay-safe — and neither can mark itself accepted.

        ``prefix`` is the candidate-id/worktree prefix ('a' solve, 'r' repair);
        ``node_prefix`` is the journal node-id prefix. Returns a Candidate, or None on infra
        failure. The attempt runs with internet OFF (iron-tight) and NO anti-fetch prompt:
        cheating is prevented STRUCTURALLY (worktree shadows site-packages + no network), not by
        limiting the model; a blocked escape is recorded as telemetry, never penalized.

        CARRY-FORWARD (``pre_apply_diff``): the running best partial diff is applied into the
        fresh worktree BEFORE the agent runs, so a module/repair agent EDITS the accumulated
        work instead of re-implementing from scratch (closes the babel/mimesis off-by-K class).
        The fresh path otherwise pre-applies NOTHING (the resume-HIT ``materialize`` callback only
        re-applies a journaled cached diff). A carry that FAILS to apply (3-way conflict) is the
        load-bearing conflict signal: we return an INDETERMINATE Candidate (meta carry_conflict=
        True, empty diff) — NOT None — so the loop distinguishes a carry-conflict (re-solve the
        module clean, NEVER erase the carry) from an infra non-result, and the prior carry is kept."""
        spec = None
        if vendor is not None:
            spec = next((s for s in self._worker_specs if s.vendor == vendor), None)
        if spec is None:
            spec = self._worker_specs[aid % len(self._worker_specs)]
        try:
            handle = self._provider.acquire(f"{self._node_ns}{prefix}{aid}")
        except Exception as exc:
            self.log(f"attempt {self._node_ns}{prefix}{aid}: worktree acquire failed: {exc}")
            return None
        try:
            wt = handle.path
            # CARRY-FORWARD seed: apply the running best partial diff BEFORE the agent edits.
            # apply_diff (worktree.py) tries strict then --3way and returns False on BOTH —
            # that False is the SOLE conflict signal. A failed carry is NEVER silently dropped
            # and NEVER no-op'd: it yields an INDETERMINATE Candidate so the caller re-solves
            # the module clean against the last-known-good carry (load-bearing; review-fix #12
            # mirror — but here we return an indeterminate Candidate instead of raising, because
            # the convergence loop must keep the carry rather than abort the whole cell).
            if pre_apply_diff and not apply_diff(wt, pre_apply_diff):
                self.log(f"attempt {prefix}{aid}: carry-forward diff conflicted (indeterminate; re-solve clean)")
                vr_c = VerificationResult(accepted=False, score=0.0, indeterminate=True,
                                          reason="carry_conflict")
                meta_c = {"vendor": spec.vendor, "model": model or spec.model, "strategy": strategy,
                          "finalization_status": "infra_nonresult", "ok": False,
                          "pass_rate": 0.0, "indeterminate": True, "gold_passed": 0, "gold_total": 0,
                          "errors": 0, "empty_diff": True, "failing_nodeids": [],
                          "failure_excerpts": "", "carry_conflict": True}
                if meta_extra:
                    meta_c.update(meta_extra)
                    meta_c["carry_conflict"] = True   # never let meta_extra clobber the signal
                cand_c = candidate_from_verification(
                    candidate_id=f"{self._node_ns}{prefix}{aid}", diff="", vr=vr_c,
                    rollout_id=aid, cluster_id=aid, meta=meta_c)
                self._all_candidates.append(cand_c)
                return cand_c
            session = self._executor.spawn(wt, spec.vendor, model or spec.model, spec=getattr(spec, "extra", {}))
            scoped = {"repo_snapshot_sha": self._provider.base_commit, "attempt": aid, "strategy": strategy}
            if scoped_extra:
                scoped.update(scoped_extra)
            if self._node_ns:           # namespace the CACHE KEY for a nested child (root unchanged)
                scoped["ns"] = self._node_ns
            res = self._engine.agent(
                # internet stays OFF (iron-tight: no network egress — the agent must NOT be able
                # to fetch the upstream package). We do NOT enable internet to avoid the abort:
                # "don't penalize/kill for trying" is handled at OUR layer instead — the worktree
                # is scored regardless of finalization, a policy_violation attempt is never
                # excluded (see solve_and_repair) and is recorded as telemetry, not punished.
                ScopedTask(prompt=prompt, sandbox=self.sandbox,
                           model=model or spec.model, vendor=spec.vendor,
                           timeout_seconds=self.per_agent_timeout_seconds, scoped_inputs=scoped),
                lambda t: session.run(t), node_id=f"{self._node_ns}{node_prefix}{aid}",
                cli_version=getattr(session, "cli_version", ""),
                materialize=lambda diff, _wt=wt: _materialize_cached_diff(_wt, diff),
                agent_type=agent_type, phase=(str(phase) if phase else ""),
                label=(str(label) if label else ""),
            )
            vr = self._scored(wt, res)
            meta = {"vendor": spec.vendor, "model": model or spec.model, "strategy": strategy,
                    "finalization_status": res.finalization_status, "ok": res.ok,
                    "pass_rate": vr.pass_rate, "indeterminate": vr.indeterminate,
                    # GOLD distance-to-solve (now that VerificationResult.total reads total_tests):
                    # gold_passed = gold ids green, gold_total = gold id count. The cut-losses
                    # detector's PRIMARY progress tier is best gold_passed, not raw pass_rate.
                    "gold_passed": int(getattr(vr, "passed", 0) or 0),
                    "gold_total": int(getattr(vr, "total", 0) or 0),
                    "errors": int(getattr(vr, "errors", 0) or 0),
                    # F1: a no-edit attempt is sterile regardless of its (candidate-id-derived) sha
                    "empty_diff": not bool((res.fs_diff or "").strip()),
                    "failing_nodeids": list(vr.failing_nodeids), "failure_excerpts": vr.failure_excerpts}
            if meta_extra:
                meta.update(meta_extra)
            cand = candidate_from_verification(
                candidate_id=f"{self._node_ns}{prefix}{aid}", diff=res.fs_diff, vr=vr,
                rollout_id=aid, cluster_id=aid, meta=meta,
            )
            # record (never penalize) escape/cheat attempts — telemetry only.
            integ = classify_attempt_integrity(res)
            if integ["attempted"]:
                cand.meta["integrity"] = integ
                self._record_integrity(f"{prefix}{aid}", spec.vendor, integ)
            self._all_candidates.append(cand)
            # checkpoint=False lets the host-side floor-probe BANK (journal) the candidate
            # for resilience WITHOUT writing the cross-process "cell solved" signal, so an
            # honest "autogen stands alone" run is not reported solved via the template floor.
            if cand.accepted and checkpoint:
                self._checkpoint_accepted(cand)
            return cand
        except Exception as exc:
            self.log(f"attempt {prefix}{aid}: {type(exc).__name__}: {exc}")
            return None
        finally:
            self._provider.release(handle, confirm_patch_extracted=True)

    def solve_attempt(self, *, strategy: Optional[str] = None, vendor: Optional[str] = None,
                      model: Optional[str] = None, prompt: Optional[str] = None,
                      attempt_id: Optional[int] = None, checkpoint: bool = True,
                      phase: Optional[str] = None, label: Optional[str] = None,
                      agent_type: str = "") -> Optional[Candidate]:
        """Run ONE isolated coding-agent attempt and score it by execution.
        Returns a ranked Candidate (accepted iff the visible suite is green) or
        None on a hard infra failure.  Worktree lifecycle is managed by ``_attempt`` —
        the generated code never touches the filesystem.

        ``phase`` / ``label`` group and name this agent in the narration UI (the per-agent
        dynamic-workflows agent() opts); ``agent_type`` tags it (e.g. route a class of work)."""
        aid = self._next_attempt_id() if attempt_id is None else _as_int_id(attempt_id)
        strat = strategy or self.strategies[aid % len(self.strategies)]
        task_prompt = prompt or self._prompt_builder(self, aid, strat)
        return self._attempt(aid=aid, prefix="a", node_prefix="attempt", prompt=task_prompt,
                             strategy=strat, vendor=vendor, model=model, checkpoint=checkpoint,
                             agent_type=agent_type, phase=phase, label=label)

    # ---- read-only schema'd sub-question: a SIGNAL, never a Candidate ----
    def ask(self, prompt: str, *, schema: Optional[dict] = None, vendor: Optional[str] = None,
            model: Optional[str] = None, agent_id: Optional[int] = None,
            max_nudges: int = 2, strict: bool = False,
            phase: Optional[str] = None, label: Optional[str] = None, agent_type: str = "ask"):
        """Ask a coding agent a READ-ONLY sub-question over the source repo (Backbone 2.2).

        Generalizes the scout: runs in a fresh **forced read-only** session (no worktree,
        no diff, no score), is journaled/replayable, and returns the agent's
        ``structured_output`` (dict OR list, when ``schema`` is given) or its final text — a
        SIGNAL the orchestrator may use to STEER compute (which files, which approach, refute
        a candidate). It can NEVER produce a Candidate or touch acceptance, so a pattern built
        on it cannot promote an unverified solve.

        SCHEMA CONTRACT (dynamic-workflows parity, guide §2.1/§2.2): the paradigm validates a
        schema'd reply "at the tool-call layer" and "retries on mismatch" — the canonical TERMINAL
        outcome is to RETURN NULL (callers ``.filter(Boolean)``), NOT to throw (verified against
        primary sources; the guide's "throws after 2 nudges" is over-precise — no source fixes a
        retry count, and the only documented throw is the distinct "subagent never called the
        structured-output tool" condition). So here: when ``schema`` is given the reply is
        VALIDATED; on a miss the question is RE-ASKED with a nudge that states the exact validation
        error, up to ``max_nudges`` times (the count is OUR choice — the paradigm fixes none) — each
        nudge a DISTINCT journaled, replay-deterministic agent call (the nudge prompt is a pure
        function of the prior, journaled reply). After exhausting nudges with no valid reply: returns
        ``None`` (default — the canonical null terminal; also keeps verify/judge fan-outs fail-open
        and degrading to plain best-of-N) or raises ``FailLoud`` when ``strict=True`` (opt-in
        fail-loud for orchestrators that want a hard stop on an unmet schema).

        ``phase`` / ``label`` group and name this agent in the narration UI (the per-agent
        agent() opts); ``agent_type`` tags it (defaults to "ask"). An explicit ``agent_id``
        makes the question replayable; when omitted the id is DERIVED from a stable hash of
        (prompt, schema, vendor, model) so an unseeded ask is still replay-deterministic under
        concurrent fan-out. Returns None on infra failure."""
        if agent_id is None:
            from ..journal.key import sha256_hex
            seed = sha256_hex("ask|" + str(prompt) + "|" + str(schema) + "|" + str(vendor) + "|" + str(model))
            aid = int(seed[:12], 16)
        else:
            aid = _as_int_id(agent_id)
        spec = None
        if vendor is not None:
            spec = next((s for s in self._worker_specs if s.vendor == vendor), None)
        if spec is None:
            spec = self._worker_specs[aid % len(self._worker_specs)]
        try:
            session = self._executor.spawn(self._source_repo, spec.vendor, model or spec.model,
                                           spec=getattr(spec, "extra", {}))
        except Exception as exc:
            self.log(f"ask {aid}: spawn {type(exc).__name__}: {exc}")
            return None

        base_prompt = str(prompt)
        cur_prompt = base_prompt
        rounds = (1 + max(0, int(max_nudges))) if schema is not None else 1
        last_err = ""
        for k in range(rounds):
            # nudge k>0 is its own journal node (distinct node_id + scoped "nudge" key); k==0
            # keeps the exact prior key shape so old journals still replay as a cache HIT.
            scoped = {"ask": (f"{self._node_ns}{aid}" if self._node_ns else aid),
                      "repo_snapshot_sha": self._provider.base_commit}
            node = f"{self._node_ns}ask{aid}"
            if k > 0:
                scoped = {**scoped, "nudge": k}
                node = f"{self._node_ns}ask{aid}n{k}"
            try:
                res = self._engine.agent(
                    ScopedTask(prompt=cur_prompt, schema=schema, sandbox="read-only",
                               model=model or spec.model, vendor=spec.vendor,
                               timeout_seconds=self.per_agent_timeout_seconds, scoped_inputs=scoped),
                    lambda t: session.run(t), node_id=node,
                    cli_version=getattr(session, "cli_version", ""), agent_type=agent_type,
                    phase=(str(phase) if phase else ""), label=(str(label) if label else ""),
                )
            except Exception as exc:
                self.log(f"ask {aid}: {type(exc).__name__}: {exc}")
                return None
            if not res.ok:
                return None  # transport failure: a schema nudge cannot fix infra -> fail-open
            if schema is None:
                return res.final_message or ""
            out = res.structured_output
            # JSON permits top-level scalars/arrays, and validate_schema handles them — so
            # validate ANY returned value; only a missing reply (None == vendor returned no
            # parsed JSON) is the "no structured value" miss. (A schema literally requiring
            # null at top level is pathological and ambiguous with "no reply" — out of scope.)
            if out is None:
                ok, err = False, "no structured JSON value was returned"
            else:
                ok, err = validate_schema(out, schema)
            if ok:
                return out
            last_err = err
            if k + 1 < rounds:
                self.log(f"ask {aid}: schema miss (nudge {k + 1}/{max_nudges}): {err}")
                import json as _json
                cur_prompt = (
                    base_prompt
                    + "\n\nYOUR PREVIOUS REPLY DID NOT MATCH THE REQUIRED SCHEMA: " + err
                    + "\nReturn ONLY a single JSON value that matches this schema EXACTLY:\n"
                    + _json.dumps(schema, sort_keys=True)[:2000])
        if strict:
            raise FailLoud(f"ctx.ask({aid}): schema not satisfied after {max_nudges} nudges: {last_err}")
        return None

    def make_attempt(self, i: Optional[int] = None) -> Callable[[], Optional[Candidate]]:
        """Return a thunk for a diversified attempt (strategy + vendor by index),
        ready to hand to ``parallel``."""
        return lambda: self.solve_attempt(attempt_id=i)

    # ---- test-driven repair: ADDITIONAL WORK ON TOP OF A ROLLOUT ----
    def repair_attempt(self, parent: Optional[Candidate], *, attempt_id: Optional[int] = None,
                       strategy: str = "repair", vendor: Optional[str] = None,
                       model: Optional[str] = None) -> Optional[Candidate]:
        """Run ONE more agent to REPAIR a genuine-but-incomplete ``parent`` attempt,
        seeded with the parent's diff + its failing tests (Reflexion-style execution
        feedback).  Runs in a fresh worktree and is re-scored by execution — so it is
        mechanically identical to ``solve_attempt`` (journal-replay safe) and can NEVER
        set ``accepted`` itself.  Returns a Candidate, or None on infra failure."""
        if parent is None:
            return None
        aid = self._next_attempt_id() if attempt_id is None else _as_int_id(attempt_id)
        pmeta = parent.meta or {}
        failing = pmeta.get("failing_nodeids") or []
        excerpts = pmeta.get("failure_excerpts") or ""
        pdiff = (parent.diff or "")[:self._repair_diff_limit]
        base_prompt = self._prompt_builder(self, aid, strategy)
        # Phase 3 (redact): the pytest failure tail is the Reflexion signal that lets the
        # repair agent target the real failures. It is now included by DEFAULT (the
        # convergence loop needs it), but ONLY after redact_excerpts() reduces it to
        # sanitized failing node ids (review-fix #6: it used to be injected raw, defeating
        # the firewall the redactor exists to enforce). Set APEX_OMEGA_REPAIR_EXCERPTS=0 to
        # drop the tail entirely. Un-redacted excerpts are a separate, narrowly-scoped
        # convergence-residual brief decision (not this general repair path).
        excerpt_block = ""
        if excerpts and os.environ.get("APEX_OMEGA_REPAIR_EXCERPTS", "1") != "0":
            try:
                from ..eval.design_contract import redact_excerpts
                safe = redact_excerpts(excerpts, arity_by_base=None,
                                       symbols=frozenset(self.repo_map.get("modules") or []))
                if safe.strip():
                    excerpt_block = "Failure node ids (sanitized):\n" + safe + "\n"
            except Exception:
                excerpt_block = ""
        repair_prompt = (
            base_prompt
            + "\n\n--- REPAIR PASS (test-driven) ---\nA prior attempt produced this patch "
            + f"(pass_rate={float(parent.public_signal_score or 0.0):.2f}) but the visible test "
            + "suite is NOT fully green:\n```diff\n" + pdiff + "\n```\n"
            + (("Failing tests: " + ", ".join(map(str, failing[:30])) + "\n") if failing else "")
            + excerpt_block
            + "Re-implement in THIS clean workspace: keep what worked, diagnose the failures, "
            + "and make the smallest correct changes to turn the whole visible suite green."
        )
        return self._attempt(
            aid=aid, prefix="r", node_prefix="repair", prompt=repair_prompt, strategy=strategy,
            vendor=vendor, model=model,
            scoped_extra={"repair_of": parent.candidate_id, "parent_sha": parent.content_sha,
                          "failing": list(failing[:30])},
            meta_extra={"repair_of": parent.candidate_id},
        )

    def solve_and_repair(self, *, attempt_id: Optional[int] = None, strategy: Optional[str] = None,
                         vendor: Optional[str] = None, model: Optional[str] = None,
                         prompt: Optional[str] = None, max_iters: int = 2,
                         **_ignored) -> Optional[Candidate]:
        """A test-driven repair LINEAGE: a base attempt, then up to ``max_iters``
        repair passes seeded by the failing tests — stopping as soon as a pass is
        accepted, progress plateaus (pass_rate not strictly improving), the base is
        a non-genuine abort (policy/infra/timeout/pass_rate==0), or budget/agent cap
        is hit.  Returns the BEST candidate of the lineage (accepted preferred, else
        highest pass_rate).  With ``max_iters=0`` this is exactly ``solve_attempt``
        (so the lineage is provably never worse than flat best-of-N).

        Accepts the same kwargs as ``solve_attempt`` (incl. ``prompt``) and tolerates
        unknown kwargs (logs + ignores) so a stochastic AUTHORED orchestrator that
        invents a keyword can never crash the whole cell."""
        if _ignored:
            self.log(f"solve_and_repair: ignoring unsupported kwargs {sorted(_ignored)}")
        aid = self._next_attempt_id() if attempt_id is None else _as_int_id(attempt_id)
        cur = self.solve_attempt(attempt_id=aid, strategy=strategy, vendor=vendor,
                                 model=model, prompt=prompt)
        lineage = [c for c in [cur] if c is not None]
        # Clamp to the context-wide ceiling (default 0 == repair OFF == flat best-of-N),
        # so an authored orchestrator cannot exceed the configured repair budget.
        eff_iters = min(max(0, int(max_iters)), self.repair_iters)
        for k in range(eff_iters):
            if cur is None or cur.accepted:
                break
            meta = cur.meta or {}
            if meta.get("finalization_status") in ("infra_nonresult", "timeout"):
                break  # a genuine non-result (nothing to build on). NOTE: a blocked escape
                       # attempt (policy_violation) is deliberately NOT excluded here — it still
                       # produced a worktree we scored, so repair may build on it. We never
                       # penalize the agent for *trying* to break out; the sandbox just denies it.
            if meta.get("indeterminate") or (cur.public_signal_score or 0.0) <= 0.0:
                break  # nothing to build on
            if not self.budget.can_start() or self.agents_used() >= self.max_agents:
                break
            # review-fix #4: derive the repair id deterministically from the (deterministic)
            # base aid + iteration index, in a namespace disjoint from base aids (<=ceiling)
            # and pattern ids (>=900000). Drawing it from the shared call-time counter made
            # concurrent repair lineages scheduling-dependent -> false cache misses on resume.
            nxt = self.repair_attempt(cur, vendor=vendor, model=model,
                                      attempt_id=700000 + (aid % 1000) * 100 + k)
            if nxt is None:
                break
            lineage.append(nxt)
            improved = (nxt.public_signal_score or 0.0) > (cur.public_signal_score or 0.0) + 1e-9
            cur = nxt
            if nxt.accepted or not improved:
                break  # solved, or plateaued -> stop spending on this lineage
        accepted = [c for c in lineage if c is not None and c.accepted]
        if accepted:
            return select_best(accepted) or accepted[0]
        ranked = sorted([c for c in lineage if c is not None],
                        key=lambda c: (c.public_signal_score or 0.0), reverse=True)
        return ranked[0] if ranked else None

    def make_repairing_attempt(self, i: Optional[int] = None, *, max_iters: int = 2) -> Callable[[], Optional[Candidate]]:
        """Thunk for a repair LINEAGE (base attempt + up to ``max_iters`` test-driven
        repairs), ready to hand to ``parallel`` exactly like ``make_attempt``."""
        return lambda: self.solve_and_repair(attempt_id=i, max_iters=max_iters)

    # ---- CONVERGENCE STRUCTURE (Phase 2): decompose -> fan-out -> reduce -> loop-until-dry ----
    # The four seams below add the missing CONVERGENCE STRUCTURE to the default orchestration.
    # All are journaled/replay-safe (they reuse _attempt / ask / _provider / apply_diff), and
    # NONE can set ``accepted`` — acceptance stays engine-owned and execution-grounded.

    def carry_best(self) -> str:
        """The running best PARTIAL diff to carry forward into the next wave: the diff of the
        VALID (non-indeterminate) candidate with the highest gold-pass count (raw pass_rate as
        the secondary tie-break). Monotone — mirrors the _best_gold_passed frontier (context.py
        parallel accounting) so a fresh worktree is always seeded with the strongest accumulated
        work, closing the off-by-K near-solve discard. Empty string when nothing usable yet."""
        best = None
        best_key = (-1, -1.0)
        for c in self.all_candidates():
            m = getattr(c, "meta", {}) or {}
            if m.get("indeterminate") or m.get("carry_conflict"):
                continue
            if not (c.diff or "").strip():
                continue
            key = (int(m.get("gold_passed", 0) or 0), float(c.public_signal_score or 0.0))
            if key > best_key:
                best_key = key
                best = c
        return (best.diff or "") if best is not None else ""

    def module_gold_ids(self, modules: Sequence[dict]) -> list:
        """Union of the per-module ``gold_test_ids`` from a decompose plan — a concrete, pure
        repair target for loop-until-dry when the merged tree ERRORS at collection (so
        ``failing_nodeids`` is empty and the old ``and residual`` guard would never engage repair).
        No LLM call; deterministic over the journaled plan."""
        out: set = set()
        for m in (modules or []):
            for i in ((m or {}).get("gold_test_ids") or []):
                out.add(str(i))
        return sorted(out)

    def modules_overlap(self, candidates: Sequence[Optional[Candidate]], *,
                        threshold: float = 0.5) -> bool:
        """DETECT (not assume) the competing-WHOLE-REPO-candidate case: True when any two VALID
        candidate diffs touch a substantially overlapping path set (Jaccard >= ``threshold``). The
        independence the fan-out ``pipeline`` primitive ASSUMES is then violated, so a textual merge
        will conflict and the paradigm says SELECT among the competing fulls rather than merge them.
        Pure over the candidate diffs (diagnostic; does not mutate candidates)."""
        sets = []
        for c in candidates:
            if c is None:
                continue
            m = getattr(c, "meta", {}) or {}
            if m.get("indeterminate") or m.get("carry_conflict"):
                continue
            paths = _diff_touched_paths(getattr(c, "diff", "") or "")
            if paths:
                sets.append(paths)
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                inter = len(sets[i] & sets[j])
                if not inter:
                    continue
                union = len(sets[i] | sets[j])
                if union and (inter / union) >= threshold:
                    return True
        return False

    def decompose(self, *, vendor: Optional[str] = None, model: Optional[str] = None,
                  agent_id: int = 700100) -> Optional[dict]:
        """Read-only, schema-validated repo DECOMPOSITION (the convergence default's wave 0).

        Runs ONE read-only ctx.ask (sandbox=read-only, effort=high, FIXED agent_id 700100 so it
        is replay-deterministic and disjoint from base/repair/pattern id namespaces) that returns
        a module breakdown: ``{"modules":[{"module","gold_test_ids","depends_on"}],"order":[...]}``.
        The number of modules becomes the PRIMARY difficulty signal (file-count stays the floor).

        FAIL-OPEN: on a schema-miss / undecomposable repo ctx.ask returns None -> we fall back to
        repo_map['modules'] (the build_repo_map top-level package names) wrapped as a degenerate
        single-module plan, or None when there is nothing — the caller then stays on best-of-N.
        Stores the chosen plan on ``self.repo_map['decomposition']`` for telemetry/replay."""
        modules = list((self.repo_map or {}).get("modules") or [])
        approach = str((self.repo_map or {}).get("approach") or "")[:1500]
        prompt = (
            "You are SCOPING a Python repository for a parallel implementation effort. Do NOT "
            "write any code. Read the source tree and the test suite and decompose the work into "
            "INDEPENDENT modules, each owning a disjoint slice of the failing/empty implementation.\n"
            "For each module return: `module` (the top-level package/sub-package or file group it "
            "implements), `gold_test_ids` (the exact pytest node-ids that module must turn green), "
            "and `depends_on` (other module names it must be implemented after). Also return "
            "`order`: a topological ordering of the module names.\n"
            + (f"Known top-level packages (a hint, not exhaustive): {modules[:50]}\n" if modules else "")
            + (f"\nScout notes:\n{approach}\n" if approach else "")
            + "\nReturn ONLY the JSON object matching the required schema."
        )
        plan = self.ask(prompt, schema=DECOMPOSE_SCHEMA, vendor=vendor, model=model,
                        agent_id=agent_id, max_nudges=2, phase="decompose", label="scope",
                        agent_type="decompose")
        if isinstance(plan, dict) and plan.get("modules"):
            mods = [m for m in plan["modules"] if isinstance(m, dict) and m.get("module")]
            if mods:
                order = [str(o) for o in (plan.get("order") or [])] or [str(m["module"]) for m in mods]
                chosen = {"modules": mods, "order": order}
                self.repo_map["decomposition"] = chosen
                return chosen
        # fail-open: degenerate plan from repo_map['modules'] (no gold subset -> caller will treat
        # <=1 module as "skip decomposition"); None when there is truly nothing to decompose.
        if modules:
            chosen = {"modules": [{"module": str(m), "gold_test_ids": [], "depends_on": []}
                                  for m in modules],
                      "order": [str(m) for m in modules]}
            self.repo_map["decomposition"] = chosen
            return chosen
        return None

    def last_residual(self) -> list:
        """The most recent full-suite residual failing node-ids (set by reduce_residuals), used to
        GROUND the goal-alignment gate's skeptics in execution reality. Falls back to the best
        candidate's failing ids when no reduce has run yet. Read-only; no agent."""
        return list(self._last_residual) if self._last_residual else self.residual_failures()

    # ---- Claude-Code-style PHASED planning (host-side; acceptance stays engine-owned) ----
    def plan_phases(self, *, plan: dict, max_phases: int, vendor: Optional[str] = None,
                    model: Optional[str] = None, agent_id: int = 700200) -> Optional[list]:
        """ONE read-only planner subagent (ctx.ask, PHASE_PLAN_SCHEMA, FIXED agent_id 700200 —
        disjoint from decompose 700100): group the decompose ``plan`` modules into <= max_phases
        ORDERED phases (manageable chunks WITH objectives + per-phase acceptance), in dependency
        order. Each phase's acceptance_gold_ids are VALIDATED against the real gold inventory (the
        union of the plan's module gold ids); hallucinated ids are dropped and, if a phase is left
        empty, its constituent modules' gold ids are substituted. Persists phase_plan.json (durable
        like ~/.claude/plans — re-read, not re-planned, on resume). FAIL-OPEN: schema-miss / < 2
        valid phases -> None (caller falls through to whole-repo converge)."""
        import json as _json
        pp = Path(self._engine.run_dir) / "phase_plan.json"
        if pp.exists():                                  # resume: re-read, never re-plan
            try:
                saved = _json.loads(pp.read_text())
                if isinstance(saved, list) and len(saved) >= 2:
                    return saved
            except Exception:
                pass
        modules = [m for m in ((plan or {}).get("modules") or []) if isinstance(m, dict) and m.get("module")]
        if len(modules) < 2:
            return None
        inventory = set(self.module_gold_ids(modules))
        gold_by_mod = {str(m.get("module")): [str(i) for i in (m.get("gold_test_ids") or [])]
                       for m in modules}
        order = [str(o) for o in ((plan or {}).get("order") or [])] or list(gold_by_mod)
        mod_brief = [{"module": str(m.get("module")), "depends_on": list(m.get("depends_on") or []),
                      "n_gold_tests": len(m.get("gold_test_ids") or []),
                      "files": list(m.get("files") or [])[:8]} for m in modules]
        goal = str((self.repo_map or {}).get("task_framing")
                   or (self.repo_map or {}).get("approach") or "")[:1500]
        prompt = (
            "You are PLANNING a repository-completion task the way a senior engineer breaks a large "
            "job into an ORDERED set of manageable phases, each with a clear objective and a concrete "
            "acceptance criterion. Do NOT write code. Group the modules below into AT MOST "
            + str(int(max_phases)) + " phases, ordered so each phase only depends on earlier ones. "
            "MERGE thin/tightly-coupled modules into one phase — do not over-split. For each phase "
            "return: name, objective (what this phase delivers), modules (the module names it covers), "
            "acceptance_gold_ids (the union of those modules' gold_test_ids — the exact tests that "
            "must be green for the phase to be DONE), files_owned (the files this phase may edit), and "
            "depends_on (earlier phase names).\n\n"
            + ("OVERALL GOAL:\n" + goal + "\n\n" if goal else "")
            + "MODULES (from decomposition):\n" + _json.dumps(mod_brief, indent=1)[:4000]
            + "\n\nTOPOLOGICAL MODULE ORDER: " + _json.dumps(order)[:1500]
            + "\n\nReturn ONLY the JSON object matching the required schema."
        )
        reply = self.ask(prompt, schema=PHASE_PLAN_SCHEMA, vendor=vendor, model=model,
                         agent_id=agent_id, max_nudges=2, phase="plan", label="phase-plan",
                         agent_type="planner")
        raw = reply.get("phases") if isinstance(reply, dict) else None
        if not raw:
            return None
        cleaned: list = []
        for ph in raw:
            if not isinstance(ph, dict):
                continue
            mods_in = [str(x) for x in (ph.get("modules") or []) if str(x) in gold_by_mod]
            ids = [str(i) for i in (ph.get("acceptance_gold_ids") or []) if str(i) in inventory]
            if not ids:                                  # planner omitted/hallucinated -> derive
                ids = sorted({i for mm in mods_in for i in gold_by_mod.get(mm, [])})
            if not ids and not mods_in:
                continue
            cleaned.append({
                "name": str(ph.get("name") or ("phase%d" % (len(cleaned) + 1))),
                "objective": str(ph.get("objective") or "")[:600],
                "acceptance_gold_ids": ids,
                "files_owned": [str(f) for f in (ph.get("files_owned") or [])][:40],
                "modules": mods_in,
                "depends_on": [str(d) for d in (ph.get("depends_on") or [])],
                "needs_custom_orchestration": bool(ph.get("needs_custom_orchestration")),
            })
        if len(cleaned) < 2:
            return None
        if len(cleaned) > int(max_phases):               # never orphan ids: fold the tail into one
            head, tail = cleaned[:int(max_phases) - 1], cleaned[int(max_phases) - 1:]
            merged = {"name": "phase_final",
                      "objective": "; ".join(t["objective"] for t in tail if t["objective"])[:600],
                      "acceptance_gold_ids": sorted({i for t in tail for i in t["acceptance_gold_ids"]}),
                      "files_owned": sorted({f for t in tail for f in t["files_owned"]}),
                      "modules": sorted({m for t in tail for m in t["modules"]}),
                      "depends_on": [], "needs_custom_orchestration": False}
            cleaned = head + [merged]
        try:
            tmp = pp.with_suffix(".json.tmp")
            tmp.write_text(_json.dumps(cleaned))
            tmp.replace(pp)
        except Exception:
            pass
        return cleaned

    def run_phase(self, phase: dict, *, carry_diff: str = "", phase_index: int = 0) -> dict:
        """Run the PROVEN converge inner loop (fanout_modules -> reduce_residuals -> loop-until-dry)
        for ONE phase, SCOPED to phase['acceptance_gold_ids'] via reduce_residuals(scope_ids=...).
        Each module agent gets the delegation contract (objective + files_owned + acceptance ids).
        Stop authority is should_continue_waves() (the SPFG+ governor) — NO new stop logic. Returns
        {merged_diff, residual, phase_passed, phase_pass_count, phase_total, accepted_full,
        candidate, conflicts}. A phase whose modules don't resolve degrades to a scoped repair loop
        on the carry; every path keeps acceptance engine-owned (only ctx.select on the full suite
        accepts).

        ``phase_index`` namespaces this phase's attempt ids (fan-out + repair) into a DISJOINT band
        so calling run_phase once per phase never collides on the journal/worktree ids — a same-id
        second phase would otherwise replay the FIRST phase's cached fan-out (the cross-phase cache
        collision bug)."""
        decomp = (self.repo_map or {}).get("decomposition") or {}
        by_name = {str(m.get("module")): m for m in (decomp.get("modules") or [])
                   if isinstance(m, dict) and m.get("module")}
        names = [str(n) for n in (phase or {}).get("modules") or []]
        mods = [by_name[n] for n in names if n in by_name]
        scope_ids = [str(i) for i in (phase or {}).get("acceptance_gold_ids") or []]
        scope_set = set(scope_ids)
        objective = str((phase or {}).get("objective") or "")
        files_owned = [str(f) for f in (phase or {}).get("files_owned") or []]
        contract = (
            "PHASE OBJECTIVE: " + (objective or "(complete the scoped tests)") + "\n"
            + ("FILES THIS PHASE OWNS (stay within these; do not edit other phases' files): "
               + ", ".join(files_owned[:40]) + "\n" if files_owned else "")
            + "Earlier phases are already implemented in this workspace — BUILD ON them, do not revert.\n"
        )
        # per-phase disjoint id bands (fan-out 734xxx..., repair 711xxx...) so phase N never
        # cache-replays phase N-1's attempts.
        pidx = max(0, int(phase_index))
        fan_base = 734000 + pidx * 4000
        rep_base = 711000 + pidx * 4000
        carry = carry_diff
        if mods:
            cands = self.fanout_modules(mods, carry_diff=carry, extra_brief=contract, id_base=fan_base)
            red = self.reduce_residuals(cands, carry_diff=carry, scope_ids=scope_ids)
        else:
            # no resolvable modules -> scoped residual repair seeded by the carry
            c = self.repair_residual(scope_ids or self.last_residual(), carry_diff=carry,
                                     attempt_id=rep_base, round=0)
            red = self.reduce_residuals([c], carry_diff=carry, scope_ids=scope_ids)
        if red.get("merged_diff"):
            carry = red["merged_diff"]
        rnd = 1
        while (self.should_continue_waves() and not red.get("accepted")
               and not red.get("phase_passed")):
            residual = [r for r in (red.get("residual_failing_ids") or []) if r in scope_set] or scope_ids
            c = self.repair_residual(residual, carry_diff=carry, attempt_id=rep_base + rnd, round=rnd)
            rnd += 1
            red = self.reduce_residuals([c], carry_diff=carry, scope_ids=scope_ids)
            if red.get("merged_diff"):
                carry = red["merged_diff"]
        return {
            "merged_diff": carry, "residual": list(red.get("residual_failing_ids") or []),
            "phase_passed": bool(red.get("phase_passed")),
            "phase_pass_count": int(red.get("phase_pass_count", 0) or 0),
            "phase_total": int(red.get("phase_total", len(scope_ids)) or 0),
            "accepted_full": bool(red.get("accepted")),
            "candidate": red.get("candidate"), "conflicts": list(red.get("conflicts") or []),
        }

    def goal_align_gate(self, plan: dict, phase: dict, *, residual_ids: Sequence[str],
                        stage: str, n: int = 3) -> dict:
        """Adversarial GOAL-ALIGNMENT review (the no-veer guard). N read-only skeptics via
        ctx.signals (no plateau accounting) each judge whether THIS phase still serves the overall
        goal G, GROUNDED in the REAL residual failing node-ids R (not the transcript — the moat over
        a transcript-only verifier). A revise/abort verdict MUST cite evidence_ids that are real
        failing ids, else it is DOWNGRADED to proceed. Grounded-majority decides; ties / no grounded
        dissent -> proceed (fail-open: the gate can STOP a veer, never stall a progressing run). A
        read-only SIGNAL: it can re-target (revise) or stop the phase loop (abort) but NEVER sets
        acceptance (C7). Off when APEX_OMEGA_GOAL_GATE=0. Returns {verdict, reason, evidence_ids,
        retarget_gold_ids}."""
        if os.environ.get("APEX_OMEGA_GOAL_GATE", "1") == "0":
            return {"verdict": "proceed", "reason": "gate disabled", "evidence_ids": [],
                    "retarget_gold_ids": []}
        import json as _json
        from ..journal.key import sha256_hex
        rid = [str(x) for x in (residual_ids or [])]
        rid_set = set(rid)
        goal = str((self.repo_map or {}).get("task_framing")
                   or (self.repo_map or {}).get("approach") or "")[:1500]
        obj = str((phase or {}).get("objective") or "")
        acc = [str(i) for i in (phase or {}).get("acceptance_gold_ids") or []]
        inventory = set(self.module_gold_ids((plan or {}).get("modules") or [])) or rid_set
        base = 700400 + int(sha256_hex(str(stage) + "|" + str((phase or {}).get("name") or ""))[:6], 16) % 80000
        nn = max(1, int(n))

        def _mk(i):
            return lambda i=i: self.ask(
                "You are an ADVERSARIAL reviewer guarding a long multi-phase run from VEERING off "
                "its goal. Decide if the current phase still serves the goal.\n\n"
                + ("OVERALL GOAL (binding):\n" + goal + "\n\n" if goal else "")
                + "CURRENT PHASE (" + str(stage) + "-check): " + (obj or "(scoped tests)") + "\n"
                + "PHASE ACCEPTANCE GOLD IDS: " + _json.dumps(acc[:40]) + "\n"
                + "REAL STILL-FAILING TEST NODE-IDS (the only admissible evidence): "
                + _json.dumps(rid[:60]) + "\n\n"
                + "Return JSON {verdict: proceed|revise|abort, reason, evidence_ids, "
                + "retarget_gold_ids}. Rules: 'proceed' if the phase is on-goal and making sense. "
                + "'revise' only if the acceptance ids are mis-scoped — put the CORRECT gold ids in "
                + "retarget_gold_ids. 'abort' only if the phase cannot serve the goal. ANY revise/"
                + "abort MUST cite evidence_ids drawn from the real failing node-ids above; with no "
                + "such evidence, return 'proceed'.",
                schema=GATE_SCHEMA, agent_id=base + i, max_nudges=1,
                phase="goal-gate", label="gate:" + str(stage), agent_type="goal_gate")

        replies = self.signals([_mk(i) for i in range(nn)])
        votes = {"proceed": 0, "revise": 0, "abort": 0}
        retarget: set = set()
        reasons: list = []
        for r in replies:
            if not isinstance(r, dict):
                continue
            v = str(r.get("verdict") or "proceed")
            ev = [str(x) for x in (r.get("evidence_ids") or [])]
            grounded = bool(rid_set) and any(e in rid_set for e in ev)
            if v in ("revise", "abort") and not grounded:
                v = "proceed"        # DOWNGRADE an ungrounded dissent (anti-hallucination)
            votes[v] = votes.get(v, 0) + 1
            if v != "proceed" and r.get("reason"):
                reasons.append(str(r.get("reason"))[:200])
            if v == "revise":
                retarget |= {str(i) for i in (r.get("retarget_gold_ids") or []) if str(i) in inventory}
        # grounded-majority: abort needs a strict majority; else revise on a strict majority; else proceed.
        if votes["abort"] > nn / 2.0:
            verdict = "abort"
        elif votes["revise"] > nn / 2.0:
            verdict = "revise"
        else:
            verdict = "proceed"
        return {"verdict": verdict, "reason": "; ".join(reasons)[:400],
                "evidence_ids": rid[:20], "retarget_gold_ids": sorted(retarget)}

    def solve_module(self, module: dict, *, carry_diff: str = "", attempt_id: Optional[int] = None,
                     vendor: Optional[str] = None, model: Optional[str] = None,
                     strategy: str = "module", prompt: Optional[str] = None,
                     extra_brief: str = "") -> Optional[Candidate]:
        """Run ONE module-scoped solve agent, seeded with the carry-forward diff applied into the
        fresh worktree BEFORE the agent edits. The agent is briefed to implement ONLY this module
        and make its gold-test subset pass; it is still scored on the FULL gold suite (the accept
        gate is unchanged). Returns a Candidate (or an INDETERMINATE carry-conflict Candidate, or
        None on infra failure)."""
        name = str((module or {}).get("module") or "module")
        gold_ids = list((module or {}).get("gold_test_ids") or [])
        aid = self._next_attempt_id() if attempt_id is None else _as_int_id(attempt_id)
        if prompt is None:
            # Prefer the eval-provided CONTRACT 1 builder (richer issue text + scout plan) when the
            # harness wired one onto repo_map; else fall back to the built-in module-scoped brief.
            _bb = (self.repo_map or {}).get("brief_builders") or {}
            _mk = _bb.get("module_solve")
            if callable(_mk):
                try:
                    prompt = _mk(self, name, gold_ids, carry_nonempty=bool((carry_diff or "").strip()))
                except Exception as exc:
                    self.log(f"solve_module: module_solve brief builder failed ({exc}); using default")
        if prompt is None:
            base = self._prompt_builder(self, aid, strategy)
            carry_note = ("\nFiles partially implemented by earlier agents are PRESENT in this "
                          "workspace — build ON them, do not revert.\n" if (carry_diff or "").strip() else "")
            ids_block = ("\n".join(map(str, gold_ids[:60])) if gold_ids else "(infer from the module's tests)")
            prompt = (
                base
                + "\n\n--- MODULE-SCOPED SOLVE ---\n"
                + f"Implement ONLY the module `{name}`. Make EXACTLY these gold tests pass — other "
                + "modules are handled by parallel agents, so do NOT reimplement the whole repo:\n"
                + ids_block + "\n"
                + "Boundaries: edit only files belonging to this module; do NOT edit/add/delete any "
                + "test file; do not touch other modules (note any genuinely-missing shared symbol "
                + "for the reducer instead of forking it).\n"
                + carry_note
                + "Run the scoped subset and iterate until that subset is green.\n"
            )
        # DELEGATION CONTRACT (graft, fact-checked: detailed objective + boundaries cure the
        # vague-delegation duplicate-work/gap failure). A phase-level objective + file-ownership
        # boundary, appended to whichever brief produced ``prompt``. Fail-open: absent -> no-op.
        if extra_brief and prompt:
            prompt = prompt + "\n\n--- PHASE DELEGATION CONTRACT ---\n" + str(extra_brief) + "\n"
        return self._attempt(
            aid=aid, prefix="m", node_prefix="module", prompt=prompt, strategy=strategy,
            vendor=vendor, model=model, pre_apply_diff=carry_diff,
            scoped_extra={"module": name, "gold_ids": list(gold_ids[:60])},
            meta_extra={"module": name},
        )

    def fanout_modules(self, modules: Sequence[dict], *, carry_diff: str = "",
                       id_base: int = 730000, extra_brief: str = "") -> list:
        """FAN-OUT per module via ctx.pipeline (no barrier, guide §4.1): each module is its own
        streaming chain, so a fast module never waits on a slow sibling. Each chain runs ONE
        module-scoped solve agent seeded with ``carry_diff``. Returns the per-module Candidate
        list (in MODULE order, never completion order), Nones filtered.

        The pipeline JOURNALS each stage output, which must be JSON-serializable — a Candidate is
        not, so the stage returns the candidate-id (a string) and we re-collect the live Candidate
        objects from ``self._all_candidates`` (where ``_attempt`` already banked them) afterward.
        This keeps the no-barrier streaming + replay determinism while preserving the real
        Candidate (with its diff/meta) for the reduce step. Module attempt-ids are deterministic
        (``id_base + index``, disjoint from base/repair/reduce namespaces) so resume replays the
        same chains."""
        mods = [m for m in (modules or []) if isinstance(m, dict)]
        if not mods:
            return []

        def _solve_stage(_prev, module, index):
            cand = self.solve_module(module, carry_diff=carry_diff, attempt_id=id_base + index,
                                     extra_brief=extra_brief)
            # forward only the JSON-safe candidate-id (the live Candidate stays in _all_candidates)
            return cand.candidate_id if cand is not None else ""

        # Key the pipeline's stage journal on id_base + module name (NOT the bare item index). The
        # engine pipeline defaults a dict item's id to its INDEX, so two fanout_modules calls in the
        # same run (e.g. the phase planner's per-phase fan-outs) would collide on "0:_solve_stage"
        # and the second phase would cache-replay the first. id_base is disjoint per phase -> unique.
        ids = self.pipeline(list(mods), _solve_stage,
                            item_id=lambda m: f"{id_base}_{(m or {}).get('module', 'm')}")
        by_id = {c.candidate_id: c for c in self._all_candidates if c is not None}
        return [by_id[i] for i in ids if i and i in by_id]

    def reduce_residuals(self, candidates: Sequence[Optional[Candidate]], *,
                         carry_diff: str = "", scope_ids: Optional[Sequence[str]] = None) -> dict:
        """REDUCE step — plain Python, NO LLM, zero tokens. Merge the per-module candidate diffs
        into ONE worktree (carry_diff first, then each candidate's diff in the order given), run
        the FULL gold suite ONCE, and return the exact residual failing node-ids.

        A per-module diff that fails to apply (apply_diff False = strict AND 3-way both failed) is
        a CONFLICT: it is recorded in ``conflicts`` (the caller re-solves it clean) and SKIPPED in
        the merge — its progress is NEVER silently erased and the carry is NEVER dropped. The
        carry itself failing to apply is the worst case (the running best can no longer be rebuilt
        here): recorded as ``__carry__`` and the merge proceeds from the bare base.

        Returns {"merged_diff", "residual_failing_ids", "accepted", "candidate", "conflicts",
        "indeterminate"}. NEVER raises on conflict.

        ``scope_ids`` (graft from the phased planner): when given, ALSO report whether THIS phase's
        acceptance gold-id subset is fully green — as a PURE SET TEST over the full-suite
        ``residual_failing_ids`` the merge already computed (NO second pytest run, strictly cheaper
        than a subset re-score). Adds ``{phase_passed, phase_pass_count, phase_total}``. The
        WHOLE-suite ``accepted`` field and its checkpoint are UNCHANGED — only ctx.select on the full
        suite ever accepts (C7). Default None == today's behaviour."""
        cands = [c for c in candidates if c is not None]
        conflicts: list = []
        indeterminate = False
        # Acquire ONE merge worktree (a deterministic id disjoint from attempt/repair namespaces).
        merge_id = 720000 + (self._next_attempt_id() % 100000)
        try:
            handle = self._provider.acquire(f"{self._node_ns}reduce{merge_id}")
        except Exception as exc:
            self.log(f"reduce_residuals: worktree acquire failed: {exc}")
            return {"merged_diff": carry_diff or "", "residual_failing_ids": [], "accepted": False,
                    "candidate": None, "conflicts": ["__acquire__"], "indeterminate": True}
        try:
            wt = handle.path
            # Strip harness scaffolding (.apex_seatbelt/) from every diff BEFORE applying: a
            # per-worktree new-file like read_jail.sb is byte-divergent across modules and would
            # conflict purely on scaffolding, sinking a genuinely-disjoint merge (defense-in-depth
            # on top of the extraction-time exclude, so even a pre-fix cached diff is safe).
            carry_clean = _strip_scaffold_hunks(carry_diff or "")
            if carry_clean.strip() and not apply_diff(wt, carry_clean):
                conflicts.append("__carry__")
                indeterminate = True
                self.log("reduce_residuals: carry diff conflicted on merge tree (re-solve from base)")
            for c in cands:
                m = getattr(c, "meta", {}) or {}
                if m.get("indeterminate") or m.get("carry_conflict"):
                    conflicts.append(str(m.get("module") or c.candidate_id))
                    continue
                d = _strip_scaffold_hunks(c.diff or "")
                if not d.strip():
                    continue
                if not apply_diff(wt, d):
                    conflicts.append(str(m.get("module") or c.candidate_id))
                    self.log(f"reduce_residuals: module diff conflicted ({m.get('module') or c.candidate_id}); "
                             "re-queued, progress preserved")
                    continue
            # capture the merged diff (everything applied on top of base) + score it ONCE.
            merged_diff = self._merged_diff(wt)
            vr = self._scored(wt, _MergeRes(merged_diff))
            meta = {"strategy": "reduce", "finalization_status": "completed", "ok": True,
                    "pass_rate": vr.pass_rate, "indeterminate": vr.indeterminate,
                    "gold_passed": int(getattr(vr, "passed", 0) or 0),
                    "gold_total": int(getattr(vr, "total", 0) or 0),
                    "errors": int(getattr(vr, "errors", 0) or 0),
                    "empty_diff": not bool(merged_diff.strip()),
                    "failing_nodeids": list(vr.failing_nodeids),
                    "failure_excerpts": vr.failure_excerpts, "conflicts": list(conflicts)}
            cand = candidate_from_verification(
                candidate_id=f"{self._node_ns}reduce{merge_id}", diff=merged_diff, vr=vr,
                rollout_id=merge_id, cluster_id=merge_id, meta=meta)
            self._all_candidates.append(cand)
            # Feed the merged full-suite measurement into the SPFG+ frontier so a climbing frontier
            # (more residual ids green across loop-until-dry rounds) RESETS the patience arms, and a
            # conflict/indeterminate reduce is neutral. No new stop logic — the existing governor
            # authority decides; should_continue_waves() consumes the updated _wave_state().
            self._observe([cand])
            if cand.accepted:
                self._checkpoint_accepted(cand)
            self._last_residual = list(vr.failing_nodeids)
            result = {"merged_diff": merged_diff, "residual_failing_ids": list(vr.failing_nodeids),
                      "accepted": bool(cand.accepted), "candidate": cand,
                      # gold_passed lets the orchestrator distinguish a CLIMBING partial (enter
                      # loop-until-dry) from a TOTAL collapse (route to SELECT/best-of-N), so a
                      # majority-conflict merge that still made progress is not thrown to best-of-N.
                      "gold_passed": int(getattr(vr, "passed", 0) or 0),
                      "conflicts": list(conflicts), "indeterminate": bool(indeterminate or vr.indeterminate)}
            if scope_ids is not None:
                # PURE SET TEST over the already-run full-suite score (no second pytest): a phase
                # passes iff ALL its acceptance gold ids are absent from the failing set AND the
                # measurement was VALID (a collection error / indeterminate never fakes a pass).
                sids = [str(s) for s in scope_ids]
                failing_set = {str(x) for x in vr.failing_nodeids}
                green = [s for s in sids if s not in failing_set]
                result["phase_total"] = len(sids)
                result["phase_pass_count"] = len(green)
                result["phase_passed"] = bool(sids and len(green) == len(sids) and not vr.indeterminate)
            return result
        finally:
            self._provider.release(handle, confirm_patch_extracted=True)

    def _merged_diff(self, wt: str) -> str:
        """The full git diff of the merge worktree vs its base commit (the carry + every applied
        module diff, captured as a single replay-safe artifact)."""
        from ..isolation.worktree import _git
        # Exclude harness scaffolding so the scored/accepted/carried artifact is worktree-path
        # INDEPENDENT (strengthens the byte-stable merged-diff cache key) and never ships .sb noise.
        res = _git("diff", self._provider.base_commit, *_SCAFFOLD_PATHSPEC, cwd=wt)
        if res.returncode == 0 and (res.stdout or "").strip():
            return res.stdout
        # fall back to the worktree-relative diff (unstaged) when the base-rev form is empty.
        res2 = _git("diff", *_SCAFFOLD_PATHSPEC, cwd=wt)
        return res2.stdout if res2.returncode == 0 else ""

    def repair_residual(self, residual_ids: Sequence[str], *, carry_diff: str,
                        excerpts: str = "", attempt_id: Optional[int] = None,
                        round: int = 0, vendor: Optional[str] = None,
                        model: Optional[str] = None, prompt: Optional[str] = None) -> Optional[Candidate]:
        """LOOP-UNTIL-DRY repair step — run ONE repair agent on the LIVE merged tree (carry_diff
        applied into the fresh worktree BEFORE the agent), scoped to the EXACT still-failing gold
        node-ids. Unlike repair_attempt (which pastes the parent diff as prompt TEXT), this seam
        APPLIES the merged tree so the agent EDITS live code — closing the off-by-K class.
        Returns a Candidate (or an indeterminate carry-conflict Candidate, or None)."""
        ids = [str(i) for i in (residual_ids or [])]
        aid = self._next_attempt_id() if attempt_id is None else _as_int_id(attempt_id)
        if attempt_id is None:
            aid = 710000 + int(round)
        if prompt is None:
            # Prefer the eval-provided CONTRACT 2 builder (state line + un-redacted excerpts when
            # enabled) when the harness wired one onto repo_map; else the built-in residual brief.
            _bb = (self.repo_map or {}).get("brief_builders") or {}
            _mk = _bb.get("residual_repair")
            if callable(_mk):
                try:
                    _best = None
                    for c in self.all_candidates():
                        cm = getattr(c, "meta", {}) or {}
                        if cm.get("indeterminate") or cm.get("carry_conflict"):
                            continue
                        if _best is None or int(cm.get("gold_passed", 0) or 0) > int((_best.meta or {}).get("gold_passed", 0) or 0):
                            _best = c
                    bm = (_best.meta if _best is not None else {}) or {}
                    passed = int(bm.get("gold_passed", 0) or 0)
                    total = int(bm.get("gold_total", 0) or 0)
                    prompt = _mk(self, ids, passed, total, excerpts=excerpts)
                except Exception as exc:
                    self.log(f"repair_residual: residual_repair brief builder failed ({exc}); using default")
        if prompt is None:
            base = self._prompt_builder(self, aid, "residual_repair")
            ids_block = "\n".join(ids[:40]) if ids else "(see the failing subset)"
            # FM-4: hard-cap the assertion tail so the repair turn stays within budget on large repos.
            excerpt_block = (("\nFailure evidence:\n" + str(excerpts)[:3000] + "\n") if (excerpts or "").strip() else "")
            prompt = (
                base
                + "\n\n--- RESIDUAL REPAIR (live merged tree) ---\n"
                + "The merged implementation is ALREADY IN THIS WORKSPACE — keep what works. "
                + f"These EXACT gold tests still FAIL; make them pass without breaking the rest:\n"
                + ids_block + "\n"
                + excerpt_block
                + "Make the smallest correct change to turn these specific tests green. Do NOT edit "
                + "tests. Re-run only the failing subset and iterate.\n"
            )
        return self._attempt(
            aid=aid, prefix="rr", node_prefix="resrepair", prompt=prompt, strategy="residual_repair",
            vendor=vendor, model=model, pre_apply_diff=carry_diff,
            scoped_extra={"residual_ids": ids[:40], "round": int(round)},
            meta_extra={"residual_repair": True},
        )

    # ---- RALPH-WIGGUM baseline: naive persistence, the IDENTICAL prompt every iteration ----
    def ralph_loop(self, *, id_base: int = 800000) -> Optional[Candidate]:
        """The ralph-wiggum baseline: ONE sequential lineage that re-runs the BYTE-IDENTICAL solve
        prompt every iteration in a PERSISTENT workspace — faithful naive iterate-until-done. The
        loop injects NOTHING between turns: NO failing-test ids, NO failure excerpts, NO diff-paste
        (any of those would make it Reflexion, not ralph). Each iteration runs in a FRESH worktree
        (a fresh context window) with the accumulated edits PRE-APPLIED — exactly replicating a ralph
        ``run.sh`` that re-invokes ``-p "$(cat prompt.md)"`` in the SAME directory each turn. The
        agent re-discovers state by reading the carried-forward workspace and runs the tests itself
        (the prompt already tells it to). NO scout / author / patterns / decomposition / parallel
        waves. Each iteration is a "wave of one" through ``ctx.parallel`` so the SAME SPFG++ governor
        that governs omega governs ralph (apples-to-apples). Stops on accept or a governor cut;
        returns the best banked candidate (no silent loss in REPORTING — the LOOP itself is naive and
        always builds on the last state, never a cherry-picked best).

        Distinct from ``baseline`` (K independent THROWAWAY rollouts, no persistence) and from omega
        (scout/author/decompose/parallel waves)."""
        # Build the prompt ONCE so every iteration re-runs the byte-identical text (ralph fidelity:
        # the prompt never varies by attempt index, strategy, or accumulated feedback).
        ralph_strategy = self.strategies[0] if getattr(self, "strategies", None) else "minimal"
        fixed_prompt = self._prompt_builder(self, id_base, ralph_strategy)
        # One agent type for the whole lineage (a single naive worker), pinned so the loop does not
        # rotate vendors by attempt index.
        ralph_vendor = self._worker_specs[0].vendor if self._worker_specs else None
        lineage: list = []
        carry = ""                       # the PERSISTENT WORKSPACE (accumulated edits) re-applied each turn
        k = 0
        while True:
            aid = id_base + k
            thunk = (lambda a=aid, c=carry: self._attempt(
                aid=a, prefix="a", node_prefix="attempt", prompt=fixed_prompt,
                strategy=ralph_strategy, vendor=ralph_vendor, model=None, pre_apply_diff=c))
            try:
                out = self.parallel([thunk])   # raises CutLosses/PlateauStop once halted
            except PlateauStop:                 # CutLosses is a subclass -> also caught
                break
            k += 1
            cand = out[0] if out else None
            if cand is None:
                continue
            lineage.append(cand)
            # PERSISTENT WORKSPACE: carry the LAST real (non-indeterminate, non-empty) diff forward.
            # Naive persistence builds on the previous iteration's state — NOT a cherry-picked best —
            # so a carry-conflict / infra non-result / no-edit turn leaves the prior workspace intact.
            m = getattr(cand, "meta", {}) or {}
            if not m.get("indeterminate") and not m.get("carry_conflict") and (cand.diff or "").strip():
                carry = cand.diff
            if cand.accepted:
                break
        return self.select(lineage)

    # ---- execution-authoritative selection (the ONLY producer of a winner) ----
    def select(self, candidates: Sequence[Optional[Candidate]]) -> Optional[Candidate]:
        cands = [c for c in candidates if c is not None]
        return select_best(cands, allow_unaccepted=not self._abl.cardinal_contract_enforced)

    def any_accepted(self, candidates: Sequence[Optional[Candidate]]) -> bool:
        return any(c is not None and c.accepted for c in candidates)

    # ---- a convenience escalation schedule (the generated code may ignore it) ----
    def plan_waves(self, schedule: Optional[Sequence[int]] = None, *,
                   start: Optional[int] = None, factor: int = 2, max_wave: int = 256) -> list:
        """Agent-cap-bounded escalation wave sizes.

        Default (``schedule=None``): a DOUBLING schedule — start with ``start``
        agents and multiply by ``factor`` each wave (1, 2, 4, 8, ...), capped per
        wave at ``max_wave`` and overall at the soft agent cap (``self.max_agents``).
        This is "fewest agents first": a couple/tens of agents on most tasks, only
        escalating toward ``max_agents`` (up to the 1000 backstop) while unsolved.

        Back-compat: passing an explicit ``schedule`` tuple keeps the old
        fixed-wave behaviour (each element bounded by the remaining cap)."""
        out, used = [], 0

        def _emit(k: int) -> bool:
            """Append a wave of size k bounded by the remaining cap. Returns False
            once the cap is exhausted (caller should stop)."""
            nonlocal used
            if used + k > self.max_agents:
                k = self.max_agents - used
            if k <= 0:
                return False
            out.append(k)
            used += k
            return used < self.max_agents

        if schedule is not None:
            for k in schedule:
                if not _emit(int(k)):
                    break
            return out

        # doubling-to-cap: start at the scout-chosen initial wave size (difficulty-
        # driven), escalate geometrically only while unsolved.
        k = max(1, int(self.initial_agents if start is None else start))
        factor = max(2, int(factor))
        while used < self.max_agents:
            if not _emit(min(k, max_wave)):
                break
            k *= factor
        return out
